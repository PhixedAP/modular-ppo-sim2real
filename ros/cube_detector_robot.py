import rospy
import sys
import gymnasium as gym
import numpy as np
import torch
import time
import torch.nn as nn
from PIL import Image as pil_image
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
import torchvision.transforms as T
import cv2
# pip install cv-bridge
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist


# cubechaser_skrl_cnn_func_cube_detector.py
class Policy(GaussianMixin, Model):
    compute_counter = 0
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum", cube_detector_neurons=6):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.cube_detector_neurons = cube_detector_neurons

        self.fc1 = nn.Linear(self.cube_detector_neurons, 2*self.cube_detector_neurons)
        self.fc11 = nn.Linear(2*self.cube_detector_neurons, 2*self.cube_detector_neurons)

        self.history_fc1 = nn.Linear(38, 76)
        self.history_fc2 = nn.Linear(76, 32)

        self.fc2 = nn.Linear(2 * self.cube_detector_neurons + 32, self.cube_detector_neurons + 16)
        self.fc3 = nn.Linear(self.cube_detector_neurons + 16, self.num_actions)

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        self.compute_counter += 1

        if self.compute_counter < 100:
            print("Policy: ", inputs["states"].shape)

        cube_detector_data = inputs["states"][:, :self.cube_detector_neurons]
        history_data = inputs["states"][:, self.cube_detector_neurons:]
        x = self.fc1(cube_detector_data)
        x = torch.tanh(x)
        x = self.fc11(x)
        x = torch.tanh(x)

        y = self.history_fc1(history_data)
        y = torch.tanh(y)
        y = self.history_fc2(y)
        y = torch.tanh(y)

        x = torch.cat((x, y), dim=1)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)
        return torch.tanh(x), self.log_std_parameter, {}


class Value(DeterministicMixin, Model):
    compute_counter = 0
    def __init__(self, observation_space, action_space, device, clip_actions=False, cube_detector_neurons=1000):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.cube_detector_neurons = cube_detector_neurons

        self.fc1 = nn.Linear(self.cube_detector_neurons, 2*self.cube_detector_neurons)
        self.fc11 = nn.Linear(2*self.cube_detector_neurons, 2*self.cube_detector_neurons)

        self.history_fc1 = nn.Linear(38, 76)
        self.history_fc2 = nn.Linear(76, 32)

        self.fc2 = nn.Linear(2*self.cube_detector_neurons + 32, self.cube_detector_neurons+16)
        self.fc3 = nn.Linear(self.cube_detector_neurons+16, 1)

    def compute(self, inputs, role):
        cube_detector_data = inputs["states"][:, :self.cube_detector_neurons]
        history_data = inputs["states"][:, self.cube_detector_neurons:]
        x = self.fc1(cube_detector_data)
        x = torch.tanh(x)
        x = self.fc11(x)
        x = torch.tanh(x)

        y = self.history_fc1(history_data)
        y = torch.tanh(y)
        y = self.history_fc2(y)
        y = torch.tanh(y)

        x = torch.cat((x, y), dim=1)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)
        return x, {}


class CubeDetectorModel:
    model_path = "/home/phil/university/thesis/IsaacLab-1.1.0/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/thesis-isaaclab-direct-cubechaser-camera/finetune/cube_detection/"
    model_name = "object_detector.pth"

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    transforms = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    image_target_width = 320
    image_target_height = 320

    def __init__(self, device):
        self.device = device
        self.model = torch.load(self.model_path + self.model_name).to(self.device)
        self.model.eval()

    def predict_image(self, image):
        # print("predict_image...")
        image = self.transforms(image).unsqueeze(0).to(self.device)
        # print("transforms done, shape:", image.shape)
        boxPreds, classPreds = self.model(image)
        # print("boxPreds:", boxPreds)
        # print("classPreds:", classPreds)
        return boxPreds, classPreds


class CubeDetectorRlAgent:
    # commits of this model:
    # https://gitlab.com/phil-masterthesis/isaaclab-direct-cubechaser-camera/-/blob/0e494eace7dc9676aa6921bb7742c85784cbd448/cubechaser_camera_env_cube_detector.py
    # https://gitlab.com/phil-masterthesis/isaaclab-direct-cubechaser-camera/-/blob/0e494eace7dc9676aa6921bb7742c85784cbd448/cubechaser_skrl_cnn_func_cube_detector.py
    model_path = "/home/phil/university/thesis/thesis-ros-jetauto/local_ws/src/fourwis_cubechaser_sb3/scripts/cube_detector/model/24-09-05_22-17-26-760051_PPO/"
    model_name = "agent_25000.pt"
    memory_size = 16
    num_envs = 1
    num_actions = 2

    size_action_history = 20
    size_robot_history = 18
    size_cube_detector_history = 10
    cube_detector_neurons = 6
    cd_neurons = cube_detector_neurons * size_cube_detector_history
    num_observations = cd_neurons + size_action_history + size_robot_history

    single_action_space = gym.spaces.Box(low=-1.0, high=1.0, dtype=np.float32, shape=(num_actions,))
    action_space = gym.vector.utils.batch_space(single_action_space, num_envs)

    single_observation_space = gym.spaces.Dict()
    single_observation_space["policy"] = gym.spaces.Box(
        low=0.0, high=1.0, dtype=np.float32,
        shape=(num_observations,)
    )
    observation_space = gym.vector.utils.batch_space(single_observation_space, num_envs)

    def __init__(self, device):
        self.action_history = torch.zeros(1, self.size_action_history, device=device, dtype=torch.float32)
        self.robot_history = torch.zeros(1, self.size_robot_history, device=device, dtype=torch.float32)
        self.cube_detector_history = torch.zeros(1, self.size_cube_detector_history * self.cube_detector_neurons, device=device, dtype=torch.float32)

        self.cube_detector_model = CubeDetectorModel(device)

        self.device = device
        cfg = PPO_DEFAULT_CONFIG.copy()
        cfg["grad_norm_clip"] = 1.0  # needed?
        cfg["state_preprocessor"] = RunningStandardScaler
        cfg["state_preprocessor_kwargs"] = {"size": self.observation_space, "device": self.device}
        cfg["value_preprocessor"] = RunningStandardScaler
        cfg["value_preprocessor_kwargs"] = {"size": 1, "device": self.device}

        self.memory = RandomMemory(memory_size=self.memory_size, num_envs=self.num_envs, device=self.device)
        self.models = {}
        self.models["policy"] = Policy(self.observation_space, self.action_space, self.device, clip_actions=True, cube_detector_neurons=self.cd_neurons)
        self.models["value"] = Value(self.observation_space, self.action_space, self.device, cube_detector_neurons=self.cd_neurons)
        self.agent = PPO(models=self.models,
                    memory=self.memory,
                    cfg=cfg,
                    observation_space=self.observation_space,
                    action_space=self.action_space,
                    device=self.device)
        self.agent.load(self.model_path + self.model_name)
        print("CubeDetectorRlAgent init done")
        time.sleep(1)
        self.agent.set_running_mode("eval")

    def get_observation(self, image):
        print("get_observation...")
        boxPreds, classPreds = self.cube_detector_model.predict_image(image)
        # print("cube detector prediction:", boxPreds, classPreds)
        if classPreds[0][0] < classPreds[0][1]:
            print("CUBE DETECTED: ", classPreds[0], boxPreds[0])
        else:
            print(boxPreds[0])
        cube_detector_output = torch.cat((boxPreds, classPreds), dim=1)
        self.cube_detector_history = torch.roll(self.cube_detector_history, shifts=-self.cube_detector_neurons, dims=1)
        self.cube_detector_history[:, -self.cube_detector_neurons:] = cube_detector_output
        # Note: robot history just zeros for now, also zeros in simulation

        obs = torch.cat((self.cube_detector_history, self.action_history, self.robot_history), dim=1)

        return obs, boxPreds, classPreds

    def get_action(self, observation):
        print("get_action...")
        action = self.agent.act(observation, 500, 10000)[0]
        self.action_history = torch.roll(self.action_history, shifts=-self.num_actions, dims=1)
        self.action_history[:, -self.num_actions:] = action
        return action


# TODO: check how good cube detector is without driving
# TODO: clean up code!
# TODO: add robot position and orientation
# TODO: check history stuff
# TODO: maybe use other model?


class RosAgent:
    def __init__(self, cd_agent):
        print("RosAgent init")
        rospy.init_node('RosAgent')
        rospy.on_shutdown(self.shutdown)
        temp_twist = Twist()
        temp_twist.linear.x = 0.0
        self.global_vel_pub = rospy.Publisher('/jetauto_controller/cmd_vel', Twist,
                                         queue_size=10)  # for some reason needs to be global...

        self.global_vel_pub.publish(temp_twist)
        time.sleep(3)

        self.cd_agent = cd_agent
        rospy.Subscriber('camera/rgb/image_raw', Image, self.image_callback_guard)

        # TODO: subscribe to robot position and orientation

        self.image_callback_running = False
        self.shutting_down = False

        self.image_counter = 0
        self.update_action_after_images_count = 10

    def image_callback_guard(self, msg):
        if self.shutting_down:
            return
        if not self.image_callback_running:
            self.image_callback(msg)

    def image_callback(self, msg):
        if self.image_callback_running:
            return
        self.image_counter += 1
        start_time = time.time()
        self.image_callback_running = True
        print("")
        print(f'image received, width: {msg.width}, height: {msg.height}, encoding: {msg.encoding}, step: {msg.step}')
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        cv2.imwrite(
            "/home/phil/university/thesis/thesis-ros-jetauto/local_ws/src/fourwis_cubechaser_sb3/scripts/cube_detector/" + "test.jpg",
            cv_image)
        print("image size:", cv_image.shape)
        cv_image_resized = cv2.resize(cv_image, (self.cd_agent.cube_detector_model.image_target_width, self.cd_agent.cube_detector_model.image_target_height))
        cv_image_resized = cv2.rotate(cv_image_resized, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # print("image size:", cv_image_resized.shape)


        # TODO: image rotation correct?
        observation, boxPreds, classPreds = self.cd_agent.get_observation(pil_image.fromarray(cv2.cvtColor(cv_image_resized, cv2.COLOR_BGR2RGB)))
        # save image to drive
        startX = int(boxPreds[0][0] * self.cd_agent.cube_detector_model.image_target_width)
        startY = int(boxPreds[0][1] * self.cd_agent.cube_detector_model.image_target_height)
        endX = int(boxPreds[0][2] * self.cd_agent.cube_detector_model.image_target_width)
        endY = int(boxPreds[0][3] * self.cd_agent.cube_detector_model.image_target_height)
        cv2.rectangle(cv_image_resized, (startX, startY), (endX, endY), (0, 255, 0), 2)
        # cv2.imwrite(
        #     "/home/phil/university/thesis/thesis-ros-jetauto/local_ws/src/fourwis_cubechaser_sb3/scripts/cube_detector/sample_images/" + "real_robot_cam_" + str(self.image_counter) + ".png",
        #     cv_image_resized)

        action = self.cd_agent.get_action(observation)
        self.send_action(action)
        rospy.sleep(2)  #TODO: remove, results in old images being displayed...
        self.image_callback_running = False
        print(f"image callback took: {time.time() - start_time}")

    def send_action(self, action):
        print(f'sending action: {action}')
        return
        if self.image_counter % self.update_action_after_images_count == 0:
            twist_msg = Twist()
            twist_msg.linear.x = action[0][0].item() * 0.1
            twist_msg.angular.z = action[0][1].item() * 0.1
            self.global_vel_pub.publish(twist_msg)

    def shutdown(self):
        print("shutting down")
        self.shutting_down = True
        rospy.sleep(1)
        self.global_vel_pub.publish(Twist())
        rospy.sleep(1)


# if __name__ == "__main__":
print("starting cube detector rl agent")
device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
cd_agent = CubeDetectorRlAgent(device)

# test_observation = cd_agent.get_observation(test_image)
# exit()

ros_agent = RosAgent(cd_agent)


print("cube detector rl agent started")
# rospy.init_node('hoge')
# global_vel_pub2 = rospy.Publisher('/jetauto_controller/cmd_vel', Twist, queue_size=10)
# temp_twist = Twist()
# temp_twist.linear.x = 0.1
rate = rospy.Rate(1)

while not rospy.is_shutdown():
    # global_vel_pub2.publish(temp_twist)
    rate.sleep()
