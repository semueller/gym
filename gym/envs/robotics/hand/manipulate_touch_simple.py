import numpy as np
import sys

from gym import utils, error
from gym.envs.robotics.hand import manipulate

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))


class ManipulateTouchSimpleEnv(manipulate.ManipulateEnv, utils.EzPickle):
    def __init__(
        self, model_path, target_position, target_rotation,
        target_position_range, reward_type, initial_qpos={},
        randomize_initial_position=True, randomize_initial_rotation=True,
        distance_threshold=0.01, rotation_threshold=0.1, n_substeps=20, relative_control=False,
        ignore_z_target_rotation=False, touch_visualisation='on_touch',
    ):
        """Initializes a new Hand manipulation environment with touch sensors.

         Args:
            model_path (string): path to the environments XML file
            target_position (string): the type of target position:
                - ignore: target position is fully ignored, i.e. the object can be positioned arbitrarily
                - fixed: target position is set to the initial position of the object
                - random: target position is fully randomized according to target_position_range
            target_rotation (string): the type of target rotation:
                - ignore: target rotation is fully ignored, i.e. the object can be rotated arbitrarily
                - fixed: target rotation is set to the initial rotation of the object
                - xyz: fully randomized target rotation around the X, Y and Z axis
                - z: fully randomized target rotation around the Z axis
                - parallel: fully randomized target rotation around Z and axis-aligned rotation around X, Y
            ignore_z_target_rotation (boolean): whether or not the Z axis of the target rotation is ignored
            target_position_range (np.array of shape (3, 2)): range of the target_position randomization
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            randomize_initial_position (boolean): whether or not to randomize the initial position of the object
            randomize_initial_rotation (boolean): whether or not to randomize the initial rotation of the object
            distance_threshold (float, in meters): the threshold after which the position of a goal is considered achieved
            rotation_threshold (float, in radians): the threshold after which the rotation of a goal is considered achieved
            n_substeps (int): number of substeps the simulation runs on every call to step
            relative_control (boolean): whether or not the hand is actuated in absolute joint positions or relative to the current state

        Additional Args for touch sensors:
            touch_visualisation (string): how touch sensor sites are visualised. default is no visualisation
                - always: always show sensor sites
                - on_touch: only show sensor sites with readings > 0
        """

        # vis type
        self.touch_visualisation = touch_visualisation

        # touch sensor mappings
        self._tsensor_id2name = {}
        self._tsensor_name2id = {}
        self._tsensor_id2siteid = {}

        # dict for inital rgba values for debugging
        self._site_id2intial_rgba = {}

        # list of dictionaries of sensors per joint
        self._tsensor_group_names = []

        self._tsensor_prefix = 'robot0:TS_'

        manipulate.ManipulateEnv.__init__(
            self, model_path, target_position, target_rotation,
            target_position_range, reward_type, initial_qpos={},
            randomize_initial_position=True, randomize_initial_rotation=True,
            distance_threshold=0.01, rotation_threshold=0.1, n_substeps=20, relative_control=False,
            ignore_z_target_rotation=False)
        utils.EzPickle.__init__(self)

        # get touch sensor ids and their site names
        for k, v in self.sim.model._sensor_id2name.items():
            if 'TS' in v:
                self._tsensor_id2name[k] = v
                self._tsensor_name2id[v] = k
                self._tsensor_id2siteid[k] = self.sim.model._site_name2id[v.replace('TS', 'T')]

                # get intial rgba values
                self._site_id2intial_rgba[self._tsensor_id2siteid[k]] = self.sim.model.site_rgba[
                    self._tsensor_id2siteid[k]].copy()

        # construct list of dictionaries, each holding a name -> id mapping
        # for each joint on each finger
        for finger in ['th', 'lf', 'rf', 'mf', 'ff']:
            for part in ['tip', 'middle', 'proximal']:
                sensor_group = {}
                if part == 'tip':
                    name = self._tsensor_prefix + finger + part + '_' + 'tip'
                    sensor_group[name] = self._tsensor_name2id[name]
                for suffix in ['front', 'back', 'left', 'right']:
                    name = self._tsensor_prefix + finger + part + '_' + suffix
                    sensor_group[name] = self._tsensor_name2id[name]
                self._tsensor_group_names.append(sensor_group)

        # don't forget the palm
        name = self._tsensor_prefix + 'palm'
        self._tsensor_group_names.append({name: self._tsensor_name2id[name]})

    def _get_obs(self):
        robot_qpos, robot_qvel = manipulate.robot_get_obs(self.sim)
        object_qvel = self.sim.data.get_joint_qvel('object:joint')
        achieved_goal = self._get_achieved_goal().ravel()  # this contains the object position + rotation

        # get touch sensor readings. if there is one, set value to 1
        touch_values = []

        for sensor_group in self._tsensor_group_names:
            value = 0

            for k, v in sensor_group.items():
                value += self.sim.data.sensordata[v]

            value /= len(sensor_group)
            b = 1 if value != 0.0 else 0
            touch_values.append(b)

        # set rgba values
        if self.touch_visualisation == 'always':
            for k, v in self._tsensor_id2name.items():
                self.sim.model.site_rgba[self._tsensor_id2siteid[k]] = self._site_id2intial_rgba[
                        self._tsensor_id2siteid[k]].copy()

        elif self.touch_visualisation == 'on_touch':
            for sensor_group in self._tsensor_group_names:
                activation = False

                for k, v in sensor_group.items():
                    if self.sim.data.sensordata[v] != 0.0:
                        activation = True

                if activation:
                    for k, v in sensor_group.items():
                        self.sim.model.site_rgba[self._tsensor_id2siteid[v]] = self._site_id2intial_rgba[
                            self._tsensor_id2siteid[v]].copy()
                else:
                    for k, v in sensor_group.items():
                        self.sim.model.site_rgba[self._tsensor_id2siteid[v]] = [0, 0, 0, 0]
        else:
            for k, v in self._tsensor_id2name.items():
                self.sim.model.site_rgba[self._tsensor_id2siteid[k]] = [0, 0, 0, 0]

        observation = np.concatenate([robot_qpos, robot_qvel, object_qvel, touch_values, achieved_goal])

        return {
            'observation': observation.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.ravel().copy(),
        }

class HandTouchSimpleBlockEnv(ManipulateTouchSimpleEnv):
    def __init__(self, target_position='random', target_rotation='xyz', reward_type='sparse'):
        super(HandTouchSimpleBlockEnv, self).__init__(
            model_path='hand/manipulate_touch_block.xml', target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type)


class HandTouchSimpleEggEnv(ManipulateTouchSimpleEnv):
    def __init__(self, target_position='random', target_rotation='xyz', reward_type='sparse'):
        super(HandTouchSimpleEggEnv, self).__init__(
            model_path='hand/manipulate_touch_egg.xml', target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type)


class HandTouchSimplePenEnv(ManipulateTouchSimpleEnv):
    def __init__(self, target_position='random', target_rotation='xyz', reward_type='sparse'):
        super(HandTouchSimplePenEnv, self).__init__(
            model_path='hand/manipulate_touch_pen.xml', target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            randomize_initial_rotation=False, reward_type=reward_type,
            ignore_z_target_rotation=True, distance_threshold=0.05)
