import numpy as np

from gym.envs.robotics import rotations, fetch_env, utils

class FetchTouchEnv(fetch_env.FetchEnv):

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type, touch_mode='binary', touch_visualisation='on_touch',
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense

        Additional Args for touch sensors:
            touch_mode (string) : representation of touch sensor readings
                - raw: uses values directly from mujoco
                - binary: 1 if touch occurred, 0 otherwise
            touch_visualisation (string): how touch sensor sites are visualised. default is no visualisation
                - always: always show sensor sites
                - on_touch: only show sensor sites with readings > 0
                - never: never shows sensors regeardless of their measurements
        """


        # binary or raw
        self.touch_mode = touch_mode
        self.touch_visualisation = touch_visualisation

        # touch sensor mappings
        self._tsensor_id2name = {}
        self._tsensor_name2id = {}
        self._tsensor_id2siteid = {}

        # dict for inital rgba values for debugging
        self._site_id2intial_rgba = {}

        fetch_env.FetchEnv.__init__(self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type)

        # get touch sensor ids and their site names
        for k, v in self.sim.model._sensor_id2name.items():
            if 'TS' in v:
                self._tsensor_id2name[k] = v
                self._tsensor_name2id[v] = k
                self._tsensor_id2siteid[k] = self.sim.model._site_name2id[v.replace('TS', 'T')]

                # get intial rgba values
                self._site_id2intial_rgba[self._tsensor_id2siteid[k]] = self.sim.model.site_rgba[
                    self._tsensor_id2siteid[k]].copy()


    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        if self.has_object:
            object_pos = self.sim.data.get_site_xpos('object0')
            # rotations
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
            # velocities
            object_velp = self.sim.data.get_site_xvelp('object0') * dt
            object_velr = self.sim.data.get_site_xvelr('object0') * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())

        # get touch sensor readings based on chosen reading type
        if self.touch_mode == 'raw':
            touch_values = [self.sim.data.sensordata[k] for k, v in self._tsensor_id2name.items()]

        else:
            touch_values = [1 if self.sim.data.sensordata[k] != 0.0 else 0 for k, v in
                            self._tsensor_id2name.items()]

        # set rgba values
        if self.touch_visualisation == 'always':
            for k, v in self._tsensor_id2name.items():
                self.sim.model.site_rgba[self._tsensor_id2siteid[k]] = self._site_id2intial_rgba[
                    self._tsensor_id2siteid[k]].copy()

        elif self.touch_visualisation == 'on_touch':
            for k, v in self._tsensor_id2name.items():
                if self.sim.data.sensordata[k] != 0.0:
                    self.sim.model.site_rgba[self._tsensor_id2siteid[k]] = self._site_id2intial_rgba[
                        self._tsensor_id2siteid[k]].copy()
                else:
                    self.sim.model.site_rgba[self._tsensor_id2siteid[k]] = [0, 0, 0, 0]
        else:
            for k, v in self._tsensor_id2name.items():
                self.sim.model.site_rgba[self._tsensor_id2siteid[k]] = [0, 0, 0, 0]


        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel, touch_values,
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }
