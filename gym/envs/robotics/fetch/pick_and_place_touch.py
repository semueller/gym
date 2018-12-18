from gym import utils
from gym.envs.robotics import fetch_touch_env


class FetchTouchPickAndPlaceEnv(fetch_touch_env.FetchTouchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', touch_mode='binary', touch_visualisation='on_touch',):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        fetch_touch_env.FetchTouchEnv.__init__(
            self, 'fetch/pick_and_place_touch.xml', has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type,  touch_mode=touch_mode, touch_visualisation=touch_visualisation,)
        utils.EzPickle.__init__(self)
