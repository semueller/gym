#!/usr/bin/env python3
from mujoco_py import load_model_from_path, MjSim, MjViewer
from gym.utils import seeding
from collections import OrderedDict
import math
import os
import numpy as np

np_random, seed = seeding.np_random(0)

def quat_from_angle_and_axis(angle, axis):
    assert axis.shape == (3,)
    axis /= np.linalg.norm(axis)
    quat = np.concatenate([[np.cos(angle / 2.)], np.sin(angle / 2.) * axis])
    quat /= np.linalg.norm(quat)
    return quat


def _sample_goal(target_position_range):


    # Select a goal for the object position.
    target_pos = None
    assert target_position_range.shape == (3, 2)
    offset = np_random.uniform(target_position_range[:, 0], target_position_range[:, 1])
    assert offset.shape == (3,)
    target_pos = sim.data.get_joint_qpos('object:joint')[:3] + offset

    assert target_pos is not None
    assert target_pos.shape == (3,)

    # Select a goal for the object rotation.
    target_quat = None

    angle = np_random.uniform(-np.pi, np.pi)
    axis = np_random.uniform(-1., 1., size=3)
    target_quat = quat_from_angle_and_axis(angle, axis)

    assert target_quat is not None
    assert target_quat.shape == (4,)

    target_quat /= np.linalg.norm(target_quat)  # normalized quaternion
    goal = np.concatenate([target_pos, target_quat])
    return goal

# model = load_model_from_path("/home/llach/projects/gym/gym/envs/robotics/assets/force_gripper/pick_and_place.xml")
model = load_model_from_path("/home/llach/projects/gym/gym/envs/robotics/assets/hand/manipulate_pen.xml")
sim = MjSim(model)
viewer = MjViewer(sim)
t = 0

range = np.array([(-0.02, 0.02), (-0.04, 0.04), (-0.04, 0.04)])

pos = _sample_goal(range)

while True:
    # sim.data.ctrl[0] = math.sin(t)
    # sim.data.ctrl[1] = math.cos(t)


    if t % 100 == 0:
        pos = _sample_goal(range)

    sim.data.set_joint_qpos('target:joint', pos)
    t += 1
    sim.step()
    viewer.render()

