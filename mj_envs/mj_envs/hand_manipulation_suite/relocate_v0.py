import numpy as np
from gym import utils
from mjrl.envs import mujoco_env
from mujoco_py import MjViewer
import os

ADD_BONUS_REWARDS = True

class RelocateEnvV0(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.target_obj_sid = 0
        self.S_grasp_sid = 0
        self.obj_bid = 0
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        self.init_palm_pos = np.array([0., 0., 0.], dtype=np.float64)
        self.init_obj_pos = np.array([0., 0., 0.], dtype=np.float64)
        mujoco_env.MujocoEnv.__init__(self, curr_dir+'/assets/DAPG_relocate.xml', 5)
        
        # change actuator sensitivity
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([10, 0, 0])
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([1, 0, 0])
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([0, -10, 0])
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([0, -1, 0])

        self.target_obj_sid = self.sim.model.site_name2id("target")
        self.S_grasp_sid = self.sim.model.site_name2id('S_grasp')
        self.obj_bid = self.sim.model.body_name2id('Object')
        utils.EzPickle.__init__(self)
        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5*(self.model.actuator_ctrlrange[:,1]-self.model.actuator_ctrlrange[:,0])
        self.action_space.high = np.ones_like(self.model.actuator_ctrlrange[:,1])
        self.action_space.low  = -1.0 * np.ones_like(self.model.actuator_ctrlrange[:,0])

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        try:
            a = self.act_mid + a*self.act_rng # mean center and scale
        except:
            a = a                             # only for the initialization phase
        self.do_simulation(a, self.frame_skip)
        ob = self.get_obs()
        obj_pos  = self.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()

        reward = -0.1*np.linalg.norm(palm_pos-obj_pos)              # take hand to object
        constraint_violation = 0
        if obj_pos[2] > 0.04:                                       # if object off the table
            reward += 1.0                                           # bonus for lifting the object
            reward += -0.5*np.linalg.norm(palm_pos-target_pos)      # make hand go to target
            reward += -0.5*np.linalg.norm(obj_pos-target_pos)       # make object go to target
        else:
            # print('Object not moved')
            # Add negative reward for constraint violation

            # out of radial axis violation
            radial_axis_penalty_distance = 0.03 # 0.01, 0.0001
            # print(obj_pos, self.init_palm_pos, palm_pos)
            dynamic_penalty = False  # SET THIS TO TRUE FOR DYNAMIC PENALTY, FALSE FOR FIXED PENALTY
            dynamic_penalty_scale = 100
            cost_penalty = 0.01

            numerator = np.linalg.norm(np.cross(obj_pos - self.init_palm_pos, self.init_palm_pos - palm_pos))
            denominator = np.linalg.norm(obj_pos - self.init_palm_pos)
            r = numerator / (denominator + 1e-20)
            
            # print(numerator, denominator)
            if r > radial_axis_penalty_distance:
                if not dynamic_penalty:
                    constraint_violation += cost_penalty
                    # reward -= cost_penalty
                else:
                    constraint_violation += ((r - radial_axis_penalty_distance) * dynamic_penalty_scale)
                    # reward -= ((r - radial_axis_penalty_distance) * dynamic_penalty_scale)

            # out of linear axis violation
            numerator = np.dot(palm_pos - self.init_palm_pos, obj_pos - self.init_palm_pos)
            l = numerator / (denominator + 1e-20)

            # print(numerator)
            if l > 1.1 or l < -0.1:
                # distance from cylinder ends
                if l > 1:
                    l -= 1
                if l < 0:
                    l = -l

                # calculate penalty
                if not dynamic_penalty:
                    constraint_violation += cost_penalty
                    # reward -= cost_penalty
                else:
                    constraint_violation += l * denominator * dynamic_penalty_scale
                    # reward -= l * denominator * dynamic_penalty_scale

        if ADD_BONUS_REWARDS:
            if np.linalg.norm(obj_pos-target_pos) < 0.1:
                reward += 10.0                                          # bonus for object close to target
            if np.linalg.norm(obj_pos-target_pos) < 0.05:
                reward += 20.0                                          # bonus for object "very" close to target

        goal_achieved = True if np.linalg.norm(obj_pos-target_pos) < 0.1 else False

        return ob, reward, False, dict(goal_achieved=goal_achieved, constraint_violation=constraint_violation)

    def get_obs(self):
        # qpos for hand
        # xpos for obj
        # xpos for target
        qp = self.data.qpos.ravel()
        obj_pos  = self.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()
        return np.concatenate([qp[:-6], palm_pos-obj_pos, palm_pos-target_pos, obj_pos-target_pos])
       
    def reset_model(self):
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)
        self.model.body_pos[self.obj_bid,0] = self.np_random.uniform(low=-0.15, high=0.15)
        self.model.body_pos[self.obj_bid,1] = self.np_random.uniform(low=-0.15, high=0.3)
        self.model.site_pos[self.target_obj_sid, 0] = self.np_random.uniform(low=-0.2, high=0.2)
        self.model.site_pos[self.target_obj_sid,1] = self.np_random.uniform(low=-0.2, high=0.2)
        self.model.site_pos[self.target_obj_sid,2] = self.np_random.uniform(low=0.15, high=0.35)
        self.sim.forward()
        self.init_palm_pos = self.data.site_xpos[self.S_grasp_sid].copy().ravel()
        self.init_obj_pos = self.data.body_xpos[self.obj_bid].copy().ravel()
        return self.get_obs()

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qp = self.data.qpos.ravel().copy()
        qv = self.data.qvel.ravel().copy()
        hand_qpos = qp[:30]
        obj_pos  = self.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()
        return dict(hand_qpos=hand_qpos, obj_pos=obj_pos, target_pos=target_pos, palm_pos=palm_pos,
            qpos=qp, qvel=qv)

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        obj_pos = state_dict['obj_pos']
        target_pos = state_dict['target_pos']
        self.set_state(qp, qv)
        self.model.body_pos[self.obj_bid] = obj_pos
        self.model.site_pos[self.target_obj_sid] = target_pos
        self.sim.forward()

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.azimuth = 90
        self.sim.forward()
        self.viewer.cam.distance = 1.5

    def evaluate_success(self, paths):
        num_success = 0
        num_paths = len(paths)
        total_cost = 0
        # success if object close to target for 25 steps
        for path in paths:
            if np.sum(path['env_infos']['goal_achieved']) > 25:
                num_success += 1
            total_cost += np.sum(path['env_infos']['constraint_violation'])
        success_percentage = num_success*100.0/num_paths
        average_cost = total_cost/num_paths
        return success_percentage, average_cost
