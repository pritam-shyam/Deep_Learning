import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, runtime=5., target_pos=None, init_pose=np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0]),         init_velocities=np.array([0.0, 0.0, 0.0]), init_angle_velocities=np.array([0.0, 0.0, 0.0]), pos_noise=None, vel_noise=None, ang_noise=None, ang_vel_noise=None):
        
        self.action_low = 400
        self.action_high = 420
        self.action_size = 1   
        self.pos_noise = 0.25
        self.vel_noise = 0.15
        
        self.target_pos = target_pos
        self.ang_noise = ang_noise
        self.ang_vel_noise = ang_vel_noise
        
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.state_size = len(self.get_state())
        self.action_b = (self.action_high+self.action_low)/2.0
        self.action_m = (self.action_high-self.action_low)/2.0

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward_loss = (self.sim.pose[2] - self.target_pos[2]) ** 2
        reward_loss += 0.1 * (self.sim.linear_accel[2] ** 2)
        reward_out = np.maximum(1 - 0.5 * 0.5 * (np.sqrt(1 + (reward_loss / 0.5) ** 2) - 1), 0)
        reward = np.clip(reward_out, -1, 1)
        return reward
    
    def get_state(self):
        """Uses position error and target to get current state."""
        error = self.sim.pose[:3] - self.target_pos
        state = np.array([error[2], self.sim.v[2], self.sim.linear_accel[2]])
        return state
        
    def get_rotor_speed(self, action):
        """Gets the rotor speeds for all four axes using numpy."""
        action_convert = (action * self.action_m) + self.action_b
        rotor_speeds = action_convert * np.ones(4)
        return rotor_speeds

    def step(self, action):
        """Uses action to obtain next state, reward, done."""
        done = self.sim.next_timestep(self.get_rotor_speed(action))
        reward = self.get_reward()
        next_state = self.get_state()
        if reward <= 0:
            done = True
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        if self.action_size == 1:
            new_random_pose = np.copy(self.sim.init_pose)
            if self.pos_noise > 0:
                new_random_pose[2] += np.random.normal(0.0, self.pos_noise, 1)
            self.sim.pose = np.copy(new_random_pose)            
        return self.get_state()