import numpy as np

class MultiGoalTask():
    def __init__(self, env, seed) -> None:
        np.random.seed(seed)
        self.env = env
        self.steps = np.zeros(4)
        self.goals = np.zeros((4, 2), dtype=np.float64)
        self.goal_reward = 2.5
    
    def generate_goal(self, x, y, idx):
        goal_x = np.random.uniform(low=x-7.5, high=x+7.5, dtype=np.float64)
        goal_y = np.random.uniform(low=y-7.5, high=y+7.5, dtype=np.float64)
        self.goals[idx] = np.array([goal_x, goal_y])
    
    def step(self, action):
        obs, _, done, info = self.env.step(action)
        self.steps += 1
        pos = np.array(obs[-2:], dtype=np.float64)
        dist = np.linalg.norm(pos - self.goals, axis=1)
        reward = np.sum(dist < 3.0) * self.goal_reward
        self.steps[dist < 3.0] = 0
        self.steps[self.steps == 50] = 0
        for i in np.nonzero(self.steps == 0):
            self.generate_goal(pos[0], pos[1], i)
        return obs, reward, done, info
