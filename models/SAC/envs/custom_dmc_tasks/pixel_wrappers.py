from collections import deque

import akro
import gym
import numpy as np
import matplotlib.pyplot as plt


class RenderWrapper(gym.Wrapper):
    def __init__(
            self,
            env,
    ):
        super().__init__(env)

        if env._domain == 'cheetah':
            l = len(env.physics.model.tex_type)
            for i in range(l):
                if env.physics.model.tex_type[i] == 0:
                    height = env.physics.model.tex_height[i]
                    width = env.physics.model.tex_width[i]
                    s = env.physics.model.tex_adr[i]
                    colors = []
                    for y in range(width):
                        scaled_y = np.clip((y / width - 0.5) * 4 + 0.5, 0, 1)
                        colors.append((np.array(plt.cm.rainbow(scaled_y))[:3] * 255).astype(np.uint8))
                    for x in range(height):
                        for y in range(width):
                            cur_s = s + (x * width + y) * 3
                            env.physics.model.tex_rgb[cur_s:cur_s + 3] = colors[y]
            env.physics.model.mat_texrepeat[:, :] = 1
        else:
            l = len(env.physics.model.tex_type)
            for i in range(l):
                if env.physics.model.tex_type[i] == 0:
                    height = env.physics.model.tex_height[i]
                    width = env.physics.model.tex_width[i]
                    s = env.physics.model.tex_adr[i]
                    for x in range(height):
                        for y in range(width):
                            cur_s = s + (x * width + y) * 3
                            env.physics.model.tex_rgb[cur_s:cur_s + 3] = [int(x / height * 255), int(y / width * 255), 128]
            env.physics.model.mat_texrepeat[:, :] = 1

        self.action_space = self.env.action_space
        self.observation_space = akro.Box(low=-np.inf, high=np.inf, shape=(64, 64, 3))

        self.ob_info = dict(
            type='pixel',
            pixel_shape=(64, 64, 3),
        )

    def _transform(self, obs):
        pixels = self.env.render(mode='rgb_array', width=64, height=64).copy()
        pixels = pixels.flatten()
        return pixels

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self._transform(obs)

    def step(self, action, **kwargs):
        next_obs, reward, done, info = self.env.step(action, **kwargs)
        return self._transform(next_obs), reward, done, info
    
    def loc(self):
        return self.env
