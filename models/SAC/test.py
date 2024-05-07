from envs.custom_dmc_tasks import dmc
from envs.custom_dmc_tasks.pixel_wrappers import RenderWrapper
import matplotlib.pyplot as plt

env = dmc.make('quadruped_run_forward_color', obs_type='states', frame_stack=1, action_repeat=2, seed=0)
env = RenderWrapper(env)
env.reset()
trajectories = {'env_infos': []}
for _ in range(100):
    obs, reward, done, extra = env.step(env.action_space.sample(), render=True)
    trajectories['env_infos'].append(extra)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
env.render_trajectories(trajectories, "red", [-25, 25, -25, 25], ax)
plt.show()