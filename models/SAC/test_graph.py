from metra import Metra
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

args = {
    "latent_low": -1,
    "discrete": True,
    "latent_high": 1,
    "latent_dim": 16,
    "n_epochs": 10_000,
    "batch_size": 256,
    "env_name": "HalfCheetah-v4",
    "lamb": 30.0,
    "lr": 0.0001,
    "episode_per_epoch": 8,
    "grad_steps_per_epoch": 50,
    "minibatch_size": 256,
    "epsilon": 0.001,
    "checkpoint_epoch": 100,
    "encoder": False,
    "pixel": False,
}
metra = Metra(**args)

# metra.run_loaded_model(78.6042)

trajectories, visited_states = metra.locomotion_metric_discrete()
metra.plot_trajectories(trajectories)
plt.figure(figsize=(10, 10))
plt.plot(np.arange(len(visited_states)), visited_states)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.gca().ticklabel_format(useMathText=True)
plt.xlabel('Steps')
plt.ylabel('State Coverage')
plt.title('HalfCheetah', weight='bold')
plt.show()

