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
metra.train()

# trajectories = metra.locomotion_metric()
# trajectories_np = np.array(trajectories)

# cmap = ListedColormap(['white', 'blue'])

# # Plot the trajectories matrix
# plt.figure(figsize=(8, 8))
# plt.imshow(trajectories_np, cmap=cmap, interpolation='nearest')
# plt.colorbar() 
# plt.xlabel('X Position')
# plt.ylabel('Y Position')
# plt.title('Agent Trajectories')
# plt.grid(False)
# plt.show()
