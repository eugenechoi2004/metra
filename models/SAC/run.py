from metra import Metra
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

args = {
    "latent_low": -1,
    "latent_high": 1,
    "latent_dim": 2,
    "n_epochs": 1_000,
    "batch_size": 256,
    "env_name": "Ant-v4",
    "lamb": 30.0,
    "lr":0.0001,
    "grad_steps_per_epoch":50,
    "minibatch_size":256,
    "epsilon":0.001,
    "checkpoint_epoch":50,
}
metra = Metra(**args)
metra.train()

# metra.run_loaded_model(78.6042)

trajectories = metra.locomotion_metric()
trajectories_np = np.array(trajectories)

cmap = ListedColormap(['white', 'blue'])

# Plot the trajectories matrix
plt.figure(figsize=(8, 8))
plt.imshow(trajectories_np, cmap=cmap, interpolation='nearest')
plt.colorbar() 
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Agent Trajectories')
plt.grid(False)
plt.show()
