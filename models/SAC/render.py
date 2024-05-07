from metra import Metra
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

args = {
    "latent_low": -1,
    "discrete": False,
    "latent_high": 1,
    "latent_dim": 2,
    "n_epochs": 100,
    "batch_size": 256,
    "env_name": "Ant-v4",
    "lamb": 30.0,
    "lr": 0.0001,
    "episode_per_epoch": 8,
    "grad_steps_per_epoch": 50,
    "minibatch_size":256,
    "epsilon":0.0001,
    "checkpoint_epoch":50,
}
metra = Metra(**args)
metra.run_loaded_model(30.2185)
