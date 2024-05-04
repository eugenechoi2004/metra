from modal import Image
from metra import Metra

args = {
    "latent_low": -1,
    "latent_high": 1,
    "latent_dim": 2,
    "n_epochs": 1000,
    "batch_size": 256,
    "env_name": "Ant-v4",
    "lamb": 30.0,
    "lr":0.0001,
    "grad_steps_per_epoch":50,
    "minibatch_size":256,
    "epsilon":0.001
}

metra = Metra(**args)
metra.train()