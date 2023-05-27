import numpy as np
import matplotlib.pyplot as plt

model_history = np.load('base_history_of_3000ep.npy', allow_pickle='TRUE').item()

plot_it = 0
for key in model_history:
    if "val" in key:
        fig, axs = plt.subplots(1, 1, figsize=(9, 4), sharey=True)

        axs.set_ylabel(key[4:])
        axs.set_xlabel("epoch")

        axs.plot([x for x in range(0, 3000)], model_history[key])
        fig.suptitle(key)
        fig.savefig(key + "_results.png")