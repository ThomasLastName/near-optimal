
import numpy as np
from matplotlib import pyplot as plt

univar_grid = np.linspace(0,1,15)
start = np.array([-7.1, -7.1])
end = np.array([5.,4.])
bivar_grid = start + np.outer( univar_grid, (end-start) )
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(bivar_grid[:,0], bivar_grid[:,1], color="red", linewidth=2)
ax.set_xlim([-3,3])
ax.set_ylim([-3,3])
plt.tight_layout()
plt.show()
