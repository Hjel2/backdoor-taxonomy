import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
        x = np.ones((10, 10))
        x[3, 4] = 0
        plt.imshow(1 - x, cmap='Greys')
        plt.xticks([])
        plt.yticks([])
        plt.show()
