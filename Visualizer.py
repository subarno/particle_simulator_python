import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
class Visualizer:
    def __init__(self, particles):
        """
        Initialize the Visualizer with a list of Particle objects.
        Each Particle must have attributes: position (np.array) and type (str).
        """
        self.particles = particles
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

    def update(self, frame):
        """
        Update the visualization for the current frame.
        """
        self.ax.clear()
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.set_zlim(-10, 10)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        # Plot particles
        positions = np.array([p.position for p in self.particles])
        types = [p.type for p in self.particles]

        # Map particle types to colors
        color_map = {"proton": "red", "electron": "blue", "neutron": "green"}
        colors = [color_map.get(ptype, "black") for ptype in types]

        self.ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=colors, s=50)

    def animate(self):
        """
        Run the animation.
        """
        animation = FuncAnimation(self.fig, self.update, frames=100, interval=50)
        plt.show()

