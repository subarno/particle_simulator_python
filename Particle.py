import numpy as np
class Particle:
    c = 3.0e8
    def __init__(self, position, velocity, mass, charge,particle_type):
        """
               Initializes a particle with position, velocity, mass, and type.
               """
        self.position = np.array(position, dtype=float)  # Position vector [x, y, z]
        self.velocity = np.array(velocity, dtype=float)  # Velocity vector [vx, vy, vz]
        self.mass = mass  # Mass of the particle
        self.type = particle_type  # Particle type (e.g., "proton", "electron")
        #self.energy = self.calculate_kinetic_energy() # Initial energy calculated
        self.charge = charge
        self.rest_energy = self.calculate_rest_energy()
        self.energy=self.calculate_rest_energy()

    def calculate_kinetic_energy(self):
        """
        Calculates the kinetic energy of the particle.
        KE = 0.5 * mass * velocity^2
        """
        velocity_magnitude = np.linalg.norm(self.velocity)
        kinetic_energy = 0.5 * self.mass * velocity_magnitude ** 2
        return kinetic_energy

    def calculate_rest_energy(self):
        return self.mass * self.c ** 2

    def calculate_momentum(self):
        return self.mass * np.linalg.norm(self.velocity)

    def total_energy(self):
        momentum = self.calculate_momentum()
        return np.sqrt((self.rest_energy) ** 2 + (momentum * self.c) ** 2)

    def update_position(self, dt):
        """
        Updates the particle’s position based on its velocity and a given time step dt.
        """
        self.position += self.velocity * dt

    def detect_decay(self):
        """
        Detects if the particle should decay. This can be based on energy or random probability.
        """
        # Basic example: particles with high energy have a chance to decay
        decay_probability = min(1, self.energy / 1000)  # Arbitrary threshold for demonstration
        if np.random.random() < decay_probability:
            print(f"{self.type} particle decayed!")
            return self.decay()
        return []

    def display_info(self):
        """
        Prints particle information: type, mass, position, velocity, and energy.
        """
        print(f"Particle Type: {self.type},Mass: {self.mass} kg,Position: {self.position},Velocity: {self.velocity},Kinetic Energy: {self.energy} J")

    def decay(self):
        """
        Decay mechanism where the particle may transform into other particles.
        Returns a list of new particles or an empty list if no decay products.
        """
        decay_products = []

        # For demonstration, let's assume that a proton decays into two lighter particles
        if self.type == "proton":
            decay_products.append(Particle(self.position, self.velocity / 2, self.mass / 2, self.charge,"neutron"))
            decay_products.append(Particle(self.position, -self.velocity / 2, self.mass / 2, self.charge,"positron"))

        # Decay can be expanded based on type and probabilities
        return decay_products