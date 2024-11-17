import numpy as np
 
class Particle:
    def _init_(self, position, velocity, mass, particle_type):
        """
        Initializes a particle with position, velocity, mass, and type.
        """
        self.position = np.array(position, dtype=float)  # Position vector [x, y, z]
        self.velocity = np.array(velocity, dtype=float)  # Velocity vector [vx, vy, vz]
        self.mass = mass                                 # Mass of the particle
        self.type = particle_type                        # Particle type (e.g., "proton", "electron")
        self.energy = self.calculate_kinetic_energy()    # Initial energy calculated
 
    def update_position(self, dt):
        """
        Updates the particle’s position based on its velocity and a given time step dt.
        """
        self.position += self.velocity * dt
 
    def calculate_kinetic_energy(self):
        """
        Calculates the kinetic energy of the particle.
        KE = 0.5 * mass * velocity^2
        """
        velocity_magnitude = np.linalg.norm(self.velocity)
        kinetic_energy = 0.5 * self.mass * velocity_magnitude ** 2
        return kinetic_energy
 
    def detect_decay(self):
        """
        Detects if the particle should decay. This can be based on energy or random probability.
        """
        # Basic example: particles with high energy have a chance to decay
        decay_probability = min(1, self.energy / 1000)  # Arbitrary threshold for demonstration
        if np.random.random() < decay_probability:
            print(f"{self.type} particle decayed!")
            return self.decay()
        return None
 
    def decay(self):
        """
        Decay mechanism where the particle may transform into other particles.
        Returns a list of new particles or an empty list if no decay products.
        """
        decay_products = []
        
        # For demonstration, let's assume that a proton decays into two lighter particles
        if self.type == "proton":
            decay_products.append(Particle(self.position, self.velocity / 2, self.mass / 2, "neutron"))
            decay_products.append(Particle(self.position, -self.velocity / 2, self.mass / 2, "positron"))
 
        # Decay can be expanded based on type and probabilities
        return decay_products
 
    def display_info(self):
        """
        Prints particle information: type, mass, position, velocity, and energy.
        """
        print(f"Particle Type: {self.type}")
        print(f"Mass: {self.mass} kg")
        print(f"Position: {self.position}")
        print(f"Velocity: {self.velocity}")
        print(f"Kinetic Energy: {self.energy} J")

 
