import numpy as np
import random
from Particle import Particle


class Collider:
    c = 3.0e8
    collision_threshold = 1e12
    energy_threshold = 100.0
    split_probability = 0.5
    fusion_probability = 0.5
    k_e= 8.9875517873681764e9

    def __init__(self, particles):
        self.particles = particles

    def distance_to(self, p1, p2):
        return np.linalg.norm(p1.position - p2.position)

    def check_collision(self, particle1, particle2):
        # if the distance between two particles is smaller then the thresold limit
        # then it means that two particles are gonna collide
        return self.distance_to(particle1, particle2) < self.collision_threshold

    def apply_momentum_conservation(self, p1, p2):
        if p1.mass != p2.mass:
            v1_final = ((p1.mass - p2.mass) / (p1.mass + p2.mass)) * p1.velocity + (
                    2 * p2.mass / (p1.mass + p2.mass)) * p2.velocity
            v2_final = ((p2.mass - p1.mass) / (p1.mass + p2.mass)) * p2.velocity + (
                    2 * p1.mass / (p1.mass + p2.mass)) * p1.velocity
            p1.velocity, p2.velocity = v1_final, v2_final
        else:
            p1.velocity, p2.velocity = p2.velocity, p1.velocity
        return p1.velocity, p2.velocity

    # conservation of electric charge and energy

    def perform_fusion(self, p1, p2, total_initial_momentum, total_charge):
        fusion_particle_list=[]
        combined_mass = p1.mass + p2.mass
        """Fuse two particles into a single new particle with combined mass and charge."""

        additional_mass = ((p1.total_energy() + p2.total_energy()) - (combined_mass * self.c ** 2)) / self.c ** 2
        new_mass = combined_mass + additional_mass
        # momentum conservation
        new_velocity = total_initial_momentum / new_mass

        fusion_particle = Particle(position=p1.position, velocity=new_velocity, mass=new_mass, charge=total_charge,
                                   particle_type="FusionParticle")
        fusion_particle_list.append(fusion_particle)
        print(f"Fusion created a new particle with mass {new_mass} and charge {total_charge}.")
        return fusion_particle_list

    def split_particles(self, particle,total_charge):
        """Randomly transform particles post-collision based on a probability."""
        transformation_probability = 0.3  # Example probability for transformation
        new_particles = []
        if np.random.rand() < transformation_probability:
            # Example of splitting a particle into two new particles
            new_mass = particle.mass * 0.5
            if new_mass < 1e-30:
                return []

            particle.mass *= 0.5
            new_particle = Particle(position=particle.position, velocity=particle.velocity,
                                    mass=new_mass,charge=total_charge, particle_type="DecayProduct")
            new_particles.append(new_particle)
        return new_particles

    def handle_collision(self, particle1, particle2):
        """Handle the collision by either merging or splitting the particles."""
        total_initial_momentum = particle1.velocity * particle1.mass + particle2.velocity * particle2.mass
        total_initial_energy = particle1.total_energy() + particle2.total_energy()
        total_charge = particle1.charge + particle2.charge
        if self.check_collision(particle1, particle2):
            # Apply momentum conservation
            self.apply_momentum_conservation(particle1, particle2)
            # fussion
            if random.random() < self.fusion_probability:
                # Try fusion
                fused_particle = self.perform_fusion(particle1, particle2, total_initial_momentum, total_charge)
                if fused_particle:
                    return fused_particle
            # split the particles
            if random.random() < self.split_probability:
                new_particles = self.split_particles(particle1,total_charge)
                self.particles.extend(self.split_particles(particle2,total_charge))
                return new_particles

        return [particle1, particle2]  # No collision, return the same particles

    def apply_forces(self):
        """Apply electric forces between all particle pairs."""
        time_step = 0.01
        with multiprocessing.Pool() as pool:
            pool.starmap(self.apply_force_pair, [(p1, p2, time_step) for i, p1 in enumerate(self.particles)
                                                 for p2 in self.particles[i + 1:]])

    def apply_force_pair(self, p1, p2, time_step):
        """Calculate and apply the force between two particles."""
        force = self.calculate_electric_force(p1, p2)
        p1_acceleration = force / p1.mass
        p2_acceleration = -force / p2.mass
        p1.velocity += p1_acceleration * time_step
        p2.velocity += p2_acceleration * time_step

    def calculate_electric_force(self, p1, p2):
        """Calculate the electric force between two charged particles using Coulomb's law."""
        distance_vector = p2.position - p1.position
        distance = np.linalg.norm(distance_vector)
        if distance == 0:
            return np.zeros(3)  # Avoid division by zero
        force_magnitude = self.k_e * (p1.charge * p2.charge) / distance ** 2
        force_direction = distance_vector / distance
        return force_magnitude * force_direction



