import unittest
from Particle import Particle
from Collider import Collider
import numpy as np
from unittest.mock import patch
from Simulation import Simulation,SimulationGUI
import tkinter as tk
from tkinter import messagebox
import threading
import random
from DataLogger import DataLogger
import pandas as pd
from unittest.mock import patch, MagicMock
from Visualizer import Visualizer


class TestCollider(unittest.TestCase):
    def setUp(self):
        self.collision_threshold = 1e12
        self.particle1 = Particle(position=np.array([0.0, 0.0, 0.0]), velocity=np.array([1.0, 0.0, 0.0]), mass=1.0,
                                  charge=1.0, particle_type="Proton")
        self.particle2 = Particle(position=np.array([1.0, 0.0, 0.0]), velocity=np.array([0.0, 1.0, 0.0]), mass=1.0,
                                  charge=-1.0, particle_type="Electron")

        self.particles = [self.particle1, self.particle2]
        self.collider = Collider(self.particles)
        self.collider = Collider(self.particles)

    def test_distance_to(self):
        distance = np.linalg.norm(self.particle1.position - self.particle2.position)
        self.assertEqual(distance, 1.0)

    def test_check_collision(self):
        result = (self.particle1, self.particle2)
        self.assertTrue(result, "Collision not")

    def test_momentum_conservation(self):
        initial_momentum_p1 = self.particle1.calculate_momentum()
        initial_momentum_p2=self.particle2.calculate_momentum()
        initial_momentum =  initial_momentum_p1 + initial_momentum_p2
        lastvelo_p1, lastvelo_p2 = self.collider.apply_momentum_conservation(self.particle1, self.particle2)
        last_momentym = lastvelo_p1 * self.particle1.mass + lastvelo_p2 * self.particle2.mass
        self.assertTrue(np.allclose(initial_momentum, last_momentym), "Momentum is conserved")

    def test_energy_conservation(self):
        initial_energy_p1 = self.particle1.total_energy()
        initial_energy_p2 = self.particle2.total_energy()
        self.collider.apply_momentum_conservation(self.particle1, self.particle2)
        final_energy_p1 = self.particle1.total_energy()
        final_energy_p2 = self.particle2.total_energy()
        total_initial_energy = initial_energy_p1 + initial_energy_p2
        total_final_energy = final_energy_p1 + final_energy_p2
        self.assertTrue(np.allclose(total_initial_energy, total_final_energy), msg="Energy not conserved.")

    def test_perform_fusion(self):
        fused_particle = self.collider.perform_fusion(self.particle1, self.particle2,
                                                      total_initial_momentum=self.particle1.mass * self.particle1.velocity + self.particle2.mass * self.particle2.velocity,
                                                      total_charge=self.particle1.charge + self.particle2.charge)

        self.assertEqual(len(fused_particle), 1)
        self.assertGreater(fused_particle[0].mass, 0, "Fusion did not create a valid particle.")

    def test_split_particles(self):
        total_mass_before_split = self.particle1.mass

        # Split the particle
        new_particles = self.collider.split_particles(self.particle1, self.particle1.charge)

        total_mass_after_split = sum([p.mass for p in new_particles]) + self.particle1.mass

        self.assertEqual(total_mass_before_split, total_mass_after_split, msg="Mass is not conserved during split.")

    def test_handle_collision(self):
        particles_after_collision = self.collider.handle_collision(self.particle1, self.particle2)

        # Check if new particles were created (for fusion or split)
        self.assertTrue(len(particles_after_collision) >= 2, "Collision did not result in new particles.")


class TestParticle(unittest.TestCase):
    def setUp(self):
        self.collision_threshold = 1e12
        self.particle1 = Particle(position=np.array([0.0, 0.0, 0.0]), velocity=np.array([1.0, 0.0, 0.0]), mass=1.0,
                                  charge=1.0, particle_type="Proton")
        self.particle2 = Particle(position=np.array([1.0, 0.0, 0.0]), velocity=np.array([0.0, 1.0, 0.0]), mass=1.0,
                                  charge=-1.0, particle_type="Electron")

    def test_detect_delay(self):
        particle = Particle(position=[0, 0, 0], velocity=[1, 1, 1], mass=1.0, charge=1, particle_type="proton")
        particle.energy = 2000


        with patch("numpy.random.random", return_value=0.0):  # This makes the random value always <= decay_probability
            decay_products = particle.detect_decay()
            self.assertGreater(len(decay_products), 0, "Decay should occur when the energy is high enough")


        particle.energy = 500  # This is below the threshold, so decay should not happen


        with patch("numpy.random.random", return_value=1.0):
            decay_products = particle.detect_decay()
            self.assertEqual(len(decay_products), 0, "Decay should not happen when the energy is low")

def test_calculate_momentum(self):
    expected_momentum = self.particle1 * np.linalg.norm(self.particle1.velocity)
    self.assertEqual(self.particle1.calculate_momentum(), expected_momentum, "Momentum calculation is incorrect")

class TestSimulation(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.simulation =MagicMock(Simulation)
        self.gui = SimulationGUI(self.root, self.simulation)
        self.gui.start_simulation = MagicMock(return_value=True)
        self.gui.pause_simulation = MagicMock(return_value=True)
        self.gui.reset_simulation = MagicMock(return_value=True)
        self.gui.resume_simulation = MagicMock(return_value=True)
    def test_start_simulation(self):
        self.assertTrue(self.gui.start_simulation(), "Simulation started")
    def test_pause_simulation(self):
        self.assertTrue(self.gui.pause_simulation(), "Simulation paused")
    def test_reset_simulation(self):
        self.assertTrue(self.gui.reset_simulation(), "Simulation resetted")
    def test_resume_simulation(self):
        self.assertTrue(self.gui.resume_simulation(), "Simulation resumed")

class TestDataLogger(unittest.TestCase):

    def setUp(self):
            self.mock_particle1 = MagicMock()
            self.mock_particle1.type = "electron"
            self.mock_particle1.position = [0, 0, 0]
            self.mock_particle1.velocity = [0, 0, 0]
            self.mock_particle1.energy = 1.0
            self.mock_particle1.id = 1
            self.mock_particle2 = MagicMock()
            self.mock_particle2.type = "proton"
            self.mock_particle2.position = [1, 1, 1]
            self.mock_particle2.velocity = [1, 1, 1]
            self.mock_particle2.energy = 2.0
            self.mock_particle2.id = 2

    def test_export_data(self):

        logger = DataLogger(filename="test.csv", log_format="csv")
        logger.log_collision(time=0.1, particle1=self.mock_particle1, particle2=self.mock_particle2)

        exported_data = logger.export_data()
        self.assertEqual(len(exported_data), 1)
        self.assertEqual(exported_data[0]['Event_Type'], "Collision")

    def test_invalid_log_format(self):

        with self.assertRaises(ValueError):
            DataLogger(filename="test.txt", log_format="txt")

    def test_preprocess_data(self):
        logger = DataLogger(filename="test.csv", log_format="csv")

        # Create sample data
        data = [
            {"Time": 0.1, "Event_Type": "Collision", "Particle_Type": "electron", "Energy": "1.0",
             "Involved_Particles": "electron_1;proton_2"},
            {"Time": 0.2, "Event_Type": "Decay", "Particle_Type": "proton", "Energy": "2.0",
             "Involved_Particles": "proton_2"}
        ]
        df = pd.DataFrame(data)

        # Preprocess the data
        X, y = logger.preprocess_data(df)


        self.assertEqual(X.shape, (2, 2))  # Should have two features: Energy and Involved_Particles_Count
        self.assertEqual(y.shape, (2,))

class TestVisualization(unittest.TestCase):
    def setUp(self):
        self.particle1 = Particle(position=np.array([1, 2, 3]), velocity=np.array([0, 0, 0]), mass=1, charge=1,
                                  particle_type="proton")
        self.particle2 = Particle(position=np.array([-1, -2, -3]), velocity=np.array([0, 0, 0]), mass=1, charge=-1,
                                  particle_type="electron")
        self.particles = [self.particle1, self.particle2]
        self.visualizer = Visualizer(self.particles)
    def test_update_visualization(self):
        self.visualizer.ax.scatter=MagicMock()
        self.visualizer.update(0) # 0 is the frame
        self.visualizer.update(0)

        # Check that scatter was called once
        self.visualizer.ax.scatter.assert_called_once()
if __name__ == '__main__':
    unittest.main()
