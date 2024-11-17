import numpy as np
import tkinter as tk
from tkinter import messagebox
import threading
import random

from Particle import Particle
from Collider import Collider
import cProfile
class Simulation:
    def __init__(self, num_particles=10, speed=1000000, time_step=0.001, max_time=10, particle_type="proton",
                 energy_ev=1e13):
        self.num_particles = num_particles
        self.speed = speed
        self.time_step = time_step
        self.max_time = max_time
        self.particle_type = particle_type
        self.energy_ev = energy_ev
        self.energy_joules = self.convert_ev_to_joules(energy_ev)
        self.particles = []
        self.time = 0
        self.running = False
        self.simulation_thread = None
        self.create_particles()

    def convert_ev_to_joules(self, energy_ev):
        """ Converts energy in electron volts (eV) to joules (J). """
        return energy_ev * 1.60218e-19

    def create_particles(self):
        """ Creates particles based on the user-defined number, type, speed, and energy. """
        self.particles = [
            Particle(np.random.rand(3) * 100, np.random.rand(3) * self.speed, random.uniform(1e-27, 1e-24),random.choice([-1, 0, 1]) * 1.6e-19,self.particle_type) for _ in range(self.num_particles)]

    def update(self):
        """ Updates the simulation, checking for collisions, particle movement, and decay events. """
        if self.running:
            self.time += self.time_step
            collider = Collider(self.particles)


            for i, p1 in enumerate(self.particles):
                for j, p2 in enumerate(self.particles[i + 1:], start=i + 1):
                    if collider.check_collision(p1, p2):
                        collider.handle_collision(p1, p2)


            for particle in self.particles:
                particle.update_position(self.time_step)
                decay_products = particle.detect_decay()
                if decay_products:
                    self.particles.extend(decay_products)

    def start(self):
        """ Starts the simulation in a new thread. """
        self.running = True
        self.simulation_thread = threading.Thread(target=self.run_simulation)
        self.simulation_thread.start()

    def pause(self):
        """ Pauses the simulation. """
        self.running = False

    def reset(self):
        """ Resets the simulation to its initial state. """
        self.time = 0
        self.running = False
        self.create_particles()

    def resume(self):
        """ Resumes the simulation from where it was paused. """
        if not self.running:
            self.running = True
            self.simulation_thread = threading.Thread(target=self.run_simulation)
            self.simulation_thread.start()

    def run_simulation(self):
        """ Runs the simulation loop until the maximum time is reached or the simulation is stopped. """
        while self.running and self.time < self.max_time:
            self.update()


# GUI for controlling the simulation
class SimulationGUI:
    def __init__(self, root, simulation):
        self.root = root
        self.simulation = simulation
        self.root.title("Particle Collider Simulation")
        self.root.geometry("400x500")

        # Main frame for input controls
        self.form_frame = tk.Frame(root)
        self.form_frame.pack(padx=10, pady=10)

        # Number of particles input field
        self.num_particles_label = tk.Label(self.form_frame, text="Number of Particles:")
        self.num_particles_label.grid(row=0, column=0, pady=5)
        self.num_particles_entry = tk.Entry(self.form_frame, width=30)
        self.num_particles_entry.insert(0, "10")  # Default number of colliding particles
        self.num_particles_entry.grid(row=0, column=1, pady=5)

        # Particle speed input field
        self.speed_label = tk.Label(self.form_frame, text="Particle Speed (m/s):")
        self.speed_label.grid(row=1, column=0, pady=5)
        self.speed_entry = tk.Entry(self.form_frame, width=30)
        self.speed_entry.insert(0, "1000000")  # Default speed
        self.speed_entry.grid(row=1, column=1, pady=5)

        # Time step input field
        self.time_step_label = tk.Label(self.form_frame, text="Time Step (seconds):")
        self.time_step_label.grid(row=2, column=0, pady=5)
        self.time_step_entry = tk.Entry(self.form_frame, width=30)
        self.time_step_entry.insert(0, "0.001")  # Default time step
        self.time_step_entry.grid(row=2, column=1, pady=5)

        # Particle type dropdown menu
        self.particle_type_label = tk.Label(self.form_frame, text="Select Particle Type:")
        self.particle_type_label.grid(row=3, column=0, pady=5)
        self.particle_type_var = tk.StringVar()
        self.particle_type_var.set("proton")  # Default particle type
        self.particle_type_menu = tk.OptionMenu(self.form_frame, self.particle_type_var, "proton", "electron",
                                                "neutron")
        self.particle_type_menu.config(width=20)
        self.particle_type_menu.grid(row=3, column=1, pady=5)

        # Energy of colliding particles input field
        self.energy_label = tk.Label(self.form_frame, text="Energy of Colliding Particles (eV):")
        self.energy_label.grid(row=4, column=0, pady=5)
        self.energy_entry = tk.Entry(self.form_frame, width=30)
        self.energy_entry.insert(0, "1e13")  # Default energy
        self.energy_entry.grid(row=4, column=1, pady=5)

        # Status Label (Simulation Status)
        self.status_label = tk.Label(self.form_frame, text="Status: Not Started", width=40, height=2, relief="sunken")
        self.status_label.grid(row=5, column=0, columnspan=2, pady=10)

        # Control buttons
        self.start_button = tk.Button(self.form_frame, text="Start", width=20, command=self.start_simulation)
        self.start_button.grid(row=6, column=0, columnspan=2, pady=10)

        self.pause_button = tk.Button(self.form_frame, text="Pause", width=20, command=self.pause_simulation)
        self.pause_button.grid(row=7, column=0, columnspan=2, pady=5)

        self.reset_button = tk.Button(self.form_frame, text="Reset", width=20, command=self.reset_simulation)
        self.reset_button.grid(row=8, column=0, columnspan=2, pady=5)

        self.resume_button = tk.Button(self.form_frame, text="Resume", width=20, command=self.resume_simulation)
        self.resume_button.grid(row=9, column=0, columnspan=2, pady=5)

    def start_simulation(self):
        """ Starts the simulation with the user inputs. """
        try:
            num_particles = int(self.num_particles_entry.get())
            speed = float(self.speed_entry.get())
            time_step = float(self.time_step_entry.get())
            particle_type = self.particle_type_var.get()
            energy_ev = float(self.energy_entry.get())
            self.simulation.num_particles = num_particles
            self.simulation.speed = speed
            self.simulation.time_step = time_step
            self.simulation.particle_type = particle_type
            self.simulation.energy_ev = energy_ev
            self.simulation.energy_joules = self.simulation.convert_ev_to_joules(energy_ev)
            self.simulation.create_particles()
            self.simulation.start()

            # Update status label
            self.status_label.config(text="Simulation Status: Running")
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numerical values.")

    def pause_simulation(self):
        """ Pauses the simulation. """
        self.simulation.pause()
        self.status_label.config(text="Simulation Status: Paused")

    def reset_simulation(self):
        """ Resets the simulation to its initial state. """
        self.simulation.reset()
        self.status_label.config(text="Simulation Status: Not Started")

    def resume_simulation(self):
        """ Resumes the simulation from where it was paused. """
        self.simulation.resume()
        self.status_label.config(text="Simulation Status: Running")