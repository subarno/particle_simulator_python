import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv
import json
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import Particle,Collider



class DataLogger:
    def __init__(self, filename="simulation_data.csv", log_format="csv"):
        """
        Initializes the DataLogger to save simulation data.

        :param filename: The name of the output file.
        :param log_format: The format to save the data, either 'csv' or 'json'.
        """
        self.filename = filename
        self.log_format = log_format
        self.data = []

        # Create a new file or append if it already exists
        if log_format == "csv":
            self.headers = ["Time", "Event_Type", "Particle_Type", "Particle_ID", "X", "Y", "Z", "VX", "VY", "VZ",
                            "Energy", "Involved_Particles"]
            self._initialize_csv()
        elif log_format == "json":
            self._initialize_json()
        else:
            raise ValueError("Log format must be either 'csv' or 'json'.")

    def _initialize_csv(self):
        """Initialize CSV file with headers if not already present."""
        if not os.path.isfile(self.filename):
            with open(self.filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(self.headers)

    def _initialize_json(self):
        """Initialize JSON file with an empty list if not already present."""
        if not os.path.isfile(self.filename):
            with open(self.filename, mode='w') as file:
                json.dump([], file)

    def log_collision(self, time, particle1, particle2):
        """Log collision event."""
        event = {
            "Time": time,
            "Event_Type": "Collision",
            "Particle_Type": particle1.type,
            "Particle_ID": f"{particle1.type}_{id(particle1)}",
            "X": particle1.position[0],
            "Y": particle1.position[1],
            "Z": particle1.position[2],
            "VX": particle1.velocity[0],
            "VY": particle1.velocity[1],
            "VZ": particle1.velocity[2],
            "Energy": particle1.energy,
            "Involved_Particles": [f"{particle1.type}_{id(particle1)}", f"{particle2.type}_{id(particle2)}"]
        }
        self._save_event(event)

    def log_decay(self, time, particle):
        """Log particle decay event."""
        event = {
            "Time": time,
            "Event_Type": "Decay",
            "Particle_Type": particle.type,
            "Particle_ID": f"{particle.type}_{id(particle)}",
            "X": particle.position[0],
            "Y": particle.position[1],
            "Z": particle.position[2],
            "VX": particle.velocity[0],
            "VY": particle.velocity[1],
            "VZ": particle.velocity[2],
            "Energy": particle.energy,
            "Involved_Particles": [f"{particle.type}_{id(particle)}"]
        }
        self._save_event(event)

    def log_fusion(self, time, fusion_particle, involved_particles):
        """Log fusion event."""
        event = {
            "Time": time,
            "Event_Type": "Fusion",
            "Particle_Type": fusion_particle.type,
            "Particle_ID": f"{fusion_particle.type}_{id(fusion_particle)}",
            "X": fusion_particle.position[0],
            "Y": fusion_particle.position[1],
            "Z": fusion_particle.position[2],
            "VX": fusion_particle.velocity[0],
            "VY": fusion_particle.velocity[1],
            "VZ": fusion_particle.velocity[2],
            "Energy": fusion_particle.energy,
            "Involved_Particles": [f"{p.type}_{id(p)}" for p in involved_particles]
        }
        self._save_event(event)

    def _save_event(self, event):
        """Save event in the appropriate format (CSV or JSON)."""
        if self.log_format == "csv":
            self.data.append(event)
            with open(self.filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([event["Time"], event["Event_Type"], event["Particle_Type"], event["Particle_ID"],
                                 event["X"], event["Y"], event["Z"], event["VX"], event["VY"], event["VZ"],
                                 event["Energy"], ";".join(event["Involved_Particles"])])

        elif self.log_format == "json":
            with open(self.filename, mode='r') as file:
                data = json.load(file)
            data.append(event)
            with open(self.filename, mode='w') as file:
                json.dump(data, file)

    def export_data(self):
        """Export collected data (CSV or JSON)."""
        return self.data

    def preprocess_data(self,df):
        # Ensure that 'Energy' is numeric
        df['Energy'] = pd.to_numeric(df['Energy'], errors='coerce')

        # Convert 'Involved_Particles' into a count of involved particles
        df['Involved_Particles_Count'] = df['Involved_Particles'].apply(lambda x: len(x.split(';')))

        # Selecting features for prediction (we'll use Energy and Involved_Particles_Count)
        X = df[['Energy', 'Involved_Particles_Count']]

        # Target variable: Particle_Type (what we want to predict)
        y = df['Particle_Type']

        return X, y
