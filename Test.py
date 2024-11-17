import time

from Particle import Particle
from Collider import Collider
from Simulation import Simulation, SimulationGUI
from Visualizer import Visualizer
from DataLogger import DataLogger
import tkinter as tk
from tkinter import messagebox
import threading
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import csv
import json
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification


def main():
    # Create random particles for testing
    particle_types = {
        "Proton": {"mass": 1.67e-27, "charge": 1.6e-19},
        "Electron": {"mass": 9.11e-31, "charge": -1.6e-19},
        "Muon": {"mass": 1.88e-28, "charge": -1.6e-19},
        "Neutron": {"mass": 1.67e-27, "charge": 0},
    }
    particles = []
    for i in range(1, 11):
        particle_type = random.choice(list(particle_types.keys()))
        properties = particle_types[particle_type]
        position = [random.uniform(-1e-9, 1e-9) for _ in range(3)]
        velocity = [random.uniform(-1e5, 1e5) for _ in range(3)]

        mass = properties["mass"]
        charge = properties["charge"]

        particle = Particle(position, velocity, mass, charge, particle_type)
        particles.append(particle)

    # Initialize the Collider
    collider = Collider(particles)
    data_logger = DataLogger("collision_data.csv", log_format="csv")
    # Detect collisions
    print("Detecting collisions...")
    start_time = time.time()
    for i in range(len(particles)):
        for j in range(i + 1, len(particles)):
            particle1 = particles[i]
            particle2 = particles[j]
            if collider.check_collision(particle1, particle2):
                print(f"Collision detected between particle {i + 1} and particle {j + 1}")
                data_logger.log_collision(time.time(), particle1, particle2)

    for particle1 in particles:
        data_logger.log_decay(time.time() - start_time, particle1)

    df = pd.read_csv("collision_data.csv")
    X, y = data_logger.preprocess_data(df)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"X_train type: {X_train.dtypes}")
    print(f"y_train type: {y_train.dtype}")
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=1))
    cm = confusion_matrix(y_test, y_pred)
    print(confusion_matrix(y_test, y_pred))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
    plt.show()

    # Step 1: Create a sample dataset with sufficient samples
    # Generating a synthetic classification dataset with more features

    X, y = make_classification(n_samples=100, n_features=5, n_informative=3, n_classes=2, random_state=42)

    # Step 2: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 3: Initialize the Random Forest model
    model = RandomForestClassifier(random_state=42)

    # Step 4: Perform GridSearchCV to tune hyperparameters
    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, 20],
        'min_samples_split': [2, 5, 10],
    }

    # Initialize GridSearchCV with 5-fold cross-validation (works with 100 samples)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

    # Fit the grid search model
    grid_search.fit(X_train, y_train)

    # Step 5: Print the best parameters and best score
    print("Best parameters found: ", grid_search.best_params_)
    print("Best score found: ", grid_search.best_score_)

    # Step 6: Evaluate the model using the test set
    y_pred = grid_search.best_estimator_.predict(X_test)

    # Step 7: Evaluate the model performance
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=1))

    # Step 8: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Step 9: Plot confusion matrix using seaborn
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=grid_search.best_estimator_.classes_,
                yticklabels=grid_search.best_estimator_.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Apply forces
    print("Applying forces...")
    collider.apply_forces()

    # Print particle states
    print("\nFinal particle states:")
    for i, particle in enumerate(collider.particles):
        print(f"Particle {i + 1}:")
        print(f" - Type: {particle.type}")
        print(f" - Position: {particle.position}")
        print(f" - Velocity: {particle.velocity}")
        print(f" - Mass: {particle.mass:.2e} kg")
        print(f" - Charge: {particle.charge:.2e} C")
        print(f" - Total Energy: {particle.total_energy():.2e} J\n")

    # Visualizer Class
    visualizer = Visualizer(particles)
    visualizer.animate()

    # Bottleneck
    # Simulation Starts....
    print("Simulation Starts...")
    root = tk.Tk()
    simulation = Simulation()
    gui = SimulationGUI(root, simulation)
    root.mainloop()


if __name__ == "__main__":
    main()
