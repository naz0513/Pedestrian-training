import numpy as np
import pandas as pd

def retrieveOriginalTrajectory(dataset, timestep, ID, traj_length=10):
    trajectory = []
    df = dataset[dataset['ID'] == ID]
    for i in range(traj_length):
        row = df[df['Time'] == (timestep + i)*0.5]
        trajectory.append([row['Positionx'].values[0], row['Positiony'].values[0]])
    return trajectory