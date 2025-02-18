import numpy as np
import pandas as pd
from pykrige.ok3d import OrdinaryKriging3D
from sklearn.metrics import r2_score

# Sample data: coordinates (x_im, y_im, z_value) and CO2 concentrations
data = {
    'x_im': [10, 20, 30, 40, 50],
    'y_im': [15, 25, 35, 45, 55],
    'z_value': [100, 150, 200, 250, 300],  # Elevation
    'co2_concentration': [400, 420, 430, 410, 415]
}
df = pd.DataFrame(data)

# Define the grid for interpolation
gridx = np.linspace(min(df['x_im']), max(df['x_im']), 50)
gridy = np.linspace(min(df['y_im']), max(df['y_im']), 50)
gridz = np.linspace(min(df['z_value']), max(df['z_value']), 50)

# Perform Ordinary Kriging in 3D
OK3D = OrdinaryKriging3D(
    df['x_im'], df['y_im'], df['z_value'], df['co2_concentration'],
    variogram_model='linear',
    verbose=False,
    enable_plotting=False
)
z, ss = OK3D.execute('grid', gridx, gridy, gridz)

# Map sample points to grid indices
def find_nearest_index(array, value):
    return (np.abs(array - value)).argmin()

interpolated_values = []
for i in range(len(df)):
    x_index = find_nearest_index(gridx, df['x_im'].iloc[i])
    y_index = find_nearest_index(gridy, df['y_im'].iloc[i])
    z_index = find_nearest_index(gridz, df['z_value'].iloc[i])
    interpolated_values.append(z[z_index, y_index, x_index])

# Calculate R-squared score
r2 = r2_score(df['co2_concentration'], interpolated_values)

print(f"R-squared score of the 3D kriging interpolation: {r2}")