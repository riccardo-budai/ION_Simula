import pyfibers

# Enable logging to see 3D fiber creation and simulation progress
pyfibers.enable_logging()

# Creating the spiral path for the fiber
import numpy as np
import matplotlib.pyplot as plt

# Define parameters for the spiral
height = 1000  # total height of the spiral
turns = 5  # number of turns in the spiral
radius = 1000  # radius of the spiral
points_per_turn = 20  # points per turn

# Generate the spiral coordinates
t = np.linspace(0, 2 * np.pi * turns, points_per_turn * turns)
x = radius * np.cos(t)
y = radius * np.sin(t)
z = np.linspace(0, height, points_per_turn * turns)

# Combine into a single array for path coordinates
path_coordinates = np.column_stack((x, y, z))

# Calculate the length along the fiber
fiber_length = np.sqrt(np.sum(np.diff(path_coordinates, axis=0) ** 2, axis=1)).sum()

# Generate the Gaussian curve of potentials
# Set the mean at the center of the fiber and standard deviation as a fraction of the fiber length
mean = fiber_length / 2
std_dev = fiber_length / 50  # Smaller value gives a narrower peak

# Create an array representing the linear distance along the fiber
linear_distance = np.linspace(0, fiber_length, len(path_coordinates))

# Generate Gaussian distributed potentials
potentials = np.exp(-((linear_distance - mean) ** 2 / (2 * std_dev**2)))

# Normalize potentials for better visualization or use
potentials /= np.max(potentials)

# Scale to a maximum of 500 mV
potentials *= 500

# Create subplots for both the 3D path and potentials
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot the generated path to visualize it
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(x, y, z)
ax1.set_title("3D Spiral Path for Fiber")
ax1.set_xlabel("X axis")
ax1.set_ylabel("Y axis")
ax1.set_zlabel("Z axis")

# Plot the Gaussian distribution of potentials along the fiber
ax2.plot(linear_distance, potentials, label='Gaussian Potentials')
ax2.set_xlabel('Distance along fiber (µm)')
ax2.set_ylabel('Potential (mV)')
ax2.set_title('Gaussian Distribution of Potentials along fiber path')
ax2.legend()

plt.tight_layout()
plt.show()

from pyfibers import FiberModel, build_fiber_3d

fiber = build_fiber_3d(
    FiberModel.MRG_INTERPOLATION, diameter=10, path_coordinates=path_coordinates
)
print(fiber)
print('Fiber is 3D?', fiber.is_3d())

# Using the resample_potentials_3d() method to resample the potentials along the fiber path
fiber.resample_potentials_3d(
    potentials=potentials, potential_coords=path_coordinates, center=True, inplace=True
)

plt.figure()
plt.plot(
    fiber.longitudinal_coordinates,
    fiber.potentials,
    marker='o',
    label='resample_potentials_3d()',
)
plt.xlabel('Distance along fiber (µm)')
plt.ylabel('Potential (mV)')
plt.title('Resampled Potentials along Fiber')

# Calculating arc lengths and using resample_potentials()
arc_lengths = np.concatenate(
    ([0], np.cumsum(np.sqrt(np.sum(np.diff(path_coordinates, axis=0) ** 2, axis=1))))
)
fiber.resample_potentials(
    potentials=potentials, potential_coords=arc_lengths, center=True, inplace=True
)
plt.plot(
    fiber.longitudinal_coordinates,
    fiber.potentials,
    marker='x',
    label='resample_potentials()',
    alpha=0.6,
    color='k',
)
plt.show()

import numpy as np
from pyfibers import ScaledStim
from scipy.interpolate import interp1d

# Setup for simulation
time_step = 0.001  # milliseconds
time_stop = 15  # milliseconds
start, on, off = 0, 0.1, 0.2
waveform = interp1d(
    [start, on, off, time_stop], [1, -1, 0, 0], kind="previous"
)  # biphasic rectangular pulse

# Create stimulation object
stimulation = ScaledStim(waveform=waveform, dt=time_step, tstop=time_stop)
ap, time = stimulation.run_sim(-1.5, fiber)
print(f'Number of action potentials detected: {ap}')
print(f'Time of last action potential detection: {time} ms')

# Calculate point source potentials at all fiber coordinates
x, y, z = 800, 800, 500  # Point source location
i0 = 1  # Current of the point source
sigma = 1  # Isotropic conductivity
fiber_potentials = fiber.point_source_potentials(x, y, z, i0, sigma, inplace=True)

# Create subplots for both the potentials and 3D visualization
fig = plt.figure(figsize=(15, 6))

# Plot potentials
ax1 = plt.subplot(121)
ax1.plot(fiber.longitudinal_coordinates, fiber.potentials, marker='o', markersize=4)
ax1.set_xlabel('Distance along fiber (µm)')
ax1.set_ylabel('Potential (mV)')
ax1.set_title('Fiber potentials from point source')
ax1.grid(True, alpha=0.3)

# Plot the fiber with the point source
ax2 = plt.subplot(122, projection='3d')
ax2.plot(x, y, z, 'ro', label='Point Source', markersize=8)
ax2.plot(
    fiber.coordinates[:, 0],
    fiber.coordinates[:, 1],
    fiber.coordinates[:, 2],
    label='Fiber Path',
    linewidth=2,
)
ax2.set_title('Fiber Path with Point Source')
ax2.set_xlabel('X axis')
ax2.set_ylabel('Y axis')
ax2.set_zlabel('Z axis')
ax2.legend()

plt.tight_layout()
plt.show()

# Generate a right angle for fiber path
x = [0, 10000, 10000]
y = [0, 0, 10000]
z = [0, 0, 0]
path_coordinates = np.array([x, y, z]).T

model = FiberModel.MRG_INTERPOLATION


def fiber_plot(ax, fiber, title):
    """Plot the fiber path and coordinates on the given axis."""  # noqa: DAR101
    # Plot fiber path
    ax.plot(x, y, lw=10, color='gray', alpha=0.3, label='fiber path')
    # Plot fiber coordinates
    ax.plot(
        fiber.coordinates[:, 0],
        fiber.coordinates[:, 1],
        marker='o',
        markersize=3,
        lw=0,
        label='fiber coordinates',
        color='black',
    )
    ax.plot(
        fiber.coordinates[:, 0][::11],
        fiber.coordinates[:, 1][::11],
        marker='o',
        markersize=3,
        lw=0,
        c='r',
        label='nodes',
    )
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.legend()


# Define fiber configurations as a list of parameter dictionaries
fiber_configs = [
    {'title': 'No shift', 'center': False, 'shift': None, 'shift_ratio': None},
    {'title': 'Shifted by 200 µm', 'center': False, 'shift': 200, 'shift_ratio': None},
    {
        'title': 'Shifted by half internodal length',
        'center': False,
        'shift': None,
        'shift_ratio': 0.5,
    },
    {'title': 'Centered', 'center': True, 'shift': None, 'shift_ratio': None},
    {
        'title': 'Centered + shifted by 200 µm',
        'center': True,
        'shift': 200,
        'shift_ratio': None,
    },
    {
        'title': 'Centered + shifted by half internodal length',
        'center': True,
        'shift': None,
        'shift_ratio': 0.5,
    },
]

# Create fibers with different shift configurations
fibers = {}
titles = {}

for config in fiber_configs:
    # Build fiber with the specified parameters
    fiber_params = {
        'diameter': 13,
        'fiber_model': model,
        'temperature': 37,
        'path_coordinates': path_coordinates,
        'passive_end_nodes': 2,
    }

    # Add optional parameters if specified
    if config['center']:
        fiber_params['center'] = True
    if config['shift'] is not None:
        fiber_params['shift'] = config['shift']
    if config['shift_ratio'] is not None:
        fiber_params['shift_ratio'] = config['shift_ratio']

    # Create fiber
    fiber = build_fiber_3d(**fiber_params)

    # Store fiber and title
    key = config['title'].lower().replace(' ', '_').replace('+', '').replace('µm', 'um')
    fibers[key] = fiber
    titles[key] = config['title']

# Create subplots for all fiber configurations
fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
axes = axes.flatten()

for i, (key, fiber) in enumerate(fibers.items()):
    fiber_plot(axes[i], fiber, titles[key])

plt.tight_layout()
plt.show()
