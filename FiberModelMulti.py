import pyfibers

# Enable logging to see multiple source simulation progress
pyfibers.enable_logging()

import numpy as np
from pyfibers import build_fiber, FiberModel, ScaledStim
from scipy.interpolate import interp1d

# create fiber model
n_sections = 265
fiber = build_fiber(FiberModel.MRG_INTERPOLATION, diameter=10, n_sections=n_sections)
print(fiber)

# Setup for simulation. Add zeros at the beginning so we get some baseline for visualization
time_step = 0.001
time_stop = 20
start, on, off = 0, 0.1, 0.2  # milliseconds
waveform = interp1d(
    [start, on, off, time_stop], [0, 1, 0, 0], kind="previous"
)  # monophasic rectangular pulse

fiber.potentials = fiber.point_source_potentials(0, 250, fiber.length / 2, 1, 10)

# Create stimulation object
stimulation = ScaledStim(waveform=waveform, dt=time_step, tstop=time_stop)

stimamp = -1.5  # mA
ap, time = stimulation.run_sim(stimamp, fiber)
print(f'Number of action potentials detected: {ap}')
print(f'Time of last action potential detection: {time} ms')

print("\nAttributi della fibra dopo la simulazione:")
print(dir(fiber))
print(help(fiber.record_sfap))
input('---> ')

fiber.potentials *= 0  # reset potentials
for position, polarity in zip([0.45 * fiber.length, 0.55 * fiber.length], [1, -1]):
    # add the contribution of one source to the potentials
    fiber.potentials += polarity * fiber.point_source_potentials(
        0, 250, position, 1, 10
    )

# plot the potentials
import matplotlib.pyplot as plt

plt.figure()
plt.plot(fiber.longitudinal_coordinates, fiber.potentials[0])
plt.xlabel('Position (μm)')
plt.ylabel('Potential (mV)')
plt.show()

# run simulation
ap, time = stimulation.run_sim(stimamp, fiber)
print(f'Number of action potentials detected: {ap}')
print(f'Time of last action potential detection: {time} ms')

potentials = []
# Create potentials from each source
for position in [0.45 * fiber.length, 0.55 * fiber.length]:
    potentials.append(fiber.point_source_potentials(0, 250, position, 1, 1))
fiber.potentials = np.vstack(potentials)
print('fiber potential shape :', fiber.potentials.shape)

plt.figure()
plt.plot(fiber.potentials[0, :], label='source 1')
plt.plot(fiber.potentials[1, :], label='source 2')
plt.legend()
plt.show()

