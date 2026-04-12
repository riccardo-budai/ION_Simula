import pyfibers
from neuron import h, load_mechanisms
import os
import numpy as np
from pyfibers import build_fiber, FiberModel, ScaledStim
from scipy.interpolate import interp1d
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Helper function for consistent plotting
def plot_fiber_potentials(
    fiber_obj, potentials, title, ax=None, show_full_span=True, use_shifted_coords=False
):
    """Plot fiber potentials with nodes highlighted.

    :param fiber_obj: The fiber object containing coordinates and properties.
    :type fiber_obj: Fiber
    :param potentials: Array of potential values to plot.
    :type potentials: array_like
    :param title: Title for the plot.
    :type title: str
    :param ax: Axes to plot on. If None, creates new figure and axes.
    :type ax: matplotlib.axes.Axes, optional
    :param show_full_span: Whether to show the full potential distribution background.
    :type show_full_span: bool, optional
    :param use_shifted_coords: Whether to use shifted coordinates if available.
    :type use_shifted_coords: bool, optional
    :return: The axes object containing the plot.
    :rtype: matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))

    # Plot the full potential distribution if requested and available
    if (
        show_full_span
        and 'supersampled_potentials' in globals()
        and 'coords' in globals()
    ):
        ax.plot(
            coords,
            supersampled_potentials,
            '-',
            linewidth=1,
            color='lightgray',
            label='potential distribution',
            zorder=-1,
        )

    # Use shifted coordinates if requested and available, otherwise use original coordinates
    if use_shifted_coords and hasattr(fiber_obj, 'shifted_coordinates'):
        plot_coords = fiber_obj.shifted_coordinates
    else:
        plot_coords = fiber_obj.coordinates[:, 2]

    # Plot all sections as dots
    ax.scatter(plot_coords, potentials, s=20, color='black', label='sections')

    # Plot nodes (every 11th point for MRG fibers) as larger red dots
    ax.scatter(plot_coords[::11], potentials[::11], c='red', s=40, label='nodes')

    ax.set_title(title)
    ax.set_xlabel('Position along fiber (μm)')
    ax.set_ylabel('Potential (mV)')
    ax.legend()

    if ax is None:
        plt.show()

    return ax

########################################################################################################################
n_coords = 10000

supersampled_potentials = norm.pdf(np.linspace(-1, 1, n_coords), 0, 0.2) * 10
coords = np.cumsum([1] * n_coords)

plt.plot(coords, supersampled_potentials)
plt.title('Extracellular potentials')
plt.xlabel('Position along fiber (\u03bcm)')
plt.ylabel('Potential (mV)')
plt.show()

fiber_length = np.amax(coords) - np.amin(coords)
fiber = build_fiber(FiberModel.MRG_INTERPOLATION, diameter=10, length=fiber_length)
print(fiber)

# Compare non-centered vs centered resampling with align_coordinates=True (default)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Non-centered: align potential coordinates to start at 0
fiber.potentials = fiber.resample_potentials(supersampled_potentials, coords)
plot_fiber_potentials(fiber, fiber.potentials, 'Non-centered', ax=ax1)

# Centered: align midpoints of both coordinate systems
fiber.resample_potentials(supersampled_potentials, coords, center=True, inplace=True)
plot_fiber_potentials(fiber, fiber.potentials, 'Centered', ax=ax2)

plt.tight_layout()
plt.show()

# Using the updated plot_fiber_potentials function with shifted coordinates enabled
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Non-centered
fiber.potentials = fiber.resample_potentials(supersampled_potentials, coords)
plot_fiber_potentials(
    fiber,
    fiber.potentials,
    'Non-centered',
    ax=ax1,
    use_shifted_coords=True,
)

# Centered
fiber.resample_potentials(supersampled_potentials, coords, center=True, inplace=True)
plot_fiber_potentials(
    fiber,
    fiber.potentials,
    'Centered',
    ax=ax2,
    use_shifted_coords=True,
)

plt.tight_layout()
plt.show()

# Demonstrate basic shifting
print(f"Fiber internodal length (delta_z): {fiber.delta_z:.1f} μm")

# Test different shift values
shift_values = [0, 250, 500]  # μm
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, shift_val in enumerate(shift_values):
    # Apply shift during resampling
    shifted_potentials = fiber.resample_potentials(
        supersampled_potentials,
        coords,
        center=True,
        shift=shift_val,
    )

    plot_fiber_potentials(
        fiber,
        shifted_potentials,
        f'Shift = {shift_val} μm',
        ax=axes[i],
        use_shifted_coords=True,
    )

plt.tight_layout()
plt.show()

print(f"Current fiber length: {fiber.length:.1f} μm")
print(f"Potential span: {coords[-1] - coords[0]:.1f} μm")
print("\\nTesting shifts with error handling...")

# Test shifts that will likely cause errors
shift_values = [0, 500, 1000]  # μm
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
successful_shifts = []
failed_shifts = []

for i, shift_val in enumerate(shift_values):
    try:
        shifted_potentials = fiber.resample_potentials(
            supersampled_potentials, coords, center=True, shift=shift_val
        )

        plot_fiber_potentials(
            fiber,
            shifted_potentials,
            f'Shift = {shift_val} μm',
            ax=axes[i],
            use_shifted_coords=True,
        )
        successful_shifts.append(shift_val)
        print(f"✓ Shift {shift_val} μm: Success")

    except ValueError as e:
        failed_shifts.append(shift_val)
        axes[i].text(
            0.5,
            0.5,
            'ERROR',
            ha='center',
            va='center',
            transform=axes[i].transAxes,
            bbox={'boxstyle': "round,pad=0.3", 'facecolor': "red", 'alpha': 0.3},
        )
        axes[i].set_title(f'Shift = {shift_val} μm (FAILED)')
        print(f"Shift {shift_val} μm: FAILED")
        print(e)

plt.tight_layout()
plt.show()

print(f"\\nSummary: {len(successful_shifts)} successful, {len(failed_shifts)} failed")
if failed_shifts:
    print("SOLUTION: Create a shorter fiber to accommodate larger shifts")
