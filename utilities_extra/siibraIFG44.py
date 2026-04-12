import matplotlib.pyplot as plt
from nilearn import plotting
import seaborn as sns
import siibra

assert siibra.__version__ >= "1.0.1"

sns.set_style("dark")

jubrain = siibra.parcellations.JULICH_BRAIN_CYTOARCHITECTONIC_ATLAS_V3_0_3
pmaps = jubrain.get_map(space="mni152", maptype="statistical")
print(jubrain.publications[0]['citation'])

specs = ["ifg 44 left", "hoc1 left"]
regions = [jubrain.get_region(spec) for spec in specs]
for r in regions:
    plotting.plot_glass_brain(
        pmaps.fetch(region=r),
        cmap="viridis",
        draw_cross=False,
        colorbar=False,
        annotate=False,
        symmetric_cbar=True,
        title=r.name,
    )
plt.show()

fig, axs = plt.subplots(1, len(regions), sharey=True, figsize=(8, 2.7))
for i, region in enumerate(regions):
    layerwise_cellbody_densities = siibra.features.get(region, "layerwise cell density")
    layerwise_cellbody_densities[0].plot(ax=axs[i], textwrap=25)
    print(layerwise_cellbody_densities[0].urls)
    axs[i].set_ylim(25, 115)
plt.show()
