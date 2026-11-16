import rasterio
import matplotlib.pyplot as plt

# Open the TIFF file
from pathlib import Path

# Get the project root
project_root = Path(__file__).resolve().parent.parent

# Build the full path
tiff_path = project_root / "data" / "Mysuru_NDVI.tif"

dataset = rasterio.open(tiff_path)

print("Dataset Information")
print("-------------------")
print(dataset)

# Read the first band
band = dataset.read(1)

print("Shape:", band.shape)
print("Minimum:", band.min())
print("Maximum:", band.max())

# Display the image
plt.imshow(band)
plt.colorbar()
plt.title("NDVI Image")
plt.savefig("outputs/ndvi_plot.png")
print("Image saved successfully!")