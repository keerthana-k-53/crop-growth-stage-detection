import rasterio
import numpy as np
import matplotlib.pyplot as plt

with rasterio.open("data/Mysuru_NDVI.tif") as src:
    ndvi = src.read(1)

classified = np.zeros_like(ndvi)

classified[(ndvi < 0)] = 0
classified[(ndvi >= 0) & (ndvi < 0.2)] = 1
classified[(ndvi >= 0.2) & (ndvi < 0.5)] = 2
classified[(ndvi >= 0.5)] = 3

plt.figure(figsize=(8,8))
plt.imshow(classified, cmap="RdYlGn")
plt.title("NDVI Land Cover Classification")
plt.colorbar()
plt.savefig("outputs/landcover_map.png", dpi=300)
plt.show()

print("Land cover map saved!")