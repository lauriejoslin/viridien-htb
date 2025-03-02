Our submission tackles Viridien's Challenge at HackTheBurgh 2025.

Frameworks, libraries, and AI tools used:
- Numpy (to store images as Tensors for PyTorch purposes)
- Matplotlib (to help in presentation and graphs)
- PyTorch / Torchvision (to implement Convolutional Neural Networks to classify land use)
- Sklearn (for scoring metrics assistance)
- PIL (for image manipulation purposes)

Our model uses the EuroSAT dataset, "a large-scale land use and land cover classification dataset derived from multispectral Sentinel-2 satellite imagery covering European continent. EuroSAT is composed of 27,000 georeferenced image patches (64 x 64 pixels) - each patch comprises 13 spectral bands (optical through to shortwave infrared ) resampled to 10m spatila resolution and labelled with one of 10 distinct land cover classes: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake."
The GitHub repo for it can be found [here](https://github.com/phelber/EuroSAT).

Our model first trains on the 64x64 images found in the EuroSAT dataset. For better predictive ability relative to the actual target image given in this challenge, we have omitted some classes that are less relevant (e.g. River).
We then slice the target RGB satellite image given in this challenge into 64x64 slices, before predicting their classes.

![classification](https://github.com/user-attachments/assets/063c936b-a27c-408a-9f65-d31a6c3bae1d)

Given that each square is 64x64 pixels, for a 10m x 10m resolution, each square is ~0.41 km square of land.
The top 4 rows (roughly where the new runway will be) contain:
- 7 residential squares (2.87 km square)
- 6 highway squares (2.46 km square)
- 14 pasture squares (5.74 km square)
- 1 Vegetation2 square (0.41 km square)

  
