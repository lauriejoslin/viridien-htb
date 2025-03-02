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
