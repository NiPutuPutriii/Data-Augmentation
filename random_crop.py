import torchvision
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open('kuciang.png')
crop = torchvision.transforms.RandomCrop(size=256)

fig, axes = plt.subplots(3, 3, figsize=(7, 8))
fig.subplots_adjust(wspace=0.3)
for ax in axes.flatten():
    ax.imshow(crop(img))