import torchvision
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open('cat.103.jpg')
brightness = torchvision.transforms.ColorJitter(brightness=0.8)

fig, axes = plt.subplots(3, 3, figsize=(8, 7))
fig.subplots_adjust(wspace=0.3)
for ax in axes.flatten():
    ax.imshow(brightness(img))