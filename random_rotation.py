import torchvision
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open('kuciang.png')
rotate = torchvision.transforms.RandomRotation(degrees=20)

fig, axes = plt.subplots(3, 3, figsize=(8, 7))
fig.subplots_adjust(wspace=0.3)
for ax in axes.flatten():
    ax.imshow(rotate(img))