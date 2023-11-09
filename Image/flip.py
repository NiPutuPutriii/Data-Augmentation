from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open('kuciang.png')

flip = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip()
])

fig, axes = plt.subplots(3, 3, figsize=(10, 8))
fig.subplots_adjust(wspace=0.3)
for ax in axes.flatten():
    ax.imshow(flip(img))