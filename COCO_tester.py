import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.datasets import CocoCaptions
import matplotlib.pyplot as plt
from clip import clip
import numpy as np
from PIL import Image
from torch.nn import functional as F
import os

N = 10
print(f'Initializing with batch of size {N}')


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, preprocess = clip.load('ViT-B/32', device = device)
print("Loading dataset (COCO)")
coco = CocoCaptions(
    root='data/coco/val2017',
    annFile='data/coco/annotations/captions_val2017.json',
    transform=preprocess
)


# Creating a suitable subset of COCO
coco = [coco[i] for i in range(N)]
images = [img for img, cap in coco]
captions = [cap for img, cap in coco]
input_images = torch.stack(images).to(device)
tokenized_captions = [clip.tokenize(caps).to(device) for caps in captions]

# Initialize embeddings 
print("Done preprocessing!")
print("Computing embeddings...")
with torch.no_grad():
    encoded_texts = [model.encode_text(tok_caps) for tok_caps in tokenized_captions]
    encoded_images = model.encode_image(input_images)

print("Embeddings done!")
print("Saving embeddings...")
labels = [i for i in range(N)]

torch.save({'image': encoded_images,
            'labels': labels
            }, 'image_embeddings_COCO.pt')

torch.save({'prompt': encoded_texts,
            'labels': labels
            }, 'text_embeddings_COCO.pt')


print(f'Embeddings saved!')
print('Job finished')