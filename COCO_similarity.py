import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision.datasets import CocoCaptions
import COCO_tester
def main():
    N = 10
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.Grayscale(3)])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Loading dataset (COCO)")
    coco = CocoCaptions(
        root='data/coco/val2017',
        annFile='data/coco/annotations/captions_val2017.json',
        transform=transform)
    coco = [coco[i] for i in range(N)]
    captions = [cap for img, cap in coco]


    image_data = torch.load("image_embeddings_COCO.pt", map_location=torch.device('cpu'))
    enc_images = image_data["image"]  # shape (N, D)
    labels = image_data["labels"]  # list of ints, length N



    text_data = torch.load("text_embeddings_COCO.pt", map_location=torch.device('cpu'))
    encoded_texts = text_data["prompt"]  # shape (10, D)
    first_caption_embeddings = [captions[0] for captions in encoded_texts]  # each is shape [D]
    first_caption_embeddings = torch.stack(first_caption_embeddings)        # shape [N, D]
    enc_prompts = first_caption_embeddings


    similarity = enc_images @ enc_prompts.T

    prediction = similarity.argmax(dim = 1).cpu().numpy()

    labels = [cap_list[0] for cap_list in captions]
    labels = np.array(labels)

    accuracy = (prediction == labels).mean()
    print(f'Accuracy: {accuracy:.4f}')
    print(len(prediction))

if __name__ == '__main__':
    main()