import torch
import clip
from PIL import Image
import json
import os
import numpy as np
import matplotlib.pyplot as plt
# from tensorboardX import SummaryWriter

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
data_dir = '../Data/test-data/masks'

with open('/home/aston/Desktop/python/CAD-Matching/Data/test-data/ade20k_instance_imgCatIds.json','rb') as f:
    instance_list = json.load(f)

all_cat = []
for i in range(len(instance_list['categories'])):
    all_cat.append("a " + instance_list['categories'][i]['name'])

with open(os.path.join(data_dir,'cat.json'),'r') as f:
    catid = json.load(f)

image = ''
# for i in range(len(catid)):
for i in range(1):
    image_path = os.path.join(data_dir,str(i+1)+'.png')

    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize(all_cat).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    max_index = np.argmax(probs)

    plt.figure()
    img = plt.imread(image_path)
    plt.imshow(img)
    plt.title(all_cat[max_index])
    plt.axis('off')
    plt.show()

    print("Label probs:", all_cat[max_index])

print(model.encode_image)