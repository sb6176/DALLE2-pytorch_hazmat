import torch
from PIL import Image
import open_clip
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# model_path = "./pretrained_models/clip/ViT-L-14.pt"
model_path = './logs/2024_07_19-11_27_30-model_ViT-L-14-lr_5e-06-b_100-j_4-p_amp/checkpoints/ViT-L-14.pt'
model_name = "ViT-L-14"

model, _, preprocess = open_clip.create_model_and_transforms(model_name = model_name, pretrained = model_path)
tokenizer = open_clip.get_tokenizer(model_name)
model.to(device)

img_path = "./dataset/images/raw/image_0040.jpg"

image = preprocess(Image.open(img_path)).unsqueeze(0).cuda(device=device)
text = tokenizer(["2 inhalation-hazard hazmat labels", "a pile of boxes in a warehouse", "a wood floor"]).cuda(device=device)

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]
