import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import pandas as pd
from dalle2_pytorch import Unet, Decoder, CLIP, DecoderTrainer, OpenAIClipAdapter
from dalle2_pytorch.train_configs import TrainDiffusionPriorConfig, TrainDecoderConfig
from accelerate import Accelerator
import open_clip

# Define your dataset class
class ImageCaptionDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        caption = self.data.iloc[idx, 1]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, caption

def main():
    # Initialize Accelerator
    accelerator = Accelerator()

    # Initialize device and distributed training
    device = accelerator.device

    # Initialize TensorBoard writer if on main process
    if accelerator.is_main_process:
        writer = SummaryWriter()

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Load datasets
    train_dataset = ImageCaptionDataset(csv_file='./dataset/captions_train.csv', transform=transform)
    val_dataset = ImageCaptionDataset(csv_file='./dataset/captions_val.csv', transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=1200, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)
    
    # Load model and tokenizer
    model_path = './logs/2024_07_19-11_27_30-model_ViT-L-14-lr_5e-06-b_100-j_4-p_amp/checkpoints/ViT-L-14.pt'
    model_name = "ViT-L-14"
    custom_clip = OpenAIClipAdapter(model_name, model_path, is_open_clip=True) 
    tokenizer = open_clip.get_tokenizer(model_name)
    
    decoder_config = TrainDecoderConfig.from_json_path("pretrained_models/decoder_v1.0.2_upsampler_config.json").decoder
    decoder = decoder_config.create()
    
    print(isinstance(decoder, Decoder))

    decoder_unet_0_state = torch.load("pretrained_models/decoder_v1.0.2.pth", map_location="cpu")["model"]
    decoder_unet_1_state = torch.load("pretrained_models/upsampler/veldrovive/latest.pth", map_location="cpu")
    decoder.load_state_dict(decoder_unet_0_state, strict=False)
    decoder.load_state_dict(decoder_unet_1_state, strict=False)
    decoder.clip = custom_clip

    # # Prepare the model and optimizer for distributed training
    # decoder, train_loader, val_loader = accelerator.prepare(decoder, train_loader, val_loader)
        
    # decoder_trainer = DecoderTrainer(
    #     decoder,
    #     lr = 1e-5,
    #     wd = 1e-2,
    #     ema_beta = 0.99,
    #     ema_update_after_step = 1000,
    #     ema_update_every = 10,
    # )

    # epochs = 10
    # for epoch in range(epochs):
    #     decoder.train()
    #     train_loss = 0.0
    #     for images, captions in train_loader:
    #         loss = decoder_trainer(
    #             images,
    #             text=captions,
    #             max_batch_size=4
    #         )
    #         decoder_trainer.update()
    #         train_loss += loss.item()

    #     train_loss /= len(train_loader)
    #     if accelerator.is_main_process:
    #         writer.add_scalar(f'Loss/train', train_loss, epoch)

    #     decoder.eval()
    #     val_loss = 0.0
    #     with torch.no_grad():
    #         for images, captions in val_loader:
    #             loss = decoder_trainer(
    #                 images,
    #                 text=captions,
    #                 max_batch_size=4
    #             )
    #             val_loss += loss.item()

    #     val_loss /= len(val_loader)
    #     if accelerator.is_main_process:
    #         writer.add_scalar(f'Loss/val', val_loss, epoch)

    #     if accelerator.is_main_process:
    #         print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss}, Val Loss: {val_loss}")

    # if accelerator.is_main_process:
    #     writer.close()

if __name__ == "__main__":
    # Make sure to set the environment variable for distributed training if using
    main()

