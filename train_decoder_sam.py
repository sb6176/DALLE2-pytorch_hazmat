import torch
from dalle2_pytorch import DALLE2, Unet, Decoder, CLIP, DecoderTrainer
from dalle2_pytorch import DiffusionPrior, DiffusionPriorNetwork, OpenAIClipAdapter, Decoder, DALLE2, Unet
from dalle2_pytorch.train_configs import TrainDiffusionPriorConfig, TrainDecoderConfig
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import open_clip
from torchvision import transforms
import pandas as pd
from PIL import Image
import numpy as np
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter

log_dir = "./logs/decoder/training_run_1"
writer = SummaryWriter(log_dir=log_dir)

# Define your dataset class
class ImageCaptionDataset(Dataset):
    def __init__(self, csv_file, transform=None, batch_size=32, tokenizer=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.batch_size = batch_size
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        caption = self.data.iloc[idx, 1]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.tokenizer([str(caption)])[0]
    
    def get_images(self):
        images = []
        for idx in range(len(self)):
            img_path = self.data.iloc[idx, 0]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            images.append(image)
        return torch.stack(images)
    
    def get_image_batches(self):
        all_images = self.get_images()
        batches = []
        for i in range(0, len(all_images), self.batch_size):
            batch = all_images[i:i + self.batch_size]
            batches.append(batch)
        return batches

get_num_parameters = lambda model, only_training=False: sum(p.numel() for p in model.parameters() if (p.requires_grad or not only_training))

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

accelerator = Accelerator()
device = accelerator.device

# model_path = "./pretrained_models/clip/ViT-L-14.pt"
clip_path = './logs/2024_07_19-11_27_30-model_ViT-L-14-lr_5e-06-b_100-j_4-p_amp/checkpoints/ViT-L-14.pt'
clip_name = "ViT-L-14"

# custom_clip = OpenAIClipAdapter(model_name, model_path, is_open_clip=True) 
clip_tokenizer = open_clip.get_tokenizer(clip_name)

# print(f"CLIP has {get_num_parameters(custom_clip)}")

# mock data
train_dataset = ImageCaptionDataset(csv_file='./dataset/captions_train.csv', transform=transform, tokenizer=clip_tokenizer)
val_dataset = ImageCaptionDataset(csv_file='./dataset/captions_val.csv', transform=transform, tokenizer=clip_tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=8)

decoder_config = TrainDecoderConfig.from_json_path("pretrained_models/decoder_v1.0.2_upsampler_config.json").decoder
decoder = decoder_config.create()

# decoder_unet_0_state = torch.load("pretrained_models/decoder_v1.0.2.pth")["model"]
# decoder_unet_1_state = torch.load("pretrained_models/upsampler/veldrovive/latest.pth")
# decoder_unet_2_state = torch.load("pretrained_models/upsampler/v1.0.2/latest.pth")["model"]
# decoder.load_state_dict(decoder_unet_0_state, strict=False)
# decoder.load_state_dict(decoder_unet_1_state, strict=False)
# decoder.load_state_dict(decoder_unet_2_state, strict=False)

# decoder.clip = custom_clip

decoder_trainer = DecoderTrainer(
    decoder,
    accelerator=accelerator,
    dataloaders={
        "train": train_dataloader,
        "val": val_dataloader
    },
    lr = 5e-7,
    wd = 1e-2,
    ema_beta = 0.99,
    ema_update_after_step = 1000,
    ema_update_every = 10,
)

decoder_trainer.load(path="pretrained_models/decoder_v1.0.2.pth", only_model=True, strict=False)
decoder_trainer.load(path="pretrained_models/upsampler/veldrovive/latest.pth", only_model=True, strict=False, model_arg=False)
decoder_trainer.load_clip(clip_name, clip_path)

epochs = 20
for epoch in range(epochs):
    if accelerator.is_main_process:
        print('Running epoch ', epoch, "/", epochs)
    for unet_number in (1, 2):
        loss = decoder_trainer(
            unet_number = unet_number, # which unet to train on
            max_batch_size = 4         # gradient accumulation - this sets the maximum batch size in which to do forward and backwards pass - for this example 32 / 4 == 8 times
        )

        decoder_trainer.update(unet_number) # update the specific unet as well as its exponential moving average
        
        print('Loss for unet#', unet_number, ': ', loss)
        
        writer.add_scalar(f'Loss/train/unet_{unet_number}', loss)
        
    # for i, optimizer in enumerate(decoder_trainer.optimizers):
    #     if optimizer is not None:
    #         for j, param_group in enumerate(optimizer.param_groups):
    #             writer.add_scalar(f'Learning_rate/optimizer_{i}/group_{j}', param_group['lr'], epoch)
      

DecoderTrainer.save(path=log_dir)

writer.close()

# after much training
# you can sample from the exponentially moving averaged unets as so

# mock_image_embed = torch.randn(32, 512).cuda()
# images = decoder_trainer.sample(image_embed = mock_image_embed, text = text) # (4, 3, 256, 256)