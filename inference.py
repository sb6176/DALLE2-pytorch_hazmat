import torch
from torchvision.transforms import ToPILImage
from dalle2_pytorch import DiffusionPrior, DiffusionPriorNetwork, OpenAIClipAdapter, Decoder, DALLE2, Unet
from dalle2_pytorch.train_configs import TrainDiffusionPriorConfig, TrainDecoderConfig
import open_clip

def compare_tensors(tensor1, tensor2, tol=1e-5):
    return torch.allclose(tensor1, tensor2, atol=tol)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model_path = "./pretrained_models/clip/ViT-L-14.pt"
model_path = './logs/2024_07_19-11_27_30-model_ViT-L-14-lr_5e-06-b_100-j_4-p_amp/checkpoints/ViT-L-14.pt'
model_name = "ViT-L-14"

custom_clip = OpenAIClipAdapter(model_name, model_path, is_open_clip=True) 
tokenizer = open_clip.get_tokenizer(model_name)
custom_clip.to(device)

prior_config = TrainDiffusionPriorConfig.from_json_path("pretrained_models/prior_config.json").prior
prior = prior_config.create().cuda()

prior_model_state = torch.load("pretrained_models/prior.pth")
prior.load_state_dict(prior_model_state, strict=True)

decoder_config = TrainDecoderConfig.from_json_path("pretrained_models/decoder_v1.0.2_upsampler_config.json").decoder
decoder = decoder_config.create().cuda()

decoder_unet_0_state = torch.load("pretrained_models/decoder_v1.0.2.pth")["model"]
decoder_unet_1_state = torch.load("pretrained_models/upsampler/veldrovive/latest.pth")
decoder_unet_2_state = torch.load("pretrained_models/upsampler/v1.0.2/latest.pth")["model"]
decoder.load_state_dict(decoder_unet_0_state, strict=False)
decoder.load_state_dict(decoder_unet_1_state, strict=False)
decoder.load_state_dict(decoder_unet_2_state, strict=False)

prior.clip = custom_clip
decoder.clip = custom_clip

dalle2 = DALLE2(prior=prior, decoder=decoder).cuda()

prompt = "a red sports car"

images = dalle2(
    [prompt],
    cond_scale = 2.
).cpu()

print(images.shape)

for img in images:
    img = ToPILImage()(img)
    img.save(f"./out/{prompt}.jpg")

#~~~~~~~~~~~~~~~

# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# # model_path = "./pretrained_models/clip/ViT-L-14.pt"
# model_path = './logs/2024_07_19-11_27_30-model_ViT-L-14-lr_5e-06-b_100-j_4-p_amp/checkpoints/ViT-L-14.pt'
# model_name = "ViT-L-14"

# clip = OpenAIClipAdapter(model_name, model_path, is_open_clip=True) 
# tokenizer = open_clip.get_tokenizer(model_name)
# clip.to(device)

# decoder_config = TrainDecoderConfig.from_json_path("pretrained_models/decoder_v1.0.2_config.json").decoder
# decoder = decoder_config.create().cuda()

# decoder_model_state = torch.load("pretrained_models/decoder_v1.0.2.pth")["model"]

# for k in decoder.clip.state_dict().keys():
#     decoder_model_state["clip." + k] = decoder.clip.state_dict()[k]

# decoder.load_state_dict(decoder_model_state, strict=True)

# # prior networks (with transformer)

# prior_network = DiffusionPriorNetwork(
#     dim = 768,
#     depth = 6,
#     dim_head = 64,
#     heads = 8
# ).cuda()

# diffusion_prior = DiffusionPrior(
#     net = prior_network,
#     clip = clip,
#     image_embed_dim = 768,
#     timesteps = 100,
#     cond_drop_prob = 0.2
# ).cuda()

# unet1 = Unet(
#     dim = 128,
#     image_embed_dim = 768,
#     cond_dim = 128,
#     channels = 3,
#     dim_mults=(1, 2, 4, 8),
#     text_embed_dim = 768,
#     cond_on_text_encodings = True  # set to True for any unets that need to be conditioned on text encodings (ex. first unet in cascade)
# ).cuda()

# unet2 = Unet(
#     dim = 16,
#     image_embed_dim = 768,
#     cond_dim = 128,
#     channels = 3,
#     dim_mults = (1, 2, 4, 8, 16)
# ).cuda()

# decoder = Decoder(
#     unet = (unet1, unet2),
#     image_sizes = (128, 256),
#     clip = clip,
#     timesteps = 1000,
#     sample_timesteps = (250, 27),
#     image_cond_drop_prob = 0.1,
#     text_cond_drop_prob = 0.5
# ).cuda()

# dalle2 = DALLE2(
#     prior = diffusion_prior,
#     decoder = decoder
# )

# prompt = "open clip exploring a github repo"

# images = dalle2(
#     [prompt],
#     cond_scale = 2.
# ).cpu()

# print(images.shape)

# for img in images:
#     img = ToPILImage()(img)
#     img.save(f"./out/{prompt}.jpg")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# prior_config = TrainDiffusionPriorConfig.from_json_path("pretrained_models/prior_config.json").prior
# prior = prior_config.create().cuda()

# prior_model_state = torch.load("pretrained_models/prior.pth")
# prior.load_state_dict(prior_model_state, strict=True)

# decoder_config = TrainDecoderConfig.from_json_path("pretrained_models/upsampler/v1.0.3/decoder_config.json").decoder
# decoder = decoder_config.create().cuda()

# with open("decoder_architecture.txt", "w") as f:
#     print(decoder, file=f)

# decoder_model_state = torch.load("pretrained_models/upsampler/v1.0.3/latest.pth")["model"]

# for k in decoder.clip.state_dict().keys():
#     decoder_model_state["clip." + k] = decoder.clip.state_dict()[k]

# decoder.load_state_dict(decoder_model_state, strict=True)

# dalle2 = DALLE2(prior=prior, decoder=decoder).cuda()

# prompt = "Hello world!"

# images = dalle2(
#     [prompt],
#     cond_scale = 2.
# ).cpu()

# print(images.shape)

# for img in images:
#     img = ToPILImage()(img)
#     img.save(f"./out/{prompt}.jpg")