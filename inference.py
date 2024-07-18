import torch
from torchvision.transforms import ToPILImage
from dalle2_pytorch import DiffusionPrior, DiffusionPriorNetwork, OpenAIClipAdapter, Decoder, DALLE2
from dalle2_pytorch.train_configs import TrainDiffusionPriorConfig, TrainDecoderConfig


prior_config = TrainDiffusionPriorConfig.from_json_path("pretrained_models/prior_config.json").prior
prior = prior_config.create().cuda()

prior_model_state = torch.load("pretrained_models/prior.pth")
prior.load_state_dict(prior_model_state, strict=True)

decoder_config = TrainDecoderConfig.from_json_path("pretrained_models/decoder_v1.0.2_config.json").decoder
decoder = decoder_config.create().cuda()

decoder_model_state = torch.load("pretrained_models/decoder_v1.0.2.pth")["model"]

for k in decoder.clip.state_dict().keys():
    decoder_model_state["clip." + k] = decoder.clip.state_dict()[k]

decoder.load_state_dict(decoder_model_state, strict=True)

dalle2 = DALLE2(prior=prior, decoder=decoder).cuda()

prompt = "a poison hazmat label on a barrel"

images = dalle2(
    [prompt],
    cond_scale = 2.
).cpu()

print(images.shape)

for img in images:
    img = ToPILImage()(img)
    img.save(f"./out/{prompt}.jpg")