# === unet_partial_finetune.py ===
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import random
import os
from diffusers import StableDiffusionPipeline, DDPMScheduler
import torch.nn.functional as F
from tqdm import tqdm

# === Настройки ===
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
CHECKPOINT_DIR = Path("unet_finetuned_checkpoints")
DATASET_DIR = Path("dataset/baroque/resized")
PROMPT = "a painting in baroque style"
BATCH_SIZE = 2
EPOCHS = 3
LR = 1e-5
IMG_SIZE = 512
SAVE_EVERY = 1  # эпох
PERCENT_PARAMS_TO_TRAIN = 0.05

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Загрузка модели ===
pipe = StableDiffusionPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
pipe.safety_checker = None
pipe = pipe.to(device)
vae = pipe.vae.eval()
unet = pipe.unet.train()
tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder.eval()

# === Размораживаем часть параметров UNet ===
all_params = list(unet.named_parameters())
num_total = len(all_params)
num_trainable = int(num_total * PERCENT_PARAMS_TO_TRAIN)
selected = set(random.sample(range(num_total), num_trainable))
for i, (name, param) in enumerate(all_params):
    param.requires_grad = i in selected
print(f"Разморожено {num_trainable} из {num_total} параметров ({PERCENT_PARAMS_TO_TRAIN * 100:.1f}%)")

# === Датасет ===
class BaroqueDataset(Dataset):
    def __init__(self, folder):
        self.files = list(Path(folder).glob("*.jpg"))
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        image = Image.open(self.files[idx]).convert("RGB")
        return self.transform(image)

# === Подготовка обучения ===
dataset = BaroqueDataset(DATASET_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, unet.parameters()), lr=LR)

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# === Обучение ===
for epoch in range(EPOCHS):
    print(f"\nЭпоха {epoch + 1}/{EPOCHS}")
    for images in tqdm(loader, desc="Обучение"):
        images = images.to(device)

        # Кодирование в латенты
        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample() * 0.18215
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (latents.shape[0],), device=device).long()
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            # Правильная генерация текстовых эмбеддингов
            text_input = tokenizer([PROMPT] * BATCH_SIZE, padding="max_length", max_length=77, return_tensors="pt").input_ids.to(device)
            encoder_hidden_states = text_encoder(text_input)[0]

        # Прогон UNet
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        loss = F.mse_loss(noise_pred, noise)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Loss: {loss.item():.4f}")

    # === Сохранение чекпоинта ===
    if (epoch + 1) % SAVE_EVERY == 0:
        save_path = CHECKPOINT_DIR / f"epoch_{epoch + 1}"
        save_path.mkdir(parents=True, exist_ok=True)
        pipe.save_pretrained(save_path)
        print(f"Чекпоинт сохранён: {save_path.resolve()}")
