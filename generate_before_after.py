from diffusers import StableDiffusionPipeline
import torch
import os
import uuid
from datetime import datetime

# Настройки
prompt = "an artwork in baroque style"
checkpoint_path = "lora_outputs/baroque/fast_safe_run/checkpoint-10500"
#checkpoint_path = "lora_outputs/baroque/checkpoint-10500"
base_model = "runwayml/stable-diffusion-v1-5"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Каталоги для сохранения
base_output_dir = "outputs"
before_dir = os.path.join(base_output_dir, "before_lora")
after_dir = os.path.join(base_output_dir, "after_lora")
os.makedirs(before_dir, exist_ok=True)
os.makedirs(after_dir, exist_ok=True)

# Загружаем базовую модель
pipe = StableDiffusionPipeline.from_pretrained(
    base_model,
    torch_dtype=torch.float32
).to("cuda" if torch.cuda.is_available() else "cpu")

# === Генерация до LoRA ===
unique_id = uuid.uuid4().hex
before_path = os.path.join(before_dir, f"before_{timestamp}_{unique_id}.png")
image_before = pipe(prompt).images[0]
image_before.save(before_path)
print(f"✅ Сохранено: {before_path}")

# === Подгружаем LoRA ===
pipe.unet.load_attn_procs(checkpoint_path)

# === Генерация после LoRA ===
unique_id = uuid.uuid4().hex
after_path = os.path.join(after_dir, f"after_{timestamp}_{unique_id}.png")
image_after = pipe(prompt).images[0]
image_after.save(after_path)
print(f"✅ Сохранено: {after_path}")
