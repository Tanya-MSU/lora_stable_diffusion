from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
from pathlib import Path

# === Настройки ===
prompt = "an artwork in baroque style"
before_dir = Path("outputs/before_lora")
after_dir = Path("outputs/after_lora")
device = "cuda" if torch.cuda.is_available() else "cpu"

# === Получаем последние изображения ===
def get_latest_image(folder):
    images = sorted(folder.glob("*.png"), key=os.path.getmtime)
    return images[-1] if images else None

before_path = get_latest_image(before_dir)
after_path = get_latest_image(after_dir)

if not before_path or not after_path:
    raise FileNotFoundError("❌ Не найдены изображения для сравнения.")

# === Загрузка CLIP модели ===
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# === Функция для вычисления CLIP similarity ===
def compute_similarity(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.cosine_similarity(outputs.image_embeds, outputs.text_embeds).item()

# === Вычисляем и сравниваем ===
before_sim = compute_similarity(before_path)
after_sim = compute_similarity(after_path)

print(f"\nBEFORE ({before_path.name}): {round(before_sim, 4)}")
print(f"AFTER  ({after_path.name}): {round(after_sim, 4)}")

improvement = after_sim - before_sim
print("\n📊 Улучшение:", round(improvement, 4))
if improvement > 0:
    print("✅ LoRA улучшила соответствие изображения промпту.")
elif improvement < 0:
    print("❌ LoRA сделала результат хуже.")
else:
    print("➖ Разницы нет.")
