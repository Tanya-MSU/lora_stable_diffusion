# import torch
# from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
# from safetensors.torch import load_file
# from pathlib import Path
# from PIL import Image
# import pandas as pd
# from pytorch_fid import fid_score
#
# # === Настройки ===
# PROMPT = "an artwork in baroque style"
# NUM_IMAGES = 100
# OUTPUT_DIR = Path("comparison_fid_100")
# MODEL_NAME = "runwayml/stable-diffusion-v1-5"
# LORA_PATH = "lora_outputs/baroque/checkpoint-10000/pytorch_lora_weights.safetensors"
# CSV_PATH = OUTPUT_DIR / "fid_result.csv"
# DATASET_PATH = Path("dataset/baroque/resized")
#
# # === Папки без удаления ===
# vanilla_dir = OUTPUT_DIR / "vanilla"
# lora_dir = OUTPUT_DIR / "lora"
# vanilla_dir.mkdir(parents=True, exist_ok=True)
# lora_dir.mkdir(parents=True, exist_ok=True)
#
# # === Генерация изображений ===
# def generate_images(pipe, output_path, num_images):
#     existing = len(list(output_path.glob("*.png")))
#     for i in range(existing, existing + num_images):
#         image = pipe(PROMPT).images[0]
#         image.save(output_path / f"{i:03d}.png")
#
# # === Загрузка модели ===
# pipe = StableDiffusionPipeline.from_pretrained(MODEL_NAME, torch_device="cuda")
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# pipe.safety_checker = None
# pipe.enable_attention_slicing()
#
# # === Генерация без LoRA ===
# generate_images(pipe, vanilla_dir, NUM_IMAGES)
#
# # === Подключение LoRA ===
# state_dict = load_file(LORA_PATH)
# pipe.unet.load_attn_procs(state_dict)
#
# # === Генерация с LoRA ===
# generate_images(pipe, lora_dir, NUM_IMAGES)
#
# # === Подсчёт FID ===
# fid_vanilla = fid_score.calculate_fid_given_paths(
#     [str(DATASET_PATH), str(vanilla_dir)],
#     batch_size=10,
#     device="cuda",
#     dims=2048,
# )
# fid_lora = fid_score.calculate_fid_given_paths(
#     [str(DATASET_PATH), str(lora_dir)],
#     batch_size=10,
#     device="cuda",
#     dims=2048,
# )
#
# # === Сохраняем результат ===
# df = pd.DataFrame([{
#     "prompt": PROMPT,
#     "num_images": NUM_IMAGES,
#     "base_model": MODEL_NAME,
#     "lora_path": LORA_PATH,
#     "fid_vanilla_vs_dataset": round(fid_vanilla, 4),
#     "fid_lora_vs_dataset": round(fid_lora, 4)
# }])
#
# CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
# df.to_csv(CSV_PATH, mode='a', header=not CSV_PATH.exists(), index=False)
#
# print(f"Готово!\nFID без LoRA: {fid_vanilla:.4f}\nFID с LoRA: {fid_lora:.4f}")
# print(f"CSV записан в: {CSV_PATH.resolve()}")

# import torch
# from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
# from safetensors.torch import load_file
# from pathlib import Path
# from PIL import Image
# import datetime
# import gc
#
# # === Настройки ===
# PROMPT = "a painting in baroquo style"
# NUM_IMAGES = 100
# MODEL_NAME = "runwayml/stable-diffusion-v1-5"
# LORA_PATH = "lora_outputs/baroque/fast_safe_run/checkpoint-10500/pytorch_lora_weights.safetensors"
# BASE_OUTPUT = Path("comparison_fid-2_100")
#
# # === Функция для создания уникальной папки ===
# def get_unique_output_dir(base_path: Path):
#     index = 2
#     while True:
#         path = base_path.parent / f"{base_path.stem.split('_')[0]}-{index}_{base_path.stem.split('_')[1]}"
#         if not path.exists():
#             return path
#         index += 1
#
# # === Создание директорий ===
# OUTPUT_DIR = get_unique_output_dir(BASE_OUTPUT)
# VANILLA_DIR = OUTPUT_DIR / "vanilla"
# LORA_DIR = OUTPUT_DIR / "lora"
# VANILLA_DIR.mkdir(parents=True, exist_ok=True)
# LORA_DIR.mkdir(parents=True, exist_ok=True)
#
# # === Функция генерации изображений ===
# def generate_images(pipe, output_path, num_images, label):
#     for i in range(num_images):
#         image = pipe(PROMPT).images[0]
#         timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#         filename = f"{label.lower()}_{timestamp}_{i:03d}.png"
#         image.save(output_path / filename)
#         print(f"✅ Сохранено: {filename}")
#
# # === Основной процесс ===
# def main():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"📁 Папка для вывода: {OUTPUT_DIR}")
#
#     # Базовая модель
#     print("🔄 Загружаем базовую модель...")
#     pipe = StableDiffusionPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
#     pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
#     pipe.safety_checker = None
#     pipe.enable_attention_slicing()
#     pipe = pipe.to(device)
#
#     # Генерация обычных изображений
#     print("🎨 Генерация изображений базовой моделью...")
#     generate_images(pipe, VANILLA_DIR, NUM_IMAGES, label="vanilla")
#
#     # Подключение LoRA
#     print("🔗 Подключаем LoRA...")
#     state_dict = load_file(LORA_PATH)
#     pipe.unet.load_attn_procs(state_dict)
#
#     # Генерация изображений с LoRA
#     print("🎨 Генерация изображений с LoRA...")
#     generate_images(pipe, LORA_DIR, NUM_IMAGES, label="lora")
#
#     del pipe
#     gc.collect()
#     torch.cuda.empty_cache()
#
#     print("✅ Генерация завершена!")
#
# if __name__ == "__main__":
#     main()


import torch
import pandas as pd
from pathlib import Path
from pytorch_fid import fid_score

# === Настройки ===
PROMPT = "a painting in baroquo style"
NUM_IMAGES = 100
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
LORA_PATH = "lora_outputs/baroque/fast_safe_run/checkpoint-10500/pytorch_lora_weights.safetensors"

# === Пути ===
OUTPUT_DIR = Path("comparison-2_fid-2")
CSV_PATH = OUTPUT_DIR / "fid-2_result.csv"
DATASET_PATH = Path("dataset/baroque/resized")
VANILLA_DIR = OUTPUT_DIR / "vanilla"
LORA_DIR = OUTPUT_DIR / "lora"

def main():
    print("⏳ Считаем FID для vanilla...")
    fid_vanilla = fid_score.calculate_fid_given_paths(
        [str(DATASET_PATH), str(VANILLA_DIR)],
        batch_size=10,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dims=2048,
    )

    print("⏳ Считаем FID для LoRA...")
    fid_lora = fid_score.calculate_fid_given_paths(
        [str(DATASET_PATH), str(LORA_DIR)],
        batch_size=10,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dims=2048,
    )

    print(f"\n✅ Готово!")
    print(f"FID vanilla vs dataset: {fid_vanilla:.4f}")
    print(f"FID lora vs dataset: {fid_lora:.4f}")

    df = pd.DataFrame([{
        "prompt": PROMPT,
        "num_images": NUM_IMAGES,
        "base_model": MODEL_NAME,
        "lora_path": LORA_PATH,
        "fid_vanilla_vs_dataset": round(fid_vanilla, 4),
        "fid_lora_vs_dataset": round(fid_lora, 4)
    }])

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, mode='a', header=not CSV_PATH.exists(), index=False)
    print(f"📁 CSV записан в: {CSV_PATH.resolve()}")

if __name__ == "__main__":
    main()
