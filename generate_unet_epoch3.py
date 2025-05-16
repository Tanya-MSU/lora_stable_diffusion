from pathlib import Path
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

# === Настройки ===
CHECKPOINT_PATH = "unet_finetuned_checkpoints/epoch_3"
OUTPUT_DIR = Path("comparison_unet_epoch3/vanilla")
PROMPT = "a painting in baroque style"
NUM_IMAGES = 100

# === Подготовка папки ===
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === Загрузка дообученной модели (всё автоматически) ===
pipe = StableDiffusionPipeline.from_pretrained(
    CHECKPOINT_PATH,
    torch_dtype=torch.float32
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
pipe.safety_checker = None

# === Генерация изображений ===
for i in range(NUM_IMAGES):
    image = pipe(PROMPT).images[0]
    image.save(OUTPUT_DIR / f"{i:03d}.png")

print(f"Сгенерировано {NUM_IMAGES} изображений в папку: {OUTPUT_DIR.resolve()}")
