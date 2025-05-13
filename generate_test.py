from diffusers import StableDiffusionPipeline
import torch

# Загружаем модель и переносим на GPU с поддержкой float16
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
).to("cuda")


# Отключаем NSFW-фильтр, чтобы избежать чёрных изображений
pipe.safety_checker = lambda images, clip_input: (images, [False] * len(images))

# Промпт — можно заменить на любой
prompt = "An elegant Neo-Rococo interior with ornate gilded furniture, pastel color palette, flowing silk drapery, floral motifs, large mirrors with golden frames, and romantic lighting from crystal chandeliers, high realism, ultra detailed, cinematic lighting, artstation, baroque influences"

# Генерация
generator = torch.manual_seed(42)  # чтобы результат был стабильным (опционально)
image = pipe(prompt, num_inference_steps=150, generator=generator).images[0]

# Сохраняем
image.save("result.png")
