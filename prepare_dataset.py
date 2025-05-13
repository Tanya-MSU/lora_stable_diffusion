import os
from PIL import Image
import pandas as pd

# Путь до папки dataset относительно проекта
root_dir = os.path.join(os.getcwd(), "dataset")
output_csv = os.path.join(root_dir, "dataset.csv")
data = []

# Проходим по всем стилям в dataset/
for style in os.listdir(root_dir):
    style_path = os.path.join(root_dir, style)
    raw_path = os.path.join(style_path, "raw")
    output_path = os.path.join(style_path, "resized")

    if not os.path.isdir(raw_path):
        continue

    os.makedirs(output_path, exist_ok=True)

    for i, filename in enumerate(sorted(os.listdir(raw_path))):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img = Image.open(os.path.join(raw_path, filename)).convert("RGB")
        img = img.resize((512, 512))

        new_filename = f"{style}_{i+1:04}.jpg"
        final_path = os.path.join(output_path, new_filename)
        img.save(final_path)

        data.append({
            "file": os.path.relpath(final_path, os.getcwd()).replace("\\", "/"),
            "prompt": f"an artwork in {style} style"
        })

df = pd.DataFrame(data)
df.to_csv(output_csv, index=False)

print(f"✅ Обработано стилей: {len(data)} изображений сохранено в {output_csv}")
