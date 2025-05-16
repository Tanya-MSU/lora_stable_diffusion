import torch
import pandas as pd
from pathlib import Path
from pytorch_fid import fid_score

# === Пути ===
DATASET_PATH = Path("dataset/baroque/resized")
UNET_PATH = Path("comparison_unet_epoch3/vanilla")
LORA_4_PATH = Path("comparison_fid_100/lora")
LORA_16_PATH = Path("comparison-2_fid-2/lora")
CSV_PATH = Path("fid_all_comparison.csv")

def compute_fid(name, gen_path):
    fid = fid_score.calculate_fid_given_paths(
        [str(DATASET_PATH), str(gen_path)],
        batch_size=10,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dims=2048,
    )
    print(f"FID {name}: {fid:.4f}")
    return round(fid, 4)

def main():
    results = []

    results.append({
        "name": "UNet fine-tuned (epoch 3)",
        "folder": str(UNET_PATH),
        "fid_vs_dataset": compute_fid("UNet", UNET_PATH)
    })

    results.append({
        "name": "LoRA (rank=4, alpha=4)",
        "folder": str(LORA_4_PATH),
        "fid_vs_dataset": compute_fid("LoRA rank=4", LORA_4_PATH)
    })

    results.append({
        "name": "LoRA (rank=16, alpha=32)",
        "folder": str(LORA_16_PATH),
        "fid_vs_dataset": compute_fid("LoRA rank=16", LORA_16_PATH)
    })

    df = pd.DataFrame(results)
    df.to_csv(CSV_PATH, index=False)
    print(f"\n CSV сохранён: {CSV_PATH.resolve()}")

if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()
    main()
