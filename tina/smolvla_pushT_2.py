from pathlib import Path

import matplotlib.pyplot as plt
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


# ---------------------------------------------------------
# 1. Putanja do tvog završenog treninga
# ---------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

RUN_DIR = REPO_ROOT / "tina" / "outputs" / "smolvla_pusht_prviOzbiljni20k"
CHECKPOINTS_DIR = RUN_DIR / "checkpoints"

# Automatski pronalazi poslednji numerički checkpoint
checkpoint_dirs = sorted(
    path
    for path in CHECKPOINTS_DIR.iterdir()
    if path.is_dir() and path.name.isdigit()
)

if not checkpoint_dirs:
    raise FileNotFoundError(
        f"Nije pronađen checkpoint u: {CHECKPOINTS_DIR}"
    )

model_path = checkpoint_dirs[-1] / "pretrained_model"

if not model_path.exists():
    raise FileNotFoundError(
        f"Nije pronađen pretrained_model folder: {model_path}"
    )

print(f"Učitavam model iz:\n{model_path}")


# ---------------------------------------------------------
# 2. GPU ili CPU
# ---------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Uređaj: {device}")


# ---------------------------------------------------------
# 3. Učitavanje tvog istreniranog SmolVLA modela
# ---------------------------------------------------------

policy = SmolVLAPolicy.from_pretrained(
    str(model_path)
).to(device).eval()


# ---------------------------------------------------------
# 4. Učitavanje procesora iz ISTOG checkpointa
# ---------------------------------------------------------

preprocess, postprocess = make_pre_post_processors(
    policy.config,
    str(model_path),
    preprocessor_overrides={
        "device_processor": {
            "device": str(device)
        },
        # Isto mapiranje koje si koristila tokom treninga
        "rename_observations_processor": {
            "rename_map": {
                "observation.image": "observation.images.camera1"
            }
        },
    },
)


# ---------------------------------------------------------
# 5. Učitavanje PushT dataseta
# ---------------------------------------------------------

dataset = LeRobotDataset(
    "lerobot/pusht",
    video_backend="pyav",
)


# ---------------------------------------------------------
# 6. Biramo prvi frame iz prve epizode
# ---------------------------------------------------------

episode_index = 0

from_idx = int(
    dataset.meta.episodes["dataset_from_index"][episode_index]
)
to_idx = int(
    dataset.meta.episodes["dataset_to_index"][episode_index]
)

frame_index = from_idx
frame = dict(dataset[frame_index])

print(f"Epizoda: {episode_index}")
print(f"Opseg frameova: {from_idx}–{to_idx - 1}")
print(f"Odabrani frame: {frame_index}")
print(f"Ključevi pre obrade: {list(frame.keys())}")

if "task" in frame:
    print(f"Instrukcija: {frame['task']}")


# Sačuvaj demonstriranu akciju pre normalizacije
recorded_action = frame["action"].detach().cpu().clone()


# ---------------------------------------------------------
# 7. Preprocessing
# ---------------------------------------------------------

batch = preprocess(frame)

print(f"Ključevi posle obrade: {list(batch.keys())}")


# ---------------------------------------------------------
# 8. Inferencija
# ---------------------------------------------------------

policy.reset()                      #cisti prethodni chunk
torch.manual_seed(0)

agent_position = frame["observation.state"][:2].cpu()

print("\nPrvih 30 akcija iz action chunka:")

with torch.inference_mode():
    for i in range(min(policy.config.n_action_steps, 30)):
        action = policy.select_action(batch)
        action = postprocess(action).squeeze(0).cpu()

        movement = action - agent_position

        print(
            f"{i:02d}: akcija={action.numpy()}, "
            f"pomak={movement.numpy()}"
        )

"""

# ---------------------------------------------------------
# 9. Prikaz slike nad kojom je izvršena inferencija
# ---------------------------------------------------------

image = frame["observation.image"].detach().cpu()

# LeRobot daje sliku kao [C, H, W], a matplotlib očekuje [H, W, C]
image = image.permute(1, 2, 0)

agent_position = frame["observation.state"][:2].cpu()

print("Trenutna pozicija agenta:", agent_position.numpy())
print("Pokret modela:", (predicted_action - agent_position).numpy())
print("Pokret demonstratora:", (recorded_action - agent_position).numpy())

if image.dtype == torch.uint8:
    image = image.float() / 255.0

plt.figure(figsize=(6, 6))
plt.imshow(image.numpy())
plt.title(
    f"Predikcija: {predicted_action.numpy()}\n"
    f"Dataset akcija: {recorded_action.numpy()}"
)
plt.axis("off")
plt.tight_layout()
plt.show()
"""