"""import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import sys
from pathlib import Path

# Dodavanje src foldera
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import torch
import numpy as np
import gymnasium as gym
import gym_pusht

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

print("=== POKRETANJE SMOLVLA MODELA U SIMULACIJI ===")

# 1. Provera uređaja
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[1/3] Koristimo uređaj: {device}")

# 2. Učitavanje modela
print("[2/3] Učitavanje SmolVLA modela...")
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").to(device)
policy.eval()
print("       SmolVLA model uspešno učitan!")

# 3. Pokretanje 3D Simulacije (PushT zadatak)
print("[3/3] Otvaranje 3D simulacionog okruženja...")
env = gym.make("gym_pusht/PushT-v0", render_mode="human")
obs, info = env.reset()

print("\nSimulacija je pokrenuta!")
print("Pritisni Ctrl+C u terminalu da zaustaviš simulaciju.")

for step in range(200):
    env.render()
    
    # Nasumičan korak za testiranje okruženja
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, _ = env.reset()

env.close()
print("Simulacija uspešno završena!")"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import gymnasium as gym
import gymnasium_robotics

print("=== POKRETANJE MUJOCO 3D SIMULACIJE ===")

# Koristimo v4 verziju okruženja
env = gym.make("FetchPickAndPlace-v4", render_mode="human")
obs, info = env.reset()

print("MuJoCo 3D okruženje je uspešno učitano!")
print("Možeš koristiti miš za rotiranje i zumiranje kamere u 3D prozoru.\n")

for step in range(5000):
    env.render()
    
    # Nasumične akcije robotičke ruke
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
print("Simulacija je uspešno završena.")