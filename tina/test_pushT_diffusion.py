import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT)) 

import gymnasium as gym
import gym_pusht
import numpy as np
import pygame
import torch

from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.diffusion import DiffusionConfig, DiffusionPolicy  #arhitektura i naucene tezine

MODEL_ID = "lerobot/diffusion_pusht"
DATASET_ID = "lerobot/pusht"
MAX_STEPS = 300


def main():

    #INICIJALIZACIJA UREDJAJA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Koristi se: {device}")

    print("Učitavanje Diffusion Policy modela...")
    policy_config = DiffusionConfig.from_pretrained(MODEL_ID)    #arhitektura U-Net mreze
    policy_config.device = str(device)

    policy = DiffusionPolicy.from_pretrained(
        MODEL_ID,
        config=policy_config,
    )
    policy.to(device)
    policy.eval()           #prazni bafer prethodnih akcija i prelazi u zakljucan mode
    policy.reset()

    print("Učitavanje metapodataka Push-T dataseta...")
    dataset_metadata = LeRobotDatasetMetadata(DATASET_ID)   #uzima min, max, mean vrednosti iz dataseta

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        dataset_stats=dataset_metadata.stats,
    )

    #inicijalizacija okruzenja
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        observation_width=96,
        observation_height=96,
        max_episode_steps=MAX_STEPS,
    )

    observation, info = env.reset()
    policy.reset()

    # Inicijalizacija Pygame prozora za prikaz simulacije
    pygame.init()
    screen_width, screen_height = 512, 512
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("PushT Diffusion Policy Simulacija")
    clock = pygame.time.Clock()

    print("Simulacija je uspešno pokrenuta!")

    total_reward = 0.0
    step = 0

    try:
        with torch.inference_mode():
            while step < MAX_STEPS:
                # Obrada Pygame događaja (omogućava zatvaranje prozora na X)
                for event in pygame.event.get():              
                    if event.type == pygame.QUIT:
                        return

                # Prikaz trenutnog stanja na ekranu
                frame = env.render()        #uzima sliku iz okruzenja
                if frame is not None:
                    # Konverzija NumPy okvira u Pygame surface
                    surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
                    surface = pygame.transform.scale(surface, (screen_width, screen_height))
                    screen.blit(surface, (0, 0))
                    pygame.display.flip()

                # Izvlačenje piksela za model
                pixels_img = observation.get("pixels")
                if pixels_img is None:
                    pixels_img = frame

                # Priprema opservacije za LeRobot
                pixels = torch.from_numpy(pixels_img).permute(2, 0, 1).float() / 255.0 #H,W,C - C,H,W format
                agent_pos = torch.from_numpy(observation["agent_pos"]).float()

                observation_batch = {
                    "observation.image": pixels.unsqueeze(0).to(device),
                    "observation.state": agent_pos.unsqueeze(0).to(device),
                }

                observation_batch = preprocessor(observation_batch)
                action = policy.select_action(observation_batch)
                action = postprocessor(action)

                action_numpy = (
                    action
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype("float32")
                )

                observation, reward, terminated, truncated, info = env.step(action_numpy) #slanje koordinate agentu

                total_reward += float(reward)
                step += 1
                clock.tick(30)  # Ograničavanje na 30 FPS radi preglednosti

                if step % 10 == 0:
                    print(
                        f"Korak: {step:3d} | "
                        f"akcija: {action_numpy.round(2)} | "
                        f"reward: {reward:.3f}"
                    )

                if terminated or truncated:
                    print("\nEpizoda je gotova.")
                    break

    finally:
        pygame.quit()
        env.close()

    print(f"Broj izvršenih koraka: {step}")
    print(f"Ukupan reward: {total_reward:.3f}")


if __name__ == "__main__":
    main()