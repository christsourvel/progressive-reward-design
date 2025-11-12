import os
import time
import numpy as np
import random
import torch
from stable_baselines3 import PPO, SAC, TD3
from env import FixedWingUAVEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

def train_agent(algorithm="PPO", config="waypoint", timesteps=300_000, model_name="agent"):
    # unique run id for logs and saved model
    run_id = time.strftime("%Y%m%d-%H%M%S")
    tb_name = f"{model_name}_{algorithm}_{config}_{run_id}"

    if algorithm == "PPO":
        # Vectorized env with normalization for more stable learning
        def make_env():
            env = FixedWingUAVEnv(config=config)
            env.reset(seed=SEED)
            return Monitor(env)
        venv = DummyVecEnv([make_env])
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
        model = PPO(
            policy="MlpPolicy",
            env=venv,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            ent_coef=0.01,
            clip_range=0.2,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log="logs/",
            seed=SEED,
        )
    elif algorithm == "SAC":
        # Vectorized env with normalization for more stable learning
        def make_env():
            env = FixedWingUAVEnv(config=config)
            env.reset(seed=SEED)
            return Monitor(env)
        venv = DummyVecEnv([make_env])
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
        model = SAC(
            policy="MlpPolicy",
            env=venv,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            ent_coef='auto',
            learning_starts=10_000,
            tensorboard_log="logs/",
            seed=SEED,
        )
    elif algorithm == "TD3":
        # Vectorized env with normalization for stability (like SAC)
        def make_env():
            env = FixedWingUAVEnv(config=config)
            env.reset(seed=SEED)
            return Monitor(env)
        venv = DummyVecEnv([make_env])
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
        model = TD3(
            policy="MlpPolicy",
            env=venv,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=1000000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            tensorboard_log="logs/",
            seed=SEED,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    model.learn(total_timesteps=timesteps, tb_log_name=tb_name)
    return model, run_id, algorithm, config

def train_all_experiments():
    """Train all algorithm x config combinations for comparison"""
    algorithms = ["PPO", "SAC", "TD3"]
    configs = ["waypoint", "heuristic"]
    timesteps = 500_000
    
    results = []
    
    for algorithm in algorithms:
        for config in configs:
            print(f"\n{'='*60}")
            print(f"Training {algorithm} on {config.upper()} configuration...")
            print(f"{'='*60}")
            
            try:
                model, run_id, alg, cfg = train_agent(
                    algorithm=algorithm, 
                    config=config, 
                    timesteps=timesteps, 
                    model_name="fixedwing"
                )
                
                model_path = f"trained_models/{alg.lower()}_fixedwing_{cfg}_{run_id}"
                model.save(model_path)
                # Save VecNormalize statistics for evaluation/finetuning
                try:
                    vec_env = model.get_env()
                    if isinstance(vec_env, VecNormalize) or hasattr(vec_env, "save"):
                        vec_env.save(model_path + "_vn.pkl")
                except Exception as e_save:
                    print(f"⚠️ Could not save VecNormalize stats: {e_save}")
                
                print(f"✅ Saved {algorithm} {config} model to {model_path}.zip")
                results.append((algorithm, config, model_path))
                
            except Exception as e:
                print(f"❌ Failed to train {algorithm} {config}: {e}")
                results.append((algorithm, config, f"FAILED: {e}"))
    
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    for alg, cfg, result in results:
        status = "✅ SUCCESS" if not result.startswith("FAILED") else "❌ FAILED"
        print(f"{alg:>5} {cfg:>6}: {status}")
    
    return results

def main():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("trained_models", exist_ok=True)

    # Train all combinations
    results = train_all_experiments()
    
    print("\nAll training experiments complete!")
    print("Models saved for comparison between algorithms and configurations.")

if __name__ == "__main__":
    main()