import os
import time
import numpy as np
import random
from stable_baselines3 import PPO, SAC, TD3
from env import FixedWingUAVEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

def train_agent(algorithm="PPO", timesteps=300_000, model_name="agent", seed=42):
    # Set seeds for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    
    # unique run id for logs and saved model
    run_id = time.strftime("%Y%m%d-%H%M%S")
    tb_name = f"{model_name}_{algorithm}_waypoint_{run_id}"

    if algorithm == "PPO":
        # PPO with VecNormalize for stable learning
        make_env = lambda: Monitor(FixedWingUAVEnv())
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
            seed=seed,
        )
    elif algorithm == "SAC":
        # Vectorized env with normalization for more stable learning
        make_env = lambda: Monitor(FixedWingUAVEnv())
        venv = DummyVecEnv([make_env])
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
        model = SAC(
            policy="MlpPolicy",
            env=venv,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=100_000,
            batch_size=128,
            gamma=0.99,
            tau=0.01,
            ent_coef='auto',
            learning_starts=10_000,
            tensorboard_log="logs/",
            seed=seed,
        )
    elif algorithm == "TD3":
        # Vectorized env with normalization for stability (like SAC)
        make_env = lambda: Monitor(FixedWingUAVEnv())
        venv = DummyVecEnv([make_env])
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
        model = TD3(
            policy="MlpPolicy",
            env=venv,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=100_000,
            batch_size=128,
            gamma=0.99,
            tau=0.01,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            tensorboard_log="logs/",
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    model.learn(total_timesteps=timesteps, tb_log_name=tb_name)
    return model, run_id, algorithm

def train_all_experiments(seed=42):
    """Train all algorithms for comparison"""
    algorithms = ["PPO", "SAC", "TD3"]
    timesteps = 500_000
    
    # Set global seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    
    results = []
    
    for algorithm in algorithms:
        print(f"\n{'='*60}")
        print(f"Training {algorithm}...")
        print(f"{'='*60}")
        
        try:
            model, run_id, alg = train_agent(
                algorithm=algorithm, 
                timesteps=timesteps, 
                model_name="fixedwing",
                seed=seed
            )
            
            model_path = f"trained_models/{alg.lower()}_fixedwing_waypoint_{run_id}"
            model.save(model_path)
            # Save VecNormalize statistics for evaluation/finetuning
            try:
                vec_env = model.get_env()
                if isinstance(vec_env, VecNormalize) or hasattr(vec_env, "save"):
                    vec_env.save(model_path + "_vn.pkl")
                    print(f"✅ Saved VecNormalize stats to {model_path}_vn.pkl")
            except Exception as e_save:
                print(f"⚠️ Could not save VecNormalize stats: {e_save}")
            
            print(f"✅ Saved {algorithm} model to {model_path}.zip")
            results.append((algorithm, model_path))
            
        except Exception as e:
            print(f"❌ Failed to train {algorithm}: {e}")
            results.append((algorithm, f"FAILED: {e}"))
    
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    for alg, result in results:
        status = "✅ SUCCESS" if not result.startswith("FAILED") else "❌ FAILED"
        print(f"{alg:>5}: {status}")
    
    return results

def main():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("trained_models", exist_ok=True)

    # Train all combinations with fixed seed for reproducibility
    seed = 42
    print(f"🌱 Using seed {seed} for reproducible training")
    results = train_all_experiments(seed=seed)
    
    print("\nAll training experiments complete!")
    print("Models saved for comparison between algorithms.")

if __name__ == "__main__":
    main()