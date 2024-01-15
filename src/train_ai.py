from pathlib import Path
import uuid
from crystal_env import CrystalEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from callback import TensorboardCallback
from stable_baselines3 import PPO


def make_env(rank, env_conf, seed=0):
    def _init():
        env = CrystalEnv(
            n_steps=env_conf["n_steps"],
            extra_buttons=env_conf["extra_buttons"],
            session_path=env_conf["sess_path"],
        )
        env.reset(seed=(seed + rank))
        return env

    set_random_seed(seed)
    return _init


if __name__ == "__main__":
    ep_length = 2048 * 10
    sess_id = str(uuid.uuid4())[:8]
    sess_path = Path(f"session_{sess_id}")
    learn_steps = 3

    env_config = {"n_steps": ep_length, "extra_buttons": False, "sess_path": sess_path}

    num_cpu = 3

    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])

    checkpoint_callback = CheckpointCallback(
        save_freq=ep_length, save_path=sess_path, name_prefix="poke_crystal"
    )

    callbacks = [checkpoint_callback, TensorboardCallback()]

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        n_steps=ep_length // 8,
        batch_size=128,
        n_epochs=3,
        gamma=0.998,
        tensorboard_log=sess_path,
    )
    for i in range(learn_steps):
        model.learn(
            total_timesteps=(ep_length * num_cpu),
            callback=CallbackList(callbacks),
        )
