from pathlib import Path
import uuid
from crystal_env import CrystalEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from callback import TensorboardCallback
from stable_baselines3 import PPO
import os
import re
import glob

# TensorBoard log directory
log_dir = "./tensorboard/"
# Create log directory if it doesn't exist
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


def make_env(rank, env_conf, seed=0):
    def _init():
        env = CrystalEnv(env_conf)
        env.reset(seed=(seed + rank))
        return env

    set_random_seed(seed)
    return _init


def find_latest_session_and_poke():
    all_folders = os.listdir()
    session_folders = [
        folder for folder in all_folders if re.match(r"session_[0-9a-fA-F]{8}", folder)
    ]

    most_recent_time = 0
    most_recent_session = None
    most_recent_poke_file = None

    for session_folder in session_folders:
        poke_files = glob.glob(f"{session_folder}/poke_crystal_*_steps.zip")
        for poke_file in poke_files:
            mod_time = os.path.getmtime(poke_file)
            if mod_time > most_recent_time:
                most_recent_time = mod_time
                most_recent_session = session_folder
                most_recent_poke_file = poke_file[
                    :-4
                ]  # Remove '.zip' from the filename

    return most_recent_session, most_recent_poke_file


if __name__ == "__main__":
    ep_length = 2048 * 10
    sess_id = str(uuid.uuid4())[:8]
    sess_path = Path(f"session_{sess_id}")

    env_config = {
        "n_steps": ep_length,
        "extra_buttons": False,
        "session_path": sess_path,
        "explore_weight": 3,
        "reward_scale": 3,
    }

    learn_steps = 6
    num_cpu = 1

    # session_folder, latest_poke_file = find_latest_session_and_poke()
    # print("\n" + latest_poke_file)
    latest_poke_file = False

    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])

    checkpoint_callback = CheckpointCallback(
        save_freq=ep_length, save_path=sess_path, name_prefix="poke_crystal"
    )

    callbacks = [checkpoint_callback, TensorboardCallback()]

    if latest_poke_file:
        model = PPO.load(latest_poke_file, env=env, tensorboard_log=log_dir)
        model.n_steps = ep_length
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()

    else:
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            n_steps=ep_length // 8,
            batch_size=128,
            n_epochs=3,
            gamma=0.998,
            tensorboard_log=log_dir,
        )

    for i in range(learn_steps):
        model.learn(
            total_timesteps=(ep_length * num_cpu),
            callback=CallbackList(callbacks),
        )
