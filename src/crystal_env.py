import sys
import uuid
import os
from math import floor, sqrt
import json
from pathlib import Path

import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt
from skimage.transform import resize
from pyboy import PyBoy
from pyboy.logger import log_level
import hnswlib
import mediapy as media
import pandas as pd

from gymnasium import Env, spaces
from pyboy.utils import WindowEvent


class CrystalEnv(Env):
    def __init__(self, n_steps, extra_buttons, session_path):
        self.extra_buttons = extra_buttons
        self.s_path = session_path
        self.s_path.mkdir(exist_ok=True)
        self.instance_id = str(uuid.uuid4())[:8]
        self.similar_frame_dist = 2_000_000.0
        self.init_state = "../PokemonCrystal.gbc.state"

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
        ]

        if self.extra_buttons:
            self.valid_actions.extend(
                [WindowEvent.PRESS_BUTTON_START, WindowEvent.PASS]
            )

        self.release_arrow = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
        ]

        self.release_button = [
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
        ]

        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(128, 40, 3), dtype=np.uint8
        )

        self.pyboy = PyBoy(
            "../PokemonCrystal.gbc",
            debugging=False,
            disable_input=False,
            window_type="SDL2",
            hide_window="--quiet" in sys.argv,
        )
        self.max_steps = n_steps
        self.print_rewards = True
        self.act_freq = 24
        self.reward_scale = 3
        self.explore_weight = 3
        self.reset_count = 0
        self.all_runs = []

        self.screen = self.pyboy.botsupport_manager().screen()
        self.pyboy.set_emulation_speed(6)
        self.reset()

    def reset(self, seed=None):
        self.seed = seed

        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)

        self.init_knn()

        self.recent_memory = np.zeros((40 * 8, 3), dtype=np.uint8)

        self.recent_frames = np.zeros(
            (
                3,
                36,
                40,
                3,
            ),
            dtype=np.uint8,
        )

        self.agent_stats = []
        self.levels_satisfied = False
        self.base_explore = 0
        self.max_opponent_level = 0
        self.max_event_rew = 0
        self.max_level_rew = 0
        self.last_health = 1
        self.total_healing_rew = 0
        self.died_count = 0
        self.party_size = 0
        self.step_count = 0
        self.progress_reward = self.get_game_state_reward()
        self.total_reward = sum([val for _, val in self.progress_reward.items()])
        self.reset_count += 1
        return self.render(), {}

    def render(self, reduce_res=True, add_memory=True, update_mem=True):
        game_pixels_render = self.screen.screen_ndarray()  # (144, 160, 3)
        if reduce_res:
            game_pixels_render = (255 * resize(game_pixels_render, (36, 40, 3))).astype(
                np.uint8
            )
            if update_mem:
                self.recent_frames[0] = game_pixels_render
            if add_memory:
                pad = np.zeros(shape=(2, 40, 3), dtype=np.uint8)
                game_pixels_render = np.concatenate(
                    (
                        self.create_exploration_memory(),
                        pad,
                        rearrange(self.recent_memory, "(w h) c -> h w c", h=8),
                        pad,
                        rearrange(self.recent_frames, "f h w c -> (f h) w c"),
                    ),
                    axis=0,
                )
        return game_pixels_render

    def step(self, action):
        self.run_action_on_emulator(action)
        self.append_agent_stats(action)

        self.recent_frames = np.roll(self.recent_frames, 1, axis=0)
        obs_memory = self.render()

        # trim off memory from frame for knn index
        frame_start = 2 * (8 + 2)
        obs_flat = (
            obs_memory[frame_start : frame_start + 36, ...].flatten().astype(np.float32)
        )

        self.update_frame_knn_index(obs_flat)

        self.update_heal_reward()
        self.party_size = self.read_m(0xDCD7)

        new_reward, new_prog = self.update_reward()

        self.last_health = self.read_hp_fraction()

        # shift over short term reward memory
        self.recent_memory = np.roll(self.recent_memory, 3)
        self.recent_memory[0, 0] = min(new_prog[0] * 64, 255)
        self.recent_memory[0, 1] = min(new_prog[1] * 64, 255)
        self.recent_memory[0, 2] = min(new_prog[2] * 128, 255)

        step_limit_reached = self.check_if_done()

        self.save_and_print_info(step_limit_reached, obs_memory)

        self.step_count += 1

        return obs_memory, new_reward * 0.1, False, step_limit_reached, {}

    def check_if_done(self):
        done = self.step_count >= self.max_steps
        return done

    def group_rewards(self):
        prog = self.progress_reward
        return (
            prog["level"] * 100 / self.reward_scale,
            self.read_hp_fraction() * 2000,
            prog["explore"] * 150 / (self.explore_weight * self.reward_scale),
        )

    def update_reward(self):
        # compute reward
        old_prog = self.group_rewards()
        self.progress_reward = self.get_game_state_reward()
        new_prog = self.group_rewards()
        new_total = sum(
            [val for _, val in self.progress_reward.items()]
        )  # sqrt(self.explore_reward * self.progress_reward)
        new_step = new_total - self.total_reward
        if new_step < 0 and self.read_hp_fraction() > 0:
            # print(f'\n\nreward went down! {self.progress_reward}\n\n')
            self.save_screenshot("neg_reward")

        self.total_reward = new_total
        return (
            new_step,
            (
                new_prog[0] - old_prog[0],
                new_prog[1] - old_prog[1],
                new_prog[2] - old_prog[2],
            ),
        )

    def init_knn(self):
        # Declaring index
        self.knn_index = hnswlib.Index(
            space="l2", dim=4320
        )  # possible options are l2, cosine or ip
        # Initing index - the maximum number of elements should be known beforehand
        self.knn_index.init_index(max_elements=20000, ef_construction=100, M=16)

    def append_agent_stats(self, action):
        x_pos = self.read_m(0xD362)
        y_pos = self.read_m(0xD361)
        map_n = self.read_m(0xD35E)
        levels = [
            self.read_m(a) for a in [0xDCFE, 0xDD2E, 0xDD5E, 0xDD8E, 0xDDBE, 0xDDEE]
        ]

        expl = ("frames", self.knn_index.get_current_count())
        self.agent_stats.append(
            {
                "step": self.step_count,
                "x": x_pos,
                "y": y_pos,
                "map": map_n,
                "map_location": self.get_map_location(map_n),
                "last_action": action,
                "pcount": self.read_m(0xDCD7),
                "levels": levels,
                "levels_sum": sum(levels),
                "ptypes": self.read_party(),
                "hp": self.read_hp_fraction(),
                expl[0]: expl[1],
                "deaths": self.died_count,
                "badge": self.get_badges(),
                # "event": self.progress_reward["event"],
                "healr": self.total_healing_rew,
            }
        )

    def read_party(self):
        return [
            self.read_m(addr)
            for addr in [0xDCD8, 0xDCD9, 0xDCDA, 0xDCDB, 0xDCDC, 0xDCDD]
        ]

    def get_map_location(self, map_idx):
        map_locations = {
            "01": "New Bark Town",
            "02": "Route 29",
            "03": "Cherrygrove City",
            "04": "Route 30",
            "05": "Route 31",
            "06": "Violet City",
            "07": "Sprout Tower",
            "08": "Route 32",
            "09": "Ruins of Alph",
            "0A": "Union Cave",
            "0B": "Route 33",
            "0C": "Azalea Town",
            "0D": "Slowpoke Well",
            "0E": "Ilex Forest",
            "0F": "Route 34",
            "10": "Goldenrod City",
            "11": "Radio Tower",
            "12": "Route 35",
            "13": "National Park",
            "14": "Route 36",
            "15": "Route 37",
            "16": "Ecruteak City",
            "17": "Tin Tower",
            "18": "Burned Tower",
            "19": "Route 38",
            "1A": "Route 39",
            "1B": "Olivine City",
            "1C": "Lighthouse",
            "1D": "Battle Tower",
            "1E": "Route 40",
            "1F": "Whirl Islands",
            "20": "Route 41",
            "21": "Cianwood City",
            "22": "Route 42",
            "23": "Mt.Mortar",
            "24": "Mahogany Town",
            "25": "Route 43",
            "26": "Lake of Rage",
            "27": "Route 44",
            "28": "Ice Path",
            "29": "Blackthorn City",
            "2A": "Dragons Den",
            "2B": "Route 45",
            "2C": "Dark Cave",
            "2D": "Route 46",
            "2E": "Silver Cave",
            "2F": "Pallet Town",
            "30": "Route 1",
            "31": "Viridian City",
            "32": "Route 2",
            "33": "Pewter City",
            "34": "Route 3",
            "35": "Mt.Moon",
            "36": "Route 4",
            "37": "Cerulean City",
            "38": "Route 24",
            "39": "Route 25",
            "3A": "Route 5",
            "3B": "Underground",
            "3C": "Route 6",
            "3D": "Vermilion City",
            "3E": "Diglett's Cave",
            "3F": "Route 7",
            "40": "Route 8",
            "41": "Route 9",
            "42": "Rock Tunnel",
            "43": "Route 10",
            "44": "Power Plant",
            "45": "Lavender Town",
            "46": "Lav Radio Tower",
            "47": "Celadon City",
            "48": "Saffron City",
            "49": "Route 11",
            "4A": "Route 12",
            "4B": "Route 13",
            "4C": "Route 14",
            "4D": "Route 15",
            "4E": "Route 16",
            "4F": "Route 17",
            "50": "Route 18",
            "51": "Fuchsia City",
            "52": "Route 19",
            "53": "Route 20",
            "54": "Seafoam Islands",
            "55": "Cinnabar Island",
            "56": "Route 21",
            "57": "Route 22",
            "58": "Victory Road",
            "59": "Route 23",
            "5A": "Indigo Plateau",
            "5B": "Route 26",
            "5C": "Route 27",
            "5D": "Tohjo Falls",
            "5E": "Route 28",
            "5F": "New Bark Town (Fast Ship)",
        }
        if map_idx in map_locations.keys():
            return map_locations[map_idx]
        else:
            return "Unknown Location"

    def run_action_on_emulator(self, action):
        # press button then release after some steps
        self.pyboy.send_input(self.valid_actions[action])
        # disable rendering when we don't need it
        for i in range(self.act_freq):
            # release action, so they are stateless
            if i == 8:
                if action < 4:
                    # release arrow
                    self.pyboy.send_input(self.release_arrow[action])
                if action > 3 and action < 6:
                    # release button
                    self.pyboy.send_input(self.release_button[action - 4])
                if self.valid_actions[action] == WindowEvent.PRESS_BUTTON_START:
                    self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
            if i == self.act_freq - 1:
                self.pyboy._rendering(True)
            self.pyboy.tick()

    def get_game_state_reward(self, print_stats=False):
        seen_poke_count = self.read_seen_poke()
        state_scores = {
            # "event": self.reward_scale * self.update_max_event_rew(),
            #'party_xp': self.reward_scale*0.1*sum(poke_xps),
            "level": self.reward_scale * self.get_levels_reward(),
            "heal": self.reward_scale * self.total_healing_rew,
            # "op_lvl": self.reward_scale * self.update_max_op_level(),
            "dead": self.reward_scale * -0.1 * self.died_count,
            "badge": self.reward_scale * self.get_badges() * 5,
            #'op_poke': self.reward_scale*self.max_opponent_poke * 800,
            "seen_poke": self.reward_scale * seen_poke_count * 400,
            "explore": self.reward_scale * self.get_knn_reward(),
        }

        return state_scores

    def get_badges(self):
        return sum(
            [self.bit_count(self.read_m(0xD857)), self.bit_count(self.read_m(0xD858))],
            0,
        )

    def save_screenshot(self, name):
        ss_dir = self.s_path / Path("screenshots")
        ss_dir.mkdir(exist_ok=True)
        plt.imsave(
            ss_dir
            / Path(
                f"frame{self.instance_id}_r{self.total_reward:.4f}_{self.reset_count}_{name}.jpeg"
            ),
            self.render(reduce_res=False),
        )

    def update_heal_reward(self):
        cur_health = self.read_hp_fraction()
        # if health increased and party size did not change
        if cur_health > self.last_health and self.read_m(0xDCD7) == self.party_size:
            if self.last_health > 0:
                heal_amount = cur_health - self.last_health
                if heal_amount > 0.5:
                    print(f"healed: {heal_amount}")
                    self.save_screenshot("healing")
                self.total_healing_rew += heal_amount * 4
            else:
                self.died_count += 1

    def read_m(self, addr):
        return self.pyboy.get_memory_value(addr)

    def read_bcd(self, num):
        return 10 * ((num >> 4) & 0x0F) + (num & 0x0F)

    def read_bit(self, addr, bit: int) -> bool:
        # add padding so zero will read '0b100000000' instead of '0b0'
        return bin(256 + self.read_m(addr))[-bit - 1] == "1"

    def read_seen_poke(self):
        addr = [
            0xDEB9,
            0xDEBA,
            0xDEBB,
            0xDEBC,
            0xDEBD,
            0xDEBE,
            0xDEBF,
            0xDEC0,
            0xDEC1,
            0xDEC2,
            0xDEC3,
            0xDEC4,
            0xDEC5,
            0xDEC6,
            0xDEC7,
            0xDEC8,
            0xDEC9,
            0xDECA,
            0xDECB,
            0xDECC,
            0xDECD,
            0xDECE,
            0xDECF,
            0xDED0,
            0xDED1,
            0xDED2,
            0xDED3,
            0xDED4,
            0xDED5,
            0xDED6,
            0xDED7,
            0xDED8,
        ]
        return sum([self.bit_count(self.read_m(i)) for i in addr])

    def read_hp(self, start):
        return 256 * self.read_m(start) + self.read_m(start + 1)

    def read_hp_fraction(self):
        hp_sum = sum(
            [
                self.read_hp(add)
                for add in [0xDD01, 0xDD31, 0xDD61, 0xDD91, 0xDDC1, 0xDDF1]
            ]
        )
        max_hp_sum = sum(
            [
                self.read_hp(add)
                for add in [0xDD03, 0xDD33, 0xDD63, 0xDD93, 0xDDC3, 0xDDF3]
            ]
        )
        max_hp_sum = max(max_hp_sum, 1)
        return hp_sum / max_hp_sum

    def get_levels_sum(self):
        poke_levels = [
            max(self.read_m(a) - 2, 0)
            for a in [0xDCFE, 0xDD2E, 0xDD5E, 0xDD8E, 0xDDBE, 0xDDEE]
        ]
        return max(sum(poke_levels) - 4, 0)

    def get_levels_reward(self):
        explore_thresh = 22
        scale_factor = 4
        level_sum = self.get_levels_sum()
        if level_sum < explore_thresh:
            scaled = level_sum
        else:
            scaled = (level_sum - explore_thresh) / scale_factor + explore_thresh
        self.max_level_rew = max(self.max_level_rew, scaled)
        return self.max_level_rew

    def get_knn_reward(self):
        pre_rew = self.explore_weight * 0.005
        post_rew = self.explore_weight * 0.01
        cur_size = self.knn_index.get_current_count()
        base = (self.base_explore if self.levels_satisfied else cur_size) * pre_rew
        post = (cur_size if self.levels_satisfied else 0) * post_rew
        return base + post

    def create_exploration_memory(self):
        w = 40
        h = 8

        def make_reward_channel(r_val):
            col_steps = 16
            max_r_val = (w - 1) * h * col_steps
            r_val = min(r_val, max_r_val)
            row = floor(r_val / (h * col_steps))
            memory = np.zeros(shape=(h, w), dtype=np.uint8)
            memory[:, :row] = 255
            row_covered = row * h * col_steps
            col = floor((r_val - row_covered) / col_steps)
            memory[:col, row] = 255
            col_covered = col * col_steps
            last_pixel = floor(r_val - row_covered - col_covered)
            memory[col, row] = last_pixel * (255 // col_steps)
            return memory

        level, hp, explore = self.group_rewards()
        full_memory = np.stack(
            (
                make_reward_channel(level),
                make_reward_channel(hp),
                make_reward_channel(explore),
            ),
            axis=-1,
        )

        if self.get_badges() > 0:
            full_memory[:, -1, :] = 255

        return full_memory

    def update_max_event_rew(self):
        cur_rew = self.get_all_events_reward()
        self.max_event_rew = max(cur_rew, self.max_event_rew)
        return self.max_event_rew

    def update_frame_knn_index(self, frame_vec):
        if self.get_levels_sum() >= 22 and not self.levels_satisfied:
            self.levels_satisfied = True
            self.base_explore = self.knn_index.get_current_count()
            self.init_knn()

        if self.knn_index.get_current_count() == 0:
            # if index is empty add current frame
            self.knn_index.add_items(
                frame_vec, np.array([self.knn_index.get_current_count()])
            )
        else:
            # check for nearest frame and add if current
            labels, distances = self.knn_index.knn_query(frame_vec, k=1)
            if distances[0][0] > self.similar_frame_dist:
                # print(f"distances[0][0] : {distances[0][0]} similar_frame_dist : {self.similar_frame_dist}")
                self.knn_index.add_items(
                    frame_vec, np.array([self.knn_index.get_current_count()])
                )

    def get_all_events_reward(self):
        #! Change the flags, actually the AI starts with 5205 'Event' points
        event_flags_start = 0x828
        event_flags_end = 0xA0A
        base_event_flags = 30
        return max(
            sum(
                [
                    self.bit_count(self.read_m(i))
                    for i in range(event_flags_start, event_flags_end)
                ]
            )
            - base_event_flags,
            0,
        )

    def bit_count(self, bits):
        return bin(bits).count("1")

    def save_and_print_info(self, done, obs_memory):
        if self.print_rewards:
            prog_string = f"step: {self.step_count:6d}"
            for key, val in self.progress_reward.items():
                prog_string += f" {key}: {val:5.2f}"
            prog_string += f" sum: {self.total_reward:5.2f}"
            print(f"\r{prog_string}", end="", flush=True)

        if self.step_count % 50 == 0:
            plt.imsave(
                self.s_path / Path(f"curframe_{self.instance_id}.jpeg"),
                self.render(reduce_res=False),
            )

        if self.print_rewards and done:
            print("", flush=True)

            fs_path = self.s_path / Path("final_states")
            fs_path.mkdir(exist_ok=True)
            plt.imsave(
                fs_path
                / Path(f"frame_r{self.total_reward:.4f}_{self.reset_count}_small.jpeg"),
                obs_memory,
            )
            plt.imsave(
                fs_path
                / Path(f"frame_r{self.total_reward:.4f}_{self.reset_count}_full.jpeg"),
                self.render(reduce_res=False),
            )

        if done:
            self.all_runs.append(self.progress_reward)
            with open(
                self.s_path / Path(f"all_runs_{self.instance_id}.json"), "w"
            ) as f:
                json.dump(self.all_runs, f)
            pd.DataFrame(self.agent_stats).to_csv(
                self.s_path / Path(f"agent_stats_{self.instance_id}.csv.gz"),
                compression="gzip",
                mode="a",
            )
