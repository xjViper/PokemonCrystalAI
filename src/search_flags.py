from pyboy import PyBoy
import os

# Abrindo o jogo Pokemon Crystal
pyboy = PyBoy("../PokemonCrystal.gbc")
# Carregando o Save Inicial
save = open("../PokemonCrystal.gbc.state", "rb")
pyboy.load_state(save)
# Inicializando a emulação
pyboy.set_emulation_speed(6)
pyboy.set_memory_value(0xD4B7, 20)


def bit_count(bits):
    return bin(bits).count("1")


def clear_terminal():
    if os.name == "nt":
        os.system("cls")
    else:
        os.system("clear")


def read_seen_poke():
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
    return sum([bit_count(pyboy.get_memory_value(i)) for i in addr])


prev_values = {
    "x_pos": None,
    "y_pos": None,
    "N_map_n": None,
    "S_map_n": None,
    "W_map_n": None,
    "E_map_n": None,
    "map_n": None,
    "map_bank": None,
    "room": None,
    "j_badges": None,
    "k_badges": None,
    "seen": None,
    "pt_num": None,
    "levels": [None, None, None, None, None, None],
}


# Loop principal para emular o jogo
while not pyboy.tick():
    # Obtendo os valores das flags
    x_pos = pyboy.get_memory_value(0xDCB8)
    y_pos = pyboy.get_memory_value(0xDCB7)
    map_n = pyboy.get_memory_value(0xDCB6)
    map_bank = pyboy.get_memory_value(0xDCB5)
    warp_n = pyboy.get_memory_value(0xDCB4)

    N_map_n = pyboy.get_memory_value(0xD1AA)
    S_map_n = pyboy.get_memory_value(0xD1B6)
    W_map_n = pyboy.get_memory_value(0xD1C2)
    E_map_n = pyboy.get_memory_value(0xD1CE)

    j_badges = pyboy.get_memory_value(0xD857)
    k_badges = pyboy.get_memory_value(0xD858)

    pt_num = pyboy.get_memory_value(0xDCD7)

    room = pyboy.get_memory_value(0xD148)

    event_flags = pyboy.get_memory_value(0xDA72)

    levels = [
        pyboy.get_memory_value(a)
        for a in [0xDCFE, 0xDD2E, 0xDD5E, 0xDD8E, 0xDDBE, 0xDDEE]
    ]

    hour = pyboy.get_memory_value(0xD4B7)
    min = pyboy.get_memory_value(0xD4B8)

    seen = read_seen_poke()

    if any(
        x != prev_values[key]
        for key, x in {
            "x_pos": x_pos,
            "y_pos": y_pos,
            "N_map_n": N_map_n,
            "S_map_n": S_map_n,
            "W_map_n": W_map_n,
            "E_map_n": E_map_n,
            "map_n": map_n,
            "map_bank": map_bank,
            "room": room,
            "j_badges": j_badges,
            "k_badges": k_badges,
            "seen": seen,
            "pt_num": pt_num,
            "levels": levels,
        }.items()
    ):
        # Atualizando os valores anteriores
        prev_values["x_pos"] = x_pos
        prev_values["y_pos"] = y_pos
        prev_values["N_map_n"] = N_map_n
        prev_values["S_map_n"] = S_map_n
        prev_values["W_map_n"] = W_map_n
        prev_values["E_map_n"] = E_map_n
        prev_values["map_n"] = map_n
        prev_values["map_bank"] = map_bank
        prev_values["room"] = room
        prev_values["j_badges"] = j_badges
        prev_values["k_badges"] = k_badges
        prev_values["seen"] = seen
        prev_values["pt_num"] = pt_num
        prev_values["levels"] = levels

        clear_terminal()

        # Imprimindo os valores das flags
        print(f"Posição X 0xDCB8: {x_pos}")
        print(f"Posição Y 0xDCB7: {y_pos}")

        # print(f"Room Player is in 0xD148: {room}")
        # print(f"Número do Wrap 0xDCB4: {warp_n}")
        print(f"Número do Mapa 0xDCB6: {map_n}")
        print(f"Map Bank 0xDCB5: {map_bank}")

        print(f"Número do Mapa Conectado ao Norte 0xD1AA: {N_map_n}")
        print(f"Número do Mapa Conectado ao Sul 0xD1B5: {S_map_n}")
        print(f"Número do Mapa Conectado ao Oeste 0xD1C1: {W_map_n}")
        print(f"Número do Mapa Conectado ao Leste 0xD1CD: {E_map_n}")

        # print(f"Event Flags 0xDA72: {event_flags}")
        print(f"Johto Badges 0xD857: {j_badges}")
        print(f"Kanto Badges 0xD858: {k_badges}")

        print(f"Time 0xD4B7:0xD4B8: {hour}:{min}")

        print(f"Pokes Vistos Def Read Seen Poke: {seen}")
        print(f"Party Size: {pt_num}")

        print(
            f"""Nível dos Pokes por Slot: 
            1 - 0xDCFE: {levels[0]}
            2 - 0xDD2E: {levels[1]}
            3 - 0xDD5E: {levels[2]}
            4 - 0xDD8E: {levels[3]}
            5 - 0xDDBE: {levels[4]}
            6 - 0xDDEE: {levels[5]}"""
        )

# Encerrando a emulação
clear_terminal()
pyboy.stop()
