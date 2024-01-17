from pyboy import PyBoy

# Abrindo o jogo Pokemon Crystal
pyboy = PyBoy("../PokemonCrystal.gbc")
# Carregando o Save Inicial
save = open("../PokemonCrystal.gbc.state", "rb")
pyboy.load_state(save)
# Inicializando a emulação
pyboy.set_emulation_speed(5)


def read_bcd(num):
    return 10 * ((num >> 4) & 0x0F) + (num & 0x0F)


def bit_count(bits):
    return bin(bits).count("1")


def read_seen_poke():
    return sum(
        [
            bit_count(pyboy.get_memory_value(i))
            for i in range(
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
            )
        ]
    )


def read_money():
    money1 = pyboy.get_memory_value(0xD84E)  # 0
    money2 = pyboy.get_memory_value(0xD84F)  # 11
    money3 = pyboy.get_memory_value(0xD850)  # 184
    return (
        (100 * 100 * read_bcd(money1))
        + (100 * read_bcd(money2))
        + int(16.11 * read_bcd(money3))
    )


# Loop principal para emular o jogo
while not pyboy.tick():
    # Obtendo os valores das flags
    x_pos = pyboy.get_memory_value(0xDCB8)
    y_pos = pyboy.get_memory_value(0xDCB7)
    map_n = pyboy.get_memory_value(0xDCB6)
    map_bank = pyboy.get_memory_value(0xDCB5)
    warp_n = pyboy.get_memory_value(0xDCB4)

    j_badges = pyboy.get_memory_value(0xD857)
    k_badges = pyboy.get_memory_value(0xD858)

    pt_num = pyboy.get_memory_value(0xDCD7)

    room = pyboy.get_memory_value(0xD148)

    event_flags = pyboy.get_memory_value(0xDA72)

    levels = [
        pyboy.get_memory_value(a)
        for a in [0xDCFE, 0xDD2E, 0xDD5E, 0xDD8E, 0xDDBE, 0xDDEE]
    ]

    money = read_money()

    # money1 = read_bcd(moneyb1)
    # money2 = read_bcd(moneyb2)
    # money3 = read_bcd(moneyb3)
    # Imprimindo os valores das flags
    print(f"Posição X 0xDCB8: {x_pos}")
    print(f"Posição Y 0xDCB7: {y_pos}")
    # Map Values:
    # 3 - Route 29
    # 4 - New Bark City
    # 5 - Elm's Lab
    # 6 - Player's House
    # 7 - Player's Room
    # 8 - Neighbor's House - New Bark City
    # 9 - Wrap 1 - Elm's House
    # 9 - Wrap 2 - Route 46
    # 13 - Guard's House - Route 29 <--> Route 46
    print(f"Número do Mapa 0xDCB6: {map_n}")
    print(f"Número do Wrap 0xDCB4: {warp_n}")
    print(f"Map Bank 0xDCB5: {map_n}")
    print(f"Event Flags 0xDA72: {event_flags}")
    print(f"Room Player is in 0xD148: {room}")
    print(f"Johto Badges 0xD857: {j_badges}")
    print(f"Kanto Badges 0xD858: {k_badges}")
    # print(f"Dinheiro Bit 1 0xD84E: {money1}")
    # print(f"Dinheiro Bit 2 0xD84F: {money2}")
    # print(f"Dinheiro Bit 3 0xD850: {money3}")
    print(f"Dinheiro Def Read Money: {money}")
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
pyboy.stop()