from pyboy import PyBoy

# Abrindo o jogo Pokemon Crystal
pyboy = PyBoy("../PokemonCrystal.gbc")
# Carregando o Save Inicial
save = open("../PokemonCrystal.gbc.state", "rb")
pyboy.load_state(save)
# Inicializando a emulação
pyboy.set_emulation_speed(5)

# Loop principal para emular o jogo
while not pyboy.tick():
    # Obtendo os valores das flags
    x_pos = pyboy.get_memory_value(0xDCB8)
    y_pos = pyboy.get_memory_value(0xDCB7)
    map_n = pyboy.get_memory_value(0xDCB6)
    map_bank = pyboy.get_memory_value(0xDCB5)
    levels = [
        pyboy.get_memory_value(a)
        for a in [0xDCFE, 0xDD2E, 0xDD5E, 0xDD8E, 0xDDBE, 0xDDEE]
    ]
    money1 = pyboy.get_memory_value(0xD84E)
    money2 = pyboy.get_memory_value(0xD850)

    # Imprimindo os valores das flags
    print(f"Posição X 0xDCB8: {x_pos}")
    print(f"Posição Y 0xDCB7: {y_pos}")
    print(f"Número do Mapa 0xDCB6: {map_n}")
    print(f"Map Bank 0xDCB5: {map_n}")
    print(f"Dinheiro Bit 1 0xD84E: {money1}")
    print(f"Dinheiro Bit 2 0xD850: {money2}")
    print(
        f"""Nível dos Pokes por Slot: 
        1 - 0xDCFE: {levels[0]} / 
        2 - 0xDD2E: {levels[1]} / 
        3 - 0xDD5E: {levels[2]} / 
        4 - 0xDD8E: {levels[3]} / 
        5 - 0xDDBE: {levels[4]} / 
        6 - 0xDDEE: {levels[5]}"""
    )

# Encerrando a emulação
pyboy.stop()
