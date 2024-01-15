from pyboy import PyBoy

# Abrindo o jogo Pokemon Crystal
pyboy = PyBoy("../PokemonCrystal.gbc", window_type="headless")
# Carregando o Save Inicial
pyboy.load_state("../PokemonCrystal.gbc.state")
# Inicializando a emulação
pyboy.set_emulation_speed(5)

# Loop principal para emular o jogo
while not pyboy.tick():
    # Obtendo os valores das flags
    x_pos = pyboy.get_memory_value(0xD362)
    y_pos = pyboy.get_memory_value(0xD361)
    map_n = pyboy.get_memory_value(0xD35E)
    levels = [
        pyboy.get_memory_value(a)
        for a in [0xDCFE, 0xDD2E, 0xDD5E, 0xDD8E, 0xDDBE, 0xDDEE]
    ]

    # Imprimindo os valores das flags
    print(f"Posição X 0xD362: {x_pos}")
    print(f"Posição Y 0xD361: {y_pos}")
    print(f"Número do Mapa 0xD35E: {y_pos}")
    print(
        f"""Nível dos Pokes por Slot: 
        1 - 0xDCFE: {levels[0]} / 
        2 - 0xDD2E: {levels[0]} / 
        3 - 0xDD5E: {levels[0]} / 
        4 - 0xDD8E: {levels[0]} / 
        5 - 0xDDBE: {levels[0]} / 
        6 - 0xDDEE: {levels[0]}"""
    )

# Encerrando a emulação
pyboy.stop()
