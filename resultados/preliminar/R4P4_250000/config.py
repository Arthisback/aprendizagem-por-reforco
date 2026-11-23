class Config:
    # Reward configs
    STEP_REWARD = -1 #valor inteiro
    WALL_REWARD = -10 #valor inteiro
    HOLE_REWARD = -500 #valor inteiro
    GOAL_REWARD = 500 #valor inteiro

    # Q-Learning configs
    ALPHA = 0.2 #valor no intervalo (0, 1]
    GAMMA = 0.99 #valor no intervalo (0, 1]
    EPSILON = 1.0 #valor no intervalo (0, 1]
    ALPHA_DECAY = 0.995 #valor no intervalo (0, 1]
    EPSILON_DECAY = 0.99 #valor no intervalo (0, 1]
    # O valor de DECAY_STEP determina a cada quantos episódios completos o ALPHA e o EPSILON
    # serão serão multiplicados pelos seus respectivos fatores de decaimento (ALPHA_DECAY e EPSILON_DECAY).
    # O progresso do treinamento é impresso a cada DECAY_STEP episódios completos.
    DECAY_STEP = 100 #valor inteiro positivo
    
    # Train or Test Model/Agent
    TOTAL_TIMESTEPS = 250000 #valor inteiro positivo
    TEST_SIMMULATION_NUMBER = 1000 #valor inteiro positivo
    TRAIN = True 
    TEST = True 
    RENDERS = False 

def STEP_REWARD() -> int:
    return Config.STEP_REWARD

def WALL_REWARD() -> int:
    return Config.WALL_REWARD

def HOLE_REWARD() -> int:
    return Config.HOLE_REWARD

def GOAL_REWARD() -> int:
    return Config.GOAL_REWARD

def ALPHA() -> float:
    return Config.ALPHA

def GAMMA() -> float:
    return Config.GAMMA

def EPSILON() -> float:
    return Config.EPSILON

def TOTAL_TIMESTEPS() -> int:
    return Config.TOTAL_TIMESTEPS

def ALPHA_DECAY() -> float:
    return Config.ALPHA_DECAY

def EPSILON_DECAY() -> float:
    return Config.EPSILON_DECAY

def DECAY_STEP() -> int:
    return Config.DECAY_STEP

def TEST_SIMMULATION_NUMBER() -> int:
    return Config.TEST_SIMMULATION_NUMBER

def TRAIN() -> bool:
    return Config.TRAIN

def TEST() -> bool:
    return Config.TEST

def RENDERS() -> bool:
    return Config.RENDERS