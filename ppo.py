"""Experimentos com o PPO (Stable-Baselines3) no ambiente do professor.

O ambiente (classe MazeEnv) é importado do RL-game.py original, sem modificação. Antes da importação o
TRAIN e o TEST do Config são desligados, para que o treino/teste do Q-Learning (que fica no nível do
módulo) não rode. As recompensas são as da configuração de recompensa do melhor modelo do Q-Learning e o
TOTAL_TIMESTEPS é o mesmo do Q-Learning.

Uso:
    python ppo.py    # treina, salva e testa os 10 modelos (depois de rodar o qlearning.py)
"""

import importlib.util
import multiprocessing
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from qlearning import RECOMPENSAS, TESTES, TOTAL_TIMESTEPS

SAIDA = Path("resultados/ppo")
PROCESSOS = 5  # modelos treinados em paralelo (limitado pela memória da máquina)
N_ENVS = 8  # cópias do ambiente usadas para coletar experiência (mesmo total de passos)

# === Configurações do PPO (o que não aparece fica com o valor padrão do Stable-Baselines3) ===
# padrão: learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
#         clip_range=0.2, ent_coef=0.0, vf_coef=0.5, rede [64, 64] para a política e para o valor
CONFIGURACOES = {
    "PPO01": dict(),
    "PPO02": dict(learning_rate=1e-3),
    "PPO03": dict(learning_rate=1e-4),
    "PPO04": dict(ent_coef=0.01),
    "PPO05": dict(gamma=0.9),
    "PPO06": dict(gamma=0.999, gae_lambda=0.98),
    "PPO07": dict(n_steps=256, batch_size=256),
    "PPO08": dict(clip_range=0.1, n_epochs=4),
    "PPO09": dict(policy_kwargs=dict(net_arch=[256, 256])),
    "PPO10": dict(n_steps=512, batch_size=256, ent_coef=0.01, gamma=0.98,
                  policy_kwargs=dict(net_arch=[128, 128])),
}


def carregar_jogo(recompensas):
    """Importa o RL-game.py original com as recompensas escolhidas, sem rodar o treino/teste do Q-Learning."""
    sys.path.insert(0, str(Path("RL-game").resolve()))
    from config import Config

    for nome, valor in recompensas.items():
        setattr(Config, nome, valor)
    Config.TRAIN = Config.TEST = Config.RENDERS = False
    spec = importlib.util.spec_from_file_location("rl_game", "RL-game/RL-game.py")
    jogo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(jogo)
    return jogo


def testar(modelo, jogo, recompensas, deterministico, episodios=TESTES):
    """Mesmo teste do RL-game.py: soma das recompensas e sucesso quando a recompensa é a do objetivo."""
    env = jogo.MazeEnv()
    total, sucesso = 0, 0
    for _ in range(episodios):
        estado, _ = env.reset()
        fim = False
        while not fim:
            acao, _ = modelo.predict(estado, deterministic=deterministico)
            estado, recompensa, fim, _, _ = env.step(int(acao))
            total += recompensa
            if recompensa == recompensas["GOAL_REWARD"]:
                sucesso += 1
    return total / episodios, 100 * sucesso / episodios


def treinar(identificador, parametros, recompensas, semente):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.env_util import make_vec_env

    torch.set_num_threads(1)
    random.seed(semente)  # o labirinto é sorteado com o módulo random
    np.random.seed(semente)
    jogo = carregar_jogo(recompensas)
    pasta = SAIDA / identificador
    pasta.mkdir(parents=True, exist_ok=True)

    class Curva(BaseCallback):
        """Guarda a recompensa e o sucesso de cada episódio de treino (para a curva de aprendizado)."""

        def __init__(self):
            super().__init__()
            self.linhas = []

        def _on_step(self):
            for fim, recompensa, info in zip(self.locals["dones"], self.locals["rewards"], self.locals["infos"]):
                if fim:
                    self.linhas.append((self.num_timesteps, info["episode"]["r"],
                                        recompensa == recompensas["GOAL_REWARD"]))
            return True

    inicio = time.time()
    env = make_vec_env(jogo.MazeEnv, n_envs=N_ENVS, seed=semente)
    modelo = PPO("MlpPolicy", env, seed=semente, verbose=0, device="cpu", **parametros)
    curva = Curva()
    modelo.learn(total_timesteps=TOTAL_TIMESTEPS, callback=curva)
    modelo.save(pasta / "modelo_ppo.zip")
    treino_s = time.time() - inicio
    pd.DataFrame(curva.linhas, columns=["passo", "recompensa", "sucesso"]).to_csv(
        pasta / "curva_treino.csv", index=False)

    recompensa_det, sucesso_det = testar(modelo, jogo, recompensas, deterministico=True)
    recompensa_est, sucesso_est = testar(modelo, jogo, recompensas, deterministico=False)
    print(f"{identificador}: sucesso {sucesso_est:.1f}% (estocástica) / {sucesso_det:.1f}% (determinística), "
          f"treino {treino_s / 60:.1f} min", flush=True)
    return {"id": identificador,
            "recompensa_media": recompensa_est, "taxa_sucesso": sucesso_est,
            "recompensa_media_det": recompensa_det, "taxa_sucesso_det": sucesso_det,
            "treino_min": treino_s / 60}


def melhores_recompensas():
    """Configuração de recompensa do modelo do Q-Learning com a maior taxa de sucesso no teste."""
    config = pd.read_csv("resultados/qlearning_config.csv")
    resultados = pd.read_csv("resultados/qlearning_resultados.csv")
    melhor = config.merge(resultados).sort_values(["taxa_sucesso", "recompensa_media"]).iloc[-1]
    print(f"Melhor modelo do Q-Learning: {melhor['id']} ({melhor['recompensas']}, {melhor['taxa_sucesso']}%)")
    return melhor["recompensas"]


def salvar_config(chave):
    """Tabela com o valor de todos os parâmetros de cada modelo (os não alterados com o padrão)."""
    padrao = dict(learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                  clip_range=0.2, ent_coef=0.0, net_arch="64, 64")
    linhas = []
    for identificador, parametros in CONFIGURACOES.items():
        p = {**padrao, **{k: v for k, v in parametros.items() if k != "policy_kwargs"}}
        if "policy_kwargs" in parametros:
            p["net_arch"] = ", ".join(map(str, parametros["policy_kwargs"]["net_arch"]))
        alterados = sorted(set(parametros) - {"policy_kwargs"} | ({"net_arch"} if "policy_kwargs" in parametros else set()))
        linhas.append({"id": identificador, "recompensas": chave, "n_envs": N_ENVS, **p,
                       "alterados": " ".join(alterados)})
    pd.DataFrame(linhas).to_csv("resultados/ppo_config.csv", index=False)


def principal():
    chave = melhores_recompensas()
    recompensas = RECOMPENSAS[chave]
    contexto = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(PROCESSOS, mp_context=contexto) as pool:
        tarefas = [pool.submit(treinar, i, p, recompensas, semente=int(i[3:])) for i, p in CONFIGURACOES.items()]
        resultados = [t.result() for t in tarefas]

    salvar_config(chave)
    tabela = pd.DataFrame(resultados)
    tabela.to_csv("resultados/ppo_resultados.csv", index=False, float_format="%.3f")
    print(tabela.round(2).to_string(index=False))


if __name__ == "__main__":
    principal()
