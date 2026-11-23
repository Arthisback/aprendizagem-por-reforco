"""Análise complementar dos modelos já treinados (não altera os resultados oficiais das tabelas).

Repete o teste de 1000 simulações de cada modelo, com o mesmo laço do RL-game.py, registrando como cada
episódio terminou (objetivo, buraco ou limite de 200 turnos). Para o Q-Learning registra também a fração
de passos em estados que o agente nunca visitou no treino e a probabilidade média que a política softmax
dá à ação preferida.
"""

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ppo import carregar_jogo
from qlearning import RECOMPENSAS, TESTES

R = Path("resultados")


def episodio(env, escolher):
    """Roda um episódio; devolve (como terminou, recompensa total, passos)."""
    estado, _ = env.reset(isnumpy=False)
    total, fim = 0, False
    while not fim:
        estado, recompensa, fim, _, _ = env.step(escolher(estado), isnumpy=False)
        total += recompensa
    if env.agent_pos == env.goal_pos:
        return "objetivo", total
    return ("limite" if env.turn >= 200 else "buraco"), total


def analisar_qlearning(jogo):
    config = pd.read_csv(R / "qlearning_config.csv")
    linhas = []
    for _, linha in config.iterrows():
        for nome in RECOMPENSAS["R1"]:
            setattr(jogo.Config, nome, int(linha[nome]))  # recompensas do modelo
        env = jogo.MazeEnv()
        agente = jogo.QLearningAgent(env.action_space, 0, 0, 0)
        agente.load_model(R / "qlearning" / linha["id"] / "q_learning_model.pkl")
        agente.epsilon = 0
        vistos = set(agente.q_table)
        cont = {"passos": 0, "novos": 0, "prob": []}

        def escolher(estado):
            cont["passos"] += 1
            if estado in vistos:
                q = agente.q_table[estado]
                p = np.exp(q - q.max())
                cont["prob"].append(p.max() / p.sum())
            else:
                cont["novos"] += 1
            return agente.get_action(estado)

        fins = [episodio(env, escolher)[0] for _ in range(TESTES)]
        plt_fechar(env)
        linhas.append({
            "id": linha["id"], "estados_q_table": len(vistos),
            **{f"fim_{k}": 100 * fins.count(k) / TESTES for k in ("objetivo", "buraco", "limite")},
            "passos_estado_novo": 100 * cont["novos"] / cont["passos"],
            "prob_acao_preferida": 100 * float(np.mean(cont["prob"])),
            "passos_por_episodio": cont["passos"] / TESTES,
        })
        print(linhas[-1], flush=True)
    return pd.DataFrame(linhas)


def analisar_ppo(jogo):
    from stable_baselines3 import PPO

    linhas = []
    for pasta in sorted((R / "ppo").iterdir()):
        modelo = PPO.load(pasta / "modelo_ppo.zip", device="cpu")
        env = jogo.MazeEnv()
        linha = {"id": pasta.name}
        for deterministico, sufixo in ((False, ""), (True, "_det")):
            escolher = lambda s: int(modelo.predict(np.array(s), deterministic=deterministico)[0])
            fins = [episodio(env, escolher)[0] for _ in range(TESTES)]
            linha.update({f"fim_{k}{sufixo}": 100 * fins.count(k) / TESTES for k in ("objetivo", "buraco", "limite")})
        linhas.append(linha)
        plt_fechar(env)
        print(linhas[-1], flush=True)
    return pd.DataFrame(linhas)


def plt_fechar(env):
    import matplotlib.pyplot as plt

    plt.close(env.fig)


if __name__ == "__main__":
    import torch

    torch.set_num_threads(4)
    random.seed(2026)
    np.random.seed(2026)
    melhor = pd.read_csv(R / "ppo_config.csv")["recompensas"].iloc[0] if (R / "ppo_config.csv").exists() else "R3"
    jogo = carregar_jogo(RECOMPENSAS[melhor])
    from config import Config  # o mesmo objeto lido pelo RL-game.py

    jogo.Config = Config
    if sys.argv[1:] in ([], ["qlearning"]):
        analisar_qlearning(jogo).to_csv(R / "analise_qlearning.csv", index=False, float_format="%.2f")
    for nome, valor in RECOMPENSAS[melhor].items():
        setattr(Config, nome, valor)
    if sys.argv[1:] in ([], ["ppo"]):
        analisar_ppo(jogo).to_csv(R / "analise_ppo.csv", index=False, float_format="%.2f")
