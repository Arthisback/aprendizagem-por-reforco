"""Experimentos com o Q-Learning.

Cada modelo é treinado e testado pelo RL-game.py original, sem nenhuma modificação: para cada combinação
é gerado um config.py próprio (a partir do RL-game/config.py) dentro de resultados/qlearning/<id>/, e o
RL-game.py é executado com essa pasta como diretório de trabalho. Assim o "from config import ..." do
jogo lê o config.py da combinação e o modelo q_learning_model.pkl é salvo na mesma pasta.

Uso:
    python qlearning.py preliminar   # testes para escolher o TOTAL_TIMESTEPS
    python qlearning.py              # as 16 combinações (4 recompensas x 4 parâmetros)
"""

import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

JOGO = Path("RL-game/RL-game.py").resolve()
CONFIG_ORIGINAL = Path("RL-game/config.py").read_text()
SAIDA = Path("resultados/qlearning")
PROCESSOS = 8  # modelos treinados em paralelo (cada um é um processo independente)

TOTAL_TIMESTEPS = 2_000_000  # fixo para todos os modelos (Q-Learning e PPO), ver testes preliminares
TESTES = 1000  # TEST_SIMMULATION_NUMBER

# === Configurações de recompensa (valores inteiros) ===
RECOMPENSAS = {
    "R1": dict(STEP_REWARD=0, WALL_REWARD=0, HOLE_REWARD=0, GOAL_REWARD=1),
    "R2": dict(STEP_REWARD=-1, WALL_REWARD=-1, HOLE_REWARD=-10, GOAL_REWARD=10),
    "R3": dict(STEP_REWARD=-1, WALL_REWARD=-5, HOLE_REWARD=-100, GOAL_REWARD=100),
    "R4": dict(STEP_REWARD=-1, WALL_REWARD=-10, HOLE_REWARD=-500, GOAL_REWARD=500),
}

# === Configurações de parâmetros do Q-Learning ===
PARAMETROS = {
    "P1": dict(ALPHA=0.1, GAMMA=0.5, EPSILON=0.1, ALPHA_DECAY=1.0, EPSILON_DECAY=1.0, DECAY_STEP=100),
    "P2": dict(ALPHA=0.1, GAMMA=0.9, EPSILON=1.0, ALPHA_DECAY=1.0, EPSILON_DECAY=0.99, DECAY_STEP=100),
    "P3": dict(ALPHA=0.5, GAMMA=0.95, EPSILON=1.0, ALPHA_DECAY=0.99, EPSILON_DECAY=0.98, DECAY_STEP=100),
    "P4": dict(ALPHA=0.2, GAMMA=0.99, EPSILON=1.0, ALPHA_DECAY=0.995, EPSILON_DECAY=0.99, DECAY_STEP=100),
}

FIM_TESTE = re.compile(r"Episodes (\d+), Mean Reward: (-?[\d.]+), Success Rate: (\d+)%")
PROGRESSO = re.compile(r"Episode (\d+), Mean Reward: (-?[\d.]+), Success Rate: ([\d.]+)")


def gerar_config(valores):
    """Preenche o config.py do professor com os valores da combinação."""
    texto = CONFIG_ORIGINAL
    for nome, valor in valores.items():
        texto, n = re.subn(rf"^(\s*{nome} = )[^#\n]*?(?=\s*#|\s*$)", rf"\g<1>{valor}", texto, flags=re.M)
        assert n == 1, nome
    return texto


def rodar(identificador, valores, semente, base=SAIDA):
    """Treina e testa um modelo com o RL-game.py original; devolve o resultado do teste."""
    pasta = base / identificador
    pasta.mkdir(parents=True, exist_ok=True)
    config = {**valores, "TRAIN": True, "TEST": True, "RENDERS": False, "TEST_SIMMULATION_NUMBER": TESTES}
    (pasta / "config.py").write_text(gerar_config(config))
    # sementes fixas só para a execução poder ser repetida; o código do jogo não é alterado
    codigo = (
        f"import random, numpy, runpy; random.seed({semente}); numpy.random.seed({semente}); "
        f"runpy.run_path({str(JOGO)!r}, run_name='__main__')"
    )
    with open(pasta / "log.txt", "w") as log:
        subprocess.run([sys.executable, "-c", codigo], cwd=pasta, stdout=log, stderr=subprocess.STDOUT, check=True)
    texto = (pasta / "log.txt").read_text()
    episodios, recompensa, sucesso = FIM_TESTE.findall(texto)[-1]
    curva = pd.DataFrame(
        [(int(e), float(r), float(s)) for e, r, s in PROGRESSO.findall(texto)],
        columns=["episodio", "recompensa_media", "taxa_sucesso"],
    )
    curva.to_csv(pasta / "curva_treino.csv", index=False)
    print(f"{identificador}: recompensa média {recompensa}, sucesso {sucesso}%", flush=True)
    return {"id": identificador, "recompensa_media": float(recompensa), "taxa_sucesso": int(sucesso),
            "episodios_treino": int(curva["episodio"].max()) if len(curva) else 0}


def preliminar():
    """Mesma combinação com vários TOTAL_TIMESTEPS, para escolher um valor que não desperdice tempo."""
    import time

    linhas = []
    for recompensa, parametros in (("R3", "P1"), ("R3", "P2"), ("R4", "P4")):
        for passos in (250_000, 500_000, 1_000_000, 2_000_000, 4_000_000, 8_000_000):
            valores = {**RECOMPENSAS[recompensa], **PARAMETROS[parametros], "TOTAL_TIMESTEPS": passos}
            linhas.append((f"{recompensa}{parametros}_{passos}", valores, (recompensa, parametros, passos)))

    def tarefa(item):
        nome, valores, (r, p, passos) = item
        inicio = time.time()
        res = rodar(nome, valores, semente=0, base=Path("resultados/preliminar"))
        return {"recompensa": r, "parametros": p, "TOTAL_TIMESTEPS": passos,
                "recompensa_media": res["recompensa_media"], "taxa_sucesso": res["taxa_sucesso"],
                "tempo_s": round(time.time() - inicio)}

    with ThreadPoolExecutor(PROCESSOS) as pool:
        tabela = pd.DataFrame(pool.map(tarefa, linhas))
    tabela.to_csv("resultados/preliminar.csv", index=False)
    print(tabela.to_string(index=False))


def principal():
    combinacoes = []
    for i, (r, recompensa) in enumerate(RECOMPENSAS.items()):
        for j, (p, parametros) in enumerate(PARAMETROS.items()):
            identificador = f"Q{4 * i + j + 1:02d}"
            valores = {**recompensa, **parametros, "TOTAL_TIMESTEPS": TOTAL_TIMESTEPS}
            combinacoes.append((identificador, r, p, valores))

    with ThreadPoolExecutor(PROCESSOS) as pool:
        resultados = list(pool.map(lambda c: rodar(c[0], c[3], semente=int(c[0][1:])), combinacoes))

    config = pd.DataFrame([{"id": c[0], "recompensas": c[1], "parametros": c[2], **c[3]} for c in combinacoes])
    config.to_csv("resultados/qlearning_config.csv", index=False)
    tabela = pd.DataFrame(resultados)
    tabela.to_csv("resultados/qlearning_resultados.csv", index=False)
    print(config.merge(tabela).to_string(index=False))


if __name__ == "__main__":
    preliminar() if sys.argv[1:] == ["preliminar"] else principal()
