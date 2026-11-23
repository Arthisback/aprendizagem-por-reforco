"""Figuras do relatório (geradas a partir dos CSVs em resultados/)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

R = Path("resultados")
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # paleta categórica (3 primeiras posições)
CINZA, TEXTO, TEXTO2, GRADE = "#b9b8b2", "#0b0b0b", "#52514e", "#e6e5e0"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": CINZA, "axes.labelcolor": TEXTO2, "xtick.color": TEXTO2,
    "ytick.color": TEXTO2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRADE, "grid.linewidth": 0.6, "legend.frameon": False, "figure.dpi": 200,
})
porcento = FuncFormatter(lambda v, _: f"{v:.0f}%")
milhoes = FuncFormatter(lambda v, _: f"{v / 1e6:g} M")


def preliminar():
    dados = pd.read_csv(R / "preliminar.csv")
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    ax.axvspan(1.6e6, 2.5e6, color=GRADE, alpha=0.8, lw=0)
    for cor, ((r, p), grupo) in zip(SERIES, dados.groupby(["recompensa", "parametros"], sort=False)):
        ax.plot(grupo["TOTAL_TIMESTEPS"], grupo["taxa_sucesso"], color=cor, lw=2, marker="o", ms=4,
                label=f"{r} + {p}")
        ultimo = grupo.iloc[-1]
        ax.annotate(f"{r} + {p}", (ultimo["TOTAL_TIMESTEPS"], ultimo["taxa_sucesso"]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=TEXTO, fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks(sorted(dados["TOTAL_TIMESTEPS"].unique()))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e6:g} M"))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.yaxis.set_major_formatter(porcento)
    ax.set_ylim(0, 100)
    ax.set_xlim(2e5, 1.6e7)
    ax.set_xlabel("TOTAL_TIMESTEPS (escala logarítmica)")
    ax.set_ylabel("Taxa de sucesso no teste")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(R / "fig_preliminar.png")


def mapa_qlearning():
    config = pd.read_csv(R / "qlearning_config.csv")
    dados = config.merge(pd.read_csv(R / "qlearning_resultados.csv"))
    tabela = dados.pivot(index="recompensas", columns="parametros", values="taxa_sucesso")
    ids = dados.pivot(index="recompensas", columns="parametros", values="id")
    cmap = LinearSegmentedColormap.from_list("azul", ["#eef4fc", "#2a78d6", "#0d3a75"])
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    ax.imshow(tabela.values, cmap=cmap, vmin=10, vmax=85)
    for i in range(tabela.shape[0]):
        for j in range(tabela.shape[1]):
            v = tabela.values[i, j]
            ax.text(j, i, f"{ids.values[i, j]}\n{v:.0f}%", ha="center", va="center", fontsize=8,
                    color="white" if v > 40 else TEXTO)
    ax.set_xticks(range(4), tabela.columns)
    ax.set_yticks(range(4), tabela.index)
    ax.set_xlabel("Parâmetros do Q-Learning")
    ax.set_ylabel("Recompensas")
    ax.grid(False)
    for lado in ax.spines.values():
        lado.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(R / "fig_qlearning_mapa.png")


def curvas():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=True)
    # Q-Learning: progresso impresso pelo RL-game.py a cada DECAY_STEP (100) episódios
    config = pd.read_csv(R / "qlearning_config.csv")
    resultados = pd.read_csv(R / "qlearning_resultados.csv")
    melhor_q = config.merge(resultados).sort_values(["taxa_sucesso", "recompensa_media"]).iloc[-1]
    linhas = config[config["recompensas"] == melhor_q["recompensas"]]
    cores = iter(SERIES + ["#eda100"])
    for _, linha in linhas.iterrows():
        curva = pd.read_csv(R / "qlearning" / linha["id"] / "curva_treino.csv")
        suave = curva["taxa_sucesso"].rolling(10, min_periods=1).mean() * 100
        a1.plot(curva["episodio"], suave, lw=1.5, color=next(cores), label=f"{linha['id']} ({linha['parametros']})")
    a1.set_title(f"Q-Learning, recompensas {melhor_q['recompensas']}", fontsize=9, color=TEXTO)
    a1.set_xlabel("Episódio de treino")
    a1.set_ylabel("Taxa de sucesso no treino")
    a1.yaxis.set_major_formatter(porcento)
    a1.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g} mil"))
    a1.legend(fontsize=7, loc="upper left")
    # PPO: sucesso dos episódios de treino em janelas de 50 mil passos
    ppo = pd.read_csv(R / "ppo_resultados.csv")
    melhor_p = ppo.sort_values(["taxa_sucesso", "recompensa_media"]).iloc[-1]["id"]
    pior_p = ppo.sort_values(["taxa_sucesso", "recompensa_media"]).iloc[0]["id"]
    for identificador in ppo["id"]:
        curva = pd.read_csv(R / "ppo" / identificador / "curva_treino.csv")
        janela = (curva["passo"] // 50_000 + 0.5) * 50_000
        media = curva.groupby(janela)["sucesso"].mean() * 100
        destaque = {melhor_p: (SERIES[0], f"{melhor_p} (melhor)"), pior_p: (SERIES[1], f"{pior_p} (pior)")}
        cor, rotulo = destaque.get(identificador, (CINZA, None))
        a2.plot(media.index, media.values, lw=1.5 if rotulo else 0.9, color=cor, label=rotulo,
                zorder=3 if rotulo else 2)
    a2.plot([], [], color=CINZA, lw=0.9, label="demais modelos")
    a2.set_title("PPO, recompensas " + str(pd.read_csv(R / "ppo_config.csv")["recompensas"].iloc[0]),
                 fontsize=9, color=TEXTO)
    a2.set_xlabel("Passo de treino")
    a2.xaxis.set_major_formatter(milhoes)
    a2.legend(fontsize=7, loc="lower right")
    a1.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(R / "fig_curvas.png")


if __name__ == "__main__":
    preliminar()
    mapa_qlearning()
    if (R / "ppo_resultados.csv").exists():
        curvas()
