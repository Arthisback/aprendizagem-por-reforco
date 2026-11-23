#!/usr/bin/env bash
# Roda os experimentos e gera o relatorio.pdf, tudo no Docker.
#   ./executar.sh              -> Q-Learning (16 modelos), PPO (10 modelos), análise, figuras e PDF (~35 min)
#   ./executar.sh preliminar   -> só os testes preliminares do TOTAL_TIMESTEPS (~4 min)
#   ./executar.sh relatorio    -> só refaz as figuras e o PDF a partir dos CSVs em resultados/
set -euo pipefail
cd "$(dirname "$0")"

docker build -q -t ia-rl . >/dev/null
rodar() { docker run --rm --user "$(id -u):$(id -g)" -v "$PWD":/trabalho ia-rl sh -c "$1"; }
relatorio='python figuras.py && python -c "import typst; typst.compile(\"relatorio.typ\", output=\"relatorio.pdf\")" && echo "PDF gerado: relatorio.pdf"'

case "${1:-tudo}" in
  preliminar) rodar "python qlearning.py preliminar" ;;
  relatorio) rodar "$relatorio" ;;
  tudo)
    rodar "python qlearning.py && python -W ignore ppo.py && python -W ignore analise.py && $relatorio"
    # deixa o melhor modelo do Q-Learning ao lado do jogo, para o modo TEST do RL-game.py
    melhor=$(rodar "python -c \"import pandas as pd; c = pd.read_csv('resultados/qlearning_config.csv').merge(pd.read_csv('resultados/qlearning_resultados.csv')); print(c.sort_values(['taxa_sucesso', 'recompensa_media']).iloc[-1]['id'])\"")
    cp "resultados/qlearning/$melhor/q_learning_model.pkl" RL-game/q_learning_model.pkl
    echo "Modelo $melhor copiado para RL-game/q_learning_model.pkl" ;;
  *) echo "uso: $0 [preliminar|relatorio]"; exit 1 ;;
esac
