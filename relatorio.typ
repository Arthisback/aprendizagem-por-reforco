// Relatório do trabalho. Gerado por ./executar.sh (tabelas lidas dos CSVs e figuras em resultados/).

#let integrantes = ("Davi Marangoni", "Arthur Bertoglio")

#set document(title: "Trabalho – Aprendizagem por Reforço", author: integrantes)
#set page(paper: "a4", margin: 2.5cm, numbering: "1")
#set text(lang: "pt", region: "BR", size: 11pt)
#set par(justify: true)
#set heading(numbering: "1.1")
#show heading: set block(above: 1.4em, below: 0.8em)
#show heading.where(level: 1): set text(size: 13pt)
#show heading.where(level: 2): set text(size: 11pt)
#set list(indent: 0.6em)
#set figure(gap: 0.9em)
#show figure: set block(above: 1.5em, below: 1.5em)
#show figure.where(kind: table): set figure.caption(position: top)
#show figure.caption: set text(size: 9.5pt)
#set table(stroke: none, inset: (x: 5pt, y: 3pt), align: center)
#show table: set text(size: 9pt)
#show table: set par(justify: false, leading: 0.5em)
#show table.cell.where(y: 0): set text(weight: "bold")
#show "Q-Learning": box // evita quebra de linha no hífen
#show "Stable-Baselines3": box
#let borda = table.hline(stroke: 0.8pt)
#let divisa = table.hline(stroke: 0.4pt)

// ---------- dados (lidos dos CSVs gerados pelos scripts) ----------
#let ler(arquivo) = csv(arquivo, row-type: dictionary)
#let qcfg = ler("resultados/qlearning_config.csv")
#let qres = ler("resultados/qlearning_resultados.csv")
#let pcfg = ler("resultados/ppo_config.csv")
#let pres = ler("resultados/ppo_resultados.csv")
#let virgula(x) = str(x).replace(".", ",")
#let num(x, casas: 2) = {
  // número com casas decimais fixas e vírgula: -57 -> "−57,00"
  let v = calc.round(float(x), digits: casas)
  let inteiro = calc.trunc(calc.abs(v))
  let frac = str(calc.round((calc.abs(v) - inteiro) * calc.pow(10, casas)))
  let sinal = if v < 0 { "−" } else { "" }
  if casas == 0 { sinal + str(inteiro) } else { sinal + str(inteiro) + "," + "0" * (casas - frac.len()) + frac }
}
#let melhor(linhas) = {
  let ordem = linhas.sorted(key: l => (float(l.taxa_sucesso), float(l.recompensa_media)))
  ordem.last().id
}
#let qan = ler("resultados/analise_qlearning.csv")
#let qmelhor = melhor(qres)
#let pmelhor = melhor(pres)
#let destaca(id, alvo, celulas) = if id == alvo { celulas.map(c => strong(c)) } else { celulas }

#align(center)[
  #text(size: 10pt)[
    Universidade Federal da Fronteira Sul -- Campus Chapecó \
    Curso de Ciência da Computação \
    GEX1225 -- Inteligência Artificial #h(0.4em) · #h(0.4em) Prof. Felipe Grando
  ]
  #v(0.9em)
  #text(size: 16pt, weight: "bold")[Trabalho -- Aprendizagem por Reforço]
  #v(0.2em)
  #integrantes.join(" e ")
  #v(0.6em)
]

= Ambiente e metodologia <sec-metodo>

O ambiente disponibilizado pelo professor (`RL-game.py`) é um labirinto de 15 × 15 células sorteado
novamente a cada episódio: o agente começa em uma posição aleatória afastada das bordas, o objetivo fica em
um dos quatro cantos, e cerca de 40% das células fora do caminho garantido viram paredes e 5% viram
buracos. A cada passo o agente escolhe uma de quatro ações (cima, direita, baixo, esquerda) e observa
apenas o conteúdo das quatro células vizinhas, qual é o canto do objetivo e a sua própria posição. O
episódio termina ao chegar ao objetivo, ao cair em um buraco ou ao completar 200 turnos. As recompensas
são `STEP_REWARD` (cada passo), `WALL_REWARD` (bater em uma parede, sem sair do lugar), `HOLE_REWARD`
(cair em um buraco) e `GOAL_REWARD` (chegar ao objetivo).

O Q-Learning do professor atualiza a tabela com
$Q(s, a) <- (1 - alpha) Q(s, a) + alpha [r + gamma max_(a') Q(s', a')]$. Um detalhe importante da
implementação é a escolha da ação: com probabilidade $epsilon$ ela é aleatória; caso contrário, ela é
*sorteada pela softmax dos valores Q* do estado, e não simplesmente a de maior valor. Isso vale também no
teste, em que $epsilon = 0$. Por isso a _escala_ das recompensas muda o comportamento do agente: quanto
maiores as diferenças entre os valores Q, mais a softmax se concentra na melhor ação.

*Como os experimentos foram executados.* O ambiente e o código do Q-Learning *não foram modificados*.
Para cada modelo, o script `qlearning.py` gera um `config.py` próprio (com os valores da combinação) e
executa o `RL-game.py` original com `TRAIN = True`, `TEST = True` e `RENDERS = False`; a recompensa média
e a taxa de sucesso das tabelas são as impressas pelo próprio `RL-game.py` ao final das 1000 simulações
de teste (`TEST_SIMMULATION_NUMBER = 1000`). Os únicos acréscimos, feitos fora do código do jogo, foram
fixar as sementes dos geradores aleatórios (para a execução poder ser repetida) e rodar vários modelos em
paralelo. O `RL-game/config.py` entregue contém a configuração do melhor modelo, e o modelo treinado
correspondente foi copiado para `RL-game/q_learning_model.pkl`. O PPO (@sec-ppo) usa a mesma classe de
ambiente, importada do `RL-game.py`.

*Taxa de sucesso.* É a porcentagem das 1000 simulações em que o agente chegou ao objetivo. Com 1000
simulações, a margem de erro dessa medida é de cerca de ±3 pontos percentuais (intervalo de 95%), então
diferenças menores do que isso não devem ser levadas muito a sério. A recompensa média só pode ser
comparada entre modelos com a mesma configuração de recompensa, pois cada configuração usa uma escala
diferente.

= Escolha do TOTAL_TIMESTEPS <sec-timesteps>

Antes dos experimentos principais, três combinações foram treinadas com 0,25 a 8 milhões de passos
(@fig-preliminar). O desempenho sobe rápido até 2 milhões de passos (por exemplo, R3 + P2 vai de 58% com
1 milhão para 76% com 2 milhões) e, depois disso, cresce pouco (78% com 4 milhões e 82% com 8 milhões),
enquanto o tempo de treino dobra a cada passo. O valor escolhido foi *`TOTAL_TIMESTEPS = 2.000.000`*,
usado em todos os modelos do Q-Learning e do PPO. Com ele, cada modelo do Q-Learning treina em cerca de
um minuto (de 26 mil a 44 mil episódios) e cada modelo do PPO em 4 a 17 minutos.

#figure(
  image("resultados/fig_preliminar.png", width: 84%),
  caption: [Testes preliminares: taxa de sucesso no teste (1000 simulações) conforme o número de passos de
    treino. A faixa cinza marca o valor escolhido. As configurações R3, R4, P1, P2 e P4 estão na
    @tab-qcfg.],
) <fig-preliminar>

= Q-Learning <sec-qlearning>

== Configurações testadas

Foram definidas 4 configurações de recompensa e 4 de parâmetros, combinadas em 16 modelos
(@tab-qcfg):

- *Recompensas.* R1 só recompensa o objetivo (todo o resto vale 0). R2, R3 e R4 seguem a mesma ideia --
  custo de $-1$ por passo para incentivar caminhos curtos, penalidade por bater na parede, penalidade
  grande por cair no buraco e recompensa grande no objetivo -- em escalas crescentes (objetivo valendo
  10, 100 e 500).
- *Parâmetros.* P1 usa $gamma = 0","5$ (o agente valoriza pouco o futuro) e exploração fixa
  ($epsilon = 0","1$, sem decaimento). P2 é a configuração "clássica": $alpha = 0","1$, $gamma = 0","9$ e
  $epsilon$ começando em 1 (exploração total) e multiplicado por 0,99 a cada 100 episódios. P3 aprende
  mais rápido ($alpha = 0","5$) e reduz $alpha$ e $epsilon$ ao longo do treino. P4 usa $gamma = 0","99$
  (horizonte longo) e também reduz os dois.

#figure(
  caption: [Identificador e parâmetros de cada modelo do Q-Learning. Em todos, `TOTAL_TIMESTEPS =
    2.000.000` e `TEST_SIMMULATION_NUMBER = 1000`.],
  table(
    columns: 12,
    borda,
    table.header(
      [ID], [Comb.], [STEP], [WALL], [HOLE], [GOAL], [ALPHA], [GAMMA], [EPSILON],
      [ALPHA\ DECAY], [EPSILON\ DECAY], [DECAY\ STEP],
    ),
    divisa,
    ..qcfg.map(l => (
      l.id, [#l.recompensas + #l.parametros], ..(l.STEP_REWARD, l.WALL_REWARD, l.HOLE_REWARD, l.GOAL_REWARD)
        .map(v => v.replace("-", "−")),
      virgula(l.ALPHA), virgula(l.GAMMA), virgula(l.EPSILON), virgula(l.ALPHA_DECAY),
      virgula(l.EPSILON_DECAY), l.DECAY_STEP,
    )).flatten(),
    borda,
  ),
) <tab-qcfg>

== Resultados

#figure(
  caption: [Resultados do teste (1000 simulações) de cada modelo do Q-Learning. Em negrito, o melhor
    modelo.],
  table(
    columns: (auto, auto, auto, auto),
    borda,
    table.header([ID], [Combinação], [Recompensa média], [Taxa de sucesso]),
    divisa,
    ..qres.enumerate().map(((i, l)) => destaca(l.id, qmelhor, (
      l.id, [#qcfg.at(i).recompensas + #qcfg.at(i).parametros], num(l.recompensa_media),
      [#l.taxa_sucesso%],
    ))).flatten(),
    borda,
  ),
) <tab-qres>

O melhor modelo foi o *#qmelhor* (R3 + P2), com #qres.find(l => l.id == qmelhor).taxa_sucesso% de
sucesso, acima dos 40% exigidos. Onze dos 16 modelos passaram de 40%. A @fig-qmapa mostra a taxa de
sucesso de cada combinação.

#figure(
  image("resultados/fig_qlearning_mapa.png", width: 62%),
  caption: [Taxa de sucesso no teste de cada combinação de recompensas (linhas) e parâmetros
    (colunas).],
) <fig-qmapa>

== Análise

Para entender os resultados, cada modelo foi testado de novo em mais 1000 simulações (script
`analise.py`), registrando como cada episódio terminou e com que probabilidade a softmax escolhia a ação
preferida do agente. As taxas de sucesso dessa repetição ficaram próximas das oficiais (diferença média
de #num(qan.zip(qres).map(((a, b)) => calc.abs(float(a.fim_objetivo) - float(b.taxa_sucesso))).sum()
/ qres.len(), casas: 1) ponto percentual). A @tab-qanalise resume os números por configuração de
recompensa.

#let media(linhas, campo) = linhas.map(l => float(l.at(campo))).sum() / linhas.len()
#figure(
  caption: [Repetição do teste agrupada por configuração de recompensa (média dos 4 modelos de cada
    uma): como os episódios terminaram e probabilidade média da ação preferida pela softmax.],
  table(
    columns: 6,
    borda,
    table.header([Recomp.], [Objetivo], [Buraco], [Limite de \ 200 turnos], [Prob. da ação \ preferida],
      [Passos em estado \ não visto no treino]),
    divisa,
    ..("R1", "R2", "R3", "R4").map(r => {
      let ids = qcfg.filter(l => l.recompensas == r).map(l => l.id)
      let ls = qan.filter(l => l.id in ids)
      (r, ..("fim_objetivo", "fim_buraco", "fim_limite", "prob_acao_preferida", "passos_estado_novo")
        .map(c => [#num(media(ls, c), casas: 1)%]))
    }).flatten(),
    borda,
  ),
) <tab-qanalise>

- *A escala das recompensas foi o fator mais importante.* Em R1 os valores Q ficam entre 0 e 1, e a
  softmax de valores tão próximos é quase uniforme: a ação preferida tinha só 26% de chance de ser
  escolhida, praticamente o mesmo que sortear entre as 4 ações. O agente anda quase ao acaso e cai em
  buracos em 69% dos episódios (R1 também não pune o buraco). Com R2, R3 e R4 as diferenças entre os
  valores Q crescem e a política fica mais decidida (35%, 69% e 81%), e as quedas em buracos diminuem.
- *Decidido demais também atrapalha.* R4 (a maior escala) caiu menos em buracos, mas passou a terminar
  29% dos episódios pelo limite de 200 turnos, contra 19% em R3. Bater na parede não encerra o episódio e
  o agente só enxerga as quatro células vizinhas: o mesmo estado aparece em labirintos diferentes, e uma
  ação que funciona em um deles pode levar a um beco em outro. Uma política quase determinística fica
  presa indo e voltando; um pouco de aleatoriedade (como em R3) ajuda a sair. O pior caso foi o Q16 (R4 +
  P4, a combinação mais "decidida", com 95% na ação preferida), que esgotou os 200 turnos em 40,5% dos
  episódios. Por isso R3 teve o melhor desempenho médio (67% de sucesso, contra 59% de R4).
- *Desconto baixo prejudica.* Com $gamma = 0","5$ (P1) a recompensa do objetivo some depressa
  ($0","5^10 approx 0","001$): longe do objetivo, as ações têm valores Q parecidos. Com R2 e R3, P1 foi o
  pior conjunto de parâmetros (34% e 58%). Comparando Q09 e Q10, que só diferem nos parâmetros, a ação
  preferida passou de 43% para 71% de probabilidade e as quedas em buracos caíram de 20% para 9%. Em R4,
  a escala maior compensou e P1 empatou com P2 e P3.
- *P2 e P3 ficaram equivalentes.* A diferença entre Q10 (74%) e Q11 (70%) está dentro da margem de
  erro; na repetição da análise, os dois ficaram em 74% e 76%. O decaimento de $epsilon$ nos dois deixa o
  agente explorar bastante no início e aproveitar o que aprendeu no fim.
- *O limite não é falta de treino.* Menos de 1% dos passos de teste caíram em estados nunca vistos no
  treino: com 2 milhões de passos, a tabela Q cobre quase tudo o que aparece no teste (cerca de 20 mil
  estados). O que limita a taxa de sucesso é a visão restrita do agente, que só enxerga as células
  vizinhas e não o labirinto inteiro.

= PPO <sec-ppo>

O PPO (_Proximal Policy Optimization_) foi aplicado com a biblioteca Stable-Baselines3 (script `ppo.py`).
Diferente do Q-Learning, que guarda um valor para cada par estado-ação em uma tabela, o PPO treina duas
redes neurais: a política (probabilidade de cada ação dado o estado) e uma estimativa do valor do estado.
A cada rodada o agente coleta experiência com a política atual e a atualiza limitando o quanto ela pode
mudar (_clipping_), o que torna o treino estável. Como é uma rede neural, o que ela aprende em um estado
se generaliza para estados parecidos.

*Configuração fixa.* As recompensas são as de *R3* (`STEP_REWARD = -1`, `WALL_REWARD = -5`,
`HOLE_REWARD = -100`, `GOAL_REWARD = 100`), a configuração de recompensa do melhor modelo do Q-Learning
(#qmelhor), que também teve a melhor média entre as quatro. O `TOTAL_TIMESTEPS` é o mesmo do Q-Learning,
2.000.000 de passos. O ambiente é o `MazeEnv` do `RL-game.py`, importado sem modificação; a observação
(as quatro células vizinhas, o canto do objetivo e a posição) é tratada pela biblioteca como
`MultiDiscrete` e convertida em um vetor _one-hot_ de 249 posições. A experiência é coletada em 8 cópias
do ambiente ao mesmo tempo (`n_envs = 8`), o que não muda o total de passos. Uma observação: a biblioteca
só para de treinar ao fim de uma coleta completa (`n_steps` × 8 passos), então o total real ficou um
pouco acima do valor pedido -- 2.015.232 passos na maioria dos modelos (0,8% a mais), 2.000.896 no PPO07
e 2.002.944 no PPO10.

*Teste.* Cada modelo foi testado em 1000 simulações, com o mesmo laço e o mesmo critério de sucesso do
`RL-game.py` (a recompensa recebida ser a do objetivo). A ação é sorteada pela política aprendida
(`deterministic=False`), que é a forma como o PPO a treina e a mesma ideia da softmax usada pelo
Q-Learning do professor. Como comparação, a tabela também mostra a taxa de sucesso escolhendo sempre a
ação mais provável (`deterministic=True`).

== Configurações testadas

Foram treinados e salvos 10 modelos (arquivos `resultados/ppo/<ID>/modelo_ppo.zip`). O PPO01 usa os
valores padrão da biblioteca e serve de referência; cada um dos outros muda um aspecto do treino
(@tab-pcfg):

- *Taxa de aprendizado:* maior no PPO02 ($10^(-3)$) e menor no PPO03 ($10^(-4)$) que o padrão
  ($3 dot 10^(-4)$).
- *Exploração:* o PPO04 soma um bônus de entropia à função objetivo (`ent_coef = 0,01`), que incentiva a
  política a não ficar determinística cedo demais.
- *Desconto:* $gamma = 0","9$ no PPO05 (horizonte curto) e $gamma = 0","999$ com `gae_lambda = 0,98` no
  PPO06 (horizonte longo).
- *Frequência das atualizações:* o PPO07 atualiza a política a cada 256 passos por cópia do ambiente
  (2048 no total), em vez de 2048 (16.384 no total), com lotes de 256.
- *Tamanho das atualizações:* o PPO08 limita mais a mudança da política a cada atualização
  (`clip_range = 0,1`) e passa menos vezes pelos dados (`n_epochs = 4`).
- *Tamanho da rede:* o PPO09 usa duas camadas de 256 neurônios, em vez de 64.
- *Combinação:* o PPO10 junta atualizações mais frequentes, bônus de entropia, $gamma = 0","98$ e uma rede
  intermediária (128 × 128).

#let alterado(l, campo, valor) = if campo in l.alterados.split(" ") { strong(valor) } else { valor }
#figure(
  caption: [Identificador e parâmetros de cada modelo do PPO. Em negrito, os valores diferentes do padrão
    da biblioteca. Em todos, recompensas R3, `TOTAL_TIMESTEPS = 2.000.000` e `n_envs = 8`; `n_steps` é o
    número de passos coletados por cópia do ambiente antes de cada atualização e "Rede" é o número de
    neurônios das duas camadas ocultas (iguais na política e no valor).],
  table(
    columns: 10,
    borda,
    table.header([ID], [learning\ \_rate], [n_steps], [batch\ \_size], [n_epochs], [gamma], [gae\ \_lambda],
      [clip\ \_range], [ent_coef], [Rede]),
    divisa,
    ..pcfg.map(l => (
      l.id,
      ..("learning_rate", "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "clip_range",
        "ent_coef").map(c => alterado(l, c, virgula(l.at(c)))),
      alterado(l, "net_arch", l.net_arch.replace(", ", "×")),
    )).flatten(),
    borda,
  ),
) <tab-pcfg>

== Resultados

#figure(
  caption: [Resultados do teste (1000 simulações) de cada modelo do PPO. Em negrito, o melhor modelo. A
    penúltima coluna é só para comparação (ação mais provável em vez de sorteada). O tempo de treino é
    indicativo: cinco modelos eram treinados ao mesmo tempo na mesma máquina.],
  table(
    columns: 5,
    borda,
    table.header([ID], [Recompensa média], [Taxa de sucesso], [Taxa de sucesso \ (determinística)],
      [Tempo de \ treino]),
    divisa,
    ..pres.map(l => destaca(l.id, pmelhor, (
      l.id, num(l.recompensa_media), [#num(l.taxa_sucesso, casas: 1)%],
      [#num(l.taxa_sucesso_det, casas: 1)%], [#calc.round(float(l.treino_min)) min],
    ))).flatten(),
    borda,
  ),
) <tab-pres>

#let pan = ler("resultados/analise_ppo.csv")
#let pbest = pres.find(l => l.id == pmelhor)
#let qbest = qres.find(l => l.id == qmelhor)
#let qbest_an = qan.find(l => l.id == qmelhor)
#let faixa(campo) = {
  let v = pan.map(l => float(l.at(campo)))
  [#num(calc.min(..v), casas: 1)% a #num(calc.max(..v), casas: 1)%]
}

Todos os 10 modelos passaram com folga dos 20% exigidos. O melhor foi o *#pmelhor* (bônus de entropia),
com #num(pbest.taxa_sucesso, casas: 1)% de sucesso e recompensa média de #num(pbest.recompensa_media).

== Análise e comparação com o Q-Learning

#figure(
  image("resultados/fig_curvas.png", width: 100%),
  caption: [Taxa de sucesso durante o treino. À esquerda, os quatro modelos do Q-Learning com as
    recompensas R3 (média móvel de 1000 episódios; inclui os episódios de exploração, por isso fica abaixo
    da taxa do teste). À direita, os 10 modelos do PPO (média em janelas de 50 mil passos).],
) <fig-curvas>

- *Os parâmetros do PPO pouco importaram.* As taxas de sucesso ficaram entre
  #num(calc.min(..pres.map(l => float(l.taxa_sucesso))), casas: 1)% e
  #num(calc.max(..pres.map(l => float(l.taxa_sucesso))), casas: 1)%, uma faixa estreita, perto da
  margem de erro de ±3 pontos. Há uma leve tendência: as mudanças que deixam o treino mais cuidadoso
  (taxa de aprendizado menor no PPO03, bônus de entropia no PPO04, atualizações menores no PPO08) e o
  horizonte longo do PPO06 ficaram em torno de 88%, enquanto a taxa de aprendizado maior (PPO02) e o
  horizonte curto (PPO05, $gamma = 0","9$) ficaram em torno de 84%. A mesma ordem aparece na taxa de
  sucesso do último milhão de passos de treino, o que indica que a tendência não é só ruído do teste.
  Uma rede maior (PPO09) não ajudou e foi a mais lenta para treinar.
- *O PPO aprende muito mais rápido.* Pela @fig-curvas, todos os modelos do PPO passam de 80% de sucesso
  no treino com 100 a 300 mil passos (5% a 15% do orçamento) e depois praticamente não melhoram.
  O `TOTAL_TIMESTEPS` de 2 milhões foi escolhido pelo Q-Learning, que ainda estava melhorando nesse ponto.
- *O PPO superou o Q-Learning* com as mesmas recompensas e o mesmo número de passos:
  #num(pbest.taxa_sucesso, casas: 1)% contra #qbest.taxa_sucesso% de sucesso, e recompensa média de
  #num(pbest.recompensa_media) contra #num(qbest.recompensa_media) (aqui a comparação é válida, pois as
  duas usam R3). A diferença vem da generalização: a tabela do Q-Learning aprende cada combinação de
  posição, vizinhança e objetivo separadamente, enquanto a rede do PPO aprende regras que valem em
  qualquer posição, como "ir na direção do canto do objetivo" e "não entrar em uma célula vizinha com
  buraco". Na repetição do teste (`analise.py`), o PPO caiu em buracos em só #faixa("fim_buraco") dos
  episódios, contra #num(qbest_an.fim_buraco, casas: 1)% do #qmelhor, e quase todas as falhas do PPO
  (#faixa("fim_limite")) foram por esgotar os 200 turnos.
- *A política precisa de aleatoriedade.* Escolhendo sempre a ação mais provável
  (`deterministic=True`), o sucesso caiu para cerca de 70% e os episódios que esgotaram os 200 turnos
  subiram para #faixa("fim_limite_det"), sem nenhuma queda em buraco. É o mesmo efeito visto no
  Q-Learning com R4: como o agente só vê as células vizinhas, uma política determinística entra em ciclos
  dos quais não sai; sortear a ação permite escapar. Isso também explica a recompensa média quase nula
  na versão determinística: cada episódio preso custa 200 passos de $-1$ e as batidas na parede.
- *Custo.* O Q-Learning treinou cada modelo em cerca de 1 minuto; o PPO levou de 4 a 17 minutos (as redes
  neurais custam mais por passo).

= Uso de IA <sec-ia>

Foi utilizado o assistente de IA Claude (Anthropic), por meio da ferramenta Claude Code, no
desenvolvimento dos scripts que automatizam os experimentos (`qlearning.py`, `ppo.py`, `analise.py` e
`figuras.py`), na escolha inicial das configurações testadas, na análise dos resultados e na redação
deste relatório.
