# Pendências — tese e protótipo

Marque `[x]` ao concluir e registre a data. Referências de seção são da tese (`metodologia_pesquisa_reescrtia.tex`, `planejamento.tex`).

## 1. Protótipo (E4 — itens pendentes da Seção "Operacionalização no protótipo")

- [x] **Monitoramento e reotimização** (2026-10-31): `experiments/monitoring.py`, em `monitoring` de cada experimento.
  - A cada Δt = 30 d na janela de validação: estado ⟨ρ, L, W̄T⟩ e Δestado por agente, mais a restrição ativa nos 90 dias anteriores.
  - Dispara quando o limiar persiste por 2 janelas (|Δρ| ≥ 0,15; ΔL e ΔW̄T ≥ 25%) ou quando a restrição muda.
  - Janelas sem visitas repetem o último W̄T. Intervalo mínimo de 60 d e no máximo 2 reotimizações.
  - No disparo: recalibra, reestima custos e roda o ADOPT de novo com tabelas fixas. A nova configuração é avaliada contra a vigente no trecho seguinte (P80 pareado).
  - Limiares provisórios (E1).
  - Correção feita junto: o corte de idade dos casos ativos no ρ amplia a janela para trás quando há poucas visitas. Antes, o ρ mensal oscilava entre 0 e 1. Isso afeta a identificação da restrição: **precisa de nova rodada**.
- [x] **Métricas estendidas de H1** (2026-10-31): espera, vazão por bloco de 28 dias, WIP por estágio, utilização, frequências de rota (variação total) e retrabalho, em `extended` / `checks_extended` de cada dobra (`evaluation/validation.py`). Critérios provisórios; só decidem H1 com `extended_enforced=True` (decisão da E1). Dependências entre objetos só sinalizadas (não simuladas).
  - Correção feita junto: a ocupação dos servidores passou a ser integrada no tempo. Antes, só contava serviços concluídos no horizonte, o que superestimava a ociosidade usada no termo β. **Os resultados do capítulo precisam de nova rodada.**
  - Achado: em Mindville, a escala de serviço de Development vai ao piso do ajuste (0,016) e a utilização simulada fica perto de zero (observada 0,23). Investigar a calibração.
- [x] **Métricas por classe de serviço** (2026-10-31): LT, WT, TH, FE e WIP por tipo de item, observados e simulados (H1 e C0–C3), mais C2 × C1 no P80 por classe (descritivo). Classes com ≥ 30 casos na calibração (até 4; resto em "Outros"). Curvas de permanência por classe opcionais (`CalibrationSettings.by_class`); o padrão mantém a curva comum.
- [x] **Descoberta de processos** (2026-10-31): `mining/discovery.py`, em `process_discovery` de cada repositório, sobre a janela de calibração.
  - Variantes e cobertura, DFG com frequência e tempo médio.
  - Candidatos: Inductive Miner (com e sem filtro de ruído 0,2) e Heuristics Miner, cada um com aderência, precisão, generalização e simplicidade, sem escore único.
  - Conformidade: modelo de maior precisão entre os de aderência ≥ 0,95; atividades com problemas; aderência por classe.
  - O Split Miner não tem implementação em Python. Fica registrado como não executado: rodar o .jar externo ou declarar na tese.
- [x] **Análise de sensibilidade** (2026-10-31): `experiments/sensitivity.py` + `scripts/run_sensitivity.py`.
  - Análise local, um fator por vez, com 18 fatores e 34 variantes. Cobre os itens da metodologia:
    - pesos γ/α/β e γ estimado, pesos e janela da restrição;
    - janela e persistência do monitoramento;
    - demanda ±20% e serviço ±10%;
    - quantil da capacidade, horizonte, janela de calibração e fim do período ativo;
    - acoplamentos, calendário, multitarefa e curvas por classe.
  - Fatorial 2^k (`--factorial fator:baixo/alto`), com efeitos principais e interações sobre o Δ% de H2.
  - Para cada variante, o resultado registra robustez de direção, de conclusão prática e de decisão de H2/H3, além de mudança de a* e de C2.
  - Intervalos dos fatores provisórios (E1).
  - Teste rápido em ZooKeeper: conclusões estáveis, exceto horizonte de 270 d e κ = 0,2, que mudam a direção de H2 (Δ ≤ 2,4%, sem relevância prática).
- [x] **Estudo-piloto do número de réplicas** (2026-10-31): R0 = 10 réplicas de todas as configurações. R é o menor valor com meia largura do IC das diferenças pareadas do P80 (C2−C1 e C2−C2_sem_ToC) ≤ 2,5% do P80 da referência, limitado a 100. Registro em `replications`. `--reps N` desliga o piloto. A precisão é provisória (E1).
- [x] **Pesos estimados na calibração** (2026-10-31), como opção. O padrão continua com os pesos fixos, como manda a metodologia: teoria primeiro, sensibilidade depois.
  - `CostSettings.gamma_mode="calibration"`: γ ∝ 1/dispersão de cada métrica normalizada nas simulações de estimação, que não usam as réplicas de avaliação.
  - `ConstraintSettings.weight_mode="entropy"`: β pelo método da entropia nos indicadores mensais da calibração.
  - Os pesos efetivos ficam em `settings`. A escolha é da E1.
- [x] **Calendários de trabalho** (2026-10-31), opção `calendar="business"` (`--calendar business`).
  - Tempo em dias úteis, sem sábados e domingos. Não há feriados, porque os contribuidores estão em vários países.
  - Todas as durações das configurações são convertidas (× 5/7), e as datas reportadas continuam no calendário civil.
  - O padrão segue em dias corridos: projetos públicos com contribuidores globais. Decisão da E1 e item da sensibilidade.
- [x] **Acoplamentos no simulador** (2026-10-31), opcionais (`--couplings`) e estimados na calibração.
  - Sincronização: ao iniciar uma visita, o caso fica bloqueado com probabilidade p(S, S'), estimada dos vínculos O2O "is blocked by"/"depends on" com sobreposição. Ele espera a saída de um caso real do estágio S'.
  - Pessoas compartilhadas: estágios com Jaccard de pessoas ≥ 0,10 formam um pool comum com até 50% dos servidores.
  - Com eles ligados, o teste de interação passa a observar esses mecanismos. Em MariaDB, a aresta Contribuidores–Mantenedores passou a existir (interação 0,10).
  - A "espera por release" continua só como indicador (`object_centric`).
- [x] **Penalidade de multitarefa** (2026-10-31): serviço × (1 + κ·max(0, L/c − 1)), com `--multitask off|estimated|κ`.
  - O κ estimado relaciona a permanência ao WIP por servidor na entrada e só é usado se a correlação de Spearman for positiva e significativa.
  - Achado: em ZooKeeper, MariaDB e Spring a associação é negativa, então o κ estimado é 0. Não há evidência de penalidade, e o padrão fica κ = 0.
- [ ] **Reprodutibilidade:**
  - [x] git (2026-10-31): `.gitignore` deixa de fora dados, resultados, `.env` e `.venv`. O código está em https://github.com/julianoribeiro/ezekiel (ramo `main`, commit inicial `5d98cec`).
  - [x] dependências fixadas (`requirements.lock`, Python 3.12) e ambiente automatizado (`scripts/setup_env.sh`: venv, instalação e testes) (2026-10-31);
  - [x] manifesto por execução (`ezekiel/manifest.py`): revisão git, SHA-256 do código, ambiente, SHA-256 dos logs e do `.meta.json`, configuração, sementes e comando. Gravado automaticamente pelos scripts (2026-10-31);
  - [ ] OCEL derivados (CC BY 4.0) publicados: o código já está no GitHub, mas os logs (`ocels/`, 14 GB) não cabem num repositório git. Publicar num depósito de dados (ex.: Zenodo, com DOI) e citar no README, com atribuição ao *The Public Jira Dataset*.
  - [x] README na raiz do repositório (objetivo, hipóteses, dados, instalação, reprodução, estrutura, citação) (2026-10-31).
  - [ ] Licença do código: escolher (ex.: MIT, como o código do *The Public Jira Dataset*) e adicionar `LICENSE`.
- [x] **Nova rodada completa** (2026-10-31): 16 repositórios, `por_papel` e `por_estagio`, mais H4, com o protótipo atualizado. Resultados em `output/hypotheses/`. Capítulo `resultados_preliminares.tex` atualizado com:
  - estudo-piloto de R;
  - métricas complementares de H1;
  - descoberta de processos;
  - classes de serviço;
  - monitoramento;
  - novos números de H2 e H4.

  A metodologia (`sec:operacionalizacao_prototipo`) também foi atualizada.
  - Conclusões: H1 rejeitada em 14 de 14 e H2/H3 validadas em 0 de 14 nas duas granularidades. O MongoDB por estágio deixou de ser validado após Holm (−1,84%, p = 0,022 sem ajuste e 0,308 com ajuste). H4 sem mudança.
  - Achado para a E1: o piloto por precisão dá R = 10. Com Holm sobre 14 repositórios, o menor p ajustado do Wilcoxon fica em cerca de 0,014, então falta um requisito de poder.
  - Mudaram os resultados: ocupação dos servidores (termo β), corte de idade no ρ (restrição ativa) e R pelo estudo-piloto.
  - Com as opções novas desligadas, o resto é idêntico (regressão verificada em ZooKeeper e MariaDB).
  - Correções feitas na primeira tentativa da rodada (2026-10-31):
    - O monitoramento falhava em `por_estagio` (NEXUS, SAK) quando a reotimização mudava o conjunto de agentes. Também falhava `PseudoTree.depth` com um componente desconexo de mais de um agente.
    - **A espera simulada dos casos iniciais começava em zero**, enquanto o *lead time* conta a idade do caso. A espera observada inclui o histórico do caso. Por isso a espera média falhava nos 14 repositórios (erro de 55–98%). Isso também enviesava o termo WT da função de custo e a eficiência de fluxo.
    - Agora o caso inicial carrega a espera acumulada até t. Com isso, o erro da espera cai para 3% no ZooKeeper e para 22% no MariaDB. Rodada refeita; a anterior ficou em `output/hypotheses_e4_espera_sem_historico/`.

## 2. Decisões com o orientador (E1)

- [ ] Fixar *a priori* os valores provisórios:
  - critérios de H1;
  - margens de H2;
  - pesos;
  - limiares do grafo.

  Os valores atuais estão em `docs/hipoteses.md`.
- [ ] Desfecho primário estimado com censura.
- [ ] Detecção de mudanças de regime entre janelas.
- [ ] Separação entre amostra exploratória e confirmatória, e o critério de seleção da amostra confirmatória.
- [ ] Enquadramento ético pela Resolução CNS 510/2016 (dados públicos, sem CEP).
- [ ] Premissa `por_papel` como modelo organizacional principal no Jira.
- [ ] Escopo de C3 e intervalo de escala de H4 (número de agentes × densidade).

## 3. Texto da tese

- [x] Palavras-chave: "sistemas multiagentes" trocado por "Teoria das Restrições" (PT e EN), coerente com o resumo (2026-10-31).
- [x] Pourbafrani no `main.bib`: não era duplicata, as duas entradas estavam erradas. Agora são BIS 2020 (LNBIP 389) e *IEEE Access* 10 (2022), com autores completos (2026-10-31).
- [x] "25 trabalhos": o planejamento passou a dizer 33 trabalhos em quatro eixos, com quadro comparativo entre classes (2026-10-31).
- [x] Arquivos antigos fora da compilação (`introducao.tex`, `fundamentacao.tex`, `metodologia_pesquisa.tex`, `relacionados.tex`, `RSL.tex`, `old/`) movidos para fora do Overleaf, em `~/Documents/Tese_arquivo_local/overleaf/` (2026-10-31).
- [x] Backups do protótipo (`_backup_antes_remocao_ado/`, `_backup_pre_hipoteses/`) movidos para `~/Documents/Tese_arquivo_local/ezekiel/` (2026-10-31).
- [ ] Compilar no Overleaf e conferir referências (Pourbafrani) e avisos do BibTeX.

## Concluído

- [x] Protótipo refeito para testar H1–H4 da `introducao_reescrita` (2026-10-16).
- [x] Agentes como unidades organizacionais (`config/organizacao/`) (2026-10-19).
- [x] Log OCEL 2.0 multiobjeto com nível de evidência das relações (2026-10-20).
- [x] Grafo de restrições por evidências e teste de interação (2026-10-21).
- [x] Capítulo de resultados preliminares reescrito com os resultados novos.
- [x] Incoerências da tese, pontos 1–5:
  - enquadramento em dados públicos;
  - numeração, rótulos e bibliografia;
  - protocolo;
  - erros de digitação e marcações em vermelho;
  - cronograma até jun/2027.
- [x] Tese e protótipo restritos a bases públicas (2026-10-31).
