# Hipóteses H1–H4 no protótipo

Este documento liga as hipóteses da introdução reescrita da tese ao código que as testa. A implementação anterior (S0/S1/S2, "MAE < 15%", Mann-Whitney sobre itens) está em `ezekiel/legacy/` e só é usada pela API de demonstração e por `scripts/run_jira_comparison.py`.

## Como rodar

```bash
python -m pytest tests                                    # verificação (ADOPT = C3 = força bruta, etc.)
python scripts/run_hypotheses.py --h4                     # 16 repositórios Jira (org. por_papel) + varredura H4
python scripts/run_hypotheses.py --org por_papel por_estagio   # comparação de granularidades
python scripts/run_hypotheses.py --only mariadb-mdev --quick
python scripts/run_hypotheses.py --only mariadb-mdev --couplings --calendar business --multitask estimated
python scripts/run_sensitivity.py --only apache-zookeeper --quick
python -m ezekiel.cli run --ocel-path <arquivo.json> --profile jira
python -m ezekiel.cli h4
```

Saídas: `output/hypotheses/<repo>.json` (completo, inclui métricas por réplica) e `output/hypotheses/summary.json` (agregado, com Holm).

## Mapa hipótese → código

| Hipótese | O que é testado | Código |
|---|---|---|
| H1 — fidelidade | Calibra em [t−365d, t) e simula a janela de validação [t, t+180d) a partir do estado real em t. Compara P50, P80 e P90 do lead time, Wasserstein, KS, vazão, WIP e cobertura de quantis. Aprovada só se **todos** os critérios forem atendidos. | `simulation/calibration.py`, `evaluation/validation.py`, `experiments/hypotheses.py::run_h1` |
| Agentes (modelo organizacional) | Unidades organizacionais documentadas; estágios sem unidade são ambiente. | `dcop/organization.py`, `config/organizacao/` |
| H2 — coordenação | C2 (ADOPT) vs C1 (decisões locais): redução do **P80** do lead time (pareado por réplica, t ou Wilcoxon) **e** não inferioridade da vazão (margem de 5%). | `dcop/costs.py::local_choice`, `experiments/hypotheses.py` |
| H3 — termos ToC | C2 vs C2 com α = β = 0 (ablação): P80 melhor, sem piora significativa de WT, WIP e FE. | `dcop/costs.py::CostModel.instance` |
| H4 — viabilidade | ADOPT vs C3 (centralizado exato): lacuna, ciclos, mensagens, tempo e memória. Na parte empírica, por repositório; na sintética, por número de agentes e densidade. | `dcop/adopt.py`, `dcop/centralized.py`, `experiments/synthetic.py` |

## Módulos da etapa E4 (out/2026)

| Item da metodologia | Código | Como ligar |
|---|---|---|
| Validação de saída estendida (espera, vazão por bloco, WIP por estágio, utilização, rotas, retrabalho) | `evaluation/validation.py::extended_metrics` | sempre reportada; decide H1 só com `ValidationCriteria.extended_enforced` |
| Classes de serviço | `EventLog.observed_metrics_by_class`, `engine.py` (classe por caso) | sempre reportadas; curvas por classe com `CalibrationSettings.by_class` |
| Estudo-piloto de réplicas | `stats.required_replications`, `hypotheses.py` | padrão (`pilot_reps=10`); `--reps N` fixa R |
| Pesos estimados na calibração | `costs.calibration_gamma`, `constraint.entropy_weights` | `gamma_mode="calibration"`, `weight_mode="entropy"` |
| Monitoramento e reotimização | `experiments/monitoring.py` | padrão (`MonitoringSettings.enabled`) |
| Descoberta de processos e conformidade | `mining/discovery.py` | padrão (`process_discovery` no resultado) |
| Calendário de trabalho (dias úteis) | `EventLog.with_business_calendar` | `--calendar business` |
| Sincronização entre objetos e pessoas compartilhadas | `engine.py` (`sync`, `pool`), `calibration.sync_rules/people_pools` | `--couplings` |
| Penalidade de multitarefa | `StageModel.multitask`, `calibration.estimate_multitask` | `--multitask estimated` ou κ |
| Sensibilidade (local e fatorial) | `experiments/sensitivity.py`, `scripts/run_sensitivity.py` | script próprio |
| Manifesto de execução | `ezekiel/manifest.py` | automático em `run_hypotheses.py` e `run_sensitivity.py` |

Ambiente fixado: `requirements.lock` e `scripts/setup_env.sh`.

## Configurações (metodologia, "Configurações de comparação")

- **C0** — política vigente: sem limite de WIP, FIFO.
- **C1** — cada agente escolhe (WIP, política) que minimiza a permanência no próprio estágio, sem custo compartilhado.
- **C2** — ADOPT sobre J(x) = Σ f_ij + α·WIP a montante de a* + β·ociosidade de a*. A pseudo-árvore tem raiz em a*.
- **C2_sem_ToC** — ADOPT com α = β = 0.
- **C3** — eliminação de variáveis exata sobre a mesma instância de C2.

Todas usam o mesmo modelo, o mesmo estado inicial e as mesmas sementes (números aleatórios comuns) e a **mesma capacidade**: o número de servidores não é variável de decisão. Os custos são estimados com sementes próprias, diferentes das de avaliação.

## Log centrado em objetos (OCEL 2.0)

- **Formato:** OCEL 2.0 JSON padrão (lido pelo PM4Py) mais um arquivo lateral `.meta.json` com o nível de evidência de cada qualificador de relação (explícita, derivada ou inferida) e a lista de relações **inferidas**, que ficam fora do log principal (`ezekiel/ocel/ocel2.py`).
- **Jira** (`scripts/build_jira_ocel2.py`, `scripts/convert_jira_ocel2_all.py`):
  - objetos: WorkItem, Version, Sprint, Component e Team (campo "Team", quando o repositório o tem);
  - relações O2O explícitas: bloqueio, dependência, duplicação, pai e filho, versão, componente, sprint e equipe.
- **Uso:**
  - o log de casos da simulação é a projeção sobre WorkItem, idêntica ao log anterior;
  - os demais objetos alimentam validação, caracterização por tipo de objeto e **indicadores de sincronização** (`scripts/characterize_ocel2.py`, saída em `output/hypotheses/object_centric.json`).
- **Simplificação declarada:** a simulação não representa a sincronização entre objetos (bloqueios, espera por release). Os indicadores quantificam o peso dessa simplificação.

## Grafo de restrições por evidências (`ezekiel/dcop/dependencies.py`)

1. **Candidatos:** para cada par de agentes, mede-se na janela de calibração:
   - fluxo direto de itens;
   - sincronização (itens com bloqueio que se sobrepõem nos estágios dos dois agentes);
   - recurso compartilhado (Jaccard das pessoas anonimizadas que movem itens);
   - correlação das variações semanais de WIP;
   - interface declarada no modelo organizacional.

   O par é candidato se alguma fonte passa do limiar.
2. **Aresta:** só se as decisões interagem na função de custo simulada, isto é, max |f(xa,xb) − f(xa,b̄) − f(ā,xb) + f(ā,b̄)| ≥ τ. Caso contrário, as decisões são separáveis e entram como custos unários.
3. **Matriz de dependências** (`dependency_matrix` no resultado): registra evidências, mecanismo, variáveis, interação e decisão. Quando a evidência vem só de mecanismos não simulados (sincronização ou pessoas), isso é dito explicitamente.

Limiares provisórios (`DependencySettings`): fluxo de 5%, 5 pares sincronizados, Jaccard de 0,10, |r| de 0,30 e τ = 0,01.

## Fonte de dados

Todos os dados vêm de bases públicas (*The Public Jira Dataset*, CC BY 4.0). O protótipo não tem conector para dados corporativos. Perfil `--profile jira` (`ezekiel/data/eventlog.py::LogProfile`): "Resolved" conclui o caso (o "Closed" posterior é administrativo); os status do Jira não separam fila de trabalho dentro de um estágio.

## Decisões de modelagem (para o texto da tese)

1. **Conclusão no Jira = "Resolved".** O "Closed" é fechamento administrativo, em lote, no release. Tratar "Resolved" como QA criava milhares de itens "parados em QA". Isso é configurável (`--profile jira`, ou `--completion-status`).
2. **Casos = episódios.** Um item reaberto gera um novo caso, e o tempo fechado não conta como trabalho.
3. **Fim da janela = quantil 0,99 dos eventos.** Esse critério evita a cauda de repositórios migrados (ex.: Spring → GitHub em 2019).
4. **Permanências por Kaplan-Meier** (truncamento à esquerda e censura à direita). A massa de probabilidade que a curva não esgota gera casos *estacionados*, que contam no WIP mas não consomem capacidade. Sem isso, itens com anos de idade eram "concluídos" pela simulação.
5. **Capacidade = P90 da simultaneidade de casos ativos** (exclui os que já passaram do P95 das permanências). Os dados não identificam esse parâmetro, então ele deve entrar na análise de sensibilidade (`CalibrationSettings.server_quantile`).
6. **Agentes são unidades organizacionais, não estágios** (`fundamentacao_reescrita.tex:129`, `metodologia:35`, `:342`, `:985`). O modelo organizacional (`config/organizacao/*.json`, `ezekiel/dcop/organization.py`) declara cada agente com:
   - estágios sob sua responsabilidade;
   - responsabilidade operacional, informação local, decisões, recursos e evidência de origem.

   Estágios sem unidade decisória (Backlog) são **ambiente**. Cada agente controla (fator de WIP, política de fila[, fração de capacidade]) sobre os seus estágios; a capacidade total do agente é conservada. A restrição ativa, a configuração C1, os termos da Teoria das Restrições e o estado ⟨ρ, L, W̄T⟩ são calculados por agente. O resultado inclui o quadro de documentação dos agentes e a matriz de dependências.
   - **Jira:** as equipes não são observáveis (o responsável é pseudonimizado no dataset). `por_papel` (contribuidores × mantenedores) é uma **premissa declarada**; `por_estagio` é uma granularidade de comparação, que reproduz exatamente a rodada anterior.
7. **Restrição ativa:** escore C_i = Σ β_k·indicador normalizado (ρ, WT, L, B), calculado em janelas mensais, com a* = líder no maior número de janelas e estabilidade reportada.
8. **Bloqueio Kanban real:** ao avançar, o caso só libera a vaga atual depois de obter a próxima. No retrabalho, libera antes, o que evita impasse.
9. **Espera × serviço (quando os status distinguem fila de trabalho; não é o caso do Jira).** Cada visita é decomposta em segmentos de fila e de trabalho. A curva de serviço usa só o tempo em status de trabalho. A espera no estágio tem duas componentes:
   - **exógena:** atraso amostrado da curva KM do tempo em fila (prioridades, lotes e bloqueios que a capacidade não explica);
   - **endógena:** disputa pelos servidores.

   A calibração aumenta a capacidade quando a espera endógena passa de metade da observada e ajusta a escala da exógena para cobrir o restante. Calibrar só a capacidade levava a um servidor por estágio, artefato que foi descartado.
10. **Kaplan-Meier com durações nulas:** em τ = 0, os casos sem truncamento estão em risco. Até a correção de 2026-10-17, o primeiro evento em 0 zerava a curva, o que afetava logs com transições simultâneas (MongoDB: 570 visitas). Os resultados anteriores estão em `output/hypotheses_v1_kmbug/` e não devem ser usados.
11. **Pseudonimização:** o *The Public Jira Dataset* já traz responsáveis e autores anonimizados; o protótipo não armazena identidades.
12. **Multitarefa:** a penalidade existe (`--multitask`), mas o κ estimado é 0 nos repositórios examinados (associação negativa entre carga e permanência). Com κ = 0, limitar o WIP só desloca a fila; os ganhos possíveis vêm de evitar bloqueios e do sequenciamento.

## Valores PROVISÓRIOS (fixar *a priori* com o orientador)

| Parâmetro | Valor | Onde |
|---|---|---|
| Erro relativo máximo de P50 / P80 / P90 | 25% / 25% / 35% | `ValidationCriteria` |
| Erro relativo máximo de vazão / WIP | 20% / 15% | `ValidationCriteria` |
| Wasserstein normalizado máximo | 0,35 | `ValidationCriteria` |
| Cobertura mínima de quantis | 2/3 | `ValidationCriteria` |
| Mínimo de casos concluídos na validação | 30 | `ValidationCriteria` |
| Margem de não inferioridade da vazão (H2) | 5% | `ExperimentSettings` |
| Relevância prática do P80 | 5% | `ExperimentSettings` |
| γ1, γ2, γ3 / α, β | 1, 1, 1 / 0,5, 0,5 | `CostSettings` |
| Pesos β1..β4 da restrição | 0,25 cada | `ConstraintSettings` |
| Escala de H4 / lacuna / tempo / memória por nó | 3–9 agentes / 1% / 10 s / 10 MB | `H4Settings` |
| Réplicas de avaliação | 30 (definir por estudo-piloto, Eq. da meia largura) | `ExperimentSettings` |
