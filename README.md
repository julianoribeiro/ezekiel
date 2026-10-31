# ezekiel

Protótipo da tese de doutorado de Juliano Ribeiro sobre **coordenação distribuída de decisões em processos interdependentes**. O protótipo combina mineração de processos centrada em objetos, simulação de eventos discretos, otimização distribuída por restrições (DCOP/ADOPT) e Teoria das Restrições.

Ele testa as quatro hipóteses da tese com dados **públicos** de repositórios Jira:

| Hipótese | Pergunta | Como é testada |
|---|---|---|
| **H1** fidelidade | O simulador calibrado reproduz uma janela de validação que não usou? | Calibra em [t−365 d, t) e simula [t, t+180 d) a partir do estado real em t. Compara quantis do *lead time*, distribuições, vazão, WIP, utilização e rotas. |
| **H2** coordenação | A coordenação distribuída (C2, ADOPT) reduz o P80 do *lead time* em relação a decisões locais (C1) sem perder vazão? | Comparação pareada por réplica, com números aleatórios comuns e teste de não inferioridade da vazão. |
| **H3** termos da ToC | Os termos da Teoria das Restrições na função objetivo melhoram o resultado? | Ablação: C2 contra C2 com α = β = 0. |
| **H4** viabilidade | O ADOPT escala e se aproxima da solução centralizada exata (C3)? | Lacuna de otimalidade, ciclos, mensagens, tempo e memória, nos repositórios e em instâncias sintéticas. |

O mapa completo entre hipóteses, código e decisões de modelagem está em [docs/hipoteses.md](docs/hipoteses.md). Os critérios marcados como **provisórios** ainda serão fixados *a priori*, antes da análise confirmatória.

## Dados

- **Fonte:** *The Public Jira Dataset* (Montgomery, Lüders e Maalej, MSR 2022), com 16 repositórios Jira públicos, licenciado sob CC BY 4.0. Os autores do conjunto anonimizaram pessoas por UUID. Disponível em [doi.org/10.5281/zenodo.5882881](https://doi.org/10.5281/zenodo.5882881).
- **Logs derivados:** cada projeto é convertido para **OCEL 2.0** multiobjeto (itens, versões, sprints, componentes e equipes). Junto vai um arquivo `.meta.json` com origem, regras de conversão e o nível de evidência de cada relação (explícita, derivada ou inferida). As relações inferidas ficam fora do log principal.
- **Fora do repositório:** os dados (`ocels/`, cerca de 14 GB) e os resultados (`output/`) não estão aqui. Siga "Reproduzir" para reconstruí-los.

Nenhum dado de organização específica é coletado ou usado.

## Instalação

Requer Python 3.12.

```bash
bash scripts/setup_env.sh          # cria .venv com as versões de requirements.lock e roda os testes
source .venv/bin/activate
```

## Reproduzir

**1. Obter e converter os dados.** É preciso ter o MongoDB instalado (`mongod` e `mongorestore`).

```bash
# baixe o dump do Zenodo para ocels/ThePublicJiraDataset/3. DataDump/
mongod --dbpath <pasta> --port 27018 --bind_ip 127.0.0.1 &
python scripts/convert_jira_ocel2_all.py              # os 16 projetos → ocels/ocel2/*.json
python scripts/convert_jira_ocel2_all.py --only apache-zookeeper
```

O conversor restaura uma coleção por vez e a apaga antes da próxima, então nunca há mais de um repositório em disco.

**2. Testar as hipóteses.**

```bash
python scripts/run_hypotheses.py --h4                              # 16 repositórios + varredura sintética de H4
python scripts/run_hypotheses.py --only mariadb-mdev               # um repositório
python scripts/run_hypotheses.py --org por_papel por_estagio       # compara granularidades de agentes
python scripts/run_hypotheses.py --only mariadb-mdev --quick       # exploração rápida
```

Opções de modelagem, desligadas por padrão e com decisão pendente:

```bash
--calendar business        # tempo em dias úteis
--couplings                # sincronização entre objetos (bloqueios) e pessoas compartilhadas
--multitask estimated      # penalidade de multitarefa estimada na calibração
```

**3. Análise de sensibilidade.**

```bash
python scripts/run_sensitivity.py --only apache-zookeeper                                  # local, um fator por vez
python scripts/run_sensitivity.py --only mariadb-mdev --factorial "demand:−20%/+20%" "couplings:ref/ligados"
```

Os resultados vão para `output/hypotheses/<organização>/` e `output/sensitivity/`. Cada execução grava um **manifesto** com:
- revisão git e SHA-256 do código;
- ambiente;
- SHA-256 dos logs e dos `.meta.json`;
- configuração completa, sementes e comando.

## Estrutura

```
ezekiel/
  data/eventlog.py          log de casos e visitas, calendário de trabalho, métricas observadas
  ocel/                     leitura, validação e caracterização de OCEL 2.0
  mining/                   restrição ativa, descoberta de processos e conformidade
  simulation/               simulador SimPy (Kaplan-Meier, capacidade, WIP, classes, acoplamentos) e calibração
  dcop/                     modelo organizacional, grafo de restrições por evidências, custos, ADOPT, C3 exato
  evaluation/               validação de H1 e estatística pareada (t/Wilcoxon, Holm, não inferioridade)
  experiments/              H1–H4, monitoramento e reotimização, sensibilidade, instâncias sintéticas de H4
  manifest.py               manifesto de execução
  legacy/                   implementação anterior às hipóteses atuais (só para a API de demonstração)
config/organizacao/         modelos organizacionais (agentes = unidades decisórias)
scripts/                    conversão dos dados, execução das hipóteses e da sensibilidade
tests/                      verificação (ADOPT = C3 = força bruta, conservação, números aleatórios comuns etc.)
docs/hipoteses.md           mapa hipótese → código, decisões de modelagem e valores provisórios
```

## Agentes

Os agentes são **unidades organizacionais**, não estágios do fluxo. No Jira público, as equipes não são observáveis, então o modelo principal (`por_papel`: contribuidores e mantenedores) é uma **premissa declarada**. A granularidade `por_estagio` serve só para comparação.

## Citação dos dados

> Montgomery, L.; Lüders, C.; Maalej, W. *An Alternative Issue Tracking Dataset of Public Jira Repositories*. In: 2022 IEEE/ACM 19th International Conference on Mining Software Repositories (MSR), 2022, p. 73–77.

## Licença

- **Dados:** os dados e os logs derivados seguem a CC BY 4.0 do conjunto original, com atribuição aos autores acima.
- **Código:** a licença ainda não foi definida.
