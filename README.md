# ArchitectureScope — Software Architecture Style Classifier

A Python program that classifies a software system into one of six
architecture styles from eight measurable structural features, using a
k-nearest-neighbours classifier built from scratch. Standard library
only — no dependencies, no API keys, no network access.

Author: **Abel Haile Magicho** — [github.com/kembAB](https://github.com/kembAB)

## The six styles

| Style | In one line |
|---|---|
| Monolith | One deployable unit holding all functionality over a shared database. |
| Layered | Functionality organised in tiers that call downward, deployed together. |
| Client-Server | Many client units doing presentation, one server side holding logic and data. |
| Microservices | Many independently deployable services, each owning its data store. |
| Event-Driven | Units communicate by publishing and reacting to events, asynchronously. |
| Pipe-and-Filter | Data flows through a sequence of independent transformation stages. |

## The eight features

| # | Feature | Meaning |
|---|---|---|
| 1 | `deployment_units` | How many separately deployed units the system has. |
| 2 | `independent_deployment` | Ratio 0.0–1.0 of units deployable without touching the others. |
| 3 | `own_database_per_unit` | 1 when each unit owns its data store, else 0. |
| 4 | `async_communication_ratio` | Ratio 0.0–1.0 of communication that is asynchronous messaging. |
| 5 | `coupling_score` | 0–10; higher means tighter coupling between units. |
| 6 | `sequential_pipeline_flow` | 1 when data mainly flows through transformation stages, else 0. |
| 7 | `client_request_driven` | 1 when work is mainly initiated by client requests, else 0. |
| 8 | `team_autonomy` | 0–10; how independently separate teams own separate units. |

## How it works

1. A labelled dataset of 36 example systems (6 per style) describes each
   system with the same eight features.
2. Features are scaled to a common 0.0–1.0 range with min-max
   normalization, so no single feature dominates by its unit of measure.
3. A new system is compared with every labelled example using Euclidean
   distance over the normalized features.
4. The three nearest neighbours vote; the majority style wins. Ties are
   broken by total distance, then by the single nearest neighbour.
5. Every classification returns its three nearest neighbours as
   evidence, so a human can audit why the decision was made.

## How to run

```bash
python architecture_classifier.py
```

The program prints the dataset summary, a leave-one-out
cross-validation evaluation, and live classifications of five new
systems that are not in the dataset. The full run is saved in
`output.txt`.

## Evaluation result (real run)

Leave-one-out cross-validation over all 36 systems — each system
classified using the other 35 as the reference set:

- **Overall accuracy: 36/36 = 100.0%**
- Per style: Client-Server 6/6, Event-Driven 6/6, Layered 6/6,
  Microservices 6/6, Monolith 6/6, Pipe-and-Filter 6/6
- Misclassifications: none

Demo classifications of five new systems, all with unanimous
neighbour evidence:

| System | Predicted style |
|---|---|
| Harbour Books online store | Monolith |
| Continental Bank service platform | Microservices |
| Factory telemetry monitoring system | Event-Driven |
| PhotoLab enhancement tool | Pipe-and-Filter |
| Dental clinic reception system | Client-Server |

## Honest limits

- The dataset is small (36 systems) and hand-crafted from the published
  characteristics of each style, so the perfect score is best read as
  an upper bound: the classifier partly recovers the definitions used
  to author the data. Real measured systems would be messier.
- Two features (coupling, team autonomy) are judgement scores rather
  than direct measurements; different assessors would score them
  differently.
- Only six styles are covered, and real systems often blend styles.
- The classifier is deliberately simple; no feature weighting or
  learned model is used.

## Files

- `architecture_classifier.py` — the complete program
- `output.txt` — the full real run
- `software-architecture-classification-thesis.pdf` — the project thesis
- `software-architecture-classifier-explainer.pdf` — the project explainer
