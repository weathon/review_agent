# All-Task Convergence and Backward Transfer in Federated Domain-Incremental Learning with Partial Participation

- Decision: Reject
- Scores: 2, 6, 8

## Abstract
Real-world federated systems seldom operate on static data: input distributions drift while privacy rules forbid raw-data sharing.  We study this setting as Federated Domain-Incremental Learning (FDIL), where (i) clients are heterogeneous, (ii) tasks arrive sequentially with shifting domains, yet (iii) the label space remains fixed. Two theoretical pillars remain missing for FDIL under realistic deployment: a guarantee of backward knowledge transfer (BKT) and a convergence rate that holds across the sequence of *all* tasks with ***partial participation***.  We introduce SPECIAL (Server-Proximal Efficient Continual Aggregation for Learning), a simple, memory-free FDIL algorithm that adds a single server-side ``anchor'' to vanilla FedAvg: in each round, the server nudges the uniformly sampled participated clients update toward the previous global model with a lightweight proximal term.  This anchor curbs cumulative drift without replay buffers, synthetic data, or task-specific heads, keeping communication and model size unchanged.  Our theory shows that SPECIAL (i) *preserves earlier tasks:* a BKT bound caps any increase in prior-task loss by a drift-controlled term that shrinks with more rounds, local epochs, and participating clients; and (ii) *learns efficiently across all tasks:* the first communication-efficient non-convex convergence rate for FDIL with partial participation, $\mathcal{O}(\sqrt{E/\left(NT\right)})$,  with $E$ local epochs, $T$ communication rounds, and $N$ participated clients per round, matching single-task FedAvg while explicitly separating optimization variance from inter-task drift. Experimental results further demonstrate the effectiveness of SPECIAL.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes SPECIAL (Server-Proximal Efficient Continual Aggregation for Learning), a theoretically grounded and memory-free algorithm for Federated Domain-Incremental Learning (FDIL) with partial client participation. The core idea is to introduce a server-side proximal anchor that links each task’s aggregated update to the previous global model. This anchor regularizes the optimization path to control cumulative parameter drift, mitigating catastrophic forgetting without storing past data or adding task-specific heads. The authors claim two main theoretical advances: (1) the first Backward Knowledge Transfer (BKT); and (2) the all-task convergence rate for FDIL.

Empirically, SPECIAL is evaluated on three vision benchmarks—Digit-10, VLCS, and PACS—using ResNet-18 as the backbone. However, while the theoretical framework is elegantly derived and the proofs appear rigorous, the experimental validation is weak and narrow in scope. All experiments are restricted to small-scale vision datasets with ResNet-18, which fails to substantiate the paper’s strong claim of achieving “all-task convergence” across diverse domains. The method is not tested on other modalities (e.g., text, multimodal) or larger-scale federated systems, leaving its generality and scalability uncertain. The improvements in ACC and BWT are modest (typically 1–2%), with no runtime comparisons. Moreover, the paper’s practical novelty is limited—the proposed server-side proximal term is a simple extension of FedProx or EWC-like regularization, repackaged with new theoretical framing. The claim of being the first “all-task convergence” method is somewhat overstated given similar analyses in prior continual and federated literature, albeit under different assumptions. The term “ALL-TASK” in particular is exaggerated relative to the empirical evidence, especially since only a few vision tasks are considered in 2025, where expectations for multimodal or large-scale evaluations are higher.

### Strengths
Good theoretical foundation: The paper presents rigorous mathematical analyses, including a Backward Knowledge Transfer (BKT) bound and an all-task convergence rate for Federated Domain-Incremental Learning (FDIL) under partial client participation — both of which are rarely addressed in prior work.

Well-written and structured: The paper is well-organized, with clear motivation and theoretical derivations.

### Weaknesses
1.	Limited experimental scope. All experiments are performed on small-scale vision datasets with ResNet-18, which severely limits the generality of the results. The “ALL-TASK” claim is overstated, given that no multimodal, NLP, or large-scale federated settings are tested.

2.	Weak empirical novelty. The method is essentially a server-side variant of FedProx/EWC with a new theoretical justification. Its algorithmic contribution is modest relative to existing regularization-based approaches.

3.	No efficiency or scalability analysis. The paper does not evaluate computational, communication, or convergence-time overhead, even though the theoretical results emphasize communication efficiency.

4.	Disconnect between theoretical claims and practice. The “all-task convergence” theorem is theoretically appealing but only partially verified empirically. The experiments do not demonstrate convergence across a large number of tasks or diverse domains.

5.	Overgeneralized claims for 2025 standards. In 2025, “ALL-TASK” convergence across domains is expected to cover heterogeneous architectures and multimodal tasks, not just ResNet-18 on vision benchmarks. The claim feels overexaggerated relative to evidence.

### Questions
See Weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The authors introduce SPECIAL, a simple, memory-free algorithm that adds a lightweight server-side proximal anchor to FedAvg to address catastrophic forgetting in federated domain-incremental learning. It theoretically guarantees both backward knowledge transfer (BKT) and all-task convergence under partial client participation, with the same communication efficiency as FedAvg. Experiments on Digit-10, VLCS, and PACS show SPECIAL achieves the best average accuracy among memory-free methods and competitive backward transfer, validating its theoretical advantages in balancing stability and plasticity.

### Strengths
(1) Adds only a lightweight proximal step at the server side without extra memory or communication. Also the authors provide theoretical proof to support this contribution.

(2) Consistent improvements in average accuracy (ACC) and competitive backward transfer (BWT) across multiple datasets, and matches FedAvg’s efficiency while handling non-IID and temporally drifting data.

(3) The problem set up is interesting, which makes a progress towards more realistic scenario.

### Weaknesses
(1) Experiments are conducted only on small-scale image datasets (Digit-10, VLCS, PACS), which limits evidence of scalability and applicability to more realistic federated or multimodal scenarios.

(2) The paper could better articulate what SPECIAL fundamentally offers beyond combining existing strategies for stability, forgetting mitigation, and partial participation. A clearer discussion or ablation contrasting SPECIAL with such composite baselines would help isolate its true contribution.

(3) The proximal weight λ must be tuned separately for each dataset (e.g., 0.25 for Digit-10, 0.4 for VLCS, 0.05 for PACS), suggesting dataset-dependent behavior that may complicate deployment in dynamic or heterogeneous real-world environments.

### Questions
Please address the comments in the weakness section.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper presents a novel federated domain incremental learning algorithm by adding a single server side anchor term to the vanilla FedAvg algorithm. This small change makes the model more stable to distribution shifts as demonstrated in the results on three benchmark datasets. Theoretical guarantees on BKT and all-task convergence rate are also provided.

### Strengths
The paper is well presented and presents the motivation, method, theory and results in a coherent manner. 

Simple addition of an “anchor” on the server side seems to provide improvements on the FedAvg algorithm.

### Weaknesses
The number of datasets is limited. The paper can probably add 1 or 2 more datasets to the analysis.

### Questions
It is not exactly required in the paper but I am curious about the following extensions:

1. How does this algorithm extend to other domains like text or time-series models? Would it easily extend to these kinds of data as well?

2. What would need to change if there are new classes that are introduced on some of the client nodes? Would the algorithm extend to this use-case as well?

### Soundness
3

### Presentation
3

### Contribution
3
