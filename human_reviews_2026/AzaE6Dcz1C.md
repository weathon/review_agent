# Single-Loop Byzantine-Resilient Federated Bilevel Optimization

- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
Federated bilevel optimization plays a crucial role in solving complex problems with nested optimization structures. However, its distributed nature makes it highly susceptible to faulty or Byzantine behaviors. Existing Byzantine-resilient approaches are either restricted to simple single-level optimization problems or rely on sub-loop updates that introduce significant computational and communication overhead. To address these limitations, we propose a family of Byzantine-resilient federated bilevel algorithms, which (i) operate within a single-loop structure, (ii) achieve optimal Byzantine resilience, and (iii) ensure computational and communication efficiency. The core of the proposed method, BR-FedBi, leverages an auxiliary variable that facilitates efficient hypergradient estimation while simultaneously solving the lower- and upper-level problems. Building on BR-FedBi, we further integrate the algorithm with Polyak’s momentum and the probabilistic gradient estimator (PAGE) (Li et al., 2021), resulting in provable optimal Byzantine resilience and optimal sample complexity. Both theoretical analysis and empirical results demonstrate the superior performance of the proposed algorithms.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies federated bilevel optimization (FBO) under Byzantine clients. It first presents an algorithm‑independent asymptotic lower bound. Algorithmically, the paper proposes a single‑loop Byzantine‑robust FBO framework BR‑FedBi, plus a Polyak‑momentum version (BR‑FedBiM) and a PAGE variance‑reduced version (BR‑FedBiP). With momentum, the method is shown to match the lower bound up to constants, claiming “optimal Byzantine robustness.” Experiments also validate the advantages of the proposed algorithm.

### Strengths
1. First Byzantine lower bound for FBO. Extends single‑level robust learning lower bounds to bilevel settings, clarifying that both upper‑ and lower‑level heterogeneity amplify error and that the breakdown depends on $B_f,B_g$. 
2. Theory–algorithm closure. After giving the lower bound, BR‑FedBiM attains it (up to constants), which is convincing.
3. Robust empirical gains across diverse attacks/aggregators with sizeable margins.

### Weaknesses
1. External validity / representativeness. Experiments focus on hyper‑representation with small models and datasets. Add a more realistic, higher‑dimensional FBO (e.g., hyperparameter optimization of weight decay/augmentation, or meta‑learning / personalized FL) and report end‑to‑end communication volume, latency, and speedups to support the single‑loop efficiency claim.
2. Lacked related works. The related works section lacks coverage of distributed stochastic bilevel optimization, particularly decentralized bilevel optimization, which has gained an increasing focus recently. Therefore, the author should include a review of relevant works on decentralized bilevel optimization in this section.
3. Near‑critical $\delta$ behavior & aggregator choice. Theory constrains $\kappa$ vs. $\delta$, but the paper lacks a systematic study when $\delta$ approaches the breakdown. Provide $\delta$ sweeps and compare $\kappa$ across aggregators to probe the near‑bound regime.
4. Assumption strength & measurability. Mean‑square smoothness and $(\zeta,B)$‑heterogeneity are standard but not easily measurable in practice. Offer empirical estimates (from client gradient statistics) for $\zeta_f,\zeta_g,B_f,B_g$ to connect constants to practice.
5. Baseline tuning fairness. Has BILANTINE been tuned fairly for communication rounds, stepsizes, batch sizes? Include ablations under equal communication budget or equal wall‑clock, and disclose full tuning tables.
6. Engineering clarifications. Clarify PAGE trigger probability $p$, mini‑batch sizes $b$, momentum/stepsize coupling (theory ranges vs. actual values). Address practical FL issues (asynchrony, partial participation, client dropouts) experimentally if possible.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes BR-FedBi, a single-loop Byzantine-resilient federated bilevel optimization framework. By introducing an auxiliary variable and integrating Polyak’s momentum with the probabilistic gradient estimator (PAGE), the method achieves efficient hypergradient estimation, optimal Byzantine resilience, and sample complexity. Both theoretical analysis and experiments show strong empirical performance under adversarial settings.

### Strengths
1. The paper addresses an important and challenging problem, Byzantine-resilient bilevel optimization in federated settings, which is timely and underexplored.

2. The proposed single-loop design is elegant and significantly reduces computation and communication overhead compared to two-loop approaches.

3. Theoretical analysis is comprehensive, providing convergence guarantees with optimal sample complexity, supported by consistent empirical results.

### Weaknesses
1. The paper’s presentation could be clearer, especially in explaining how the auxiliary variable interacts with upper and lower levels in the single-loop update.

2. Experimental evaluation is somewhat limited; it would be more convincing to include larger-scale or more diverse federated settings.

3. The robustness analysis mainly relies on existing aggregation rules; more ablation on the aggregation mechanism itself would strengthen the contribution.

4. While the proposed single-loop design is interesting, the experimental section lacks direct comparison with other single-loop bilevel approaches, which would help isolate the benefit of the proposed structure itself.

### Questions
see weakness.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
this paper studies the vulnerability of federated bilevel optimization against Byzantine attacks and the inefficiency of existing sub-loop-based defenses by proposing a family of single-loop Byzantine-resilient algorithms (FBO). The proposed algorithms leverage auxiliary variables, Polyak’s momentum, and variance reduction to balance robustness, computational and communication efficiencies.  The experiments on a set of hyper-representation tasks show the algorithms outperform existing tasks under four attacks and six robust aggregators. Authors also give theoretic analysis regarding algorithm-independent lower bound for FBO under Byzantine attacks.

### Strengths
The paper studies two important problems in federated bilevel optimization including Byzantine vulnerability and computational inefficiency of sub-loop defenses.   It also provides first-of-its-kind theoretical guarantees for Byzantine-resilient federated bilevel optimization.  Experimental results have shown consistently that the proposed method outperform the existing methods in the literature.

### Weaknesses
Experiments use fixed attacker ratios and static attack types, but real-world Byzantine attacks often involve dynamic attacker counts e.g., sudden spikes in faulty clients or adaptive attack strategies. The paper does not test how algorithms adapt to such changes.

### Questions
Can authors elaborate the potential applications of the proposed method in real-world scenarios ?

### Soundness
3

### Presentation
3

### Contribution
3
