# Learning a Stable Reservoir from a Single Trajectory via Persistent Loops and Markov Flow

- Decision: Reject
- Scores: 6, 8, 4, 4

## Abstract
We study whether embedding global topology and local transport into a fixed reservoir can improve phase tracking and prediction. From a single delay‑embedded trajectory, we build a recurrent operator in two parts: (i) long‑lived $H_1$ classes from persistent cohomology are converted to circular coordinates whose average phase velocities instantiate stable $2\times2$ rotation blocks, and (ii) short‑horizon transition counts over a coarse partition define a Markov model whose action is lifted back to neuron space through sparse, stochastic pooling and lifting maps. A convex blend of these topological and flow components is scaled by power iteration to a preset operator‑norm bound, yielding a leaky ESN with a straightforward echo‑state guarantee; only a ridge‑regularized linear readout is trained. The resulting reservoir is fixed, interpretable, and analyzable: its internal oscillators reflect the attractor’s dominant loops, while its couplings align with observed local transport. In experiments on chaotic systems and real‑world series, the method is data‑efficient and maintains the computational profile of standard ESNs, while delivering improved phase tracking and competitive—often superior—multistep forecasts relative to tuned random reservoirs of the same size. Overall, the framework offers a principled alternative to sampling‑based wiring by learning the reservoir once from data.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
## Summary

This paper introduces the Persistent Homology Reservoir (PHR), a novel framework for designing the recurrent operator (W) of an Echo State Network (ESN). Different from random fixed weights, PHR learns the reservoir structure offline from a single observed trajectory by explicitly embedding the system's global topology and local dynamics.

The proposed reservoir $W$ is constructed as a convex blend of two main components: Topological Operator which aims to capture long-term recurrent dynamics and Flow Operator which captures short-term local transport. The state space is partitioned by k-means, and a Markov transition matrix is estimated based on observed transitions. This coarse dynamic is lifted back to the reservoir space via stochastic pooling and lifting maps.

Experiments are conducted on several chaotic systems and real-world time series forecasting demonstrate that PHR consistently outperforms various baselines, particularly in long-horizon forecasting.

### Strengths
## Strengths
- This paper makes a substantive advance in reservoir computing. Instead of heuristic random initialization, the authors propose a structured, data-driven design from system geometry and dynamics. It integrates topological data analysis (TDA) for global oscillatory structure with operator-theoretic methods for local flow, with theoretical support and stronger empirical results.
- PHR demonstrates superior empirical performance over several strong baselines and cover a comprehensive sets of benchmarks. The improvements in Valid Prediction Time on chaotic systems are particularly attractive, showing better emulation of the underlying autonomous dynamics rollouts.
- The evaluation is rigorous and comprehensive with a detailed ablation study effectively isolates the value of the topology and flow components and clearly demonstrates the contributions of each component.

### Weaknesses
## Weakness
- The primary weakness is the extreme density of the presentation and the high complexity of the methodology. The construction relies heavily on advanced concepts from algebraic topology (e.g., persistent cohomology, cocycles, Rips skeletons, discrete harmonic extension, Dirichlet energy). This presents a substantial barrier to entry for the broader machine learning community. The paper is mathematically dense and assumes a high level of familiarity with TDA.

- While the training (readout fitting) remains efficient, the offline construction of W involves computationally expensive steps, notably the O(n²) cost for the distance matrix and the subsequent persistent homology computations. Although subsampling is used to mitigate this, the construction phase is significantly heavier than the near-instantaneous initialization of a standard ESN.

### Questions
## Questions 
- Could the authors provide a more intuitive explanation of how the harmonic extension produces meaningful circular coordinates for a non-expert reader? Furthermore, while the ablation study shows PH-derived coordinates outperform than PCA, could you elaborate on why this approach is necessary and what specific advantages it offers over simpler geometric estimations? I suggest include these explantions in the camera-ready version, which would greatly improve the presenation and readibility of the paper.

- How sensitive is the performance of PHR to the choice of delay embedding parameters and the number of clusters? Are the dominant H1 classes and the subsequent Markov flow robustly identified across different embeddings and partitions?

- Lemma 3.1 indicates that the fidelity of the flow channel depends on the pool-lift calibration defect $AB−I_Q$. How close to the identity is $AB$ in practice with the current stochastic construction of $A$ and $B$, and how does the sparsity affect this defect?

I am not an expert in reservoir computing. If the authors can address the concerns above, I am willing to increase my score accordingly.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
Summary

The manuscript describes a novel form of reservoir computing
which learns the structure of a reservoir from the data to be
predicted. It is applied to chaotic time-series prediction problems.
The main idea is construct the dynamics of the reservoir such that
two characteristics are mixed:
First, the occurrence of oscillatory limit cycles. And second,
stochastic transitions between them.
A linear readout is finally trained on the reservoir activity
to perform prediction.
The power of the presented approach is demonstrated on a number
of established benchmark problems (chaotic attractors) as well
as on real-world benchmark tests. The presented approach in most
cases provides smallest root mean square error compared to the other
approaches.

Soundness

The manuscripts appears to be written with much care and mathematical
derivations are presented in detail in the appendix. Since the work
lies outside of my core expertise, I could not check the proofs in
detail. The empirical evidence, that the presented algorithm outperforms
a large number of alternatives is a convincing and potentially impactful
result.

Presentation

The presentation is surely gauged to the expert reader who is familiar
with the employed mathematical tools. For such a reader, I believe, the
manuscript provides the important steps to reproduce the work. Also the
authors strive to move detailed proofs to appendices, thus improving
readability for readers outside their core community.
The main ideas are also presented verbally and one gets an idea of the
approach even if not from the core field.
Some of the text is, however, still very dense.



Contribution

The main contribution of this work is to propose an algorithm to construct
a reservoir from direct observations of the data, rather than using, for example,
randomly coupled networks as a reservoir.
A main result is that on the one hand, the authors are able to guarantee
stability criteria (echo state property) and on the other hand they
demonstrate competitive performance of the approach, in parts by far exceeding
what alternative methods achieve. This is a remarkable improvement.

### Strengths
see "contribution" above

### Weaknesses
The presentation in the main text would in my opinion benefit from
a less technical presentation to reach a larger readership, providing
explanations in less technical terms, that may not be known to many
participants of the conference.
However, I don't consider myself an expert in this very field, so
I would weigh the assessment by more expert reviewers higher.

### Questions
The method of construction of the reservoir, by implementing 2 x 2 rotations,
seems to be gauged towards time-series problems that contain period orbits.

Also it seems that the algorithm is best suited for time-series prediction,
as its aim appears to be a faithful reconstruction of the observed
time series.
Did the authors think about other applications, such as classification?
(for example language, spoken digit classification). I don't propose to
extend the experiments, but rather would like the authors to comment,
whether they expect advantages in such applications as well.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes Persistent Homology Reservoir, a new framework for constructing a stable and interpretable recurrent operator for Echo State Networks (ESNs). Instead of using random reservoir weights, the authors learn a fixed reservoir directly from one observed trajectory by combining two components: (1) Persistent cohomology-derived circular coordinates that define internal oscillators capturing long-lived topological loops, and (2) A lifted Markov operator that encodes local flow transitions between coarse partitions of the embedded data. A power-iteration scaling step ensures the Echo State Property by bounding the operator norm. The resulting reservoir is fixed, stable, and interpretable, with a simple ridge regression readout.

Experiments on chaotic and real-world time series show improved forecasting accuracy and longer stable horizons compared to standard and structured ESNs. The theoretical section is detailed, including proofs of contraction and eigenpair persistence, and the method is backed by thorough ablations.

However, the paper is very difficult to follow: derivations are overly dense, notation is heavy, and key intuitions are buried in long proofs and algorithm boxes. Although the framework is looks promising, the conceptual message is obscured by excessive formalism and long mathematical detours.
Empirical studies, while strong, focus mostly on low-dimensional chaotic systems; scalability and generalization to modern high-dimensional ML benchmarks remain unclear.

Overall, the idea of learning a reservoir from topological and flow structure is novel and interesting, but the presentation lacks accessibility and practical clarity. The paper would benefit from a simpler exposition, clearer ablations, and discussion of computational limits.

### Strengths
The paper introduces a novel, principled way to learn a stable reservoir directly from data by combining persistent-homology-based oscillators with a lifted Markov flow operator. It provides a stability guarantee (Echo State Property) through explicit norm scaling, and the resulting reservoir is interpretable, with internal modes corresponding to data-driven loops and flows.
Experiments on several chaotic and real-world series show competitive or superior performance to standard ESNs, supported by ablations. Overall, it offers an innovative and theoretically grounded direction that connects topology, dynamics, and reservoir computing.

### Weaknesses
I personally don't like the theoretical burnen in this paper. Without proper demonstration just figure 1 (a sketch for the model), the rest are all theory which creates trouble to verify in the short peorid of time.

### Questions
1. Can you validate on more complexed senorio like 2D kolmogorov flow?
2. How's the scalablility of this framework?

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
Authors proposes PHR (Persistent Homology Reservoir): instead of drawing an ESN reservoir at random and scaling it, they learn a fixed reservoir once from a single delay-embedded trajectory by combining (1) a topology-driven rotation operator built from persistent cohomology and (2) a lifted Markov flow operator built from short-horizon transition counts, then (3) power-scale the blend to get an explicit echo-state/stability guarantee. Only the linear readout is trained.

### Strengths
+ Technical novelty: The persistent-homology-informed reservoir and lifted Markov flow operator are both novel and concretely implemented. 

+ Empirical support: the method is consistent across synthetic and real datasets.

### Weaknesses
- Clarity / completeness: Some parts (auto-tuning details, how unused reservoir units are filled, PH parameter sensitivity) need more explicit description for reproducibility.

- Scope: Evaluation sticks to ESN-style comparisons; missing baselines from Koopman or DMD families limits broader impact.

- Robustness analysis: Still somewhat heuristic; PH and k-means steps might be brittle for noisy or non-stationary data.

- Presentation: I'd suggest moving much math to the appendix and include Figures summarizing the method and motivation. Current Fig1 is very hard to read.

### Questions
- “W_top is formed … and randomly permuted to distribute the oscillator pairs across the reservoir; remaining coordinates receive decaying radii.” what is the rule for filling the rest of the reservoir when you have, say, X units but only Y loops. Can you give a concrete example to help undertsand it.

- Line 1080 "The noise channel breaks degeneracies and complements the basis without compromising
stability, as its contribution is explicitly budgeted by ξ and then squashed by the global
scaling to ρ⋆." can you illustrate this part?

### Soundness
3

### Presentation
2

### Contribution
3
