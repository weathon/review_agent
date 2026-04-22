# Wasserstein Motifs: Non-deterministic Alignment of Ecological Networks

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 4

## Abstract
We study the problem of ecological network (food web) alignment, where we seek to identify structural equivalences among species and uncover backbones of interactions that represent shared functional substructures. These fundamental properties reveal the functional relationships that sustain ecosystems, enabling more accurate predictions of biodiversity responses to environmental change. Existing methods are computationally expensive, not scalable, and hard to interpret ecologically. We provide a first rigorous formalization of food web alignment based on network motifs, and show existing methods popularized in the ecological community are equivalent to minimizing a Fused Gromov-Wasserstein-like cost functional, termed *Wasserstein Motifs*. Moreover, we propose an interpretable and provably correct algorithm that efficiently computes non-deterministic alignments between food webs by leveraging their representation as feature measure networks. As a byproduct, we introduce a novel approach for identifying the non-deterministic backbones of interactions. Experiments on a continental-scale dataset of 129 Sub-Saharan African mammal food webs demonstrate significant gains in accuracy, scalability, and interpretability over state-of-the-art methods. Our results establish a principled bridge between ecological network science and optimal transport, opening avenues for the analysis of complex structured data.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper formalizes food-web, an ecological network, alignment via a motif-based, non-deterministic OT objective. The core optimization (Eq. 1) fuses a zeroth-order feature term with a first-order topology term and introduces a “self-alignment” penalty to avoid trivial mappings. The paper connects an influential ecological alignment heuristic to a special case of their objective (α=1), propose a KL-BAPG variant with Dykstra projections to compute transport plans under ≤ marginal constraints, and leverage the learned transport to define “non-deterministic backbones” and a fractional triangle-style transitivity score across networks. Experiments on 129 Sub-Saharan mammal food webs report faster runtimes than simulated annealing and qualitatively more ecologically coherent alignments than FGW/partial-FGW baselines, plus meta-alignment analyses by diet/biomass groups and top-k backbone diagnostics).

### Strengths
1. The paper gives clear formalization of a long-standing ecological heuristic. Proposition 2 shows that in the deterministic case with $\alpha$=1 and Pearson-based feature discrepancy, the objective (Eq. 1) matches the annealing score of Mora et al. up to an additive constant, hence they share minimizers.  This provides a principled geometric footing for that heuristic within their OT-style framework.

2. The formulation presented in the paper naturally supports many-to-many correspondences via a transport plan $T$ under $\leq$ marginal constraints. The objective splits into a zeroth-order feature term <T, C> by using motif-role features and a first-order, adjacency-aware term <T,A_1 (T⊙C) A_2> that promotes neighborhood-consistent matches. A self-alignment penalty controlled by ε discourages degenerate self-mass solutions, improving ecological plausibility. Algorithm 1’s KL-BAPG with Dykstra projections gives a lightweight routine and enforces the ≤ marginal constraints by simple multiplicative rescaling, making the method easy to implement and numerically stable.


3. The paper generalizes the backbone concept to non-deterministic mappings by introducing a role-similarity score and a fractional, triangle-based transitivity. Empirically, the resulting top-k backbones are more connected and more transitive than a than a null model at small k shown in Fig. 12.

### Weaknesses
1.	The objective’s first-order term and the linear feature term look close to FGW-style formulations. Without fair, feature-matched baselines, for example, FGW using the same motif-role features, FGW with motifs+traits, it’s hard to isolate what is new: the objective design, the penalty, or simply the features.
2.	The paper does not systematically study the behavior or provide robust defaults for the tradeoff parameter $\alpha$, self-alignment regularization parameter $\varepsilon$  or  step-size $\gamma$ , nor how these interact.
3.	The evaluation is potentially unfair and weak. The paper compares its method using motif-role features against FGW/Partial-FGW baselines using trait features, so improvements may reflect feature choice, not the objective/solver.

### Questions
1. How do results change when FGW uses the same motif features and traits+motifs? Also, could the authors do ablation study of the following: (1) first-order term off, (2) self-alignment penalty off?

2.	If you keep the graph directed in the first-order term (e.g., separate in/out adjacency), do alignments/backbones improve?

3. see weakness 2.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper provides a formal mathematical framework for aligning ecological networks (food webs) using network motifs to identify structurally similar species and shared functional substructures. The authors show that existing motif-based approaches can be interpreted as minimizing a Fused Gromov-Wasserstein-like cost functional, which they call Wasserstein Motifs. Based on this formulation, they introduce an interpretable and theoretically grounded algorithm that efficiently computes non-deterministic (many-to-many) alignments between food webs represented as feature measure networks. The method also includes a procedure for identifying non-deterministic backbones of interactions, which capture functional redundancy among species.

### Strengths
- Captures many-to-many species relationships and functional redundancy
- Achieves computational speedup over previous motif-based approaches
- Identifies non-deterministic backbones revealing shared structural patterns across ecosystems

### Weaknesses
A central limitation is the absence of direct ecological ground truth. The evaluation assumes that identical species across networks have similar functional roles, an indirect validation that weakens confidence in the biological accuracy of the alignments and backbones. Without empirical or expert-labeled data, ecological correctness cannot be firmly established.

The dataset also contains inferred predator-prey interactions based on body size and taxonomy rather than direct field observations. While standard in metaweb studies, such inferred data can introduce bias and uncertainty, potentially affecting the reliability of the results.

Experimental design is limited by the use of fixed hyperparameters without sensitivity or ablation analysis, reducing confidence in the method's robustness and generalizability. Moreover, computing non-deterministic transitivity remains computationally demanding, restricting analysis to small backbones and limiting scalability.

No implementation code or supplementary materials are provided, which undermines reproducibility and prevents independent verification. Finally, the experimental results are largely qualitative and descriptive, lacking rigorous statistical evaluation, making it difficult to draw decisive or generalizable conclusions.

### Questions
Has the framework been tested or considered for other types of ecological or interaction networks? Demonstrating broader applicability could reinforce the method's general utility.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies the problem of identifying similarities in different ecosytems by formulating it as a network alignment problem. Given a directed graph, where nodes are species and edges as predation, the goal is to identify species across different ecosystems (nodes across graphs) that play functionally similar roles. The authors propose an objective function that captures node features, graph structure and a regularizing term to disallow self-alignments (node mapped to itself). They then provide an algorithm that minimizes this objective and prove convergence to stationary point. Experiments are provided on real world data to show that the alignments obtained by their methods are faster and more coherent.

### Strengths
The problem considered and the challenges in solving it are well motivated. The main contributions are the formalization of alignment problem and showing that the heuristic approach used widely in the field, is a special case for their formulation. The algorithmic approach seems to be relied mostly on previous work, but they show that optimizing for alignment under their cost results in more coherent results. As there is no ground-truth, the evaluations are conceptual. The non-deterministic transitivity score is well specified and also consistent with their goal.

### Weaknesses
The theoretical claims may be seen as a bit of oversell. The convergence is proven under an assumption that is not justified well or stated under which conditions might hold, but instead some empirical evidence is provided that this holds. This is a bit non-standard for theoretical proofs. In the comparison against baselines for runtime, is motif computation, enumeration considered? For methods with and without that might be a big difference. Also, the $L1$ alignment distribution is not reported for the baselines.

### Questions
The matrix $C$ is assumed to have positive entries, but with $\varepsilon=1$ and correlation as the dissimilarity, the entries could be negative. Was some other dissimilarity measure used for part of experiments?

### Soundness
2

### Presentation
1

### Contribution
3
