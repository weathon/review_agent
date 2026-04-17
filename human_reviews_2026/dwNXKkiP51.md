# Initialization Schemes for Kolmogorov–Arnold Networks: An Empirical Study

- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Kolmogorov–Arnold Networks (KANs) are a recently introduced neural architecture that replace fixed nonlinearities with trainable activation functions, offering enhanced flexibility and interpretability. While KANs have been applied successfully across scientific and machine learning tasks, their initialization strategies remain largely unexplored. In this work, we study initialization schemes for spline-based KANs, proposing two theory-driven approaches inspired by LeCun and Glorot, as well as an empirical power-law family with tunable exponents. Our evaluation combines large-scale grid searches on function fitting and forward PDE benchmarks, an analysis of training dynamics through the lens of the Neural Tangent Kernel, and evaluations on a subset of the Feynman dataset. Our findings indicate that the Glorot-inspired initialization significantly outperforms the baseline in parameter-rich models, while power-law initialization achieves the strongest performance overall, both across tasks and for architectures of varying size. This work underscores initialization as a key factor in KAN performance and introduces practical strategies to improve it.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper investigates initialization strategies for spline-based Kolmogorov-Arnold Networks (KANs), a relatively new neural network architecture. The authors argue that proper initialization is crucial for KAN performance but remains under-explored, and methods from traditional MLPs are not directly applicable. The paper proposes and evaluates several initialization schemes: two theoretically-motivated approaches inspired by LeCun and Glorot/Xavier variance preservation principles (including a variant using batch-normalized spline bases), and an empirical power-law family with tunable exponents. These schemes are evaluated through large-scale grid searches on function fitting and forward PDE approximation tasks, analysis of training dynamics (loss curves, Neural Tangent Kernel spectrum), and testing on a subset of the Feynman dataset. Key findings suggest that the Glorot-inspired scheme performs well, especially for larger models, while the empirical power-law initialization consistently achieves the best performance across tasks and architectures.

### Strengths
1: Addresses a critical and timely aspect of training KANs – initialization,  and provides the first systematic comparison of different initialization strategies specifically tailored for spline-based KANs. 

2: The proposed power-law initialization demonstrably leads to superior performance across a wide range of settings in the experiments, offering a practical and effective strategy. The finding that small $\alpha$ and large $\beta$ are generally preferred provides a useful heuristic.

### Weaknesses
1. The paper is purely empirical, lacking a theoretical justification for the effectiveness of methods. For example, why do small $\alpha$ and large $\beta$ work well for power-law initialization? If relying solely on empirical search, there may be a suspicion of parametric-innovation.

2. The data in experimental is relatively simple, considering only function fitting and PDE, which limits the generalizability of the author's conclusions.

3. Lack of in-depth analysis on the reasons for poor initialization performance, such as for LeCun-normalized.

### Questions
1. Could you comment on how the performance of KANs with the proposed initializations compares to standard, well-initialized MLPs on some of the benchmark tasks?

2.Could the sensitivity to the variance equipartition assumption in the LeCun/Glorot derivations be investigated? What happens if different partitions are assumed?

3.Can the authors provide more insight into why the LeCun-normalized scheme failed to perform well? Is there an interaction between batch normalization and the spline basis learning?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper explores three new initialization schemes for KANs, identifying a current gap that KAN implementations rely on the standard initialization presented in the original KAN paper. The first two initialization methods explored LeCun and Glorot, are based in theory, the first driven by variance preservation, in which the variance of the inputs should match the variance of the outputs (LeCun) and the other driven by stabilizing variance for activations and gradients across layers. Additionally, the paper explores a third, empirical initialization based on the power-law scaling of KAN structure, that includes tunable exponents on the KAN's architecture (termed "power-law" in the paper). The paper goes on to evaluate the accuracy of these initialization methods on two main tasks, function fitting on five 2D target functions, and forward PDE problems. On both of these tasks, power-law initialization significantly outperforms other methods, with Glorot initialization as a distant second method. Additionally, the authors explore training stability, comparing loss curves and eigenvalue spectra of the NTK matrix.

### Strengths
The paper addresses an open question in KAN training and successfully demonstrates different initialization variants and their relative benefits. The paper motivates well the use of the two theoretically-driven initialization methods, and selects a good set of tasks and demonstrations to illustrate the relative benefits of each method. This is very fundamental work that is important to enabling the community to practically use KANs.

### Weaknesses
I appreciated the theoretical motivation of LeCun and Glorot initializations but was disappointed that there wasn't much discussions on why these methods behaved in the way they did. Table 1 for example shows that only about 15% of runs with LeCun initialization outperform the baseline, but there isn't an exploration into why this might be the case, or an exploration of the dynamics of this method of initialization. The same holds true for the results in Table 2. 

I'm not quite sure I understand what is being demonstrated in Figure 1 for the large architecture, as training instability for the power law initialization dominates the dynamics. The authors say "The more pronounced oscillations observed in its curves stem from the fact that the model already reaches very low loss values under a fixed learning rate, leaving no smoothing effect from adaptive adjustments." but the learning rate in these plots is fixed to 10e-3 so I'm not sure what is meant by "no smoothing effect from adaptive adjustments" unless the learning rate is in fact changing. The same instability concern holds true for the PDE case.

Overall, I have concerns with the power law initialization, as I believe this method is prone to overfitting, and that results of this method are artificially inflated. I'm not quite sure that comparing the performance of this method with the theoretically motivated initializations.

### Questions
I would love to understand better what is driving the dynamics of the LeCun and Glorot initialization methods. I would also like to better understand the decision to add tunable parameters to an initialization method, whether there is a precedent here, and why we believe that would not result in overfitting.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes three families of initialization for spline‑based KANs: (i) LeCun‑inspired schemes (one ``numerical,'' one using batch‑normalized spline bases), (ii) a Glorot‑inspired scheme that balances forward/backward variances, and (iii) an empirical power‑law that scales residual/basis weights via exponents $\alpha, \beta$. Experiments include 2‑D function fitting, three forward PDEs (Allen–Cahn, Burgers, 2‑D Helmholtz), and a subset of the Feynman dataset; training‑dynamics are visualized with both loss curve analysis and NTK spectra. The numerics show that both Glorot and tuned power‑law tends to beat the original KAN baseline, and tuned power‑law is the strongest overall.

### Strengths
1. Overall the paper is clearly written with clear problem statement and clean formulation for the KAN layer. The LeCun/Glorot initializations are derived explicitly.
2. The paper include broad and thorough experimentation. The NTK spectra are a nice lens to visualize conditioning differences for different initialization.
3. Even though the paper is empirical, better initialization recipes in different scenarios for KANs are of practical relevance.

### Weaknesses
1. Although the authors claim that the power-law initialization has the best overall performance, its performance is based on a grid search over $\alpha,\beta$ and only the best‑performing configuration is retained. In practice, such tuning can be costly, and the paper offers no low‑budget recipe to make the scheme usable out of the box.
2. The LeCun/Glorot initialization are derived by applying the standard variance‑preserving idea to the KAN layer and rely on a strong equipartition assumption across the 
$G+k+1$ terms. The paper itself notes this simplification and that an alternative split performed worse. Meanwhile, the best results come from the empirical power‑law, for which the paper offers no mechanistic explanation beyond trends in heatmaps. The authors acknowledge this limitation, but as it stands, the conceptual contribution feels incremental and the empirical win might due to a tuned heuristic.

### Questions
The author's best region favors small $\alpha$ and larger $\beta$. Could the authors provide a heuristic/NTK‑based rationale and show how eigenvalue conditioning varies across $(\alpha,\beta)$?

### Soundness
2

### Presentation
3

### Contribution
3
