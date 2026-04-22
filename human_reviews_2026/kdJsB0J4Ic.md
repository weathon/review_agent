# Structure Learning from Time-Series Data with Lag-Agnostic Structural Prior

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Learning instantaneous and time-lagged causal relationships from time-series data is essential for uncovering fine-grained, temporally-aware interactions. Although this problem has been formulated as a continuous optimization task amenable to modern machine learning methods, existing approaches largely neglect the use of coarse-grained, lag-agnostic causal priors, an important form of prior knowledge that is often available in practice. To address this gap, we propose a novel framework for structure learning from time series to integrate lag-agnostic priors, enabling the discovery of lag-specific causal links without requiring precise temporal annotations. We introduce formulations to precisely characterize the lag-agnostic prior, and demonstrate their consequential and process-equivalence to priors, maintaining consistency with the intended semantics of the priors throughout optimization. We further analyze the challenge for optimization due to the increased non-convexity by lag-agnostic prior constraints, and introduce a data-driven initialization to mitigate this issue. Experiments on both synthetic and real-world datasets show that our method effectively incorporates lag-agnostic prior knowledge to enhance the recovery of fine-grained, lag-aware structures.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper considers  the integration of coarse-grained lag-agnostic causal priors. 
The main claim is that lag-agnostic priors can enable the discovery of lag-specific causal links.
The main effort of the paper is put on how to integrate such prior during the practical optimization procedure, and it provides both theoretical and empirical analysis.

### Strengths
- lag-agnostic structural priors is an interesting formulation, and it can be useful in practice
- The theoretical analysis is clear and rigorous. It reveals the challenges during the optimization and why the proposed  logic-dual Formulation is essential.

### Weaknesses
- In Section 4.2 and Figure 1, it would be more comprehensive if a baseline using equation (9) can be added. Such empirical results would further justify the discussion about process-equivalent approach.
- The concrete research problem is not sufficiently introduced. For example, the concept of "process equivalence" lacks a more formal definition. I suggest a minor adjustment to emphasize this part.

### Questions
- What is the connection and differences between the Lag-Agnostic Structural Prior and those in the related work? Especially the "order" of causal variables, like Partial Orders [1] and Causal Orders [2]. I suggest an additional discussion.
- Please consider improving the notation about $\Theta\_{ij,s}\:=\\{\\theta\\mid \|(W\_s(\\theta))\_{ij}\| \\geq \\delta \\}$. Readers may expect $\\Theta\_{ij,s}(0)\:=\\{\\theta\\mid\|(W\_s(\\theta))\_{ij}|\\geq 0\\}$, which does not match the actual definition of $\Theta_{ij,s}(0)$.

[1] Differentiable Structure Learning with Partial Orders

[2] Causal Order: the Key to Leverage Imperfect Experts in Causal Inference

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the problem of causal structure learning from time-series data when only lag-agnostic prior knowledge is available. The authors propose a continuous optimization framework that integrates such priors into time-series structure learning. The paper first identifies the process-inequivalence issue in naive maximum-based formulations for lag-agnostic priors, which causes bias toward specific lags during optimization (Section 3.2, Proposition 1). To address this, it introduces two process-equivalent formulations, a binary-masked formulation (Eq. 10) and a logic-dual formulation (Eq. 11), that preserve the semantics of lag-agnostic priors throughout optimization (Section 3.3). The authors further analyze the non-convexity induced by lag-agnostic constraints and propose a data-driven initialization strategy (Section 3.4) to mitigate convergence to poor local optima. Finally, through comprehensive experimental validation on synthetic data, non-linear and non-stationary datasets (using LIN and RHINO backbones), and real-world DREAM4 gene regulatory networks (Sections 4.1–4.4), the paper demonstrates that the proposed framework improves causal recovery and stability compared to both data-only and lag-specific prior methods, particularly when temporal information is noisy or incomplete.

### Strengths
Originality: The notion of lag-agnostic structural priors is novel and fills a clear gap between lag-specific causal discovery (e.g., Sun et al., 2023) and coarse-grained prior-based static structure learning (e.g., Zheng et al., 2018). The formal distinction between consequence equivalence and process equivalence is particularly insightful and provides new conceptual clarity (Proposition 1–4, Section 3).

Technical quality: The theoretical analysis is rigorous. The proofs of process-equivalence (Appendix B) and the illustration of increased non-convexity via Example 1 are convincing. The data-driven initialization strategy is well-motivated and empirically validated.

Clarity: The paper is well-structured, with clear notation and well-separated sections. Figures 1–2 and Tables 1–2 (pages 8–9) are clear and they effectively support the claims.

Significance: The work provides a general, modular mechanism that can be integrated into various differentiable structure learning frameworks (DYNOTEARS, LIN, RHINO). The experiments demonstrate consistent improvement across backbones, suggesting wide applicability.

### Weaknesses
Scalability considerations: The computational complexity of the binary-masked and logic-dual penalties is not analyzed. As both require operations across all lags and node pairs, their efficiency and scaling to large graphs (e.g., d > 100) remain unclear.

Interpretability of lag selection: Although the process-equivalent formulations prevent early bias, the final lag assignments are driven primarily by data fitting. The paper could elaborate more on how reliably the method identifies the true lag rather than simply satisfying priors (discussion in Section 3.3).

Ablation clarity: While Appendix E reportedly contains ablations, the main text provides limited discussion of which component (formulation type vs. initialization) contributes most to performance gains.

### Questions
Complexity and scalability: How does the proposed penalty (especially the product-based logic-dual term) scale with increasing lag length and variable count?

Initialization sensitivity: How sensitive is the method to the choice of unconstrained pre-training in Eq. (14)? Does using different unconstrained learners (e.g., VAR vs. neural backbones) affect performance?

Interpretability of lag selection: Can the authors quantify how often the method identifies the correct lag (when available) rather than merely satisfying the lag-agnostic constraint?

Practical deployment: Are there specific guidelines for tuning λₚ when validation ground truth is unavailable?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper addresses structure learning for multivariate time series when practitioners have coarse, lag-agnostic causal priors (edge exists but unknown lag). Authors show a straightforward "max over lags" penalty is consequentially equivalent to the desired prior but process inequivalent (induces early bias to a single lag). They propose two process-equivalent penalties that act across all lags: 1) Binary-masked loss $p_\text{bin}$: activates only when all lag-specific edges are below threshold, then pushes them jointly; 2) Logic-dual/product loss $p_\text{or}$: product of ReLU terms (OR semantics), with a normalization to reduce scale sensitivity. They also argue lag-agnostic constraints increase non-convexity and propose a two-stage, data-driven initialization.
Experiments on synthetic VAR graphs (ER-k, Gaussian/Exponential noise), nonlinear/non-stationary backbones (LIN, RHINO), and DREAM4 show consistent gains in SHD/F1/AUROC and lower regression loss vs data-only baselines and vs lag-specific priors with wrong lags.

### Strengths
- Lag-agnostic prior knowledge is common; formalizing it is useful.
- Simple, plug-in losses applicable to multiple backbones (DYNOTEARS, LIN, RHINO).
- Initialization story is well-motivated.
- Broad evaluation

### Weaknesses
- Propositions establish equivalence of penalties but there's no optimization-theoretic guarantee (e.g., convergence to a correct lag under identifiability conditions).
- The non-convexity example is illustrative but small; more formal landscape analysis would strengthen claims.
- The product loss can suffer vanishing gradients when many lags are near but below $\delta$; the normalization helps but may not fully address scale with larger L.
- How robust are results to noisy/incorrect presence and incorrect absence priors (false positives/negatives in $C_p$, $C_a$)?
- Baselines focus on NOTEARS-family and a “random-lag” variant. Important time-series causal methods like DYNOTEARS with group-sparsity across lags isn't compared.
- No comparison to softmax/log-sum-exp surrogates for max (temperature-controlled) which are a natural alternative to address process-inequivalence.

### Questions
-  Any empirical comparison to LSE or Gumbel-softmax over lags?
- What happens when $C_p$ contains 20-40% spurious pairs or $C_a$ wrongly forbids true edges?
- For $p_\text{or}$, how often do you observe near-zero gradients early? Does the normalization fully fix it as L grows?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes a lag-agnostic prior constraint for incorporating prior knowledge into causal discovery algorithms for time series data. The authors highlight the drawback of outcome-equivalent constraints like maximum-based prior and theoretically justify the effectiveness of their process-equivalent prior. Empirically, the paper shows the effectiveness of their method in incorporating the provided prior information.

### Strengths
1. The paper is very well-written and easy to read.
2. The theoretical results are well-motivated and clearly stated. The examples are well-constructed to illustrate the point that the authors are trying to convey.
3. The empirical results are extensive, and include several ablation studies.
4. The authors tackle an important problem that is often overlooked. Priors coming from domain-experts are almost always lag-agnostic so it's nice to see a paper that tackles this challenge.

### Weaknesses
1. Although Proposition 1 is illustrative, it is proven under a very strong assumption, i.e. $\nabla{|(W_\tau)_{ij}|} \mathcal{L} \geq 0$ for all $\tau$. This is unrealistic, since many edges in real-world applications are, in fact, not forced to 0 by the data. It is unclear whether the principle being illustrated would still hold in such cases.

2. The main experiments in Section 4 do not include the outcome equivalent baseline "maximum-based formulation". Although the authors report some experiments in Appendix E.2, the difference between the two methods seems quite small (especially for the init data setting). This is an important baseline that the authors should consider including in Table 1, Figure 1 and Table 2.

3. A minor weakness is that practical sources and reliability of the availability of such lag-agnostic priors could be discussed more concretely, especially in the motivation.

### Questions
1. Do the authors have intuition for why data driven initialization works?
2. As noted in the limitations section, the model can go wrong due to incorrect priors. How sensitive is the model to incorrect priors?

### Soundness
3

### Presentation
4

### Contribution
3
