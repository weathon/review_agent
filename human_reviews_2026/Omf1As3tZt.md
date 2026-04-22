# Wasserstein Distributionally Robust Minimax Regret Optimization for Multimodal Machine Learning

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 2, 6, 4

## Abstract
Learning robust multimodal predictors under distributional uncertainty remains challenging, as empirical risk minimization (ERM) is brittle to modality-specific perturbations and standard distributionally robust optimization (DRO), by minimizing worst-case risk, may yield overly conservative solutions under heterogeneous noise. 
We introduce **Wasserstein Distributionally Robust Minimax Regret Optimization (WDRO-MRO)**, a framework that unifies Wasserstein DRO with minimax regret. By minimizing worst-case *regret* relative to the oracle predictor, WDRO-MRO provides a decision-centric robustness notion that directly bounds performance degradation under heterogeneous shifts. A modality-weighted Wasserstein cost further enables selective protection of vulnerable modalities. 
Theoretically, WDRO-MRO establishes a solid foundation: existence and uniqueness of minimax regret solutions under convex losses, convexity and strong duality of the formulation, and sensitivity characterizations of optimal regret with respect to ambiguity radii and modality weights. We also provide statistical guarantees including consistency, finite-sample generalization bounds, $O(N^{-1/2})$ convergence rates, and explicit sample complexity. Algorithmically, WDRO-MRO admits tractable convex reformulations (LP, SOCP, SDP, and power-cone programs) and introduces a dual-game algorithm that couples strong-dual reformulations with an exponentiated-weights adversary update, yielding an oracle-free no-regret procedure.
Empirically, on the HANCOCK multimodal healthcare dataset, WDRO-MRO maintains competitive average accuracy and improves robustness and fairness compared to ERM and standard DRO, without incurring excessive conservatism.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes WDRO-MRO, a framework combining WDRO with min-max regret to improve robustness in multimodal ML. It provides theoretically grounded formulation, tractable convex reformulations and a provable convergent dual-game solver. Experiments have shown strong robustness and fairness over ERM and DRO.

### Strengths
1.	The unified formulation is novel and the theoretical foundations are strong.

2.	It gives tractable convex reformulation, interpretability and regularization links.

3.	Experimental results show good robustness and fairness.

### Weaknesses
1.	This paper is very technical and notations are too heavy. It is hard to readers to track their theory and proofs. I suggest the authors to introduce proof sketches for each lemma or theorem to make readers easy to follow. Furthermore, a notation table is welcome to improve the readability.

2.	The experimental evaluation is not that enough. Only one dataset, HANCOCK is evaluated. Only logistic regression is considered. There are lack of multimodal models/datasets to verify the effectiveness.

3.	Although WDRO-MRO shows superiority against WDRO, the overall performance seems not satisfactory. AUCs of WDRO-MRO are less than 0.7, it is very hard for hospitals and doctors using WDRO-MRO as their machine learning paradigm to analyze cancers. Therefore, the proposed method may remain theoretically and not be practical. Is HANCOCK dataset too hard for learning?

4.	WDRO-MRO may induce further computational overhead compared with standard DRO.

### Questions
1.	How sensitivie are results to the choice of modality weights $\alpha_k$? Is there any automatic tuning?

2.	Can the dual-game solver extent to non-convex case?

3.	How is fairness measured across modalities rather than patient subgroups?

### Soundness
3

### Presentation
2

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
The paper addresses the challenge of robust multimodal machine learning under modality-specific distributional shifts, where standard ERM fails and traditional DRO can be overly conservative. It proposes Wasserstein Distributionally Robust Minimax Regret Optimization (WDRO-MRO), which minimizes worst-case regret relative to an oracle using a modality-weighted Wasserstein ambiguity set to provide decision-centric robustness. Key results include theoretical guarantees on optimization and statistics, tractable convex reformulations, a dual-game algorithm, and empirical improvements in accuracy, robustness, and fairness on the HANCOCK healthcare dataset compared to ERM and standard DRO.

### Strengths
1. The paper provides a comprehensive theoretical foundation for the WDRO-MRO framework. The analysis in Section 3 is thorough, covering basic optimization properties (existence, uniqueness, convexity, duality) , detailed computational properties (tractable reformulations) , and strong statistical guarantees (consistency, finite-sample bounds, convergence rates) .
  
2. The paper addresses a gap: robust multimodal learning under heterogeneous, modality-specific noise. The synthesis of minimax regret (to avoid DRO's conservatism ) with a modality-weighted Wasserstein cost is a novel and intuitive solution to this problem.
  
3.  A significant strength is showing that this seemingly complex min-max-min-max problem can be reformulated into a variety of standard, tractable convex programs (LP, SOCP, SDP). This makes the framework practical and usable.

### Weaknesses
1. Several symbols appear before being defined or are insufficiently specified, e.g.:
   - $F_m$ and $g$ (line 164)
   - $\lambda_{\max}$ (line 236)
   - $\Delta([N])$ (line 253)
   - $\sigma_k^2$ (line 311)
   - $d_k$ (line 320)
   - $L_\ell$ (line 323)
   - $s_i(f_t,\lambda_t)$ (line 231)

2. Notation inconsistencies: for instance, $\mathcal U$ (line 94) and $\mathcal B$ (line 298) appear to denote the same ambiguity set.

3. Empirical reporting (in the provided extract) lacks detail on metrics, baseline protocols, and statistical significance testing.

4. Novelty may partially overlap with prior regret-based DRO; the multimodal extension could read as incremental without stronger empirical breadth.

5. Proposition 2.1 states a standard strong duality result in DRO; it would be cleaner to cite a canonical reference and relegate proof details to the appendix.

6. In Proposition 3.2, $\phi(f)$ (or $\varphi(f)$) is used before being defined.

7. The hierarchy and labeling of Lemmas/Propositions/Theorems sometimes do not reflect their relative importance in the argument.

8. Consider moving Lemmas 3.1–3.4 to the appendix to streamline the main narrative.

9. The extension to nonconvex multimodal deep fusion is deferred to future work, leaving current evidence limited to (e.g.) logistic models.

10. Experiments use a single dataset (HANCOCK) and two baselines (ERM-LR, WDRO-LR); comparisons to group-DRO and modern multimodal architectures, plus ablations over $p$, $\alpha_k$, and $\rho$, are missing.

### Questions
1. How were hyperparameters tuned for ERM vs WDRO vs WDRO-MRO to ensure parity? Provide a table.
  

2. Can you compare to group-DRO and at least one modern multimodal deep baseline to assess gains beyond linear models?
3. Please expand the main-text assumptions and proof sketch for Lemma 3.6 and the regularization equivalences (Lemmas 3.10–3.12). What exact tail/geometry assumptions are needed?
4. Your theoretical tractability results rely on convexity. For the non-convex deep learning case , would the "Learner / Oracle updates" in Algorithm 1 be replaced by, for example, a single stochastic gradient step?
5. Can you provide runtime comparisons for the reformulations (e.g., LP vs. SDP) on real data?

### Soundness
2

### Presentation
2

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
The paper presents a new framework for distributionally robust optimization relying on (i) expressing the min-max objective in terms of regret and not just adversarial expectations, (ii) using a separable metric to define Wasserstein distances adapted to various modalities.

The authors analyze first the soundness of their framework: does a solution exits, is the problem convex or not, can we reformulate it? Then the authors instantiate the generic objective they propose with various distances and various losses. For each case they present a formulation of the dual problem amenable to solvers for convex optimization. They propose a generic optimization algorithm and present optimization convergence guarantees for it. They present sensitivity analyses and statistical properties. 

The authors conduct experiments using their framework with a logistic loss on a dataset of medical records with various modalities. Their approach appears to outperform a basic empirical risk minimization and a standard distributionally robust optimization method using Wasserstein distances.

### Strengths
- The results are extensive. In total, the paper consists in 22 theoretical results (lemmas/theorems/propositions).
- The authors analyzed the core properties of their framework (optimization/statistical properties) as well as interesting reformulations in primal and dual space.
- The experiment illustrates well the framework, though more details could help.
- Lemmas 3.10, 3.11 make interesting connections with simple regularized forms of ERM
- Many other results may be of particular interest but the presentation makes it hard to appreciate each of them.

### Weaknesses
- This is not a conference article. There is not enough space to introduce properly each result, (see many questions/remarks below).
- It is unclear how incremental the framework is on some of its contributions. Typically, considering the problem in terms of regret may shift perspectives on performance criterions towards more sensible ones, and it implies many changes I believe compared to usual Wasserstein based frameworks. On the other hand, using a separable distance seems to be of small novelty. That said, even if the proposed changes are incremental, the paper is quite complete about it. All "required" results are there.
- The assumptions may lack some details (see questions below).
- The experimental and implementation parts lack details. See questions below.

### Questions
1. Assumptions
a. Assumption 2.2 is quite restrictive. Most losses are not bounded. Even Lipschitz-continuity may fail to hold in many cases.
b. What do you mean by "closed convex class"? Is it a closed convex set of functions? What metric is used to define closed sets here? Or did you mean a class of closed convex functions?
c. What is the prediction range?
d. What is exactly the outer objective?
e. Why is Proposition 2.1 appearing below the assumptions? What's the point of this proposition?

3. Algorithms
a. What is the empirical dual subgradient? 
b. What do you mean by "implementable" line 398? Other methods can be implemented a priori.

4. Experiments, 
a. What is exactly the method that is used? Three methods are presented in 4.1. 
b. What distance is used at the end? Given all the possible cases of norm, were they all compared?

5. Conclusion,
The authors say line 471 that "WDRO-MRO provides a decision-centric notion of robustness that naturally connects performance and fairness within a tractable optimization framework". Can they justify this sentence? 

Remarks:
- line 65: Define somewehre the wasserstein cost for the unfamiliar reader. That way, the reader will directly understand where c intervenes.
- line 94: What the p in $W_p$? I suppose it is the norm used in the $d_k$ defining c? Maybe remove p since the distances were not defined in terms of norms in the first place.
- line 143: phi(f) has not been defined
- line 165: "standard in machine learning" give a reference to justify.
- Proposition 3.6: proposition 3.6 is quoted inside proposition 3.6
- line 430: add a reference about SMOTE oversampling

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
4

### Summary
This paper proposes a framework named Wasserstein Distributionally Robust Minimax Regret Optimization (WDRO-MRO) to address multimodal machine learning problems under distributional uncertainty. The proposed framework extends classical Empirical Risk Minimization (ERM) and standard Distributionally Robust Optimization (DRO), demonstrating significant improvements in performance, robustness, and fairness in practical multimodal scenarios.

From my perspective, WDRO-MRO is conceptually derived from [1] and [2]. The author leverages modality-weighted Wasserstein costs to define an ambiguity set within the Minimax Regret Optimization (MRO) framework [2], which better captures modality-specific heterogeneity. Compared with [1], the author generalizes the formulation to arbitrary convex loss functions and Wasserstein norms, while [1] only considered the type-2 Wasserstein distance. The theoretical contributions are substantial: under mild assumptions, the paper establishes the existence and uniqueness of minimax regret solutions, strong duality, statistical consistency, finite-sample generalization bounds, and an O(N⁻¹ᐟ²) convergence rate. Building on these results, the author presents tractable convex reformulations under different losses and Wasserstein norms, and proposes an oracle-free iterative dual-game algorithm to solve WDRO-MRO efficiently. Moreover, the paper interprets WDRO-MRO as a form of implicit regularization that enhances robustness. Altogether, the theoretical development provides a solid foundation for understanding and accepting this novel framework.

[1] Agarwal, A. & Zhang, T. (2022). Minimax Regret Optimization for Robust Machine Learning under Distribution Shift. arXiv:2202.05436
[2] Al Taha, F., Yan, S., & Bitar, E. (2023). A Distributionally Robust Approach to Regret Optimal Control using the Wasserstein Distance. arXiv:2304.06783

### Strengths
- WDRO-MRO is a general and theoretically well-grounded framework applicable to any convex loss function and Wasserstein norm.
- The application to logistic regression is clearly presented, with detailed derivations of the objective and upper bound.
- Experiments on the HANCOCK multimodal dataset show that WDRO-MRO substantially outperforms baseline methods across multiple metrics reflecting performance, robustness, and fairness.

### Weaknesses
- The conceptual framework of WDRO-MRO is not entirely new, as similar formulations have appeared in previous works [1, 2]; thus, the paper’s novelty is somewhat limited.
- The HANCOCK dataset may not fully reflect complex multimodal interactions. If I understand correctly, the preprocessing step produces a tabular feature matrix divided into several feature groups, which feels closer to assigning group-wise weights rather than fusing genuinely distinct modalities.
- The experimental comparisons are somewhat limited: the paper only compares with ERM and standard Wasserstein DRO, while many improved DRO variants (e.g., Group DRO, f-DRO, or divergence-based DRO) could provide a more convincing benchmark.
- Regarding modality weighting, WDRO-MRO requires specifying weights αₖ for each modality. In modern multimodal learning, we often fuse information via cross-attention or other adaptive mechanisms that learn inter-modal relevance automatically. It would therefore strengthen the work if the author compared WDRO-MRO with such modern multimodal fusion approaches.

[1] Agarwal, A. & Zhang, T. (2022). Minimax Regret Optimization for Robust Machine Learning under Distribution Shift. arXiv:2202.05436
[2] Al Taha, F., Yan, S., & Bitar, E. (2023). A Distributionally Robust Approach to Regret Optimal Control using the Wasserstein Distance. arXiv:2304.06783

### Questions
1. In Algorithm 1, each iteration requires computing sᵢ for all N data points over T iterations, and each sᵢ involves evaluating the Wasserstein distance (O(n³ log n)). Could this lead to computational scalability issues in high dimensions?
2. The paper does not discuss how the modality weights αₖ are chosen. Are they tuned via cross-validation or simply set uniformly?
3. In Section 4.1, the author claims that larger αₖ leads to weaker shrinkage on wₖ. Is this behavior empirically observable in the reported experiments?
4. Since many DRO frameworks focus on distribution shift, do the evaluation metrics used in the paper effectively reflect this? For instance, how well would WDRO-MRO handle domain shifts such as training on one hospital and testing on another in clinical applications?

### Soundness
3

### Presentation
3

### Contribution
2
