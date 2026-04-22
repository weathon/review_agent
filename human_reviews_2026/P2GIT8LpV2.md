# Minimax Sample Complexity of Graph Neural Networks: Lower Bounds and Structural Effects

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Graph Neural Networks (GNNs) achieve strong empirical performance across domains, yet their fundamental statistical behavior remains poorly understood. This paper develops a minimax analysis of ReLU message-passing GNNs with explicit architectural assumptions, in  both inductive (graph-level) and transductive (node-level) settings. For arbitrary graphs without structural constraints, we show that the worst-case generalization error scales as $\sqrt{\log d / n}$ with sample size $n$ and input dimension $d$, matching the $1/\sqrt{n}$ behavior of feed-forward networks. Under a spectral--homophily condition combining strong label homophily and bounded spectral expansion, we prove a stronger minimax lower bound of $d/\log n$ for transductive node prediction. We complement these results with a systematic empirical study on three large-scale benchmarks (ogbn\_arxiv, ogbn\_products\_50k, Reddit\_50k) and two controlled synthetic datasets representing the worst-case and structured regimes of our theory. All benchmark graphs we study fall in the slow-mixing, bottlenecked regime captured by our spectral-homophily condition, and ratio-based scaling tests show error decay consistent with the $d/\log n$ rate in real and structured settings, while the worst-case synthetic dataset follows the $\sqrt{\log d / n}$ curve. Together, these results indicate that practical GNN tasks often operate in the spectral-homophily regime, where our lower bound $d/\log n$ is tight and effective sample complexity is driven by graph topology rather than universal $1/\sqrt{n}$ behavior.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper studies sample complexity of ReLU-based GNNs. The authors prove (i) a worst-case minimax lower bound for graph-level prediction and (ii) a structure-aware node-level lower bound. Experiments on Cora, Reddit, QM9, Facebook with GCN/GAT/GraphSAGE study test-error scaling vs. the number of nodes. The work positions itself as delivering the first sharp lower bounds that explicitly reflect graph structure, complementing expressivity-driven upper bounds.

### Strengths
S1: Strong mathematical rigor and novelty in lower bounds. Theorem 1 establishes a worst-case minimax lower bound for ReLU MPNNs with explicit architectural assumptions. Theorem 2 links spectral–homophily to a structure-dependent (node-level, transductive) lower bound.

S2: The experimental section provides validation over theorem 1 and 2, as multiple datasets and architectures fit to the proposed lower-bounds.

S3: The paper carefully contrasts its lower-bound focus with recent expressivity/upper-bound works, clarifying complementary scope.

### Weaknesses
W1: Experimental granularity is thin. Figures sample only a handful of n values (6 per dataset; and n lower than 1,000 for Cora/Facebook due to dataset limits), which makes distinguishing the different asymptotics fragile. More densely sampled n (especially mid-range) and more seeds would increase robustness. 

W2: Scope of datasets and tasks. Only four benchmarks; the link-prediction result is limited to Facebook. Adding synthetic graphs with controlled spectral gaps/homophily and additional real datasets (e.g., ogbn-* datasets) would better stress-test the structural claim. 

W3: Verifying the structural condition. Empirical sections do not demonstrate the inequality which Theorem 2 requires numerically. Providing this would more tightly link the assumption to observed scaling.

Im inclined to raise my score once the experimental concerns above are addressed.

### Questions
Q1: Can you resample Figures 1–4 with more n points (esp. between 500 and 10k where curves diverge) and more seeds to provide confidence intervals and slope diagnostics on log–log plots? This directly addresses the robustness of the scaling claims. 

Q2: For each dataset and n, please compute $\lambda_2$ of the induced labeled-node subgraph and report whether the condition required by Theorem 2 holds for a reasonable K, or provide an empirical estimate of K. This would concretely validate Theorem 2’s premise.

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
This paper derives lower bounds to the sample complexity/generalization of GNNs using minmax theory. The paper derives such bounds both for inductive (graph level) and transductive (node level) settings. This is achieved using Fano’s inequality and by constructing packing numbers in the two settings.

### Strengths
This paper develops nontrivial analysis to prove an important property of GNNs. Lower bounds to the sample complexity of GNNs are important and novel, and missing from the current literature on GNN generalization. Especially, analyzing node-level tasks is very novel, as most other papers analyze graph-level tasks.

### Weaknesses
This is potentially a strong theoretical paper. My main comment is about making the paper more accessible to the GNN community. Deep learning is an interdisciplinary subject, and hence good papers in deep learning should give short “tutorial-like” introductions, possibly in the appendix, to any nontrivial theoretical tool used in the paper. 

Since this paper is both for experts on GNNs and experts in minmax bounds, and most often there is no intersection between the two communities, I recommend writing an introduction to minmax theory in the appendix and referring to it in the main paper. Specifically, it is not direct to see how the setting of regression is a special case of the general minmax risk, so I would explicitly write the general formulation of the minmax, and then write explicitly how regression is a special case. Moreover, a reference to minmax theory is missing, e.g., Chapter 15 in [A]. 

You should also explicitly recite Fano’s inequality, and all other tools required in the proofs, like packing number. As it stands, many nontrivial tools are never explicitly defined in the paper, which makes it impossible for most potential readers to follow. It is also important to define all norms in the paper, as different people have different normalization standards for the same norm.

For motivation, I would also explain how the spectral gap/spectral–homophily is related to “bottleneckedness.”

Once the authors add such a section to the revised pdf I will raise my score.

The other direction  (Explaining GNNs to the statistical learning community) is already covered in the paper - GNNs are explicitly defined.


My second comment is about choosing a less generic title. Since there are many papers about GNN generalization bounds, I recommend (but don’t demand) writing a title that directly expresses that you derive a minmax analysis. This will make it easier for researchers in the future to search for the paper. Moreover, I would emphasize in the abstract and introduction that you also analyze transductive learning, since most other papers focus on inductive graph-level settings.



[A] Martin.J. Wainwright. High-dimensional statistics : A non-asymptotic viewpoint. Cambridge Uni. Press, 2019.

### Questions
See Weaknesses

### Soundness
4

### Presentation
2

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
This paper presents a theoretical foundation for the sample complexity of ReLU-based graph neural networks by using minimax analysis for lower bounds of sample complexity. The paper finds that while GNNs can in principle match the $1/\sqrt n$ scaling of feed-forward networks, realistic structural assumptions for graphs force risk to decay much slower. The paper also includes proof-of-concept experiments.

### Strengths
- The paper provides the first sharp sample complexity lower bounds for GNNs under realistic structures.
- Brief implementation code is provided.
- The problem of generalization error and sample complexity is significant.

### Weaknesses
- The proof of Theorem 2 can be put in the appendix with only a sketch of proof provided in the main text. The extra remaining space can include more motivations and/or insights.
- Sec. 3 mentions that three prediction regimes will be studied, but no theoretical results for link prediction have been provided (just one task experiment).
- Sec. E and Sec. F seem to be the same but with an extra section title in the appendix.
- Raw experiment results are not shown but only an aggregated table.

### Questions
Can you add some discussions on the link prediction task?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies the sample complexity of ReLU-based GNNs, deriving minimax lower bounds on generalization error ( $\sqrt{\frac{\log d }{n}}$ for arbitrary graphs and $\frac{\log d }{n}$ under spectral-homophily). Experiments on standard datasets with GCN, GAT, and GraphSAGE are presented to support the scaling claims.

### Strengths
(1) Theoretical novelty is high, providing the minimax analyses for GNNs. The bounds are clearly stated, and connecting theory to practice is an important goal.  

(2) The results establish a crucial baseline for the statistical efficiency of GNNs and quantify the theoretical value of graph structure.

(3) The bounds are stated clearly and the paper distinguishes between the general and structured graph scenarios.

### Weaknesses
(1)  Vacuous spectral-homophily assumption: The condition $\lambda_2(L) \le \kappa / \log n$ can be trivial for small $n$, making the sharper lower bound vacuous. $\kappa$ is absorbed as a constant in the bound, raising questions about whether the bound meaningfully captures the role of graph structure.

(2) Experimental verification of bounds: Curve fitting to candidate functional forms ($1/\sqrt{n}, 1/n, 1/\log n, 1/n^\gamma$) **does not rigorously validate** the minimax lower bound. A more appropriate approach would compare the empirical test error directly to the derived bound, including constants, or estimate an empirical constant $C^\star$ to show tightness.

(3) Limited sample size regime: Experiments are restricted to $100 < n < 1000$, a narrow range insufficient to assess asymptotic scaling. Asymptotic bounds should ideally be tested over broader regimes, including **very small and very large** $n$, and with graphs approaching the worst-case constructions used in the theory. I recommend more experimental verification on synthetic datasets and OGB benchmarks.

(4) Tightness claims:  The current experimental evidence does not demonstrate the tightness of the bounds. Without direct comparison of the experimental test error with the theoretical minimax bound, it is unclear whether the bounds are meaningful in practice.

### Questions
(1) How do you justify the omission of  $\kappa$ from the sharper lower bound in Theorem 2, given its role in the spectral assumption?

(2) Have you considered comparing empirical errors directly to the theoretical bound (including constants) instead of curve fitting?

(3) Have you conducted controlled experiments varying $n$ and $d$ on synthetic graphs to confirm the scaling laws? What is the rationale for excluding large-scale benchmarks with samples $n>10,000$ or in worst-case constructions used in the minimax proof?

### Soundness
2

### Presentation
3

### Contribution
3
