# Learning to Price Bundles: A GCN Approach for Mixed Bundling

- Avg Score: 3.33
- Decision: Reject
- Scores: 4, 4, 2

## Abstract
Bundle pricing refers to designing several product combinations (i.e., bundles) and determining their prices in order to maximize the expected profit. It is a classic problem in revenue management and arises in many industries, such as e-commerce, tourism, and video games. However, the problem is typically intractable due to the exponential number of candidate bundles. In this paper, we explore the usage of graph convolutional networks (GCNs) in solving the bundle pricing problem. Specifically, we first develop a graph representation of the mixed bundling model (where every possible bundle is assigned with a specific price) and then train a GCN to learn the latent patterns of optimal bundles. Based on the trained GCN, we propose two inference strategies to derive high-quality feasible solutions. A local-search technique is further proposed to improve the solution quality. Numerical experiments validate the effectiveness and efficiency of our proposed GCN-based framework. Using a GCN trained on instances with 5 products, our methods consistently achieve near-optimal solutions (better than 97\%) with only a fraction of computational time for problems of small to medium size. It also achieves superior solutions for larger size of problems compared with other heuristic methods such as bundle size pricing (BSP). The method can also provide high quality solutions for instances with more than 30 products even for the challenging cases where product utilities are non-additive.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper explores the usage of graph convolutional networks (GCNs) in solving the bundle pricing problem. They first develop a graph representation of the mixed bundling model, then train a GCN to learn the latent patterns of optimal bundles, and propose two inference strategies to derive high-quality feasible solutions. Numerical experiments validate the effectiveness and efficiency of the proposed GCN-based framework.

### Strengths
- **Well-motivated, practical problem.** Clearly ties mixed bundling to real applications (e-commerce, subscriptions, tourism) and to revenue-management/econ literature.
- **Clear hybrid idea.** Learning-guided **GCN → pruning → MILP**, with an LP-guided local search; balances tractability and exact optimization.
- **Readable pipeline.** Figures convey end-to-end flow; the overall method is easy to reason about and potentially easy to deploy.

### Weaknesses
- **Missing ML baselines.** No head-to-head with neural recommenders or neural CO (e.g., BGCN-style methods); even a proxy comparison or discussion would help.
- **OOD/generalization unclear.** Claims about training on small (n) and testing on larger (n) lack systematic sweeps across utility families and distribution shifts.
- **Interpretability.** No qualitative bundle visualizations or case studies to show why bundles are pruned/kept versus classical methods.

### Questions
1. **Generalization & OOD.** How does performance change when test utilities differ sharply from training (e.g., (\log(1+x)), strong complementarities/substitutes, correlated item utilities)? Please report a sweep.
2. **Baselines & positioning.** Can you add a quantitative comparison (or careful proxy) against recent GNN/Transformer bundle recommenders or neural CO solvers? If not, discuss trade-offs (objective mismatch, scalability, interpretability).
3. **Cutoff & hyperparameters.** How were the 0.5 cutoff, GCN depth/hidden size/dropout chosen? Provide sensitivity curves and validation protocol; any adaptive/learned cutoff variants?
4. **Monotonicity & constraints.** In practice, are final prices monotone (assumption used in the subadditivity equivalence)? If not, do you enforce monotonicity, or how often is it violated?
5. **Interpretability & case studies.** Can you include qualitative examples (bundle heatmaps, per-segment top-(P_{kj}) items, before/after pruning sets) to illuminate what the GCN learns?
6. **LS–MILP relation.** Do you observe cases where LP-guided improvements fail to translate into MILP gains? If so, how do you handle acceptance criteria or numerical stability?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper reframes the classic bundle-pricing problem as a graph-learning task and introduces a GCN-based pipeline that first prunes the exponential bundle space and then solves a reduced MILP.  Extensive experiments on synthetic data show >97 % of the small-instance optimum in seconds and clear gains over bundle-size heuristics for up to 100 products.

### Strengths
- Original combination of GCNs with the Hanson–Martin MILP.

- Scalable to 100+ products where exact methods fail. 

- Solid ablation of pruning variants and local-search refinement.

### Weaknesses
- Theoretical insight is limited—no approximation guarantees or generalization bounds.

 - All instances are synthetic with concave utility; realism of this model is asserted but not validated on real catalog or transaction data.

- Scalability claims rely on a fixed segment size (m=10) and ignore the potentially dominant MILP time when candidate bundles grow.

- Missing baselines from recent neural-OR literature (e.g., neural diving or RL-based pricing).

### Questions
- The authors hold the assumption that each bundle has distinct products, so what about the scenarios where repeated products are collected in one bundle?

- The adopting GCN for bundle modeling is not very new (see all kinds of papers on bundle recommendation, bundle generation, and so on). What is the special difficulty in the current settings? And what contribution do the authors make to the current adoption?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a GCN (graph convolutional network)-based approach to price bundles of products to customers. The network is firstly determined to output a small selection of bundles. Then, the selection of bundles is given to an MILP solver to determine the optimal prices on these bundles.

Baselines include an optimal but costly MILP approach and a fast but suboptimal Bundle-Size-Pricing (BSP) approach. Experiments show that the GCN-based approach preserves $>97.5\%$ optimal revenue and lower running time in small settings. In large settings, the GCN-based approach still outperforms BSP in terms of revenue and running time.

### Strengths
Learning to price bundles is an interesting idea. The neural network architecture is novel (as far as I can tell). The presentation is easy to understand. The experiments show the potential of strategic bundle-selection towards MILP-solving-based approaches to some extent.

### Weaknesses
The paper has some major weaknesses, any of which is a great challenge for the acceptance of this paper.

1) The paper does not explain how to train the GCN. In my understanding, the GCN is parameterized so training is a necessary step before using in experiments. In addition, the output of GCN is followed by an MILP solver, the latter of which is (probably) non-differentiable. Differentiating the target w.r.t. GCN parameters seems to be impossible.
2) The paper also does not clearly explain the experimental settings. For example, it is expected that the determination of all model parameters, such as $c _j^u, c _k^s, u _{kj}$, should be explained before experimental results.
3) The choice of baseline is questionable. The authors only select two baselines, only one of them (BSP) works in large settings. The authors have also mentioned that BSP is possibly suboptimal as it does not consider the heterogeneity of products. It seems likely that the authors should have considered more appropriate baselines as mentioned in Section 2, or at least given justifications on why other approaches are not appropriate baselines.

See below questions for my other concerns.

### Questions
1) The paper does not mention learning-based approaches on pricing bundles. Have the authors made literature review on this topic? Or is it the case that this topic remains blank in literature? (Although I believe the latter is quite unlikely.)
2) The GCN structure is designed with feature engineering without mathematical justifications. A simple MLP (with a post-processing sigmoid) may also work. Have the authors considered MLPs before GCN? Could the authors provide explanations on why GCN might be more appropriate than MLP? (Note that Ockham's Razor principle suggests that GCN is an overkill if it can be replaced with a simpler MLP without loss.)
3) The authors have emphasized that the $R _{kb}$ is non-additive in line 80-81. However, since there are only finite possible outcomes, why the non-additive utilities provide more difficulties than additive utilities? It seems to me that the additivity or non-additivity does not matter so much as the problem itself is finite.

### Soundness
2

### Presentation
1

### Contribution
2
