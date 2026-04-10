## Summary
This paper introduces DyCO-GNN, an unsupervised learning framework for dynamic combinatorial optimization (DCO) that requires no training data. The method adapts the "shrink-and-perturb" technique to warm-start a GNN-based optimizer across temporal graph snapshots, aiming to accelerate convergence while preserving solution quality. Experiments on dynamic MaxCut, MIS, and TSP show consistent improvements over static and naively warm-started PI-GNN under varying time budgets.

## Strengths
- **Novel problem setting**: The work is the first to propose a learning-based, training-data-free approach for dynamic combinatorial optimization, addressing a clear gap between instance-specific learning and real-world dynamic problems.
- **Simple and effective method**: The adaptation of shrink-and-perturb (SP) to mitigate the local-optima trapping of naive warm-starting is straightforward yet yields robust gains across three CO problems and multiple real-world/synthetic dynamic graphs, often achieving better solutions than converged static PI-GNN in a fraction of the runtime.
- **Comprehensive empirical evaluation**: The paper tests DyCO-GNN on MaxCut, MIS, and TSP with different GNN architectures, time budgets, and sensitivity analyses (degree of change, SP parameters), providing solid evidence of its effectiveness within the evaluated scope.

## Weaknesses
### Major:
- **Limited core algorithmic novelty**: The central algorithmic idea—applying shrink-and-perturb (SP) to warm-start a neural optimizer—is directly borrowed from Ash & Adams (2020), a supervised learning technique. While the application to unsupervised dynamic CO is new, the methodological advance is incremental. For a top-tier conference, this adaptation alone may not suffice without deeper theoretical or mechanistic innovation.
- **Insufficient explanation for design choices and performance variation**: The paper evaluates three ways to apply SP (embedding layer, GNN layers, full network) with no single variant dominating across tasks/datasets (Tables 1–3). This inconsistency is not analyzed or explained, leaving users without principled guidance on which configuration to choose for a new problem. The paper also lacks a mechanistic analysis of *why* SP helps in this specific optimization context (e.g., how it affects the QUBO loss landscape or gradient dynamics).
- **Narrow exploration of dynamic scenarios**: Experiments are limited to edge additions/deletions (MaxCut/MIS) and a single moving node (TSP). More complex and realistic dynamics—node additions/deletions, simultaneous structural and constraint changes, or adversarial perturbations—are not tested, undermining claims of general applicability to DCO.
- **Weak theoretical connection to the main method**: Theorem 1 analyzes perturbation in the Goemans–Williamson (GW) algorithm for MaxCut, which uses SDP relaxation and randomized rounding. This provides only analogical support for DyCO-GNN, which performs gradient-based optimization of a relaxed QUBO objective via GNNs. The theorem does not offer direct insight into the GNN-based optimization process, leaving the method’s success as an empirical observation.

### Minor:
- **Limited baseline comparisons beyond the PI-GNN family**: While the paper focuses on instance-specific methods and includes some non-neural baselines in Appendix D.3, it does not compare against established dynamic CO algorithms or reoptimization heuristics from the optimization literature. This makes it difficult to assess the practical competitiveness of DyCO-GNN in the broader DCO landscape.
- **Hyperparameter choices presented as universal without full justification**: The SP parameters (λ_shrink=0.4, λ_perturb=0.1) are fixed across all experiments with a claim of "no further tuning." Although sensitivity analysis is provided in the appendix, the main text does not discuss the robustness of these choices or how they might need adjustment for different problems or dynamic regimes.

### Trivial:
- **Occasionally confusing metric notation**: In tables, the notation "Values closer to 1 are better (↑/↓)" is slightly ambiguous for TSP (where lower ApR is better), though the context clarifies the meaning.

## Nice-to-Haves
- Developing an adaptive SP mechanism that adjusts λ_shrink/λ_perturb or the layers to perturb based on snapshot similarity or gradient signals.
- Extending evaluation to dedicated dynamic CO benchmarks or synthetic dynamic graphs with controlled change properties (e.g., node arrivals/departures, large rewiring).
- Providing a deeper ablation study comparing SP to alternative stabilization techniques (e.g., learning rate resets, gradient clipping) to isolate the contribution of the specific SP formulation.

## Removed Points
*These points are flagged to be removed, treat them with caution*

**Strengths removed:**
- "The paper is well-written" – generic strength.
- "The topic is important" – generic strength.
- "The experiments are extensive" – already covered by specific empirical evaluation strength.

**Weaknesses removed:**
- "The ground truth acquisition using Gurobi with a 60-second time limit is unreliable" – The paper explicitly states Gurobi is used with a time limit, and this is standard practice for obtaining reference solutions; doubting the existence or availability of Gurobi violates the hard rule.
- "Missing comparison to all existing dynamic CO algorithms" – This is scope creep; the paper focuses on learning-based, instance-specific methods and includes relevant non-neural baselines in the appendix. Demanding exhaustive comparison to every traditional algorithm is unreasonable.
- "Hyperparameters like σ for the noise ε^t are not specified" – This is a reproducibility nitpick about implementation details; the hard rule removes such trivial hyperparameter complaints.
- "Formatting issues in tables make data hard to read" – Pure formatting/style nitpick, removed per hard rule.
- "The method does not scale to millions of nodes" – The paper evaluates graphs up to thousands of nodes, which is reasonable for a research submission; requesting arbitrarily larger scales is a generic one-size-fits-all weakness.

## Suggestions
- In the revision, add a concise discussion in the main text explaining the performance variation across SP application strategies (emb/GNN/full) and provide practical guidance on selecting a configuration based on problem characteristics.
- Expand the experimental section to include at least one more complex dynamic scenario (e.g., node additions/deletions) to better demonstrate generalizability.
- Strengthen the connection between Theorem 1 and the GNN method by adding a brief discussion on how the intuition from GW perturbations might translate to gradient-based optimization with SP, or explicitly note the theorem's role as analogical support.