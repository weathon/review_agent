=== CALIBRATION EXAMPLE 15 ===

# Final Consolidated Review
## Summary
This paper proposes **FGL\_AC**, a federated graph classification framework that combines two ideas: (i) client-side spectral clustering as a preprocessing step, and (ii) server-side attention-weighted aggregation of client updates instead of plain FedAvg/FedProx. Experiments on MUTAG, ENZYMES, and PROTEINS under several partition settings report modest gains over FedAvg/FedProx variants.

## Strengths
- **The paper tests heterogeneous partition regimes rather than a single toy split.** In Table 2 and the accompanying discussion, the authors evaluate balanced/unbalanced and overlap/no-overlap settings, which is more informative than evaluating only one federation scenario.
- **The method combines two levers at different stages of the pipeline.** The design is not just “replace FedAvg with attention”: it attempts to improve both local data handling (via spectral clustering preprocessing) and server aggregation (via adaptive weighting), and the ablation figures are at least directionally consistent with both components helping.
- **There is some evidence of empirical improvement over the chosen baselines.** Table 2 shows FGL\_AC often achieving the best accuracy and sometimes the best F1 among the listed GCN/SAGE + FedAvg/FedProx baselines, though the gains are uneven and sometimes small.
- **The paper raises an intuitively meaningful hypothesis for federated graph learning:** not all client updates should contribute equally, especially under heterogeneous graph distributions. That intuition is plausible and worth exploring.

## Weaknesses

### Major:
- **The clustering component’s claimed efficiency/communication benefit is not supported by the method as described.**  
  This is a central issue because the abstract, introduction, and methodology repeatedly claim that spectral clustering “reduces the overall model training burden,” “improves communication performance,” and “reduces communication overhead.” However, Section 3.2 only describes clustering graphs and then says: *“Then the clustering results are mapped back to the space of the original solution, which is used as the input of the graph classification task.”* There is no defined mechanism showing reduced transmitted message size, fewer communication rounds, fewer local examples, graph coarsening for transmission, or any measured reduction in bandwidth/runtime/FLOPs. As written, this is a preprocessing step that may alter the input representation, but the paper does not substantiate the stronger systems-style claims attached to it.

- **The core attention-based aggregation mechanism is underspecified at the level of the main contribution.**  
  Section 3.3 is the heart of the paper, yet important pieces are missing or ambiguous. Eq. (8) defines a GAT-style attention score over client feature vectors \(c_i, c_j\), but the paper never clearly defines what these client-level features are, how model parameters \(Z_i\) are represented as \(c_i\), or how the attention parameters \(W\) and \(a\) are learned in the federated optimization loop. The text says weights are assigned *“according to the contribution to the server aggregation”* and elsewhere *“according to the different training effects of the clients,”* but no concrete mapping from training effect to attention input is provided. Eq. (9) then jumps to a weighted sum of client parameter sets. This leaves the paper’s main novelty insufficiently specified to evaluate technically.

- **The connection between spectral clustering and the downstream graph classifier is unclear.**  
  Section 3.2 does not explain how the clustering output is actually consumed by the GNN/classifier. The paper says each subgraph is treated as a point, clustered, and then the result is “mapped back” and used as input, but does not specify whether cluster assignments become graph features, whether representative graphs are selected, whether graphs are relabeled/reweighted, or whether the model trains on clustered prototypes. This is not a minor implementation omission: it is necessary to understand what the preprocessing actually changes and why it should help classification.

- **The experimental evidence is too limited for the breadth of the claims.**  
  The evaluation uses only three datasets, appears to use only three clients throughout, and compares primarily to GCN/SAGE with FedAvg/FedProx. For a paper claiming a new federated graph learning framework with better training effect and reduced burden, this is not enough to establish robustness or generality. The paper also does not report variance across runs, statistical uncertainty, runtime, communication volume, or convergence-speed comparisons, despite making claims about burden/efficiency. This matters especially because some reported gains are small and some metrics are tied or inconsistent.

- **Some of the paper’s claims overstate what the reported results show.**  
  The abstract claims improvements of “2.63% - 4.03% compared to other federated graph learning frameworks,” but Table 2 is more mixed: not all gains are in that range, and FGL\_AC is not uniformly best on every metric. For example, on PROTEINS under one unbalanced-overlap F1 row, FGL\_AC is worse than a baseline. Similarly, Section 4.3 concludes the method *“also has certain advantages for centralized model training,”* but the experiment mainly shows that clients participating in federation benefit from shared information compared with an isolated client; it does not establish a meaningful centralized-learning contribution.

### Minor
- **The novelty is incremental rather than strong by ICLR standards.**  
  Spectral clustering as preprocessing and GAT-style attention for weighting are both established ideas; the contribution here is mainly their combination in a federated graph classification pipeline. That can still be publishable with strong execution, but the paper currently does not provide enough technical depth or evidence to elevate the combination into a compelling methodological advance.

- **Ablation evidence is directionally useful but too narrow to isolate the proposed effects convincingly.**  
  The ablations are only shown on MUTAG, only with accuracy curves, and do not investigate key design choices such as number of clusters, KNN parameter, scale parameter \(\psi\), or the structure of the attention mechanism. Since the paper’s contribution is exactly “clustering + attention,” stronger component analysis is needed.

- **The setup with only three clients is a poor stress test for a client-attention method.**  
  A mechanism meant to adaptively weight heterogeneous clients is more convincing when demonstrated beyond a trivial small-client regime. With only three clients, the heterogeneity and scaling claims remain weakly supported.

- **Notation and algorithm specification are confusing in places.**  
  For example, Table 1 uses \(L\) both for local iterations and Laplacian matrix, and Algorithm 1 omits crucial detail for the aggregation step. This is not just cosmetic: it contributes to the difficulty of understanding and reproducing the method.

- **Figure 2 suggests differential privacy, but the method does not describe or evaluate any DP mechanism.**  
  Since the diagram explicitly labels communication with “Differential Privacy,” the absence of any corresponding methodological or experimental treatment is misleading and should be corrected.

### Trivial
- **The claimed degeneration to FedAvg is asserted rather than demonstrated.**  
  The idea is plausible if attention weights become uniform, but the paper does not formally derive the equivalence under the actual update rule and weighting scheme.

## Nice-to-Haves
- Add sensitivity analyses for clustering hyperparameters (number of clusters, KNN \(k\), scale \(\psi\)) and show how clustering affects downstream accuracy.
- Visualize learned client attention weights over rounds to demonstrate whether the mechanism learns meaningful non-uniform weighting or effectively collapses to near-uniform averaging.
- Report per-client performance trajectories to support the claim that stronger clients help weaker ones through attention-weighted aggregation.
- Strengthen Section 4.3 by comparing FGL\_AC against other FL methods under the same setup, rather than only contrasting federated participation with isolated local training.

## Removed Points
These points are flagged to be removed, treat them with caution:

- **Requests to cite or compare with specific external methods as a hard criticism.**  
  Several reviews asked for comparisons to named methods or additional related work. It is reasonable to say the baseline set is limited, and that stronger/more directly relevant baselines would improve the paper. However, I am not turning “missing related work” into a standalone criticism, since that requires external verification beyond the submission.

- **Pure reproducibility nitpicks about omitted appendix details or complete implementation specifics.**  
  The real issue is not missing minor hyperparameters; it is that the main method itself is underspecified in core places (especially Section 3.3 and the clustering-to-classifier pipeline). Those substantive concerns are kept above.

- **Any concern questioning the existence/availability of cited tools, models, datasets, or references.**  
  Such criticisms are not valid grounds here.

## Novel Insights
The most important synthesis across the reviews is that the paper’s main weakness is not merely “needs more experiments,” but a mismatch between **what is claimed** and **what is actually specified and validated**. The paper would be substantially stronger if reframed more modestly as an empirical combination of clustering-based preprocessing and adaptive aggregation for graph classification, rather than as a method that also reduces communication burden. Right now, the clustering piece reads like an input transformation with unclear downstream use, while the attention piece reads like a promising intuition without a complete federated learning formulation. That gap in specification, more than the incremental novelty alone, is what prevents the current submission from reading as technically solid at ICLR level.

## Suggestions
- **Precisely define the attention aggregation pipeline.** State what \(c_i\) is, how it is computed from each client, how \(W\) and \(a\) are optimized, and whether the method produces one global model or client-specific personalized models.
- **Explicitly describe how spectral clustering alters the training data or features.** A step-by-step mapping from raw graphs to clustered representation to GNN input is necessary.
- **Either remove the communication-efficiency claims or support them with measurements.** Report communication bytes, rounds to convergence, runtime, and/or local computation cost.
- **Expand experiments beyond three clients** and show that the adaptive weighting remains useful under more realistic client counts and heterogeneity.
- **Report results across multiple random seeds** with mean and variance; otherwise many of the small gains in Table 2 are hard to interpret.
- **Broaden and strengthen ablations** to cover clustering hyperparameters and the attention design itself.
- **Revise Figure 2 or add the missing methodology** if differential privacy is meant to be part of the contribution.
- **Tighten the claims in the abstract and conclusion** to reflect the actual evidence, especially where improvements are mixed rather than uniform.

# Actual Human Scores
Individual reviewer scores: [3.0, 1.0, 3.0, 3.0]
Average score: 2.5
Binary outcome: Reject
