# FedMomentum: Preserving LoRA Training Momentum in Federated Fine-Tuning

- Decision: Reject
- Scores: 4, 8, 4

## Abstract
Federated fine-tuning of large language models (LLMs) with low-rank adaptation (LoRA) offers a communication-efficient and privacy-preserving solution for task-specific adaptation. Naive aggregation of LoRA modules introduces noise due to mathematical incorrectness when averaging the downsampling and upsampling matrices independently. However, existing noise-free aggregation strategies inevitably compromise the structural expressiveness of LoRA, limiting its ability to retain client-specific adaptations by either improperly reconstructing the low-rank structure or excluding partially trainable components. We identify this problem as loss of training momentum, where LoRA updates fail to accumulate effectively across rounds, resulting in slower convergence and suboptimal performance. To address this, we propose FedMomentum, a novel framework that enables structured and momentum-preserving LoRA aggregation via singular value decomposition (SVD). Specifically, after aggregating low-rank updates in a mathematically correct manner, FedMomentum applies SVD to extract the dominant components that capture the main update directions. These components are used to reconstruct the LoRA modules with the same rank, while residual components can be retained and later merged into the backbone to preserve semantic information and ensure robustness. Extensive experiments across multiple tasks demonstrate that FedMomentum consistently outperforms prior state-of-the-art methods in convergence speed and final accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
FedMomentum is a federated fine-tuning framework for LoRA-based models that aggregates client updates in the multiplication of LoRA-B and LoRA-A space and then applies SVD to the aggregated updates. The server divides singular components to major components, residual components, and negligible components, and reconstructs LoRA using major components. Residual components are tracked as a residual term and can be merged into the backbone. Experiments across multiple LLM tasks (e.g., reasoning and code generation) emphasize convergence speed and final accuracy.

### Strengths
1. The experiments show consistent gains across multiple datasets.

### Weaknesses
1. FlexLoRA[1], which also uses SVD-based aggregation, though mentioned in the related works, should be thoroughly discussed, compared and included as an important baseline. The main incremental contribution of FedMomentum is its insights, but the method is highly similar to the closest work FlexLoRA and has not been fully compared; in the absence of rigorous positioning and systematic ablation, the existing performance improvement is difficult to be clearly attributed. 
2. It is not clear how to determine the number of residual components, could the author formally define it? 
3. SVD is not new in aggregation for LoRA in FL. The computational and communication overhead induced by SVD should be studied comprehensively in different FL scenarios (different models, different client numbers).
4. Tables 2–5 should also report, for each method, the communication cost required to reach the corresponding accuracy.

[1] Federated Fine-tuning of Large Language Models under Heterogeneous Tasks and Client Resources. Bai et al. NeurIPS 2024

### Questions
1. Could the author evaluate the impact of extreme low rank (rank-1 and rank-2)?
2. Could the author scale up the client number to 30 or more to test the effectiveness of the proposed aggregation?
3. How is the non-IID setting constructed for datasets without labels, such as GSM8k or HumanEval?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces FedMomentum, an SVD-based aggregation scheme for federated LoRA fine-tuning that (1) aggregates client updates in the correct BA form, (2) performs (randomized) SVD on the summed update to recover top-r principal components, (3) balances singular values across B and A to avoid gradient anisotropy, and (4) carries residual components that are later merged into the backbone to preserve semantics, thereby maintaining training momentum across rounds. Experiments on math, commonsense, and code benchmarks show faster convergence and higher accuracy than baselines.

### Strengths
1. Consistent improvements on math, commonsense, and code benchmarks.
2. Splitting Σ as Σ1/2 across B and A is a low-cost fix for singular-value skew.

### Weaknesses
1. All results are on LLaMA2-7B; behavior on newer or larger models is unknown.
2. Experiments use 10 clients with Dirichlet β=0.5. It’s unclear how momentum preservation holds under more extreme non-IID, partial participation, or straggler scenarios.

### Questions
1. How does FedMomentum perform on a recent larger model?
2. What residual-energy threshold and merge cadence are recommended across tasks?
3. Could residuals/SVD factors leak client characteristics?

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
4

### Summary
The paper proposes FedMomentum, a federated LoRA fine-tuning framework that aggregates delta weights (BA) across clients and then applies (randomized) SVD to (i) reconstruct LoRA modules from the top-r components with a balanced √Σ split between A and B to stabilize gradients, and (ii) carry forward residual components that are later merged into the backbone. This aims to avoid aggregation noise and preserve “training momentum” that is otherwise lost by reinitialization or partial freezing strategies. Experiments on math reasoning, commonsense reasoning, and code generation with LLaMA2-7B show faster convergence and higher accuracy than FedIT, FLoRA, FFA-LoRA, and FRLoRA; ablations indicate both the balanced split and residual path contribute to gains.

### Strengths
Clear problem framing around momentum loss in federated LoRA and a principled SVD-based fix.

Consistent improvements and faster convergence over strong baselines, with thorough ablations.

Practicality considered via randomized SVD and reporting of runtime overhead.

### Weaknesses
“Momentum” story isn’t operationalized:

The paper motivates momentum preservation with spectra/visuals but never measures optimization continuity directly (e.g., gradient-direction alignment across rounds, cosine to prior updates, curvature drift). Without such probes, the claimed mechanism remains a narrative rather than an evidenced cause.

Residuals add hidden communication/state costs:

FedMomentum keeps a residual subspace (until ~99% energy is retained) and ships it back for backbone merges. Early rounds can have sizable residual rank; yet the paper reports aggregation time, not the bytes/round for residuals vs. plain LoRA A/B. This makes “communication-efficient” hard to verify under realistic uplink constraints.

Fragile hyperparameters, little sensitivity analysis:

Key knobs (LoRA rank r, the residual energy threshold, randomized-SVD oversampling/power iterations, and the √Σ split) are fixed or lightly ablated. There’s no sweep showing accuracy vs. residual threshold, nor stability across seeds/clients—leaving robustness and reproducibility unclear.

### Questions
Please see weaknesses

### Soundness
2

### Presentation
3

### Contribution
2
