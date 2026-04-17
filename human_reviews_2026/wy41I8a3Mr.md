# Federated Sketching LoRA: A Flexible Framework for Heterogeneous Collaborative Fine-Tuning of LLMs

- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Fine-tuning large language models (LLMs) on resource-constrained clients remains a challenging problem. Recent works have fused low-rank adaptation (LoRA) techniques with federated fine-tuning to mitigate challenges associated with client model sizes and data scarcity. Still, the heterogeneity of resources remains a critical bottleneck: while higher-rank modules generally enhance performance, varying client capabilities constrain LoRA's feasible rank range. Existing approaches attempting to resolve this issue either lack analytical justification or impose additional computational overhead, leaving a wide gap for efficient and theoretically-grounded solutions. To address these challenges, we propose federated sketching LoRA (FSLoRA), which leverages a sketching mechanism to enable clients to selectively update submatrices of global LoRA modules maintained by the server. By adjusting the sketching ratios, which determine the ranks of the submatrices on the clients, FSLoRA flexibly adapts to client-specific communication and computational constraints. We provide a rigorous convergence analysis of FSLoRA that characterizes how the sketching ratios affect the convergence rate. Through comprehensive experiments on multiple datasets and LLM models, we demonstrate FSLoRA's performance improvements compared to various baselines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Federated Sketching LoRA (FSLoRA), a heterogeneous federated fine-tuning framework for LLMs that keeps a large global LoRA rank on the server but lets each client update only a submatrix of the LoRA adapters via a random-k diagonal sketching matrix. By choosing client-specific sketching ratios ki​/r, FSLoRA reduces per-client compute/communication while retaining the expressivity of a higher global rank. The authors give a convergence analysis that quantifies how sketching ratios rescale smoothness constants and thus the convergence rate, and they report consistent accuracy gains vs. HeteroLoRA, FlexLoRA, and FLoRA and commonsense reasoning (LLaMA-3.2-3B), with comparable or lower GPU hours.

### Strengths
1. Random-k sketching gives unbiased submatrix updates with a clean convergence analysis that recovers FedAvg as ki→r.
2. Reduces both computation and communication without SVD or adapter merging.

### Weaknesses
1. FSLoRA adds a server-side duty each round to sample sketches, reconstruct sparse updates, and aggregate them.
2. Random-k is unbiased and makes the theory go through, but may discard consistently useful columns/rows. The authors test simple importance metrics and find them worse, partly because they break unbiasedness

### Questions
1. Your theory assumes random-k diagonal sketching with unbiasedness and yields scaled smoothness in the bound. In practice, random-k may repeatedly miss high-utility columns/rows. Can you extend the analysis to a data-dependent, non-uniform sketch policy？
2. You aggregate [B;A] and claim compatibility with secure aggregation, but there are no end-to-end systems measurements. How does this choice affect latency under many layers or clients？

### Soundness
3

### Presentation
4

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
This paper addresses the challenge of fine-tuning large language models on resource-limited clients, where data scarcity and heterogeneous computational capabilities pose significant obstacles. It proposed Federated Sketching LoRA (FSLoRA) to allow clients to update selected submatrices of global LoRA modules according to their resources, with adjustable sketching ratios to balance efficiency and performance. A rigorous convergence analysis is provided, and extensive experiments demonstrate consistent improvements over strong baselines across multiple datasets and LLMs.

### Strengths
1. The paper is clearly written and well-structured, making it easy to follow.
2. The paper provides a solid theoretical analysis.
3. The method introduces a novel approach with clear innovation.

### Weaknesses
1. The study is restricted to RoBERTa and LLaMA, and testing on more models would further demonstrate the generality of the approach.
2. The paper lacks a discussion of the method’s limitations, which would provide more insight into its scope and potential application.
3. The method currently aggregates sketches from different clients without explicitly considering potential conflicts between diverse update directions, which may affect convergence and global model stability.
4. Some formulas are not numbered.

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes Federated Sketching LoRA (FS-LoRA) to address resource heterogeneity in federated LoRA fine-tuning. The method maintains a high-rank global LoRA module, while clients train only a sparse subset of it, defined by a client-specific "sketching matrix". This allows the training sparsity to be adapted to each client's resources. The authors provide a convergence analysis and show that FS-LoRA avoids the overheads of prior methods, demonstrating superior performance and efficiency in extensive experiments.

### Strengths
- **Flexible and Intuitive Method:** The core idea of using sketching to let clients train a sparse subset of a larger global LoRA module is an elegant solution. The adaptable sketching ratio provides a flexible way to handle client-specific resource constraints.
- **Strong Empirical Results:** FS-LoRA is shown to be highly effective, consistently outperforming all SOTA heterogeneous LoRA baselines (HETLoRA, FlexLoRA, FLORA) in accuracy across multiple models and datasets.
- **Thorough Ablation Studies:** Key design choices are well-justified through comprehensive ablations, which validate the benefits of the sketching mechanism, the use of a high-rank global module, and the impact of the sketching ratio

### Weaknesses
- Communication cost for download: In table 3, it seems the FS-LoRA’s communication cost for download is ≥ q, but actually for methods like HETEROLoRA and FlexLoRA, the download is $k_iq/r$, if the number of rank is assume to be a public information for each client. However, q communication cost seems to be required for FS-LoRA.
- Seems sketching shares very similar intuition with sparse training, which can be another approach to solve the heterogeneous rank problem. While author has illustrated FS-LoRA’s effectiveness by theory, an empirical comparison with other sparse FL method will be strongly support FS-LoRA’s effectiveness.

### Questions
see weakness

### Soundness
4

### Presentation
3

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
FSLoRA addresses heterogeneous client resources in federated LoRA fine-tuning by maintaining a high-rank global LoRA on the server while each client updates only a randomly selected subset of LoRA channels (“sketches”) sized to its compute and bandwidth. This preserves the accuracy benefits of a larger global rank yet makes client-side training and communication sparse and controllable; the authors provide convergence analysis showing how the sketching ratio governs the efficiency–accuracy trade-off. Empirically, FSLoRA achieves stronger results than prior heterogeneous-LoRA baselines on language understanding and reasoning tasks at comparable or lower resource cost, with negligible overhead from transmitting sketch indices.

### Strengths
Resource-adaptive and easy to deploy: server keeps a high-rank global LoRA while clients update sketch-selected channels, avoiding heavy SVD/merging and adding negligible downlink overhead.

Unbiasedness and convergence guarantees with the sketching ratio (k/r) cleanly controlling the accuracy–compute/communication trade-off, recovering standard FedAvg as k→r.

Robust empirical gains: consistently outperforms heterogeneous-LoRA baselines on GLUE and commonsense tasks at equal or lower cost, remaining stable with many clients and pronounced heterogeneity.

### Weaknesses
Random sketches can neglect useful channels over time; without smart re-sampling or scheduling, some capacity may stay undertrained. This manifests as persistent “cold” rows/columns in the LoRA factors, widening dispersion in per-channel gradient magnitudes, and a mismatch between channel salience and training frequency. The effect is amplified under non-IID data and non-stationary tasks, leading to biased adaptation where frequently sampled channels overfit client subpopulations while underexposed channels lag, degrading generalization and stability across rounds.

System realities like stragglers/partial participation aren’t deeply addressed; uneven participation could skew which channels get trained. When only a subset of clients reports in a given round, the corresponding sketched channels receive disproportionate updates, creating temporal and channel-wise imbalance. This induces drift in the effective training distribution, slows or destabilizes convergence, and can entrench performance disparities across clients whose data or resources systematically limit participation.

Baseline clarity is lacking: the paper doesn’t clearly report each model’s pre-finetuning (zero-shot or supervised) accuracy per dataset, making it hard to quantify absolute gains; without transparent “from–to” numbers by task/model, it remains ambiguous whether the observed improvements reflect real utility over strong base models or are largely within noise and tuning variance.

### Questions
Please see weaknesses

### Soundness
3

### Presentation
3

### Contribution
2
