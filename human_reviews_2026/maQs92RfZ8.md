# Exploring Federated Pruning for Large Language Models

- Decision: Reject
- Scores: 6, 6, 4, 2

## Abstract
LLM pruning has emerged as a promising technology for compressing LLMs, enabling their deployment on resource-limited devices. However, current methodologies typically require access to public calibration samples, which can be challenging to obtain in privacy-sensitive domains. To address this issue, we introduce FedPrLLM, a comprehensive federated pruning framework designed for the privacy-preserving compression of LLMs. In FedPrLLM, each client only needs to calculate a pruning mask matrix based on its local calibration data and share it with the server to prune the global model. This approach allows for collaborative pruning of the global model with the knowledge of each client while maintaining local data privacy. Additionally, we conduct extensive experiments to explore various possibilities within the FedPrLLM framework, including different comparison groups, pruning strategies, and the decision to scale weights. Our extensive evaluation reveals that one-shot pruning with layer comparison and no weight scaling is the optimal choice within the FedPrLLM framework. We hope our work will help guide future efforts in pruning LLMs in privacy-sensitive fields. Our code is available at https://anonymous.4open.science/r/FedPrLLM-15594.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces FedPrLLM, a federated pruning framework for privacy-preserving compression of Large Language Models (LLMs). Traditional LLM pruning often requires public calibration data, which is problematic for privacy-sensitive domains. FedPrLLM addresses this by letting each client compute a local pruning mask matrix using its private calibration data, which is then aggregated by a central server to prune the global model.

### Strengths
- The integration of federated learning (FL) and LLM pruning is a new direction. FedPrLLM fills a clear research gap in *privacy-preserving model compression*
- The experiments are extensive and methodical, covering multiple models, pruning ratios, datasets, and local methods. This level of rigor strengthens the validity of the findings.
- The paper isolates three well-defined research questions (comparison group, scaling, pruning strategy), which makes the investigation systematic and easy to follow.

### Weaknesses
- Although the framework is federated, experiments appear to use *simulated clients* (split calibration data) rather than real-world heterogeneous environments with non-IID data. This limits the external validity in true FL settings.
- The study ignores structured pruning, which is often preferred for real deployment.
- While communication costs are briefly reported, there’s limited exploration of real-time efficiency, scalability, or energy savings.

### Questions
- Add analysis or experiments showing that sharing mask matrices does not leak sensitive information. Techniques like differential privacy could be integrated.
- Include empirical results on communication time, computation cost, and memory savings, not just perplexity, to demonstrate real deployment viability.

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
This paper introduces FedPrLLM, a general federated pruning framework for large language models (LLMs) that allows clients to perform local unstructured pruning and share only binary masks to preserve data privacy. The authors systematically explore three key design dimensions—comparison group (layer/row/column), weight scaling, and one-shot vs. iterative pruning—across various LLMs, sparsity levels, and datasets. They conduct extensive experiments (6 LLMs, 10 datasets) and derive practical insights (e.g., “layer-wise voting without weight scaling and one-shot pruning works best”).

This paper makes a practical and well-executed contribution by empirically benchmarking federated pruning strategies for LLMs. While the framework is not fundamentally novel and lacks theoretical depth, its findings are well-supported, reproducible, and directly useful for the community. A stronger baseline comparison and heterogeneity analysis would elevate it further.

### Strengths
- The paper enables LLM pruning under federated settings with privacy preservation, which is interesting and underexplored.

- The paper did large-scale experiments with 6 LLMs, 10 datasets, multiple sparsity levels. Ablation studies support key claims.

- The three pruning dimensions are practical and grounded, covering critical decisions in collaborative model compression. Shows that simpler design choices (e.g., no scaling, one-shot pruning) often outperform complex alternatives, saving computation and communication.

### Weaknesses
- The core framework is a combination of known components; prior work (e.g., FedSpaLLM) has used mask voting for federated LLM pruning.

- The paper lacks formal analysis of why certain choices (e.g., weight scaling degrades performance) work or fail.

- All experiments assume IID or single-dataset settings; real-world FL often involves non-IID, skewed data, which may affect mask voting. Can authors clarify on this?

### Questions
See weaknesses.

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
3

### Summary
This paper introduces **FedPrLLM**, a comprehensive framework for **federated unstructured pruning of large language models** (LLMs) under strict data privacy constraints. In FedPrLLM, each client computes a local pruning mask using its private calibration data and uploads only the binary mask (not model weights or raw data) to the server. The server aggregates these masks via summation and selects the top-*k* entries (i.e., weights most clients agree to prune) to construct a global mask for pruning the shared LLM.

The work systematically investigates three key design choices within this framework:  
1. **Comparison group**: Should pruning decisions be made across the entire layer, per row, or per column?  
2. **Weight scaling**: Should retained weights be scaled based on client consensus (e.g., inverse pruning frequency)?  
3. **Pruning strategy**: Is iterative (layer-by-layer) pruning worth its high communication cost compared to one-shot pruning?

Through extensive experiments across **6 open-source LLMs**, **2 local pruning methods** (Wanda, SparseGPT), **3 sparsity levels**, **10 datasets**, and thousands of GPU hours, the authors derive three robust empirical findings:
- **Layer-wise comparison** is consistently superior and robust across settings.
- **Weight scaling harms performance**, despite intuitive appeal.
- **One-shot pruning matches iterative pruning in accuracy** while halving communication cost.

### Strengths
1. **High Practical Relevance**: Addresses a critical gap—how to compress LLMs in privacy-sensitive, decentralized settings (e.g., healthcare, finance)—where public calibration data is unavailable.
2. **Rigorous and Systematic Evaluation**: The scale of experiments (6 LLMs, multiple sparsities, datasets, and methods) is exceptional for a systems/ML paper. The ablation studies are thorough and convincing.
3. **Clear, Counterintuitive Insights**: The findings—especially that weight scaling hurts performance and that one-shot pruning suffices—are surprising yet well-supported, challenging assumptions from centralized pruning literature.
4. **Strong Engineering Contribution**: FedPrLLM is simple, communication-efficient, and compatible with existing local pruning methods. The framework is modular and extensible.

### Weaknesses
1. **Limited Baseline Comparison**: While the paper compares against “Local-only” and “Centralized” baselines, it does not benchmark against concurrent or prior federated compression methods (e.g., FedSpaLLM [Bai et al., 2024], mentioned in Related Work). A direct comparison would strengthen impact claims.
2. **Assumption of Public Pre-training**: The framework assumes access to a public pre-trained LLM. While standard, the paper does not discuss implications if pre-training data were private—a growing concern in DP-ML.
3. **Calibration Data Efficiency**: Each client uses only 2 samples (64 clients × 2 = 128 total). While realistic, the sensitivity to ultra-low calibration data (e.g., 1 sample/client) is not explored.
4. **Communication Cost Nuance**: Although one-shot reduces rounds, the paper does not analyze bandwidth vs. latency trade-offs or heterogeneous client capabilities.

### Questions
As described in weakness.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces FedPrLLM, a federated pruning framework for large language models (LLMs) aimed at privacy-preserving compression. Clients compute local pruning masks using private calibration data and share them with a server, which aggregates masks to prune the global model. The authors explore three design choices: (1) comparison group (layer, row, column), (2) weight scaling, and (3) pruning strategy (one-shot vs. iterative). Experiments across LLaMA variants and multiple datasets conclude that layer-level comparison, no weight scaling, and one-shot pruning are preferable.

### Strengths
- The paper addresses a timely problem: pruning LLMs under privacy constraints in federated settings.
- Provides extensive empirical evaluation across multiple models, sparsity levels, and datasets.
- Clear takeaways (layer comparison, no scaling, one-shot pruning) that practitioners can adopt.
- Framework is simple and easy to implement, making it accessible for real-world experimentation.

### Weaknesses
- The approach is a straightforward mask aggregation; similar ideas exist (e.g., FedSpaLLM[1]).
- Weak privacy claim. No secure aggregation or differential privacy. Masks may potentially leak sensitive information
- No comparison to advanced pruning methods (OWL[2], BESA[3], SliceGPT[4]) or structured sparsity approaches. While the authors can argue that some of tghe methods are structured pruning methods, it is important to compare against them.
- Communication costs lack units; no runtime, memory, or speedup metrics; catastrophic perplexities for column comparison unexplained.

### Questions
Given the strengths, I have the following questions:

1. The paper positions itself as the first systematic study of federated pruning for LLMs, but prior work (e.g., FedSpaLLM) already explores similar aggregation strategies. Specifically, the \ell_0‑aware aggregation, adaptive mask expansion, layer sampling and related variants from FedSpaLLM that aggregate sparse models or masks to meet global sparsity budgets. How does FedPrLLM fundamentally differ from FedSpaLLM or other federated pruning approaches. The differences that you included in the related work section is not sufficient?

2. The authors noted that previous approaches are not optimal, but there is no comparison against such approaches. The paper limits local pruning to Wanda and SparseGPT but omits competitive post‑training LLM pruning methods that have non‑uniform sparsity (OWL), blockwise reconstruction (BESA), or structured slicing (SliceGPT), each of which has shown superior performance or concrete speedups. Without these baselines, it is unclear whether FedPrLLM’s “best recipe” remains best when local candidates are stronger or structured. Can the authors include these baselines and clarify whether the proposed framework remains competitive.

3. My major concern is in the calculation of the local pruning mask matrix using the private calibration data. When this is sent to the global server, what are the privacy risks involved? Sending the mask without any privacy protection could potentially make the mask vulnerable to attacks.
The paper claims “privacy‑preserving compression” because only binary masks are shared. But masks are data‑dependent and can leak information (presence/absence of features/examples). The FL literature documents strong leakage from shared updates (gradient inversion, membership/property inference). Without secure aggregation or differential privacy analysis, there is no formal protection; moreover, secure aggregation would change communication/computation budgets materially, which the paper does not account for. Can the authors clarify this?

4. I also have a concern around the settings. All experiments shard IID C4 into 128 sequences and hand exactly 2 sequences per client; there is no non‑IID partitioning, no client sampling/dropouts, and no heterogeneity in compute/network, all central to FL. As a result, the conclusions may not transfer to realistic cross‑silo or cross‑device FL deployments. Can the authors clarify this?

5. Communication costs are reported as raw numbers (“6.476B”, “12.952B”) without units (bits? parameters? bytes?). It is relatively unclear what this mean. Also, there is no textual description to clarify this.

6. The catastrophic perplexities for column comparison (e.g., 311,468.53 on LLaMA‑3‑8B at 50% sparsity) indicate numerical instability or a mis‑specified comparison group; the paper does not analyze or mitigate these failures. In addition, Eq. (1) imposes $\|M_\ell\|_0 \ge k$ but the server later enforces exact top‑k (fixed sparsity). The mismatch weakens the formulation.



[1] Bai et al. FedSpaLLM: Federated Pruning of Large Language Models

[2] Yin et al. Outlier Weighed Layerwise Sparsity (OWL): A Missing Secret Sauce for Pruning LLMs to High Sparsity

[3] Xu et al. BESA: Pruning Large Language Models with Blockwise Parameter-Efficient Sparsity Allocation

[4] Ashkboos et al. SliceGPT: Compress large language models by deleting rows and columns

### Soundness
2

### Presentation
3

### Contribution
2
