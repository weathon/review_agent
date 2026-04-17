# Towards Universal & Efficient Model Compression via Exponential Torque Pruning

- Decision: Reject
- Scores: 4, 4, 4

## Abstract
The rapid growth in complexity and size of modern deep neural networks (DNNs) has increased challenges related to computational costs and memory usage, spurring a growing interest in efficient model compression techniques. Previous state-of-the-art approach proposes using a Torque-inspired regularization which forces the weights of neural modules around a selected pivot point. Whereas, we observe that the pruning effect of this approach is far from perfect, as the post-trained network is still dense and also suffers from high accuracy drop. In this work, we attribute such ineffectiveness to the default linear force application scheme, which imposes inappropriate force on neural module of different distances. To efficiently prune the redundant and distant modules while retaining those that are close and necessary for effective inference, in this work, we propose Exponential Torque Pruning (ETP), which adopts an exponential force application scheme for regularization. Experimental results on a broad range of domains demonstrate that, though being extremely simple, ETP manages to achieve significantly higher compression rate than the previous state-of-the-art pruning strategies with negligible accuracy drop.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Exponential Torque Pruning (ETP), a regularization-based structured pruning method that replaces the linear “force” schedule of Torque with an exponential schedule: nearby modules (a pivot) receive weak penalties while distant modules are penalized strongly, encouraging them toward zero and enabling structured removal. ETP is evaluated across vision (CIFAR-10/100, ImageNet-1k), NLP (GLUE: SST-2, MRPC on BERT/RoBERTa), graphs (PPI/GAT) and time-series (ETTh1/Informer). It also includes a small LLM case study on OPT-350M/WikiText at 50% sparsity (perplexity metric). Reported results show better speed-accuracy trade-offs than strong structured pruning baselines (e.g., DepGraph, Torque) and competitive/post-training LLM methods (SparseGPT, Wanda, LLM-Pruner) under the stated settings.

### Strengths
- Simple, general idea with clear intuition. Replacing linear with exponential penalization aligns with the goal “strongly suppress far modules, preserve near ones,” and is easy to add as a loss term next to the task loss. The method is architecture-agnostic and demonstrated across several domains.
- Broad empirical coverage. Results on CNNs/ViT, BERT/RoBERTa, GAT and Informer show favorable accuracy at equal MACs speed-up, and robustness under more aggressive compression.

### Weaknesses
- Section 4.4 reports the 50% sparsity perplexities on OPT-350M/WikiText and compares to SparseGPT, Wanda, DepGraph, LLM-Pruner, but does not state whether ETP was used with additional training/finetuning (task loss + ETP loss) or applied in a purely post-training fashion; there are no training hyperparameters for OPT in Appendix 7.1, unlike other tasks. This raises questions about the exact pipeline and reproducibility.
- The paper says, “All methods are constrained to 50% sparsity for a fair comparison” for OPT-350M, but does not explicitly confirm whether SparseGPT/Wanda/LLM-Pruner were run on public dense checkpoints (the standard) or on a model that had already been trained with ETP. Appendix 7.1 states that experiments follow baselines’ provided implementations, which suggests baselines are run in their own standard setting, not on an ETP-trained model—but this is not spelled out for OPT. Please clarify
- Impact of ETP regularization on downstream task performance (LLMs) is not evaluated. For LLMs, only WikiText perplexity on OPT-350M is reported; there is no evaluation on GLUE-style downstream tasks (e.g., CoLA, STS-B, RTE, MNLI, QNLI, QQP) to assess whether adding ETP loss to the task objective harms downstream utility. By contrast, for RoBERTa/BERT the paper evaluates only SST-2 and MRPC (two GLUE tasks), not the broader suite.
- Beyond geometric intuition/plots, the paper lacks formal analysis on optimality/stability or why exponential scheduling should yield provably superior sparsity patterns (e.g., per-layer sparsity heterogeneity, convergence, or KKT-style characterizations). This weakens the “universal” claim.

### Questions
1. Is it possible to combine ETP with  quantization?

2. Appendix 7.1 indicates $\beta$ is a single scalar selected via grid search. Is it a global one for all layers or per-layer one?

3. Will ETP work for small model but hard task? Like Resnet18-ImageNet?

4. For ResNet-50-ImageNet, after structured removal, how do you handle BatchNorm?

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
4

### Summary
This paper proposes Exponential Torque Pruning (ETP), a regularization based structured pruning method that generalizes prior Torque style penalties. Instead of using a linear increase of penalty with distance from a pivot index, the proposed ETP applies an exponential schedule. The objective adds a weighted L2 penalty on module weights (filters, channels, heads, neurons) where the weight grows exponentially with the index distance from a pivot. The goal is to approximate a step like selection that keeps near pivot modules almost unpenalized while strongly shrinking far modules. Experimental results are evaluated on vision backbones (VGG, ResNet, ViT), NLU encoders (BERT, RoBERTa), a graph model (GAT), and a time series Transformer (Informer). Across these tasks ETP typically improves the accuracy versus compression tradeoff relative to linear torque and several structured baselines.

### Strengths
The strengths can be summarized as follows.

- (1) The idea is clear and simple. Replacing a linear distance weight with an exponential weight is easy to adopt and aligns with the intended near keep and far shrink behavior of torque style penalties.

- (2) The applicability is wide. The same formulation works across CNNs, Transformers for vision and NLP, a graph model, and a time series model, which supports the claim of generality.

- (3) The empirical gains over the chosen baselines are consistent. The proposed ETP often achieves higher accuracy at the same or larger speed up than linear torque and several structured pruning baselines, with multi-seed averages and significance tests that improve confidence.

### Weaknesses
Also, the weaknesses are summarized as follows.
- (1) Novelty is relatively not high. The contribution is a schedule change within an existing regularization template rather than a new pruning paradigm or analysis. A theoretical treatment that connects the exponential weight to an ideal selection boundary would strengthen the work.
- (2) The SOTA claim is not fully established. Since only one paper published in 2025 has been cited, the comparison set omits several recent and strong methods for LLMs and ViTs. Without these results the universal and efficient claim remains unproven outside the selected baselines.
- (3) Limited LLM scale. The study focuses on BERT and RoBERTa on GLUE subsets rather than decoder only LLMs where much of the current pruning literature concentrates.
- (4) Hardware metrics are missing. Results are reported in MACs or FLOPs based speed ups but there are no measurements of real latency, throughput, or energy on CPUs and GPUs, which are critical for structured sparsity.
- (5) Sensitivity to pivot choice is under explored. Since distance from a pivot drives the penalty, performance may depend on how the pivot and indexing are chosen. This

### Questions
Q1. For a stronger SOTA claim, can you add comparisons to recent structured and semi-structured pruning methods for ViTs and LLMs at matched sparsity or compute budgets, and include leading post training unstructured methods for context on the accuracy versus compute frontier

Q2. Can you report hardware results, such as end to end latency and throughput on one CPU and one GPU, and possibly energy per inference, to show that MACs based gains translate to deployment gains for structured sparsity?

Q3. Do the conclusions hold for decoder only LLMs such as LLaMA 7B or 13B with head or neuron level structured pruning? If not, what obstacles arise when applying ETP at that scale?

Q4.  Can you provide a theoretical insight or bound that explains when and why an exponential schedule is closer to an ideal keep or prune boundary than linear or logarithmic schedules, and how the steepness parameter should be set as a function of desired sparsity?

### Soundness
2

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
This paper addresses a key limitation of traditional Torque Pruning, which applies a linear force scheme and thus causes insufficient pruning for distant modules while over-penalizing nearby essential modules. To mitigate this issue, the authors propose Exponential Torque Pruning (ETP) — a general structured pruning framework that applies an exponential regularization force proportional to λd\lambda^{d}λd, where ddd denotes the distance from the pivot module. This exponential formulation allows ETP to apply much stronger constraints on distant redundant modules while preserving the weights of closer, necessary ones. Extensive experiments are conducted across four domains — vision (VGG19, ResNet-50, ViT-B/16), NLP (BERT, RoBERTa), graph (GAT), and time-series forecasting (Informer).ETP consistently outperforms prior SOTA methods such as DepGraph, Torque, and GReg in both compression ratio and accuracy retention. For instance, ETP achieves 9× speed-up with only 2.2% accuracy drop on CIFAR-100 (VGG19) and 42× speed-up with 2.4% drop on BERT (SST-2). The method also shows promising results on large models (OPT-350M), improving perplexity over SparseGPT and DepGraph under the same sparsity budget.

### Strengths
1.	Improved Regularization Design: ETP introduces an exponential force application scheme that better aligns with the intuition of structured pruning, effectively addressing the imbalance in the original Torque method.
2.	Strong Empirical Results Across Domains:The experiments span vision, language, graph, and time-series tasks, demonstrating the universality and robustness of the approach across architectures.

### Weaknesses
1.	Limited Novelty: The core idea — replacing the linear regularization force in Torque with an exponential one — is conceptually straightforward. Similar non-linear or adaptive regularization ideas (e.g., GReg, Wang et al., 2020) have been explored before. The theoretical novelty could be better articulated.
2.	Lack of Real Hardware Validation: While the paper claims ETP is suitable for edge deployment, it only reports computational reduction via MACs, not real inference latency or energy consumption on hardware devices. Since MACs reduction does not always correlate with actual latency, this limits the practical evidence for “efficiency.”
3.	Outdated Experimental Baselines: The study primarily evaluates on older architectures (VGG-19, ResNet-50, BERT) without including modern 2024+ models such as LLaMA 3 or Qwen 3, which restricts the relevance to current research trends.

### Questions
1.	Have the authors conducted real-device evaluations (e.g., on GPU, CPU, or edge hardware) to confirm the claimed efficiency gains in latency or energy?
2.	How is the exponential base λ selected across different architectures and layers? Is there any theoretical or adaptive mechanism beyond grid search?

### Soundness
3

### Presentation
3

### Contribution
2
