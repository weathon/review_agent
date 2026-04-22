# RAPA: Recursively Aligned Pathway Adaptation of Large Language Models

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 6, 2

## Abstract
Parameter-Efficient Fine-Tuning (PEFT) adapts large language models (LLMs) by training only a small fraction of parameters. Adapter-based approaches reduce compute per step but introduce practical overhead from the additional adapter path (e.g., extra kernel launches and activation storage). Adapter-free approaches avoid this structural overhead by directly updating pretrained weights; however, per-layer random index selection can fragment the trainable subspace, attenuating gradient flow and limiting accuracy. We propose **Recursively Aligned Pathway Adaptation (RAPA)**, an adapter-free PEFT method that forms index-consistent pathways through depth. RAPA follows two principles: (i) selecting balanced submatrices that maximize the number of weights alignable across layers, and (ii) recursively aligning these indices across layers and residual connections. In experiments, RAPA matches or surpasses strong PEFT baselines across most benchmarks while preserving adapter-free efficiency with minimal memory and compute overhead. Code is available at \url{https://anonymous.4open.science/r/rapa}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes RAPA, which is an adapter-free PEFT method for LLM fine-tuning. The core idea is to (i) reallocate fixed parameter budget into a balanced square submatrix per linear layer rather than LoRA’s low-rank adapters (such that total trainable params unchanged) and (ii) align the selected row/column index sets recursively across layers and through residual connections, so that gradients propagate along trainable pathway. The trainable square submatrix yields higher effective rank (compared to LoRA), and avoids adapter induced activation overheads. Empirically, RAPA matches or outperforms strong adapter-based (LoRA/DoRA) and adapter-free (PaCA) baselines.

### Strengths
1, Cross-layer index-aligned subspace tuning is a novel idea, backed up by explanations using backprop for why aligned indices improve gradient consistency.
2, Balanced submatrix offers higher ranks under equal trainable params budget
3, The paper provides substantial evaluation which show RAPA outperforms baselines under equal parameter budget

### Weaknesses
1, While the squared matrix maximizes the trainable matrix’s rank in theory, in practice it’s not clear why “rank” should be the only direction to optimize for. Explanations for why this optimality aligns with practical goals are absent.
2, The paper focuses on dense arch. MoE, another popular choice of LLM arch, is underexplored. Especially, it’s non-trivial and unclear whether router and expert FFN admit similar training dynamics and the same technique is applicable.
3, Different modules (such as kqv projections VS FFN) could favor different shaped submatrix or alignment mechanisms, but this is not well understood with the existing analysis. It’s also not clear how the embedding matrix (and final projection) is handled; considering its completely different function and large proportion of memory it takes, handling embedding is a subtle and non-trivial topic which was not touched.
4, The paper covers adapter based and adapter free methods, but left out one important area based on side-tuning, which relies on additional trainable parameters without triggering non-linear activations through backbone by design. Please include references to this line of work:
	Read: Recurrent adaptation of large transformers, John Nguyen, Sid Wang, Ke Li, and Carole-Jean Wu. 2023.
        Side-tuning: a baseline for network adaptation via additive side networks. Jeffrey O Zhang, Alexander Sax, Amir Zamir, Leonidas Guibas,and Jitendra Malik 2020

### Questions
1, While square trainable matrix maximizes rank under a fixed budget, could the authors explain why maximizing rank alone is the right optimization objective in practice? Are there either empirical or theoretical insights connecting rank-optimality to downstream task performance, beyond heuristic intuition presented in the text?
2, How does RAPA extend to MoE architectures, where routers and expert FFNs may follow different learning dynamics? Do the authors expect the same alignment mechanism to apply to both router and expert parameters, or are modifications necessary to maintain stability?
3, Could different modules (e.g., kqv projections vs. FFN layers) benefit from different submatrix shapes or alignment strategies? Additionally, how are token embeddings and the LM head treated in RAPA? Given their distinct functional roles and large parameter footprint, have the authors explored whether they require different handling?
4, The related-work discussion omits side-tuning based adapter-free methods that add parallel trainable subnetworks without modifying backbone nonlinearities (e.g., Zhang et al., 2020; Nguyen et al., 2023). Could the authors compare RAPA against or at least position it relative to this class of methods?
5, Have you tested on 34B/70B (or larger) models? Given your kernel-launch argument, do you anticipate the relative efficiency gap to widen or narrow at scale?
6, Given the recursive nature of your approach, can you refine your theoretical analysis by deriving bound on gradient variance iteratively along the pathway?

### Soundness
2

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
4

### Summary
This paper proposes RAPA (Recursively Aligned Pathway Adaptation), a new adapter-free parameter-efficient fine-tuning (PEFT) algorithm for large language models (LLMs). It selects balanced square submatrices to maximize expressive capacity under a fixed parameter budget and recursively aligns trainable indices across layers, blocks, and residual connections. This alignment aims to create coherent gradient pathways, reducing discontinuities and accelerating convergence. Experiments in the paper show RAPA matching or surpassing baselines like LoRA, DoRA, and PaCA on benchmarks such as MMLU, with lower memory and training time.

### Strengths
- Efficiency Gains: RAPA minimizes training overhead by avoiding adapter modules, which can lead to fewer kernel launches and lower activation memory. Experiments indicate it uses around 41-53 GB of memory for models like LLaMA2-7B/13B, often outperforming adapter-based methods in speed and resource use.
- Actionable Ablation Studies: Figure 5 on p.7 demonstrates that shaping the row/column ratio for the tuned submatrix and applying index alignment both improve task performance, supporting the design decisions empirically as well as theoretically.
- Performance Improvements: By aligning pathways recursively, RAPA stabilizes gradient propagation, leading to faster convergence and higher accuracy on benchmarks such as MMLU (e.g., 54.6% average for LLaMA2-7B). This structured selection of balanced submatrices enhances representational capacity under fixed parameter budgets.

### Weaknesses
- Reliance on Random Index Selection: While RAPA's primary contribution is the recursive alignment of pathways, the initial selection of this pathway is based on a randomly chosen index set $I$ (Appendix E). A concern arising from this design is that a purely random selection, even if structurally coherent, may not be optimal. It seems likely that this method could miss weights that are critical for task adaptation, potentially limiting its performance relative to importance-driven or calibrated approaches.
- Questionable Statistical Significance of Gains: The stability of the method's reported gains warrants closer inspection. Although Appendix A analyzes robustness to random seeds, the reported fine-tuning performance across four seeds shows a non-trivial performance spread (53.7% to 54.3%, +0.6%, from Table 5). This fluctuation resulting from the random seed choice is comparable in magnitude to the performance improvement claimed over prior work, such as PaCA (+0.6% on LLaMA2-13B, Table 2). This raises questions about the statistical significance of the reported gains and suggests that the method's advantages might be highly sensitive to the specific random seed chosen for index selection.
- Unsubstantiated Contribution of “Rank Expansion”: This benefit is not experimentally disentangled from the paper's other core contribution: index alignment. It is plausible that the observed performance gains are predominantly attributable to the index alignment strategy, rather than the putative “rank expansion”. The paper would be significantly strengthened by an ablation study that isolates these two effects (e.g., by applying RAPA's “Rank Expansion” parameterization to other adapter-free methods like PaCA.)

### Questions
- Can the method be extended or modified to support dynamic, input- or data-dependent index selection, without incurring the calibration cost that motivates RAPA’s simplicity? What are the trade-offs?
- In Figure 3, the current diagram does not clearly visualize how these three different types of alignment are implemented or how they relate to each other. The labels (1), (2), and (3) are present, but their graphical representation is ambiguous. Could the authors revise it to provide a more explicit and differentiated illustration of each of the three alignment strategies? This would significantly improve the paper's clarity and help readers grasp the precise mechanics of the RAPA method.
- Could the authors comment on the method's sensitivity to other hyperparameters and design choices that were not tested? For example, potential confounders such as the distribution of selected indices (e.g., contiguous vs. sparse) are not explored. This leaves a gap in understanding the method's robustness to its own hyperparameter changes.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents RAPA, an adapter-free PEFT method for LLMs. RAPA selects balanced square submatrices of parameters for adaptation and recursively aligns these subspaces across all layers and residual connections, with the goal of improving gradient flow while minimizing the practical overhead associated with adapter-based approaches. The approach is supported by mathematical motivation and empirical validation across several benchmarks, consistently outperforming or matching PEFT baselines without increasing compute and memory costs.

### Strengths
S1: This paper provides a thorough analysis of the limitations of both adapter-based and adapter-free PEFT methods, identifying sources of overhead and the fragmentation issue in per-layer random adaptation. It presents a clear motivation and offers insightful mathematical analysis, demonstrating how recursive index alignment enhances gradient flow.
S2: Extensive experiments and ablation studies across multiple benchmarks convincingly demonstrate the effectiveness of the proposed method.

### Weaknesses
W1: All experiments are conducted on standard Transformer architectures, and the method’s adaptability to architectural variants (e.g., MoE or deep convolutional Transformers) remains untested, limiting the generality of its claimed universality.
W2: The current implementation employs a fixed global index set across all layers, which may restrict its ability to adapt to tasks with heterogeneous per-layer requirements.

### Questions
Q1: Can the authors provide controlled experiments that isolate the effect of increased parameter subspace rank from cross-layer alignment? Does alignment alone (with fixed small rank) provide benefit, or does improvement scale almost linearly with subspace size?
Q2: How does RAPA perform in Transformers featuring more complex or unconventional residual connection patterns?
Q3: Is there empirical evidence demonstrating its effectiveness in architectures that differ substantially from the standard Transformer block structure used in current experiments?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a PEFT method which finetunes only a subset of the parameters in a pretrained LLM and backprops error only through the finetuned part of the model.  The novelty is that the same parameter indexes are used across layers, so that error passing through the trained weights and error passing through their skip connections are aligned (i.e. either both blocked or both computed).  This leads to small but consistent improvements over previous PEFT methods with similar computation budgets.

### Strengths
The idea is clearly defined and executed.  The results show consistent improvements across several benchmark tasks.

### Weaknesses
My main concern is that the contributions of this paper are small.  As I understand it, it is really a minor variation on PaCa, implementing an idea that is rather straightforward.  Also, the resulting improvements are not very substantial.

The technical descriptions of how and why the parameter selection is done are too vague.  I can understand how they apply to FFN layers, but it is not at all clear to me how they apply to attention layers.  In that case, it isn't clear if this method only applies to the value and output matrices, or if it also applies to the query and key matrices, and if so how.  And it is not clear how the index selection interacts with the different attention heads, or the low-rank nature of the query-key interactions.

### Questions
How do equations (2)-(7) apply to the attention layers?

Note that saying that frozen weights block error backprop is technically incorrect.  This is a choice based on efficient computation; there would be no problem computing these error terms if you wanted to.

### Soundness
4

### Presentation
2

### Contribution
2
