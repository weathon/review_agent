# Taming Momentum: Rethinking Optimizer States Through Low-Rank Approximation

- Decision: Accept (Oral)
- Scores: 6, 6, 6

## Abstract
Modern optimizers like Adam and Muon are central to training large language models, but their reliance on first- and second-order momenta introduces significant memory overhead, which constrains scalability and computational efficiency. 
In this work, we reframe the exponential moving average (EMA) used in these momenta as the training of a linear regressor via online gradient flow. 
Building on this equivalence, we introduce LoRA-Pre, a novel low-rank optimizer designed for efficient pre-training. 
Specifically, LoRA-Pre reduces the optimizer's memory footprint by decomposing the full momentum matrix into a compact low-rank subspace within the online linear learner, thereby maintaining optimization performance while improving memory efficiency. 
We empirically validate LoRA-Pre's efficacy by pre-training models from the Llama architecture family, scaling from 60M to 1B parameters. 
LoRA-Pre achieves the highest performance across all model sizes.
Notably, LoRA-Pre demonstrates remarkable rank efficiency, achieving comparable or superior results using only 1/8 the rank of baseline methods. 
Beyond pre-training, we evaluate LoRA-Pre's effectiveness in fine-tuning scenarios. 
With the same rank, LoRA-Pre consistently outperforms all efficient fine-tuning baselines.
Specifically, compared to standard LoRA, LoRA-Pre achieves substantial improvements of 3.14 points on Llama-3.1-8B and 6.17 points on Llama-2-7B, validating our approach's effectiveness across both pre-training and fine-tuning paradigms.
Our code is publicly available at https://github.com/mrflogs/LoRA-Pre.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes LoRA-Pre, a new optimizer framework that reinterprets momentum as an online linear regression process and replaces traditional exponential-moving-average (EMA) updates with a low-rank closed-form solution. By decomposing momentum into low-rank factors and updating them analytically, LoRA-Pre significantly reduces optimizer-state memory while preserving training stability. The method is applied to both Adam and Muon, yielding LoRA-Pre-Adam and LoRA-Pre-Muon variants. Experiments on large-scale pretraining (LLaMA-1B and smaller models) and instruction fine-tuning (LLaMA-8B, LLaMA-7B) show that LoRA-Pre matches or surpasses strong baselines such as Galore, DoRA, and ReLoRA, while offering notable memory savings and rank efficiency.

### Strengths
The paper introduces a clear and technically sound rethinking of momentum optimization by framing EMA updates as an online regression problem and deriving a low-rank closed-form solution. This perspective is both conceptually original and practically valuable, bridging optimization theory with efficient model training. The proposed LoRA-Pre method is well-integrated with existing optimizers like Adam and Muon, achieving strong empirical results with lower memory costs. The experiments are comprehensive, covering both large-scale pretraining and fine-tuning, and the presentation is mathematically rigorous and well-organized.

### Weaknesses
The empirical evaluation primarily compares with projection-based low-rank optimizers; incorporating a wider range of modern baselines (e.g., Sophia, Shampoo) would provide a more complete assessment of practical advantages.

The ablation studies emphasize rank efficiency but do not clearly disentangle the effect of the low-rank momentum representation from other implementation factors such as learning-rate scaling or normalization.

The scalability and computational trade-offs of the proposed updates, particularly under distributed or large-batch training, are not fully analyzed, leaving some uncertainty about deployment efficiency in large-scale settings.

### Questions
Can the authors clarify how the proposed low-rank updates scale computationally in distributed or large-batch training? An analysis of communication and synchronization cost would help assess real-world deployment.

How sensitive is LoRA-Pre to the chosen rank or initialization of the low-rank factors? A more detailed study could clarify whether the observed gains primarily come from rank efficiency or improved optimization dynamics.

Have the authors considered extending comparisons beyond projection-based methods to include modern adaptive or second-order optimizers, to better quantify the relative benefits of the proposed approach?

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
4

### Summary
This paper introduces LoRA-Pre, a low-rank optimizer that reinterprets the exponential moving average (EMA) in momentum computation as an online linear regression problem. By exploiting this equivalence, the authors compress the optimizer’s momentum matrices via low-rank factorization (m_B,m_A) and derive closed-form Newton-style EMA updates without requiring back-propagation through the optimizer state. This leads to  LoRA-Pre achieves similar or better performance using only 1/8 of the rank, and it generalizes effectively to other optimizers such as Muon.

### Strengths
1.	The relation between EMA and online linear regression is novel and conceptually clean, providing a unified foundation for memory-efficient optimizer design.
2.	The method applies seamlessly to multiple modern optimizers (Adam, Muon), offering a general solution for optimizer-state compression in large-scale model training.
3.	LoRA-Pre consistently outperforms both projection-based and fine-tuning low-rank optimizers across different scales and tasks. Rank-efficiency studies reinforce the robustness under constrained memory.

### Weaknesses
1.	Missing Key Baseline in main table: Although Fira [1] is mentioned in the related work and ablation (Table 3), it is missing from the main results table (Table 1). This omission is non-trivial, since Fira operates under **exactly the same setting** for pre-training and it also deals with the optimizer state itself under a low-rank constraint, making it a direct and essential baseline.
The paper should include a comparison with Fira-Adam in the main results.
Even if LoRA-Pre does not necessarily outperform Fira, the paper should explicitly analyze their differences and trade-offs.

2.	This paper does not analyze approximation error or convergence fidelity under low-rank constraints.

3.	The ablation focuses solely on rank. Adding sensitivity analysis to other hyperparameters (like $\beta$ and $\gamma$) would offer a more comprehensive understanding.

[1] Fira: Can We Achieve Full-rank Training of LLMs Under Low-rank Constraint?

### Questions
see weaknesses

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
3

### Summary
This paper introduces LoRA-Pre, a new memory-efficient optimizer designed to reduce the memory footprint of optimizer states during large language model (LLM) pre-training and fine-tuning. The central insight reframes the exponential moving average (EMA) momentum update as an instance of online linear regression, then leverages this equivalence to propose a low-rank factorization of the optimizer's momentum matrices. Detailed update rules are derived for Adam and Muon optimizers, with closed-form solutions shown via Newton's method. Extensive experiments on Llama-family models (from 60M to 1B parameters) demonstrate that LoRA-Pre achieves competitive or superior performance with much smaller rank than baseline low-rank methods, both for pre-training and memory-efficient fine-tuning.

### Strengths
__1) Theoretical Insight:__ The paper establishes a non-trivial mathematical equivalence between EMA momentum updates and online linear regression (Section 3.2), which is both elegant and enables a new way to think about optimizer state compression beyond ad-hoc engineering.

__2) Methodological Contribution:__ Closed-form Newton-based update rules are derived for the low-rank factors, with careful exposition (Section 3.3, Theorem 3.1, Appendix A). This enables stable and efficient momentum updates without full-rank storage.

__3) Empirical Validation:__ Experiments are thorough, covering both pre-training (Section 4.1, Table 1) and fine-tuning (Section 4.2, Table 2) across a range of Llama models and tasks (GSM8k, MATH500). 

__4) Competitive baselines:__ Adam, Muon, LoRA, Galore, and others are compared, and LoRA-Pre outperforms or matches the best across most benchmarks.

### Weaknesses
__1) Incomplete related work coverage.__ The literature review appears incomplete. The paper omits several relevant lines of work where optimizer momenta or gradient statistics are decomposed or constrained in low-rank form. For example, MLorc [1] and MoFaSGD [2] address momentum/gradient factorization. Within PEFT, papers like LoRA-Pro [3] and LoFT [4] are highly relevant (the latter yielding updates that resemble those proposed here). Another closely related direction is Riemannian low-rank pretraining/optimization [5], which explicitly performs muon-like steps on a low-rank manifold. While the paper does cite LoRA-Right and LORO, this seems insufficient to position the contribution within the broader space of low-rank momentum and manifold-constrained optimization. In addition, AdaPM [6] also decomposes momentum into a low-rank representation, conceptually aligning with the spirit of the present work. Although the specific algorithmic designs differ, ADAPM appears to be a highly relevant work. I recognize that ADAPM was posted after the ICLR submission deadline, but it should be cited and discussed in the camera-ready to properly contextualize the contribution.

__2) Memory accounting is unclear and potentially unfavorable in some regimes.__ LoRA-Pre reduces the persistent state, but each step still materializes full momentum via $m = m_B m_A$ and momentum with bias correction to apply the update. This allocation increases peak memory, unlike methods that avoid full momentum materialization (e.g., standard LoRA or GaLore’s projected states). For small models (for example with 1 layer), LoRA-Pre can yield no benefit and may increase both memory and runtime. A complete peak-memory breakdown (parameters, activations, gradients, optimizer states, and transient products) is missing.

__3) Computational overhead and numerical stability are underexplored.__ Per-step inversions of $r \times r$ matrices add nontrivial overhead and can be ill-conditioned as $r$ grows. The paper does not specify stabilization strategies nor quantify runtime overhead relative to Adam/Muon/GaLore. Empirical stability analysis is absent.

__4) Fairness of comparisons and the $r/d{model}$ control variable.__ Using $r/d_{model}$ as the primary control for fairness is inadequate: peak memory and step time depend on whether full momentum is materialized, number of layers, matrix operations performed at each step, etc. Comparisons should be matched on peak memory and/or throughput wall-clock time.

__5) Typo:__ in Algorithm 1 line 12 it should be $(1 - \beta_2)$, not $(1 - \beta_1)$.

### Questions
1) How does the method handle ill-conditioning when computing matrix inverses for the low-rank updates, especially as $r$ increases? Is there any regularization, and how is numerical stability ensured in practice?

2) Under the settings of Table 1, which has the larger peak memory: LoRA-Pre (Adam) or plain Muon?

3) In Table 1, classical Muon consistently outperforms LoRA-Pre(Muon) (and all other methods), while LoRA-Pre(Adam) always outperforms Adam. Also as training tokens increase, LoRA-Pre(Adam) overtakes LoRA-Pre(Muon), whereas among “full” methods Muon remains superior to Adam. What interactions between Muon’s orthogonalized updates and factorized momentum explain this gap, and can they be mitigated?

4) Your Theorem 3.1 derives closed-form factor updates via an online Newton step on $L(m_B, m_A, g)$. How does this update relate to the closed-form updates used in LoRA-Pro [3] and LoFT [4], which did not explicitly employ Newton’s method?

[1] Shen W. et al. MLorc: Momentum Low-rank Compression for Large Language Model Adaptation //arXiv preprint arXiv:2506.01897. – 2025.

[2] Shivagunde N. et al. Approximations may be all you need: Towards pre-training LLMs with low-rank decomposition and optimizers. – 2024.

[3] Wang Z. et al. Lora-pro: Are low-rank adapters properly optimized? //arXiv preprint arXiv:2407.18242. – 2024.

[4] Tastan N. et al. LoFT: Low-Rank Adaptation That Behaves Like Full Fine-Tuning //arXiv preprint arXiv:2505.21289. – 2025.

[5] Bogachev V. et al. LoRA meets Riemannion: Muon Optimizer for Parametrization-independent Low-Rank Adapters //arXiv preprint arXiv:2507.12142. – 2025.

[6] Zhang Y., Liu Y., Fang C. AdaPM: a Partial Momentum Algorithm for LLM Training //arXiv preprint arXiv:2510.09103. – 2025.

### Soundness
2

### Presentation
2

### Contribution
2
