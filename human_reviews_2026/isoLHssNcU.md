# Memory Savings at What Cost? A Study of Alternatives to Backpropagation

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 4, 6, 6

## Abstract
Forward-mode automatic differentiation (FmAD) and zero-order (ZO) optimization have been proposed as memory-efficient alternatives to backpropagation (BP) for gradient computation, especially in low-resource settings. However, their practical benefits remain unclear due to two key gaps: a lack of comparison against memory-efficient BP variants, such as activation checkpointing, and a lack of systematic characterization of tradeoffs between accuracy, memory, and computation efficiency when comparing these methods. This work presents a comprehensive empirical comparison of BP, FmAD, and ZO methods. We first conduct theoretical analysis to present intuitions that, while FmAD and ZO can reduce memory usage, they incur significant costs in accuracy, convergence speed, and computation compared to BP with checkpointing. These drawbacks worsen with larger models or constrained perturbation budgets. Empirical experiments on large language and vision-language models show that BP with checkpointing outperforms FmAD and ZO variants, including those enhanced with variance reduction, achieving up to 31.1% higher accuracy, 34.8% faster convergence, and 3.8$\times$ fewer computations at comparable memory usage. We also investigate specific failure modes in FmAD and ZO, including instabilities in Jacobian-vector products that can destabilize training. Our results highlight fundamental limitations of FmAD and ZO, and the effectiveness of BP with checkpointing for model training, under memory-constrained settings.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper revisits claims that forward-mode automatic differentiation (FMAD) and zeroth-order (ZO) optimization can serve as memory-efficient alternatives to standard backpropagation (BP) for training large models. The authors provide an extensive empirical comparison of BP (including checkpointed variants), FMAD, and ZO methods across language and vision-language tasks. Their key message is that while FMAD and ZO can reduce activation memory, they do so at substantial cost to convergence rate, accuracy, and total compute. BP with activation checkpointing achieves similar memory usage but consistently outperforms these alternatives in accuracy, convergence speed , and computational efficiency.

### Strengths
$\textbf{Comprehensive empirical coverage.}$

The experiments are unusually broad, spanning multiple datasets (AGNews, GSM8K, MMLU, VQAv2, etc.), architectures (BERT, OPT, LLAMA, QWEN), and numerous variants of FMAD and ZO (such as variance reduction: gradient accumulation and adaptive perturbation sampling).
The empirical results are well-documented, and the authors provide open-source code. This offers useful reference points for practitioners evaluating the memory-accuracy trade-off.

---

$\textbf{Clear empirical conclusion.}$

The message is consistent: BP with checkpointing remains the dominant choice even under tight memory budgets, as FMAD and ZO cannot compete once compute and convergence stability are taken into account.

### Weaknesses
$\textbf{Theoretical contribution is minimal.}$

The paper repackages well-known results: FMAD and ZO introduce variance proportional to dimensionality and thus degrade convergence as dimension-dependent. The provided asymptotic bounds in Table 2 mirror classical results in standard literature. There is little new theory beyond presentation.

---

$\textbf{Empirical design conflates purpose with performance.}$

Demonstrating that FMAD and ZO are inferior to checkpointed BP under standard differentiable training does not advance understanding: this outcome is expected from basic principles.
In particular, the argument that these methods are “memory-efficient alternatives” to BP ignores the fact that checkpointing already occupies this design space. The resulting conclusion (“BP with checkpointing is better”) is tautological.

---

$\textbf{Overemphasis on scale without insight.}$

The inclusion of billion-parameter models makes the study impressive in scale but not necessarily more insightful. The experimental results largely reiterate the same qualitative pattern seen on smaller models.

---

$\textbf{Lack of forward-looking or constructive perspective.}$

The paper ends with a negative verdict on FMAD/ZO, but does not explore when these methods should be used (e.g., non-differentiable, privacy-restricted, or hardware-limited cases). As written, the paper reads more as a rebuttal to prior enthusiasm than a constructive contribution.

---

$\textbf{Questionable motivation for the comparison.}$

The central premise, directly comparing BP (a first-order method) with ZO (a derivative-free scheme) and FMAD (a forward-mode derivative estimator), is conceptually weak. These methods serve distinct purposes and are not competing paradigms for the same optimization regime.

### Questions
* What practical or conceptual scenario do the authors envision where FMAD or ZO could plausibly compete with BP-checkpointing? Without a defined use case, the study risks becoming a “straw-man” comparison.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies alternatives to backpropagation (BP), specifically forward-mode automatic differentiation (FM-AD) and zero-order (ZO) optimization, for training neural networks. The authors advocate using BP with checkpointing as the primary method and argue that the memory savings offered by FM-AD and ZO do not justify their performance loss and additional computational cost. They consider different variants of these algorithms and provide a theoretical analysis of their convergence rates, memory usage, and time complexities. In addition, they present extensive experiments on various models and datasets to compare these methods against each other.

### Strengths
- The problem under investigation is relevant and practically significant.
- Theoretical analysis compares different important aspects of these algorithms.
- Experiments are extensive, covering different models and datasets.
- The paper is generally well written and easy to read.

### Weaknesses
- The paper is a study of existing methods and benchmarks them on different models and datasets. The theoretical results for FM-AD are not new and are already well known in the community; however, I am not familiar with the novelty of the results for ZO. The overall message of the paper is not surprising and aligns with most previous works comparing these methods to BP. While the authors provide extensive experiments to compare these approaches, the text largely reiterates what is already visible in the figures, rather than offering deeper insights into why certain behaviors occur or providing more intuitive explanations of the results. For a paper of this nature, readers would expect more conceptual understanding and insightful remarks, rather than a lengthy restatement of numerical findings.


- The directions used in FM-AD and ZO are drawn from a normal distribution, and no alternative distributions or selection strategies are considered. I believe the performance of these algorithms may strongly depend on the choice of directions, making this an important factor for comparison. For example, in FM-AD, when n=D, one can set the directions as basis vectors and recover the gradient exactly, without error. This point is neither mentioned nor included in the comparisons. By treating normally distributed directions as an inherent part of the algorithms, the paper risks presenting a potentially misleading picture of their performance.

### Questions
- Why is the compute in Table 2 for the parallel versions of the methods the same as for the sequential versions?

- Do you normalize the directions drawn from the normal distribution? Without normalization, the gradient magnitudes are altered, which requires extra care in adjusting and scheduling learning rates for FM-AD and ZO.

- In Figure 2, some graphs converge initially but begin to diverge after some time. Why does this happen?

- In Figure 3, why is gradients + optimizer state + misc higher for FM-AD? Also, the activations memory for this method should be roughly 2× that of ZO (one vector for forward pass activations and another for JVP), so why does it appear significantly larger in the figures?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This study investigates some alternatives to backpropagation aimed at training large model with minimal memory usage. These methods are compared against standard backpropagation, as well as checkpointed backpropagation.
After both theoretical analysis and empirical measurements on several models and fine-tuning tasks, the authors show that forward only methods suffer from poor convergence properties, due to noisy gradient estimation. Checkpointed backpropagation, on the other hand, computes exact gradient with similar memory requirements, and hence appears as a more viable method for fine-tuning large models under strict memory constraints.

### Strengths
- The claim that checkpointed BP can be much better than forward only alternatives while achieving the same memory requirements, is well motivated. It may also encourage the community to include it as a strong baseline when performing experiments, since it appears to be missing in many works.
- A theoretical analysis of each method is performed, followed by empirical validation.
- The study covers multiple aspects of the methods including accuracy, memory, time, FLOPs, and convergence speed.
- The evaluation setup is diverse, with models between 110M and 13B parameters, for language and vision-language tasks.

### Weaknesses
- The HiZOO training algorithm [1] may deserve some discussion in the paper. This forward-only method, tries to address the slow convergence of MeZO by leveraging second-order information, which is what is criticized in the current study.
- The theoretical analysis provides time and memory complexities for each method. However the big O notations hide constant factors, which are important to consider since for instance a backward pass is about 2x slower than a forward pass. Since the methods rely on VJP, JVP and/or forward passes, they have different constant factors.
- The complexities are presented without accounting for parallelism. Forward-only methods remove the backward locking, which unlocks new ways of parallelizing the computations across devices. Similarly, ZO methods involve multiple forward passes which could be run in parallel, thus reducing the effective training time.

Minor:
- Some notations are a bit confusing. For instance in Table 2, why use $D$ for the number of layer when it was defined as $p$ before (line 133).

[1] Second-Order Fine-Tuning without Pain for LLMs:A Hessian Informed Zeroth-Order Optimizer, Zhao et al., ICLR 2025

### Questions
- Why does the (vanilla) FMAD method have a similar FLOPs/iter to BP-CHECKPOINTING in Table 4? I would expect that the methods would be ordered like: FMAD < BP < BP-CHECKPOINTING, since FMAD involves a JVP during the forward pass that is usually noticeably faster than doing a backpropagation, and checkpointed BP adds a computational overhead over BP.
- In Table 2, I do not get why the computational complexity of BP with checkpointing is in $\mathcal{O}(d \log D)$.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper provides a comprehensive theoretical and empirical comparison of backpropagation (BP) with activation checkpointing against forward-mode automatic differentiation (FMAD) and zero-order (ZO) optimization methods for gradient computation. 

It highlights trade-offs in accuracy, convergence speed, memory, and computation, introduces variance reduction techniques (gradient accumulation and adaptive perturbation sampling) for FMAD and ZO, and analyzes their failure modes, concluding that BP with checkpointing generally outperforms alternatives on large language and vision-language models across diverse tasks.

### Strengths
The paper's main contribution is highlighting gaps of prior work on FMAD and ZO methods, particularly the lack of comparisons to memory-efficient BP variants like activation checkpointing and incomplete evaluations of trade-offs beyond memory savings. Its unified theoretical framework (Table 2) effectively distills convergence bounds, memory costs, and compute complexities. 

The empirical evaluation is broad, spanning large-scale models (Llama 3.1 8B, Qwen 2 VL 7B), tasks (text classification, generation, visual QA), and method variants covering discussions around accuracy, convergence time, memory, and FLOPs. 

The variance reduction techniques (Gradient Accumulation, Perturbation Sampling) proposal over vanilla FMAD/ZO demonstrates a significant improvements in accuracy. The failure mode analysis (JVP instabilities, optimizer interactions) offers practical insights into why these alternatives underperform, enhancing the paper's significance for practitioners in low-resource settings.

### Weaknesses
While the theoretical analysis is insightful, it relies on assumptions like non-convex but L-smooth functions, which may not fully capture the complexities of modern transformers (attention mechanisms or quantization effects in QLORA); providing sensitivity analyses or relaxations could strengthen claims. 

Empirically, the focus on QLORA with low ranks (r=1) limits generalizability—higher ranks or full fine-tuning scenarios might alter trade-offs, especially for FMAD/ZO's scalability in high dimensions. 

Some baselines feel incomplete: for instance, comparisons to more advanced ZO variants like Zo-SignSGD or hybrid methods (e.g., combining ZO with BP) are missing, and the vision tasks are limited to one model (Qwen 2 VL), potentially overlooking multimodal-specific challenges.

### Questions
In the theoretical bounds (Table 2), how sensitive are the convergence errors for FMAD and ZO to the choice of perturbation budget n in practice? Could you provide ablation results on varying n for a larger model like OPT-13B?

The new variance reduction techniques improve accuracy, but how do they interact with other optimizers beyond AdamW? Did you observe similar JVP spikes or instabilities?

Why focus primarily on QLORA with r=1? Would results hold for higher ranks or adapter-free fine-tuning, where parameter count d increases dramatically?

### Soundness
3

### Presentation
3

### Contribution
3
