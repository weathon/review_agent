# AccLoRT: Efficient Large Language Models Pretraining through Low-Rank Accumulation

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Pretraining large language models (LLMs) poses significant computational challenges, particularly due to the memory requirements that exceed the capabilities of standard GPU devices. To address these issues, we introduce a fully low-rank approach for LLMs pretraining to improve the memory efficiency. Specifically, our approach sequentially trains low-rank matrices and accumulates them into a frozen high-rank matrix until convergence. Notably, our approach enables the low-rank traning without a warm up phase with full parameter, therefore achieving memory efficiency in the entire training process. We provide a comprehensive theoretical analysis for our proposed method by establishing the upper and lower bounds for the rank of multiple matrix sums and analyzing the rank dynamics in low-rank adapters. The results show that with finite accumulation steps, the accumulated low-rank training is equivalent to full-rank training. Extensive experiments on both synthetic reduced rank regression and practical Llama models (60M to 1B parameters) validate the effectiveness of the proposed approach in pretraining, demonstrating its potential to make LLM development more accessible and efficient.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes AccLoRT (Accumulated Low-Rank Training), a memory-efficient pretraining framework for large language models (LLMs). The method sequentially trains multiple low-rank matrices and accumulates them into a frozen high-rank matrix, enabling fully low-rank training throughout the entire pretraining process without any full-parameter warm-up. The authors provide theoretical analyses on the rank bounds of matrix sums, LoRA’s rank evolution under SGD, and the asymptotic equivalence between AccLoRT and full-rank training. Extensive experiments on Llama models (60M–1B) demonstrate that AccLoRT achieves superior perplexity and memory efficiency compared to existing methods such as GaLore, ReLoRA, and SLTrain.

### Strengths
1. The paper  derives upper/lower rank bounds and theorems describing LoRA’s rank evolution and AccLoRT’s convergence, giving a clear mathematical understanding of why accumulation works.

2. Empirical evaluations span a wide range of model sizes (60M–1B) shows improvements.

3. The proposed idea is good and reasonable.

### Weaknesses
1. The paper completely omits "Fira: Can We Achieve Full-rank Training of LLMs Under Low-rank Constraint?"[1], a closely related approach that also performs full-rank pretraining under low-rank constraints. Both share nearly identical experimental settings (rank, optimizer, and dataset) and similar motivation (achieving full-rank training). Without detailed discussion in motivation, methodology, memory consumption, and experimental performance comparison, the claimed advantages of AccLoRT remain incomplete and potentially overstated.

2. Current model size in experiments is small. It would be better to see pretrain experiments on 7B models. In this experimental setting, both Fira [1], Galore [2],  APOLLO [3] all conduct the 7B experiments.

3. Illustration is poor. For example, the font size in Figure 1 and Figure 3 is too small to read for potential readers.


[1] Fira: Can We Achieve Full-rank Training of LLMs Under Low-rank Constraint?

[2] GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection

[3] APOLLO: SGD-like Memory, AdamW-level Performance

### Questions
see weaknesses

I will adjust my score according to the rebuttal.

### Soundness
3

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
The paper proposes a method to pretrain model by accumulating low rank updates. In the theory part, the paper provides upper and lower bounds for the weight matrix rank after accumulating the low rank updates and bounds for the ranks of adapter matrices $A,B$ with gradient updates. In experiment, the paper proposes to separate between the first initialization and subsequent ones after each merge, where in the first initialization $A,B$ are obtained by the truncated QR decomposition of a normal matrix and in the subsequent initializations, $A$ is 0 and $B$ is normal. The experiment shows that the proposed method has better perplexity on the pretrain task using Llama, and is competitive with other methods on fine-tune tasks.

### Strengths
- There have been several works on using accumulating LoRA, but understanding the rank evolvement during training remains not well studied. This paper provides both upper and lower bounds for this question, which I appreciate.
- The proposed method is simpler than similar existing approaches, such as ReLoRA. In particular, ReLoRA requires a warm up stage for the full model while this method doesn't.
- Empirically, the method performs better than existing approaches with memory footprint.

### Weaknesses
- First of all, the paper lacks clarity in the representation.I'm not sure I really understand the proposed method. 
  - In particular, how is the full model $W$ initialized? Is it to 0? If this is the case, it very surprising to me. 
  - Theorem 3.3 also doesn't state the necessary assumptions, such as the initialization of $A_0$, $B_0$. Its proof in the appendix is also not clearer. For example, line 1304 says the input $x$ is drawn from a discrete set referred in line 1274 - 1284 where some assumptions about this set is made, which I'm also not sure I understand.
- The algorithmic contribution of this paper to me is quite incremental. It boils down to finding a different initialization of $A,B$. The idea of merging the adapters to the base model has been explored quite a lot before.
- To this end, I'm not sure I understand where the improvements in the experiment comes from. In particular, compared with ReLoRA which basically uses the same merging idea, and even warms up the model before the low-rank update phase, why does the proposed method perform better?

### Questions
- Please see the weaknesses.
- In table 2, can the authors provide the comparison with the other methods in Table 3?
- Line 215 mentioned $W_{acc}$, Is there a difference with the weight $W$ in Algorithm 1?
- In Algorithm 1, what is the goal of the if-else statement in Line 224-227? It doesn't seem the be explained in the text.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces AccLoRT, a framework for memory-efficient pretraining of large language models (LLMs) based on sequential accumulation of low-rank matrices. The core idea is to sidestep the memory bottleneck of full-rank training by progressively training and freezing low-rank adapters. Theoretical analyses are provided, establishing upper and lower bounds for the rank of summed matrices, and elucidating rank dynamics during training. Extensive experiments on synthetic regression and Llama models from 60M to 1B parameters are presented, showing empirical benefits in memory usage and perplexity relative to prior methods. The approach is evaluated both in pretraining and fine-tuning scenarios.

### Strengths
1. The memory efficiency challenge in LLM pretraining is timely and relevant. The authors provide a compelling motivation, emphasizing the increasing computational inaccessibility of fundamental LLM research.
2. AccLoRT’s approach to accumulating low-rank matrices is carefully described, including specific details on initialization, memory trade-offs, and update mechanisms. 
3. The experiments cover both toy and large-scale settings. Figure 3 reveals the effect of initialization on loss and perplexity for Llama models, giving actionable insight into implementation choices.
4. Table 3 demonstrates AccLoRT achieving strong or comparable perplexity to full-rank and other efficient training methods (like GaLore, ReLoRA, SLTrain) across various Llama model sizes—often with meaningful memory savings (as detailed in Table 2).

### Weaknesses
1. While the related work section summarizes many recent memory-efficient fine-tuning and pretraining approaches, it overlooks several directly relevant recent advances in low-rank and model compression for LLMs.
2. While Table 2 comprehensively compares model sizes and memory/parameter usage, Table 3's comparison is focused on perplexity only. 
3. The framing and experiments focus exclusively on LLMs and linear regression; generalizability to vision models, multi-modal LLMs, or other architectures is not tested, even though low-rank techniques are often portable.

### Questions
1. Table 3 shows that AccLoRT marginally underperforms GaLore/full-rank at the 1B parameter scale, but outperforms on smaller models; do the authors attribute this to the hyperparameters, the method itself, or intrinsic limitations of low-rank accumulation as dimensionality increases?
2. Is there an avenue for automating the choice of rank $r$ and accumulation frequency $T$ (possibly dynamically during training) as opposed to hand-tuning or grid search?
3. Could the memory savings be further quantified in terms of wall-clock time, power consumption, or batch size enabled, especially on single-GPU or low-resource devices (beyond the current synthetic/LLama runs)?

### Soundness
3

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
3

### Summary
The paper introduces AccLoRT, a method for memory-efficient pre-training of LLMs with fully low-rank training. The core mechanism is a periodic accumulation and re-initialization strategy, which trains the adapters from scratch without initial pretrained weights, merges into a fixed high rank matrix, and continues with reinitialized adapters. The paper provides a theoretical basis which shows that the accumulated low-rank training is equivalent to full-rank training after a finite number of accumulation steps.

Experiments on pretraining of Llama models and finetuning of RoBERTa-base models validate the method.

### Strengths
- Despite a simple strategy, the method achieves good results across pretraining setup.
- Some discussion and experiments on the initialization of low rank matrices.
- Extensive experiments including those that help understand training progression.

### Weaknesses
- For large model pretraining, AccLoRT did not achieve significant gain over GaLore. Although the memory usage is lower at early stage, AccLoRT approaches the same memory usage level as GaLore if using the same rank. How does the extra memory at early stage benefit training?
- Would the plateau towards the end of each training cycle slows down the training in general and complicate the choice of ranks and iteration steps? Could the authors provide perplexity progression throughout training and compare with GaLore.
- The method is not efficient in fine-tuning. The performance is sometimes not as good as LoRA despite a larger adapter size (8 x num iterations). Also, the accumulation frequency varies across settings. What is the principal for choosing the rank and accumulation frequency? How sensitive the results are for these choices of hyper-parameters?

### Questions
- In table 1, for LoRA, should the total be + 4r(n + m) for both AccLoRT and LoRA?
- L450: "In the 1B parameter setting, while GaLore achieves the best perplexity of 15.64, AccLoRT maintains competitive performance at 16.61.", but in the table there's no 16.61 but 15.49. Is it a typo?
- Since ReLoRA is a very close approach, could you add more discussion for the difference in methodology that leads to the improvement. It seems that ReLoRA is only mentioned as requiring a full parameter warm up pretraining.

### Soundness
2

### Presentation
3

### Contribution
2
