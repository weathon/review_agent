# A Free Lunch in LLM Compression: Revisiting Retraining after Pruning

- Decision: Reject
- Scores: 2, 6, 2, 2

## Abstract
While Neural Network pruning typically requires retraining the model to recover pruning-induced performance degradation, state-of-the-art Large Language Model (LLM) pruning methods instead solve a layer-wise mask selection and reconstruction problem on a small set of calibration data to avoid full retraining, as it is considered computationally infeasible for LLMs. Reconstructing single matrices
in isolation has favorable properties, such as convexity of the objective and significantly reduced memory requirements compared to full retraining. In practice, however, reconstruction is often implemented at coarser granularities, e.g., reconstructing a whole transformer block against its dense activations instead of a single matrix. In this work, we study the key design choices when reconstructing or retraining the remaining weights after pruning. We conduct an extensive computational study on state-of-the-art GPT architectures, and report several surprising findings that challenge common intuitions about retraining after pruning. In particular, we observe a free lunch scenario: reconstructing attention and MLP components separately within each transformer achieves the same model performance as more coarse-grained reconstructions while being the most resource-efficient. Most importantly, this Pareto-optimal setup achieves better performance than full retraining, despite requiring only a
fraction of the memory. Furthermore, we demonstrate that simple and efficient pruning criteria such as Wanda can outperform much more complex approaches when the reconstruction step is properly executed, highlighting its importance. Our findings challenge the narrative that retraining should be avoided at all costs and provide important insights into post-pruning performance recovery for LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper studies the performance of retraining after pruning under different setups and tries to explore the optimal setup for achieving the best performance. The paper provides experimental evidence that simple pruning methods like Wanda can achieve even better performance than other more complex pruning methods when combined with a good retraining strategy.

### Strengths
The retraining problem after pruning is worth studying. The paper considers different cases for retraining across various inputs, targets, and pipeline setups, and shows experimental results comparing the performance under different setups.

### Weaknesses
- My main concerns lie in the presentation of this paper. I don’t mean that the paper contains a lot of mistakes, but the overall presentation—such as the definition of equations and terms—makes it quite hard to understand. See the Questions section for some examples.
- I’m also concerned about the contribution of this paper. Most of the key statements are already well-known from previous work. For example, the paper shows that per-matrix reconstruction consistently underperforms. However, this is already a well-known result and, in fact, per-matrix reconstruction is not a commonly adopted formulation in practice.
- I’m wondering about the rationale for combining losses at different levels. Would performance improve if we selectively used certain losses instead? For example, the paper combines per-matrix reconstruction loss, attention component loss, and MLP loss—similar to what was explored in [1] by NVIDIA, which also investigated retraining strategies for pruned models and gave much convincing conclusions.
- The experiments mainly focus on perplexity. However, perplexity alone cannot fully represent a model’s performance after retraining. It’s expected that retraining on next-word prediction will lead to improvements on that same task. What really matters is the model’s generalization ability on downstream tasks. Unfortunately, I did not see sufficient experimental coverage in this regard.

Overall, I think the presentation, contributions, and experimental setup need to be improved before the paper is ready for acceptance.

[1] Compact Language Models via Pruning and Knowledge Distillation

### Questions
- Line 189: It’s hard for me to capture the idea quickly. Some matrices are bolded and others are not. Also, in Line 195, should the X actually be \hat{X}?
- Line 220: Terms like “block size one-half” are hard for the reader to understand at a glance.
- Later in the paper, there are many terms like block size 1/2, 1, 12, which are also difficult to parse quickly. It’s unclear to the reader what exactly these refer to, making it harder to follow the main ideas.
- The differences in the results are quite limited and not sufficient to support strong conclusions. For example, in Table 3, the perplexity differences between Wanda and SparseGPT are mostly within 0.1. It’s difficult to justify the claim of “outperformance” with such a small margin. If you change the seed for calibration data selection or adjust the batch size, the conclusion could easily be reversed.

### Soundness
2

### Presentation
2

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
This paper re-examines the post-pruning reconstruction process in large language models by systematically investigating design choices including propagation strategies, loss functions, and reconstruction granularities. The study reveals several findings that challenge conventional wisdom, particularly that reconstructing attention and MLP components separately within each transformer block (block size 1/2) achieves a favorable balance between resource efficiency and model performance. This approach not only requires less memory but also demonstrates competitive or superior performance in perplexity and zero-shot accuracy compared to full model retraining in certain scenarios.

### Strengths
1.Comprehensive research design covering comparisons of multiple propagation strategies, loss functions, and reconstruction granularities, providing systematic experimental analysis.

2.Extensive experimental scope across multiple mainstream models (OPT-1.3B, OPT-6.7B, LLaMA-2-13B, LLaMA-3-8B), enhancing the generalizability of findings.

3.Identification of block size 1/2 reconstruction as a favorable balance point between resource efficiency and performance, offering practical guidance for applications.

### Weaknesses
1.When the reconstruction granularity is small, more iterative training is required, so comparing only peak memory usage is unfair; training time comparisons should also be included.

2.This paper demonstrates through detailed experiments that a block size of 1/2 yields the best results, but it lacks certain theoretical explanations.

### Questions
1.Could the authors provide some insights to explain why a block size of 1/2 achieves optimal performance? This would strengthen the paper's contributions and provide deeper understanding.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper conducts an extensive study across multiple LLMs, exploring how different reconstruction settings affect post-pruning recovery. By isolating factors such as propagation strategy, loss function, and reconstruction granularity, the work provides valuable empirical insights into how local reconstruction actually works in practice. Demonstrating that simple pruning methods like Wanda can outperform more complex schemes when coupled with proper reconstruction offers a practical contribution.

### Strengths
A thorough empirical study investigates the impact of various reconstruction configurations on post-pruning performance recovery. I appreciate the paper’s well-defined experiments and its simple yet effective findings. My main concern lies in the level of novelty, though I believe this point would be best discussed among reviewers and potential readers.

### Weaknesses
- Although the paper provides strong empirical evidence, its experimental novelty is somewhat limited. The work mainly explores reconstruction granularity variations rather than introducing a new methodological direction.
- The pruning methods considered feel narrow in scope (only unstructured and semi-structured pruning). Including structured pruning experiments would strengthen the paper’s generality and practical relevance.
- The statement "Reconstructing at block size 1/2 is most effective." (page 6) appears slightly overstated. Although this trend holds for OPT models, the LLaMA series exhibits comparable or even different optimal settings, as already noted in the paper.
- The calibration data appear to be limited to sequences sampled from C4. It would be valuable to analyze how varying the type of calibration data (e.g., using a different dataset or an instruction-tuning corpus) affects reconstruction performance.
- When we refer to retraining after pruning for performance recovery, we usually mean retraining the model on a next-token prediction task using logits with a sufficient amount of data. However, I think the term full retraining in this paper does not correspond to that setting. Moreover, it is difficult to find a clear and direct comparison between the proposed reconstruction scheme and the kind of fully retrained model that we usually mean by retraining.
- It would be valuable to include a comparison or discussion of other regression-based performance recovery or compensation methods in LLM pruning.
  * SlimLLM: Accurate Structured Pruning for Large Language Models ( https://arxiv.org/abs/2505.22689 )
  * Olica: Efficient Structured Pruning of Large Language Models without Retraining ( https://arxiv.org/abs/2506.08436 )
  * Fluctuation-based Adaptive Structured Pruning for Large Language Models ( https://arxiv.org/abs/2312.11983 )

### Questions
Please refer to the Weakness part.

### Soundness
2

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
The paper studies “partial training” for LLMs where only selected components are optimized while the rest are reconstructed or kept fixed. It claims that partial training outperforms full-parameter training with lower memory and compute, and shows results on small to mid-size models and standard language modeling metrics.

### Strengths
- Reports an observation that partial training can beat full-parameter training on some setups. This is interesting and, if robust, practically useful.
- Simple pipeline. Minimal code changes in principle.

### Weaknesses
- **Model scale and diversity.** Experiments are limited to small or mid-size models. No validation on modern ≥13B–70B models. Generality is unproven.
- **Missing LoRA and strong PEFT baselines.** No comparison to LoRA/QLoRA, adapters, prefix/prompt tuning, or modern pruning/distillation. The central claim “better than full training” must also be “non-trivially competitive with PEFT,” which is not shown.
- **Hyperparameter opacity.** Training schedules, LR ranges, warmup, weight decay, gradient clipping, batch sizes, number of steps/epochs, and early-stopping criteria are under-specified. Reproducibility is weak.
- **Dataset coverage.** Only a narrow calibration/training distribution is used. No dataset ablation. Distribution shift is not analyzed.
- **Training dynamics absent.** No loss/valid curves, gradient norms, or representation-shift diagnostics to justify the claim.
- **Compute and memory accounting.** Comparisons focus on peak memory during the proposed procedure, not end-to-end FLOPs, wall-clock, or activation I/O when reconstructing with the dense teacher. This weakens “cheaper than full training.”
- **Seeds and variance.** Several tables, i.e., Table 1-3, appear single-seed. No mean±std. Claim strength is overstated.
- **Over-claiming.** Abstract and conclusions generalize beyond the tested setting. Baseline tuning appears narrow and may be under-optimized, yet conclusions are broad.

### Questions
1. More experiments as noted above: larger models, more datasets, PEFT baselines, multi-seed with mean±std, full compute/memory accounting.
2. Please report training dynamics: train/valid loss over steps, PPL vs steps, gradient norms, layerwise updates, and parameter movement norms.
3. Provide mathematical intuition: what objective is the partial training implicitly optimizing relative to full training? Is there a projection, subspace, or spectral alignment view that explains when partial training wins?
4. How are components selected for optimization vs reconstruction? Any criterion beyond engineering convenience (e.g., sensitivity, curvature, Fisher, activation energy)?
5. How many steps and samples until partial training surpasses full training? Is the effect transient or stable after long training?

### Soundness
2

### Presentation
2

### Contribution
1
