# dParallel: Learnable Parallel Decoding for dLLMs

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Diffusion large language models (dLLMs) have recently drawn considerable attention within the research community as a promising alternative to autoregressive generation, offering parallel token prediction and lower inference latency. Yet, their parallel decoding potential remains largely underexplored, as existing open-source models still require nearly token-length decoding steps to ensure performance. To address this, we introduce dParallel, a simple and effective method that unlocks the inherent parallelism of dLLMs for fast sampling. We identify that the key bottleneck to parallel decoding arises from the sequential certainty convergence for masked tokens. Building on this insight, we introduce the core of our approach: certainty-forcing distillation, a novel training strategy that distills the model to follow its original sampling trajectories while enforcing it to achieve high certainty on masked tokens more rapidly and in parallel. Extensive experiments demonstrate that our method can dramatically reduce the number of decoding steps while maintaining performance. When applied to the LLaDA-8B-Instruct model, dParallel reduces decoding steps from 256 to 30 on GSM8K, achieving an 8.5× speedup without performance degradation. On the MBPP benchmark, it cuts decoding steps from 256 to 24, resulting in a 10.5× speedup while maintaining accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper aims to further enhance the parallel decoding capability of Diffusion Large Language Models (DLLMs) through training. The authors argue that the suboptimal performance of current DLLMs in parallel decoding stems from their inadequate confidence distribution, which fails to support effective parallel generation. To address this, they propose a training strategy that explicitly encourages high-confidence predictions across multiple positions. Extensive experiments demonstrate that the trained model achieves significantly improved reasoning efficiency compared to the original model.

### Strengths
1. The acceleration of parallel decoding in DLLMs is a promising and worthwhile direction for further exploration.
2. Studying parallel decoding from the perspective of confidence distribution offers a novel and interesting angle.
3. The presented experimental results demonstrate a clear and substantial acceleration effect.

### Weaknesses
1. I have significant concerns regarding the motivation of the proposed method. For example, in Figure 3, the authors’ approach is based on the assumption that the confidence distribution shown in the lower figure is superior to that in the upper one. However, since a given prefix can often be completed in multiple reasonable ways, it is natural and intuitive for the confidence to be higher near the prefix. The authors should provide further clarification and evidence to justify this assumption.

2. I also have concerns regarding the role of Equation (7). Directly optimizing Equation (5) would inherently contribute to optimizing Equation (7) as well. Therefore, the authors should provide a more convincing explanation of the specific role and necessity of Equation (7), for instance, from an optimization perspective or through an analysis of the gradient behavior.

3. Since both the training and test datasets consist of similar mathematical and coding problems, I am concerned about whether the proposed method can truly serve as a generalizable paradigm for accelerating inference through fine-tuning, or if the observed acceleration merely results from overfitting to specific training data.

### Questions
See Weaknesses

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
The paper introduces dParallel, a method to unlock the parallel decoding potential of diffusion large language models (dLLMs).
While dLLMs conceptually support parallel token prediction, existing models still require nearly token-length decoding steps to maintain quality.

To overcome the bottleneck that token confidence builds up left-to-right, they propose certainty-forcing distillation (CFD), a self-distillation technique that encourages the model to reach high certainty across masked tokens in parallel, without disrupting the original decoding trajectory.

When applied to LLaDA-8B-Instruct and Dream-7B-Instruct, dParallel dramatically reduces decoding steps (e.g., from 256 → 30 on GSM8K, ≈ 8.5× speedup) while maintaining or slightly improving accuracy.

### Strengths
1. The paper well identifies sequential certainty convergence as the fundamental bottleneck in parallel decoding is a sharp and well-motivated observation.

2. Certainty-forcing distillation is conceptually clear and easy to implement. It reuses a pre-trained dLLM as both teacher and student, requiring only modest computational cost (LoRA fine-tuning on consumer GPUs).

3. Across math and code benchmarks (GSM8K, MATH, MBPP, HumanEval), dParallel achieves 5–10× decoding acceleration with minimal or no accuracy drop.

4. The algorithm and implementation details (LoRA config, pseudocode, appendix) are precise and easy to follow and visualization of token confidence propagation (Figures 2 & 5) nicely supports the paper’s core claim about parallel convergence.

### Weaknesses
1. The “certainty-forcing” idea extends existing self-distillation and entropy-minimization concepts rather than introducing a wholly new framework. Its originality lies mainly in the application to dLLMs.

2. The method doesn’t fundamentally enhance reasoning or knowledge capability, and experiments rely on ~ 100 k self-generated math/code prompts; scaling behavior on larger, more diverse datasets or multilingual tasks remains unexplored.

3. The experiments  lack of natural-language, long-form, or open-ended generation tasks makes it unclear how general the method is beyond structured tasks.

4. While empirically convincing, there’s no formal discussion of how certainty dynamics relate to convergence or stability of diffusion decoding, which could also benefit to reveal why easoning or knowledge capability is not enhanced. 

5. Recent works like Block Diffusion (Arriola et al. 2025) or Dimple (Yu et al. 2025b), which also target parallel decoding, are cited but not empirically compared.

### Questions
1. How does certainty-forcing behave on non-mathematical text (e.g., story generation or instruction following)?

2. Does parallel certainty hold for open-ended or ambiguous tokens?

3. The paper shows that confidence correlates with correctness, but could the model become over-confident and harm diversity?

4. How robust are results to the choice of β (balance weight) or temperature T in the entropy loss?

5. Is there a theoretical or empirical limit to how few decoding steps dLLMs can reach before degradation (e.g., 5–10 steps)?

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
2

### Summary
The paper introduces dParallel to address sequential certainty convergence in Diffusion LLMs. dParallel minimizes the entropy of top tokens to boost confidence in the model’s predictions, and makes inference semi-autoregressive by breaking down parallel decoding into blocks wherein tokens within a block are predicted in parallel, and blocks themselves follow autoregressive decoding. dParallel significantly reduces the latency in comparison to several recent works without taking a hit on accuracy.

### Strengths
* The latency reductions obtained by dParallel are substantial.
* The paper provides clear ablations on various components of the training algorithm.

### Weaknesses
* dParallel uses self distillation to train the LoRA parameters which requires running a large number of forward passes through the base model making training computationally expensive.

### Questions
* How would this fit with speculative decoding with a large autoregressive model (sharing the same vocabulary) since dParallel seems to satisfy all the requirements for a drafter.
* What would the final diffusion model look like if it was trained from scratch with dParallel’s objective (without the distillation)?
* How does LoRA compare to full finetuning for dParallel?

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
This paper proposes dParallel, a learnable method to accelerate dLLMs by reducing the number of required decoding steps. The authors first identify a key bottleneck to parallel generation: the "sequential certainty convergence," where the model's confidence in its predictions propagates from left-to-right rather than emerging in parallel. To address this, they introduce "certainty-forcing distillation," a training strategy that fine-tunes a pre-trained dLLM using a new objective. This objective combines a standard consistency loss (to follow the original model's generation trajectories) with a certainty-forcing loss (an entropy minimization term) that encourages the model to become highly confident in its correct predictions more rapidly. The method is evaluated on top of two SOTA academia-setting models LLaDA and Dream models, shwoing significant reductions in decoding steps (e.g., 8.5x on GSM8K) while maintaining the original model's accuracy.

However, I am recommending a borderline score at this time due to two major reservations that temper the paper's claims and require clarification:
1.  **Conflation of Training and Inference Strategies:** The experimental setup makes it difficult to isolate the true benefit of the proposed *training* method. The dParallel-trained model is evaluated with a specific entropy-based sampler, while the baselines (which are untrained) use different samplers. The gains may be attributable to the superiority of the entropy sampler itself, rather than the certainty-forcing distillation. (I could be wrong but feel free to correct me on this authors)

2.  **Concerns About Specialized Fine-Tuning:** The distillation process uses a relatively small (100k samples) dataset generated from specific domains (e.g., math and code). This raises the concern that dParallel is effectively a form of specialized fine-tuning, and its impressive performance might not generalize to broader, out-of-domain tasks.

I am open to increasing my score if the authors can provide clean explanations or cleanly ablate the effects of their training method and demonstrate its generalizability beyond the distillation domains or have discussions on addressing this weakness too.

### Strengths
1. The identification and visualization of "sequential certainty convergence" is a clear and valuable contribution that helps formalize why parallel decoding in dLLMs is challenging.

2. The certainty-forcing loss, which minimizes entropy only on correctly predicted tokens, is an elegant way(IMO) to encourage higher confidence without penalizing the model during its exploration phase.

3. The reported speedups (8.5x-10.5x) with little to no performance drop on challenging reasoning benchmarks are substantial and demonstrate the high potential of the approach.

### Weaknesses
- **Experimental Design Conflates Variables:** This is the most critical weakness. The paper compares `(dParallel-trained model + entropy sampler)` against `(original model + other samplers)`. To truly validate the contribution of the *certainty-forcing distillation*, the essential baseline is missing: `(original model + entropy sampler)`. Without this comparison, it is impossible to disentangle the gains from the training method versus the gains from the inference-time sampler. It is plausible that the entropy sampler is simply a better fit for dLLMs, and the training method provides only a marginal extra benefit.

- **Overfitting danger**: The distillation data is sourced from math and code problems. While the main evaluation is on math and code benchmarks, this raises a serious question about generalization. Does this specialized training harm the model's performance on general-purpose language tasks like commonsense reasoning (e.g., MMLU, HellaSwag) or instruction following? The lack of evaluation on these standard benchmarks is a major omission and leaves the true "cost" of this training unclear.

-   **Insufficient Ablation of the Method:** The paper ablates the training objective components but provides limited analysis on key design choices. For example, the certainty loss is applied only to tokens the model already predicts correctly. What percentage of tokens does this typically affect during training? How does this dynamic change over the course of training? Furthermore, the impact of the temperature `T` and the loss weight `β` are not explored, making it hard to assess the method's sensitivity. I understand that one paper might not cover everything but it is worth including some sort discussions and inform the future researchers who might be using this method.

### Questions
1.  To isolate the effect of your proposed training method, could you provide results for the original, untrained LLaDA model using the *same entropy-threshold sampler* you use for dParallel? This is a crucial experiment to demonstrate that the gains are primarily from certainty-forcing distillation and not just the choice of sampler.

2.  Have you evaluated dParallel on broader, general-purpose benchmarks (e.g., MMLU, ARC, HellaSwag) to confirm that the specialized distillation on math/code data does not cause performance regression on out-of-domain tasks?

3.  Regarding the certainty-forcing loss (Eq. 7), what is the typical size of the set `Mc` (correctly predicted tokens) relative to `Ma` (all masked tokens) during training, which kinda of distribution does it follow w.r.t. `t` ? I might miss some details feel free to correct me? Does this ratio change significantly, and how does it affect the stability and effectiveness of the learning signal?

4.  The paper argues that consistency distillation alone is insufficient. However, in Table 1, the performance drop for "Consistency Distillation" is quite severe (e.g., 75.7% -> 69.9% on GSM8K). This seems worse than simple few-step decoding. Could you clarify why this baseline performs so poorly? Does this suggest a potential issue with the self-distillation setup itself?

Minor questions:
- some discussion of the semi-autoregressive masking strategy during training would be beneficial. Why was a block size of 32 chosen for LLaDA but 256 for Dream? How does this choice interact with the goal of promoting parallel certainty?

### Soundness
3

### Presentation
3

### Contribution
3
