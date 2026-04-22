# Inpainting-Guided Policy Optimization for Diffusion Large Language Models

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 8

## Abstract
Masked diffusion large language models (dLLMs) are emerging as promising alternatives to autoregressive LLMs, offering competitive performance while supporting unique generation capabilities such as inpainting. We explore how inpainting can inform RL algorithm design for dLLMs. Aligning LLMs with reinforcement learning faces an exploration challenge: sparse reward signals and sample waste when models fail to discover correct solutions. While this inefficiency affects LLMs broadly, dLLMs offer a distinctive opportunity—their inpainting ability can guide exploration. We introduce IGPO (Inpainting Guided Policy Optimization), an RL framework that strategically inserts partial ground-truth reasoning traces during online sampling. Unlike providing full solutions, inpainting steers exploration toward promising trajectory spaces while preserving self-generated reasoning, bridging supervised fine-tuning and reinforcement learning.
We apply IGPO to group-based optimization methods such as GRPO, where exploration failures cause zero advantages and gradients. IGPO restores meaningful gradients while improving sample efficiency. We also propose supervised fine-tuning on synthetically rewritten concise traces that better align with dLLM generation patterns. With additional techniques including entropy-based filtering, our training recipe yields substantial gains across four mathematical benchmarks—GSM8K, Math500, AMC and Minerva—achieving new state-of-the-art results for full-attention masked dLLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces IGPO (inpainting guided policy optimization) for optimizing diffusion language models with RL which uses ground truth correct trajectories in order to learn in questions where all of the generations are incorrect. To do this, the algorithm resamples more trajectories for questions where all generations are wrong, but inserting text from the correct ground truth trajectory to be inpainted around. Followed by a standard GRPO update on it. This provides "hints" for the model during generation. The authors show that this increase performance in math reasoning tasks.

### Strengths
- this paper is well written
- this paper considers an important problem, both improving overall performance and utilizing ground truth responses.
- the results of this paper are encouraging for the method.

### Weaknesses
- while it does seem to work, the entropy-based gradient filtering seems like a method in order to deal with the off policy-ness of the hint responses. Being able to deal with this off-policy would be desirable instead (although harder).
- the improvement is encouraging, but it is unclear whether it comes from increasing generation number or the hints (see questions)
- My overall feeling is this seems to improve performance slightly, but there is more work to be done to show that this is the *right* way to use gold trajectories. Ie. How does it compare to distillation methods?

### Questions
- how many of these new responses get clipped? I feel like clipping would null many of these since the trajectory would be off policy?
- I believe that the fair comparison would actually be to GRPO which oversampled on incorrect trajectories. Ie. Do hint tokens actually matter?
- does `without Inpaint` in Figure 4 mean that one adds oversampled correct trajectories into the batch as well?
- does SFT on the rewritten trajectories move it closer to on policy?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes Inpainting-Guided Policy Optimization (IGPO), a reinforcement learning framework designed to enhance exploration in diffusion-based large language models under sparse reward settings. The key idea is to inject partial ground-truth reasoning traces as inpainting hints into masked regions during RL sampling, allowing the model to receive guided feedback while retaining its own generative reasoning. This strategy effectively mitigates the zero-advantage problem in group-based policy optimization methods such as GRPO, where uniformly incorrect responses yield zero gradients and inefficient learning. Experiments on mathematical reasoning datasets demonstrate the effectiveness of the proposed method.

### Strengths
The proposed use of inpainting for guided exploration in RL for dLLMs is a creative and well-motivated exploitation of architecture-specific inductive bias.

The approach is validated with thorough quantitative results on four widely recognized mathematical reasoning benchmarks,substantial gains over both prior masked dLLM methods and non-diffusion LLM baselines are reported
Ablation studies are thorough.

### Weaknesses
The formulation of IGPO objective in the sampling procedure(Eq 5)  is difficult to follow. 

What does "Advantages $A_{i}$ are computed normally" mean?

The theoretical analysis is weak, provides limited justification for the proposed inpainting mechanism.

### Questions
Could the authors provide theoretical analysis to justify why the proposed inpainting mechanism improves policy optimization performance?

How robust is the proposed trace rewriting method? Have the authors evaluated it against simpler rewriting heuristics or with different base models for inpainting?

For domains lacking high-quality reasoning traces, how does IGPO’s performance degrade or adapt under less informative supervision?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces IGPO, a reinforcement learning framework that leverages the inpainting capability of masked diffusion large language models to overcome exploration inefficiency in RL finetuning. Instead of relying on full solutions, IGPO injects partial ground-truth reasoning hints during generation, guiding exploration toward promising reasoning paths while trying to maintain on-policy learning. Combined with a Length-Aligned Supervised Fine-Tuning stage using concise rewritten reasoning traces and entropy-based gradient filtering for stability, IGPO achieves state-of-the-art performance on GSM8K, Math500, AMC, and Minerva, improving sample efficiency and robustness over baseline methods.

### Strengths
1. The paper is well-written and thoughtfully presented, making its core ideas clear and easy to follow.

1. The proposed algorithm elegantly exploits the unique bidirectional and inpainting capabilities of diffusion LLMs to form a guided exploration strategy that resolves the zero-advantage problem in RL finetuning.

2. It demonstrates clear empirical gains, achieving state-of-the-art results across multiple mathematical reasoning benchmarks.

### Weaknesses
1. The proposed method relies heavily on the unique characteristics of diffusion LLMs and is therefore not a general solution to the zero-advantage problem applicable to broader classes of language models.

2. Compared with standard GRPO, IGPO additionally requires the RL dataset to include ground-truth reasoning paths, which further limits its applicability and use scenarios.

3. The use of reasoning hints violates the on-policy requirement of underlying RL algorithms, potentially leading to bias in the optimization objective.

4. Using reasoning hints during the RL process may constrain the model's ability to explore freely, making the final performance dependent on the quality of the provided expert reasoning hints.

### Questions
1. IGPO assumes access to accurate ground-truth reasoning traces during RL. How robust is the method to imperfect or noisy reasoning annotations? Have the authors considered evaluating IGPO with reasoning traces of varying quality to assess its sensitivity?

2. Would it be possible to include a discussion or comparison between IGPO and more recent approaches addressing the zero-advantage problem, such as [1]?

[1] Le, Thanh-Long V., et al. "No prompt left behind: Exploiting zero-variance prompts in llm reinforcement learning via entropy-guided advantage shaping." arXiv preprint arXiv:2509.21880 (2025).

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a policy optimization for diffusion-based LLMs that bridges supervised fine-tuning and reinforcement learning. Specifically, it introduces Inpainting Guided Policy Optimization (IGPO), which leverages the unique inpainting capabilities of full-attention dLLMs by strategically injecting partial ground-truth reasoning traces into online policy optimization. With this design as well as some other techniques such as entropy-based filtering, they achieve better sample efficiency than GRPO and achieve SOTA among full-attention masked dLLMs across four mathematical reasoning benchmarks.

### Strengths
+ The paper presents the first work to utilize the unique inpainting capabilities of diffusion LLMs
for reinforcement learning, which is novel.
+ The method achieves SOTA results on four benchmarks for full-attention based dLLMs.
+ The ablation studies demonstrate insights of the proposed method, such as how self-generated inpainted traces provide a better learning signal than ground truth traces.

### Weaknesses
The proposed method is a hybrid between supervised learning (or imitation learning in the general RL field) and online reinforcement learning. The authors should compare or at least discuss other guided exploration methods in the general RL domain (especially in areas like diffusion policy) to give more insights into how this method is particularly good for dLLMs and how it might be or might not be transferable to tuning other diffusion-based models (say, video diffusion models or diffusion policies). Admittedly, the full-attention dLLMs have their unique properties.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
