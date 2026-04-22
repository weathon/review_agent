# GigaVideo-1: Advancing Video Generation via Automatic Feedback with 4 GPU-Hours Fine-Tuning

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 2, 4, 8

## Abstract
Recent advances in diffusion models have significantly improved the quality of video generation. However, their real-world deployment often requires fine-tuning for physical constraints, a process that depends on human annotations and large-scale computational resources.
In this paper, we propose GigaVideo-1, an efficient fine-tuning framework that advances video generation without additional human supervision. Rather than injecting large volumes of high-quality data from external sources, GigaVideo-1 unlocks the latent potential of pre-trained video diffusion models through automatic feedback. 
GigaVideo-1 focuses on two key aspects: data and optimization. On the data side, we design a prompt-driven data engine that constructs diverse, weakness-oriented training samples. On the optimization side, we introduce a reward-guided training strategy, which adaptively weights samples using feedback from pre-trained vision-language models with a realism constraint.  
GigaVideo-1 offers a flexible optimization framework adaptable to various capability dimensions. To demonstrate its versatility, we instantiate the framework on VBench-2.0's 17 evaluation dimensions as concrete application instances. Using Wan2.1 as the baseline, GigaVideo-1 yields consistent improvements, with an average gain of $\sim$4\% using only 4 GPU-hours. Requiring no manual annotations and minimal real data, GigaVideo-1 shows both effectiveness and efficiency. Code, model, and data will be publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents an algorithm for the rapid scaling of video diffusion models, reporting up to a 4% performance gain with only 4 GPU hours. The approach involves two main components: a prompt-driven data engine to generate diverse synthetic training data and a VLM-based scoring mechanism to weight this data during training.

However, I am puzzled by the formulation of the proposed KL divergence constraint. The authors frame the objective as minimizing the distance between $u(z_t,p,t;\theta)+z_0$ and $z_1$. This is mathematically equivalent to minimizing the distance between the model's direct output, $u(z_t,p,t;\theta)$ and the target $z_1-z_0$. Given this equivalence, the decision to add $z_0$ to the model's prediction before computing the loss appears superfluous.

### Strengths
1. High Novelty: The core idea of actively identifying, synthesizing, and then reweighting challenging videos is highly innovative. This "target-seeking" optimization strategy, which explicitly pushes the model beyond its current capabilities, is a new direction for training diffusion models. To the best of my knowledge, no prior work has taken a similar approach.

2. Thorough Experimental Validation: The experiments are comprehensive and well-designed. The ablation studies are particularly insightful, systematically justifying the components of the proposed method. The authors have diligently tested their reweighting strategy across various training paradigms (including SFT, RL, and gradient-based methods), concluding that the offline reweighting approach is most effective.

3. Strong Empirical Results: The method achieves impressive results. Notably, the fact that this approach outperforms a strong baseline like Flow-GRPO provides compelling evidence for its effectiveness and practical value.

### Weaknesses
1. Request for Intuition on Method's Effectiveness: I find the core mechanism of the paper's success to be somewhat perplexing, despite the strong empirical results. The methodology involves training on videos synthesized from challenging prompts. Intuitively, videos generated for highly challenging prompts would be of lower quality, receive a lower VLM score, and therefore contribute less to the training objective due to the reweighting. It is therefore surprising that this strategy yields such a significant (nearly 5-point) improvement over standard SFT. Could the authors provide a more detailed explanation or intuition for why this approach is so effective?

2. Clarification on Data Blending in Table 2: Regarding the experiments in Table 2, when the datasets PsVs, PrVs, and PrVr are used concurrently, does this mean they are simply blended together for training? If so, what is the anticipated effect of also incorporating the PsVr dataset into this mixture?

3. Suggestion for Showcasing Qualitative Results: The static figures presented in the paper are not sufficient to fully demonstrate the superiority of the proposed method's video generation quality. While the supplementary videos are helpful, I would strongly suggest that the authors create an anonymous GitHub Pages site or a similar web-based platform. This would allow reviewers to more easily and directly compare the results and appreciate the qualitative improvements

### Questions
No

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes an automated pipeline for fine-tuning video diffusion models. Its core idea is a variant of DPO, first use LLM to generate polished prompts, then use the polished prompts to generate videos to performe DPO. During the fine-tuning, use the VLM to score the specific generated video and performe RL. 

Its core concept is a straightforward combination of existing ideas, lacking genuine conceptual innovation or technical breakthrough. It resembles more of a robust engineering frame work than a pioneering research method. I think the authors need to clarify the contributions and significance of this paper, and provide more insights behind this method.

### Strengths
This paper is complete, constructing an end-to-end automated fine-tuning system that includes all stages from data generation and evaluation to optimization. The experiment results on VBench 2.0 seems good.

### Weaknesses
1. Lack of core innovation. This is the most critical flaw, and the authors need to clarify their contributions and significances. From my perspective, the method proposed in this paper is essentially "Automated DPO/RWR". The entire pipeline can be summarized as: LLM generates targeted prompts -> base model generate the videos -> MLLM scores -> score-weighted loss training. Every module in this paradigm is off-the-shelf, and the combination method is straight forward.

2. Lack of in-depth experiments. All experiments in this paper are conducted in VBench 2.0. The pipeline heavily relies on the dimensions defined by VBench 2.0. Does this suggest limited generalizability of the framework?

3. The paper claims to address deep semantic issues like "physical consistency". However, the MLLM itself, trained on large-scale web data, possesses "physical knowledge" and "common sense" that are similarly superficial and biased. Using a biased judge to correct a biased generative model might only be optimizing a model consensus rather than truly approximating physical consistency.

### Questions
No.

### Soundness
3

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
3

### Summary
This paper introduces GigaVideo-1, a lightweight and data-efficient fine-tuning framework designed to enhance video diffusion models (e.g., Wan2.1-T2V-1.3B). The core of the paper consists of two modules: the Prompt-Driven Data Engine and the Reward-Guided Optimization module. According to the paper, their method can automatically improve the performance of video generation models on challenging dimensions (such as physical dimensions like thermodynamics) while requiring minimal computational resources (4 GPU hours).

### Strengths
1. The paper is presented with clarity and is easy to understand. The contributions are clearly articulated (1. the data engine; 2. the optimization method).

2. The framework demonstrates strong generalizability. As validated in the appendix, it brings consistent performance gains when applied to various video model backbones (e.g., CogVideoX, HunyuanVideo), proving it is a versatile and portable solution rather than a model-specific trick.

3. A major contribution of this paper lies in its well-chosen research issue — addressing the poor performance of general video models in physical dimensions — and its proposed automated mechanism for rapid targeted fine-tuning, which has achieved substantial performance gains.

### Weaknesses
1. I'm curious about the statement on **line 314**: "the synthetic dataset is generated by different pre-trained T2V models." Specifically, which T2V models were used for this purpose? I mean, if you're fine-tuning a Wan2.1-1.3B model but the synthetic data is generated using Wan2.1-14B, wouldn't the time required for synthetic data generation be excessively long?

2. Another concern centers on the unvalidated effectiveness of the MLLM-based evaluation. Without a correlation analysis between MLLM scores and human judgment, the entire optimization process risks "reward hacking"—improving automated metrics at the cost of human perception, potentially resulting in high-scoring but perceptually poor videos. 

3. While the paper avoids the massive human annotation required by methods like DPO, it introduces a different bottleneck: the manual selection of seed prompts to generate videos exhibiting specific flawed dimensions.  You must first manually identify the model's weak dimensions and then design corresponding seed prompts, which is inherently subjective.

### Questions
1. The entire optimization process relies on the MLLM providing accurate and meaningful scores. Could the authors provide a correlation analysis between the MLLM's dimension-specific scores and human subjective judgments?

2. The fine-tuning is highly targeted on specific weak dimensions. Did the authors evaluate whether this targeted improvement comes at the cost of performance on other, non-optimized dimensions? For example, after fine-tuning on "Human Interaction," did the model's performance on "Aesthetic Quality" or "Background Consistency" degrade?

3. Could an automated system use a large set of diverse prompts, generate videos, and use the MLLM's own failure signals (low scores on certain prompt types) to automatically cluster and identify new weak dimensions without human pre-definition?

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
The paper presents GigaVideo-1, a lightweight post-training framework that boosts pre-trained text-to-video diffusion models without any human annotations or extra real videos. Key technical components are Prompt-Driven Data Engine and an LLM-based generator. With only 4 GPU-hours of full-parameter fine-tuning on Wan2.1-1.3B, the system raises the average VBench-2.0 score by ~4 percentage points, outperforming several larger-scale competitors. Extensive ablations, user studies, and cross-backbone transfers are provided.

### Strengths
- 4 GPU-hours is orders of magnitude cheaper than prior SFT/RL works, making the method attractive for practitioners.
- The prompt engine explicitly amplifies failure modes, leading to a stronger training signal than random web videos.
- 17 dimensions, 5 strong baselines, user study, ablation of data source & reward strategy, and tests on four different architectures (2B–13B).

### Weaknesses
- Sec. 4.3 shows that mixing synthetic prompts with synthetic videos ($P_sV_s$+$P_rV_s$) actually hurts accuracy, hinting that some LLM-generated captions are too exotic and push the model away from realism.
- How do you filter or validate the LLM-generated captions to prevent physically impossible or nonsensical queries (e.g., “a person with three elbows”)?  Could such cases bias the model toward hallucination?
- Have you tried a single, unified reward model (e.g., training a small diffusion critic on MLLM pseudo-labels) instead of switching between MLLM and specialist models?
- What is the expected GPU-hour scaling for larger models?  Does the cost grow linearly with parameter count, or does the targeted small-data regime keep it sub-linear?
- In joint training, did you explore dynamic loss-balancing techniques (grad-norm, uncertainty weighting, or Pareto optimisation) to mitigate the observed interference between dimensions?
- Some related work discussed is helpful for improving manuscript quality, like InstructVideo and Lumos-1 etc.

[1] InstructVideo: Instructing Video Diffusion Models with Human Feedback, CVPR.

[2] Lumos-1: On autoregressive video generation from a unified model perspective, Arxiv.

### Questions
Please see WEAKNESS.

### Soundness
3

### Presentation
3

### Contribution
3
