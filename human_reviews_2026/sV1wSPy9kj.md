# Reward-Agnostic Prompt Optimization for Text-to-Image Diffusion Models

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
We investigate a general approach for improving user prompts in text-to-image (T2I) diffusion models by finding prompts that maximize a reward function specified at test-time. Although diverse reward models are used for evaluating image generation, existing automated prompt engineering methods typically target specific reward configurations. Consequently, these specialized designs exhibit suboptimal performance when applied to new prompt engineering scenarios involving different reward models. To address this limitation, we introduce RATTPO (Reward-Agnostic Test-Time Prompt Optimization), a flexible test-time optimization method applicable across various reward scenarios without modification. RATTPO iteratively searches for optimized prompts by querying large language models (LLMs) *without* requiring reward-specific task descriptions. Instead, it uses the optimization trajectory and a novel reward-aware feedback signal (termed a "hint") as context. Empirical results demonstrate the versatility of RATTPO, effectively enhancing user prompts across diverse reward setups that assess various generation aspects, such as aesthetics, general human preference, or spatial relationships between objects.
RATTPO surpasses other test-time search baselines in search efficiency, running 4.8 times faster than naive reward-agnostic test-time search baseline on average. Furthermore, with sufficient inference budget, it can achieve comparable performance to learning-based baselines that require reward-specific fine-tuning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a well-motivated approach (RATTPO) for reward-agnostic prompt optimization for text-to-image generation. It iteratively refines an initial prompt by querying LLMs at test time: one optimizer LLM proposes new prompts conditioned on the optimization history and a hint-generator LLM to provide reward-aware feedback. Empirical results demonstrate the versatility and effectiveness of PATTPO across a wide range of rewards, including human preference, text-to-image consistency, and holistic MLLM assessment. RATTPO also shows higher search efficiency compared to other test-time search baselines.

### Strengths
1. The primary strength is the reward-agnostic nature of RATTPO, which is convincingly demonstrated across different diverse reward functions.

2. The method is training-free and gradient-free and exhibits superior generalization when compared to learning-based baselines.

3. The hint is formatted as natural language feedback, making the optimization process transparent and potentially human-interpretable.

### Weaknesses
1. Lack the ablation of using single prompting loop for both prompt generation and hint generation.
2. The method is computationally demanding, requiring two LLMs (optimizer and hint generator) in an iterative loop and necessitating multiple costly image generation and reward function calls (up to 160 generated prompts) to achieve good performance.

### Questions
1. Have you encountered the reward hacking problem in your optimization framework?
2. Why do you not consider integrating both optimizer and hint-generator in a single loop?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper presents RATTPO, a reward-agnostic test-time prompt optimization method for text-to-image diffusion models. Unlike previous approaches that are tailored to specific reward functions, RATTPO can flexibly enhance prompts across various evaluation scenarios by leveraging LLMs and a novel reward-aware feedback signal. Experimental results show that RATTPO significantly improves search efficiency and prompt quality for diverse reward models. With adequate inference budget, RATTPO achieves performance comparable to specialized learning-based baselines without requiring task-specific tuning.

### Strengths
- The motivation and significance of the proposed scenario are clearly articulated and highly relevant.
- The experimental results convincingly demonstrate the superiority of the proposed method over existing approaches.

### Weaknesses
- The paper is poorly written, with an overly brief description of the methodology. It lacks essential details about the input prompts used for the first LLM to generate candidate prompts for image generation, the input prompts for the second LLM, and the specific format of the "hint" texts, all of which are critical to understanding the core approach.
- The paper lacks a clear diagram illustrating the overall workflow of the proposed method; Algorithm 1 alone is insufficient for conveying the process.
- Lines 054-057 contain two sentences that redundantly express the same idea.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
In this paper, the authors propose a new method that uses LLMs to perform automated prompt engineering for text-to-image generation. In particular, they suggest using an LLM as “prompt optimizer” and another LLM as “hint generator” to iteratively improve the generated prompts based on images generated from a text-to-image generative model and an external reward model. At every iteration, the LLM prompt optimizer will generate a prompt given the history of improvements, then the text-to-image model will generate images based on the new prompt and the reward model will output the reward w.r.t. the generated images. The LLM hint generator then produces edit suggestions for the prompt and then the LLM prompt optimizer will improve the prompts produced based on these suggestions. They also conduct experiments to show the effectiveness of their method in comparison to multiple baselines on various datasets.

### Strengths
1. The experimental results look very promising, especially in Figure 1 where they show great test time scaling.
2. The algorithm is fairly simple and easy to implement.
3. The paper is well written and easy to understand.

### Weaknesses
1. My main concern about the paper is regarding its novelty. The idea of both LLM as automated prompt generator and as a judge/hint giver has been thoroughly explored both in the context of LLM self-improvement/RLAIF [2,3,4] [(Madaan et al., 2023; Wang et al.,
2023a; Shinn et al., 2023) from the paper] and text-to-image generation [1] [(Yang et al.,
2024; Fernando et al., 2023; Du et al., 2024; He et al., 2024; Mañas et al., 2024) from the paper]. In fact, the algorithm proposed in this paper is strikingly similar to [1] and He et al. 2024. It is unclear to me the marginal changes made in this paper are significant enough.
2. The experiment results, while showing a lot of promise, do seem a bit selective and incomplete. For example, 

    (i) In Figure 1, OPT2I only shows up in one out of eight subplots, which is also the only place where this baseline is compared. Given the extreme similarity of the methodology, it would make sense for the authors to include OPT2I in all comparisons that they conduct. Similarly, somehow not all baselines are compared in all experiments.

    (ii) When comparing the inference time, the authors denote the wallclock time for their method as “Time, RATTPO at win”, which seems to indicate that they are only accounting for the cases where their method outperformed the baseline. It is very unclear why they would make this selection, i.e. why not just calculate the wallclock time for all RATTPO runs?
3. Besides the concerns above, the authors should also consider adding the following experiments to strengthen the paper:

	(i) The authors should include the comparison against [1], as it is a highly related and similar work (specifically section 6 in [1])

	(ii) The authors only use the Gemma model family in their experiment and they should consider other MLLMs like the GPT family, etc.


Reference:

[1] Liu et al. Language Models as Black-Box Optimizers for Vision-Language Models. 2024.

[2] Chao et al. Jailbreaking black box large language models in twenty queries. 2023.

[3] Wang et al. Self-Instruct: Aligning Language Models with Self-Generated Instructions. 2022.

[4] Huang et al. Large Language Models Can Self-Improve. 2022.

[5] Lee et al. RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback. 2024.

### Questions
How transferable are the prompts that are optimized for one text-to-image model to another one?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces RATTPO (Reward-Agnostic Test-Time Prompt Optimization), a novel framework for optimizing prompts in text-to-image (T2I) diffusion models without requiring reward-specific training or modifications. The method uses a dual-LLM approach:

An optimizer LLM iteratively refines prompts based on historical optimization trajectories.
A hint-generator LLM provides reward-aware feedback ("hints") derived from optimization history, replacing manual task descriptions.
RATTPO is training-free, gradient-free, and adaptable to diverse reward models (e.g., human preference, text-image alignment, multimodal LLM assessments). Experiments show it outperforms baselines in search efficiency (4.8× faster) and matches reward-specific methods with sufficient inference budget.

### Strengths
Extensive experiments across 8 reward setups, showing versatility and efficiency.

### Weaknesses
- Computational cost: Despite efficiency gains, RATTPO requires multiple image generations per iteration (line 7, Algorithm 1). Potential optimizations (e.g., caching) are unexplored.
- Prompt length constraints: The impact of initial prompt length on optimization is not analyzed.
- Novelty limited, because iteratively prompt optimization is trivial.

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
2
