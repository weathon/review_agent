# VIPO-R1: Cultivating Video Reasoning in MLLMs via Verifier-Guided Iterative Policy Optimization

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 4, 2

## Abstract
Applying Reinforcement Learning (RL) to Multimodal Large Language Models (MLLMs) shows significant promise for complex video reasoning. However, popular Reinforcement Fine-Tuning (RFT) methods, such as outcome-based Group Relative Policy Optimization (GRPO), are limited by data preparation bottlenecks (e.g., noise or high cost) and exhibit unstable improvements in the quality of long chain-of-thoughts (CoTs) and downstream performance. To address these limitations, we propose **VIPO-R1**, a **V**erifier-guided **I**terative **P**olicy **O**ptimization method designed to gradually enhance MLLMs' ability to generate long-term reasoning chains for challenging VideoQA. The core component is the Rollout-Aware Verifier, positioned between the GRPO and Direct Preference Optimization (DPO) training phases to form the GRPO-Verifier-DPO training loop. This verifier leverages small LLMs as a judge to assess the reasoning logic of rollouts, enabling the construction of high-quality contrastive data, including reflective and contextually consistent CoTs. These curated preference samples drive the efficient DPO stage (7x faster than GRPO), leading to marked improvements in reasoning chain quality, especially in terms of length and contextual consistency. This training loop benefits from GRPO's expansive search and DPO's targeted optimization. Experimental results demonstrate: 1) Faster and more effective optimization compared to standard GRPO variants, yielding superior performance; 2) Our trained models exceed the direct inference of large-scale instruction-tuned Video-LLMs, producing long and contextually consistent CoTs on diverse video reasoning tasks; and 3) Our model with one iteration outperforms powerful MLLMs (e.g., Kimi-VL) and thinking models (e.g., Video-R1), highlighting its effectiveness and stability.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes VIPO-R1, a training loop for video-reasoning MLLMs that alternates GRPO (online RL) with a rollout-aware verifier that curates contrastive CoT pairs, followed by DPO to refine the policy, then iterates. The verifier filters/labels rollouts using accuracy checks (e.g., Math-Verify or task metrics), reasoning–answer consistency checks, repetition and length checks, and builds several preference types (penalty, consistency, reflection). Compared to using GRPO alone, the loop aims to (i) stabilize optimization, (ii) lengthen CoTs without drifting, and (iii) reduce “right answer, wrong reasoning” inconsistencies. Experiments on VSI-Bench, Video-MMMU/MMMVU, TOMATO, and Video-MME report consistent gains over a Qwen2.5-VL-7B base and over strong baselines such as Video-R1/Kimi-VL-Thinking; the authors also highlight faster progress due to the DPO stage being ~7× cheaper per sample than GRPO.

### Strengths
1. Clear training recipe that combines known pieces in a useful way. The GRPO→Verifier→DPO cycle is well-motivated; the verifier’s multi-aspect filtering (accuracy, consistency, repetition, length) and construction of contrastive/reflection pairs is detailed and easy to reproduce at a high level.

2. Empirical benefits across multiple video benchmarks, with especially large gains on hard video-reasoning tasks (e.g., +7.9% over GRPO on VSI-Bench; +5.6% over Video-R1 on Video-MMMU, as reported in the text) and reductions in reasoning-answer inconsistency. Case studies qualitatively show fewer spurious steps and better self-correction.

### Weaknesses
1. Verifier dependency & potential circularity:\
Several measurements of “consistency” and sample selection rely on LLM-based verification. If the same (or closely related) verifier is also used to curate training pairs, the evaluation may inherit its biases. A stronger case would use (i) external, task-specific non-LLM checks where possible and (ii) a held-out verifier for reporting consistency.

2. Metric choice.\
The paper emphasizes response length and Acc-Cons. Length is an input-dependent proxy; a human study or task-specific rubric (e.g., step validity on math/physics cases) would better validate that longer CoTs are meaningfully better. Suggest adding human evaluation on the necessity of the lengths.

3. Suggested references.\
Authors are also encouraged to discuss and reference the following models:\
[1] Wang et al. "Video-RTS: Rethinking Reinforcement Learning and Test-Time Scaling for Efficient and Enhanced Video Reasoning", https://arxiv.org/abs/2507.06485 \
[2] Sun et al. "video-SALMONN-o1: Reasoning-enhanced Audio-visual Large Language Model", https://arxiv.org/abs/2502.11775

### Questions
1. Which verifier LLM(s) are used in practice, and are they different from the base policy? Any evidence that results hold with multiple verifiers? 

2. How sensitive are results to the consistency/length/repetition thresholds and clustering settings used in data curation? Provide a sensitivity plot.

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
3

### Summary
This paper starts from the observation that existing Reinforcement Fine-Tuning (RFT) methods like GRPO are inefficient, unstable, and struggle to generate high-quality, long chain-of-thought (CoT) reasoning.
Therefore, the authors introduce a training paradigm called VIPO, which uses a verifier to generate pair-wise preferences from rollouts of the GRPO method,  forming a GRPO-Verifier-DPO training loop.
The performance surpasses standard GRPO on several benchmarks.

### Strengths
1. The authors utilize the immediate outputs of GRPO for pair-wise preference sample generation for DPO, which is a novel idea to my knowledge.
2. The observation of the drawbacks of the current GRPO is accurate, namely, extra data preparation and inconsistency of the reasoning trace with the final output.

### Weaknesses
1. The presentation of this paper is not clear, e.g., the introduction of the verifier, and the font size in Fig.1.
2. The proposed training loop is quite complex, while the performance improvement is not surprising. For example, there are two RL frameworks (OpenRLHF & TRL) used.
3. The quality of DPO is influenced by the verifier and the rollouts, while traditional DPO methods are usually built on human-aligned, clean datasets. Therefore, the effectiveness of DPO may be limited.
4. In Tab.2 & Tab.4, the standard GRPO & RFT methods lag behind the base model. Why?

### Questions
1. I am wondering whether this method is mentioned in the previous research.
2. How is the performance of Qwen2.5-VL (thinking) in Tab.2 obtained? Does Qwen2.5-VL support thinking mode originally as a vision-language reasoning model?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes VIPO-R1. The core of its methodology is the GRPO-Verifier-DPO training loop: rollouts from the GRPO stage are filtered through a verifier and then utilized for DPO training. This training method enables VIPO-R1 to generate long-term reasoning chains for challenging VideoQA tasks.

### Strengths
- The paper introduces a novel approach by utilizing the rollouts from the GRPO process for subsequent DPO optimization, which is an interesting attempt.

### Weaknesses
My primary concern with this work is whether the GRPO-Verifier-DPO training loop is effective in a practical sense:
- In Table 3, the DPO stage provides almost no benefit to **Accuracy**, with the main improvements seen in **Consistency** and **Acc-Cons**. However, there might be an issue with these two metrics. The "consistent answers" are judged by a small LLM, and it is possible that due to differences in model capabilities, a large LLM can derive the correct answer from a given reasoning path while the small verifier LLM cannot.
- This concern seems to be reflected in the results of Table 3: while Consistency and Acc-Cons increase over multiple iterations, the accuracy remains almost unchanged. This might imply that the consistency metric is not meaningful.
- Furthermore, Table 5, which analyzes the verifier's components, does not report accuracy. This casts doubt on the actual contribution of the verifier to the model's practical reasoning capabilities.
- According to Table 4, the most substantial improvement for VIPO-R1 comes from the "Reasoning Activation" step, rather than the GRPO-Verifier-DPO loop. In fact, the accuracy even shows a slight decrease after applying DPO.

### Questions
- What is the specific LLM used in the Verifier?
- Regarding the claim in Line 339, "our approach reduces training time from 63 hours to 49 hours," where is this demonstrated? Shouldn't introducing a new DPO stage make the overall process more time-consuming?

### Soundness
3

### Presentation
2

### Contribution
2
