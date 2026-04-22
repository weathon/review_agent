# RoboAlign: Reinforcement Learning for Action-Aligned Multimodal Large Language Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
In recent years, state-of-the-art vision–language–action models (VLAs) have been built upon pre-trained multimodal large language models (MLLMs). However, how to systematically train MLLMs to improve VLA performance remains an open problem. While prior approaches primarily focus on strengthening embodied reasoning via linguistic actions, the modality gap limits the transferability of language-based knowledge to non-linguistic low-level actions produced by VLAs. To address this problem, we propose a novel framework RoboAlign that aligns MLLM representations with low-level actions, thereby producing MLLMs well-suited for VLA. Specifically, we achieve action alignment through reinforcement learning, where the model generates action tokens via zero-shot reasoning in natural language. To validate the effectiveness of RoboAlign, we train VLAs by adding a diffusion-based action head on top of an MLLM backbone and evaluate them on major robotics benchmarks. Specifically, training base MLLMs with RoboAlign improves the performance on robotic tasks by 17.5%, 18.9%, and 106.6% on LIBERO, CALVIN, and real-world robotic environments, respectively. Moreover, RoboAlign outperforms models aligned only with language-described actions or with supervised fine-tuning based approaches such as ECoT, demonstrating its effectiveness and broad applicability.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes ROBOALIGN, a reinforcement learning–based training framework that aligns multimodal large language models (MLLMs) with low-level action generation for embodied manipulation. The method aims to strengthen the connection between high-level reasoning and executable actions while retaining the model’s general-purpose knowledge. ROBOALIGN first applies supervised fine-tuning (SFT) to let the model generate low-level actions via zero-shot reasoning, and then employs GRPO optimization to refine this process through action-accuracy rewards. This RL-based alignment encourages diverse embodied reasoning trajectories, improves reasoning-action coupling, and mitigates catastrophic forgetting compared with SFT alone.

### Strengths
1. In my view, the proposed approach effectively reduces downstream training costs (e.g., on LIBERO and CALVIN) by freezing the MLLM backbone and training only the action head. This design is, in my opinion, the most significant contribution of the paper.

2. The authors conduct both simulation and real-robot experiments, which substantially enhance the credibility and practicality of the results.

3. The idea of using GRPO-based alignment to strengthen the MLLM’s capability in low-level action generation is novel and insightful.

### Weaknesses
1. As mentioned in the strengths, the most valuable contribution of this work is its ability to reduce downstream training costs. However, the paper does not clearly emphasize or analyze this aspect. From the experimental results alone, the performance on LIBERO is still lower than many recent approaches, such as the representative work OpenVLA-OFT [1].

2. The practical impact of this contribution appears limited. In current downstream VLA training scenarios, freezing the VLA backbone provides only minor time savings, while achieving higher task success rates remains the main focus. Therefore, the proposed method may lack strong practical relevance or real-world applicability.

3. The paper provides limited details and analysis regarding the experimental setup and results. It would benefit from a more comprehensive discussion or additional analyses in the appendix to help readers better understand the experimental design and the effectiveness of the proposed method. (Suggestion, not affect rating.)

[1] Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success.

### Questions
1. During the GRPO experiments, did the authors observe genuine improvements in the model’s robotic generalization ability? Specifically, can the model trained on Bridge V2 fast-token data truly enhance its capacity to generate accurate low-level actions?

2. In the SFT stage, the authors fine-tune the MLLM on several reasoning datasets. Are these datasets truly meaningful for improving downstream embodied tasks such as LIBERO and CALVIN? I am skeptical about their practical value, as such reasoning data may have limited impact while increasing training cost. It would be helpful if the authors could provide ablation studies to validate this.

Minor Issues / Typos:

1. It is recommended that all equations include numbering for easier reference and citation.

2. In line 276, within the reward formulation, the index i should start from 1 rather than 0.

### Soundness
2

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
3

### Summary
The authors present RoboAlign, a training method for VLAs that aims to address the difficulty of translating improvements in MLLM quality to low-level action generation. RoboAlign consists first of a supervised fine-tuning stage, in which the MLLM is trained to generate FAST action tokens using a curated mixture of VQA and MCQA datasets formatted specifically to encourage reasoning. Then RoboAlign performs a reinforcement learning fine-tuning stage (GRPO), in which low-level action tokens output are encouraged to match the target tokens for as long as possible in the sequence. The method is implemented on top of the Qwen2.5VL-7B-Ins MLLM and experiments are performed on Libero and Calvin to test whether RoboAlign improves over typical training. On Libero, RoboAlign is compared to two other action alignment strategies.

### Strengths
- Method design allows many VLA architectures.
- Interesting and promising performance shown on multimodal benchmarks.

### Weaknesses
- Only one MLLM backbone is used in the experiments. The results would generalize if improvements could be shown with another MLLM.
- Baselines do not include another VLA such as GR00T N1 [1] or OpenVLA [2].

[1] GR00T N1: An Open Foundation Model for Generalist Humanoid Robots. Bjorck et al., arXiv preprint arXiv:2503.14734 2025.
[2] OpenVLA: An Open-Source Vision-Language-Action Model. Kim et al., PMLR, 2025.

### Questions
- Does RoboAlign work only for VLAs that generate action tokens? Certain works such as LLaRA [1] or TraceVLA [2] output actions in image coordinate space.
l. 185 How long does sampling take? How do you compute the reward based on the response?
Table 3: What is the difference between language-only and action-only SFT?
Section 5.3 Real Robot Experiments: How do other baselines perform on these real world tasks?

[1] LLaRA: Supercharging Robot Learning Data for Vision-Language Policy. Li et al., ICLR 2025.
[2] TraceVLA: Visual Trace Prompting Enhances Spatial-Temporal Awareness for Generalist Robotic Policies. Zheng et al., CoRL 2024.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces RoboAlign, a framework designed to better align MLLM representations with low-level action generation in a VLA. The key contribution is applying GRPO, a reinforcement learning method, after the standard SFT stage to enhance policy alignment and control performance. Experiments using Qwen2.5VL-7B-Instruct demonstrate that RoboAlign improves robot rollout success across popular benchmarks like CALVIN and LIBERO, as well as in real-world robotic settings.

### Strengths
This paper is clearly presented and well-organized, which makes it easy to follow. The authors covered all the necessary preliminary knowledge to understand the paper, and the experiments cover both two well-known simulation benchmarks and real-world environments.

### Weaknesses
The core contribution of this paper is to empirically find that applying GRPO could be helpful for VLAs (or MLLMs) and further improves the performance after SFT. However, since GRPO itself is not a novel method and its application to a VLA—especially one adapted from a VLM—is relatively intuitive, the work’s conceptual novelty is somewhat limited. Nevertheless, the demonstration of its practical success in robotics is still valuable. 

The key limitation that contributes a lot to the review is that the proposed approach is evaluated only on a single VLM, Qwen2.5VL-7B-Instruct, making it unclear whether the observed improvements generalize to other base models or architectures. The authors are encouraged to validate the generality of their method by testing it with additional models; otherwise, the paper is more similar to a tech report instead of a research paper.

### Questions
N/A

### Soundness
2

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
5

### Summary
This paper presents ROBOALIGN: a system for directly aligning multimodal LLMs (MLLM) with lower-level action generation via a two phase training pipeline: 
1- It uses supervised fine-tuning (SFT) to equip the model with the initial ability to predict action tokens.
2- It uses reasoning-incentivized reinforcement learning (following other work like Deepseek-R1) improve performance and reduce problems like catastrophic forgetting.

The SFT phase is done on a data mixture that was curated from various VQA and other CoT datasets, while the RL is using offline GRPO with a subset of the SFT CoT data. FAST is used to tokenize the actions into tokens. 

The paper tests ROBOALIGN on a couple of robotic simulation benchmarks (CALVIN, LIBERO) as well as real-world robots. Showing uplift.

### Strengths
- The paper is well-presented and easy to read.

- The paper shows that ROBOALIGN training process keeps the baseline performance on multimodal benchmarks or improves it.

- The paper reports success rates on a real Franka Research 3 robot arm. This sets it apart from other work that just tests on simulations. And it shows significant uplift against the baseline Qwen model.

### Weaknesses
- Many of the observations/design decisions that the paper makes are now considered the standard for action models (i.e. reasonable action tokenization from continuous vectors to discrete tokens, choosing a reasonable data mixture and doing some kind of SFT first, then some kind of RL -potentially with CoT- as a final step). The real contribution is not in the full pipeline in my opinion since it seems to be similar to previous works. The real contributions is the combination of algorithmic choices (e.g. FAST for tokenization, GRPO for RL), and their own curated data mixture including "ROBOALIGN VQA" dataset.

- The ROBOALIGN training pipeline uses only offline RL with repeated data from the SFT. It does not use any online RL training (e.g. with gym/mujoco simulation environments or similar).

- While in some evaluation cases the improvements are significant, some evaluations show degradation (especially on long-horizon tasks)

- More ablations studies and evaluations would be interesting in my opinion and would make the contribution stronger, examples:
-- Other base models apart from Qwen-2.5 to see if these findings translate to other baselines
-- Two benchmarks are good but the work could benefit from including more benchmarks in both training and evaluation (e.g. metaworld)

### Questions
Typos:
- line 101: "of is embodied"

### Soundness
3

### Presentation
4

### Contribution
2
