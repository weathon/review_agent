# AceReason-Nemotron 1.1: Advancing Math and Code Reasoning through SFT and RL Synergy

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 4, 6, 8, 8

## Abstract
In this work, we investigate the synergy between supervised fine-tuning (SFT) and reinforcement learning (RL) in developing strong reasoning models. We begin by curating the SFT training data through two scaling strategies: increasing the number of collected prompts and the number of generated responses per prompt. Both approaches yield notable improvements in reasoning performance, with scaling the number of prompts resulting in more substantial gains. We then explore the following questions regarding the synergy between SFT and RL:
(i) Does a stronger SFT model consistently lead to better final performance after large-scale RL training?
(ii) How can we determine an appropriate sampling temperature during RL training to effectively balance exploration and exploitation for a given SFT initialization?  
Our findings suggest that (i) holds true, provided effective RL training is conducted, particularly when the sampling temperature is carefully chosen to maintain the temperature-adjusted entropy around 0.3, a setting that strikes a good balance between exploration and exploitation. Notably, the performance gap between initial SFT models narrows significantly throughout the RL process. Built on a strong SFT foundation and SFT–RL synergy, our AceReason-Nemotron-1.1 7B model significantly outperforms AceReason-Nemotron-1.0 and achieves new state-of-the-art performance among Qwen2.5-7B-based reasoning models on challenging math and code benchmarks, thereby demonstrating the effectiveness of our post-training recipe.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies the synergy between supervised fine-tuning and reinforcement learning in math and coding domain and and introduces AceReason-Nemotron-1.1-7B model that achieves high performance on math and coding problem. More specifically, they study the effects of scaling SFT data by increasing both prompt diversity and the number of responses per prompt. And they then apply a staged RL curriculum with increasing context length, analyzing how SFT strength, sampling temperature, and overlong response handling affect learning.

### Strengths
1. The paper collects a large amount of SFT and RL data and conducts thorough preprocessing. In addition, it performs extensive SFT and RL experiments along with many ablation studies, and evaluates the model across comprehensive math and coding benchmarks. Overall, the experiments are fairly thorough.


2. The final model achieves leading performance on multiple metrics compared to models of similar size, demonstrating the effectiveness of the proposed approach.

### Weaknesses
1. Although this paper features a very large workload and extensive experiments, I believe the generalization of some of its findings and conclusions is questionable. Specifically, all experiments are conducted on Qwen2.5-Math-7B and focus on math and coding generation tasks. From the model perspective, multiple works [1][2] have already pointed out the uniqueness of Qwen2.5-Math-7B, so the findings of this paper (e.g., sampling temperature choices) may not transfer to other models. From the task perspective, the experiments are primarily on math and coding. Would these conclusions still hold for more challenging tasks such as web agents or SWE-style agent tasks?

2. More broadly, the findings feel more like a report on tuning methods and hyperparameters for a particular model and task type, rather than offering deeper empirical and theoretical insights that generalize to different settings in the community. Therefore, I hope the authors can clearly distinguish which conclusions are generalizable and which only apply to the specific setup studied in the paper.

3. In the RL training phase, the authors divide the process into many stages, but no clear justification is provided. In Line 196, the authors cite AceReason-Nemotron-1.0 and follow a similar setup, yet do not explain the rationale behind these design choices in more detail.

[1] Zeng, Weihao, et al. "Simplerl-zoo: Investigating and taming zero reinforcement learning for open base models in the wild." arXiv preprint arXiv:2503.18892 (2025). 

[2] Shao, Rulin, et al. "Spurious rewards: Rethinking training signals in rlvr." arXiv preprint arXiv:2506.10947 (2025).

### Questions
I have raised most of my concerns in the Weaknesses section, and I would like the authors to address them. Below are a few additional questions:

1. In Lines 396–398, the authors suggest keeping the entropy around 0.3. However, according to Figure 5, the entropy later deviates significantly from 0.3 and approaches 0.4. Could the authors clarify why this happens, and whether this suggestion is universal across settings (e.g. models, datasets)?

2. The paper conducts math and code training in separate stages. Have the authors considered joint training that mixes math and code signals instead of a staged approach?

### Soundness
2

### Presentation
3

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
This work presents a comprehensive study of SFT and RL training on a 7B reasoning model. The authors investigate different aspects, including the impact of SFT model on RL performance, the sampling temperature and overlong penalty during RL training. The whole RL training recipe starts from math training to code training, with a short-to-long generation length. The resulted model achieves competitive results on both math and code benchmarks.

### Strengths
1. The paper presents a comprehensive SFT&RL recipe for a 7B reasoning models.
2. Ablation studies from different aspects are conducted, making this work as a solid technical work and providing valuable empirical insights.
3. The resulted model achieves competitive performance on both math and code benchmarks.

### Weaknesses
1. The question "Is theMath-onlyStage-1(8K) trulynecessary" is not answered in the submission.

### Questions
1. Why does the authors choose to apply RL training on math and code in seperate stages, instead of apply RL training on math and code data simultaneously?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper conducts a systematic analysis of the alternating interplay and integration between SFT (Supervised Fine-Tuning) and RL (Reinforcement Learning) training. It explores how expanding the SFT dataset improves initialization — including the number of prompts, the number of responses per prompt, and the number of training epochs. The authors also investigate the impact of using different SFT models for initializing RL, as well as how varying the temperature during RL training affects entropy and accuracy. In addition, they examine whether filtering overlong samples during RL affects the outcomes. Through detailed experiments, the authors study all these questions and demonstrate that SFT based on Qwen2.5-7B surpasses DeepSeek-R1-Distill-Qwen in performance. Furthermore, they confirm that stage-wise RL can achieve further improvements on stronger SFT models.

### Strengths
1. Compared with most technical reports that provide only brief descriptions of SFT and RL details, this paper conducts a systematic study on the interplay and integration between SFT and RL using extensive resources. This work is highly significant for understanding the training dynamics of RL initialized from SFT in the research community.

2. The authors analyze SFT data in detail — including the number of prompts, the number of responses per prompt, and their effects on SFT performance — across both Math and Code domains. They also study how different SFT models influence RL training dynamics, offering valuable insights. For example, scaling the number of prompts yields more significant improvements, and stronger SFT models lead to higher RL starting points, though the performance gap narrows as RL training progresses.

3. The authors analyze how different temperatures in RL affect entropy and performance, experimentally illustrating healthy entropy dynamics and corresponding temperature settings during RL training.

4. They further study the impact of overlong sample filtering, providing detailed analyses under various maximum-length settings.

### Weaknesses
The experiments are mainly conducted on Qwen2.5-based 7B models. It is unclear whether the authors tested larger models, such as 32B, to verify the generalizability of their conclusions. However, given the large-scale data and the computational resources required, the absence of such experiments is understandable.

### Questions
See in Weakness

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper studies how supervised fine-tuning (SFT) and reinforcement learning (RL) interact to improve long chain-of-thought reasoning for math and code. It scales SFT along two axes with more unique prompts and more responses per promptand analyzes epoch effects. It then applies a stage-wise, strictly on-policy GRPO RL curriculum (math stages at 8K -> 16K-> 24K-> 32K tokens, interleaved with code RL at 24K and 32K) and examines starting-point sensitivity, sampling temperature, and handling of over-length trajectories. The resulting 7B model, AceReason-Nemotron-1.1, built on Qwen2.5-Math-7B, reports improved results on AIME24/25, MATH500, HMMT2025, BRUMO2025, EvalPlus, and LiveCodeBench v5/v6, with analyses of temperature-adjusted entropy and overlong filtering.

### Strengths
- This is a well-executed empirical paper that systematically probes how supervised fine-tuning and reinforcement learning interact for long chain-of-thought reasoning, with clear takeaways (e.g., entropy/temperature rule-of-thumb, stage-wise curriculum, and overlong-filtering policy). The paper offers generalizable training guidance.

- The empirical program is rigorous and targeted: clearly defined questions, extensive ablations, and transparent evaluation settings.

- The model is evaluated on a wide range of mathematical and coding benchmarks, using repeated sampling avg@n for variance reduction and consistent default decoding across all models.

### Weaknesses
- Please report GPU-days per stage to contextualize the recipe’s practicality.
- Adding one/more non-Qwen-7B model would validate generality.
- The authors did not release code, which may limit reproducibility.
- In Figure 4, why is the “final accuracy achieved at the end of each training stage” for different models not shown at the same step? Does this mean that different models require a different number of training steps in a given stage?
- Regarding the results in Figure 4, have you conducted experiments where RL is applied directly to the base model (skipping SFT)? Would this lead to similar conclusions? Additionally, the reviewer is also curious whether applying RL directly on an existing instruct model would produce more interesting comparison results.
- During different stages of RL training, is there any overlap between the samples used in each stage? How many samples were used in total for each stage?
- How is the number of training steps for each RL stage determined? What are the exact numbers?
- In the SFT experiments on the number of responses per prompt, does the diversity of the responses per prompt have a significant impact? When sampling multiple responses, how do the authors ensure diversity?

### Questions
See Weaknesses

### Soundness
4

### Presentation
3

### Contribution
4
