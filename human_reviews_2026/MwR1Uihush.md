# One RL to See Them All: Visual Triple Unified Reinforcement Learning

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4, 4

## Abstract
Reinforcement learning (RL) has significantly advanced the reasoning capabilities of vision-language models (VLMs). However, its application beyond reasoning remains largely unexplored, especially for perception-intensive tasks like object detection and grounding. We propose V-Triune, a Visual Triple Unified Reinforcement Learning system that enables VLMs to jointly learn visual reasoning and perception tasks within a single training pipeline. V-Triune comprises three complementary components: Sample-Level Data Formatting to unify diverse inputs, Verifier-Level Reward Computation to deliver modular rewards via specialized verifiers, and Source-Level Metric Monitoring to enable fine-grained diagnostics. A key innovation within the verifier component is the proposed Dynamic IoU reward, which provides adaptive and progressive feedback for several perception tasks. Leveraging V-Triune, we develop Orsta (7B, 32B), a family of models built upon open-source backbones. Jointly training Orsta on a diverse dataset of eight representative reasoning (math, puzzle, etc.) and perception (detection, grounding, etc.) tasks leads to consistent improvements across both domains. As a result, Orsta achieves substantial gains on MEGA-Bench Core, with improvements ranging from +2.1 to +14.1 over its baselines, and these benefits extend to a wide range of downstream tasks. These results establish V-Triune as an effective and scalable system for building more comprehensive VLMs. Code is provided in the supplementary materials.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes V-Triune, a unified RL system for multimodal models that jointly optimizes visual reasoning and perception within a single pipeline. Implementation-wise, V-Triune uses a sample-wise definition of different rewards (reasoning/perception) that can be used to route into different verifiers. 

Using V-Triune, the authors train Orsta (7B, 32B) and report consistent gains (e.g., +2.1 to +14.1 on MEGA-Bench Core) and improvements on several downstream tasks.

A notable technical contribution is the Dynamic IoU curriculum that tightens the IoU threshold during training (0.85 -> 0.95 -> 0.99), balancing reward ambiguity vs. sparsity for detection/grounding.

Overall, I see limited technical contribution in this work and lean to recommend rejection pending author’s discussion.

### Strengths
1. Empirical effectiveness is validated through experiments on different scales of models, though it's not clear that the proposed method is the direct cause of the improvement.

### Weaknesses
1. Limited technical contribution. In section 3, 3.1 mainly discussed the sample-wise reward dataset format and its corresponding verifier design, 3.2 elaborated a curriculum learning paradigm for IoU-based reward, and 3.3 presented how authors monitor the training process. I think all of them do not provide technical contributions to the current visual RL community.

2. Incomplete ablation studies in the proposed sample-level reward mechanism. In section 4.3, the authors conducted ablations with:	

   - Different data mix (reasoning, perception, reasoning + perception) 
   - w/ and w/o dynamic IoU annealing strategy 

    However, an important ablation is missing – what if we use full data but only switch off the sample-wise reward (i.e. with standard RLVR)? Otherwise there is no way to differentiate the impact of the high-quality training data and the proposed sample-wise reward design.

### Questions
1. In L310, the authors mentioned on-/off-policy configuration of their method without explaining. How do you define on-/off-policy in your method? Is your on-policy setup based on a single synchronous loop and the off-policy setup based on a replay buffer?

### Soundness
2

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
4

### Summary
This manuscript presents V-Triune, a unified reinforcement learning (RL) framework for jointly training vision-language models (VLMs) on high-level reasoning and low-level perception tasks. It addresses key challenges—data heterogeneity, incompatible reward schemes (e.g., exact-match vs. spatial metrics), and opaque training dynamics—through a three-tier design: Sample-Level Formatting ,Verifier-Level Rewards and Source-Level Monitoring. Using V-Triune, the authors train the Orsta-7B and Orsta-32B models (based on Qwen2.5-VL), which achieve consistent gains over strong baselines on benchmarks like MEGA-Bench, COCO, and MathVista. The work also highlights key engineering insights, such as freezing the Vision Transformer during RL training to avoid gradient explosion and performance collapse.

### Strengths
1.Introduced a unified, co-designed, and extensible framework that addresses the intricate challenge of jointly optimizing a single model for both high-level reasoning and fine-grained perception tasks.

2.A "Dynamic IoU Reward" mechanism is proposed for perception tasks, employing a curriculum-based approach that incrementally tightens the IoU threshold to progressively raise the difficulty of earning rewards as training advances.

3.The efficacy and superiority of the proposed method are comprehensively demonstrated through extensive experiments, achieving performance gains across a diverse benchmarks.

### Weaknesses
1.While the manuscript posits a synergistic effect from jointly training perception and reasoning tasks, the results on the ScreenSpot-Pro benchmark are not fully convincing. Specifically, the Orsta-7B model demonstrates only a marginal 1.2% performance gain, which lags substantially behind the 50%-60% performance typically achieved by models of the 7B scale on this benchmark. This significant discrepancy casts considerable doubt on the general applicability of the claim that these two task types are always mutually beneficial.

2.The proposed Dynamic IoU Reward schedule, while effective, appears to be based on a pre-defined, heuristic-driven curriculum. This raises the question of whether an adaptive scheduling strategy could yield better or more robust results. For instance, have the authors considered methods where the IoU threshold adjusts automatically based on the model's real-time performance, such as maintaining a target success rate for reward acquisition?

### Questions
Regarding the training data composition, the paper reports that 18 distinct datasets were aggregated. For the sake of clarity and reproducibility, could the authors provide a more granular breakdown? Specifically, we request the sample counts for each of the 18 source datasets, detailing the numbers both prior to and following the data curation pipeline. Furthermore, what is the final proportional balance between the samples allocated to reasoning tasks versus those for perception tasks in the curated training corpus?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes V-Triune, a unified reinforcement learning framework that integrates sample-level formatting, verifier-level modular reward computation, and source-level metric monitoring. Based on this framework, the authors train Orsta (7B/32B) models across eight visual reasoning and perception tasks. The method achieves consistent improvements over the base model Qwen2.5-VL, showing stronger performance on both reasoning-heavy and perception-heavy benchmarks.

### Strengths
- The paper is clearly structured and easy to follow.
- Compared to the base model Qwen2.5-VL, Orsta surpasses it comprehensively across both reasoning and perception benchmarks.

### Weaknesses
- The related work section lacks discussion of prior multi-task or joint training research, and the paper provides no in-depth analysis of this key challenge, despite it being central to the proposed framework.
- In general, the technical novelty is relatively limited. The dynamic IoU reward is a reasonable practice but not particularly insightful; meanwhile, the proposed “source-level metric monitoring” appears to be more of a system or logging design rather than a methodological contribution (Sec. 3.3).
- The task-composition ablation in this paper does not clearly illustrate the difference between per-task fine-tuning and multi-task unified training. It would be better to break down the overall score into specific categories. I am curious whether, on perception benchmarks, the combination of "perception + reasoning" still outperforms "perception" alone.
- The evaluation against other baselines (e.g., MM-Eureka, VL-Rethinker) mainly relies on MEGA-Bench, with insufficient comparisons to other representative benchmarks (e.g., MMMU, MathVista), which are commonly used for evaluation.

### Questions
Why does the result in Figure 3(a) not match the main table? The main table reports a score of 38.3, but in Figure 3(a) it is below 37.5.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes V-Triune, a unified reinforcement learning framework that trains a vision-language model (VLM) on both visual reasoning and perception tasks within one pipeline. The V-Triune system has three key components: (1) Sample-Level Data Formatting (2) Verifier-Level Reward Computation (3) Source-Level Metric Monitoring.  Instead of using a fixed intersection-over-union threshold to reward predicted bounding boxes, the IoU threshold is gradually raised over the course of training. The Dynamic IoU reward provides adaptive and progressive feedback.
Empirically, the RL training yields consistent improvements in both domains: Orsta achieves substantial gains on the MEGA-Bench Core multi-task benchmarks.

### Strengths
# Strengths

1. The design of V-Triune demonstrates thoughtful architecture engineering. The Sample-Level Formatting overall is a good implementation strategy, and Verifier-Level Rewards achieves modularity and scalability. The highlighted  Dynamic IoU mechanism gradually tightening the IoU threshold over training to guide the model from coarse to fine localization, which match some insights from previous vision works.


2. The empirical results on multiple benchmarks show the strengths of the proposed methods.

### Weaknesses
# Weaknesses

1. Motivation for RL in Perception Tasks: The core premise—that reinforcement learning is the right tool for perception-heavy tasks such as object detection and grounding—is not entirely convincing.

2. Experiments: Although many baselines are reported, the critical point—why the method works—is not sufficiently supported or justified. For example, why not train directly on the collected dataset? Since compute is spent on training, is the proposed method the best way to use it?

3. One detailed observation is that the training is very resource-intensive (64 GPUs, probably many hours for just 3 epochs). RL fine-tuning here generates 8 candidate sequences per prompt and only then gets a reward. That’s a lot of computation for one gradient step. Please show that the performance boost is worth the heavy computation.

4. The proposed approach degrades inference speed (inference efficiency issue) by introducing reasoning steps. For perception tasks where fast responses are essential, this extra time is a drawback; please quantify end-to-end latency/FPS and discuss mitigation (e.g., a no-reasoning fast path).

### Questions
# Questions

Does the work freeze the vision backbone? In line 302, the authors state, “Freeze the ViT to prevent gradient explosion.” Does this mean the authors aim to improve perception by updating the LLM while assuming the vision backbone is not the bottleneck?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes V-Triune, a unified reinforcement learning system for post-training vision-language models on both reasoning (math, science, charts, puzzles) and perception (detection, grounding, OCR, counting) tasks. The system has three components: sample-level data formatting, verifier-level reward computation and source-level metric monitoring. A key technical contribution is a Dynamic IoU reward curriculum that tightens the IoU threshold over training, mitigating reward ambiguity vs. sparsity in detection/grounding. Built on this system, the paper trains Orsta-7B/32B on 47.7K curated samples from 18 sources across 8 tasks.

### Strengths
1. Unified RL system design with clean separation of concerns.
2. Dynamic IoU curriculum addresses reward ambiguity vs sparsity; backed by ablations on COCO multi-object and OVDEval negation subsets.
3. Clear training/eval configs; data schema; anonymized code/checkpoints promised in supplementary, and provide a 47k dataset to the community.

### Weaknesses
1. The core contributions skew toward system integration and scaling—per-sample routing, modular verifiers, and an IoU-based curriculum are each incremental/known ideas; the work reads more like a strong engineering recipe than a new algorithmic principle.
2. The proposed solution involves a lot of moving parts and task-specific logic, which could raise concerns about its scalability and maintainability. Each new task requires custom handling (formatting rules for inputs/outputs and a specialized verifier to compute rewards). For the eight tasks in the paper, the authors defined multiple reward components, e.g. exact answer checks for QA, format compliance checks, IoU calculation for detection, etc., sometimes even with sample-specific weightings. This heavy reliance on manual configuration means that scaling V-Triune to “all” vision-language tasks would demand significant human effort. If one wanted to incorporate a new modality or task (say, segmentation masks or video question answering), they would likely need to design new verifiers or reward functions from scratch. The paper frames V-Triune as extensible, but it does not demonstrate an automatic or learned way to extend to novel tasks, it is not a plug-and-play solution for arbitrary tasks without additional engineering.

### Questions
1. How are the various tasks balanced in the unified RL training loop? For example, If a particular task’s reward signal is weaker or noisier, how did you prevent it from being overshadowed by others?
2. How does the RL-fine-tuning approach compare to a multi-task sft on the same collection of tasks? For example, did you train a version of orsta using conventional supervised learning on the eight tasks, and if so, how did its performance differ from the RL-trained orsta?

### Soundness
3

### Presentation
2

### Contribution
2
