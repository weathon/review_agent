# MMedAgent-RL: Optimizing Multi-Agent Collaboration for Multimodal Medical Reasoning

- Decision: Accept (Poster)
- Scores: 6, 8, 4, 4

## Abstract
Medical Large Vision-Language Models (Med-LVLMs) have shown strong potential in multimodal diagnostic tasks. However, existing single-agent models struggle to generalize across diverse medical specialties, limiting their performance. Recent efforts introduce multi-agent collaboration frameworks inspired by clinical workflows, where general practitioners (GPs) and specialists interact in a fixed sequence. Despite improvements, these static pipelines lack flexibility and adaptability in reasoning. To address this, we propose MMedAgent-RL, a reinforcement learning (RL)-based multi-agent framework that enables dynamic, optimized collaboration among medical agents. Specifically, we train two GP agents based on Qwen2.5-VL via RL: the triage doctor learns to assign patients to appropriate specialties, while the attending physician integrates the judgments from multi-specialists and its own knowledge to make final decisions. To address the inconsistency in specialist outputs, we introduce a curriculum learning (CL)-guided RL strategy with dynamic entropy regulation, progressively teaching the attending physician to balance between imitating specialists and correcting their mistakes. Experiments on five medical VQA benchmarks demonstrate that MMedAgent-RL outperforms both open-source and proprietary Med-LVLMs. Notably, it achieves an average performance gain of 23.6\% over strong baselines.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes MMedAgent-RL, a multi-agent framework based on RL designed to optimize collaboration for multimodal medical diagnostic reasoning. The framework trains a triage doctor agent to assign cases and an attending physician agent, which uses a curriculum learning strategy, to integrate opinions from various specialist agents and make a final decision.

### Strengths
The proposed curriculum-based RL strategy to train an attending agent to evaluate and integrate potentially noisy or conflicting advice from specialists is a novel and well-motivated approach to improving the robustness of multi-agent systems.

### Weaknesses
My major comments are:

1. The specialist agents that provide the core domain knowledge are powerful proprietary models like GPT-4o and o3. This makes the framework's performance heavily dependent on external, closed-source models. The experiments do not clearly show how the system performs when only open-source models are used as specialists.
2.  The paper claims a significant performance gain over an SFT method. However, this SFT baseline is not clearly defined in the main text. It is uncertain if this is a model fine-tuned on ground-truth answers, on specialist responses, or another configuration. 
3. The paper states the triage doctor is optimized using GRPO. However, the ablation study in Table 2 only presents a "w/o Triage" condition, which removes the step entirely. It fails to compare the RL-optimized triage agent against a simpler, non-RL baseline (e.g., a standard SFT-trained classifier). Given the near-perfect triage accuracy reported, the necessity of using RL for this component is not well-justified.
4.  The curriculum learning strategy splits training data based on specialist accuracy (easy, medium, hard). The paper then presents an analysis of performance on test data that is also split by difficulty. The method for splitting this test data is not explained. If it is based on the specialists' accuracy on the test set, this constitutes data leakage, as it uses information about the test answers to perform the analysis.

### Questions
Please see the weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper develops a RL-based framework, namely MMedAgent-RL, for optimizing multi-agent collaboration in medical VLM reasoning. The proposed framework is well-motivated and empirically strong, with evaluations on both in-domain and out-of-distribution datasets.

### Strengths
(1) This framework develops a machine that can adjust collaboration policies based on task difficulty. The integration of C-MARL for entropy control is theoretically motivated.
(2) The evaluation and experimental design are comprehensive. The proposed framework shows good performance in 5 public datasets, including both in-domain and out-of-distribution datasets.

### Weaknesses
(1) The framework is inspired by the 'triage–specialist–attending'. The authors need to find more evidence to demonstrate that this aligns with the real hospitalization process. Within different sections in a hospital, the workflow may differ.
(2) This work lacks the involvement of human experts. 
(3) Some of the technical details are missing. For example, are the first GP and the second GP updated simultaneously?

### Questions
The authors use Qwen2.5-VL as the base model. Is this framework transferable to other base models?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces MMedAgent-RL, a reinforcement learning (RL)-based multi-agent framework that overcomes the rigidity of existing collaboration systems by enabling dynamic and optimized cooperation among medical agents. The framework mimics a clinical "triage-and-referral" system, utilizing a curriculum RL strategy with dynamic entropy regulation to train a primary model to intelligently integrate and resolve noisy or conflicting inputs from various specialist agents.

### Strengths
1. The paper is well-written, logically clear, and easy to follow.
2. The theoretical derivations are fairly sound.
3. Extensive experiments demonstrate the superiority of the proposed MMedAgent-RL.

### Weaknesses
1. The middle part of Figure 1(a) does not reflect the practical workflow of Multi-Agent collaboration; it seems to lack representation of the General Practitioner, which leads to ambiguity.
2. Section 3.1 mentions optimizing the triage doctor using GRPO, so it would be worthwhile to discuss the triage doctor's capability (quantitatively) as well as its reasoning process.
3. The underlying mechanism for the entropy regularization term in Equation 3.1 needs to be explained, and the rationale behind the choice and range of values for $\gamma_s$ should also be elaborated.
4. Figure 3 mentions the selection of three 'o3' models as specialist doctors? Does this mean each specialty uses three 'o3's? If so, there seems to be no differentiation between the specialties. Why was 'o3' chosen over a specially designed medical MLLM?
5. The experimental setup in Figure 4 is not clearly described, and more details should be provided.

### Questions
Please refer to Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes MMedAgent-RL, a reinforcement learning framework for multi-agent medical reasoning. The system simulates clinical workflows with a triage doctor routing cases to specialists (proprietary LVLMs), then an attending physician trained via curriculum RL to aggregate specialist opinions. The key innovation is a curriculum learning strategy with dynamic entropy regulation that progressively teaches the model to handle specialist outputs of varying reliability. Experiments on 5 medical VQA benchmarks show significant gains over baselines.

### Strengths
1. Framing multi-agent medical reasoning as a curriculum RL problem with dynamic entropy control is well-motivated by the reality of imperfect expert judgments.

2. Strong empirical results, 23.6% average gain over baselines and excellent OOD generalization (72.6% on MMMU/OmniMedVQA) demonstrate effectiveness.

3. The three-stage curriculum (easy/medium/hard based on specialist accuracy) with corresponding entropy coefficients (0.0001/0.005/0.03) is principled and clearly explained.

### Weaknesses
1. Missing critical baselines: No comparison with simpler alternatives that could  possible achieve similar results, eg. single GPT-4o or Qwen2.5-VL sampling N diverse outputs using different prompts or high temperatures → majority voting or trained aggregator. These would test if the complex triage+multi-expert pipeline is necessary.

2. The paper claims the attending physician learns to "correct specialist mistakes," but provides no quantitative evidence on hard cases where all specialists fail. A fairer and more rigorous evaluation is needed to determine how much of the performance gain is due to routing and how much is due to aggregation. 

3. Best performance requires OpenAI llms, but medical data cannot be sent to external APIs in privacy-critical environments. Where is the evaluation with deployable open-source specialists ?

I will reconsider my rating if the author addresses these questions well.

### Questions
1. In "w/o Triage" (Table 2), what exactly happens? Random specialist? All specialists? Please clarify and consider: single model with diverse sampling might achieve similar diversity without routing.

2. Can you provide results progressively adding components (base → +triage → +multi-expert → +curriculum RL) to quantify each contribution?

### Soundness
2

### Presentation
3

### Contribution
2
