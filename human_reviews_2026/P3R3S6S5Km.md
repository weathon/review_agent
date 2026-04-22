# RLBFF: Binary Flexible Feedback to bridge between Human Feedback & Verifiable Rewards

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Reinforcement Learning with Human Feedback (RLHF) and Reinforcement Learning with Verifiable Rewards (RLVR) are the main RL paradigms used in LLM post-training, each offering distinct advantages. However, RLHF struggles with interpretability and reward hacking because it relies on human judgments that usually lack explicit criteria, whereas RLVR is limited in scope by its focus on correctness-based verifiers. We propose Reinforcement Learning with Binary Flexible Feedback (RLBFF), which combines the versatility of human-driven preferences with the precision of rule-based verification, enabling reward models to capture nuanced aspects of response quality beyond mere correctness.  RLBFF extracts principles that can be answered in a binary fashion (e.g. accuracy of information: yes, or code readability: no) from natural language feedback. Such principles can then be used to ground Reward Model training as an entailment task (response satisfies or does not satisfy an arbitrary principle). We show that Reward Models trained in this manner can outperform Bradley-Terry models when matched for data and achieve top performance on RM-Bench (86.2\%) and JudgeBench (81.4\%, \#1 on leaderboard as of September 24, 2025). Additionally, users can specify principles of interest at inference time to customize the focus of our reward models, in contrast to Bradley-Terry models. Finally, we present a fully open source recipe (including data) to align Qwen3-32B using RLBFF and our Reward Model, to match or exceed the performance of o3-mini and DeepSeek R1 on general alignment benchmarks of MT-Bench, WildBench, and Arena Hard v2 (at $<5$\% of the inference cost).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces Reinforcement Learning with Binary Flexible Feedback (RLBFF), a novel paradigm designed to bridge the gap between RLHF and RLVR. RLBFF combines the versatility of human-driven preferences with the precision of rule-based verification. The method shows strong performance on RM-Bench and JudgeBench.

### Strengths
- The paper clearly articulates the distinct limitations of RLHF and RLVR. The proposed RLBFF is a well-motivated solution that effectively bridges these two paradigms by incorporating the flexibility of human judgment with explicit, binary, and verifiable criteria.

- Reward models trained with RLBFF achieve high performance on major alignment benchmarks.

### Weaknesses
The paper is technically sound and well-executed. I find no discernible weaknesses.

### Questions
N/A

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Reinforcement Learning from Binary Flexible Feedback (RLBFF) proposes to use binary feedback derived from sets of explicit, human-interpretable principles to train scalar reward models and align large language models (LLMs). It mainly focuses on converting HelpSteer3-Feedback, an open-source dataset of natural language feedback, into structured binary evaluations along principles such as clarity, accuracy, or adherence to requirements. Unlike traditional RLHF, which relies on pairwise human preferences, or RLVR, which focuses narrowly on correctness, RLBFF extracts principle–fulfillment pairs (yes/no) to enable more transparent and fine-grained reward modeling. Using these data, the authors train flexible scalar and generative reward models that outperform previous methods on RM-Bench, JudgeBench, and the newly introduced PrincipleBench, achieving top leaderboard performance. They further apply RLBFF to align Qwen3-32B, achieving results comparable to proprietary systems like o3-mini and DeepSeek R1 on general alignment benchmarks such as MT-Bench, WildBench, and Arena Hard v2, while maintaining efficiency and interpretability at less than 5% of the inference cost.

### Strengths
* Merging RLHF and RLVR represents an important research direction, aiming to combine the interpretability and verifiability of rule-based feedback with the flexibility and broad applicability of human preference learning.
* RLBFF reports strong empirical results, where they outperform BT models trained on similar data and other Generative RMs on both in-distribution regarding training data (the proposed PrincipleBench) and other benchmarks(RM-bench, JudgeBench). 
* The presentation is clear and easy to follow

### Weaknesses
* Extracting principles is not possible for other datasets that do not contain textual feedback. The format or quality of the textual feedback may be critical to this method's performance. This may limit the generalizability of this approach to other datasets.

### Questions
* The Skywork-8B-v2 model seems to report higher results on RM-bench, although on limited categories. Could you provide a comparison with your models if possible?
* In Table 5, can you include the BT baseline model in comparison with '+RLBFF'? This will help analyze whether the dataset is, or the reward model pipeline, is more critical for improving performance.

### Soundness
3

### Presentation
4

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
This paper introduces Reinforcement Learning with Binary Flexible Feedback (RLBFF), aiming to bridge RL with human feedback (RLHF) and RL with verifiable rewards (RLVR) in the context of large language model alignment. The authors argue that RLHF suffers from ambiguity and reward hacking, whereas RLVR is too narrow, and propose RLBFF, which uses human-written, principle-grounded, binary feedback to make reward assignments both interpretable and precise. They extract over one thousand fine-grained principles from HelpSteer3-Feedback, train both scalar and generative reward models on this filtered data, and evaluate models on RM-Bench, JudgeBench, and the newly curated PrincipleBench. Their Scalar Reward Model supports user-specified principles at inference, runs efficiently, and achieves state-of-the-art results on several benchmarks. The authors also show how the approach can align Qwen3-32B to strong performance on major open-source alignment benchmarks with substantial cost savings.

### Strengths
1 Clear Motivation and Formulation: The paper gives a lucid account of the deficiencies of RLHF and RLVR, motivating the need for reward models that are both interpretable and principled (see Section 1, especially Table 1).

2 Methodological Innovation: The principle-grounded binary feedback format is an interesting compromise between free-form preferences and rigid verifiable criteria. The pipeline for extracting principles via LLM prompting, with grounding in supporting text spans and consensus filtering via embedding similarity, is thoughtfully designed (see Section 3 and the discussion around the filtering strategy).

3 Strong Empirical Results: The Flexible Principles reward models set a new high-water mark on RM-Bench and JudgeBench (see Table 2), and further excel on PrincipleBench, which focuses on adherence to explicit principles beyond correctness. This outperforms both scalar and generative baselines, including those requiring much heavier computation.

### Weaknesses
1 Limited Theoretical Analysis of Loss Design and Convergence: While the scalar reward modeling and binary prediction setup is elegant, there is little substantive mathematical analysis of the properties of the loss, especially with respect to class imbalance (Section 4.3). For example, while the principle fulfillment distribution is only slightly imbalanced ($35.4%$ no; $64.6%$ yes), no explicit techniques are described for dealing with such bias (e.g., reweighting, calibration). This omission is relevant since reward hacking and label bias are major themes of the work. More quantitative analysis of the loss landscape, as well as calibration studies, would add rigor.

2 Benchmarks and Biases: While Table 2 and Table 3 provide strong empirical evidence, most results are for benchmarks (RM-Bench, JudgeBench, PrincipleBench) that are closely related to the training data or evaluation principle. The generalizability to tasks or principles outside the extracted HelpSteer3-Feedback distribution is not explored in depth. Are there domains or principle types where the model underperforms or fails?

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces Reinforcement Learning with Binary Flexible Feedback (RLBFF) that combines the versatility of RLHF with the precision of RLVR, enabling reward models (RMs) to capture aspects of response quality beyond correctness. RLBFF extracts principles that can be answered in a binary manner from natural language feedback. They create this training data with principles from Helpsteer3-Feedback, and use it to train reward models and also align LLMs. Experimentally, they demonstrate that RMs trained using principles outperform Bradley-Terry models and other external baselines on RM-Bench, JudgeBench, and a newly proposed PrincipleBench (designed to evaluate RMs ability to follow arbitrary principles). Additionally, they also show that a Qwen model aligned using RLBFF and their RM matches or exceeds the performance of o3-mini and Deepseek-R1 on general alignment benchmarks.

### Strengths
The paper proposes RLBFF, which combines the benefits of RLHF and RLVR, enabling reward models to capture arbitrary aspects of response quality beyond simple correctness. RLBFF extracts principles that can be answered in a binary fashion from feedback data and uses them to train both ScalarRMs and GenerativeRMs, improving their performance over standard Bradley–Terry models and external baselines on reward model benchmarks.

The proposed Flexible Principles Scalar RM is the first scalar reward model to enable grounding by user-specified principles, allowing for rapid inference while achieving enhanced performance over baselines.

The authors also perform alignment on a Qwen model using RLBFF and a trained principle GenRM, achieving performance that matches or exceeds o3-mini, Claude-Sonnet, and DeepSeek-R1 at a fraction of the inference cost.

Furthermore, the paper details the recipe used to construct the principle data from HelpSteer3-Feedback, improving reproducibility. Additionally, the authors propose PrincipleBench, a new reward model benchmark designed to evaluate the ability of RMs to follow arbitrary principles—a capability that was previously lacking.

### Weaknesses
The authors use Generative RMs in their alignment setup. However, inference with Generative RMs is more expensive than with Scalar RMs. A comparison against alignment performed using Scalar RMs would have been enlightening, as it would clarify the performance trade-off relative to the faster inference offered by Scalar RMs. Additionally, given the comparatively poorer performance of GenRMs compared to ScalarRMs on PrincipleBench, which was hypothesized to result from GenRMs being initialized from reasoning models that typically excel at correctness tasks, ScalarRMs might be a better choice for general purpose alignment.

It would also have been valuable to see whether RLBFF provides significant gains over approaches using standard Bradley–Terry RMs at smaller model scales.

### Questions
1) Could you provide alignment results obtained using a Scalar RM, and contrast its performance with that achieved using a Generative RM?

2) Given the comparatively poorer performance of GenRMs compared to ScalarRMs on PrincipleBench, which was hypothesized to result from GenRMs being initialized from reasoning models that typically excel at correctness tasks, wouldn’t ScalarRMs be a better choice for alignment in scenarios requiring broadly good performance, rather than just correctness?

3) Are there any results comparing RMs trained with RLBFF and Bradley–Terry models at smaller model scales (e.g., 3B, 8B)?

4) Will the principle dataset created from HelpSteer3-Feedback be released? While the recipe is provided, access to the dataset itself would greatly facilitate adoption and reproducibility.

### Soundness
3

### Presentation
3

### Contribution
3
