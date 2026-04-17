# DriveAction: A Benchmark for Exploring Human-like Driving Decisions in VLA Models

- Decision: Reject
- Scores: 2, 2, 4, 4

## Abstract
Vision-Language-Action (VLA) models have advanced autonomous driving, but existing benchmarks still lack scenario diversity, reliable action-level annotation, and evaluation protocols aligned with human preferences. To address these limitations, we introduce DriveAction, the first action-driven benchmark specifically designed for VLA models, comprising 16,185 QA pairs generated from 2,610 driving scenarios. DriveAction leverages real-world driving data proactively collected by drivers of autonomous vehicles to ensure broad and representative scenario coverage, offers high-level discrete action labels collected directly from drivers’ actual driving operations, and implements an action-rooted tree-structured evaluation framework that explicitly links vision, language, and action tasks, supporting both comprehensive and task-specific assessment. Our experiments demonstrate that state-of-the-art vision-language models (VLMs) require both vision and language guidance for accurate action prediction: on average, accuracy drops by 3.3% without vision input, by 4.1% without language input, and by 8.0% without either. Our evaluation supports precise identification of model bottlenecks with robust and consistent results, thus providing new insights and a rigorous foundation for advancing human-like decisions in autonomous driving.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces DriveAction, a new VLA benchmark built from real-world driving data. The dataset contains QA pairs — including action questions — collected from deployed autonomous vehicles, and uses driver-labeled discrete action decisions. The benchmark covers a wide range of scenarios such as intersections, lane changes, and ramp merges that are useful for evaluating corner-case decision making. The authors propose an action-rooted evaluation framework and show results across various VLMs to demonstrate benchmark sensitivity.

### Strengths
The dataset comes from actual self-driving deployments, not synthetic or open-source simulator data. This includes real corner cases, high-value for both research and industry.

Human-validated action labels ensure the dataset has clean supervision and avoids spurious driving behaviors.

The scenarios selected are indeed realistic and relevant — these are good assets for the community to study.

### Weaknesses
I’m really unsure about the usefulness and motivation of this benchmark for VLA-based driving.
The actions are framed as multiple-choice. In real driving, there is no predefined set of choices popping up like a test. So I’m not convinced how this maps to actual autonomous driving:

Where would these choices come from at runtime?

Generated from another model? If so, that introduces another huge source of error.

Many examples feel tailor-crafted to match the scenario — unlikely to generalize.

Modern VLAs are being used to directly produce actions (i.e., trajectories or control tokens). In that case, direct prediction is simpler and more aligned with real-time operation than asking them to pick from provided abstract options.

To truly show real-world relevance, the authors need to demonstrate that better DriveAction performance → better driving.
For example:
1. improvements in collision rate
2. lower displacement error
3. higher success rate in closed-loop evaluation
None of that is measured here.

If the intention is instead to evaluate general reasoning, then the benchmark is too narrow — it only includes driving scenarios. In that framing, its impact would be limited because it can’t tell you anything about general VLM robustness across domains.

So either:
• The benchmark evaluates driving performance — then connect results to actual driving metrics.
• The benchmark evaluates reasoning — then it’s too domain-specific.
Right now it’s stuck in the middle — neither fully useful for driving systems nor for reasoning more broadly.

### Questions
Can the authors show that performance on DriveAction correlates with real driving outcomes (DE, CR, etc.)?

Can the authors justify how these multiple-choice actions would exist in an actual autonomous vehicle system?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces DriveAction, the first action-driven benchmark specifically designed for Vision-Language-Action (VLA) models in autonomous driving. DriveAction focuses on human-like decision-making. Extensive evaluations on 12 general VLMs and two domain-specific models (Non-MoE vs MoE) reveal how vision and language inputs affect final decisions and expose task-specific bottlenecks (e.g., navigation and traffic-light understanding).

### Strengths
1. Collecting 16k QA pairs from 2,610 real-world driving scenarios contributed by professional drivers.
2. Using real-time driver actions as ground-truth labels to capture authentic human decision intent.
3. Proposing an action-rooted tree-structured evaluation framework that connects vision, language, and action layers.

### Weaknesses
As we all know, VLA models are inherently action-centric, and thus the action dimension should play a more decisive role in evaluation. However, DriveAction primarily emphasizes open-loop QA assessments on Dynamic, Static, Navigation, and Efficiency tasks, rather than measuring closed-loop driving behavior that reflects real-time control and long-horizon decision consistency. So it makes me confused. I think the author needs to discuss more about the importance of this benchmark in the community.

### Questions
Same to Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces DriveAction, a benchmark specifically designed for Vision-Language-Action (VLA) models in autonomous driving. It aims to fill gaps in scenario diversity, action-level annotation, and human-aligned evaluation. DriveAction includes 16,185 QA pairs across 2,610 driving scenarios, derived from driver-contributed real-world data, featuring action-rooted, tree-structured evaluation connecting vision, language, and action tasks. Experiments with 12 VLMs (e.g., GPT-4o, Claude 3.7, Gemini 2.5 Pro) reveal performance drops when either vision or language modalities are removed, highlighting multimodal dependence.

### Strengths
1. Proposes DriveAction, the first benchmark explicitly designed for Vision-Language-Action (VLA) evaluation in autonomous driving, addressing missing links between vision, language, and action reasoning.

2. Action labels are collected directly from real-time driver operations, faithfully capturing human decision intent rather than post-hoc annotations.

3. The action-rooted, tree-structured framework enables interpretable, modular analysis across V-L-A components, offering fine-grained evaluation flexibility.

4. Evaluates 12 state-of-the-art VLMs under four modality configurations (V-L-A / V-A / L-A / A), systematically showing modality dependencies.

### Weaknesses
1. While the benchmark is well-structured, its main finding (that models need both vision and language inputs) is intuitive and not conceptually groundbreaking.

2. Previous works like DriveLM (Sima et al., 2024) and Reason2Drive (Nie et al., 2024) already explore end-to-end reasoning or goal-driven evaluation, weakening the “first action-driven” claim.

3. Evaluation focuses on accuracy without deeper breakdowns (e.g., statistical variance, error typology, or causal reasoning analysis).

### Questions
1. How is inter-driver variability handled in “driver-contributed” data to ensure label consistency?

2. Could authors clarify whether the action labels are categorical only or also include continuous control values?

3. How are QA pairs validated for bias or ambiguity given LLM assistance in generation?

4. Do the results generalize to unseen city environments, or is there domain overfitting?

### Soundness
3

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
This paper introduces DriveAction, an action-driven benchmark for Vision-Language-Action (VLA) models in autonomous driving. It includes over 16K QA pairs across diverse real-world scenarios with human-annotated action labels and a tree-structured evaluation framework, enabling comprehensive assessment of vision, language, and action reasoning.

### Strengths
1. The paper introduces DriveAction, a well-structured benchmark.

2. Dataset quality is high, with real-world, driver-contributed data with diverse scenarios.

### Weaknesses
1. I suggest that the authors include a discussion of recent studies on VLM-generated datasets for autonomous driving that are built on different foundations. For example, some works such as [1][2] generate data based on existing datasets like nuScenes or nuPlan, while others use internal datasets. Highlighting these distinctions would help the community better understand the overall differences and positioning of this work.

[1] Y. Xu et al., “VLM-AD: End-to-End Autonomous Driving through Vision-Language Model Supervision,” CoRL, 2025

[2] Z. Zhou et al., “AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning,” NeurIPS, 2025.

2. Another concern is dataset quality. Although human verification is mentioned, the annotation and quality-control process could be described in greater detail to improve transparency and reproducibility.

3. While the benchmark design is strong, the paper mainly focuses on dataset construction and evaluation, with limited methodological novelty. I am not sure whether it fits better under a benchmark or dataset track, if such a category exists.

### Questions
1. How is DriveAction’s action taxonomy defined and maintained to prevent overlap or ambiguity between tasks (e.g., “navigation lane change” vs. “efficiency lane change”)?


2. Were there any efforts to balance action categories, given the natural bias toward simple maneuvers (e.g., going straight)?

### Soundness
3

### Presentation
3

### Contribution
2
