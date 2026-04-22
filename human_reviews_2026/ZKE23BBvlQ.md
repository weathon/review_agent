# Human Behavior Atlas: Benchmarking Unified Psychological And Social Behavior Understanding

- Avg Score: 5.20
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4, 4

## Abstract
Using intelligent systems to perceive psychological and social behaviors, that is, the underlying affective, cognitive, and pathological states that are manifested through observable behaviors and social interactions, remains a challenge due to their complex, multifaceted, and personalized nature. Existing work tackling these dimensions through specialized datasets and single-task systems often miss opportunities for scalability, cross-task transfer, and broader generalization. To address this gap, we curate Human Behavior Atlas, a unified benchmark of diverse behavioral tasks designed to support the development of foundation models for understanding psychological and social behaviors. Human Behavior Atlas comprises over 100,000 samples spanning text, audio, and visual modalities, covering tasks on *affective states*, *cognitive states*, *pathologies*, and *social processes*. Our unification efforts can reduce redundancy and cost, enable training to scale efficiently across tasks, and enhance generalization of behavioral features across domains. On Human Behavior Atlas, we train three models: Omnisapiens-7B SFT, Omnisapiens-7B BAM, and Omnisapiens-7B RL. We show that training on Human Behavior Atlas enables models to consistently outperform existing multimodal LLMs across diverse behavioral tasks. Pretraining on Human Behavior Atlas also improves transfer to novel behavioral datasets; with the targeted use of behavioral descriptors yielding meaningful performance gains. The benchmark, models, and codes can be found at: https://github.com/MIT-MI/human_behavior_atlas.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
By integrating diverse modalities and heterogeneous datasets under a single LLM-based paradigm, it demonstrates the potential of large models to generalize across emotional, cognitive, and behavioral domains. While some architectural or methodological innovations are limited, the work’s scale, data diversity, and behavior-centric vision represent a meaningful contribution that could serve as a foundation for future research in multimodal affective computing.

### Strengths
1. The authors successfully harmonize heterogeneous datasets and tasks (e.g., emotion recognition, personality, mental health indicators) under a single large language model (LLM)-based paradigm, which demonstrates strong potential for generalization across domains.
2. Leverages an unprecedented scale of multimodal behavioral data, enabling the model to learn rich representations that reflect both emotional and cognitive dimensions of human behavior.
3. Experimental results demonstrate solid transfer capabilities across a wide range of downstream behavioral and affective computing tasks

### Weaknesses
1. The authors designed two types of output: a dedicated classifier for categorical predictions, and a decoder that generates open-ended responses from the final hidden states. However, this classifier+decoder structure seems inconsistent with the logic of using large language models. Ideally, we expect an LLM to use a single decoder to produce all outputs, while the classifier approach merely leverages the model’s encoding capability, which deviates from the intended design philosophy.
2. The main contribution lies in integrating various tasks and datasets into a unified LLM-based training paradigm. However, the authors did not introduce any specific architectural designs to enhance model capability, for example, handling behavioral temporal dynamics, or effectively leveraging facial landmarks, acoustic cues, or pose keypoints. Instead, these modalities are simply fed into the model as additional data for brute-force learning, which shows limited innovation.
3. Currently, LLMs perform poorly on regression tasks, yet affective computing tasks such as valence-arousal estimation or depression assessment require fine-grained regression rather than simple binary classification. The authors’ choice to convert PHQ-9 into a binary classification problem thus oversimplifies the task and fails to reflect real-world application scenarios.
4. A common issue in multi-task learning is negative transfer or gradient conflict, but the authors did not address whether their method incorporates any targeted solutions to mitigate these problems.
5. The superior performance in transfer learning could be attributed to the use of large-scale and diverse multimodal data (visual/acoustic cues) rather than the proposed model architecture or data structure itself, the paper does not clarify this distinction.

### Questions
1. Although the authors devote substantial discussion to explaining how behavioral descriptors benefit model training (a point already well established in many multi-task affective models) it is unclear why the model itself is not designed to predict these descriptors directly. Doing so would align more closely with the paper’s title, “Human Behavior Atlas”, which implies an explicit mapping or prediction of behavioral factors.
2. In the implementation section, the authors carefully adjust minibatch sizes across different tasks. However, it would be more informative if they could quantitatively present the impact of task diversity on training. Moreover, since the datasets vary greatly in size, it remains unclear how the authors address data imbalance during training or does imbalance affect the model performance.
3. The comparison models were not trained on such a large and diverse dataset, which raises concerns about fairness. A more rigorous and convincing evaluation would involve cross-dataset validation, for example, testing affective models across IEMOCAP, Aff-Wild2, or AffectNet, to demonstrate the generalization ability of the proposed approach.

### Soundness
3

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
2

### Summary
The paper introduces **Human Behavior Atlas (HBA)**, a unified multimodal benchmark for psychological and social behavior understanding. Key contributions include:
- A standardized **prompt–target interface** across 14 datasets and 10+ behavioral tasks.
- A **Behavioral Adapter Module (BAM)** for injecting external behavioral descriptors (e.g., pose, acoustic features) into a frozen backbone.
- Three training variants: **SFT, BAM, and RL (GRPO)**, evaluated comprehensively across tasks.

### Strengths
- **Unified benchmark** with broad task coverage and standardized metrics.
- **Clean and modular design** of BAM, enabling non-invasive feature injection.
- **Consistent empirical gains** of behavior-specialized models over general MLLMs.

### Weaknesses
#### **1. Insufficient Ablation Studies on BAM and RL**
- **BAM Design Choices**: The paper does not justify key design decisions—e.g., why the adapter is inserted at the **penultimate layer**, why **mean and standard deviation pooling** is used for temporal descriptors, or why a **two-layer FFN** is chosen. A systematic sweep over injection points, hidden dimensions, and pooling strategies is missing.
- **Descriptor Contribution**: The relative importance of **visual vs. acoustic descriptors** is not analyzed. Ablations disabling each stream would clarify their individual and synergistic effects.
- **RL Hyperparameters**: The use of **β = 0 for KL regularization** is not motivated, and no sensitivity analysis is provided for group size, clipping epsilon, or reward scaling. Training stability and reward hacking risks are unexamined.

#### **2. Evaluation Protocol Gaps**
- **LLM-as-Judge Reliability**: The open-ended evaluation relies solely on a **single closed-source LLM judge** (GPT-5-nano). No human–judge agreement, inter-judge consistency, or prompt robustness tests are reported.
- **Prompt Isolation & Contamination**: The paper does not clarify whether the judge and model prompts are isolated, whether sessions are reset, or whether tools are disabled—raising concerns about evaluation integrity.

#### **3. Label Granularity Reduction**
- Sentiment labels in datasets like **MOSEI (7-point scale)** are collapsed to **binary positive/negative**, discarding nuanced distinctions. This limits comparability with prior work and may mask model failures on fine-grained sentiment.

#### **4. Metric Heterogeneity and Cross-Task Comparability**
- Discrete tasks use **weighted F1 or weighted accuracy**, while open-ended tasks use **LLM-judged TRUE-rate**. This inconsistency complicates cross-task aggregation and model ranking.
- No unified scoring mechanism (e.g., normalized score per task family) is proposed.

#### **5. Lack of Computational Efficiency Analysis**
- The computational cost of BAM (e.g., latency, memory overhead) is not reported, nor is its impact on inference speed or scalability—key for real-world deployment.

#### **6. Limited Discussion on Multimodal Fusion Strategy**
- The fusion of text, audio, and video is performed via **simple concatenation** in the embedding space. More advanced fusion mechanisms (e.g., cross-attention, gating) are not explored or motivated.

### Questions
1. **LLM-as-Judge Reliability**: Please report human–judge agreement (κ or ρ) on a 500-sample subset, add at least one open-source judge, and show sensitivity to prompt variations.
2. **BAM & RL Ablations**: Systematically vary BAM’s injection layer, hidden size, and descriptor streams. For RL, sweep β, group size, and reward terms—report accuracy vs. compute and training stability.
3. **Fine-Grained Evaluation**: Provide results on full label sets (e.g., 7-point sentiment) and justify binarization decisions with class-wise F1 and confusion matrices.
4. **Metric Harmonization**: Propose a unified scoring scheme (e.g., normalized score per task family) and report confidence intervals over multiple runs.
5. **Prompt and Protocol Transparency**: Document judge–model isolation measures and include a finalized, typo-free version of the judge rubric.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose an unified benchmark of diverse behavioral tasks designed to support the development of unified models for understanding psychological and social behaviors. The aim is to take the opportunities for scalability, cross-task transfer, and broader generalization.

### Strengths
- The paper is well organized and written
-  The unified benchmark named HUMAN BEHAVIOR ATLAS comes in a timely manner for the field. It covers a large spectrum of situations.

### Weaknesses
None

### Questions
What will be the restrictions about the use of this benchmark?

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
The paper proposes HUMAN BEHAVIOR ATLAS, a unified multimodal benchmark focused on understanding psychological and social behaviors. The benchmark combines 13 existing datasets into a prompt-response instruction format across text, audio, and video modalities, totaling around 101k samples spanning 10 behavioral task categories. Three model variants are trained and evaluated: OMNISAPIENS-7B SFT, OMNISAPIENS-7B BAM, and OMNISAPIENS-7B RL. Results show that these models outperform general multimodal LLMs (Qwen 2.5, Gemma-3, HumanOmni) on both multi-task and transfer-learning evaluations.

### Strengths
1. The benchmark is a useful contribution that could help move multimodal LLM research toward more holistic behavioral understanding. 
2. The experimental setup is thorough, covering 10 in-domain datasets and some "transfer" datasets. 
3. The benchmark and code will become publicly available.

### Weaknesses
1. No new data is collected - the benchmark repackages existing datasets into a unified format. While valuable, this limits novelty.
2. The models are fully fine-tuned on the same datasets used for evaluation. In the transfer-learning section, the model is again fine-tuned for a few epochs on the "held-out" datasets, and only Qwen 2.5-Omni-7B is used for comparison. This setup mainly measures fine-tuning efficiency, not true zero-shot generalization.
3. The construction of prompts is unclear. It’s not specified whether they were hand-crafted or automatically generated, nor whether prompt robustness was tested.

### Questions
1. In Table 4, can you add results using only the behavioral descriptors (e.g., OpenSMILE, MediaPipe) to show how much these features alone contribute?
2. Since the models are fine-tuned on the same datasets used for evaluation, have you checked whether fine-tuning affects general-purpose abilities (e.g. text generation)?
3. Did you perform any true zero-shot evaluations to measure actual generalization?
4. How were prompts generated - hand-written or LLM-generated?
5. In Table 5, it would be interesting to add results before fine-tuning on the held-out datasets and also show results for the multimodal LLMs used in Table 4 for comparison.

### Soundness
3

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
This paper introduces HUMAN BEHAVIOR ATLAS, a large-scale multimodal benchmark with numerous tasks for general psychological and social behavior understanding. The core data contribution is curation and standardization of existing public datasets. Three variants of OMNISAPIENS-7B are evaluated to show the effectiveness of multi task training and transfer learning.

### Strengths
* Benchmark, source code, models will be released.
* Atlas is a large-scale multimodal benchmark with numerous tasks for general psychological and social behavior understanding.
* Paper is well-written, and easy to follow.
* Promising performances for multi-task training and transfer learning are shown.

### Weaknesses
* My major concern is the limited contributions
  * The paper does not propose a new model architecture.
  * The core data contribution is curation and standardization of existing public datasets instead of new data collection.
* The comparison in Table 4 is unfair because the OMNISAPIENS-7B variants were trained directly on the HUMAN BEHAVIOR ATLAS data, while the general multimodal LLM baselines were evaluated in zero-shot inference mode without fine-tuning on this specific benchmark. The performance gain largely reflects the benefit of SFT or RL on the target tasks, not necessarily the inherent superiority of the model's architecture or the benchmark itself.
* To demonstrate the superiority of the ATLAS for unified modeling, the authors should compare OMNISAPIENS-7B variants to models trained exclusively on another large, existing social behavior or affective computing dataset (e.g., a comprehensive version of CMU-MOSEI, MELD, or a large synthesized dataset like HumanOmni) and then test all models across the full range of ATLAS tasks.
* A similar issue in the transfer learning experiment in Section 4.2 and Table 5.

### Questions
Please refer to the weaknesses section.

### Soundness
2

### Presentation
2

### Contribution
2
