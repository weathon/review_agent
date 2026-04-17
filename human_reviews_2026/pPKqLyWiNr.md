# LLaVAction: evaluating and training multi-modal large language models for action understanding

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Understanding human behavior requires measuring behavioral actions. Due to its complexity, behavior is best mapped onto a rich, semantic structure such as language. Emerging multimodal large language models (MLLMs) are promising candidates, but their fine-grained action understanding ability has not been fully examined. In this work, we reformulate EPIC-KITCHENS-100, one of the largest and most challenging egocentric action recognition datasets, into a MLLM benchmark (EPIC-KITCHENS-100-MQA). We demonstrate that when we sample difficult answers based on specialist models as distractors, leading MLLMs struggle to recognize the correct actions. How can we increase the performance of MLLMs? We curated a supervised finetuning dataset that includes `hard' action recognition, temporal detection, captioning, and free-form question answering to improve models' diverse action understanding capabilities. We introduce a new model called LLaVAction that adds an action token to boost models' attention on visual tokens and a two-stage pipeline to obtain structured actions. LLaVAction greatly improves the MLLMs' ability of action understanding, achieving strong improvements on both MLLM benchmarks (21 points in accuracy over GPT-4o on EPIC-KITCHENS-100-MQA) and established action recognition benchmarks, suggesting that our methods prepare MLLMs to be a promising path forward for complex action tasks. 
Code, data, the benchmark, and models are available at https://github.com/AdaptiveMotorControlLab/LLaVAction.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The technical contributions, especially the comprehensive training data mixture, clearly demonstrate improved performance over baselines like GPT-4o. 

However, the effectiveness of the new benchmark is weakened by concerns regarding the **validity of its hard negative samples**, which undermines the strength of the evaluation methodology. 
Furthermore, the specialized Action Token unnecessarily introduce task-specific constraints that **limit the model’s generalization capability**. The two-stage structured action prediction pipeline is an **unnecessary design**.

Given that the benchmark validity is questionable and the proposed methods either lack necessity or generality, but recognizing the significant technical gains achieved, the paper lands at a borderline reject.

### Strengths
LLaVAction integrates structured training tasks like temporal detection and temporal order learning to teach the model action boundaries and natural continuity. Such training data mixture strategy improves action understanding ability of MLLMs.

### Weaknesses
The Hard Example Mining strategy used for the new EPIC-KITCHENS-100-MQA benchmark is questionable due to the validity of hard negatives. 
The generated "hard distractors"—which are chosen for being highly confusable errors—may be semantically correct actions due to potential annotation ambiguity/near synonyms, or missing labels. The EPIC-KITCHENS dataset has 4K action labels, which is hard to annotate exhaustively in the original dataset.
Although the mechanism guarantees the exclusion of the exact ground truth class ID, it does not ensure the distractors are true semantic negatives.



The model’s Action Token design compromises MLLM generality by employing explicit classification heads to predict a finite set of nouns and verbs. While intended to enhance visual feature utilization, this hard-coded classification is specialized for the datasets with closed set labels, moving away from the flexible, open-vocabulary text generation paradigm characteristic of general MLLMs.



The proposed two-stage structured action prediction pipeline is unnecessary. 
The authors contend that using this pipeline is essential because "One could put all the possible actions in the question prompt and let MLLMs choose, but this will become impossible when the number of action types increases" (e.g., 4k actions). However, this argument is questionable because simpler, direct classification methods exist that avoid context overflow, such as storing the video+prompt+question in the KV Cache and calculating the loss by appending individual class names to perform 4K classification natively.

### Questions
The authors could discuss more about the Hard Example Mining strategy, since the other shortcomings are hopeless.

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
5

### Summary
This study focuses on improving the understanding of complex human actions by emerging multimodal large language models (MLLMs). The researchers reformulate the EPIC-KITCHENS-100 dataset into a new benchmark, EPIC-KITCHENS-100-MQA, to evaluate MLLMs' fine-grained action recognition. They find that leading MLLMs struggle with identifying correct actions, especially when presented with challenging distractors. To address this, they develop a supervised fine-tuning dataset encompassing tasks like hard action recognition, temporal detection, and free-form question answering. They also propose a new model, LLaVAction, which improves attention to visual tokens and utilizes a two-stage pipeline to better understand structured actions. The model significantly enhances performance, achieving notable accuracy gains on both MLLM and action recognition benchmarks, highlighting its potential for tackling complex human behavior tasks.

### Strengths
1. The paper is well-written, presenting the methodology and experimental results in a clear and systematic manner.
2. The research topic is highly practical and meaningful. Enhancing MLLMs' ability to better understand human actions could benefit a wide range of real-world applications.
3. By approaching the problem from an action recognition perspective, the paper introduces a new, challenging benchmark. This dataset holds potential to contribute valuable insights to research within the relevant domain.
4. The proposed LLaVAction model not only demonstrates strong performance in action recognition tasks but also achieves improvements across more generic benchmark datasets. This is a noteworthy phenomenon, showcasing the potential value and versatility of the approach to broader applications.

### Weaknesses
1. While the paper focuses on action understanding, its evaluation is limited to action recognition tasks. Comprehensive human action understanding involves other dimensions, such as temporal dynamics, the meaning or intent behind actions, and human interactions. Previous works like MotionLLM [1] and HAIC [2], which explore the use of MLLMs for human action understanding beyond just action recognition, have introduced broader benchmarks and methodologies covering these aspects. This type of works are highly relevant to this paper's domain and should be discussed as part of the related work. Furthermore, demonstrating LLaVAction's effectiveness on these more comprehensive action understanding benchmarks could strengthen the proposed method.
2. Although the paper shows that LLaVAction improves performance on general video understanding benchmarks, a detailed analysis of the reasons behind these improvements is missing. Investigating which aspects of the proposed training methodology contribute to these gains would provide deeper insights into the strengths of the approach.

[1] Chen, Ling-Hao, Shunlin Lu, Ailing Zeng, Hao Zhang, Benyou Wang, Ruimao Zhang, and Lei Zhang. "Motionllm: Understanding human behaviors from human motions and videos." arXiv preprint arXiv:2405.20340 (2024).

[2] Wang, Xiao, Jingyun Hua, Weihong Lin, Yuanxing Zhang, Fuzheng Zhang, Jianlong Wu, Di Zhang, and Liqiang Nie. "HAIC: Improving Human Action Understanding and Generation with Better Captions for Multi-modal Large Language Models." ACL 2025.

### Questions
Will the datasets, models, and related resources be made publicly available?

### Soundness
2

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
4

### Summary
This paper tackles the challenge of action understanding in Mutli-modal Large Language Models (MLLMs). The paper address this challenge by introducing a new and challenging benchmark EPIC-KITCHENS-100-MQA, which a multiple-choice, fine-grained action-understanding benchmark by reformulating the EK100 datasets into a multiple-choice question-answering (MQA) task. More importantly, the paper makes its contribution by utilizing hard distractors, leveraging predictions from latest action recognition models to generate semantically similar and difficult distraction choices. The paper then proposes LLaVAction, which introduces an “action token” to enhance focus on visual information and a two-stage inference pipeline to produce structured action outputs. On EPIC-KITCHENS-100-MQA, LLaVAction outperforms baseline MLLMs and achieves SOTA results on several other action recognition and video understanding benchmarks.

### Strengths
•	Benchmark Novelty: The method of using SOTA specialist models is not novel. However, the paper tackles the efficiency and effectiveness of MQA benchmarks by creating a challenging evaluation framework which successfully exposes the weaknesses of latest MLLMs model shown in Table.1.

•	Strong Performance: The proposed LLaVAction model demonstrates its significant performance gain over GPT-4o, LLAVA-OV, and LLAVA-Video on the new benchmark and also shows SOTA performance and strong generalization abilities on origin EK100 dataset and etc.

•	The paper is well-written and easy to follow, the model itself is concrete and effective: 1. Using learnable action taken with independent heads and utilize 2. Two-stage structured prediction pipeline that narrow downs the candidates choice.

### Weaknesses
•	Using strong external models as support: The construction relies on specialized model and other action recognition model as strong support. It might create confusion that dependence on TIM and other action recognition model will effectively improve the result rather than improvement from the model itself.

•	Possible OOD problem:
The paper does partially address my concern that they use AVION as training support vs TIM as evaluation. However, the reported 65% top-1 agreement and 45% top-5 overlap between AVION and TIM’s results on EK-100 evaluation set. The out of distribution itself will not proofed by claiming of using two different mining models. The paper might need a more robust way of concluding the OOD Distractor by possibly including a domain test on EK-100.

•	Novelty for two-stage pipeline: The paper claims the two-stage pipeline predictor as one of their contributions.  While in Section 3.3, the paper demonstrates they use a specialist model to generate Top-K candidates, it acts like a classifier for post-processing the output.

•	Specialized model confusion: In Section 4.3, the paper claims the task is zero-shot generalization. However, the specialized model is used to construct the candidate sets and also to ensure fair comparison against other model. It looks like a blur result rather than claiming as zero-shot since there is external help for inference.

### Questions
•	For the possible OOD problem, can you provide a clearer explanation or experiment to proof the current train/val setting remains OOD?

•	In your claim, the GPT-4o reformulation alone downgrades the MQA performance but the two-stage pipeline itself relies on GPT-4o for caption during training process. Can you elaborate the balance between benefits and harms?

•	How do results vary with K in both the miner and the two-stage pipeline? Is there a robust way to pick K beyond empirical tuning?

•	Minor notation issue: In Section 1.3, Formula 1 shows how the author formulate the MQA task. However, the author didn't definite p in the formula. Does it refer to the correct narration?

### Soundness
3

### Presentation
4

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
The paper introduces LLaVAction, a novel framework designed to evaluate and enhance the fine-grained action understanding capabilities of multi-modal large language models (MLLMs). The authors first reformulate the EPIC-KITCHENS-100 dataset into a video multiple-choice QA benchmark (EPIC-KITCHENS-100-MQA) using hard example mining with state-of-the-art specialist models such as TIM to generate challenging distractors. This benchmark exposes the current limitations of existing MLLMs (e.g., GPT-4o) in fine-grained action discrimination.
To improve MLLM performance, the authors curate a supervised fine-tuning dataset encompassing tasks like hard action recognition, temporal detection, captioning, and free-form QA. The proposed LLaVAction model introduces an action token to strengthen visual information utilization and a two-stage structured prediction pipeline for fair comparison with specialized models.
Empirical results show significant improvements—over 21-point accuracy gains on EPIC-KITCHENS-100-MQA compared to GPT-4o--and strong generalization across multiple benchmarks, suggesting that LLaVAction effectively enhances MLLMs’ action understanding.

### Strengths
Comprehensive evaluation across multiple datasets, benchmarks, and modalities.


Innovative benchmark creation using model-based hard example mining—a principled and efficient alternative to human-generated distractors.


Clear empirical improvement over strong baselines (e.g., GPT-4o, LLaVA-Video).


Thoughtful model design, integrating both architectural and data-centric enhancements.


Strong generalization across domains, including unseen datasets like EPFL-Smart-Kitchen-30 and Meccano.


Interpretability analysis (Section 4.5) provides convincing evidence of improved visual-text alignment.

### Weaknesses
While comprehensive, the paper’s novelty leans more on system-level integration than on introducing fundamentally new model architectures.


The action token mechanism—though effective—could benefit from deeper theoretical or ablation-based exploration (e.g., why it captures action semantics more effectively than other token-aggregation methods).


The reformulation pipeline is heavily reliant on existing specialist models (e.g., TIM, AVION), which could limit generalizability to non-egocentric domains.


The benchmark’s dependence on EPIC-KITCHENS-100 might introduce dataset bias, and expanding evaluation to non-cooking activities would further strengthen claims of generalization.

### Questions
Could the authors discuss how LLaVAction’s two-stage pipeline scales to open-vocabulary or long-horizon tasks beyond EPIC-KITCHENS?


Are there any failure cases where hard distractors cause misalignment between textual and visual cues?


How sensitive is LLaVAction to the choice of the base MLLM (e.g., Qwen2 backbone)?


Would the authors consider releasing the adversarially generated distractor datasets for community benchmarking?

In related works, the authors should talk about how their method compares to other methods like InsTALL [Nguyen et al., 2025 arXiv]. 

How do the results look with more traditional methods like ResNet3D, InceptionNet-3D (I3D), etc.?

### Soundness
3

### Presentation
2

### Contribution
2
