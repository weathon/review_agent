# Reasoning-Enhanced Large Language Models for Molecular Property Prediction

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 6, 2

## Abstract
Molecular property prediction is crucial for drug discovery and materials science, yet existing approaches suffer from limited interpretability, poor cross-task generalization, and lack of chemical reasoning capabilities. Traditional machine learning models struggle with task transferability, while specialized molecular language models provide little insight into their decision-making processes. To address these limitations, we propose $\textbf{MPPReasoner}$, a multimodal large language model that incorporates chemical reasoning for molecular property prediction.
Our approach, built upon Qwen2.5-VL-7B-Instruct, integrates molecular images with SMILES strings to enable comprehensive molecular understanding. We develop a two-stage training strategy: supervised fine-tuning (SFT) using 16,000 high-quality reasoning trajectories generated through expert knowledge and multiple teacher models, followed by Reinforcement Learning from Principle-Guided Rewards (RLPGR). RLPGR employs verifiable, rule-based rewards that systematically evaluate chemical principle application, molecular structure analysis, and logical consistency through computational verification.
Extensive experiments across 8 datasets demonstrate significant performance improvements, with MPPReasoner outperforming the best baselines by 7.91\% and 4.53\% on in-distribution and out-of-distribution tasks respectively. MPPReasoner exhibits exceptional cross-task generalization and generates chemically sound reasoning paths that provide valuable insights into molecular property analysis, substantially enhancing both interpretability and practical utility for chemists.
Code is available at https://anonymous.4open.science/r/MPPReasoner-12687.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This submission focuses on the molecular property prediction problem. Prior methods exhibit poor cross-task generalization and a lack of interpretability. The author argues that the fundamental limitation shared by previous methods is the absence of effective chemical reasoning, where models need to analyze the molecule and explain the prediction under chemical principles. To address this limitation and introduce chemical reasoning to the model, the submission proposes MPPReasoner, a reinforcement learning (RL) approach to instill domain-specific reasoning ability to the multi-modal large language model (MLLMs). MPPReasoner consists of two training stages: supervised fine-tuning (SFT) for instilling reasoning ability to the model, and Reinforcement Learning from Principle-Guided Rewards (RLPGR) for further enhancing the chemical reasoning ability. The submission conducts extensive empirical studies and justifies the effectiveness of the proposed method.

### Strengths
- The submission focuses on the trendy LLM reasoning problem, effectively enhancing the model's reasoning ability on the molecular property prediction task by proposing a domain-specific reward to guide the training process. 
- The submission is generally well-written, with clear illustrations and tables.
- Extensive experiments have been conducted to provide a good insight into the components of the proposed method.

### Weaknesses
- The overall pipeline is similar to the existing framework [1], where the model is first SFT with distilled reasoning trajectories and followed by RLVR. The proposed RLPGR differs by employing domain-specific rule checks via RDKit. Further experiments are suggested to investigate the underlying effectiveness of the chemistry layer. For example, whether the reward truly contributes to the quality and chemical validity of the reasoning trajectories.
- The quality control of the reasoning trajectories for SFT could be further enhanced beyond rejection sampling. Since the submission emphasises the chemical reasoning ability, the data for SFT could be checked by the LLM-as-Judge to ensure the reasoning quality. In addition, the reward definitions are not clear. For example, the reasoning layer proposes to evaluate the reasoning quality from logic consistency and a comparative perspective. However, neither of which are not provides implementation details, such as the adopted keyword for the logical consistency reward. 
- The rationale for adopting an MLLM instead of a purely textual LLM requires further justification. Since SMILES strings already encode molecular structure information, the claimed benefit of incorporating 2D molecular images is not convincingly supported by experiments. Moreover, most domain-specific baselines (e.g., BioT5-Plus, MolecularGPT) operate purely on textual modalities, making the comparison potentially unfair. The authors should provide ablations or control experiments isolating the contribution of the visual modality.

[1] DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning. In Nature, 2025.

### Questions
1. Could you provide further discussion on the adaptation of MLLM instead of a purely textual LLM?
2. Could you provide discussion or experiments on the reasoning quality of the SFT data?
3. Could you provide more detailed information on the reasoning layer in Section 3.2.2?

### Soundness
2

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
4

### Summary
This paper proposes MPPReasoner, a framework for learning chemical reasoning in molecular property prediction. Specifically, it tackles the fact that existing models, such as GNNs or molecular LLMs, are limited in providing a chemical reasoning process while predicting. The proposed method uses a two-staged training pipeline. First, the model is supervised fine-tuned (SFT) on the ChemCoT dataset using curated instructions that embed task-specific knowledge. Second, it is further finetuned with GPRO using a composite reward that combines foundational rewards (answer accuracy and format compliance), reasoning rewards (logical consistency and use of few-shot examples), and chemical rewards (chemical concept correctness and structural analysis accuracy). The experimental results show that the proposed training algorithm outperforms task-specific specialists and LLM-based generalists on both in-distribution and OOD tasks. Further ablation study demonstrates the contribution of each reward component.

### Strengths
* The proposed reward is designed for comprehensively assessing the chemical reasoning process.
* The ablation study shows the effectiveness of each reward component.

### Weaknesses
* My major concern is the novelty. Sequentially training SFT and RL is a well-known technique for fine-tuning LLMs, and the constructed dataset is borrowed from prior work (ChemCoT). The novel aspect of this paper is the design of the reward function, but its contribution is limited.

* The claimed mitigation of limitations in existing approaches is not sufficiently persuasive. Recent molecular LLMs [1,2,3,4] provide informative reasoning processes. If such models demonstrate interpretability, the necessity of the proposed training algorithm should be clearly articulated. I recommend revising the overall motivation so that the limitations being addressed are aligned with the proposed reward design. For example, it would be better to present an analysis of whether the reasoning processes are well aligned with the final predictions or whether the predicted chemical concepts are correct, and then explain the motivation for the reward design on that basis.

* The general-purpose baselines are limited to GPT, DeepSeek, and the base model (e.g., Qwen2.5-VL). I suggest comparing against a broader range of models such as GPT-5, Qwen3, Gemma, and other fine-tuned multimodal LLMs [1,2,3,4].

* It is unclear why large vision–language models (LVLMs) are used as the baseline. Molecules can be represented in diverse forms such as text (e.g., SMILES) and molecular graphs. Nevertheless, the paper does not clearly justify why the proposed model needs to be built on top of an LVLM.

[1] Li et al., "Towards 3d molecule-text interpretation in language models." ICLR, 2024.

[2] Park et al., "LLaMo: Large Language Model-based Molecular Graph Assistant", NeurIPS, 2024.

[3] Kim et al., "Mol-LLaMA: Towards General Understanding of Molecules in Large Molecular Language Model", ArXiv, 2025.

[4] Wang et al., "TxGemma: Efficient and Agentic LLMs for Therapeutics", ArXiv, 2025.

### Questions
* Could the authors report the experimental results with greedy sampling (i.e., temperature = 0)
* Could the authors provide the experimental results on other CYP datasets such as CYP 3A4, 1A2, and 2C9?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Summary:
This paper introduces MPPReasoner, a multimoda large language model designed to predict molecular properties by incorporating chemical reasoning. The model is trained using a two-stage strategy involving supervised fine-tuning on expert-generated reasoning trajectories and a novel reinforcement learning method called RLPGR, which use rule-based rewards to evaluate the model's application of chemical principle. Experiments show that MPPReasoner outperforms existing models, particularly on out-of-distribution tasks, and generates chemically sound, interpretable explanations for its predictions.

### Strengths
Pros:
- The paper introduces MPPReasoner, a multimodal LLM for predicting molecular properties with chemical reasoning.
- The model achieved better perfromance compared with baseslines, especially on out-of-distribution tasks.

### Weaknesses
Cons:
- The framework primarily relies on SMILES and 2D molecular data. What if we want to consider the 3D molecular structural information as well?
- Ablation study on whether using 2D images helps is encouraged to be conducted.
- Computational cost may be an issue. Generating detailed, step-by-step reasoning paths creates more computational overhead than models that make direct predictions.
- There is also a quite related study that is worth discussing in the paper [1].

[1] Zheng, Y., Koh, H. Y., Ju, J., Nguyen, A. T., May, L. T., Webb, G. I., & Pan, S. (2025). Large language models for scientific discovery in molecular property prediction. Nature Machine Intelligence, 1-11.

### Questions
See Weaknesses

### Soundness
1

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces MPPReasoner, a multimodal large language model that aims to bring reasoning into molecular property prediction. The model integrates SMILES and molecular images using Qwen2.5-VL-7B-Instruct as its backbone and employs a two-stage training strategy - 1) Supervised fine-tuning (SFT) on 16 000 reasoning trajectories curated from expert knowledge and teacher models. 2) Reinforcement Learning from Principle-Guided Rewards (RLPGR).  Results across eight MoleculeNet and TDC datasets show performance gains over baselines.

### Strengths
1. Novel attempt at structured chemical reasoning: The use of rule-based verifiable rewards to check logical and chemical consistency is an interesting direction in molecular property prediction beyond standard RLHF.

### Weaknesses
1. No justification for the image modality: The paper gives no ablation demonstrating whether the 2D molecular images add value over textual SMILES alone. Since the same structural information is encoded in SMILES, the inclusion of images is redundant.
2. Missing transparency for teacher prompts and expert guidance - “Expert-Guided Task-Specific Generation” is crucial for reproducibility, yet the actual prompts are omitted from the appendix and code. Without these, it is difficult to assess whether reasoning quality stems from prompt design or model capability.
3. No evaluation on dataset correctness: The dataset correctness stems entirely from "teacher" models. No human/expert annotations are conducted to assess the quality. This is unlike ChemCoT, which has been curated by 13 PhD Chemistry PhD students with thousands of hours of annotation time. 
4. Logical consistency is assessed by GPT-4o using a rubric, not by human experts. This risks circular evaluation bias.
5. Evaluation fairness and significance: Competing models (e.g., GPT-4o, o3-mini) are zero-shot whereas MPPReasoner is fine-tuned, so Table 1 does not represent a controlled comparison. Moreover, no statistical tests (e.g. t-test) are reported; it is unclear if gains are significant.
6. Dataset is unavailable: At least a subset of the reasoning dataset is not provided with the submission. Without that, it is not possible to evaluate the quality of the dataset.

To summarise, there are major concerns regarding the reliability and correctness of the dataset. The significance of the results is unclear. The information available in the submission is not sufficient to enable faithful reproducibility.

### Questions
1.	Why are molecular images included? Can the authors show an ablation demonstrating improvement from images vs. SMILES alone?
2.	How were expert prompts designed? The appendix lists them as blank. Could you release a few illustrative examples?
3.	Are improvements statistically significant? Please report standard deviations or confidence intervals across random seeds.
4.	Could you release a subset of the reasoning trajectories be released for verification? Are there any expert validation statistics to judge the correctness of the reasoning paths ?

### Soundness
1

### Presentation
2

### Contribution
1
