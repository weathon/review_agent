# OpenReviewer: Predicting Conference Decisions with LLMs and Beyond

- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
The rapid growth of AI conference submissions has strained the peer-review system, motivating interest in AI-assisted review. Yet it remains unclear how reliably such systems approximate human judgment, which relies on domain expertise and nuanced reasoning. To address this challenge,, we introduce \open, a model designed to directly pred conference acceptance decisions rather than generate full reviews. Using ICLR 2024–2025 data, we evaluate large language models (LLMs), vision–language models (VLMs), and interpretable statistical models. Results show that text-only LLMs with continual pre-training outperform multimodal counterparts, achieving up to 78.5\% accuracy on balanced datasets (vs.\ 50\% random baseline). White-box statistical models further provide interpretability through feature analysis, revealing that structural attributes (e.g., paper length, section balance, citation engagement) are consistently predictive. Beyond average accuracy, a confidence-stratified utility analysis shows that the top 10\% most confident predictions reach 92.92\% precision, enabling reliable triage of ``obvious” accepts and rejects while exposing areas of uncertainty. Overall, our findings demonstrate both the promise and limitations of AI-involved peer review: current models can reduce workload and aid submission reviewing, but fall short of reliably replacing expert judgment.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a model aimed at predicting conference acceptance decisions rather than generating full reviews, addressing the strain on peer review from the surge in AI submissions. Using ICLR 2024–2025 data, the study compares LLMs, VLMs, and interpretable statistical models, finding that text-only LLMs with continual pre-training achieve the best accuracy (up to 78.5%) and that structural features like paper length and citation engagement are strong predictors. A confidence-based analysis shows highly precise predictions (92.9%) for the most confident cases, suggesting the potential of AI to assist human reviewers.

### Strengths
- The paper is among the first to incorporate Vision–Language Models (VLMs) into automated peer review, extending evaluation beyond textual semantics to include visual understanding of scientific figures. Although performance lags behind text-only LLMs, this represents a notable step forward in the emerging field of AI-assisted reviewing.

- The authors present interesting and interpretable findings through large-scale feature-level statistical analyses using recent ICLR datasets. These analyses meaningfully reveal factors influencing paper acceptance, contributing to transparency and interpretability in automated review systems.

- Beyond reporting average accuracy, the authors perform a confidence-stratified utility analysis, showing that the top 10% of most confident predictions achieve over 92% precision, providing practical insights for triaging clear accept/reject cases.

### Weaknesses
- The model only predicts accept/reject outcomes without generating full reviews. This makes the system a black box: the reasoning behind its predictions remains opaque, and without textual feedback, authors gain little actionable insight on how to improve their work.

- The experiments are limited to 8,000+ ICLR 2024–2025 papers, focusing only on four categories (LLM, CV, RL, Theory). This narrow scope raises concerns about the scale and diversity of the dataset, potentially limiting generalizability.

- The problem formulation (predicting acceptance from the original submission text only, without considering peer-review or rebuttal interactions) seems simplistic and detached from the real peer-review process. Moreover, since a basic feature-based machine learning model already achieves 74.2% accuracy, the performance gain from large language models is modest.

- The proposed model, OpenReviewer, does not introduce an innovative architecture; it relies on standard continual pre-training (CPT) + fine-tuning pipelines without task-specific adaptation or new methodological insights.

- The comparison between VLMs and LLMs is not fully fair: LLMs receive the entire paper text, while VLMs are fed only the abstract and introduction, limiting the latter’s access to crucial information. Consequently, the conclusion about VLM inferiority and figure utility is not well supported.

- No zero-shot large model baselines (e.g., GPT-5, DeepSeek-V3.1, GPT-4o, DeepSeek-R1) are included, which would have provided a stronger reference for evaluating model capability without fine-tuning.

### Questions
1. Why do VLMs perform worse than LLMs? Is this primarily because VLMs were provided only with the abstract and introduction? If full-text input were given, would their performance improve? Given that VLMs should benefit from visual information in figures, this phenomenon deserves a detailed explanation.

2. Why are the experiments restricted to ICLR 2024–2025? Since ICLR data is openly available, it’s understandable to use it, but extending to additional years could better validate the model’s robustness.

3. The Re2 dataset [1] includes multi-year, multi-conference review and rebuttal data. Could the authors extend their analysis to this dataset to test the cross-conference and cross-year generalizability of their findings? Such an extension would greatly strengthen the paper.

4. How would zero-shot large language models (e.g., GPT-5, DeepSeek-V3.1, GPT-4o, DeepSeek-R1) perform on this task without any fine-tuning? A comparison could reveal whether fine-tuning truly adds value.

5. Since peer review and rebuttal interactions strongly influence acceptance outcomes, have the authors considered integrating rebuttal-phase information (e.g., reviewer–author exchanges, additional experiments, clarifications) into the predictive model? This would make the task more realistic and practically meaningful.

> [1] Re2: A Consistency-ensured Dataset for Full-stage Peer Review and Multi-turn Rebuttal Discussions.

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
4

### Summary
This paper introduces OpenReviewer, a novel framework designed to directly predict the acceptance decisions of academic conference papers (e.g., for ICLR) rather than generating full review text. It formulates the task as a binary classification problem (accept/reject). The study comprehensively evaluates various models, including text-only Large Language Models (LLMs), Vision-Language Models (VLMs), and interpretable statistical models, using data from ICLR 2024 and 2025. Key findings indicate that text-only LLMs, especially after Continual Pre-Training (CPT), achieve the highest accuracy (up to 78.5% on a balanced dataset). Furthermore, a confidence-stratified analysis reveals that the model's most confident predictions (top 10%) achieve over 92% precision, suggesting potential for reliable triage of submissions. The study also analyzes discriminative features through white-box models, highlighting the importance of structural attributes like paper length and citation engagement.

### Strengths
1. The paper provides a thorough comparative analysis across different model families (LLMs, VLMs, statistical models).
2. Moving beyond aggregate accuracy metrics, the paper introduces a confidence-based utility analysis.

### Weaknesses
1. The training data is too single, which may cause some bias in model training. For example, if only ICLR submissions are used as training data, then the model is likely to fit the ICLR conference's acceptance preferences. After all, we all know that different conferences have different acceptance preferences.
2. Your model almost completely fails to model the features of the paper (neither the model structure nor the prompt design reflects it). This makes using your model to predict acceptance a black box process. However, we all know that whether a paper is accepted or not depends not only on its text and image. Many other factors include whether the paper's motivation is reasonable, whether the method proposed in the paper is correct and innovative enough, and whether the experimental part of the paper is solid.
3. I think the reason why the performance of many settings does not reach the random score (i.e., 50%) is precisely because of the unexplainability and crudeness of your model design.
4. The paper lacks discussion and citation of related work, such as [1].

[1] Llms assist nlp researchers: Critique paper (meta-) reviewing.

### Questions
I'd like to know if you used your proposed OpenReviewer to predict your submission? Don't take it too seriously, it's just a joke :)

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces OpenReviewer, a framework for predicting paper acceptance decisions at AI conferences, specifically using ICLR 2024–2025 data. The approach leverages large language models (LLMs) with continual pre-training (CPT) and prompt-based fine-tuning, vision-language models (VLMs), and interpretable statistical classifiers to directly forecast binary outcomes (accept/reject) from paper content, including text, figures, and engineered features. Key contributions include empirical demonstrations of up to 78.5% accuracy for text-only LLMs on balanced datasets, insights into the limited benefits of multimodal inputs, and a confidence-stratified analysis highlighting triage potential (e.g., 93.09% precision in top-10% confident accept predictions). The work addresses the growing strain on peer review by proposing AI-assisted decision prediction rather than full review generation, with an emphasis on interpretability and workload reduction.

### Strengths
- The focus on binary acceptance prediction, rather than full review generation, is a relevant extension of prior AI-assisted review efforts (e.g., Sukpanichnant et al., 2024; Ye et al., 2024). It could inform author strategies and committee triage, aligning with documented challenges in review sustainability (Lawrence, 2022; Beygelzimer et al., 2023).
- The inclusion of LLMs (Qwen-3 variants), VLMs (Qwen2.5-VL, Gemma-3), and white-box classifiers (e.g., Random Forest at 74.2% accuracy) offers a broad comparative baseline. The statistical feature analysis (29 features across categories like structure and methodological rigor) yields interpretable signals, such as the predictive value of paper length and citation density.
- Continual pre-training (CPT) demonstrates clear gains over vanilla fine-tuning (e.g., 9.5% accuracy improvement at 8B scale), providing a useful methodological contribution for domain adaptation in generative models.
- The statistical models also perform well and even surpass the VLM models and sometimes text models. It suggested that the statistical models also provide actionable predictions and interpretations.

### Weaknesses
- Severe Limitations in VLM Design and Evaluation: A critical flaw is the VLM input configuration, which restricts analysis to only the abstract, introduction, and the first two figures per paper. This represents a minimal subset of the full manuscript, severely constraining the models' ability to assess holistic quality signals such as experimental results, appendices, or later methodological details—core elements in peer review. Consequently, VLM performance (e.g., 68.2% accuracy for Qwen2.5-VL) appears artificially inflated in "Image Helps" cases while ignoring potential gains from comprehensive multimodal processing. This ad hoc choice not only prohibits fair comparison to text-only LLMs (which use full anonymized body text) but also invalidates claims about multimodal superiority, rendering the VLM results unreliable and unrepresentative of real-world applicability.
- Dataset and Generalizability Constraints: The dataset is confined to ICLR submissions, with keyword-based subdomain partitioning excluding non-LLM/CV/RL/Theory papers and deferring broader analysis. The natural 34/66 accept/reject imbalance is partially mitigated by balanced subsets, but cross-conference validation (e.g., NeurIPS, ICML) is absent, limiting claims to a single venue.

### Questions
- Why limit VLM inputs to abstract/introduction and only two figures? What experiments justify this choice, and how would full-manuscript multimodal processing impact results?
- How do predictions vary with evolving conference criteria across years/venues? Could multi-conference data mitigate ICLR-specific biases?
- What bias audits were conducted on LLMs/VLMs, particularly for subdomain imbalances or structural proxies favoring certain writing styles?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper empirically investigates the capability of LLM/VLM on predicting conference paper acceptance decisions. Experimental results demonstrate the promise and limitations of LLM/VLM-based conference paper review.

### Strengths
1. The topic of leveraging LLM/VLM to predict conference paper acceptance decisions is interesting.
2. Extensive analysis is conducted to show the capability of LLM/VLM on paper review.

### Weaknesses
1. Although extensive experiments and analysis are conducted, I feel that Sections 4.4 and 4.5 are not that informative. The main topic of this paper is to investigate the capability of LLM/VLM on predicting conference decisions. What is the underlying reason for evaluating the performance of traditional machine learning methods?

### Questions
1. Why do we need to train the model with the vocabulary decoupled label loss instead of vanilla cross-entropy loss (just treat the answer as text)?

### Soundness
2

### Presentation
2

### Contribution
2
