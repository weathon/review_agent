# So-Fake: Benchmarking Social Media Image Forgery Detection

- Avg Score: 6.00
- Decision: Reject
- Scores: 6, 4, 8, 6

## Abstract
Recent advances in AI-powered generative models have enabled the creation of increasingly realistic synthetic images, posing significant risks to information integrity and public trust on social media platforms. While robust detection frameworks and diverse, large-scale datasets are essential to mitigate these risks, existing academic efforts remain limited in scope: current datasets lack the diversity, scale, and realism required for social media contexts, and evaluation protocols rarely account for explanation or out-of-domain generalization.
To bridge this gap, we introduce \textbf{So-Fake}, a comprehensive social media-oriented dataset for forgery detection consisting of two key components. First, we present \textbf{So-Fake-Set}, a large-scale dataset with over \textbf{2 million} photorealistic images from diverse generative sources, synthesized using a wide range of generative models. Second, to rigorously evaluate cross-domain robustness, we establish \textbf{So-Fake-OOD}, a novel and large-scale (\textbf{100K}) out-of-domain benchmark sourced from real social media platforms and featuring synthetic imagery from commercial models explicitly excluded from the training distribution, creating a realistic testbed that mirrors actual deployment scenarios. Leveraging these complementary datasets, we present \textbf{So-Fake-R1}, a baseline framework that applies reinforcement learning to encourage interpretable visual rationales. Experiments show that So-Fake surfaces substantial challenges for existing methods. By integrating a large-scale dataset, a realistic out-of-domain benchmark, and a multi-dimensional evaluation protocol, So-Fake establishes a new foundation for social media forgery detection research.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces So-Fake, a large-scale benchmark aimed at evaluating forgery detection on social media imagery. It includes two core datasets — So-Fake-Set (2M images for training/validation) and So-Fake-OOD (100K out-of-distribution samples) — designed to simulate real-world generative diversity using varied GAN- and diffusion-based models. The accompanying model, So-Fake-R1, integrates reinforcement learning to jointly optimize detection, localization, and explanation tasks. Experiments demonstrate So-Fake’s realism, diversity, and diagnostic value for understanding model generalization.

### Strengths
1. Comprehensive dataset design: The paper presents a well-structured, large-scale benchmark that captures both in-domain and cross-domain conditions, using authentic social media content and realistic generation pipelines.
2. Unified evaluation and interpretability: The incorporation of reinforcement learning (GRPO) for multi-objective optimization across detection, localization, and explanation is technically sound and represents a meaningful advancement over prior benchmarks that treat these tasks independently.
3. Authentic OOD evaluation: So-Fake-OOD leverages real Reddit data paired with commercial generative models entirely disjoint from training, creating a realistic testbed that mirrors actual deployment scenarios with genuine distributional shifts.
4. Extensive ablation demonstrate the effectiveness of the proposed training recipe of So-Fake-R1.

### Weaknesses
1. Insufficient ablation on scalability: The paper lacks detailed analysis on how dataset size or class diversity impacts generalization performance, which would strengthen claims of robustness.
2. Potential domain limitations: While Reddit imagery provides diversity, reliance on a single platform may bias the dataset’s domain distribution and restrict its representation of global social media conditions.

### Questions
See weakness.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes So-Fake, which includes two datasets (So-Fake-Set and So-Fake-OOD) and a detector named So-Fake-R1. From the dataset perspective, it covers multiple data sources, 12 semantic classes, and adopts various generation methods for synthesis. Additionally, the OOD dataset collects a large number of images from social media platforms (e.g., Reddit) to simulate real-world scenarios. From the detector perspective, a reinforcement learning-based joint detector is proposed to simultaneously handle three tasks: detection, localization, and explanation. Experimental results show that the proposed method outperforms some existing works under the condition of using the same training set.

### Strengths
- Proposes a new dataset that covers richer semantic categories, synthesis methods, and real-world social media data.
- Presents a reinforcement learning-based detector with aligned rewards designed for the three tasks of detection, localization, and explanation.
- Conducts sufficient comparative experiments and ablation studies to verify the effectiveness of the proposed detector.

### Weaknesses
Dataset-related Issues:
- The paper claims that existing datasets have a "Narrow Categorical Scope", yet the proposed dataset does not significantly expand the semantic categories (only 12 categories as shown in Figure 2). Compared with existing datasets, what types of data are completely newly included? Furthermore, for the AIGC Detection task, is there a genuine need for such a large number of semantic categories? Does covering more semantic categories truly contribute to improving generalization performance?
- Regarding the claim of addressing "Outdated Generation Quality", similarly, is it really necessary to cover so many synthesis methods to achieve good out-of-domain (OOD) detection performance? Could a training set composed of specific combinations (e.g., StyleGAN3 + Latent Diff) yield unexpected OOD performance? Although the paper invests significant effort in creating new data using various synthesis algorithms, it fails to analyze whether these data are beneficial for building a detector with stronger generalization. Exploring the dataset from this perspective and deriving conclusions would better address future generation algorithms.
- Although many social media images are collected as OOD and real-scenario data, what are the differences between these data and other datasets? If they are more difficult to detect, what are the underlying reasons? Is it due to unique traces introduced by processing on social platforms? These questions are worth investigating.
- In Section 3.3 "Data Generation", the data generation approach seems similar to that of previous datasets. Are there any unique innovations? Otherwise, the mere application of different generation algorithms makes this section more like an experimental report rather than an academic research contribution.

Method:
- The techniques involved, such as cold start and GRPO training, are all based on existing works. Although there may be certain improvements in designing feedback rewards for the three tasks (detection, localization, and explanation) simultaneously, the core originality is insufficient.

Experimental Limitations
- All experiments in this paper are based on the proposed So-Fake-Set. Although the paper states that comparative methods are fine-tuned on this dataset, it cannot guarantee that the performance achieved through these fine-tunings is optimal. Adding cross-dataset experiments would better verify the effectiveness of the proposed method. For example, training the proposed So-Fake-R1 on the dataset introduced by CAT-Net (IJCV'22), then comparing it with methods like CAT-Net and TruFor on other OOD datasets.

Justification: Although the paper invests substantial effort in constructing the dataset, it fails to conduct effective information mining on the constructed data to derive innovative conclusions. While this dataset may serve as a baseline to inspire future work, it is difficult for me to give it a highly positive rating at this stage.

### Questions
Please see Weaknesses.

### Soundness
2

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
5

### Summary
This paper studies the task of forgery analysis on static images, including detection, localization, and anomaly explanation. Targeting the social media domain, where AI-generated images are widespread, the authors propose **So-Fake-Set** and **So-Fake-OOD**, along with a reinforcement learning-based baseline method named **So-Fake-R1**. Specifically, to address the scarcity of social media image data, the authors collect and construct a large-scale dataset and a challenging OOD benchmark tailored for social media images, which holds significant potential for advancing the community. In addition, the introduction of reinforcement learning into the forgery analysis task achieves impressive results.

### Strengths
1. The proliferation of AI-generated images on social media is indeed one of the pressing issues in today’s society. The proposed dataset effectively alleviates the data scarcity problem in this area and carries significant research value at the frontier of the field.  
2. With a simple reward function design, **So-Fake-R1** achieves state-of-the-art performance, opening up new possibilities for applying reinforcement learning to forgery image analysis tasks.  
3. The paper is clearly written, and the figures and visual analyses are comprehensive and easy to follow.

### Weaknesses
As a benchmark, the evaluation should be expanded to include as many Vision-Language Models (VLMs) as possible — for example, **DeepSeek-VL2** and several closed-source Multimodal Large Language Models (MLLMs) such as **GPT**. In addition, experiments should also be conducted across different model sizes to provide a more comprehensive assessment.

### Questions
Please refer to the *Weakness* section for detailed explanations.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces So-Fake, a large-scale benchmark for social media image forgery detection, and a unified baseline model So-Fake-R1. So-Fake provides diverse, realistic, and out-of-distribution data that reflect real-world social media conditions. Built upon this benchmark, So-Fake-R1 employs a two-stage training framework combining supervised fine-tuning and reinforcement learning (GRPO) to jointly perform these three tasks with structured reasoning outputs.

### Strengths
1.	This paper is easy to follow.
2.	Large-scale dataset. The proposed benchmark is significantly larger and more diverse than existing datasets, covering 12 semantic categories from real-world social media sources
3.	Well-designed OOD setting. So-Fake-OOD explicitly uses commercial generators that do not overlap with those in the training set, effectively simulating real-world scenarios where models must handle unseen generative tools.

### Weaknesses
Since So-Fake-OOD real samples come from Reddit and the training set aggregates from COCO and other sources, there might be near-duplicate or visually similar images across domains. The paper should report duplicate detection to ensure strict separation between training and OOD evaluation sets.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
