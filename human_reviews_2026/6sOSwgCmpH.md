# Learning Self-Critiquing Mechanisms for Region-Guided Chest X-Ray Report Generation

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 2, 8, 4

## Abstract
Automatic radiology reporting assists radiologists in diagnosing abnormalities in radiology images, where grounding the automatic diagnosis with abnormality locations is important for the report interpretability. However, existing supervised-learning methods could lead to learning the superficial statistical correlations between images and reports, lacking multi-faceted reasoning to critique the relevant regions on which radiologists would focus. Recently, self-critical reasoning has been investigated in test-time scaling approaches to alleviate hallucinations of LLMs with increased time complexity. In this work, we focus on chest X-ray report generation with particular focus on clinical accuracy, where self-critical reasoning is alternatively introduced into the model architecture and their training objective, preferred by the real-time automatic reporting system. In particular, three types of self-critical reasoning are proposed to critique the hypotheses of grounded abnormalities compared to i) alternative abnormalities, ii) alternative patient's X-ray image, and iii) potential false negative abnormalities. To realize this, we propose a novel Radiology Self-Critiquing Reporting (RadSCR) framework, which constructs the abnormality proposals for each localized abnormality region and verify them by the proposed self-critiquing mechanisms accordingly. The critiqued results of the abnormality proposals are then integrated to generate the completed report with interpretable diagnostic process. Our experiments show the state-of-the-art performance achieved by RadSCR in the grounded report generation and diagnosis critiquing, demonstrating its effectiveness in generating the clinically accurate report.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a self-criticism mechanism learning framework (RadSCR) for chest X-ray report generation, aiming to improve the clinical accuracy and interpretability of report localization. Unlike traditional models that rely solely on visual-text correlation training, RadSCR simulates the "self-reflection" behavior of radiologists during the diagnostic process, re-evaluating initially detected abnormal areas through three complementary mechanisms: based on surrogate anomalies, surrogate patient images, and critical verification of potential missed detections. The model employs a weakly supervised localization and retrieval-based report generation paradigm, embedding the self-criticism mechanism into the network structure to achieve end-to-end training, invoking the LLM inference chain when inference is not required.

### Strengths
This paper focuses on the core clinical behavior of "self-correction," transforming it into a learnable critical mechanism within the model. Methodologically, it combines multi-source contrastive learning with retrieval-based generation, improving anomaly localization accuracy while enhancing semantic alignment. Experiments cover three tasks (report generation, retrieval, and localization), providing comprehensive validation with reasonable indicator selection.

### Weaknesses
While the paper's overall design is clear and the results are convincing, there remains some ambiguity between the concept and implementation of its "self-criticism mechanism." The three current critical paths (substitute anomaly, substitute patient, and potential missed detection) are more like structured perturbations of the input space than a rigorous "self-reflection" process. Therefore, the model's critical learning ability still relies on a pre-defined data augmentation paradigm rather than genuine adaptive induction from errors. Furthermore, although these mechanisms appear complementary in the experiments, the paper fails to systematically reveal their interactions and lacks quantitative analysis of the relative contributions or sensitivities of different critical signals. This makes "multidimensional criticism" theoretically more like parallel regularization than organic collaboration.

### Questions
1.	Will the three critique mechanisms (surrogate anomalies, surrogate patients, and potential missed detections) in the paper adaptively adjust their importance or weights during training through model learning? If there are interactions between the mechanisms, can these relationships or independent contributions be quantified?
2.	While the three critique mechanisms have shown advantages in experiments, can their gains be validated through further ablation experiments, especially regarding the impact of different critique signals on performance?
3.	Currently, the model's self-criticism only functions during the training phase. Has some form of "dynamic self-correction" mechanism been considered for the inference phase to address new error types or unseen biases?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
In this paper, the authors present an improved approach to report generation utilizing the critique principle. The basic idea is to form a triplet (V,B,E) representing the visual information in a region V, bounding box B, and concept E (presumably the finding itself), and contrasting triplets (V',B',E') representing variations due to substitutions with incorrect findings, incorrect locations, and image content replacements to indicate if any other possibilities exist other than the correct triple (V,B,E). The actual report generation is then done by retrieving the relevant sentences using these triplets as keys and assembling the overall report.

### Strengths
The idea of setting up contrasts with incorrect placements and incorrect findings or image appearances that can indicate two different findings is in general a good idea. The results on standard datasets appear to outperform comparables on the dimensions analyzed which include regional grounding performance as well as accuracy of findings classification.

### Weaknesses
The description of the method could be improved by giving examples of each case to make it convincing. The critique principle should be explained clearly. What is the model architecture? Its inputs and outputs are not clearly explained. 

Overall this idea is similar to the one proposed in the recent paper on phrase-grounded fact-checking(Ref1) where the idea of synthetic data generation also explored real and fake finding cases but built instead a discriminative classifier to separate the findings. While that paper proposed it for inference-time checking, this paper is proposing a similar one for report generation itself. The number of findings examined is much more limited in this case, so the performance improvements perhaps cannot be compared. 

Comparison with newer models needs to be done ( GPT4o, XRAYGPT and many other representative decoder models including MAIRA-2).

1. "Phrase-grounded Fact-checking for Automatically Generated Chest X-ray Reports",  https://papers.miccai.org/miccai-2025/paper/3526_paper.pdf

### Questions
What happens if the above operations in critique generation overlap and produced redundant triplets?
Line 236 says all sentences with the same finding are concatenated. How does that help? Wouldn’t it make the encoding worse in representing the finding with a mixture of sentences?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work presents RadSCR, a new methodology for report generation on chest x-ray images using self-critiquing mechanisms. The method utilizes a triplet to retrieve text descriptions for the abnormalities detected by the visual model. This triplet comprises: the image representation obtained using a vision encoder; local information provided by Grad-CAM; and learned concept embeddings for each abnormality. The framework is trained end-to-end to align image and text embeddings using contrastive losses. The final report is generated by post- processing the retrieved phrases using an LLM. The results show that RadSCR not only surpasses the state-of-the-art for report generation but also improves the capabilities of the visual model to detect abnormalities.

### Strengths
· Presented a novel and sound methodology to combine concept learning with local information provided by post-hoc xAI techniques (Grad-CAM in this case) to improve report generation.
· Extensive comparison between RadSCR and the state-of-the-art methods across different metrics and datasets, showing significant improvements for report generation.
· Showed that using RadSCR on top of a visual encoder improved the abnormality detection performance when compared to an MLP.
· Performed an ablation study that gives more insights about how each part of the proposed method impacts the overall performance.

### Weaknesses
· Figure 1 should be improved; it does not give the reader a broad idea of how the method works.
· Some details about inference are hard to grasp, which raises concerns regarding the performance evaluation of the retrieval results. A schematic would benefit the understanding of the inference process.
· More details about the Clinical Efficacy metrics are missing. The target used for some of the metrics is also not clear.
· Even though the authors briefly discuss the work's limitations, they are unclear to the reader. That section should also be extended with further discussions.

### Questions
· In section 3.3, what is usually the sentence length l for the s_bert variable? Since this variable was created by concatenating all sentences for the same abnormality, its length raises concerns regarding implementation.

· When computing the metrics, are the ground truth annotations used as targets? Or the predictions for the visual model? Both cases would offer valuable information for the reader, since in the first one, it checks how the report matches the ground truth, and in the
second, how well it describes the visual model predictions.

· Regarding the inference, is the search space for the sentences only among the subsets of a given abnormality? For instance, if the model predicted abnormality A as present, is the search space for the sentences only the set of sentences that describe A, or are all the
sentences in the dataset? This would have implications for the retrieval results, since the first case would be easier.

· The authors mentioned the lack of localization for the false negatives as a limitation. Nevertheless, it is possible to apply Grad-CAM to the abnormalities that are not predicted. Did the authors try to use this information to estimate the negative box B0 better and overcome this problem?

Minor Comments:
In the related works, what do the authors mean when it is said that localizing anatomical parts is not precise enough for grounding?
Some misspellings in the text:
- Line 63: triplet is misspelt
- Line 224: detection is misspelt
- Line 258: replace production by product
- Line 305: positive sample and negative sample are misspelt
- Line 323, 326: approaches is misspelt
- Line 338: replace “by the following metrics” with “with the following metrics”
- Line 341: which also considers is also misspelt
- The term “in the sequel” appears in the text; it is recommended to replace it with the term “in
the following section” or “below”.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces RadSCR (Radiology Self-Critiquing Reporting), a region-guided framework for chest X-ray report generation that learns self-critiquing mechanisms end-to-end—without relying on costly test-time reasoning. It mimics radiologists’ diagnostic verification through three critiques: (1) comparing predicted abnormalities against alternative ones, (2) testing patient-specificity via other patients’ images, and (3) checking for potential false negatives. Validated region proposals are then used for retrieval-based sentence generation with an LLM decoder. RadSCR significantly outperforms state-of-the-art methods on MIMIC-CXR and related datasets in both clinical accuracy and abnormality localization, offering interpretable and reliable radiology reports.

### Strengths
The paper presents a highly original motivation inspired by the self-critiquing process of radiologists, bridging human diagnostic reasoning and machine learning. Instead of relying on expensive LLM-based reasoning, the authors design learnable self-critiquing mechanisms (alternative abnormalities, alternative patients, and false negative checking) that elegantly emulate how clinicians verify findings before writing reports.

The technical quality is strong: the proposed framework integrates region grounding, self-reflection, and retrieval-based generation into an end-to-end trainable architecture, supported by comprehensive experiments across multiple datasets.

In terms of clarity, the paper is well structured, with intuitive explanations, clear equations, and visual schematics that make the workflow easy to follow. The significance is substantial for both the medical AI and vision-language communities. It provides a scalable way to enhance interpretability and reliability in clinical report generation without the overhead of LLM reasoning. The self-critiquing paradigm could inspire broader applications in trustworthy generative modeling beyond radiology.

### Weaknesses
The main limitation of this paper lies in the design rationale and justification of the alternative feature mechanisms that form the foundation of the proposed self-critiquing framework. While the idea of emulating radiologists’ reasoning through alternative hypotheses is conceptually appealing, the paper does not sufficiently explain why these particular forms of “alternatives” (abnormality-level, inter-patient, and false-negative critiques)were selected, nor how they were parameterized or sampled during training.

For example, in Alternative Abnormalities Critiquing, the paper describes selecting an alternative abnormality embedding $E’_m$ from either negatively predicted classes or non-overlapping positive predictions. However, it remains unclear how this selection impacts the discriminative strength of the critiqu, whether the alternatives are chosen to be visually similar, semantically distinct, or dynamically learned during training. Without such clarification, it is difficult to assess the robustness, interpretability, and reproducibility of the proposed mechanism.

In Alternative Patients’ Images Critiquing, the model substitutes the visual feature V with that of a “randomly selected” patient. While this offers cross-patient contrast, it may introduce semantic and anatomical misalignment, as random images can vary significantly in projection type (PA vs. AP), patient posture, or imaging equipment characteristics. A more controlled sampling strategy (such as selecting patients with comparable acquisition parameters or anatomical embeddings) could yield more meaningful contrastive feedback. An analysis of the sensitivity of this mechanism to inter-patient variability would strengthen the empirical validity.

Regarding Potential False Negatives Critiquing, the current implementation averages embeddings of all predicted negatives into a single global representation $E_0$. This coarse aggregation may dilute subtle abnormalities and risks introducing false positives due to noisy global pooling. It is unclear how this global feature contributes to identifying missed abnormalities or whether it merely adds uninformative signals. A more structured approach, such as region-aware aggregation or uncertainty-guided negative sampling, might enable more precise critique and localization.

Conceptually, the idea of constructing contrastive or counterfactual pairs to improve visual-language grounding is not entirely novel. Prior work, such as [1] shares a similar objective of forming counterfactual pairs to enhance discriminative representation learning and reduce hallucination. The present paper would be strengthened by explicitly discussing this connection. To clarify how RadSCR’s self-critiquing differs conceptually and methodologically from counterfactual reasoning, what additional benefits it introduces, and how the two paradigms could complement each other.

[1] Li, M., Lin, H., Qiu, L., Liang, X., Chen, L., Elsaddik, A., & Chang, X. (2024, ECCV). Contrastive Learning with Counterfactual Explanations for Radiology Report Generation.

### Questions
1. In Alternative Abnormalities Critiquing, the paper states that the alternative embedding $E’_m$ is chosen from negatively predicted abnormalities or positive ones in non-overlapping regions. Have you examined whether “hard” (visually similar) versus “easy” (semantically distant) alternatives affect training stability or discriminative power?
2. Sampling Strategy for Alternative Patients
The paper describes replacing the current image feature V with that of “a randomly selected other patient.” How is randomness controlled here, do you ensure the same projection view (PA/AP) or similar patient metadata to avoid distribution drift?

### Soundness
3

### Presentation
3

### Contribution
2
