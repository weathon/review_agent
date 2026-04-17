# Omni-IML: Towards Unified Interpretable Image Manipulation Localization

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 8

## Abstract
Existing Image Manipulation Localization (IML) methods rely heavily on task-specific designs, making them perform well only on the target IML task, while joint training on multiple IML tasks causes significant performance degradation, hindering real applications. To this end, we propose Omni-IML, the first generalist model designed to unify IML across diverse tasks. Specifically, Omni-IML achieves generalization through three key components: (1) a Modal Gate Encoder, which adaptively selects the optimal encoding modality per sample, (2) a Dynamic Weight Decoder, which dynamically adjusts decoder filters to the task at hand, and (3) an Anomaly Enhancement module that leverages box supervision to highlight the tampered regions and facilitate the learning of task-agnostic features. Beyond localization, to support interpretation of the tampered images, we construct Omni-273k, a large high-quality dataset that includes natural language descriptions of tampered artifacts. It is annotated through our automatic, chain-of-thoughts annotation technique. We also design a simple-yet-effective interpretation module to better utilize these descriptive annotations. Our extensive experiments show that our single Omni-IML model achieves state-of-the-art performance across all four major IML tasks, providing a valuable solution for practical deployment and a promising direction of generalist models in image forensics. Our code and dataset are available at https://github.com/qcf-568/OmniIML.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents Omni-IML, a generalist model for Image Manipulation Localization. Its core methodological contributions are threefold: a Modal Gate Encoder for adaptive modality selection, a Dynamic Weight Decoder with task-aware filters, and an Anomaly Enhancement module that uses box supervision to learn task-agnostic features. The proposed single-model solution aims to unify multiple IML tasks, marking a promising step towards generalist models in image forensics.

### Strengths
Unified image forgery locazation is an study-worthy problem, and this paper also presents a promising solution for this problem. The paper is well organized and easy to follow.

### Weaknesses
1, My main concern is which is the primary focus of this paper?  a unified framework or interpretable? both hold many pages in methodology, but their connection is low. The explanation module integrates the IML results and finetune a MLLM for explanation text. However, the explanation seems does not benefit the IML accuracy in return, leading to the explanation section is relatively separate from the main focus. To me,  the localziation accuracy  is more important than explanation for IML problem, so it is reasonable and intuitive to promote localizaiton accuracy with explanation guidance. 

2, Does the MLLM joinly tune with the localization networks?

3, how is the bbox supervision obtained in anmaly enhancement ?

4, During the dataset construction,  how to split the image to several tamperred objects from the binary mask? was it achieved with regulations or some algorithms?

5, As the localization mask has been given before generating the explanation, with this, the tampered region seem too simple. Besides, the artifacts description  is also too simple and general.

6, the parameter and inference time should be compared.

### Questions
see the weaknesses.

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
3

### Summary
The paper introduces Omni-IML, a unified model for interpretable image manipulation localization (IML) across four major domains: natural images, documents, faces, and scene texts. It addresses the limitations of task-specific IML models by proposing a generalist architecture with key components: a Modal Gate Encoder for adaptive modality selection (vision-only or vision+frequency), a Dynamic Weight Decoder for sample-adaptive filtering, and an Anomaly Enhancement module using bounding box supervision to highlight tampered regions. Additionally, the authors construct Omni-273k, a large dataset with 273k samples annotated via a chain-of-thoughts (CoT) pipeline using GPT-4o for natural language descriptions of tampering artifacts. An interpretation module based on a multimodal large language model (MLLM) is introduced to generate textual explanations.

### Strengths
1. Unified Generalist Approach: This is the first work to effectively unify IML across diverse domains (natural, document, face, scene text) in a single model, achieving SOTA results on multiple benchmarks. It provides a practical solution for real-world deployment, reducing the need for multiple specialized models and addressing high maintenance costs.

2. Large-Scale Interpretable Dataset: Omni-273k is a comprehensive, high-quality dataset covering all four IML domains, with structured JSON annotations for tampered content, positions, and artifacts. The CoT annotation pipeline improves quality over prior single-prompt methods, enabling finer-grained evaluation and supporting interpretable forensics.

3. Effective Architectural Innovations for Generalization: Components like the Modal Gate (for modality adaptation) and Dynamic Weight Decoder (for handling diverse tampering clues) enable robust performance across tasks. The Anomaly Enhancement module leverages box supervision to enhance task-agnostic features, and the simple interpretation module (using visual prompts) boosts reliability without altering the MLLM structure.

4. Strong Empirical Validation: Extensive experiments on representative benchmarks show minimal performance degradation from joint training, with Omni-IML outperforming domain-specific SOTAs. The work also highlights interpretability, a underexplored aspect in unified IML.

### Weaknesses
1. The CoT annotation pipeline, while effective, feels incremental and assembled from existing ideas (e.g., step-by-step prompting and self-examination in GPT-4o). It may not be necessary for low-level tampering artifacts (e.g., texture or edge inconsistencies), which are often straightforward and do not require repeated reasoning or context-heavy analysis, potentially overcomplicating the process without strong justification.

2.  Modules like the Modal Gate and Dynamic Weight Decoder rely on well-established techniques (e.g., gating mechanisms for modality selection, dynamic filtering akin to those in dynamic networks). This makes the contributions appear more as combinations of prior art rather than groundbreaking innovations, reducing the perceived technical depth.

3. The focus on low-level artifacts (e.g., font/alignment inconsistencies in documents) might limit generalizability to emerging tampering methods, such as those from advanced generative models. The evaluation could benefit from more diverse, real-world scenarios beyond the curated datasets.

4. While the CoT pipeline is claimed to improve annotation quality, there is insufficient evidence (e.g., human evaluations or comparisons to non-CoT baselines) to demonstrate its necessity, especially for simpler domains where context information is weak.

5. SThe interpretation module operates as a standalone component, not integrated with the localization process (e.g., the Dynamic Weight Decoder). This separation may reduce the model's ability to provide cohesive, context-aware explanations, potentially missing opportunities for joint optimization of localization and interpretability.

### Questions
1.  Why is repeated reasoning (e.g., self-examination) necessary for low-level artifacts like texture inconsistencies, which seem more perceptual than contextual? Could a simpler single-prompt approach suffice for document or scene text domains, and what ablation results support the added complexity?
2.  How do the Modal Gate and Dynamic Weight Decoder differ substantially from existing gated or dynamic mechanisms (e.g., in multimodal fusion or dynamic convolutions)? Can you provide comparisons to show that these are not just incremental adaptations?

3. In cases where tampering is subtle and low-level (e.g., no strong semantic context), does the visual prompt risk introducing hallucinations in the MLLM? What metrics were used to evaluate this, and how does it perform on edge cases like multi-instance documents?

4. Given the incremental nature of the pipeline, how does Omni-IML handle novel tampering methods not seen in the training datasets, such as those from future diffusion models? Are there cross-domain transfer experiments to validate true generalization?

5. Since the interpretation module is separate from the localization pipeline, how does this design choice impact the model's ability to provide consistent and context-aware artifact descriptions? Would integrating interpretability into the Dynamic Weight Decoder or other core components improve overall performance or coherence?

### Soundness
3

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
The paper proposes Omni-IML, aiming to unify four major image manipulation localization (IML) tasks—natural images, documents, faces, and scene text—within a single general model. The core architecture comprises three components: a Modal Gate Encoder (sample-adaptive modality selection), a Dynamic Weight Decoder (sample-adaptive decoder filters), and Anomaly Enhancement (box-level supervision to boost contrast of manipulated regions during training). In addition, the paper presents an explanation module that overlays the predicted mask as a visual prompt to an MLLM, and constructs a large-scale dataset with structured text annotations, Omni-273k. Experiments claim single-model SOTA across multiple benchmarks in the four IML categories with minimal degradation under joint training. The authors commit to releasing code and data.

### Strengths
1. Introduces the first practical general IML model, Omni-IML, addressing the long-standing issue that existing methods suffer significant performance drops under joint training.
2. Novel and effective architectural design: the Modal Gate Encoder elegantly resolves the “double-edged sword” of frequency features—very helpful for document images but potentially noisy for natural images—by adaptively selecting the optimal modality. The Dynamic Weight Decoder adjusts filters dynamically to cope with diverse manipulation types. The Anomaly Enhancement module leverages box supervision during training to strengthen anomaly cues without adding inference cost, which is a neat design.
3. Builds a large-scale, high-quality, and comprehensive dataset, Omni-273k, providing stronger foundations for the field, along with a novel automatic annotation pipeline.
4. Extensive experiments validate the effectiveness of the proposed method.

### Weaknesses
1. The main contribution is performance, but it comes with higher computational cost. As shown in Table 16, Omni-IML (152M parameters) is larger than most methods (e.g., DTD 66M, SparseViT 50M) and its inference speed (7.1 FPS) is slower than most methods. Considering practical deployment, such high computational cost could be a barrier.
2. The localization and explanation modules are independent. The explanation module simply uses the localization mask as a visual prompt; there is no end-to-end or error-correction feedback mechanism. If the localization module misses regions, the explanation module receives no prompt there, likely leading to explanation failure. This loose coupling limits the system from being a truly end-to-end interpretable solution.
3. The paper states a cross-domain sampling ratio of 10:5:2:20 for natural, document, face, and scene-text images, respectively. However, the scene-text training set has only 229 images yet receives the highest weight. Such extreme re-sampling may introduce generalization/bias side effects. The paper currently lacks sensitivity analyses or rationale justifying this choice.
4. Numerous writing and typesetting issues hinder readability. For example, typos in the first line of the intro (“Mmanipulated”, “servious”), the title of Figure 5 (“Anomanly Enhancement”), and “Singe and multiple” in Table 1.

### Questions
1. Can the overall architecture be light-weighted for deployment (e.g., smaller backbones, distillation, pruning)? How does the end-to-end latency compare to methods with similar or larger model sizes (e.g., PIM, UPOCR) under the same input resolution, given that the paper reports FPS only?
2. Can the localization and explanation modules be jointly trained, and what are the effects? How does the explanation module behave under missed detections? Any attempts at feedback from explanation to localization for error correction?
3. Can you provide justification for the 10:5:2:20 cross-domain sampling ratio? Given the very small number of scene-text samples, does heavy re-sampling cause overfitting or bias in practice?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces an unified model named Omni-IML for the localization and interpretation of image manipulation. Three novel modules are proposed in the Omni-IML, effectively addressing the task conflicts in the unified training. An interpretation module is proposed to improve model’s forgery interpretation accuracy by prompting the multi-modal large language model with the localization model’s prediction. The proposed Omni-IML achieves state-of-the-art average performance across four major image manipulation localization tasks.
This paper also contributes the Omni-273k dataset to support better training on image forgery interpretation, it is initially constructed by a chain-of-thought pipeline, effectively reducing human-washing cost. The proposed structured label format enables detailed and reasonable evaluation.

### Strengths
Unified generalist model across four IML domains
- A single model achieves strong generalization and SOTA performance on natural, document, face, and scene text tasks without per-task tuning.
- Addresses fragmentation and lack of generalization in prior task-specific solutions with a well-motivated unified design.

Sample-adaptive architecture design
- Modal Gate Encoder adaptively selects the optimal modality (RGB vs frequency+vision) per sample.
- Dynamic Weight Decoder introduces per-sample dynamic filters to mitigate joint-training degradation and improve feature fusion of low- and high-level features.
- Anomaly Enhancement module is cleverly designed to enhance forged-region contrast using box supervision, with empirical support in ablations.
- Visual prompts assist correct tampered-region identification, yielding more accurate artifact descriptions compared to prior methods.

The Omni-273k dataset as a substantive contribution
- Large-scale coverage across four major IML tasks, enabling comprehensive training and evaluation.
- Structured JSON annotations enable fine-grained, reasonable evaluation and analysis, improving over unstructured labels in prior work.
- Chain-of-thought automatic annotation pipeline enhances data availability and quality.

Extensive experimental validation
- Thorough experiments, ablation studies, and comparisons convincingly demonstrate the effectiveness of each proposed module and the unified system.
- Results consistently verify performance and generalization across diverse tasks.

Technical soundness and clarity
- Methodology is technically sound, with coherent component design and integration.
- Writing and paper structure are clear and easy to understand.

### Weaknesses
- Regarding the Modal Gate, other auxiliary modalities, except for frequency may produce better results. Also, a soft gate weighting fusion may be better than the hard gate in the current version.
- The model’s parameter size and computational burden are relatively heavy; it includes a multi-modal large model in both training and inference. 
- The font size figure2 and figure3 is too small. It is suggested to simplify the content and enlarge the font size.

### Questions
See the concerns raised in the 'Weaknesses' section.

### Soundness
4

### Presentation
3

### Contribution
4
