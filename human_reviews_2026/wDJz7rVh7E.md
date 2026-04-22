# LLaVA-RadZ: Can Multimodal Large Language Models Effectively Tackle Zero-shot Radiology Recognition?

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 4

## Abstract
Recently, Multimodal Large Language Models (MLLMs) have demonstrated exceptional capabilities in visual understanding and reasoning across various vision-language tasks.However, we found that MLLMs cannot process effectively from fine-grained medical image data in the traditional Visual Question Answering (VQA) pipeline, as they do not exploit the captured features and available medical knowledge fully, results in MLLMs usually performing poorly in zero-shot medical disease recognition.Fortunately, this limitation does not indicate that MLLMs are fundamentally incapable of addressing fine-grained recognition tasks.From a feature representation perspective, MLLMs demonstrate considerable potential for tackling such challenging problems.Thus, to address this challenge, we propose $\textbf{\textit{LLaVA-RadZ}}$, a simple yet effective framework for zero-shot medical disease recognition via utilizing the existing MLLM features.Specifically, we design an end-to-end training strategy, termed $\textit{Decoding-Side Feature Alignment Training ($\textbf{DFAT}$)}$ to take advantage of the characteristics of the MLLM decoder architecture and incorporate modality-specific tokens tailored for different modalities.Additionally, we introduce a $\textit{Domain Knowledge Anchoring Module ($\textbf{DKAM}$)}$ to exploit the intrinsic medical knowledge of large models, which mitigates the $\textit{category semantic gap}$ in image-text alignment.
Extensive experiments demonstrate that our LLaVA-RadZ significantly outperforms traditional MLLMs in zero-shot disease recognition, achieving the comparable performance to the well-established and highly-optimized CLIP-based approaches.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces **LLaVA-RadZ** which is a medical multi-modal large language model for zero-shot radiology disease recognition by mining the decoder’s penultimate-layer features with special tokens (global + local) and aligning them via symmetric cross-modal InfoNCE (**DFAT**). It further adds a **Domain Knowledge Anchor Module (DKAM)** that uses LLM-written disease descriptions as category prototypes, pulling image/text features toward disease-level semantics. On several chest X-ray benchmarks, the method is competitive with CLIP-style baselines in zero/low-shot settings and remains strong with 1% labeled data.

### Strengths
1) **DFAT**: Add modality-specific special tokens (⟨ImgCls⟩/⟨TxtCls⟩), extract penultimate-layer embeddings as global features, pool the rest as local features, and train with symmetric global and local InfoNCE to align image-text pairs.

2) **DKAM**:  Auto-generate disease descriptions with the LLM and map them into a semantic vector set; add a category-guided contrastive loss so image/text features align at the disease-category level.

3) **Novelty**: the proposed method improves MLLMs with SOTA CLIP-style models for zero-shot CXR classification

### Weaknesses
1) **Domain scope is narrow**: the study provides results for chest X-ray only. There is no CT, MRI, ultrasound, so the generalization of radiology data (including X-ray, CT, utrasound) is unproven.

2) **Zero-shot protocol**: the paper mainly focus on the capacity of **DOMAIN KNOWLEDGE ANCHOR MODULE**. However, the protocol for label mapping are unclear. For examples, **RSNA Pneumonia** is a single binary label dataset, how to mapping from the pre-trained anchors to the target labels set. Additionally, COVIDx CXR-2 is collected from COVID-19 patients which does not have compatible labels with classic pneumonia. How does the proposed method generate anchors or mapping labels across datasets? Zero-shot performances heavily depend on what labels are used for training. If the target labels are used for training, that is **target-aware labels supervision**.

3) **Goal clarity**: the proposed method focus on enhance one ability of MLLM for one data modality, instead of improve the general abilities of MLLM for one specific disease or image modality. For a **radiology** MLLM meant to answer all relevant CXR questions such as grading, disease recognition, localization, or generating report or answer VQA questions. Please clarify your goal?

4) **Interpretability**: the proposed method does not provide explanation or evidence localization. for the final diagnosis. In clinic setting, the explanation or evidence is very important for radiologist review.

### Questions
1) Performance is sensitive to special-token counts and depth of hidden layers. Please quantify or provide explanation about it?

2) Please provide more details for compute, memory usage, and latency versus other baselines.

3) Please explain if the other baselines are fine-tuning on the same dataset with the proposed method.

4) Please clarify how the anchors are generated for the target datasets.

5) how sensitive are results to prompt wording, anchor length, and LLM choice for generating disease descriptions?

6) can the proposed method handle with label-space shift (new disease unseen in the training anchors)

7) How are ontology mismatches handled? Please refer to the Weakness section

8) Do you used fixed anchors derived from the training dataset's labels set? or do you include the target datasets' labels into the training anchors?

9) what are the impacts of **temperature**?  Please provide an ablation study

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
3

### Summary
This paper proposes LLaVA-RadZ, a simple framework for zero-shot radiology disease recognition that leverages medical MLLM-based features instead of the standard VQA pipeline.

### Strengths
1. The paper has a clear and coherent narrative flow: it starts from the problem formulation, introduces the proposed method, and then validates it through extensive experiments. 
2. The experimental results are well-presented and show competitive performance. 
3. The method is also evaluated on multiple datasets, together with thorough ablation studies to justify the design choices.

### Weaknesses
1. The methodological novelty is relatively narrow. The framework mainly adapts an existing MLLM with feature-alignment tricks and prompt tokens for zero-shot classification, which feels incremental rather than conceptually new. The core idea—repurposing MLLM features for contrastive alignment—is more of an engineering extension than a fundamentally new direction.
2. The problem formulation underutilizes the capacity of modern MLLMs. Given that MLLMs are designed for rich medical reasoning and compositional understanding, constraining such a large model to a basic zero-shot disease classification task seems mismatched with its strengths and does not convincingly motivate why MLLM reasoning ability is necessary here.
3. Fairness of comparison remains unclear. The parameter scale of MLLMs is fundamentally different from CLIP-based baselines, making the comparison difficult to interpret. The gains reported against optimized medical CLIP-style models (e.g., MAVL) are often modest or inconsistent across datasets, and in some cases marginal in absolute terms, which raises questions about practical significance and whether the additional model complexity is justified.
4. From a clinical utility perspective, the improvements may be limited. The work focuses on common radiology benchmarks rather than clinically high-value or safety-critical scenarios, and several metrics exhibit only small deltas over strong prior art. Without demonstrating improved interpretability, robustness, or real-world benefit, the incremental accuracy gains translate weakly to actionable clinical impact.
5. The computational and training overhead is non-trivial. Leveraging a full MLLM backbone and LoRA finetuning introduces substantial resource cost compared to more lightweight radiology-specific encoders. The paper lacks a clear cost–benefit discussion, and the additional complexity may not justify the marginal performance improvements for practical deployment settings.

### Questions
See Weaknesses.

### Soundness
3

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
The paper proposes LLaVA-RadZ, a framework designed to enhance zero-shot medical image recognition using multimodal large language models (MLLMs). It introduces two main components:

First, DFAT, which extracts and aligns multimodal features from the decoder side using specially designed tokens <ImgCls> and <TxtCls>.

Second DKAM: which integrates domain-specific semantic anchors generated from medical knowledge to guide cross-modal alignment.

The motivation is to bridge the performance gap between generative MLLMs and discriminative vision-language models in medical zero-shot settings.

### Strengths
1. Reasonable token design.
The introduction of <ImgCls> and <TxtCls> tokens is intuitive and aligns well with the goal of disentangling modality-specific information. The authors provide convincing evidence via clustering analysis and ablations, though I would appreciate more direct visualization of attention distributions to confirm that these tokens truly capture the intended modality cues.

2. Concept-level extraction of medical knowledge.
The DKAM module essentially performs a structured concept extraction and anchoring step for medical entities, which is conceptually coherent and consistent with similar attempts in recent multimodal medical reasoning works.

### Weaknesses
1. Performing zero-shot classification using MLLMs is a rather niche setting. In realistic medical scenarios, simpler and far more efficient CNN or vision transformer models would be more appropriate and interpretable for image classification tasks. Thus, the motivation for adapting a large MLLM to this task feels weak.

2. Using contrastive learning for adaptation appears superficial for an MLLM. Given that token-level tuning or adapter-based fine-tuning is feasible, the proposed approach feels like an indirect workaround rather than a deep integration into the model’s architecture.

3. The reported gains (e.g., in Table 2) are marginal compared to recent state-of-the-art classification models. The improvements, while consistent, are not substantial enough to justify the added complexity and computational cost of the proposed framework.

### Questions
1. In equation 1 and 2, why use "+" to represent concatnation? This is confusing.
2. Can you show us the attention hotmap for the special token introduced?

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
3

### Summary
This paper introduces the LLaVA-RadZ framework, a novel approach to zero-shot medical disease recognition using Multimodal Large Language Models (MLLMs). Despite the impressive performance of MLLMs in general tasks, the paper identifies that they struggle with fine-grained medical image recognition, particularly in radiology disease classification, due to issues with feature extraction and alignment between images and corresponding medical texts.

The paper proposes LLaVA-RadZ as a solution, which includes several innovative components:

Decoding-Side Feature Alignment Training (DFAT): A strategy designed to leverage the strengths of the MLLM decoder architecture by introducing special tokens for image and text modalities. This enables the effective extraction of global representations from both the image and the text.

Domain Knowledge Anchoring Module (DKAM): This module utilizes the inherent medical knowledge within the model to enhance the understanding of medical diseases and bridge the semantic gap between images and text descriptions. It helps align image-text pairs based on disease categories.

Cross-modal Contrastive Loss: To further improve the alignment between image and text features, this loss function optimizes both global and local feature representations across modalities.

### Strengths
Innovative Framework: The LLaVA-RadZ framework introduces a novel approach to zero-shot medical disease recognition by leveraging MLLM features directly. The inclusion of DFAT and DKAM for enhancing cross-modal alignment is a strong contribution, as it enables the model to utilize its inherent capabilities without task-specific fine-tuning.

Strong Results: The experiments provide compelling evidence that the LLaVA-RadZ model significantly outperforms traditional models, such as CLIP and other state-of-the-art methods, in various disease recognition tasks. This includes improved accuracy and F1 scores across multiple medical datasets, showcasing the model’s potential in practical medical applications.

Domain Knowledge Integration: The integration of domain-specific medical knowledge through the DKAM module effectively bridges the gap between image and text data, improving both feature alignment and category-level understanding. This enhances the robustness and interpretability of the model in medical contexts.

Scalability and Generalization: The framework’s ability to work in zero-shot settings and generalize across various medical disease datasets without requiring extensive labeled data makes it highly valuable for real-world medical applications, where such datasets are often limited.

### Weaknesses
Limited Domain Coverage: While the framework performs well on several benchmark datasets, a broader evaluation on other medical imaging datasets (e.g., CT scans, MRIs) or more diverse diseases would strengthen the claims about the generalizability of the model across different medical conditions.

Fine-Tuning Sensitivity: Although the framework shows improvement over standard methods, it still relies on fine-tuning to achieve the best results. A deeper exploration of how the model behaves in pure zero-shot settings without any fine-tuning would be beneficial, especially in terms of robustness to unseen diseases or rare conditions.

Model Complexity and Efficiency: While the results are promising, the computational complexity of the LLaVA-RadZ framework, particularly the use of multiple training strategies (DFAT, DKAM, contrastive loss), should be better analyzed. Understanding the trade-off between model complexity and performance is important for real-world deployment.

Interpretability: While the framework offers improved feature alignment, more explicit methods to interpret the model’s decision-making process could increase trust in clinical settings. Providing visualization tools or methods to explain how image features and text descriptions are aligned could further enhance the framework's usability.

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
