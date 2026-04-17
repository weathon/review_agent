# Phase-aware Memory Thought for 3D Medical Image Report Generation

- Decision: Reject
- Scores: 0, 4, 4, 4

## Abstract
Multi-phase 3D contrast-enhanced imaging is indispensable for clinical diagnosis, yet current vision–language models (VLMs) inadequately capture temporal dynamics across imaging phases, thereby limiting their reliability in automated medical report generation. We propose the \textit{Phase-aware Memory Thought} (PhoT) framework, a novel paradigm that integrates temporal progression patterns in multi-phase CT with structured clinical reasoning. PhoT incorporates: (i) phase-aware pretraining to learn temporally aligned visual representations; (ii) parameter-efficient fine-tuning to adapt these representations for report generation; and (iii) a structured inference mechanism (“Phase of Thought”) that leverages diagnostic templates to enhance clinical fidelity. We curate and evaluate PhoT on a large-scale dataset comprising 12,230 multi-phase CT series from 61,332 patient cases. Experimental results demonstrate that PhoT consistently outperforms strong baselines in both retrieval and report generation, achieving superior accuracy and interpretability. This work establishes PhoT as a clinically grounded, temporally aware VLM, advancing automated diagnostic reporting in complex medical imaging scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper proposes PhoT, a framework for 3D multi-phase contrast-enhanced CT report generation. It comprises three main contributions: Phase-aware Medical Alignment, which learns unified representations across multi-phase CT sequences; Phase-aware Diagnosis Generation, which efficiently adapts these representations for report generation through an adaptor and a large language model; and Phase of Thought Inference, a structured inference paradigm that uses diagnostic templates to improve the interpretability and clinical relevance of the generated reports. The authors also curate a large-scale multi-phase CT dataset covering diverse anatomical regions, and experiments on this dataset demonstrate that PhoT consistently outperforms 2D and 3D baselines in both retrieval and report generation tasks.

### Strengths
It introduces a large-scale multi-phase 3D CT dataset, compares PhoT against numerous 2D and 3D baselines, and evaluates its performance across both retrieval and report generation tasks. In addition, thorough ablation studies demonstrate PhoT’s effectiveness across different foundation models and its robust generalizability on the external CT-RATE dataset.

### Weaknesses
1. The citation format is entirely inconsistent throughout the paper, which disrupts readability and does not follow the official ICLR template (Refer to template’s Section 4.1 Citations within the text).
2. There are multiple writing inconsistencies, such as redundant word usage (e.g., repeated ‘which’, line 175) and missing or unclear cross-references to the appendix (line 311).
3. In Section 3.2, the text states that the visual encoder is frozen but does not explain how the LLM is trained, while Figure 2 implies the use of LoRA, leading to inconsistency between text and figure.
4. Section 4.1.1 provides no justification for comparing PhoT with 2D baselines, which are not directly comparable to the 3D setting.
5. Section 4.1.2 reports only lexical metrics (BLEU, ROUGE, METEOR, BERT-F1) for report generation, omitting key clinical metrics (RadGraph-F1[1], RadCliQ[2], CheXbert[3], etc.), reducing evaluation credibility.
6. Section 4.2 introduces the CT-RATE dataset without describing its background or explaining its suitability for assessing generalization.
7. The paper lacks an Ethics Statement, which is required since it involves medical imaging data.

[1] RadGraph: Extracting Clinical Entities and Relations from Radiology Reports

[2] Evaluating Progress in Automatic Chest X-Ray Radiology Report Generation
[3] CheXbert: Combining Automatic Labelers and Expert Annotations for Accurate Radiology Report Labeling Using BERT

### Questions
- In Section 3.2 (Phase-aware Fine-tuning), the paper mentions fine-tuning for report generation. After this process, does PhoT retain any instruction-following capability to support the structured inference described in Phase of Thought (Section 3.3)?
- Given that PhoT is designed for 3D multi-phase CT imaging, why does Table 1 include comparisons with 2D models, which are not directly comparable to the 3D setting?
- What are the clinical metric results (e.g., RadGraph-F1, RadCliQ, CheXbert), and do they support the conclusions?
- What are the detailed LoRA parameters (e.g., rank, alpha) used in the fine-tuning process as illustrated in Figure 2?

### Soundness
1

### Presentation
1

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
This manuscript presents a Phase-aware Memory Thought (PhoT) framework on vision-language models (VLM), for integrating temporal progression patterns in multi-phase computed tomography (CT). This framework has three parts: phase-aware pretraining, parameter-efficient fune-tuning, and structured inference. PhoT is then evaluated on over 12,000 multi-phase CT series, outperforming baselines.

### Strengths
- (Originality) Two-step training of input transformer and adapter/LoRA fine-tuning

 - (Quality) Extensive evaluation across various retrieval set sizes

### Weaknesses
- Lack of details on CT phases. In particular, while  two-, three- and four-phase samples are described (presumably corresponding to the non-contrast, arterial, venous and delayed phases in Figure 1), it is unclear as to whether they only occur in fixed combinations (e.g. if two-phase, then non-contrast+arterial), and why different patients have different numbers of phases recorded (and if there is certain bias or distribution amongst patients with different numbers of phases)

 - Lack of details about the LLM used for phase-aware fine-tuning (LLaMA-3.1-8B?), including parameter settings

 - Lack of details on PhoT caption template development and optimization process, which is the defining contribution

 - Ablation study was performed against the VLM, and not the proposed framework steps (phase-aware pretraining, parameter-efficient fune-tuning, structured inference); also, it is unclear whether using additional performance metrics and datasets is generally considered under ablation

### Questions
1. In the Abstract, it is stated that 12,230 multi-phases CT series were collected from 61,332 patient cases. However, Section 4 appears to imply that these "patient cases" are actually individual phases, that may belong to the same patient. If so, it could be considered to rephrase the description, and possibly provide details of any patient exclusion(s).

2. In Figure 1, "Extractd" might be "Extracted".

3. In Section 3.1, it might be clarified as to whether the inputs for different individual phases $t$ are tagged (or zero-padded, where that phase does not exist for a particular series)

4. It might also be clarified as to whether the spatial dimensions (assumed one of which represents temporal phase length) are constant, for all phases.

5. Likewise, for phase aggregation, the treatment for non-existent phases in a series might be explained, especially as the final phase T appears possibly different across patients.

6. In Section 4.1.2, the significance of different test set sizes is unclear; it appears more common to examine the effect of different training set sizes. This might be clarified.

7. In Section 4.2, for the confidence interval analysis, it is unclear as to the appropriateness of computing SD over multiple test set sizes (instead of seeds). This could be further justified.

### Soundness
3

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
2

### Summary
This paper proposes the Phase-aware Memory Thought (PhoT) framework for automated report generation of multi-phase 3D medical images (primarily multi-phase enhanced CT). The method consists of three parts: phase-aware pre-training of multi-phase representations, efficient parameter fine-tuning to adapt to report generation, and a structured generation template based on "Phase of Thought." The authors evaluated the framework on a self-built large-scale multi-phase CT dataset, showing slight improvements in text metrics and retrieval tasks compared to several baselines, and argue that the framework can improve cross-phase semantic consistency and clinical interpretability.

### Strengths
This paper focuses on the practical needs of multi-phase CT, covering the entire pipeline from pre-training to fine-tuning to generation. The engineering implementation is solid, and the joint evaluation of report generation and retrieval tasks demonstrates the method's versatility to some extent. From problem setting to motivation explanation, it accurately grasps the clinical pain point of "phase consistency." With visualization and qualitative examples, it showcases the framework's potential value in cross-phase information aggregation and text stability.

### Weaknesses
1.	The system lacks innovation, and its core mechanism resembles an engineered combination of existing memory and attention mechanisms.
2.	The definition of phase and the source of supervision are unclear, making it difficult to directly verify the effectiveness of phase-awareness.
3.	Phase of Thought is closer to templated prompts, lacking verifiable reasoning modeling.
4.	"Phase awareness" and "thought" are merely semantic encapsulations, lacking clear task definitions or mathematical representations.

### Questions
1.	The paper describes a "phase-aware" module that captures dynamic changes in multi-phase CT scans, but it doesn't explain how these "phases" are defined or acquired. Are they manually labeled, timestamped, or learned by the model itself? If it's a self-learning mechanism, can it be proven that the learned phase distribution aligns with clinical definitions?
2.	The core idea of the "Phase of Thought" module is to mimic a physician's reasoning chain, but its implementation appears closer to structured prompt generation. Could the authors further explain what new learnable signals or explicit constraints this module introduces at the reasoning level?
3.	The model claims long-range memory capabilities, but the paper doesn't demonstrate performance changes with different video lengths or number of phases. Does this mechanism remain stable even with significant increases in the input sequence?
4.	The experimental improvement is relatively limited (approximately 1–2%). Were multiple random seeds or statistical significance validation performed? If not, could the authors provide a more reliable explanation of the performance variance?
5.	The report generation task uses common linguistic metrics such as BLEU and ROUGE. Have you considered incorporating structured assessments that better align with the characteristics of medical reports (such as RadGraph or entity consistency metrics) to more comprehensively support the conclusion of "improved clinical consistency"?

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
The paper proposes a new vision–language framework for generating medical reports from multi-phase 3D contrast-enhanced CT scans. They focus mainly on modeling temporal dynamics across imagig phases, which is crucial for clinical interpretation.
The model has three main components phase-aware pretraining to train a multi-scale Vision Transformer features and learn temporally aligned visual representations from multi-phase CT sequences. They then freeze the visual encoder and use a lightweight spatial adaptor to map visual tokens to a language model for report generation. Fnally, they introduce structured diagnostic templates that guide reasoning and improve report generation.

### Strengths
1. The explicit temporal learning seems interesting to allow the model to capture clinically important contrast progression patterns.
2. The model introduces a template-guided reasoning process that compared with generic chain-of-thought seems to provide better outcomes.
3. Reported results outperform several other models or baselines.
4. Excellent dataset size in this specific application.

### Weaknesses
1.	The clinical applicability of the problem is limited. Although modelling temporal data is important, capturing physiological details for contrast CT scans is not well motivated. Although different body parts or abnormalities react differently to amount of contrast in the body, the work does not support their assumptions with strong clinical arguments.
2.	Although the data size is excellent, it is limited in terms of diversity. In fact you mentioned the hospital name which you should not. This restricts diversity in imaging protocols and may limit generalization to other hospitals or scanners which significantly impact model performance.
3.	The paper lacks proper real clinical validation: yes the conference is not a clinical conference but relying on quantitative and simulated metrics (such as BLEU, ROUGE, GREEN) is not sufficient. Comparing the generated reports blindly by radiologists would add huge value to the results.
4.	The author claims memory based learning but what seems to be happening is sequential learning from multi phased data instead. The word memory is properly not the appropriate choice of word.
5.	There is confusion in the problem formulation in terms of chosen variable names, e.g., C is the number of channels in the tensor, C’ is the feature representation size, then in eq 8, you mention C(LLM), and on page 8, C’ is mentioned as channel dimension. There are other occurrences where you get lost in these variables.

### Questions
Check the weaknesses and address them please.

### Soundness
2

### Presentation
2

### Contribution
3
