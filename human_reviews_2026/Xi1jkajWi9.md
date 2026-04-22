# CerebraGloss: Instruction-Tuning a Large Vision-Language Model for Fine-Grained Clinical EEG Interpretation

- Avg Score: 5.60
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6, 6

## Abstract
Interpreting clinical electroencephalography (EEG) is a laborious, subjective process, and existing computational models are limited to narrow classification tasks rather than holistic interpretation. A key bottleneck for applying powerful Large Vision-Language Models (LVLMs) to this domain is the scarcity of datasets pairing EEG visualizations with fine-grained, expert-level annotations. We address this by introducing CerebraGloss, an instruction-tuned LVLM for nuanced EEG interpretation. We first introduce a novel, automated data generation pipeline, featuring a bespoke YOLO-based waveform detector, to programmatically create a large-scale corpus of EEG-text instruction data. Using this data, we develop CerebraGloss, the first model of its kind capable of unified, generative analysis—performing tasks from detailed waveform description to multi-turn, context-aware dialogue. To evaluate this new capability, we construct and release CerebraGloss-Bench, a comprehensive benchmark for open-ended EEG interpretation. CerebraGloss demonstrates strong performance, surpassing leading LVLMs, including proprietary models like GPT-5, on this benchmark and achieving a new state-of-the-art on the TUSZ seizure detection task. Models, benchmark and tools are available at https://github.com/iewug/CerebraGloss.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
CerebraGloss introduces an instruction-tuned large vision-language model (LVLM) for fine-grained clinical EEG interpretation. The paper contributes (i) a programmatic data engine that converts raw EEG into structured annotations via a bespoke YOLO-based waveform detector (CerebraGloss-YOLO) plus background/artifact analyzers; (ii) a two-stage post-training pipeline on top of Qwen2.5-VL (3B/7B) that first aligns EEG visual concepts and then performs instruction tuning for generative and conversational tasks; and (iii) CerebraGloss-Bench, a 90-segment, expert-verified benchmark assessing descriptions, complex MCQs, QA, and dense channel-wise waveform detection. Empirically, CerebraGloss surpasses general LVLMs (including GPT-5) on the new benchmark (e.g., MCQ 80.0%, ROUGE-1 44.19, QA 4.76/10 judged by GPT-5) and achieves new SOTA balanced accuracy on TUSZ seizure detection, while remaining competitive on HMC sleep staging. Ablations support the two-stage design (notably very short Stage-1 alignment) and suggest scaling benefits at 7B.

### Strengths
*  Reframes EEG analysis from narrow classification to unified generative interpretation with multi-turn dialogue; introduces a YOLO-based waveform detector tailored to channel-wise EEG graphoelements; offers a practical two-stage alignment/tuning recipe that is efficient (≈4h per stage) and scales to 7B. 
*  Solid multi-facet evaluation: (i) CerebraGloss-Bench with free-text, MCQ, QA, and detection; (ii) standard clinical tasks (TUSZ/HMC); (iii) ablations on Stage-1 duration, Stage-2 composition, and model size. Clear reporting of balanced accuracy for imbalanced tasks and $mAP@0.5$ for detection. 
*  Pipeline is clearly described with concrete examples (e.g., montage, artifacts, spindles, K-complexes), plus a concise clinical EEG primer that defines background rhythms and graphoelements. Limitations and ethics are explicitly discussed. 
* The data engine and benchmark could catalyze research on general-purpose neuro-intelligent system*; reported SOTA on TUSZ indicates that the generative framing and instruction tuning can still deliver strong discriminative performance. Planned open-sourcing increases community value.

### Weaknesses
While the system is competently engineered, it largely follows a conventional recipe—fine-tuning a general-purpose foundation model on a domain-specific corpus with task-aligned instructions—without introducing a clearly new algorithmic idea. The components (instruction data curation, adapter/LoRA-style tuning, and standard loss/training schedules) appear incremental, and the paper does not articulate a principle or mechanism that would generalize beyond this application. As a result, the contribution feels more like a careful system instantiation than an advance in learning algorithms.

### Questions
1. Can the authors report per-class precision/recall (or error taxonomies) for **CerebraGloss-YOLO** against expert labels on a held-out set, and estimate the noise rate in the generated instruction data? This would contextualize hallucination rates. 
2. Beyond GPT-5 judging, did clinicians perform blinded scoring (with inter-rater κ) on a subset of QA/description outputs? If not, could the authors add a small human study in the rebuttal? 
5.  Can the model surface **calibrated uncertainty**, highlight **low-confidence regions**, or defer to human review automatically? This is especially important given acknowledged hallucinations and ethics.

### Soundness
3

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
4

### Summary
This paper introduces CerebraGloss, a large vision-language model (LVLM) instruction-tuned for clinical EEG interpretation. The key innovation is treating EEG waveforms as visual data and adapting LVLMs to interpret them through natural language. The authors develop an automated data generation pipeline featuring CerebraGloss-YOLO for waveform detection, create a large-scale instruction dataset (94K examples), and establish CerebraGloss-Bench as the first benchmark for open-ended EEG interpretation. The model achieves state-of-the-art performance on seizure detection (TUSZ) but not sleep staging (HMC) and shows convincing results on their novel benchmark (but no comparisons are available).

### Strengths
- Shifts from narrow classification to comprehensive, interpretative analysis of EEG, mimicking expert clinical practice
- Novel approach in treating EEG data as images
- CerebraGloss-Bench fills a critical gap with 90 expert-validated segments covering diverse clinical phenomena 
- New SOTA results on seizure detection (but not sleep staging)
- Includes ablation studies on training stages, data composition, and model scaling
- Open-source commitment: Plans to release model, benchmark, and tools to advance the field
- Nice additional supplementary material for clinical context, more clinical examples and useful information on used prompts

### Weaknesses
- The practical interpretability of “neuro-glosses” remains somewhat subjective, and quantitative evaluation of explanation quality is limited.
- Primarily applies existing LVLM techniques (Qwen2.5-VL) to a new domain without substantial architectural innovations
- Treats EEG as images rather than native time-series signals (a comparison with methods operating directly on the time-series data would be helpful)
-  CerebraGloss-Bench might be too small with 90 expert-validated segments (for robust validation)
- Heavy reliance on automated pipeline introduces noise; authors acknowledge hallucination issues from false positives (also pointed out as limitation by the authors)
- Limited set of comparisons (regarding multimodal or EEG models)

### Questions
- Validation: Have you conducted any pilot studies with neurologists comparing CerebraGloss interpretations to expert readings? What is the inter-rater agreement between your automated annotations and expert clinicians on a held-out validation set?
- Have you conducted systematic error analysis on CerebraGloss-YOLO's outputs? What are the most common failure modes?
- Could you provide confusion matrices showing which waveform types are most frequently confused?
- Regarding potential hallucinations: Would incorporating uncertainty quantification improve clinical safety?
- Have you tested extending context beyond 10 seconds using sliding windows or hierarchical processing?
- Regarding ablation: Could you test the model without the Gemini-generated data to isolate the contribution of automated vs. LLM-augmented data?
- Why only 90 segments in CerebraGloss-Bench? Is this sufficient for robust evaluation?
- Further comparisons: Could you compare against multimodal medical models like Med-PaLM or BiomedGPT? What about domain-specific EEG models that don't use vision-language approaches?
- Can you elaborate a bit more on what specific challenges prevent direct time-series encoding and why you are in the favor of treating EEG data as images?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors introduce CerebraGloss, a Large Vision-Language Model (LVLM) that is instruction-tuned for the interpretation of clinical EEG. The authors' core claim is that this model moves beyond the field's current focus on narrow classification tasks (like seizure detection) to enable unified, generative analysis. This allows the model to generate detailed waveform descriptions and engage in multi-turn, context-aware dialogue about an EEG segment.

To overcome the lack of suitable training data, the authors developed a novel, automated data generation pipeline to create a large-scale corpus of EEG-text instruction data. To evaluate the model's new capabilities, they also introduce and release CerebraGloss-Bench, a benchmark for open-ended EEG interpretation. The authors demonstrate that CerebraGloss outperforms 'off-the-shelf' LVLMs (including proprietary models) on this new benchmark and also performs strong classification on the TUSZ seizure detection classification task.

### Strengths
- Novel and valuable contribution: The paper aims to start a broader shift around machine learning for EEG. Moving from isolated classification to a unified, generative, and conversational model for EEG interpretation is a novel and interesting direction for the field - certainly worthy of exploration.
- Significant data generation effort: The development of the automated data generation pipeline, especially the bespoke detector for waveform events, represents a substantial engineering effort. This pipeline is a key contribution to addressing the primary bottleneck (lack of fine-grained, large-scale paired data) that has held back this line of research.
- New generative bnenchmark: The release of their benchmark is a welcome contribution. The field so far I believe entirely lacks benchmarks for evaluating generative and conversational abilities, and this dataset (though small) provides a much-needed tool for measuring progress in this new research direction.

### Weaknesses
- Limited classification evaluation: The paper introduces CerebraGloss as a model capable of both classification and generation. However, the classification evaluation seems limited. Given that the data generation pipeline sourced data from numerous public datasets (TUAB, TUEV, TUAR, etc.), it is surprising that the model's classification performance was not also benchmarked on tasks from these datasets (e.g., abnormality detection on TUAB or a more complex event classification from TUEV). This would have provided a much more comprehensive picture of its capabilities as a "unified" model.
- Missing comparisons: The related work section discusses prior work like Gijsen & Ritter (2025) but dismisses it, stating it "struggles to ground textual descriptions." It is unclear if this claim is based on a direct comparison or is a hypothesis. This work (together with EEG-CLIP (https://doi.org/10.3389/frobt.2025.1625731) although that one is 'concurrent' so comparisons cannot be requested, yet citation seems appropriate), seems like an important baseline. It is unclear why other existing models, which also bridge EEG and language, were not further discussed or included in the comparisons.
- Small scale of the new benchmark: While CerebraGloss-Bench is a really valuable type of contribution, its scale of only 90 segments seems small. This raises concerns about the statistical robustness of the evaluation results presented in Table 3 and how well they might generalize.

I would be happy to increase my score if the evaluations are extended to better support the claims about the classification ability of the model or if the paper's presentation is appropriately adapted to focus on the other strengths of the authors' work.

### Questions
- Why was the classification evaluation limited to TUSZ and HMC? Could the model's performance be reported on other standard tasks from the datasets you already used for data generation?
- Why do the authors not provide any comparisons with Gijsen & Ritter, 2025 (given that EEG-CLIP is technically 'concurrent work')? It seems the most relevant work to yours.
- Could you elaborate on the decision to limit CerebraGloss-Bench to 90 samples?  Given the small size, how confident are you in the benchmark's ability to robustly differentiate model performance?
- Do the authors have any future intentions to release any of the data annotations or to expand the benchmark?
- I'm wondering about the trade-off of presenting the EEG as an image versus operating in 'signal/timeseries' space. I presume it is necessary to move to images to be able to leverage the pretrained VLMs, but have the authors considered integrating a timeseries encoder with an LLM? I would not expect any analyses on this but it seems an important design decision about which it would be interesting to know the authors' reasoning.
- In the Stage 1 ablation study (Section 6.3), you find the optimal checkpoint is at just 0.05 epochs to avoid overfitting and "overwriting its powerful, pre-existing reasoning capabilities." This suggests the model starts to overfit almost immediately. Could you provide more intuition on why this overfitting on the "template-heavy captions" happens so rapidly, before even a single pass over the data? Is this visible in the model outputs?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces CerebraGloss, an instruction-tuned Large Vision-Language Model (LVLM) for nuanced interpretation of clinical EEG waveforms. The authors propose a full data-engine pipeline that generates synthetic EEG-text instruction data. The paper also introduces CerebraGloss-Bench, a benchmark of 90 EEG segments annotated for four evaluation formats (description, QA, MCQ, waveform detection). Experiments show improvements over general LVLMs (e.g., GPT-5, Qwen2.5-VL-3B) and new state-of-the-art results on TUSZ seizure detection.

### Strengths
1. EEG interpretation is a clinically valuable yet underexplored problem in multimodal AI. The authors frame the shift from narrow classification to generative interpretation.

2. The automated data engine is an engineering contribution. The combination of YOLO-based detection, spectral feature extraction, and rule-based/LLM caption synthesis is practical for a data-scarce field.

### Weaknesses
1. Although the pipeline is innovative, the core dataset is automatically generated with only light manual supervision. This raises concerns about noise propagation and factual validity in fine-grained annotations.

2. The authors suggest “occasional hallucinations” in generated interpretations in ethics statement, but more quantitative analysis of annotation accuracy or comparison to human expert labels would increase confidence.

3. CerebraGloss-Bench contains only 90 test segments.

4. Comparisons are restricted to general-purpose LVLMs or LLaVA-Med which is relatively out-dated. Some more recent and stronger specialized LVLMs should be included for comparison.

### Questions
1. More analysis on the generated data can be conducted. For example, the quantitative analysis of annotation accuracy or comparison to human expert labels.

2. Some stronger specialized baselines can be included.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents CerebraGloss, the first Large Vision-Language Model (LVLM)  fine-tuned for clinical EEG interpretation through instruction tuning. It transforms the traditional EEG interpretation task from a closed classification problem into a generative, multi-turn conversational explanation task, supported by an automated EEG–text instruction data generation pipeline and a new evaluation benchmark called CerebraGloss-Bench. The model demonstrates strong performance on standard datasets (TUSZ, HMC) and a custom benchmark, showing promising potential for both research and practical applications.

### Strengths
CerebraGloss represents an important step toward generative and multi-task EEG interpretation empowered by LVLMs

### Weaknesses
Data quality and validation are insufficient: The impact of noise in the automated data pipeline is not quantitatively analyzed. The paper lacks comparison with models trained on manually annotated data, making it unclear how noise affects performance. The representativeness of the data distribution is also not well explained—although the training data come from several public datasets, it is uncertain whether they cover different ages and pathological EEG types. The model’s generalization ability on heterogeneous data remains questionable.


Methodological depth could be improved: Converting EEG signals into images may lose important phase and amplitude dynamics, reducing diagnostic fidelity. The instruction tuning strategy heavily depends on an existing framework (Qwen2.5-VL), with limited architectural innovation introduced by the paper itself.


Writing style and tone are somewhat promotional: Some phrases sound overly promotional, such as “we pioneer” and “significantly outperforms GPT-5”. There are also minor spelling errors (e.g., “Descripition” should be “Description”).

### Questions
Evaluation lacks objectivity and completeness: The dialogue evaluation relies on GPT-5 scoring, which may introduce subjectivity and bias. No human expert evaluation is provided, making it hard to assess the true clinical relevance of the generated content. In addition, CerebraGloss-Bench contains only 90 EEG segments, which limits the statistical significance of the results.

### Soundness
2

### Presentation
4

### Contribution
4
