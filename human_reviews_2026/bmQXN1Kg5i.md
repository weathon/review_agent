# TumorChain: Interleaved Multimodal Chain-of-Thought Reasoning for Traceable Clinical Tumor Analysis

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Accurate tumor analysis is central to clinical radiology and precision oncology, where early detection, reliable lesion characterization, and pathology-level risk assessment directly guide diagnosis, staging, and treatment planning. Chain-of-Thought (CoT) reasoning is particularly critical in this setting, as it enables stepwise interpretation from imaging findings to clinical impressions and pathology-level conclusions, ensuring traceability and reducing diagnostic errors. Here, we target the clinical tumor analysis task and build a large-scale benchmark that operationalizes a multimodal reasoning pipeline, spanning findings, impressions, and pathology predictions. 
We curate TumorCoT, a large-scale dataset of 1.5M CoT-labeled VQA instructions paired with 3D CT scans, with step-aligned rationales and cross-modal alignments along the “findings → impression → pathology” trajectory, enabling standardized evaluation of both final accuracy and reasoning consistency.
We further propose TumorChain, a multimodal interleaved reasoning framework that tightly couples 3D imaging encoders, clinical text understanding, and organ-level vision-language alignment. 
Through cross-modal alignment and iterative interleaved causal reasoning, TumorChain grounds visual evidence, aggregates conclusions, and issues pathology predictions after multiple rounds of self-refinement, improving traceability and reducing hallucination risk.
TumorChain demonstrates consistent gains over strong unimodal and pipeline baselines in lesion detection, impression quality, and pathology classification, and successfully generalizes to the public DeepTumorVQA benchmark. Ablations validate the key contributions of interleaved reasoning and clinical CoT. Clinically, these advances lay the groundwork for reliable, interpretable tumor assessment to support real-world decision-making. To advance safe, explainable, and reproducible multimodal reasoning for high-stakes tumor analysis, detailed information about our project can be found on our project homepage at https://github.com/ZJU4HealthCare/TumorChain.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces TumorChain, a comprehensive framework for traceable clinical tumor analysis. The authors address limitations in current Medical Large Vision-Language Models (Med-LVLMs), which often lack specialized oncology reasoning, fine-grained 3D analysis, and interpretable decision-making. The authors propose: (i) TumorCoT-1.5M, a large-scale, multimodal dataset of 1.5 million Chain-of-Thought (CoT)-annotated Visual Question Answering (VQA) samples, built from 3D CT scans and corresponding radiology/pathology reports, focusing on five major digestive organs. (ii) TumorChain Model, a multimodal interleaved reasoning framework for stepwise reasoning from findings to impressions to pathology predictions. (iii) TumorChain-Eval, a specialized evaluation protocol that moves beyond conclusion accuracy to assess the stepwise logical consistency and clinical relevance of the generated reasoning chains. The paper demonstrates state-of-the-art performance on their TumorCoT benchmark and shows generalization on the public DeepTumorVQA dataset.

### Strengths
1. The idea of Chain-of-Thought (CoT) reasoning is timely for current Med-LVLMs.
2. The curated TumorCoT-1.5M dataset is a major contribution.
3. The proposed model outperforms state-of-the-art LVLMs.
3. The paper is well organized and easy to follow.

### Weaknesses
1. The iterative interleaved reasoning process, which involves multiple calls to the segmentation model and the LLM with progressively growing context, could be computationally expensive. A discussion of the inference time/latency compared to single-pass baselines would be valuable, especially for potential real-time clinical applications.
2. The data engine uses GPT-4o-mini/Claude/GPT-5 variants, CoTe also relies on an LLM scorer. Even with logic calibration, it is unsafe to rely on these models, as the authors show in their experiments that general-purpose LVLMs perform bad on these medical tasks. Providing human adjudication rates, inter-rater agreement, and sensitivity to the scorer model would be more convincing for people to use the proposed data, especially in this safety-critical domain.
3. Again, the experimental results are primarily quantitative, lacking human reader studies or case-level blinds to confirm clinical utility and error modes (e.g., missed small lesions, false positives in complex anatomies).

### Questions
1. For the benchmark setting, are the 9:1 splits patient-level and cross-institutional? How are radiology/pathology reports prevented from leaking textual supervision across splits?
2. How is the “calibrated uncertainty” computed and validated?
3. What are the exact backbones (3D encoder, LLMs) for 3B/7B? What is $P(\cdot)$’s architecture and $Cl(\cdot)$’s head? Any per-organ heads or class imbalance handling?
4. Any blinded reader study or qualitative error taxonomy? Examples where LLM-only baselines hallucinate but TumorChain corrects?

### Soundness
2

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
4

### Summary
This paper addresses the limited tumor-specific reasoning and lack of reasoning depth in current medical large vision-language models (Med-LVLMs), by introducing a standardized formulation of tumor reasoning that mirrors the oncology workflow—from radiology findings to impressions to pathology-level predictions. It constructs TumorCoT-1.5M, a large-scale dataset of 1.5M multimodal chain-of-thought (CoT) instructions aligned with 3D CT scans, and proposes TumorChain, an interleaved multimodal CoT reasoning framework that integrates 3D imaging encoders, segmentation experts, and large language models for organ-guided, iterative reasoning. The resulting performance on the TumorCoT and DeepTumorVQA benchmarks demonstrates substantial improvements in classification, lesion analysis, and reasoning traceability compared to existing Med-LVLM baselines

### Strengths
1. The creation of TumorCoT-1.5M, a comprehensive CoT dataset, provides a valuable resource for the community.
2. The proposed TumorChain framework, enables deep reasoning steps and achieves significant performance gains on the TumorCoT and DeepTumorVQA benchmarks

### Weaknesses
1. The modality of the dataset and the empirical experiment is limited to CT modality only, It remains unclear whether the approach generalizes to other medical reasoning tasks or modalities.
2. The CoT data generation are relying on LLMs, lacking Human Clinical Validation. This raises concerns about potential biases, and hallucination.

### Questions
1. Are baseline models trained on the training set of TUMORCOT-1.5M? If not, and your model is instead trained on it, this is not a fair comparison.
2. Is the CoT data validated by human experts?

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
3

### Summary
This paper builds a tumor‑centric, end‑to‑end multimodal reasoning pipeline for 3D CT that goes from radiology findings,impressions,pathology‑level conclusions. It first constructs TumorCoT‑1.5M (~1.5M CoT‑labeled VQA instructions paired with CT scans across five digestive organs) via a multi‑agent, knowledge‑graph–guided data engine, enabling standardized, step‑aligned supervision. It then introduces TumorChain—an interleaved reasoning framework that tightly couples a 3D vision encoder, an organ segmentation expert, a lightweight abnormality classifier, and an LLM to iteratively align global context with organ‑level ROI evidence, produce traceable chains of thought, and calibrate uncertainty. Finally, it proposes TumorChain‑Eval/CoTe, which parses CoTs into triplet‑based chains (Finding, Impression, Long Reasoning) for stepwise scoring. Experiments show consistent gains over strong generalist and medical LVLM baselines on TumorCoT and strong generalization to DeepTumorVQA, with ablations validating the value of interleaved reasoning and clinical CoT.

### Strengths
1. This paper introduces a reasoning paradigm aligned with clinical workflows, featuring a traceable closed-loop process of 'Findings,Impressions,Pathology'.
2. This paper presents TumorCoT, a large-scale dataset
of 1.5M CoT-labeled VQA instructions paired with 3D CT scans, with stepaligned rationales and cross-modal alignments along the “findings→impression→pathology” trajectory.
3.  This paper proposes TumorChain, a multimodal interleaved reasoning framework that tightly couples 3D imaging encoders, clinical
text understanding, and organ-level vision-language alignment.

### Weaknesses
1. The authors use a proposed multi-agent, knowledge-graph-guided engine to generate chain-of-thought reasoning, which may introduce bias. Could the authors conduct a small-scale CoT evaluation with medical experts to validate the reasoning quality?
2. Does the use of the segmentation model increase inference or training latency, potentially reducing efficiency? How do the authors address this issue?
3. The comparison with commercial models only includes GPT-5-Mini and Gemini2.0-Flash. How do the latest models, such as Gemini 2.5 Pro and GPT-4o, perform on this benchmark?
4. Table 1 shows that all baselines are non-CoT models. How would the proposed method perform when compared against CoT-enabled models? like LLava-COT?

### Questions
see Weaknesses

### Soundness
3

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
4

### Summary
The paper presents TumorChain, an interleaved multimodal chain-of-thought (CoT) framework for clinical tumor analysis on 3D CT. The authors also release TumorCoT-1.5M, a large CoT-labeled dataset spanning findings → impressions → pathology, and propose TumorChain-Eval for stepwise reasoning assessment. TumorChain integrates a 3D vision encoder, an organ segmentation expert, an auxiliary abnormality classifier, and an LLM that iteratively reasons over global and organ-level tokens. On TumorCoT and the public DeepTumorVQA, TumorChain outperforms strong generalist and medical LVLMs; ablations suggest CoT and the iterative organ-guided loop contribute meaningfully.

### Strengths
+ The TumorCoT-1.5M dataset is a highly valuable contribution to the field. A 3D, instruction-tuned dataset of this scale, specifically designed for traceable, multi-step reasoning in a high-stakes domain like oncology, is a significant achievement and will likely enable new research directions.
+ The TumorChain model's core mechanism, Iterative Interleaved Reasoning (IIR), is an effective way to handle the high dimensionality of 3D medical images.
+ TumorChain achieves top performance across subtasks and generalizes to DeepTumorVQA; ablations show the value of CoT/IIR and of the auxiliary classifier.

### Weaknesses
- The dataset relies heavily on an MLLM-based data-generation pipeline. This introduces a risk of factual errors, bias, or hallucinations. The paper does not sufficiently discuss the quality control process for this generated data, beyond mentioning expert reviews. Furthermore, critical details on data splitting (e.g., ensuring splits are at the patient level to prevent leakage) are not detailed in the main text.
- The paper uses terms like "causality-guided inference" and "iterative causal reasoning". However, the method appears to be an exemplary case of step-by-step reasoning that follows a clinical logic path (Findings $\rightarrow$ Impression $\rightarrow$ Pathology), rather than a formal causal inference model (e.g., using structural causal models, do-calculus, etc.).
- The main results in Table 1 compare the TumorChain model (trained on the new 1.5M TumorCoT dataset) against baselines (like RadFM, Lingshu, GPT-5) that were not trained on this data. This makes it impossible to disentangle the performance gains from the novel architecture versus the massive, high-quality dataset. A fairer comparison would require fine-tuning the baseline models on TumorCoT.

### Questions
(1) Can you please clarify the data-splitting strategy for TumorCoT-1.5M? Are the training, validation, and test sets split at the patient level to prevent data leakage?

(2) The CoT data engine is impressive. Beyond the expert reviews mentioned, was there a formal, quantitative evaluation by clinicians to measure the factual accuracy and clinical plausibility of the LLM-generated reasoning chains?

(3) For the open-ended report generation task, what mechanisms are in place to prevent the model from hallucinating pathological findings that are not grounded in the visual evidence?

(4) Would you consider providing results for a baseline (e.g., Qwen2.5-VL-7B) fine-tuned on TumorCoT to create a more direct comparison?

### Soundness
2

### Presentation
3

### Contribution
3
