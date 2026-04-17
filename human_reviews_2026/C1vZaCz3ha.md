# CorBenchX: Large-Scale Chest X-Ray Error Dataset and Vision–Language Model Benchmark for Report Error Correction

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
AI-driven models have shown great promise in detecting errors in radiology reports, yet the field lacks a unified benchmark for rigorous evaluation of error detection and further correction. To address this gap, we introduce \textbf{CorBenchX}, a comprehensive suite for automated error detection and correction in chest X-ray reports, designed to advance AI-assisted quality control in clinical practice. We first synthesize a large-scale dataset of 26,326 chest X-ray error reports by injecting clinically common errors via prompting DeepSeek-R1, with each corrupted report paired with its original text, error type, and human-readable description. Leveraging this dataset, we benchmark both open- and closed-source vision–language models (\eg, InternVL, Qwen-VL, GPT-4o, o4-mini, and Claude-3.7) for error detection and correction under zero-shot prompting. Among these models, o4-mini achieves the best performance, with 50.6 \% detection accuracy and correction scores of BLEU 0.853, ROUGE 0.924, BERTScore 0.981, SembScore 0.865, and CheXbertF1 0.954, remaining below clinical-level accuracy, highlighting the challenge of precise report correction. To advance the state of the art, we propose a multi-step reinforcement learning (MSRL) framework that optimizes a multi-objective reward combining format compliance, error-type accuracy, and BLEU similarity. We apply MSRL to QwenVL2.5-7B, the top open-source model in our benchmark, achieving an improvement of 38.3\% in single-error detection precision and 5.2\% in single-error correction over the zero-shot baseline.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents CorBenchX, a large-scale benchmark for automated error detection and correction in chest X-ray radiology reports. The authors generate 26,326 error-injected reports from MIMIC-CXR covering five clinically relevant error types, paired with original text, span-level edits, and descriptions, and validated through a multi-stage quality-control pipeline. Nine vision-language models (VLMs), including open and closed, are evaluated under zero-shot settings which reveals that even the best model achieves only ~50% detection accuracy. This highlights the difficulty of precise medical report corrections. To improve performance, the paper proposes a Multi-Step Reinforcement Learning (MSRL) approach leveraging the GRPO method that sequentially predicts error type, describes the error, and corrects the report with tailored rewards. MSRL applied to Qwen2.5-VL yields substantial gains in both detection and correction metrics and generalizes to an out-of-distribution IU-Xray test set. Overall, the work provides a comprehensive benchmark and a structured RL method for advancing radiology report quality control.

### Strengths
* The paper addresses an important problem by shifting focus from radiology report generation to error detection and correction, which has direct clinical relevance. 

* It introduces a large and carefully constructed dataset with realistic clinical error types, span-level annotations, and a multi-stage validation pipeline including radiologist checks. 

* The paper proposes a multi-step reinforcement learning strategy that explicitly separates error identification, explanation, and correction, offering a structured and more interpretable alternative to conventional single-step prompting or RL methods. 

* An extensive evaluation is performed across nine state-of-the-art vision-language models with diverse lexical, semantic, and clinically grounded metrics, along with out-of-distribution testing, offering strong empirical support for the benchmark and method.

### Weaknesses
* The dataset relies entirely on synthetically injected errors via DeepSeek-R1, which may not faithfully represent real clinical error distributions. Including even a small real-world error set or expert-annotated subset for validation would strengthen ecological validity and reduce concerns about model overfitting to synthetic artifacts. 

* The OOD evaluation is constructed with the same error taxonomy and prompting style, leading to uncertainty about whether the demonstrated generalization reflects robustness or shared generation biases. 

* Although radiologists participated in dataset quality control, there is no human evaluation of model-corrected reports, making it hard to assess whether MSRL edits are safe, clinically faithful, and free from hallucinations. 

* The evaluation primarily includes general-purpose vision-language models (GPT-4o, Claude-3.7, Qwen2.5-VL) and omits medical domain–specialized VLMs such as CheXagent, LLaVA-Med, Med-PaLM, and BioViL-T, which are specifically trained on radiology or clinical datasets. Incorporating these domain-adapted models would provide a more rigorous baseline for assessing the difficulty of radiology report correction and clarify whether MSRL’s improvements stem from its learning strategy rather than general model capacity or domain mismatch.

### Questions
* All errors are injected by DeepSeek-R1. Would the method generalize if errors were introduced by a different VLM (e.g., GPT-4 or Claude) or by human annotators? 

* The RL objective uses BLEU for correction, which may penalize clinically appropriate paraphrasing. Did the authors experiment with RadGraph-F1 or CheXbert-based rewards during RL, or can they justify why BLEU was chosen over clinically grounded metrics for optimization? 

* Have the authors considered comparing against medical-tuned VLMs such as CheXagent, LLaVA-Med, Med-PaLM, and BioViL-T? 

* In Table 2, BLEU and ROUGE scores drop for sentence-level correction after applying MSRL to QwenVL2.5 models. Can the authors clarify why MSRL leads to reduced lexical-overlap metrics at the sentence level, while all metrics improve at the report level?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This manuscript introduces the CorBenchX benchmark, for automated error detection and correction in chest X-ray reports. A dataset of over 26,000 reports is synthesized via prompting DeepSeek-R1 to inject clinically common errors, with these error reports paired against the original error-free text as ground truth. The full CorBenchX benchmark benchmark of error reports is then applied to multiple common vision-language models, with o4-mini found to achieve the best performance on common text metrics. A multi-step reinforcement learning (MSRL) framework is then used with QwenVL2.5-7B to improve error detection further.

### Strengths
- (Originality) Large, curated X-ray report dataset generated by LLM for public usage

 - (Quality) Fairly comprehensive human-in-the-loop pipeline to minimize synthetic error report genration failure

### Weaknesses
- Minimal details about how actual errors are synthesized, for example what types of words are inserted for insertion errors, and what location/side text is recognized for side confusion errors etc.

 - Injection methodology does not appear to take actual X-ray image into account, and thus the errors may not reflect actual probabilities derived from image appearance (e.g. whereas the LLM may think that some injected error is medically possible, it may be extremely unlikely from the actual image)

 - Further, reliance on LLMs to insert errors by prompts does not automatically assure that errors are well-distributed; for example, for side confusion errors, what cardinalities (e.g. left, top right, etc.) are recognized/preferred?

### Questions
1. In the Dataset Source and Sampling subsection, it is implied that records where one of the "Findings" and "Impressions" sections are empty, may be sampled for synthetic error injection. It might be clarified as to whether reports with potentially nonexistent findings would be relevant.

2. In the Error Injection Procedure subsection, are the ratios between error types for single-error reports, and between single-error and multiple-error reports, empirically determined?

3. In the Quality Control Pipeline subsection, it might be briefly discussed as to whether the 900 additional reports flagged for review from the remaining approximately 24,000 reports, are in similar proportion to the number of imperfectly-generated reports from human review in Stage 1.

4. In the Multi-step Reinforcement Learning section, three separate queries (in three stages) are shown to be used in the MSRL GRPO training pipeline. It might be clarified as to whether this multi-query chain-of-thought promption was attempted for VLM evaluation.

5. In the Multi-step Reinforcement Learning section, errors in earlier stages would appear to affect later queries/stages. It might be clarified as to whether the training process provides the correct input for each query, regardless of the output for the previous query - or if each query is expected to attempt reward maximization on potentially incorrect inputs.

6. While there are separate prompts for single error and multiple error correction (Appendix B.1.1/B.1.2), this does not appear to be a realistic assumption for real-life application, since the total number of errors in a report should be unknown. As such, it may be more appropriate to apply the same multiple error prompt to both single and multiple-error tasks.

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
In this paper, the authors introduce a benchmark called CorBenchX, which is designed for automated error detection and correction in chest X-ray reports. The aim is to build a benchmark that can evaluate the extent to which off-the-shelf models are capable of detecting and correcting errors in radiology reports. Contributions include (1) a dataset with 26k reports where errors are injected via an LLM, (2) a systematic benchmarking study across open and closed-source VLMs, and (3) a reinforcement learning framework that optimizes a multi-objective reward to perform this task.

### Strengths
- This paper focuses on an important and underexplored application area - detecting and correcting errors in radiology reports. The work has potential for real-world impact.
- The paper is well-written and methods are clearly described
- The dataset will be a useful contribution to the community, especially as it is substantially larger than datasets included in previous work and was reviewed by expert radiologists

### Weaknesses
- **Settings without errors:** The dataset and benchmark focus on detecting errors in reports that have 1-3 injected errors, and this work does not consider settings where there are no errors. Being able to distinguish reports that have no errors from reports that have 1 or more errors is important for clinical utility.
- **Additional details on benchmark composition and need for more analysis:** While the authors provide some basic statistics on the benchmark in Table 1, the paper could benefit from (1) more descriptive statistics surrounding the composition of the dataset and (2) more detailed analysis into performance of benchmarked models across various stratifications.
    - For example, what  types of clinical findings are omitted, and are certain omissions easier than others for the benchmarked models to detect/correct? When mistakes in measurements are injected, how large are the deviations from the true values? What is the mean number of words modified per injected error? Generally, I find myself not having a clear picture of the composition of the created dataset.
- **Additional evaluations on reinforcement learning framework:** The authors state that their reinforcement learning framework is a key contribution of this work, yet the utility of this framework is not thoroughly evaluated.
    - In particular, the paper would benefit from additional ablations with respect to the design choices of this method. Additional evaluations of the trained Qwen+MSRL models on an external benchmark could also be useful.

### Questions
In addition to the weaknesses above, additional questions are listed below: 

- Could the authors provide further analysis on why different trends are observed for report-level vs. sentence-level metrics?

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
This paper introduces CorBenchX, a large-scale benchmark for automated error detection and correction in chest X-ray radiology reports. The dataset consists of 26,326 reports with both single-error and multi-error cases, synthetically generated from MIMIC-CXR using DeepSeek-R1 prompting. Each report includes annotations for error type, span, and human-readable descriptions. The authors benchmark nine VLMs and also propose a multi-step RL method that sequentially optimizes for format compliance, error-type accuracy, and correction quality. MSRL improves single-error detection precision by 38.3% and correction by 5.2% over baselines.

### Strengths
- a large and detailed benchmar for radiology report error correction, addressing gaps in existing resources
- Comprehensive benchmarking, lots and diverse comparison across VLMs with relevant metrics
- Multi-step reinforcement learning approach improves reasoning and interpretability
- Includes ablation and out-of-distribution evaluation

### Weaknesses
- The dataset still relies on artificial perturbations; real-world reporting errors may exhibit different distributions or linguistic patterns, depends heavily on a single generative model
- Focused solely on chest X-rays; generalization to CT, MRI, or multimodal clinical contexts is not yet demonstrated

### Questions
- How well do synthetic errors reflect real clinical error distributions? Have you compared CorBenchX against real-world error frequencies or radiologist revisions?
- Could you elaborate on the human validation effort, how many samples were reviewed by radiologists, and how was inter-rater agreement measured?
- Why did you not consider using multiple generative models to synthesize the dataset to reduce potential bias?

### Soundness
2

### Presentation
3

### Contribution
3
