# Medical thinking with multiple images

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Large language models perform well on many medical QA benchmarks, but real clinical reasoning is harder because diagnosis often requires integrating evidence across multiple images rather than interpreting a single view. 
We introduce MedThinkVQA, an expert-annotated benchmark for thinking with multiple images, in which models must interpret each image, combine cross-view evidence, and solve diagnostic questions under intermediate supervision and step-level evaluation. The dataset contains 10,067 cases, including 720 test cases, with an average of 6.68 images per case, substantially denser than prior work (earlier maxima $\leq$ 1.43). 
On the test set, the best closed-source models, Claude-4.6-opus, Gemini-3-pro, and GPT-5.2-xhigh, achieve only 54.9%--57.2% accuracy, while smaller proprietary variants, GPT-5-mini/nano, drop to 39.7% and 30.8%. Top open-source models perform worse overall, with Qwen3.5-397B-A17B (52.2%) and Qwen3.5-27B (50.6%) leading, followed by Lingshu-32B (43.2%), InternVL3.5-38B (40.7%), and MedGemma-27B (31.8%).
Further analysis points to a single-core bottleneck: current models struggle with grounded multi-image reasoning, i.e., reliably extracting, aligning, and composing evidence across views before higher-level inference can help. 
This is supported by three consistent findings: adding expert-provided single-image cues and integrating cross-image evidence improve performance, whereas replacing them with models’ self-generated intermediates reduces accuracy. 
Step-level analysis shows that over 70% of errors come from image reading and cross-view integration, with reasoning failures increasing on decisive steps. 
Scaling results show that while accuracy increases with more images, additional inference-time computation is beneficial only when the underlying visual grounding is already reliable. When early evidence extraction is weak, longer reasoning yields limited or unstable gains and can even amplify misread cues. 
Together, these results show that the main barrier is not simply insufficient reasoning length or depth, but the lack of reliable mechanisms for grounding, aligning, and composing distributed evidence across real-world, cross-view, multimodal clinical inputs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces an expert-annotated benchmark that probes the capabilities of multimodal LMMs on multi-image cross-view medical reasoning. Moreover, a step-wise evaluation protocol is proposed that can help pinpoint where models fail in their reasoning. State-of-the-art models are evaluated on the benchmark, and their mistakes are categorized into different error groups, highlighting the tendency for these models to fail on medical image understanding.

### Strengths
- The motivation of the paper is clear: it is important to assess multi-view capabilities of current models as it is closer to realistic diagnostic scenarios.
- The step-wise evaluation and breakdown into error categories is valuable. It helps us better understand where these models fail in complex medical reasoning tasks. 
- The study on the utility of adding expert hints vs. self-generated captions is interesting and highlights the benefits of human-in-the-loop diagnostic workflows.

### Weaknesses
- Overall, the paper is hard to follow at parts due to multiple factors. First, the structure is unusual. The introduction could be more high-level without much details about methodology. Details about methodology is dispersed between the introduction and various other sections. Second, the paper assumes significant background knowledge on the source dataset used to create the benchmark. Is the ground truth reasoning trace part of the source dataset? What are the different sections of the source data and what kind of information do they exactly contain (e.g. "Integrated Imaging Summary" or "Image Hint")?

- The step-wise evaluation is a bit unclear to me. How exactly is the generated output broken down into these steps? Furthermore, a rigorous definition of the error categories would be helpful, e.g. what does "Clinical Scenario Error" entail?

- How is it supported in the work that multiple images are *necessary* to answer the questions accurately? If the problems can be tackled by a single view, then the key claim of probing cross-view synthesis is not well-supported.

- Human performance is missing in the benchmark, making it difficult to gauge the gap between SOTA models and human experts on cross-view medical reasoning.

- It is unclear what is the practical use-case of generating the teaching note in this benchmark. It seems only loosely connected to cross-view medical reasoning.

### Questions
- I would recommend restructuring the paper with clearer delineation between different sections. 
- How are steps defined in the step-wise evaluation and how the error categories are defined and determined?
- How is the benchmark strictly probing multi-view reasoning? What happens if we remove some images? If the benchmark is truly probing cross-view capabilities, this would significantly degrade performance.
- What is the expert performance on the benchmark?

Minor: 
- Table number is missing on line 362.

### Soundness
2

### Presentation
1

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
This paper introduces MedThinkVQA, a benchmark built from Eurorad teaching cases, each pairing a clinical scenario with multi-image studies and expert reasoning artifacts. The task is framed as multi-choice diagnosis with a multi-step “think-with-images” pipeline: per-image findings, case-level integrated summary, differential-diagnosis reasoning, and a long-form medical education discussion generation. The evaluation goes beyond accuracy with stepwise checks, error-type tags, ROUGE/RadCliQ, and rubric-based/LLM-judge scoring of discussions, with human studies supporting reliability.

### Strengths
1. The problem formulation is well-motivated: prior medical VQA sets tend to be single-image, answer-centric, or automatically labeled; here the authors emphasize cross-view fusion and expert-authored intermediate signals that better mirror real diagnostic practice. 

2. Dataset quality and evaluation design are thoughtfully engineered. The paper details option sets derivation from expert differentials, textual leakage detection, pruning of items solvable by text-only LLMs, mitigation of surface biases, and wide coverage spanning 20/22 ICD-10 chapters. The evaluation is also staged and fine-grained.

3. Engineering efforts in data curation and model evaluation bring concrete analysis and conclusions.

Generally, this is a good paper and I did not see major flaws.

### Weaknesses
1. Dataset statistics are inconsistent. Both the abstract and Table 1 state an average of 6.51 images per case, yet Line 200 in Sec. 3.1 mentions 8.3. 

2. Some details about the dataset are missing. For example, how are multiple images in one case gathered? Are they longitudinal studies from the same patient, or complementary imaging modalities (X-ray, CT, MR, etc), or both? If both are involved, I suggest analyzing them separately since two scenarios assess different capabilities.

3. The evaluation relies on commercial LLMs (GPT-5) as LLM-judge, which might entail model/version drift risk since neither the API nor any specific snapshot is guaranteed to be available forever. Given this, I am wondering whether open-source models (e.g. Qwen series) are able to serve this and how the evaluation results will vary (e.g., will the scores differ drastically, or will there be any bias?). 


4. Lack of discussion with related work. For example, Medical-Diff-VQA [1] , ICG-CXR [2], MedFrameQA [3] are not mentioned in Table 1 and the manuscript, although they explicitly feature in visual reasoning with multiple imaging studies from the same patient.

[1] Expert Knowledge-Aware Image Difference Graph Representation Learning for Difference-Aware Medical Visual Question Answering (KDD 2023)

[2] Towards Interpretable Counterfactual Generation via Multimodal Autoregression (MICCAI 2025)

[3] MedFrameQA: A Multi-Image Medical VQA Benchmark for Clinical Reasoning (arXiv 2025.05)

---

Below are minor issues:
- Each abbreviation should be expanded when it first appears for better readability. For example, “QA” (question-answering) in Line 010 and Line 035; “MCQ” (multiple-choice question) in Line 104.
- In Lines 362--363: “Table shows representative model accuracy on the held-out test set.” Table index seems missing. 
- The prompts in the appendices (Secs. E, F, and G) are overflowing the right margin. Enabling automatic hyphenation or manually inserting hyphens may fix these issues.
- I suggest copying the “without SFT” results in Fig. 3 to Tab. 8 which presents model performance after SFT, or combine these results in one single figure. In this way, the readers will see the value of the curated training examples more clearly.
- I suggest adding a graph showing the imaging modality distribution, instead of plain description in Sec. I.

### Questions
Do cases in MedThinkVQA include redundant images (e.g. imaging study that does not provide valid information, or sometimes even contradictory information)? This aligns more to the practice where clinical users would not always do an image pre-filtering for the MLLM assistant. If that is the case, will such data produce a robust MLLM when used for model training and test models to ignore noisy information when used for model evaluation?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces MedThinkVQA, a benchmark for evaluating multi-image diagnostic reasoning in medical imaging. The benchmark features 8,481 cases (751 test) averaging 6.51 images per case, sourced from Eurorad and expert-annotated with three-step supervision: (1) per-image findings, (2) case-level imaging summaries, and (3) differential diagnosis reasoning. This paper evaluates various VLMs (GPT-5, Qwen2.5-VL, MedGemma, InternVL) and find that current models struggle significantly (GPT-5: 57.39% accuracy), with the primary bottleneck being cross-image evidence extraction and integration rather than language reasoning. The benchmark includes beyond-accuracy evaluation with error-type tagging and medical education case discussion generation.

### Strengths
1. This paper presents the largest expert-annotated multi-image medical QA benchmark.
2. This paper presents the well-designed three-step evaluation framework that mirrors clinical diagnostic workflow.
3. This benchmark performs rigorous dataset curation with multiple quality control measures, such as, leakage detection, confusion-aware pruning.

### Weaknesses
Weakness:
1. Dataset relies entirely on Eurorad cases, potentially limiting generalizability despite broad coverage of radiology subspecialties.
2. The benchmark data sourced from Eurorad may have been included in the pre-training corpora of some evaluated models.
3. The case analysis is not presented, such as, failed case analysis.

### Questions
1.  How do you ensure that Eurorad cases haven't been seen during pre-training of evaluated models?
2. Only the train set (small scale dataset) is used to train the vlm? the detailed train (sft) strategy.
3. Why not compared with the expert radiologist?
4. The present question from benchmark is verified by human expert for quality control? If not, why?

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
3

### Summary
This paper introduces MedThinkVQA, a large-scale benchmark explicitly designed to evaluate multimodal large language models (MLLMs) in multi-image diagnostic reasoning. Built from 8,481 expert-curated teaching cases (average 6.5 images per case), it formalizes a three-step workflow that mirrors clinical reasoning: (1) per-image findings, (2) cross-view imaging summary, and (3) differential-diagnosis reasoning, followed by a medical-education discussion task. 
Evaluation goes beyond simple accuracy by introducing step-level correctness, error-type tagging, and educational-value scoring. Baseline experiments on diverse MLLMs, show that even the best model reaches only 57.4 % accuracy, revealing that cross-image integration is the major bottleneck for current medical VLMs.

### Strengths
This is a high-quality benchmark paper with clear motivation and substantial novelty. It moves beyond single-image VQA toward multi-image, step-supervised diagnostic reasoning, something not covered by prior datasets such as OmniMedVQA or MedXpertQA. The design is conceptually elegant mirroring how clinicians think and technically meticulous, from option-wise pruning and leakage checks to fine-grained error analysis and human-LLM judge validation (κ≈0.8). 
The beyond-accuracy evaluation framework, including RadCliQ metrics and teaching-discussion scoring, sets a new standard for interpretable benchmarking in medical AI. The authors also demonstrate strong ethical and reproducibility practices, releasing code, annotation scripts, and bias audits. Overall, the work is novel, comprehensive, and clinically grounded, offering a meaningful step toward trustworthy multimodal reasoning in medicine.

### Weaknesses
While the dataset is impressively detailed, it is built entirely from Eurorad cases, which may bias the distribution toward educational rather than real clinical imaging; cross-institutional validation or inclusion of temporal cases would further strengthen robustness. 
The current evaluation focuses on four families of models and could benefit from broader comparisons to recent foundation-level MLLMs such as Gemini 2.5 Pro or Claude 3.5 Sonnet to fully situate difficulty. The study identifies the image-fusion bottleneck clearly but offers relatively limited prescriptive insight—there is little discussion of architectural directions or training strategies that could overcome this barrier.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
3
