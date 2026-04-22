# Error Notebook-Guided, Training-Free Part Retrieval in 3D CAD Assemblies via Vision-Language Models

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 2

## Abstract
Effective specification-aware part retrieval within complex CAD assemblies is essential for automated engineering tasks. However, using LLMs/VLMs for this task is challenging: the CAD model metadata sequences often exceed token budgets, and fine-tuning high-performing proprietary models (e.g., GPT or Gemini) is unavailable. Therefore, we need a framework that delivers engineering value by handling long, non-natural-language CAD model metadata using VLMs, but without training. We propose a 2-stage framework with inference-time adaptation that combines corrected Error Notebooks with RAG to substantially improve VLM-based part retrieval reasoning. Each Error Notebook is built by correcting initial CoTs through reflective refinement, and then filtering each trajectory using our proposed grammar-constraint (GC) verifier to ensure structural well-formedness. The resulting notebook forms a high-quality repository of specification-CoT-answer triplets, from which RAG retrieves specification-relevant exemplars to condition the model's inference. We additionally contribute a CAD dataset with human preference annotations. Experiments with proprietary models (GPT-4o, Gemini, etc) show large gains, with GPT-4o (Omni) achieving up to +23.4 absolute accuracy points on the human-preference benchmark. The proposed GC verifier can further produce up to +4.5 accuracy points. Our approach also surpasses other training-free baselines (standard few-shot learning, self-consistency) and yields substantial improvements also for open-source VLMs (Qwen2-VL-2B-Instruct, Aya-Vision-8B). Under the cross-model GC setting, where the Error Notebook is constructed using GPT-4o (Omni), the 2B model inference achieves performance that comes within roughly 4 points of GPT-4o mini.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This manuscript discusses the specification aware part retrieval in complex 3D CAD assembly, which is a task that leads to VLM failure due to long context and poor reasoning ability. The author proposes a novel, non training framework that utilizes Error Notebook and RAG to enhance proprietary VLM. This notebook is constructed by collecting initial erroneous thought chains (CoT) inference from VLM, and then using VLM self reflection to generate corrected CoT trajectories. During inference, RAG retrieves these corrected examples as minority shot examples to guide VLM's inference. This article also contributes a new manual annotation CAD dataset and a two-stage pipeline to solve the problem of long context. Experiments have shown that GPT-4o has improved accuracy by 23.4% on a human preference benchmark

### Strengths
1. The authors contribute a new multimodal CAD dataset that includes human preference annotations. This would be  a valuable resource for the community if it can be made public.

2. The concept of leveraging corrected reasoning exemplars as reusable Error Notebooks is novel and elegant.

3. This work provides a new approach for solving the problem of excessively long STEP file inputs, and the part retrieval in assemblies that it focuses on is also a broad application in CAD design.

### Weaknesses
1. Some typos need further correction, such as missing commas after some equations and missing spaces between some words.

2. The latency and resource overhead of RAG-based prompting are not quantified (e.g. the time for doing one complete inference).

3. The core viewpoint is that retrieving corrected error is crucial. However, the experiment (Table 1) only compared "w/E-Notebook" (its method) with "w/o-E-Notebook" (zero sample). Missing a key baseline: a standard few shot method based on RAG that only retrieves correct examples (i.e. (query, correct_CoT, correct_answer)). Without this comparison, it is hard to know whether the performance gain comes from  error corrections or solely from known benefits provided by RAG with any relevant contextual examples.

### Questions
1. This paper notes that CoT performance drops with 50 exemplars due to long prompt lengths. Does this suggest a fundamental limitation? Have authors explored strategies to mitigate this, such as summarizing the retrieved CoTs or dynamically selecting only the most relevant steps from each CoT?

2. What is the average token length and inference time per query? This is quite important for practical applications, as it is difficult to apply if the output result is too slow due to the long time of doing CoT.

3. Could authors please quantify the computational cost of building the Error Notebook? For example, how many VLM calls are required per sample to generate the corrected CoT?

4. Are the geometric constraints required for part assembly essentially obtained from the retrieved STEP files, rather than being calculated by the LLM/VLM directly?

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
3

### Summary
This paper introduces a training-free framework for 3D CAD part retrieval using Vision-Language Models (VLMs). Its core contribution is the Error Notebook, a mechanism that uses Retrieval-Augmented Generation (RAG) to guide VLM inference with corrected reasoning exemplars, avoiding costly fine-tuning.

Key contributions: i) A training-free reasoning framework (Error Notebook + RAG) that significantly improves accuracy (up to 23.4% absolute gain) without fine-tuning. ii) A new, human-annotated CAD assembly dataset with relational specifications, based on the Fusion 360 Gallery. iii) A two-stage VLM pipeline (part description generation, then retrieval) to handle long, complex STEP file inputs.

### Strengths
The Error Notebook is a creative solution for improving proprietary, "black-box" VLMs that cannot be fine-tuned. It effectively adapts self-correction concepts to an inference-time strategy.

The evaluation is comprehensive, testing multiple VLMs (GPT-4o, Gemini) across various assembly complexities. The method shows robust and consistent improvements for all models, such as boosting GPT-4o (Omni) from 41.7% to 65.1% accuracy.

The two-stage pipeline is a practical engineering solution for processing lengthy, non-natural language STEP file metadata, making the approach viable for real CAD workflows.

The new human-preference dataset is a significant contribution, enabling more human-centric evaluation by filtering out ambiguous cases with multiple correct answers.

The ablations provide valuable insights, such as CoT reasoning being most beneficial for complex assemblies (>10 parts). The finding that performance is similar for 1 to 50 exemplars is also a key practical takeaway.

### Weaknesses
The paper is empirically strong but lacks a theoretical justification for why the Error Notebook works. There is no formal analysis of its properties or potential failure conditions.

The approach requires ground-truth labels to build the Error Notebook. This reliance limits its generalizability and scalability to new domains where labels are unavailable or expensive to acquire.

The primary comparison is against the same models without the Error Notebook. The paper needs comparisons against other training-free methods (e.g., standard few-shot learning, self-consistency) to isolate the specific benefit of the Error Notebook.

The computational overhead (API costs, token usage, latency) of constructing the notebook and performing RAG-enhanced inference is not discussed.

### Questions
- What specific function is used for RAG retrieval? How sensitive are results to this choice?
- How can the Error Notebook be constructed or applied in settings where ground-truth labels are unavailable or expensive to obtain?
- How does the Error Notebook compare to other training-free methods like inference-time scaling (for example self-consistency)?
- What are the computational overheads (API calls, tokens, latency) for notebook construction and RAG-enhanced inference compared to the baseline?
- How often do the VLMs successfully generate valid reasoning corrections given the ground truth? What happens when the correction generation itself fails?

### Soundness
3

### Presentation
2

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
The authors propose a part retrieval framework that uses error notebooks + RAG for refined prompt engineering. Furthermore, they contribute a human-in-the-loop CAD dataset for evaluating the proposed approach. Evaluations are performed on closed source models and achieve upto a 23% accuracy improvement on human-preference benchmarks.

### Strengths
- error notebooks idea is a creative solution for test-time reasoning augmentation
- contribution of human-in-the-loop dataset

### Weaknesses
- comparisons are primarily on proprietary models. It would be great to see them on open-source models as well.
- the eval metrics are somewhat under-developed. For example, there is not analysis on retrieval relevance.
- baseline comparisons against conventional CAD part retrieval methods are missing

### Questions
1. why was the dataset size limited to using a single set (752 assemblies) of Fusion 360 Assembly Dataset?
2. i might have missed this, but, how are the corrected reasoning trajectories verified?
3. are there scenarios where the model is not able to self-correct with prompting?
4. what kind of assemblies does the method struggle with?

### Soundness
3

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
4

### Summary
This paper considers a specific sub-problem in CAD tasks automatization, mainly part retrieval in the assembly. For this task a new approach, based on error notebooks on RAG is proposed, which can be used on top of any LLM model without fine-tuning or adaptataion. Experimental validation shows validity of the proposed approach.

### Strengths
1) The proposed method can be used without fine-tuning on top of any LLM model

### Weaknesses
1) Very limited scope of work due to selected problem
As far as I understand, the considered problem is retrieval of specific part in assembly by using its textual description. Each assembly in question consists of very limited amount of parts, less then 50. So the problem is like text retrieval from corpus of up to 50 texts. Yes, texts are very specific (CAD description of parts), but nonetheless the problem is small. This limits both the significance and impact of the work.

2) No related works or similar methods or comparision with existing methods. 
How such problems are solved now? Are there any approaches for part retrieval, other then LLMs? Has anyone considered the same problem before, or it is a first time?

3) Unclear description and presentation. 
Up till section 2.2 i have been struggling to understand what the problem in question is. Overall the presentation is very unclear.

### Questions
Please address the weakness #2 and comment on problem statement and significance/impact of your work.

### Soundness
2

### Presentation
1

### Contribution
2
