# Do 3D Large Language Models Really Understand 3D Spatial Relationships?

- Decision: Accept (Poster)
- Scores: 6, 8, 8, 6

## Abstract
Recent 3D Large-Language Models (3D-LLMs) claim to understand 3D worlds, especially spatial relationships among objects. Yet, we find that simply fine-tuning a language model on text-only question-answer pairs can perform comparably or even surpass these methods on the SQA3D benchmark without using any 3D input. This indicates that the SQA3D benchmark may not able to detect if the model exploits textual shortcuts rather than engages in 3D-aware reasoning. To address this issue, we introduce Real-3DQA, a more rigorous evaluation benchmark that filters out easy-to-guess questions and introduces a structured taxonomy to assess various aspects of 3D reasoning. Experiments on Real-3DQA confirm that existing 3D-LLMs struggle with spatial relationships once simple cues are removed. We further propose a 3D-reweighted training objective that leverages negative samples via explicit 3D-relation alignment, substantially enhancing 3D-LLMs’ performance in spatial reasoning tasks. Our findings underscore the need for robust benchmarks and tailored training strategies to advance genuine 3D vision-language understanding.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper finds that in the current benchmarks for 3D-aware reasoning, as SQA3D, there are many easy-to-guess questions or questions that do not require any 3D input and information. Therefore, the paper introduces Real-3DQA, which is a more rigorous benchmark that filter out the unproper questions for 3D understanding. Furthermore, the paper proposes a 3D-reweighted training objective to guide the model to focus on 3D features when performing 3D reasoning. Experimental results show that the previous high scores in 3D understanding benchmarks largely do not reflect the true 3D reasoning capability of the current 3D-LLMs. The performance on the proposed more rigorous benchmark can more truthfully reflect the real 3D understanding and reasoning capability of 3D-LLMs.

### Strengths
++ The paper spotted an interesting yet important issue in the current evaluation of 3D-LLMs, which can raise the awareness of the community on evaluating the true 3D understanding capability of 3D-LLMs.

++ The way of filtering out the 3D-irrelavant questions by comparing the answers of with and without 3D input is reasonable and effective.

++ The writing of this paper is mostly clear. The current issue is clearly stated, followed by the reasonable solutions to build a more reasonable benchmark, and finally shows the performance drop of current method on the more 3D-aware benchmark. Therefore, the overall paper flow is smooth, except for the viewpoint rotation score section which I have some confusions stated below.

### Weaknesses
-- Basically, 3D-LLMs are considered to be general purpose AI assistants. Therefore, in real applications, not all questions have to be related to 3D reasoning. In this case, the performance drop on the questions that can be shortcut-solvable by language priors becomes a real issue, because it means that the general usability of the system will get harmed, as there may only be a portion of tasks that truly require 3D reasoning and understanding. Therefore, the proposed 3DR-FT method has flaws that fails to preserve the original reasoning capabilities of LLMs.

-- I do not quite get the meaning and motivation to design such a viewpoint rotation score in Section 3.2. Basically, I think from Section 3.1, the 3D-independent questions have already been filtered out when comparing the answers. Therefore, there seems to be no need to put the effort to create questions that rotate the viewpoints, as creating these questions for all the four viewpoints may need some manual labor.

-- Typo: Line 461: "3D-RFT" should be "3DR-FT".

### Questions
-- For the viewpoint rotation score, how are the questions for creating the viewpoint variants getting chosen? From my understanding, it may require heavy manual work, as first we need to select questions that can have different answers when the viewpoints vary. Second, we need to manually label what the answers are for each directions. This is why I think the viewpoint rotation score part feels unnecessary, as it requires manual labor but is not quite justified to be useful.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents three main contributions to the development and evaluation of 3D large language models (3D-LLMs). 1) it introduces Real-3DQA, a benchmark that filters out questions answerable through textual shortcuts, thereby providing a more rigorous assessment of true 3D reasoning abilities. 2) it proposes the Viewpoint Rotation Score (VRS), a metric that evaluates a model’s spatial consistency under viewpoint changes, reducing the influence of superficial linguistic cues. 3) it develops a 3D-aware Reweighted Fine-Tuning (3DR-FT) method that re-emphasizes 3D-dependent information to enhance spatial understanding. Extensive experiments show that existing 3D-LLMs still exhibit limited real 3D reasoning capabilities on Real-3DQA, while the proposed 3DR-FT significantly improves their 3D comprehension and robustness.

### Strengths
- The paper is clearly written and easy to follow, with a well-structured presentation of ideas. The authors’ approach to analyzing the limitations of current 3D-LLMs is insightful and well-motivated, making the paper engaging to read.
- It provides a comprehensive and well-organized overview of existing 3D-LLMs and their evaluation benchmarks.
- The proposed Real-3DQA dataset and VRS metric are meaningful and valuable additions to the 3D-LLM research community. The comparative evaluation of existing 3D-LLMs on both the original and filtered benchmarks is thorough and revealing.
- The ablation studies on the proposed 3DR-FT method are well-designed and conducted across multiple models and datasets. The results are consistent and convincingly demonstrate the effectiveness of the proposed fine-tuning strategy and dataset.

### Weaknesses
- The proposed 3D-aware Reweighted Fine-Tuning (3DR-FT) method improves performance on the Real-3DQA benchmark but results in degraded performance on the original datasets. This suggests that the approach may either reduce the model’s generalization to language-prior-heavy questions or bias it toward producing answers that deviate from the dominant linguistic mode when 3D cues are absent. However, in real-world scenarios, many questions that 3D-LLMs encounter are likely to rely heavily on linguistic priors. Therefore, a more detailed discussion on how to balance 3D dependence and language adaptability would strengthen the paper.
- The overall dataset size of Real-3DQA is relatively small, and the variation in scene types, spatial relations, and question templates appears limited. This may restrict the benchmark’s ability to comprehensively evaluate diverse aspects of 3D reasoning and generalization across different environments. The authors are suggested to discuss how to further improve the 3D-LLM evaluation.

### Questions
Minor typo: line 226 "Mx. The ..." -> "Mx, the ..."

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper asks whether current 3D-LLMs truly understand 3D spatial relationships. The authors first show that a **text-only “blind” fine-tuned LLM** can match or surpass several 3D-LLMs on SQA3D, implying the benchmark allows shortcutting via linguistic priors rather than genuine 3D reasoning. They therefore construct Real-3DQA, filtering out questions that both a vision-conditioned model and its blind counterpart (and then a general LLM) can answer correctly without 3D input, yielding a set more dependent on 3D evidence. They further introduce a rotation-robustness metric, the Viewpoint Rotation Score (VRS), which evaluates consistency across rephrased, viewpoint-rotated variants of the same question. Experiments reveal sharp drops in accuracy when requiring correctness across all rotated views, highlighting poor rotation consistency in recent 3D-LLMs. Finally, the paper proposes 3D-aware Reweighted Fine-Tuning (3DR-FT), which uses a blind-model reference to upweight samples that are hard to guess from text alone, substantially improving performance on Real-3DQA (and a similarly filtered Real-ScanQA) while encouraging reliance on 3D cues.

### Strengths
The paper is original in framing “real” 3D understanding via the Real-3DQA pipeline and the rotation-consistency metric (VRS), moving beyond shortcut-prone benchmarks. Methodologically it’s solid: the blind-vs-vision contrast, rotated rephrasings, and multi-stage QC make the evidence credible, and 3DR-FT is a clear, effective objective. The writing and figures communicate the pipeline and metrics cleanly. Empirically, the work exposes meaningful brittleness in 3D-LLMs and shows a practical path (3DR-FT) to increase true 3D reliance—making the contribution significant.

### Weaknesses
Rotation robustness is evaluated using GPT-generated viewpoint texts. Although quality control is thorough, this approach still relies on linguistic rather than geometric variation and limits the test’s realism. Using scene-graph or pose-based rotations with automatic answer recomputation would more directly assess spatial consistency. Coverage is limited to four fixed views; expanding to denser yaw and pitch/roll, plus reporting uncertainty would strengthen claims. The Real-3DQA set is much smaller post-filtering, so testing robustness to alternative filtering unions would address potential distributional shifts. 3DR-FT is demonstrated on two backbones and trades off SQA3D performance; broader ablations and a multi-objective/curriculum variant could maintain gains on Real-3DQA without degrading shortcut-heavy sets.

### Questions
1. How sensitive is Real-3DQA to the choice of “blind” baseline? Would using different text-only LLMs (e.g., smaller or instruction-tuned ones) change which questions are filtered out?
2. Could the authors validate that the GPT-generated rotated descriptions correspond to actual geometric viewpoint shifts rather than linguistic paraphrases? A small human-verified subset or rendered 3D examples would help.
3. What are the main failure modes of 3DR-FT—does it hurt generalization to unseen object types or tasks beyond QA?
4. The four fixed rotation angles cover only azimuth changes. Would adding elevation or roll perturbations further reduce accuracy?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates whether current 3D large language models (3D-LLMs) truly understand 3D spatial relationships, rather than exploiting language priors. The authors show that text-only finetuned LLMs can perform competitively on existing 3D QA benchmarks, suggesting strong linguistic bias. To address this, they propose Real-3DQA, a refined benchmark that filters out 3D-independent questions and introduces viewpoint-rotation consistency evaluation. Additionally, they introduce a simple fine-tuning strategy, 3D-Reweighted Fine-tuning (3DR-FT), which assigns higher weights to questions that are less answerable from text alone. Experiments demonstrate significant performance drops of existing 3D-LLMs on Real-3DQA, confirming the issue, and modest gains from 3DR-FT.

### Strengths
- The finding that text-only models perform nearly as well as 3D-LLMs on standard benchmarks is valuable to the community.

- Real-3DQA improves evaluation fairness by filtering 3D-independent samples and introducing rotation consistency, which is sound.

- The authors evaluate multiple existing 3D-LLMs and provide both quantitative and qualitative analyses that convincingly demonstrate the identified problem.

### Weaknesses
Limited novelty in methodology: The paper diagnoses dataset bias effectively but does not propose new modeling architectures or mechanisms to fundamentally improve 3D reasoning.
1. The Real-3DQA benchmark largely refines existing datasets via filtering and simple viewpoint augmentations; while useful, it feels more like an engineering refinement than a conceptual leap.
2. The 3DR-FT method adds a weighting term to the loss based on text–3D prediction discrepancy; the idea is intuitive but technically minor.

The work’s value is primarily diagnostic rather than methodological.

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
2
