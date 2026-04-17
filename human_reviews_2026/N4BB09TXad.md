# JanusCoder: Towards a Foundational Visual-Programmatic Interface for Code Intelligence

- Decision: Accept (Poster)
- Scores: 8, 6, 6

## Abstract
The scope of neural code intelligence is rapidly expanding beyond text-based source code to encompass the rich visual outputs that programs generate. This visual dimension is critical for advanced applications like flexible content generation and precise, program-driven editing of visualizations. However, progress has been impeded by the scarcity of high-quality multimodal code data, a bottleneck stemming from challenges in synthesis and quality assessment. To address these challenges, we make contributions from both a data and modeling perspective. We first introduce a complete synthesis toolkit that leverages reciprocal synergies between data modalities to efficiently produce a large-scale, high-quality corpus spanning from standard charts to complex interactive web UIs and code-driven animations. Leveraging this toolkit, we construct JanusCode-800K, the largest multimodal code corpus to date. This powers the training of our models, JanusCoder and JanusCoderV, which establish a visual-programmatic interface for generating code from textual instructions, visual inputs, or a combination of both. Our unified model is a departure from existing approaches that build specialized models for isolated tasks. Extensive experiments on both text-centric and vision-centric coding tasks demonstrate the superior performance of the JanusCoder series, with our 7B to 14B scale models approaching or even exceeding the performance of commercial models. Furthermore, extensive analysis provides key insights into harmonizing programmatic logic with its visual expression. Our code and checkpoints are available at \url{https://github.com/InternLM/JanusCoder}.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces JanusCoder, a family of multimodal code-generation models that integrate textual, visual, and programmatic information. Its key contributions include:
1.  JanusCode-800K Dataset: A new dataset covering both text-centric and vision-centric tasks;
2.  DTVBench Benchmark: A new benchmark for dynamic theorem visualization using Manim and Mathematica;
3.  Strong Performance: Comprehensive evaluations on eight benchmarks show JanusCoder performs on par with or surpasses GPT-4o and specialized open-source models.

### Strengths
1.  This paper introduces a comprehensive dataset (JanusCode-800K), which fills a clear gap by integrating visual, textual, and programmatic modalities at scale;
2. The paper is well-written, with clear motivation and structured sections to detail the data curation process;
3. JanusCoder obtains strong empirical results across >8 benchmarks;

### Weaknesses
1. DTVBench’s limited scale (~102 tasks) may constrain statistical reliability;
2. Why does JANUSCODERV-8B perform worse than InternVL3.5-8B on DesignBench and WebCode2M?
3. When considering visual information, besides benchmarks purely focused on visualization, there are also some algorithmic or reasoning-related benchmarks [1]. Has the paper evaluated or discussed model performance on such tasks?
4. A question mark appears at line 253.


[1] MMCode: Benchmarking Multimodal Large Language Models for Code Generation with Visually Rich Programming Problems

### Questions
See the Weakness part.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
JanusCoder expands code intelligence from purely text-based inputs to jointly reasoning over code and visual outputs. It introduces a scalable multimodal code-data synthesis pipeline and builds JANUSCODE-800K, the largest multimodal code corpus to date. The authors develop unified models that handle text-centric and vision-centric coding tasks, achieving performance comparable to or surpassing commercial systems while offering insights into aligning program logic with visual expression.

### Strengths
1. The paper introduces a unified visual-programmatic interface and a complete multimodal code data synthesis toolkit, enabling code models to reason jointly over textual and visual programming tasks.

2. The work builds JANUSCODE-800K, the largest multimodal code corpus, and conducts extensive experiments across diverse benchmarks and modalities, demonstrating careful design and thorough empirical evaluation.

3. Results show strong performance, with 7B, 14B models approaching or surpassing commercial systems across text-centric and vision-centric code tasks.

### Weaknesses
1. Unclear data release plan: The data synthesis pipeline and JANUSCODE-800K corpus are central contributions, yet public availability is not guaranteed at submission time, raising concerns about reproducibility and community impact. The paper should explicitly clarify the dataset release schedule and scope.

2. Judge-based evaluation bias: The Stage-3 refinement heavily depends on LLM/VLM judges, which risks evaluation circularity and bias toward the same models used for filtering. More human evaluation or cross-model adjudication would strengthen reliability and reduce bias.

3. Limited methodological novelty: Despite strong engineering effort, the paper reads largely as a dataset and data-pipeline contribution. The model setup closely follows existing architectures and training paradigms, making the work feel more like a large-scale data release than a method advance.

### Questions
JANUSCODE-800K is central to the contribution. Will the full dataset be publicly released at camera-ready?

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
This paper presents JanusCoder, a suite of multimodal models aiming to unify code generation and visual reasoning through a “visual–programmatic interface.” The authors introduce a large-scale data synthesis toolkit and release JanusCode-800K, a multimodal corpus covering charts, WebUIs, visual artifacts, and animations. Two models—JanusCoder (text-centric) and JanusCoderV (vision-centric)—are trained on this corpus using Qwen and InternVL backbones. The authors also propose DTVBench, a benchmark for dynamic theorem visualization tasks. Experiments across seven benchmarks show that JanusCoder models outperform open-source baselines and sometimes rival GPT-4o. Ablation studies indicate that cross-domain synergies and reward-based data filtering contribute significantly to performance.

### Strengths
- Ambitious and timely goal: a unified model bridging code logic and visual semantics.

- The JanusCode-800K dataset appears large, diverse, and potentially impactful for future research.

- Strong empirical results on both text- and vision-centric benchmarks, including new ones created by the authors.

### Weaknesses
- Model novelty is limited. The architecture largely reuses Qwen and InternVL; the “unified interface” claim feels more conceptual than technical.

- Weak quantitative evidence for data quality improvements. The reward model and filtering pipeline are described but not systematically validated.

- Overextended scope. The paper attempts to be both a dataset, benchmark, and model paper, which dilutes its main scientific contribution.

- Lack of human evaluation for subjective visual tasks (animations, WebUIs).

### Questions
- What exactly differentiates the “unified visual-programmatic interface” from standard multimodal fine-tuning?
- Could you quantify the impact of reward modeling (e.g., pre- vs. post-filtering data quality metrics)?

### Soundness
3

### Presentation
3

### Contribution
3
