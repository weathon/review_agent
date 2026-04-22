# Part-X-MLLM: Part-aware 3D Multimodal Large Language Model

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 6, 4, 6

## Abstract
We introduce Part-X-MLLM, a native 3D multimodal large language model that unifies diverse 3D tasks by formulating them as programs in a structured, executable grammar. Given an RGB point cloud and a natural language prompt, our model autoregressively generates a single, coherent token sequence encoding part-level bounding boxes, semantic descriptions, and edit commands. This structured output serves as a versatile interface to drive downstream geometry-aware modules for part-based generation and editing. By decoupling the symbolic planning from the geometric synthesis, our approach allows any compatible geometry engine to be controlled through a single, language-native frontend. We pre-train a dual-encoder architecture to disentangle structure from semantics and instruction-tune the model on a large-scale, part-centric dataset. Experiments demonstrate that our model excels at producing high-quality, structured plans, enabling state-of-the-art performance in grounded Q&A, compositional generation, and localized editing through one unified interface. Project page: https://chunshi.wang/Part-X-MLLM/

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes Part-X-MLLM, a native 3D multimodal LLM designed to operate not on whole 3D objects as monolithic shapes, but at part granularity. The key idea is to cast all 3D tasks (part captioning, grounding, Q&A, generation, and edit control) as autoregressive program generation in a grammar that interleaves tokens for bounding boxes, part semantics, and edit operators.

### Strengths
1. This paper is well-written and very easy to follow.
2. The paper identifies a real gap: existing 3D LLMs “understand scenes” or “generate geometry” but do not provide a unified, language-native, executable control surface over 3D parts. Casting all tasks as program generation is clean and technically coherent.
3. Unlike prior 3D MLLM benchmarks that focus on scene captioning or open QA, UniPart-Bench directly tests structured part-aware program generation (BBox alignment, part-level grounding, edit planning). That makes the evaluation target exactly match the claimed contribution, instead of proxy tasks.
4. The model shows consistent improvements over OmniPart and PartField baselines in structural planning metrics, not marginally but substantially (+2-5 IoU), which is not trivial in box-level matching.

### Weaknesses
1. All experiments are conducted on clean, part-annotated synthetic assets with well-formed AABBs. There is no evidence that the model works under realistic conditions (partial scans, occlusion, noise, incomplete parts, inconsistent normals).
2. UniPart-Bench adopts the same grammar, annotation style, and task templates used in training, and the test split is extremely small relative to the training volume. Thus the results may reflect memorization of format rather than robust capability.
3. The claimed downstream benefits (editing / generation quality) are entirely mediated by third-party geometry backends; the paper does not isolate how much of the improvement is attributable to the proposed planning interface itself versus the backend strength.

### Questions
1. Can the model sustain the same part-aware behavior when the input is not a clean synthetic asset but a noisy or incomplete real-world scan? Are there any stress-tests quantifying degradation under distribution shift?
2. Given that the benchmark shares annotation grammar and task templates with the training corpus and the test slice is very small, how do we exclude the possibility that the model is primarily exploiting training-time format priors rather than demonstrating generalizable planning ability?

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
3

### Summary
This paper introduces Part-X-MLLM, a novel part-aware 3D Multimodal Large Language Model. Its central contribution is the unification of diverse 3D tasks—including generation, editing, and question-answering—under a single framework by formulating them as programs within a structured, executable grammar of parts. The model takes an RGB point cloud and a natural language prompt as input and autoregressively generates a unified token sequence that encodes part-level bounding boxes, semantic descriptions, and edit commands, forming an executable "plan". This approach effectively decouples symbolic planning from geometric synthesis, allowing any compatible geometry engine to be controlled through this single, language-native interface. The authors employ a dual-encoder architecture to disentangle structural from semantic information and instruction-tune the model on a large-scale, part-centric dataset. Evaluated on a comprehensive benchmark (UniPart-Bench) spanning 11 task families, the model demonstrates a strong capability to produce high-quality structured plans, enabling state-of-the-art performance in grounded tasks.

### Strengths
- The research on part-based 3D generation is highly practical, and the authors have designed a unified framework that integrates 3D generation, understanding, and editing, which is very valuable.
- The paper not only proposes a large model, Part-X-MLLM, but also introduces a new benchmark and includes extensive experimental comparisons in both 3D generation and understanding, demonstrating substantial effort.
- The writing is clear and easy to follow, and the figures are professionally designed and visually appealing.

### Weaknesses
See the "Questions" section.

### Questions
First, part-based 3D is not my primary research area, so I am not fully aware of the most cutting-edge advancements in this field. I would like to see other reviewers' opinions on whether the comparative methods used in this paper are sufficiently state-of-the-art across various tasks. I believe the paper presents extensive work, and the unified framework for processing 3D patches as input is highly valuable. I am currently leaning toward a borderline accept score and am inclined to recommend acceptance. My final score may be adjusted based on other reviewers' comments and the authors' rebuttal.

Additionally, part-based 3D generation could potentially be generalized to 3D scene generation. It would be beneficial if the authors could briefly analyze this possibility. The following papers on scene compositional generation may also be considered for citation:
[1] GALA3D: Towards Text-to-3D Complex Scene Generation via Layout-guided Generative Gaussian Splatting \
[2] Layoutdreamer: Physics-guided layout for text-to-3d compositional scene generation \
[3] Semantic score distillation sampling for compositional text-to-3d generation \
[4] CompGS: Unleashing 2D Compositionality for Compositional Text-to-3D via Dynamically Optimizing 3D Gaussians

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
5

### Summary
This paper builds a multimodal framework and benchmark focused on 3D part-level understanding based on Qwen2.5-VL. The authors propose a dual-encoder architecture and a multi-stage training pipeline to jointly model geometric structures and semantic appearance for 3D objects.

### Strengths
1.Addressing part-level 3D multimodal modeling is timely and necessary.

2.The proposed dual-encoder design effectively encodes complementary attributes of 3D objects.

3.The use of task-specific prompts and special tokens enables diverse part-centric tasks within a unified framework.

### Weaknesses
1.Evaluation metrics rely mainly on traditional natural-language metrics; consider including LLM-based scoring (e.g., GPT-judge) for more robust assessment.

2.Baselines: comparison is limited; please include strong 2025-era SOTA 3D multimodal models on QA and grounding tasks (e.g., Mini-GPT-3D).

3.Benchmark: experiments are primarily on the authors' dataset; please evaluate on established public 3D benchmarks, such as the Point-LLM test suite, and include metrics for point resolution sensitivity and generative quality.

### Questions
1.Evaluation metrics are primarily limited to standard natural-language measures. It would strengthen the evaluation to incorporate LLM-based automatic assessment frameworks (e.g., GPT-based judging protocols) to better capture semantic correctness and reasoning quality.

2.Baseline comparisons appear insufficient. Please include strong contemporary 3D foundation models (2025 SOTA) for part-level QA and grounding tasks, such as Mini-GPT-3D, to more convincingly demonstrate the advantages of the proposed approach.

3.Benchmark evaluation is largely conducted on the authors’ own dataset. To better demonstrate generalization and fairness, we recommend evaluating on established public 3D multimodal benchmarks (e.g., the Point-LLM test suite) and additionally reporting performance under varying point-cloud resolutions as well as part-level generation quality metrics.

### Soundness
2

### Presentation
2

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
Novel, but paper writing could be improved

### Strengths
This paper presents Part-X-MLLM, a 3D large language model for diverse 3D tasks by formulating them as programs in an executable grammar. Overall, the work is decent, includes a large curated dataset, and is generalizable across 11 tasks. 

+ The two-stage training process is interesting to help the model learn the underlying 3D structure and associate the pretrained language knowledge with it.

+ Semantic Granularity Control is an interesting part of the work. The part-aware synthesis is useful for many practical analyses.

### Weaknesses
- Small typo in line 192 'boxe'

- The writing is a bit hard to follow in places. For example, in line 35, I am not sure why Part-X-MLLM is 'native'. Similarly, the mention of 'structural opaqueness' in line 53 is not clear.

- The distinction with past works is not clear enough. I would love to see a table comparing past works and Part-X-MLLM.

- For the qualitative analysis, it would have been great to see a small-scale study with real participants and evaluate the performance of Part-X-MLLM qualitatively.

### Questions
n/a

### Soundness
3

### Presentation
3

### Contribution
3
