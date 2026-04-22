# Ref-Adv: Exploring MLLM Visual Reasoning in Referring Expression Tasks

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Referring Expression Comprehension (REC) links language to region level visual
perception. Standard benchmarks (RefCOCO, RefCOCO+, RefCOCOg) have
progressed rapidly with multimodal LLMs but remain weak tests of visual reasoning and grounding: (i) many expressions are very short, leaving little reason-
ing demand; (ii) images often contain few distractors, making the target easy to
find; and (iii) redundant descriptors enable shortcut solutions that bypass genuine
text understanding and visual reasoning. We introduce Ref-Adv, a modern REC
benchmark that suppresses shortcuts by pairing linguistically nontrivial expressions with only the information necessary to uniquely identify the target. The
dataset contains various expressions on real images, curated with hard distractors and annotated with reasoning facets including negation. We conduct comprehensive ablations (word order perturbations and
descriptor deletion sufficiency) to show that solving Ref-Adv requires reasoning
beyond simple cues, and we evaluate a broad suite of contemporary multimodal
LLMs on Ref-Adv. Despite strong results on RefCOCO, RefCOCO+, and RefCOCOg, models drop markedly on Ref-Adv, revealing reliance on shortcuts and
gaps in visual reasoning and grounding. We provide an in depth failure analysis
and aim for Ref-Adv to guide future work on visual reasoning and grounding in
MLLMs.
The dataset is available at \url{https://ref-adv.github.io/}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Ref-Adv, a new benchmark for Referring Expression Comprehension (REC) designed to address the limitations of existing datasets like RefCOCO, RefCOCO+, and RefCOCOg. The authors argue that these benchmarks are overly simplistic and allow models to exploit shortcuts rather than perform genuine visual reasoning. Ref-Adv is constructed using a semi-automated pipeline combining LLM-generated expressions and human verification, with a focus on hard distractors, minimal sufficiency, and negation. The paper includes comprehensive ablation studies and evaluations of modern MLLMs, showing a significant performance drop on Ref-Adv compared to traditional benchmarks.

### Strengths
1. The paper addresses a well-known and important issue in the REC community — the overestimation of model capabilities due to dataset shortcuts. The motivation is clear and well-argued.
2. The two-stage pipeline (LLM-authored + human verification) is well-designed. The inclusion of hard distractors, minimal sufficiency, and negation adds meaningful complexity.
3. The evaluation covers a wide range of MLLMs (both open and closed-source), includes ablation studies, and uses multiple IoU thresholds, which strengthens the empirical analysis.

### Weaknesses
1. While the dataset is more challenging, the core task (REC) remains unchanged. The paper does not propose a new task formulation or evaluation protocol beyond traditional bounding box accuracy. The idea of “hard distractors” and “minimal sufficiency” is not entirely new — similar ideas have been explored in prior work (e.g., Cops-Ref, FineCops-Ref).
2. The dataset is curated from COCO and OpenImages. It would be beneficial to explore how well Ref-Adv generalizes to more diverse or out-of-domain images
3. The reliance on IoU-based accuracy is standard but limited. It does not capture partial correctness or reasoning steps. For example, the evaluation should reflect that the performance drop is only from the difficulty of object localization or distractors. The model locates the distractor or cannot localize the object correctly.
4. As shown in Table 3 and 4, the word order removal and descriptor deletion are used to demonstrate that the proposed benchmark avoids the model from relying on the Grounding Shortcut. However, the performance gap is marginal compared with RefCOCO.
5. The integration of COT also brings marginal performance gain, e.g., the performance of Qwen vl-3B decreases with COT. Therefore, we cannot conclude that the proposed benchmark relies on the reasoning.
6. This paper only gives a conclusion and a benchmark. However, no solution is presented, which limits the contribution of this paper. In addition, only the evaluation data is provided. Even though we are aware of this conclusion, what can the community do to address such a problem?

### Questions
Please refer to Weaknesses.

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
This paper keenly identifies a core problem in the current field of Referring Expression Comprehension (REC): although Multimodal Large Language Models (MLLMs) have achieved near-saturation performance on classic benchmarks like RefCOCO(+/g), this high performance is due to models exploiting "shortcuts" (e.g., overly short expressions, lack of same-category distractors, redundant descriptors) rather than genuinely mastering visual reasoning.

To address this, the paper contributes Ref-Adv, a new, high-quality REC evaluation benchmark. The benchmark is constructed through a meticulously designed pipeline that combines LLM (GPT-4o) generation with rigorous human verification. This pipeline is specifically designed to generate scenarios containing "hard distractors" and create "minimally sufficient" referring expressions for them, while significantly increasing the proportion of complex logic like negation.

### Strengths
The paper not only points out the saturation of classic benchmarks but also deeply analyzes the three specific causes with data and examples.

The proposed four-stage data pipeline is outstanding. The two-stage LLM process, particularly "Similarity Judgement" and "Minimally Sufficient Expression Generation," is cleverly designed. The extremely strict three-annotator verification process ensures the dataset's exceptionally high quality and trustworthiness.

The CoT experiment is an especially insightful finding, as it clearly reveals that Ref-Adv measures reasoning ability of MLLMs.

### Weaknesses
1. Limitation of an Evaluation-Only Benchmark: Ref-Adv (5k samples) is an evaluation benchmark, not a training set. It excels at diagnosing the flaws of current models but does not provide a pathway for models to learn how to solve these complex reasoning tasks. Given its high construction cost (low keep rate), how to scale this up into a training set is an open question.

2. The paper repeatedly claims to test "visual reasoning" and "multi-step reasoning." However, the scope of reasoning evaluated is quite narrow. It primarily focuses on combinatorial logic (e.g., attributes A and B, but not C) and spatial relationships. The benchmark is missing deeper forms of reasoning, such as:
   - Functional Reasoning: (e.g., "the object used for cutting," "the item that can hold water")
   - Intent/Causal Reasoning: (e.g., "the person about to jump," "the dog that knocked over the vase")

    By focusing only on discriminative visual attributes, the paper overclaims the breadth of reasoning it actually tests.

### Questions
No

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
This paper targets the gap between classic REC benchmarks (RefCOCO series) and real visual–language reasoning by constructing a hard, shortcut-resistant benchmark of thousands of instances. 
Referring expressions are produced via a two-stage LLM pipeline that first extracts attributes and then composes a minimal-sufficient description, followed by tri-annotator human verification for the correctness.
Evaluation uses Acc@IoU at multiple thresholds and tests a broad slate of MLLMs (open and closed, with/without CoT). 
Results show strong models on traditional datasets RefCOCO drop substantially on Ref-Adv. Anti-shortcut ablations demonstrate that Ref-Adv requires order-sensitive, compositional grounding rather than keyword matching, which is an potential issue in the traditional benchmarks.

### Strengths
It proposes anti-shortcut design including bag-of-words shuffling and descriptor-deletion both cause larger drops than on legacy benchmarks. This is helpful for the need for compositional, order-aware grounding.
Its data construction is considerable with strong filter pipeline and covering negation. The final benchmark is processed with strict 3-human-annotator agreement. 
From experiments the benchmark exposes failure modes that legacy REC underestimates, creating a clear diagnostic “stress test” where many strong MLLMs collapse. Reporting across multiple IoU thresholds and conditional slices yields informative analysis.
Model coverage spans major open/closed families and CoT settings.

### Weaknesses
Coverage analysis of this paper is limited. There is no thorough breakdown of the 2,833 images / 5,000 instances (categories/attributes/relations/occlusion, long tails) or side-by-side coverage vs RefCOCO/+/g traiditional widely used benchmarks.

The CoT conclusion on RefCOCO conflicts with prior work in top venue. ARGUS[1] reports grounded CoT improves MLLM performance on RefCOCO/+/g, but this paper finds CoT can hurt the performance on RefCOCO, making the conlcusion not convincing. 

There are gaps and missing baselines in the evaluation. SoM + Semantic-SAM may propagate segmentation errors to IoU in evaluation with no sensitivity analysis(quantitative or qualitative) in this work. Also, this paper does not analyze the answer–grounding consistency, which is about the inconsistency between the text prediction and the actual grounding. Key baseline like LLaVA-OneVision does not been tested.

[1] ARGUS: Vision-Centric Reasoning with Grounded Chain-of-Thought, CVPR 2025

### Questions
Could the author please add category/attribute/relation/occlusion histograms/stats for the benchmark as a comparison vs RefCOCO(/+/g). 
It will be better if the author can clarify the CoT setup and explain the different conclusions with ARGUS on RefCOCO/+/g. Please consider address other evaluation pipeline & baselines.

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
This paper introduces a Referring Expression Comprehension (REC) benchmark dataset, Ref-Adv, which intentionally removes redundant descriptions and focuses on distractors, to make the dataset more challenging for VLMs. The samples are from the validation and test splits of COCO and OpenImages v7, and are labeled with GPT-4O before human reviews. This benchmark reveals the potential weaknesses of a few common VLMs, including GPT-4o, Gemini 2.5, InternVL-3, Qwen2.5-VL, and GLM-4.5v.

### Strengths
1. The benchmark is generated based on a sound rationale. This seems effective at revealing weaknesses in current REC models, although these models tend to overfit the traditional benchmarks.
2. The evaluation is comprehensive and overall convincing.

### Weaknesses
1. One thing to keep in mind is that the GPT-4O generated captions are a bit biased compared to the human captions, in that GPT-4o apparently generates more negations and humans usually unintentionally avoid negations. This is not necessarily bad, since in real applications, users may need to query with negations.
2. The analysis of experimental results is minimal. In particular, I wish the authors could analyze the performance on captions containing negations.

### Questions
1. GPT-4o has a low performance on Ref-Adv, why? It makes so many mistakes on the captions generated by itself?

### Soundness
2

### Presentation
2

### Contribution
2
