# FRABench and UFEval: Unified Fine-grained Evaluation with Task and Aspect Generalization

- Avg Score: 5.50
- Decision: Accept (Oral)
- Scores: 4, 4, 6, 8

## Abstract
Evaluating open-ended outputs of Multimodal Large Language Models has become a bottleneck as model capabilities, task diversity, and modality rapidly expand. Existing ``MLLM-as-a-Judge'' evaluators, though promising, remain constrained to specific tasks and aspects (i.e., specific evaluation criteria such as fluency for text and image quality for images). In this paper, we argue that, on one hand, based on the interconnected nature of criteria, learning specific aspects can generalize to unseen aspects; on the other hand, jointly learning to assess multiple visual criteria and tasks may foster a synergistic effect. To this end, we propose UFEval, the first unified fine-grained evaluator with task and aspect generalization for four evaluation tasks --- Natural Language Generation, Image Understanding, Image Generation, and Interleaved Text-and-Image Generation. However, training such a unified evaluator is hindered by the lack of a large-scale, multi-modal, and aspect-level resource. To address this gap, we introduce FRABench, a comprehensive fine-grained evaluation dataset. Specifically, (1) We first construct a hierarchical aspect taxonomy encompassing 112 distinct aspects across the aforementioned four tasks. (2) Based on this taxonomy, we create FRABench, comprising 60.4k pairwise samples with 325k evaluation labels obtained from a combination of human and GPT-4o annotations. (3) Finally, leveraging FRABench, we develop UFEval, a unified fine-grained evaluator. Experiments show that learning on specific aspects enables UFEval to generalize to unseen aspects, and joint learning to assess diverse visual tasks and aspects can lead to substantial mutual benefits.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes UFEval, a unified fine-grained evaluator designed to achieve task and aspect generalization for MLLMs. To support this, the authors build a hierarchical taxonomy with 112 distinct aspects across four task types, and based on this, construct FRABench, a large-scale, multi-modal, aspect-level evaluation benchmark, obtained through a mix of human and GPT-4o annotations. They then train UFEval on FRABench and evaluate its generalization ability through in-domain and out-of-domain experiments, demonstrate UFEval’s strong performance as an evaluator compared with baselines, and explore its application in preference alignment for both image understanding and image generation tasks.

### Strengths
1. The work introduces UFEval, the first unified fine-grained evaluator designed with both task and aspect generalization capabilities. It covers four diverse evaluation tasks: Natural Language Generation (NLG), Image Understanding (IU), Image Generation (IG), and Interleaved Text-Image Generation (ITIG). Complementary to this, the paper constructs FRABench, a large-scale, multi-modal, and aspect-level comprehensive evaluation dataset, which directly addresses the critical lack of resources required for training such a universal evaluator.

2. The trained UFEval evaluator demonstrates good performance across various benchmarks.

3. The study shows a substantial amount of work and extensive experiments.

### Weaknesses
1. While the paper proposes an ambitious taxonomy covering 112 distinct aspects, there are potential weaknesses in how this taxonomy is constructed. The authors note that they directly adopted existing aspect tree structures from previous studies and incorporated additional aspects "based on their definitions". However, the methodology for reconciling differences or conflicts between these source taxonomies is not clearly articulated. Moreover, the paper provides no evidence of expert validation, inter-rater agreement, or any form of consistency checking to support the robustness of the taxonomy.

2. The inclusion of 112 aspects also raises concerns about semantic redundancy. Many aspects seem to overlap substantially in meaning, making them difficult to distinguish in practice. For example:

   - Accuracy and Instruction Following are strongly correlated: when a model accurately completes a task, it typically also follows instructions, and deviations from instructions often result in inaccuracy.

   - Creativity-related aspects such as Appeal, Engagingness, and Creativity appear to measure very similar qualities and may not represent truly distinct evaluative dimensions.

   The paper argues that the FRA-OOD test set contains "unseen task-specific aspects" and uses this as evidence of UFEval's generalization ability. However, if these "unseen" aspects are semantically overlapping or near-synonymous with training aspects, the evaluator may not be demonstrating genuine generalization. Instead, it may simply be transferring learned evaluation criteria to synonymous labels, which is conceptually weaker than true out-of-domain generalization. This issue undermines the validity of the claimed aspect-level generalization. 

3. The paper’s core claim is that jointly learning to assess multiple visual aspects and tasks leads to synergistic effects. To support this, the authors compare models trained on a single task (e.g., “w/ IU”) with the multi-task model ("Ours"), and report performance improvements in Tables 4 and 5. However, the "Ours" model is trained on a larger and more diverse dataset than the other ablations. This makes it difficult to disentangle whether the observed gains actually stem from cross-task synergy or simply from exposure to more varied training data. Without a controlled experiment where the total training data volume is held constant across conditions, the evidence provided does not conclusively support the claimed synergistic effect. A data-controlled ablation would strengthen the argument for true multi-task synergy.

### Questions
1. What specific methodology was used to reconcile conflicts between the source taxonomies?

2. What empirical evidence (e.g., expert validation) can be provided to support the robustness and distinctness of the 112 aspects?

3. Given the high semantic overlap among many aspects, how can the authors prove that the performance on "unseen" aspects represents true generalization to novel concepts, and not just inference on synonymous labels?

### Soundness
2

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
This paper addresses the critical challenge of evaluating open-ended outputs from MLLMs. The authors argue that existing "MLLM-as-a-Judge" evaluators are too narrowly focused on specific tasks and aspects, limiting their generalizability. To overcome this, they propose two main contributions: 1) FRABench, a large-scale, fine-grained evaluation benchmark covering four major multimodal tasks (NLG, IU, IG, ITIG) across 28 sub-tasks, and 2) UFEval, a unified evaluator model trained on FRABench. The core hypotheses are that learning to evaluate diverse tasks and aspects fosters synergistic benefits, and that evaluators can generalize to unseen tasks and aspects due to their inherent semantic connections. The paper presents a comprehensive hierarchical taxonomy of 112 evaluation aspects, which forms the foundation of FRABench's 60.4k pairwise samples. Experiments demonstrate that UFEval achieves strong performance on out-of-domain tasks and aspects, is competitive with specialized evaluators on public benchmarks, and can be used to generate high-quality preference data for downstream model alignment via DPO.

### Strengths
1. The paper tackles a well-recognized bottleneck in MLLM research. As model capabilities expand, the need for scalable, reliable, and nuanced evaluation frameworks is paramount. The authors correctly identify the limitations of current approaches and propose a compelling, unified vision for evaluation. 
2. The development of FRABench is a major contribution in its own right. Creating a benchmark of this scale (60.4k pairs, 325k labels) that is multi-task, multi-modal, and fine-grained is a significant undertaking. The hierarchical aspect taxonomy is also a valuable conceptual contribution, providing a structured and comprehensive framework for thinking about MLLM quality that could be widely adopted.
3. The paper presents a strong set of results. UFEval consistently outperforms strong baselines on out-of-domain generalization, particularly on the human-annotated sets. Furthermore, it achieves competitive, and sometimes superior, performance against state-of-the-art, specialized evaluators (e.g., Themis, LLaVA-Critic) on established public benchmarks, despite being a single, unified model.

### Weaknesses
1. This is the most significant concern regarding the methodology. A substantial portion of the 325k training labels are generated by GPT-4o. While the authors use human-annotated data for testing, the training data's quality is fundamental to the final model's performance. The paper's validity rests on the assumption that GPT-4o is a sufficiently reliable and unbiased annotator for 112 diverse aspects.
2. While the resulting aspect tree is a strength, the process of its creation is described somewhat briefly. The paper mentions adapting existing structures and manually organizing the rest based on definitions. This process can be highly subjective.
3. The paper excels at presenting strong aggregate performance metrics. However, a deeper understanding would be gained from analyzing where UFEval fails. A qualitative analysis comparing UFEval's judgments to human judgments, especially in cases of disagreement, would be highly insightful.
4. There are few baseline comparisons, and qwen2-vl is already a year-old model. Authors should compare more recent large multimodal models and experiment with newer models

### Questions
1. In Table 1, the number of aspects for each evaluator is listed. For instance, AUTO-J is listed with 332. Are these aspects comparable in scope and granularity to the 112 aspects in your taxonomy? A direct numerical comparison might be misleading if the definition of an "aspect" differs significantly across studies. A clarifying footnote could be helpful.
2. The base model for UFEval is Qwen2-VL-7B. How much of the observed performance gain is attributable to the FRABench data versus the inherent strengths of this particular base model? A useful baseline might be to fine-tune Qwen2-VL-7B on a single-task, single-aspect dataset (e.g., the data used for Themis) and compare its performance on that specific task against UFEval. This would help further isolate the benefits of the unified training approach.
3. Regarding the GPT-4o annotations for UAs , you mention providing only the response to avoid bias from query-correctness. This is a very thoughtful design choice. Did you explore or observe any other biases in the GPT-4o annotations, and did you employ any other strategies to mitigate them during data creation?

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
This paper introduced a benchmark / evaluation dataset called FRABench for measuring so-called "aspects" for text, image generation and image understanding. Such aspects can be coherence, grammaticality, semantic consistency, etc. Also, it introduces UFEval, an multimodal LLM as a judge for such tasks and aspects, built by fine-tuning Qwen2-VL-7B-Instruct on the new FRABench dataset.

### Strengths
* FRABench is a substantial contribution: 60.4k pairwise samples and 325k aspect-level labels covering 112 distinct aspects organized in a clear hierarchical taxonomy.

### Weaknesses
* Vague terminology and presentation: The paper repeatedly uses the term “aspects” in the abstract and introduction without defining it or providing concrete examples. This makes it difficult for the reader to understand what exactly is being evaluated until much later (around page 4). Introducing a brief definition or illustrative examples early on would substantially improve readability and accessibility.

* Although the dataset includes some human labels, most of the 325k aspect-level annotations are GPT-4o-generated which raises the question of bias dna weakness propagation.

* UFEval is built only on Qwen2-VL-7B-Instruct, which raises concerns about how much the findings depend on that particular architecture. How do the findings generalise for other backbones, such as LlaVA, mplug-owl-3, etc?

* it would have been useful to have a qualitative analysis of failure modes.

* Generalisation tests: While UFEval is evaluated on several external benchmarks (e.g., Winoground, Pick-a-Pic), all of its training and main “aspect generalization” analyses are conducted within the FRABench framework. Since both training and out-of-domain splits rely on the same hierarchical taxonomy of aspects, it remains somewhat unclear how well UFEval’s claimed aspect-level generalization would hold under a completely independent fine-grained evaluation scheme.

### Questions
L016-017 there is so much repetitions of the word "aspects"

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
In this paper, the authors have developed a new evaluation dataset, FRABench, and used it to train the UFEval model, a unified evaluator of multimodal LLMs (MLLMs) open ended responses.

Their FRABench evaluation dataset contains 112 hierarchical, universal and task-specific eval aspects related to major MLLM tasks. The dataset contains 60,400 sample pairs and 325,000 evaluation labels, created using both human reviewers and GPT-4.0.

The authors argue that because these evaluation aspects are interconnected, it leads to better performance and enable the model to generalize to unseen aspects.

### Strengths
- They created a comprehensive, fine-grained dataset for MLLM tasks.
- The paper is well written, with detailed experiments, ablation studies, and comparisons on several public benchmarks.
- They show that FRABench can be used to fine-tune smaller 7B models, improving their performance to match larger 72B+ models for MLLMs tasks
- They show that UFEval can be used to automatically generate high-quality preference data.

### Weaknesses
- Heavy reliance on GPT-4o annotations. It is not shown how much they correlate with human labels. 
- UFEval still underperform relative to the larger models.

### Questions
Could FRABench be used for retrieval, providing few-shot examples to improve the performance of bigger models via in-context learning?

### Soundness
4

### Presentation
4

### Contribution
4
