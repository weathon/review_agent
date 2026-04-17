# Towards Universal Video Retrieval: Generalizing Video Embedding via Synthesized Multimodal Pyramid Curriculum

- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
The prevailing video retrieval paradigm is structurally misaligned, as narrow benchmarks incentivize correspondingly limited data and single-task training. Therefore, universal capability is suppressed due to the absence of a diagnostic evaluation that defines and demands multi-dimensional generalization. To break this cycle, we introduce a framework built on the co-design of evaluation, data, and modeling. First, we establish the Universal Video Retrieval Benchmark (UVRB), a suite of 16 datasets designed not only to measure performance but also to diagnose critical capability gaps across tasks and domains. Second, guided by UVRB's diagnostics, we introduce a scalable synthesis workflow that generates 1.55 million high-quality pairs to populate the semantic space required for universality. Finally, we devise the Modality Pyramid, a curriculum that trains our General Video Embedder (GVE) by explicitly leveraging the latent interconnections within our diverse data. Extensive experiments show GVE achieves state-of-the-art zero-shot generalization on UVRB. In particular, our analysis reveals that popular benchmarks are poor predictors of general ability and that partially relevant retrieval is a dominant but overlooked scenario. Overall, our co-designed framework provides a practical path to escape the limited scope and advance toward truly universal video retrieval.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper tackles the problem of universal video retrieval by proposing a co-designed framework jointly targeting evaluation, data, and training aspects. The authors introduce the Universal Video Retrieval Benchmark (UVRB), design a scalable data synthesis pipeline (V-SynFlow) yielding over 1.55 million high-quality, multimodal pairs (UVRD), and propose a Modality Pyramid curriculum to train a General Video Embedder (GVE) model for robust generalization. Extensive experiments demonstrate state-of-the-art zero-shot performance across challenging retrieval scenarios and provide new insights into the strengths and weaknesses of current paradigms.

### Strengths
1.The paper simultaneously introduces a novel dataset and a new video retrieval method, presenting a substantial and comprehensive contribution.

2.The paper conducts extensive experiments to demonstrate the effectiveness of the proposed method.

### Weaknesses
1.The authors claim that a major contribution of this work is the introduction of a new benchmark; however, the related work section lacks a thorough discussion of existing benchmarks, and the paper fails to provide comparative analyses with them.

2.The authors claim to have introduced a new benchmark; however, as evidenced in Section 3.1, this benchmark appears to merely aggregate existing datasets with only basic categorization. Moreover, the authors do not perform any substantial data cleaning or in-depth analysis, raising concerns about the comprehensiveness and validity of the proposed benchmark.

3.The experiments in the paper are conducted exclusively on the authors’ newly proposed benchmark, which undermines the persuasiveness of the claimed effectiveness of the video retrieval method, as its generalizability and robustness remain unverified on established or diverse datasets.

4.The paper contains citation formatting errors: in numerous places where the citep command should have been used, the authors have failed to apply it correctly.

5.The GVE method appears to involve only minor modifications to the training strategy built upon the Qwen-VL model. To more rigorously validate the effectiveness of GVE, the authors should also evaluate the model without this method under otherwise identical conditions. Ideally, such an analysis would be included in an ablation study. However, the notation used in the current ablation study—specifically “GVE-s” and “GVE-i”—is not clearly defined, making it difficult to interpret what components or variants these labels refer to.

### Questions
1.Is UVRB merely an aggregation of existing data? If so, beyond simply combining datasets, what additional efforts did the authors undertake to ensure the benchmark offers distinct advantages? If not, how was the data in the benchmark generated?

2.What do GVE-i and GVE-s respectively represent in the ablation study?

### Soundness
2

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
This paper proposes a holistic video retrieval research framework built on the co-design of evaluation, data, and modeling. The authors introduce the Universal Video Retrieval Benchmark (UVRB), a suite of 16 datasets to diagnose model capabilities across diverse tasks (e.g., textual, composed, visual) and domains (e.g., coarse-grained, fine-grained, long-context). Guided by diagnostics from this benchmark, they present V-SynFlow, a scalable workflow to synthesize a high-quality, multi-task dataset of 1.55 million video-text pairs (UVRD). Finally, they propose the Modality Pyramid, a curriculum learning strategy to train their General Video Embedder (GVE), an MLLM-based model. Experiments show that GVE achieves state-of-the-art zero-shot performance on UVRB, and the analysis reveals new insights, such as the finding that performance on partially relevant retrieval tasks is a better predictor of universal capability than traditional benchmarks.

### Strengths
Holistic Framework: The main strength is the ambitious and well-executed "evaluation-data-training" co-design. This approach moves beyond incremental model improvements to address a systemic issue in the field.

Comprehensive Benchmark (UVRB): The creation of UVRB is a major contribution that allows for a much more nuanced and diagnostic evaluation of video retrieval models than was previously possible.

Strong Empirical Results: The GVE model demonstrates superior zero-shot performance across nearly all tasks and domains, validating the effectiveness of the proposed data and training curriculum. The fact that a smaller 3B parameter GVE outperforms larger 7B baselines is particularly compelling.

### Weaknesses
Reliance on Synthetic Data: While the synthesis pipeline is sophisticated, it relies on an MLLM captioner. This introduces a potential for model-inherent biases or systematic errors in the training data that may not reflect real-world human annotations. The authors were clearly aware of the "garbage in, garbage out" problem. Their V-SynFlow pipeline includes a "Multi-granular Quality Control" stage as a first line of defense. This pre-filtering aims to ensure the MLLM captioner starts with a clean, coherent set of videos, reducing the chance of generating nonsensical descriptions. However, this is still automated, not human, validation.

### Questions
Regarding the Modality Pyramid curriculum: How sensitive is the task scheduling to the choice of the initial "prober" model ($\Psi_{1}$)? Would starting with a weaker or architecturally different prober (e.g., a CLIP-based model instead of GME-7B) significantly alter the training trajectory?Your finding that partially relevant (PR) retrieval is the best proxy for universal capability is fascinating. Do you have a hypothesis as to why this is the case? Does it require a more robust understanding of semantics to distinguish subtle relevance from complete irrelevance?In your V-SynFlow pipeline, what measures were taken to audit for and mitigate potential factual inaccuracies or hallucinations from the MLLM captioner? Could these artifacts inadvertently penalize models that are better grounded during evaluation?The performance degradation when scaling spatial tokens beyond 400 (Figure 13) is an interesting result. Does this suggest that the vision encoder or the projection layer is not effectively summarizing high-resolution features, or is it more of an attentional issue within the LLM?

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
3

### Summary
The paper introduces a universal video retrieval framework along with a new benchmark that encompasses multiple tasks—including textual, composed, and visual retrieval—across various domains such as coarse-grained, fine-grained, and long-context scenarios. It further presents the Universal Video Retrieval Dataset (UVRD) and the General Video Embedder (GVE), which leverages synthetic data for training. The effectiveness of GVE is demonstrated through evaluations on the UVRD benchmark, showing performance improvements over baseline methods.

### Strengths
- The paper investigates a new Universal Video Retrieval (UVR) task and evaluates the proposed method across a diverse set of benchmarks, demonstrating strong performance relative to existing baseline approaches.
- The proposed method and architecture are relatively simple in design, yet they prove to be effective across a wide range of video retrieval tasks.

### Weaknesses
- While the paper argues that UVRB is a new benchmark, it seems like the benchmark is just a combination of prior works. 
- The distinction between the proposed approach and prior work, such as UNITE, is not clearly articulated, making it difficult to assess the novelty of the contribution.
- The data generation pipeline should be compared with existing baselines; however, such comparisons are either missing or insufficiently discussed, limiting the understanding of its advantages or uniqueness.
- The motivation and corresponding evaluation appear somewhat weak. For instance, the paper claims that mastering perceptual primitives first is beneficial; however, this claim is only supported by improvements in final performance. A more carefully designed experimental setup is needed to explicitly validate this hypothesis.

### Questions
- Could the authors elaborate on the key differences or advancements introduced in this work?
- How does this approach differ in design or effectiveness from existing data generation methods?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the limitations of narrow video retrieval paradigms by proposing a co-designed framework for universal video retrieval. Core contributions include: (1) the UVRB, a suite of 16 datasets covering multi-task and multi-domain scenarios for diagnostic evaluation; (2) V-SynFlow, a scalable data synthesis pipeline generating 1.55M high-quality multi-task training pairs (UVRD); (3) the Modality Pyramid curriculum, which leverages task hierarchies to train the GVE based on Qwen2.5-VL; and (4) extensive experiments showing GVE achieves state-of-the-art zero-shot generalization on UVRB.

### Strengths
- Holistic Framework: The central strength is the novel co-design of evaluation, data, and modeling. This holistic approach breaks the cycle of narrow benchmarks leading to specialized models and provides a scalable path forward.
- Comprehensive Benchmark: The creation of a large-scale, diagnostic benchmark is a significant and lasting contribution that will benefit the entire research community.
- SOTA Performance: The proposed GVE model demonstrates impressive state-of-the-art performance in a strictly zero-shot setting, validating the effectiveness of the entire framework.
- Insightful Analysis: The paper goes beyond reporting metrics and provides a deep dive into the dimensional capabilities of different models. The findings on the importance of partially relevant retrieval and the performance divergence between CLIP and MLLM-based architectures are particularly insightful.

### Weaknesses
- Narrow Domain Coverage: UVRB does not include specialized domains (e.g., medical, industrial, surveillance), where visual semantics and query intent differ significantly. Extending the benchmark to these domains would enhance generalizability claims.

### Questions
How does the Modality Pyramid’s temperature scheduling (σ(t)) affect training dynamics? Are there scenarios where alternative scheduling strategies (e.g., task-specific temperatures) yield better results?

### Soundness
3

### Presentation
3

### Contribution
3
