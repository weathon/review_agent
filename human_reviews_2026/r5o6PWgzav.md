# LumosX: Relate Any Identities with Their Attributes for Personalized Video Generation

- Decision: Accept (Poster)
- Scores: 8, 4, 6, 6

## Abstract
Recent advances in diffusion models have significantly improved text-to-video generation, enabling personalized content creation with fine-grained control over both foreground and background elements. However, precise face–attribute alignment across subjects remains challenging, as existing methods lack explicit mechanisms to ensure intra-group consistency. Addressing this gap requires both explicit modeling strategies and face-attribute-aware data resources.  We therefore propose LumosX, a framework that advances both data and model design. On the data side, a tailored collection pipeline orchestrates captions and visual cues from independent videos, while multimodal large language models (MLLMs) infer and assign subject-specific dependencies. These extracted relational priors impose a finer-grained structure that amplifies the expressive control of personalized video generation and enables the construction of a comprehensive benchmark. On the modeling side, Relational Self-Attention and Relational Cross-Attention intertwine position-aware embeddings with refined attention dynamics to inscribe explicit subject–attribute dependencies, enforcing disciplined intra-group cohesion and amplifying the separation between distinct subject clusters. Comprehensive evaluations on our benchmark demonstrate that LumosX achieves state-of-the-art performance in fine-grained, identity-consistent, and semantically aligned personalized multi-subject video generation.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses **personalized multi-subject video generation** where each subject is defined not only by identity (face) but also by **subject-specific attributes** (e.g., clothing, accessories) and an explicit **background/object** context. On the data side, the authors build a complete pipeline that re-captions videos, extracts entities/attributes, and produces **reference image groups** for faces, per-subject attributes, independent objects, and a clean background—then binds these to **prompt tokens** for training. On the modeling side (built on Wan 2.1, 1.3B), they introduce three key components—**R2PE**, **CSAM**, and **MCAM**. Experiments focus on both identity consistency and **multi-subject group consistency**, with ablations indicating each component’s contribution.

### Strengths
- **Clear problem focus & delta.** Goes beyond prior “multi-face-only” personalization to **multi-subject** customization with explicit attribute and background/object control, making it applicable to more realistic scenes.
- **Well-engineered data pipeline.** The end-to-end collection & preprocessing pipeline—especially **binding reference images to prompt tokens**—is practical and valuable; if released, it would benefit future work.
- **Methodological novelty.** The proposed **R2PE**, **CSAM**, and **MCAM** are sensible and targeted to the identified failure modes; ablations support their effectiveness.
- **Stronger multi-subject results.** Improvements are most evident in **multi-subject group** consistency compared to recent baselines.
- **Reproducibility mindset.** Implementation details are reasonably complete; the paper frames a clear benchmark setting derived from the same pipeline.

### Weaknesses
- **Baseline parity/inputs.** It’s not always clear that baselines ingest the same **attribute/object/background** references rather than face-only cues; this affects fairness in multi-subject comparisons.

### Questions
1. **Subject-consistency setting vs. baselines.** Both **SkyReels-A2** and **Phantom** claim multi-subject capability. In your **subject-consistency** experiments, did they receive exactly the same bundle of inputs (per-face, per-attribute, per-object, and background references), or only faces? If not identical, please clarify the input protocol and why the comparison is fair.

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
5

### Summary
This paper introduces LumosX, a multi-subject video personalization framework which aims at maintaining the identity and attribute consistency across subjects. To achieve this, LumosX integrates both data-level and model-level innovations: 1) on the data side, the authors uses Qwen2.5-VL to infer subject–attribute relations and generate explicit relational priors; 2) on the modeling side, they introduce relational self-attn & cross-attn, which incorporate relational RoPE and custom attention masks to enhance intra-group cohesion and suppress cross-group interference. Extensive experiments show that LumosX achieves SOTA performance compared to the baseline models like Phantom and SkyReels-A2, particularly in maintaining identity consistency and semantic alignment.

### Strengths
- This paper focuses the problem of identity-attribute consistency, which sounds interesting but has been well explored topic in the domain of multi-subject video personalization.
- From the Tables 1~3, LumosX shows strong quantitative results comparing the SOTA models, like Phantom and SkyReels.
- The paper is well written and easy to follow. The figures are well-plotted and informative which can make readers quickly understand the core ideas.

### Weaknesses
- Although the paper focuses on maintaining consistent identity-attribute pairing, the author only show one generated example with multiple identity-attribute pairs (the right example in Figure 5). With such one sample, it is unconvincing to claim that LumosX is capable of solving the issue of inconsistent identity-attribute pairing.
- While interesting, it is unclear whether the issue of inconsistent identity-attribute pairing is an actual problem. First, in the only multi-subject example provided in the paper, both the comparing models (Phantom and SkyReels) do not exhibit attribute-swapping issues. Further, most of the existing models also supports full-body personalization which can easily fix such problem without the proposed relational design. If the problem rarely appears or can be easily fixed by the current models, the value of the proposed framework is limited.
- The relational modeling is too simplistic and only considers pairwise identity-attributes grouping. Therefore, the model cannot handle more complex spatial or interaction relations, which are quite common in real-word videos. For example, "the person holding the cup" vs. "the cup near the person" (which requires the spatial modeling) or "person-person or attribute-attribute" interactions.
- Another problem of the proposed relational modeling is that it is determined during the dataset collection phase. Such design assumes the relations are fixed throughout the video and fails to handle more complex scenarios where the scene evolves, such as someone turns around / changes pose / are occluded. The authors could consider dynamic re-binding design which could address the limitation by adjusting the grouping relation during video generation process.
- While the paper introduces multiple masks, such as CSAM and MCAM, to bias the attention, it is unclear how attention actually changes under these biases. The author could visualize the attention maps or compute the attention scores to better interpret / understand the proposed relational self-attn / cross-attn.
- All experiments are conducged on a private benchmark collected by the authors themselves, which makes it hard to reproduce the results. Could the authors also provide the scores on the public benchmarks, such as MSRVTT-personalization [1] and ​​OpenS2V-5M-Eval [2]?

[1] "Multi-subject Open-set Personalization in Video Generation", CVPR 2025

[2] "A Detailed Benchmark and Million-Scale Dataset for Subject-to-Video Generation", NeurIPS 2025 D&B

### Questions
- Could the authors provide more examples with multiple identity-attribute pairs? Since this is the main focus of the paper, it is difficult to justify the model performance without such samples.
- Could the authors also provide the examples that the existing models fail on handling identity-attribute consistency?
- Could the authors discuss the extension with more complex relational and dynamic modeling? The current relational modeling is too simplistic to cover the complexity of real-world videos.
- Could the provide more theoretical/empirical analysis on the proposed CSAM and MCAM?
- Could the authors also provide the scores on the public benchmarks?
- [Minor] The "3D VAE Decoder" block in Figure 3 is upside-down.
- [Minor] What is CSMA in Section 4.3 and Appendix C?

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This paper proposes LumosX, a framework for personalized multi-subject video generation that addresses the critical challenge of face–attribute misalignment in existing text-to-video (T2V) models. The core innovation lies in explicitly modeling face-attribute dependencies at both the data and model levels, enabling fine-grained, identity-consistent, and semantically aligned video synthesis.

### Strengths
1. Combines MLLM-driven data annotation with relational attention to solve a misalignment problem—an innovative fusion of NLP (entity extraction) and computer vision (positional embedding/attention masking) techniques.
2. The problem statement (face-attribute misalignment) is clearly articulated with examples (e.g., "A man on the left... and a man on the right..." causing confusion).
3. Enables flexible multi-subject customization (foreground/background control) that prior models lack.

### Weaknesses
1. The training dataset only includes 1–3 subjects, the paper acknowledges instability for 10+ subjects due to RoPE extrapolation. No preliminary results or mitigation detailsare provided beyond a future work note.
2. All evaluations rely on automated metrics. Human judgment of face-attribute alignment, video naturalness, and prompt adherence would strengthen claims (e.g., do viewers perceive fewer misassignments in LumosX-generated videos?).

### Questions
1. Do you plan to conduct a human study to assess perceived face-attribute alignment, video naturalness, or prompt adherence? If not, why do you believe automated metrics alone are sufficient to validate LumosX’s practical utility?
2. Can you provide quantitative metrics to support FLUX’s "superior realism" over other inpainting models? How do background quality variations affect downstream video generation?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a novel framework for personalized video generation with controllable facial attributes. The authors introduce a dedicated dataset curation pipeline tailored to this task and propose model designs to capture subject-attribute dependencies. Experimental results demonstrate satisfactory performance.

### Strengths
- The paper addresses an important and challenging problem in video generation: enabling both personalization and semantic control, areas where previous works have struggled.
- The paper introduces the dataset for this task and the designs on model side.

### Weaknesses
- The focus on generating videos based solely on facial attributes may limit the method’s generalizability to broader contexts.
- It is unclear how the approach performs when handling videos with multiple subjects (three or more), as the datasets support up to three subjects.
- The implementation details lack specifics on computational requirements such as GPU count and training duration.
- The definition of a "subject" during data curation is ambiguous. Figures 3 and 4 suggest that a subject may consist of a single face and its corresponding attribute, but the criteria for data curation and the types of attributes used require further clarification. While some details are mentioned in the appendix, a clear and formal definition should be included in the main text.
- The maximum number of attributes that can be assigned to one subject is not specified.
- It remains unclear whether the method supports using textual attributes in the absence of images. Although text prompts may control generation, the paper would benefit from further discussion and experiments on text-based attribute control in the context of multi-subject video personalization.

### Questions
- In the current data curation pipeline, human subjects are extracted from frames at the start, middle, and end of the video, and all input subjects appear in the full dataset. How does the method handle scenarios where a particular subject is present only partway through a video (e.g., featured in the first half but absent in the second)? While autoregressive generation may naively address this by using a sliding window of subjects, clarification on handling such cases would be helpful.
- Since attributes are provided as images, can existing multi-subject image personalization models be used to generate the initial image, followed by video generation? A comparison with such an approach would strengthen the experimental evaluation.

### Soundness
3

### Presentation
3

### Contribution
3
