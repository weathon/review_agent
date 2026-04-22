# FineSplat: Fine-Grained 3D Open-Vocabulary Language Gaussian Splatting

- Avg Score: 5.33
- Decision: Reject
- Scores: 8, 4, 4

## Abstract
Existing open-vocabulary scene understanding methods are primarily limited to coarse-grained understanding at the object category level, making them incapable of handling fine-grained queries. In this paper, we introduce a challenging task of fine-grained open-vocabulary scene understanding and propose a novel fine-grained 3D language gaussian splatting framework, FineSplat for short. Unlike prior methods that rely on the vision-language alignment model, such as CLIP, FineSplat models the feature field solely from textual captions, transforming the cross-modal feature matching challenge into a retrieval process between queries and captions. Specifically, we design the Fine-Grained Caption Generation (FGCG) strategy to obtain captions containing multi-dimensional fine-grained attributes. Then, the Fine-Grained Feature Field Modeling (FGFFM) strategy is introduced to encode generated fine-grained captions into object-level semantic features, which subsequently supervise the training of 3D Gaussian representations. Furthermore, we construct Fine-OVS, a benchmark to support research and evaluation of the fine-grained open-vocabulary scene understanding task. Extensive experiments conducted on the Fine-OVS demonstrate that our FineSplat framework significantly outperforms existing state-of-the-art methods.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces a challenging task of fine-grained open-vocabulary scene understanding, which is of great significance for real-world interaction. A novel language gaussian splatting framework, FineSplat, is proposed to address this task. This paper also introduces a novel benchmark dataset, Fine-OVS, to address the gap in previous benchmarks where the test queries are limited to simple category-level labels. The authors conducted extensive qualitative and quantitative experiments to verify the effectiveness of the proposed method from multiple perspectives. Overall, this is an interesting paper that makes a meaningful contribution to the community.

### Strengths
1.The paper is well written and well organized, and the proposed FineSplat is easy to follow.
2.This paper has a strong motivation and breaks through the CLIP-based paradigm by modeling the feature field using only fine-grained captions, which is a novel idea.
3.To support research and evaluation on fine-grained scene understanding tasks, this paper constructs a novel benchmark, Fine-OVS, which includes 8 fine-grained attributes. 
4.Extensive quantitative and qualitative experimental results demonstrate that FineSplat exhibits stronger fine-grained understanding capabilities compared to baseline methods.

### Weaknesses
1.The paper lacks ablation studies on Fine-Grained Feature Field Modeling. Specifically, if this process is treated as feature matching rather than text retrieval, how the choice of encoder affects performance remains unclear.
2.In Figure 2, the two text encoders are shown using the same color. However, according to the paper’s description, they are two different encoders, so the visual representation in Figure 2 should be adjusted.
3.The authors should clarify whether they plan to release the benchmark publicly. I believe this benchmark could significantly advance research in fine-grained scene understanding tasks.

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
4

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
This paper introduces a method for semantic 3D feature fields. Authors observes that CLIP often behaves like a bag-of-words under phrase/sentence-level queries and fails to capture rich compositional context. To address this, they propose an embedded 3DGS that ingests long, descriptive text features. The main idea is introduction of coarse-to-fine captioning pipeline that leverages a DAM and an MLLM to extract fine-grained textual signals, and they adjust the text encoder to handle unbalanced captions (mismatched information density between user queries and generated captions). To evaluate fine-caption-conditioned querying, they curate a new dataset for 3D localization from fine captions and show improvements over prior methods on tasks requiring detailed textual understanding.

### Strengths
- Clear motivation : Identification of CLIP’s bag-of-words-like behavior under long queries and the proposal to inject richer text signals into 3DGS. The problem framing (descriptive queries vs. keyword queries) is persuasive and grounded in practical use cases.

- The captioning pipeline and dataset can catalyze follow-up research on fine-grained 3D language grounding.

### Weaknesses
**More Ablation experiments?** 
- It is unclear whether the method truly understands each fine-grained meaning; ablations are needed (e.g., Observation on if they can distinguish same category with different materials, or when spatial relations are identifiable). 

**Multi-view consistency**
- Multi-view inconsistency remains a concern for text signals.
Prior works note instability from view-dependent cues due to inconsistent SAM segmentation and occlusions. 
Suggest text multi-view bootstrapping experiments: does aggregating captions from different views converge to consistent 3D semantics?

**Compatible with existing dataset**
- Do results on existing datasets remain comparable? While expressiveness seems improved, it is unclear whether coarse information is still well represented.

**About generalization study** 
- In the generalization study (Appendix E.4), comparisons should focus only on changed factors; since material accounts for only 29.2%, small overall changes may not indicate true generalization. How does performance differs using changing cases only.

### Questions
- Is there an ablation without the unbalanced-query/caption mechanism?
- Other questions are covered under Weaknesses above.

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
4

### Summary
The authors identify a limitation in current open-vocabulary 3D understanding methods, which are mostly limited to coarse, object-level recognition. They propose a new task called "Fine-grained Open-vocabulary 3D Scene Understanding" and a framework, FineSplat, to address it. The core idea of FineSplat is to avoid using vision-language models like CLIP for feature matching. Instead, it generates detailed textual captions for all objects in a scene and models the 3D feature field using only text-based embeddings. This reframes the problem from cross-modal (vision-text) matching to intra-modal (text-text) retrieval. The method involves a Fine-Grained Caption Generation (FGCG) strategy and a Fine-Grained Feature Field Modeling (FGFFM) strategy. The authors also introduce a new small-scale benchmark, Fine-OVS, to evaluate this fine-grained task

### Strengths
1. **Excellent Motivation:** The paper is well-motivated. It correctly identifies a key weakness in current 3D language-field methods (e.g., LangSplat, GAGS) that rely on CLIP: they struggle with fine-grained attribute binding and compositionality, often behaving like a "bag-of-words" . The examples in Figure 1 clearly illustrate this failure mode.
2. **Novel Problem Formulation:** The core idea of reformulating the task from vision-text matching to text-text retrieval is clever. This directly sidesteps the known attribute-binding issues of models like CLIP and grounds the scene representation in a semantically richer (text-only) space.
3. **New Benchmark Contribution:** The paper introduces Fine-OVS, a new benchmark specifically designed to evaluate this more challenging fine-grained task, which is a valuable contribution to the community.

### Weaknesses
1. **Extremely Complex Pipeline:** The primary weakness of this paper is the immense complexity of the proposed pipeline, which feels more like a heavy engineering effort than a clean, scalable method. The "Fine-Grained Caption Generation" (FGCG) strategy alone (see Fig. 2) requires running multiple, large foundation models in sequence: (1) Run SAM to get masks ; (2) Run DAM (a caption model) on *every* mask ; (3) Run an MLLM (Qwen-VL) with complex multi-modal prompts (including blurred and highlighted images) to *refine* the captions. This multi-stage, computationally massive data-generation process seems impractical to scale.
2. **Scene-Specific Components:** The method's scalability is severely limited by the fact that it requires training a *new, scene-specific* autoencoder for *every single scene*. This is also mentioned as a limitation by the authors. This means FineSplat is not a generalizable, "train-once" model, but rather a pipeline that must be partially re-trained for any new scene it encounters, which is a significant drawback.
3. **Very Limited Evaluation:** The new Fine-OVS benchmark is extremely small, consisting of only **8 scenes**. While the method shows strong performance on this custom-built benchmark, this is not a comprehensive evaluation. It is unclear if this complex pipeline is feasible or effective on larger-scale datasets.
4. **Poor Generalization to Standard Tasks:** The authors' own experiments on the standard (coarse-grained) LERF benchmark (Table 8) show that FineSplat performs *worse* than the baseline LangSplatV2 . This strongly suggests that the method has been over-specialized for its own narrow, fine-grained task and has lost the ability to perform well on general, coarse-grained queries. This supports the idea that this is a "niche" solution that does not advance general scene understanding.

### Questions
1. Could the authors provide a full computational cost analysis for the *entire* pipeline? This should include the FGCG data generation (cost of running SAM, DAM, and MLLM on all views) and the FGFFM (cost of training the *per-scene*autoencoder). How many hours/VRAM does it take to process one scene from start to finish, compared to LangSplatV2?
2. The poor performance on LERF (Table 8) is concerning. Does this imply a fundamental trade-off, where the model gains fine-grained accuracy by sacrificing coarse-grained accuracy? Can the model no longer reliably answer simple queries like "find the mug"?

### Soundness
2

### Presentation
3

### Contribution
2
