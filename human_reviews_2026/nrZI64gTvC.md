# OmniCT: Towards a Unified Slice-Volume LVLM for Comprehensive CT Analysis

- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Computed Tomography (CT) is one of the most widely used and diagnostically information-dense imaging modalities, covering critical organs such as the heart, lungs, liver, and colon. Clinical interpretation relies on both slice-driven local features (e.g., sub-centimeter nodules, lesion boundaries) and volume-driven spatial representations (e.g., tumor infiltration, inter-organ anatomical relations). 
However, existing Large Vision–Language Models (LVLMs) remain fragmented in CT slice versus volumetric understanding: slice-driven LVLMs show strong generalization but lack cross-slice spatial consistency, while volume-driven LVLMs explicitly capture volumetric semantics but suffer from coarse granularity and poor compatibility with slice inputs. The absence of a unified modeling paradigm constitutes a major bottleneck for the clinical translation of medical LVLMs. 
We present OmniCT, a powerful unified slice–volume LVLM for CT scenarios, which makes three contributions: 
(i) Spatial Consistency Enhancement (SCE): volumetric slice composition combined with tri-axial positional embedding that introduces volumetric consistency, and an MoE hybrid projection enables efficient slice–volume adaptation; 
(ii) Organ-level Semantic Enhancement (OSE): segmentation and ROI localization explicitly align anatomical regions, emphasizing lesion- and organ-level semantics; 
(iii) MedEval-CT: the largest slice–volume CT dataset and hybrid benchmark integrates comprehensive metrics for unified evaluation. 
OmniCT consistently outperforms existing methods with a substantial margin across diverse clinical tasks and satisfies both micro-level detail sensitivity and macro-level spatial reasoning. More importantly, it establishes a new paradigm for cross-modal medical imaging understanding. Our project is available at https://github.com/ZJU4HealthCare/OmniCT.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose OmniCT, a unified LVLM that handles both slice-level and volume-level CT understanding. This is achieved through two key modules: Spatial Consistency Enhancement (SCE) and Organ-level Semantic Enhancement (OSE). SCE comprises volumetric slice composition, tri-axial positional embeddings, and a slice/volume MoE hybrid projection. OSE utilizes organ masks (generated using TotalSegmentator) to localize and analyze 117 anatomical structures. Subsequently, adaptive token aggregation and fusion with global tokens are employed. Additionally, the authors introduce MedEval-CT, a comprehensive dataset, benchmark, and toolkit suite for CT. MedEval-CT boasts approximately 1.7 million slice/volume VQA samples across seven distinct task types, for organ- and task-balanced evaluation. Extensive experiments were conducted across various public 2D and 3D CT benchmarks, including SLAKE, VQA-RAD, OmniMedVQA, RadFig-VQA, M3D, CT-RATE, and 3D-RAD. The results demonstrate that OmniCT (3B/7B) outperforms prior general and medical LVLMs.

### Strengths
- The proposed SCE/OSE design directly addresses the gap between slice-driven detail sensitivity and volume-driven spatial reasoning, which are squarely motivated by clinical reading patterns.

- OmniCT shows consistent improvements over some models across diverse 2D and 3D CT benchmarks, with well-documented ablations supporting the claims.

- Table 1 (SCE/OSE ablation) and studies on mixed-data training, encoder choices (2D vs 3D), and organ/task-level heatmaps add useful insight and support the design claims. 

- Volumetric Slice Composition + Tri-Axial Positional Embedding adds volumetric awareness while keeping slice compatibility. The MoE Hybrid Projection is a pragmatic way to route slice vs volume tokens.

### Weaknesses
- While MedEval-CT is positioned as “largest” (~1.7M) and “holistic,” the data sources, licensing, and deduplication across training vs evaluation (and vs existing public benchmarks) are not detailed enough to rule out contamination/leakage.

- The pipeline leans on large models (Qwen2.5-VL-72B, Qwen3-237B-A3B) for selection/mapping/refinement. This raises bias propagation questions and potential circularity if similar model families are used in training/eval. 

- The 2D-encoder-dominant findings are interesting, but the native-3D comparisons seem limited (e.g., M3D-CLIP vs DINOv3/SigLIP). Stronger baselines (recent 3D ViTs / 3D CLIP variants with tuning matched to your token budgets) would make the '2D encoders suffice' claim more convincing.

- The presentation of the work, particularly the figures, could be enhanced by using more aesthetically pleasing boxes or eliminating the shadows to improve readability.

### Questions
- Could you provide a clear breakdown of MedEval-CT’s data sources, licenses, and how overlap with public evaluation benchmarks (e.g., SLAKE, VQA-RAD, CT-RATE) is prevented? Details on your deduplication or leakage-checking strategy would greatly strengthen the dataset’s credibility.

- Could you report sensitivity analyses on the token unshuffle parameter m and its computational trade-offs? How does varying m impact memory usage, inference latency, and accuracy for 2D vs 3D settings?

- Since OSE relies on TotalSegmentator masks, how robust is OmniCT to imperfect or noisy organ segmentations? Have you tested performance degradation under synthetic noise or partial-mask conditions?

- The results suggest 2D encoders outperform 3D ones for volumetric reasoning. Could you add or discuss comparisons against stronger native-3D models (e.g., VideoMAE-3D, Uni3D-ViT) with matched token budgets to solidify this claim?

### Soundness
3

### Presentation
2

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
The authors propose OmniCT and MedEval-CT. OmniCT is a new architecture, aimed at allowing to combine slice and volumetric information in vision-encoders, allowing their LVLM to be applied to both, 3D and 2D data. Additionally, they introduce Spatial Consistency Enhancement (SCE) and Organ-level semantic enhancement (SCE).
Regarding the MedEval-CT Dataset and Benchmark, they unify the existing 2D and 3D VLM benchmarks and unify their evaluation.

### Strengths
The paper exceeds their baselines across many benchmarks, yielding performance improvements and pushing the state-of-the-art of medical image understanding. 
Moreover they introduce a way to leverage both, 2D and 3D medical images.

### Weaknesses
My main concern with the paper two-fold: 

Firstly and most-importantly, the motivation of the paper is unclear to me. The authors motivation originates largely from the fact that there exist 2D and 3D CT medical image datasets, however by default all CT images are 3D. Subsequently, i don't know (and the authors don't motivate well) why 2D slices are needed to be integrated. How would one get the 2D slice selection? If a professional is in the loop, why would I need the LVLM? This question puts the authors method as well as the unification of their benchmark in question.

Secondly, clarity. The paper is squeezing a lot of content into the 9 pages, which leads to a lot of important context missing or being deferred somewhere else. E.g. details on the large dataset the authors propose (MedEval-CT-Dataset) are hard to comeby. Maybe it's a collection of all already publicly available datasets, maybe it's new data? All they provide is 1.7 million 2D slices. This glossing over important details is prevalent in the paper.
 

Other minor concerns are:
1. The idea of leveraging 2D encoders and packing slices into RGB channels is not novel with an old example being [1] and joint training existing as well in [2].
2. The idea of leveraging organ-masks to aggregate embeddings is not new either, as it was done by fVLM before already [3].

[1] Draelos, Rachel Lea, et al. "Machine-learning-based multiple abnormality prediction with large-scale chest computed tomography volumes." Medical image analysis 67 (2021): 101857.
[2] Xie, Yutong, et al. "Unimiss: Universal medical self-supervised learning via breaking dimensionality barrier." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2022.
[3] Shui, Zhongyi, et al. "Large-scale and Fine-grained Vision-language Pre-training for Enhanced CT Image Understanding." The Thirteenth International Conference on Learning Representations.

### Questions
Q1: MedEval-CT-Dataset: Where does this data come from? 1.7 million samples means what in Volumes/Slices?
Q2: MedEval-CT-Bench: Why is a joint benchmark necessary? Why not just use the existing slice-wise benchmarks and the existing 3D benchmarks?
Q3: Did the authors evaluate how LLMs are able to solve the VQA (in particular the multiple-choice) questions? There is work that shows that question design alone can leak information on the correct answers. 
Q4: Did you evaluate quantitative metrics? I.e. CT-RATE report generation has a RadBERT classifier to yield abnormality labels, which would circumvent the potential VQA question leakage.

### Soundness
2

### Presentation
1

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
This paper introduces OmniCT, a unified Large Vision-Language Model (LVLM) for CT interpretation that jointly models 2D slice-based and 3D volume-based inputs within one framework.
The key motivation is that current medical LVLMs are fragmented:
- Slice-driven LVLMs (e.g., Med-VLM, LLaVA-Med) have good 2D generalization but lack spatial consistency across slices.
- Volume-driven LVLMs (e.g., CT-CHAT, M3D-LaMed) capture 3D structures but are coarse-grained and not compatible with slice inputs.

OmniCT aims to bridge this gap through three main innovations:
1. Spatial Consistency Enhancement (SCE):
Introduces Volumetric Slice Composition, Tri-axial Positional Encoding, and a Mixture-of-Experts (MoE) hybrid projection to unify 2D and 3D visual tokens. This allows both slices and volumes to share a common latent space while maintaining volumetric spatial consistency.
2. Organ-level Semantic Enhancement (OSE):
Injects anatomical priors via organ segmentation and region-of-interest masking, followed by adaptive feature aggregation to emphasize small organs and reduce redundancy in large ones.
3. MedEval-CT Benchmark:
A large-scale dataset (1.7M samples) and evaluation suite unifying 2D and 3D CT tasks with multi-level metrics and a standardized evaluation toolkit.

### Strengths
The paper is strongly motivated and addresses a real, clinically relevant gap in current medical LVLMs: 
1. Strong and clinically grounded motivation and 
2. Unified slice–volume modeling framework

It proposes a coherent architectural design (SCE + OSE + MoE projection) that enables shared representation learning between slice-driven and volume-driven inputs. the fragmentation between 2D slice-based and 3D volume-based CT understanding. Its unified slice–volume framework is conceptually coherent and technically well-designed, combining Spatial Consistency Enhancement (SCE), Organ-level Semantic Enhancement (OSE), and a Mixture-of-Experts hybrid projection to jointly model local detail and global volumetric context. The integration of tri-axial positional encoding and anatomy-guided ROI alignment contributes to more spatially consistent and semantically rich representations. Empirically, the work demonstrates high-quality experimentation and reporting, including detailed ablations, multi-task evaluations, and architecture analysis that isolate the effects of each proposed module. Finally, the paper is well-written and logically structured.

### Weaknesses
1. Conceptual novelty and differentiation:
The “unified slice-volume” design is valuable but not fundamentally new, similar goals have been pursued by Med-2E3, Med3DInsight, and hybrid 2D/3D LVLMs. Can the authors clarify more clearly articulate what OmniCT does differently (e.g., tri-axial positional encoding vs. cross-slice attention in Med-2E3, why tri-axial positional encoding is important? Is there any ablation for this?).

2. Architectural complexity vs. simplicity:
In the MoE routing, only 2 expert model is used for 2D and 3D accordingly. However, it is difficult for us whether the 3D expert is really learning the 3D representations from tokens, same as 2D. Can you clarify more on how to well distinguish these two experts are learning distinctive representation to each other?

### Questions
1. Methodological distinction:
How does the proposed SCE differ from Med-2E3’s dynamic cross-slice fusion or Med3DInsight’s encoder alignment? It will be great if there is an explicit mathematical or architectural comparison.
2. Organ-level supervision:
Purely curious. The OSE module depends on pre-computed segmentation masks (TotalSegmentor). Does this supervision introduce label bias? How does OmniCT perform if segmentation is noisy or missing? Is the aggregated token more than enough for fusion. Wondering if the performance will enhance while the aggregated tokens has a longer context length for concatenation?
3. Cross-modality transfer:
The paper claims OmniCT trained on 2D can generalize to 3D tasks. Is this because of the tri-axial positional encoding or MoE routing? It will be great to provide an ablation to separate their contributions.

### Soundness
3

### Presentation
3

### Contribution
3
