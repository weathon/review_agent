# WIMFRIS: WIndow Mamba Fusion and Parameter Efficient Tuning for Referring Image Segmentation

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Existing Parameter-Efficient Tuning (PET) methods for Referring Image Segmentation (RIS) primarily focus on layer-wise feature alignment, often neglecting the crucial role of a neck module for the intermediate fusion of aggregated multi-scale features, which creates a significant performance bottleneck. To address this limitation, we introduce WIMFRIS, a novel framework that establishes a powerful neck architecture alongside a simple yet effective PET strategy. At its core is our proposed HMF block, which first aggregates multi-scale features and then employs a novel WMF module to perform effective intermediate fusion. This WMF module leverages non-overlapping window partitioning to mitigate the information decay problem inherent in SSMs while ensuring rich local-global context interaction. Furthermore, our PET strategy enhances primary alignment with a MTA for robust textual priors, a MSA for precise vision-language fusion, and learnable emphasis parameters for adaptive stage-wise feature weighting. Extensive experiments demonstrate that WIMFRIS achieves new state-of-the-art performance across all public RIS benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
WIMFRIS introduces a neck-heavy, parameter-efficient RIS framework that aggregates multi-scale DINOv2 features, fuses them with CLIP text via a windowed Mamba block, and adaptively re-weights each stage, setting new SOTA mIoU on RefCOCO/+/g with < 3 % trainable params.

### Strengths
1. First to plug a windowed SSM neck (WMF) into RIS; mitigates exponential decay of vanilla Mamba.
2. Learnable emphasis per stage is simple yet novel for PET.
3. Exhaustive ablations: window size, kernel configs, PET modules all explored.
4. Plug-in HMF boosts ETRIS & DETRIS (Table 1), proving generic utility.

### Weaknesses
1. All results are fine-tuned; real-world deployment often lacks target-domain labels.
2. WMF prepends text to windows, but vision never feeds back to text; may miss visual disambiguation cues.
3. Parameter efficiency ≠ inference speed; window partitioning + SSM may hurt parallelism.

### Questions
See weakness

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
This paper proposes a novel parameter-efficient tuning (PET) method named WIMFRIS for referring image segmentation. In contrast to existing PET methods that primarily focus on layer-wise feature alignment and are struggle to aggregate multi-scale features, the proposed approach introduces a simple yet effective neck architecture based on the Mamba module. WIMFRIS achieves state-of-the-art performance on standard RIS benchmarks, demonstrating both efficiency and strong segmentation capability.

### Strengths
- The paper proposes a new efficient parameter-efficient tuning (PET)–based referring image segmentation (RIS) approach named WIMFRIS.
- The proposed algorithm enhances efficiency by replacing conventional blocks with an HMF block that actively leverages the Mamba architecture. In addition, it introduces several novel components—an SSM-based MTA, an MSA robust to multiple receptive fields, and an RFMixer—which together contribute to more precise vision-language fusion.
- The method achieves state-of-the-art performance on popular RIS benchmarks, demonstrating both effectiveness and robustness.

### Weaknesses
- Structural Issues in Writing
   - In the Abstract, abbreviations such as HMF and WMF appear without their full names or descriptions, making it difficult for readers to understand them.
   - Figure 1 lacks an explanation of the HMF module, requiring readers to infer that WMF is a sub-module of HMF only from context.
- #Params of PET and Performance Comparison
   - When comparing with existing PET methods, it would be fair to keep the number of PET parameters (#params) consistent across models. According to Table 1, when DINOv2-B/14 is used as the vision encoder, the proposed method shows only a slight improvement in performance compared to DETRIS, even though it uses more parameters. This raises concerns that the effectiveness of WIMFRIS may not be scalable.
- Limited Novelty
   - The paper proposes several modules (e.g., WMF, HMF, MSA, MTA), but the architectural novelty of each component seems limited. For instance, the HMF module appears to replace multiple cross-attention layers with a more efficient Mamba-based structure, but the use of Mamba itself is not novel. Similarly, the MSA and RFMixer are designed to handle multiple receptive fields, but this concept is not entirely new.
   - The paper would benefit from additional discussion or evidence to substantiate the novelty of these architectural contributions.
- Lack of Ablation Studies
   - As mentioned above, the paper lacks experiments that demonstrate the effectiveness and novelty of the proposed modules. For example, it would strengthen the work to include comparisons between MSA/RFMixer and baseline or vanilla methods for handling multiple receptive fields.
   - Table 3-(a) appears more like an engineering-oriented study rather than one providing clear scientific insight.

### Questions
Please provide your responses with reference to the weaknesses mentioned above.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces WIMFRIS, a framework for Referring Image Segmentationthat focuses on both a novel intermediate fusion neck architecture (the Hierarchical Mamba Fusion, or HMF, block) and a parameter-efficient tuning strategy. The HMF block leverages a Window Mamba Fuser module to effectively aggregate and fuse multi-scale vision and language features, using window partitioning to tackle the exponential decay in information typical of state-space models. The PET strategy employs adapters to efficiently align textual and visual representations and a learnable stage-wise emphasis mechanism. Extensive experiments are conducted on major RIS benchmarks, demonstrating state-of-the-art results for WIMFRIS compared to both PET-based and full fine-tuning methods.

### Strengths
- WIMFRIS achieves state-of-the-art or highly competitive performance across all standard RIS benchmarks (RefCOCO, RefCOCO+, G-Ref), outperforming previous parameter-efficient and full-tuning baselines. Table 2 clearly demonstrates these gains, including mixed-data setups.
- Multiple ablation tables systematically dissect the contributions of each module and architectural choice.
- The schematic diagrams provide clear breakdowns of the model pipeline, supporting the text’s descriptions of modular design and the flow of visual and textual feature processing. The visualizations  offer compelling qualitative evidence for improved segmentation, especially in challenging situations (e.g., clutter, occlusion).
- The paper carefully characterizes the underlying exponential decay issue in SSM-based fusion, and the model’s windowed approach is well justified both mathematically and empirically.
- WIMFRIS demonstrates competitive results while tuning a very small fraction of backbone parameters, highlighting the value for practical deployment.
- The explicit, detailed description of contrastive, dice, and alignment losses (and their weighting) makes reproduction feasible and testable.

### Weaknesses
- While MSA adapters and MTA are described and visualized in Figure 2, the specific methodology for choosing insertion layers for adapters in different backbones is only loosely justified. There is a missed opportunity for a principled, possibly automated or analytical policy for placement, and no ablation on layer choice is provided.
- Although Table 3 (a) explores performance trade-offs for window size, the choice of optimal $4 \times 4$ is only empirically justified. There is little theoretical or dataset-specific reasoning for why this size generalizes, and exploring task- or scale-adaptive policies would strengthen claims of robustness.
- There are several grammatical errors and awkward phrasings, as well as the use of slightly non-standard abbreviations in the tables (e.g., "vol", "m/s/6", "m/sfI" in Table 1), which may disrupt readability and hinder quick assimilation for a broad audience.

### Questions
- Can the authors provide a rationale for the placement of PET adapters (MSA, MTA) at specific depths in the vision/text backbone? Have they considered or tested more adaptive/learned strategies for insertion, and can they provide ablations or guidelines for optimal selection?

- How is the concatenation between text class tokens and visual patch windows actually handled in practice (e.g., with respect to normalization, possible channel mismatch, and possible overfitting due to repetitive text tokens)? Would normalization before SSM scans improve performance or stability?

- Have the authors empirically measured the actual decay rate of long-range dependencies for varying window sizes in SSM, and if so, can those be reported? Is the optimal window size truly dataset/task dependent?

- Are there notable scenarios where the windowed approach harms segmentation accuracy, e.g., in very small or oddly-shaped object instances, or when referring expressions are ambiguous or highly context-dependent?

- Will the complete code (including all adapter implementations and ablation regimes) be released for reproducibility, and if so, under what license and conditions?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper presents a parameter-efficient framework that integrates a window-based intermediate fusion neck (HMF) and lightweight adapters (MTA, MSA, and emphasis parameters) to enhance vision–language alignment for referring image segmentation.

### Strengths
- The paper introduces a Hierarchical Mamba Fusion (HMF) block, which performs intermediate vision–language fusion by aggregating multi-scale features and applying a window-based Mamba module (WMF).
- A parameter-efficient tuning (PET) strategy is presented, consisting of a Mamba Text Adapter (MTA) for modeling textual priors, a Multi-Scale Aligner (MSA) with RFMixer and cross-attention for visual–text alignment, and learnable emphasis parameters for adaptive layer weighting.
- The overall framework, WIMFRIS, integrates these components and is experimentally compared against existing PET-based and full fine-tuning methods on multiple RIS benchmarks.

### Weaknesses
* Lack of Novelty

The paper shows limited novelty. The **PET part** closely follows DETRIS, essentially extending its parameter-efficient tuning framework with minor Mamba-based modifications. The **neck design** heavily overlaps with the fusion architecture in fixation phase in SaFiRe, both adopting window-based Mamba fusion for intermediate vision-language alignment. Overall, the work mainly integrates these existing ideas rather than introducing a substantively new contribution.


* Incomplete Manuscript

The paper appears **incomplete**. Section 3.2 is unfinished, and the crucial description of the **task decoder** is missing. This omission disrupts the continuity between Sections 2.3 and 2.4. The authors should carefully verify whether the submitted version is the complete manuscript.


* Unfair and Limited Comparison

For Table 1


1. **Unfair Comparison :**
To ensure fairness, (1) the parameters of PET-based methods should be adjusted to achieve **comparable model sizes**, and (2) the **backbones of all compared methods** should be unified.

2. **Limited Comparison with State-of-the-Arts:**
More PET-based approaches should be included, as previous works (e.g., ETRIS, DETRIS, RISCLIP) have done, especially those involving **backbone-side modality fusion** in RIS, such as **PWAM in LAVT**, **SDF in VLT**, and **CFE in RISCLIP**, as well as classical parameter-efficient tuning methods like **LoRA** and **Adapter**.

3. **Marginal Improvement of the WMF Neck:**
Compared with **DETRIS**, the improvements achieved by the proposed **WMF Neck** are quite marginal.

4. **Insufficient Comparison :**
A more comprehensive comparison is needed to substantiate the claimed advantages of the proposed neck method, including detailed analyses of **parameter counts**, **computational cost (GFLOPs)**, and **inference speed**, particularly in comparisons with **ETRIS/DETRIS necks**.

For Table 2

1. **Inconsistent Metrics:**
   Table 2 mixes **mIoU** and **oIoU** without clarification. While RISCLIP, DETRIS, and WIMFRIS use **mIoU**, most other methods report **oIoU**. In particular, for works like **CGFormer** and **Polyformer**, which provide both metrics, the authors still report their **oIoU** values. Since **mIoU** is generally higher than **oIoU** on the RefCOCO family datasets, this inconsistency makes the performance comparison **unreliable**.
2. **RISCLIP Issue:**
   According to the authors’ own definition (line 44, “…keeping the vast majority of the backbone parameters frozen”), RISCLIP also freezes its CLIP backbone and should be considered a parameter-efficient tuning method. Moreover, the results of **RISCLIP-L** are missing, which appear **significantly higher** than those of the proposed “Ours-L” model (trained on RefCOCO+, mIoU: **RISCLIP-L** 74.38 / 78.77 / 66.84 vs. **Ours-L** 71.9 / 76.2 / 67.2).


*  Efficiency Analysis

Although this work emphasizes the **PET framework** and uses the **efficient Mamba architecture**, more detailed **efficiency analyses** should be provided—specifically **GFLOPs**, **inference speed**, and preferably **FPS**.


* Minor Issues

In **Table 3(a)**, the content does not match the caption: *4×4* is **not** the smallest window size.



***I would be happy to revise my score if the author addresses these points.***



---

**References:**

DETRIS: Densely Connected Parameter-Efficient Tuning for Referring Image Segmentation AAAI2025

SaFiRe: SaFiRe: Saccade-Fixation Reiteration with Mamba for Referring Image Segmentation NeurIPS 2025

LAVT: Language-Aware Vision Transformer for Referring Image Segmentation CVPR2022

VLT: Vision-Language Transformer and Query Generation for Referring Segmentation TPAMI2023

RISCLIP:Extending CLIP’s Image-Text Alignment to Referring Image Segmentation NAACL2024

LoRA: Low-Rank Adaptation of Large Language Models. ICLR2022

Parameter-Efficient Transfer Learning for NLP. ICML2019

CGFormer: Contrastive Grouping with Transformer for Referring Image Segmentation CVPR2023

PolyFormer: Referring Image Segmentation as Sequential Polygon Generation CVPR2023

### Questions
*  Could you clarify the **task decoder design**?

*  In Table 1, which IoU metric is used—**mIoU** or **oIoU**? ETRIS reports oIoU from the original paper, but DETRIS uses mIoU.

* In Table 2, please clarify metric issue and the RISCLIP issue mentioned in W1-B.

* What are the **inference speed** and **GFLOPs** of the proposed model?

### Soundness
2

### Presentation
2

### Contribution
2
