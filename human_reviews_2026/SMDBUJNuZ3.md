# A Novel Query-Driven Multi-Stage Alternating Feature Extraction and Interaction Network for Image Manipulation Localization

- Decision: Reject
- Scores: 2, 2, 4, 8

## Abstract
Image Manipulation Localization (IML) aims to identify and localize the tampered regions within edited images. Many studies employ a dual-branch backbone to extract tampering features from dual modalities, followed by feature fusion at the final stage. In this process, the extraction and fusion of dual-modality features is relatively independent, which fails to fully leverage the complementarity between different modalities and thus diminishes sensitivity to tampering artifacts. Inspired by the way humans continuously integrate multi-faceted knowledge to understand the world, we propose QMA-Net, which contains a novel Multi-stage Alternating Feature Extraction and Interaction architecture. At each stage, we deeply explore the intrinsic relationships and mappings between different modality features. Feature extraction and interaction are performed alternately, constructing complementary dual-modality tampering feature representations and enhancing sensitivity to tampering artifacts. Additionally, we introduce a lightweight, Query-driven Multi-level Feature Decoding. This mechanism progressively aggregates key information from multi-level dual-modality tampering features through multiple sets of learnable tamper-aware queries, effectively filtering out irrelevant features. Finally, multi-level queries are used to refine discriminative features, enabling precise localization of tampered regions. Extensive experiments demonstrate that our framework outperforms current state-of-the-art models in localization accuracy and robustness across multiple public datasets, achieving a favorable balance between performance and efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper addresses the insufficient modal interaction in existing dual-stream networks for Image Manipulation Localization (IML) by proposing a novel framework named QMA-Net. The core of this framework is a Multi-stage Alternating Feature Extraction and Interaction architecture, which deeply fuses RGB and noise modality information throughout multiple stages rather than performing a single fusion at the end. Building upon this, the paper further designs a lightweight, Query-driven Multi-level Feature Decoder that employs learnable "tamper-aware queries" to actively screen and aggregate critical tampering cues from multi-level features. Experimental results demonstrate that this method achieves competitive performance on several public IML benchmarks.

### Strengths
- The paper proposes a structured, multi-stage fusion framework that attempts to address the common issue of insufficient modal interaction in dual-stream networks. The authors validate their method through systematic experiments, evaluating it on multiple public datasets and using detailed ablation studies to analyze the contributions of its key components.
- The experimental results show that the proposed method outperforms several existing baseline models across multiple evaluation metrics. This indicates that the proposed components, when combined, form an effective manipulation localization system, demonstrating its potential for this specific task.

### Weaknesses
- The paper successfully demonstrates the effectiveness of the "tamper-aware queries" but fails to provide a deep exploration of their working mechanism. It is unclear what patterns these query vectors learn or whether they exhibit preferences for different artifacts (e.g., edges, textures, semantics) at different network levels. The lack of such visualization or quantitative analysis significantly undermines the model's interpretability.
- QMA-Net introduces several elaborate modules (e.g., DFCM), which increase the model's overall complexity. Although ablation studies show that removing these modules degrades performance, the paper lacks a comparison with simpler alternatives (e.g., using simple feature concatenation or addition for interaction within the same multi-stage framework). This makes it difficult for readers to determine whether the current design's complexity is necessary to achieve high performance.
- The paper mentions that its noise extraction front-end (MNFM) fuses multiple features from SRM, BayarConv, and NoisePrint++. While this is a powerful combination, the paper does not detail how these features are fused, nor does it experimentally analyze their individual contributions. An internal ablation study of the MNFM module would make the methodology description more complete and rigorous.

### Questions
- The ablation study in the appendix (Table 9) shows that performance degrades when the number of queries is increased from 16 to 32. Could you provide an explanation for this interesting non-monotonic phenomenon? Does this suggest that image manipulation features are inherently sparse, causing an excessive number of queries to introduce redundant information or noise that interferes with the final localization?
- Could you quantitatively explain the specific advantages of feeding the interacted features back into the backbone for the next stage of extraction? How much would performance drop if a simpler architecture were used—one without feedback that simply aggregates the fused features from all stages at the decoder end? This would help us better understand the core value of your design.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes QMA-Net, a query-driven multi-stage framework for Image Manipulation Localization (IML). It addresses two core limitations of existing methods: insufficient interaction between RGB (semantic boundaries) and high-frequency noise (tampering artifacts like compression errors) modalities, and redundant multi-level features. QMA-Net consists of two key components: (1) a Multi-stage Alternating Feature Extraction and Interaction module, which alternates "feature extraction–deep interaction" via cross-modal alignment and a Dual-modal Feature Cross-guided Module (DFCM) to build complementary bimodal representations; (2) a lightweight Query-driven Multi-level Feature Decoding module, which uses learnable tamper-aware queries to filter irrelevant information and aggregate critical features via a Multi-domain Feature Aggregation Module (MFAM). Experiments on 4 public datasets show QMA-Net outperforming baselines like CAT-Net and TruFor.

### Strengths
1. Unlike traditional "extract-then-simple-fusion" methods, QMA-Net enables stage-wise alternating interaction between RGB and noise modalities.
2. Learnable tamper-aware queries (16 queries, verified optimal via experiments) dynamically aggregate multi-level key information via MFAM, filtering background noise.
3. Experiments cover traditional tampering (splicing, copy-move) and AI-generated tampering (COCOGlide, AutoSplice), with robustness tests under Gaussian blur/JPEG compression. Results consistently outperform baselines.

### Weaknesses
1. Comparisons lack state-of-the-art methods (e.g., IMDPrompter, FakeShield) that excel in fine-grained or AI-generated tampering. This fails to demonstrate QMA-Net’s breakthrough.
2. Insufficient Theoretical Support for Key Designs: Critical choices (e.g., "RGB-aligned-to-noise" interaction order, DFCM’s Group Convolution grouping number) rely on empirical settings without quantitative analysis (e.g., gradient variance tests) or alternative design comparisons.
3. It is hard to clearly understand the key mechanisms from Fig.2.

### Questions
1. Comparisons lack state-of-the-art methods (e.g., IMDPrompter, FakeShield) that excel in fine-grained or AI-generated tampering. This fails to demonstrate QMA-Net’s breakthrough.
2. Insufficient Theoretical Support for Key Designs: Critical choices (e.g., "RGB-aligned-to-noise" interaction order, DFCM’s Group Convolution grouping number) rely on empirical settings without quantitative analysis (e.g., gradient variance tests) or alternative design comparisons.
3. It is hard to clearly understand the key mechanisms from Fig.2.

### Soundness
3

### Presentation
2

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
This paper proposes QMA-Net, an image manipulation localization (IML) framework that alternates feature extraction and cross-modal interaction between RGB and noise modalities, combined with a query-driven multi-level feature decoder for adaptive fusion. Experiments on several benchmarks show modest but consistent improvements over prior SoTA methods.

### Strengths
- According to reported results, the framework outperforms state-of-the-art methods across benchmarks.
- From a design perspective, the idea of alternating between extraction and interaction (rather than simply branching + fusing) is intuitive and potentially beneficial.

### Weaknesses
## Main Weakness
- The approach is fundamentally a refinement of the dual-branch + fusion paradigm. The “alternating” schedule, while reasonable, does not constitute a paradigm shift. The query-based decoding has been used in other domains (e.g., DETR, Visual Prompt Tuning, Mask2Former) which weakens the novelty claim.
- There is heavy reuse of ideas such as multi-scale or multi-modal feature extraction and fusion in existing literature. The paper does not convincingly show clear dissimilarity to or break from those approaches like MVSS or Mesorch.
- The authors do not provide the model’s complexity, and the frequent feature fusion raises concerns that it may be a network with very large parameter count and FLOPs.
- The manuscript contains many places where it does not meet mature academic writing standards. For example, in Figure 1 it is not self-contained: the term ‘HF’ in ‘OursHF’ is undefined, and the word ‘classic’ is used without clarifying which specific model is meant. In line 53 the text refers to ‘(columns 5 and 6)’ but earlier no clear reference is given to what image or table those columns correspond to. There are many similar instances throughout the paper. From the standpoint of writing and presentation, this falls short of the level expected at conferences like ICLR.

### Questions
Please reference to Weakness section.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposes a novel framework named QMA-Net, which primarily comprises two key components: a Multi-stage Alternating Feature Extraction and Interaction architecture and lightweight, Query-driven Multi-level Feature Decoding. At each stage, the intrinsic relationships and mappings between RGB modality and noise modality features are deeply explored through the Cross-modal Feature Alignment and the DFCM. Simultaneously, the most critical information is selectively extracted and condensed from the dual-modality features while filtering out irrelevant interference via the MFAM and Tamper-Aware Queries. The experimental results demonstrate that QMA-Net outperforms existing state-of-the-art methods in both localization accuracy and robustness.

### Strengths
1. The proposed Multi-stage Alternating Feature Extraction and Interaction architecture constructs complementary bimodal feature representations, thereby effectively enhancing the sensitivity to tampering artifacts.
2. The unique decoding mechanism effectively focuses on key regions while filtering out irrelevant interference.
3. The writing is easy to understand. 
4. To enhance interpretability, the authors visualized features from different levels. An extensive set of ablation studies was conducted, which thoroughly validates the model's design.
5. The overall model has fewer parameters and lower computational requirements.

### Weaknesses
1. There is no analysis of failure cases, which limits transparency about when and why the model may fail in real-world forensic scenarios.
2. In Figure 2's QMA-Net pipeline, the notations (e.g., R1 through R4) are somewhat compact. The overall layout could be made more spacious.
3. Fail to compare with more effective models such as APSC-Net [1], or discuss them in related works, limiting the demonstrated effectiveness.
4. There are a few typos in the paper.

[1] Qu C, Zhong Y, Liu C, et al. Towards modern image manipulation localization: A large-scale dataset and novel methods[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 10781-10790.

### Questions
none

### Soundness
4

### Presentation
4

### Contribution
3
