# Vision-Language Preference Optimization for Weakly Supervised Temporal Action Localization

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 4

## Abstract
Weakly supervised temporal action localization (WS-TAL) aims to localize actions in untrimmed videos using only video-level labels. Due to the absence of frame-level annotations, classification predictions during the initial training phase predominantly rely on the prior knowledge embedded in pre-trained video foundation models.
However, the foundation model's inherent erroneous biases persist uncorrected during training, resulting in compounding error propagation throughout the learning process.
To address this issue, we develop a dual-branch framework called Vision-Language Preference Optimization (VLPO) that enhances WS-TAL tasks through systematic integration with vision-language model (VLM). 
Our framework introduces two key components: 
(1) The Vision-Language Fine-Tuning (VLFT) branch, 
which effectively establishes a multimodal feature alignment mechanism through video-level supervision, conducts online adaptive fine-tuning on the vision-language features. This significantly enhances the semantic sensitivity of temporal localization under weakly-supervised conditions;
(2) The Preference Driven Optimization (PDO) branch, through the predictive preferences provided by VLM, optimizes the traditional WSTAL framework and actionness learning at the snippet-level from both class-aware and class-agnostic perspectives, significantly enhancing the accuracy of action localization.
Extensive experiments on WS-TAL benchmarks demonstrate that VLPO significantly outperforms state-of-the-art methods, showcasing its effectiveness in WS-TAL.
The source code will be released upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
1. Insufficient theoretical novelty: Dual-branch (VLFT/PDO) and modules (CM-AFA/DSP) are incremental, lacking groundbreaking frameworks
2. Limited generalization: Only VideoCLIP-XL and 2 datasets tested; no complex scenario/other VLM validation 
3. Incomplete experiments: Omitted recent SOTA comparisons; superficial ablation without hyperparameter rationale
4. Weak qualitative analysis: Few examples; evasive "LongJump" failure attribution
5. Minor strength: Outperforms some SOTA on THUMOS14/ActivityNet 1.3 with low inference overhead

### Strengths
This paper proposes the VLFT branch to enable effective fine-tuning of VLM under weak supervision and introduces the PDO branch, which optimizes the WS-TAL task by refining the prediction preferences of VLM from both class-aware and class-agnostic perspectives.

### Weaknesses
1. The core design of the VLPO framework lacks groundbreaking theoretical innovation. First, the dual-branch (VLFT + PDO) paradigm for weakly supervised temporal action localization (WS-TAL) is not novel—existing works (e.g., PVLR [19], Li et al. [13]) have already explored integrating vision-language (VL) signals with WS-TAL via multi-branch structures.
2. Key modules like Cross-Modal Anchored Feature Alignment (CM-AFA) and Dynamic Selection Pooling (DSP) only implement incremental adjustments to existing cross-modal alignment (e.g., CLIP’s contrastive learning) and temporal pooling (e.g., Top-k pooling in MIL) methods, without proposing a new theoretical framework or mathematical mechanism to justify their superiority. 
3. The framework’s generalization ability is not adequately verified, leading to doubts about its practicality. First, the authors only use VideoCLIP-XL as the vision-language model (VLM) without testing other mainstream VLMs (e.g., BLIP-2, FLAVA, LLaVA-1.5) or lightweight VLMs (e.g., CLIP-ViT-B/32). This makes it impossible to confirm whether VLPO’s performance gains come from the framework itself or the specific VLM’s pre-trained features. 
4. The experimental design is incomplete, and the analysis lacks rigor, failing to fully support the framework’s effectiveness. Specifically,  the comparison with state-of-the-art (SOTA) methods is incomplete. On ActivityNet 1.3, the authors only compare with 11 methods (mostly pre-2024) and omit recent SOTA works (e.g., 2024’s AFPS [17] variants, 2025’s non-VLM-based WS-TAL methods), making the "SOTA outperformance" claim unconvincing. 
5. The ablation study is superficial. For example, in Table 3, the authors only verify the "presence/absence" of modules but do not analyze the impact of key hyperparameters (e.g., the anchoring strategy in CM-AFA, the threshold selection in DSP) on performance. The hyperparameter robustness test (Appendix A.3) only reports fluctuations but does not explain why λ₁=300, αₕ=0.9 are optimal—no rationalization or cross-dataset validation of hyperparameter selection is provided

### Questions
See the above comments

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Vision-Language Preference Optimization (VLPO), a dual-branch framework for weakly supervised temporal action localization (WS-TAL) that leverages vision-language models (VLMs). VLPO addresses the bias and error propagation in pre-trained video models through two components: (1) Vision-Language Fine-Tuning (VLFT), which aligns multimodal features and enhances semantic sensitivity via adaptive fine-tuning, and (2) Preference Driven Optimization (PDO), which refines snippet-level actionness learning using VLM-guided preferences. Experiments on WS-TAL benchmarks show that VLPO achieves superior performance over existing state-of-the-art methods.

### Strengths
1. The paper presents an innovative dual-branch weakly supervised framework (VLPO) that leverages vision-language preference optimization to correct model bias and enhance cross-modal feature interaction.

2. The proposed approach achieves significant performance improvements in temporal action localization by mitigating error accumulation and improving snippet-level localization accuracy.

### Weaknesses
1. The overall framework is complex, and the multi-module design increases computational cost during both training and inference.

2. The method relies heavily on the quality and generalization ability of the pre-trained vision-language model, which may reduce robustness in domain-shifted or low-quality datasets.

3. The Related Work section could be further enriched to provide a more comprehensive comparison and discussion of existing WS-TAL and vision-language integration methods.

### Questions
Please refer to the Weakness part.

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
This paper proposes a set of adaptive modules that leverage complementary information from vision language models (VLMs) to improve weakly supervised temporal action localization (WS-TAL). Instead of directly applying a VLM to WS-TAL, the authors introduce CM-AFA and DSP to better align visual and textual features with the WS-TAL setting, and PPG and APR to further optimize the localization objective.

### Strengths
1. The method achieves strong performance and clearly outperforms existing approaches.
2. The ablation studies are comprehensive and support the effectiveness of the proposed modules.

### Weaknesses
1. The motivation for the proposed modules is not sufficiently developed. While the design seems reasonable for addressing the generalization gap of VLMs on WS-TAL, the paper does not deeply analyze why these modules are necessary or how each specifically mitigates the mismatch between VLM features and the WS-TAL objective.
2. In the ablation studies in the appendix, it would be more informative to report results at multiple IoU thresholds (e.g., 0.3/0.5/0.7) rather than only the average score, so readers can better judge the localization quality.
3. The captions for the qualitative results are too brief. Please provide more detailed observations, especially regarding how the proposed modules affect different action classes or challenging temporal patterns.
4. It would strengthen the paper to include failure cases and a short discussion of the method’s limitations.

### Questions
1. The paper mentions that VLMs are not good at direct deploying on WS-TAL, but the motivation for each adaptive module is not fully analyzed. Could you clarify what specific mismatch each module is designed to address?  
2. Why are CM-AFA and DSP the right choices for aligning VLM visual/textual features to WS-TAL? Did you consider alternative designs?  
3. In the appendix ablation, it only reports average performance. Can you provide results across multiple IoU thresholds?  
4. The qualitative results have very short captions. Could you add more detailed descriptions?  
5. Can you show a few failure cases and analyze their causes? What are the current limitations of the approach?

### Soundness
2

### Presentation
2

### Contribution
2
