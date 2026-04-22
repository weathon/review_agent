# MRAD: Zero-Shot Anomaly Detection with Memory-Driven Retrieval

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Zero-shot anomaly detection (ZSAD) often leverages pretrained vision or vision-language models, but many existing methods use prompt learning or complex modeling to fit the data distribution, resulting in high training or inference cost and limited cross-domain stability. To address these limitations, we propose Memory-Retrieval Anomaly Detection method (MRAD), a unified framework that replaces parametric fitting with a direct memory retrieval. The train-free base model, MRAD-TF,  freezes the CLIP image encoder and constructs a two-level memory bank (image-level and pixel-level) from auxiliary data, where feature-label pairs are explicitly stored as keys and values. During inference, anomaly scores are obtained directly by similarity retrieval over the memory bank. Based on the MRAD-TF, we further propose two lightweight variants as enhancements: (i) MRAD-FT fine-tunes the retrieval metric with two linear layers to enhance the discriminability between normal and anomaly; (ii) MRAD-CLIP injects the normal and anomalous region priors from the MRAD-FT as dynamic biases into CLIP's learnable text prompts, strengthening generalization to unseen categories. Across 16 industrial and medical datasets, the MRAD framework consistently demonstrates superior performance in anomaly classification and segmentation, under both train-free and training-based settings. Our work shows that fully leveraging the empirical distribution of raw data, rather than relying only on model fitting, can achieve stronger anomaly detection performance. The code has been publicly released at https://github.com/CROVO1026/MRAD.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper propose a two-level feature–label memory built from an auxiliary dataset using a frozen CLIP ViT-L/14. Inference is similarity retrieval from this memory (MRAD-TF). Here, MRAD-FT learns only two linear layers to calibrate the retrieval metric, while MRAD-CLIP injects region priors (normal/anomalous) from MRAD-FT as dynamic biases into learnable CLIP prompts to improve localization and cross-domain generalization. The approach is evaluated on 16 industrial/medical datasets and reports competitive train-free performance and new SOTA with the lightweight variants.

### Strengths
1.The paper rethinks zero-shot anomaly detection from a retrieval perspective, replacing complex prompt tuning or residual modeling with a memory-driven similarity framework.

2.The progression from MRAD-TF → MRAD-FT → MRAD-CLIP is logical and empirically validated. The fine-tuning stage (two linear layers + similarity dropout) improves separability while remaining lightweight, and the final CLIP-based variant integrates region-level priors as dynamic biases to guide attention and localization.

3.The authors evaluate across 16 datasets spanning industrial and medical domains, reporting both image- and pixel-level metrics. The consistent performance gains over prior CLIP-based ZSAD models (e.g., WinCLIP, AnomalyCLIP, FAPrompt) highlight strong generalization.

### Weaknesses
1.The approach depends on computing similarities against thousands of prototypes (≈3k here), yet the paper omits latency, memory, and scalability studies. As memory grows with more datasets or higher patch granularity, retrieval time could become a bottleneck. 

2.Although the model is tested on diverse datasets, there is little reporting of variance (seed/template effects), robustness to distribution shifts (e.g., lighting, noise), or per-category breakdowns. The current results might reflect dataset bias or favorable template selection.

### Questions
1.What is the minimal supervision needed to construct the memory? Can MRAD-TF be instantiated with only normal data (no masks) + synthetic anomaly patches, and how would performance change? Any results with image-level labels only?

2.In Table 2 the class-token bias hurts performance.  Would multi-scale priors (coarse-to-fine) or attention-pooled region priors help?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a novel method for ZSAD called Memory-Retrieval Anomaly Detection (MRAD). It replaces parametric fitting with with direct memory retrieval, facilitating anomaly detection without extensive training. The framework constructs a two-level memory bank using feature-label pairs, allowing efficient similarity retrieval and demonstrating superior performance. The work highlights the potential of leveraging the empirical data distribution for effective anomaly detection, offering a fresh perspective on ZSAD.

### Strengths
This paper proposes a novel approach that replaces parametric fitting with a direct memory retrieval to ZSAD, offering a fresh perspective on anomaly detection. It demonstrates soundness in both theoretical grounding and empirical validation. Also, the paper gives clear definitions and explanations of methodologies, making it accessible to readers.

### Weaknesses
Major:
1. All experiments use VisA or MVTec-AD as the auxiliary dataset. Could other datasets be used as the auxiliary dataset?
2. MRAD-CLIP injects region priors as additive biases into CLIP’s learnable prompts. It remains unclear whether this choice of design is optimal or merely sufficient.
3. MRAD-FT adds 2.76M parameters, but the fine-tuning efficiency remains under-explored. This may result in the inadequately quantified “lightweight” claim.

Minor:
1. Memory bank size scales with the auxiliary training data, which could be optimized further.
2. The number of the medical datasets used for image-level ZSAD is relatively small.

### Questions
1. Is there any related work about similarity retrieval in the field of ZSAD before? 
2. Are the quality and diversity of the feature-label pairs stored in the memory bank needed to be controlled or ensured?
3. What are the known limitations of MRAD, particularly in scenarios with highly imbalanced datasets or extreme domain shifts?

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
This paper proposes MRAD, a unified framework for zero-shot anomaly detection (ZSAD) that replaces parametric modeling with direct memory-based retrieval. The method freezes the CLIP image encoder and builds a two-level memory bank (image-level and pixel-level) from auxiliary data. The MRAD-TF variant operates in a fully training-free manner and already achieves competitive results through similarity-based retrieval. The MRAD-CLIP variant further injects normal and anomalous region priors (derived from MRAD-FT) into learnable text prompts. Experiments on 16 industrial and medical datasets demonstrate the effectiveness of the approach.

### Strengths
1. The framework is simple and effective.

2. The paper is clearly written and easy to follow.

3. Extensive experiments on both industrial and medical benchmarks support the claims.

### Weaknesses
1. The fine-tuning stage adopts two linear projection layers, but no ablation compares against shallower (e.g., 1-layer) or deeper variants, making it unclear whether the chosen depth is optimal or arbitrary.

2. The method emphasizes the benefit of two-level memory (image + pixel), but there is no ablation where one level is removed to show whether both levels are truly necessary.

3. No sensitivity analysis is provided for key hyperparameters (e.g., similarity mask ratio ρ, top-k selection, thresholding strategy).

### Questions
1. The fine-tuning stage adopts two linear projection layers, but no ablation compares against shallower (e.g., 1-layer) or deeper variants, making it unclear whether the chosen depth is optimal or arbitrary.

2. The method emphasizes the benefit of two-level memory (image + pixel), but there is no ablation where one level is removed to show whether both levels are truly necessary.

3. No sensitivity analysis is provided for key hyperparameters (e.g., similarity mask ratio ρ, top-k selection, thresholding strategy).

4. The approach relies on a specific auxiliary dataset, yet there is no experiment showing whether the performance is stable when using different auxiliary datasets.

### Soundness
3

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
3

### Summary
This paper proposes MRAD, a memory-retrieval anomaly detection framework for zero-shot anomaly detection (ZSAD). Instead of learning a parametric mapping from features to labels, MRAD retrieves directly from a feature–label memory bank, avoiding training overhead and potential information loss. The authors design three variants: MRAD-TF, a training-free model using frozen CLIP encoders; MRAD-FT, which introduces two lightweight linear layers for fine-tuning retrieval metrics; and MRAD-CLIP, which incorporates region priors from MRAD-FT into CLIP’s learnable prompts for improved cross-modal alignment. Experiments on 16 industrial and medical datasets show that MRAD consistently outperforms state-of-the-art baselines such as AnomalyCLIP, FAPrompt, AdaCLIP, and WinCLIP, achieving superior performance with high efficiency and robustness.

### Strengths
1. MRAD reframes zero-shot anomaly detection as a non-parametric retrieval problem rather than a traditional model-fitting task. The proposed two-level (image- and pixel-level) memory bank is conceptually simple yet effective, marking a meaningful departure from existing CLIP-based prompt-learning approaches.

2. The experimental evaluation is extensive, covering 16 datasets across both industrial and medical domains. The results demonstrate the robustness and generalization ability of the proposed method.

### Weaknesses
1. While the empirical results are strong, the paper lacks theoretical justification or analytical insight into why a retrieval-based framework can outperform traditional parametric fitting approaches. The memory mechanism has been explored extensively in few-shot anomaly detection, and the novelty here lies in extending it to the zero-shot setting. Therefore, the authors should provide a more detailed discussion on why and how features extracted from the source domain can generalize effectively to target-domain detection tasks.

2. The overall reading flow of the paper could be improved, as certain sections are difficult to follow, and the mathematical formulations appear unnecessarily dense, which affects readability. The authors are encouraged to simplify equations where possible

3. It would be valuable to investigate whether AnomalyCLIP, when equipped with a similar “vanilla” memory mechanism, could achieve comparable performance. This comparison would help to more clearly demonstrate the effectiveness of the proposed memory-retrieval design, which is positioned as the main innovation of this work.

### Questions
See Weaknesses

### Soundness
3

### Presentation
2

### Contribution
2
