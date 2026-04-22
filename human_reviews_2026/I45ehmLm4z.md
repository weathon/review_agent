# Describe-to-Score: Text-Guided Efficient Image Complexity Assessment

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 4, 4, 8

## Abstract
Accurately assessing image complexity (IC) is essential for many vision tasks, yet existing approaches rely almost exclusively on visual features and therefore fail to capture the high-level semantics that humans often use when judging complexity. We introduce a multimodal perspective for IC modeling by integrating visual representations with caption-derived textual semantics. This integration enriches the representational space and provides complementary structural cues that are difficult to infer from vision alone. From an information theoretic and representation viewpoint, we offer an idealized analysis suggesting how semantic guidance can regularize the hypothesis space and support more stable generalization. We present D2S (Describe-to-Score), a framework that generates natural-language descriptions using a pretrained vision–language model and aligns the visual encoder with textual structure through feature alignment and entropy distribution alignment. These mechanisms encourage the visual backbone to internalize semantic regularities during training. Importantly, D2S employs text only during training and maintains a vision-only inference pipeline with no additional computational overhead. Experiments show that D2S achieves state-of-the-art performance on the IC9600 benchmark and remains competitive on no-reference image quality assessment (NR-IQA) tasks. Additional studies demonstrate robustness across captioners and prompt designs, and improved semantic transferability on downstream probing tasks, highlighting the effectiveness of multimodal guidance for complexity-related modeling.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces a text-guided vision encoder only method for image complexity assessment (ICA).
In the training phase, the text feature from CLIP text encoder and the visual feature from the vision encoder are aligned with proposed entropy distribution alignment (EAL) and feature alignment (FAL).
The vision encoder then obtains a vision-text aligned (multi-modal) feature which has effectively reduced the empirical Rademacher complexity and improved generalization, after the training.
In experiments, the proposed method outperforms previous works on IC9600 benchmark, and it shows fast adaptation in early stage of the training.

### Strengths
The proposed method performs a form of vision-text fusion for IC modeling.
- Through theoretical background (Section 2), it demonstrates that utilizing text features can achieve increased accuracy and generalization, from which the core components of the proposed method (EAL and FAL) are derived.
- Compared to existing studies that utilize high-level information such as object counts (Shen et al., 2024) and motion trends (Li et al., 2025), the proposed method can leverage more flexible high-level information through text.

Key Features of the Proposed Method
- For vision-text feature alignment, EAL employs energy distance loss (Szekely & Rizzo, 2013) while FAL adopts InfoNCE loss (van den Oord et al., 2019).
- The text encoder is kept frozen while only the vision encoder is trained, enhancing efficiency during inference by utilizing only the vision encoder.
- D2S outperforms existing methods on the IC9600 benchmark.

Experimental Advantages
- Demonstrates advantages over existing methods in a small samples training (Table 2).
- Conducts ablation study (Table 5) to show that both EAL and FAL are necessary.
- Shows applicability to downstream tasks, including NR-IQA.

### Weaknesses
Lack of empirical justification for optimal image captioning template design.
- The core idea of the proposed method is to reduce effective feature dimensions by utilizing effective semantic (text) features (Theorem 2).
- To achieve this goal, designing an optimal image captioning template (Figure 5) is considered crucial. However, the paper lacks sufficient discussion regarding the criteria used for this design and the underlying rationale. For instance, questions remain unanswered: Is the template sufficiently rich in textual description? Is the template selected through extensive experimental validation? Clear justification for the template design choices is not provided.

Limited novelty in EAL and FAL architectures.
- The forms of EAL and FAL represent common architectures for vision-text feature alignment that have been widely adopted in other tasks beyond ICA. These structures are not novel from an architectural perspective, nor can they be considered specifically tailored for ICA.

Minor experimental design concerns.
- Caption model selection: Why was BLIP used instead of more recent, superior captioning methods?
- Vision encoder choice: Why was ResNet employed instead of CLIP's vision encoder?
- Limited benchmark evaluation: Why were results compared only on IC9600 without evaluation on other benchmarks?
- Comparative analysis: Figure 7 would benefit from direct comparison with vision-only approaches for more reasonable evaluation.

### Questions
Regarding Proposition 1 (Eq2).
- I am not sure even after reading A.1.
- For example, is it valid when alpha=0.1, beta=0.1 especially for the left inequality?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Describe-to-Score (D2S) tackles image complexity assessment by first generating captions with BLIP, then aligning vision and text to predict a scalar complexity score from visual features alone.  It introduces entropy distribution alignment (EAL) to match modality entropy statistics and a CLIP-style feature alignment (FAL) in a shared space.  The authors motivate D2S with information- and generalization-theoretic arguments (higher fused entropy; reduced effective dimensionality) and implement learnable pooling for efficient inference.  Experiments on IC9600 show state-of-the-art correlations and faster latency, while transfer to NR-IQA yields competitive results, notably on KADID-10K.

### Strengths
- The paper proposes a distinctive “describe → align → score” pipeline for image complexity: captions from a VLM guide visual features during training, while inference remains image-only—a way to inject semantics without runtime cost.  It further introduces Entropy Distribution Alignment (EAL) with an energy-distance loss and buffers/EMA to stabilize cross-modal statistics, plus CLIP-style Feature Alignment (FAL)—a creative combination that is new in ICA.    The information- and generalization-theoretic motivation (higher fused entropy; reduced effective dimension) gives the method conceptual clarity and elevates its potential impact on complexity assessment.  

- Method details are concrete: the paper specifies the projection/connector, formulates EAL analytically, and illustrates the full training/inference workflow with clear figures and a prompt template.     Empirically, D2S attains state-of-the-art correlations on IC9600 with notable latency advantages, and shows competitive transfer to NR-IQA—evidence of real-world significance beyond a single benchmark.

### Weaknesses
- The paper posits that “entropy increases → richer representation → closer to true complexity,” and relies on entropy‐distribution alignment plus feature alignment (EAL/FAL), but gives no operational, reproducible definition for estimating $p(\cdot)$ or verifying the premise that “semantic compression reduces effective dimension.” Please add a concrete mapping from features to distributions (e.g., temperature-scaled softmax or KDE), run controlled synthetic tests to validate (or falsify) the “dimension compression” hypothesis, and include ablations that hold effective dimension fixed while toggling text guidance.  

- Only one captioner/encoder pairing is explored. Please provide a 2D grid over captioners (e.g., BLIP variants) × prompt designs (length/order/style), measure performance vs. compute, and analyze which caption attributes (entity count accuracy, relation coverage) correlate with complexity prediction.

### Questions
See Weaknesses.

### Soundness
2

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
This paper proposes D2S, a model for Image Complexity Assessment (ICA) that integrates vision-language learning. D2S leverages BLIP to generate textual descriptions (captions) for images, thereby injecting high-level semantic information into the visual encoder during training.
The method introduces two alignment mechanisms — Feature Alignment (FAL) and Entropy Distribution Alignment (EAL) — to align the textual and visual feature spaces.
Experiments show that D2S achieves competitive results on both ICA and NR-IQA tasks while maintaining low parameter count and inference latency, demonstrating its efficiency and scalability.

### Strengths
1. The paper explores an interesting idea: leveraging textual information for image complexity assessment. The training-time multimodal alignment with BLIP-generated captions and visual features is technically well-motivated.
2. The theoretical derivation from information theory and generalization theory provides conceptual depth and connects intuition to formal analysis.
3. The model achieves good performance–efficiency tradeoff, with low parameter count and short inference time.

### Weaknesses
1. Caption quality and reliability are critical yet under-analyzed. Figures 17 and 18 mention that the final generated text (“the overall visual complexity is...”) can often be incorrect, but the paper does not further analyze how such errors influence D2S’s performance. A detailed study or ablation on the four BLIP prompts would make this much stronger.
2. The paper could better discuss the role of the Projection module and whether it affects the training of the core visual encoder, especially since it is not used at inference time.
3. The main experiments (Table1)  are limited to IC9600, making it difficult to confirm generalization across other ICA datasets.
4. Possible typographical errors exist in Table 4 (e.g., TOPIQ, LoDa), which need verification.

### Questions
1. Since the projection module (as shown in Figure 4) is used during training but discarded during inference, could its presence unintentionally affect the optimization of the visual encoder?
2. Why were generalization experiments conducted only on datasets such as Nagle4k and Savoias, without performing full-scale experiments similar to IC9600?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper presents D2S (Describe-to-Score), a novel framework for image complexity assessment (ICA) that integrates visual and textual semantic information. The method first uses a pre-trained vision-language model (BLIP) to generate image captions and then aligns visual and textual features through two key mechanisms: Entropy Distribution Alignment (EAL) and Feature Alignment (FAL). Importantly, D2S employs multimodal information during training but only requires visual input at inference, achieving both semantic richness and computational efficiency.
Comprehensive experiments on multiple datasets (IC9600, KADID-10K, and others) show that D2S attains state-of-the-art (SOTA) performance with significantly reduced inference latency. Theoretical analyses based on information theory and Rademacher complexity further justify the proposed design.

### Strengths
- The combination of text-guided semantics with visual complexity assessment is original and well-motivated, bridging a key gap in prior visual-only ICA approaches.
- The paper provides clear theoretical arguments using entropy and generalization theory to explain the advantages of multimodal fusion.
- By discarding the text branch during inference, D2S achieves SOTA performance while maintaining low latency — a practical and elegant design choice.
- The experiments cover supervised, unsupervised, small-sample, cross-dataset, and cross-task settings. Results are consistently superior or competitive across diverse benchmarks.
- Ablation studies and error analyses effectively demonstrate the contribution of each module (EAL, FAL, AttnPool) and the benefits of semantic guidance.
- The manuscript is clearly written, logically organized, and provides sufficient implementation details for reproducibility.
- Demonstrating transfer to no-reference image quality assessment (NR-IQA) further enhances the general interest and robustness of the method.

### Weaknesses
- While D2S improves performance, the interpretability of what textual semantics contribute (beyond activation histograms) could be elaborated, for example with qualitative examples of captions’ influence.
- Since captions are generated automatically, the performance might depend on BLIP’s accuracy; this dependency and its robustness are not deeply analyzed.
- Although the ablation studies are extensive, additional comparisons with other text-guided approaches (e.g., CLIP-based fusion or textual embeddings without caption generation) would further strengthen the validation.
- Some theoretical derivations (e.g., in Proposition 1) are concise and could benefit from clearer notation or discussion of assumptions.

### Questions
- How sensitive is D2S to the quality or type of captions generated by BLIP? Would fine-tuning BLIP or using alternative VLMs (e.g., CLIP-ViT-L or Florence-2) affect results significantly?
- Have the authors evaluated how the accuracy or reliability of BLIP captions impacts image complexity estimation? Since BLIP is not a perfect captioning model and may produce incomplete or incorrect descriptions, it would be valuable to understand whether such caption errors significantly affect the downstream complexity predictions.
- Could the entropy alignment mechanism generalize to other multimodal tasks beyond ICA (e.g., aesthetics or memorability prediction)?
- During inference, since the text branch is discarded, to what extent are the visual encoders truly semantically informed versus statistically regularized by text during training?
- Is there any noticeable trade-off between performance and training time introduced by the entropy buffers and momentum model in EAL?
- Could the authors provide qualitative examples showing how textual descriptions guide the visual branch — for example, comparing visual attention maps with and without text alignment?

### Soundness
4

### Presentation
4

### Contribution
3
