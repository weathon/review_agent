# SeMoBridge: Semantic Modality Bridge for Efficient Few-Shot Adaptation of CLIP

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 6, 2

## Abstract
While Contrastive Language-Image Pretraining (CLIP) excels at zero-shot tasks by aligning image and text embeddings, its performance in few-shot classification is hindered by a critical limitation: intra-modal misalignment. This issue, caused by a persistent modality gap and CLIP's exclusively inter-modal training objective, leaves the embedding spaces uncalibrated, making direct image-to-image comparisons unreliable. Existing methods attempt to address this by refining similarity logits or by computationally expensive per-sample optimization.
To overcome these challenges, we introduce SeMoBridge, a lightweight yet powerful approach that directly addresses the misalignment. Our method maps images into the text modality, while keeping their semantic content intact through what we call a Semantic Modality Bridge. SeMoBridge is closed-form and can optionally be trained through multi-modal supervision, combining image and text-alignment losses to optimize the projection. Experiments show that the trained version, SeMoBridge-T, requires only a fraction of the training time while overall outperforming other methods, particularly in low-data scenarios (1, 2, and 4 shots).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper pinpoints an overlooked limitation of CLIP in few-shot classification—namely intra-modal misalignment caused by the modality gap—and proposes SeMoBridge, a lightweight projection that maps image embeddings into the text modality. A closed-form, training-free version and an optionally fine-tuned variant (SeMoBridge-T) are introduced; the latter is trained with a multi-modal loss while keeping CLIP frozen. Experiments show consistent state-of-the-art accuracy in 1/2/4-shot regimes on several benchmarks with 5-10× less computation than prior methods such as Cross-the-Gap.

### Strengths
(1) Clear motivation and visual evidence of intra-modal misalignment; (2) Simple global projection applicable to all inputs, avoiding per-sample optimisation; (3) Efficient fine-tuning that never back-propagates through CLIP encoders; (4) Strong empirical gains and detailed ablations; (5) Low engineering overhead, making the method practical.

### Weaknesses
(1) Theoretical justification is limited; why a single linear map suffices is not deeply analysed; (2) Evaluation is restricted to image classification with ViT-B/16; effect on larger backbones, other modalities, and tasks (retrieval, detection) is unknown; (3) Only image-to-text projection studied—no comparison with text-to-image or shared latent bridges; (4) Dependence on prompt engineering is unclear; (5) Robustness across seeds and potential degradation of zero-shot performance are not reported.

### Questions
(Q1) What exact closed-form solution is used—Procrustes on class centroids or something else? (Q2) Does constraining the projection’s rank change results? (Q3) What happens if both query and support images are projected before comparison? (Q4) Can a bridge trained on one dataset transfer to another without re-tuning? (Q5) How much does SeMoBridge affect original CLIP zero-shot accuracy?

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
4

### Summary
The paper introduces SeMoBridge, a lightweight method designed to address the challenge of intra-modal misalignment in CLIP (Contrastive Language-Image Pretraining) when applied to few-shot classification tasks. CLIP, while highly effective in zero-shot settings, struggles with few-shot learning due to its inability to calibrate the image modality during training. The authors propose a Semantic Modality Bridge (SeMoBridge) to map images into the text modality, thus enabling more reliable comparisons within CLIP's shared embedding space. The method operates with minimal computational cost and offers significant improvements in accuracy, particularly in low-data settings (1-4 shots). A trained version of SeMoBridge (SeMoBridge-T) further enhances performance through multi-modal supervision. Extensive experiments across 11 datasets confirm that SeMoBridge outperforms state-of-the-art methods, especially in low-shot scenarios.

### Strengths
1. The proposed SeMoBridge effectively resolves the intra-modal misalignment that impacts CLIP's performance in few-shot learning tasks. The approach of mapping images into the text modality for alignment is both novel and practical.

2. SeMoBridge is training-free, reducing computational cost by eliminating the need for expensive per-sample optimization. Even in the trained version (SeMoBridge-T), the method requires minimal training time compared to existing alternatives, such as APE and Tip-Adapter.

3. The paper reports extensive experiments across various benchmarks, demonstrating SeMoBridge’s effectiveness in low-shot scenarios (1, 2, 4 shots). The experiments also show SeMoBridge’s robustness to distribution shifts, making it applicable to real-world settings.

### Weaknesses
1. While the method is demonstrated to work well on low-shot settings, there is a lack of evaluation on very large-scale datasets beyond the commonly used benchmarks like ImageNet, Caltech101, and others. Testing on more diverse datasets could provide a clearer picture of SeMoBridge's generalization capabilities.

2. The method heavily relies on the CLIP model for both image and text embeddings. While CLIP is a state-of-the-art model, this dependence limits the method's applicability to tasks where CLIP's pre-trained features may not be optimal or where fine-tuning CLIP is necessary.

3. SeMoBridge's mapping from image embeddings to the text modality, while effective for alignment, might potentially obscure fine-grained visual details in some cases, as it primarily leverages textual descriptions for image alignment.

4. The multi-modal supervision in SeMoBridge-T shows significant performance improvement, especially in low-data settings. However, the requirement for text supervision may not always be practical in real-world applications where text descriptions for every class might not be readily available.

### Questions
1. How does SeMoBridge handle scenarios where textual prompts or descriptions are unavailable or difficult to generate for specific classes? Would a zero-text version of SeMoBridge be possible, and how would it compare in performance?

2. How does SeMoBridge perform in settings with noisy or ambiguous class descriptions? Are there mechanisms in place to mitigate the potential issues arising from imperfect text prompts?

3. Can SeMoBridge be extended to multi-modal tasks beyond few-shot image classification, such as multi-modal retrieval or object detection? What are the implications of using SeMoBridge for tasks where visual details are more critical than text alignment?

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
1

### Summary
CLIP variants and few-shot image classification are out of my expertise, and I write these review comments based on my general knowledge and understanding.

This paper focuses on the intra-modal calibration problem, where images with different classes may introduce confusions. To address this problem, the authors propose SeMoBridge, which integrates the fine-grained text modality to optimize image representations. Therefore the performance on few-shot image classification is improved. The overall structure is lightweight and very easy to train, and it also supports a training-free setting.

### Strengths
- The training paradigm is very efficient and lightweight, and only a projector/adapter is needed. Approximately less than 1M parameters are required and the whole training process requires less than one minute.
- The proposed method is effective and outperforms other baselines. Besides, SeMoBridge is better in fewer shots.
- The writing and overall presentation is good, and the structure is well organized.
- The experiments are solid and 11 datasets are used for evaluation.

### Weaknesses
- More shots lead to smaller performance gap compared to the baselines in Figure 4, 5, and Figure 6 (right). Further explanations should be provided.
- It seems SeMoBridge could be a plugin to any CLIP-related tasks, but current paper does not extend it to further domains and tasks.
- Besides Figure 2 right and Figure 7, there lacks of a quantitative metric to explicitly express the levels of intra-modal calibration problem. Currently, all the results are based on general final performance.

### Questions
- Does few-shot text classification also has such a intra-modality calibration problem, and can we map these texts to images for effective calibration?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper considers the problem of the modality gap in CLIP's vision–language modeling, which leads to an embedding space with potentiallhy undesirable properties and arises from its inter-modal contrastive training. The paper focuses in particular on undesirable intra-modality comparisons in the visual embeddings. 

To address these, the paper proposes a simple mapping of visual embeddings into the textual embedding space, which is particularly fast to obtain, as it can be done in a training-free setup or with quick fine-tuning.

### Strengths
- The approach is super-fast.

### Weaknesses
- The paper proposes a simple technique to obtain embeddings of images in the text embedding space and claim that this addresses the modality gap. While it is true that the resulting embeddings reside in the same space, it was always clear that you can achieve such a projection to have embeddings in the same space. However, genuinely solving CLIP's modality gap refers to something else: It involves developing an improved learning paradigm that leads to embeddings that enable us to do everything we can normally do with CLIP plus also have an embedding space without any modality gap. The submission investigates a single use case (image few-shot classification) without considering the other requirements.

- While the modality gap is well-established, the motivation in Figure 1 seems more abstract: The left side shows an idealized example where a dog image is closer to a cat centroid. However, it is  unclear whether such a case (with these specific example images) would ever occur with CLIP. I imagine this would happen more with easier to confuse classes. The right side of Figure 1 appears to be a more data-driven analysis, but again lacks crucial details for reproducibility, e.g. the rows and columns are not labeled with any classes, so it is hard to know in what instances this problem might occur.

- The approach is directly based on SD-IPC (Ding et al., 2023), a method to create a textual embedding for images. The authors adapt this approach for CLIP-based few-shot classification.

- The success of the approach hinges a lot on the quality of the textual descriptions. The paper uses custom prompts for each dataset to obtain appropriate class descriptions instead of a generic solution.

### Questions
- Can you report additional experiments on standard CLIP downstream tasks?

- Is there a way to avoid the use of class-specific bias (CSB) vectors? They appear to severely limit the generalizability of the approach.

### Soundness
2

### Presentation
2

### Contribution
1
