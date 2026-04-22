# VTBench: Comprehensive Benchmark Suite Towards Real-World Virtual Try-on Models

- Avg Score: 6.00
- Decision: Reject
- Scores: 6, 8, 4

## Abstract
While virtual try-on has achieved significant progress, evaluating these models towards real-world scenarios remains a challenge. A comprehensive benchmark is essential for three key reasons: (1) Current metrics inadequately reflect human perception, particularly in unpaired try-on settings; (2) Most existing test sets are limited to indoor scenarios, lacking complexity for real-world evaluation; and (3) An ideal system should guide future advancements in virtual try-on generation.
To address these needs, we introduce the **V**irtual **T**ry-on **Bench**mark (**VTBench**), the first-ever hierarchical try-on benchmark suite that systematically decomposes virtual image try-on into hierarchical, disentangled dimensions, each equipped with tailored test sets and evaluation criteria. VTBench exhibits three key advantages: 1) Multi-Dimensional Evaluation Framework: The benchmark encompasses five critical dimensions for virtual try-on generation (*e.g.,* overall image quality, texture preservation, complex background consistency, cross-category size adaptability, and hand-occlusion handling). Granular evaluation metrics of corresponding test sets pinpoint model capabilities and limitations across diverse, challenging scenarios. 2) Human Alignment: Human preference annotations are provided for each test set, ensuring the benchmark’s alignment with perceptual quality across all evaluation dimensions. 3) Valuable Insights: Beyond standard indoor settings, we analyze model performance variations across dimensions and investigate the disparity between indoor and real-world try-on scenarios. To foster the field of virtual try-on towards challenging real-world scenarios, VTBench will be open-sourced, including all test sets, evaluation protocols, generated results, and human annotations.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a hierachical virtual try-on benchmark for virtual try-on evaluation on multple aspects, including both the overall and localized texture quality of generated images. The authors use CLIP, DINO and QWen models to construct specific metrics for each aspect and address the image evaluation under the unpaired setting. The proposed benchmark also includes human preference labels to test the metrics' perceptual alignment with human.

### Strengths
1) The motivation of this paper precisely targets the major challenge in virtual try-on evaluation: coarse similarity metrics that are inconsistent with human perception and unsuitable for texture detail evaluation.

2) Font texture similarity is a novel metric that has been overlooked in prior work, but is very important in real-world setting where it's necessary to preserve brand's text logo.

3) The proposed benchmark is valuable to the virtual try-on community.

### Weaknesses
1) Using VLM model to determine size fitness is not convincing. It is difficult to evaluate if a generated garment fits the original body shape because of clothing-body occlusion in the clothed model image. In addition, size fitness itself can also be decomposed into three categories: oversized/loose fitting, normal fitting and tight fitting. The authors provide human alignment scores in the experiments, but it's better to see more evidence showing that the size evaluation in VLM is accurate. Perhaps some visualizations on randomly-selected triplets and their QWen responses can be helpful.

2) The hand consistency metric evaluates hand/joint structure but does not include various hand artifacts that are common in many virtual try-on images. I would suggest changing the name to avoid confusion or including hand artifacts detection (e.g., skin color incosistency and blurry finger edge).

3) I suggest the author provide more details of the data filtering criteria and human evaluation, including sources and number of human annotators.

### Questions
1) When evaluating background semantic consistency, what's the justification of choosing DINO instead of QWen VLM?

2) How is the overall rank calculated in Table 1? I assume it is not a simple average since the six metrics all have different magnitudes.

3) The sentence at L417 is cut off.

4) Figure 3 and Figure 5  show results on five dimensions. I suggest the authors also include results of fidelity. It is important as it serves as a baseline to show how mismatched FID/KID is with human judgment.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposes VTBENCH , a comprehensive benchmark suite for evaluating image-based virtual try-on (VTON) models. The authors argue that existing evaluation methods (like FID/KID) align poorly with human perception and lack consideration for real-world complex scenarios. VTBENCH addresses these issues with a hierarchical evaluation framework that decomposes VTON quality into three main categories and six dimensions: General Image Quality (Similarity, Aesthetics), Garment Preservation (Texture, Size), and Auxiliary Consistency (Background, Hand) .

### Strengths
- The paper's most outstanding strength is its large-scale human preference annotation study. The results  robustly demonstrate that the proposed new metrics (e.g., for cross-category, texture, hand, and background consistency) are highly correlated with human perceptual judgments , a feature broadly lacking in existing metrics. The VLM-based "Size Fitness" metric and the OCR-based "Font Texture Similarity" (FTS) metric  are highly innovative. They elevate evaluation from low-level pixel similarity to high-level semantic and logical correctness.

- The curation of four new specialized test sets (CBC, FTF, CSF, HOC) is a significant contribution, specifically targeting common failure cases like hand occlusions , complex backgrounds , and cross-category try-ons  that are ignored by previous benchmarks.

- The comprehensive benchmark analysis of 16 SOTA models provides valuable insights to the community regarding the pros and cons of different architectures (GAN vs. UNet vs. DiT) .

### Weaknesses
- **Questionable Efficacy of Visual Texture Metric**: The paper computes the cosine similarity between CLIP or DINO embeddings of the original garment and the cropped garment region from the generated image  to judge visual texture. However, CLIP and DINO are trained via contrastive or self-supervised learning, which are not inherently designed to enhance fine-grained details. For example, generative methods like IP-Adapter, which use CLIP as an image encoder, often fail to restore reference image details. Furthermore, AnyDoor uses DINO as an encoder but still requires a high-frequency filter to ensure detail consistency. This casts doubt on the ability of CLIP or DINO to reliably measure fine-grained texture.
- **High Dependency of Background Consistency Metric on Masks**: In the measurement of background consistency , most try-on methods are mask-based, and the masked regions differ between methods. The fit and extent of a given mask will heavily influence the paper's background consistency metric. When evaluating the 16 baselines on the CBC dataset, was a standardized, pre-computed mask provided to all models?
- **Limited Scope of Hand Consistency Metric**: The Hand Consistency metric  focuses on the consistency of the model's preserved regions, but focusing only on hands is somewhat limited. Since try-on images are often half- or full-body, the hand region has a low pixel ratio. Other elements like the model's hair, face, skin tone, and body shape are visually more critical to the realism of the result and should also be evaluated for consistency.

### Questions
- When evaluating the 16 baselines on the CBC dataset, was a standardized, pre-computed mask provided to all models? If so, how was this mask generated? If not (i.e., each model used its own internal mask), how does this metric fairly compare different models (as the score might reflect the mask's size more than the background preservation quality)?

- Why did the authors choose to focus exclusively on hand consistency? Are there plans to expand this consistency evaluation to other non-edit regions that are critical for perceptual realism, such as facial fidelity, hair details, and skin tone consistency?

- Given the known limitations of CLIP/DINO in capturing high-frequency details (as noted in the "Weaknesses" section), can the authors provide more evidence that $E_{\epsilon}$  reliably measures fine-grained texture (rather than just style or the overall shape)? Have the authors considered supplementing this with other detail-focused metrics (e.g., high-frequency variants of LPIPS or SigLIP which is used in FLUX Redux) to enhance the "Texture Fidelity" dimension's evaluation?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces VTBench, a benchmark for evaluating virtual try-on (VTON) models. The key contributions are three-fold: 1 A hierarchical evaluation framework that decomposes virtual try-on quality into six fine-grained dimensions; 2 It introduces several unpaired evaluators to overcome the lack of paired ground-truth images; 3 It provides four custom test datasets with human preference annotations, where a model comparison study is conducted to evaluate different VTON models.

### Strengths
+ This work establishes a foundation for future research toward realistic and perceptually aligned virtual try-on systems.
+ It evaluates 16 state-of-the-art models across multiple paradigms, offering valuable comparative insights for the community.
+ It tailors existing models for virtual try-on evaluation from different perspectives.

### Weaknesses
- The aesthetic metric correlates poorly with humans, significantly discrediting the reliability of its use in evaluating and comparing VTON models.
- It lacks comparison with commonly used full-reference and no-reference image quality assessment metrics.
- Its claim to guide the development of future VTON models is somewhat overstated, since the benchmark itself doesn’t propose novel generative methods or optimization strategies.

### Questions
N/A

### Soundness
3

### Presentation
2

### Contribution
3
