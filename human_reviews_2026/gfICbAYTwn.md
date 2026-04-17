# RefAM: Attention Magnets for Zero-Shot Referral Segmentation

- Decision: Reject
- Scores: 4, 6, 6

## Abstract
Most existing approaches to referring segmentation achieve strong performance
only through fine-tuning or by composing multiple pre-trained models, often at the
cost of additional training and architectural modifications. On the other hand, large-
scale generative diffusion models encode rich semantic information, making them
attractive as general-purpose feature extractors. In this work, we introduce a new
method that directly exploits features, attention scores, from diffusion transformers
for downstream tasks, requiring neither architectural modifications nor additional
training. To systematically evaluate these features, we extend benchmarks with
vision–language grounding tasks spanning both images and videos. Our key insight
is that stop words act as attention magnets: they accumulate surplus attention
and can be filtered to reduce noise. Moreover, we identify global attention sinks
(GAS) emerging in deeper layers and show that they can be safely suppressed or
redirected onto auxiliary tokens, leading to sharper and more accurate grounding
maps. We further propose an attention redistribution strategy, where appended
stop words partition background activations into smaller clusters, yielding sharper
and more localized heatmaps. Building on these findings, we develop RefAM,
a simple training-free grounding framework that combines cross-attention maps,
GAS handling, and redistribution. Across zero-shot referring image and video
segmentation benchmarks, our approach consistently outperforms prior methods,
establishing a new state of the art without fine-tuning or additional components.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a training-free approach to improve referring segmentation by leveraging attention patterns in diffusion transformers. The key insight is that certain tokens, referred to as attention sinks or magnets, absorb a disproportionate amount of attention. By analyzing and redistributing this attention, the proposed method enhances the quality of segmentation maps without requiring fine-tuning or architectural changes. The approach, achieves SOTA results across multiple benchmarks.

### Strengths
* The paper is clearly written and well organized.

* The approach is training-free, yet achieves SOTA performance across several benchmarks.

* The method is well motivated by an empirical study of attention in diffusion transformers.

### Weaknesses
* The idea of leveraging attention sinks/magnets for feature map visualization has been explored in prior works (e.g. [1]). This paper extends such ideas to a different application.

* The inference pipeline may be computationally expensive, as it relies on both DiT and SAM. Can the proposed approach be integrated into existing referring segmentation models or simplified for more practical deployment.

[1] Darcet, Timothée, et al. "Vision transformers need registers." arXiv preprint arXiv:2309.16588 (2023).

### Questions
Please check the weakneses section

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a method for zero-shot referral segmentation, named RefAM. The model adopts a Diffusion Transformer (DiT) as its backbone and utilizes its attention maps to predict accurate segmentation maps. To mitigate the issue of global attention sinks (GAS) that hinder accurate prediction, the authors introduce the concept of attention magnets, which encourage the model to focus on more relevant regions. Furthermore, by applying an attention redistribution strategy, the segmentation maps are refined for higher accuracy. The proposed method achieves state-of-the-art performance on both image and video referred segmentation tasks.

### Strengths
- This method attempted to apply DiT as the backbone for referring segmentation and observed performance improvements. Furthermore, conducted an in-depth analysis and identification of Global Attention Sinks (GASs).
- This paper proposed stop-word-based attention magnets and a method to predict a relatively accurate segmentation map in a training-free manner through attention redistribution.
- This method achieved state-of-the-art performance in the field of referring segmentation for both images and videos.

### Weaknesses
- Generalization Issue
   - It is unclear whether the proposed mechanisms, such as GAS (Global Attention Sink) mitigation and stop-words as attention magnets, work effectively when applied to backbones other than DiT. For instance, it would be valuable to see whether these components consistently show effectiveness across different variants of CLIP or DiT.

- Dependency on Off-the-Shelf Models
   - The paper emphasizes its training-free property as its strength. However, it also appears to rely heavily on off-the-shelf models such as SAM. It would be important to evaluate how much the performance depends on these external pretrained models, and to provide a fair comparison with existing methods under equivalent conditions.

- Justification for Using DiT
    - While the paper highlights the application of Diffusion Transformer (DiT) as a key design choice, the rationale for this choice is not fully convincing. More quantitative or qualitative analysis demonstrating why DiT is particularly suitable—or superior—would strengthen the argument.

- Background on Stop Words as Attention Magnets
   - The experiments clearly show that introducing stop words as attention magnets improves performance. However, the theoretical or empirical background supporting this design choice is not well discussed. Providing references or related studies that justify the role of stop words in guiding attention could make the paper more comprehensive and grounded.

### Questions
Please provide your responses with reference to the "weaknesses" mentioned above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
RefAM aims to address referring object segmentation in a zero-shot, training-free manner. It utilizes large pre-trained diffusion transformer models (DiTs) that encode cross-attention maps between text and image tokens. The key idea is the introduction of "attention magnets", i.e., stop words or additional tokens, that help redirect or absorb unwanted attention in the cross-attention maps. This leads to a more precise grounding of the referred object.

### Strengths
1. The discovery that stop words act as attention magnets and "global attention sinks" in diffusion transformers provides valuable insights.  
2. This approach utilizes pre-trained diffusion models combined with attention manipulation, allowing the referring segmentation task to be performed without any additional training. As a result, it is widely applicable without the need for specialized labeled data.  
3. On benchmarks such as RefCOCO, RefCOCO+, RefCOCOg, and video referring segmentation (e.g., Ref-DAVIS 17), it outperforms previous training-free methods.

### Weaknesses
1. The paper mentions "using diffusion transformer models," but it's crucial to note that the specific details such as the model variant, pretraining dataset, version, and hyperparameters are significant. Any variation in these aspects could influence the results. Since RefAM employs more robust vision backbones and pretrained models, it is difficult to attribute the improvements solely to the methodology rather than to these stronger backbones.
2. There is a reliance on large pretrained models, which necessitates powerful diffusion transformer backbones (DiTs) and segmentation models like SAM. This may lead to constraints in computational power or memory.
3. The quality of the masks may vary. The method extracts heatmaps and utilizes a segmentation model instead of being trained end-to-end for precise masks. As a result, the masks may be less accurate in complex scenes featuring occlusions or ambiguous expressions.
4. The stop-word and attention magnet strategy could be heuristic in nature. This strategy involves adding tokens and filtering them, which might require fine-tuning or may not generalize effectively across all languages or types of expressions.

### Questions
1. Which variants of the diffusion transformer (DiT) and FLUX were utilized in the RefAM experiments?
2. Can the same backbones and pretrained model be used for baseline comparisons?
3. How does RefAM perform when employing weaker or older vision-language backbones? Are performance gains still noticeable when adjusting for backbone strength?
4. How does RefAM compare to other zero-shot methods in terms of latency and memory usage?
5. Is the same set of magnets effective across different backbones and benchmarks?

### Soundness
3

### Presentation
3

### Contribution
3
