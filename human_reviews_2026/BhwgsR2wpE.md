# Recycling Pretrained Classification Heads for Efficient Vision-Language Alignment

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Vision-Language Models (VLMs) with separate image and text encoders, such as CLIP, excel at tasks like zero-shot classification or cross-modal retrieval. They achieve this by embedding images and text into a shared representation space. However, their success relies on end-to-end training with large volumes of paired samples, entailing prohibitive data and computational costs. Existing post-hoc vision-language alignment methods, which map independently trained image and text encoders into a shared representation space using lightweight functions, reduce training costs but still require substantial paired data. We introduce a data augmentation approach that recycles classification head weights from ImageNet-21K pretraining and combines them with a reduced number of image-text pairs to achieve vision-language alignment. These recycled weights significantly mitigate the need for large alignment datasets, while the combination with a reduced number of image-text pairs extends alignment beyond the original ImageNet domain. We demonstrate that integrating our augmentation approach with several state-of-the-art post-hoc alignment techniques consistently boosts accuracy in cross-modal retrieval, zero- and few-shot classification tasks. Experiments confirm that our approach provides a versatile and data-efficient solution for vision-language representation alignment.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper aims to achieve vision-language alignment of individually trained image and text encoders using a small dataset of image-text pairs.
Instead of collecting a large-scale image-text dataset, the paper proposes to utilize the classification head weights of a pre-trained ImageNet-21K classifier.
Since each row of the weight matrix can be viewed as a prototype of each class, pairs of the weight rows and corresponding class name texts can be used as an additional dataset for training a model (e.g., MLP) that maps text embedding space to image embedding space.
Experimental results demonstrate that the proposed method outperforms baselines with a limited amount of image-text pair data.

### Strengths
- S1: Image-text alignment with only a small dataset is helpful for practitioners.
- S2: The proposed method can be easily implemented by simply extracting the head weights of a pre-trained classifier.

### Weaknesses
- W1: The motivation for aligning individually trained image and text encoders is ambiguous since many pre-trained CLIP models are available in general classification domains.
- W2: Recent rich unimodal image encoders are mainly trained with self-supervised learning, such as contrastive learning (e.g., SimCLR, MoCo) or masked autoencoders, rather than classification. Can the proposed method also be applied to such models?
- W3: A theoretical explanation of why the classifier weights trained with cross-entropy can be used for prototypes of a cosine classifier would be beneficial.
- W4: A direct evaluation of alignment in the embedding space could be a verification of alignment, which is the primary purpose of the paper. For example, visualizing embedding space and cosine similarity distribution, as is done in [a] and [b], would be helpful.
- W5: The generality of the proposed method is limited since it can achieve competitive accuracy with CLIP only when the domain is close to ImageNet-21K. Can other classification heads trained on other datasets be used?


[a] Eslami and de Melo, Mitigate the Gap: Investigating Approaches for Improving Cross-Modal Alignment in CLIP, ICLR 2025.   
[b] Liang et al., Mind the gap: Understanding the modality gap in multi-modal contrastive representation learning, NeurIPS 2022.

### Questions
Please refer to the weaknesses.

### Soundness
3

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
This paper proposes recycling classification head weights from ImageNet-21K pretraining as data augmentation to achieve data-efficient vision-language alignment. Experiments on cross-modal retrieval, zero- and few-shot classification tasks demonstrate the effectiveness of proposed method.

### Strengths
1. This paper proposes a simple-yet-effective way for data-efficient post-hoc vision-language alignment, which recycles classification head weights from ImageNet-21K pretraining as augmented training data. 
2. This paper also provides analysis on why we can treat pretrained classification head weights as image representations
3. Experiments on several downstream tasks show the effectiveness of proposed method.

### Weaknesses
1. The authors only highlight the improvements of their method over existing post-hoc alignment techniques such as CSA, Text-to-Concept, and MLP alignment in the context of cross-modal retrieval. However, it is also essential to validate these gains on zero- and few-shot classification tasks, as such comparisons are fundamental to demonstrating its overall effectiveness. Furthermore, it is recommended that these results be included as a table in the main paper.

2. The authors are encouraged to provide t-SNE visualizations in the embedded space, comparing the original image/text representations with the classification head weights. Such visualization would help reveal the potential modality gap and offer a more comprehensive understanding of the alignment effect.

3. Regarding the third zero-shot configuration (ImageNet-21K & 1 Image-Caption pair per class), the inclusion of one image from each class in the evaluation datasets raises the question of whether this setting constitutes a strictly zero-shot scenario.

4. The proposed method appears to bind the classification head weights to a specific image encoder. For instance, when using head weights derived from ViT-B/16 pre-training, the image encoder must also be the pre-trained ViT-B/16. If this understanding is correct, it would limit the flexibility and general applicability of the approach. Besides, the pretrained classification head weights only contain 21K samples, which may cause scalability concerns.

5. Can the pretrained classification head weights be leveraged to enhance a pre-trained CLIP model? Specifically, can we fine-tune adapters on the 21K data samples derived from these weights to improve CLIP performance?

Minor Issue:
In Line 304-305, $g$ and $\overline{g}$ should be exchanged. $g$ is the identity and $\overline{g}$ is the MLP.

### Questions
See Weaknesses.

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
3

### Summary
This paper proposes a pragmatic, data-efficient method to perform post-hoc alignment between independently pretrained image and text encoders. The key idea is to recycle the row vectors (class-weight vectors) from classification heads learned during ImageNet-21K pretraining and pair each weight vector wi with a text embedding of the corresponding class name fT(ti). These (wi, fT(ti)) pairs are used as additional training data (Dweights) and combined with a (potentially small) set of image–text pairs (Dimgtxt) to learn lightweight alignment mappings (g, g) with existing post-hoc methods (e.g., CSA, text-to-concepts, or a small MLP).

### Strengths
1. Recycling classification head weights is simple, computationally cheap, and leverages resources (pretrained weights) that are commonly discarded.
2. The authors test multiple alignment methods (CSA, text-to-concepts, MLP) and several vision encoders, evaluate on retrieval (Flickr30K) and a diverse set of nine classification benchmarks, compare ImageNet-1K vs ImageNet-21K weights, and present ablations (adding one image-caption per class).

### Weaknesses
1.The authors observe a significant geometric separation between weight vectors and image features (an "image-weight modality gap") but proceed by simply concatenating both modalities into the same alignment dataset. There is little exploration of principled ways to handle this gap (e.g., modality-specific normalizations, learned projections for weight vectors, or reweighting strategies).
2. The approach requires access to the pretraining classifier weights and associated human-readable class names. Many commercial or proprietary vision models do not expose their classification-layer weights or class-label vocabulary.

### Questions
1. You report a statistically significant separation between weight vectors and image features. Have you tried learning a dedicated lightweight projection (or simple normalization / scaling) for the weight vectors prior to alignment, or adding modality-specific alignment heads? If so, does that improve retrieval and classification compared to the current union strategy? If not, can you comment on the expected benefit and computational cost?
2. Your strongest results rely on ImageNet-21K classification heads. For target domains with little or no overlap with ImageNet concepts (e.g., medical imaging, non-photographic domains), what is the recommended strategy? Can you (or will you) provide experiments using alternative large-scale label sets (e.g., Places365, iNaturalist) or show how selecting a subset of ImageNet-21K weights relevant to the target domain affects performance?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents a data-efficient post-alignment method for aligning vision and language representations. Building on post-hoc alignment techniques, it reuses classification head weights from ImageNet21K pretraining as a form of data augmentation， by treating the head weight as concept representation, combining them with a smaller set of image–text pairs. This approach reduces training cost while maintaining strong performance. When integrated with existing post-hoc alignment methods, it consistently improves results on cross-modal retrieval and zero- or few-shot classification, offering a versatile and efficient alternative to full-scale end-to-end vision–language training.

### Strengths
The idea of enhancing post-hoc alignment by averaging multiple image representations per class to form a prototype is novel and appealing. It improves efficiency by consolidating representations, though it sacrifices some diversity in the text–image relationship. As a result, the model becomes primarily classification-oriented, losing finer details such as compositional or structural information present in the original images and text.

### Weaknesses
Insufficient related work discussion: Lines 136–161 largely replicate content from [1], yet fail to include a proper comparison or discussion, even though [1] demonstrates stronger performance than the chosen baseline.


Contradictory motivation: The motivation emphasizes using minimal paired data for post-hoc alignment, but the main experiments rely on a CLIP text encoder trained on massive paired datasets—undermining the stated goal.


Experimental issues:


- The first experiment trains and tests on Flickr30k, which fails to test generalization—the key strength of CLIP. The improvement over the baseline is partly due to adding 21k new classes, making comparisons unfair. A fair baseline should use 21k ImageNet images with captions and apply the vanilla post-hoc alignment method.


- The second experiment aligns the text encoder to a classification head and evaluates it on overlapping datasets with ImageNet21k, essentially exploiting the evaluation setup. This converts a strong non–zero-shot model into a pseudo–zero-shot one rather than demonstrating true zero-shot ability.


Overall, the experiments lack evidence of generalization or transfer learning benefits, which are central to CLIP’s purpose.

[1] Zhang, Le, Qian Yang, and Aishwarya Agrawal. "Assessing and Learning Alignment of Unimodal Vision and Language Models." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

### Questions
how would the model really transfer to downstream task, say coco retrieval or other tasks such as MMVP, Winoground, which tests fine-grained and compositional understanding of the model beyond simple classificaiton?

### Soundness
2

### Presentation
2

### Contribution
2
