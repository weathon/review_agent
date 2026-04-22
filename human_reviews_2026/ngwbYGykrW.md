# FG-CLIP 2: A Bilingual Fine-grained Vision-language Alignment Model

- Avg Score: 5.33
- Decision: Reject
- Scores: 6, 6, 4

## Abstract
Fine-grained vision-language understanding requires precise alignment between visual content and linguistic descriptions, a capability that remains limited in current models, particularly in non-English settings. While models like CLIP perform well on global alignment, they often struggle to capture fine-grained details in object attributes, spatial relations, and linguistic expressions, with limited support for bilingual comprehension. To address these challenges, we introduce FG-CLIP 2, a bilingual vision-language model designed to advance fine-grained alignment for both English and Chinese. Our approach leverages rich fine-grained supervision, including region-text matching and long-caption modeling, alongside multiple discriminative objectives. We further introduce the Textual Intra-modal Contrastive (TIC) loss to better distinguish semantically similar captions. Trained on a carefully curated mixture of large-scale English and Chinese data, FG-CLIP 2 achieves powerful bilingual performance. To enable rigorous evaluation, we present a new benchmark for Chinese multimodal understanding, featuring long-caption retrieval and bounding box classification. Extensive experiments on 29 datasets across 8 tasks show that FG-CLIP 2 outperforms existing methods, achieving state-of-the-art results in both languages. We will release the model, code, and benchmark to facilitate future research on bilingual fine-grained alignment.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a bilingual fine-grained vision-language alignment model for English and Chinese. It extends CLIP-style architectures by integrating fine-grained supervision signals and introducing a new Textual Intra-modal Contrastive (TIC) loss. The training follows a two-stage paradigm: global alignment with long and short captions, and fine-grained regional alignment using multiple discriminative objectives. Authors curate large-scale bilingual datasets and construct several Chinese benchmarks to evaluate fine-grained and bilingual understanding. Experiments on 29 datasets and 8 tasks show consistent improvements over existing methods in retrieval, detection, segmentation, and as a vision encoder in large multimodal models (LMMs)

### Strengths
This paper proposes a comprehensive bilingual fine-grained vision-language alignment model for English and Chinese. The two-stage training design systematically transitions from global to fine-grained alignment, combining both caption lengths and region-level signals. Explicit bilingual integration with curated English and Chinese datasets totaling over ~2.4 billion pairs supports multilingual robustness. The approach bridges the gap between previously disjoint English fine-grained and Chinese global models, enhancing cross-lingual generalization. The resulted model FG-CLIP2 verified on 29 datasets and 8 tasks shows consistent improvements over existing methods

### Weaknesses
- The presentation in this paper could be improved, especially in the Approach section. For example, Section 3.2 does not introduce the $L_{FGV}$ and $L_{FGT}$ loss. If these losses are defined in FG-CLIP (or elsewhere), they need be mentioned. In Section 3.3 on training data,  which is the key factor enabling the model’s bilingual capability, the paper lacks essential visualizations and analysis, such as dataset construction details, sample examples, and composition distribution. Moreover, there appears to be no experimental analysis of how different data components contribute to bilingual performance improvement.For example, is there evidence that the inclusion of Chinese training data improves the model’s English alignment capability, given the larger overall training corpus?
- As one of the core contributions, the Textual Intra-modal Contrastive (TIC) mechanism lacks sufficient experimental validation. For instance, no visualization or embedding analysis is provided to support the claimed improvement in semantic separability.
- Similarly, the two-stage training strategy from coarse to fine alignment is not thoroughly evaluated. All results are reported only for the final model, without distinguishing between training stages. Therefore, the claim that the second stage effectively enhances fine-grained capability is insufficiently verified.
- Finally, the comparison with previous works should include the amount of training data used for each model.

### Questions
Section 4.1 mentions that the first-stage training uses ASCEND 910B NPUs, while the second stage uses H800 GPUs. How long did each stage cost, and why was the second-stage training not continued on NPUs?

### Soundness
3

### Presentation
2

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
This paper designs a bilingual vision-language model to enhance fine-grained alignment between visual content and linguistic descriptions for both English and Chinese. It adopts a two-stage training paradigm with multiple discriminative objectives and is trained on large-scale bilingual datasets, outperforming existing models across 29 datasets and 8 tasks.

### Strengths
1. The paper is well-written and easy to follow.

2. The proposed bilingual pre-training framework and the design of loss objectives exhibit certain innovativeness.

3. Experimental results fully demonstrate the effectiveness of the proposed method.

### Weaknesses
1. FG-CLIP 2 has achieved impressive performance improvements compared to FG-CLIP. I wonder which part of the design contributes the most significantly to the performance improvement? Does it come from the gain brought by using SigLIP 2 for initialization? Does the simultaneous use of bilingual data have a mutually promoting effect?

2. Five losses used in the paper are assigned different weights, yet there seems to be no ablation experiment to justify why such weight settings are adopted. In addition, the results of the ablation experiment in Table 7 are not very significant.

3. Some related papers (e.g., [1][]2) should be discussed and compared.

[1] UMG-CLIP: A Unified Multi-Granularity Vision Generalist for Open-World Understanding. ECCV 2024.

[2] Contrastive Localized Language-Image Pre-Training. ICML 2025.

### Questions
Please refer to the 'weaknesses' part.

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
The paper presents FG-CLIP2, a bilingual fine-grained vision-language model, designed to advance fine-grained alignment for both English and Chinese. Trained on a carefully curated mixture of large-scale English and Chinese data, FG-CLIP 2 achieves SOTA performance. Additionally, a new benchmark for Chinese multimodal understanding is contributed.

### Strengths
1. Addresses the need for English-Chinese bilingual fine-grained vision-language understanding, catering to non-English scenario demands.
2. A new benchmark suite to advance evaluation in Chinese multimodal understanding is contributed.
3. FG-CLIP 2 outperforms existing methods, achieving state-of-the-art results in both languages.

### Weaknesses
1. The paper’s focus is confusing. For bilingual capability, it only mentions adding English-Chinese mixed data without other innovations. The differences between FG-CLIP2 and previous fine-grained vision-language models are not clarified.
2. There is no analysis of how previous methods would perform when trained on this dataset, affecting the fairness of experimental comparisons.
3. The training objectives involve too many hyperparameters, but the paper does not explain how these hyperparameters are determined, raising questions about the reliability of the training process.

### Questions
refer to Weaknesses

### Soundness
2

### Presentation
2

### Contribution
2
