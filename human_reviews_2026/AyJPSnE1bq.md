# WOW-Seg: A Word-free Open World Segmentation Model

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Open world image segmentation aims to achieve precise segmentation and semantic understanding of targets within images by addressing the infinitely open set of object categories encountered in the real world.  However, traditional closed-set segmentation approaches struggle to adapt to complex open world scenarios, while foundation segmentation models such as SAM exhibit notable discrepancies between their strong segmentation capabilities and relatively weaker semantic understanding. To bridge discrepancies, we propose WOW-Seg, a Word-free Open World Segmentation model for segmenting and recognizing objects from open-set categories. Specifically, WOW-Seg introduces a novel visual prompt module, Mask2Token, which transforms image masks into visual tokens and ensures their alignment with the VLLM feature space. Moreover, We introduce the Cascade Attention Mask to decouple information across different instances. This approach mitigates inter-instance interference, leading to a significant improvement in model performance. We further construct an open world region recognition test benchmark: the Region Recognition Dataset (RR-7K). With 7,662 classes, it represents the most extensive category-rich region recognition dataset to date. WOW-Seg attains strong results on the LVIS dataset, achieving a semantic similarity of 89.7 and a semantic IoU of 82.4. This performance surpasses the previous SOTA while using only one-eighth the parameter count. These results underscore the strong open world generalization capabilities of WOW-Seg. The code and related resources are available at https://github.com/AAwcAA/WOW-Seg-Meta.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces WOW-Seg, a novel word-free, open-world segmentation model that leverages a Vision-Language Model (VLM) to recognize objects from visual prompts alone. The core contributions include the Mask2Token module, which converts image masks into visual tokens aligned with the VLM's feature space , and the Cascade Attention Mask, designed to mitigate interference between multiple object instances during processing. The authors also present RR-7K, a new large-scale region recognition dataset with 7,662 categories, to better evaluate open-world models.

### Strengths
**SOTA Performance with Exceptional Efficiency.** WOW-Seg achieves outstanding results, setting a new state-of-the-art on challenging benchmarks like LVIS and PACO by a significant margin. This is particularly impressive given its remarkable efficiency; the 1B parameter model consistently outperforms much larger competitors, highlighting the architecture's effectiveness.


**New Benchmark for the Community.** The authors contribute the RR-7K dataset, a large-scale and challenging new benchmark for open-world evaluation. With over 7,600 categories, it addresses a critical need for more comprehensive testing and provides a valuable resource to drive future research in model generalization.

### Weaknesses
### **Major**

**Unfair Comparison of Models**

The paper emphasizes that the 1B-parameter WOW-Seg outperforms larger models like PAM (3B) and DAM (8B). This comparison is potentially misleading as WOW-Seg is built upon InternVL3-1B, a highly advanced (and fictionally dated 2025) backbone. The baseline models may use older, less capable backbones, where a larger parameter count does not guarantee better performance. To ensure a fair comparison, the authors should provide context on the architectures and pre-training of the baseline models or conduct experiments with same base MLLM.

**Limited Motivation**
The motivation for the Cascade Attention Mask is underdeveloped and lacks a clear, compelling justification. The paper posits that objects should be treated as independent during segmentation to avoid being "affected by such correlations" that exist in the natural world (e.g., a cup and water).

While this principle is stated, the paper fails to sufficiently explain why this correlation is detrimental to the model's performance. The authors mention preventing "inter-instance interference" and "detrimental information leakage", but these concepts are presented without concrete examples or analysis. 


**Missing Citations**

The paper relies on several models and datasets without providing formal citations , including Qwen2.5-VL, InternVL-3.0, LVIS, PACO, Sentence BERT, Cityscapes, ADE20K, PASCAL COCO Stuff, etc.


### **Minor**

Inconsistent Naming: The model is referred to as "WOW-SEG" , "WOW-Seg" , and "wow-seg". Component names also vary, such as "Mask Generater" in a figure and "Mask Generator" in the text. Consistency should be maintained.

Vague Definitions: The term "mask generators" is used without specifying which one (e.g., SAM, SAM 2) was employed for the experiments or how it was configured.

### Questions
Please check Weaknesses

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces a novel approach to open-world segmentation without predefined label space. The authors present RR-7K, a new benchmark with significantly more evaluable categories than existing ones. Their proposed two-stage segmentation pipeline first uses a universal mask generator to produce instance masks and then employs a multi-modal large language model for mask region classification. This pipeline achieves superior results on both official benchmarks and the new RR-7K dataset.

### Strengths
1. Benchmark Contribution: The proposed RR-7K benchmark represents a critical step for open-vocabulary segmentation, featuring a substantially larger category set than previous datasets and thus offering greater real-world relevance.

2. Performance: The novel method achieves strong, state-of-the-art results on both the new RR-7K benchmark and other established datasets.

3. Presentation: The paper is clearly presented with effective and high-quality writing.

### Weaknesses
1. **Insufficient Dataset Analysis and Justification**
The paper introduces a new benchmark, RR-7K, but provides insufficient deep analysis and justification for its design choices. While basic class distributions are provided, the authors fail to rigorously validate the dataset's claims, specifically:

    **Real-World Relevance**: The paper lacks compelling evidence that the adopted label system and class distribution are a closer approximation to real-world data compared to existing large-scale benchmarks (e.g., ImageNet-22K).

    **Source of Distribution**: It is unclear whether the resulting class distribution is an intended design choice or merely a bias introduced by the specific mask generation and classification models used in the dataset curation pipeline.

   A more detailed, comparative analysis of RR-7K's categorical structure (including granularity and scale) against established benchmarks is needed to solidify its importance and justify its adoption.

2. **Limited Novelty of the Segmentation Pipeline**
The proposed two-stage segmentation pipeline, while effective, appears to have limited algorithmic novelty. The core architecture—using a universal mask generator followed by a classification stage—closely resembles several existing methods in the open-vocabulary segmentation literature (e.g., [1], [2], [3]).  While the integration of a Multi-modal Large Language Model (MLLM) for classification is a modern substitution, the paper does not convincingly demonstrate that this change is a sufficiently novel contribution to anchor the method as the primary technical advance. The main contribution appears to reside more in the RR-7K dataset and its extensive evaluation, which is conflict with the layout of current submission.

[1] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4015–4026, 2023. (Mentioned in the text prompt part)


[2] Ding, Z., Wang, J. and Tu, Z., 2022. Open-vocabulary universal image segmentation with maskclip. In Proceedings of the 40th International Conference on Machine Learning.

[3] Xu M, Zhang Z, Wei F, Lin Y, Cao Y, Hu H, Bai X. A simple baseline for open-vocabulary semantic segmentation with pre-trained vision-language model. In European Conference on Computer Vision 2022 Oct 22 (pp. 736-753).

### Questions
1. A deeper analysis of the proposed benchmark is needed.

2. Will the dataset be released?

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
4

### Summary
This paper introduces WOW-Seg, a Word-free Open World Segmentation model designed for robust segmentation and semantic understanding across open-set object categories. It incorporates Mask2Token, which converts image masks into visual tokens aligned with the VLLM feature space, and Cascade Attention Mask, which reduces inter-instance interference. The authors also propose RR-7K, a large-scale region recognition benchmark with 7,662 categories. WOW-Seg achieves state-of-the-art performance on LVIS while using only one-eighth of the parameters, demonstrating strong open-world generalization and efficiency.

### Strengths
**Originality:** The paper introduces two novel and effective components. The first is Mask2Token, a visual prompt module that transforms masks into tokens, which cleverly embeds mask features directly into the VLLM's existing feature space. The second is the Cascade Attention Mask , a custom attention mechanism designed to solve inter-instance interference. This allows for efficient multi-mask processing by explicitly decoupling the features of different masks and their corresponding text outputs during autoregressive generation.

**Significance:** WOW-Seg achieves state-of-the-art results, surpassing prior models on LVIS (e.g., +4.1 Semantic IoU) and PACO , while being substantially more parameter-efficient (1B parameters vs. 3B-8B for SOTA competitors). Furthermore, the introduction of the RR-7K dataset is a major contribution, providing a much-needed benchmark to evaluate true open-world generalization, on which prior SOTA models perform poorly.

**Clarity:** The paper is well-written and easy to follow.

### Weaknesses
- Dependency on Mask Generator Quality: The model's "word-free" recognition is fundamentally dependent on the quality of the masks provided by an external "Mask Generator" (like SAM) during inference.
- The Mask2Token module, while effective, appears computationally intensive. It requires cropping each mask region and feeding it through the visual encoder to extract tokens. The computational cost and inference speed are not analyzed.
- Several closely related works [1,2,3,4] are missing from the discussion. The authors should provide a more in-depth analysis of these studies and incorporate them into the comparative evaluation to ensure a comprehensive and fair comparison.
- The paper lacks a quality analysis of the newly proposed RR-7K benchmark.

[1] Kawano, Yasufumi, and Yoshimitsu Aoki. "Tag: Guidance-free open-vocabulary semantic segmentation." arXiv preprint arXiv:2403.11197 (2024).

[2] Ülger, Osman, Maksymilian Kulicki, Yuki Asano, and Martin R. Oswald. "Auto-vocabulary semantic segmentation." In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 24266-24275. 2025.

[3] Rewatbowornwong, Pitchaporn, Nattanat Chatthee, Ekapol Chuangsuwanich, and Supasorn Suwajanakorn. "Zero-guidance segmentation using zero segment labels." In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1162-1172. 2023.

[4] Shin, Heeseong, Chaehyun Kim, Sunghwan Hong, Seokju Cho, Anurag Arnab, Paul Hongsuck Seo, and Seungryong Kim. "Towards open-vocabulary semantic segmentation without semantic labels." Advances in Neural Information Processing Systems 37 (2024): 9153-9177.

### Questions
- PAM was trained on a large-scale dataset, yet it performs poorly on the RR-7K dataset. In contrast, WOW-Seg, which was trained only on a combination of LVIS, PACO, and COCO-Stuff datasets, achieves significantly higher performance on RR-7K. The authors are encouraged to provide an analysis explaining this discrepancy.
- In the open-vocabulary segmentation results presented in Table 2, it would be beneficial to include comparisons with more recent state-of-the-art methods, such as CAT-Seg, to provide a stronger benchmark evaluation.
- The proposed Cascade Attention Mask is designed to ensure that object masks are independent of each other. However, the results indicate that the multiple-masks-per-sample (MM) strategy outperforms the single-mask-per-sample (SM) strategy. The authors should clarify why MM remains effective if the masks are intended to be independent.

### Soundness
3

### Presentation
3

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
This article focuses on the problem of open world image segmentation. It introduces a word-free open world segmentation model for segmenting and recognizing objects from open set categories. The core idea is to encode multiple masks as visual prompts and autoregressively identifies each mask. It develops Mask2Token for mask encoding and  Cascade Attention Mask for detection. Further, the article introduces a new dataset, RR-7K, which features with 7000+ categories.

### Strengths
- The work contributes a new dataset, RR-7K, which holds notable potential to benefit the research community. 
- The proposed method achieves state-of-the-art performance across multiple benchmarks

### Weaknesses
- The motivation behind the cascade attention mechanism is not sufficiently justified or explained. While the authors acknowledge that “in the natural world, there are inherent correlations between objects,” they ignore this prior knowledge in their design — which seems counterintuitive.

- Furthermore, the proposed cascade attention scheme lacks a formal mathematical description. The current exposition relies heavily on intuition.

- For the dataset, I am unsure whether we indeed need to recognize 7K categories. As observed in Table 7, many categories appear semantically overlapping or synonymous (e.g., "audience" vs. "audience member", "ad board" vs. "advertisement board"). This suggests potential redundancy that could inflate category counts without adding meaningful discriminative power. Moreover, the paper provides no statistical analysis of the dataset, such as, label distribution.

- The current visualizations focus primarily on specific objects, however, it is expected to show the overall segmentation results of entire images. 

- As the leading information that authors intend to share, the caption of Fig. 1 is minimal and lacks sufficient context. TBH, it is not easy to grasp all details at the first glance.

### Questions
- Why authors disregard the inherent semantic relationships among objects when designing the cascade attention mask?

- Could you provide details on the label distribution across the categories in the RR-7K dataset?

- While InternVL-78B is employed to filter hallucinations during data annotation, how can you ensure that the model itself does not introduce hallucinations?

- Osprey exhibits poor performance on established benchmarks like LVIS and PACO, yet achieves relatively good results on the proposed RR-7K dataset. What explains this discrepancy? A thorough investigation into this is essential to validate RR-7K as a fair, representative, and necessary benchmark.

### Soundness
3

### Presentation
2

### Contribution
2
