# ReLayout: Integrating Relation Reasoning for Content-aware Layout Generation with Multi-modal Large Language Models

- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
Content-aware layout aims to arrange design elements appropriately on a given canvas to convey information effectively. Recently, the trend for this task has been to leverage large language models (LLMs) to generate layouts automatically, achieving remarkable performance. However, existing LLM-based methods fail to adequately interpret spatial relationships among visual themes and design elements, leading to structural and diverse problems in layout generation. To address this issue, we introduce ReLayout, a novel method that leverages relation-CoT to generate more reasonable and aesthetically coherent layouts by fundamentally originating from design concepts. Specifically, we enhance layout annotations by introducing explicit relation definitions, such as region, salient, and margin between elements, with the goal of decomposing the layout into smaller, structured, and recursive layouts, thereby enabling the generation of more structured layouts. Furthermore, based on these defined relationships, we introduce a layout prototype rebalance sampler, which defines layout prototype features across three dimensions and quantifies distinct layout styles. This sampler addresses uniformity issues in generation that arise from data bias in the prototype distribution balance process. Extensive experimental results verify that ReLayout outperforms baselines and can generate structural and diverse layouts that are more aligned with human aesthetics and more explainable.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes an approach, ReLayout, to add layout constraints to an existing layout dataset. ReLayout is based on two methods, Layout Relation-CoT Construction method to recover and reconstruct the relationships of the design elements in a hierarchical manner, and Layout prototype Rebalance Sampler to adjust the sample distribution via weighted sampling.

### Strengths
1. The paper is generally easy to follow.

2. The relation data added to the dataset may be useful.

### Weaknesses
1. Novelty of the proposed idea: This work tries to incorporate some low-level constraints (the relationships among the design elements, such as negative space, alignment, non-overlapping, and saliency) into graphic design layouts. This paper presents it in a way like it is never considered before. In fact, it is largely studied by existing works. For example, [1] explicitly models different types of layout constraints for graphic design; [2] considers the empty space (or negative space) and overlapping among design elements; and [3] considers saliency information in layout generation. This paper does not discuss what existing works have done.

2. Problems with the Layout Relation-CoT construction method. First, I am not sure why the generation of relations is related to CoT. While the relations may be generated in a hierarchical manner, I do not see how it is a step-by-step generation. In fact, I am really confused how this method can even be remotely related to CoT. Second, the recovery of the relations among the design elements is actually rather straightforward. I do not see how this method is novel.

3. Problems with the Layout Prototype Rebalance Sampler. First, by adjusting the sample distribution, I agree that less frequent samples will get sampled more frequently. However, would this also create a negative effect that popular designs become less popular while rare designs become popular? Second, why is the layout prototype rebalance sampler novel? Where is it novel? This is not discussed in the paper.


[1] Peter O’Donovan, Aseem Agarwala, and Aaron Hertzmann. Learning Layouts for Single-Page Graphic Designs. TVCG, 2014.
[2] Xinru Zheng, Xiaotian Qiao, Ying Cao, and Rynson Lau. Content-aware generative modeling of graphic design layouts. TOG (ACM SIGGRAPH), 2019.
[3] Daichi Horita, Naoto Inoue, Kotaro Kikuchi, Kota Yamaguchi, and Kiyoharu Aizawa. Retrieval-Augmented Layout Transformer for Content-Aware Layout Generation. In CVPR, 2024.

### Questions
See my comments/questions in the Weaknesses section.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The goal of this paper is to integrate relation reasoning for content-aware layout generation with existing MLLMs. A new data transformation mechanism is proposed to add relation annotations by decomposing the overall layout into a hierarchical structure. A layout prototype rebalance sampler is further used to enhance the layout diversity. The experiments demonstrate the effectiveness of the proposed method in terms of layout quality and diversity.

### Strengths
1. The proposed method is a reasonable solution to improve the performance of existing MLLMs on the layout generation task. The experiments on two public datasets should show the improvements in terms of layout structure and diversity.
2. The proposed data construction and resampling mechanism produces additional annotation details on existing layout datasets, which could be useful for future research in the community.

### Weaknesses
1. There is a gap between the motivation and the proposed method. The motivation of this paper is inspired by the Chain-of-thought (CoT) that progressively obtains element relations step by step. However, CoT belongs to an inference-time technique, while the proposed method belongs to data transformation for training. I am not sure how the stated relation-CoT is used in MLLM training and inference. The structure-level understanding and the high-level layout design concepts (L82-L84) should be illustrated more clearly.
2. The technical contributions are relatively limited. The two mechanisms proposed by this paper, i.e., layout relation-CoT construction and prototype rebalance sampler, belong to dataset engineering works. First, the construction mechanism mainly uses several heuristic rules to decompose layout element relationships into a hierarchical structure. These rules have been thoroughly discussed in traditional graphic design works. Second, the rebalance sampler utilizes a simple clustering method to reconsider the effect of different layout clusters.
3. Both quantitative and qualitative results cannot show a significant performance improvement between the proposed method and existing works. Based on the results in Table 1, compared to RALF and LayoutPrompter, the proposed method can only achieve comparable or slightly better results on the CGL dataset. Take the third column in Fig.6 as an example. The graphic design created by the proposed method also contains an alignment issue.
4. The saliency map is derived from an existing Infrared Small Target Detection work (L691-L693). The choice of adopting object detection work here is questionable. In addition, visual saliency detection on natural images is different from that on graphic designs.
5. The limitations stated on the last page are not very clear. It would be better to discuss some failure cases with detailed examples.

### Questions
1. What is the advantage of using HTML, rather than other formats like JSON, as the output sequence?
2. Why use 8 feature culsters in the layout prototype rebalance sampler? Are there any examples to show the typical layout difference between these 8 clusters? Fig. 5 only shows some examples in 3 clusters.

### Soundness
2

### Presentation
2

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
The paper studies the problem of content-aware layout generation.

The paper shows that explicit modeling of element relationships in layout generation models based on multi-modal large language models (MLLM) can improve the generation performance. The paper contributes a CoT-like approach that generates element relationships besides element coordinates to output more structured layouts, and a clustering-based sampling method that balances training samples of different styles for generating more diverse layouts.

### Strengths
1. Content-aware layout generation is an important research problem to study.

2. The idea of explicitly modeling element relationships in content-aware layout generation is interesting and novel, which could inspire the layout generation community.

3. The augmented HTML-based layout representation with element relationship information is well designed, and the proposed sampling method is shown to be effective.

### Weaknesses
1. The quantitative results of the proposed method on PKU are not satisfactory. As shown in Table 1, on PKU, the proposed method is not very helpful for improving the performance of MLLMs on the content metrics, i.e., readability and occlusion. 

2. The evaluation is not complete. First, this work represents element relationships in terms of hierarchical regions and element margins. Comparison with some alternative representations is needed, but is missing in the current paper. For example, one possible alternative is to extract pairwise location and size relationships between elements from existing layout annotations, as in prior work (e.g., LayoutFormer++), and represent them in HTML or JSON format. Second, in the ablation study (Section 4.5), the effectiveness of the added margin attribute in Section 3.2 is not tested, and the importance of the three feature vectors ($\mathbf{f}_i^{\text{s}}$, $\mathbf{f}_i^{\text{r}}$, $\mathbf{f}_i^{\text{e}}$) in Section 3.3  are not evaluated. Third, the effect of the number of clusters in Section 3.3 is not tested.

### Questions
1. In Section 3.3., what statical features are extracted from $\mathcal{R}_i$?
2. What is the actual number of samples used for training the proposed model on CGL and PKU, respectively?
3. In Table 4, V0 is better than V1, V2 and V3 in terms of readability. What is the reason for it?

### Soundness
2

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
The paper introduces ReLayout, a method for content-aware layout generation. The authors identify two main challenges in previous approaches: structural and diversity problems. To address these issues, they present relation-CoT, decoupling the output space into three dimensions: saliency, region, and element. Furthermore, they also propose a rebalance sampling strategy to ensure the output diversity. Experiments on PKU and CGL datasets demonstrate improvements over baseline models.

### Strengths
The paper clearly demonstrates the problems (overlap, alignment errors, lack of diversity) in existing approaches and proposes a well-designed method to address the issues. The relation-CoT methodology is simple and effective, introducing minimal modification to the layout format while achieving noticeable performance improvements. In addition, the authors conduct extensive experiments, including quantitative, qualitative comparison, and user studies to show ReLayout's superior performance in all aspects.

### Weaknesses
- It is confusing that how ReLayout could preserve the aspect ratio of elements using the proposed methodology. For example, as shown in Figure 1, due to the error alignment of the text boxes, PosterLlama produces distorted elements. It is not clear how this issue is addressed by introducing layout relation-CoT.
- When predicting the elements, its style contains both the margin attributes and the bounding box information. In such formulation, each element have 5 attributes, which inevitably causes positioning conflicts. What is the motivation to use this output format? Are there any ablation studies on removing the margin attributes from the output?
- The implementation details of the features $f_i^s$ and $f_i^r$ are not included in the paper.
- Due to the rebalance sampling strategy, ReLayout appears to be inefficient. How many samplings are needed to ensure coverage of all 8 clusters? The experiment results do not include relevant analysis on this.

### Questions
1. I understand that ReLayout organizes the layouts into a more structured format. However, why is the relation-CoT paradigm fundamentally better than direct coordinate prediction? Specifically, there may still be overlap between two regions. Therefore, placing elements within regions does not fundamentally solve the problem of overlap between elements. 
2. Will the predicted element exceed the boundary of its region?
3. Do the authors evaluate the reading order in the generated layout? A good layout should take the reading order of elements into consideration.

### Soundness
2

### Presentation
3

### Contribution
2
