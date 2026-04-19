# Visually Consistent Hierarchical Image Classification

- Decision: Accept (Poster)
- Scores: 3, 5, 8, 5, 6

## Abstract
Hierarchical classification predicts labels across multiple levels of a taxonomy, e.g., from coarse-level \textit{Bird} to mid-level \textit{Hummingbird} to fine-level \textit{Green hermit}, allowing flexible recognition under varying visual conditions.  It is commonly framed as multiple single-level tasks, but each level may rely on different visual cues.  Distinguishing \textit{Bird} from \textit{Plant} relies on {\it global features} like {\it feathers} or {\it leaves}, while separating \textit{Anna's hummingbird} from \textit{Green hermit} requires {\it local details} such as {\it head coloration}.  

Prior methods improve accuracy using external semantic supervision, but such statistical learning criteria fail to ensure consistent visual grounding at test time, resulting in incorrect hierarchical classification.  We propose, for the first time, to enforce \textit{internal visual consistency} by aligning fine-to-coarse predictions through intra-image segmentation. Our method outperforms zero-shot CLIP and state-of-the-art baselines on hierarchical classification benchmarks, achieving both higher accuracy and more consistent predictions. It also improves internal image segmentation without requiring pixel-level annotations.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The paper explores hierarchical image classification (HIC) task, which is old but worthy to explore. I have experienced many bad reviews in this domain, for instance, “the proposed method relies on coarse labels and therefore not useful in the real world”. However, I think they are the misunderstanding of this area and I want to give some new one.

### Strengths
NA

### Weaknesses
Overlap with former work? I noticed that an important reference, TransHP: Image Classification with Hierarchical Prompting (NeurIPS 2023), is not cited in your paper. This work may be directly relevant, as it appears to share similarities with your proposed method, specifically in the use of different ViT blocks for different levels of hierarchy. 

Limited novelty. Your proposed approach introduces elements such as Superpixel and Graph pooling. While these are effective, both are well-established techniques in computer vision. A more detailed explanation of how these additions provide novel contributions within the hierarchical framework would clarify the unique aspects of your work.

Limited evaluation. ImageNet, as a large-scale hierarchical dataset, might provide a stronger test of your method's capabilities compared to the smaller datasets currently used. 

Formatting problem: Please ensure the readability of all components of the paper. For instance, the font size in the tables is small, which may make it difficult for readers to check the data presented

### Questions
Please specific what is the difference between your work and TransHP?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This work aims to tackle hierarchical image classification. It is motivated by a hierarchical image segmentation work, which also conducts image classification. This work makes an extension on this basis and proposes a tree KL loss to deliver semantic consistency predictions regarding hierarchy. Evaluation on three datasets verifies the effectiveness over the baseline, which focuses on segmentation rather than classification.

### Strengths
1. The overall presentation is easy to follow. There is no difficulty in understanding the work.
2. The improvement on the baseline is considerable.

### Weaknesses
1. The technical contribution is not enough. Specifically, the proposed method contains visual consistency and semantic consistency. However, the major design and implementation of visual consistency are directly borrowed from CAST [1]. The claim in L260-264 is not convincing. Simply adding supervision in each decoding level can not be considered a vital contribution compared to directly using the overall architecture of CAST.

[1]. Learning hierarchical image segmentation for recognition and by recognition. ICLR 2024

2. The review only contains hierarchical image classification. However, pixel-level hierarchical classification (i.e., hierarchical image segmentation [2,3,4]) can also provide insights for this work. In fact, the visual consistency mentioned in this work is conducting segmentation on the image, and the baseline method (CAST) is also a hierarchical image segmentation work. Therefore, a literature review on hierarchical image segmentation should be included.

[2] AIMS: All-Inclusive Multi-Level Segmentation for Anything. NeurIPS 2023.

[3] Hierarchical Open-vocabulary Universal Image Segmentation. NeurIPS 2023

[4] LOGICSEG: parsing visual semantics with neural logic learning and reasoning. ICCV 2023. 


3. Concatenate labels from all levels to create a distribution has already been explored in [3,4]. The difference is [3,4] using cross-entropy loss, while this work uses KL divergence. 

4. The motivation for using KL divergence instead of cross-entropy is unclear. Since CE loss contains both the KL term to minimize the differences between distribution and a punishment term to minimize the uncertainty. Why KL divergence is better than CE? Both experiments and the theoretical explanation should be provided.

5. The evaluation of the proposed method is limited. 

      i) The comparison to hierarchical counterparts only contains out-of-date methods published before 2022 and the baseline (focusing on segmentation rather than classification). The top-leading solutions (e.g., [5]) are all ignored.

      ii) The label hierarchy is shallow, with only up to 3 hierarchical levels.

[5]. TransHP: Image Classification with Hierarchical Prompting. NeurIPS 2023

6. The claims in 4.6 are similar to the insights provided in [1].

Overall, the technical contribution, evaluation, and insights provided by this work are all limited.

### Questions
Why KL divergence is better than CE? Both experiments and the theoretical explanation should be provided.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This article deals with hierarchical image classification using neural networks. This classification is actually composed of several classifications performed at different levels of precision (coarse to fine). It is important that the classes detected for an image at different levels match the logical organization of the labels. Labels are organized in a tree structure, where each level of precision is a level of the tree. Therefore, there must be a direct path between the different classes identified in an image for the classification to be correct. 

So we understand that in hierarchical classification, there are different levels at which to judge the quality of the classification. At each level, there's the accuracy with respect to the expected label, and there's the logical connection between classifications at different levels.

A simple solution would be to use a single encoder to represent the image and different classification heads for each hierarchical level. The article explains that this is a bad solution because it puts the different hierarchical levels in competition with each other because they require different levels of representation. The current state of the art divides the architecture into independent branches, each of which generates a representation for a different hierarchical level. The authors have found that this separation has led to a non-causality in the classification of the different levels (decisions are independent), which could lead to inconsistencies on the relationship between classifications of different levels. 


The article focuses on CAST, a variant of the transformer network that does not slice the image into patches but into superpixels and integrates a hierarchical structure by including superpixel merging. The architecture thus goes from the fine to the coarse level. Thus, the authors have found that this architecture is adapted to the problem of hierarchical classification by adopting a fine-to-coarse classification logic. Therefore, they have adopted the same architecture, but instead of classifying the different levels in parallel like the state-of-the-art, they classify them sequentially. Thus, the super pixel embedding and the class token pass through this architecture, and in the course of this processing, the class token is classified several times. The classifier heads follow each other and classify at increasingly coarse levels. 

The authors also propose to combine two losses, one which is the cross entropy of the independent classification at each level, and one which is the concatenation of the probability of all classes (renormalized) where several classes are expected (one for each level). Therefore, the losses promote a local and a global level of good classification. 

In the experimental section, the authors show that their sequential classification allows a good improvement in the collection of metrics (local and structural) compared to architectures that process the task in parallel. In addition, parallel architectures are larger because they have to divide the network into as many branches as there are hierarchical levels. The authors show that a smaller network can outperform a larger one if it is designed to match the structure of the problem to be solved. 

The authors also added an ablation study to investigate the advantage of fine-to-coarse over coarse-to-fine, and the influence of both local and structural loss. The authors also show that hierarchical training improves segmentation results.

### Strengths
The paper is well written and organized. The authors have made a good effort to put the problem in the context of the state of the art. The problematic behavior of the state-of-the-art architecture is well illustrated in Figures 1 and 2. In general, the illustrations and explanations allow a non-expert to understand the specificity of hierarchical classification, the state-of-the-art choices, and the resulting problems. I also appreciated the care taken in explaining the metrics to understand what they represent and how they complement each other. 

The solution proposed by the authors makes sense, as they have found that CAST's architecture is very well suited to solve this problem. The explanations and schematics of the architecture are clear, I just had to go to the CAST article to understand how super-pixel pooling is done (which is not critical to understanding the article). The two losses also make sense, as the authors realized that the problem of the state of the art comes from an independent resolution at each level of the hierarchy, and added a loss that forces a (more) globally correct classification. The experimental part (Figure 4, TICE metric) shows that their model makes fewer structural errors (compared to the tree structure), even if it is smaller.

Table 2 and the different ablations clearly show the impact of the improvements made. The last part, which shows that hierarchical learning can improve segmentation, is an interesting addition, which may lead us to think of an opening towards hierarchical segmentation (for which super-pixel-based embedding seems to be well suited).

On a more personal note, at a time when the tendency is to build gigantic, high-consumption networks, I find it appreciable to see methods demonstrating that designing a solution specifically for a problem can increase its efficiency.

### Weaknesses
In part 4.4, I'm not sure how to interpret the result. We can assume that the result is a failure if the superpixel segmentation doesn't make sense. But to me, this just shows that superpixel segmentation is not necessarily adapted to a semantic problem, because it's done at the color level, and semantic in images is not necessarily associated with color.  

In Figures 1 and 4, I find that the space between class names (like "Chimpanzeeferret" in Figure 1) and their position/alignment in the tree can be difficult to read. In two-word classes, you might want to break a line, because at first sight the second part of the name seems to be another class link to another node.

### Questions
The following paragraphs are meant to be an open reflection. I'm not an expert in hierarchical classification, so I may have missed some key parts of the problem, modeling, and overall reflection. 

I think fine-to-coarse works well here because it fits the network architecture, not because it's a better choice overall. Indeed, if the hierarchical structure of the labels is a tree, one could simply put all the effort into classifying the fine level and infer the higher levels by going up the tree, since each class has only one parent. Of course, this would mean that a single error on the fine level would invalidate all other classifications. 

Overall, I think the coarse-to-fine direction makes more sense, since it's all about descending a tree and thus reducing the search field of the lower layers. I just don't think this architecture is suited for that. However, it may be possible to design a sequential architecture in coarse-to-fine logic, for example a U-net. We could imagine a U-net or an encoder-decoder using the CAST structure with a symmetric decoder. It would then be necessary to find a way to divide the super-pixels. The classification will be perform sequentially by going up the decoder

Whether it's one way or the other (but rather the other), I think that the sequential classifications should directly influence each other (like in a Markov chain or a transition table), and not just as different normalizations of the classification probabilities.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This work proposes H-CAST, a model built upon CAST for hierarchical image classification, by addressing both visual consistency and semantic consistency across predictions at different hierarchical levels. To achieve this, hierarchical supervision at different network layers and a tree loss are introduced. Experiments on three datasets verify the proposed method can achieve better performance than the baseline method.

### Strengths
1. The motivation for visual consistency is sound.
2. This work introduces new metrics for hierarchical classification, which can measure the coherence of hierarchical predictions.
3. The proposed method achieves SOTA on all datasets.

### Weaknesses
1. The key point of this paper, which is making classifiers at different levels attend to consistent visual cues, lacks support from quantitative results or theory. For example, if wrongly classified cases are all associated with incorrect CAMs? What is the transfer rate after adopting the proposed model? Qualitative comparisons are subjective and lack statistical significance. In addition, Grad-CAM is an approximation and does not truly explain how the network operates.
2. This work relies heavily on CAST. Though the authors propose the concept of visual consistency, its implementation is directly borrowed from CAST. Consequently, the primary contributions of this work are merely the hierarchical supervision loss and tree KL loss, neither of which can ensure visual consistency. Additionally, the fine-to-coarse training strategy is also derived from CAST. The tree-path KL loss is trivial.
3. The tree KL loss defines the ground truth distribution according to the number of hierarchical levels (i.e., 1/L). It would be beneficial to address whether the proposed solution can effectively manage larger hierarchies, particularly those with depths of up to 10 or 20.
4. While the clustering module would potentially guide the model to attend to spatially-coherent regions, it is unclear why the model would “ensure that each hierarchical classifier focuses on the same corresponding regions”. In fact, the model still has the opportunity to find shortcuts (attending to different regions in different levels as Fig. 2) and meanwhile deliver correct classification results. In addition, I expect visualizations of Grad-CAM results in different hierarchical levels from the proposed model. 
5. The experiments are somewhat questionable. Firstly, the latest hierarchical competitor, HRN, published in CVPR'22, is relatively outdated. Secondly, the experimental results of HRN differ from those reported in the original publication. Thirdly, there are several competitors [a-b] (might be more, not carefully checked), released before two months prior to the DDL, that outperform this work.
6. Results in Fig. 5 do not totally make sense to me. Examples in a) and b) are not equally recognizable, i.e., all examples in b) are much harder to distinguish/group than those in a). As a result, it is hard to confirm whether poor clustering of pixels in b) is the cause or the effect of incorrect predictions. One way for improvement is to examine for similar hard-level images in a) whether correct predictions are achieved along with better clustering results. 
7. According to [c], existing work has explored various loss functions for hierarchical classification. It would be useful to compare the effectiveness of tree KL loss against these alternatives. Of course, the analysis should also be provided.
8. The comparison to hierarchical approaches is unfair. While FGN and HRN use ResNet-50 as the backbone， this work adopts ViT-S.
9. More top-leading hierarchical classification work should be included in the comparison.
10. The datasets used for evaluation are small. larger datasets with complex hierarchy (ie.g., ImageNet-1K, iNaturalist) should be evaluated to better assess effectiveness.
11. How about the training and inference speed of the proposed method? Given the incorporation of superpixels and segmentation in addition to classification, it is necessary to provide a comparison of resource costs, including both time and memory usage.
12. The literature review is far from complete. In addition to [a, b, c], numerous efforts in hierarchical scene parsing is totally missing; see  related work section in [d, e]. As a top-conference paper, a comprehensive literature review is a basic requirement. I believe the reference part should be greatly extended. 
13. In fact, the hierarchical loss function in [d] is superior to the proposed Tree-PATH KL loss, in that it guarantees hierarchy-aware coherent predictions while the proposed loss cannot. A strict quantitative comparison of the two loss functions should be provided.
14. Minor issues include: a) the usage of the terms "interpretability" (L447) and "explainability." b) citation format. c) vector images. d) missing period after Eq. 2. e) the presentation of Eq. 3.
15. A Conclusion section should be added to properly conclude the work and offer insights of downside of impact of the work. 
[a] HLS-FGVC: Hierarchical Label Semantics Enhanced Fine-Grained Visual Classification.
[b] Hierarchical multi-granularity classification based on bidirectional knowledge transfer.
[c] Hierarchical classification at multiple operating points. NeurIPS 2022
[d] Deep Hierarchical Semantic Segmentation. CVPR 2022
[e] LogicSeg: Parsing Visual Semantics with Neural Logic Learning and Reasoning, ICCV 2023

### Questions
Please see the above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes H-CAST architecture for hierarchical classification tasks. The architecture builds on top of prior CAST work: superpixels are fed into a ViT network, where periodic graph pooling operation aggregates the tokens of high similarity. This produces a fine-to-course hierarchy of features. Linear layers at each level of the hierarchy are used as classification heads. The paper also presents tree-path KL loss, where the entire path in the hierarchical class tree is matched. The method shows strong performance over baselines in several benchmark datasets.

### Strengths
(1) The model shows better results than prior works.

(2) The examples given in the introduction help to explain and illustrate the reasoning behind the hierarchical focus.

(3) The ablations confirm that the added additional loss contributes to the performance.

### Weaknesses
(1) In L61, the difference in available labelling is presented as one of the motivations for hierarchical classification. However, the presented method assumes that all levels of the hierarchy are available. Can the technique work if the finest levels of supervision are not available?

(2) Similarly, given the availability of the finest-level label, the other course levels in the tree are implied, so perhaps a more appropriate flat baseline would be a ViT that only predicts finest-level classes (and thus parent nodes by simple aggregation). It would also provide a more appropriate comparison in terms of architecture.

(3) Given the relatively "small" sizes of the datasets (Tab 1.) by modern standards and some occasional closeness to the flat baselines in the scores (Tab 2.) Has there been any significant variability observed in the results? Would it be possible to include some measure of variance for some key results?

### Questions
Please see questions listed alongside weakness.

### Soundness
2

### Presentation
2

### Contribution
2
