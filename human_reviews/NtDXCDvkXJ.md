# The Renaissance of Classic Feature Aggregations for Visual Place Recognition in the Era of Foundation Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 8, 3, 6

## Abstract
Visual Place Recognition (VPR) addresses the retrieval problem in large-scale geographic image databases through feature representations. Recent approaches have leveraged visual foundation models and have proposed novel feature aggregations. However, these methods have failed to grasp the core concepts of foundational models, such as leveraging extensive training sets, and have also neglected the potential of classical feature aggregations, such as GeM and NetVLAD, for low-dimensional representations. Building on these insights, we revive classic aggregation methods and create more fundamental VPR models, abbreviated SuperPlace. First, we introduce a supervised label alignment method that combines grid partitioning and local feature matching. This allows models to be trained on diverse VPR datasets within a unified framework, similar to the design principles of foundation models. Second, we introduce G$^2$M, a compact feature aggregation with two GeMs, in which one GeM learns the principal components of feature maps along the channel direction and calibrates the other GeM's output. Third, we propose the secondary fine-tuning (FT$^2$) strategy for NetVLAD-Linear (NVL). NetVLAD first learns feature vectors in a high-dimensional space and then compresses them into a low-dimensional space using a single linear layer. G$^2$M excels in large-scale applications requiring rapid response and low latency, while NVL-FT$^2$ is optimized for scenarios demanding high precision across a broad range of conditions. Extensive experiments (12 test sets, 14 previous methods, and 11 tables) highlight our contributions and demonstrate the superiority of SuperPlace. Specifically, SuperPlace-G$^2$M achieves state-of-the-art results with only one-tenth of the feature dimensions compared to recent methods. Moreover, SuperPlace-NVL-FT$^2$ holds the top rank on the MSLS challenge leaderboard. We have submitted a ranking screenshot, the source code, and the original experimental records in the supplementary materials.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The work outlines three contributions to Visual Place Recognition (VPR). The first two are intended to improve the aggregation modules, such as NetVLAD (NVL-FT$^2$) and GeM (G$^2$M) of DINOv2 foundational features. At the same time, the third contribution addresses the discrepancy in existing VPR benchmarks and proposes a unified one that combines four datasets with a consistent labeling scheme.

### Strengths
The paper extensively and thoroughly presents the historical overview of the VPR field (introduction and related works sections).

### Weaknesses
1. The paper is hard to follow, and the main storyline of the work is missing. While some limitations are addressed (lack of consensus on the VPR model supervision, limitation of classical aggregation methods for FM features), the main challenge that unites the three contributions is unclear and is not explicitly stated. Experiments should be designed and presented according to the storyline, demonstrating the effectiveness of the proposed method in tackling a specific issue.

2. The experimental part contains a lot of information with a mixture of different settings (training data, pre-trained backbone models) and requires careful discussion and self-containing captions (Tables 3 - 8, 10, 11). Instead of a numbered summary in Section 4.2 it would be beneficial to discuss the differences between the performance of the proposed method and SOTA approaches, stressing on the advantages and disadvantages of the former on different configurations.

3. Novelty of the work is limited. Dimensionality compression of NetVLAD features has been proposed before [1, 2], while GCA components have been used as a simple baseline in feature fusion works [3].

4. The decision to quantize the training labels while evaluation still abides by the common practices of utilizing continuous coordinates appears to be a limitation. This concern should be addressed with further baselines (Table 6) and qualitative figures demonstrating the superiority of one convention over another.

[1] Gong et al. "Ghost-dil-NetVLAD: A Lightweight Neural Network for Visual Place Recognition", 2024

[2] Griange et al. "Design Space Exploration of Low-Bit Quantized Neural Networks for Visual Place Recognition", 2023

[3] Wu et al. "Asymmetric Feature Fusion for Image Retrieval", 2024

### Questions
1. I believe that the work in its current state requires a significant rewrite. There are two possible ways of approaching this task: (1) finding a common problem that can be effectively addressed with the new aggregation modules and multi-dataset training; (2) separating contributions into different works with more detailed discussions and extensive evaluations.
2. It is unclear what "SuperPlace" is and how it differs from DINOv2 or other foundational model (FM) backbone.
3. It is unclear if Supervised Label Alignment is performed during training or if it is a pre-processing algorithm.
4. The rationale behind combining four datasets, namely GSV-Cities, SF-XL, Pittsburgh-250k, and MSLS, is unclear. Does the main motivation lie solely in increasing the amount of training data or its diversity? 
5. What does "rank" in GCA refer to (Section 4.4, Table 8)?
6. How do PCA components of feature maps look after training GCA in alignment with the location-based information (Figure 2)?
7. How are G$^2$M and NVL-FT$^2$ trained (loss function)?
8. It would be interesting to replace GeM with GCA

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper investigates how modern foundation model DINOv2 can be utilized for the task of visual place recognition. The paper has two main contributions. First, by revisiting the already existing feature aggregating methods with some minor tweaks, the paper observes that the resulting methods achieve very competitive results while in many cases being simpler than previous methods. Second, the paper proposes a method to chain multiple place recognition datasets into one to benefit from large-scale training. At last, the paper thoroughly evaluates and ablate their main components of the proposed method on multiple datasets and in comparison to other methods.

### Strengths
Overall, the contributions of this paper are interesting and worth publication, in particular, revisiting time-tested methods, improving them in a clever way, and showing that with a few small changes, they can still achieve SOTA performance.

To summarize:
- The proposed method is simple and elegant, yet achieves strong performance on multiple benchmarks.

- The proposed method improves widely used GeM and NetVLAD pooling layers for use with DINOv2 foundation model.

- The paper proposes a scheme to chain the training datasets together and it seems to be outperforming the recently published method (SALAD) to which authors compare.

- The different components of the proposed method are thoroughly ablated.

- The resulting descriptors are relatively compact (768D for SP-G2M model and 8192D for SP-NVL-FT2 model) making the method applicable for large-scale applications.

### Weaknesses
The biggest issue is in the way the paper defines the supervised label alignment technique as well as some training details. While this contribution seems to be an important part of the paper (as shown in Table 6), it is not really clear how it works. The only bits of information can be found in Table 1, but it is not clear how to interpret it. Is the SimplePlace model trained on each dataset separately with different label schemes and loss functions? Or is it chaining all the datasets into one by finding a labeling scheme that can be derived from all of them? Adding more details and explaining this in the rebuttal would be beneficial.

### Questions
1. While I understand that this method is based on DINOv2, is there a way on how to test the generalizability of G2M and FT2 approaches on other foundation models?

2. Can you please clarify step-by-step how supervised label aligning works? What loss function is used in the training?

3. Would it be also possible to report results (on at least one dataset) with DINOV2 and the unmodified NetVLAD and GeM pooling methods?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper presents two kinds of foundation model based compact descriptors for visual place recognition. In this paper, based on two classical GEM and NetVlad descriptors, the authors proposed the corresponding variants. In addition, a label alignment strategy is proposed to be tailored for the foundation vision model pre-training. Extensive experiments are conducted on 12 test datasets to demonstrate the effectiveness of the proposed framework.

### Strengths
1. The paper is readable, where the organization and presentation of this paper are good.
2. Extensive experiments are conducted to verify the proposed framework.

### Weaknesses
1. The main concern on this paper is that the contribution of this paper is incremental. With the pre-trained DINO-v2 foundation model, the authors proposed two variants of GEM and Netvlad, respectively. By introducing an additional branch to project GEM into the low-dimensional space with MLP, the G^2M descriptor is proposed. By fine-tuning the backbone network and linear layer of Netvlad, the NVL-FT^2 descriptor is developed. In fact, the training strategies are usually employed in the design of the network structure. Moreover, this paper lacks deep analysis of the core modules in the proposed framework, i.e.,  the reasons why the simple operations on the classical visual place recognition descriptors can lead to significant improvement.
2. What is the relationship between the proposed G^2M descriptor and NVL-FT^2 descriptor? It seems that the proposed method is not self-contained. The same insights are conveyed to readers from the proposed descriptors? Therefore, I am confused that why the authors proposed two independent variants for the visual place recognition task.
3. The authors claim that the proposed G^2M descriptor is suitable for the real-time applications while the NVL-FT^2 descriptor is suitable for high-performance cases. However, in the experimental section, I did not see the comparison of the inference time of the G^2M descriptor to verify the real-time performance. In addition, the idea case is that a real-time and high-performance visual place recognition descriptor is desired. How to deal with the case with the proposed variants?
4. In this paper, there are too many abbreviations, which might be confused for readers. For example, in Line 417, what is SLA?

### Questions
Please see the weakness section.

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
4

### Summary
The paper presents SuperPlace, a VPR model that improves classic feature aggregation techniques like GeM and NetVLAD. Two main improvements are introduced: (1) G2M, a compact two-layer GeM module designed to optimize feature maps along the channel direction, and (2) a secondary fine-tuning method (FT2) for NetVLAD-Linear, which refines high-dimensional representations before compressing them. Additionally, a supervised label alignment method is proposed to facilitate multi-dataset training, inspired by Visual Foundation Models. Extensive benchmarking indicates that SuperPlace achieves top results on several VPR datasets while using fewer feature dimensions.

### Strengths
1. The paper demonstrates that classical aggregation techniques, when carefully refined, can match or outperform modern complex methods. This insight is valuable for the community to reconsider the recent trend of utilizing VLMs for VPR tasks.
2. The authors provide a thorough evaluation across multiple benchmark datasets, comparing against recent state-of-the-art VPR methods.

### Weaknesses
1. The contribution of the two proposed methods are largely incremental. The channel attention module is widely used to aggregate information in different channels, as is in the SE paper that the authors already cited. I don't see any difference of the proposed module to previous practices, and I don't think that the "motivations, usage, and design details are novel." Also, two-stage training is a very common practice in fine tuning pretrained models.
2. It is not clear on necessity of label alignment. If using triplet loss for supervision, the only information needed is the positive and negative sets for each query image, which can be easily determined by applyting thresholds on the location and direction.
3. Although the authors demonstrated the superior performance with the DINOv2 representation, there is no experiments showing what is the improvement for other backbones like VGG, ResNet, and ViT. This significantly weakens the generalizability of the proposed method on different backbones and representations.
4. While I appreciate the insights provided in Figure 2, it is not discussed in depth. If the location information is dominant in the first component, why not simply empoly it?
5. The illustration in Figure 1 is confusing. For example, different colors are used for the same component (patch tokens, feature map). Same color is used for different component (GeM, VFM, SoftMax, etc.). It is also not clear what's the difference between the two discriptors in NVL-FT^2.

### Questions
I appreciate the authors' insight on why NV-Linear doesn't give exptected performance, but this is not very convincing. Why does the training of NV-Linear happens in a "a lower-dimensional space"?

If the above question is answerd and concerns addressed, I'm happy to raise my rating.

### Soundness
3

### Presentation
2

### Contribution
3
