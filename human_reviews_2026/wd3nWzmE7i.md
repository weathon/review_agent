# Multimodal Few-Shot Point Cloud Segmentation via Agent Adaptation and Discriminative Deconfusion

- Avg Score: 4.50
- Decision: Reject
- Scores: 8, 4, 4, 2

## Abstract
Few-shot 3D point cloud segmentation (FS-PCS) aims to leverage a limited amount of annotated data to enable the segmentation of novel categories. Most existing studies rely on single-modal point cloud data and have not fully explored the potential of multimodal information. In this paper, we propose a novel FS-PCS framework, Multimodal Agent Adaptation and Discriminative Deconfusion (MAD).  MAD incorporates three modalities: images, point clouds, and category text embeddings. To fuse multimodal information, we propose the Multimodal Semantic Agents Correlation Aggregation (M-SACA) module, which fuses multimodal features through agent-level correlation and uses text affinity for category semantic learning. To alleviate semantic gaps between the support set and query set in multimodal features, we propose the Semantic Agents Prototypes Adaptation (SAPA) module, which generates multimodal agents for query and support sets, adjusting prototypes to adapt the query feature space. To alleviate intra-class confusion, we introduce the Discriminative Deconfusion (DD) module, which preserves intra-class consistency through residual adapters and generator weights. 
Experiments on the S3DIS and ScanNet datasets demonstrate that MAD attains state-of-the-art performance, improving mIoU by 
3%–7%. Our method can significantly improve segmentation results and suggest valuable insights for future studies. The code will be publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces Multimodal Agent Adaptation and Discriminative Deconfusion, a framework for multimodal few-shot 3D point cloud semantic segmentation. MAD simultaneously models multimodal information in both query and support sets. The framework consists of three main components: Multimodal Semantic Agents Correlation Aggregation, Semantic Agents Prototypes Adaptation, and Discriminative Deconfusion. Experiments on the S3DIS and ScanNet benchmarks demonstrate that MAD consistently improves performance compared to strong baselines.

### Strengths
1. The writing is generally clear, and the figures help in understanding the technical flow.
2. The framework design is well-motivated, addressing challenges about cross-modal fusion, semantic gap adaptation, and intra-class confusion.
3. Extensive comparisons on two benchmarks and detailed ablation studies demonstrate the effectiveness of the proposed approach.

### Weaknesses
1. In the Discriminative Deconfusion section (lines 313–315), the definition of the generator is unclear. While generator-based adaptive weighting improves results, the paper does not provide intuition for why it performs better than averaging or summing.
2. The ablation study should also include an analysis of Eq. (13), specifically by comparing the predictions of p^{TxT} and p^{DD}.
3. Jointly fusing query and support multimodal correlation features is computationally expensive. As reported in the paper, MAD’s computational cost is noticeably higher than that of MM-FSS. A deeper analysis of scalability to larger or more complex scenes would strengthen the work.
4. Minor grammar issues remain. For example:
   - '... fuses multimodal features for query and support set fusion through multimodal semantic agents correlation aggregation …' (lines 83–85).
   - '... eliminate discriminative deConfusion from the residual connection …' (lines 309–310).

### Questions
Please refer to the weakness part and address the concerns there.

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
3

### Summary
This paper addresses the issues in the few-shot 3D point cloud segmentation (FS-PCS) field, where existing methods rely on single-modal data, suffer from insufficient multi-modal fusion, semantic gaps, and intra-class confusion, by proposing a Multi-modal Agent Adaptation and Discriminative Deconfounding framework (MAD). This framework integrates three modalities—image, point cloud, and text embeddings—and achieves synchronized multi-modal association feature fusion between query and support sets through the M-SACA module. The SAPA module adaptively adjusts prototypes to mitigate semantic gaps, while the DD module eliminates intra-class confusion via residual adapters and dynamic weighting. Experiments on the S3DIS and ScanNet datasets show that compared with the current state-of-the-art method MM-FSS, MAD improves mIoU by 3%–7%, and the code will be made publicly available. The manuscript is well-structured, the technical approach is clear, focuses on the core challenges of the FS-PCS field, and has definite research value.

### Strengths
1.Novel multimodal alignment perspective: Existing multimodal FS-PCS methods only handle query-side correlations. This work simultaneously constructs multimodal correlations and aligns prototypes for both support and query, making it conceptually more unified and technically a natural progression from correlation modeling → prototype adaptation → discriminative deconfounding in a three-step pipeline. 
2.Significant and consistent improvements: Outperforms the strong baseline MM-FSS across multiple settings on S3DIS and ScanNet, including 1-way/2-way and 1-shot/5-shot scenarios, with clear average gains and visualization results. 
3.Clear modularity and thorough ablation: Independent ablations are conducted on the three main modules, loss terms, modality contributions, and generator weights, allowing the sources of performance to be pinpointed.

### Weaknesses
The manuscript mentions 'the first exploration of simultaneously fusing multimodal association features of query and support sets,' but it does not provide a detailed comparison with the multimodal fusion approaches used in vision-language models (such as 3D VLMs and GFS-VL), nor does it clearly define the fundamental differences between 'agent association aggregation' and existing cross-modal attention mechanisms, which may result in a lack of uniqueness in the proposed innovation

### Questions
1. Clarity of assumptions and dependencies: What assumptions does the method make about the availability of images and class texts during the testing phase? The experiments seem to assume that all three modalities are provided at inference; robustness and degradation strategies when any modality is missing are not reported. Is the text encoder (LSeg's text/visual encoder) frozen during training/testing? The impact of different freezing strategies on performance is not specified.2. Statistical significance and variance: Few-shot learning is usually sensitive to episode sampling. Tables 1 and 2 do not provide mean ± variance or confidence intervals, nor report statistical tests.3. Generalization and extrapolation: Validation is conducted only on indoor datasets S3DIS/ScanNet. There is no evaluation on outdoor LiDAR data or cross-dataset transfer (e.g., A→B few-shot), making it difficult to assess robustness to changes in scene or sensor.4. Fairness with strong multimodal baselines: The paper mentions that all three use the "same 2D-aligned pre-trained backbone," but specific details on alignment/pre-training (data, image resolution, handling alignment errors) are not fully described; differences in 2D-3D alignment quality could directly affect the results.5. Method details still need to be filled in: Implementation details of Fagent (clustering radius/adjacency definitions, clustering and reverse assignment after FPS, sensitivity to NA/NP) are not fully explained; hyperparameter sensitivity and ablation studies (e.g., β, τ, channel dimension D′) are missing; time/memory costs and episode-level throughput are not reported, only FLOPs/parameters.6. Writing and terminology consistency: The paper occasionally mixes SA-CAPA with SAPA/M-SACA; a unified naming is recommended. Some statements and figure captions could be further polished to facilitate reproducibility.

### Soundness
2

### Presentation
2

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
This paper addresses the challenge of few-shot 3D point cloud segmentation by proposing a multimodal framework that leverages both RGB-text and 3D-text pairs during training. The key idea is to learn modality-agnostic cross-modal alignment, such that at test time, the model can perform segmentation using only 3D-text input, even though 2D data was available during training. The core contributions include: a modality-complementary contrastive learning loss that aligns RGB-text and 3D-text embeddings jointly, a modality-invariant meta-learning pipeline for better generalization across modalities, experiments on ScanNet, S3DIS, and ScanRefer demonstrating the effectiveness of their method under both standard and generalized few-shot settings.

### Strengths
- Novel Setting: The proposed training-testing mismatch (RGB-text used only during training, 3D-text at test time) is practical and not widely explored in current literature. This setting is well-motivated and could inspire further work.
- Multimodal Fusion Strategy: The method uses a well-designed dual-branch architecture with contrastive losses that promote shared representation learning across modalities. The idea of learning from richer modalities but testing on limited modalities is both elegant and impactful.  
- Good Clarity: Figures and architecture diagrams are clear; the training/testing pipelines are easy to follow. The motivation is well-articulated.

### Weaknesses
- Incremental Contribution: While the training-testing mismatch setting is interesting, the core method is essentially a combination of known techniques: contrastive learning for modality alignment, standard meta-learning with prototypes, feature fusion with transformers
These parts are not fundamentally novel on their own. There’s no architectural breakthrough or new theoretical formulation.
- Over-Reliance on Pretrained CLIP: The RGB and text encoders are frozen CLIP backbones. The method’s performance may heavily depend on this pretrained knowledge, making it unclear how much the proposed training scheme contributes beyond CLIP’s strong prior.
- Besides, the authors use CLIP as text encoder, but do not cite the paper. "Category names are encoded as text embeddings using the text encoder in LSeg Li et al. (2022)", while in LSeg they mentioned "and we use the pretrained Contrastive Language–Image Pre-training (CLIP) throughout (Radford et al., 2021)"
- Missing Real-World Applications: The paper motivates the method as applicable to real-world tasks where only 3D data is available at test time, but no concrete application or real robot / AR / scene understanding demonstration is provided.
- Limited Analysis of Modal Gap: The paper assumes that RGB-text and 3D-text can be aligned effectively, but doesn’t deeply analyze or quantify the actual modality gap. A t-SNE or retrieval-based analysis comparing embeddings could help.

### Questions
See weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
Authors present a method for few-shot point-cloud segmentation. As far
as I can see, the contributions are the integration of textual
guidance and joint embedding of query and support sets. Experiments on
two datasets show very impressive results.

### Strengths
+ Results are quite good.
+ Integration of textual guidance is an interesting and popular
  direction.

### Weaknesses
- The method description is not very detailed and clear.
  + The problem definition does not present a segmentation problem but
    rather a classification problem.
  + Authors use N to represent the number of samples in the query set
    as well as the number of categories.
  + The definition of the mask $M$ and where it comes from is
    unclear. It seems like the mask is the segmentation and both the
    query and the support sets have it.
  + Notation starts being used without introduction, e.g. $F_q^i$
    on line 179.
  + It is unclear how authors obtain a "background" text embedding?
  + Why are IF and UF defined and why are they different?
  + What does "agent clustering" mean and what are "agent points"?
    
- This paper is very difficult to read. The proposed method is
  explained through many acronyms and new module names. This makes it
  very difficult to follow. Furthermore, justification and intuition
  behind different choices do not seem to be well
  substantiated. On top, the paper uses a very heavy notation but it
  is not always well explained.

### Questions
- Is it necessary to have a different name for each model component?
  In my opinion, this is not necessary and even further, makes it
  extremely difficult to follow the text.
- I do not understand the difference between the 1st and 2nd claimed
  contribution on lines 83 to 89. Can you please explain why these are
  two different contributions?
- I do not understand why extensive experiments showing the proposed
  model achieves good results is a main contribution. Isn't this
  simply the necessary condition for writing a scientific article?
- I could not understand your paper despite spending considerable
  amount of time. I can suggest you to spend time to explain the
  method clearer.

### Soundness
2

### Presentation
1

### Contribution
2
