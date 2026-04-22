# TerraFM: A Scalable Foundation Model for Unified Multisensor Earth Observation

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 6, 8, 6

## Abstract
Modern Earth observation (EO) increasingly leverages deep learning to harness the scale and diversity of satellite imagery across sensors and regions. While recent foundation models have demonstrated promising generalization across EO tasks, many remain limited by the scale, geographical coverage, and spectral diversity of their training data, factors critical for learning globally transferable representations. In this work, we introduce TerraFM, a scalable self-supervised learning model that leverages globally distributed Sentinel-1 and Sentinel-2 imagery, combined with large spatial tiles and land-cover aware sampling to enrich spatial and semantic coverage. By treating sensing modalities as natural augmentations in our self-supervised approach, we unify radar and optical inputs via modality-specific patch embeddings and adaptive cross-attention fusion. Our training strategy integrates local-global contrastive learning and introduces a dual-centering mechanism that incorporates class-frequency-aware regularization to address long-tailed distributions in land cover. TerraFM achieves strong generalization on both classification and segmentation tasks, outperforming prior models on GEO-Bench and Copernicus-Bench. Our code and pretrained models will be publicly released.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces TerraFM, which is a foundation model for remote sensing / earth observation. TerraFM leverages the following novel strategies: modality-specific patch embedding, natural augmentations, cross-attention fusion, and class-frequency-aware centering (in contrastive learning). TerraFM can process Sentinel-2 images (multispectral optical), Sentinel-1 images (synthetic aperture radar), or both. TerraFM achieves SOTA results across standard remote-sensing benchmarks.

### Strengths
- TerraFM achieves SOTA performance across standard RS benchmarks and over standard RS baseline methods
- The paper is, in general, clearly written and presented
- The cross-attention token-fusion strategy seems novel (although is unclear, see weaknesses)
- The class-frequency-aware centering strategy is novel to the best of my knowledge

### Weaknesses
- The novelty of the proposed strategies is often overstated and sometimes incorrect, in my view. For example, "modality-specific patch embedding" is just a different patch embedding layer per modality, which is commonly used (e.g., Galileo or even SatMAE). Treating sensor modalities as natural augmentations for contrastive learning is also not novel, e.g., CROMA performs radar-optical contrastive learning. Thus, the cross-attention fusion block and centering strategies are the only methods that I consider novel to TerraFM.
- The cross-attention fusion block is unclear to me. My understanding is that the images are patchified according to their modality, which become keys/values in cross-attention with learnable queries. However the number of learnable queries is unclear, as this determines the sequence length for the transformer backbone, this is crucial. In the appendix, it implies that there is 1 query per spatial location. Does this mean that with two modalities (2x196 patches), there are 196 learnable queries attending to 392 patches? The bottom of section 3.1 implies that each query only attends to the patches at the same spatial location. Can you please provide code to make this clear for me?

### Questions
- Please clarify the cross-attention fusion
- In Table 5, centering is ablated. Does this mean that both centering terms are removed together? If so, how can we know the effect of the novel centering term introduced?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces TerraFM, a large-scale self-supervised foundation model for Earth Observation (EO) that unifies multi-sensor data—specifically Sentinel-1 SAR and Sentinel-2 optical imagery (L1C, L2A)—under a common framework. TerraFM is built on a student-teacher contrastive learning framework (similar to DINO) and introduces three main technical contributions to address challenges specific to EO data.The authors conduct extensive experiments on the GEO-Bench and Copernicus-Bench benchmarks. The results show that TerraFM consistently outperforms prior state-of-the-art models in both classification and segmentation tasks

### Strengths
1. Well-written and organized; clear motivation and architectural diagrams.
2. Strong empirical results: Their method outperforms all baselines on GEO-Bench and Copernicus-Bench for both classification and segmentation tasks.
3. Clear ablation studies isolate the impact of each component—modality augmentation, fusion, and dual centering—showing substantial performance gains

### Weaknesses
1. Although Major-TOM sampling is described, the paper gives limited discussion of potential geographic or seasonal biases in the curated dataset and how these might affect global transferability.
2. The model is designed for three specific inputs: S1, S2-L1C, and S2-L2A. While it generalizes well (Appendix C.3), the paper doesn't discuss the practical steps needed to extend TerraFM to a new modality (e.g., Landsat, hyperspectral).
3. The paper margin is smaller than the official format provided.
4. The cross-attention fusion module is activated "When multiple modalities M ⊆ M are selected" (L235-236). Given the stochastic assignment to student and teacher networks, this needs clarification. Does this mean fusion occurs within the student network if it happens to be assigned both S1 and S2 while the teacher does not use fusion?
5. The paper claims an advantage from using "large spatial tiles" (534px) to capture broader spatial context. However, the model's actual input resolutions are standard (224×224 for global crops, 96×96 for local crops). it is unclear to a reader that how much of the performance gain comes from this cropping strategy.

### Questions
1. How does the fusion module handle missing modalities at inference time? For example, if Sentinel-1 or Sentinel-2 data are unavailable, is there a fallback strategy?
2. Can the proposed modality-specific patch embedding extend easily to more than three modalities (e.g., DEM, hyperspectral data)?
3. Does TerraFM consider temporal alignment between Sentinel acquisitions, or is it purely spatial? If not, how might the framework extend to temporal FMs?
4. How sensitive is the model to the choice of the α parameter (balancing global and frequency-aware centers)? Was this tuned per dataset?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors introduce TerraFM, a S1/S2 foundation model for earth observation (EO) data. The paper proposes a modality-specific self-supervised pre-training recipe aimed specifically at EO domain-specific data characteristics. The authors incorporate a custom patch embedding mechanism catering to the multispectral nature of the data, treat different modalities as separate views for a multiview-learning setting, introduce an attention-based fusion mechanism to incorporate modalities within the model, and propose a dual-centering mechanism to handle the skewness of the underlying data distribution. Experiments indicate that the model robustly outperforms comparable foundation model efforts on two extensive benchmarking datasets.

### Strengths
- This paper has strong contributions to the literature, proposing a self-supervised pretraining strategy based on several well-motivated, domain-specific architecture components, including the double-centering strategy which is novel. 
- The experiments are comprehensive and indicate strong performance, especially when including the supplement, building confidence in the quality of the model.

### Weaknesses
- The paper exceeds length requirements and ignores formatting guidelines (e.g. margins), this has to be fixed. 
- The authors do not justify the choice not to incorporate any other augmentions in the pre-training framework. Many successful works  in the EO literature have previously done joint-embedding learning based on heavier augmentation strategies that are absent here, is there a reason?
- The main text (and appendix) lacks any details on model tuning or justification for hyperparameter choices.
- Table 2 and 3 use underlining to highlight second best models, this should be clear from the table description. Further, Table 4 does not use the same highlighting but should at least be consistent. 
- Table 5 should show improvement from baseline SS performance for subsequent rows instead of the absolute scores, this would make it much more readable.

### Questions
- An complementary alternative explored to the dual-centering done to encourage learning long-tailed features of the data is altering dataset curation. While data sampling in this work is done based on land cover priors, have the authors considered replicating efforts like dynamic dataset curation (arXiv:2504.06962) instead or in tandem with the dual-centering strategy explored here to see how that impacts model performance?
- Similarly, since resolution varies between S1 and S2 imagery and acquisition modes, have the authors considered further augmenting the ViT backbone with domain-specific mechanisms, such as the scale information incorporated by ScaleMAE (arXiv:2212.14532)?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper presents TerraFM, a self-supervised remote sensing foundation model trained with Sentinel-2 and Sentinel-1 data. Compared to other remote sensing FMs, TerraFM has a much larger spatial input size. The model is trained using contrastive learning with different modalities used to construct augmentation pairs. The modalities are encoded separately and fused with cross-attention. The paper introduces a novel dual centering strategy that adds an additional prior about LULC distributions beyond the DINO centering strategy. Experiments on  Geo-Bench and Copernicus-Bench show that TerraFM outperforms most other models across most tasks.

### Strengths
- The paper is well-written and well-organized. It is easy to read and follow the technical details.
- The paper presents extensive experiments on Geo-Bench, Copernicus-Bench, and other datasets in the appendix. The ablation experiments test the contribution of each component.
- The performance is good across all tasks, with similar magnitudes across GeoBench tasks. This is particularly evident in Figure 1.

### Weaknesses
- The paper claims several novel contributions to the FM design: modality-specific patch embedding, modalities as augmentations, cross-attention fusion block, and dual-centering strategy. I think the dual-centering strategy is new (although very similar to the DINO centering), but it seems that all of the others have been proposed in prior work. The paper does not discuss how their contributions compare to these prior works.
    - modality-specific patch embedding: This is a common strategy in remote sensing FMs today. Maybe the authors realize this since it was not part of the ablation. I don’t think it can be claimed as a contribution. For example, SatMAE, Galileo, RingMo, and more all use separate embedding layers/tokenizers for each modality (or groups of channels within modalities). This was also done in vision with [MultiMAE](https://arxiv.org/pdf/2204.01678).
    - modalities as augmentations: The idea of using colocated pairs of images from different modalities as augmentation pairs for contrastive learning is also well established in prior work (e.g., [Jain et al., 2022](https://arxiv.org/abs/2209.02329); [Prexl & Schmitt 2023](https://openaccess.thecvf.com/content/CVPR2023W/EarthVision/papers/Prexl_Multi-Modal_Multi-Objective_Contrastive_Learning_for_Sentinel-12_Imagery_CVPRW_2023_paper.pdf); and CROMA).
    - cross-attention fusion block: I believe the same thing was done in [CrossMAE](https://openaccess.thecvf.com/content/CVPR2024/papers/Guo_CrossMAE_Cross-Modality_Masked_Autoencoders_for_Region-Aware_Audio-Visual_Pre-Training_CVPR_2024_paper.pdf) and other remote sensing specific work like [Chan-To-Hing & Veeravalli 2024](https://arxiv.org/abs/2401.02764).
- There is no discussion of performance variability and stability across runs/seeds or samples. Were the experiments repeated for different random seeds and/or subsets of the test data to quantify the standard error of performance metrics?
- I have some questions about comparisons between TerraFM and previous models (taken from Galileo paper) on downstream tasks:
    - Tseng et al. 2025 (Section C) describes the hyperparameter sweep done for evaluating downstream tasks. From the TerraFM paper’s Section C with evaluation implementation details, it seems there are similarities but not the same protocols compared to Galileo. I would expect TerraFM to use the same protocol as Galileo if they are trying to compare directly to numbers from Galileo’s tables.
    - The paper says the authors used linear probing, UperNet probing, k-NN, and fine-tuning for downstream task evaluation. It seems that all segmentation tasks except the ablation analysis used linear probing, but the ablation used UperNet probing for cashew plant. Why was the protocol changed here? Why was the cashew plant task chosen here? From my experience the labels in this dataset are noisier and lower resolution (and very sensitive to scale) compared to other geobench tasks.
- Some formatting and writing issues
    - The margins are smaller than allowed. This gives the paper a lot more space than other authors have to work with on their submitted papers. I expected this to constitute a desk reject.
    - The in-text citations need to be changed to use \citet (bracketed)
    - The style of Table 4 is notably different than the prevoius tables
- Minor issues
    - Table 1 has several confusing or unexplained details: I was confused that “Scale” represents the number of training samples not the number of model parameters. I would also have expected to find the # parameters in this table. Why is Benchmarks a relevant column here? How is Pixels (~T) computed?
    - There are some missing or unclear details about the pretraining data. The paper says “global distributional priors” were used - what priors/where did these come from? The paper says they “enriced each grid cell with metadata” - what metadata exactly and how was this formatted/interpreted by the model?
    - The “Impact of Components” paragraph discusses gain % for individual components, but those components are not isolated. For example, the 14% gain for cashew plant is from all of the components added, not just dual centering. Can the authors add rows to this table that isolate the contribution of each component?
    - The caption of Figure 6 talks about GeoBench as a whole, but the results are only shown for m-EuroSat.

### Questions
- What are the differences between the paper’s contributions and the works I referenced in the Weaknesses question?
- Were the experiments repeated for different random seeds and/or subsets of the test data to quantify the standard error of performance metrics?
- What are the differences between the evaluation protocols for the results taken from the Galileo paper and those applied for TerraFM?
- Why was the segmentation protocol changed from linear probe to upernet probe for the ablation experiment?
- WorldCover is known to have highly variable accuracy across the world, with especially poor performance in Africa (see [Kerner et al., 2024](https://www.nature.com/articles/s41597-024-03306-z) for one analysis). Since this is used as a prior in the dual centering method, do you expect performance to be worse in regions where WorldCover has worse performance?
- Can the authors add rows to the ablations table that isolate the contribution of each component?

### Soundness
2

### Presentation
3

### Contribution
2
