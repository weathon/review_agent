# UrbanFusion: Stochastic Multimodal Fusion for Contrastive Learning of Robust Spatial Representations

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 2, 8, 4

## Abstract
Forecasting urban phenomena such as housing prices and public health indicators requires the effective integration of various geospatial data. Current methods primarily utilize task-specific models, while recent generic models for spatial representations often support only limited modalities and lack multimodal fusion capabilities. To overcome these challenges, we present UrbanFusion, a spatial representation model that features Stochastic Multimodal Fusion (SMF). The framework employs modality-specific encoders to process different types of inputs, including street view imagery, remote sensing data, cartographic maps, and points of interest (POIs) data. These multimodal inputs are integrated via a Transformer-based fusion module that learns unified representations. An extensive evaluation across 41 tasks in 56 cities worldwide demonstrates UrbanFusion’s strong generalization and predictive performance compared to state-of-the-art GeoAI models. Specifically, it 1) outperforms prior models on location-encoding, 2) allows multimodal input during inference, and 3) generalizes well to regions unseen during training. UrbanFusion can flexibly utilize any subset of available modalities for a given location during both pretraining and inference, enabling broad applicability across diverse data availability scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents UrbanFusion which is a multimodal geospatial representation learning framework designed to integrate multiple data sources, e.g., remote sensing imagery, street-view imagery, POI and OSM, and location information (lat – lon). The paper uses a self-supervised learning strategy, termed SMF, which combines contrastive learning for aligning representations from different modality subsets with a modality reconstruction objective for fusion.

### Strengths
A pertinent challenge for spatial data and spatial foundation model, especially the fusion of data of different modalities.
The tasks are comprehensive, including many downstream tasks and 56 cities.
The paper is generally well-written and clearly structured.

### Weaknesses
Since the features from different modalities are extracted from frozen backbone encoders, the the primary contribution seems to be on fusion, which refers to specific combination strategy SMF involving contrastive learning and reconstruction with modality masking. This combination is applied effectively to spatial data, but the underlying principle of combining these two types of self-supervised objectives is not entirely new in the broader representation learning literature. It is noteworthy that empirical evidence is ample, while I doubt this is the focus of ICLR.
To the ICLR community, the novel insight given specifically by the paper (if we talk about those beyond showing that combining these existing techniques works well for this multimodal geospatial setting) is limited. This means that it seems to be more of empirical finding paper. 
There are some minor points as well. Using image encoder for OSM naturally leads to information loss and there are papers using object-level information for feature extraction. An example is 
Bai, Lubian, et al. "GeoLink: Empowering Remote Sensing Foundation Model with OpenStreetMap Data." arXiv preprint arXiv:2509.26016 (2025).
The rationality of the baseline comparisons is questionable. The pretrained backbone of the compared models differs (e.g., some are based on CLIP, others on GeoCLIP), which undermines the fairness of the comparison. Would it be more appropriate to compare with multimodal remote sensing foundation models that integrate OSM data? It would be also useful to compare a baseline with all of your frozen features concatenated, since the main contribution is on fusion.

### Questions
See weakness
Further details of section 4.5 would be useful (not easy to follow)

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes UrbanFusion, a multimodal framework that fuses several inputs (street view images, satellite images, POI data and map data) using a transformer based module. One of the main contributions of the paper is the proposed stochastic multimodal fusion training strategy, which applies a combined contrastive and reconstruction loss on random subsets of modalities to learn a unified representation. The approach is evaluated on 41 downstream urban prediction tasks.

### Strengths
[S1] The work addresses a timely problem of learning unified location embeddings for heterogeneous spatial modalities.

[S2] The proposed stochastic multimodal fusion module provides a flexible approach for aligning and reconstructing random subsets of modalities.

[S3] The paper is well written and the methodology is easy to follow.

[S4] The empirical results show improvements over the selected baselines.

### Weaknesses
[W1] One of my main **concerns** and confusion is the decision to **append the raw coordinates** to the learned embeddings during downstream evaluation: *"Following prior work [1], raw geographical coordinates are concatenated to the model embeddings for evaluation."* Location is highly correlated with many of the downstream labels (e.g. house prices), so it is unclear if the performance gains come from the approach or from leveraging location information. Moreover, the cited reference [1] is misinterpreted: In SATCLIP, location embedding (not raw coordinates) are concatenated with image features to perform image classification.

[W2] The fusion of different urban modalities has been studied extensively, and more sophisticated fusion approaches exist in the literature (e.g., [2, 3]). For the specific modalities handled here, the authors seem to bypass the core challenge of different spatial resolutions by operating at the patch level [4]. Moreover, the handling of the cartographic basemap is fairly naïve, as it relies on rasterization, rather than leveraging spatial and topological structure, which has been addressed more effectively in prior work [5, 6]. Given that, the SMF component seem to be the main technical contribution here. However, its evaluation is weak: it is validated on a toy example, and the real-world ablation shows that the combination of the reconstruction loss and contrastive loss is not the best overall in the majority of the cases (Appendix C.3.).

[W3] The foundation model claim is an overstatement given the small-scale pretraining dataset and the lack of diversity across tasks. Specifically, the evaluated tasks are variations of point-based prediction problems (regression or classification) confined to the urban domain, which does not demonstrate the broad, cross-domain applicability or diverse reasoning capabilities expected of a true foundation model.

[W4] There are instances where the authors categorize SatCLIP and GeoCLIP as unimodal, although both are inherently multimodal: *"Both models clearly outperform unimodal baselines such as SatCLIP and GeoCLIP, demonstrating the advantages of
multimodal representation learning for geographic generalization"*.

[W5] Similar to W1, several "new" features are introduced when testing on housing value prediction (i.e., # of bedrooms, # of bathrooms etc.), which are already strong predictors of the housing prices on their own. Again, it's unclear whether the improvement comes from the learned embeddings.

[1] Satclip: Global, general-purpose location embeddings with satellite imagery. AAAI 2025.

[2] Urbanclip: Learning text-enhanced urban region profiling with contrastive language-image pretraining from the web. WWW 2024.

[3] Refound: Crafting a foundation model for urban region understanding upon language and visual foundations. KDD 2024.

[4] Gair: Improving multimodal geo-foundation model with geo-aligned implicit representations. arxiv 2025.

[5] Poly2Vec: Polymorphic Fourier-Based Encoding of Geospatial Objects for GeoAI Applications. ICML 2025.

[6] Geo2Vec: Shape-and Distance-Aware Neural Representation of Geospatial Entities. arxiv 2025.

### Questions
Could the authors address the issues mentioned in the weaknesses section? Specifically, are raw geographical coordinates appended to the embeddings for all baselines, or only for your model? Can the authors provide results on some of the tasks without appending the raw coordinates?

### Soundness
2

### Presentation
3

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
### Problem:
- The authors aim to build better geospatial foundation models that can understand and predict geospatial dynamics and urban phenomena at scale
- A good foundation model should:
    - Be task-agnostic
    - Be multi-modal
    - Synthesize information across modalities
    - Be robust to missing modalities
    - Have good generalization to unseen geographic regions


### Solution

- Authors propose a novel foundation model called UrbanFusion
    - UrbanFusion processes several different modalities (street-view, satellite imagery, maps, POIs, and coordinates)
    - Each modality is encoded separately, then fused with a Transformer
    - UrbanFusion is trained via a novel objective called Stochastic Multimodal Fusion (SMF), which combines contrastive learning with self-supervised re-construction
        -  Modalities are randomly masked
        - The model is trained to align representations contrastively across different modality subsets
        - Reconstruct missing modality embeddings from the fused representation

### Strengths
- UrbanFusion is flexible wrt what modalities are available inference time. Even though it is outperformed by some baselines, this flexibility is a significant benefit to the community
- Strong (though not dominant) empirical results
- UrbanFusion can effectively generalize to new cities not seen in training
- Effectively integrates cross-modal information, while preserving intra-model information

### Weaknesses
- Empirical performance is not dominant. On tasks/datasets, UrbanFusion is outperformed by baselines
- Some overlap with prior work. Likely worth citing some of these:
    - Jenkins et al. CIKM’19. Unsupervised Representation Learning of Spatial Data via Multimodal Embedding
    - Yan et al. 2024. UrbanCLIP: Learning Text-enhanced Urban Region Profiling with Contrastive Language-Image Pretraining from the Web
- No estimates of variability (e.g., mean + std across several training runs) in experiments.

### Questions
- How do they define the locations? How much engineering is required to get UrbanFusion to work in a new city? What if the spatial resolution of the data in the new city does not match?
- What is the impact of the image encoder? DinoV3 (Siméoni et al. 2025) was pretrained extensively on satellite imagery; perhaps this would yield a meaningful performance increase? But this likely would not be a fair comparison to GeoCLIP. Could be worth studying in the future
- Why does the multimodal fusion encoder use average pooling instead of the CLS token?

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
3

### Summary
This paper presents UrbanFusion, a geofoundation model that learns unified spatial embeddings through Stochastic Multimodal Fusion (SMF). The model encodes heterogeneous urban data — including street-view images, remote sensing, maps, and POIs — using modality-specific encoders, and fuses them via a Transformer-based architecture. It is trained with a contrastive alignment objective between complementary modality subsets, together with latent reconstruction to address missing-modality scenarios. Evaluated across 41 tasks in 56 cities, UrbanFusion achieves state-of-the-art performance and strong cross-region generalization, demonstrating a practical and scalable paradigm for multimodal GeoAI.

### Strengths
•  Innovative training paradigm: The proposed Stochastic Multimodal Fusion (SMF) framework provides a practically meaningful contribution by integrating stochastic modality masking, symmetric InfoNCE alignment, and latent modality reconstruction into a unified training process.
•  Comprehensive evaluation: The paper conducts extensive and well-structured experiments across multiple tasks and scenarios, including coordinate-only, multimodal, and cross-region zero-shot generalization settings.
•  Reproducibility and practicality: The framework is conceptually simple, reproducible, and demonstrates clear potential for deployment in real-world multimodal GeoAI applications.

### Weaknesses
•  Limited architectural novelty: While the training objective is interesting, the overall fusion design (“multimodal tokens + a single-layer Transformer + average pooling”) closely resembles standard multimodal architectures. Consequently, the primary novelty lies in the learning strategy rather than in architectural design.
•  Insufficient ablation analysis: The paper lacks systematic ablations on key factors such as masking strategy and ratio, the impact of fusion depth (single- vs. multi-layer), and pooling mechanisms. Including these analyses would strengthen the methodological insight and clarify design choices.

### Questions
•  Could you report cross-domain results without concatenating raw coordinates, and analyze the model's sensitivity to coordinate normalization or projection?
•  Could you include ablations on the masking ratio/strategy, pooling mechanisms, and number of fusion layers (single vs. multi-layer)?
•  The paper positions multimodal fusion as a core contribution, yet most baselines are single-modal. This makes it difficult to disentangle the effect of the fusion mechanism from the benefit of using multiple modalities. While the "concatenated multimodal embeddings + linear probing" baseline is informative, it is not equivalent to native multimodal fusion. Please consider adding one or two native multimodal fusion baselines for a fairer comparison.

### Soundness
2

### Presentation
3

### Contribution
2
