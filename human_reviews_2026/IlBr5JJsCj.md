# MoRA: Mobility as the Backbone for Geospatial Representation Learning at Scale

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 8

## Abstract
Representation learning of geospatial locations remains a core challenge in achieving general geospatial intelligence, with increasingly diverging philosophies and techniques. While Earth observation paradigms excel at depicting locations in their physical states, we propose that a location’s full characterization requires grounding in both its physical attributes and its internal human activity pattern, the latter being particularly crucial for understanding its human-centric functions. We present MoRA, a human-centric geospatial framework that leverages a mobility graph as its core backbone to fuse various data modalities, aiming to learn embeddings that represent the socio-economic context and functional role of a location. MoRA achieves this through the integration of spatial tokenization, GNNs, and asymmetric contrastive learning to align 100M+ POIs, massive remote sensing imagery, and structured demographic statistics with a billion-edge mobility graph, ensuring the three auxiliary modalities are interpreted through the lens of fundamental human dynamics. To rigorously evaluate the effectiveness of MoRA, we construct a benchmark dataset composed of 9 downstream prediction tasks across social and economic domains. Experiments show that MoRA, with four input modalities and a compact 128-dimensional representation space, achieves superior predictive performances than state-of-the-art models by an average of 12.9\%. Echoing LLM scaling laws, we further demonstrate the scaling behavior in geospatial representation learning. We open-source code and pretrained models at: https://github.com/ylzhouchris/MoRA.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
MoRA presents a human-centric geospatial representation paradigm that uses billion-scale mobility graphs as the backbone: it encodes H3 grid cells into 128-D vectors with GNNs, aligns satellite imagery, POIs and demographics via contrastive learning, boosts downstream socio-economic tasks by 12.9% on average, demonstrates the first scaling laws in GeoAI, and releases a privacy-preserving distilled model that maps coordinates directly to high-quality embeddings.

### Strengths
MoRA leverages a billion-edge mobility graph as its backbone, aligns imagery, POIs and demographics into a unified 128-D space via contrastive learning, delivers a 12.9% average gain across nine nationwide socio-economic tasks, demonstrates the first scaling laws in GeoAI, and releases a privacy-preserving distilled model that infers embeddings from coordinates alone.

### Weaknesses
1. The design of downstream evaluation does not align with main stream methodology.
Why utilize LightGBM as the downstream prediction model instead of a linear probing method that is employed by many existing works [1,2,3,4]

2. The coverage of dataset is limited. MoRA only utilized dataset from China. How to ensure the generalization of the induced conclusion?
It is suggested to compare more cities in othe counties.


3. The technique contribution is limited. Actually, combining mobility with POI and Satellite images is a common practice in region representation learning.
What MoRA has done is move this paradigm into location embedding, while the requirement of multiple different kinds of data modality also diminish the generalization and practicability. That is why previous location embedding work [1,2,3] applied global application while this work only limited on China data.

4. I am confused of the inference practice. The paper states that during inference, it still needs 3 kinds of data (POI, demographic, satellite images) for prediction. If so, what is the meaning of contrastive learning in pretraining stage? The CLIP-based contrastive learning should inject the information of POI, image, demographics data modality into mobility backbone through contrastive alignment in pretraining. In inference, the common practice is to only utilize the pretrained mobility backbone.

---
[1] Klemmer, Konstantin, et al. "Satclip: Global, general-purpose location embeddings with satellite imagery." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 39. No. 4. 2025.

[2] Hao, Xixuan, et al. "Nature makes no leaps: Building continuous location embeddings with satellite imagery from the web." Proceedings of the ACM on Web Conference 2025. 2025.

[3] Vivanco Cepeda, Vicente, Gaurav Kumar Nayak, and Mubarak Shah. "Geoclip: Clip-inspired alignment between locations and images for effective worldwide geo-localization." Advances in Neural Information Processing Systems 36 (2023): 8690-8701.

[4] Hao, Xixuan, et al. "Urbanvlp: Multi-granularity vision-language pretraining for urban socioeconomic indicator prediction." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 39. No. 27. 2025.

### Questions
See weakness.

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
SUMMARY: The authors propose a new geospatial location representation learning framework, leveraging multimodal input data, namely imagery, POI data and demographic information (the latter two available as text). The most unique aspect about this work is that instead of leveraging raw geographic context (e.g. pure coordinates, as used in SatCLIP or GeoCLIP), the authors build a spatial graph between locations based on whether locations are associated with each other in the WeChat Payment system. While this obviously assumes a lot of prior knowledge, it is an interesting approach to explore specifically for geo representation learning in urban areas, where we can assume such data to be available. This gives their whole approach a strongly informed "geographic prior". The authors test their geo embeddings on a set of different socio-economic tasks, showing impressive performance and outperforming recent embeddings such as GeoCLIP or AlphaEarth.

### Strengths
STRENGTHS:

- I like the framing of two paradigms for geo representation learning, the human and EO centric ones. Not sure I necessarily agree but this is an interesting framing to motivate the paper! The argument about the relative nature of a "place" (i.e. a location being majorly defined by its relationship with other locations) is thought provoking and interesting. Overall, really great motivation section!

- Exploring scaling laws for geospatial representation learning is a critical research challenge, e.g. also mentioned here [1]. The authors do that a little bit (though I would have liked to see more of that).

- I like that the authors clearly define distinctions from PDFM, which is conceptually closest.

### Weaknesses
- I would love to have seen a comparison to a simple neighborhood graph; the way understand it, edges between cells/nodes are based on a-priori known WeChat Pay interactions. This is great if you have access, but what if you didn't have that data? How would this method perform if you simply built your graph based on direct neighborhood/adjacency of cells? How would the model perform then? This seems like a crucial ablation missing, unless I am missing it. This also would allow the authors to compare their approach to PDFM, which is only available for the US - and which would be an important comparison.

- In the abstract, the authors say that "While Earth observation paradigms excel at depicting locations in their physical states, we claim that a location’s comprehensive “meaning” is better grounded in its internal human activity pattern". I don't want to start a philosophical discussion about "place", but this sentence feels a bit strong / definitive. Human activity might give a location "meaning" relevant to some applications - especially human centric ones, but might be irrelevant for other, e.g. natural processes.

- Given that your paper basically argues for a geographic prior for SSL geo-representations (in your case based on payment data), I think it would be important to link your work back to the origins of work on "geographic priors", especially [2].

- I don't like averaging over R2 values of different tasks in Tab 2; this does not seem very rigorous and I'd recommend to remove that column.

- Fig 2: sizes of legend circles don't match the circle sizes in the plot

### Questions
Overall this is an interesting, well motivated paper. I like both the methodological innovations and the experiments, though there could be more details and an important ablation is missing. To help with my understanding and the final assessment, I would ask the authors to address my concerns and questions outlined in the "Weaknesses" sections.

References:

[1] Rolf, Esther, et al. "Mission Critical--Satellite Data is a Distinct Modality in Machine Learning." arXiv preprint arXiv:2402.01444 (2024).
[2] Mac Aodha, Oisin, Elijah Cole, and Pietro Perona. "Presence-only geographical priors for fine-grained image classification." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.

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
3

### Summary
### Problem

The authors tackle the problem geospatial intelligence. Specifically, the seek to convert locations into useful vector embeddings to support tasks such as predicting socio-economic indicators or urban dynamics. They argue existing approaches fall into two silos:
	1.	Earth-observation-centric: learning from satellite imagery and physical features.
	2.	Human-activity-centric: learning from mobility and demographic patterns.

These paradigms rarely integrate well, and current models miss long-range functional relationships between places (e.g., commuter flows) and lack scalable benchmarks. The field also lacks foundation-style models trained at national or global scale. The authors argue that (2) Human-centric representations are fundamental the semantics of place and are a first class citizen in their solution


### Solution

The authors propose MoRA, a framework that leverages mobility as the backbone for geospatial representation learning at scale. Their method has the following properties:
	•	Aggregates real mobility flows across millions of locations and billions of edges.
	•	Builds a nationwide mobility graph using H3 spatial grids.
	•	Learns human-centric location embeddings with a graph neural network 
	•	Aligns mobility with three auxiliary modalities using asymmetric CLIP-style contrastive learning.
	•	Demonstrates scaling laws in geospatial representation learning.
	•	Releases a benchmark of 9 socio-economic tasks and a distilled privacy-preserving model.

MoRA consistently outperforms state-of-the-art representations across prediction tasks.

### Strengths
- Strong empirical results. Mora outperforms all baselines across several experiments
- Ablation studies validates the additional complexity of their method
- Experiments are done across several training runs, informing reproducibility. Means and standard deviations are provided. 
- The introduction of the new geospatial benchmark is noteworthy
- The experiments around scaling laws are useful for the GeoAI community

### Weaknesses
- Heavy reliance on proprietary mobility data: 
    - This causes concerns related to reproducibility. I don't believe this resource is publicly available
    - What about rural areas where human activity is sparse?
- Single country mobility data: The mobility data is unique to china, casting fairly large doubts on the generalizability of the method. For example, does this method generalize to Europe or the US?
- The new benchmark primarily describes human-centric socio-economic tasks. It would make sense that mobility data would be more useful here (as demonstrated in their experiments). Other tasks such as land cover classification would likely see better results under the more Earth-observation-centric representation learning strategy. Unfortunately, this trade-off is left unexplored.

### Questions
- My main question is about generalization outside of China? Can the authors demonstrate that MoRA can perform well in other national geographies?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a graph-based location representation learning framework that is based on relative location transitions from one (hexagonal) grid cell to another. The framework, despite architecturally complex, is well justified and experimentally strong across spcial and economic tasks. While I would also appreciate benchmarking on natural downstream tasks (mentioned in Figure 1, but not benchmarked against), I understand that the premise of mobility mainly works on social and economical processes. 

Overall, I find the paper well-written and well-presented and the results convincing.

### Strengths
* Clear and well-justified framework for graph-based location representation learning via relative transitions across hexagonal grid cells.
* Strong empirical performance on spatial and socioeconomic downstream tasks.
* Well-written and clearly presented.

### Weaknesses
* Fairly complex methodology including single-modality encoders and a separate graph neural network. A joint unifying architecture would be less engineering-focused. However, ablations justify the individual components well
* It would have been nice to also capturing natural tasks, where mobility matters. For instance, species distribution modelling, like iNaturalist of BirdSnap would be good choices here

### Questions
* With MORA, would it be possible to pre-compute and store embeddings for all H3 cells in a region — similar to how AlphaEarth maintains a precomputed embedding database?
* At inference time, does one need to supply POI/satellite imagery or demographics to obtain an embedding, or can the trained GNN alone generate embeddings without requiring additional data downloads?

### Soundness
3

### Presentation
3

### Contribution
3
