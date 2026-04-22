# Needle: A Generative AI-Powered Multi-modal Database for Answering Complex Natural Language Queries

- Avg Score: 5.20
- Decision: Reject
- Scores: 6, 4, 8, 4, 4

## Abstract
Multi-modal datasets, like those involving images, often miss the detailed descriptions that properly capture the rich information encoded in each item. This makes answering complex natural language queries a major challenge in this domain. In particular, unlike the traditional nearest neighbor search, where the tuples and the query are represented as points in a single metric space, these settings involve queries and tuples embedded in fundamentally different spaces, making the traditional query answering methods inapplicable. Existing literature addresses this challenge for image datasets through vector representations jointly trained on natural language and images. This technique, however, underperforms for complex queries due to various reasons.
This paper takes a step towards addressing this challenge by introducing a Generative-based Monte Carlo method that utilizes foundation models to generate synthetic samples that capture the complexity of the natural language query and represent it in the same metric space as the multi-modal data.
Following this method, we propose Needle, a database for image data retrieval. 
Instead of relying on contrastive learning or metadata-searching approaches, our system is based on synthetic data generation to capture the complexities of natural language queries. Our system is open-source and ready for deployment, designed to be easily adopted by researchers and developers.
The comprehensive experiments on various benchmark datasets verify that this system significantly outperforms state-of-the-art text-to-image retrieval methods in the literature.
Any foundation model and embedder can be easily integrated into Needle to improve the performance, piggybacking on the advancements in these technologies.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces NEEDLE, a novel system for complex natural language query answering in multi-modal databases, specifically focusing on image retrieval. The core innovation is a Generative AI-powered Monte Carlo method that reframes the text-to-image search problem as an image-to-image search. Instead of relying on traditional contrastive learning, NEEDLE uses foundation models to generate synthetic "guide" images from the text query. These guide images, acting as stochastic representations of the query's intent, are then used with an ensemble of image embedders to perform nearest-neighbor search against the database. The system is enhanced with several optimizations, including a query complexity classifier to bypass expensive generation for simple queries and an implicit metadata generation loop for continuous improvement.

### Strengths
1. The proposed method of using generative models to create synthetic guide images is a novel and promising approach to bridge the semantic gap between complex text queries and image data. It cleverly transforms a cross-modal retrieval task into a more direct image-to-image search problem.

2. The system design is thorough and practical. Beyond the core theoretical contribution, the authors have developed a complete, open-source system with crucial efficiency optimizations like the Query Complexity Classifier and an Implicit Metadata Generation mechanism, which address the practical latency concerns of on-the-fly image generation.

3. The ablation studies are extensive and insightful. They effectively dissect the system's performance, analyzing the impact of key hyperparameters such as the number of guide images, the number of embedders, the choice of foundation models, and the effect of outlier detection, which validates the design choices.

### Weaknesses
1. The primary performance bottleneck and a significant practical challenge is the latency and computational cost associated with on-the-fly image generation by large foundation models. While the paper proposes optimizations to mitigate this, the reliance on this step for complex queries may still hinder real-time performance in large-scale, interactive applications.

2. The effectiveness of the system is heavily dependent on the quality of the generative foundation models and the image embedders. Any inherent biases, limitations, or failures of these underlying models (e.g., difficulty with compositional ordering, as noted by the authors) will be directly inherited by NEEDLE, potentially impacting its reliability and fairness.

3. The system's complexity is quite high, involving multiple components (generative models, multiple embedders, a vector store, a query classifier, etc.). This could present challenges for deployment, maintenance, and reproducibility compared to simpler, end-to-end contrastive learning models.

### Questions
N/A

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
4

### Summary
This paper propose *Needle*, a retrieval system for image databases that answers complex natural-language queries by using a text-to-image generator to create several "guide" images for the query. The authors propose to embed those guides with an ensemble of image embedders, by running multiple k-NN lookups over the target image collection, and aggregating the rankings (with optional per-topic "trust weights"). The core idea is framed as a Monte Carlo estimator of the (unknown) query-image distance: average distances between many guide-image embeddings and candidate images to approximate the true semantic distance. The authors report their results on CLIP, ALIGN, FLAVA and a BLIP+MiniLM pipeline on object-centric datasets and compositional retrieval benchmarks (COLA, Winoground, NoCaps, SentiCap).

### Strengths
- The proposed method is simple yet effective: the authors turn a complex text-to-image retrieval problem into many image-to-image searches with an engineering and system architecture focus (vector store, caching, outlier filtering, embedder weighting)

- The proposed system is modular: any generator and any embedder can be swapped (with weighting embedders and filtering out bad guides)

- Strong performance on R@10/MAP/MRR and pairing accuracy against the included baselines.

### Weaknesses
- The paper claims to be the first to use generative models to synthesize query-side signals for multimodal retrieval, but prior work has already explored generative modeling for image retrieval [1, 2, 3].

- In section 2, theorem 1 is based on the assumption that "Each guide tuple is considered an i.i.d. sample from a distribution whose mean is the ideal (but unknown) tuple that perfectly represents the query." However, this relies heavily on each guide tuple to be unbiased. There is no experiment estimating the bias/variance of the estimator as a function of number of guides and embedders, nor any stress test when generators systematically miss relations, which is particularly common for compositional prompts.

- This paper omits several strong modern baselines widely used for retrieval and compositional reasoning, e.g., OpenCLIP ViT-bigG/EVA-CLIP, SigLIP families, or recent compositional adapters for CLIP/FLAVA trained specifically on attribute-object binding (which COLA’s authors show matter).

- In the evaluations, the hard set definition uses CLIP AP < 0.5 to decide difficulty and then reports big gains over CLIP on that subset. 

- Consequently, having the Related Work in the supplementary material is a big downside -- given the ICLR reviewer guidelines (It is not necessary to read supplementary material (...)), this seems a major flaw in this paper. I strongly recommend moving this section to the main paper for future resubmissions; this helps reviewers and readers understand the position of the paper, its claims and differences with prior and concurrent work.

[1] Zhang, Yidan, et al. "Irgen: Generative modeling for image retrieval." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024.

[2] Li et al. Generative Cross-Modal Retrieval: Memorizing Images in Multimodal Language Models for Retrieval and Beyond (ACL 2024)

[3] Kim, Sungyeon, et al. "GENIUS: A generative framework for universal multimodal search." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

### Questions
- When generators systematically miss relations (e.g., "a lightbulb surrounding some plants"), how biased is the distance δ between the query φ and any tuple t?

- Why not including SigLIP, OpenCLIP bigG/EVA-CLIP, and COLA-adapted multimodal layers?

- How sensitive are results to prompt phrasing and seed selection for the guide images?

This is more of a suggestion: SIGIR or CIKM seems like a better fit for this submission.

### Soundness
3

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
To improve the accuracy and speed of text-based image retrieval, the authors propose a system called NEEDLE. Instead of relying on contrastive learning to match text queries to images, NEEDLE re-formulates the task as an image-to-image search problem by using image foundation models to generate "guide images" based on the query text, which are then used to conduct the actual retrieval. The authors propose and integrate several practical optimizations to transform NEEDLE into a robust and efficient system, which results in low-latency (0.203s) retrieval on a 2xTesla T4 GPU machine, a latency competitive with that of non-generative methods such as a CLIP search. Quantitative benchmark numbers and a 20-user user study show that NEEDLE achieves state-of-the-art performance on text-based image retrieval compared to concurrent baselines.

### Strengths
Very solid baselines to compare against! Baselines include (in L239): CLIP [58], ALIGN [28]; FLAVA [65] and CoCa [85]; BLIP and MiniLM [39, 77]; and PlugIR [34]. 

A wide range of datasets for evaluating retrieval performance (L249)!

Very solid ablations that include the choice of hyperparameters (e.g. number of guide images and embedders in FIgure 4), choice of image generation models for creating guide images (Figure 5), and the separate effect of using generated guide images (Table 5).

The engineering seems very well thought-out to me. The paper has taken measures to improve the efficiency of the system for providing an actually good user experience, and not solely chasing benchmark numbers. The authors also provide details on the test system setup and include breakdowns for the inference time for running NEEDLE on LVIS, which I believe is fast enough for real-time user interactions with a handful of users despite the test system having somewhat outdated hardware (Tesla T4 GPUs).

It looks like the authors have made a fully-functional prototype for NEEDLE, which is great especially given the paper posits itself as making a "production-ready" system (L100).

User studies in Figure 12 shows NEEDLE is strongly favored over naive CLIP retrieval on LVIS.

### Weaknesses
Using Local Outlier Factor (LOF) to detect poor-quality generated guide images sounds like it would also eliminate any image that's "novel" or "unique" or "creative", because these would also cause a guide image to have a high LOF. If the user is an artist who draws distinct images (compared to the average internet images), and wants to search for one of their own creations among other more "nomral-looking" images, I believe the LOF rule would accidentally discard what may be valid guide images. Can the authors provide further justification for using LOF, or show that this unwanted elimination does not happen in practice? I'd be happy to raise my rating given further corroboration,.

### Questions
For Theorem 1 to hold, I think we also need some assumptions about the independence among the embeddings from different embedders to apply the Chernoff bound? That is, I think Theorem 1 should require $\{\mathcal{E}_l(\bar{g}_j)\}$ to be all independent. Yes, the embedders are distinctly trained, but the architectural and data biases would probably not make their embeddings of the same image independent. This may not cause a major issue in practice, but I believe the non-independence means the bound is less effective than Theorem 1 would suggest. It should not depend on $l$ as we cannot guarantee the embedders to give independent outputs. 

I'm very interested in the discussion on "Reliance on Existing Models" (Appendix A). I wonder what new biases using generative image models introduces to image retrieval. An extended discussion with experiments would be appreciated.

Minor comments:
It'd be great to see more examples of the queries the users wrote during the user study. How complex are they? What "topics" did they talk about?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces NEEDLE, a generative-AI-powered multimodal retrieval system designed to answer complex natural language queries over image datasets. Instead of relying on contrastive joint embeddings, the authors propose a Generative Monte Carlo framework that transforms a text query into multiple synthetic “guide” images using foundation models. These generated images are then embedded with multiple vision encoders (EVA, RegNet, etc.) and used to perform image-to-image nearest-neighbor search. The paper presents both a theoretical justification for Monte Carlo sampling, a system-level implementation with dynamic embedder trust weighting and outlier filtering, and an optimized inference pipeline that includes a query complexity classifier to bypass unnecessary image generation. Extensive experiments on benchmarks show consistent gains over prior vision–language models in retrieval metrics.

### Strengths
-  The idea of translating text queries into synthetic multimodal representations before retrieval is conceptually interesting. It reframes the retrieval task as a generative sampling problem, bridging generation and search.
- The paper goes beyond an algorithmic prototype and implements a deployable system with modular embedders, anomaly filtering, and caching. The inclusion of practical optimizations shows strong engineering effort.
- Comprehensive experiments are provided across multiple datasets with ablations on guide-image count, number of embedders, and foundation models. This gives a good sense of robustness under different configurations.

### Weaknesses
- The paper only compares against 2023-era contrastive models (CLIP, ALIGN, FLAVA, CoCa, BLIP). Recent multimodal embeddings such as UniIR[1] and E5-V[2] and VLM2Vec[3] using MLLM are now standard baselines for multimodal retrieval. Without evaluating against these stronger MLLM representations, it is unclear whether NEEDLE remains competitive in the modern multimodal landscape.

- The paper reports a total inference time of 0.203 s comparable to CLIP’s 0.184 s despite including computationally heavy image generation and multi-embedder inference. Hardware details, batch size, image resolution, and generator configuration are missing, making this efficiency claim scientifically unreliable.

- The evaluated benchmarks (COCO, LVIS, COLA, Winoground) test static image retrieval and compositional captions, not complex multimodal or reasoning-based queries. The claim of “natural-language query answering over multimodal databases” is overstated; the method is still restricted to image retrieval.

- The evaluation focuses only on static image retrieval tasks (COCO, LVIS, Winoground). This evidence is insufficient to support the paper’s claim of handling “complex natural-language queries.” If the authors truly intend to demonstrate compositional multimodal reasoning, a retrieval-augmented generation (RAG)–based visual question answering or multimodal reasoning setup would be far more appropriate. As it stands, the current retrieval-only setup does not convincingly show such complexity, and it is doubtful that the system would handle open-domain or knowledge-grounded VQA (e.g., Wikipedia-style) questions, where reasoning requires world knowledge.

- NEEDLE depends on multiple vision encoders, yet the paper provides no standalone evaluation of each embedder’s retrieval performance, parameter count, or computational footprint. 

[1] Uniir: Training and benchmarking universal multimodal information retrievers. Wei et. al, ECCV24
[2] E5-v: Universal embeddings with multimodal large language models, Jiang et. al.,
[3] Vlm2vec: Training vision-language models for massive multimodal embedding tasks, Jiang et. al.,

### Questions
- Recent MLLM-based embedding models such as E5-V and VLM2Vec achieve much stronger performance and already support complex multimodal query understanding through instruction tuning. Have the authors compared NEEDLE against these newer baselines?

- The reported inference latency (0.203 s) is nearly identical to CLIP despite including foundation-model image generation and multiple encoders. Could the authors clarify the exact measurement conditions such as hardware, batch size, resolution, and generation model used?

- NEEDLE relies on encoders such as EVA and RegNet, but their individual parameter sizes, FLOPs, and retrieval accuracy are not reported. Could the authors provide quantitative comparisons that isolate the extent to which the proposed generative sampling contributes to the performance gain compared to embedders?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes using image generation model to perform Monte Carlo sampling, transforming complex natural language queries into multiple image queries for conducting complex natural language-to-image retrieval. Based on this framework, the paper designs a series of optimization methods to improve the efficiency of the overall system workflow. In the experiments, the proposed method outperforms the compared multimodal alignment approaches, particularly on the hard set.

### Strengths
The paper is logically well-structured. It is progressing smoothly from problem analysis, motivation, to the proposed solution, and then to optimization strategies. The approach is coherent and reasonable.
From a system perspective, this paper addresses and optimizes practical issues faced by the generate-and-retrieve paradigm, tackling problems of notable significance. 
The experimental analysis is sufficiently thorough, making the results convincing.
The code framework is well-opened and complete, making deployment very convenient.

### Weaknesses
Using image generation models for cross-modal retrieval is not novel, as many related works [1,2,3] have conducted similar studies. 
Although the paper emphasizes the use of foundation models, it only experiments with image generation models and does not include VLM, which intuitively might be more suitable for text-image retrieval tasks.
As Figure 5(d), using image generation model is significantly less efficient compared to general retrieval methods. Although the paper proposes several optimization strategies, the performance gap remains substantial.
[1] Zijun Long, et al. Diffusion Augmented Retrieval: A Training-Free Approach to Interactive Text-to-Image Retrieval. 
[2] Ran Zuo, et al. SceneDiff: Generative Scene-Level Image Retrieval with Diffusion Models. 
[3] Lan Wang, et al. Generative Zero-Shot Composed Image Retrieval.

### Questions
Since the method already employs a closed-source image generation model, it might be worthwhile to compare against more recent and stronger baselines, such as VLM-based embedding models. I am curious about the advantages of using an image generation model when compared with other methods of similar scale and complexity.

### Soundness
3

### Presentation
3

### Contribution
2
