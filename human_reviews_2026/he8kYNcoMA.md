# ST-SimDiff: Balancing Spatiotemporal Similarity and Difference for Efficient Video Understanding with MLLMs

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 8

## Abstract
Multimodal Large Language Models (MLLMs) face significant computational overhead when processing long videos due to the massive number of visual tokens required. To improve efficiency, existing methods primarily reduce redundancy by pruning or merging tokens based on importance or similarity. However, these approaches largely overlook a critical dimension of video content, i.e., changes and turning points, and they lack a collaborative model for spatio-temporal relationships.
To address this, we propose a new perspective: similarity is for identifying redundancy, while difference is for capturing key events. Based on this, we designed a training-free framework named ST-SimDiff. We first construct a spatio-temporal graph from the visual tokens to uniformly model their complex associations. Subsequently, we employ a parallel dual-selection strategy: 1) similarity-based selection uses community detection to retain representative tokens, compressing static information; 2) temporal difference-based selection precisely locates content-changing points to preserve tokens that capture key dynamic shifts. This allows it to preserve both static and dynamic content with a minimal number of tokens. Extensive experiments show our method significantly outperforms state-of-the-art approaches while substantially reducing computational costs.
Our code is available in [https://github.com/bingjunluo/ST-SimDiff](https://github.com/bingjunluo/ST-SimDiff).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper aims to address the problem of extremely high computational costs caused by the massive number of visual tokens when multimodal LLMs process long videos. It makes three main contributions: 1) It is the first to focus on inter-frame differences in videos, emphasizing that change is key to video understanding. 2) It proposes a method for constructing spatiotemporal graphs that uniformly models spatial similarity and temporal continuity. 3) It introduces a new token selection strategy: for token clusters with high similarity in the graph, only a few representative tokens are retained, while tokens with extremely high dissimilarity on temporal edges are preserved.

### Strengths
This paper proposes ST-SimDiff, a training-free framework for video token compression in multimodal large language models. The core innovation lies in simultaneously leveraging similarity (to identify redundancy) and dissimilarity (to capture key events) for token selection. The method constructs a spatiotemporal graph and achieves dual-path parallel selection by combining community detection and temporal difference detection. Experiments demonstrate state-of-the-art performance on VideoMME, LongVideoBench, and EgoSchema, while significantly reducing computational costs. The work exhibits strong novelty, effectively achieves its research objectives, and thoroughly validates the method's effectiveness across multiple models and datasets.

### Weaknesses
1.Does the compressed video still retain the audio modality? If yes, how is the alignment between video and audio achieved? If not, does the absence of audio bring negative impacts? If so, how can the negative effects caused by the missing audio be mitigated?
2.For the same input, this framework seems to produce only one type of output regardless of how the text query instruction varies (rather than automatically adapting based on instruction changes). For different video analysis objectives, is this fixed, unified video pruning framework insufficiently flexible?
3.Constructing the spatiotemporal graph requires traversing all visual tokens. For videos with extremely high resolution or extremely long duration, is this framework still applicable? In other words, what is the estimated upper limit of visual tokens that this framework can handle?
4.Considering only changes between adjacent frames may lead to the model being insufficiently sensitive to slowly changing scenes.

### Questions
1. Does the community detection algorithm actually use Louvain or connected components? If it's the latter, why not use the superior Louvain algorithm?
2. In Table 1, the ST-SimDiff results at Token Retain Ratio (r=50%) even exceed the Token Retain Ratio (Full Performance) results (63.3). Why does this phenomenon occur where the compressed video performs better than the original video? The paper should provide a discussion on this.
3. Could you provide some concrete examples showing which tokens are selected by SRTS and which are selected by DETS?
4. When selecting difference frames, the video only considers changes between a few adjacent frames, which may lead to the model being insufficiently sensitive to slowly changing scenes.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the computational overhead MLLMs face when processing long videos. The authors propose ST-SimDiff, a training-free token reduction framework. The core idea is that video compression must balance two aspects: using similarity to identify and compress redundant static content, and using difference to preserve key dynamic events or turning points. The method constructs a spatio-temporal graph to model token relationships. It then uses community detection for similarity and temporal difference thresholding for key events to generate a compact, information-rich token subset. Experiments show SOTA performance on multiple benchmarks.

### Strengths
1. The paper is well-written and clearly articulates a significant and practical problem. The core concept of striking a balance between spatiotemporal similarity and temporal difference constitutes a valuable contribution to this field.
2. The proposed framework is training-free, making it highly practical, and easily applicable as a plug-and-play module. The use of a spatio-temporal graph to uniformly model complex token relationships is good way to implement the core idea.

### Weaknesses
1. Discrepancy in baseline performance. The paper's validation on Qwen2.5-VL (Appendix, Table 4) suffers from a baseline inconsistency. The paper reports the "Upper Bound (Full Performance)" for Qwen2.5-VL as 62.9 on VideoMME and 59.2 on LongVideoBench. However, the official Qwen2.5-VL technical report states a score of 65.1 on VideoMME and 56.0 on LongVideoBench. The authors should address or explain this discrepancy.
2. Lack of Image-based Validation. The paper focuses exclusively on video datasets. However, the core mechanism, particularly the Similarity-based Representative Token Selection (SRTS, is designed to handle spatio-temporal redundancy. The spatial component of this logic should be directly applicable to compressing redundant tokens in static images. Experiments on standard image-based VLM benchmarks could showcase the robustness of the similarity-based compression module.
3. Qualitative Visualization. The paper's core hypothesis relies on the intuitive concepts of "similarity" (static content) and "difference" (key events). While the quantitative results are strong, the paper provides no qualitative visualizations. It would be highly beneficial to include figures that show, for specific video examples, which tokens are selected by the SRTS module versus the DETS module. This would provide direct and intuitive evidence that the framework is truly capturing static backgrounds and dynamic turning points as intended.

### Questions
See weaknesses above.

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
5

### Summary
This paper presents ST-SimDiff, a balanced framework that explicitly models the trade-off between spatial-temporal similarity and difference in video representations. The method aims to improve video understanding by aligning similar semantics while preserving meaningful diversity across frames. The authors introduce novel similarity–difference guided modules that can be integrated into existing video backbones with minimal modifications. Extensive experiments on standard benchmarks demonstrate consistent improvements over strong baselines, showing superior performance across multiple tasks with manageable computational overhead.

### Strengths
This well-executed, clearly written paper addresses the critical issue of balancing temporal consistency and diversity in video representation learning. The ST-SimDiff framework is elegant, conceptually sound, and easily integrates into existing architectures, delivering a significant performance boost without the need for additional supervision or costly retraining.
The paper effectively highlights how previous methods either overemphasize similarity or difference, while ST-SimDiff strikes a balanced, principled approach. The motivation is compelling, the technical formulation is clear, and the presentation flows smoothly.
The design of interpretable components effectively clarifies how spatial-temporal cues interact. The experimental validation is comprehensive, covering diverse datasets, multiple backbones, and various evaluation metrics, with ablations that isolate the contribution of each component. Results are strong and consistent.

### Weaknesses
The paper lacks crucial qualitative visualizations. It does not show failure cases or ambiguous scenarios, making it difficult to understand the mechanism's practical limitations or how it behaves when it produces an incorrect interpretation.
The framework's performance boundaries are not explored. The paper fails to investigate video types or specific tasks where the proposed balancing paradigm might be suboptimal or ill-equipped.
The paper does not address the critical misalignment problem that arises from sequence pruning. While pruning/dropping is a common efficiency method, it creates a gap between the dense data the model was trained on and the sparse data it receives at inference, for which the model has no explicit handling mechanism.
The paper lacks a method for error attribution, making it difficult to determine whether a specific failure originates from the similarity module, the difference module, or their interaction.

### Questions
Can the authors provide qualitative visualization results, especially highlighting where the similarity–difference mechanism fails or produces ambiguous interpretations?
Where are the performance boundaries of this framework? Are there specific video tasks/types that ST-SimDiff is ill-equipped to handle?
How does the framework solve the input misalignment problem after pruning?
If the model makes an incorrect judgment on a video, how can you attribute the failure to the similarity module versus the difference module?

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
The paper presents a novel and insightful framework for video token compression, effectively balancing similarity-based redundancy reduction with difference-based key event capture. This training-free, dual-path approach is technically sound and demonstrates strong, generalizable performance across multiple models and significant computational savings.

### Strengths
1.The paper makes a significant contribution by addressing the need for efficient video understanding through the dual focus of redundancy compression (via similarity) and key event capture (via difference). This approach moves beyond prior work that primarily focused on similarity or importance-based pruning alone.\
2.The proposed ST-SimDiff framework is technically sound, employing a spatio-temporal graph to model token relationships uniformly. The dual-path SRTS and DETS effectively implement the core motivation, demonstrating significant and consistent performance improvements across a range of state-of-the-art, model-agnostic baselines, including LLaVA-Video, NVILA-Video, and Qwen2.5-VL.\
3.As a training-free framework, ST-SimDiff introduces minimal overhead and operates with substantially improved efficiency. Its computational complexity is analyzed as O(Nd), much more efficient than the $O(N^2d)$ self-attention it aims to reduce. Empirical results show significant reductions in inference time and peak GPU memory usage.

### Weaknesses
1.The final pruning step, where the initial candidate set exceeds the target budget, is underexplained in Section 4.5. It is unclear which layers are selected, how the scores are aggregated, and the impact of this step on performance.\
2.The paper lacks qualitative examples, such as visualizations showing which tokens are selected. This makes it challenging to build intuition about the method’s behavior and understand its potential failure modes.\
3.There is no comparative analysis or justification to explain why the connected components algorithm is chosen in SRTS. The impact of this choice is less investigated.

### Questions
1.Which exact shallow layers are used in the final attention-based pruning step? What is the specific aggregation function for attention scores across heads and layers?\
2.Provide qualitative visualizations that highlights which tokens are preserved by the proposed method to help build intuition and identify failure modes.\
3.Why do you choose the connected components algorithm compared with other mainstream community detection methods, especially in terms of both understanding performance and processing efficiency?\
4.Could the framework's performance be further enhanced by making parts of it learnable? What are the potential benefits versus the cost?\
5.The SRTS and DETS paths are currently combined, seemingly uniformly. Is it possible to dynamically weight these two paths for better performance?

### Soundness
3

### Presentation
3

### Contribution
3
