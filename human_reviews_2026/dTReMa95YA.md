# Modality Curation: Building Universal Embeddings for Advanced Multimodal Information Retrieval

- Decision: Reject
- Scores: 4, 6, 6, 6

## Abstract
Multimodal information retrieval (MIR) faces inherent challenges due to the heterogeneity of data sources and the complexity of cross-modal alignment. While previous studies have identified modal gaps in feature spaces, a systematic approach to address these challenges remains unexplored. In this work, we introduce UNITE, a universal framework that tackles these challenges through two critical yet underexplored aspects: data curation and modality-aware training configurations. Our work provides the first comprehensive analysis of how modality-specific data properties influence downstream task performance across diverse scenarios. Moreover, we propose Modal-Aware Masked Contrastive Learning (MAMCL) to mitigate the competitive relationships among the instances of different modalities. Our framework achieves state-of-the-art results on multiple multimodal retrieval benchmarks, outperforming existing methods by notable margins. Through extensive experiments, we demonstrate that strategic modality curation and tailored training protocols are pivotal for robust cross-modal representation learning. This work not only advances MIR performance but also provides a foundational blueprint for future research in multimodal systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper addresses modality heterogeneity and cross-modal alignment in Multimodal Information Retrieval (MIR) by proposing UNITE, a universal embedding framework with two core innovations: systematic modality curation and Modal-Aware Masked Contrastive Learning (MAMCL). Trained via a two-stage pipeline, UNITE achieves SOTA on 40+ tasks and supports simultaneous fine-grained/instruction-based retrieval for text/image/video/fusions.

### Strengths
1.	UNITE achieves leading results across 40+ retrieval tasks (coarse-grained, fine-grained, instruction-based), outperforming both smaller specialized models and larger competitors (e.g., 2B UNITE surpasses 7B VLM2Vec on WebVid-CoVR).
2.	Systematic analysis reveals T-V pairs excel in general retrieval and even outperform T-I pairs in image-text tasks—contradicting traditional assumptions and guiding more efficient MIR data curation. It's a interesting finding.

### Weaknesses
1.	Inadequate Theoretical and Modal Coverage in Modality Curation. The paper focuses on analysis of T-V pair effectiveness but lacks theoretical explanations for why certain data types (e.g., T-V outperforming T-I in image-text tasks) yield better results. Additionally, while it emphasizes "curating modality data" as a core contribution, the analysis is mostly limited to T→V retrieval conclusions. It fails to clarify how other modalities (e.g., text-text, image-text) should be managed—whether they are still mixed as in traditional methods, or if there are optimized proportion strategies?
2.	MAMCL’s design lacks sufficient novelty, as prior contrastive learning works (e.g., sampling batches from a single data source to ensure uniform modality) have already addressed cross-modality interference, making MAMCL a similar but not groundbreaking approach. Moreover, MAMCL masks negative examples of different modalities, which wastes a portion of negative samples. Given that the quantity and quality of negative examples are critical for contrastive learning performance, this waste may limit the model’s ability to learn discriminative representations.

### Questions
1.	Table 7 (1→2) shows that MAMCL reduces the performance of WebVid CoVR. Does clarifying this mean that MAMCL will cause losses in specific circumstances?
2.	What other functions does the conclusion obtained in Analysis 5 have, apart from guiding the addition of T-V data in retrieval adaptation?  Is the data- curation meaningless, merely indicating that the data of TV needs to be included? Because in the end, all the data were mixed together, just like the previous work. Moreover, the proportion of different modal data has not been explored either.
3.	The article does not provide specific prompts for Instruction-based Retrieval. Since instructions determine the tuning results, omitting them will hinder replication. Supplement these in appendix.

### Soundness
2

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
This paper introduces UNITE, a multimodal embedding training paradigm in two stages -- retrieval adaptation followed by instruction tuning, and a loss variant called Modal-Aware Masked Contrastive Learning (MAMCL). 
Specifically, MAMCL restricts negatives to the same target-modality type to reduce inter-modal interference. 
The authors emphasize modality data curation (proportions and sequencing of TT/TI/TV pairs) and report state-of-the-art results on instruction-based and fine-grained retrieval benchmarks, notably MMEB and WebVid-CoVR.

### Strengths
1. Broad generalization and strong performance in video retrieval: the proposed UNITE models perform strong across various retrieval scenarios, tasks, and granularities. On WebVid-CoVR, UNITE_instruct-7B exceeds baselines under their reported settings.
2. Proper ablations: The paper includes a dedicated MAMCL ablation (Table 7) and a full training-data composition analysis (TT/TI/TV mix, under fixed data budget).

### Weaknesses
1. Marginal performance of the MAMCL component: while MAMCL is conceptually sound, its average gains are small (about +0.3 overall on MMEB, avg of +0.5 on WebVid-CoVR with 7B parameters), and it can trade off specific metrics (e.g., CoVR R@5 at 7B). I recommend deeper analysis on when/why it helps.
2. Lack of efficiency analysis: MAMCL changes the effective negative set via a modality mask, but the paper does not report compute comparisons to standard InfoNCE; only high-level training setup (e.g., 64×A100, single-epoch runs, time) is provided. Adding a theoretical analysis or a wall-clock comparison would strengthen this paper.

### Questions
See weaknesses

### Soundness
4

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
The paper proposes UNITE, a framework for building unified embeddings for multimodal information retrieval, supporting text, images, videos, and their combinations. A key contribution is the introduction of MAMCL (Masked Contrastive Learning), which mitigates interference by masking contrastive terms between candidates of different modality types, thereby improving cross-modal alignment in the shared embedding space. The training methodology is structured into two stages: a retrieval adaptation phase, which leverages diverse modality pairs to construct a robust embedding space, and an instruction tuning phase, where the model is fine-tuned on instruction-based datasets to handle more complex and nuanced retrieval queries. Experiments on multiple benchmarks show consistent improvements, and ablation studies confirm MAMCL’s effectiveness across various instruction-based retrieval tasks.

### Strengths
The paper introduces a novel Modality-Aware Masked Contrastive Learning (MAMCL) approach that extends contrastive learning to better accommodate heterogeneous modalities. The framework's ability to integrate video retrieval alongside text and image retrieval broadens its applicability and underscores the model's versatility in handling complex multimodal data.
Comprehensive evaluations across diverse multimodal benchmarks substantiate the benefits of both MAMCL and the modality-aware data design. The inclusion of both in-distribution and out-of-distribution analyses strengthens the empirical validity of the claims.
The paper is generally well-structured and clearly written. The motivation, methodology, and experimental results are communicated with precision, and the technical formulations are clearly presented and easy to follow. While some training details are appropriately included in the appendix, certain key aspects—such as the specific large language model (LLM) used and the preprocessing steps for each modality, particularly for video—would be better placed in the main body to improve transparency and reproducibility.
This work makes a notable contribution by proposing the MAMCL approach, which mitigates cross-modal interference and enhances alignment across heterogeneous modalities. The proposed framework advances efforts toward building universal multimodal embeddings that can jointly handle text, image, and video retrieval tasks. The results demonstrate clear improvements on several benchmarks, indicating the approach's practical value and potential for broader application.

### Weaknesses
While the paper presents a well-motivated and empirically supported approach, several aspects could be strengthened to enhance its clarity and overall impact.
1- Frozen projector and vision encoder:
The authors freeze the projector and vision encoder, but do not analyze the implications of this choice. It remains unclear how fine-tuning these components—particularly during instruction tuning—might affect multimodal alignment and retrieval performance. An ablation study comparing frozen versus trainable projectors would provide valuable insight into the trade-off between stability and adaptability, strengthening the paper’s empirical analysis.
2- Limited model diversity:
Extending experiments to other state-of-the-art vision–language models for multimodal information retrieval would provide more substantial evidence of robustness and generality, and help disentangle the contribution of MAMCL from the underlying model architecture.
3- Suboptimal video sampling strategy
The paper employs a uniform frame sampling rate of one frame per second for the video modality, which may be too coarse to capture meaningful temporal dynamics. This approach could overlook key motion cues or fine-grained visual changes that are crucial for accurate retrieval. Exploring more efficient or adaptive sampling strategies could improve video representation quality and strengthen overall multimodal retrieval performance.
4- Lack of qualitative success and failure examples:
The paper focuses heavily on quantitative results but omits illustrative examples of both successful and failed retrieval cases. Including such visual or textual samples would help readers better understand the model’s strengths, limitations, and failure patterns—particularly for instruction-based or fine-grained retrieval scenarios.

### Questions
- Have the authors considered conducting an ablation study with the projector unfrozen during training to examine its effect on learning dynamics, stability, and retrieval performance?
- Since MAMCL is presented as a general loss function, could the authors clarify why it was evaluated only on Qwen2-VL-2B?
Do you expect similar performance trends if integrated into other VLMs such as CLIP, BLIP-2, or LLaVA?
If computational limits prevented broader testing, could you provide reasoning or partial results indicating its generalizability?
- Have the authors considered testing alternative video sampling strategies (e.g., motion-based, content-aware, or adaptive sampling) instead of the fixed rate of one frame per second to evaluate their effect on retrieval accuracy and temporal representation quality?
- Would it be possible to include qualitative examples of both successful and failed retrieval cases (especially for instruction-based tasks)? Such examples could clarify where MAMCL contributes most effectively and where it struggles—insights that would be very helpful to the community.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a training strategy for large multimodal models (LMMs) for the task of cross-modal retrieval. The proposed strategy involves a masking matrix that informs the loss function about the type of data modality involved in the data used to calculate the contrastive objective. In addition, the paper conducts experiments that evaluate the importance of different data compositions for training, identifying data curation regimes that improve performance. The experimental results indicate that performance improves in various tasks and datasets.

### Strengths
* Comprehensive evaluation across datasets and tasks.
* Simple, yet effective strategy.
* Good experimental results that indicate effectiveness of the proposed strategy.
* Generally speaking the study seems well designed and well conducted.

### Weaknesses
* Writing style is too distracting. Every time the paper indicates that something is critical or crucial, it is not.
* The contribution seems incremental with tweaks to existing models and based primarily in data curation.
* The paper reports results that they say contradicts previous observations, but no reference results or citations are provided. 
* The interpretation of results and the insights is limited to highlighting numeric differences.

### Questions
* Are the results really surprising? How can these observations be explained beyond performance differences? Any hypotheses that can be tested or just black box performance differences that cannot be explained?
* Are the data curation results just a selection of data useful to solve the benchmarks rather than learning generalizable reasoning to match content?

### Soundness
3

### Presentation
2

### Contribution
3
