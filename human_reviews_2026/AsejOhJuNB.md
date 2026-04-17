# Learning and Evaluating Visual Similarity Discovery under Incomplete Labeling

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 4, 2

## Abstract
Visual Similarity Discovery (VSD) focuses on retrieving positives: images of distinct objects that exhibit perceptual similarity to a given query. This is a core need in applications like e-commerce and visual search. This work advances VSD research through several key contributions. First, we introduce a new VSD dataset in the furniture domain with over 63K labeled image pairs, providing a valuable resource for VSD learning and evaluation. Second, we propose two evaluation metrics that enable more reliable and consistent VSD performance assessment under incomplete labeling. Third, we show that supervised finetuning of multiple pretrained models on VSD labels significantly improves VSD performance. Moreover, we present Soft Positive Augmentation, a method that leverages existing VSD labels to infer soft positive relations among unlabeled pairs via weighted graph transitivity. Augmenting the VSD labels with these inferred soft positives during finetuning yields additional performance gains. Our code and dataset will be made publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the challenge of Visual Similarity Discovery (VSD) under incomplete labeling, where many visually similar image pairs remain unlabeled due to generator-based retrieval bias. This limitation affects both evaluation reliability and model training.

The paper makes three main contributions:

- VSD-Furniture Dataset:
A new human-annotated dataset of over 63K image pairs in the furniture domain, extending prior VSD benchmarks beyond fashion using the Efficient Discovery of Similarities (EDS) paradigm.

- Evaluation Metrics for Incomplete Labeling:
The authors propose two new metrics to improve robustness under sparse annotations:
Discounted Credit Score (DCS) emphasizes top-ranked retrievals and mitigates AUC’s limitations, while Estimated Hit-Ratio at K (EHR@K) normalizes for unlabeled results to ensure consistent evaluation.

- Supervised Fine-tuning with Soft Positive Augmentation (SPA):
Fine-tuning pretrained models such as CLIP, DINOv2, and BEiT on VSD labels significantly improves performance. The proposed SPA method further enhances results by inferring soft positive relationships among unlabeled pairs via weighted graph transitivity.

Overall, the paper presents a unified and practical framework for learning and evaluating VSD models in realistic, partially labeled settings.

### Strengths
- Clearly explains the necessity and significance of the Visual Similarity Discovery (VSD) task and its differences from traditional retrieval or recognition settings

- Provides comprehensive experiments with multiple pretrained models (CLIP, DINO, BEiT, etc.) and proposes quantitative evaluation using diverse metrics (AUC, BPREF, DCS, EHR@K).

- Introduces practical methods (DCS, EHR@K, SPA) that effectively address the issue of incomplete labeling and improve fine-tuning results.

### Weaknesses
- The main contribution is unclear. It combines dataset extension, metric design, and fine-tuning without a single coherent focus. The motivation for creating a new furniture-domain dataset using the same EDS pipeline is not fully convincing, as it seems an incremental domain extension.

- The logical flow around DCS is confusing. The paper links generator bias (EDS limitation) with AUC’s triplet limitation but does not clearly explain how DCS specifically resolves these issues.

- Presentation clarity is limited. The paper lacks a figure illustrating the full motivation and pipeline, and supervised fine-tuning is not novel—SPA should be emphasized more as the true methodological contribution.

### Questions
- Could the authors include a figure or diagram illustrating the overall motivation and pipeline, showing how dataset construction, metric design, and SPA are connected?

- Beyond supervised fine-tuning, have the authors considered alternative learning approaches more suited for VSD, such as semi-supervised, contrastive, or retrieval-specific adaptation methods?

- The paper mentions using multiple generator models to aggregate top-K retrievals—could the authors clarify how this aggregation is performed (e.g., union, weighted ranking, or score fusion)?

### Soundness
2

### Presentation
2

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
The paper introduces a dataset created by experts on the topic of VSD. VSD (Visual Similarity Discovery) is the task of finding objects that are similar to the query object, but preferably not the exact same object. This is very relevant for the task of product suggestions, which we find in all commerce sites. They highlight the problem of incomplete labeling when training models for these tasks. In addition to the dataset, propose two evaluation metrics designed for VSD in the case of missing labels, and lastly empirical evidence showing that fine-tuning on VSD datasets significantly improves performance.

### Strengths
The authors address a very real and important problem in the VSD literature, namely incomplete labels in a VSD dataset. Furthermore, the authors have performed rigorous testing of the proposed metrics and show that they are "better" when using them on VSD with incomplete labeling.

### Weaknesses
"Using cosine similarity alone to generate candidate pairs systematically biases the dataset toward the geometry of the pretrained embedding space, which may not reflect perceptual or semantic similarity. The Circle-loss paper explicitly shows why cosine similarity is an incomplete measure of true pair similarity. Suggestions on what else to do could be to fine-tune the embedding models using the circle-loss, or similar. Other approaches might be to combine or use an ensemble of similarity metrics. Also, a minor formatting error: overlapping text in figure and paragraph on line 206-207.

\cite{https://arxiv.org/abs/2002.10857}

### Questions
I would like to see a test of the use of the similarity metrics, or at least a discussion as to why you think solely using cosine-similarity is enough.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a study on Visual Similarity Discovery (VSD)—retrieving visually similar but non-identical items—through a new dataset (VSD-Furniture), two evaluation metrics for handling incomplete labeling (Discounted Credit Score and Estimated Hit-Ratio@K), and a fine-tuning strategy called Soft Positive Augmentation (SPA). The authors position VSD as distinct from visual search or duplicate detection, focusing on perceptual similarity beyond exact matches. Experiments on the proposed dataset and fashion benchmarks show consistent gains from supervised fine-tuning and SPA.

### Strengths
The paper is well-written. It identifies a gap between standard visual search and perceptual similarity discovery. The newly proposed metrics try to mitigate some limitations of existing metrics—how incomplete labeling can bias retrieval evaluation—and the empirical validation may be technically right.

### Weaknesses
The main conceptual concern lies in the unclear distinction between visual similarity discovery and conventional visual search. From a technical standpoint, the two tasks share almost identical pipelines, and a standard visual search system can naturally serve as a discovery engine by filtering out identical products. They share a similar goal, i.e., to rank the most visually similar objects higher based on retrievals. 

This weakens the motivation for defining VSD as a separate problem space. The necessity of introducing a new dataset becomes questionable, considering strong benchmarks such as Stanford Online Products and In-Shop Clothes Retrieval–among others–already exist for evaluating visual similarity and retrieval models. The proposed VSD-Furniture dataset is limited to a single product category and modest in scale, which restricts its generalizability and practical impact. 

Furthermore, while the proposed metrics and fine-tuning strategies yield some improvements, the gains appear limited. Standard metrics such as Recall@K can already provide reasonable ranking–use different K values–estimates even under partial labeling, reducing the necessity of introducing new ones. Moreover, the reported benefits from fine-tuning are modest and do not seem to be significant observations.

### Questions
Claiming MAP is “less effective” or should be excluded entirely seems to be a stretch. It could be better if the authors provided stronger evidence, rather than relying on a single reference to justify this decision.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper focuses on Visual Similarity Discovery under scenarios with incomplete labeling and presents four core contributions: (1) constructing the VSD-Furniture dataset, a VSD dataset in the furniture domain; (2) designing two evaluation metrics—Discounted Credit Score (DCS) and Estimated Hit-Ratio at K to adapt to incomplete labeling; (3) verifying that supervised finetuning using VSD labels can improve model performance; and (4) proposing the Soft Positive Augmentation (SPA) method, which mines potential similarity relationships in unlabeled samples via weighted graph transitivity to further enhance finetuning effects. Experiments on the VSD-Furniture and Fashion datasets validated the consistency of the proposed metrics and the effectiveness of the finetuning methods.

### Strengths
1. Visual Similarity Discovery is a scientifically significant problem. It addresses a core demand in practical applications such as e-commerce and visual search.

### Weaknesses
1. **Notation and Formatting Issues:** The manuscript appears to be very hastily written, with numerous formatting and notation flaws. For instance, in the Abstract, there is ambiguity about an extra quotation mark after the phrase "retrieving positives". Additionally, the title of Figure 1 overlaps with the body text, making it difficult to read. These issues hinder the clarity of the manuscript and suggest a lack of careful proofreading. The authors should carefully polish and revise the manuscript to fix these formatting and notation problems.
2. **Outdated Experimental Comparisons:** The methods used in the experiments are relatively outdated, and the paper lacks comparisons with more recent VSD-related methods developed in recent years. 
3. **Weak Theoretical Basis for Metrics:** The proposed metrics are only validated experimentally, with no proof of key mathematical properties. This lack of theoretical guarantees casts doubt on the metrics’ generalizability to other domains.

### Questions
Please refer to the "Weaknesses" section for relevant questions and suggestions.

### Soundness
2

### Presentation
1

### Contribution
2
