# Video Summarization Pretraining with Self-Discovery of Informative Frames

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
The rapid proliferation of videos makes automated video summarization (VS) an essential research problem: 
"Which abridged video best conveys the whole story?" The limited size of datasets is known to constrain the generalization of advanced VS methods, requiring advanced pretraining techniques to capitalize on unlabeled videos. Several pretraining methods for VS have been proposed. Yet, they heavily rely on fixed pseudo-summaries, often fail to capture the diverse frame importance, resulting in narrow generalization. To resolve conflicts between pseudo-summaries and downstream tasks, our idea is: First, pretraining should enable the summarizer to learn how to distinguish more meaningful summaries from unlabeled videos, without perspective differentiation; In this way, finetuning only requires adapting the pretrained multifaceted importance to the downstream perspective, facilitating supervised learning.
Our pretraining approach, named ViSP, is free of pseudo-summaries, expecting to better align with the ill-posed nature of defining keyframes. The pre-trained model can be fine-tuned to create the SOTA summarizers by leveraging the knowledge base behind frame saliency. ViSP is conceptually simple and empirically powerful, and it can be used to pre-train any neural video summarizer. Extensive experiments on two benchmark datasets (SumMe and TVSum) demonstrate the superiority of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper addresses the limitation of existing video summarization (VS) pretraining methods—their heavy reliance on fixed pseudo-summaries that enforce a single perspective, failing to capture the subjective, diverse nature of valid summaries and hindering generalization. To resolve this, this paper proposes ViSP, a pseudo-summary-free pretraining framework that enables VS models to learn multifaceted frame importance from unlabeled videos. This approeach uses mutual information (MI) to measure the information overlap between sampled summaries and the original video, leverages contrastive learning with InfoNCE loss to approximate MI maximization, and adopts concrete-relaxation reparameterization to handle discrete summary sampling for gradient flow, supplemented by regularization terms for compactness and a binarization penalty to avoid trivial solutions

### Strengths
ViSP effectively targets a well-documented gap in VS pretraining by eliminating fixed pseudo-summaries, aligning its design with the ill-posed, subjective nature of VS and addressing the generalization conflict between prior pretraining methods and downstream tasks

The framework is conceptually simple and modular, with its core components decoupled from the base summarizer’s architecture, making it easy to integrate with existing models.

This paper provide a detailed ablation studies to validate the efficacy of key components.

### Weaknesses
Despite claiming ViSP works for “any neural video summarizer,” the paper exclusively uses CNN-based model for experiments. No tests are conducted on other SOTA architectures, such as Transformer-based models or multimodal (e.g., A2Summ). This raises doubts about ViSP’s compatibility with models that rely on sequential memory  or cross-modal cues. Additionally, frame features are extracted using a frozen GoogleNet, while modern VS models often use task-specific encoders

Experiments are confined to two small, legacy datasets (SumMe: 25 videos, TVSum: 50 videos) that lack diversity. Other newer, larger datasets (e.g., Daliy, CNN) are ignored, limiting ViSP’s validity for practical applications.

To validate ViSP’s advantage over pseudo-summary-based methods, the paper constructs “high-quality pseudo-summaries” using ground truth labels (e.g., top 15% frames from TVSum/SumMe) for ablation. This is unrealistic for real-world pretraining, where ground truth is unavailable.

### Questions
Can authors test ViSP on non-CNN architectures (e.g., Transformer-based models, GNN-based models) to validate its claim of working for “any neural video summarizer”

It would be interesting to test ViSP on newer/larger datasets to improve its practical validity beyond SumMe/TVSum?

Will you supplement F1 score (or other traditional metrics) to enable direct comparisons with legacy VS baselines?

Please add more case studies or failure cases analysis to better demonstrate ViSP’s efficacy across diverse video scenarios?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The authors propose ViSP, a pretraining method that avoids pseudo-summaries and teaches models to identify meaningful summaries from unlabeled videos. This approach allows fine-tuning to adapt to specific downstream tasks, improving supervised learning. ViSP is simple, versatile, and empirically effective, achieving state-of-the-art performance on benchmark datasets SumMe and TVSum.

### Strengths
1. The authors propose a novel pretraining framework designed to learn versatile frame importance by examining diverse summaries within each unlabeled video. 
2. They develop the ViSP foundation summarizer, which distinguishes and explores more representative summaries using mutual information estimation and learning-based sampling. 
3. Their results show that ViSP effectively enhances the performance of state-of-the-art video summarizers.

### Weaknesses
1. The work does not provide sufficient evidence of generalization. Experiments are conducted only on a single model (CTSA), leaving it unclear whether the proposed method would yield consistent gains when applied to other summarization architectures.
2. It is not clearly justified how the proposed framework enables the model to capture multifaceted frame importance, given that the model is still fine-tuned using ground-truth summaries.
3. The method adopts CTSA as the video encoder during pretraining. To obtain more comprehensive video and summary representations, more advanced video encoders should be considered instead of reusing the same encoder as the downstream summarizer.
4. The authors claim that the framework learns “versatile” frame importance, but the manuscript does not provide convincing evidence for this claim, nor does it clearly articulate the benefit of such versatility.
5. The evaluation relies on small and outdated datasets. Validation on larger and more recent benchmarks is necessary to establish the method’s effectiveness.

### Questions
Please refer to the Weaknesses section.

### Soundness
3

### Presentation
2

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
- This paper introduces a representation learning framework for video summarization that enhances the quality of summary clip features by maximizing their mutual information with the original video features. The approach aims to capture diverse frame importance without relying on pseudo-summaries produced by existing pretrained models. Specifically, the method employs an InfoNCE loss to maximize a lower bound on the mutual information between the sampled summary and the source video, treating them as a positive pair. To enable differentiable frame selection, the approach leverages reparameterization techniques such as Gumbel-Softmax and Concrete relaxation, allowing for a softened binary mask. The model further regularizes summary length and temporal smoothness during training to control the budget and mitigate issues of sparsity or fragmentation. Evaluation focuses on rank-based metrics (Kendall/Spearman), using a fixed summary pipeline (KTS segmentation and knapsack under a 15% budget). Experiments on SumMe and TVSum demonstrate consistent improvements over strong baselines, highlighting the framework’s practical effectiveness for learning frame importance without reliance on fixed pseudo-summaries.

### Strengths
- The paper is clearly written and technically accessible, with a transparent connection between the method and its implementation, particularly in how frame scores and video-level representations are incorporated into the contrastive objective.
- The evaluation protocol is well-justified for the task, employing rank-based metrics and a fixed summary construction pipeline, which enhances the comparability of results.
- The approach is practical, which adopts differentiable selection via Gumbel-Softmax/Concrete relaxation, combined with lightweight deployment since the pretraining encoder can be omitted at inference, which facilitates straightforward integration with existing video summarization pipelines.

### Weaknesses
- The paper argues that sampling diverse summaries during pretraining is beneficial. However, the empirical evidence primarily compares "diverse sampling" to "fixed summaries," without directly analyzing between-sample diversity for the same video. A more convincing analysis would include metrics such as the distribution of pairwise IoU/Jaccard indices across multiple independently sampled summaries for each video, or the entropy/effective support size of the per-frame selection probabilities.
- While ranking correlation metrics capture the quality of frame importance ordering, they do not necessarily reflect the quality of final summaries produced after KTS and knapsack selection. The robustness of the reported gains to alternative segmentation strategies (e.g., uniform segments or VSUMM) or to omitting the knapsack step is not demonstrated, leaving a gap between improvements in ranking and actual end-to-end summary quality.
- The “unseen” evaluation setting, which leverages partial in-domain data, is reasonable; however, the process for sampling and de-duplication against test videos, including near-duplicate filtering, should be described more rigorously to ensure the integrity of the evaluation.

### Questions
- For each video, could you sample k≥5 summaries and report the distribution of pairwise IoU, along with the entropy or effective support size of the per-frame selection probabilities? This would more directly support the claim of capturing “diverse perspectives” beyond coverage with a single reference.
- If sweeping the temperature or regularization parameter to modulate diversity, do you observe a monotonic relationship between the chosen diversity metric and Kendall/Spearman correlation? Even a limited sweep could help clarify causality.
- Do results hold when using a different shot segmenter or omitting the knapsack step? Reporting rank-biased overlap (RBO) or basic coverage/diversity metrics on the final summaries would better bridge the gap between improved rankings and true end-to-end summary quality.
- How are in-domain videos sampled for the “unseen” pretraining pool, and what steps are taken to prevent leakage of test videos or near-duplicates into pretraining? For out-of-domain scaling, a brief analysis of domain distance (e.g., using CLIP-space statistics) versus performance gains would further strengthen the conclusions.

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
4

### Summary
The paper proposes an annotation-free pretraining framework for video summarization. It evaluates the frame importance without relying on pseudo summaries. It maximizes the mutual information between the original video and its sampled summary through contrastive learning, and uses reparameterization sampling for differentiable frame selection. Authors conduct some interesting experiments on SumMe and TVSum to demonstrate the performance improvements.

### Strengths
1. It introduces a well-motivated pseudo-summary-free pretraining framework that learns informative frame representations by maximizing mutual information. 
2. The theoretical foundation is clearly established through the use of a contrastive lower bound on mutual information.
3. Experimental results are mostly promising by consistent gains over baseline methods.

### Weaknesses
1. Some of the mathematical formulations, particularly the sampling and optimization sections, are presented too tersely. The role of auxiliary variable Z is not rigorously derived.

2. Experimental evaluation is limited to relatively small datasets (SumMe and TVSum), and does not test the model on larger data set, such as VTW (2529 videos) or multimodal benchmarks. This restriction limits confidence in generalization capability to more complex or real-world scenarios.

### Questions
1. For the auxiliary variable Z and Eq. (9): under what data distributions does marginalization with Z lead to a better approximation? Can its influence on variance estimation and gradient stability be quantified, for example by providing an error bound or empirical comparison?

2. Regarding model generality: beyond CSTA, how does ViSP perform when transferred to other summarization architectures? Does it still yield statistically significant improvements across architectures?

3. For cross-domain and long videos (over 30 minutes) or high-motion scenes (e.g., sports or documentaries), how does the method behave and where does it fail? Would hierarchical sampling or temporal segmentation be necessary to stabilize the InfoNCE objective?

4. In a multimodal setting, if audio or subtitle streams are introduced as additional modalities, how should ViSP’s mutual information estimation be reformulated—would a multi-view InfoNCE objective be appropriate to mitigate single-view bias? This question should be considered in the future work.

### Soundness
2

### Presentation
3

### Contribution
2
