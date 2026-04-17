# Disentangling Multimodal Knowledge Preservation and Editing via Low-rank Adaptation

- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Knowledge editing facilitates precise and targeted updates in Large Language Models (LLMs) and Multimodal Models (LMMs) without the need for full retraining. Although existing editing methods achieve remarkable performance in textual modality, they still struggle to simultaneously preserve pre-trained knowledge and generalize new knowledge in intricate multimodal contexts. To address this challenge, we propose ELoRA, a novel solution that disentangles the conflicting editing objectives. Specifically, ELoRA decomposes the standard Low-Rank Adaptation (LoRA) update into two complementary subspaces: (1) a null space aligned with preserved knowledge, constructed via multimodal initialization to maintain the model's general capabilities, and (2) a knowledge space extracted from the model's internal states to multimodal perturbations, capturing invariant semantics of updates. Extensive experiments on various LMMs, including LLaVA-v1.5-7B, Qwen2.5-VL-7B, and Phi-4-multimodal, show that ELoRA outperforms most LoRA-based methods by an average of 14.2% accuracy across three metrics: reliability, generality, and locality under rigorous LLM-as-a-Judge evaluation, which demonstrates that ELoRA can achieve superior real-world editing quality.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes ELoRA, a LoRA-based framework that decomposes the parameter update space into a null space for preserving existing knowledge and a knowledge space for injecting new information, aiming to balance locality and generality in multimodal editing.

### Strengths
The paper is clearly written and well-organized, with fluent language that make the presentation easy to follow.

### Weaknesses
1.	The authors claim that directly leveraging LoRA for multimodal KE faces challenges, including low locality and low generality. However, these two challenges do not appear to be specific to multimodal KE, and the paper does not clearly differentiate them from those already existing in unimodal KE.
2.	The title emphasizes mitigating the challenges of multimodal knowledge editing, yet the proposed method does not seem to provide further insights or methodological innovations from a multimodal perspective.
3.	The proposed null space idea is highly related to AlphaEdit [1], and the initialization strategy also resembles that of CorDA [2], which substantially reduces the method’s novelty. The authors should clarify the differences and additional contributions compared to these two methods.
4.	The authors introduce Gaussian noise and other perturbations to construct the knowledge space, but the robustness of this approach is questionable. Could such perturbations risk losing important information or even induce additional hallucinations?
5.	In Fig. 1 (a), the colors of the different lines are too similar, making them difficult to distinguish.

[1] AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models

[2] CorDA: Context-Oriented Decomposition Adaptation of Large Language Models for Task-Aware Parameter-Efficient Fine-tuning

### Questions
1.	As noted in the weaknesses, could the authors elaborate on the differences and unique contributions of their method compared with AlphaEdit, LoRA-Null, and CorDA?
2.	Why does AlphaEdit perform significantly worse on the MMKE-Bench dataset? Could the authors provide an analysis or explanation for this phenomenon?
3.	Since the paper emphasizes multimodal knowledge editing, could the authors clarify which parts of the proposed method are explicitly designed to handle multimodal data or interactions?

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
This paper proposes ELoRA, which achieves precise and targeted updates for multimodal models. The method projects the A matrix in LoRA to a null space aligned with preserved knowledge, and projects the B matrix to a knowledge space extracted from perturbed internal states, in order to maintain pre-trained knowledge and generalize new knowledge during editing.

### Strengths
- The paper proposes a new editing method ELoRA for multimodal models, addressing the challenge of simultaneously preserving pre-trained knowledge and generalizing new knowledge during editing.
- The authors considered the problem that using a single edit sample can easily cause overfitting, and solved this issue through data augmentation by adding Gaussian noise to the samples.
- The authors effectively combined the advantages of editing methods and LoRA methods.

### Weaknesses
- ELoRA successfully transferred the application of null space from editing methods to LoRA, but I think this still seems like a combination of two methods, with somewhat limited innovation.
- The paper lacks introduction to some experimental settings, such as what r is used in ELoRA and which module in the 7th layer is the object being edited.
- The paper lacks discussion on the editing targets. As a LoRA-like method, ELoRA should be applicable to every module in the model (FFN or Attention), but the paper does not discuss whether there would be differences when various modules serve as editing targets.

### Questions
- What r is used in ELoRA?
- Which module is the object being edited?

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
This paper introduces ELoRA, a novel knowledge editing framework that enables precise and generalizable updates. The method addresses a key challenge that existing approaches struggle to balance knowledge preservation and new knowledge generalization in multimodal contexts. Build upon LoRA, ELoRA decomposes LoRA updates into two subspaces: (1) a null space for preserving knowledge, and (2) a knowledge space for promoting generalizable updates. ELoRA achieves both high locality and high generality with this subspace decomposition. Experiments demonstrate consistent gains over standard LoRA-based methods across reliability, generality, and locality. Overall, this paper makes a technically sound and empirically validated contribution by providing a framework that balances knowledge preservation and semantic generalization in multimodal knowledge editing.

### Strengths
+ The combination of subspace decomposition and LoRA fine-tuning is creative in multimodal knowledge editing. 
+ The theoretical analysis of null space and knowledge space constructions are well-motivated, supported by clear mathematical formulations. The experiments are comprehensive, covering both triplet-structured knowledge and free-form real-world multimodal knowledge. The performance gains are consistent across multiple models.
+ The paper is well-organized and clearly written. The figures are well-designed.
+ The work is impactful for both the knowledge editing and parameter-efficient fine-tuning. It offers a general framework to update multimodal models.

### Weaknesses
+ While the authors claim that the target knowledge is encoded within a low-dimensional, invariant manifold in the model's internal representations, this assumption appears to be empirically motivated rather than theoretically or experimentally substantiated. This hypothesis is conceptually appealing but lacks rigorous analytical or empirical verification.
+ The paper acknowledges that constructing the projection matrix for the knowledge space requires approximately 45 seconds per edit when using 511 perturbed samples, which is a nontrivial cost for real-time or batch editing applications. Although the authors claim that this cost can be linearly reduced by decreasing the number of perturbed samples, the paper does not analyze how the number of perturbed samples affects editing performance, making the efficiency-performance trade-off unclear.
+ The authors highlight the necessity of applying separate perturbations for visual and textual modalities; however, the paper does not clearly specify how the covariance matrix from the two modalities are fused or weighted when constructing the knowledge space. Is the overall covariance matrix obtained through direct concatenation or another aggregation strategy? If so, how does the method ensure that the fusion process balances different modalities to prevent one modality from dominating the overall subspace representation?
+ The paper does not clarify how the text-image pairs used to construct the null space are selected. Since the null space is intended to represent preserved knowledge, the sampling strategy could affect its coverage. Without specifying selection criteria, or sensitivity analysis, it is difficult to assess the stability of the null space.
+ The paper reports locality scores but lacks an analysis of the edited models' downstream task performance, which would help quantify the broader impact of editing on the models' general capabilities.

### Questions
Please see the above Weaknesses.

### Soundness
3

### Presentation
4

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
In this paper, the authors propose ELoRA, a novel solution that disentangles the conflicting editing objectives. Specifically, ELoRA decomposes the standard Low-Rank Adaptation (LoRA) update into two complementary subspaces: (1) a null space aligned with preserved knowledge, constructed via multimodal initialization to maintain the model’s general capabilities, and (2) a knowledge space extracted from the model’s internal states to multimodal perturbations, capturing invariant semantics of updates.

### Strengths
1. The authors propose ELoRA, a novel solution that disentangles the conflicting editing objectives. Specifically, ELoRA decomposes the standard Low-Rank Adaptation (LoRA) update into two complementary subspaces: (1) a null space aligned with preserved knowledge, constructed via multimodal initialization to maintain the model’s general capabilities, and (2) a knowledge space extracted from the model’s internal states to multimodal perturbations, capturing invariant semantics of updates.

2.  The authors conduct extensive experiments on various LMMs, including LLaVAv1.5-7B, Qwen2.5-VL-7B, and Phi-4-multimodal, show that ELoRA outperforms most LoRA-based methods by an average of 14.2% accuracy across three metrics:

### Weaknesses
1. The construction of the knowledge space is contingent on several key hyperparameters: the window length for visual masking, the noise variance (σ) for text, the similarity threshold, and the number of perturbed samples (B). Figure 3 and 4a show that performance is sensitive to these choices. However, the process for selecting the final values used in the main experiments (e.g., why a specific θ was chosen) is not adequately justified. A more systematic analysis or a principled strategy for setting these parameters is needed to ensure reproducibility and robustness.

2.​​ The perturbation strategies seem somewhat arbitrary. The paper would benefit from a deeper discussion on whythese specific perturbations are suitable for extracting "invariant semantics." Are there more principled, modality-specific augmentation techniques that could be more effective?

3. Some widely-used baselines, such as ROME、MEMIT, should be compared

### Questions
NA

### Soundness
2

### Presentation
3

### Contribution
2
