# Imagine How To Change: Explicit Procedure Modeling for Change Captioning

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Change captioning generates descriptions that explicitly describe the differences between two visually similar images. Existing methods operate on static image pairs, thus ignoring the rich temporal dynamics of the change procedure, which is the key to understand not only what has changed but also how it occurs. We introduce ProCap, a novel framework that reformulates change modeling from static image comparison to dynamic procedure modeling.
ProCap features a two-stage design: The first stage trains a procedure encoder to learn the change procedure from a sparse set of keyframes. 
These keyframes are obtained by automatically generating intermediate frames to make the implicit procedural dynamics explicit and then sampling them to mitigate redundancy.
Then the encoder learns to capture the latent dynamics of these keyframes via a caption-conditioned, masked reconstruction task.
The second stage integrates this trained encoder within an encoder-decoder model for captioning.
Instead of relying on explicit frames from the previous stage---a process incurring computational overhead and sensitivity to visual noise---we introduce learnable procedure queries to prompt the encoder for inferring the latent procedure representation, which the decoder then translates into text. The entire model is then trained end-to-end with a captioning loss, ensuring the encoder's output is both temporally coherent and captioning-aligned. Experiments on three datasets demonstrate the effectiveness of ProCap. Code and pre-trained models are available at https://github.com/BlueberryOreo/ProCap.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper presents a novel perspective on the task of Change Captioning. Instead of merely focusing on static image pairs, it emphasizes modeling the changing process between similar image pairs. This process captures rich spatial-temporal information, enabling the model to better understand how changes occur and to generate more precise descriptions of them. Extensive experiments demonstrate that, even without a complex framework design, the proposed method effectively leverages the learning of dynamic change processes to overcome challenges caused by distractor variations, achieving strong and robust performance.

### Strengths
1. The paper provides a fresh and meaningful perspective on the change captioning task by modeling the dynamic process of change instead of focusing on static image pairs, which makes the approach both novel and insightful. 
2. The proposed framework, ProCap, demonstrates an elegant design that effectively captures rich spatial–temporal information by explicitly modeling the changing process. Moreover, its use of implicit procedure queries enables efficient and effective inference. Experiments show that ProCap surpasses existing non-LLM-based methods on two datasets and achieves competitive results on the remaining one.

### Weaknesses
1.As shown in Table 1, the method performs slightly worse on Spot-the-Diff; further analysis would be informative.
2.Presenting some failure cases would provide deeper insights into its limitations.

### Questions
Please refer to weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper addresses the task of Change Captioning, which aims to generate textual descriptions of the differences between two similar images. The authors argue that existing methods, which rely on static image pair comparison, fundamentally fail to model the rich temporal dynamics of how a change occurs. To address this limitation, the paper proposes ProCap, a novel two-stage framework that reformulates the task from static comparison to dynamic procedure modeling. 
Stage 1: Explicit Procedure Modeling: This stage trains a "procedure encoder" to capture the latent dynamics of a change.
Stage 2: Implicit Procedure Captioning: This stage handles the final captioning task.

### Strengths
1.	Novel Problem Formulation: The paper introduces a conceptual shift by reformulating change captioning from a static image comparison task into a dynamic procedure modeling problem. This directly addresses a key limitation of prior work, which largely ignores the rich temporal dynamics of how a change unfolds.
2.	Efficient Architecture: 
(1)	Stage 1 effectively learns a rich representation of spatio-temporal dynamics by training an encoder on explicitly generated and sampled keyframes.
(2)	Stage 2 introduces learnable procedure queries as an efficient solution. This allows the model to implicitly reason about the change process during inference, bypassing the computational cost and noise associated with synthesizing frames.

### Weaknesses
1.	It does not show a clear performance advantage on Spot-the-Diff over other SOTA methods.
2.	As shown in Table 4, the consistency loss does not yield a substantial performance gain.

### Questions
1.	When there are significant scene differences between the two images, how does the procedure modeling module generate intermediate frames that are meaningful for the change analysis?
2.	The ablation studies were performed exclusively on the CLEVR-Change dataset, which is insufficient for evaluating the proposed method's effectiveness in real-world scenarios (such as Spot-the-Diff).

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
In this paper, the authors focused on the task of change captioning. Considering the existing methods ignore the temporal dynamics of the change procedure, the authors proposed the ProCap framework. Specifically, ProCap explicitly models the spatio-temporal changes between image pairs. Based on the first stage, the second stage can generate rich descriptions by implicitly reasoning over the modeled change procedure.

### Strengths
1. The designed framework reformulates change captioning from static comparison to dynamic procedure modeling, which can capture the rich temporal dynamics.
2. The proposed explicit procedure modeling module can produce continuous frames between static image pairs, which facilitates the change captioning.

### Weaknesses
1. The method’s performance across the three datasets does not consistently outperform baseline methods, indicating that its overall robustness may not be sufficiently strong.
2. ProCap employs a non-LLM-based backbone, and it remains unclear how its procedure modeling module would still perform when integrated with a powerful LLM-based backbone.

### Questions
As listed above.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes ProCap, a two-stage framework that reformulates change captioning from static image comparison to dynamic procedure modeling. Stage 1 uses a pre-trained Frame Interpolation model to generate intermediate frames, samples informative keyframes via confidence scoring, and trains a Procedure Encoder through caption-conditioned masked reconstruction with three losses (reconstruction, alignment, consistency). Stage 2 replaces explicit frames with learnable procedure queries for efficient inference, fine-tuning the encoder end-to-end with a text decoder. The method achieves competitive results on three datasets, claiming that modeling "how" changes occur improves understanding beyond just comparing "what" changed.

### Strengths
1. Novel Paradigm and Strong Motivation: The paper presents a fresh perspective by shifting from static image comparison to dynamic procedure modeling, addressing a limitation in existing methods that ignore temporal dynamics. 

2. Two-Stage Design Decoupling Learning and Inference: The framework separates explicit procedure learning (Stage 1) from implicit inference (Stage 2). Using learnable queries instead of generating frames at test time is technically sound.

3. Comprehensive Experimental Validation: The paper provides thorough evaluation across three diverse datasets, extensive ablations, rich visualizations (Figures 6-10), and detailed analysis of individual components. The experiments demonstrate consistent improvements over non-LLM baselines and competitive performance against LLM-based methods while being more efficient.

### Weaknesses
1.	This paper models the change procedure to improve captioning, but (132-137) the mapping γ_T: [0,1] → I is inherently non-bijective with an exponentially large solution space. For any given (I_bef, I_aft) pair, infinitely many valid procedures exist, yet the method relies on a generated sequence from FI without justifying why this particular realization should be canonical or optimal. This makes procedure modeling more difficult than directly describing the change—a small model must navigate a continuous, high-dimensional function space rather than a discrete caption space. Additionally, the confidence-based sampling assumes frames at the "semantic midpoint" (Eq. 2) are most informative, but no experiments validate this hypothesis. 
2.	Table 2's ablation shows that adding implicit procedure queries alone (row 2, k=1) yields almost no improvement over the baseline (+2.2). The gain only appears in row 3 after incorporating the full Stage 1 pre-training. This demonstrates that the learnable queries contribute negligibly. The actual performance boost comes from the expensive Stage 1 pre-training itself, not from the "procedure modeling" concept. 
3.	Stage 1 trains the encoder to reconstruct sampled frames, yet Stage 2 replaces them with learnable queries. If the frames are good training targets, why not use them directly for captioning as a simpler end-to-end approach? Additionally, the multi-granularity masking rationale (lines 236-237) is vague, and the specific schemes/probabilities (0.1, 0.7, 0.1, 0.1) appear without ablation justification.
4.	The model design is relatively complex, consisting of two training stages, with the first stage further divided into three submodules. Compared to other non-LLM-based models, the overall performance improvement is not significant, and some metrics (M, R) are relatively low.

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
