# SCALE-VLP: Soft-Weighted Contrastive Volumetric Vision–Language Pre-training with Spatial-Knowledge Semantics

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 4

## Abstract
Vision–language models (VLMs) have demonstrated strong cross-modal capabilities,
yet most work remains limited to 2D data and assumes binary supervision
(i.e., positive vs. negative pairs), overlooking the continuous and structured dependencies
present in volumetric data such as CT. Existing approaches often treat
volumetric scans as independent 2D slices, compromising spatial coherence and
underutilizing rich clinical semantics. We propose SCALE-VLP, a soft-weighted
contrastive vision-language pre-training framework that integrates (i) volumetric
spatial semantics to preserve anatomical structure and (ii) domain-aware,
knowledge-infused semantics (e.g., radiological ontologies) to guide alignment.
This yields structurally consistent and semantically grounded representations under
limited supervision, demonstrating strong cross-task transferability (retrieval,
report generation, and classification), and cross-domain generalizability with consistent
gains without further fine-tuning. In particular, compared to the previous
state of the art, SCALE-VLP achieves up to 4.3× higher top-1 CT–report retrieval,
improves abnormality classification by 10 points, and reaches ROUGE-L 0.44 and
BERT-F1 0.89 for report generation. Further, in zero-shot evaluation on an outof-
domain external dataset, we observe consistent gains, indicating the cross-task
and cross-domain generalization ability of SCALE-VLP.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces SCALE-VLP, a soft-weighted contrastive vision-language pre-training framework tailored for 3D volumetric medical data, specifically CT scans and paired radiology reports. The method embeds volumetric spatial-coherent semantics and domain-specific clinical knowledge to enhance alignment, leveraging Soft-Weighted Contrastive Alignment (SWCA) objective that replaces binary supervision with dense, semantics-aware similarity matrices. SCALE-VLP is benchmarked for retrieval, classification, and report generation on large CT-report datasets, consistently outperforming existing models.

### Strengths
The paper proposes a single pretraining framework (SCALE-VLP) that supports three clinically relevant downstream tasks — report retrieval, report generation, and abnormality classification — using a shared encoder. 

The method is benchmarked on both in-domain (CT-RATE) and out-of-domain (BIMCV-R) datasets, with detailed quantitative results across  retrieval, classification, and generation metrics. 

The paper conducts ablation on both the spatial and knowledge weighting terms in SWCA.

### Weaknesses
1. The core motivation is confusing. The paper claims that "A central bottleneck lies in learning effective 3D representations that preserve the intrinsic spatial and semantic structure of the data". However this has already been explored in CT-CLIP, Merlin and fVLM which all use 3D backbones to encode volumetric information. As seen in Sec 3.1 model architecture, the authors also use 3D ViT to encode this information. The paper states "Each CT scan is paired with its radiology report, and both modalities are encoded using a 3D Vision Transformer and a medical-language encoder respectively". However, this modeling is not new and does not address an underexplored problem in medical imaging. 

2. The proposed Soft-Weighted Contrastive Alignment (SWCA) objective is confusing to read. It seems like the authors want to propose a method to reweight a sigmoid contrastive objective using intra-modal similarities. How does this improve the "semantics" between CT and reports? 

3. The authors claim SWCA "improves sample efficiency under limited supervision". There is no where in the paper that supports this claim. 

4. The method heavily downsamples the input volume to 256 x 256 x 32. After patch embedding, the tokens along the axial dimension effective becomes 2 (given patch size of 16). How is it possible for the model to learn 3D “spatially coherent” distance? Furthermore, it is rather uncommon to only report the volume resolution without the actual spacing that the volume was resampeld to. 

5. The “Medical knowledge fusion” is also overclaimed. This paper effectively embeds reports with a frozen medical LLM and computes a report-report similarity matrix to weight the loss. However, the image side receives no structured knowledge signal. In practice, this is a textual label-smoothing prior, not “knowledge fusion” between CT and text.

6. There is no analysis of potential bias or risk introduced by using an external LLM for "knowledge fusion". 

7. The qualitative example in Figure 7 suggests a hallucinated finding (a renal calculus); can the authors provide a broader quantitative error analysis on hallucination or missing findings for report generation?

### Questions
Can the authors clarify, with empirical or theoretical evidence, why the spatial and knowledge priors are necessary? 

How would SCALE-VLP perform under diverse domain shifts (e.g., different CT protocols or free-text reports from other healthcare systems)?

What are the training and inference costs for using SCALE-VLP?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
SCALE-VLP introduces volumetric spatial information and medical knowledges into vision-language pretraining through a soft contrastive objective. Specifically, the weights of the soft objective are defined through a volumetric kernel that encodes spatial structure, while medical knowledge is obtained through embeddings of a pretrained Medical VLM. The authors evaluate across multiple downstream tasks such as cross-modal retrieval, report generation, abnormality classification, and zero-shot generalization under domain shift.

### Strengths
- This seems like a novel extension of soft contrastive learning to an important problem of vision language alignment for CT VLMs.
- Performance in cross-modal retrieval and Report generation seems to improve

### Weaknesses
- **Unclear Method description**: 
    - Line 179: Unclear where $z_i$ came from. Is i indexing a volume patch or an entire sample? 
    - Equation 6: What score is computed to get $r_{i,m}$? What are the inputs and outputs of this score?
    - Equation 8: Since the two exponential terms likely differ in scale, their direct multiplication may be dominated by one component. How do the authors determine $\kappa_\mu$ and $\kappa)_\Sigma$ hyperparameters to prevent one component from dominating? Shouldn’t these parameters be set dynamically during training as $\mu$ and $\Sigma$ get updated?
    - Can the authors clarify what “pool size” (line 312) is referring to? What dimension are we pooling over? And why does increasing it lead to a decrease in performance in table 1? How would a practitioner choose this hyperparameter?
- **Clarity could be significantly improved**: 
    - In line 58-60, the authors argue that the problem with fVLM is the use of discrete masks, but do not indicate why this is a limitation. Can the authors clarify what causes discrete masks a poor modeling choice? And furthermore clarify what type of “continuous structure” in volumetric data is available to exploit? What happens to the learned representation when using discrete mask during contrastive pretraining and why is this unfavorable?
    - Can the authors also point to specific examples of what “fine-grained spatial details” is referring to (line 56)?
    - Line 134: The authors claim: “current frameworks fall short in considering spatially grounded and clinically guided semantic alignment”. I’m not sure I understand where this claim is referring to. Can you strengthen this argument and point to specific shortcomings? What is the algorithmic limitation that causes these models to fall short? 
    - Line 314-315: The authors claim “SCALE-VLP surpasses it, suggesting that our spatial– and knowledge–aware alignment effectively capture fine-grained clinical semantics.” However, it is unclear to me the connection between improved performance in cross-modal retrieval and the impact of spatial/knowledge-aware alignment on fine-grained clinical semantics. Can the authors state this connection more clearly? Since the knowledge-alignment depends on a global embedding of the entire report, I’m not sure how this can possibly lead to fine-grained semantics.
- The current ablation study is missing "SCALE-VLP w/o knowledge". Would it be possible to include this ablation so that we can understand the individual contribution of the spatial alignment? 
- In the experiments, I dont understand what is the learning objective is for “SCALE-VLP w/o spatial and knowledge”. If the full pretraining objective is defined by equation 11, what is left in the learning objective after removing both spatial and knowledge terms?
- In table 3: Its not clear to me that ScaleVLP does better in abnormality classification. Specifically, it seems like fVLM has stronger cluster-average accuracy. Can the authors clarify why a practitioner would prefer ScaleVLP over an existing method in abnormality classification?

Minor
- In conclusion: “iii)” and “ii)” seem to come out of order
- citations should be \citep instead of \cite when used as references. Otherwise, the citations read as part of the sentence.

### Questions
Please refer to weaknesses for the majority of the questions.
- How sensitive is the method to the Medical-Knowledge VLM? Can the authors test on multiple Medical VLMs to show that the method generalizes beyond a single backbone? 
- Why is the 3d vision encoder frozen? Why wouldn't it be more effective to jointly optimize the learned features end-to-end? If its about limited compute requirements, why not use LORA for the encoder?

### Soundness
2

### Presentation
2

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
This paper introduces SCALE-VLP, a volumetric vision–language pre-training framework for CT–report alignment. The authors propose a Soft-Weighted Contrastive Alignment (SWCA) loss that models continuous (rather than binary) image–text similarities, and integrate two additional weighting schemes: (i) a spatially coherent semantic alignment derived from 3D geometry, and (ii) a knowledge-infused semantic alignment based on medical large language models (LLMs). The method aims to improve sample efficiency and enable robust cross-task and cross-domain transfer. Experiments on CT-RATE and BIMCV-R datasets show strong performance on retrieval, report generation, and classification.

### Strengths
- The paper addresses an important and underexplored problem — volumetric (3D) vision–language pre-training for medical imaging.
- The idea of soft-weighted alignment for partial image–report correspondence is reasonable and clearly explained.
- Empirical results are extensive, covering multiple tasks (retrieval, report generation, classification) and datasets (CT-RATE, BIMCV-R).
- The writing is clear, and the overall system is well-structured.

### Weaknesses
The proposed Soft-Weighted Contrastive Alignment is a modest variant of prior sigmoid-based contrastive objectives (e.g., SigLIP, CWCL), extended with heuristic spatial and knowledge-based weighting. These additions are conceptually straightforward and lack clear theoretical justification or novel insight into vision–language alignment.

The methodological design is largely empirical, with multiple hyperparameters (κµ, κΣ, α) introduced without sensitivity or ablation analysis. The “knowledge-infused” component is loosely motivated and functions mainly as a text-similarity reweighting step, without evidence of genuine knowledge transfer or clinical reasoning benefit.

The experimental evaluation, though extensive, remains narrow in scope and rigor. All tests are CT-based, limiting the claimed cross-domain generalization. Reported gains are modest and lack statistical validation. Qualitative results are shallow, with no expert verification to support the claimed clinical plausibility.

Overall, the work is a careful system extension rather than a substantive methodological advance. While performance improvements are consistent, the contribution is incremental and lacks the conceptual depth expected for ICLR.

### Questions
Please refer to the Weaknesses section.

**I am willing to raise my score according to the rebuttal.**

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces SCALE-VLP, a dual-encoder framework for 3D CT–report pretraining using a new Soft-Weighted Contrastive Alignment (SWCA) loss. Instead of binary matches, SWCA uses continuous weights that blend spatial coherence in 3D scans (via centroid/covariance kernels) with text similarity from a medical LLM. The setup freezes a RadImageNet 3D ViT and fine-tunes a BioClinicalBERT text encoder, updating only lightweight projection layers. On the CT-RATE benchmark, the model improves retrieval (e.g., +13% R@1), report generation (BLEU-4 0.35, ROUGE-L 0.44), and abnormality classification, with additional zero-shot gains on BIMCV-R. The paper also argues SWCA is more efficient than softmax-based InfoNCE and includes solid ablations for its main components.

### Strengths
1. The paper tackles a practical and important issue of how to learn good representations from 3D volumetric scans like CT; this is a step up from the 2D-based work . The authors correctly identify the main issues in this setting: data scarcity , preserving spatial coherence, and the difficulty of integrating complex medical knowledge. The proposed method does address these issues. 
2. Consistent multi-task improvements. On CT-RATE: retrieval improves across different pool sizes. The generated reports and classification results seem to improve on all reported metrics. Zero-shot BIMCV-R results are also favorable. The proposed method also shows good generalization. 
3. This paper is well-written and generally easy to follow.

### Weaknesses
1. Contradiction in the ablation study.  In sec 4.1.1, the authors summarize that "spatial reasoning is the dominant contributor," while removing spatial reasoning results in 2-6% drop but removing both spatial and knowledge results in 14-18% drop. This suggests that knowledge, instead of spatial, accounts for the larger share of the total gain. To the very least the claim that spatial reasoning is the "dominant" is hardly verified. This is evident both in this paragraph and in tab 1 and tab 2. 
2. The authors should clarify their main contribution difference vs. prior weighted contrastive losses. The proposed SWCA is close in spirit to SigLIP (pairwise sigmoid) and continuously weighted objectives like CWCL; here weights are built from intra-modal similarity and external priors. The authors should consider a stronger baseline comparison that uses SigLIP/CWCL-style weighting without spatial/knowledge. 
3. Although many n-gram related metrics are used for evaluation, this paper does not consider assessment from actual radiologists (human expert), especially for verifying the real-world usefulness of this type of clinically-targeted solutions. The authors should consider doing such eval on a small subset at least.

### Questions
Please see weakness.

### Soundness
1

### Presentation
3

### Contribution
2
