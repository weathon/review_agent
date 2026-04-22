# Multi-Source Knowledge-Fusion for Source-Free Domain Adaptation in Object Detection

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 8, 4, 2

## Abstract
Source-free domain adaptation (SFDA) enables adaptation to a target domain without access to source data or labeled target samples, making it particularly valuable in privacy-sensitive applications such as military operations and healthcare. To leverage complementary and transferable knowledge from multiple source domains, multi-source-free domain adaptation (MSFDA) extends SFDA by collectively adapting pre-trained models from multiple sources. However, a key challenge in MSFDA is the significant distribution shift among multiple source and target domains, which often leads to suboptimal performance, especially in complex tasks like object detection. To address this, we propose a novel multi-source knowledge-fusion framework that effectively aggregates knowledge from multiple sources and mitigates distribution discrepancies. We first conduct text-driven feature augmentation that narrows the semantic gap by transforming unlabeled target images into source-stylized images using only textual descriptions of each source domain, such that the pre-trained source models are directly applicable. Each domain expert is then updated with its respective stylized target images, while the aggregator undergoes both local and global updates to ensure stable adaptation. To further improve pseudo-label quality, peer network-based confidence selection is performed to filter out noisy labels. Our method achieves state-of-the-art performance on multiple real-world datasets, demonstrating its effectiveness in multi-source free domain adaptation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates an interesting and important problem, namely multi-source-free domain adaptation for object detection. The authors propose a new framework that effectively aggregates knowledge from multiple sources and mitigates distribution discrepancies across domains. Experimental results are provided to verify the effectiveness of the proposed approach.

### Strengths
-	The paper addresses an interesting and important research topic in domain adaptation, focusing on the multi-source setting for object detection.

-	The proposed framework first generates stylized target datasets and subsequently performs multi-source knowledge fusion. To alleviate knowledge conflicts among different sources, a globally updated model aggregator is introduced, which is conceptually sound.

### Weaknesses
-	The definition of the loss term $\mathcal{L}_{\text{Gram}}$ (L230) is confusing. In addition, the theoretical or intuitive rationale for why it enables texture and appearance similarity alignment is not clearly articulated. 

-	The presentation lacks clarity in several technical sections, which makes it somewhat difficult to follow.

-	The evaluation of the proposed Target Feature Augmentation (TFA) module is not sufficiently comprehensive. A comparison with semantic augmentation techniques such as Clip the Gap [1], which also aim to enhance feature diversity, would strengthen the claims.

**Reference:**

[1] Clip the gap: A single domain generalization approach for object detection. CVPR, 2023.

### Questions
-	How is the aggregator model $\theta^{agg}$ obtained? The description is not sufficiently clear. It appears that a model-merging technique might be employed to fuse source models into an aggregator, but the specific merging strategy is not specified. Moreover, it is unclear how the method handles the potential incompatibility among distinct source models, as Eq. (8) seems to assume that the source models are mergeable.
-	As stated in L281–L304, the paper employs a small meta-network $\mathcal{F}(\cdot)$ to assign $\alpha_i$. However, the input to this meta-network is not explicitly defined and requires clarification.
-	The type of image decoder used in the reconstruction of stylized images is also not mentioned and should be clarified.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a Multi-Source Knowledge Fusion (MSKF) framework for Multi-Source Source-Free Domain Adaptation (MSFDA) in object detection. It leverages CLIP’s vision-language space to stylise unlabelled target images using only textual descriptions of source domains. This bridges semantic and appearance gaps between multiple source models and the target. Then the authors proposed to use one model serves as an aggregator (updated via EMA from domain experts), while each domain expert performs local self-training. A meta-learned contribution network dynamically learns expert weighting based on entropy minimisation, promoting consistency. At last, a co-teaching-like pseudo-label refinement, where the aggregator and domain experts filter each other’s noisy labels. The experiment results shows the improvement of the new proposed solution. The paper is clearly written and well-structured. The technical methodology and equations are sound, and implementation details are clearly stated.

### Strengths
1. TFA is an elegant use of vision-language models for domain adaptation. It effectively bridges domain gaps using text alone, which is a simple and good idea that avoids image-based generators.
2. The aggregator–expert paradigm, with meta-learned EMA weighting, offers a stable and interpretable way to integrate heterogeneous source models.
3. Mutual confidence selection (co-teaching) mitigates pseudo-label noise, a long-standing challenge in SFDA.
4. Detailed comparisons and ablations demonstrate robust gains across datasets, classes, architectures, and hyperparameters.

### Weaknesses
1. Novelty mainly lies in method combination and cross-modal extension, deeper theoretical or analytical insights are limited. The paper lacks a formal analysis of the meta-learning process for the contribution network. Such as Eq. (9) is intuitive but does not clarify optimisation dynamics or stability.
2. The performance of TFA may depend on the quality of textual descriptions. For example, even Table 13 explores variants qualitatively, there is no quantitative correlation between prompt similarity and adaptation success.
3. The qualitative results are uniformly positive. Including examples of failure (e.g., rare lighting or weather conditions) would help illustrate limits.

### Questions
1. A typo on line 357; should it be LPLD rather than LDLP, and same as the following experiment results?
2. Table 2, the max value of AP-Bus, AP-Rider, AP-Truck should be attributed to other algos, not "Ours". That gives a "out-performing rate" of 4/8, rather than 6/8. Am I correct?
3. Page 8: NC DR DF scores I think could have been defined better? Seems like abbreviations are defined in appendix only.
4. Experiment does not seem to specify how many runs or the error bars/variance of results?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies Multi-Source Source-Free Domain Adaptation (MSFDA) for object detection, a more practical but challenging setting where multiple source models are provided, but source data is unavailable. The authors propose a multi-source knowledge fusion framework consisting of text-driven feature augmentation, an aggregator–expert model structure, and mutual confidence selection. Experiments on several cross-domain datasets demonstrate the effectiveness of the proposed method.

### Strengths
1. MSFDA problem in object detection is a very practical task.
2. The authors provide strong empirical performance with extensive comparisons.
3. The authors provide a detailed appendix that makes the conclusions convincing.

### Weaknesses
1. Limited novelty of key components. TFA is a lightweight variation of text-guided CLIP-based augmentations, which reduces the novelty of the proposed method.
2. Writing clarity is insufficient, causing many technical design details to remain unclear. For example, in Sec. 3.1, an image decoder is suddenly introduced, yet no details or introduction are ever provided afterward. In Sec. 3.2, the description of the multi-source knowledge fusion framework suggests that each source model becomes an aggregator trained in parallel. If this is the case, it remains unclear how the final prediction is generated at inference time. Neither the main text nor Algorithm 2 clarifies this.
3. The writing lacks logical coherence. The authors typically present loss functions first and then loosely describe each component without establishing conceptual connections. Sec. 3.1 is particularly difficult to follow; it reads as a collection of independent strategies rather than a unified design.
4. More quantitative evidence should be moved into the main text of the paper (Dataset Art?). Additionally, Table 2 contains misleading formatting: several results from the proposed method are not the best, yet they are still incorrectly bolded.
5. The computational efficiency of the method is not discussed. TFA requires generation per target sample and per source domain, which could be prohibitively slow at scale. Moreover, updating $M$ aggregators in parallel leads to high training cost.

### Questions
1. How sensitive is TFA to the quality of text prompts?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
A method for multi-source-free domain adaptation, including a style translation module and a noisy label filtering module.
The performance achieves state-of-the-art.

### Strengths
- The idea is simple and easy to follow.

- The performance looks good.

### Weaknesses
- The performance of the method is good. However, the applicability of this method is limited. It requires pre-training the feature augmentation module and generating modules, which is not an elegant method.

- The novelty is limited. Whether it is style transfer or noise label filtering, they are common solutions in domain adaptation. There is less insight, and the motivation for using these two strategies is not novel.

- Can this method only handle object detection tasks? This method is a general idea and does not seem to be specially designed for detection tasks. If it is also feasible for other tasks, this method will be more solid. It is only useful for detection tasks. The author needs to analyze some reasons.

- There is a significant improvement in the truck, but a lot of decline in the motor class in Table 2. What is the reason? Some limitations analysis may be required.

### Questions
See weaknesses

### Soundness
2

### Presentation
3

### Contribution
2
