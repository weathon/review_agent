# Semantic Data Inflation: Adaptive Augmentation for Contrastive Representation Learning

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Self-supervised representation learning requires semantically meaningful data augmentations to learn effective features. However, current augmentation strategies either disrupt semantic structures or risk semantic drift. We present Semantic Data Inflation (SDI), a novel framework inspired by the human visual system that leverages explicit semantic guidance from pre-trained models to enhance representation quality. SDI extracts multi-level semantic cues to create consistent augmented views while maintaining critical object identities. Our multi-scale adaptive mechanism dynamically selects optimal semantic extraction strategies based on image characteristics, ensuring robust performance across diverse conditions. Extensive experiments demonstrate that SDI consistently outperforms baseline and generative methods across multiple contrastive learning frameworks. Crucially, we validate the scalability of our approach on ImageNet-1k, demonstrating significant gains over standard baselines. On ImageNette, our approach reaches 95.75\% linear evaluation accuracy, surpassing standard (+3.88\%) and generative (+3.65\%) methods. Further analysis confirms SDI produces more discriminative features with improved semantic consistency. Our code is available at https://anonymous.4open.science/r/Semantic-Data-Inflation-8D7D.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a new approach to data augmentation within the contrastive learning framework. Unlike the widely used methods that rely on predefined transformations or generative models, the proposed approach employs deterministic models such as YOLO and SAM for detection and segmentation. This design enables data-adaptive transformations and produces more reliable outputs compared to conventional generative approaches.

### Strengths
The main strength of this paper lies in its originality and balance between flexibility and reliability. If no similar work has been previously published, the idea itself is quite innovative and promising. The proposed approach takes advantage of task-specific deterministic models to guide data augmentation, which could indeed offer a meaningful alternative to generative methods. In addition, the numerical experiments show consistent improvements in both performance and auxiliary metrics, such as running efficiency and out-of-domain generalization. These results support the potential of the method in practical applications.

### Weaknesses
However, I find two important weaknesses in the paper. The first concerns the theoretical analysis in Section 4.2 and the additional proofs presented in the appendix. This part is unclear and lacks rigor. It is not evident what the main theoretical claim is or why such analysis is necessary. For example, the claim that a larger mutual information value results from the assumption “semantic-guided augmentation preserves more semantic content” (line 723) is problematic. The notion of “semantic content” is not formally defined, making the argument vague and not mathematically rigorous, as the authors suggest in line 215. As a result, this section adds little theoretical value and even weakens the paper’s logical foundation.

The second weakness is that the usefulness of the proposed approach appears to depend heavily on the choice of the detection or segmentation model. Different underlying models may lead to different transformations and hence very different performance outcomes. This sensitivity should be systematically investigated to establish the robustness of the proposed framework. While I appreciate that the authors partially acknowledge this limitation, placing it only in the “Limitations” section is insufficient. A thorough empirical analysis on how the choice of model influences the results is essential to strengthen the paper’s contribution.

### Questions
I have no further questions about the paper. Please address the second weakness I mentioned above, as a more comprehensive study on the dependency of the approach on model selection would make the work more convincing. I would be happy to adjust the score if the authors can provide additional evidence showing that the proposed approach has been sufficiently investigated beyond simple performance evaluation.

LLM Usage Disclosure:
This review was refined using a large language model (OpenAI GPT-5) to improve clarity and grammar. The assessment, analysis, and opinions expressed are entirely my own.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper considers data augmentation strategies for contrastive representation learning. This work proposes an adaptive augmentation strategy by employing pretrained detection/segmentation models and generate augmentations based on their outputs to better preserve semantic consistency. Experimental results show the effectiveness of the proposed methods on image datasets.

### Strengths
+ Data augmentation for contrastive representation learning is an important topic.

+ The proposed method sounds interesting, and the performance gain is consistent throughout experiments.

### Weaknesses
- What is Raw Duplication? It requires explanation. Does it mean that all positive pairs have identical inputs?

- The description on the proposed method is insufficient. Figure 3 is not enough to fully understand the proposed method. What are "Multi-scale adaptive mechanism", "Segmentic guided Augmentation", and "Semantic Feature Enhancement"? 

- The proposed policy introduces hyperparameters, while no hyperparameter analysis on them is provided.

- The theoretical analysis seems not so related to the proposed method, and the statement is not justified. For example, the claim in L246, "semantic guidance reduces the conditional entropy of the original representation given the augmented one" is not justified. Considering out-of-domain transfer learning, the pretrained model might lack sufficient domain knowledge for target tasks.

- Experiments are limited to small image datasets; for example, ImageNette has only 10 classes. Hence, it is not sure if the observation is scalable.

- No out-of-domain transfer learning results, which is crucial for representation learning.

- If handcrafted augmentations are experimented on CPU following the standard deep learning library implementations, then the authors might want to try GPU-based augmentation with libraries such as Kornia for a more fair comparison, particularly around Figure 5.

### Questions
Please address concerns in Weaknesses above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
I think this work tackles a problem in contrastive learning: current augmentations are either fast but semantically "blind," creating "false positives" , or generative, which are slow and can cause "semantic drift" . So the authors propose Semantic Data Inflation (SDI), which is a  solution that uses existing models like YOLO and SAM as semantic oracles to guide augmentations to ensure the main object is always preserved . What I find most important is its multi-scale adaptive mechanism, which they try to first analyze image quality and then dynamically chooses between coarse bounding boxes (for low-quality images) or precise segmentation masks (for high-quality images) . So I think this method proves effective, and they also outperform baselines while being 4-5x faster than generative methods. The resulting features also show generalization to new domains and ViT architectures

### Strengths
I think the primary strength is the simple solution to the "false positive" problem. Using off-the-shelf models as "semantic oracles" is an effective way to achieve both semantic consistency and computational efficiency . In my opinion, the multi-scale adaptive mechanism selects guidance granularity based on image quality, a design that is justified by the ablation studies in Table 7. This design is backed by  several results. The authors also show the performance gains with a +3.8-4.1% average on ImageNette (Table 2). Finally, I think the method's practicality is a plus: it's 4-5x faster than generative alternatives and it generalizes well to ViT architectures and other downstream tasks like object detection.

### Weaknesses
Here are some major limitations after reading this paper.

In my opinion, the first weakness is that the entire framework is dependent on the quality of the upstream semantic oracles. If YOLO or SAM fail to find an object (maybe it's a class they weren't trained on, or the image is too abstract), then SDI would presumably fail as well. It's transferring semantic understanding, not learning it. I wonder if the author could elaborate this with several analysis? what if the image is complex which has multi-object composition.

I think while it's much faster than generative models, it's still more computationally expensive than the standard augmentation pipeline. It requires an extra forward pass from a powerful model like YOLO or SAM for each image. Table 1 shows it's about 2-3x slower than the standard augmentation. I just wonder if this training scheme is actually needed during the large-scale training. I think the authors should have more analysis related to this effect instead of just showing the average time difference.

### Questions
Please see the weakness above.

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
3

### Summary
Semantic Data Inflation (SDI) addresses the “semantic consistency–efficiency–diversity” trilemma in self-supervised contrastive learning. SDI first leverages off-the-shelf detection/segmentation models (e.g., YOLO and SAM) to extract multi-scale semantic cues, then applies transformations within these “semantic protection zones,” and uses an image-quality-weighted adaptive mechanism to choose among detection-level, segmentation-level, or mixed-level operations. The authors report that SDI outperforms manual augmentations and generative data inflation across various contrastive frameworks and datasets (e.g., achieving 95.75% linear evaluation on ImageNette, a +3.88% improvement over standard augmentation), and demonstrates modest gains across domains, architectures, and downstream detection/segmentation tasks. Overall, SDI is a systematic engineering integration of “semantics-guided data augmentation,” emphasizing improved semantic fidelity and representation discriminability without significantly increasing training overhead.

### Strengths
1. decouples “semantic extraction — enhancement within semantics — adaptive selection,” making it easy to integrate into existing contrastive learning frameworks.

2. shows consistent gains across multiple frameworks (SimCLR, MoCo, BYOL, Barlow) on datasets like ImageNette; provides transfer results to ViT-S, small-scale downstream detection/segmentation, and medical imaging.

3. positioned as a lightweight alternative to diffusion-based data inflation, avoiding the high overhead and semantic drift risks of generative methods (with reasonable justification).

4. Some interpretability: offers intuition from mutual information/invariance perspectives, and formalizes “adaptive scale selection” in the appendix.

### Weaknesses
1. he rationalization mainly relies on mutual information and tightening inequalities, with limited novelty; the relationship to hard negatives and in-batch distribution in practical training is not thoroughly explored. Compared with systematic baselines guided by “saliency/self-supervised segmentation features/other semantic priors,” the comparisons are insufficient; the main contribution novelty lies in engineering integration and heuristic/empirical strategies.

2. lacks pretraining validation at mainstream scales such as ImageNet-1k; systematic evaluation and failure-case analysis for more complex scenarios (multi-object, heavy occlusion, small objects) are not sufficient.

3.  heavily depends on detectors/segmenters trained on large-scale labeled data, introducing supervised priors; the paper under-discusses whether comparisons against SSL baselines that do not rely on external models are fair, and whether there are risks of data overlap/information leakage.

4. the appendix formulates the selection strategy as a learnable policy, but the implementation in the main text appears heuristic, lacking end-to-end learning and results on stability/generalization. It is recommended to supplement empirical results and ablations for the “learned policy.”

### Questions
- Has your adaptive selection been truly trained end-to-end? Please report comparisons against heuristic thresholds, the stability of learned α, β, γ, and cross-dataset/architecture transfer performance.

- How do you control fairness and potential leakage from external model priors? Please specify their overlap with pretraining/evaluation data, and how performance degrades when replaced with weaker or cross-domain models.

- Please supplement ImageNet-1k scale pretraining and systematic evaluations in more complex scenarios (multi-object/occlusion/small objects), along with visualizations of failure cases.

- Can you provide sensitivity curves to upstream errors (e.g., by modulating YOLO/SAM confidence/recall) and the corresponding robustness strategies (multi-candidate fusion/confidence gating)?

- Can the theory section further integrate analyses of hard negatives and in-batch distributions to explain SDI’s concrete impact on InfoNCE training dynamics?

### Soundness
3

### Presentation
3

### Contribution
2
