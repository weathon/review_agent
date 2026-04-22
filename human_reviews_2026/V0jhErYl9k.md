# Label-Free Privacy-Preserving Learning for Zero-Shot Action Recognition

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
Traditional action recognition relies on labeled data and closed-set assumptions, limiting adaptability to novel actions and environments. Vision-Language Models (VLMs) offer a more flexible alternative through text-image alignment, enabling zero-shot action recognition. However, using raw video data poses privacy risks due to sensitive visual content. Privacy-Preserving Action Recognition (PPAR) aims to anonymize videos while preserving action-relevant semantics. Existing learning-based PPAR approaches often require both action and privacy annotations and retraining of recognition models on anonymized data, limiting their flexibility and compatibility with powerful pretrained VLMs. We propose LaF-Privacy, a novel label-free privacy-preserving framework for zero-shot action recognition. Our method is trained without any manual annotations, using two complementary objectives: preserving high-level action-relevant features and suppressing low-level appearance cues between raw and anonymized videos. We adopt a video transformer encoder for spatio-temporal learning and introduce an Action-Aware Masking Module (AAMM) to discard irrelevant regions, further enhancing privacy. LaF-Privacy enables direct use of pretrained VLMs for zero-shot inference on anonymized videos. Experiments on VP-UCF101 and VP-HMDB51 demonstrate that our approach achieves state-of-the-art trade-offs between privacy protection and zero-shot recognition performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents a label-free privacy preservation method for zero-shot action recognition using VLMs. Their proposed method minimizes visual similarity while maintaining embedding similarity, maintaining performance on action recognition tasks. Notably, their method does not require action or private attribute labels.

### Strengths
1. Handling VLM anonymization is a solid motivation that is underexplored in the field.
2. The idea is clean, results show only a small utility decrease with moderate privacy increase.

### Weaknesses
1. Anonymization performance is weak. Even with the cMAP justification, there is room for better anonymization. In prior (trained) methods, it is near impossible to make out private attributes. Even in some qualitative examples (Figure 7), some attributes are visually identifiable. Could a tradeoff curve be analyzed by scaling the relative privacy-utility weights?
2. It is difficult to tell if the anonymizer is specific for the zero-shot action recognition task. The results should explore other types of VLM tasks such as retrieval or even a captioning task to see if general performance of the VLM is retained.
3. The results without the AAMM appear to demonstrate natural masking, but the reasoning is unclear. The patch-level representations are regularized to have similar representations to the original patches. It would be helpful to see more analysis on how it learns to mask specific tokens.

### Questions
1. Could a privacy objective be up-weighted to result in more "self-masking" (see W3) and eliminate the need for the additional masking module? Also, would this result in lower cMAP/lower visibility with some (ideally slight) decrease to utility?
2. Does the natural masking before AAMM imply that those representations are already close to noise?
3. Can this anonymizer + VLM be applied to additional tasks beyond zero-shot action classification?
4. In the ablation where just the CLS/video token is used in the utility loss, how come the private attribute prediction scores increase? This is counterintuitive, since the visual patches are responsible for the privacy-preservation.
5. How do the privacy results look on the VISPR dataset shown in prior anonymization works?
6. Is the anonymization method robust against attacks? Meaning, can a model learn to reconstruct/denoise back to the original input after anonymization?

### Soundness
3

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
The paper introduces LaF-Privacy, a label-free and privacy-preserving framework for zero-shot action recognition. It aims to anonymize videos while retaining action-relevant semantics so that pretrained Vision-Language Models (VLMs) (e.g., CLIP, X-CLIP, ActionCLIP) can perform zero-shot action recognition without retraining. The approach is built on a video transformer encoder with an Action-Aware Masking Module (AAMM) that dynamically masks uninformative or privacy-sensitive patches.

### Strengths
A. Clever supervision design – Uses pretrained CLIP-style embeddings to preserve semantics without labels. The overall model is a  general and plug-and-play – Works with different VLMs (X-CLIP, ActionCLIP) without retraining them.

B. Experimental results achieve overall good balance between action and privacy, in a zero-shot setting for action and supervised testing for privacy. 

C. The overall approach is simple to understand, though the building blocks seem huge.

### Weaknesses
A. There seems to be an error on the Table 1 - privacy F1 of ours (0.632) of VP-HMDB51 should be the bold best rather than the underline 2nd best for the zero-shot session. Please confirm the number or correct it if wrong. 

B. The proposed method requires quite amount of learning (training) and, from Table 1, it seems the performance does not achieve the best on both the action recogntion and privacy hiding with this training efforts. For example, 2x down-sampling is achieving the best action recogntion and the blackening is the best model for privacy across datasets. As the paper mentioned, the proposed model is playing as a trade-off maker here. This reviwer would like to put up some challenge here:
-  Can the similar trade-off be obtained through a mix of 2x-down sampling and blackening (with a variety of blackening levels), which is not only unsupervised but also learning free? 
    - If so, then the complicated training schema from this paper would seems unnecessary or at last less efficient. For example, 2 × downsampling + mild blurring or masking could plausibly approximate the “moderate visual difference” LaF-Privacy achieves — at far lower training cost. After all, LaF-Privacy’s F1 and cMAP are just 0.03–0.05 lower than each single transformation baseline.

C. The training of privacy model is using the distorted frames, rather than the raw frames. This reviewer would like to point out that the VLM embedding might also have a chance carrying privacy semantic, which is mostly preserved due to the L2 loss of the Action Module. Since there is no re-traiing of VLMs, it is possible that VLM embeddings will leak privacy, which is over-looked by this study. This reivewer is wondering any thoughts from authors on this matter.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a framework called LaF-Privacy, designed to achieve label-free privacy-preserving learning for zero-shot action recognition. The method anonymizes videos without requiring any action or privacy annotations, enabling direct compatibility with pretrained VLMs for zero-shot inference. The framework consists of a video transformer encoder, an AAMM, and a multi-objective loss function that integrates visual dissimilarity maximization, action feature preservation, and masking regularization. Experimental results are reported on the VP-UCF101 and VP-HMDB51 datasets.

### Strengths
The paper meaningfully combines zero-shot recognition and privacy preservation, addressing the dual practical demands of data privacy and model generalization in real-world scenarios.

The experimental design is generally comprehensive, including main experiments, ablation studies, cross-VLM evaluations, and visualization analyses, which evaluate the proposed approach from multiple perspectives.

The method requires no costly privacy annotations, lowering the practical barrier for deployment and demonstrating potential applicability.

The writing is clear, and the figures and tables effectively convey the core ideas and experimental outcomes.

### Weaknesses
The paper does not provide a fair comparison with a broader range of unsupervised or self-supervised learning approaches, such as reconstruction-based or contrastive learning methods. This makes it difficult to discern whether the reported improvements are primarily due to the proposed architectural design or simply the strength of VLM-based representations.

The authors mention in the appendix that cross-VLM generalization performance degrades, but they do not provide an analysis of the underlying causes. While they claim the performance drop is acceptable, the magnitude of this decrease is non-negligible when compared to the reported performance gains. Furthermore, the evaluation lacks validation across a broader range of VLMs to assess generalization. If the framework is heavily dependent on a specific VLM and cannot transfer effectively to others, its practical applicability in diverse unsupervised scenarios would be severely limited—contradicting the paper's claimed contribution of being a flexible, label-free solution compatible with pre-trained models.

Although the paper emphasizes the notion of an “SOTA trade-off,” the absolute reductions in privacy metrics (F1, cMAP) are relatively limited, particularly for attributes like “relationships”, where the improvement is marginal. This raises concerns about the robustness of the framework against stronger inference attacks, which the paper does not discuss in depth.

Although the combination of "label-free" and "zero-shot" learning has practical value, the proposed method primarily integrates existing VLM and Transformer frameworks without introducing novel theoretical mechanisms or significant architectural innovations. Moreover, the training process heavily relies on pre-trained VLMs, making it susceptible to domain biases inherent in these models and thus limiting its applicability.

The design of the AAMM closely resembles the Learned Token Pruning (LTP) mechanism. While the application domain is different, the paper lacks sufficient novel theoretical justification or architectural innovation to substantiate its unique contribution in this context.

The privacy protection evaluation relies solely on F1 and cMAP metrics, lacking analysis against more targeted attacks.

### Questions
Ablation results show that the model without AAMM (“Unmasked”) achieves similar performance. Could the authors further analyze specific scenarios where AAMM is crucial? Are there explicit cases demonstrating that dynamic masking offers significant advantages over a fixed masking ratio?

The current privacy evaluation relies on a ViT-based classifier trained on anonymized videos. Have the authors considered stronger reconstruction-based privacy attacks to assess the robustness of the proposed framework? For attributes such as “relationships”, which are difficult to protect, what are the authors’ potential improvement directions or future plans?

### Soundness
2

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
4

### Summary
This paper tackles the problem of preserving privacy in action recognition without access to labels. The paper uses an Action-Aware Masking Module (AAMM) to mask out irrelevant information from the decoded input video and uses a loss to push the masked output away from the original input. To ensure the action recognition performance is retained, a pre-trained VLM is used to align the representations before and after augmentation. A mask loss is used to maximize the amount of tokens that are masked in the process. Overall the paper achieves some improvement over generic baselines in the zero-shot setting such as downsampling, blackening, and blur.

### Strengths
- The ability to perform privacy preservation on action recognition without requiring privacy labels is novel and interesting.

- The modules used in the pipelined are thoroughly detailed for ease of replication. 

- The proposed method achieves an average improvement compared to the baselines over both classification accuracy and privacy preservation

### Weaknesses
- Ablation results reveal that the visual loss and applying the mask does not have a significant impact on the privacy results despite being the portion of the method designed to focus on privacy.

- The cross-VLM results are not very convincing, with a significant drop suggesting that this method is not generalizable. In principle, masking the input should be a model-agnostic privacy technique.

- In Figure 2, it is not apparent which modules are being trained and which are frozen.

- There is a lack of clarity in figure captions. In figures 3 and 4 it is unclear what the takeaway is from the shown masks, as many of them look very similar.

- There is no ablation on the scaling weights for the action and vision losses. It would be helpful to see the curve as these are varied to better understand the tradeoff between classification accuracy and privacy score.

- Minor note: the filesize for this paper is very large, likely due to high quality images in the ablations, please reduce in future revisions.

### Questions
- How  is the MLP in the AAMM initialized and how does it estimate token importance?

- The baselines adopted in this paper do not show much of a drop in privacy score compared to raw data, while in the cited work [1] there is more of a significant change. What is the reason for this change in behavior?

- How does the method perform on the VISPR1[2] dataset?

[1] Ishan Rajendrakumar Dave, Chen Chen, and Mubarak Shah. Spact: Self-supervised privacy preserva tion for action recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pp. 20164–20173, 2022

[2] Tribhuvanesh Orekondy, Bernt Schiele, and Mario Fritz. Towards a visual privacy advisor: Understanding and predicting privacy risks in images. In IEEE International Conference on Computer Vision (ICCV), 2017.

### Soundness
2

### Presentation
3

### Contribution
3
