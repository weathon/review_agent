# EMBridge: Enhancing Gesture Generalization from EMG Signals Through Cross-modal Representation Learning

- Avg Score: 6.67
- Decision: Accept (Poster)
- Scores: 6, 6, 8

## Abstract
Hand gesture classification using high-quality structured data such as videos, images, and hand skeletons is a well-explored problem in computer vision. Alternatively, leveraging low-power, cost-effective bio-signals, e.g. surface electromyography (sEMG), allows for continuous gesture prediction on wearable devices. In this work, we aim to enhance EMG representation quality by aligning it with embeddings obtained from structured, high-quality modalities that provide richer semantic guidance, ultimately enabling zero-shot gesture generalization. Specifically, we propose EMBridge, a cross-modal representation learning framework that bridges the modality gap between EMG and pose. EMBridge learns high-quality EMG representations by introducing a Querying Transformer (Q-Former), a masked pose reconstruction loss, and a community-aware soft contrastive learning objective that aligns the relative geometry of the embedding spaces. We evaluate EMBridge on both in-distribution and unseen gesture classification tasks and demonstrate consistent performance gains over all baselines. To the best of our knowledge, EMBridge is the first cross-modal representation learning framework to achieve zero-shot gesture classification from wearable EMG signals, showing potential toward real-world gesture recognition on wearable devices.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a cross-modal framework that aligns EMG signals with hand pose embeddings to enable zero-shot gesture recognition. The approach is technically solid and yields consistent improvements.

### Strengths
1.The proposed solution represents a well executed and effective integration of recent state-of-the-art multimodal representation learning approaches, such as BLIP-style Q-Formers, into the EMG–pose alignment setting. The framework adapts these techniques in a way that is coherent and practically meaningful for wearable gesture recognition.
2.The experimental results are comprehensive and show consistent improvements across multiple datasets and evaluation protocols, including both in-distribution and zero-shot gesture classification, demonstrating the practical effectiveness of the proposed method.

### Weaknesses
1.While the proposed Community-Aware Soft Contrastive Learning objective introduces a soft alignment mechanism, the community structure itself is derived from a hard K-means clustering, which inherently imposes discrete partitions on the pose embedding space. This feels contradicting the goal of modeling continuous semantic similarity. Should a probabilistic model such as a Gaussian Mixture or kernel-based affinity estimation better capture soft neighborhood structures?
2.The construction of communities purely from geometric proximity in the pose latent space overlooks available semantic priors about gesture types. Integrating gesture semantics (e.g., coarse action categories) or using supervised/semi-supervised clustering could produce more meaningful communities and stronger cross-modal supervision.
3.While the paper is well structured and the experimental settings are clearly described, the background context for this specific application domain remains somewhat underdeveloped. In particular, the work would benefit from a clearer definition and motivation of the pose modality (e.g., what exactly constitutes a “pose” sample, how gestures are defined, and what practical application scenarios are targeted). For such a specialized domain, providing stronger domain background and contextual grounding would make the contribution easier to interpret and appreciate.

### Questions
NA

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
3

### Summary
This paper investigates the task of hand gesture recognition. The authors propose a cross-modal representation learning framework that bridges the semantic gap between EMG representations and poses to achieve zero-shot gesture generalization. Specifically, they introduce a Q-Former, a masked pose reconstruction loss, and a community-aware soft contrastive learning objective to enhance EMG representation learning. Extensive experiments are conducted to validate the effectiveness of the proposed framework.

### Strengths
1. The proposed cross-modal representation learning method for hand gesture recognition is a novel and interesting attempt.

2. The proposed cross-modal representation learning strategy effectively improves the quality of EMG representations.

3. The authors proposed a community-level structural similarity framework for soft contrastive learning.

4. The authors performed zero-shot classification on both in-distribution and unseen gesture categories.

### Weaknesses
1. The proposed community-aware soft contrastive learning mainly stems from previous work on contrastive learning. However, the motivation for introducing community-level structural similarities is not well explained.

2. The proposed method is only validated on the pose–EMG paired data. Could the proposed method also be applied to RGB–EMG data?

### Questions
1. In Appendix A.6, the authors claim that batch size does not affect the proposed EMBridge, while it degrades the performance of CPEP and Q-Former. Do these conclusions come from experimental results?

2. The evaluation is conducted on the emg2pose and NinaPro datasets. Only four gestures are used in the unseen setting, would this setup reduce the difficulty of the zero-shot evaluation?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper focuses on gesture generalization from EMG signals and pose. The authors propose EMBridge, a cross-modal representation learning framework consisting of three components: (1) a querying transformer that extracts pose-informative features from EMG signals and aligns them with pose features using an asymmetric setup, (2) a masked pose reconstruction loss to enrich the pose representations, and (3) a community-aware soft contrastive learning objective to account for varying pose similarities that are not captured by standard contrastive learning. The authors conduct a detailed ablation study on different components of the framework, evaluate performance on both in-distribution and unseen gesture classification tasks, and demonstrate improvements over baseline methods.

### Strengths
The paper is well written, with clear motivation and challenges. It addresses an important and interesting problem in EMG-based gesture generalization to achieve zero-shot gesture classification. The proposed framework is original, well thought out, and carefully designed to meet domain-specific needs, such as incorporating pose communities for community-aware soft contrastive learning, which is particularly interesting. The experimental section is detailed and includes ablation studies on different components of the framework, along with analyses on three EMG datasets.

### Weaknesses
I do not see any major weaknesses in this work. However, a more detailed discussion of the limitations and potential future directions would strengthen the paper. While reproducibility details are sufficient, sharing the code in the future would further benefit the community.

### Questions
- What is the motivation behind using an asymmetric setup with the pose estimator and EMG encoder? Have the authors experimented with a setup where the pose estimator is also trained during the contrastive learning process (in the Query Transformer)?

- Regarding the relationship between paired gestures in experiments, are they independent or combinatorial? If they are combinatorial (for example, open hand + down), is this relationship captured in the pose similarity matrix, or could it be included as prior knowledge?

### Soundness
3

### Presentation
3

### Contribution
3
