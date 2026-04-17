# ContinualCropBank: Object-Level Replay for Semi-Supervised Online Continual Object Detection

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 2, 2, 2

## Abstract
Deep learning has achieved remarkable progress in object detection, but most advances rely on static, fully labeled datasets$\textemdash$an unrealistic assumption in dynamic, real-world environments. Continual Learning (CL) aims to overcome this limitation by enabling models to acquire new knowledge without forgetting prior tasks; however, many approaches assume known task boundaries and require multiple passes over the data. Online Continual Learning (OCL) offers a more practical alternative by processing data in a single pass; however, it remains limited by its dependence on costly annotations. To address this limitation, Label-Efficient Online Continual Object Detection (LEOCOD) extends OCL with a semi-supervised formulation, enabling detectors to leverage unlabeled data alongside limited labeled samples. In this paper, we propose ContinualCropBank, an object-level replay module for LEOCOD that stores object patches cropped from bounding box regions and pastes them into stream images during training. This solution enables fine-grained replay, mitigating catastrophic forgetting while addressing foreground–background imbalance and the scarcity of small objects. Experiments on two benchmark datasets demonstrate that incorporating ContinualCropBank improves detection accuracy and resilience to forgetting, achieving gains of up to $9.57$ percentage points in average accuracy and reducing degradation from forgetting by up to $2.32$ points.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper addresses the problem of supervised and semi-supervised online continual learning for object detection, and proposes a novel replay module, ContinualCropBank, inspired by CropBank, initially developed for SSOD. The module is based on the idea of storing object patches cropped from the bounding-box regions of labeled images, and pasting them on streaming images during training. The module can be used on top of other modules such as Efficient-CLS, and they can both be used on top of other methods. Methods from the literature combined with these modules are tested in a fully supervised and a semi-supervised scenario, and exhibit improvements with respect to the standalone methods.

### Strengths
The authors present a novel module that draws inspiration from the online continual learning literature and adapts it to the semi-supervised setting of LEOCOD. Although the concept of CropBank was previously introduced in the literature, its adaptation to the LEOCOD framework, particularly when combined with Efficient-CLS, is non-trivial. The proposed module demonstrates notable improvements over standalone methods, as well as over methods combined with Efficient-CLS alone. The ablation study further supports the effectiveness of the module and strengthens the overall evaluation.

### Weaknesses
My main concern is on whether the comparison with previous approaches is appropriate. Since the proposed method is suitable for Semi-supervised Online Continual Learning for Object Detection, it should be compared against methods designed for subareas of this problem such as:
- Continual Learning for object detection
- Online Continual Learning for Object Detection
- Semi-supervised Continual Learning for object detection
Indeed, a method from the Semi-supervised Continual Learning for object detection literature could have a good performance (maybe analogous to the proposed method) even if it isn’t strictly created for Online CL settings. Similarly, methods for Online Continual Learning for Object Detection should be tested in the “fully supervised” subsection of the experiments. Instead, the authors only compare against methods from the general continual learning literature, hence the improvements that they report could be misleading.
I would suggest adding comparisons with such methods, or better justifying the lack of such comparison.

Minor concerns:
Lines 053-065 are difficult to follow, especially for how LEOCOD, online continual learning, SSOD, efficient-CLS are tied together. Table 2 is a little bit confusionary, I would like to see also the improvement of Efficient-CLS alone.

### Questions
Besides the concerns raised in the weaknesses section, here are additional questions:
- In Table 2 we can notice that there is a great improvement regarding methods paired with efficient-cls and methods alone. From my understanding, however, efficient-cls was created for the semi-supervised settings. Why is the improvement this strong in a fully supervised setting?
- In table 2, sometimes some metrics improve thanks to ContinualCropBank, while others decrease. Why do you think that there is this mismatch sometimes? Why is forgetting the metric that decreases most often?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper studies Label-Efficient Online Continual Object Detection, an online one-pass continual detection setting where each mini-batch contains only a small number of labeled images and the rest are unlabeled images, and the model is trained using a teacher–student framework similar to Efficient-CLS. The proposed module, ContinualCropBank, stores ground-truth object crops in a per-class FIFO memory and pastes these rescaled crops back into current labeled and unlabeled images to increase foreground density and expose more small and rare objects.

### Strengths
LEOCOD is realistic and under-explored: it is online, single-pass, and only partially labeled at each step. ContinualCropBank is simple to integrate into Efficient-CLS and is reported to improve CAP and FAP and reduce Forgetfulness F under different supervision ratios.

### Weaknesses
1.	Novelty is not clearly demonstrated. ContinualCropBank is conceptually very close to prior instance-bank and copy-paste style object replay methods in semi-supervised object detection. The paper claims that per-class FIFO storage, using only ground-truth crops instead of pseudo-label crops, and explicitly targeting catastrophic forgetting are the main contributions. However, the paper does not include a baseline that directly applies a naïve CropBank-style replay strategy under the same one-pass, partially labeled LEOCOD protocol. This baseline should be added to show that existing object-level replay does not already solve this setting.
2.	The paper does not disentangle continual replay from stronger augmentation. ContinualCropBank pastes rescaled crops into current training images, injecting extra foreground content and creating additional small objects, which by itself can improve object detection performance regardless of solving catastrophic forgetting. A baseline should be included to verify that the gains in CAP and FAP and the reductions in Forgetfulness F actually come from mitigating catastrophic forgetting rather than simply from aggressive data augmentation.
3.	Technical details important for reproducibility are under-specified. The paper does not precisely define how severe occlusion is avoided when pasting crops, how small, medium, and large objects are defined for the rescaling policy, or how sensitive performance is to the fixed per-class FIFO capacity of stored crops. These thresholds, definitions, and sensitivity analyses should be stated explicitly.

### Questions
As listed above:
1. What is the novelty of the proposed method?
2. How to prove the gains of the paper come from the proposed modules?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper targets label-efficient online continual object detection. Building on Efficient-CLS, it integrates the CropBank technique from semi-supervised object detection, augments it with FIFO and per-class capacity limits, and applies scaling and related data-augmentation strategies, yielding the proposed ContinualCropBank method. Compared with classic baselines (e.g., EWC) and the domain baseline Efficient-CLS, the approach achieves substantial performance improvements.

### Strengths
By integrating and adapting existing methods to the streaming-data setting, this paper improves upon prior LEOCOD approaches and offers a clear design rationale supported by strong empirical evidence.

### Weaknesses
1.The paper stacks the CropBank component from ACRST (semi-supervised object detection) onto the prior LEOCOD method Efficient-CLS and adds FIFO, per-class quotas, and data augmentation. This is essentially a scenario-specific adaptation rather than a genuine innovation.
2.Random placement, fixed scaling ranges, and basic occlusion checks may introduce artificial artifacts, leading to unrealistic context/occlusion patterns and texture repetition.
3.Compared with CropBank, the proposed ContinualCropBank stores patches only from labeled images and discards pseudo-labeled ones, but the rationale for this design choice is not explained.
4.The paper does not analyze sensitivity to key hyperparameters (number of pastes per image K, scaling ranges, memory capacity, IoU/occlusion thresholds) nor justify why FIFO is preferable to alternative policies.

### Questions
1.Please clarify how the proposed ContinualCropBank differs from CropBank. In particular, if CropBank were applied directly to the LEOCOD setting, would its results differ from those reported for ContinualCropBank?
2.Please justify—theoretically or empirically—the advantage of extracting patches only from labeled images (i.e., discarding pseudo-labeled images) compared with the original CropBank design.
3.Please demonstrate the sensitivity of key hyperparameters (e.g., number of pastes per image K, scaling ranges, memory capacity, IoU/occlusion thresholds) to support the choices made in the paper.
4.Please provide runtime, memory usage, and throughput overhead to quantify the additional computational cost introduced by the method.

### Soundness
2

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
5

### Summary
The paper introduces ContinualCropBank, an object-level replay module for label-efficient online continual object detection. It stores cropped object patches from labeled frames and pastes them into new labeled or unlabeled images during training, providing fine-grained replay that reduces forgetting and improves detection of small or rare objects. The experiments are evaluated on two benchmarks: OAK and the EgoObjects datasets.

### Strengths
- The paper addresses an important challenge in label-efficient online continual learning, where object annotations may be unavailable during training.
- The proposed approach introduces a simple yet effective object-level replay mechanism that is both memory-efficient and easily integrable with existing frameworks.
- The paper is well written and easy to follow.

### Weaknesses
- Conceptually inconsistent: The replay mechanism pastes objects randomly without considering the scene or spatial context, which can lead to unrealistic or semantically implausible compositions. This randomness may introduce anti-causal or spurious correlations in the semi-supervised setting; for example, placing a laptop on the roof of a moving car, causing the model to learn associations that rarely occur in real-world data. While such augmentation may improve accuracy, the gains could partly arise from exploiting these spurious correlations rather than genuine generalization.

- Weaker experimentations and missing relevant works: Baseline comparisons are limited to older continual learning methods; stronger memory-based or transformer-based COD baselines are missing. For instance, transformer-based: OW-DETR [1], transformer-memory-based: MD-DETR [2].

- The method assumes accurate bounding boxes for labeled samples, making it less robust to noisy or weak annotations.

[1] Gupta, Akshita, et al. "Ow-detr: Open-world detection transformer." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022

[2] Bhatt, Gaurav, James Ross, and Leonid Sigal. "Preventing catastrophic forgetting through memory networks in continuous detection." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024.

### Questions
- MD-DETR [1] generates pseudo labels through background relegation while training on current tasks, effectively functioning as a form of semi-supervised continual detection. Could the authors clarify how their approach compares to or differs from MD-DETR in terms of the semi-supervised strategy it employs?
- Is there any relationship between the selected object patch and the target image where it is pasted? How does random object insertion compare to context-aware pasting in terms of performance and realism?
- Could the authors include comparisons with more recent transformer-based and memory-augmented transformer baselines, such as [1] and [2]?


[1] Bhatt, Gaurav, James Ross, and Leonid Sigal. "Preventing catastrophic forgetting through memory networks in continuous detection." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024.

[2] Gupta, Akshita, et al. "Ow-detr: Open-world detection transformer." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022

### Soundness
2

### Presentation
3

### Contribution
2
