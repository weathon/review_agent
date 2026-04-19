# Simplifying Self-Supervised Object Detection Pretraining

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 5

## Abstract
Object detectors are often trained by first training the backbone in a self-supervised manner and then fine-tuning the whole model on annotated data. An unsupervised detector pretraining stage can also be interleaved, further improving the final performance and facilitating convergence during the supervised fine-tuning stage. However, existing unsupervised pretraining methods typically rely on low-level information to create pseudo-proposals that the model is then trained to localize, and ignore high-level class membership. The absence of class semantics from the pretraining objective causes a task gap between the pretraining and the downstream scenario, where detection is class-aware (e.g. given an image of a chair, the detector's task is to \textit{both }localize it and assign the ``chair'' class to the corresponding bounding box). This gap results in suboptimal detector pretraining. We propose a framework that better aligns the pretraining and downstream stages. It consists of three simple yet key ingredients: (i) richer, semantics-based initial proposals derived from high-level feature maps, (ii) discriminative training using object pseudo-labels produced via clustering, (iii) self-training to take advantage of the improved object proposals learned by the detector. We report two main findings: (1) Our pretraining outperforms previous works on the full and low data regimes by significant margins across detector architectures. (2) We show we can pretrain detectors from scratch (including the backbone) directly on complex image datasets like COCO, paving the path for unsupervised representation learning using object detection directly as a pretext task.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
- This paper proposes a new method for unsupervised object detector pre-training. The main pipeline is in three steps: 1) finding object proposals in an unsupervised way; 2) clustering the proposals to form pseudo class labels ; 3) training an object detector using the proposals and their labels. 

- On the downstream object detection task, the proposed method achieved comparable performance with previous methods.

### Strengths
1. The paper demonstrates an enhanced performance on the benchmark COCO object detection test, even if the improvement is marginal.
2. The proposed method has a good performance in data-limited scenarios, outperforming preceding methodologies substantially.
3. The manuscript is well-written, it is easy to follow, and the technical details are well explained. 
4. The idea of generating a detection dataset and using it to pre-train a detector in an unsupervised way is interesting.

### Weaknesses
1. My main concern for this paper is its significance and potential impact for future works:
    1. While the proposed method surpasses some recent methods in terms of detection AP, the advancements are quite minimal. For instance, For example, on Mask R-CNN it only outperform CutLER by 0.3 AP. (Table 1)
    2. One anticipated advantage of unsupervised pre-training would be the ability to exponentially scale the training dataset, subsequently enhancing the model's efficacy. This paper seems to miss out on this potential. The proposed clustering of object proposals becomes increasingly complex with dataset expansion, which may not even tractable when the dataset becomes very large, i.g., web scale. 
    3. Even if the clustering challenges are addressed, the method doesn't appear to capitalize on a larger dataset. For example, when switching the training set from COCO to OpenImage, which is much larger, the model performance keeps the same. (Table 6)
    4. he proposed multi-stage training only bring marginal improvement: increasing an extra stage of training brings less than 1.0 AP improvment. (Table 13)
2. The novelty of this work is also limited. The shift to generating object proposals from generating from low-level features is a simple extension of previous works. Additionally, the object detector's training mechanism seems heavily reliant on pre-existing methodologies.
3. I think the one of the intersting point of this method is its supervority on data-scarce settings (Table 4 and Table 5). However, the paper lacks in depth study about this superiority.

### Questions
I am wondering if the *Clustering and Classifying* paradigm can be replaced by some other methods, like contrastive learning. If this can work, it will greatly improve this work's impact.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper addresses the challenges in object detection training, where conventional methods involve a two-phase approach: self-supervised training of the backbone followed by supervised fine-tuning using annotated data. Many existing unsupervised pretraining techniques tend to depend on low-level data, neglecting high-level class semantics, resulting in a gap between the pretraining and actual detection tasks. To tackle this issue, the authors introduce a novel framework that emphasizes semantics-based initial proposals, employs discriminative training with object pseudo-labels, and utilizes self-training. This innovative approach not only surpasses preceding techniques but also facilitates the pretraining of detectors from scratch on complex datasets, such as COCO.

### Strengths
1. This paper includes the experiments that pre-train on COCO, which is a good exploration. When pre-trained on COCO, the proposed method outperforms the previous methods on the linear evaluation on ImagNet.
2. The method is validated on both transformer and cnn based detectors.

### Weaknesses
1.	From tab 7, we can see that the models pre-trained on COCO still can not outperform the models pre-trained on ImageNet, which has been shown in the previous papers. From this point, the exploration of pre-training on COCO did not bring novelty.
2.	“Utilizing semantic information from self-supervised image encoders to produce rich object proposals and coherent pseudo-class labels” has been explored in the previous papers such as [1].
3.	In the abstract, the authors claimed that “However, existing unsupervised pretraining methods typically rely on low-level information to create pseudo-proposals that the model is then trained to localize, and ignore high-level class membership.”I do not agree with it. In fact, the Moco and Mocov2 address on the high-level semantic information, while the later works[2,3] focus low-level information and localization. So you can not say that the existing pretraining methods typically rely on low-level information. The challenge is to create a pre-training method that can balance both localization and classification.
4. I am also curious about the comparison with MAE pre-training methods.
[1] Deep Spectral Methods: A Surprisingly Strong Baseline for Unsupervised Semantic Segmentation and Localization, L. Melas-Kyriazi, C. Rupprecht, I. Laina and A. Vedaldi, Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2022
[2] Fangyun Wei, Yue Gao, Zhirong Wu, Han Hu, and Stephen Lin. Aligning pretraining for detection via object-level contrastive learning. Advances in neural information processing systems.
[3] Zhenda Xie, Yutong Lin, Zhuliang Yao, Zheng Zhang, Qi Dai, Yue Cao, and Han Hu. Selfsupervised learning with swin transformers. arXiv preprint arXiv:2105.04553, 2021c.

### Questions
See the weakness.
I fail to discern how this work differs from prior efforts.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This study proposes a new self-supervised pretraining method called SEER for object detection. SEER first generates pseudo proposals using spectral clustering of the feature map generated by a pretrained feature extractor. The generated pseudo proposals are then fed to the detector to yield an end-to-end self-supervised an object detector. The pretrained network is validated on the MS-COCO and PASCAL VOC datasets.

### Strengths
- The proposed framework is simple and effective. By utilizing a pretrained feature extractor to generate proposals in an unsupervised manner, this method can obtain proposals of higher quality and better semantic meaning. The proposal filtering and pseudo-class label generation also require strong engineering insights to make the pipeline work for end-to-end self-supervised object detector pretraining.

- The presentation is easy to follow, and the contribution of the paper has been made clear after comparing it with existing object detector pretraining methods in its related works section.

- The obtained results are competitive, as the method is able to achieve 46.7 AP using the Deformable DETR detector and 49.6 AP using the ViDT+ detector.

- The evaluation on different tasks, including few-shot and semi-supervised learning, and the study of different pretraining datasets are appreciated. This helps demonstrate broader significance by investigating common interesting questions.

### Weaknesses
- The contributions of this work may be overstated. This paper presents SEER as a unique end-to-end object detection pretraining method that can train the backbone from scratch without freezing backbone parameters. However, previous works like JoinDet and [1] have also shown the potential ability to train the backbone without freezing, so unfreezing the backbone is not an entirely new contribution.
    

- Moreover, the ability to train the backbone of an object detector is an overstatement to some extent. As the method still requires a pretrained backbone model to generate pseudo proposals, it then leverages the generated pseudo proposals to train its backbone feature. Given that a pretrained backbone is already provided, forcing the network to retrain a backbone from scratch is not viewed as a fully end-to-end self-pretraining method.
    

- Additionally, [1] has already explored the possibility of pretraining an object detector in a fully self-supervised manner without requiring an extra pretrained backbone. When comparing [1] to this study, the pipeline of [1] seems simpler and is able to train the whole model from scratch.
    

- The pipeline relies on clustering over a pretrained network for its pseudo proposals, which inevitably introduces many hyperparameters. For example, the number of clusters for both local and global clustering is critical to the method's performance.
    

References:

[1] G Jin, et al., Self-Supervised Pre-training with Transformers for Object Detection, Neurips workshop 2022. ([https://sslneurips22.github.io/paper_pdfs/paper_4.pdf](https://sslneurips22.github.io/paper_pdfs/paper_4.pdf)).

### Questions
- This study highlights its ability to train the backbone from scratch. I would like to raise the reverse question: How does model performance compare when using a frozen pretrained backbone versus training the backbone from scratch?
    
- The performance on semi-supervised results lags far behind recent studies in semi-supervised object detection [2]. What if SEER adopts the same architecture and compares results with traditional semi-supervised object detection using self-training?
    

References:

[2] J. Zhang, et al. Semi-DETR: Semi-Supervised Object Detection with Detection Transformers. CVPR 2023.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
