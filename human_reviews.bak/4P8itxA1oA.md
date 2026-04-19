# Set Features for Anomaly Detection

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 5, 5

## Abstract
This paper proposes set features for detecting anomalies in samples that consist of unusual combinations of normal elements. Most methods, discover anomalies by detecting an unusual part of a sample. For example, state-of-the-art segmentation-based approaches, first classify each element of the sample (e.g., image patch) as normal or anomalous and then classify the entire sample as anomalous if it contains anomalous elements. However, such approaches do not extend well to scenarios where the anomalies are expressed by an unusual combination of normal elements. In this paper, we overcome this limitation by proposing set features that model each sample by the distribution of its elements. We compute the anomaly score of each sample using a simple density estimation method. Our simple-to-implement approach outperforms the state-of-the-art in image-level logical anomaly detection (+5.2%) and sequence-level time series anomaly detection (+2.4%).

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The authors present an approach based on set features to detect logical anomalies. This anomaly type consists of samples with unusual combinations of normal elements. The authors extract elements at multiple scales, apply random projections, and compute histograms in the projected spaces. The method is demonstrated in image and time series anomaly detection.

### Strengths
- The paper is well written and easy to follow.
- Anomaly detection is an interesting and timely topic.

### Weaknesses
- The method is applicable to a narrow case, specifically when the anomalies are represented by an unusual distribution of normal elements. This setting does not apply to detecting anomalies in general. The proposed method should be integrated within a method able to detect other kinds of anomalies.
- There are several methods reporting much better results on MVTec-LOCO (see [1]). The authors claim to achieve state-of-the-art results, but according to [1], this is clearly not the case. Since the method does not surpasses competing methods, its benefits are not well justified from a practical point of view.
- The method is demonstrated for only one bachbone: ResNet. It is not clear how the method generalizes to other backbones.
- It is not clear if the method applies to large scale datasets, since it relies on kNN.
- The inference time of the presented method is not discussed.
- With three lines, the conclusion is too short. The authors did not seem to manage the space too well. The conclusion should be more consistent.
- There are a few typos here and there that should be corrected:
  - "Fig.2. The average of a set of" => "Fig. 2, the average of a set of".

[1] https://paperswithcode.com/sota/anomaly-detection-on-mvtec-loco-ad

### Questions
The authors can refer to the identified weaknesses.

### Soundness
2 fair

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
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper introduces a novel approach to address the challenging problem of anomaly detection, specifically focusing on logical anomalies. The key idea revolves around modeling individual samples using their elemental components and corresponding distributions, ultimately calculating anomaly scores through density estimation on these set descriptors. The study conducts experiments encompassing both visual anomaly detection and time-series anomaly detection.

### Strengths
(1) This work tackles the intricate issue of logical anomalies in anomaly detection, a problem not commonly addressed.

(2) The paper is clear and easy to follow.

(3) The inclusion of experiments covering both image anomaly detection and time-series datasets adds to the paper's breadth and applicability.

### Weaknesses
(1) The method relies on a strong assumption that individual elements within a sample are normal, but their combination leads to anomalies (i.e., logical anomalies). However, there is no foolproof way to validate this assumption when a query sample is introduced, casting doubt on the practicality of the algorithm.

(2) The evaluation lacks comprehensiveness as it does not include state-of-the-art (SOTA) anomaly detection algorithms like CutPaste, RD4AD, SimpleNet, PaDim, CS-Flow, etc., in its experiment results (e.g. Figure 1). Over the past year, many anomaly detection algorithms have demonstrated promising results on the MVTec-LOCO benchmark. A comparison against these SOTA algorithms would better establish the effectiveness of the proposed method.

(3) Contemporary anomaly detection algorithms typically generate anomaly score maps, pinpointing the location or segment of abnormality within the query data. It remains unclear whether this method is capable of producing such anomaly localization results.

(4) In implementation details, the authors mention that combining multiple crops of a query sample enhances the proposed method's performance. However, there is no mention of the computational overhead introduced by this approach or whether there is a tradeoff between computational cost and algorithm accuracy. Moreover, it would be valuable to know if applying the strategy of using multiple crops to other methods, such as patchcore, yields similar or better results than the proposed method.

(5) The authors claims in the first page that the usually anomaly detection procedure follows the paradigm of detection-by-segmentation. This claim is too strong and not true. The majority of anomaly detection algorithms are based on data reconstruction and embedding similarity quantification.

### Questions
Please refer to the weakness section for my questions. In addition, I am wondering if there is a code published for this algorithm?

### Soundness
1 poor

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
The paper presents an algorithm to detect anomalies based on unusual combinations of set features. Multiple 'set' features for each instance might be computed using hand-crafted feature encoders, deepnets, etc. These set features are projected on to multiple histograms to compute histogram sketches that are then concatenated together to form the feature vector for anomaly detection. The objective here is that the featurization should help in detecting unusual combinations of higher-level elements rather than low-level patterns.

### Strengths
- The intuition behind the histogram descriptors is good

### Weaknesses
1. The paper should present the algorithm better. Currently, some vital information is only in the Introduction (last para on page 1). That should be moved to Section 3 for clarity.



2. Section 3.1 second paragraph "The typical way ...": While the discussion here refers to only the pooling aspect of deep networks, it overlooks their property of detecting higher-level abstractions. There are multiple feature maps in the layers of deep nets and each feature map detects one type of abstraction. Across multiple feature maps, it can be argued that they learn multiple 'sets' of features and the final task (classification/sequence prediction/anomaly detection) will be a function of the sets of features. Moreover, deepnets can learn the feature map finetuned to the task. In that respect, the proposed algorithm (SINBAD) is limited by the fixed set of histogram projections which cannot be finetuned. For example, such an ability to learn would automatically avoid projections along original axes if they are not discriminative. The point here is that, in theory, there is nothing fundamental about the proposed 'set' based design that a general purpose deepnet cannot do (e.g., for images, ResNet).



3. The procedure for generating the histogram descriptors should be illustrated either with a simple figure or in algorithm format.

### Questions
1. The number of set 'elements' (N_E in Section 3.2) used for each dataset should be presented. How large should this be for good accuracy?



2. N_P, N_D have not been properly defined before introducing them in Section 3.3. How are the number of histogram projections determined?



3. Do all set feature dimensions (not their histogram descriptors) need to be the same? i.e, (say) features extracted from different layers of ResNet? The projection matrix P (eqn 1) seems to suggest so.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper proposes set-based representations for effective anomaly detection on datasets of logical defects (such as MVTec-Loco [1] ). The set representation is constructed as the histogram of projected feature values. In the end, the anomaly score is computed by the Mahalanobis score in the set-representation space.

[1] https://www.mvtec.com/company/research/datasets/mvtec-loco

### Strengths
1. The usage of set features as presented in the paper (i.e., random projections of feature histograms and Mahalanobis distance scoring) is novel for the anomaly detection task to my best understanding.
2. The method has not only been image but time-series datasets

### Weaknesses
1. The method is not clearly presented. The notations are unclear; What is f_i[j] in Sec. 3.3? How is f in Eq. (1) exactly obtained?
2. One of the papers main claim that it is the sota on MVTec-Loco, is not valid; the proposed method is outperformed by EfficientAD [2], which is 2 years already outdated. Particularly, the proposed method significantly underperforms in the structural anomaly detection.
3. The method has not been tested on conventional benchmarks MVTec-AD [3] and VisA [4].
4. There is no computation cost analysis of the method in the paper. Is this efficient compared to EfficientAD and PatchCore?
5. Although the usage of histograms can be regarded as new in the anomaly detection task, the method itself is quite a classical one. 

[2] Batzner, Kilian, Lars Heckler, and Rebecca König. "Efficientad: Accurate visual anomaly detection at millisecond-level latencies." arXiv preprint arXiv:2303.14535 (2023).
[3] Bergmann, Paul, et al. "MVTec AD--A comprehensive real-world dataset for unsupervised anomaly detection." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019.
[4] Zou, Yang, et al. "Spot-the-difference self-supervised pre-training for anomaly detection and segmentation." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2022.

### Questions
1. It would be highly helpful if the authors provide a precise Algorithm of the proposed method.
2. What is the main difference between the proposed method and PNI [5]?

[5] Bae, Jaehyeok, Jae-Han Lee, and Seyun Kim. "Pni: industrial anomaly detection using position and neighborhood information." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair
