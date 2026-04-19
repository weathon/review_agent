# Out-Of-Distribution Detection With Smooth Training

- Decision: Reject
- Scores: 6, 5, 6, 6

## Abstract
Detecting out-of-distribution (OOD) inputs is important for ensuring the safe deployment of machine learning models in real-world scenarios. The primary factor impacting OOD detection is the neural network's overconfidence, where a trained neural network tends to make overly confident predictions for OOD samples. A naive solution to mitigate the overconfidence problem of neural networks is label smoothing. However, our experimental observations show that simply using label smoothing doesn't work. We believe that this is because label smoothing is applied to the original ID samples, which is the opposite of the goal of OOD detection (high confidence for ID samples and low confidence for OOD samples). To this end, we propose a new training strategy: smooth training (SMOT) where label smoothing is applied to the perturbed inputs. During the smooth training process, input images are masked with random-sized label-related regions, and their labels are softened to varying degrees depending on the size of masked regions. With this training approach, we make the prediction confidence of the neural network closely related to the number of input image features belonging to a known class, thus allowing the neural network to produce highly distinguishable confidence scores between in- and out-of-distribution data. Extensive experiments are conducted on diverse OOD detection benchmarks, showing the effectiveness of SMOT.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper addresses the issue of label smoothing in OOD detection. It first identifies the cause of overconfidence prediction is cross-entropy loss in neural networks. It then proposes a new training scheme of label smoothing, SMOT, for perturbed inputs. SMOT proposes to train models on the confidence of different areas of mask-out regions.

### Strengths
1. It applied label smoothing beyond training data where it demonstrated labeling smoothing on ID data is not enough.
2. The proposed training method without using auxiliary OOD dataset, but perturbation of masking is very promising where OOD auxiliary dataset is not available.  
3. The supplementary sections justify many decisions made in the main paper, such as why masking is chosen. The paper is generally complete and clear.

### Weaknesses
The discussions/conclusions from Theorem 1 and 2 are too abrupt and not very obvious, such as the paragraph just before the Sec.3.2. More explanations are needed, especially for Theorem 2. It makes it less self-contained.

### Questions
1. What are the intuitions/interpretations of $\sqrt{\frac{C}{n}}$ in Equ(4), and the equations of equations of theorem 1 and 2? The connection between the equations and the implications is not clear. 
2. Can SMOT use the updated model trained on the fly to get the mask for perturbation?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes SMOT, a smooth training algorithm for OOD detection. SMOT is based on the heuristic that masking out certain features from the input image should correspondingly leads to decrease in the network's prediction confidence. Specifically, SMOT leverages CAM together with random-thresholding to determine the masking region, and the soft label (or essentially prediction confidence encoded in the training target) is determined according to the threshold (which is related to the area of the masking region if I understand correctly). Experiments on CIFAR-10/100 and ImageNet-200 show that SMOT exhibits (moderate) performance improvements over certain existing methods.

### Strengths
- The manuscript is in general clearly written.

### Weaknesses
### Theoretical Investigation

I find that Sec. 3.1 is somewhat hard to follow. The message / motivation it tries to convey is unclear to me. See specific comments or questions below.

1. The conclusion of Theorem 1 is "given a sufficient amount of training data and a small optimal risk, ..., the issue of over-confidence for ID data is highly probable to arise". However, the equation is only related to "over-confidence" (which I assume refers to excessive maximum softmax probability according to Eq. 2) when the loss is exactly cross-entropy loss. If we use label smoothing as the loss (although it is later empirically shown not to work), then there won't be over-confidence by looking at Eq. 4.

2. My same argument could be applied to Theorem 2 as well. Furthermore, I can't see how exactly the "over-confidence in OOD data" is reflected in Theorem 2. More elaboration and clarification is necessary.

3. The concluding paragraph under Theorem 2 makes me lost again. Why we want to "access real OOD data to reduce the distribution discrepancy during training"? What does it mean to "reduce the distribution discrepancy" (the $d(\theta)$ in Theorem 2?) between ID and OOD? Meanwhile, why suddenly "limited training ID data", "overfitting", and "the failure of ID classification" become issues for OOD detection?

4. Lastly, where are the proofs of the Theorems (or where are the references if they were proved by existing works)?



### Design of SMOT

1. Eq. 9 seems a little arbitrary. Why using a temperature-scaled exponential function? What's the intuition behind it? Why (t - 255)? What is the value range of t?

### Experiments

1. One limitation of the experiments and presented results is the fact that all considered OOD datasets are far-OOD which are easier to be detected. I expect to see more results on near-OOD splits (e.g., CIFAR-100 or Tiny ImageNet for CIFAR-10, SSB or NINCO for ImageNet), which are more likely to translate to real-world where the OOD images can be extremely similar to ID images.

2. The baseline selection seems a bit arbitrary. How does SMOT compare with recent top-performing methods (e.g., ASH [1] as identified by OpenOOD [2])? Also, a highly relevant baseline is missing (see below "Related Work" for details).

3. Why the training budget and learning rate scheduler is different between base models and SMOT models? Specifically, base models are trained for 200 epochs, while the "final model" with the proposed SMOT loss is trained for 300 epochs). Meanwhile, the base model adopts a step-wise learning rate decay schedule, while the final model uses the more advanced cosine decay. Is this a fair comparison, especially given that both longer training and sophisticated scheduler exactly benefit OOD detection (Table 5 in [3])?

4. Lastly, an important ablation study that I believe should be included is how SMOT compares with random masking / cropping. This would better justify SMOT's design of leveraging CAM to determine the masking region.

### Related Work
Sec. 5 should be more thorough and informative. Specifically, notice that the general idea of using corrupted / perturbed images associated with soft labels has been explored in at least two works in the field of OOD detection [4, 5]. Among these, [4] is in particular relevant to this work. I put up a table below making high-level comparison between [4] and this work.

|      | soft target |  perturbation  | needs a pre-trained model? | 
|------|-------------|---|------|
| [4]  |  $y_\epsilon=(1-\epsilon)\cdot y + \epsilon / K \cdot u$ (see their Eq. 3)  |   image corruptions defined by ImageNet-C   | a pre-trained classifier for determining $\epsilon$ |
| SMOT |  $y_\epsilon=(1-\epsilon)\cdot y + \epsilon / K \cdot u$ (Eq. 5 in this work)   |  masking  | a pre-trained classifier for generating CAM mask |

From the above table, it is not obvious what advantages SMOT can offer over [4] (e.g., less compute, not requiring a pre-trained model). Therefore, I believe that [4] should be not only referenced but also included as an actual baseline to show that CAM-based masking and the associated method for assigning $\epsilon$ value are better than the designs of [4].

### Format
The references are inserted absurdly (e.g. "ResNet18 He et al. (2016)") which I believe are not in the most appropriate format. There are also some mis-formatting, e.g. "Eq.equation 8" in "Training details".

------------------------------------------------------
[1]  Extremely simple activation shaping for out-of-distribution detection

[2] OpenOOD v1.5: Enhanced Benchmark for Out-of-Distribution Detection 

[3] Open-Set Recognition: A Good Closed-Set Classifier Is All You Need

[4] Bridging In- and Out-of-distribution Samples for Their Better Discriminability

[5] Mixture Outlier Exposure: Towards Out-of-Distribution Detection in Fine-grained Environments

### Questions
Please see the questions in Weaknesses.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes label smoothing training framework for OOD detection. The authors use the CAM to identify the regions that have a strong correlation to the true label, and generate a masked input image and corresponding soft label for smooth training. Extensive experiments show that the smooth training strategy greatly improves the OOD performance with different score functions.

### Strengths
1. The proposed smooth training (SMOT) strategy, where soft labels are applied to the perturbed inputs, is technical sound to relieve overconfidence problem.
2. The image masking and label smoothing strategy is quite novel and makes sense.
3. The paper is well structured and in good presentation and writing.

### Weaknesses
1. It is a little bit expensive to use CAM for identifying those label-correlated regions. I would like to see the OOD detection performance with the randomly generated masks. For example, randomly masking 30%-70% of the image for smoothing training.
2. The proposed SMOT utilizes data augmentation for OOD detection. Therefore, the author should introduce and compare more related methods that investigate the effectiveness of data augmentation in OOD detection. I believe there have been many papers that exploring data augmentation for Calibration or OOD detection [1,2]
3. The SMOT framework is similar to the Outlier Exposure (OE) framework, the author should also compare the proposal with other Outlier exposure (OE) based methods, and discuss the advantages compared with the OE framework.

[1] RankMixup: Ranking-Based Mixup Training for Network Calibration
[2] OUT-OF-DISTRIBUTION DETECTION WITH IMPLICIT OUTLIER TRANSFORMATION

### Questions
1. What model is used in Table 2/3/4?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a new training strategy called Smooth Training (SMOT) to improve out-of-distribution (OOD) detection performance. The key idea is to apply label smoothing to perturbed inputs rather than original inputs during training. Specifically, the authors randomly mask label-relevant regions of input images identified by class activation maps. The labels for these masked images are softened proportional to the size of the masked regions. This forces the model to output lower confidence for partial inputs, widening the gap between in- and out-of-distribution examples.

### Strengths
* The proposed smooth training strategy is intuitive and simple to implement, requiring only small modifications to the standard training procedure.
* Thorough theoretical analysis is provided on how the commonly used cross-entropy loss leads to overconfidence, and how smooth training can mitigate this issue.
* Comprehensive experiments on CIFAR and ImageNet-200 benchmarks demonstrate SMOT consistently improves OOD detection across different base models, scoring functions, and datasets. Improvements are also shown when fine-tuning CLIP.
* Ablation studies validate the efficacy of key components like the label smoothing function and masking threshold sampling distribution.

### Weaknesses
* Although smooth training enhances Out-Of-Distribution (OOD) detection, there's a minor decrease in in-distribution accuracy compared to conventional training. An in-depth exploration of this trade-off could be beneficial.
* The authors employ class activation maps to pinpoint label-relevant regions for masking, necessitating a pre-trained model. Studying other perturbation techniques that don't require a pre-trained model could expand the method's applicability.
* Further analysis could be devoted to how the results are sensitive to variations in hyperparameter settings. For instance, how essential is the use of a CAM heatmap to guide masking? What's the optimal way to establish the relationship between mask size and label smoothing hyperparameter ?

================================

Comments after Rebuttal: 

Thanks for your response. 

1. The paper would greatly benefit from additional experimental analysis regarding the hyper-parameter $\lambda$. There is a discernible balance to be struck between in-distribution (ID) accuracy and OOD performance, which varies with $\lambda$. The sensitivity of this trade-off to different datasets is not sufficiently addressed. I encourage the authors to explore this aspect further to enhance the utility of their approach.

2. The SMOT technique appears to depend on a pre-trained model to generate the masking (noting that the use of random noise results in subpar OOD performance), a process which adds computational complexity. The paper does not elaborate on alternative methods for 'dirtying' clean images beyond masking or the application of random noise. Exploring and discussing potential alternative techniques would provide a more comprehensive understanding of the method's applicability and limitations.

Given these considerations, I am inclined to raise my score. I am hopeful that the authors will take these comments into account and address them in the final manuscript to strengthen the paper's contribution.

### Questions
see weakness

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good
