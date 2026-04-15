# GPS-SSL: Guided Positive Sampling to Inject Prior into Self-Supervised Learning

- Decision: Reject
- Scores: 5, 6, 5

## Abstract
In this paper, we propose Guided Positive Sampling Self-Supervised Learning (GPS-SSL), a general method to embed a priori knowledge into Self-Supervised Learning (SSL) positive samples selection. Current SSL methods leverage Data-Augmentations (DA) for generating positive samples and their performance heavily relies on the chosen set of DA. However, designing optimal DA given a target dataset requires domain knowledge regarding that dataset and can be costly to search and find. Our method designs a metric space where distances better align with semantic relationship thus enabling nearest neighbor sampling to provide meaningful positive samples. This strategy comes in contrast with the current strategy where DA are the sole mean to incorporate known properties into the learned SSL representation. A key benefit of GPS-SSL lies in its applicability to any SSL method, e.g. SimCLR or BYOL. As a direct by-product, GPS-SSL also reduces the importance of DA to learn informative representations, a dependency that has been one of the major bottlenecks of SSL. We evaluate GPS-SSL along with multiple baseline SSL methods on multiple downstream datasets from different domains when the models use strong or minimal data augmentations. We show that when using strong DA, GPS-SSL outperforms the baselines on under- studied domains. Additionally, when using minimal augmentations –which is the most realistic scenario for which one does not know a priori the strong DA that aligns with the possible downstream tasks– GPS-SSL outperforms the baselines on all datasets by a significant margin. We believe that opening a new avenue to impact the SSL representations that is not solely based on altering the DA will open the door to multiple interesting research directions, greatly increasing the reach of SSL.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
A method using prior knowledge to sample the positive data is proposed. It is supposed to mitigate the importance of data augmentation in self-supervised learning. The proposed GPS-SSL has shown superior capability over the methods with existing augmentation strategies.

### Strengths
+ Studying new strategies that rely less on data augmentations in self-supervised learning is worthwhile to the representations learning fields.
+ Exploring the pre-trained models (CLIP, Supervised models, VAE) for improving SSL might be interesting.

### Weaknesses
+ The proposed method needs a heavier component (such as a neural network ResNet-50) to generate the positive data sample, which is significantly computational compared to a simple calculation of data augmentation even for strong augmentations with a series of cropping, color jittering, distortion, hue, etc...

+ With the aid of a strong knowledge (and heavy) model trained on millions or hundred million of data (CLIP, ImageNet) the performance of the proposed method brings minimal advantage even worse than the existing SSL method such as VICReg in Table 2 with strong augmentation. In the weak augmentation setting, GPS-SSL may give better performance but still lag significantly behind the optimal setting (strong augmentation) of both streams, making it questionable about the contribution of the proposed method.

+ SSL contains another branch that is also very promising with the fine-tuning accuracy on downstream tasks such as MAE [1], this approach also depends very little on data augmentation (only cropping or without any augmentation already made the very good performance). This example (MAE method) will challenge the proposed method in terms of dependency on augmentation because the proposed method could not work without augmentation. I believe that modern SSLs should include this metric (fine-tune accuracy) and compare both contrastive learning and MAE approaches.

+ It should also include the linear evaluation of the only CLIP RN50 or supervised RN50 model when they have been used as the feature extractor for the downstream tasks on each considered dataset. It is to see without any training, how well these pre-trained model can perform, and based on that we can assess their contribution to the GPS-SSL (which is a combination of existing SSL + pre-trained CLIP/RN50).

+ Another point is that the experimental setting is not practical and sufficient to demonstrate the effectiveness of GPS-SSL when evaluating self-supervised contrastive learn is that they only consider pretraining with 200 epochs, which is very few epochs required by SSL models to fully converge. As shown in SimSiam or many SSL (MoCo, BYOL, Barlow Twins, VICREG,... ) the performance is best achieved with long enough self-supervised pretraining (800-1000 epochs). As a result, the comparison in long training should be considered for both methods.

+ It is not clear what is the metric they have shown in Table 1. Reading its caption, it is challenging to capture what metric they are comparing, top-1 ACC or error or something else.

[1] Masked Autoencoders Are Scalable Vision Learners, CVPR 2022

### Questions
See weaknesses

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors proposed the Guided Positive Sampling (GPS) approach to
finding positive pairs in self-supervised learning, without data
augmentation.  For each instance, a nearest neighbor is found in an
embedding space pretrained with another dataset or with a variational
autoencoder on the same dataset.  The corresponding instance becomes
the positive instance for self-supervised learning.

In their experiments, they consider using GPS with SIMCLR, BYOL,
Barlow, and VICreg on five datasets.  For GPS, they use embeddings
from supervised training, CLIP or VAE.  Generally, empirical
results indicate that using GPS outperforms, particularly with weak
augmentations.

### Strengths
Not relying on heavy handcrafting of data augmentation for
self-supervised learning is interesting.  Using prior knowledge based
on a pretrained encoder, they propose to find a nearest neighbor to
form a positive pair.  Generally, empirical results indicate that using
GPS outperforms, particularly with weak augmentations.

### Weaknesses
With prior knowledge, GPS seems to have an advantage over regular SSL,
which generally does not use prior knowledge.  According to Figure 1,
data augmentation is used in GPS-SimCLR.  So GPS seems to differ only
in the use of prior knowledge to find positive pairs.

Details are in questions below.

### Questions
1.  Theorem 1: GPS-SSL: employing eq (2) or (3) into eq (1)?

2.  Table 2: why are two different kinds of prior knowledge is used?

3.  How is $Tau$ set in Equation 3?

4.  With prior knowledge from another encoder, GPS has an advantage.
    Hence, comparison with methods that don't have prior knowledge
    might not be fair.  Could the regular SSL (with augmentation) also
    use prior knowledge?  For example, the encoder is initialized by
    prior knowledge and then regular SSL is performed.

5.  Sec 4.1, how do you predict if the classes do not overlap in the
    training and test sets (unseen classes branches/chains)?

--------  after response from authors ---

I think the authors performed experiments that remove the advantage of prior knowledge used in GPS and the results indicate GPS can improve performance over regular SSL.

### Soundness
3 good

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
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes Guided Positive Sampling Self-Supervised Learning (GPS-SSL), a method that integrates prior knowledge into Self-Supervised Learning (SSL) to improve positive sample selection and reduce reliance on data augmentations. Based on pretrained visual models and target dataset, GPS-SSL creates a metric space that facilitates nearest-neighbor sampling for positive samples. The method is applicable to various SSL techniques and outperforms baseline methods, particularly when minimal augmentations are used.

### Strengths
- Extensive experiments show the effectiveness of the GPS strategy.
- The paper is easy to follow.

### Weaknesses
- The employment of prior knowledge, specifically in the form of a pretrained visual model and the target dataset, diverges from the fundamental principles of Self-Supervised Learning (SSL).
- The incorporation of such prior knowledge raises concerns about the fairness of comparisons with existing SSL methods. There is a potential risk that the pretrained visual model and target dataset might leak additional information into the model, thereby skewing results and leading to issues of unfairness.
- The difference between GSP-SSL and NNCLR lies primarily in their respective positive sampling strategies. However, the novelty of the proposed strategy is limited.

### Questions
- It would be better to make prior knowledge in an unsupervised manner, except using pretrained visual model and target dataset.
- The supervised results are supposed to be shown in Table 2.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
