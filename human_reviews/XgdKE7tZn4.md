# HyperDisGAN: A Controllable Variety Generative Model Via Hyperplane Distances for Downstream Classifications

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 1

## Abstract
Despite the potential benefits of data augmentation for mitigating the data insufficiency, traditional augmentation methods primarily rely on the prior intra-domain knowledge. On the other hand, advanced generative adversarial networks (GANs) generate cross-domain samples with limited variety, particularly in small-scale datasets. In light of these challenges, we propose that accurately controlling the variation degrees of generated samples can reshape the decision boundary in the hyperplane space for the downstream classifications. To achieve this, we develop a novel hyperplane distances GAN (HyperDisGAN) that effectively controls the locations of generated cross-domain and intra-domain samples. The locations are respectively defined using the vertical distances of the cross-domain target samples to the optimal hyperplane and the horizontal distances of the intra-domain target samples to the source samples, which are determined by Hinge Loss and Pythagorean Theorem. Experimental results show that the proposed HyperDisGAN consistently yields significant improvements in terms of the accuracy (ACC) and the area under the receiver operating characteristic curve (AUC) on two small-scale natural and two medical datasets, in the hyperplane spaces of eleven downstream classification architectures. Our codes are available in the anonymous link: https://anonymous.4open.science/r/HyperDisGAN-ICLR2024.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a framework for data augmentation using GANs, inspired by CycleGANs. The main part of the method is to provide an explicit control between intra-class augmentation and cross-domain augmentation. This controllability is achieved by accessing the decision boundary of a pre-trained hinge classifier. The proposed data augmentation mechanism is then evaluated on several binary classification problems on image data.

### Strengths
* The proposed method is sound and reasonable. 

* The experiments show that the method allows to improve the performance of a base classifier, and that it (often) outperforms other augmentation strategies.

### Weaknesses
* For now, the method is limited to binary classification. Could it be extended to multi-class classification? Self-supervised learning? Which are settings that benefit a lot from data augmentation. 

* Experimental validation: I feel that the comparisons are not strong enough and would require a more careful and detailed evaluation. Notably, it is surprising that the Traditional augment often lowers the performance of the standard network, while data augmentation are most of the times beneficial in deep learning. It seems that the authors could have tried different settings and could have tuned better traditional data augmentations. For example, you test different hyper-parameters on your method. You should also test different hyper-parameters on the concurrent methods, otherwise it is natural that the max of HyperDisGan is higher than the max of the other methods. 

* The proposed pipeline is a quite heavy pipeline with lots of hyper-parameters to tune. 

* Clearness: the paper would benefit, at the beginning of Section 3, from a clear definition/formalization of the setting of the proposed method, e.g. input/output of the generator. This comes partly in the section 3.3, but it is not completely clear. For example, it is stated that the generator is a function from X (or Y) to X (or Y), but then, in the loss functions, the generator takes as input two variables, such as $G_{x2x}(x_1,-d_h(x_1,x_2))$. It is only in Section 4 that it is stated that the generator takes as input an image along with a distance variable replicated on the spatial dimension. 

* Minor remarks on writing/typos: several use of "an" instead of "a", e.g. Figure 2 caption: "An data augmentation" or Section 3.2 "an pre-trained"; page 5 "intro-domain"; page 5: "transformating" -> "transforming"; Table 2: "HyerDisGan" -> "HyperDisGan".

### Questions
* What about latent transformations in pre-trained GANs? It has been shown that hyperplanes separate classes or modes in the latent space of a pre-trained unconditional GAN. Could you extend your method to leverage pre-trained generators? For practitioners who want to apply your method, it would be way less costly.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
1: strong reject

### Rating Number
1

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper studies data augmentation with generative adversarial networks (GANs). The paper proposes a Cycle-GAN based method, HyperDisGAN, which takes into account the hyperplane distance between and within classes to generate samples that are useful for training downstream classifiers. HyperDisGAN uses a classifier pre-trained by hinge loss to learn transformations so that the distance on the hyperplane is as large as possible for inter-class sample transformations, and so that the distance is small for intra-class sample transformations. Experiments were conducted mainly to compare the proposed method with the Cycle GAN baselines and confirmed that the proposed method slightly outperforms the baseline in accuracy and AUC.

### Strengths
+ The paper proposes an interesting data augmentation method based on CycleGAN that generates inter-class and intra-class samples by transforming real samples.

### Weaknesses
- The motivation to introduce the proposed method is weak. The issue presented by the paper is expected to be solved by a data augmentation method that interpolates between samples, such as mixup [a], but the paper introduces data augmentation by generative models without discussing this perspective at all.
- The proposed method is not practical. The proposed method introduces a pre-trained classifier $C_\text{aug}$, an inter-class generator $G_{x2y}$, and a intra-class generator $G_{x2x}$. These are not practical because they increase in proportion to the number of classes in the downstream classification task. In fact, the paper only evaluates the data set with a small number of classes.
- Despite the significant increase in computational complexity, the performance gain given by the proposed method is negligible.
- Experimental baseline is insufficient. Since the proposed method is a type of data augmentation, its performance should be evaluated by comparison with traditional/generative data augmentation methods. For example, traditional data augmentation methods such as mixup [a], CutMix [b], SnapMix [c], and generative data augmentation methods such as MetaGAN [d] and SiSTA [e] are appropriate as experimental baselines.
- Writing quality is poor. The paper contains many undefined words (e.g., "domain," "location," and "hyperplane"), which confuse the reader. In addition, the overall algorithm and procedure are not explained, making it difficult to grasp the overview of the proposed method. In general, the paper does not meet the quality required for an academic paper.

[a] Zhang, Hongyi, et al. "mixup: Beyond empirical risk minimization." International Conference on Learning Representations (2018).

[b] Yun, Sangdoo, et al. "Cutmix: Regularization strategy to train strong classifiers with localizable features." Proceedings of the IEEE/CVF international conference on computer vision. 2019.

[c] Huang, Shaoli, Xinchao Wang, and Dacheng Tao. "Snapmix: Semantically proportional mixing for augmenting fine-grained data." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 35. No. 2. 2021.

[d] Zhang, Ruixiang, et al. "Metagan: An adversarial approach to few-shot learning." Advances in neural information processing systems 31 (2018).

[e] Thopalli, Kowshik, et al. "Target-Aware Generative Augmentations for Single-Shot Adaptation." International Conference on Machine Learning (2023).

### Questions
Nothing to ask. Please see the weaknesses.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
1 poor
