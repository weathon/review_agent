# A Latent Generative Model for Closed-set and Open-set Recognition

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 8

## Abstract
The classic recognition problem assumes that all possible classes in testing are known in advance during training, which can be termed closed-set recognition (CSR). As a natural extension, open-set recognition (OSR) requires models to reject samples of unknown classes that are not encountered in the training phase. Traditional discriminative models struggle to learn decision boundaries for OSR due to the absence of unknown samples. This has led to existing methods focusing on either CSR or OSR, as optimizing one often results in performance degradation of the other. In this paper, we offer a formalization for OSR based on learning theory, demonstrating that CSR and OSR share the same goal for generative models. Motivated by this core insight, we introduce a neural Latent Gaussian Mixture Model (L-GMM) accompanied by a collaborative training algorithm. The model consists of an encoder that maps inputs to a latent space, and a density estimator that computes probability densities. The end-to-end training algorithm, designed in a collaborative manner, learns the density estimator through maximum likelihood estimation and trains the encoder using a discriminative loss derived from the generative model. This framework yields a model capable of performing both CSR and OSR. Experimental results show that L-GMM outperforms its discriminative counterparts in image recognition and segmentation in CSR with models trained from scratch. These models also outperform other specialized methods when directly applied to OSR without any modifications or prior knowledge.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents a method called a neural Latent Gaussian Mixture Model (L-GMM) accompanied by a collaborative training algorithm in order to enhance the performance of open-set recognition (OSR). Typically, the method includes an encoder which maps the input to latent space and a density estimator.

### Strengths
The solution of using latent Gaussian mixture model in the encoder to tackle the close-set recognition is valid.

### Weaknesses
(1) The novelty of the paper is pretty limited as similar ideas for using Gaussian mixture model in variational inference to tackle open-set classification has been studied and published.
For instance, "Open-Set Recognition with Gaussian Mixture Variational Autoencoder" by A. Cao at in AAAI 2021

(2) Compared to the previous work as mentioned, this paper did not provide any new insight and fails to clarify the distinction with the similar paper mentioned above.

(3) The derivation of latent Gaussian mixture model is very common and similar derivation has been shown in previous published work including 
https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=5fda8f8bd7ff10db1ee3ab558fd41e552f008fb3
(i)"Robust Estimation of Gaussian Mixtures from Noisy Input Data", CVPR 2008
and 
(ii) https://ieeexplore.ieee.org/abstract/document/9523628, TNNLS 2023 "Robust Semisupervised Deep Generative Model Under Compound Noise"
These papers have clearly shown that Gaussian mixture model can effectively handle the outliers and noisy labels as well as tell samples from unknown class for recognition in generative model including VAE and GAN. In this case, it is not clear if there is really new insight and contribution that the current manuscript conveys. Modules including discrimination loss, density estimators have also been well studied in the previous work. If we look at the math, there is really very little difference except small tweaking.

### Questions
It is important to increase the novelty by providing new insights with new approaches instead of providing small extensions with existing method.

It is not clear what is the main contribution of this paper as the novelty is too limited and the discussion with the existing methods is also important and necessary.

It is necessary to show the distinction of the current paper from the previous work shown in 3(i) and 3(ii) in order to stand on the safe ground in terms of novelty.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
In this paper, the authors propose a method that can be used for both closed and open set recognition. The authors first employ an encoder to map the data onto a latent space and then a density estimator is applied in the latent space.  The data is assumed to have a Gaussian mixture model. The method can be seen as a hybrid method that combines the generative and discriminative approaches.

### Strengths
The main strengths of the paper can be summarized as follows:
i) Transforming data onto a latent space makes sense especially for high-dimensional inputs since the mean and variance parameters can be better approximated in a lower-dimensional latent space.
ii) The proposed method uses both generative and discriminative approaches and using discriminative approach bring additional benefits for open set recognition.
ii) Accuracies on closed set recognition are comparable to the classical networks using  the softmax loss function.

### Weaknesses
Main weaknesses of the paper can be summarized as follows:
i)  The discussion on related works is limited and misleading. The authors state typical OSR solutions train discriminative models with cross-entropy loss on known classes and a thresholding is applied on probabilities to detect unknown samples. Although earlier open set recognition methods used this approach, recent methods use different loss functions that estimate compact class acceptance regions as in [R1,R2,R3]. These methods are also similar to the proposed method since they employ both generative and discriminative approaches. Some methods that follow this principle are cited in the paper, but they are not discussed.
ii) The strongest aspect of deep learning is the easy creation of the desired feature space. One can enforce the class samples to compactly cluster around some centers as in [R1-R3]. In fact there are theoretical proofs that the data samples cluster around the vertices of a regular simplex [R4,R5]. Therefore, using a single Gaussian distribution is more practical than using a mixture model since the latter has more parameters to estimate. 
iii) The authors used obsolete methods for comparison on open set recognition datasets, and the authors do not use standard open datasets and protocols for evaluation. Please note that, the open set recognition methods are typically tested on Mnist, Cifar-10, SVHN, Cifar+10, Cifar+50 and TinyImageNet datasets. The authors use only Cifar+10 and TinyImageNet datasets and their reported accuracies are very low compared to state-of-the-art, please see [R2].
iv) The improvements over the classical softmax loss function on closed set recognition are very minor and they are not significant.

Minor Issues:
There are some sentences that are vague. They need to be explained more. For example, the sentence given on page 5  “the loss function should output a high cost when the model assigns a high probability for the sample to be any known class.”   First of all which sample, an unknown sample or a sample belonging to a known class. Also, please keep in mind that there is no unknown class sample in the training phase.

References 
[R1] T. Kasarla, G. J. Burghouts, M. van Spengler, E. van der Pol, R. Cucchiara, and P. Mettes. Maximum class separation as inductive bias in one matrix. In Advances in Neural Information Processing Systems, 2022.
[R2] H. Cevikalp, B. Uzun, Y. Salk, H. Saribas, O. Köpüklü, From anomaly detection to open set recognition: Bridging the gap, Pattern Recognition, Volume 138, 2023.
[R3] A.R. Dhamija, M. Gunther, T.E. Boul, Reducing network agnostophobia, Neural Information Processing Systems (NeurIPS), 2018 .
[R4] V. Papyan, X.Y. Han, and D. L. Donoho. Prevalence of neural collapse during the terminal phase of deep learning training. Proceedings of the National Academy of Sciences, 117:24652–24663, 2020.
[R5] H. Cevikalp, H. Saribas, Deep simplex classifier for maximizing the margin in both euclidean and angular spaces, Scandinavian Conference on Image Analysis, 2023.

### Questions
1) The authors state that they apply a threshold to detect unknown samples. What is it? Also using the softmax probabilities is quite wrong for open set recognition since the returned probabilities must sum up to 1. In that case, if the distances from unknown samples to known class samples the returned probabilities will be shared among the classes and this can be smaller than a specified threshold. But, this is quite unlikely in real applications.
2) I wonder how many mixtures the authors used and how that number if decided.
3) Applying proposed methodology to classification is straightforward but I wonder how the authors adopted it for semantic segmentation. Please give more details.
4) Cifar datasets and ImageNet have fixed train and test sets. How did the authors get standard deviations?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The claim of the paper is that it offers a formalization for OSR based on learning theory, demonstrating that Closed Set Recognition (CSR) and Open Set Recognition (OSR) share the same goal for generative models.
The idea is simple: starting from the Bayes formula (a posteriori = likelihood x prior), the authors inject a latent generative model over the likelihood which has an encoder that maps the input x to a latent variable z, and (2) a probabilistic generative model (a density estimator) that outputs a probability density of the latent variable z given the label y. The encoder is driven by discriminative learning, while the density estimator is updated by generative learning. The very innovative part is that the authors use a closed-form distribution for z, which gives great flexibility in the problem modeling, also demonstrating that it minimizes the empirical risk.  In terms of CSR and OSR, the encoder produces discriminative features for both CSR and OSR, and is learned by cross entropy on the discriminative probability, while the density estimator has to recognize samples from unknown classes.
As an instantiation of this idea, the authors propose a Latent Gaussian Mixture Model (L-GMM) using a Gaussian Mixture Model (GMM) as the generative model for very clear reasons: GMM learns very good approximations of any densities, does respect the closed form requirement and is avoiding the mode collapse being composed by multiple components, enforced also by the use of two appropriate regularizers.
The paper offers also a view over CSR and OSR from a learning-theoretic perspective introducing a loss function, the open-safe loss.  In particular, the goal of OSR is to reject samples from unknown classes, the loss function should output a high value. The open safe loss respects this requirement, so that any sample of unknown classes will have a low score for any known class.

### Strengths
A very sound theory, which I tried to summarize above. The main idea is very simple, and acts on the standard Bayes formula.and shows how the likelihood can be factorized into a generative and discriminative counter part. As an example, a novel model. latent GMM, is proposed. The paper is clear about how to proper train such a model

The approach applies to open set recognition, and shows to be effective for that problem. Also, it gives a novel loss for the specific problem, the open safe loss, demonstrating its effectiveness in theoretical terms (it brings CSR and OSR to be aligned for generative models)

Every score about originality, quality, clarity, are high to me, except significance (see below).

The state of the art is fascinating, taking from very old work to recent research.

The experiments are exhaustive: they demonstrate that L-GMM performs better than its discriminative counterparts on CSR (despite being better by a moderate margin, 0.2x% on average) and against SOTA methods on OSR. Two tasks are taken into account: image classification and segmentation. Experiments on Out-of-distribution recognition are also shown, which exhibit potential strong advancement wrt SOTA.

### Weaknesses
The abstract is not clear at all since 1) is not giving a clue about what is the common goal shared by CSR and OSR, 2) which kind of collaboration does hold in the collaborative end-to-end training.

The introduction adds a little about the common goal shared between CSR and OSRt. It seems something related to the risk, saying that considering the risk , the goals of CSR and OSR are identical. In terms of collaborative learning, the introduction helps in understanding that it is a collaboration between a dirscriminative and a generative counterpart. 

There are specific questions on the experiments, that I put below.

### Questions
1)On closed set segmentation, results definitely below the state of the art, but the authors did not explain why (Tab.5). They should. 

2)On Open set image segmentation, results are better than sota, but approaches like segmenter or Mask former are not used. Why? 

3)Finally, the ablative won the number of components are not convincing me, since the differences in terms of top1 accuracy are very similar. Is it possible to show results with 5 and 6 components?

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
3 good
