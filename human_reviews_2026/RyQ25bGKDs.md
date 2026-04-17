# Subjective Neural Networks: Bayesian Dropout with Trust-Aware Opinions

- Decision: Reject
- Scores: 2, 2, 2

## Abstract
Deep neural networks achieve remarkable predictive accuracy but often fail to convey meaningful uncertainty, which limits their reliability in safety-critical applications. Existing approaches such as Evidence Deep Learning (EDL) and Bayesian dropout either treat uncertainty as deterministic evidence or approximate it with sampling, but they lack an explicit interpretation of subjective trust. In this work, we introduce the Subjective Neural Network (SNN), a framework that combines Bayesian variational inference with subjective logic. Neuron activations are controlled by Beta–Bernoulli dropout, where the Beta distribution encodes a subjective trust opinion and the Bernoulli mask determines whether a neuron participates in inference. During prediction, we apply a nested sampling procedure: sampling trust probabilities from Beta distributions, generating dropout masks, and aggregating outputs into Dirichlet distributions. This process produces predictions that can be directly mapped into subjective opinions of beliefs and uncertainty over class labels. Empirical results on image classification benchmarks show that SNN achieves competitive accuracy while providing calibrated and interpretable uncertainty estimates. Our work establishes a principled connection between Bayesian deep learning and subjective logic, offering a pathway toward trust-aware neural networks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduce the Subjective Neural Network (SNN), a framework that combines Bayesian variational inference with subjective logic.

### Strengths
This paper proposed a novel framework to unify the Bayesian Variational Inference with subjective logic.

### Weaknesses
The proposed method is mainly evaluated on small-scale datasets, which makes it hard to convey the effectiveness of the proposed model. 

The motivation is not strong, as to why we need to combine the Bayesian Variational Inference with subjective logic.

The experiment result is weak.

### Questions
My main concern lies in why you want to combine the Bayesian variational inference with subjective logic.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors study uncertainty estimation for image classification. They propose Subjective Neural Networks (SNNs), an approach that combines Beta–Bernoulli Dropout with subjective logic.

They evaluate the approach on MNIST and CIFAR-5 using small CNNs, comparing with MC-dropout and EDL.

### Strengths
- The paper is well written in the sense that it contains basically no typos or similar issues.
- The proposed approach is simple and makes some sense overall.

### Weaknesses
- The paper is just ~7.5 pages long.
- The experimental evaluation is extremely limited, the proposed approach is just evaluated on MNIST and CIFAR-10 using very small CNNs.
- Even in the very limited evaluation, the proposed method does not seem to perform particularly well, the predictive performance on CIFAR-5 is significantly below MC-dropout in Table 1.

### Questions
- Could you please extend the paper to 9 full pages?
- Could you please significantly extend the experimental evaluation with up-to-date datasets and models?

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
4

### Summary
The authors of the paper introduce Subjective Neural Network (SNN), a framework that combines Bayesian variational inference with the domain of subjective logic, in order to tackle the problem of uncertainty quantification within the context of deep neural networks (DNNs). Their method adopts principles from Subject logic, in order to make more accurate predictions, namely the representation of beliefs opinions (instead of fixed probabilities), the epistemic uncertainty and prior information.
In order to do so they introduce a framework at which (a) neuron activations are controlled by Beta–Bernoulli Dropout, an extension of classical dropout where, instead of a fixed rate across all neurons, each neuron has a distribution over dropout probabilities (b) the representation of opinion maps to a Dirichlet distribution. Given that direct inference of the posterior over latent trust probabilities is intractable, they adopt a variational approach and also employ the Kumaraswamy distribution, in order to enable gradient-based learning.
At inference time, they approximate the posterior predictive distribution through nested sampling.
They conclude their work with an experiments section on CIFAR-10 and MNIST data where they test the performance of their method against Dropout and Evidential Deep Learning (EDL).

### Strengths
The authors introduce the idea of combining two frameworks, namely Bayesian variational inference with subjective logic. 
They develop at which dropout probabilities for each neuron are not fixed, but rather stem from a hierarchical distribution ( $z_j \sim Beta(a_j, b_j),   \ p_j \sim Bernoulli(z_j), \foreach j $)
In order to avoid the problem of intractable posterior they use they approximate each $p_j$ with a variational Beta distribution and enable gradient learning through Kumaraswamy distribution which approximates the Beta.
Finally, they use nested sampling for inference.

### Weaknesses
-- The introduction of the paper has no references;
-- Although the idea is interesting, the method does not seem to outperform existing ones at the experiments section (eg. Table 1);
-- Figure 1 is not very clear; moreover dropout is tested against different sets of digits at the digit 9 case;
-- Figure 2 does not include Dropout;
-- In CIFAR case (which is presented at the Appendix) SNN is outperformed across all tests (also same problem is observed as at Figure 5, methods are not tested against the same set of labels)

### Questions
Authors would be kindly asked to

1) please provide computational times of Dropout, EDL and SNN;
2) please include more results with respect to MNIST and CIFAR (more digits and labels at the classification of rotated digits and samples, respectively;
3) please regenerate Figure 1, bottom row so that (a) all methods are tested against the same set of digits; (b) symbols are consistent to digits across graphs;
4) please regenerate Figure 2, including Dropout curve

### Soundness
2

### Presentation
2

### Contribution
2
