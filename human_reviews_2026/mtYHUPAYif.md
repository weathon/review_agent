# Epistemic Wrapping for Uncertainty Quantification

- Decision: Reject
- Scores: 2, 2, 6, 4

## Abstract
Uncertainty estimation is pivotal in machine learning, especially for classification tasks, as it improves the robustness and reliability of models. We introduce a novel `Epistemic Wrapping' methodology aimed at improving uncertainty estimation in classification. Our approach uses Bayesian Neural Networks (BNNs) as a baseline and transforms their outputs into belief function posteriors, effectively capturing epistemic uncertainty and offering an efficient and general methodology for uncertainty quantification. Comprehensive experiments employing various BNN baselines and an Interval Neural Network for inference on the MNIST, Fashion-MNIST, CIFAR-10 and CIFAR-100 datasets demonstrate that our Epistemic Wrapper significantly enhances generalisation and uncertainty quantification.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper is about epistemic uncertainty estimation using a combination of interval networks and belief functions via a Dirichlet distribution. The authors propose a new method called Epistemic Wrapping that takes a trained Bayesian neural network (BNN) and transforms its weight distributions first into a dirichlet distribution, then into an interval representation to be used in an interval neural network, from where predictions can be made.

The contributions are:
- Modeling of epistemic uncertainty in parameter space using belief functions
- The Epistemic Wrapper method concept applying to any trained BNN
- Results showing how the proposed method compares with the state of the art

### Strengths
- The paper's writing is okayish.
- Uncertainty in parameter space is an open problem and at least this paper is trying to go in the right direction.

### Weaknesses
- There are some claims that are not supported by citations or are completely wrong:
  * "Still, current efforts model (epistemic) uncertainty in the model’s target space, rather than its parameter space." I don't think this is true, as BNNs explicitly model uncertainty in parameter space.
  * "Since Dirichlet sampling yields interval-based representations" Also I don't think this is true but I could be wrong, sampling from a dirichlet distribution should produce vector values, not intervals, and according to https://stats.stackexchange.com/questions/69210/drawing-from-dirichlet-distribution there are ways to sample vector values from a dirichlet distribution, I would appreciate clarification here or a relevant citation supporting this statement.

- The results are not impressive, actually they are very dissapointing and I believe the baselines are not representative of state of the art performance. In MNIST with a deep ensemble it is very easy to get over 99% test accuracy, as with other BNNs such as Flipout and Variational Inference, while the proposed method struggles to obtain around 90% accuracy on MNIST. Same applies for other datasets like Fashion MNIST or CIFAR10, so overall from the results in the paper, I do not see how the proposed method is an improvement over the state of the art.
- The results improve considerably (still below state of the art) after fine-tuning the INN, so what is the point of the proposed method? Using Epistemic Wrapper basically destroys a lot of knowledge in the network since accuracy decreases significantly, which undermines the need for the proposed method.
- OOD detection results are in the similar trend of being worse than the state of the art, in particular for MNIST vs Fashion MNIST and CIFAR10 vs SVHN, the authors can refer a comparison to https://arxiv.org/abs/2111.09808 , which shows AUROC larger than 0.9 for those datasets, while this paper obtains best AUROC around 0.85, this reference also shows that the main results on accuracy are also far from the state of the art.
- To me the proposed method does not make sense, the paper proposes to take a trained BNN, then transform the posterior distribution of the weights by dynamic truncation, then transform it into a belief function along intervals, then fit a dirichlet distribution using the L-moment method, then sample from this dirichlet distribution to obtain interval distributions and then build an interval neural network. The paper does not argue why these are good ideas or what is the motivation for each step, only describes what is done, but not the "why", and this is much more important considering the poor results.
- The paper is lacking in proper description of the experimental results, it quickly jumps into results, for example in the OOD results, its not clear which datasets are used as ID and which as OOD, these are usually presented as pairs (like MNIST vs Fashion MNIST), but this is unclear from the paper, and only some of the ID-OOD pairs are actually described in the appendix (see captions in Figure 9).
- It is likely that the baselines are not properly trained, specially for MNIST and Fashion MNIST datasets, as the results for the baseline BNN is below what I personally usually obtain on those datasets, a BNN on MNIST should be able to reach at least 97% accuracy, and similarly for Fashion MNIST it should reach at least 80%

### Questions
- Can you provide references on why sampling from a dirichlet distribution would produce an interval value? This does not seem to be backed up by evidence or references.
- How do you explain the results that are significantly below the state of the art? Particularly for MNIST and Fashion MNIST datasets.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
In this paper, the authors introduce an "Epistemic Wrapping", a methodology that is used to improve uncertainty quantification in classification. Specifically, the approach enables the transformation of BNN outputs into belief function posteriors, and by utilizing Interval Neural Networks, it leads to an approach that improves uncertainty quantification metrics over certain image datasets.

### Strengths
1. The paper is easy to follow.

2. Potentially, the approach might be interesting.

### Weaknesses
1. I do not understand the motivation behind the approach in the paper.
The authors say that the "current efforts model epistemic uncertainty in the model's target space, rather than its parameter space", which is somewhat true. But this target space is induced by the parameter space. Existing approaches take the parameter space, and I do not clearly understand in what regard the paper under review differs from existing papers.

2. Conceptually, why is it a good idea to make this "discretization" of the parameter space? This injects additional uncertainty about the discretization, and therefore, it is unclear why one should do it.

3. Experimental results are somewhat strange. The authors report a base accuracy of 72% for the MNIST dataset. But MNIST is known to be a straightforward classification problem to tackle. The fact that the base accuracy is only 72% indicates that something is wrong with the training process. The same holds for other datasets. This looks very strange to me.

4. The paper provides neither pseudo-code nor actual code. Therefore, I am unable to verify the experimental results myself, which is frustrating in light of the previous weakness.

=====

Minor comments:

1. Amusant typo in line 711: "...initially introduced by Dempster (**2008**), and later formalized by Shafer (**1976**)".
2. Lines 174-175: same letters both, for prior parameters and for posterior parameters.

### Questions
1. In what sense do the existing approaches work in the target space?

2. Why, conceptually, should the approach presented in the paper improve uncertainty estimates?

3. Why are the base classification models so bad?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces "Epistemic Wrapping," a new method to improve uncertainty estimation in classification tasks by transforming Bayesian Neural Network outputs into belief function posteriors. Experimental results across several datasets show that this approach significantly enhances both generalization and the accuracy of uncertainty quantification.

### Strengths
* Epistemic Uncertainty in parameter space
* Purely visible improvements in results on public datasets

### Weaknesses
* There is no information on the speed of the algorithm, especially in comparison to other baselines
* No formal derivation on why $\alpha_{k}$ can be expressed through L-moments like it was used in Eq. (6)
* No analysis on why INNs to be better Before Fine-Tuning in Table 1 for Random-Selection, and why INNs are good especially for the deep NNs (n = 8) After Fine-Tuning?
* Some minor remarks:
  * Figure 1 is very small
  * line 130: No clarification of what is "SPDE" 
  * Table 3, for Cifar-10 the last line (BNNR for VGG-16) contains bold font at some incorrect places
  * Table 6, the same issue with bold fonts for Cifar-10, LeNet-5 BNNR

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes Epistemic Wrapping, a method for improving uncertainty quantification in Bayesian Neural Networks (BNNs) by transforming parameter posteriors into belief-function posteriors. The idea is to model epistemic uncertainty in the parameter space rather than only in the output (prediction) space. The approach combines concepts from belief function theory, random set representations, and Dirichlet fitting, leading to interval-based inference via an Interval Neural Network (INN). The authors claim improved robustness, calibration, and out-of-distribution (OoD) detection on MNIST, CIFAR-10/100, and Fashion-MNIST benchmarks.

### Strengths
1. The idea of wrapping Bayesian posteriors in belief functions to capture higher-order epistemic uncertainty in the parameter space is quite novel (at least I haven't seen much).

2. The methodology is general well connected and nicely fit with each other.

3. Across datasets and architectures, this methods yields consistent accuracy and OOD gains.

### Weaknesses
1. While the method combines existing ingredients in an interesting way, several components (belief functions, Dirichlet evidential layers, random-set neural networks, interval networks) have all appeared in prior works. The authors also point out the novel lies in combing these techniques but I think then this is considered as rather incremental work maybe more suitable for a journal paper given nowadays AI conference requires more novel concepts.

2. With so much components is it hard to know which contributes the most to the final outcome, I know it might not make sense to ask the authors to do ablation on the components. But it is not clear which component contribute to what.

3. The writing of this paper is quite dense to parse, with many components referenced from existing work without explaining much of the details or intuition, which make it less accessible for the general audience.

### Questions
Please see my comments for the weakness section.

### Soundness
2

### Presentation
3

### Contribution
2
