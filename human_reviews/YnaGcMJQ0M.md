# Detecting Out-of-Distribution Samples via Conditional Distribution Entropy with Optimal Transport

- Avg Score: 4.50
- Decision: Reject
- Scores: 3, 1, 8, 6

## Abstract
When deploying a trained machine learning model in the real world, it is inevitable to receive inputs from out-of-distribution (OOD) sources. For instance, in continual learning settings, it is common to encounter OOD samples due to the non-stationarity of a domain. More generally, when we have access to a set of test inputs, the existing rich line of OOD detection solutions, especially the recent promise of distance-based methods, falls short in effectively utilizing the distribution information from training samples and test inputs. In this paper, we argue that empirical probability distributions that incorporate geometric information from both training samples and test inputs can be highly beneficial for OOD detection in the presence of  test inputs available. To address this, we propose to model OOD detection as a discrete optimal transport problem. Within the framework of optimal transport, we propose a novel score function known as the \emph{conditional distribution entropy} to quantify the uncertainty of a test input being an OOD sample. Our proposal inherits the merits of certain distance-based methods while eliminating the reliance on distribution assumptions, a-prior knowledge, and specific training mechanisms. Extensive experiments conducted on benchmark datasets demonstrate that our method outperforms its competitors in OOD detection.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes a new OOD detection method based on the optimal transport distance between the test input and that of the ID training samples. The transport plan is calculated by standard entropic regularized optimal transport between the empirical distributions of training and test sets. Empirically, the method demonstrates superior OOD detection performance on small-scale benchmarks such as CIFAR-10 and CIFAR-100.

### Strengths
- The empirical evaluations and ablations on small-scale benchmarks are comprehensive.

### Weaknesses
- The organization and writing of the paper need to be improved. Notations are overly complicated and scattered around. Multiple theorems and definitions can be omitted or miss citations.
  - For example, basic concepts in machine learning such as Definition 3.1 (entropy of two random variables) and Definition 3.4 (condition distribution) can be moved to the Appendix or omitted. 
  - Theorem 3.2 should be cited or put as a footnote or remark, as it is a classic result of the entropic regularized optimal transport problem Eq (3). The dual formulation (Proposition 3.3) should also be cited. It might be better to put the standard formulations and results in the Preliminaries Section instead of the Method Section. 
 
   - The formal definition of the OOD detection problem is missing. Therefore, it could be hard to understand why "the OOD detection problem can be formulated as a discrete optimal transport problem with entropic regularization" in P4 (Sec 3.1). Note that the OOD detection problem is a binary classification problem, which is related but **not equal to** solving the optimal transport problem.

  - A transition is needed from Sec 3.2 and Sec 3.3. It is unknown why optimal transport is related to contrastive learning. It seems ill-motivated why "in this work, we employ supervised contrastive training" (Sec 3.3, P6). Is the feature quality of models trained with standard cross entropy loss undesirable? 

- The methodology is somewhat vague. It remains unclear how OOD scores are calculated. If I understand correctly, $\nu$ (p3) is the empirical distribution of the validation set, where in practice we only have access to the ID validation set. For OOD inputs, how is the distance calculated?

### Questions
- Can authors explain how OOD scores are calculated? As one only has access to the ID validation set, how is the empirical distribution used for OOD detection? 

- Why is the method better than KNN or SSD for hard OOD tasks? (Table 3) In principle, it seems that hard OOD samples are challenging for distance-based approaches due to the proximity of OOD samples and ID samples.

### Soundness
2 fair

### Presentation
2 fair

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
The key idea is to employ optimal transport for OOD detection which is especially useful in settings where OOD detection is perform on an entire test set of samples rather than a single test sample. The intuition is that, when estimating Wasserstein distance between a training ID set and a test set, OOD test points will be treated differently from ID test points. While I like the general direction of research, I have problems with details on the approach, especially empirical evaluation and it's similarities to recent works.

Post Rebuttal:
As per my opinion, the authors seem to lack understanding of literature, missing/ignoring many important baselines. Evaluation is far from thorough. The arguments presented in rebuttal are in particular erroneous. For instance, all the well known baselines include (Garg et al, 2023) are unsupervised, not utilizing or assuming knowledge of a validation set when training OOD detector. Using a validation set obtained from ID set itself~(Hendrycks et al., 2019), is a standard practice for tuning the OOD detector so as to maximize OOD detection on OOD validation set while maintaining 5% OOD rate for ID set. While this is an optional step in all the present methods including (Garg et al., 2023), there is no reason to not utilize outlier exposure in any method. Ignoring this ground breaking work by (Hendrycks et al., 2019) as a justification for missing important baselines is misleading to the community. So I am updating my score to strong rejection. 

DEEP ANOMALY DETECTION WITH OUTLIER EXPOSURE. Hendrycks et al. ICLR 2019.

### Strengths
The general idea of employing information theory for OOD detection, not having to relying upon distributional assumptions, is highly appealing. More such works are required in this direction.

### Weaknesses
My concerns are as following. Similar work is accomplished in (Garg et al, 2023) where, instead of Wasserstein distance, KL-divergence is estimated in its dual form between an ID training set and a test set. From that perspective, these two works are highly similar. Especially, in the dual form, expression for Wasserstein distance and optimal transport are similar. While the authors cite the work in the very beginning of the paper in an abstract fashion, the connections between the two works are completely ignored, be it experiments, theory or methodology. Furthermore, as per my understanding, existing methods perform OOD detection for a given test point whereas the proposed method utilizes entire test set. While it is appreciated that, in continual learning settings, a test set is indeed available for better OOD detection rather than restricting, I would have liked to see details on this in the evaluation setup. How do you relate these two different class of methods in same setup? A lot more details are required on this particular aspects. Furthermore, while the framework of optimal transport is appealing despite being a straightforward extension of the previous work (Garg et al, 2023), the other proposed techniques of using contrastive learning, feature extraction, seem unnecessary. As per understanding, as in (Garg et al, 2023), dual function being a neural or kernel similarity like function, would take care of representation learning (feature extraction, contrastive learning), etc. Otherwise, the approach becomes unnecessarily complex, hard to decipher, and looses it appeal to some extent, especially in the era of end to end learning. Last but not least, experimental evaluation should be more thorough in general and it must include experiments on Imagenet as ID dataset as done in the recent works. In particular, the comparison between CIFAR10 vs CIFAR100 as (ID vs OOD) is not intuitive.

### Questions
Please see above.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes to use discrete optimal transport for OOD detection. The arguments are as follows:

In short, 
- a test dataset is not OOD if it equal or close in empirical distribution to the training set 
- W_p is a measure of distance between two distributions 
- That distance is the inf over couplings of a distance between samples , where a coupling is a joint over pairs from two distributions.
- OT algorithms recover this inf coupling which lets one compute a distance
- Under the coupling, given a test point (a point from one distribution) the higher the entropy of the train points (the other distribution), the "less paired" the two distributions are under the couplings
- This means the two distributions are sufficiently different 
- The conditional entropy is taken to be a scoring function for OOD detection of an instance from a test set.

### Strengths
I really like this paper. While OT has caught on in generative models, it is nice to see it used in distinct but closely related fields like OOD detection. 

The presentation of W_p + OT is clean (crisp + no extra details).

The use of the entropy of the transport plan is well motivated.

The numerical experiments are extensive.

### Weaknesses
In general the paper is good.

1. (exploring relationship between OT and dim. of encoding in more detail)

Dimension is not discussed too too much in this work, but greatly affects usefulness + ability-to-solve OT and approx. OT such as Sinkhorn. Given that I believe the work is fairly complete as far as other aspects of experiments go (datasets, benchmarks etc) I will choose to push on this. Do you have any experimental results varying the encoder/feature-space dimension? It would be great to know how this affects performance, especially for higher dim image datasets.

2. Could discuss the role of the whole test set in making decisions on each test point, in more detail. See QUESTIONS.



3.  (less a weakness, more a suggestion for helping readers)

If there is a main thing to improve, I think it would be help the not-familiar-with-OT reader gain a little more intuition for higher entropy of the transport plan (resulting from running Sinkhorn, etc) correlating with more-likely-to-be-OOD.

There is a worked example with a few datapoints in figure 1, that is good. You could maybe supplement it and help the reader by pointing out some other visual examples like e.g. if P and Q are same and you have very large N for both, you would get closer to the identity mapping being the solution , which has zero variance/entropy, etc...

In short, I think your typical reader might be quite familiar with OOD but not familiar with OT so you could add just a little more hand-holding in that part of the text since the main intuition for the method comes from understanding why the conditional entropy should be high or low.

### Questions
- See Weakness (1) (discussing dimension of encoding)

- Related to the dimension of the encoding is the dimension of the data. Could you clarify all image dataset resolutions? E.g. for LSUN and Imagenet? 

-  If I understand correctly, the score for one test point is determined in part by the whole test set's W_p distance to the train set. This is both interesting and raises questions. One analogy is the subset of OOD methods that try to fit a density to the test points to do likelihood ratio tests (the density is evaluated on each test datapoint but is the result of learning on all test points). 

If I understand correctly, though you briefly acknowledge using geometry of the whole test set as a motivation, you eventually do not return to a discussion on this particular aspect of the setup. I think it's worth differentiating between methods that use the whole test set for each test point's decision or not. Could you provide more discussion on this (what to take away from it, and what could be inspired by it in the future)? You could potentially consider introducing some new tasks as well such as making decisions on sets of points. Apologies if you discuss this in more detail and I missed it.

- (minor) Could you consider changing the terminology "score function" to "scoring function" or something similar? "Score function" already has a few meanings so for example it's hard to read "conditional distribution score function" and not think of "nabla_u log p(u|v)

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
4 excellent

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work proposes to perform OOD detection using _conditional distribution entropy_ score, which is the average entropy of the conditional distribution of training data given the test example (Eq 4) under the optimal transport probability (Prop 3.3) which is computed based on the pairwise cosine similarity matrix based on the learned feature representation of a DNN extractor. Furthermore, author proposes to learn the feature representation using supervised contrastive learning (SupCon, Section 3.3). Authors conducted experiments on simple academic benchmarks (CIFAR-100 vs SVHN/iSUN/LSUN/Textures/Places365) using basic architecture (ResNet18), and compared to previous methods with and without contrastive learning, illustrating strong performance. Authors also conducted detailed ablation study on the effect of temperature, contrastive learning, training recipe, and choice of metrics.

### Strengths
1. A novel approach to OOD detection via conditional entropy under optimal transport distribution.
2. The technical approach is clearly described with sufficient background information and detailed derivation.
3. The experiment section is done with extensive baselines and academic benchmarks.

### Weaknesses
1. The experiments are mostly based on basic architecture and rather stylized academic benchmarks. Experiments on larger models (ResNet-50 / ViT) and more realistic benchmarks (e.g., ImageNet-O) can make the result considerably more convincing.
2. It would be great to see an ablation study of how OOD performance is impacted by the choice of "distance matrix" (C vs P)  and the choice of metrics (probability, entropy, conditional probability, conditional entropy). This may also be related to my question about Table 4 below.
3. In author's description of Section 3.2, there is a slight notation disconnect for how $\pi(u|v)$ / $\pi(u, v)$ / $\pi(v)$ are related to $P^*$ described in previous section. It would be great to spell that out explicitly in terms of how these quantities can be obtained from the matrix (e.g., in the paragraph "Uncertainty Modeling" ).

### Questions
Table 4: How are the result from "w/o OT" rows obtained? Specifically, what metric is used for OOD detection?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
