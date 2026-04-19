# ConjNorm: Tractable Density Estimation for Out-of-Distribution Detection

- Decision: Accept (poster)
- Scores: 5, 6, 8, 6

## Abstract
Post-hoc out-of-distribution (OOD) detection has garnered intensive attention in reliable machine learning. Many efforts have been dedicated to deriving score functions based on logits, distances, or rigorous data distribution assumptions to identify low-scoring OOD samples. Nevertheless, these estimate scores may fail to accurately reflect the true data density or impose impractical constraints. To provide a unified perspective on density-based score design, we propose a novel theoretical framework grounded in Bregman divergence, which extends distribution considerations to encompass an exponential family of distributions. Leveraging the conjugation constraint revealed in our theorem, we introduce a \textsc{ConjNorm} method, reframing density function design as a search for the optimal norm coefficient $p$ against the given dataset. In light of the computational challenges of normalization, we devise an unbiased and analytically tractable estimator of the partition function using the Monte Carlo-based importance sampling technique. Extensive experiments across OOD detection benchmarks empirically demonstrate that our proposed \textsc{ConjNorm} has established a new state-of-the-art in a variety of OOD detection setups, outperforming the current best method by up to 13.25\% and 28.19\% (FPR95) on CIFAR-100 and ImageNet-1K, respectively.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a new scoring method for post-hoc out-of-distribution (OOD) detection, by considering the OOD detection problem as a density estimation over the exponential family. Using the connections between the exponential family of distributions and the Bregman divergence, the original density estimation problem over the exponential family is converted into the problem of finding the Bregman divergence. Then, to reduce the search space for selecting Bregman divergence, the authors propose a pair of conjugate functions and reframe the original problem into the problem of finding the optimal norm coefficient $p$ against the given dataset. The partition function is estimated using the Mont Carlo-based importance sampling technique. Experimental results demonstrate the efficacy of the proposed score, with varying $p$ depending on the dataset (p=2.2 for CIFAR-10, 2.5 for CIFAR-100, 1.5 and 1.8 for ImageNet-1k on ResNet50 and MobileNetv2).

### Strengths
- Proposed a unified scoring method for post-hoc OOD detection using a general exponential family.
- Converted the original search problem over the expansive function space of Bregman divergence into a simple problem of selecting optimal norm coefficient.
- Demonstrated the effectiveness of the proposed score on CIFAR-10/100, ImageNet-1k, and scenarios including hard OOD detection and long-tailed OOD detection.

### Weaknesses
- The main search problem of optimal coefficient $p$ for OOD scoring is remained as a hyperparameter search, which may constrain the practicality of the proposed score. Furthermore, the OOD detection performance (FPR95) is quite sensitive to the value of $p$ (Figure 4), which implies that the performance of the proposed score highly depends on the hyperparameter $p$.

### Questions
- Can the authors provide any reasonable method to choose $p$ given the training dataset and the corresponding features given NN architecture without the hyperparameter search?
- The partition function is estimated using the importance sampling-based approximation. For long-tailed OOD detection when the ID training data exhibits an imbalanced class distribution, I guess the accuracy of the importance sampling-based estimation may decrease given a limited number of tail-class predictions. Given that, can the authors elaborate why their method still outperforms in the long-tailed scenarios?

### Soundness
3 good

### Presentation
3 good

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
The paper studies the task of out-of-distribution (OOD) data detection for supervised learning tasks. The proposed method is essentially along the lines of density level set thresholding, and the density estimates leverage a particular type of exponential family (uniform) mixtures originating from a Bregman-divergence-related framework. After choosing specific parameters for computational tractability and performance, the authors compared their method with multiple baselines and prior works to demonstrate its effectiveness.

### Strengths
The paper reads well and introduces background and prior works properly. In addition, the authors leverage concrete examples and visualizations, such as plots and tables, to help the reader get the most critical points. All these benefit the readability and clarity of the submission. 

Regarding novelty and originality, the proposed method differs from existing approaches algorithmically and originates from a more general framework. It is good that the authors spend efforts building theoretical justifications and showing underlying motivations. 

The experiment results (seem to) suggest that the new method has advantages over the prior works and often outperforms its predecessors, at least within the experimental setting of the authors. Sometimes, there are seven to ten competitors and two to five benchmark datasets, such as ImageNet and CIFAR-10. From my perspective, this is the most substantial contribution added by the paper, showing the method's practicality.

### Weaknesses
One area for improvement is that while the method comes from a theoretical framework, I need to find explicit theoretical guarantees and technical claims to justify its effectiveness. So, the lack of theoretical justification is a weakness worth addressing, potentially deriving some for simple cases like the Gaussian one or explaining why the algorithm tends to perform well for small p values. 

The second weakness is that the prior work (Morteza & Li, 2022) already proposed a method based on Gaussian assumptions and Mahalanobis distance. The extension in this paper, at least logically, is relatively straightforward, i.e., from Gaussian to Exponential Family and from Mahalaobis distance to Bregman-divergence (an extension). Maybe it's worth adding a section summarizing the paper's technical novelty. 

Another weakness is that while the framework's formulation seems general, multiple assumptions come along the way. For example, the authors assumed a uniform prior and set $\psi$ to the $l_p$ norm. Of course, these might be necessary for a computationally tractable approach and could be acceptable in their current form.

### Questions
Following the comments on the weaknesses, it would be helpful if the authors could
- Provide theoretical justification (guarantees) for the proposed method.
- Explain and summarize the technical novelty in addition to prior works.
- List out all the assumptions made leading to the final method.

In addition, it would be good if the authors could make the notations more distinguishable, e.g., addressing the overuse of $\hat{p}_\theta(\star)$.

### Soundness
3 good

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
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors present a data density estimation method targeted towards out of distribution detection. The authors parameterize Bregman Divergences, which in turn are shown to parameterize Exponential Families up to an approximable normalization constant.

The authors conduct extensive evaluations on numerous OOD tasks and include ablations, showing improvements over benchmarks on many tasks.

In general the mathematical presentation is careful, results are contextualized, and the reader learns both about OOD in general and about the specific presented method, picking up some tricks on parameterizing densities along the way.

### Strengths
The authors conduct extensive evaluations on numerous OOD tasks and include ablations, showing improvements over benchmarks on many tasks.

In general the mathematical presentation is careful, results are contextualized, and the reader learns both about OOD in general and about the specific presented method, picking up some tricks on parameterizing densities along the way.

### Weaknesses
Nothing particularly bad and things seem correct and well reported and well explored. 

Mostly, the weakness would just be the lack of discussion of Deep Generative Models. The paper seems to present density estimation as the main challenge of OOD detection. For some of the considered benchmarks such as GEM, the main criticism is the specific distribution assumptions, hence the exploration in this work across the Exponential Family.

On the other hand, Deep generative models (DGMs) are a flexible approach for modeling data distributions without making distributional assumptions. Not all DGMs give the user a computable density, but there are some that do such as Normalizing Flows and more recently methods like Flow Matching, Stochastic Interpolants, and others to name a few. More generally some models give you un-normalized log densities, which also seem to be okay for this work considering that this work is willing to estimate certain normalization constants.

It's totally okay to explore a non-DGM-based method in this work, but I think it would strongly benefit from some contextualization and an attempt to answer this question in at least one way: 

For high dimensional data such as images, why should someone not pick generic deep generative models that admit densities (or un-normalized densities, or maybe log density lower bounds) and why instead should someone stick with search within the exponential family (for which you give good methods, algorithms, etc, and for which you get good results)?

There is some older work on role of DGMs in OOD detection (such as https://arxiv.org/abs/2107.06908) but I think a LOT of progress has been made in image DGMs since then (like DDPM https://arxiv.org/abs/2006.11239, Interpolants, https://arxiv.org/abs/2209.15571, Diffusions in latent space, https://arxiv.org/abs/2212.09748, etc)

### Questions
1)

Could you please clarify this phrase? I re-read it a few times and just did not understand its meaning

"Without loss of generality, we employ latent features z extracted from deep models as a surrogate for the original high-dimensional raw data x. This is because z is deterministic within the post-hoc framework."

2)

Please answer my main question in Weaknesses, on why no discuss of the role in DGMs for flexible density estimation in OOD detection.

3) small comment:
please tell the reader more about why learning the natural parameter of an Exp. Family intractable.

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
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies the problem of density estimation for density-based out-of-distribution detection. The authors firstly point out that existing logit, distance, and density based OOD methods can be either inconsistent or too restrictive. Then the authors utilize the property of exponential family and its relation to Bregman divergence to induce a modeling of the density function using conjugate norms. They further combine different method for the estimation of partition functions and finally experimentally validate their methods.

### Strengths
1. The assumption of exponential family is mild and the authors utilize the property of conjugate functions to derive a concise formulation of density functions.
2. The estimation of partition functions can be combined with different methods, which indicates the flexibility of the proposed method. 
3. The experimental results on widely-used benchmark datasets validate the usefulness of the proposed method.

### Weaknesses
The theoretical results of this paper are based on the exponential family of distribution. I think this is an explicit assumption on the prior distribution, which contradicts your answer to question ♠. An analysis on the potential extension of your results can be helpful.

### Questions
Please see the weaknesses.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
