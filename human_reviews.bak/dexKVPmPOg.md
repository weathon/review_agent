# Efficient Recomputation of Marginal Likelihood upon Adding Training Data in Gaussian Processes and Simulator Fusion

- Decision: Reject
- Scores: 5, 5, 6

## Abstract
To reduce generalization loss in line with the bias-variance trade-off, machine learning engineers should construct models based on their knowledge of the modeling target and, as training data increases, choose more flexible models with reduced dependence on that knowledge if that knowledge is unreliable. 
To achieve this automatically, methods have been proposed to determine the amount of model's assumed prior knowledge directly from training data, rather than relying solely on an engineer's intuition.
A widely studied approach involves using both a flexible model and a knowledge-dependent simulator, selectively incorporating simulator-generated data into the flexible model's training data.
While neural networks have been used as flexible models, Gaussian processes are also candidates due to their flexibility and ability to output prediction uncertainty.
However, direct methods for adding simulator-generated data to Gaussian process training data remain unstudied. The Subset of Data (SoD) method, the closest alternative, often adds inappropriate data due to its assumption about the true distribution.
The log marginal likelihood, grounded in theory, determines the inclusion of generated data. However, its computation in Gaussian processes is costly. We propose a faster method considering the Cholesky factor and matrix element dependencies.
Experiments indicate that, in terms of MSE, metrics using exact negative log likelihood outperform Subset of Data and other basic methods.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper describes a method to incorporate extra training data to learn a particular prediction tasks using Gaussian processes. The proposed method consists in following a particular approach to decide whether a particular data instance should be incorporated to the training data or not. The criterion followed consists in using the negative log likelihood given by the predictive distribution of the GP after incorporating a particular training instance. This is equivalent to changing the prior GP to another GP that is expected to perform better (since it gives a better marginal likelihood estimation). The proposed method is expensive if a naive implementation is followed, with O(N^4) cost on the number of data points or iterations to follow. The authors proposed a clever implementation that takes into account partial updates of Cholesky factors. The method is validated on synthetic datasets both in terms of performance and in terms of computational cost.

### Strengths
The paper is very well written and clearly explained. Apart from that, I cannot find any other particular strength. My overall impression is that the paper is still in an early stage and needs more work before it can be accepted for publication. In particular, the authors should address the weaknesses described below.

### Weaknesses
The experimental section is too weak. It only considers synthetic datasets. It is not clear at all if the proposed method has a practical utility since no real world problems are considered in the paper. This questions the significance of the results. The authors should given particular examples of the expected utility of the proposed approach in a real-world setting.

        The paper lacks a solid related work section. It is not clear at all if this problem has been already studied in the literature and if some methods have already been devised for it. In the introduction there are some related methods described. However, they seem methods proposed for a different setting that may be adapted to the particular setting considered by the authors.

        The proposed method has a very large computational cost that is cubic w.r.t. to the number of training points or the points to be added to the training set. This is a limitation since only a few thousand points may be considered at most. The authors should try to scale the method to larger experimental settings, considering e.g., approaches for sparse GPs.

        The use of Cholesky factors that are updated efficiently is not new within the GP literature.

### Questions
Why do not each method in Fig. 1 start from the same initial value?

Could you approach be extended to take advantage of sparse GPs approaches to scale to large datasets?

### Soundness
2 fair

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work proposes to use the negative log marginal likelihood of the Gaussian process as a criterion when selectively adding simulator-generated data to the training data. Since evaluating each candidate training data point using the negative log marginal likelihood can be time-consuming, the authors propose a method for fast computation by considering the so-called Cholesky update and take advantage of the dependencies between matrix elements.

### Strengths
Originality: Probably, the faster re-computation of the marginal likelihood might be the originality of the work. 

Quality: The experiments provided in the paper are useful to provide an idea of the approach, though they are limited to just presenting synthetic scenarios. 

Clarity: The methodology sections are generally well written and not difficult to follow, though they present some inconsistencies in the mathematical notation.

Significance (importance): The work has its strength in the efficient computation of the marginal likelihood.

### Weaknesses
-The idea of accepting data to be added as part of the training set by improving the marginal likelihood was previously explored by Titsias M. in "Variational Learning of Inducing Variables in Sparse Gaussian Processes" section 3.1. In the context of Titsias' work was used to generated pseudo-inputs (or inducing points).

-The experiments provided are limited to just presenting synthetic scenarios. 

-The introduction does not properly motives the research problem to engage the reader with the work. Also, the introduction lacks of references to better support different phrases or claims. The methodology sections are generally well written and not difficult to follow, though they present some inconsistencies in the mathematical notation.

-The work has its strength in the efficient computation of the marginal likelihood, but the main aim of the work was not to compare such an algorithm with other approaches that improve such computation, but to introduce a direct method of selectively adding simulator-generated data to training data when using Gaussian processes.

### Questions
---Specific comments---

-In Abstract, it sounds contradictory to say that we rely on knowledge that is unreliable: "construct models based on their knowledge of the modeling target and, as training data increases, choose more flexible models with reduced dependence on that knowledge if that knowledge is unreliable."
Maybe the last sentence would be better understood if read as:"...if that knowledge becomes unreliable" or simply get rid of last part and leave: "construct models based on their knowledge of the modeling target and, as training data increases, choose more flexible models with reduced dependence on that knowledge."

-In Abstract, it is not clear what it is the intention of "We propose a faster method considering the Cholesky factor and matrix element dependencies." There is something missing to properly connect with all the previous text.

-In the Introduction, there is probably a sentence missing at the very beginning regarding modelling issues or modelling challenges than allows the reader understand where the idea or problem of bias-variance trade-off comes from. Also, it is necessary to include a strong reference regarding "bias-variance trade-off" to support the text.

-In the introduction, the phrase that reads: "On the other hand, the method of selectively adding generated
simulator data to the training data only requires that data can be generated from the simulator" seems ambiguous or needs rewording.

-Please include references to support: "The criterion for selecting important data is the diversity of the training data. Various methods to measure this
diversity have been proposed."

-Please include references to support: "The negative log marginal likelihood is a metric that measures the model’s ﬁt to the training data and has a theoretical foundation that it matches, on average, the KL divergence between the true distribution and the model’s distribution."

-Where it reads: "Within this category, although
Auto Data Augmentation is efficient, The knowledge transferred", lower case "..., The knowledge..." to "..., the knowledge..."

-Introduce the acronyms KL, BIC, GPs, NLL and NML!

In section 2.1: 
-There seems to be inconsistency in the notation. I do not see the benefit of referring to $\mathbf{X}$ as a random variable. There is no information or specification of the distribution that $\mathbf{X}$ follows. I would suggest to refer as an input variable $\mathbf{x} \in \mathbb{R}^d$ instead of $\mathbf{X} \in \mathbb{R}^d$. Also $y \in \mathbb{R}$ instead of $\mathbf{y} \in \mathbb{R}^1$, these to be congruent with Eq. (1).

-Also, I suggest to use $\mathbf{y}^N=(y_1,y_2,...,y_N)^\top$ and $\mathbf{X}^N=(\mathbf{x}_1,\mathbf{x}_2,...,\mathbf{x}_N)^\top$ to be more consistent instead of the current notation in the paper.

In Eq. (1), the Covariance matrices $\mathbf{K}_{N,m^*}$ 

and $\mathbf{K}^{\top}_{N,m^*}$ 

might be swapped of quadrant. 

It is more intuitive to think that the pair $N,m^*$ refers to rows,columns respectively. 

Add period "." at the end of the equation. 

Putting the $\mathbf{x}_1...\mathbf{x}_N$ and 

$\mathbf{x}_{1^*}$ ...  

$\mathbf{x}_{m^*}$ 

inside the equation looks strange as if a vector were multiplying the covariance matrix. Maybe a footnote should be added to avoid confusion.

-If $\mathbf{K}_{N,m^*}$ 

is swapped by $\mathbf{K}^{\top}_{N,m^*}$ then the equations that use these matrices should be corrected.

-Similar comment to the one before applies to Eq. (2).

-After Eq. (2) in $F_{m+1^*}$ the $y^{m+1}$ is missing "*".

-In Eq. (2) and (3) the identity matrices $\mathbf{I}$ should be different at each quadrant since they do not have the same dimensions.

In Eq (3) the negative sign is not applied, previously it was introduced $F_{m+1^*}=-\log \mathcal{N}...$. Also, as per Eq. (4) the operation in Eq. (3) should be 

$(\mathbf{y}^{m+1}-\boldsymbol{\mu}_{m+1})$

instead of 

$(\mathbf{y}^N-\boldsymbol{\mu}_{m+1})$

-Add a comma "," after Eq. (3) and (4), then period "." after Eq. (5).

-In section 3: write $(m+1)\times(m+1)$ instead of $m+1 \times m+1$. Indeed, in the equations should be better to write, say, 

$\mathbf{y}_{(m+1)^*}$ or 

$\mathbf{K}_{(m+1)^*}$.

-In section 3: it reads: "with a total cost of 

$\mathcal{O}(M^2N + MN^2)$, 

keeping it within the cubic order", shouldn't it be within the quadratic order?

-Before Eq. (6): what is $\mathbf{K}_{+m}$? typo?

-Before section 3.2: 

$(\mathbf{L}_{m+1}$ 

$\mathbf{L}^{\top}_{m+1})^{-1}\mathbf{y}^{(m+1)^*}$ 

instead of 

$(\mathbf{L}_{m+1}$

$\mathbf{L}^{\top}_{m+1})^{-1}\mathbf{y}^{m+1}$

-Where it reads: "Lalchand \& Faul
(2018) described in Section 1, promote diversity of training data." should be "promotes" since you are referring to the method or work.

-Typo where it reads: "then using the likelihood of the all output data y", should be "...of all the output data..."

-In section 4.2: "the number of training data candidates generated from the simulator was 1,000,", you mean "1000" or 1?

-In the figure 3: it is not possible to visualise the Training data (brown-ish colour) for SoD. 

-In the conclusion: "the algorithm we proposed is specialized for regression models", not regression models in general, but a regression model particularly with a Gaussian likelihood.

-Generally, there is either a comma or period missing after the equations.

-Initial capital letter in the bibliography, words like: Gaussian and Cholesky. 

-Why is there a distribution $q(\mathbf{X}^N)$ in appendix H for Eq. (21)? Aren't we saying in $KL(q(.|\mathbf{X}^N)||p(.|\mathbf{X}^N))$ that $\mathbf{X}^N$ is given? I do not think the Eq. (21) is correct.

---Other Questions---

Is this method feasible to different statistical data types for the outs $y$ or we should assume that $y$ is always in the real values?

We fit the GP hyperparameters with the training data, but are those hyperparameters tuned again when adding simulator data?

-The experiments shown seem to have an appropriate number of N data observation so that the GP model fits quite well for the range of input data $\mathbf{x}$, so due to the conditioning properties of a Normal distribution it is expected to only accept data that could improve the conditional distribution $p(\mathbf{y}^N|\mathbf{X}^N,\mathbf{y}^{m^*},\mathbf{X}^{m^*})$. What would it happen if the GP has a smaller number of data observation, or lack of data in regions such that the predictive distribution was less uncertain? How would the acceptance and rejection would behave in such a region?

-It seems that the Log marginal likelihood metric gives priority to the model fitting, so when do we trust the simulator?

-What if the simulator is actually quite close to the true distribution, but we have a small number of data observations for which we fit a GP with the hyper-parameters tuning a distribution not that close to the true distribution?


What ways to measure a trade-off, as mentioned in the introduction, to achieve an appropriate bias-variance in our last model that contains training and simulator data? 

-If I fit the GP and generate data from such a GP and use it as simulator data, wouldn't I expect to achieve improvements in the Log marginal likelihood? Wouldn't the fitted GP be simply the best data simulator?

-The work is missing to show a real world application to additionally assess the performance of the approach, for instance an example as claimed in appendix C.

-What would be the effect of using different data simulators? For instance, a simulator less similar to the real distribution. 

-For the practitioner, How is a data simulator generally built or where does it come from?

-What if the dataset we are fitting presents a heteroscedastic noise, how could this affect the method approach for accepting training data candidates?

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
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper focuses on selectively adding data for training a low-variance model, which is an important topic. The authors propose to deploy GP along with marginal likelihood as the metric to evaluate the quality of simulated data samples. The paper first talks about the method to selectively add more training data using GP. Then, it introduces the algorithm for faster implementation.  The experiments show the improvement.

### Strengths
- The GP for adding simulated data seems to better perform than other baseline methods. 
- The algorithm is faster. 
- The discussion is well-rounded.

### Weaknesses
- The novelty of GP on this topic is a bit limited. GP is not a new method at all. The algorithm that makes it faster is more interesting but no major breakthrough. 
- It seems no real data set is experimented.

### Questions
1. Why gray points are not adopted in Figure 3?
2. It is said that the hyperparameters of GP are learned from initial training data. Do those hyperparameters change after it is learned? If it is not, does the initial training data affect the selection process? If it is not, how does it change?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
