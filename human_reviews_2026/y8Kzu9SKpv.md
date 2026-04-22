# InfoBridge: Mutual Information estimation via Bridge Matching

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Diffusion bridge models have recently become a powerful tool in the field of generative modeling. In this work, we leverage their power to address another important problem in machine learning and information theory, the estimation of the mutual information (MI) between two random variables. Neatly framing MI estimation as a domain transfer problem, we construct an unbiased estimator for data posing difficulties for conventional MI estimators. We showcase the performance of our estimator on three standard MI estimation benchmarks, i.e., low-dimensional, image-based and high MI, and on real-world data, i.e., protein language model embeddings.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces a new formulation of mutual information (MI) estimation as a domain transfer problem. In particular, it estimates MI by quantifying the difference between the drift functions of diffusion bridge processes associated with the joint distribution and the product of marginal (independent) distributions of the variables. The idea is interesting and improves previously studied diffusion-based estimators, to which the authors compare by running extensive benchmarks.

### Strengths
- The authors theoretically validate the correctness of their new approach to use Bridge Matching for MI estimation.
- The difference with prior work on diffusion-based estimator MINDE is well-expressed and theoretically grounded.
- A rich set of standard benchmarks and ablation studies has been conducted.

### Weaknesses
- The main weakness in the experimental evaluation is that the authors only consider very large sample sizes; 100k training datapoints, while classical approaches have also been tested for 5k or less. Although the main competitor, MINDE, also used 100k training points, it would be interesting to see if the method can also perform well with significantly fewer samples, e.g., 10k.
- While the MINDE estimator requires only the learned score functions to estimate mutual information, InfoBridge relies on full sampling of two diffusion bridge trajectories to approximate the drifts. It is not clear if this has implications for computational cost and scalability. It would be relevant for the authors to address and discuss the computational cost difference between the two methods.
- The paper refers several times to regularity conditions, such as the existence of finite first moments, as prerequisites for the main theorem (Theorem 4.1). However, these assumptions are not stated explicitly in the main text nor appendix. Since these are key hypotheses for the validity of the theoretical results, it would be important to summarize them clearly.
- The volatility coefficient $\epsilon$ is studied in the 16×16 Gaussian and protein embedding experiments, leading to a rule-of-thumb setting $\epsilon = 1$. However, the paper does not provide a detailed analysis of how $\epsilon$ influences the estimator’s bias and variance, nor a principled procedure for selecting it.
- Methods for MI estimation between random variables of different dimension are mentioned but not considered for experiments.
- As mentioned by the authors, heavy-tailed distributions cannot be modelled well.

Minor remarks:

- Some formulas and tables violate the space constraints (see e.g., page 3).
- Some references in the appendix are not rendered correctly.
- Reference: lee & Rhee missing year.

### Questions
- For the experiment in Table 1, some methods do not require training. Do you only provide them with 10k test samples, or do they get access to 110k samples, i.e., the full train and test split. How would the method perform with only 10k samples for training and testing jointly?
- Do you have an intuition as of why higher MI values seem to lead to worst estimation in the image gaussian 16x16 case, for all estimators?
- “First, by leveraging finite-time bridge processes instead of infinite-time diffusion (which in practice is approximated with a finite-time horizon), we provide an unbiased estimator.” Why does discretization on the bridges process not induce a bias while it does for noise-to-data diffusion?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents a novel unbiased MI estimator based on diffusion bridge models. The proposed estimator is tested on different low-dimensional and high-dimensional datasets.

### Strengths
- The proposed estimator relies on diffusion bridge models, thus leveraging a new technique for MI estimation
- The proposed generative MI estimator is unbiased, differently from MINDE which is biased
- The experimental results show the effectiveness of the proposed estimator

### Weaknesses
- Although the proposed estimator leverages diffusion bridge models, which differ from diffusion models, the proposed framework appears as an extension of MINDE
- The paper is not well-written. The clarity could definitively be improved. For instance, SDE is not defined (stochastic differential equation). The equation in line 145 should be written in two lines, as it exceeds the borders of the paper. There are many typos: line 96 (conditioned its on values), line 264 (as a domain transfer task), multiple wrong references to equation (2) while I assume the authors wanted to refer to other equations. The same wrong reference to (2) happens also in line 300. Table 2 reports Mean Average Error, but I assume the authors used the Mean Absolute Error.
- In the experiments  the authors compare the proposed method with f-DIME. However, there is no reference of f-DIME in the related work section, so it should be inserted in the Discriminative estimators paragraph.

### Questions
- What is $\delta()$ in equation (3)? It is not defined.
- In line 148 the authors refer to Problem (2). Does it refer to the second equation? That would be the MI definition. In the line below the authors write “minimizes (2)”, but they are not referring to the minimization of MI, I guess they are referring to the minimization of the long equation in line 145.
- In the pseudocode of algorithm 1 and in the code provided in the supplementary material, the authors perform a random permutation to obtain the samples from the product of marginal densities: 
y_samples_permuted = y_samples[torch.randperm(y_samples.shape[0])].
However, using such a function leads a certain number of elements of the shuffled vector to be positioned in the same initial position (before the permutation). This strategy was highlighted in f-DIME’s paper as problematic and in that specific case the authors showed that using a specific  discriminative architecture that mistake would lead to an upper bound in the MI estimate. For InfoBridge I do not see any upper bound. However, performing such a random permutation is logically incorrect and the authors should perform the experiments after correcting such a mistake. How does the performance of the proposed algorithm change after this correction? If it does not change, why? 
- Where are the experiments justifying the sentence in lines 314-315 about the additional binary input?
- Can the authors show a comparison of the computational complexity between the different algorithms used?
- Why the reference Lee and Rhee has no year?
- The discriminative models used during comparisons have the same size (number of parameters) of the generative estimators?
- Isn’t 100k train samples a very large number? How do the algorithms perform with a lower number of training samples?
- In Figure 2 the authors specify which version of MINDE they are using, but do not specify which f-DIME version. Can the authors specify it?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose InfoBRIDGE, a mutual information (MI) estimator that reframes MI estimation as a domain transfer problem between a joint distribution and the product of marginals using reciprocal processes built from mixtures of Brownian bridges. The required drifts are learned via conditional bridge matching with a neural net; training uses samples from the joint and a permuted batch to emulate independence, and evaluation computes the integral by Monte Carlo over Brownian-bridge samples. The authors argue this yields an unbiased MI estimator (given perfect drift learning) and report results on low-dimensional synthetic, image-based, high-MI benchmarks, and protein-embedding data.

### Strengths
* Conceptual novelty: Clear, principled decomposition of MI into a time-integral of drift differences between two bridge processes; different viewpoint than score-based MINDE.
* Methodological soundness: Builds on established conditional bridge matching; training and estimation procedures are straightforward (Alg. 1–2). 
* Practicality: Uses only joint samples and Brownian-bridge simulation (no explicit density estimation), which is attractive for complex data like images/embeddings.

### Weaknesses
* Unclear dependence on ϵ: The estimator formula scales with 1/(2ϵ). Please analyze or empirically demonstrate ϵ-invariance of the resulting MI and give guidance for choosing ϵ.  
* Bias/variance in practice: “Unbiased” is only in the idealized limit. Did you check the empirical calibration with confidence intervals vs. ground-truth MI? Did you check sensitivity to network size, training steps, and t-discretization?
* Independence construction: The independent process uses permuted pairs (x_0,\hat{x}_1). What is possible bias when datasets contain duplicated or highly correlated samples? Will a two-sample protocol (fresh marginal draws) materially changes results?
* Scope of experiments: The authors mention low-dimensional, image, high-MI, and protein settings; more detail is needed on metrics, baselines, and runtime/sample efficiency, especially versus strong discriminative bounds (e.g., InfoNCE variants).

### Questions
* Is MI provably independent of the choice of ϵ? in Theorem 4.1? If yes, please add a formal statement and a numeric ablation sweeping ϵ.
* What is the estimator variance as a function of t-grid resolution and batch size? Any control variates that reduce variance when t → 1 (where the (1-t)^{-1} factor amplifies noise)?  
* Does training two drifts (joint vs. independent) materially increase sample complexity vs. one-model approaches? Could a shared backbone with binary conditioning stabilize learning?  
* How would you share a NN between variables of different dimensionality?
* How does InfoBRIDGE behave on long-tailed or heavy-tailed marginals, beyond the cited remarks that classical estimators struggle there? Please add such stress tests. 
* Table 1: What are the reported metrics? Please add to caption description of the metric presented in the table.
* Figure 1: why MI estimation undershoots for Gaussians (a,b) but not for rectangles (c,d)?
* Discussion of basic implementation, training, and hyper-parameters details should be included in the main body. Include discussion of the computational cost.

Comment:

* Limitations should be discussed in the main body, not only in the appendix.

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
5

### Summary
This paper propose an unbiased estimator of mutual information constructed on bridge matching.
The evaluation is solely conducted by measuring the discrepancy from a ground truth mutual information.

### Strengths
+ The main contribution is much sought after: An unbiased, low variance estimate of mutual information has impact across fields. 

+ The paper's literature review is apt and does a great job motivating the problem.

+ The paper is full of illuminating insights tying information theory, diffusion models and stochastic calculus.

### Weaknesses
+ The evaluation is solely conducted by measuring the discrepancy from a ground truth mutual information. This is sound an necessary but also limited by itself. The paper would be more convincing if the proposed estimate of the mutual information were 
used as an objective, or its learned representations evaluated on downstream tasks.

+ The work could be made much more accessible. The background section runs two full pages, and the authors only introduce their method in page 5. We suggest the more direct route of introducing a simple example right after the introduction, and introduce the stochastic calculus concepts as needed.

+ The paper differentiate itself from MINDE[1] by 1) being unbiased while the latter is only asymptotically unbiased, 2) By presenting itself as a domain transfer task. Unfortunately, this shift in conceptual framing while thought provoking is only  indirectly evaluated through a reduction of the estimators in variance. This is a bit unfortunate as eliminating bias and reducing variance is enough to differentiate well enough from MINDE.

[1] Franzese, Giulio, Mustapha Bounoua, and Pietro Michiardi. "MINDE: Mutual information neural diffusion estimation." arXiv preprint arXiv:2310.09031 (2023).

### Questions
The typical condition for the application of Girsanov's theorem is that of Novikov. How do you suggest relaxing it to use InfoBridge with long tailed distributions as suggested in section 6?

### Soundness
4

### Presentation
2

### Contribution
3
