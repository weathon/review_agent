# Neural Mutual Information Estimation with Reference Distributions

- Decision: Reject
- Scores: 1, 5, 5, 5, 5

## Abstract
Estimating mutual information (MI) from data is a fundamental task in machine learning and data science, yet it remains highly challenging even with state-of-the-art estimators. This work proposes a new distribution-free MI estimator based on reference distributions. Unlike existing works that only discern between the joint distribution and the marginal distribution, which can easily overfit in high-MI settings, our method compares them with extra reference distributions. These artificial distributions share the same marginals as the original distributions but have known dependence structures,  providing additional signals for more accurate dependency modeling. Experiments on synthetic tasks with non-Gaussian, high-dimensional data and real-world applications including Bayesian experimental design and self-supervised learning demonstrate the potential of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
1

### Rating Number
1

### Confidence
5

### Summary
This paper considers alternate marginal-preserving base measures for the ratio-based MI estimator of Srivastava et al. Unfortunately, this paper does not provide much evidence for this choice, and the paper is dishonest in its obfuscation of its relationship to Srivastava et al., and the trivial theoretical section is apparently obfuscated to falsely appear to be more complex. While I do think the underlying idea is perhaps worth exploring, there is little in the paper as currently written that could be viewed as a contribution.

### Strengths
The idea of marginal preserving reference distributions might be worth exploring

### Weaknesses
While the idea is nice, there is little to no discussion or experiments exploring why the choice of marginal-preserving is better than the choices in [23]. The theoretical results are not related to this choice, and the experimental results are limited to hand-designed parametric distributions. The advantage in the BED results is not clear and not sufficiently explored, and the SSL results do not include [23] as a baseline and so do not contribute. 

Throughout the paper, its relationship with [23] is hidden rather than discussed. The paper seems to present the idea of using reference distributions as its own (see eg title), rather than an exact copy of the framework of [23] (the only difference is the choice of reference). [23] is mentioned briefly in the related work at the end of the paper, but should be front and center in the introduction with extensive discussion motivating the change in reference distribution. Indeed, this type of discussion should have been the key contribution of this paper as it would have helped the field move forward towards better reference distributions!

The theoretical results continue this pattern of dishonesty. Proposition 1 seems to be just a direct repackaging of a result from [23], obfuscated by an additional term that is identically zero so that the triangle inequality could be invoked. Proposition 2 is trivial and does not show anything, this fact is obfuscated by the false complexity of this constant \lambda which is trivially equal to 3. 

While this pattern is enough for me to not trust the experimental results, the experiments are also very lacking, mostly being tests of simple synthetic distributions. The BED experiment is not diverse or repeated, so it is impossible to tell if the method simply got lucky. Section 6.3 does not compare to [23] and therefore adds nothing.

Please use brackets [] for references to papers, not (). Using () - the same as equation references - is confusing. 

Lastly, the two-step framework is not valid unless different data splits are used for the two steps. If the reference distribution depends on the samples of the distribution of interest, the samples are no longer independent and the density ratio proof does not work.

### Questions
see above

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
In this work, the so-called *high-discrepancy issue* is addressed,
This issue arises from the task of mutual information estimation between high-dimensional or highly dependent random variables,
and has been shown to target critic-based parametric mutual information estimators.
The authors suggest resolving this problem via introducing additional reference distributions to the conventional critic-based framework.
The distributions are selected to mimic the marginal distributions of the original random variables.
Throughout the work, it is argued that such approach reduces the risk of overfitting, thus mitigating the high-discrepancy issue.

### Strengths
1. The main idea and the underlying intuition are presented clearly.

1. The authors employ their method in several setups with different MI-related tasks,
   including plain MI estimation benchmarking on synthetic data,
   Bayesian experimental design, and self-supervised learning.
   The results indicate moderate practical advantages of the proposed technique.

### Weaknesses
1. The synthetic examples used to evaluate MI estimators can be considered outdated.
   A year ago, a comprehensive set of MI estimation benchmarks has been proposed in [1].
   Additionally, recent papers on MI estimation feature high-dimensional image-like datasets,
   see [2] or (20,43) from the in-paper list of references.
   Finally, there is a very recent article which also proposes new image-like datasets to benchmark MI estimators [3].
   All these tests cover many important corner cases, such as different types of long-tailed distributions,
   sparse interactions, high dimensionality, non-trivial manifolds of images.

1. Proposition 2 seems to be a direct corollary of the telescopic summation equality and the triangle inequality.
   This result does not utilize the fact that $q$ is chosen to have the same marginals as $p$.
   Thus, it gives no insights why the proposed method should work well.

1. As a consequence of the previous point, the novelty of the proposed estimator is questionable.
   Without a proper theoretical justification,
   this estimator can be seen as a version of the Multinomial Ratio Estimator with some moderate technical improvements.

1. The overview of the related works is somewhat misleading.
   Firstly, not every parametric MI estimator falls into the category of critic-based methods (lines 35--36).
   For example, MINDE estimator cited on the same lines does not employ critic networks
   and does not optimize any variational lower bound of MI (lines 37--38).

   Secondly, DV and NWJ estimators can also be viewed as the density ratio estimators (lines 73--90):
   it is known that the optimal $f$ approximated by these methods is, up to some constant,
   the density ratio in question (see section E.1 in [1]).
   The difference between DV/NWJ estimators and the cited density ratio-based approach (from the work of Hjelm et al.)
   is that in the second case, a variational bound on the Jensen–Shannon divergence is optimized
   instead of the variational bound on the KL divergence.

   Finally, the normalizing flows estimator from (43), cited on lines 236--238, is not based on density estimation.
   The technique proposed in this work allows for either a direct MI calculation
   (through the covariance matrix formula after the bi-Gaussianization)
   or a direct point-wise MI ("density ratio") calculation.
   PDF estimation is just a by-product of this method, and is not used to estimate MI.

1. Additional complexity arising from the usage of the generative models is under-addressed.
    Section B.3 should be extended to (a) include all the relevant experimental details,
    (b) provide a comprehensive analysis regarding real-life complex and high-dimensional datasets
    (such data will definitely require more sophisticated generative models to be used).

**Minor:**

1. Judging by the current state of the Bayesian experimental design section,
   it is hard to understand if the proposed estimator offers any significant practical advantages.
   The work would greatly benefit from the direct downstream performance comparison conducted for both of the tasks.
1. In equation (2), it should be mentioned that the equality holds if $\mathcal{F}$ is wide enough.
   The same goes for the NWJ bound (lines 79--80).

### Questions
1. What does the conditioning on $h$ in the equation (7) mean?
   From the current notation, I see that $h$ is not a random variable, but a classification (logit) function.
   The same question for similar cases of conditioning (e.g., line 365).
   Additionally, this notation seems to be inconsistent: see lines 714--716.
1. I am puzzled by the usage of the law of large numbers in Section A.1.
   It seems that the expectation and the empirical expectation (averaging) is used interchangeably (lines 741--746).
   Also, plugging $\hat I$ from lines 725--727 to the expression on lines 730--732 sets the first right-hand-side term to zero.
   I kindly ask the authors to address my concerns regarding this proof.
1. How is the time complexity in Section B.3 measured?
   It is known that InfoNCE is more resource-intensive, but it also usually requires much fewer epochs to converge.
   Are such properties fairly represented in the experimental setup?
1. I do not understand the claim on lines 80--82: as DV and NJW both approximate lower bounds on MI,
   both of them should be biased.
   I kindly ask to clarify the statement and provide a corresponding citation or proof to support it.

**Additional references:**

[1] Czyż et al. Beyond Normal: On the Evaluation of Mutual Information Estimators. *Proc. of NeurIPS 2023.*

[2] Butakov et al. Information Bottleneck Analysis of Deep Neural Networks via Lossy Compression. *Proc. of ICLR 2024.*

[3] Lee K., Rhee W. A Benchmark Suite for Evaluating Neural Mutual Information Estimators on Unstructured Datasets. *Proc. of NeurIPS 2024.*

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The authors propose a new method to address the 'high discrepancy' that neural esimtaors of mutual information suffer from (i.e., instability in high dims/ high MI regime). The authors incorporate reference distributions that  can be used to address the 'overfitting' that leads to such high discrepancy, among other phenomena. The authors present the utility and power of proposed method on several benchmarks and applications. Furthermore, the authors present several formal guarantees for their method.

### Strengths
- In my opinion the paper is well written and not too hard to follow. 
- The presented results and experiments are easy to understand and the main contribution is overall clear.
- The experimentation is extensive and covers several benchmarks.
- an application to bayesian experiment design is presented.

### Weaknesses
1. Experiments - I would expect a comparison with SOTA MI estimators that were designed to deal with high dimensional data, which is known to be a caveat of neural estimation. The literature is dense. but I would choose 1-2 estimators to compare performance with (e.g. the unnormalized statistics estiamro or a normalizing flow-based estimator [1,2,3]), as the current estimators you compare to are not the best known for some time now. Such an experiment would help emphasize the advantage over current state of the art.

2. Unfortunately, I am not confident in the overall novelty of this work. The concept of reference distributions was previously explored for neural estimation [4,5]. While this work does not exactly leverage the reference distribution in the same sense as the ones referenced I would expect to see the contribution on high dimensional real world data, for which neural estimation is most crucial (when MI is just high but the data manifold is rather 'simple' both classical methods and flow-based method are expected to perform very well) beyond the weighting against the InfoNCE. One concrete example can be the design of an 'infomax-like' experiment, which shows the utility of estimated MI for representation learning, or some SSL learning which shows that MIME helps (while the current SSL experiment shows that the best approach is to consider only InfoNCE - correct me if I'm wrong).

3. I believe some technical claims can be made more rigorous and some important parts of this work can be better streamlined, see the questions section for more details.



[1] Duong, Bao, and Thin Nguyen. "Diffeomorphic information neural estimation." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 37. No. 6. 2023.‏
[2] Guo, Qing, et al. "Tight mutual information estimation with contrastive fenchel-legendre optimization." Advances in Neural Information Processing Sy
[3] Wang, Xu, et al. "Smoothed InfoNCE: Breaking the log N Curse without Overshooting." 2022 IEEE International Symposium on Information Theory (ISIT). IEEE, 2022.‏
[4] Chan, Chung, et al. "Neural entropic estimation: A faster path to mutual information estimation." arXiv preprint arXiv:1905.12957 (2019).‏
[5] Tsur, Dor, et al. "Neural estimation and optimization of directed information over continuous spaces." IEEE Transactions on Information Theory 69.8 (2023): 4777-4798.‏

### Questions
1. The authors discuss the approximation of the llr (proposition 2) - while we can bound the overall approximation with the maximal 'single-llr' approximation, how good do we perform a single-llr approximation?

2. Could the authors elaborate on the contribution of MIME to the SSL experiment? is the bottom line that weighting the InfoNCE with another estimator does not improve performance?

3. In equation (5), what's h_k? is h_i a logit corresponding to the ith class? this is the center of this work and I believe clearly and carefully explaining notation would significantly improve clarity.
4. what is \epsilon_{\leq d}?

5. They authors give formal guarantees (asymptotic) under the assumption of a uniformly bounded classifier. Is it guaranteed by design? can we have any conditions fo this assumption to hold?

6. 'Gaussian in the middle' (lines 211-213) - Are we guaranteed that the Gaussian copula lies between the joint and product of marginals? additionally? in what sense is the distribution between them? DKL does not have a natural geometric interpretation (actually, it does not even follow the triangle inequality..) 

7. In the SSL experiment - you mention the 'true mutual information' - what is the notion of true MI in real world data stem from? do we have any mean of estimating\calculating the ground truth?


technical comments 
1. I think \cal{D} definition (line 68) misses i=1 \to n notation.
2. I think a graph of the error vs n. may benefit to understand the performance of the estimator, while we don't have a non-asymptotic bound it will be good how is the convergence w.r.t. n, especially compared to other known estimators. For the y-axis, the author can consider some standard absolute error w.r.t ground truth (i.e., by consider a simple Gaussian case where ground truth is closed form).

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper under review proposes a method for estimating Mutual Information (MI) based on a multinomial classification problem involving samples from a reference distribution. In a way, it can be seen as a hybrid of a discriminative and a generative approach to MI estimation. The main challenge that this paper is focused on is the so-called high-discrepancy issue, when the difference between the joint distribution and the product distribution is large. The paper included theoretical analsysis (consistency, error bounds) as well as extensive experiments comparing the new approach (named MIME) with existing methods on synthetic data and in the context of some applications.

### Strengths
The proposed approach is novel and very interesting. The ideas of using a reference distribution based on Gaussian copulat theory is original. The paper includes a sound consistency and error analysis and the proofs are, as far as I can tell, correct. The submission includes solid experimental comparisons with popular methods such as MINE, InfoNCE, MRE and Difference of Entropy (DoE) estimators. The experiments are applied with synthetic data and in the context of Bayesian Experimental Design and self-supervised learning. The new method shows promising results and beats some existing methods in the datasets considered. The method performs well, in particular, in addressing the high-discrepancy issue.

### Weaknesses
* The introduction is good, but could be improved a bit further by elaborating on the classification approach to density ratio estimation, as in Srivastava et.al.

* While the method is new and original, it is not clear from the experimental results provided to what extent it constitutes a significant improvement. Most of the experiments are just mix of Gaussians or cubed Gaussians and the results in Swiss Roll not significant. On the other hand, this is somewhat mitigated by the improved time complexity of the proposed method.

* The authors fail to reference the DEMI paper: Ruizhi Liao, Daniel Moyer, Polina Golland, and William M. Wells III. DEMI: discriminative estimator of mutual information. CoRR, 2020 This paper is a classic example of a classifier-based estimator using similar ideas. 

* The paper mentions limitations of generative models such as doing well only on Gaussian mixtures. It seems that only a specific version of DoE is implemented and a new generative model, MINDE (cited in the paper) is left out of the analysis. It is not clear whether the limitations mentioned are inherent to generative models or a feature of the specific implementation of DoE used. 

* The analysis on the last experiment 6.3 is somewhat limited and could be considered as known result In fact, if the mutual information estimates are non-decreasing, one could obtain similar conclusions with other methods. * InfoNCE was well analysed by Poole, 2019 and Song, 2020.

### Questions
In addition to addressing the above mentioned weaknesses,
* In question (5), you don't say what $K$ is.
* Could you relate your work to DEMI (mentioned above)
* A discussion of MINDE as an example of a new generative model for the sake of comparison could be helpful in assessing the merit of the method
* It would be interesting to see how the method performs with distributions such as Student t, where long-tails is known to be an issue.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper presents a novel method, MIME (Mutual Information Multinomial Estimation), for estimating mutual information (MI) in high-dimensional data, addressing challenges in traditional MI estimators, especially in cases of high mutual information or high data dimensionality.

MIME introduces "reference distributions" that mimic the marginal distributions of original data while differing in dependency structures, helping to mitigate overfitting by offering more fine-grained signals for modeling dependencies. MIME formulates MI estimation as a multinomial classification problem, where a classifier is trained to distinguish between samples from the joint distribution, product of marginals, and additional reference distributions.

The authors validate MIME's effectiveness with experiments on synthetic non-Gaussian datasets and real-world tasks like Bayesian experimental design and self-supervised learning, showing MIME's resilience in complex dependency settings where traditional methods often overfit.

### Strengths
The methodology is novel and the problem considered in the work is important for the community.

The MIME  approach demonstrates key strengths, particularly in its innovative handling of high-dimensional and high-MI data, where traditional estimators often struggle due to overfitting. By incorporating reference distributions with similar marginals but varied dependency structures, MIME provides a more nuanced, stable estimation, which is essential in scenarios prone to high discrepancy issues. 

The method’s design as a multinomial classification problem with multiple reference points is an innovative strategy to alleviate the challenges seen in conventional approaches, such as overfitting and limited robustness.

### Weaknesses
Something important which is missing is the discussion about whether the proposed method is or not a bound of the MI. If it is a bound, while Proposition 1 is correct (see comments later), it does not provide intuition about the eq.(4) problem.
It is not immediately clear that the statement in proposition 2 constitutes in practice an advantage. 

One further drawback is the high computational complexity of the proposed method, which combines generative models and discriminators.
 
The work lacks in terms of experimental validation. Given the structural similarity between this work and [43], it would be interesting to see a comparison on the performance. Also, I would include other techniques like the one described in [20]. 
Moreover, the number of experiments on synthetic data is rather limited, for example in the benchmark [49] there are tens of distributions.

Concerning the experiments in Section 6.2 I think that showing the bound on MI is not conclusive. Being there no ground truth, it is not possible to claim that one method is better than an other (what if MIME is over estimating the real MI?) I think in such a simple scenario it would be better to compare estimators of $\theta$ given the dataset X under the different policies, which is ultimately what we care about. 

The limitation in the number of competitors also affect the results in Section 6.3. There are many different neural estimators which could have been used of the maximization of MI between views, showing that MIME (with beta=0.5) outperforms InfoNCE is not sufficient to claim that the method is robust.

### Questions
Can the authors expand the experimental validation as described in weaknesses?

The authors should better clarify the relationship between the technique of eq. (8) and [43], where a mutual information estimator based on normalizing flows is proposed. Can the methods be combined? What are the pros of the (seemingly) more complex procedure in which a flow model and a discriminator is learned?

Can the authors explain how is it possible that in Table 1 (appendix B.3) there is such a huge difference between the running time of DoE and MIME?

I suggest the authors to include some extra details (or assumptions) to make sure that the claim in line 734 is correct (the law of large numbers applies only in certain conditions). Also, is it a typo the \hat{h}_t in line 737?

Is the footnote 1) also containing a typo? (f=log(p(x,y))p(x)p(y)). If this is not the case I don’t understand what is written in lines 406-407.
  
Please clarify line 308 and the relationship with (8)

Is the proposed method a bound of the true MI? Please specify if the result of eq(4) affects the estimate and why the proposed method is not affected by log(B) (cfr [69] and the statement in line 447)

### Soundness
3

### Presentation
3

### Contribution
2
