# Information Bottleneck for Active Feature Acquisition

- Decision: Reject
- Scores: 6, 3, 6, 8

## Abstract
Traditional supervised learning typically assumes that all features are available simultaneously during deployment. However, this assumption does not hold in many real-world scenarios, such as medicine, where information is acquired sequentially based on an evolving understanding of a specific patient's condition. Active Feature Acquisition aims to address this problem by dynamically selecting which feature to measure based on the current observations, independently for each test instance. Current approaches either use Reinforcement Learning, which suffers from training difficulties; or greedily maximize the conditional mutual information of the label and unobserved features, which inherently makes myopic acquisitions. To address these shortcomings, we introduce a novel method using information bottleneck. Via stochastic encodings, we make acquisitions by reasoning about the features across many possible unobserved realizations in a regularized latent space. Extensive evaluation on a large range of synthetic and real datasets demonstrates that our approach reliably outperforms a diverse set of baselines.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper identifies shortcomings of CMI methods (i.e. it marginalizes unobserved features and thus makes myopic acquisitions, it doesn't distinguish between possible answers) and RL (unstable training) and proposes a simple yet effective method for AFA. They propose to uses several stochastic encoders to embed each feature individually into a latent space and then make predictions. An information bottleneck is added to regularize the latent space. For acquisition, they design an objective based on the sensitivity of the label to individual features and encourage them to pick features that distinguish high-likelihood answers. Experiments show that this method achieves better performance on both synthetic and real-world datasets compared to several baselines.

### Strengths
1. They identify several shortcomings of CMI methods and explain these limitations well with examples.
2. The proposed methods are simple but effective.
3. experiments are thorough.

### Weaknesses
1. This paper should compare to more recent RL baselines, such as [1].
2. The acquisition objective is based on the sensitivity of class wrt features. Could justify why using sensitivity is a good choice here?

[1] Yang Li and Junier Oliva. Active feature acquisition with generative surrogate models. In International Conference on Machine Learning, pp. 6450–6459. PMLR, 2021.

### Questions
1. How do we compute the integral over z in eq 1?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This work considers the active feature acquisition (AFA) problem, where you make selections sequentially based on partial information to achieve high prediction accuracy with a small budget. The authors begin by describing the main existing approaches that use RL or greedily maximize the conditional mutual information (CMI), and point out specific issues that arise with the CMI approach: namely 1) a failure to account for how selections might pay off after one or more additional selections, and 2) a focus on minimizing entropy rather than other measures of predictive uncertainty. 

They then propose a new approach that they call "information bottleneck for feature acquisition" (IBFA). The idea is to train a model that predicts $y$ given a partially observed input $x_S$ via a stochastic latent variable $z$. The model is trained to make accurate predictions subject to a penalty that pushes $p(z \mid x_S)$ towards a prior $p(z)$, typically with a very small weighting $\beta \in [10^{-3}, 10^{-2}]$. At inference time, they make selections based on an expected gradient sensitivity measure of the model's predictions to different latent dimensions, which are set up to correspond to different input dimensions. 

The results show encouraging performance in two senses: 1) fast selection of the correct features in simulated datasets, and 2) high accuracy under small acquisition budgets for several real datasets.

### Strengths
- The beginning of the paper is nicely written. It covers many related works, points out the main issues with each approach, and is generally easy to follow.
- The section on limitations of the CMI approach is useful. It makes the issue of myopic selections more concrete through a specific and intuitive example with the indicator variable.
- The results are consistently strong across a large number of datasets.
- The IBFA approach seems to have some underlying idea that’s useful, although I’m not sure what it is from reading the paper.

### Weaknesses
My main issue with this work is that the IBFA approach is also greedy, only it greedily maximizes a new criterion with no clear probabilistic interpretation. This was a bit disappointing, because the work began by making some reasonable points about flaws in the greedy CMI approach, and as a reader I was hoping for a nice clear solution to those problems. Instead, the authors propose a gradient-based sensitivity measure that seems a bit random, which they calculate in expectation over a stochastic latent variable, and it's not obvious why this works or what it has to do with the problems they discuss for the greedy CMI approach. It empirically works well so there's clearly something to it (although I have a couple qualms with the experiments that I'll mention below), but as a reader I can't tell what's useful about IBFA.

I'll describe a few issues from throughout the paper below.

In the exposition of the AFA problem (Section 3), there are a couple aspects of the presented objective that seem odd. 1) The $S$ optimization needs to be over a distribution of data points, not a single data point as currently shown. 2) The measure of predictive accuracy can arguably be one of a few things, but the max class probability is confusing - why not the probability of the true class? And 3) the constraint $|S| \leq B$ and penalty $\lambda |S|$ look redundant - you could formulate the problem with both, but your experiments only seem to consider the constrained problem (unlike some works that also learn stopping criteria).

In the discussion of limitations of the CMI approach, there are a couple parts that could be improved:

- The claims about the necessity and sufficiency of "considering values of the other unobserved features" aren't substantiated beyond the simple indicator scenario. Relatedly, it's unclear whether IBFA incorporates this idea of considering other feature values beyond the current candidate. Its acquisition criterion considers multiple sampled values for each $z_{G_i}$ which is somewhat like considering multiple values for the unobserved features, but it also samples latents for the known features, and the sampled latents for unobserved features fail to condition on the observed features. So ultimately I'm not sure this whole perspective is relevant to IBFA. 
- In Proposition 2, the criterion you present 1) is intractable, 2) is only claimed to be applicable to this indicator example, and 3) has little to do with IBFA. The failure cases of this approach are easy to see: in a scenario with duplicated features, this measure would assess every feature as having zero value because $I(X_i ; Y \mid x_O, x_U) = 0$ for all $x_U$. Your general point that *selections that fail to account for future selections might be bad* is clear, but this idea doesn't seem to generalize beyond the toy problem and is unrelated to your solution.
- In the part about how minimizing entropy is bad, it would be good to be more precise. In what sense is [0.7, 0.15, 0.15] preferable to [0.5, 0.5, 0]? The current vague explanation is that it's "more favorable for making a prediction." If you formalize that as an alternative uncertainty measure, you could perhaps greedily minimize that at each step and improve over existing CMI methods. You could probably even incorporate it into an existing method like DIME. If you think your approach is effectively doing some related reformulation of the greedy selection criterion, it would great to say so explicitly.

About the IBFA method design:

- Like I said above, the main issue I see with IBFA is a missing probabilistic interpretation for the greedy selection criterion. There's arguably some interpretation to the learning objective (although I think your math is wrong, see below), but I don't see any reasonable interpretation to your selection criterion $R(x_O, i)$ that relies on a heuristic gradient sensitivity measure. The main justification I saw in the paper was that it pays more attention to high-probability classes, but there are easier ways to achieve this. 
- The information bottleneck (IB) idea seems non-integral to the method. I noticed you have an experiment studying sensitivity to $\beta$, which often results in very small values relative to the main predictive objective (more on that below). The reasoning provided in Section 5.3 is also quite vague: "we reason about the acquisition in the latent space, we want to use representations that only contain information relevant to predicting the label." How would the network be incentivized to learn anything else? Also, even though the IB bit helps, it's not central (the results are reasonably good without it), so it seems like an odd choice to name the IBFA method after it. A more accurate name should probably allude to the gradient-based selection criterion, although admittedly I'm not sure what you would call it because it seems a bit heuristic.
- In designing the learning objective, the authors take motivation from the greedy CMI approach when choosing the term $\mathbb{E}[- \log(p_\phi(y \mid z))]$, and they seem to think of this as a form of regularization. That's a very odd perspective, and I'd like to share another interpretation: the first term is simply an upper bound on the standard predictive loss. You can see this in the following derivation: $- \log p_\phi(y \mid x_S) = - \log \int p_\phi(y \mid z) p_\theta(z \mid x_O) \leq - \int p_\theta(z \mid x_O) \log p_\phi(y \mid z)$ (the inequality is just Jensen's). The third term here is what you initially show in line 312 (which is a potentially loose upper bound), and the second is what you eventually use in eq. 3. The exposition of your objective is overly complicated, it seems like you're just learning to predict $y$ via $z$ while imposing some posterior variance via a KL divergence penalty.

There are a couple mistakes in the derivations behind IBFA and its objective:

- In your appendix E proof of Theorem 1, I think your problems start on line 1212 when you conflate underlying probability distributions with your parametric model. Under your model you've factorized the prediction as $p_\phi(y \mid z) p_\theta(z \mid x_O)$, but that's not the same thing as conditional independence of the form $p(y \mid z, x_O) = p(y \mid z)$. The observed variables $x_O$ could provide information about $y$ not captured by $z$ depending on how it's learned. For some arbitrary $p_\theta(z \mid x_O)$ distribution like the one you have at initialization, $z$ certainly does not provide all of $x_O$'s information about $y$.  I believe everything after this point is therefore wrong, including the overall claim. 
- In your appendix F proof about the IBFA objective function, I think your problems start on line 1241 when you claim that $\int p_\phi(y \mid z)p_\theta(z \mid x_O) = p(y \mid x_O)$. Again, you're conflating your parametric model with underlying probability distributions, and you can't guarantee that this holds. I believe this invalidates the later parts of this section and the claim in Theorem 2.

About the experiments:

- I saw the IB ablations in the main text and appendix, and they mostly confirm my concern that the IB part of IBFA is non-essential. In Table 2 it seems helpful, but note that 1) the method basically works without this ingredient, 2) it could be helpful mainly because these are tiny synthetic datasets. In Figure 10 we see that the best $\beta$ values are very small, even for these toy datasets. In the real-data results in Table 4, the benefits of IB are very small - in one case it even slightly hurts performance. Tables 12-13 also reflect that the $\beta$ values here are even smaller than for the toy datasets, with $\beta = 0.001$ being the largest possible value. Overall, IB seems more like a secondary implementation detail of your method, and the paper's focus on it is a bit misleading.

### Questions
Some questions about the experiments:

- I wanted to sanity check some of your results, and I noticed that the DIME and GDFS accuracies for MNIST are both worse here than in the original works (you can see this for 5 and 10 acquisitions). Why do you think that is? 
- I noticed that a pre-processing step for many of your real-data experiments is reducing the number of candidate features using STG. You claim it's so EDDI and the VAE method can work, but I'm curious if it's also hard for IBFA to operate on larger numbers of features. If so, that would be somewhat limiting and merit some discussion. 
- I saw that in Table 5 you provide "scaling laws" (this is called computational complexity) for each of the algorithms. I'm wondering if you could provide wall-clock times, because it's hard to get a sense for how slow/fast IBFA is, particularly at inference time where it requires 1) potentially many latent samples $z$ and 2) iterating over classes to calculate the gradient sensitivity measure. 
- I noticed that you adjusted the class balance for MiniBoone. That's odd, are your results different under the original version of the dataset?
- I noticed that for IBFA you pre-processed features using a copula transform. Did you try ablating this to see if it's important for strong performance? If so, that would be interesting and merit further discussion. I would also be curious to know if it helps to incorporate into other methods, which seems pretty straightforward to try.

A few other questions about IBFA:

- How sensitive is performance to your specific gradient sensitivity measure in eq. 2? For example, what if all the norms were squared? What if you didn't normalize? 
- Perhaps you can try providing intuition for the gradient sensitivity measure by considering a scenario example where the predictor conditioned on the available features is linear in the latents for the unobserved variables?
- How would you incorporate feature costs into IBFA? Some existing methods to do by weighting expected improvements in loss against the cost of acquiring features (see DIME for example). Doing so might be tricky here because the units of your selection criterion $R$ are unclear.

### Soundness
3

### Presentation
3

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
The paper proposes to use IB for scoring the utility of features, so to enable dynamic AFA. The idea is interesting and seems legit to me. Unlike the widely used CMI (which leans towards myopic feature selection) and RL (which is intractability in high-dimensional feature spaces), the proposed IBFA achieves non-greedy acquisitions using stochastic encodings sets and eliminates the sparse rewards and the deadly-triad problem in RL. Empirical performance of the proposed IBFA is robust across diverse datasets, and it consistently outperforms CMI-based, RL-based, and other baseline methods in predictive accuracy and feature acquisition efficiency.

### Strengths
+ The paper provides IBFA as a viable alternative to CMI and RL for AFA problems.

+ The idea of applying IB to regularize the latent space, so to capture only label-relevant information to reduce reliance on noisy or redundant features, is simple but solid. This regularization enables IBFA to make feature acquisitions by considering the impact of unobserved feature realizations on prediction, which improves existing AFA methods that generally ignore these dependencies.

+ The adapted synthetic experiment is useful. Ablations on the number of latent samples, variational IB regularization, and acquisition methods highlight the necessity of each model component for optimal performance.

### Weaknesses
- Why Theorems 1 & 2 matter? What are their findings (in an AFA context)?

-  Although IBFA offers robust empirical results, the paper provides limited information on computational requirements, especially in terms of scaling the model to larger or real-time applications, i.e., dynamic AFA. 

- The interpretability of its feature acquisition process remains somewhat opaque. For example, providing more interpretative insight, such as visualizing how different features affect the latent space, could improve transparency for end-users in sensitive fields like healthcare, which appears to be the very motivating example of this paper.

- My main concern to give an even high rating is that the approach could be perceived as a reapplication of IB rather than a fundamentally new method. Its novelty primarily lies in adapting IB within an AFA context rather than a new technique in IB itself. With that being said, the paper could strengthen its contribution by exploring how IBFA might be adapted to more generalized tasks or by testing it against other latest AFA models with hybrid feature acquisition strategies.

### Questions
I reproduce the weaknesses as questions to be addressed: 

- Why Theorems 1 & 2 matter? What are their findings (in an AFA context)?

-  Although IBFA offers robust empirical results, the paper provides limited information on computational requirements, especially in terms of scaling the model to larger or real-time applications, i.e., dynamic AFA. 

- The interpretability of its feature acquisition process remains somewhat opaque. For example, providing more interpretative insight, such as visualizing how different features affect the latent space, could improve transparency for end-users in sensitive fields like healthcare, which appears to be the very motivating example of this paper.

- My main concern to give an even high rating is that the approach could be perceived as a reapplication of IB rather than a fundamentally new method. Its novelty primarily lies in adapting IB within an AFA context rather than a new technique in IB itself. With that being said, the paper could strengthen its contribution by exploring how IBFA might be adapted to more generalized tasks or by testing it against other latest AFA models with hybrid feature acquisition strategies.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a novel approach called IBFA to address the problem of AFA. AFA is the task of dynamically selecting which features to measure during test time to improve the prediction performance, while minimizing the number of acquired features. The key contributions are:

a) Identifying limitations of existing approaches based on RL and CMI maximization, such as training difficulties in RL and myopic feature selection in CMI.

b) Proposing a new method, IBFA, that uses an encoder-predictor architecture with an IB regularized latent space for feature acquisition. The acquisition objective considers the effect of unobserved features and focuses on distinguishing between the most likely classes.

c) Extensive evaluation on synthetic and real-world datasets, including cancer classification tasks, demonstrating that IBFA consistently outperforms various baselines.

### Strengths
- The paper proposes a novel approach to AFA by leveraging the Information Bottleneck principle and stochastic encodings. The idea of using a regularized latent space for feature acquisition and considering the effect of unobserved features is unique and well-motivated.
- The authors provide a thorough theoretical analysis of the limitations of existing CMI-based methods and the motivations behind their proposed approach. The experiments are extensive, covering synthetic and real-world datasets, including challenging medical tasks. The results and ablation studies are well-presented and support the claims made.
- The paper is well-written, and the methods are explained clearly. The use of block diagrams and examples aids in understanding the proposed approach.
- AFA is an important problem in many real-world scenarios, such as medical diagnosis, where features are acquired sequentially based on evolving observations. The proposed IBFA method addresses key limitations of existing approaches and demonstrates promising results, potentially leading to more effective and efficient feature acquisition in practical applications.

### Weaknesses
- Limited theoretical analysis of the IB regularization: While the authors provide a thorough analysis of the CMI objective's limitations, the theoretical justification for using the IB regularization in the proposed method could be stronger. A more in-depth discussion of how IB addresses the identified issues would be beneficial.
- The paper does not discuss the computational complexity of the proposed method, particularly for the gradient estimation step. An analysis of the time and memory requirements, especially for large-scale datasets, would be valuable for assessing the practical applicability of IBFA.

### Questions
- The proposed method relies on stochastic encodings and sampling from the latent space during acquisition. How sensitive is the performance to the number of samples used, and how can this be optimized for different datasets or computational constraints? Also, how sensitive i the method to the number of latent factors per feature?
- The paper mentions that the proposed method is currently limited to classification tasks. Could you discuss potential extensions or modifications to handle regression tasks or other problem settings where the notion of class separation is not well-defined?
- The authors mention that the factorized encoding of features to latent components introduces a modeling restriction. Could you elaborate on the implications of this restriction and discuss potential alternative architectures or formulations that could address this limitation while maintaining the benefits of the proposed approach?

### Soundness
3

### Presentation
4

### Contribution
3
