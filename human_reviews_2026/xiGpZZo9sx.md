# Scalable Continuous-Time Hidden Markov Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
As a probabilistic tool for irregularly-sampled data, the Continuous-Time Hidden Markov Model (CTHMM) inherently handles real phenomena with uncertainties modelled by distributions. However, CTHMM is affected by (i) the costly matrix exponentiation (cubic time complexity \wrt the number of hidden states) involved in the estimation of transition probabilities, and (ii) the use of simplistic parametric observation models (e.g., Gaussian). Thus, we propose scalable algorithms for CTHMM on traditional problems (learning, evaluation, decoding) to ensure tractability. Firstly, we factorise states of CTHMM into multiple binary states (e.g., several $2\times 2$ sub-problems) leading to a distributed closed-form exponentiation. We also accelerate matrix-vector products, reducing the complexity from quadratic to linearithmic. Secondly, the simplistic parametric distributions are replaced by the normalising flows (that transform simple distributions into complex data-driven distributions), accelerated by sharing few invertible neural networks among groups of hidden states. Training our approach takes few hours on a GPU, while standard CTHMMs with mere 10 hidden states take few weeks.  On the largest dataset, our method scales favourably (up to 1024$\times$ larger hidden states than naive CTHMM and outperforms it by 4.1 in log-likelihood). We also outperform competing HMMs with advanced solvers on downstream tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work considers continuous-time hidden Markov models (CTHMM) whose latent state state at time $t$, $H(t)$, takes values in $\{1, ..., M\} =: [M]$ for some large integer $M$ (this work considers $M \leq 2^{12} = 4096$). Standard recursions for performing inference in such models scale exponentially with $M$ and are thus prohibitively costly.

To circumvent this problem, the authors propose a continuous-time analogue of the factorial HMM. Specifically, assuming that $m := \log_2(M) \in \mathbb{N}$, they specify $m$ independent continuous-time binary-state-space Markov chains  $(H_1(t))$, ..., $(H_m(t))$ and then set
$$
  H(t) := b(((H_1(t), ..., H_m(t)))),
$$
for some suitable bijection $b: \{0,1\}^m \to [M]$. Exploiting the fact that the binary-state-space Markov chains $(H_l(t))$ are thus conditional independent given the observations construction then makes inference in such models possible even if $M$ is large.

To further reduce the computational cost, the authors specify the observation densities/probabilities through $\bar{m}$ normalising flows $f_1, ..., f_{\bar{m}}$, where $\bar{m}$ is much smaller than $M$ (the authors consider $\bar{m} \leq 8$). That is, they specify another mapping $b' : [M] \to [\bar{m}]$ and then assume that the observation density/probability satisfies
$$
  p(o_t|H(t)) = p(o_t|f_{b'(H(t))}, \mu_{H(t)}, \varSigma_{H(t)}).
$$
Here, $\mu_j$ and $\varSigma_j$, for $j \in [M]$, are the parameters of the base distribution for the normalising flow used to parametrise the observation density.

### Strengths
This paper is fairly well written and thus easy to read. I particularly appreciate Figures 1 and 2 which make the proposed approach very clear. The idea of extending factorial HMMs to the continuous-time setting seems sensible and is novel to my knowledge. The method outperform state-of-the-art alternatives in the metrics considered and the authors take great care to demonstrate, in the ablation studies, the benefits of different aspects of their methodology in isolation.

### Weaknesses
**Main comment:**

Line 357 states that the models were trained on the first 80 % of each observation sequence and then metrics (e.g., the negative log-likelihood) were computed on 100 % of the observation sequence. This leaves a chance that the superior performance of the proposed method stems from overfitting on the first 80 % of the observations.

Thus, I would like the authors to include comparisons of the different methods in terms of out-of-sample predictions for each data set, too. Presumably, prediction is one of the main tasks for employing such methods anyway.



**Minor issues:***

- I would add brackets around the second product and the term immediately following in Equation 1.
- I think it would help the reader to mention much earlier in the paper (e.g. early on in the introduction and maybe even in the abstract) that the proposed method is essentially a continuous-time analogue of the factorial HMM (with a particular type of observation models specified through normalising flows).
- There are some very minor typos (e.g., "compare" in L47, "CTHMM" (i.e., singular), in L44 & L48)
- The text in the panels of Figure 4 is much too small.
- The bibliography has a large number of typos and inconsistencies, e.g., missing capital letters in proper nouns or in journal names, inconsistencies in formatting (abbreviated journal names vs unabbreviated; naming of conferences), some author first names are abbreviated but others aren't, use of "." (or not) in abbreviated author names.

### Questions
1. How different are the parameters $\mu_j$ and $\varSigma_j$ for those states that use the same normalising flow? If they are similar, then the proposed model becomes essentially a standard $\bar{m}$-state CTHMM (because only $\bar{m}$ states are then identifiable) with some restrictions placed on the $\bar{m} \times \bar{m}$ transition matrix.

2. Why is does the computational (training) cost of the FaHMM scale so much poorer than the proposed methodology? Don't both approaches exploit the same type of conditional independence structure?

3. What is the observation model used by the standard CTHMM baseline?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work proposes a strategy to scale Continuous-time Hidden Markov models (CTHMM) using a factorisation of the state space. In particular they model the latent state transition as the evolution of independent Markov processes, with binary state spaces. They then show that the evolution of the collection is follows a transition matrix that can be expressed as the Kronecker product of the transition matrices of the independent processes, which in turn allows fast algorithms through clever use of Kronecker algebra. The loss of expressivity, as a consequence of the independence assumption is counteracted by introducing flexible emission densities through the use of normalising flows. The experimental results some improvement in terms of negative log likelihood metric in comparison to a vanilla HMM, a factorial HMM (both discrete-time HMMs). Furthermore, it was shown that by the virtue of allowing a larger state-space than the competing discrete-time HMMs, the proposed model has better performance in downstream tasks.

### Strengths
The major strength of the paper is that it suggests a straightforward approach to have a large state-space within the CTHMM framework. This when combined with a normalising flow (NF) does indeed produce an expressive model. The idea of using parallel independent latent processes coupled with fast Kronecker algebra appears to be novel.

### Weaknesses
I believe the major weaknesses stem from the fact that the experiments do not adequately justify the main claim i.e. the usefulness of this model over available alternatives.

Major: 
 - As far as I understand, in Table 2 the metrics were obtained on the full sequence (involving the training part as well). A model having a larger state space and a NF can overfit the training data and may come out best in terms of the negative Log Likelihood (NLL) metric. With NLL as a metric (and inclusion of training data) the ablation studies also remain inconclusive, again the improvements can be due to overfitting. 
- The downstream task performance perhaps is a bit more clearer indication of the usefulness of having larger state spaces. But here the improvement over a vanilla HMM is not substantial. Especially considering the additional computational cost associated with this method.
- The experiments do not probe the limitations of the independence assumption. How well this model learns the true latent state sequences? It is difficult to scale CTHMM, but the proposed approach of scaling is (in a way) shifting the goal post by introducing a different model. Thus, the authors should have shown how well this "different" model is still able to capture the "same" underlying process.

- The introduction of a NF is not well justified. I mean one can add such a NF easily to a vanilla HMM (and its other derivatives).
 
Minor:
 - The paper structuring needs improvements. The model should have been specified first and then all ther notations should be introduced in sequence.

- There should be concise description of how the model parameters are learnt, what the forward/Viterbi algorithms are. Without a clear flow from the model to the learning and finally to the inference, the presentation seems haphazard.
- I am not sure whether Figure 2 is adding any value. I had to read the text carefully to understand the NF construction.

### Questions
- The real question is whether adding more states (following this approach) provides better performance than other approaches, for standard sequence prediction tasks, without incurring additional computational cost? This aspect is not tested adequately.  It would have been interesting to see a comparison on just unseen trajectory. 

- Following Theorem 2.1, one would care to know how well $\hat{\mathbf{Q}}$ approximates $\mathbf{Q}$? The authors need to answer this unambiguously, though either theoretical tools or additional experiments.

- Why was [1] not included in the benchmarking? 

- How was $\Delta t_{k}$ chosen?

- How does one come-up with the optimal grouping (of the NF) number?

1. Yu-Ying Liu, Shuang Li, Fuxin Li, Le Song, and James M Rehg. Efficient learning of continuous-time
hidden markov models for disease progression. Advances in neural information processing systems,
28, 2015.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper improves Continuous Time Hidden Markov Models (CTHMM) by addressing their computational and modeling limitations. First, the authors decompose the hidden state space with M values into $\log_2(M)$ independent binary Markov processes. This structure allows transition probabilities to be computed in closed form and evaluated in parallel, avoiding the high cost and instability of large matrix exponentiations. The independence of each chain limits the expressiveness of the model. To counter this, authors replace simple emission models such as Gaussians or Mixture of Gaussians with conditional normalizing flows, which can represent more complex observation distributions. These changes make CTHMMs more practical for large datasets with irregular time sampling, enabling training in reasonable time. The approach achieves higher likelihoods and better downstream performance than vanilla CTHMM and discrete-time HMM models.

### Strengths
- The authors propose improvements that significantly enhance the scalability of CTHMMs.
- The method achieves superior results compared to standard CTHMMs and discrete-time HMMs reported in the literature.
- The manuscript is clearly written and easy to follow.

### Weaknesses
- Lines 356 and 454: "Models are trained on the first 80% of each sequence (…) and evaluated on each complete sequence." This train–test split, used across all experiments, does not provide a reliable measure of generalization because 80% of the test data overlaps with the training sequences. This issue is particularly critical for downstream tasks, where the classifier may simply memorize hidden-state patterns from the first 80% of each sequence rather than learning class-level features.
- Line 447: "For each method, we use the trained model with the largest number of hidden states." While this allows comparing models at their limits, it makes it unclear what drives the performance differences — the use of normalizing flows or the larger latent space dimensionality.
- The authors introduce up to eight independent normalizing flow models for grouped latent states, but no ablation studies are provided to justify this design choice.
- Similarly, different Gaussian parameters are used for each latent state as the prior distribution for the normalizing flow, yet there is no evidence showing that this design improves performance. Since the flow is already conditioned on the hidden state, using separate priors may not be necessary.

### Questions
- (Ad W1) Would it be possible to validate the models only on the remaining 20% of each sequence (unseen during training)? Alternatively, the authors could train on K% of complete sequences and evaluate on the remaining (100–K)% of entirely unseen sequences to better assess generalization.
- (Ad W2) Table 2 perfectly shows the effect of varying latent dimensionality for each model. Could a similar analysis be performed for the downstream tasks? For example, Figure 5 could be split into four subfigures (one per task) to compare methods directly (rather than one method for each of tasks).
- Minor: In Equation (1), the variable "n" is not defined explicitly and it has to be deduced from the context. Clarifying it in the text would improve readability.

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper develops methods to make continuous time HMMs (CT-HMMs) scale to large state spaces. The proposed approach has two key steps, diagrammed in Fig 1. First, the usual 2^m x 2^m transition rate matrix is factorized into many 2x2 matrices, that each govern one of m separate hidden binary variables with factorization similar to a Factorial-HMM (Ghahramani & Jordan '97). Then, the emission model uses a conditional normalizing flow (Dinh et al. '17) to produce a density over observed vectors at each time stamp, which depends on the current state of the m binary variables.

The first assumption, which assumes m binary variables make up the state at each time, is a simplification of the usual CT-HMM which allows more general Markov distributions over the 2^m possible states. However, the computational advantages of this simplification are argued to be worthwhile, especially since the flexible normalizing flow emission model can hopefully make up for any loss in expressivity of the latent state model.

For each of the 2^m hidden state configurations, indexed by j, there is learned a separate mean vector $\mu_j$ and covariance $\Sigma_j$. However, a state-specific normalizing flow might be too expensive, so instead only $\tilde{m} < 2^m$ distinct normalizing flows are learned, and shared to all $2^m$ states via modular arithmetic indexing (the set of every $\tilde{m}$-th states in index order share a normalizing flow model, but have distinct emission models). 

Experiments evaluate this CT-HMM approach against classic CT-HMM methods as well as discrete time HMMs with Gaussian likelihoods and factorized versions of those.

### Strengths
I think there's plenty to like about this paper. 

* Focus on irregular time series modeling, a problem with lots of practical applications but also elegant theory/methods

* The presented method's runtime speedup is a neat and clever achievement. Avoids standard runtime cost of evaluating CT-HMM likelihood that is cubic O(M^3) in number of states M. Instead, proposed factorization achieves O(M log M)

* Use of normalizing flows is welcome as an engineering win to get flexible emissions

* Experiments demonstrating that they can fit models with 2^12 = 4096 states on real datasets


### Notes on Novelty

For me the key innovation here is the clever use of binary factorization with the efficient evaluation in Algorithm 2 to get runtime speed-up from cubic to "linearithmic". 

The use of conditional normalizing flows as an emission model is interesting, but the usage here is a bit "off the shelf" and I don't see new methods or insights about CNFs that were not already widely known (e.g. see how another paper by Lorek et al. 2022 uses CNFs as an emission density for discrete time HMMs).

### Weaknesses
# Weaknesses

Here's a brief summary of the issues I see with current manuscript, that prevent me from giving a higher rating at present:

* C1: Need better clarity about likelihood of o_t given hidden state
* E1: Assessment seems to be mostly on training data, not proper generalization
* E2: Is it fair to compare continuous and discrete time likelihoods?
* E3: Missing comparisons to important alternative flexible probabilistic models of irregular time
* E4: Convergence of training needs to be verified
* E5: Focus only on 2-dimensional datasets is a limitation


## Clarity issues

### C1: Clarity about likelihood of o_t given hidden state

There seems to be to conflicting formulas:

* in line 164, the log likelihood $\log p(o_t | b(H(t)) = j)$ is given as a log Normal PDF plus a sum of log determinants
* in line 3 of Alg 1, this same likelihood is given using only the log Normal PDF evaluation, without any determinants

I'm pretty sure the determinants are needed to do the change of variables correctly... so is there something missing in Alg 1?

## Experimental design issues/questions

### E1: Assessment seems to be mostly on training data, not proper generalization

In the experiment descriptions, I see the text

> Models are trained on the first 80% of each sequence to minimise average negative log-likelihood (per observation) and evaluated on each complete sequences

So are reported likelihoods on the full 100% of each sequence? If that's true, this seems like you are mostly assessing *training* quality, not generalization quality.

Typically, models are empirically measured to assess generalization (to new sequences, or to parts of sequences unseen in training). I don't know if the ranking of methods based on this "complete sequence" likelihood is measuring the right thing.

Is the CRPS measure also assessed on "complete sequences"? if so, the same concern applies.

### E2: Is it fair to compare continuous and discrete time likelihoods?

I don't think the main paper provides enough reproducible details about how the provided data with irregular timings was discretized. The paper says:

> discretise irregular intervals by computing mean time differences across the data, normalising, and rounding to integers

So, I first compute the mean $\delta t$ across the dataset, but then how do I normalize? Divide by the empirical standard deviation? Something else? Then I take the transformed time steps and round to the nearest integer? I think an example (worked out in the supplement) would help here.

Imagine a case where some adjacent measurements are quite close to one another, but the normalized mean is much larger. Would these ever get grouped into the same timestep? I can imagine this is possible if both measurements round to the same integer.

I worry if the latter occurs (one discrete time interval holds multiple measurements), that the quantities being evaluated in CT vs discrete time (DT) models are no longer apples-to-apples. We need more certainty that the evaluations in Table 1 are "apples-to-apples" fair.

**Related issue:** the compared DT-HMMs in experiments use only Gaussian likelihoods. I'd expect a comparison to the CNF-based likelihoods for DT-HMMs, using e.g. methods from Lorek et al NeurIPS 2022.

### E3: Missing comparisons to important alternative flexible probabilistic models of irregular time

I appreciate the focus on CT-HMMs and close relatives, but I think for a venue like ICLR most folks in the audience will wonder "how does this compare to other probabilistic models of irregular time series"? 

For example, you could consider:
* Schirmer et al's continuous recurrent units: https://proceedings.mlr.press/v162/schirmer22a.html
* work on CNFs for irregular time series by Yalavarthi et al. at AAAI 2025: https://ojs.aaai.org/index.php/AAAI/article/view/35494
* GRU-ODE-Bayes from NeurIPS 2019: https://proceedings.neurips.cc/paper/2019/hash/455cb2657aaa59e32fad80cb0b65b9dc-Abstract.html
* work on stochastic ODEs
* work on scalable Gaussian processes (GPs)

I'd expect to see at least 1 or 2 models not in the CT-HMM or DT-HMM family for comparison on a dataset or two. To be clear, I don't necessarily need the present paper's method to beat this alternative, just for a fair evaluation so the audience can understand the strengths/disadvantages of the present approach. 

Given the complex landscape of irregular time models available, now, this experiment is important for establishing the significance of this present paper. 


### E4: Did runs converge? Need to verify

I see that all methods were run for 100 epochs, or a 160 hour limit. I respect the time limit for its practicality. But for runs that terminated after 100 epochs, were there sanity checks that convergence occurred?

### E5: Focus on two-dimensional datasets is a limitation

All 3 tested datasets focus on time series that describe

> two-dimensional geographic locations data over extensive spatial areas

This focus on 2D data is a limitation. Many time series of interest to CT-HMM users, such as patient vital signs over time, have much greater dimensionality than 2D. 

Is there a reason to think the presented approach wouldn't scale well to higher dimensions? Perhaps estimating the state-specific covariance $\Sigma_j$ becomes difficult when $D$ gets large and there are many states?

### Questions
Answering the questions raised above, especially those in E1-E5, are probably most important for me to consider raising my score.

I also have this general question: Is the speedup obtained via the efficient factorization into binary states plus careful evaluation of Kronecker products in Alg 2 here exclusive to the *continuous-time* HMM setting? Would any part of this possibly accelerate discrete time HMMs with factorized binary states?

### Soundness
2

### Presentation
4

### Contribution
2
