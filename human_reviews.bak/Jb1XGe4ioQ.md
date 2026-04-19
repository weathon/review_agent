# Stream-level flow matching from a Bayesian decision theoretic perspective

- Decision: Reject
- Scores: 5, 3, 3

## Abstract
Flow matching (FM) is a family of training algorithms for fitting continuous normalizing flows (CNFs). Conditional flow matching (CFM) exploits the fact that the marginal vector field of a CNF can be learned by fitting least-square regression to the so-called conditional vector field specified given one or both ends of the flow path. We show that viewing CFM training from a Bayesian decision theoretic perspective on parameter estimation opens the door to generalizations of CFM algorithms. We propose one such extension by introducing a CFM algorithm based on defining conditional probability paths given what we refer to as "streams," instances of latent stochastic paths that connect pairs of noise and observed data. Further, we advocate the modeling of these latent streams using Gaussian processes (GPs). The unique distributional properties of GPs, and in particular the fact that the velocity of a GP is still a GP, allows drawing samples from the resulting stream-augmented conditional probability path without simulating the actual streams, and hence the "simulation-free" nature of CFM training is preserved. We show that this generalization of the CFM can substantially reduce the variance in the estimated marginal vector field at a moderate computational cost, thereby improving the quality of the generated samples under common metrics. Additionally, we show that adopting the GP on the streams allows for flexibly linking multiple related training data points (e.g., time series) and incorporating additional prior information. We empirically validate our claim through both simulations and applications to image and neural time series data.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper extends the Bayesian framework of flow matching, a generative algorithm that learns a vector field (parameterized by a neural network) that transforms noise to data. It extends conditional flow matching, which considers a target vector field conditioned on a data point, to have the target vector field conditioned on a stream, a complete path over the generation. It is shown that the minimizer of the resulting loss is the same as conditional flow matching. The streams are modeled by Gaussian Processes for analytical and computational convenience. Experiments are done on synthetic two-Gaussian datasets, MNIST, and HWD+ showing that this extension decreases the Wasserstein distance of the generations to the real data and enables control over the generative path.

### Strengths
- The paper is written clearly and straightforwardly.
- The proposed framework is theoretically sound, flexible, and novel. It shows that flow matching is general enough to be extended to different types of conditioning.
- Experiments show the value of the framework in a statistically significant way.

### Weaknesses
Overall, I feel that the paper does not quite fully back up some claims that it makes.
- The paper mentions that the computational cost is moderate, but does not explicitly discuss this in Section 3.2 or show measurements in Section 4. It would greatly strengthen the paper to include this, as GP posteriors can be expensive to compute.
- The HWD+ dataset is not usually used for time series, and so the construction of that experiment seems a bit synthetic.

### Questions
- Is the assumption of independence among the dimensions in the GP too restrictive? Does it allow generating data with arbitrary dependence structures?
- Have the authors done experiments on classic time series datasets, e.g. any of the UCR time series datasets (https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper proposes to use Gaussian Process (GP) models to define interpolants for flow matching. This is motivated by first introducing a “Bayesian decision theoretic perspective” on flow matching, essentially corresponding to the observation that the marginal flow can be obtained as a posterior expectation of a conditional flow, which in turn is the solution to a least squares regression problem. The paper also considers the case of conditioning the GP interpolant on intermediate values of the flow (that is, if the flow transports q_0 to q_1 over the time interval t \in [0,1], then we can condition on observations of the trajectories at, say, t=0.5).

### Strengths
Using GPs to define stochastic interpolants gives a flexible framework that makes use of several convenient properties of GPs, specifically that the derivative of a GP is also a GP (used in the definition of the objective function, since we regress the learned flow on the velocity field induced by the interpolant), and the fact that we can compute a conditional GP by conditioning on an arbitrary collection of points (used to enable conditioning on intermediate samples, e.g. at t=0.5). It also opens up for using different tools and techniques from the GP regression literature in the context of flow-based generative models.

### Weaknesses
The novelty of the work is questionable.
* As far as I can tell, the “Bayesian perspective” developed in Section 2, as well as the “per-stream perspective” develop in Section 3.1 is known and covered in prior work, e.g. https://arxiv.org/abs/2303.08797 
* The use of “GP regression models” to define the interpolant is (to the best of my knowledge) novel, but at the same time quite similar to using SDEs, see e.g. Section 3.1 in https://arxiv.org/abs/2303.08797 and methods based on Schrödinger bridges, e.g. https://arxiv.org/abs/2303.16852 (the latter is cited in the current paper, but only in passing and no discussion about similarities with this work is provided). Note that these models based on “diffusive interpolants” also result in GPs, although defined indirectly through the corresponding SDE. While not identical to the current paper (which is based on directly defining the GP through mean and covariance functions) I would have liked to see an in-depth discussion about similarities and differences/pros and cons.

Based on this, I would recommend rewriting the paper and move section 2 and 3.1 into a background section (relating these to prior work) and instead elaborating on the (as far as I can tell) novel contribution of this work, namely the use of GPs to define interpolants. You could for instance elaborate on the difference between k11 and c11 (which is not so clear in the current paper), how you define the unconditional GP marginals for the endpoints x0, x1 (I assume that you use an unconditional reference GP which is then conditioned on the non-Gaussian data points to obtain the interpolant?), and pros and cons between the GP-based model compared to other diffusive interpolants/bridges.

### Questions
The possibility of conditioning on intermediate states is interesting, but I wonder how this is related to multi-marginal optimal transport, e.g. https://arxiv.org/abs/2310.03695 How does the proposed method differ from the multi-marginal setting?

A minor comment is that, in my opinion, using all of z, x_t, and s to refer to more of less the same thing is unnecessary and the paper would be easier to read if you harmonize the notation.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper presents an informal view of the CFM framework from the perspective of the Bayesian approach. A new vector field learning algorithm (GP-CFM) is presented, which leads to more intuitively expected trajectories and slightly improves the metrics in simple cases, compared to the classical I-CFM and OT-CFM.

### Strengths
- Python code is provided
- The paper is rather clear written

### Weaknesses
- There are no theoretical estimates, such as the accuracy of the presented Algorithm, rate of the convergence, variance reduction etc. Thus, the paper is empirical in essence. 
- Experiments have been performed only on synthetic data or low-dimensional datasets like MNIST. I would like to see more experiments including experiments on higher dimensional data, especially since the paper is empirical.
The results on CIFAR10 that Tong et al 2024b cite would be much more convincing. 
The classic paper by Lipman et al “Flow Matching for Generative Modeling”, 2023, considers an ImageNet dataset up to $128\times128$. I recommend present the results also on ImageNet at different dimensions.
- In the same way, I suggest publishing other metrics for comparison, such as NLL and NFE (as in Lipman et al).
- As the results of Fig. 5 show, the difference in metrics between the presented method and I-CFM (or OT-CFM) is extremely small. Thus, the method does not show a significant gain in efficiency. I would expect a gain of at least 10%--15% or more. However, as I wrote in the previous point, it would be much more convincing to experiment on multiple high-dimensional datasets rather than two similar ones.
- If I am not mistaken, the only significant difference between the presented algorithm and classical CFM is the presence of empirical points within the trajectory (i.e., points for time $t$ other than 0 or 1). However, this formulation goes beyond the original formulation -- if we want to sample from an unknown distribution given by a set of samples, we have no interior points in the trajectory, only a set of endpoints. Moreover, different FM models may use different maps, and hence different intermediate points for given source and target distributions. Could you please elaborate how your method handle the setting with only two (end-)points and how your implementation takes into account different maps?
- In [1] there is an example of an experiment with points lying inside the trajectory. It would be interesting to see the application of the presented method to this data, since such a problem statement falls exactly under the presented framework. This is a recent article, and it just discusses time-dependent processes (of cell evolution).
- In Fig. 1 C, I-CFM appears to be simply undertrained. Can you elaborate on the details of the experiments and how calculate the metric in Table on this Fig? Can you present another metrics, like NLL, in this case? Am I correct that the pictures on C show the worst results among 100 runs, not the typical results?
- The theoretical part of the paper is written in a way that is not rigorous and sloppy in terms of mathematics (although it claims some theoretical results). These include:
  - It is not always clear for which variables the mathematical expectation is taken. 
     - For example, L173--174: $\mathbb{E}_{t,q(s),\delta(x-x_t(s))}$. The first variable on which integration is taken is $t$ (and we know from the context that it is uniformly distributed in the interval $[0,1]$). But then there is $q(s)$ under the sign $\mathbb E$. This is already a distribution function (or is it? It has a fixed argument), not a variable. Such confusion leads to heavy reading and increases the possibility of errors in the text itself. Further $,\delta(x-x_t(s))$ is under $\mathbb E$, where there are already several variables under the function sign.  I advise (at least the first time you encounter a formula) to write clearly on which variable the integral is taken and what the measure (or probability density function) of this random variable is, or better yet, to write all the key mathematical expectations that result from this paper in the form of integrals (Riemann or Lebesgue) in the Appendix.
     - As a corollary to the previous point. In probability theory and mathematical statistics, mathematical expectation is nothing but an integral over a measure. In the paper, L185--186, what is meant by $\mathbb E_{x_t(s)}$, over which variable (measure) is the integration performed? If this variable has a distribution density function, could you give it explicitly?
  - L141-142: what is the strict definition of $s$?  Is $s$ is just a curve: $s\colon [0,1]\to \mathbb R^d$, $s(0)=x_0$, $s(1)=x_1$? Or is it stochastic process https://en.wikipedia.org/wiki/Stochastic_process? (For example, a Brownian bridge is a continuous-time gaussian process with the given start and end points). In both cases it is customary to write $s(t)$ (or $s_t$), not $x_t(s)$. I suggest to clarify definitions, and use the standardized designations adopted for these definitions. The same applies to the definition of so-called ``stream velocity'',  L155-156. Namely, if $X(t)$ is stochastic process, then $dX_t = \mu_t\ dt + \sigma_t\ dB_t$, where $B(t)$ is Wiener process and the concept of a derivative has to be further defined. Thus, the notation $dX(t)/dt$ is ambiguous. Please clarify these concepts that are key to your paper.
  - It is not a strict designation just a vertical line, such as L150--151, or L239-240. Does it denote the conditional expectation (https://en.wikipedia.org/wiki/Conditional_expectation)? I suggest to use strict mathematical notation (such as conditional expectation) to avoid misunderstandings, at least in places where the essence of the presentation is theoretical exploration. 
  - Usually, the symbol $\mathcal N(\mu, \sigma)$ refers to the (multivariate) Gaussian distribution. How then to **strictly** understand sampling of both $x_t$ and its derivative simultaneously in L239-240 (from a single Gaussian distribution with time-dependent parameters)? How is the conditional expectation on $x_\text{obs}$ taken during sampling? Can you please give a detailed step-by-step description of the calculations, in particular, how exactly you sample $x_t$  and  $\dot x_t$, how conditional expectation is taken (if it was taken)?

 - L073-074 ` to integrate related observations`. It is not clear what are `related` observations in general. Can you clarify what these are and in which real-world problem statements they occur? Further on you indicate that for MNIST dataset related are pictures with numbers are 0, 8, 6. This seems like an _ad hoc_ statement. How can you extend this concept (of `related` observations) to, for example, CIFAR10 dataset and clarify how can it help in the solving initial task of sampling from unknown distribution?

 - L472--474 `Here, we consider the task of transforming from“0” (at t = 0) to “8” (at t = 0.5), and then to “6” (at t = 1).` Such a formulation seems too artificial (and beyond the scope of the classical FM problem formulation). Can you please explain the significance of this formulation, and/or cite similar real-world problems that include such a formulation, and/or papers with such a problem formulation?



[1] Alexander Y. Tong, Nikolay Malkin, Kilian Fatras, Lazar Atanackovic, Yanlei Zhang, Guillaume Huguet, Guy Wolf, and Yoshua Bengio. Simulation-free Schr¨odinger bridges via score and flow matching. In Sanjoy Dasgupta, Stephan Mandt, and Yingzhen Li (eds.), Proceedings of The 27th International Conference on Artificial Intelligence and Statistics, volume 238 of Proceedings of Machine Learning Research, pp. 1279–1287. PMLR, 02–04 May 2024b.

### Questions
In addition to the questions in the weakness section, please answer the following questions:

- L247--249: What is the relationship between your method and OT-CFM? How does sample pairing, which is explicitly present in OT-CFM (Tong et al, 2024) in the form of OT-minibactch, arise in your method? Moreover, OT-minibatch can use different metrics to find the distance between points, which one do you use?

- Can you explain in a little more detail how you found the FID for MNIST? In the zip archive there is a file metric/Fid_score.py, there basic functions require an image size of $299\times299$ (e.g. line 53 of this file: `assert x.shape[1:] == (3, 299, 299), “Expected input shape to be: (N,3,299,299,299)” +\`). However, in line 186 resize is commented out: `#im = cv2.resize(im, (299, 299))`, and the images from MNIST are $28\times28$ in size.

- On L071--072 `We demonstrate that adjusting the GP streams can reduce the variance of the estimated marginal vector field with moderate computational cost.` Can you explain in more detail where in the article you showed this and in what form (theoretically, empirically)?

### Soundness
2

### Presentation
3

### Contribution
2
