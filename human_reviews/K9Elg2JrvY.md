# FROM LOW TO HIGH-VALUE DESIGNS: OFFLINE OPTIMIZATION VIA GENERALIZED DIFFUSION

- Decision: Reject
- Scores: 6, 5, 6

## Abstract
This paper studies the black-box optimization task which aims to find the maxima of a black-box function using only a static set of its observed input-output data. This is often achieved via learning and optimizing a surrogate function using such offline dataset. Alternatively, it can also be framed as an inverse modeling task which maps a desired performance to potential input candidates that achieve it. Both approaches are limited by the limited amount of offline data. To mitigate this limitation, we introduce a new perspective which casts offline optimization as a diffusion process mapping between an implicit distribution of low-value inputs (i.e., offline data) and a superior distribution of high-value inputs (i.e., solution candidates). Such diffusion process can be learned using low- and high-value inputs sampled from synthetic functions that resemble the target function. These synthetic functions are constructed as the mean posterior of multiple Gaussian processes fitted with different parameterizations on the offline data, alleviating the data bottleneck. Experimental results demonstrate that our approach consistently outperforms previous methods, establishing a new state-of-the-art performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a new perspective on offline optimization, framing the optimization task as a distributional translation from low-value points to high-value points. It uses diffusion models to learn this translation. The paper shows the effectiveness of this new perspective through theoretical analysis and experimental validation.

### Strengths
1. The new perspective of viewing the optimization problem as a diffusion translation problem is quite novel and intuitive.

2. The method proposed in this paper is direct and clear: It uses synthetic data from fitted gaussian process for data augmentation and then employs a diffusion model to learn the distributional translation from low-value to high-value regions.

3. The presentation of this paper is clear and well-structured.

4. The experiments are comprehensive.

### Weaknesses
The method needs to fit multiple Gaussian processes to generate synthetic data, which might have high time complexity in high-dimensional cases. Besides, the reverse diffusion process also requires multiple denoising steps, which also increases the time complexity.

### Questions
NA

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper considers the offline optimization problem, where offline (x, y) pairs are given, and the goal is to return samples x with better value. The paper proposes a distribution shifting approach by first learning the underlying function via a mean estimate of the Gaussian process under difference kernel functions and creating two sets of samples $X^+$ and $X^-$ by taking gradient ascent and gradient descent based on the estimated function starting from the offline samples. Then a diffusion model is learned via translating the data from $X^-$ to $X^+$ via assumed conditional Gaussian distribution. Experiments are conducted on the variance of benchmarks and the proposed method is compared to be comparable/superior to some offline linear optimization baselines.

### Strengths
The problem of offline optimization is important and has many potentials. 

The experiments are presented clearly and the proposed method indeed has improvements compared to baselines.

### Weaknesses
The originality of viewing the offline optimization as a distribution shift is not novel. See [a]. 

The derivations are not correct. The proposed method is not theoretically justified, merely a theoretically motivated heuristic. For example, the Assumption of Eq (13) cannot be justified given Eq (12), thus giving no evidence for Eq (14) as combining both equations. Thus the overall learning procedure lacks of justifications. Take another example of Eq (3), the optimal solution x^* is a function of the unknown function g, and thus g(x^*) conditioned on the offline data is not Gaussian. It is the maximum of several Gaussian since $x^*$ is also a random variable. 

The diffusion approach generates data outside of the distribution of the samples (outside of the distribution coverage) thus leading to critical generalization issues. This issue is not sufficiently addressed. 

Overall, it is a work based on a wrong assumption/mathematics though has engineering improvement. 

[a] Zhang, Q., Zhou, R., Shen, Y. and Liu, T., 2024. From Function to Distribution Modeling: A PAC-Generative Approach to Offline Optimization. arXiv preprint arXiv:2401.02019.

### Questions
In the paper, notation $\omega$ is used in multiple places. Is it a random variable indicating the randomness of the underlying function and offline data? In Eq (12) and follows, is it $\omega^*$ or $\omega$?

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
This is a high-potential paper which could be much better written. I suggest a full review of the paper for readability. For example, the second line of the abstract is 4 lines long.  The goal of the paper is not clear from reading the introduction. 

The paper seeks to learn surrogate parameterized functions which learn the data distribution. Using these surrogates, the data is augmented, and this augmented data can then be used to train a model. The issue with such augmentation is that both the (x,y) pairs must be learnt, and predicting the x may lead to OOD generations. This is usually solved through inverse mappings, or search. This paper proposes ideas from Brownian bridge diffusion, which was previously proposed in computer vision to extend to tasks in the Design Bench.

### Strengths
- The use of diffusion in design tasks is quite novel. 
- The results are strong. But note that the best of 128 samples is reported in Table 1. The algorithm itself may not have access to "best score". So while the algorithm reaches good results in best case, median or avg may have to be considered.

### Weaknesses
- Why use the brownian bridge diffusion process? How do other diffusion processes perform?
- Section 3.1 what is the stochastic translation operator? why use terms before defining them?
- The paper has spent a lot of space on Section 3.1 which comes without explanation of what the goal is. For example what is g in this setting? Are these the equivalents of the noise estimates in DDPM? What is the reason for the specific form in equation 13? Please rewrite thsi section with what you are attempting in this section. While reading this section, it is extremely unclear what the mathematics is leading towards, and hence why a stochastic translation is required.
- There is no algorithm section translating the mathematics of Section 3 into the experiments of Section 4. I do not see what was the optimization implemented, or how it was obtained.
- The paper does not report running times of the algorithms. Given diffusion can be significantly costlier it is worth reporting the wall-clock time.
- The paper reports max score in Table 1, but without external evaluation, it is not clear that the algorithm may know what is best. Hence this is not the expected performance. Can  you also show the mean score?

### Questions
- Why do you set the value of zeta to t/T? and delta_t to zeta_t(1-zeta_t)?

### Suggestions:

- Could you consider revising your abstract? The goal of the paper was not immediately clear.
- Please consider a rewrite of section 3.1. Start with what the optimization function is. Then follow through with the derivations on how it was obtained. Please mention how this is used in an algorithm.
- Happy to raise my score given some changes to the manuscript.

### Typos
- Line 233: "as follows"

### Soundness
4

### Presentation
2

### Contribution
3
