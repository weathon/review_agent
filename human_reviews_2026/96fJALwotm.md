# Complexity Analysis of Normalizing Constant Estimation: from Jarzynski Equality to Annealed Importance Sampling and beyond

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 2, 6, 6

## Abstract
Given an unnormalized probability density $\pi\propto\mathrm{e}^{-V}$, estimating its normalizing constant $Z=\int_{\mathbb{R}^d}\mathrm{e}^{-V(x)}\mathrm{d}x$ or free energy $F=-\log Z$ is a crucial problem in Bayesian statistics, statistical mechanics, and machine learning. It is challenging especially in high dimensions or when $\pi$ is multimodal. To mitigate the high variance of conventional importance sampling estimators, annealing-based methods such as Jarzynski equality and annealed importance sampling are commonly adopted, yet their quantitative complexity guarantees remain largely unexplored. We take a first step toward a non-asymptotic analysis of annealed importance sampling. In particular, we derive an oracle complexity of $\widetilde{O}\left(\frac{d\beta^2{\mathcal{A}}^2}{\varepsilon^4}\right)$ for estimating $Z$ within $\varepsilon$ relative error with high probability, where $\beta$ is the smoothness of $V$ and $\mathcal{A}$ denotes the action of a curve of probability measures interpolating $\pi$ and a tractable reference distribution. Our analysis, leveraging Girsanov's theorem and optimal transport, does not explicitly require isoperimetric assumptions on the target distribution. Finally, to tackle the large action of the widely used geometric interpolation, we propose a new algorithm based on reverse diffusion samplers, establish a framework for analyzing its complexity, and empirically demonstrate its efficiency in tackling multimodality.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper provides a full theoretical analysis of the error in Annealed Importance Sampling (AIS), accounting for both the sampling process that generates particles and the estimator of the normalizing constant computed from these samples. The authors derive a general upper bound on the estimation error as a function of the prescribed probability path that guides the sampling process. Two important path choices are examined: the standard geometric interpolation path, which is shown to lead to exponential complexity in the difficulty of the problem, and the reverse diffusion path, which achieves only polynomial complexity—provided oracle access to score functions (otherwise, an additional approximation error appears). The difficulty of the problem is quantified in terms of the between-mode distance of the target distribution.

### Strengths
This is a welcome analysis. There is a long literature that analyzes the error of Annealed Importance Sampling, but it usually:
- makes strong assumptions: assuming equilibrium sampling, using the asymptotic regime where the number of samples is big, etc.
- is not very interpretable: the formula of the error does not clearly tell the user how to design the prescribed probability path

It has been a goal in that literature to produce a general formula of the error that is interpretable enough to inform the design choice of the probability path. To my knowledge, this is the first such analysis.

### Weaknesses
The writing could be clearer in some parts, but overall the paper is clear.

### Questions
Q1. Theorem 4 is particular to the geometric interpolation path? Is it normal that the number of samples $N$, the number of SMC iterations $M$, and the discretization of the Langevin process, do not appear in the error bound (Eq 11)?

### Soundness
4

### Presentation
2

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper uses a novel method to analyze and bound the non-asymptotic estimation error of the normalization constant for an energy functions in terms of smoothness of the potential and the action of annealing curve. 
The method relies on conditions that are less restrictive than the commonly used isoperimetric constraints for this type of analysis and applies to non-log-concave distributions. The analysis sheds light on the increased complexity of estimation with geometric interpolation in comparison to OU reversal.

### Strengths
I believe the main contribution of the paper, according to itself, is the non-asymptotic analysis (in the number of path samples) for TI and AIS (LMC) which is presented in THM2 and 4. THM2 proves constant probability bound on the error of the normalization factor estimation ratio in the path integral of Thermodynamic integration. THM4 proves an upper bound on the number of oracle calls $M$ in AIS (LMC, Alg. 1).

### Weaknesses
1 - Some important recent literature is missing from the work. How do you compare your method to Adjoint sampler or methods approximating the Kantorovich potential of the RDS.

2 - Organization of the paper is very synthetic. The paper is super technical, lacks a properly structured background section and multiple theoretical results are presented without a straight forward relevance to each other. The notations are not clearly defined in the text ($B_t$ and $B_t^\leftarrow$ in Eq 2 and 4, $n$ in line 444) and abbreviations are missing (TI, RN, SDE). Navigating through the paper is very difficult. I don't think the inclusion of Lem1 or THM1 and THM3 and their proofs are necessary for the text - this context can be built in a less technical and more cohesive manner. I recommend the authors restructure the paper and this is the main reason I'm in favor of rejecting this version. In the following I provide a list of modifications that are going to influence my decision.

3 - Use uniform notation across different theorems, sections and appendix as much as possible. For instance, $k$ and $l$ for step indices, forward and backward SDEs in proof of THM2 reuse the same notation.

4 - Simplify the representation of the Theorems 

4.1 - The use of scaling $t/T$ in THM2 is not necessary. It appears that the action $\mathcal A = \int_0^1 |\dot \pi_t|^2dt$ is proportional to $T$ which is used for scaling $t$. Therefore, the last line of proof implies a constant upper bound on the path integral for any speed and not just the specified $T$ (see proof of THM2). 

4.2 - Prop1 and Prop2 and THM6 seem out of place with respect to the rest of the paper. Is there a corollary missing? It might be the case that sample complexity bound depends on the action, but the relations are only derived for the geometric interpolation (Eq 7) in THM4. It is not enough to suggest that the lower bound on action in geometric and upper bound in RDS translates directly to better upper bound for RDS sample complexity. 

4.3 - In THM5 what is the use of $\delta$ in practice? This Theorem is adapted from previous work with the addition of $delta$ for which I'm not able to see the motivation.

4.4 - Appendix E.5 doesn't provide any analysis on the sample complexity of the mentioned algorithms, only references. Are the orders taken from the references? If so should be stated like that in the paper as opposed to analysis (line 445). 

5 - Throughout the paper different units of complexity are used to compare different algorithms. It would be helpful to write down the dependence of all complexity bounds in terms of the action, bounding the action, and with the median trick in a table for clarity.

6 - I'm not certain how Alg. 2 is used in "the series of new algorithms based on reverse diffusion samplers". I'm also assuming the experiment section somehow incorporates Alg. 2. But RDMC, RSDMC, ZODMC, SNDMC all are reference algorithms with their own state update steps and importance weight computation. It is not clear how Alg. 2's state and weight update is incorporated in the reference algorithms to get new results. 

7 - Experiments in $\mathbb R^2$ tend to be too simple and lead to overlapping results (confidence bounds) from RDS based algorithms. Results in more complex setups could prove more interesting.

### Questions
1 - It feels counter intuitive that a nested linear ($\theta$ in Eq. 10) and exponential (line 317 for $\lambda$) discretization of the schedule would reduces the overall estimation error in comparison to purely linear or exponential schedule (line 324). Since the proof of THM4 bounds the error with $r = 1$, I don't see the motivation for the exponential $\lambda$. Can you provide some examples that show how the cost (11) depends on $r$? What are the conditions for it to improve with $r > 1$? Is there a benefit of exponential cascading of $\lambda$?

2 - As the main issue with multimodal distributions arises in high dimensions, is there any explanation on how the action depends on $d$ in PROP1?

3 - Can you give a brief explanation of mass teleportation an mode switching phenomenom (line 394)?

4 - The order of OU sampling error (reversing the noising process) is known to be independent of $d$ and similar to the OU process accuracy [De Bortoli - NeurIPS2021]. Where does the dependence on $d$ come from in your derivation in line 444?

5 - Can the authors detail how they incorporate Alg. 2 in the mentioned algorithms to get new results?

6 - Alg. 2 seems like a typical diffusion sampler weight update, e.g. similar to that of PIS [Zhang & Chen 2022]. Is the difference with PIS in decomposition of the Brownian motion?

Minor comments:
Typo: line 316 ALMC -> LMC

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
The authors show query complexity bounds for normalization constant estimation using annealing-based methods including Jarzynski equality for SDE's and annealed importance sampling. The details for JE and AIS are different (being continuous/discrete), but they share a similar analysis framework. 

The bounds are in terms of the "action", or total squared metric derivative through the annealing sequence. The authors give an example where this is exponentially large in a situation with mass teleportation, and use this to motivate applying JE to reverse diffusion samplers through a polynomial bound on the action only under smoothness and 2nd moment.

### Strengths
This is a very timely topic given the increasing presence of these algorithmic methods (especially Jarzynski's equality which has been adapted for sampling with flow-based neural network models) and the limited theory for them. Normalizing constant estimation is a fundamental problem in ML and statistics, and the authors do a good job of giving examples of applications. The paper is technically well-written, and I especially appreciate the careful treatment of the forward/backwards SDE's. I expect the theory developed in this paper will be useful for future analyses on this and similar algorithms.

The application to reverse diffusion samplers is an especially nice result. This shows that efficient normalizing constant estimation follows from general conditions given an accurate score estimate.

### Weaknesses
The proof is written just for geometric interpolation, though it would be better to have a general bound that works for an arbitrary annealing sequence and added drift, as this flexibility is often important in the literature, especially in approaches that depend on learning the annealing sequence and drift (e.g. Máté and Fleuret, 2023; Albergo and Vanden-Eijnden, 2025). Would a similar proof work, or is the proof specialized to the annealing sequence? What changes are necessary? Relatedly, would this framework apply to give guarantees for sampling?

The main downside is the polynomial dependence in all bounds on the "action." While this is weaker than isoperimetry, it's not clear whether this matches other bounds for annealing/tempering-type methods. In fact, I doubt this covers many cases where we'd want to use annealing-based methods: The authors show a lower bound for the action in the case of a mixture of 2 (1-D) gaussians using power tempering. It seems this analysis can also be adapted to show a lower bound in any case with well-separated modes where their relative weights change, a situation which some other analyses of tempering-based methods are able to tackle; this would mean that in multimodal settings, the bounds in the paper are only effiicient when the relative weights don't change. Can the authors please clarify when they expect the action to be polynomial, and how this assumption relates to other "beyond isoperimetry" assumptions in the literature (e.g. multimodal/mixture settings)? Under what settings do we expect the action to be reasonable? (The reverse diffusion sampler is an example where this is nice, which I appreciate.)

I would be happy to increase my score if these points are satisfactorily addressed.

### Questions
In Theorem 4, the authors interpolate from a log-concave distribution, with the partition function for the log-concave distribution estimated separately via TI. However, one could also attempt to use JE through the entire sequence from the Gaussian approximation. How would these compare theoretically or experimentally? Is there a reason to prefer one or the other besides convenience of theoretical analysis?

The statement of Lemma 6 is a bit confusing, since what is meant is that the conditions are to be satisfied for small enough constants, not arbitrary constants. Please clarify this in the lemma statement (right now this is only in Remark 5).

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The work proposes a scheme for normalizing constant estimation based on the annealed Langevin framework. It shows, via a relatively novel annealing scheme, that one can achieve multiplicative accuracy with number of queries depending polynomially on the action of the curve, defined as the integrated norm of the metric derivative (in a Wasserstein sense).

### Strengths
I think this is an interesting and useful result, providing a principled means of normalizing constant estimation with interpretable guarantees. I concur with the authors that this is an important problem in computational statistics.

The connections between this and annealed sampling algorithms are elegant and nicely parallel the connection in discrete settings. This fact is also noticed by the authors.

The connections to statistical mechanics are always nice to see.

### Weaknesses
I find the action of a curve to be a somewhat inscrutable quantity from the perspective of algorithm design. The authors do a good job of trying to make this quantity accessible to unfamiliar readers but it certainly merits further investigation.

### Questions
(2) and (3) do not seem to be time reversals of each other? Or is ``backward’’ meant in a more pedestrian sense?

23: Girsanov theorem -> Girsanov’s theorem

111: denoising diffusion model -> denoising diffusion models

136: estimating normalizing constant -> estimating the normalizing constant

287: for the study non-asymptotic -> for the study of non-asymptotic

390: full proof is in -> The full proof is in

463: biased estimate -> biased estimates

### Soundness
3

### Presentation
3

### Contribution
3
