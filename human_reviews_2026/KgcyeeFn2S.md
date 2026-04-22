# Expanding the Chaos: Neural Operator for Stochastic (Partial) Differential Equations

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
Stochastic differential equations (SDEs) and stochastic partial differential equations (SPDEs) are fundamental tools for modeling stochastic dynamics across the natural sciences and modern machine learning. Developing deep learning models for approximating their solution operators promises not only fast, practical solvers, but may also inspire models that resolve classical learning tasks from a new perspective. In this work, we build on classical Wiener–chaos expansions (WCE) to design neural operator (NO) architectures for SPDEs and SDEs: we project the driving noise paths onto orthonormal Wick–Hermite features and parameterize the resulting deterministic chaos coefficients with neural operators, so that full solution trajectories can be reconstructed from noise in a single forward pass. On the theoretical side, we investigate the classical WCE results for the class of multi-dimensional SDEs and semilinear SPDEs considered here by explicitly writing down the associated coupled ODE/PDE systems for their chaos coefficients, which makes the separation between stochastic forcing and deterministic dynamics fully explicit and directly motivates our model designs. On the empirical side, we validate our models on a diverse suite of problems: classical SPDE benchmarks, diffusion one-step sampling on images, topological interpolation on graphs, financial extrapolation, parameter estimation, and manifold SDEs for flood prediction, demonstrating competitive accuracy and broad applicability. Overall, our results indicate that WCE-based neural operators provide a practical and scalable way to learn SDE/SPDE solution operators across diverse domains.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a neural operator that solves stochastic differential equations by transforming the random noise into deterministic features using a Wiener Chaos Expansion. This allows the model to learn the solution operator and generate full trajectories in a single computation, providing a fast and accurate method for a wide range of problems.

### Strengths
1. The proposed method offers theoretical approximation guarantees and is formulated with rigor.

2. Two distinct neural operators, F-SPDENO and SDENO, are introduced to address SPDE and SDE problems, respectively.

3. The model's performance is comprehensively validated across multiple domains, demonstrating its broad applicability.

### Weaknesses
1. The multiple theorems, definitions, and propositions appearing in the text should be clearly delineated to indicate which are newly proposed in this paper and which are derived from classical theories.

2. The authors have not adequately addressed the limitations of this study or its potential for future development.

### Questions
1. If I understand correctly, a fundamental limitation of this work lies in the requirement of prior knowledge about the entire noise process W. However, in practical applications, W is often difficult to observe and may exhibit complex dynamical behaviors—potentially coupled with chaotic dynamics—making it challenging to recover this noise process solely from observable data.

2. When applied to diffusion generative models, the authors demonstrate that a one-step sampler can be achieved. However, the denoising process (reverse noise) is still required here. Does this imply that the purpose of applying this method is solely to accelerate the denoising process? In fact, existing studies have explored one-step generation schemes. Therefore, it is necessary to compare the proposed approach with these existing schemes, or at least highlight the potential advantages of the method presented in this paper.

3. See weaknesses.

### Soundness
2

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
4

### Summary
The paper proposes a novel neural operator (NO) for SPDEs and SDEs based on their solution’s Wiener Chaos Expansion (WCE). Experiments across diverse tasks, including SPDE benchmarks, diffusion one-step sampling, topological interpolation, data extrapolation, parameter estimation, and manifold SDE, demonstrate the performance of the proposed method.

### Strengths
- The paper proposes a new neural operator specific to SPDEs and SDEs.
- The paper introduces the theoretical foundation of this method, Weiner Chaos Expansion, in detail.
- The experiments span many areas, from the numerical solutions of SPDEs to application problems like diffusion sampling and parameter estimation.

### Weaknesses
- The experiment does not consider these two models [Peiyan Hu, Qi Meng, Bingguang Chen, Shiqi Gong, Yue Wang, Wei Chen, Rongchan Zhu, Zhi-Ming Ma, and Tie-Yan Liu. Neural operator with regularity structure for modeling dynamics driven by
spdes. arXiv preprint arXiv:2204.06255, 2022.], [Shiqi Gong, Peiyan Hu, Qi Meng, Yue Wang, Rongchan Zhu, Bingguang Chen, Zhiming Ma, Hao Ni, and Tie-Yan Liu. Deep latent regularity network for modeling stochastic partial differential equations. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 7740–7747, 2023.], which are mentioned in the related work.
- The experiment of the Diffusion One-step Sampler does not contain the **quantitative** results during **evaluation** like FID, which makes the results not convincing enough. Also, as a one-step distillation method, there should be comparisons with other well-known methods, like consistency models. The experiments of Topological Interpolation, Extrapolation, Meta-SDENO, and WCE on Manifold also have the same problems. I think it's better to have one or two experiments on SDEs, but with more comprehensive results, at least quantitative evaluation metrics, and baselines.
- There is no reproducibility statement and code repository for reference, which limits the credibility of the experiments.

### Questions
- In F-SPDENO, the authors represent $u_α(t, x)$ in a temporal basis $ϕ_k(t)$. The approximately equal sign in line 235 indicates that the relation is obtained through an approximation, meaning that an approximation is involved. In other words, the linear combination of $\phi_k(t)$ cannot exactly represent all possible $u_\alpha(t, x)$. Is that correct? If so, will this affect the expressiveness and performance?
- What is the inference time of the proposed model compared with other baselines?

### Soundness
1

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The submission proposes a neural operator for stochastic (partial) differential equations (SPDES). The main idea is to Wiener-Chaos-expand the solution of a given SPDE, in which case the coefficients of said expansion solve systems of deterministic partial differential equations (PDE). Then, this system of PDEs can be solved with existing neural operator techniques, like Fourier neural operators.
The resulting SPDE-neural operator is then benchmarked on a wide range of experiments, including stochastic Navier Stokes, diffusion samplers, ensemble Kalman filters, and manifold SDEs.

**Summary of my recommendation**
Although I find the range of the experiments quite impressive, I recommend rejecting this work. The reason is that the submission is presented as a method and theory paper, as opposed to predominantly experimental work, but I find the theory and method contributions are not as novel and significant as the introduction and abstract state.
Furthermore, the volume of the submission (50 pages) is not representative of the volume of its actual contributions; more on this below.
I would have given the paper a better score if it had been presented as experiment-centric work: namely, if
- known results were removed from the paper (for instance, proofs of orthonormality of Wick polynomials or Doob's Martingale inequality);
- existing work on neural operators (and PINNs) for SPDEs and Wiener Chaos expansions was acknowledged more openly; 
- quantitative experimental results were expanded upon;

then, I think the contribution would have been more convincing.
Since implementing these changes would require a major rewrite of the submission, I do not think this is in scope for a revision during the rebuttal period. Therefore, unless I have missed something (in which case I am open to revising my score), I recommend rejecting this work. Details on each point follow below.

### Strengths
**I appreciate that all the technical explanations are rigorous.** Sometimes, the literature on machine learning with SPDEs can be a bit loose with technicalities (eg filtrations), so it is nice to see the submission take the maths seriously. At times this perspective might be pushed a bit too far, e.g., by including proofs of widely known statements from stochastic analysis, which distracts from the rest of the paper. Still, overall, I appreciate a rigorous treatment of SDEs in the machine learning literature.

**The range of experiments is impressive.** The numerical demonstrations include superconductivity, Navier-Stokes, diffusion samplers, topological interpolation, parameter estimation, and manifold SDEs, and  as such, the submission goes beyond what I would have expected from a method paper. I consider this scale and range to be the biggest strength of this submission. 
As mentioned under "Summary", I think that the submission would have been stronger had the experiments received a significantly more prominent role in the manuscript. Removing the known theoretical results would have also given space for further quantitative benchmarks; currently, Tables 1 and 2 show results for superconductivity and Navier-Stokes, respectively, but the remaining four experiments are entirely qualitative demonstrations instead of strict benchmarks. I understand the point made in lines 356f that the diffusion sampler is a proof of concept, not necessarily intended to improve the state of the art, but I think more qualitative results (that is, reporting more runtimes and RMSEs) would strengthen the evaluation. Nonetheless, I think the experiments are a strength of the submission.

### Weaknesses
**I think that the contributions claimed in the abstract and intro are slightly overstated.** Concretely, the abstract mentions that Wiener Chaos Expansion (WCE) theory is being extended; however, Theorems 1 and 2 discuss the WCE of SPDEs, which is a standard result (e.g., Neufeld & Schmoker, 2024; Lototsky & Rozovskii, 2006). The remaining definitions, lemmas, and analyses in Appendices B and C are also standard results from stochastic analysis. 
Regarding the analysis of the SPDENO in Appendix D: the analysis hinges on the assumption in Equation (92) that the error decomposes into the sum of three terms (which need not be the case), and then each term is bounded by existing results (Neufeld & Schmocker, 2024; Kovachki et al., 2023). Overall, I think that the lack of theoretical contributions of this work, together with the fact that the abstract and intro claim the contrary, is a major weakness of the submission.



**Novelty of the method:** I agree with the statement in line 069 that the submission is the first work to deploy neural operator models for S(P)DEs at scale (with a focus on "at scale"). 
However, the method itself is not new. Breaking SPDEs down into systems of coupled PDEs, and using PDE solvers, is a typical strategy for numerical treatment of SPDEs (eg Foster et al., 2020, which I mention under "Questions" below). 
There have also been machine-learning works that use this deterministic representation of stochastic equations, for example, Eigel and Miranda (2024), Salvi et al. (2022), Neufeld and Schmocker (2024); see also: 

> Jared O’Leary, Joel A Paulson, and Ali Mesbah. Stochastic physics-informed neural ordinary differential
equations. Journal of Computational Physics, 468:111466, 2022.

> Ling Guo, Hao Wu, and Tao Zhou. Normalizing field flows: Solving forward and inverse stochastic
differential equations using physics-informed flow models. Journal of Computational Physics, 461:
111202, 2022.

Importantly, Eigel and Miranda (2024) propose the same neural operator for SPDEs as the submission. They don't conduct experiments at the same scale, but the method exists. As such, I think that the novelty of the proposal is limited, and since the novelty of the method is a major component of the claimed contributions in this submission, I consider novelty to be a major weakness.



**Volume:** Related to novelty, I am a bit unsure whether the volume of contributions in this paper warrants a 50-page submission. Especially since the theoretical results are all essentially known, and could be handled with an appropriate citation instead of the full derivation in Appendices B and C. I think such unnecessary proofs distract from the main contributions of this paper (which, in my view, is demonstrating SPDE neural operators at scale, see the discussion under "Strengths").

In summary, I think the submission contains interesting results about neural operators for SPDEs, but in its current presentation, there are too many weaknesses to warrant acceptance. 
I am not saying that every ICLR paper needs to present novel theory and a novel method and excessive experiments to be accepted. However, the presentation of the results needs to match the actual contributions of the paper, and for the present submission, there is a mismatch in both theoretical (which is oversold) and experimental novelty (which is undersold). This mismatch is why I recommend rejecting this work.

### Questions
The following questions do not affect my score, but I think discussing these points would be a useful addition to a future version of this manuscript.


- Section 3.1 constructs Q-Wiener processes from Brownian increments and, like Neufeld and Schmocker (2024), uses Wick polynomials for Polynomial Chaos Expansions. There are alternative approaches for polynomial approximation of stochastic processes (eg Foster et al. below); why choose the Wick polynomials? Is there an optimal choice for basis representations?

> Foster, James, Terry Lyons, and Harald Oberhauser. "An optimal polynomial approximation of Brownian motion." SIAM Journal on Numerical Analysis 58.3 (2020): 1393-1421.

- Tables 1 and 2 list relative L2 errors. The proposed method performs best, but the numbers are quite close. It would be great to include standard deviations in both tables to get a feeling for whether the results are statistically significant. Is this possible?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper learns the solution operators of SDEs and SPDEs by using the solutions’ Wiener Chaos Expansion. It achieves this by decomposing the noise in the orthonormal Wick-Hermite feature space and learning the corresponding propagator coefficients. The method is applied to several SPDE and SDE learning problems. Experimental results demonstrate its effectiveness.

### Strengths
1. The method is well established under rigorous analysis. Detailed theoretical analysis and guarantees are provided. But I did not check all the details of the derivations.
2. The evaluations are conducted on various kinds of tasks, covering physical, image, graph, financial, and manifold domains, demonstrating the superiority and generalization of the proposed method.

### Weaknesses
1. My main concern is that the presentation of the practical aspect of the method is not good enough. The workflow of using the method is ambiguous. From the implementation perspective, many details are missing. Thus, it is hard to connect the established theory with practice. It is better to formalize the problem setup in Section 2. In Section 4 and Figure 2, where do Q-Brownian realization increments come from? Why does the workflow start with Q-Brownian motion? How to compute Wick features using Q-Brownian realization? Does the computation of deterministic PDE involves some learning procedure using deep neural networks?
2. Some experimental results are not complete for convincing comparison. For example, 

- (1) In Figure 3, more visualisation results of baselines should be provided for comparison. 

- (2) In Figure 5, it seems result on only one instance is presented. More statistics should be provided on a collection of test instances.

- (3) In Section 5.4 Manifold SDEs, the presentation of results (Figure 8) is rather limited, and there is no comparison with baselines. Therefore, it is hard to appreciate the performance.

### Questions
1. For Figure 1, what conclusion does the comparison between Brownian and Q-Brownian motions aim to reveal? In the right subfigure, it seems that the reconstruction using $N=5$ is a biased estimation (much lower compared to other lines)?
2. For Table 1, why does using more training trajectories ($N$=1,000 vs 10,000) not improve the performance of NSPDE and the proposed F-SPDENO?
3. For Figure 5, for clarification, the top row and bottom row should be labelled as TSBM and G-SDENO (as I guess), respectively. Besides, it should be pointed out how the 1-Wasserstein distance (WD) metric reflects performance (whether larger or smaller values are preferred?).
4. The last line in Definition 1 seems like a lemma. It is better to add a citation here for reference.

### Soundness
3

### Presentation
2

### Contribution
3
