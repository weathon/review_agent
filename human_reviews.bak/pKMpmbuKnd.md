# Constrained Posterior Sampling: Time Series Generation with Hard Constraints

- Decision: Reject
- Scores: 8, 8, 6, 5

## Abstract
Generating realistic time series samples is crucial for stress-testing models and protecting user privacy by using synthetic data. In engineering and safety-critical applications, these samples must meet certain hard constraints that are domain-specific or naturally imposed by physics or nature. Consider, for example, generating electricity demand patterns with constraints on peak demand times. This can be used to stress-test the functioning of power grids during adverse weather conditions. Existing approaches for generating constrained time series are either not scalable or degrade sample quality. To address these challenges, we introduce Constrained Posterior Sampling (CPS), a diffusion-based sampling algorithm that aims to project the posterior mean estimate into the constraint set after each denoising update. Notably, CPS scales to a large number of constraints ($\sim100$) without requiring additional training. We provide theoretical justifications highlighting the impact of our projection step on sampling. Empirically, CPS outperforms state-of-the-art methods in sample quality and similarity to real time series by around 10\% and 42\%, respectively, on real-world stocks, traffic, and air quality datasets.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
In this paper, the authors proposed a diffusion-based sampling algorithm - Constrained Posterior Sampling (CPS), to generate time series data meeting certain constraints. The basic idea is solving a regularized optimization task in every denoising step and the penalty term  will be strengthened such that the resulting data fall within the constrained set. Overall, I think this paper is well-written, has clear presentation and motivation. I will consider raising my score if the authors handle my questions properly.

### Strengths
**1** This paper has extensive experiments to show the effectiveness of their method.

**2** A theoretical result is provided for validating the effectiveness under a linear constraint case. 

**3** Proper baselines are chosen for comparison and comprehensive literature review is provided.

**4** The proposed method basically provides some interesting ideas of handling constrained generation in diffusion models. I think the method can be extended to other scenarios.

### Weaknesses
**1** In each step, a constrained optimization task needs to be solved, which can be computationally intensive. Additionally, the current framework is more like as a general framework without imposing specific restrictions on the constraints. It would be beneficial to develop a universal algorithm that aims to minimize the constrained optimization for a general class of constraints, even if the optimal solution is not achieved.

**2**  The fidelity of generated data is not properly evaluated in this paper. I suggest using classification method to for fidelity evaluation. That is using a classifier to discriminate real and synthetic data and use the testing error as fidelity metric.

### Questions
**Q1** Page 5, Lines 240-244 "Our key intuition is to design $\gamma(t)$ as a strictly **decreasing** function of $t$ that takes small values for the initial denoising steps (t close to T) and tends to $\infty$ for the final denoising steps". Does the function form of $gamma(t)$ affect the result? Is it possible to select others?

**Q2** The main contribution is the adaptive projection step that solves the penalized optimization task. From my understanding, this is more like a soft constraint and it becomes a hard one when $t=1$. However, in practice, how to choose $T$?

**Q3** In every denoising step, the protection step solves a regularized optimization task. However, the optimization problem might be difficult if some non-linear constraints are chosen. Is it guaranteed that the optimal or local optimal solution can be obtained?

**Q4** In Theorem 2, what is design parameter k? It seems missing in the algorithm.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes a method called CPS to sample time series from a diffusion model with given constraints. The trick is projecting the intermediate values in the diffusion model sampling process towards the constraint set. In later iterations of the sampling process, the diffusion model corrects the projected time series so it stays close to the data generating distribution. The paper derives some theoretical properties of their sampler, including a convergence theorem in the case of Gaussian data. The paper also contains experiments on several real and toy datasets, where CPS outperforms baselines.

### Strengths
The paper is very well-written, easy to read, and makes clear arguments. The proposed method CPS is simple, and seems easy to apply. It may even be applicable to other domains, like tabular data, where meaningful constraints can be specified as simple inequalities. The results are very good, with CPS outperforming baseline methods in almost all experiments.

### Weaknesses
I'm not sure how meaningful the similarity metrics comparing the generated sample and the test sample that satisfies the all constraints is, since an optimal generator may not get the best possible score. This is because any finite set of constraints is likely to leave some freedom to generate different outputs, which would not be identical to the reference sample from the test set. This is shown in Table 1, where none of COP, COP-FT and CPS violate any constraints on several datasets, but have different values for the DTW and SSIM metrics.

A comparison with Loss DiffTime would still be interesting, since it would show how beneficial training specifically for a given set of constraints is.

### Questions
- In Section 3, should $\hat{z}_0$, and various related symbols, have the subscript $t$? If not, why is 0 used there?
- What does the form of the distribution that CPS samples from, given in Theorem 1, tell us?
- How many constraints are imposed in Table 1 and Figure 6?
- How badly are the generated time series in Figure 6 violating the constraints?
- How is it possible that CPS is effectively copying samples (see traffic conditional and stocks datasets in Figure 6) from the test set that the generator has presumably not seen during training? Are these time series really characterised so precisely by a relatively small number of numbers?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates generating time series data samples with specific constraints. The authors propose a training-free constrained posterior sampling method based on $z-\hat{z}_{0,pr}$ optimization problem within objective function. Furthermore, they theoretically justify the use of imposing condition as a conditional sampling. They empirically prove their idea across various settings, including different setups and datasets.

### Strengths
• The paper investigate the very important and practical issues in time series generation, which encompasses generating particular examples with specific condition. This is the unique properties in time series generation compared to vision tasks.

• The authors propose a convex optimization based methods which is much more effective and simple for time series than existing methods.

• The paper conducts experiments on various datasets, including well-known ones, demonstrating the effectiveness of the constrained synthesis.

• The authors establish a theoretical background of their methods.

### Weaknesses
Please refer to the Questions section.

### Questions
•	Even though they propose new method, but the reviewer think the method is ot novel. It poses regularize similar to (Coletta et al., 2024), but with convex optimization strategy.

•	The author mentioned that the guided gradient is not enough for restricting the constraints. Could you clarify the main differences of two methods? In terms of convex optimization with global optimum, two of them should be same with proper set of hyperparameters.

•	Furthermore, the authors mentioned about the constraints sets as convex. However, the objective $||z-\hat{z}_{0,pr}||^2$ that the author proposed might not be optimum for non-convex optimization.

•  The comparison includes only (Coletta et al., 2024). Could you consider additional comparison methods, such as the instance-aware guidance strategy in [1]?

•	In Theorem 2, why does $k$ approach infinity as $T$ goes to infinity? More explanation on the convergence analysis would be helpful.

•	Please correct minor errors. Such as generation(Yoon et al.) in line 140 and Recently (Coletta ~) -> use citet.

[1] DIFFUSION-TS: INTERPRETABLE DIFFUSION FOR GENERAL TIME SERIES GENERATION, ICLR 2024.

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
The paper introduces Constrained Posterior Sampling, a diffusion-based method for generating time series that adhere to hard constraints. The approach is theoretically grounded and the empirical evaluations demonstrate CPS's effectiveness in producing high-quality, constraint-compliant time series. 
However, the novelty of CPS is somewhat limited due to its conceptual similarities with existing methods (see below for extended discussion) .

### Strengths
- The paper provides a solid theoretical basis for CPS, including proofs of convergence and constraint satisfaction.
- Experiments on real-world datasets demonstrate CPS's ability to generate high-quality and constraint-compliant time series.
- It seems like that the framework can handle a large number of constraints, and all at sampling time.

### Weaknesses
- The key weakness of this approach is novelty. CPS shares significant conceptual similarities with existing methods. 
In particular :
[A] also uses a projection step within diffusion processes to enforce constraints, and consider arbitrary constraint sets, while providing theoretical guarantees for convex constraint sets. The algorithm proposed in [A] is also very similar to the proposed one. 
[B] implements hard constraints on the outputs of autoencoders for linear constraints, as those considered in this paper. 
[C] uses “mirror mappings” which apply to convex constraint sets. 
[D] further extend the classes of constraints can be handled handling constraints represented by convex polytopes. 

- Given the above, the paper lacks a thorough comparison with existing methods and this omission makes it difficult to assess CPS's relative advantages or limitations.

- The projection step in CPS can be computationally intensive, especially for high-dimensional constraints, how can this be handled? 

[A] Jacob K Christopher, Stephen Baek, Ferdinando Fioretto. Constrained Synthesis with Projected Diffusion Models. NeurIPS 2024 - ArXiv: https://arxiv.org/abs/2402.03559v1 (first appeared on ArXiv on Feb 2024)

[B] Thomas Frerix, Matthias Nießner, and Daniel Cremers. Homogeneous linear inequality constraints for neural network activations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pages 748–749, 2020.

[C] Guan-Horng Liu, Tianrong Chen, Evangelos Theodorou, and Molei Tao. Mirror diffusion models for constrained and watermarked generation. NeurIPS 2023. ((first appeared on ArXiv on Oct 2023)

[D] Nic Fishman, Leo Klarner, Emile Mathieu, Michael Hutchinson, and Valentin De Bortoli. Metropolis sampling for constrained diffusion models. NeurIPS 2023. ((first appeared on ArXiv on July 2023)

### Questions
see limitations above

### Soundness
3

### Presentation
4

### Contribution
2
