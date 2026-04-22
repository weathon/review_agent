# Efficient Regression-based Training of Normalizing Flows for Boltzmann Generators

- Avg Score: 6.67
- Decision: Accept (Poster)
- Scores: 6, 6, 8

## Abstract
Simulation-free training frameworks have been at the forefront of the generative modelling revolution in continuous spaces, leading to large-scale diffusion and flow matching models. However, such modern generative models suffer from expensive inference, inhibiting their use in numerous scientific applications like Boltzmann Generators (BGs) for molecular conformations that require fast likelihood evaluation. In this paper, we revisit classical normalizing flows in the context of BGs that offer efficient sampling and likelihoods, but whose training via maximum likelihood is often unstable and computationally challenging. We propose Regression Training of Normalizing Flows (RegFlow), a novel and scalable regression-based training objective that bypasses the numerical instability and computational challenge of conventional maximum likelihood training in favour of a simple $\ell_2$-regression objective. Specifically, RegFlow maps prior samples under our flow to targets computed using optimal transport couplings or a pre-trained continuous normalizing flow (CNF). To enhance numerical stability, RegFlow employs effective regularization strategies such as a new forward-backward self-consistency loss that enjoys painless implementation.  Empirically, we demonstrate that RegFlow unlocks a broader class of architectures that were previously intractable to train for BGs with maximum likelihood. We also show RegFlow exceeds the performance, computational cost, and stability of maximum likelihood training in equilibrium sampling in Cartesian coordinates of alanine dipeptide, tripeptide, and tetrapeptide, showcasing its potential in molecular systems. Code available at: https://github.com/danyalrehman/RegFlow.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes to train normalizing flows by regression to alleviate the training instability in traditional MLE training to transform samples from a prior $q$ to a target distribution $p$, ie $x_0\sim q$ and $x_1=T(x_0)\sim p$. To enable regression-based training, it requires that there exists an invertible map between $x_0$ and $x_1$. Finding an exact solution is intractable, and therefore the author proposes two different approximations: 1. through coupling $(x_0, x_1)$ with optimal transport; 2. through coupling $(x_0, x_1)$ by training an additional CNF and letting $x_1$ be generated from $x_0$ through the trained CNF.

The author justifies the usage of their CNF-coupling by showing the error bound wrt wasserstein distance between the learned distribution and the target distribution. The experimental results also showcase the effectiveness of the proposed method, through various of peptide tasks.

In summary, the reviewer gives a weak acceptance.

### Strengths
1. The proposed method is simple and effective, which improves the training stability of NFs

2. The author provides mathematical justification for using the CNF-coupling, by showing the wasserstein distance between trained model and target.

3. Though the CNF-coupling sounds expensive at the first glance, the author shows that it requires much less computational overhead in table 4.

4. The experimental results are good

### Weaknesses
1. The main concern lies in the invertibility of coupling, as both OT-coupling and CNF-coupling are approximations. It would be great if the author could elaborate on when this approximation would be broken and, in such a case, how the classic MLE objective would help. Intuitively speaking, if $p_0$ and $p_1$ are too separate, the true velocity might be less smooth and the (t-dependent) Lipschitz constants can be large, which means the Wasserstein bound can be very loose.

2. The experiments in this paper focus on training NFs and then doing importance sampling to get equilibrium samples. However, in the plots, such as Figure 2, the author only shows the reweighted energy histogram but not the resampled Ramachandran plots. The reviewer thinks it is important as well to show the rama-plots with recalibrated samples.

### Questions
1. Can the author also show the training time comparison analogous to table 4 on other benchmarks?

2. Table 3 is a bit confusing. Why are the inference times of the same NF but different training methods (MLE/RegFlow) different?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This manuscript outlines a new approach to train one-step generative models, specifically normalizing flows, which allow exact sample likelihood evaluations. This work is important as likelihood evaluation is a bottleneck for applications of NF in the sciences, as most prominently outlined here: Boltzmann Generators. The idea outlined in this paper is simple and effective: train a regular NF against a pre-specified flow (e.g. either a pre-trained Continuous NF, or a pre-computed OT flow). This allows for faster and more stable training and more efficient sample likelihood evaluations in most cases.

While the method seems like it still has some ways to go to be ready for prime-time, I find that the paper, overall, is an interesting conceptual step. Consequently, i am willing to increase my score if the concerns below are addressed.

### Strengths
- Conceptually clear and well written manuscript. Numerous insightful comments about normalizing flows.
- Well thought-out experiments and evaluations. All fairly standard in the field now, but still well done.
- Clear performance gain, in terms of compute-time, over most of the included baselines.

### Weaknesses
- Claims and attribution. The proposed TFEP method is closely related to the ambient thermodynamic interpolant approach by Moqvist et al https://arxiv.org/abs/2411.10075 
- Sample quality and scaling. ESS remain fairly low. Scaling to tetra peptides is nice several other recent works e.g. https://arxiv.org/abs/2502.18462 demonstrate scaling to significantly larger systems. 
- Lack of error estimates on evaluation statistics.

### Questions
- How do the authors envision this approach scale to larger systems? Comparing figures 7 and 8 it seems like the RegFlows miss important details that might be important for performance, and one would expect this to only increase with system size.
- Are observables --- e.g. free energies --- computed under the DiT-based CNF meaningfully different from those under the reweighed values from the presented RegFlow approach?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper describes REGFLOW, an approach for training a normalizing flow that performs better than traditional maximum likelihood training.  The approach regresses to predetermined flows, either from another model or precomputed optimal transport couplings. This approach results in substantial better normalizing flow models than traditional training.

### Strengths
The paper addresses a chronic problem with normalizing flows.  The described regression loss is intuitive an simple to implement (once couplings have been determined).  The approach results in dramatic improvements compared to MLE training using the same models and data.  Sensitivity to some parameters (e.g. regularization) is explored. I appreciate the evaluation of free energy.  The paper is well written and easy to follow.

### Weaknesses
Although the improvement compared to NF models is extreme, the results aren't necessarily state-of-the-art compared to other models.

### Questions
Why are NFs trained with REGFLOW substantially faster at computing likelihoods?

How does increasing the number of OT couplings improve performance? What if the OT is approximate?

What is the basis for the statement that beyond a certain level of regularization that numerical invertability is guaranteed? Is this an empirical statement, or is there are proof (if the former, than perhaps an alternate wording would be more accurate).

### Soundness
4

### Presentation
4

### Contribution
4
