# Empirical NTK tracks task complexity

- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
Mathematical properties of the neural tangent kernel (NTK) have been related--both theoretically and empirically--to convergence of optimization algorithms and the ability of trained models to generalize. However, most existing  theoretical results hold only in the infinite width limit and only for standard data distributions. In the present work, we suggest a practical approach to investigating the NTK for finite-width networks, by understanding the parameter-space symmetries of the network in the presence of finite data sets. In particular, the NTK Gram matrix associated to any finite data set can naturally be regarded as an empirical version of the NTK. Moreover, its rank agrees with the functional dimension of the data set, the number of independent parameter perturbations affecting the model’s outputs on the data set. In this work, we explore the evolution of the functional dimension of deep ReLU networks during training, focusing on the relationship to data set complexity, regularization, and training dynamics. Empirically, we find that functional dimension of deep ReLU networks: (1) tracks data set complexity, (2) increases during training until function stabilization, and (3) decreases with stronger weight decay, suggesting that gradient-based optimization algorithms are biased towards simpler functions for ReLU networks. Moreover, our experiments provide strong evidence that--contrary to conventional wisdom--the empirical NTK for deep finite-width ReLU networks is typically rank-deficient at initialization. We offer a potential theoretical explanation for this empirical phenomenon in terms of certain data-dependent hidden equivalences, emphasizing the  connection between these equivalences and the geometry of the loss landscape. We also establish a theoretical upper bound on functional dimension in terms of the number of linear regions sampled by the data set.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies the rank of the empirical NTK (eNTK) during training, both empirically and theoretically. The main contributions are:
(1) Empirical evidence showing that the eNTK computed on a batch is typically rank-deficient at the beginning of training and increases during training for several small synthetic problems.
(2) The observation that the final eNTK rank increases with task complexity and decreases with weight decay strength in the same synthetic problems.
(3) A theoretical bound on the eNTK rank for ReLU networks, formulated in terms of the number of neurons that remain always-active or always-inactive on the whole batch. The theoretical result is derived by calculating the dimension of the network’s symmetry group, thereby connecting eNTK rank deficiency to intrinsic parameter-space symmetries of ReLU architectures.

### Strengths
**(S1) Empirical evidence of rank deficiency in the eNTK:** The empirical results showing that the eNTK is rank-deficient at initialization are interesting, as they contradict the common assumption of full rankness often made in the NTK-regime literature.

**(S2) Relationship between symmetries of the output function and NTK rank:** The observation that the eNTK rank can be upper-bounded by the dimension of the symmetry group of the network’s output function is also interesting. While this connection is intuitive, it appears to be novel. However, it is unclear how practically useful this observation is, as discussed in the Weaknesses section.

### Weaknesses
**(W1) Connection between theory and experiments:** The empirical results and the theoretical bound seem largely disconnected. Although the authors show that the eNTK is rank-deficient empirically, they do not compare the theoretical bound from Thm.1 to the actual eNTK ranks observed in practice. The related issues discussed below suggest that the bound may not be predictive in the studied settings.

**(W2) Clarity of Thm.1 statement:** After reading the statement of Thm. 1, one could think that the stated upper bound may be negative with large enough $n_\ell^{\text{fixed}}$. For example, consider the architecture used in the 1D experiments, with widths $(1,15,15,15,15,1)$ and suppose $n_\ell^{\text{fixed}}=14$ in each hidden layer. Then the total number of parameters is $D=15\times 2+15\times 16\times 3+16=766$ and $d = 4\times 14\times 14+4=788$, giving $D-d<0$. Since $d$ is the orbit dimension, it cannot exceed $D$, so I suppose this case is hidden in the "almost any parameter" part of the statement (nontrivial stabilizers). However, it is not clearly stated in the theorem and does not appear in any discussion, as far as I can see.

**(W3) Is the bound in Thm. 1 mostly vacuous?** Using the same setup as in the example above but with $n_\ell^{\mathrm{fixed}}=13$, we obtain $D-d=82$, already much larger than the experimental eNTK ranks in all the provided figures. Reducing $n_\ell^{\mathrm{fixed}}$ further quickly hits the trivial upper bound on the empirical NTK rank (the “batch functional dimension” in the paper's language) given by the batch size ($100$ in the experiments). This raises the question of how often the provided bound is actually non-vacuous.

**(W4) Writing and presentation:** The paper is quite difficult to follow. Many results supporting the main claims (e.g., Figs. 7, 8, 10, 11, Proposition C.3) appear only in the appendix, while much of the main text is devoted to describing a general framework (Section 3) that is not novel, accompanied by a few very large figures. There are also editing problems, such as missing references (line 305) and incorrect appendix links (line 344).

**(W5) Scope of experiments:** The experimental scope is quite limited, and the motivation for the chosen setups is not clearly discussed. Experiments on real datasets or with more varied task structure and complexity would strengthen the results.

Overall, while the paper contains an interesting idea, the current version is not sufficiently convincing in my opinion. Therefore, I am leaning toward rejection. My assessment could improve if the authors respond to the concerns regarding the main theoretical bound, its relationship to the experiments, and the experimental scope.

### Questions
- What is the relationship between the main result (Thm.1) and the “roughly monotonic” assumption (A.1)?
- Did the authors compute the actual values of the bound in Thm.1 for their experimental settings?
- How do the results connect to prior work that typically observes a reduction in *effective rank* of the eNTK at the start of training? See, for example, experiments in Baratin et al. (2021) [1].

## References

[1] Baratin, Aristide, et al. “Implicit regularization via neural feature alignment.” *International Conference on Artificial Intelligence and Statistics (AISTATS)*, PMLR, 2021.

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper analyzes the empirical NTK rank and its change during training. Claims are made based on two experiments on simple synthetic data and a simple model. A bit theory is developed on the symmetry of the network, but lacks connection with the paper’s major goal.

### Strengths
n/a

### Weaknesses
1: A connection of the theory of the paper, simply Theorem 1, and the major claims of the paper is missing. The paper mainly claims about the rank of empirical NTK and its change during training. However, (a) Theorem 1 seems to be independent of the training procedure. (b) what is its implication on the NTK rank?


2: The presentation of the paper can be largely improved, especially in Section 4.1. 
In most of this section, the paper is just filled with a few technical mathematical concepts which have no clear connection to the paper’s context. This makes it hard to understand the discussion, even though I have the knowledge of these math concepts. 

The connection to the paper’s context is made clear only in Remark 4.3 (quite near the end of the section), where permutation and scaling actions are discussed. I believe a better way is to (1) put the permutation and scaling actions early on (also not under a remark), and (2) put the technical concepts, such as Lie Group, orbits etc, into appendix.

In addition, given the symmetry (permutation and scaling) is quite limited and simple, I doubt the necessity of introducing the concept of Lie groups. 

3: As an empirical work, this paper only experimented on extremely simple synthetic data, which are far from enough to be conclusive. I expect verifications of the empirical claims across several real-world datasets and network architectures.

### Questions
no further questions

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
4

### Summary
This paper investigates the empirical Neural Tangent Kernel (NTK) in finite-width deep ReLU networks, focusing on its relationship with task complexity and training dynamics. The authors show that, contrary to common belief, the empirical NTK is typically rank-deficient at initialization. They demonstrate that its rank increases during training until function stabilization, correlates positively with dataset complexity, and is suppressed by stronger weight decay. Theoretically, they link low functional dimension to data-dependent parameter-space symmetries and provide an upper bound based on the number of linear regions sampled by the data. These findings offer new insights into optimization biases and implicit regularization in deep ReLU networks.

### Strengths
1. The paper presents a highly original and significant empirical finding by demonstrating that the empirical NTK for deep, finite-width ReLU networks is consistently rank-deficient at initialization, contrary to conventional wisdom.

2. The work provides creatively linking the NTK's rank to the novel concept of data-dependent parameter-space symmetries and providing an upper bound based on linear regions, it offers a compelling and elegant mechanistic explanation for the observed low-rank phenomenon.

3. The paper achieves remarkable clarity by grounding abstract concepts like the NTK and functional dimension in concrete, measurable quantities (matrix rank, linear region count).

### Weaknesses
1.  The experimental validation is conducted exclusively on small, synthetic, noiseless datasets (univariate and bivariate functions). This leaves it unclear whether the central finding—that the NTK rank tracks task complexity—generalizes to high-dimensional, noisy, real-world data (e.g., CIFAR or ImageNet), where the notion of "complexity" is less well-defined.

2. While the paper provides a compelling theoretical explanation (data-dependent symmetries from fixed activation patterns), it lacks direct empirical measurement to substantiate this as the primary cause. Quantifying the number of "always-active/inactive" neurons throughout training and directly correlating this count with the observed drops in functional dimension would significantly strengthen the theoretical results.

3. Proposition C.3 establishes an upper bound for functional dimension based on the number of linear regions, but the results show the actual dimension is "well below" this bound. The paper does not sufficiently investigate what factors determine this gap.

### Questions
1. Your theoretical framework and experiments are compelling on low-dimensional synthetic data. Could you provide evidence or discuss whether the strong correlation between empirical NTK rank and "task complexity" holds for higher-dimensional, real-world datasets (e.g., CIFAR, ImageNet )? In such domains, how would you propose to define or measure "task complexity" independently to validate this relationship?

2. You propose that data-dependent symmetries (from always-active/inactive neurons) are a primary cause for the low-rank NTK. Did you track the proportion of such neurons throughout training? Could you show that a drop in functional dimension coincides with an increase in neurons achieving a fixed activation status?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper investigates the empirical behavior of Neural Tangent Kernels (NTKs) across various ML algorithms, with an emphasis on understanding how NTK dynamics reflect or correlate with the underlying complexity of the algorithms. The authors provide a comprehensive experimental and analytical study on the evolution of the empirical NTK during training, examining its relation to key measures of task difficulty.

### Strengths
- The paper shows extensive experimental evaluations. 

- The conceptual innovation of functional dimension as a data and task dependent metric is significant.

### Weaknesses
A few important discussions with existing literature are missing.

### Questions
How robust is the correspondence between functional dimension, as defined via the NTK spectrum, and the capacity of a network to generalize?

Can you compare your results with the Scaling neural tangent kernels via sketching and random features? 

What are examples of nonstandard data distributions for which the analysis in 'Fine-Grained Analysis of Optimization and Generalization for Overparameterized Two-Layer Neural Networks' fails to accurately capture the behavior of finite-width networks?

### Soundness
3

### Presentation
4

### Contribution
3
