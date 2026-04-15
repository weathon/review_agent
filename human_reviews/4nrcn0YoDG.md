# Global Identifiability of Overcomplete Dictionary Learning via L1 and Volume Minimization

- Decision: Accept (Poster)
- Scores: 5, 6, 6, 8

## Abstract
We propose a novel formulation for dictionary learning with an overcomplete dictionary, i.e., when the number of atoms is larger than the dimension of the dictionary. The proposed formulation consists of a weighted sum of $\ell_1$ norms of the rows of the sparse coefficient matrix plus the log of the matrix volume of the dictionary matrix. The main contribution of this work is to show that this novel formulation guarantees global identifiability of the overcomplete dictionary, under a mild condition that the sparse coefficient matrix satisfies a strong scattering condition in the hypercube. Furthermore, if every column of the coefficient matrix is sparse and the dictionary guarantees $\ell_1$ recovery, then the coefficient matrix is identifiable as well. This is a major breakthrough for not only dictionary learning but also general matrix factorization models as identifiability is guaranteed even when the latent dimension is higher than the ambient dimension. We also provide a probabilistic analysis and show that if the sparse coefficient matrix is generated from the widely adopted sparse-Gaussian model, then the $m\times k$ overcomplete dictionary is globally identifiable if the sample size is bigger than a constant times $(k^2/m)\log(k^2/m)$ with overwhelming probability. Finally, we propose an algorithm based on alternating minimization to solve the new proposed formulation.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper presents a novel  formulation for dictionary learning with the dictionary matrix being overcomplete.  Under certain conditions, the authors demonstrate that the novel formulation guarantees global identifiability on the overcomplete dictionary. Finally, the authors design  an alternating optimization algorithm to solve the proposed formulation.

### Strengths
It is impressive that the proposed formulation can guarantee  global identifiability over dictionary learning with an overcomplete dictionary matrix under some conditions.

### Weaknesses
1. It is not easy to verify whether $A$ and $S$ satisfy the Assumptions 3-4. Hence, it is difficult to evaluate the practical applicability of the theoretical results. 
2. The paper provides only a simple simulation experiment, and the results are somewhat unconvincing.
2. The theoretical results are related to the optimal solution to equation 2. However, the proposed optimization algorithm for solving equation 2 cannot  guarantee convergence to a global optimum.

### Questions
1. In Lemma 1, it seems that $\Phi=I$ only when the optimal solution to equation 2 is unique. Hence, if there are multiple optimal solutions, does Lemma 1 still hold? If not, how to demonstrate that the optimal solution to equation 2 is unique? 
2. How to prove that $A$ in Assumption 4 must exist? In addition, note that $A$ needs to satisfy Assumption 1 as well.
3.  In line 363, the authors state that they aim to check whether the optimal value of equation 12 equals to 1. However, Theorem 2 only gives the probability that the maximum value is greater than 1. What's the relationship between them?
4. Are optimization problems 14 and 2 equivalent? How to determine $\lambda$?
5. For the synthetic experiment, using the estimation error to evaluate the algorithm's performance is somewhat unconvincing. It is more reasonable to show that there exist a permutation matrix and a diagonal matrix that can convert the learned dictionary into the real one. In addition, multiple experiments should be conducted to record the corresponding success probability.   
6. Why didn't the authors compare the proposed algorithm with other dictionary learning algorithms in the experiment? Currently, only a simple experiment is available.
7. Where is the Figure mentioned in line 466?
8. Many sentences in Introduction overlap with Hu and Huang (2023a).

### Soundness
2

### Presentation
2

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
The paper proposes an approach for dictionary learning that uses a loss that mixes a modified, weighted version of the ell-1 norm of the mixture matrix coefficients (with different weights for different rows) with the volume of the dictionary matrix. It identifies a condition for successful identification of the mixing matrix called strong scattering. Similar to existing results, the likelihood of strong scattering for random mixing coefficient matrices such as sparse Gaussian, finding a scaling low for the number of vectors used in learning to scale like $\mathcal{O}\left(\frac{k^2}{m} \log \frac{k^2}{m}\right)$, where $k$ is the number of dictionary elements and $m$ is the data dimension. An alternating minimization algorithm for the proposed optimization is included as well.

### Strengths
The formulation appears novel and the analytical results are comprehensive.
A sound identifiability condition is presented.

### Weaknesses
As with other conditions for sparse learning and recovery, it appears that the required strong scattering condition cannot be efficiently checked.

It is difficult to assess how much stronger the sufficient scattering condition is versus "that of complete dictionary learning".

Some specific arguments are not clear (see questions).

A figure in the experimental section (cf. Line 466) is missing.

### Questions
Line 165: is a square power missing outermost in the second term? Why does this line imply $\alpha = 1$?

Line 171: Why is Assumption 1 reasonable? Is this equality always possible? If so, can that be shown as a lemma?

Line 188: if $\mathcal{B}_m \subseteq \mathcal{S}$, then isn't $\mathcal{B}_m \cap \mathcal{S} = \mathcal{B}$?

Line 246: Assumption 4 has not yet been introduced - can you move the definition earlier?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the identification problem in over-complete dictionary learning by introducing a new formulation. The authors primarily build on the analysis from [Huang & Hu, 2023], extending the concept of "sufficiently scattered" to the over-complete setting. By combining this extension with scaling and independence conditions for $A$ and $S$, the authors argue that "sufficiently scattered" serves as a sufficient condition for the identifiability of $A$ under the proposed formulation (2). Additionally, they provide a theoretical guarantee that this "sufficiently scattered" condition holds with high probability under the commonly used Bernoulli-Gaussian distribution.

### Strengths
The idea is well-motivated, and the problem is relevant to the community. While previous work typically relies on column incoherence for $A$, the authors propose a novel sufficient condition of $S$ for the global identifiability of the over-complete dictionary learning problem under their formulation. This is achieved by extending the "sufficiently scattered" condition from non-negative matrix factorization (NMF) to the context of dictionary learning.

### Weaknesses
1) The connection between the proposed "sufficiently scattered" condition and the conditions outlined in [3] remains unclear. Could the authors clarify this relationship?

2) The paper appears to be incomplete. For instance, the figure for the experimental section is missing, and in line 187, it seems that $\mathcal{S} \subseteq \mathbb{R}^k$ should be used.

### Questions
Given that the "sufficiently scattered" condition has been previously introduced in NMF and topic modeling, and that similar identifiability conditions appear in [1,2], could the authors discuss the specific technical challenges posed by applying this condition in the dictionary learning (DL) setting compared to the NMF/topic modeling context? 

[1] Kejun Huang, Nicholas D Sidiropoulos, and Ananthram Swami. Non-negative matrix factorizationrevisited: Uniqueness and algorithm for symmetric decomposition. IEEE Transactions on Signal Processing, 62(1):211–224, 2013.

[2] Kejun Huang, Xiao Fu, and Nikolaos D Sidiropoulos. Anchor-free correlated topic modeling: Identifiability and algorithm. Advances in Neural Information Processing Systems, 29, 2016.

[3] P. Georgiev, F. Theis and A. Cichocki, "Sparse component analysis and blind source separation of underdetermined mixtures," in IEEE Transactions on Neural Networks, vol. 16, no. 4, pp. 992-996, July 2005

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
3

### Summary
The paper introduces a new formulation for the overcomplete dictionary learning problem. The authors show global identifiability of the dictionary and sources up to permutation and scaling provided that the atoms are sufficiently sparse.

### Strengths
The paper seems mathematically sound (be careful with the dimensions, see the detailed comments below). Its positioning with respect to the existing literature should be better documented though. Two results appear as particularly related: Hu and Huang 2023 and Agarwal et al/ Rambhatla et al. It would help to have a clear discussion on the improvement of the paper compared to those results.

### Weaknesses
Sparse coding or sparse dictionary learning are not new

### Questions
Detailed Comments:

- Maybe recall what complete and overcomplete (no orthogonality) dictionary mean
- Formulation (2) should be better introduced. Why is 
- line 106, you say that A should be a dictionary that guarantees exact recovery of all s-sparse vectors. Do you mean that min ||x||_1 s.t y= Ax should have a unique solution for all s-sparse vectors?
- line 107, what is the cellular hull?
- you should clarify the notion of scattered cellular hull before introducing your results.
- Statement of Lemma 1 is misleading. First of all, from what I understand the weights d_{*c} reach the maximum of \sum_c d_c ||e_cS_*|| under the constraint \|d\|\leq m. Secondly, if the max is attained for (3), why not just optimize the l1 norm squared?
- Is it always possible to scale the columns of A_{\#} and rows of S_{\#} to satisfy (5). This is not obvious to me
- If I understand well you want the set S to be reduced to canonical vectors p? and S could include vectors that are not in the span of Q but all vectors in span(Q) must be of the form q/||q||?
- From your definition of B_m, the set is a subset of R^k (i.e. it is given by some linear combinations of the columns of Q). Moreover S is also a subset of R^k so how can the intersection of those subsets be a subset of R^m (i.e given by rows of Q)? Maybe you mean the columns of Q?
- line 158-159, I would add just one sentence, to explain that for the correlation to be maximum, you need the cosine of the angle between the vectors to be maximum which implies d_c = \alpha \|e_c^T S_*\| for all c
- lines 164-165, there are alphas missing. 
- line 178 and Figure 1. If I understand well, the set B_m is an intersection of spheres of dimension m. If my understanding is correct, I think it would be worth mentioning it somewhere because it looks as if the points clouds in Fig 1 have non empty inerior (especially the 2-strongly scattered one) while my guess is there are empty. 
- lines 241-244, in your proof sketch, again if I understand well, you define your matrix Q from the left factor of the SVD of A_#. I.e. if you have A_# = U\Sigma V^T, then you define Q as V. Then why not say it like that. I feel this is simpler and much more clear
- On line 248, you refer to assumption 4 which does not appear anywhere (the hyperlink does not work)
- line 251-253, shouldn’t the pseudo inverse be applied on the right of S_*, i.e. from line 252, the dimensions of W seems to be n\times n to me. Moreover, what you need to project to have the decomposition of line 251 are the rows of S_* not the columns. 
- One lines 268-269, if I’m not wrong you mulyiply both sides by S_# and not S_*
- On line 272, there is a transpose missing on the second A_#
- On line 272, the last equality in Equation (8) is not completely clear to me. Isn’t ||e_c^T S_*||_1 = ||w_c^T S_#||_1 and not ||e_c^T S_#||_1 ? why is ||w_c^T S_#||_1 = ||e_c^T S_#||_1 ? Does the relation follow from (5) and the fact that A_# = A_*D\Pi ? It would help to have even a short additional explanation here.
- lines 303-305, I don’t understand the sentence. You say that the sparsity is implicitely implied in (5)? How come ?
- lines 302 - 303 should be rephrased. I think what you mean is that “sparsity is required to have the strongly scattered condition used in the statement of Theorem 1” instead of “sparsity is implied in Assumption 1”
- line 308 “does not necessarily mean that the sparse coefficients S_# is identifiable” —> “are identifiable” ?
- lines 313 -320, Assumption 4 seems quite strong (or quite vague) on the dictionary. Is it easy to find such dictionaries? (I.e. you don’t provide any numerical illustration). It would be perhaps good to have a short comment such as the one at the beginning of section 2.3
- lines 339-340, “the most crucial condition is assumption 3 that cell ..” —> “the most crucial condition is assumption 3, or the fact that cell(S_#) should be generated …”?
- lines 341-342: “and show that when it satisfied assumption 3” —> do you mean “and show that it satisfies assumption 3”?
- Section 2.3., lines 337-346, I don’t really understand why, if you can make it work in the sparse Gaussian model, you can’t make it work in the Bernoulli Gaussian model. If the probability in the Bernoulli distribution is set to s/n, can’t you get a result similar to what you have with sufficient probability? Even if you can’t be at least s-sparse, isn’t “at least s-sparse” with sufficient probability enough?
- line 348 - 349 “if for every column of S” —> “if every column of S”
- lines 362-363 : “is equal to” or “equals” but not “equals to” 
- line 380, I would remove the line “which is a good sign that the bound is tight ”
- line 383 “even if identifiability of S_# is not required”, what do you mean “is not required”? Aren’t all your result focusing on the identifiability of S_# ? i would remove the paragraph starting from “On the other hand” because it makes everything unclear.
- line 389 - 390, the sentence “Due to the novel formulation (2) for overcomplete …” does not make sense either. Do you mean “We will now design an algorithm for formulation (2) for which uniqueness (up to permutation and scaling) of the dictionary and sources was shown above”
- line 427 “which is not preferable as one step of an iterative algorithm ” just remove.

### Soundness
3

### Presentation
3

### Contribution
3
