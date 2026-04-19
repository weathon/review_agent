# Robust Barycenter Estimation using Semi-Unbalanced Neural Optimal Transport

- Decision: Accept (Poster)
- Scores: 8, 5, 6, 6

## Abstract
Aggregating data from multiple sources can be formalized as an *Optimal Transport* (OT) barycenter problem, which seeks to compute the average of probability distributions with respect to OT discrepancies. However, in real-world scenarios, the presence of outliers and noise in the data measures can significantly hinder the performance of traditional statistical methods for estimating OT barycenters. To address this issue, we propose a novel scalable approach for estimating the *robust* continuous barycenter, leveraging the dual formulation of the *(semi-)unbalanced* OT problem. To the best of our knowledge, this paper is the first attempt to develop an algorithm for robust barycenters under the continuous distribution setup. Our method is framed as a $\min$-$\max$ optimization problem and is adaptable to *general* cost functions. We rigorously establish the theoretical underpinnings of the proposed method and demonstrate its robustness to outliers and class imbalance through a number of illustrative experiments. Our source code is publicly available at https://github.com/milenagazdieva/U-NOTBarycenters.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
In this work, authors propose to solve the continuous unbalanced optimal transport barycenter. To do that, they derive a dual formulation to the problem, which is a min-max problem over potentials and conditional distributions. They solve the problem by parametrizing the potential and conditional distributions with neural networks. Finally, they show that their method works on three experiments, demonstrating that they recover the right barycenters, that the barycenter is robust to outliers and to class imbalance, and that it can handle using different costs.

### Strengths
The paper is overall well written and nice to read.

It provides one of the first method to solve continuous UOT barycenters leveraging a dual formulation and neural networks parametrizations, which is an interesting contribution.

Several convincing experiments are done demonstrating that the method learns well the UOT barycenter, that it is robust to outliers and class imbalance, and that it works with several costs.

### Weaknesses
The main weakness in my opinion is that the method feels really incremental compared to [1]. The main difference is that it it adapted to the UOT problem, which just changes slightly the formulation of the dual, and how to sample from the barycenter for inference.

Another weakness, which is classical with UOT, is that the choice of the unbalancedness parameters does not seem easy.

The method also needs to solve a min-max problem, which is probably unstable and very costly.

### Questions
A term seems to be missing in the definition of $\psi$-divergence, see Definition 1 in [2].

The sentence line 227 is not clear to me. The $m$-congruence is a constraint of the problem. So I do not understand why it is written "if the potentials satisfy the m-congruence, then the optimal value of the SUOT barycenter can be derived by solving (8)". The SUOT barycenter can always be derived by solving (8) and the potentials necessarily satisfy this constraint when solving (8).

In Corollary 1, it is stated that the sup is taken over $f_{[1,K]}$. Isn't it also taken over $m$?

In Theorem 2, equation (12), shouldn't it be an argmin?

In Section 5.1, it is stated that the UOT problem is equivalent with the OT problem between rescaled distribution. Is it truly equivalent? And could we solve the OT barycenter between the rescaled distributions instead of solving the SUOT barycenter?

Is it proved that $T_{1\\#}\mathbb{P}_1$ gives the UOT barycenter?

I think that the reference [3] is missing. In their experiments, they compute robust OT barycenters via unbalancedness.

Typos:
- Line 11: The first sentence of the abstract feels weird: I don't see what is the "common challenge"
- Line 107: "This transition is valid since we work the infimum in weak"
- Line 142: "wights"
- Line 365: what is $T$ in $T_1 = \lambda_1 Id + \lambda_2 T$
- Legend of Figure 4: $\mathbb{P}_0$ -> $\mathbb{P}_1$ and $\mathbb{P}_1$ -> $\mathbb{P}_2$

[1] Kolesov, A., Mokrov, P., Udovichenko, I., Gazdieva, M., Pammer, G., Burnaev, E., & Korotin, A. (2024). Estimating Barycenters of Distributions with Neural Optimal Transport. arXiv preprint arXiv:2402.03828.

[2] Séjourné, T., Peyré, G., & Vialard, F. X. (2023). Unbalanced optimal transport, from theory to numerics. Handbook of Numerical Analysis, 24, 407-471.

[3] Séjourné, T., Bonet, C., Fatras, K., Nadjahi, K., & Courty, N. (2023). Unbalanced optimal transport meets sliced-Wasserstein. arXiv preprint arXiv:2306.07176.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
In this paper, the authors investigate the problem of finding the barycenter with semi-unbalance optimal cost (SUOT) from perspective of dual theory. In particular:

(1) They derive the theory for the dual form of SUOT barycenter with reformulation, and necessary condition based on marginal plan and conditional plan

(2) Then, from the necessary condition,  they propose the an algorithm (Algorithm 1) to approximate the optimal transport plan from the solution. This algorithm is based on training the neural network of optimal plan.  

(3) Then, they empirically validate the efficacy of the algorithm on both synthetic and real dataset.

### Strengths
1. The paper provides an important and novel dual theory for the SUOT barycenter problem.

2. The proposed algorthm works well and persistent to the imbalance and outliers. 

3. The paper is well-structured and easy to follow.

### Weaknesses
1. The necessary condition seems to be not difficult to obtain, which makes it not convincing enough. So, would it be possible to obtain some sufficient conditions? 

2. It would be better if the authors can highlight the advantages of finding the optimal plan over the optimal solution. 

3. As stated in the limitations, the authors did not provide a rigorous analysis for the convergence of the algorithm.

### Questions
See Weaknesses section.

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
This paper proposes a scalable approach for estimating the robust continuous barycenter by using the dual formulation of the (semi-)unbalanced OT problem. They first attempt to develop an algorithm for robust barycenters under continuous distribution setup. They model this problem as a min-max optimization problem and provide theoretical underpinnings, along with experimental results on synthetic and real-world datasets to demonstrate the robustness and adaptability of their method.

### Strengths
- This paper introduces a continuous SUOT barycenter estimation method that addresses the issue of robustness against outliers and imbalances in real-world datasets.
- Rigorous derivation of the SUOT-based framework and solid theoretical support for the proposed model give the approach a solid foundation.

### Weaknesses
Experiments are not sufficient and the baseline method is not comprehensive.

### Questions
- Given that the Wasserstein barycenter of Gaussian distributions has a closed-form solution, could the authors provide experimental verification for the Gaussian distribution case?
- Can the robustness parameter $\tau$ be automatically optimized during training rather than manually tuned? If not, for different proportions or types of noise, different $tau$ is usually required. Will this limit the practicality of the scheme?
- How does the method work with high-dimensional data? In other words, the support size of the measures may be large in practical applications. While the support size of the measures in your experiments seems rather small.
- Why don't you compare the methods [1,2] based on robust OT? Please add corresponding experiments.

[1] Nietert S, Goldfeld Z, Cummings R. Outlier-robust optimal transport: Duality, structure, and statistical analysis[C]//International Conference on Artificial Intelligence and Statistics. PMLR, 2022: 11691-11719.

[2] Wang X, Huang J, Yang Q, et al. On Robust Wasserstein Barycenter: The Model and Algorithm[C]//Proceedings of the 2024 SIAM International Conference on Data Mining (SDM). Society for Industrial and Applied Mathematics, 2024: 235-243.

### Soundness
3

### Presentation
3

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
This paper proposes a neural network-based method to estimate continuous barycenter via the dual formulation of the semi-unbalanced OT.

### Strengths
1. The proposed method may be the first continuous robust barycenter estimation approach with proper theoretical support and practical validation.

2. The proposed method is robust and can estimate the barycenter based on the data containing the outliers.

### Weaknesses
My main concern is the novelty of the paper. The primary method seems to be a combination of barycenter calculation, neural optimal transport, and semi-unbalanced optimal transport. In my view, simply combining methods may not be sufficient for publication at ICLR, so I give a negative score. I will consider raising my score if the authors can demonstrate unique contributions that go beyond a straightforward combination of these methods.

### Questions
Q1. How does the time efficiency of U-NOTB compare with other baselines?

Q2. The experiments seem to lack details about the size of the training/testing sets and some training specifics, such as learning rate and other hyperparameters.

Q3. Since the overall training is similar to GANs with a max-min approach, is the training of U-NOTB stable?

### Soundness
2

### Presentation
2

### Contribution
1
