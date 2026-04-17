# Mitigating Spurious Correlation via Distributionally Robust Learning with Hierarchical Ambiguity Sets

- Decision: Accept (Poster)
- Scores: 6, 6, 8

## Abstract
Conventional supervised learning methods are often vulnerable to spurious correlations, particularly under distribution shifts in test data. To address this issue, several approaches, most notably Group DRO, have been developed. While these methods are highly robust to subpopulation or group shifts, they remain vulnerable to intra-group distributional shifts, which frequently occur in minority groups with limited samples. We propose a hierarchical extension of Group DRO that addresses both inter-group and intra-group uncertainties, providing robustness to distribution shifts at multiple levels. We also introduce new benchmark settings that simulate realistic minority group distribution shifts—an important yet previously underexplored challenge in spurious correlation research. Our method demonstrates strong robustness under these conditions—where existing robust learning methods consistently fail—while also achieving superior performance on standard benchmarks. These results highlight the importance of broadening the ambiguity set to better capture both inter-group and intra-group distributional uncertainties.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose a hierarchical uncertainty set approach that takes care of inter-group and intra-group shifts based on robust optimization. Their method is effective in improving the worst group accuracies under certain data shifts scenarios, especially those with shifts in distribution beyond just changing group proportions. They have evaluated their approach on multiple datasets and obtain small improvements against competing approaches.

### Strengths
- The proposed method is effective in improving the worst group accuracy in handling distribution shifts, compared to GroupDRO and competing methods on a variety of datasets including CMNIST, Waterbirds, and CelebA. 

- The main idea of the method is clear and well-presented: in addition to uncertainty in shifts in group proportions, add a second layer of uncertainty in the within-group samples.

### Weaknesses
- Compared to vanilla GroupDRO, the improvement is relatively small (Table 2) for the unshifted case. 

- As with other forms of robust optimization that handles uncertainty in the samples, there is a tradeoff between optimizing for average and worst-case accuracies. Although we could obtain better worst case accuracies with the authors' method, it suffers worse average accuracies compared to GroupDRO in Table 1. It could be difficult to determine this tradeoff in practice.

### Questions
- How do the authors determine the size of the within-group ambiguity set? This could be a difficult parameter to tune.

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
3

### Summary
This paper addresses a limitation of existing methods for mitigating spurious correlations, such as GroupDRO, which are robust to shifts in group proportions (inter-group shifts) but fail when the data distribution within a minority group changes between training and testing (intra-group shifts). The authors propose a hierarchical distributionally robust optimization (DRO) framework that models both levels of uncertainty through a hierarchical ambiguity set. This set forces the model to be robust not only to the worst-case mixture of groups but also to the worst-case data variations within each group, formulated using a Wasserstein distance in a latent space. To validate their approach, they introduce new challenging benchmark datasets designed to exhibit these minority-group shifts, demonstrating that their method achieves stronger robustness where state-of-the-art methods fail, while also maintaining top performance on standard benchmarks.

### Strengths
This paper presents a well-motivated extension to distributionally robust optimization (DRO) for mitigating spurious correlations. It identifies and addresses a failure mode in existing robust learning methods: vulnerability to intra-group distribution shifts. 

While prior work like Group DRO focuses on robustness to changing group proportions (inter-group shifts), this paper argues that the distribution within small minority groups can itself be non-stationary. 

The proposed hierarchical ambiguity set is a theoretically sound solution that unifies robustness at both the inter- and intra-group levels. The authors introduce new, modified datasets specifically designed to induce the targeted minority-group shifts, providing a direct test of their hypothesis. 

The motivation is made intuitive through clear diagrams (Fig. 1), the mathematical formulation is precise, and the results are analyzed thoroughly, including qualitative insights from Grad-CAM.

### Weaknesses
I've identified several gaps and issues:

The proof applies Lemma A.0.1 (from Staib & Jegelka 2017) but then immediately relaxes it:

$$E_{P_g}\left[\sup_{x: d(z(x), z(X)) \leq \epsilon_g} L(f_\theta^L(z(x)), Y)\right] \leq E_{P_g}\left[\sup_{z': d(z', z(X)) \leq \epsilon_g} L(f_\theta^L(z'), Y)\right]$$

- This inequality can be strict (not tight) when $z(\cdot)$ is not surjective onto the $\epsilon$-ball

Also proposition B.1 assumes:
- "the feature map $z(x)$ is **fixed** w.r.t. $\theta$"

But $z(x)$ is defined (equation 7) as:

$z(x) := f_\theta^{L-1}(f_\theta^{L-2}(\ldots f_\theta^1(x)))$

This is a direct contradiction - $z(x)$ clearly depends on $\theta$ through layers 1 to $L-1$. The convergence analysis is therefore invalid for the actual algorithm being run.

The cost function (page 5) sets:

$c((x,y), (x',y')) = \begin{cases} \|z(x) - z(x')\|, & \text{if } y = y' \\\\ \infty, & \text{otherwise} \end{cases}$

This means $W_p(P, Q) = \infty$ whenever marginal distributions of $Y$ differ. 

- The paper claims "this definition does not cause any issues" because $G = (Y, A)$, but this severely restricts the ambiguity set - distributions can only shift within same-label groups

The paper doesn't compare against:
- Standard Wasserstein DRO methods (Kuhn et al. 2019, cited but not compared)

Also as a notation note, $P$ vs $\hat{P}$ usage is sometimes unclear.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposed a robust optimization scheme with hierarchical distribution shifts. It optimizes the worst-case distribution from a mixture of shifted group distributions Eq.(5). As a concrete instance, the mixture coefficients can be arbitrary and the group drift is constrained within a Wasserstein ball as in Eq.(8). A practical algorithm is introduced with convergence guarantee. Experiments on common robust optimization benchmark datasets show that the algorithm can outperform several baselines in the robust optimization literature.

### Strengths
1. Writing and presentation are very clear and easy to follow

2. The algorithm is theoretically motivated, and a practical algorithm is introduced with a convergence guarantee. 

3. The algorithm performs better than several baselines in terms of worst-case accuracy.

### Weaknesses
1. A hierarchical framework is introduced but with only one instantiation. It would be better to showcase the flexibility of the framework by using the $\rho$ in (5). More importantly, it is unclear why such a $\rho$ constraint is meaningful in real-world scenarios.

2. Similarly, only the Wasserstein distance is considered as the measure of distribution shift. It would be worthwhile to discuss other options like f-divergence, and whether they will be feasible in practice.

### Questions
What other options would be possible and practical for the general framework specified in (5)?

### Soundness
3

### Presentation
4

### Contribution
3
