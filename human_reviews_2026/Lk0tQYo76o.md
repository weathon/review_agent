# Improved Analysis for Sign-based Methods with Momentum Updates

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
This paper presents enhanced analysis for sign-based optimization algorithms with momentum updates. Traditional sign-based methods obtain a convergence rate of $\mathcal{O}(T^{-1/4})$ under the separable smoothness assumption, but they typically require large batch sizes or assume unimodal symmetric stochastic noise. To address these limitations, we demonstrate that signSGD with momentum can achieve the same convergence rate using constant batch sizes without additional assumptions. We also establish a convergence rate under the $l_2$-smoothness condition, improving upon the result of the prior momentum-based signSGD variant by a factor of $\mathcal{O}(d^{1/2})$, where $d$ is the problem dimension. Furthermore, we explore sign-based methods with majority vote in distributed settings and show that the proposed momentum-based method yields convergence rates of $\mathcal{O}\left( d^{1/2}T^{-1/2} + dn^{-1/2} \right)$ and $\mathcal{O}\left( \max \\{ d^{1/4}T^{-1/4}, d^{1/10}T^{-1/5} \\} \right)$, which outperform the previous results of $\mathcal{O}\left( dT^{-1/4} + dn^{-1/2} \right)$ and $\mathcal{O}\left( d^{3/8}T^{-1/8} \right)$, respectively. Numerical experiments also validate the effectiveness of the proposed methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents a new analysis for sign-momentum. First, it presents a convergence rate of $O(\frac{\sqrt{\|\mathbf{L}\|_1+\Delta}+\|\mathbf{\sigma}\|_1}{T^{1/4}})$ for the centralized case under a separable smoothness assumption. Next, for the standard smoothness assumption, it presents a convergence rate of  $O(\frac{d^{1/2}(L+\Delta+\sigma)}{T^{1/4}})$. In the distributed setup, it achieves better results than (Sun et al., 2023; Jin et al., 2021). The paper provides experimental results on CIFAR-10 for the proposed approaches against existing baselines.

### Strengths
* The theoretical results appear to improve upon existing work, although I am less familiar with this literature.
* The results on CIFAR-10 look good.

### Weaknesses
* The $S_G$ procedure depends on unknown parameters. It is also unclear how the authors implemented $S_G$ and which version was used in the distributed setup. Could the authors provide more details? It would also be helpful if they compared results across the proposed versions (v1 and v2) to support the theoretical claims better.
* Regarding Theorem 5, the idea of applying an unbiased estimation twice is interesting. However, in this case, shouldn’t we also observe a reduction in stochastic variance with the number of workers and, consequently, accelerated convergence, as is typically seen in standard distributed learning?
* The authors did not provide any supplementary material. Since no code for the empirical evaluation is available, it is difficult for the community to verify and assess the contribution of the proposed approach.

### Questions
See above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper shows that signSGD with momentum can achieve a convergence rate of $\mathcal{O}(T^{-1/4})$ using constant batch sizes without additional assumptions such as large batch sizes or unimodal symmetric stochastic noise. Experimental results are shown on CIFAR to demonstrate convergence rates.

### Strengths
Original tighter error analysis: 
Bounds $\sum_i |[\nabla f(x_t)]_i| \cdot \mathbb{P}(\text{sign mismatch})$ directly by $\|\nabla f(x_t) - v_t\|_1$ instead of probability inequalities requiring $\mathcal{O}(\sqrt{T})$ batches or symmetric noise assumptions

Removes assumptions: 
No $\mathcal{O}(\sqrt{T})$ batches or symmetric noise for $\mathcal{O}(T^{-1/4})$ rate that are required for prior analysis

Experimental validation: 
Experiments with ResNet on CIFAR dataset show faster gradient norm decay that matches theoretical results

### Weaknesses
Incremental improvements:
The improvements are incremental rather than paradigm shifting. The improved bound is useful but may not have a large practical impact

Weak experimental results:
The experimental results are with ResNet on CIFAR datasets. These do not reflect modern uses cases of Sign-based methods. Experimental results on large models would make the paper stronger.

### Questions
How would the method perform on a larger model and more realistic datasets?

Assumption 8 $\sup_x \|\nabla f_j(x;\xi)\|_\infty \leq G$ is strong for neural networks, even though the authors cite previous works that use similar assumptions. Do your experiments satisfy this? What are observed \max_{t,j} \|\nabla f_j(x_t;\xi)\|_\infty values?

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
2

### Summary
The paper revisits sign-based stochastic optimization with momentum (a.k.a. Signum) and provides tighter convergence analyses. First, under separable smoothness and separable bounded noise, the authors show that signSGD with momentum achieves the classical non-convex rate $O(T^{-1/4})$ without the large-batch or unimodal-symmetric-noise assumptions used in earlier work. Under the standard $\ell_2$-smoothness and bounded-variance assumptions, they further prove an $O(d^{1/2}T^{-1/4})$ rate, improving the prior $O(dT^{-1/4})$ dependence for momentum-based sign methods. For distributed majority-vote settings, they propose an unbiased server-side sign operator and derive improved rates, e.g., $O(d^{1/2}T^{-1/2}+dn^{-1/2})$ and $O(\max(d^{1/4}T^{-1/4}, d^{1/10}T^{-1/5}))$, that outperform prior results in both $d$ and $T$. Empirically, CIFAR-10 (centralized) and CIFAR-100 (distributed, 8 nodes) experiments with 10 seeds support the claims.

### Strengths
1. The paper delivers a theoretical tightening for sign-based momentum methods in both centralized and distributed settings. On the centralized side, it attains the classical non-convex rate $O(T^{-1/4})$ for Signum under separable smoothness and bounded noise without resorting to large-batch or restrictive noise-shape assumptions, and under standard $\ell_2$-smoothness it improves the dimension dependence from $d$ to $d^{1/2}$. On the distributed side, introducing an unbiased server-side sign operator leads to sharper dependencies on $d,T$, and the number of workers $n$, strengthening the case for majority-vote style aggregation. 

2. The technical pivot, controlling sign error through a cleaner estimator-error recurrence, reads as careful and broadly applicable, and the presentation is clear, with assumptions and algorithms spelled out and tables that position the results against prior analyses. 

3. Empirically, though modest in scale, the CIFAR-10/100 studies with multiple seeds are consistent with the theoretical story and demonstrate that the proposed analyses map to competitive performance in practice.

### Weaknesses
1. The empirical scope is narrow: results focus on CIFAR-10 (centralized) and CIFAR-100 with eight nodes (distributed), leaving open how the methods behave in larger-scale, highly heterogeneous, or bandwidth-constrained regimes. 

2. The theory relies on specific schedules for step size and momentum, yet the experiments use grid-tuned constants without ablations that test sensitivity to the prescribed schedules, which blurs the link between bounds and practice. 

3. The comparative breadth could be stronger: beyond SGDM/AdamW and selected sign baselines, results omit contenders such as error-feedback compressors or recent variance-reduced sign methods that would sharpen the empirical positioning.

### Questions
See Weaknesses. In particular:
1. Can you extend the evaluation beyond CIFAR-10 (centralized) and CIFAR-100 with eight nodes (distributed), e.g., to larger-scale, in order to assess how the methods behave in those settings?

2. Can you add ablation testing sensitivity to the theorem-prescribed step-size and momentum schedules versus the grid-tuned constants, to clarify the link between the bounds and practice?

3. Could you include additional baselines, such as error-feedback compressors or recent variance-reduced sign methods?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper establishes theoretical convergence guarantees for a class of stochastic optimization algorithms that employ gradient-sign updates (signSGD) augmented with momentum. 

The analysis primarily focuses on the **Signum** algorithm in centralized settings and its distributed variants based on majority voting. 

The main motivation is to address limitations of prior analyses, which required either large batch sizes or restrictive noise assumptions to attain the optimal convergence rate of (O(T^{-1/4})). 

Empirical results on image classification tasks (CIFAR-10/100) show fast convergence compared to several established baselines.

### Strengths
1. The paper shows that their algorithm reduces the large-batch requirements and improves the dimension dependency in the theory of sign-based optimization.

2. The analysis is conducted under multiple standard assumptions, making the results robust and widely applicable, and the distributed analysis also accounts for heterogeneous data settings.

3. The experimental results demonstrate superior performance in both centralized and distributed environments.

### Weaknesses
1. It remains unclear how the proposed algorithm compares with Ref. [1]. The method in Ref. [1] uses a fixed mini-batch size and imposes no noise assumptions, yet achieves an O(T^{-1/3}) complexity, whereas the present work reports only O(T^{-1/4}). Although the authors note that Ref. [1] assumes component-wise smoothness while this paper assumes global smoothness, both assumptions appear mild.

2. While the theory significantly improves the dependence on the dimension $d$, the experiments do not explicitly validate this. Performance gains are demonstrated on fixed-dimension tasks (e.g., CIFAR), but an ablation study varying $d$ would have provided stronger empirical support for the theoretical claims.

3. The related work section mentions error-feedback (Karimireddy et al., 2019) but does not discuss in depth how the proposed methods compare to these techniques in theory or practice, particularly regarding robustness and performance when the sign operation introduces significant bias.

Ref [1]. Wei Jiang, Sifan Yang, Wenhao Yang, and Lijun Zhang, Efficient Sign-Based Optimization: Accelerating Convergence via Variance Reduction, NeurIPS 37, pp. 33891–33932, 2024.

### Questions
We already have the Signum algorithm—why is there a need to introduce MVSM? Could you better illustrate the motivation behind MVSM?

### Soundness
2

### Presentation
2

### Contribution
1
