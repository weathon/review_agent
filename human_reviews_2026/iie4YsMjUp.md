# Certified Robustness Training: Closed-Form Certificates via CROWN

- Avg Score: 2.50
- Decision: Reject
- Scores: 0, 2, 6, 2

## Abstract
Adversarial training reshapes neural network decision boundaries by pushing them away from adversarial examples, but this approach ignores a crucial geometric factor: the local curvature that determines how steeply network outputs change with input perturbations. We introduce a fundamentally different approach that optimizes certified robustness by directly reshaping decision boundary geometry during training. Our key insight is that CROWN's linear bounds encode both the safety margin and input sensitivity needed for closed-form certified radius computation, transforming expensive verification into efficient geometric analysis. We derive differentiable expressions that enable direct optimization of the margin-over-slope ratio underlying certified robustness, creating networks with inherently robust decision regions rather than boundaries hardened against specific attacks. Our hybrid training method combines adversarial training's broad coverage with geometric certified objectives applied to hard examples, achieving 98.33% clean accuracy and 71.1% certified robustness at epsilon = 0.03 on MNIST — outperforming both PGD adversarial training (61.7%) and randomized smoothing (53.1%) in ReLU-based networks. On DC optimal power flow regression, we demonstrate controllable accuracy–safety trade-offs critical for engineering applications. By making certified robustness certificates both computationally tractable and differentiable, our approach enables robustness-aware learning that produces networks robust by geometric design rather than adversarial accident.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper utilizes the CROWN framework to acquire estimates of classification margins for ReLU neural networks. Subsequently, by penalizing the training loss, they seek to increase the margins of classification, thereby increasing the robustness of the classifiers and defending against adversarial attacks. Experimental results are provided on the MNIST dataset and a power system control.

### Strengths
Unfortunately, nothing specific caught my attention to be strength-worthy.

### Weaknesses
This paper does not rise to the level of an ML conference, yet ICLR. It is more fitting for a course project. 

The novelty in this paper is indeed very limited and the authors do not properly establish baselines or compare with them. There are works that are indeed close to this that are not properly cited or compared with. 

The paper is repetitive and this raises concerns regarding the potential significant use of LLMs in writing the paper.

The authors make some geometrical claims over and over (in line with the repetitiveness) without really providing any theoretical, empirical, or even geometrical justifications.

The methodology of the paper is in essence similar to [1] and [2], neither of which have been cited or compared with. More specifically, [1] has performed the same line of thought and reasoning, but instead of using CROWN, they utilize the Lipschitz constant to provide bounds. Indeed, CROWN can potentially provide tighter bounds, but it is important to give credit to the original work. I am aware that [1] studies $ l_2 $ bounds instead of the $ l_\infty $ done in here, but nonetheless, the core of the methods is the same.

There are many baselines that are missing in the experiment section. Comparison has to be made against many adversarial training methods like [2], TRADES, etc.

The experiments are very limited and very small-scale, not meeting the standards of ICLR. There is only an experiment on the MNIST dataset, which is a very simple task and pretty saturated, and then on a power system setup, which is even smaller. These are not acceptable. 
Moreover, the power system experiment lacks any comparison with other work.


Some other comments:
- I understand the appeal of the differentiability of the CROWN bounds. However, the authors are, in a way, selling this too much, given that they are using ReLU networks, which are technically not differentiable!
- 33 (and many other places): The authors claim that adversarial training just moves the decision boundaries without changing geometry. What is the proof for this? How can you claim that this is indeed what is happening.
- fig 1: the boundaries are all the same in the left and right. what has changed between adversarial training vs your training? No margin has changed.
- 81-82: 'this transforms expensive post-hoc verification into an ...' The expensive part of the verification is due to the branching, which you are not doing, right? If so, you haven't really done anything and haven't added any contribution and are just doing bare CROWN. So, why are you saying that this has transformed ...?
- 144: Can you use any scalar function s(x)?? Doesn't CROWN only work with linear functions of the output of the network? Can you for example, handle the log-sum-exp on the output of the network?

CROWN provides linear upper and lower bounds. How is this showing the local geometric structure that you keep claiming it does? Between these two hyperplanes that CROWN finds, the geometric structure might as well be any complex non-linear non-convex function.




[1] Fazlyab M, Entesari T, Roy A, Chellappa R. Certified robustness via dynamic margin maximization and improved lipschitz regularization. Advances in Neural Information Processing Systems. 2023 Dec 15;36:34451-64.

[2] Xu Y, Sun Y, Goldblum M, Goldstein T, Huang F. Exploring and exploiting decision boundary dynamics for adversarial robustness. arXiv preprint arXiv:2302.03015. 2023 Feb 6.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper presents a novel certified training scheme aiming to maximise a lower bound to the certified robustness radius.
The authors propose to compute this lower bound through alpha-CROWN, a well-known method to compute local linear approximations of neural networks.
Experimental results on MNIST classification and DC optimal power flow regression are presented.

### Strengths
The idea to do certified training through direct regularization of a bound to the certified radius is simple yet novel to the best of my knowledge.

### Weaknesses
I am confused about the practical technicality of the lower bound. The bound itself will rely on intermediate pre-activation bounds, which, according to the appendix, are computed using a given l-infinity ball (at most $\epsilon=0.03$, for MNIST). Given that the CROWN linear network approximation relies on these intermediate bounds, I am not sure that the associated lower bound to the certified radius can say much beyond the largest radius employed for the intermediate bounds. In this case, if the lower bound turned out to be greater than $\epsilon=0.03$, I am not sure that would be valid. This is a very important point, as one may as well directly employ CROWN bounds to the network output for $\epsilon=0.03$ perturbations, as commonly done in the certified training literature.

Related to the above, the authors fail to compare against any baseline from the deterministic certified training literature. At the very least, the authors should compare to a certified training loss (an upper bound to the adversarial cross entropy) computed using the same CROWN bounds used to compute the radius lower bound. I think it would also be extremely important to compare against more recent certified training approaches, such as [1, 2]. 
It is also crucial to compare against Lipschitz-based methods, such as [3], given that the geometric intuitions behind the work closely mirror the idea of lowering the Lipschitz constant of a network.
It is impossible to otherwise appropriately judge the merits of the proposed approach.

Furthermore, I am not sure the comparisons to PGD and RS are fair: for the first, the radius employed is different than the one for the authors' method. For the second, it operates on l2 perturbations, whereas the authors focus on l-infinity benchmarks.

Finally, the MNIST comparison with the baselines is carried out over an extremely small network, and using very small certified radii. For instance, [1] gets 99.23/98.22 standard/certified accuracy at $\epsilon=0.1$. Experiments on networks close to the state-of-the-art are fundamental to appropriately assess the work.

[1] Certified Training: Small Boxes are All You Need, Mueller et al., ICLR23

[2] Expressive Losses for Verified Robustness via Convex Combinations, De Palma et al., ICLR24

[3] Rethinking Lipschitz Neural Networks and Certified Robustness: A Boolean Function Perspective, Zhang et al., NeurIPS22

### Questions
- How loose is the bound from theorem 3.1 compared to bisection search? Why not using it directly at this point?
- When you compute the radius bound, what is the input domain you use for the verifier? Do you use the entire input space? 
- Could you clarify the point in the weaknesses concerning the interaction between the intermediate bounds and the validity of the radius lower bound?
- The certified training literature commonly relies on IBP bounds owing to their superior scalability and differentiability, which allow them to use significantly larger networks than those employed in the paper. Could the authors provide experiments on larger CNNs?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work advances certified robustness by embedding geometric awareness into the training objective, leveraging the affine relaxations of CROWN/β-CROWN to derive closed-form, differentiable certification radii optimized end to end. The design centers on a margin-over-slope objective that jointly enlarges safety margins and suppresses input sensitivity, yielding decision regions that are robust by construction rather than through after-the-fact defenses. Methodologically, it builds on a closed-form ℓ∞ radius (generalizable to ℓp) via half-space distances, local exactness under activation stability, and sensitivity-control guarantees. Optimization employs a smooth multi-constraint aggregator to stabilize gradients, along with optional β-CROWN joint-α tightening to achieve tighter relaxations with moderate overhead. Once coefficients are computed, radius evaluation runs in O(d). In practice, a hybrid training regimen applies adversarial training for broad coverage and selectively adds certification penalties to hard examples, converting verifier feedback into stable, differentiable training signals. Experimental results show that the proposed method outperforms strong adversarial and smoothing baselines on a standard classification benchmark and reveals actionable accuracy–safety Pareto trade-offs in a power-system regression task, demonstrating that embedding tractable, differentiable certification signals into training yields scalable and verifiably robust models.

### Strengths
1. The work leverages the affine relaxations of CROWN/β-CROWN to derive closed-form, differentiable certification radii and optimizes a margin-over-slope objective end to end. This joint objective simultaneously enlarges quantifiable safety margins and suppresses input sensitivity, transforming certification from a post-hoc verification step into a geometry-aware training signal.
2. Using half-space distances as the analytical bridge, the approach grounds robustness in a closed-form ℓ∞ radius with theoretical extensibility to ℓp norms, establishes local exactness under activation stability, and provides sensitivity-control guarantees. Training is stabilized through smooth multi-constraint aggregation, while β-CROWN joint-α tightening enhances relaxation tightness with moderate computational overhead. 
3. The method integrates verifier feedback into a hybrid regimen that combines broad adversarial training with selective certification penalties on hard examples. It achieves improved performance relative to adversarial and smoothing baselines on a standard classification benchmark and reveals actionable accuracy–safety Pareto trade-offs in a power-system regression task, demonstrating a practical route toward constructing scalable, verifiably robust models without reliance on after-the-fact defenses.

### Weaknesses
1. The classification evaluation centers on a standard classification benchmark, and the engineering assessment presents a single DC‑OPF case study. Accordingly, claims about scalability and robustness would be stronger with evidence on larger‑scale, multi‑class, or non‑vision settings and with modern architectures (e.g., deeper/wider networks with attention or normalization), as well as checks for cross‑norm consistency.

2. The approach relies on CROWN/β‑CROWN bounds being sufficiently tight and on locally stable activation patterns. In architectures with non‑piecewise‑linear components or in deeper/wider models, relaxations may be looser and activation switches more frequent, potentially attenuating certified radii and making optimization more sensitive to relaxation quality. Optional β‑CROWN joint‑α tightening can mitigate this at added computational cost.

3. Although radius evaluation runs in O(d) time once coefficients are available, overall training cost is largely driven by (β‑)CROWN bound propagation and grows with input dimensionality and the number of constraints. The work would benefit from deeper ablations that disentangle the contributions of margin enlargement versus sensitivity suppression, clarify the roles of smooth aggregation and tightening.

### Questions
1. Can the method maintain both certified robustness and clean accuracy advantages on more complex datasets? Is there empirical evidence of cross-norm consistency (e.g., from ℓ∞ to other norms), as well as measurements of scalability and throughput/latency in different experimental settings.

2. To what extent does the loss of CROWN/β-CROWN tightness under frequent activation switching or in networks with non-piecewise-linear components weaken the certified radii and the stability of training signals? Could provide comparative analyses (e.g., original CROWN vs. β-CROWN tightening, and the gap against small-scale exact verification) to quantify the deviation between theoretical assumptions and practical behavior, thereby clarifying the method’s applicability boundaries?

3. What are the respective and joint contributions of margin enlargement and sensitivity suppression to the observed performance gains? How significant are the marginal effects of smooth aggregation and tightening strategies in improving stability and robustness? Additionally, how does the overall training cost scale with input dimensionality and the number of constraints?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This manuscript studies robustness verification and provable robust learning from the geometric aspect. Specifically, the authors derive a differentiable bound of the distance based on CROWN between the data point and the decision boundary. Then this bound, because it is differentiable, is incorporated into adversarial training to boost certified robustness.

### Strengths
++ There are not too many works studying certified robustness from the geometric perspective. The proposed method is sound and easy to follow.

++ The computational overhead is analysed and tractable.

### Weaknesses
1. My major concern is the novelty, the motivation and the proposed method is very similar to the paper "Training Provably Robust Models by Polyhedral Envelope Regularization" (2023).

    * The mentioned paper also use CROWN to derive the bound and calculate the distance between the data input and the approximated (linearized) decision boundary and incorporate it into adversarial training to boost certified robustness, almost identical to Equation (15) in this manuscript.

    * The mentioned paper considers a finer constraint, the perturbed image is inside the [0, 1] box and applicable to general $l_p$ norm. This manuscript does not consider the [0, 1] bounding box and focus only on $l_\infty$ norm bounds.

    * The mentioned paper has larger-scale experiments, including more architectures (multilayer CNN) and CIFAR10, while this manuscript's experiments only include smaller-scale results like MNIST.

2. The computational complexity of CROWN is significantly larger than just forward / backward propagation, because CROWN is quadratic w.r.t. depth while backward propagation is linear w.r.t depth. Therefore, I believe the computational complexity, compared with normal training, would be significant for deeper neural networks. I agree with the claim in this manuscript "the complexity is polynomial", but "polynomial" is not enough. The authors may consider using CROWN-IBP to derive the bound more efficiently, but this bound would be looser. I suggest the author add one section discussing about the trade-off between complexity and tightness of the bound.

3. As pointed out in the points above, the experiment part is weak and should include larger scale dataset and models. The authors should at least discuss the algorithm in the general $l_p$ cases.

4. The slope derived by CROWN is not always differentiable [line 79]. I agree we can have a differentiable and tight bound for ReLU functions. However, for functions like sigmoid and tanh, if we try to obtain the tight approximation, then we have to utilise some numerical method and this will make the slope non-differentiable. On the other hand, we can have a differentiable but looser bound. The authors should clarify this.

Minor issues:

1. The template used is not identical to the official one for ICLR 2026 submission. I cannot see the text "Under review as a conference paper at ICLR 2026".

In general, the authors should clearly clarify the novelty, the contribution and make the experimental part comprehensive before this work being considered for publication. I welcome the authors to address my concerns in the rebuttal and will reconsider the manuscript after rebuttal.

### Questions
Please answer the questions raised in the weakness part:

[Weakness 1] Please compare this work with "Training Provably Robust Models by Polyhedral Envelope Regularization" (2023) and **clearly point out what is the novelty and the contribution of this work**.

[Weakness 2] Please add a detailed discussion about the complexity when the model gets deeper.

[Weakness 3] Please add larger-scale experiments to validate the effectiveness of your proposed method. Please consider non-$l_\infty$ cases as well.

[Weakness 4] Please clearly point out how to pick the differential factors in linear approximation by CROWN.

### Soundness
3

### Presentation
2

### Contribution
1
