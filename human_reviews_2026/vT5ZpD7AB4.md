# A Robust Certified Machine Unlearning Method Under Distribution Shift

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 6, 2

## Abstract
The Newton method has been widely adopted to achieve certified unlearning. A critical assumption in existing approaches is that the data requested for unlearning are selected i.i.d. (independent and identically distributed). However, the problem of certified unlearning under non-i.i.d. deletions remains largely unexplored. In practice, unlearning requests are inherently biased, leading to non-i.i.d. deletions and causing distribution shifts between the original and retained datasets. In this paper, we show that certified unlearning with the Newton method becomes inefficient and ineffective under non-i.i.d. unlearning sets. We then propose a better certified unlearning approach by performing a distribution-aware certified unlearning framework based on iterative Newton updates constrained by a trust region. Our method provides a closer approximation to the retrained model and yields a tighter pre-run bound on the gradient residual, thereby ensuring efficient $(\epsilon,\delta)$-certified unlearning. To demonstrate its practical effectiveness under distribution shift, we also conduct extensive experiments across multiple evaluation metrics, providing a comprehensive assessment of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper explores exact certified unlearning for non-linear neural networks under the non-IID retain data assumption. The authors point out that, in this setting, a single-step Newton update with added noise is inefficient. To address this, they propose performing multiple Newton updates within a trust region (TR) framework, demonstrating that this approach yields an unlearned model that more closely approximates the retrained model.

### Strengths
1. The paper is well written and easy to follow.

2. The problem statement is novel — the authors address the non-IID data deletion scenario, which is often overlooked in existing certified unlearning literature.

3. Extensive theoretical rigors are demostrated, providing proofs and analyses to support the proposed method.

### Weaknesses
1. Although the paper provides extensive theoretical analysis, the experimental evaluation is limited. Only a single baseline is compared, whereas other exact unlearning methods based on Newton updates should have been included for a more comprehensive comparison.

2. In addition, the experiments are conducted solely on plain MLP architectures, which limits the practical relevance of the results — evaluation on more complex or modern architectures (e.g., CNNs or Transformers) would strengthen the claims.

3. Although the paper claims that a single-step Hessian update is unreliable for non-IID retain sets, I did not find a formal proof supporting this statement. Moreover, an explicit example or illustration of what constitutes a non-IID retain set and when such a scenario arises would greatly help readers understand the motivation. I could not locate this explanation in the experiments section either—perhaps I overlooked it. Since the entire premise of the paper relies on this assumption, providing a formal justification and a concrete example would significantly strengthen the work.

### Questions
1. How is the local Lipschitz constant computed at each step to solve the optimization problem? Additionally, is it practically feasible to estimate this constant for large-scale neural networks, where computing exact Hessians or higher-order terms can be computationally prohibitive?

2. What is the effect of the forget data size on the performance of the proposed method? Specifically, how does increasing or decreasing the number of samples to be unlearned influence the convergence, stability, or closeness of the unlearned model to the retrained model?

### Soundness
4

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
This paper studies machine unlearning problem and proposes to use trust region to perform certified machine unlearning with non-i.i.d. data deletion. Instead of one-step Newton, the paper proposes to iteratively perform update to keep track of data shifts. The method is then validated by numerical studies.

### Strengths
The machine unlearning problem is timely. Being able to handle non iid data removal is important. The paper is generally clearly written. By using iterative updates, the method only rely on local Lipschitz constants. The use of trust region method make it possible to accommodate non-iid data removal and distribution shift.

### Weaknesses
The novelty is somewhat limited, this paper combines a few existing ideas. More importantly, unlearning is usually from the context of multi-agent setup. The usage of TR would make it very difficult to generalize to the scenarios of federated learning setting, since coordinated global search required by TR is either very expensive or not possible to implement in distributed setting. The authors need to really motivate well the usage of centralized unlearning methods in any practical settings.

The proposed TR method is iterative, requiring multiple steps, whereas the baseline is a single-step update. The TR method is inherently more computationally expensive. The paper completely omits any analysis of runtime or computational overhead, making it impossible to assess the trade-off between the claimed utility improvements and efficiency.

### Questions
Can the authors compare their method against Qiao or other works that efficiently approximates Newton step? While it is believable that those method would fail under non iid setting, there is no convincing evidence to show improvement. It would be good to add that comparison.

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
3

### Summary
This paper introduces TR Certified Machine Unlearning, a trust region based framework that addresses a key limitation of certified unlearning methods based on Newton updates, namely the assumption that deleted samples are independent and identically distributed. The authors point out that in practical scenarios, deletion requests are often biased, leading to distribution shifts that undermine existing theoretical guarantees. TR Certified Machine Unlearning incorporates a trust region constraint into the Newton update process, adaptively controlling the update radius and step size to ensure that each iteration remains within a locally reliable region.

### Strengths
1. The paper establishes a clear mathematical link between biased deletion and distribution shift and shows why this invalidates conventional Newton-based unlearning. Theoretical derivations, including the local smoothness assumption, descent lemma, and convergence bounds, are carefully constructed and logically consistent, giving the work strong analytical credibility. 

2. The experimental design closely mirrors the theoretical claims. The authors quantitatively relate posterior KL divergence to performance degradation and demonstrate that TR-Certified Unlearning effectively mitigates this effect. The alignment between theory and empirical outcomes makes the proposed framework both interpretable and verifiable. 

3. TR-Certified Unlearning achieves smaller gradient residual bounds and thus requires less noise for certification while preserving model accuracy. This balance between formal privacy guarantees and practical model performance represents a meaningful improvement over existing certified unlearning approaches.

### Weaknesses
1. The experiments are based on small datasets such as MNIST and CIFAR-10, and the models used are relatively simple. It is uncertain whether the proposed method would still show a clear advantage when tested on larger and more complex networks. As model size grows, the extra computation and time required by the trust-region updates may reduce or even cancel out the observed gains. 

2. It would be nice if the paper can include more baselines in the comparisons.

3. The paper briefly mentions potential applications but does not discuss how the proposed method would integrate into existing unlearning frameworks or data deletion pipelines. A short discussion on deployment practicality would make the contribution more complete.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper addresses certified unlearning under biased deletions that induce distribution shift. Instead of a single damped-Newton step, it wraps approximate Newton updates in a trust-region loop: build a local quadratic model, cap the step by a clipped radius, accept or reject using a model-agreement ratio, and iterate. The theory provides a pre-run gradient bound based on local Lipschitz constants estimated in the current trust ball, and the implementation relies on HVPs with truncated CG/LiSSA. The claim is that this yields tighter certificates and tracks the retrained solution better than a one-step Newton method when the deletion set is non-i.i.d.

### Strengths
1.	The trust-region wrapper around an approximate Newton step is technically coherent: it preserves a Newton-like search direction while controlling step size by a model-agreement rule under a locally defined ball.

2.	The certificate is tied to quantities the algorithm actually estimates in the loop, namely a local gradient Lipschitz constant in the current ball and a spectral-norm bound from power iteration, which makes the residual bound operational rather than symbolic.

### Weaknesses
1. The clipping rule sets r_t=∥g_t∥/L_t, while L_titself is defined as the local gradient-Lipschitz constant on the ball B(w_t,r_t). Without an a priori radius or an update scheme that defines L_tbefore r_t, the pair (r_t,L_t)is not well defined and the step size cannot be computed from the specification as written.

2. The paper asserts that, under distribution shift, the trust-region scheme “achieves a closer approximation to the retrained model than a one-step Newton update,” yet the formal results shown in the same section are a pre-run gradient bound and a descent control based on L_max (T). There is no explicit inequality or counterexample in the same metric that compares the trust-region iterate to the one-step Newton solution, so the central comparison is stated but not actually established.
3. To secure positive definiteness of the approximate Hessian, the analysis assumes a damping shift by λI. This is equivalent to inserting strong l_2regularization on the retained objective. The regularization level simultaneously makes the theory go through and changes the very objective being certified; the text acknowledges the damping and its analogy to l_2 but does not separate the effect on certifiability from the change in the target function.

### Questions
Please refer to the weakness above.

### Soundness
2

### Presentation
3

### Contribution
3
