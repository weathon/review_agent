# Towards Robust Regularization with Central-difference and Momentum Lookahead

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Sharpness-Aware Minimization (SAM) is an effective technique for improving generalization by guiding optimizers towards flat minima through parameter perturbations. However, extending such regularization strategies to multi-step settings often leads to instability, where naive iterative updates degrade rather than enhance generalization. To overcome this limitation, we propose Central-difference Momentum Lookahead Regularization (CMLR), a framework that performs momentum lookahead through central-difference probing of the loss landscape. By constructing the perturbation direction from symmetric gradient evaluations, CMLR realizes a momentum lookahead update that is inherently more robust and exhibits reduced variance, while requiring no additional gradient evaluations. This design ensures smooth optimization trajectories and reliable improvements at low computational cost. We establish formal convergence guarantees together with a variance reduction analysis for CMLR, and empirically demonstrate that it consistently improves generalization across diverse architectures and datasets.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper attempts to overcome the limitation of forward-difference aprroximation of SAM and proposes a SAM variant, CMLR, that incorporates central-difference and momentum lookahead mechanism. The authors argue that the proposed optimizer is more robust and exhibits reduced variance. The empirical results further suggest that CMLR consistently performs better across various architectures and datasets.

### Strengths
This papers uncovers an important observation that a high concentration of similarity between consecutive updates is critical to generaliztion. Based on this observation, the authors propose two tricks, central differencing and mometum lookahead, to mitigatigate the problem of generalization degradation in mutli-step SAM. The authors further demonstrate the superiority of the proposed optimizer, theoretically and emprically. The paper is overall well-motivated and easy to follow.

### Weaknesses
While the empirical results suggest that CMLR is effective, I still have a few comments as follows:

- In Section 2.2, the authors state that **The consensus points to gradient instability as the main cause**. What is the meaning of gradient instability? Do you mean gradient explosion? Some works[1][2] otherwise refer it to being easily get trapped in the saddle points. Please clarify.

[1] Kim et al., Stability Analysis of Sharpness-Aware Minimization, 2023.

[2] Tan el al., Stabilizing sharpness-aware minimization through a simple renormalization strategy, JMLR, 2025.
 - In Theorem 4.2, Equation (12) does not imply that $\nu_{CMLR} \leq \nu_{LR}$. So, I am wondering how the statement **CMLR...exhibits reduced variance** is concluded?
 - In Algorithm 1 line 9-10, the true gradient $\nabla L(w)$ is approximated by the mean of $g_k^+$ and $g_k^-$. Actually, this operation could be highly unstable when $\rho$ is relatively large, at least for $\rho\geq0.05$. Could the authors provide some experiment results to show that this approximation is very close to $\nabla L(w)$?
 - Experiments on ImageNet-1K should be included to further validate the efficacy of CMLR. Moreover, how the smoothing factor $\beta_s$ and $\beta_e$ affect the generalization should also be investigated.

### Questions
see Weaknesses.

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
The authors provide a novel optimizer for ANNs called Central-difference Momentum Lookahead Regularization (CMLR).
CMLR extends SAM, aiming to reduce instabilities that occur when training with SAM.
Following related work, the authors interpret SAM as a gradient regularization method approximating second-order terms by a simple forward difference scheme.
The authors claim to reduce the resulting instabilities by replacing the forward difference with a central difference.
CMLR implements this forward difference together with a gradient approximation based on previous gradients into the two-step lookahead framework.
The authors show a convergence analysis and empirical evaluation on CIFAR10/100.

### Strengths
The idea is well presented and well motivated. The empirical results shown lead to performance improvements.

### Weaknesses
1. The empirical evaluation is insufficient. Studying only CIFAR10/100 is not very meaningful. Especially, VGG16 is very outdated. The authors should perform additional experiments on larger (ImageNet) as well as more diverse (e.g., NLP) datasets.
2. The baseline performances are all not good. Especially, the ViTs' performance is very bad in all experiments. The authors should select a solid baseline before claiming to achieve performance improvements with their method.
3. Reproducibility of the empirical results is not given. Neither are the general hyperparameters (learning rate, perturbation strength, and weight decay) given, nor are the results of the hyperparameter search for CMLR given.
4. The choice of hyperparameters is not clear. The choice of $\beta_e$ is not discussed. How exactly was the HP search performed? Multi-dimensional grid search over all HPs? On a validation set or the training set? Were different parameters chosen per model/dataset? The full HP optimization results should be reported. About 1\%-2\% test gain is not much for introducing 6 new hyperparameters ($\rho, \beta_s, \beta_e, \alpha, K, \lambda$) and tuning these on the test set.
5. The impact of the introduced hyperparameters should be studied in more detail. This especially holds for $K$, which is simply fixed by the authors.
6. Regarding the consecutive gradient updates, the authors claim in lines 53-54, "a higher concentration of similarity scores away from 0 strongly correlates with better generalization". This claim is not supported at all. I am not a fan of putting Fig. 1 in the introduction. The interpretation is very unclear. It does not help the reader to understand the motivation of the method. Why is the cosine similarity even bimodal? And why are there values larger than 1 shown in the figure?
7. What was the compute budget for the different methods reported in Tab. 1 and 2? If the authors keep the number of epochs fixed but perform $K=10$ steps per batch, this would result in a 10x higher computational budget for the reported results. Is this correct?
8. In Fig. 6 in the appendix, the authors show that $\alpha = 1$ achieves the best results for CIFAR100 and the second-best results for CIFAR10. $\alpha = 1$ results in the slow weights matching the fast weights, which collapses the lookahead mechanism to default SGD with multiple steps per batch. Why did you choose the lookahead mechanism, if it is not beneficial?

Minor Comments:
- In line 113, there should be no $\nabla$ in "training loss $\nabla L(w)$".
- Fig. 2 is placed on page 4 but only referenced on page 6. In general, I do not think Fig. 2 helps to understand the algorithm.
- When giving an approximation, you should use $\approx$ instead of = (e.g., Eq. 5).
- There should be an "Eq." or similar on line 257.
- Algorithm 1 is not in Appendix A as stated in line 285.
- In Appendix F.1.1, the discussion of $\gamma_{interp}$ should be in a separate section (starting from line 1360 "A key component...").

### Questions
1. In line 224, you claim, "The core idea of this formulation is to shift the update's focus. It relies more on the standard gradient at the original point $w$...", however, you later approximate $\nabla L(w)$ by the mean of the two perturbed gradients, so your method does not rely on the gradient at the original point $w$ at all. How does your approximation of the gradient align with your motivation?
2. I did not really get the conclusion of the convergence analysis (I have to admit that I did not study the appendix in detail). What is the interpretation of Eqs. 12 and 13?
3. How does your method perform if you do not anneal $\beta_k$? It is not clear to me why the annealing is needed.
4. What is SAM-N in the legend of Fig. 3? "normal"?
5. $\gamma_{interp}$ is not used in the main paper; instead, the mean of both gradients is used, i.e., $\gamma_{interp}=0.5$, right? Please state that in the relevant appendix chapter. Also, Fig. 5 is pure noise. There is no meaningful effect (which is quite surprising).

### Soundness
2

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
The paper reframes SAM and related sharpness-aware methods through Gradient Regularization (GR) and argues that SAM’s multi-step instability stems from its forward-difference approximation. It proposes a more stable central-difference GR update and embeds it in a momentum Lookahead inner–outer loop to form CMLR

### Strengths
The GR lens makes SAM’s multi-step instability traceable to a first-order forward-difference; replacing it with a second-order central-difference reduces approximation error and stabilizes updates

Convergence for non-convex objectives ($O(\lambda)$ neighborhood) and variance-reduction analysis, including a spectral treatment for CMLR

The extensive ablation studies and the good empirical breadth are also a strength 

The inner loop uses $g^+$ and $g^-$ and a no-extra-grad momentum lookahead update for $v_{k+1}$, keeping cost comparable to SAM while enabling multi-step smoothing.

### Weaknesses
The paper stresses “no additional gradient evaluations,” but CMLR still requires two perturbed gradients per inner step (as does SAM’s two-pass). Please add wall-clock breakdowns (forward/backward counts, memory) vs. SAM/GSAM/Lookbehind-SAM under equal budgets, not only vs. CLR

Convergence guarantees hinge on smoothness, Lipschitz Hessian, and bounded moments; it would help to diagnose failure modes (e.g., very large $\rho$, poor $\lambda$, or aggressive $\beta$ annealing) and relate them to the cosine-similarity stability metric (Fig. 1).

the novelty margin w.r.t. CR-SAM (curvature-regularized SAM), GSAM, and Lookbehind-SAM could be sharpened beyond empirical deltas; a conceptual map clarifying where CMLR differs mathematically from CR-SAM’s curvature proxy would help

There exist dynamic learning-rate schedules that also adapt to local landscape/sharpness (e.g., SALR: Sharpness-Aware Learning Rate Scheduler for Improved Generalization) which pursue flatter minima without inner maximization by coupling the learning rate to sharpness estimates. A conceptual comparison between CMLR’s update-direction regularization and LR-based dynamic exploration could enrich the paper.

### Questions
Could you report (i) number of forward/backward passes per parameter update, (ii) GPU hours, and (iii) peak memory for CMLR vs. SAM/GSAM/Lookbehind-SAM under the same training time? The current efficiency result is mainly CLR vs. CMLR

Can you consider adding a small Transformer experiment (e.g., text classification) or a speech benchmark to demonstrate modality robustness ?

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
3

### Summary
This paper investigates the reason behind the failure of multi-step SAM, attributing it to excessive variance, and proposes using central difference instead of one-sided difference to address this issue. By combining this idea with momentum lookahead, the authors introduce CMLR and demonstrate consistent performance improvements in their experiments.

### Strengths
The strength of this paper lies in offering a new perspective to explain the poor performance of multi-step SAM, namely that the one-sided difference used to approximate the Hessian–vector product leads to excessively high variance. The argumentation is sound, and the authors propose a practical solution to this issue, using CMLR with central difference instead. The effectiveness of the proposed method is validated through experiments.

### Weaknesses
The main weakness of this paper is that I do not believe multi-step SAM is used in practice, as it results in a severalfold increase in computational cost. In the experiments, the authors use k=10, which, to my understanding, implies that the training time is multiplied by ten. This is clearly impractical. The practical significance of studying and proposing algorithms under such a setting is quite limited. In addition, the presentation of the paper is somewhat confusing, making it difficult to follow.

### Questions
I could not find the full name of FMLR in the paper, and terms such as CMLR and ML-SAM appear before their complete forms are introduced (lines 72, 75, 78), which makes the text difficult to follow.

The authors claim that “The test accuracies show a clear hierarchy (CMLR > FMLR > CR > FR > SAM > ML-SAM)”, but I could not find the corresponding performance results in the paper.

Did the authors ensure a fair comparison in the main experiments? I only see that their method is run with 
k=10. For the baselines, were multi-step versions of SAM, GSAM, or CR-SAM used? Was SGD allowed to run for ten times longer? Is there any comparison under equal computational cost?

Why does ViT perform much worse than ResNet-18? Did the authors conduct sufficient hyperparameter tuning, or consider using a larger dataset such as ImageNet, or fine-tuning a pretrained ViT?

The paper also does not report the additional computational overhead of CMLR and related algorithms, such as wall-clock time.

The authors claim that central-difference gradients can be computed in parallel, but I do not see why this is the case. From the pseudocode, it appears that the gradients need to be computed separately after perturbing in both directions. Could the authors explain this point in more detail?

The paper does not cite Becker et al. (2024), “Momentum-SAM: Sharpness Aware Minimization without Computational Overhead”, which appears to be closely related.

### Soundness
2

### Presentation
2

### Contribution
2
