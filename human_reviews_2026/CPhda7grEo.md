# On the Convergence of Muon and Beyond

- Decision: Reject
- Scores: 2, 6, 4, 6, 4

## Abstract
The Muon optimizer has demonstrated remarkable empirical success in handling matrix-structured parameters for training neural networks. However, a significant gap remains between its practical performance and theoretical understanding. Existing analyses show that the Muon variants achieve only a suboptimal iteration complexity of $\mathcal{O}(T^{-1/4})$ in stochastic non-convex settings, where $T$ denotes the number of iterations. To explore the theoretical limits of the Muon framework, we analyze two Momentum-based Variance-Reduced variants: a one-batch version (Muon-MVR1) and a two-batch version (Muon-MVR2). We provide the first rigorous proof that incorporating variance reduction enables Muon-MVR2 to attain the optimal iteration complexity of $\tilde{\mathcal{O}}(T^{-1/3})$, thereby matching the theoretical lower bound for this class of problems. Furthermore, our analysis establishes last-iterate convergence guarantees for Muon variants under the Polyak-Łojasiewicz (PŁ) condition. Extensive experiments on vision (CIFAR-10) and language (C4) benchmarks corroborate our theoretical findings on per-iteration convergence. Overall, this work offers the first proof of optimality for a Muon-style optimizer and clarifies the path toward developing more practically efficient, accelerated variants.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work studies the convergence properties of the Muon optimizer and two variants under suitable assumptions. In particular, for the stochastic non-convex setting, the authors provide convergence guarantees for the standard Muon algorithm and demonstrate improved guarantees for two variance-reduced versions. Additionally, the convergence of Muon and its two variants is examined under the Polyak-Lojasiewicz condition. Finally, the authors present experimental results on vision and language tasks, evaluating Muon and its two variants alongside other popular optimizers.

### Strengths
- The paper is for the most part clearly written and its structure is easy to follow.
- The last-iterate convergence results under the PL condition are interesting.
- The convergence guarantees of the approximate variance reduction scheme (MVR1) without the additive error-term improve previous results.
- The experimental results showcase the improved performance of the proposed algorithms alongside other popular optimizers.

### Weaknesses
- The proposed algorithmic variants of Muon are not novel. Prior work [1] has proposed Shampoo (a generalization of Muon) with both one-batch and two-batch momentum variance reduction and also uses the additional $\gamma$ parameter to control the variance-reduction term. In addition, the algorithm in [1] uses weight decay. However, it is important to note that [1] does not provide convergence guarantees for their algorithm.
- The convergence of standard Muon in the non-convex setting has been studied in previous works, as also mentioned in the paper. For example, [2] shows an iteration complexity of $O(T^{-1/4})$ with a batch size of $O(1)$. In contrast, this works shows an iteration complexity of $O(\log(T)/ T^{1/4})$, that is with an additional $\log(T)$ factor. Additionally, [3] also shows that the stochastic first-order oracle calls in Muon are $O( T^{-1/4})$. However, [3] studies Muon with weight decay.
- The convergence of the variant MVR2 (with $\gamma =1$ and additional weight decay) in the non-convex setting was also studied in a previous work [3]. This work claims that their advantage lies in the use of a batch size of $O(1)$, as opposed to [3]. However, [3] demonstrates an interaction complexity of $O(T^{-1/3})$ using a large batch size only in the first iteration and for all subsequent iterations the batch size is set to $O(1)$. Therefore, the amortized average batch size is also $O(1)$. Furthermore, the complexity in [3] does not involve a logarithmic factor, as opposed to this work, which demonstrates a complexity of $O(\log(T)/ T^{1/3})$. Additionally, Assumption 3.3 of this work is stronger than Assumption 2 in [3].
- This work claims that the MVR1 variant is equivalent to Muon with Nesterov momentum for a specific choice of parameters (equation (4)). I do not see how this is possible, though I may be wrong.
- Some parts of the non-ergodic section are a bit unclear. For example, $\Gamma_t$ is not specified in Remark 3.5.
- The introduction of an additional hyperparameter $\gamma$ makes tuning even more tedious in practice.
- The optimizers in the experiments use weight decay, which is not supported by the theoretical results.
- It is not specified how the variance parameter $\gamma$, momentum parameter $\beta$ and weight decay parameter were tuned in the experiments.
- One minor presentation issue is that standard Muon is referred to as Muon-MVR1 with $\gamma=0$, which is a bit misleading, since in that case no variance reduction is applied.


[1] MARS: Unleashing the Power of Variance Reduction for Training Large Models - Yuan et al.

[2] Convergence Bound and Critical Batch Size of Muon Optimizer - Sato et al.

[3] Lions and Muons: Optimization via Stochastic Frank-Wolfe - Sfyraki and Wang

### Questions
- I understand that $\gamma$ is used to control the variance reduction term, as in [1]. However, this scheme is different from the originally proposed STORM scheme in [4]. I also see that in practice $\gamma$ is set to a very small value, which effectively reduces the impact of the variance reduction. Can the authors explain the motivation for including $\gamma$?
- Can the authors elaborate on how equation (4) is obtained from Algorithm 1?



[1] MARS: Unleashing the Power of Variance Reduction for Training Large Models - Yuan et al.

[4] Momentum-Based Variance Reduction in Non-Convex SGD - Cutkosky and Orabona

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper analyzes the theoretical convergence of the Muon optimizer and introduces two momentum-based variance-reduced variants, Muon-MVR1 and Muon-MVR2. The authors show that Muon-MVR2 attains the optimal stochastic non-convex iteration complexity of $\tilde{\mathcal{O}}(T^{-1/3})$ and establish last-iterate convergence under the Polyak–Łojasiewicz condition. The analysis is elegant and unified: it extends the stochastic momentum framework to the orthogonalized Muon setting, with proofs that are logically consistent and mathematically transparent. Empirical results on CIFAR-10 and C4 further support the theoretical claims.

### Strengths
1. **Clear theoretical contribution.** The paper provides a rigorous convergence analysis for Muon-MVR, bridging the gap between Muon’s empirical success and previously incomplete theory. The Lyapunov-style recursion and variance-control arguments are clean and easy to follow.

2. **Well-structured and consistent analysis.** The logical progression from assumptions to lemmas and theorems is smooth, and the proofs are well-motivated and technically precise. The framework appears general enough to extend to other momentum–variance-reduction methods.

3. **Readable exposition.** Despite the mathematical density, the writing is accessible and the notation—aside from minor issues—remains consistent. The paper is pleasant to read, and the proofs are presented with notable clarity.

### Weaknesses
1. **Overstated claims in presentation.** The statement that the method “overcomes the bottleneck of traditional momentum methods, achieving the optimal iteration complexity” is exaggerated. Sfyraki & Wang (2025) already analyze a Muon+VR variant (under a stochastic Frank–Wolfe formulation) achieving the same $\tilde{\mathcal{O}}(T^{-1/3})$ rate. The novelty here lies primarily in the unconstrained SGD-style analysis, not in attaining a new theoretical complexity.

2. **Ambiguity in Table 1.** The superscripts $c$ and $d$ are undefined. Their meaning should be clearly explained in the caption or a footnote.

3. **Formula inconsistency (Eq. (4)).** As written,
   $$
   M_t = \mu M_{t-1} + \nabla f(X_t, \xi_t) + \mu \big(\nabla f(X_t, \xi_t) - \nabla f(X_{t-1}, \xi_{t-1})\big)
   $$
   is not equivalent to Eq. (2). It should instead be
   $$
   C_t = \mu C_{t-1} + (1-\mu)\nabla f(X_t,\xi_t), \quad
   M_t = \mu C_t + (1-\mu)\nabla f(X_t,\xi_t).
   $$

4. **Possible overclaim in Remark 3.2.** Although the analysis avoids a growing-batch assumption, the comparison with Sfyraki & Wang (2025) is not fully aligned, as their problem setting, objective, and convergence metric differ. Characterizing this as a “significant theoretical advantage” is therefore overstated—especially without empirical evidence that fixed-batch training achieves the claimed $\tilde{\mathcal{O}}(T^{-1/3})$ scaling.

5. **Experimental limitations.** The empirical section primarily plots standard training curves and does not directly test the predicted scaling laws. Moreover, there is no comparison against other Muon-type optimizers. As a result, the experiments feel empirical rather than diagnostic.

6. **Typographical and constant issues.**
   - (1) Page 15 (L799): should read $\epsilon = \frac{1-\beta_{t+1}}{\beta_{t+1}}$.
   - (2) Page 15 (L801), Eq. (6): the right-hand side contains an extra “$t$”.
   - (3) Page 22 (L1187): the second term’s constant should be $2\eta_{t+1}^2(4L^2 n + \sigma^2)$; assuming $\eta_t \le 2\eta_{t+1}$, we have $2\eta_t^2 L^2 n \le 8\eta_{t+1}^2 L^2 n$, which affects multiple subsequent constants.

### Questions
The main improvement of Muon lies in its orthogonalization step. Could the authors elaborate on why orthogonalization improves optimization dynamics over standard SGD? Is there a formal argument or empirical evidence that the equalization of singular values accelerates convergence?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work integrates Momentum Variance Reduction (MVR)—including a MARS‑style $\gamma$ factor—into Muon. It analyzes three variants (standard Muon/MVR1 with $\gamma=0$, MVR1 with $\gamma_t=t^{-1/2}$, and MVR2 with two gradients on the same mini‑batch), proving ergodic rates $\tilde{\mathcal O}(T^{-1/4})$ for MVR1 and $\tilde{\mathcal O}(T^{-1/3})$ for MVR2, plus PŁ last‑iterate rates $\tilde{\mathcal O}(T^{-1/2})$ (MVR1) and $\tilde{\mathcal O}(T^{-2/3})$ (MVR2). Experiments on CIFAR‑10/ResNet‑18 and LLaMA2‑130M/C4 are provided.

### Strengths
- The core idea is reasonable and timely: MARS-style variance reduction has worked well in practice, and combining MVR with Muon yields provable benefits, making the approach promising for practical impact.
- The analysis of a more practical, one-batch MVR variant appears novel at the algorithmic level; however, I am uncertain about the theoretical novelty/difficulty of this result.
- The main results are stated cleanly and appear correct.
- The algorithmic setup is presented clearly: MVR1 vs. MVR2, schedules, and update rules.
- The empirical section covers reasonable benchmarks (CIFAR-10/ResNet-18, LLaMA2-130M on C4) and broadly supports the claimed advantages, especially for MVR2.

### Weaknesses
- Despite its practical motivation, the paper’s theoretical analysis appears to offer limited novelty. The proof technique closely mirrors that of normalized SGD with momentum (Cutkosky & Mehta, 2020)—which is, surprisingly, not cited—and is similar to prior analyses of Muon’s convergence. Moreover, MVR in this context has already been discussed (e.g., the OptML notes: https://optmlclass.github.io/notes/optforml_notes.pdf), and parameter-agnostic results under comparable assumptions (e.g., Yang et al., 2023) are not addressed. As a result, beyond analyzing a more practical $\gamma$-factor variant, the contribution seems incremental: with a decreasing schedule, $\gamma_t$, the analysis does not improve the bound over $\gamma=0$ (it essentially recovers the SGD rate). The paper should add a clear, dedicated discussion of technical novelty, explicitly situating its results relative to these works and clarifying what is genuinely new in the Muon-specific setting.

- The $\gamma$‑factor with a decreasing schedule ($\gamma_t=t^{-1/2}$) recovers the same $\tilde{\mathcal O}(T^{-1/4})$ rate as $\gamma=0$, so the analysis does not show a theoretical gain from the MARS‑like correction.
- Related‑work positioning: compares against relatively weak baselines and omits important works such as Kovalev (2025); needs a more detailed comparison to Sfyraki & Wang (2025).
- Optimality beyond the $T$‑exponent is unclear: the dependence on $L,\sigma,n$ (especially for Theorem 3.2) is not discussed relative to known parameter‑agnostic results.
- Missing important baseline: MARS‑AdamW should be included.
- Missing/unclear experimental details: whether Muon is combined with AdamW for some parameter groups, and the separate learning rates; absence of ablations on $\gamma$, weight decay, and batch size; on C4, a single fixed LR is used for all optimizers, which is hard to justify.
- “Nesterov‑accelerated” wording for Option MVR1 is incorrect for the presented update.

---
- Cutkosky, A., & Mehta, H. “Momentum improves normalized SGD.” ICML, 2020.
- Yang, Junchi, et al. “Two sides of one coin: the limits of untuned SGD and the power of adaptive methods.” NeurIPS 36, 2023.
- Kovalev, Dmitry. “Understanding gradient orthogonalization for deep learning via non-Euclidean trust‑region optimization.” arXiv:2503.12645, 2025.
- OptML notes on MVR in this context: https://optmlclass.github.io/notes/optforml_notes.pdf￼

### Questions
- Technical novelty: how do the proofs differ materially from normalized‑SGD‑with‑momentum (Cutkosky & Mehta, 2020) and prior Muon analyses; why are these not cited? Please also discuss connections to the OptML MVR notes and parameter‑agnostic results (Yang et al., 2023).
- Positioning vs. prior work: add a careful comparison to Sfyraki & Wang (2025) and include Kovalev (2025); specify which assumptions/estimators.
-  Baselines: add MARS‑AdamW on CIFAR‑10 and C4.
- Empirical details: clarify if Muon is used only for matrix‑shaped parameters while AdamW is used elsewhere; list parameter groups and per‑group learning rates; report weight decay values.
- LLaMA2/C4 setup: justify using a single LR for all optimizers or include at least a small per‑optimizer LR sweep.
- Constants: discuss the dependence on $L,\sigma,n$ in Theorems 3.1–3.2 and whether any factors are tight or can be improved.

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
This paper aims to establish a rigorous theoretical foundation for the Muon optimizer, which has recently gained attention for its remarkable empirical performance. The authors point out the limitations of existing Muon analyses and propose two variants to address these issues.

### Strengths
The primary strength is the first proof that Muon-MVR2 achieves optimal iteration complexity in the stochastic non-convex function. The paper doesn’t stop at presenting theoretical results. It validates the algorithm through experiments on CIFAR-10 and C4 datasets.

### Weaknesses
While it's commendable that this study used a large dataset like C4, the testing was conducted on only one model, LLaMA2-130M. Although the paper's main contribution is theoretical, the lack of evaluation on more than one model and more than one dataset remains its biggest drawback.

### Questions
1.	Why wasn't the wall-clock time shown in Figure 2? While O-notation is important, what is practically significant is essentially the wall-clock time.

2.	Muon is primarily known as an optimizer frequently used for LLMs. Therefore, I question the choice of using an older model from 2016 (ResNet-18) specifically for image classification experiments. If the goal was to demonstrate its effectiveness in the vision domain as well, why weren't large-scale datasets like ImageNet/MS COCO and more recent vision models such as CLIP or DINO employed? The chosen combination of ResNet18 & CIFAR-10 might inadvertently lead readers to suspect that the method failed on more modern and larger-scale vision tasks, and thus only results from this simpler setup were presented.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents analysis of two momentum-based variance-reduced Muon-MVR1 and Muon-MVR2 algorithms, which achieve sublinear convergence rates. In addition, the paper also provides convergence analysis details for the standard Muon and Muon-MVR1 methods. Muon-MVR2 variant offers some additional analysis on achieving optimal iteration complexity of $O(T^{-1/3})$ for non-convex optimization. Primary comparisons are between Muon-MVR2 versus the theoretical lower bound. The paper presents some small-scale experiments (such as with CIFAR-10 and LLaMA2), and practical results imply that those looking for the best theoretical convergence rate should use the Muon-MVR2 variant.

### Strengths
- The main contribution of this paper is it's rigorous theoretical proof of optimality for the Muon-class optimizer. There has been some recent interest in understanding the performance of Muon [1], and aiming to close the gap between Muon's practical usability versus its theoretical understanding is valuable. 

- The proposed Muon-MVR2 achieves its optimal convergence using a constant batch size $O(1)$. This is a significant improvement over prior work [2] that needed large initial batch sizes. The last-iterate non-ergodic convergence analysis is also rigorous, showing $O(T^{-2/3})$. 

- There are experiments in both vision and language domains, although the main contribution of the paper remains its theoretical clarification of the Muon framework. There is a clearly expressed tradeoff between the "cheaper" Muon-MVR1 variant and the theoretically backed MVR2 variant. 



[1] Shen, Wei, et al. "On the convergence analysis of muon." arXiv preprint arXiv:2505.23737 (2025).

[2] Sfyraki, Maria-Eleni, and Jun-Kun Wang. "Lions and muons: Optimization via stochastic frank-wolfe." arXiv preprint arXiv:2506.04192 (2025).

### Weaknesses
- The most concerning point is that the proposed Muon-MVR2 variant requires 2 gradient evaluations per step. This practically introduces almost double the amount of computational overhead. 

- This practical-vs-theory tradeoff is shown in the paper's own experiments. The authors transparently note that the overhead can be prohibitive, especially visible in Fig 1C where the theoretically slower MVR1 is actually faster in practice per wall-clock time. Since the more practically usable MVR1 variant is stuck at the suboptimal $O(T^{-1/4})$ convergence rate, this means most people will likely use this lower-cost variant. Making the implications of the paper's main headline results less impactful. 

- Experiments are small-scale. CIFAR-10 is an outdated and small dataset, but more critically LLaMA2-130M is only trained for 10,000 steps. At this very small scale, it is unclear how the benefits outweigh the costs of Muon-MVR2. The variance reduction technique also introduces new hyperparameters ($\gamma$), which becomes another expensive step to tune in any practical setting.

### Questions
- When is Muon-MVR2 the better choice? Fig1 A/B seems to show it's better per-epoch, but it appears slower than Muon-MVR1 in wall-clock time on Fig 1C. Is there a point after N number of epochs where its faster convergence overcomes its 2x per-step cost? A larger experiment is needed to examine this possibility. 

- How well does Muon-MVR2 compare against variance-reduced optimizers? Such as SPIDER[3] and SVRG? 

- The theoretical contributions of the paper largely depend on perfect orthogonalization. But in practice this is an approximation (Newton-Schultz) for efficiency. Does using the approximation break these theoretical guarantees? 

[3] Fang, Cong, et al. "Spider: Near-optimal non-convex optimization via stochastic path-integrated differential estimator." Advances in neural information processing systems 31 (2018).

[4] Gower, Robert M., et al. "Variance-reduced methods for machine learning." Proceedings of the IEEE 108.11 (2020): 1968-1983.

### Soundness
3

### Presentation
2

### Contribution
3
