# Stability Bounds for Domain Generalization under Limited Data

- Decision: Reject
- Scores: 4, 2, 4, 6

## Abstract
The less-sample learning problem is challenging for machine learning, as it leads to unstable model estimation, i.e., the risk gap between the empirical risk and the expected risk for models increases as the size of the training data decreases. To address this, the classical VC-bound suggests reducing the VC-dimension of models through regularization. However, the data in domain generalization are not independent and identically distributed (i.i.d.), which implies that such bounds fail to provide effective guidance for learning. To fill this gap, we present stability bounds. Specifically, we derive a general exponential-decay upper bound based on the notion of stability for models and McDiarmid’s inequality. Based on this, we then present stability bounds for models obtained by regularization-based learning methods. Finally, we apply this result to a classification case and develop a learning method. We also study the stability and generalization error bounds of the proposed learning method, as well as its convergence properties. Additionally, we conduct experiments using datasets with different data sizes to analyze the effectiveness of our methods in real-world applications.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses limited-data domain generalization by deriving stability-based generalization bounds and proposing DLAERM, a domain-level L2-regularized method. The theory shows error decays exponentially with stability, and the algorithm ensures good generalization even with few domains, validated by theoretical guarantees and experiments.

### Strengths
- The paper derives new, exponential-decay generalization bounds for DG using stability theory. These bounds do not rely only on i.i.d. data. 

- A new algorithm, DLAERM, is proposed. The method is proven to have a linear convergence rate, ensuring efficient optimization.

- Experiments on benchmarks (PACS, OfficeHome) confirm the theory.

### Weaknesses
- Most proofs appear valid, but Theorem 1 requires caution; the inequality in Eq.(3) may still depend on the number of domains $E$.

- The paper does not define a proper partial order on function spaces; for example, when writing $h \ge h'$, the meaning is not specified.

- Paper structure is complete and the main thread is clear, but notation and semantic roles (e.g., $R,R',R_{\exp}$) switch without clear clarification, increasing cognitive burden.

- Experimental interpretation is insufficient. Figure 1 shows trends but is not explicitly linked to theoretical quantities such as $\beta$.

- Missing discussion of classic and recent DG literatures, e.g.

[1] Gulrajani & Lopez-Paz (2021) In Search of Lost Domain Generalization, ICLR.

[2] Tong et al. (2023) Distribution Free Domain Generalization, ICML.

### Questions
1. Page 14, Lines 733. In the proof of Theorem 1, should the right hand side of the inequality be multiplied by an additional factor $E$? If so, this modification would propagate to the main results and may alter the final bounds.

2. The current proof of Lemma 1 is not mathematically rigorous with respect to the use of order relations and the treatment of subgradients.

3. (If possible) Demonstrating empirical performance on large scale datasets would further strengthen the credibility of the proposed method.

4. See Weaknesses

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper studies domain generalization (DG) via algorithmic stability. It adapts the classical Bousquet–Elisseeff framework (with McDiarmid’s inequality) to a multi-domain setting by assuming independence (but not necessarily identical distribution) across domains, and then instantiates the bound for RKHS ERM with $\ell_2$-regularization. The main statement yields a stability parameter of the form
$\beta \le \tfrac{2\sigma^2\kappa^2}{\lambda E N}+\beta^\*$,
leading to a tail bound of McDiarmid type (“exponential decay”) and a convergence result under strong convexity and plain GD.

### Strengths
- The paper is well-structured; the stability-to-generalization pipeline is explicit and easy to follow.

- Instantiated stability for RKHS ERM. Provides a closed-form stability bound in terms of $(\lambda, E, N, \kappa, \sigma)$ that is interpretable and matches classical intuition (regularization improves stability).

- Simple empirical illustration. The toy experiments align with the qualitative prediction that stronger regularization helps in small-sample regimes.

### Weaknesses
- Limited novelty vs. classical stability. The core technical path (uniform stability $+$ McDiarmid tail) closely mirrors standard results; the “exponential-decay bound” is the usual concentration tail, not a new rate or DG-specific mechanism.

- Strong/atypical assumptions. The bespoke stability notion on $R(h,D)$ vs. $R(h,D_v)$, existence of $h^\*$ minimizing $|R-R'|$ with $\nabla R(h^\*)\ge0$, bounded kernel $\kappa$, and $\sigma$-admissibility collectively make the scope narrow; the role and size of $\beta^\*$ are left largely implicit.

- Theory–practice mismatch. Convergence is proved under strong convexity with plain GD, whereas experiments use deep models where assumptions do not hold; no bridge (e.g., surrogate convex surrogates or local strong convexity) is provided.

### Questions
Clarify the necessity and typical size of $\beta^\*$; provide conditions under which $\beta^\*\approx0$ or explicit upper bounds.

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
3

### Summary
This paper addresses the less-sample problem in domain generalization (DG), where limited training samples per domain lead to unstable model estimation and degraded generalization. To overcome the limitations of classical VC-bound–based generalization analysis (which assumes i.i.d. data), the authors propose a new stability-based theoretical framework using McDiarmid’s inequality to derive an exponential-decay generalization bound under non-i.i.d. data.
Building on this theory, the authors propose a regularization-based DG method, named DLAERM (Domain-Level Adaptive Empirical Risk Minimization), that adaptively controls the regularization strength λ according to data size and domain count. They also prove the method’s β-stability bound and Q-linear convergence, showing that stronger regularization improves stability for small-sample regimes.
Empirically, the method is validated on PACS and OfficeHome datasets using ResNet-50 as the feature extractor. DLAERM and its variants (e.g., DLAFISH, DLAEQRM) consistently outperform existing IRM-, ERM-, and regularization-based baselines, particularly under smaller training datasets (OfficeHome).

### Strengths
The paper provides a rigorous derivation of generalization bounds based on stability theory rather than the classical VC-bound, making it suitable for non-i.i.d. DG settings. The use of McDiarmid’s inequality to relax the independence assumption is theoretically elegant and relevant.

The proposed DLAERM framework naturally generalizes multiple regularization-based DG methods (LASERM, DLAFISH, etc.) through the same stability-based perspective. This offers a unified explanation for how λ controls stability and generalization.

Experiments on PACS (large sample) and OfficeHome (small sample) clearly show that DLA-based methods improve classification accuracy, demonstrating robustness to data scarcity.

### Weaknesses
Only two datasets (PACS, OfficeHome) are used, both for image classification. The method’s applicability to more complex DG benchmarks (e.g., TerraIncognita, VLCS, DomainNet) is not explored.

Although λ plays a central theoretical role, there is no systematic study of how varying λ affects stability, generalization gap, or convergence speed in practice.

Since the paper claims to solve small-sample problems, it would be insightful to compare with sample-efficient or augmentation-based methods (e.g., Mixup) that target similar challenges.

### Questions
How sensitive is the method to the choice of λ? Could an adaptive or learnable λ further improve performance?

How does the DLAERM framework behave when the number of domains E is very small (e.g., 2 or 3)? The bound suggests dependence on E, but experiments seem to focus on 4-domain setups.

Is there any trade-off observed between stability (small β) and model expressiveness (underfitting risk) when λ is too large?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper develops a new theoretical framework to analyze and mitigate instability in \textbf{domain generalization (DG)} when training data per domain are scarce.  
The authors argue that traditional VC-based generalization bounds are ineffective in DG since data are not i.i.d., and they instead derive stability-based exponential-decay generalization bounds using McDiarmid’s inequality.  
Their main results show that for regularization-based learning methods, the generalization gap satisfies $\varepsilon \leq \mathcal{O}\left(\frac{1}{\sqrt{\lambda E N}}\right)$,
where $\lambda$ is the regularization factor, $E$ the number of domains, and $N$ the number of samples per domain.  They extend these results to learning in a reproducing kernel Hilbert space (RKHS) and demonstrate how the bounds motivate a new learning method, **DLAERM** (Domain-Level Aggregated Empirical Risk Minimization), which uses inter-domain sparsity to achieve stable generalization without requiring many domains.  

Empirical evaluations on PACS and OfficeHome datasets with ResNet-50 validate the theoretical insight: DLAERM and its variants consistently outperform classical ERM and IRM-based baselines, especially in low-data regimes.

### Strengths
**Originality:**
1.  Introduces a stability-based theoretical framework tailored to domain generalization, extending Bousquet and Elisseeff’s stability notion beyond the i.i.d. setting.
2.  Derives exponential-decay generalization bounds using McDiarmid’s inequality, linking data size, regularization strength, and generalization gap in a novel way.
3. Proposes the DLAERM algorithm, which integrates theoretical insights into a practical learning objective with inter-domain sparsity.

**Quality:**
1.  The theoretical analysis is rigorous and carefully derived, with clear proofs and explicit assumptions (e.g., $\sigma$-admissibility, RKHS setup, convexity).
2.  Empirical results, though limited in scope, support the theoretical claims—especially the inverse proportionality between $\lambda$ and $EN$.
3.  The paper clearly articulates differences from prior work (VC-bounds, Ben-David et al. 2010, Bousquet \& Elisseeff 2002) and justifies the relevance of its assumptions.


**Clarity:**
1. The structure of the paper is logical and systematic: from motivation to theory, learning method, and experiments.
2.  Mathematical exposition is detailed yet readable, with intuitive remarks following each theorem and corollary.
3.  Figures and tables (especially Table 1 and Figure 1) effectively summarize empirical trends.


**Significance:**
    1. Provides a theoretical foundation for DG under small data, a highly relevant but under-explored regime.
    2. Offers an interpretable link between stability theory and practical DG learning algorithms.
    3. May inspire future research on stability-based regularization or generalization analysis in non-i.i.d. settings.

### Weaknesses
**Limited empirical validation.** Experiments are confined to two datasets and one backbone (ResNet-50). Further tests on non-vision tasks (e.g., text DG) or larger architectures would strengthen claims of generality.
  
**Comparative scope.** The experiments mainly modify existing baselines by adding the DLA regularizer. Stronger DG benchmarks or recent transformer-based backbones (e.g., ViT) are missing.

### Questions
1. Theorem 2 and Corollary 1 depend critically on the assumption that data across domains are independent. How sensitive are the stability bounds to mild violations of this assumption (e.g., correlated domain sampling)?
    

2.  In practice, how should $\lambda$ be tuned when both $E$ and $N$ vary? Is there an adaptive rule (e.g., $\lambda \propto 1/(EN)$) that matches empirical results?

3. Given that $\beta*$ can dominate the bound when large, is there a measurable or estimable way to assess $\beta*$ in real datasets?

### Soundness
3

### Presentation
3

### Contribution
3
