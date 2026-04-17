# Composite Optimization with Error Feedback: the Dual Averaging Approach

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 8, 8

## Abstract
Communication efficiency is a central challenge in distributed machine learning training, and message compression is a widely used solution. However, standard Error Feedback (EF) methods (Seide et al., 2014), though effective for smooth unconstrained optimization with compression (Karimireddy et al., 2019), fail in the broader and practically important setting of composite optimization, which captures, e.g., objectives consisting of a smooth loss combined with a non-smooth regularizer or constraints. The theoretical foundation and behavior of EF in the context of the general composite setting remain largely unexplored. In this work, we consider composite optimization with EF. We point out that the basic EF mechanism and its analysis no longer stand when a composite part is involved. We argue that this is because of a fundamental limitation in the method and its analysis technique. We propose a novel method that combines _Dual Averaging_ with EControl (Gao et al., 2024), a state-of-the-art variant of the EF mechanism, and achieves for the first time a strong convergence analysis for composite optimization with error feedback. Along with our new algorithm, we also provide a new and novel analysis template for inexact dual averaging method, which might be of independent interest. We also provide experimental results to complement our theoretical findings.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the failure of standard EF methods in composite optimization, where the objective includes a non-smooth component. The authors argue that the proximal step used in this setting disrupts the additive structure essential for traditional EF analysis. To resolve this, they propose a novel algorithm that combines Dual Averaging with a modern EF variant (EControl). This approach successfully restores the necessary structure, enabling the first strong convergence guarantees for composite optimization with communication compression.

### Strengths
- Fills a gap in optimization theory by providing the theoretically-backed EF method for the general composite setting.

- The use of Dual Averaging is a conceptually clean and logical solution that directly targets the structural issue at the core of the problem.

### Weaknesses
- The claims of the paper are not backed by strong experimental evidence. The experiments mentioned are not detailed enough to validate the practical advantages of the proposed method over existing techniques.

- While the authors suggest their analysis provides a versatile template for other domains (e.g., safe reinforcement learning), the paper does not demonstrate these applications.

### Questions
N/A

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
The work deals with reducing the communication in distributed optimization cost via compression, in the setting where there is an additional regularization term whose proximity operator is applied by the server.

### Strengths
Reducing the communication cost of distributed optimization is an important topic. Combining compression and stochastic gradients is not fully understood and this work is a step forward in this direction.

### Weaknesses
** I have the following concerns

1)  "In the convex regime, it is believed that the convergence of proximal EF21 critically relies on the bounded domain assumption, which we do not assume in our work" This is not correct. There is no such restriction for EF21. The Islamov et al. paper is for non-smooth functions, which is a different setting.

2) "EF21, which is more closely related to gradient-difference compression methods". I don't know why you claim that EF21 might not be really an EF method. The control variates are defined recursively, so they accumulates past compression errors. 

3) A refined analysis of EF21 in the composite case is in Condat et al. “EF-BV: A Unified Theory of Error Feedback and Variance Reduction Mechanisms for Biased and Unbiased Compression in Distributed Optimization,” Neurips 2022

4) It seems that the focus of the work is about dealing with stochastic gradient errors and linear speedup with respect to such errors (linear speedup with respect to compression errors with independent unbiased compressors is equally important but is not addressed).

5) Explain why convexity is critical to the analysis. Convexity is a restrictive assumption. This is fine to me, but on the other hand in convex settings, Nesterov acceleration becomes available. So, there is an important gap between your guarantees and the convergence speed that can be obtained if $\sigma\rightarrow 0$ and $\delta\rightarrow 1$.

** Typos and minor comments

* Beznosikov et al: the reference is given twice.

* Fatkhullin et al. 2021: published in Journal of Machine Learning Research, 26, 2025

* Algorithm 1 line 4: :=:=

* unlesss

* There should be a period at the end of equations, e.g. 15, 16, 17

### Questions
1) The $f_i$ can be non-convex? It seems that yes, but it is important to state it clearly.

2) What is the real challenge in dealing with $\Psi$? In algorithms such as EF21 or DIANA, dealing with $\Psi$ is almost trivial, using nonexpansiveness of the prox. (Exploiting strong convexity of Psi to get linear convergence is less straightforward). This should be clearly explained as this is the main contribution of the work.

3) Can you prove linear convergence if $f$ or $\Psi$ is strongly convex?

### Soundness
3

### Presentation
3

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
This paper studies distributed **composite** optimization, $\min_x f(x)+\psi(x)$ with smooth $f$ and (possibly) non-smooth/prox-friendly $\psi$, under **compression with error feedback (EF)**. The authors combine **Dual Averaging (DA)** with **EControl** and analyze “virtual iterates,” claiming *“for the first time a strong convergence analysis for composite optimization with error feedback”* (Abstract; Sec. 1). They give an inexact-DA analysis template and experiments on synthetic $\ell_1$-regularized softmax and FashionMNIST with Top-$k$ compression, comparing against proximal EF and proximal EF21.

### Strengths
* **Clarity of Presentation:** The paper is well-written, logically structured, and easy to follow. The theoretical development is presented step-by-step, making the complex analysis accessible. The distinction between the real and virtual iterates, a key and often confusing point in EF literature, is handled with consistent clarity.
- **Conceptual:** Clear identification that naïve EF analyses break in the composite case; DA is a natural vehicle to handle the prox term. (Sec. 3–4 table of contents suggests an organized build-up.) 
- **Analysis template:** An inexact-DA framework and a sampling scheme for virtual iterates (Secs. 3, 3.1) that could be reusable beyond EControl. 
- **Mechanistic clarity:** A separate technical section on EControl’s properties (App. G, H), and a discussion of real vs virtual iterates (App. I). 
- **Experiments** provide both synthetic composite and a FashionMNIST $\ell_1$-regularized logistic example with Top-$k$ compression (δ=0.1), and include a *real vs virtual iterate* comparison. (Sec. 5; Fig. 1–2).

### Weaknesses
- **Overstated novelty (major):** The paper claims the first “strong convergence analysis for composite optimization with EF.” However, **composite EF** existed well before:
  - EC-ProxSGD and **EC-RDA (dual averaging)** handle composite finite-sum with error compensation. (OPT 2020).   
  - **EC-LSVRG/Quartz/SDCA** support composite settings with linear/convex guarantees (arXiv 2021).   
  - **EF-BV** (NeurIPS 2022) treats composite objectives with an explicit proximal step within a unified EF/variance-reduction theory.   
  - **Eco-FedSplit** (ICASSP 2022) integrates error-compensated compression with proximal FL splitting.   
  - **EF21-Prox** makes EF21 applicable to composite problems and states convergence (JMLR/ArXiv 2021).
  The manuscript must **qualify** its “first” claim (e.g., *first DA+EControl-based virtual-iterate rate under assumptions X, Y*), and clearly delineate what “strong” means vs prior composite-EF rates.

[1] Qian, Xun, et al. "Error compensated proximal SGD and RDA." _Proc. 12th Annu. Workshop Optim. Mach. Learn._. 2020.

[2] Qian, Xun, et al. "Error compensated loopless SVRG for distributed optimization." _Workshop on Optimization for Machine Learning_. 2020.

[3] Qian, Xun, et al. "Error compensated loopless svrg, quartz, and sdca for distributed optimization." _arXiv preprint arXiv:2109.10049_ (2021).

[4] Condat, Laurent, Kai Yi, and Peter Richtárik. "EF-BV: A unified theory of error feedback and variance reduction mechanisms for biased and unbiased compression in distributed optimization." _Advances in Neural Information Processing Systems_ 35 (2022): 17501-17514.

[5] Khirirat, Sarit, Sindri Magnússon, and Mikael Johansson. "Eco-fedsplit: Federated learning with error-compensated compression." _ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_. IEEE, 2022.

[6] Fatkhullin, Ilyas, et al. "EF21 with bells & whistles: Practical algorithmic extensions of modern error feedback." _arXiv preprint arXiv:2110.03294_ (2021).


- **Related-work coverage (major):** Sec. A emphasizes uncomposite EF and Safe-EF, but does **not** cite EC-ProxSGD/EC-RDA (OPT’20), EC-LSVRG/Quartz/SDCA (’21), EF-BV (NeurIPS’22), Eco-FedSplit (ICASSP’22). Add these and discuss differences in assumptions (e.g., convexity, finite-sum vs online), or the novelty claim is misleading.
- **Baselines (major):** Experiments compare to “proximal EF” and “proximal EF21,” but **omit** EC-ProxSGD/EC-RDA, EF-BV, and EC-LSVRG/Quartz/SDCA—precisely the most relevant **composite+EF** competitors. Include these, or justify why not.
- **Virtual vs real iterate gap:** Theory primarily addresses a sample of **virtual** iterates; experiments show similarity to **real** iterates on one setup. Provide bounds bridging real iterates (App. I) to the main claims, or broaden empirical validation.
- **Scope/assumptions clarity:** Precisely state smoothness/convexity/stochastic-noise assumptions used to obtain the claimed “strong” rate, and contrast to prior composite-EF (e.g., whether rates are stationarity vs suboptimality, and dependence on $\delta$, $n$, variance). The Abstract currently suggests generality that theorems may not cover. 
* **Gap Between Virtual and Real Iterate Guarantees:** The strong theoretical results, including linear speedup, are proven only for the *virtual iterates*. The analysis for the *real iterates* in Appendix I is substantially weaker, yielding a rate of $\mathcal{O}(1/(\delta^4 \epsilon^2))$ that lacks linear speedup and has a much worse dependency on the compression parameter $\delta$. The experiments show that real and virtual iterates perform similarly, which suggests the analysis for the real iterates is not tight. The proposed sampling procedure to recover the virtual iterate guarantee is impractical as it requires storing uncompressed cumulative gradients and an extra round of uncompressed communication, undermining the method's efficiency goals.

### Questions
1. Please **restate** the novelty claim precisely: relative to EC-RDA (dual averaging) and EC-ProxSGD (OPT’20), what is mathematically new in your DA+EControl analysis (assumptions, iterates, rate form)? How does “strong convergence” differ from prior composite-EF rates? Could the authors please clarify their novelty claim in light of these specific prior works? Why were these methods not cited or compared against?
2. Can you add **EC-ProxSGD/EC-RDA, EF-BV, and EC-LSVRG/Quartz/SDCA** as baselines—or explain incompatibilities?  
3. Is there a **theoretical link** from virtual to **real** iterates beyond empirical similarity (Fig. 1b)? Can you deliver a bound for the real iterate under your stepsize choices? 
4.  The analysis of real iterates in Appendix I is much weaker than the analysis of virtual iterates, yet experiments suggest their performance is nearly identical. What is the fundamental analytical barrier preventing a tighter analysis for the real iterates? Is there a specific step in the proof (e.g., handling the proximal operator directly) where the analysis loses tightness?
5.  Given that Qian et al. (2020) already analyzed an error-compensated RDA, could you elaborate on the specific contributions of your inexact dual averaging analysis template (Section 3)? Does it address limitations in the analysis of Qian et al. or offer a different perspective that enables the stronger (virtual iterate) result?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents a novel convergence analysis for distributed algorithms that incorporate error feedback mechanisms, referred to as **EControl**, to address **composite optimization problems**. The analysis is based on the **dual averaging framework**, which enables bounding the difference between the virtual and true iterates generated by the algorithms. This bound is a crucial step in demonstrating the advantages of error feedback, such as improved solution accuracy in algorithms using communication compression. 

The paper establishes the convergence guarantees for distributed proximal algorithms using **EControl** under standard assumptions, including **Lipschitz smoothness** of the objective function and **unbiased**, **variance-bounded** stochastic gradients. Moreover, the **iteration complexity** of these algorithms for composite (constrained) optimization problems matches that of their unconstrained counterparts.

Finally, empirical results validate the effectiveness of distributed proximal algorithms with **EControl**, showing that they achieve **linear speed-up** with respect to the number of participating agents.

### Strengths
This paper addresses an important gap in the study of **error-feedback algorithms** for distributed optimization by proposing novel analytical tools for handling **composite optimization problems** that involve the **proximal operator** under **communication constraints**. The authors effectively leverage the **dual averaging framework** to establish novel convergence analysis. In particular, Section 3 demonstrates how bounding the difference between the virtual and true iterates—a key aspect of analyzing error-feedback algorithms for unconstrained problems—can be extended to the composite setting.

The theoretical analysis is built upon standard and well-justified assumptions commonly used in gradient-based algorithms with communication compression, including:

1. **Convexity and Lipschitz smoothness** of the objective function  
2. **$\alpha$-contractive** compression operators  
3. **Unbiased** and **variance-bounded** stochastic gradients  

Extensive literature on error feedback and communication compression, which is closely related to this paper, has been included in **Appendix A**. 

The empirical evaluation is comprehensive. The proposed methods are tested on both synthetic data (softmax with $\ell_1$-regularization) and real data (logistic regression with $\ell_1$-regularization on FashionMNIST). Key findings include:
- The proposed algorithms support a significantly **larger step size** compared to baseline methods, which likely contributes to its superior convergence behavior.  
- The proposed algorithms exhibit **linear speed-up** with respect to the number of clients, confirming their scalability and efficiency in distributed settings.

### Weaknesses
Algorithm 2 requires access to $\bar{g}$ in Line 16 to execute, which is **impractical**, even though full gradient communication occasionally occurs in Line 4. The key concern is that existing error-feedback methods for unconstrained problems—such as **EControl** and **EF21**—do **not** depend on periodic full gradient communication, making Algorithm 2 comparatively less communication-efficient.

Moreover, the empirical results in this paper simulate stochastic gradients by adding Gaussian noise to the true gradients. A more realistic setup would be to construct the problems such that stochastic gradients are obtained through **minibatching** over the local training samples available to each client, rather than by artificially injecting noise. This modification would allow for evaluating the impact of minibatched training samples on the convergence behavior of the proposed algorithms.

### Questions
1. I believe Theorem 4.4 indicates that the iteration complexity of **EControl with Dual Averaging (Algorithm 2)** for composite problems matches that of the **vanilla EControl** method for quasi-convex unconstrained problems, as presented in *Gao et al. (2024a)*. Could the authors please verify this point? If confirmed, emphasizing this result would further strengthen the paper’s contribution by highlighting the efficiency of the proposed convergence analysis techniques.

2. $m$ is not defined on Pages 2 and 4. Could the authors clarify what $m$ represents?

3. On Page 6, the definition of $A_t$ is unclear. I assume it is given by $A_t = \sum_{\tau \leq t} a_\tau$. Please confirm or clarify this notation.

4. In Section 5, the authors mention that “we simulate the stochastic gradient by adding Gaussian noise to the gradients.” What is the **standard deviation** of the Gaussian noise used in these experiments? This information is missing from the main text.

5. **Assumption 2.5 vs. Assumption 3.1:**  The weaker condition in Assumption 2.5,  i.e. $\mathbb{E}\|\| g_i(x;\xi^i) - g_i(y;\xi^i) \|\|^2 \leq \ell^2 \|\| x - y \|\|^2$,  seems potentially redundant with Assumption 3.1. Is it possible to remove Assumption 3.1 from the analysis, or are both required for specific parts of the proof?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper studies Error Feedback (EF) methods for distributed composite optimization problems.
Existing EF analyses focus only on smooth (non-composite) settings, since the usual virtual-iterate technique used for EF fails when a non-smooth regularizer is present.
The authors address this issue by replacing standard gradient descent with a dual-averaging approach, which restores the additive structure needed for EF analysis.
By combining this idea with the EControl mechanism (a state-of-the-art EF variant), they obtain the first convergence guarantees for EF in the composite setting and achieve state-of-the-art iteration complexity.

### Strengths
1. The paper is well written and clearly organized.

2. The motivation is intuitive: the authors reformulate the EF method using dual averaging, which restores the additive structure (gradients plus accumulated compression error) and enables virtual-iterate analysis in the composite case.

3. The theoretical results are strong, with convergence rates matching the best-known results for the non-composite (smooth) setting.

### Weaknesses
1. It would be better if the paper provided more intuition about the **EControl** mechanism, since it plays a central role in the proposed method.  
2. There are several typos and minor presentation issues:  
   - Double $\coloneqq$ in Algorithm 1.  
   - $A_T$ and $\tau_t$ are not defined.  
   - $R_0$ is also not defined; it would be good to cite the work whose rate the paper matches in the non-composite case.  
   - Several displayed equations are missing commas or periods at the end.

### Questions
Is it possible to extend EF21 to the composite setting instead of EF? Did you try?

### Soundness
3

### Presentation
3

### Contribution
3
