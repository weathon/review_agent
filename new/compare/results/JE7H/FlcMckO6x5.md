---
job_id: 159e5e3d-d5fb-43ff-b080-75328043414d
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: FlcMckO6x5.pdf
paper: Separable Neural Networks: Approximation Theory, NTK Regime, and Preconditioned Gradient Descent
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper develops theory and optimization methods for a specific neural architecture (separable neural networks), including NTK analysis and preconditioned gradient descent, which fits squarely within ICLR’s scope on learning theory, optimization, and representation learning, with applications to INRs and PINNs.

## Minimum Quality
Pass ✅.  
The paper is in English and includes Abstract, Introduction, theoretical methodology (approximation, NTK, SepPGD), experiments, and Conclusions/Discussions. The contributions are non‑trivial, proofs are detailed, and experiments cover several tasks. I do not see a fundamental methodological or evaluation flaw that would justify immediate desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not find any hidden instructions or attempts to manipulate automated reviewing systems in the provided content.

---

# Expected Review Outcome:

## Summary

The paper studies separable neural networks (SepNNs), which factor multivariate functions into linear combinations of univariate neural networks combined via tensor decompositions (CP, TT, Tucker).  

First, it proves a universal approximation theorem for multivariate SepNNs, using Stone–Weierstrass plus standard universal approximation, covering CP/TT/Tucker variants. Second, it derives NTK regimes for CP‑type SepNNs: with both width and rank going to infinity, the NTK converges to a deterministic kernel; with infinite width but fixed rank, it converges to a random kernel governed by Gaussian processes. Third, it proposes a separable preconditioned gradient descent (SepPGD) that applies NTK-based preconditioning at the factor level, achieving an $\mathcal{O}(nD)$ per‑iteration complexity on $n^D$ grid samples and empirically faster convergence and reduced spectral bias on KRR, INR image/surface representation, and grid‑based PINNs.

---

## Strengths

1. **Clean and fairly general approximation result for SepNNs.**  
   The universal approximation theorem in **Theorem 1** (Pages 3–4, detailed proof Pages 19–25) is technically solid and conceptually satisfying: it shows that CP, TT, and Tucker SepNNs can approximate any continuous function on a compact subset of $\mathbb{R}^D$. The use of Stone–Weierstrass plus a vector‑valued extension of standard universal approximation is a neat and unified proof technique, much simpler and more general than prior bivariate, sine‑specific arguments. The detailed closure/separation proofs (Pages 19–22) show real care.

2. **NTK regime characterization for separable architectures.**  
   Lemma 1 and **Equation (4)** provide an interpretable decomposition of the SepNN’s NTK as a sum over factor‑NTKs weighted by products of other factors, which is a natural and useful structural insight. **Theorem 2** (Page 5) and **Corollary 1** (Page 6) carefully distinguish between the “infinite width & infinite rank” deterministic NTK regime and the “infinite width & fixed rank” random NTK regime, connecting to classical NNGP/NTK theory but in a genuinely separable setting. The derivations in Sections A.6–A.8 are worked out in detail, including the diagonalization and law‑of‑large‑numbers arguments.

3. **Conceptually interesting link between separability and efficient NTK computation.**  
   **Lemma 3** and **Equations (9)–(10)** (Pages 18–19) show that, on grid inputs, the SepNN NTK matrix decomposes into Kronecker products of $n\times n$ matrices, yielding efficient exact or pseudo‑NTK computation. This is a genuine architectural advantage over standard MLPs; it also underpins the efficiency of SepPGD. The connection between tensor decompositions and Kronecker‑structured kernels is explicitly and clearly exploited.

4. **Separable PGD is well‑motivated and tied to spectrum control.**  
   The design of SepPGD in **Definition 1** and **Equations (7)–(8)** (Page 8) is consistent with the NTK analysis: factor preconditioners $\{S_d\}$ are built by eigenvalue modulation on factor NTKs (via sum‑of‑logits pseudo‑NTKs), and **Lemma 2** (Page 9) establishes an algebraic equivalence, in the bivariate grid case, between SepPGD and a full NTK-based PGD with a Kronecker‑structured preconditioner $\hat S = S_1\otimes I + I\otimes S_2$. This is a nice piece of linear‑algebra engineering that justifies the spectral‑bias claims.

5. **Real computational advantage and thorough complexity discussion.**  
   **Table 1** (Page 8) compares the complexity of Hessian-based methods, full NTK spectrum modification, mini‑batch NTK preconditioning, and SepPGD. The scaling reduction from $O(n^{D})$ (or $O(n^{3D})$ for constructing a full NTK preconditioner) to $O(nD)$ and $O(D(n^3 + n^2P))$ is consistent with the separable structure and is non‑trivial for practice, particularly on large INRs/PINNs grids.

6. **Empirical support across diverse tasks and good use of figures.**  
   The experiments cover kernel ridge regression, image INRs, surface representation, and 3D PDEs with PINNs.  
   - **Figure 1** (Page 5) corroborates the NTK theory: (a) shows lack of convergence to a deterministic NTK with fixed rank, (b) shows convergence as width and rank jointly grow, (c) shows NTK staying nearly fixed during training, and (d) shows the steep eigenvalue decay that motivates spectral bias and preconditioning.  
   - **Figure 2** (Page 9) and **Figure 6** / **Figure 7** in the appendix show that, for KRR and image INRs, SepPGD consistently improves convergence in wall‑clock time over vanilla SepNN and NTK‑MSK variants, in both noiseless and noisy settings.  
   - **Figure 3** (Page 10) and **Figure 8–12** display sharper image details and higher IoU for surface representation with SepPGD vs. baselines.  
   - **Figure 4**, **Figure 13**, and **Figure 14** demonstrate similar acceleration for separable PINNs on 3D diffusion, Klein–Gordon, and Helmholtz equations.

7. **Useful ablation studies and hyperparameter sensitivity.**  
   The appendix tables (Tables 2–8, Pages 16–17) explore modulation functions $g(\lambda)$, number of eigenvalues $k$, rank $R$, width, preconditioner update frequency, activation functions, and noise robustness. These ablations show that SepPGD’s benefit is quite robust to $k$ and update frequency, and explain where it overfits high‑frequency noise, which is important for practical use.

---

## Weaknesses

1. **Approximation theory is qualitative only; no rates or rank/width dependence.**  
   While **Theorem 1** convincingly establishes denseness, it is purely existential. There is no guidance on how the required rank $(R$ or $(R_d))$ or width $W$ scales with function smoothness, intrinsic dimensionality, or target accuracy $\epsilon$. The proof in Section A.5 ultimately chooses $\delta=\epsilon/(2RDM^{D-1})$ etc., but does not track how $R$ itself must grow with properties of $f$. For a class of architectures whose main selling point is efficiency via low ranks, the lack of any quantitative relation between approximation error and rank/width is a significant gap: in practice, users must choose $R$, and the paper gives no theoretical insight into how large $R$ needs to be even for simple function classes (e.g., bandlimited or Sobolev functions). This limits the scientific value of the approximation result beyond “universal approximator” rhetoric, which the field already has in abundance.

2. **NTK analysis is restricted and somewhat classical; limited new insights beyond structure.**  
   The NTK results are largely a separable restatement of known phenomena: **Theorem 2** recovers the usual deterministic NTK in the double limit $W\to\infty$, $R\to\infty$, with a separable kernel $\sum_d k(x_d,x_d')\prod_{d'\neq d}c_{d'}(x_{d'},x'_{d'})$; **Theorem 3** (Appendix A.4) re‑derives “NTK remains fixed during training” with variance bounds of order $O(t/R)+O(t/W)$; **Corollary 1** mirrors the standard NNGP/NTK story under fixed rank. The main novelty is the explicit dependence on the separable architecture via the $a_d(\cdot)$ weights in **Equation (4)**, and the Kronecker structure in **Lemma 3**, which are indeed valuable. However, the paper does not exploit these to derive any new qualitative behavior specific to SepNNs (e.g., how separability changes eigenvalue decay, implicit regularization, or generalization beyond the generic “spectral bias” story that applies to any kernel). As written, this section risks being perceived as “SepNN = generic NTK + algebra” rather than uncovering genuinely new training phenomena.

3. **Spectral‑bias alleviation theory for SepPGD stops at plausibility, not rigorous guarantees.**  
   On Page 9, the argument that $K\hat S$ “has better spectrum (i.e., smaller condition number) than $K$” is heuristic. It uses the fact that $S_d$ improves the spectrum of $K_{\Theta_d}$ and that $\hat S = S_1\otimes I + I\otimes S_2$ inherits good eigenvalues, then informally asserts that if $\hat K = K_{\Theta_1}\otimes I + I\otimes K_{\Theta_2}$ is close to the “true” NTK $K$, then $K\hat S$ is better conditioned. There is no theorem, no explicit bound on the condition number of $K\hat S$, and no quantitative convergence‑rate statement for gradient descent under SepPGD. Given that the claimed contribution is to “provably adjust the eigenvalue distribution” and “effectively alleviate spectral bias”, the absence of a formal result here is a noticeable gap. At minimum, a precise statement along the lines of “if $\|K-\hat K\| \le \delta$ and $S_d$ satisfy X, then $\kappa(K\hat S) \le c \kappa(K)$” would significantly strengthen the story.

4. **SepPGD’s conceptual novelty is moderate; heavily relies on prior NTK‑PGD ideas.**  
   Algorithmically, SepPGD is essentially an efficient factorized implementation of existing NTK‑based preconditioning methods (Geifman et al., 2024; Shi et al., 2025). **Lemma 2** explicitly proves equivalence in the $D=2$ grid case: SepPGD corresponds to classical PGD with a Kronecker structured preconditioner $\hat S$. This is a nice computational insight, but it means the method’s *optimization dynamics* are not new, only the exploitation of separability is. The paper is quite honest about this on Page 9 (“equivalent to the classical NTK‑based PGD”), but the abstract and introduction might slightly oversell the conceptual novelty by framing SepPGD as a “new method” rather than as an efficient separable instantiation of existing PGD techniques. For an ICLR‑level optimization contribution, one might expect either a new preconditioning principle or a deeper theoretical analysis beyond algebraic equivalence.

5. **Experimental evaluation, while broad, lacks stronger baselines and statistical rigor.**  
   Most experiments compare SepPGD only to: vanilla MLP, vanilla SepNN, and NTK‑based MSK/PGD on small‑scale problems, mostly with full‑batch training. There is no comparison to other “spectral-bias‑alleviating” methods such as curriculum learning, Fourier features plus more aggressive regularization, or newer INR/PINN architectures like KAN-based INRs or non‑separable tensorial fields that might already reduce spectral bias. Results are mostly shown as single curves without confidence intervals; for example, **Figure 2** and **Figure 4** show convincing but purely qualitative gaps. For image INRs, the baselines are rather classical (SIREN MLP, SIREN SepNN), and there is no comparison with modern high‑performance INRs or kernel‑based methods. This makes it harder to assess the practical significance of SepPGD beyond “it improves my own baseline”.

6. **Potential issues / ambiguities in complexity claims and Table 1.**  
   In **Table 1**, the complexity entries are somewhat confusing. The row “Modified NTK spectrum (Geifman et al., 2024)” lists complexity $O(nD)$ with the condition “$nD < P$”, while the main text on Page 7 explains that the classical NTK-based method scales as $O(n^D)$ for $n^D$ samples. This inconsistency should be fixed; presumably the table intended $O(n^D)$, or it conflates the *number of parameters* $P$ with the *sample dimension* $n^D$. Similarly, in **Remark 4**, the eigen‑decomposition of an $n\times n$ matrix is said to be $O(n^3)^3$, which seems like a typo. Since complexity is claimed as a core advantage, these inaccuracies matter and should be clarified carefully (ideally with explicit constants and memory considerations).

7. **Some mathematical details could be tightened or clarified.**  
   - **Equation (4)** expresses the NTK as a sum of bilinear forms in $a_d(\cdot)$ and $K_{\Theta_d}$. This derivation assumes independence and certain scaling; while Appendix A.6 outlines the steps, it would help to explicitly state any assumptions on the initialization across $d$ and $r$ and clarify the role of the $1/\sqrt{R}$ prefactor in ensuring variance $O(1)$.  
   - In **Theorem 2** and its proof (Appendix A.7), the order of limits $W\to\infty$, $R\to\infty$ and the almost sure convergence are argued via separate law‑of‑large‑numbers for $W$ and $R$. A more formal justification of exchanging expectations and limits, especially given products of random variables across $d$ and $r$, would improve rigor.  
   - In **Corollary 1**, the definition of $V_d(\boldsymbol{x},\boldsymbol{x}')$ has a typo: the product includes $(f_{\Theta_{d'}}(x_{d'}') )_{r'}$ but $r'$ is undefined; presumably it should be the same $r$.  

8. **Limited discussion of limitations of NTK/linearization regime and generalization.**  
   The paper relies heavily on NTK theory, but does not really engage with the known limitations of linearized models in explaining generalization or feature learning, especially for finite‑width and finite‑rank SepNNs that are of most practical interest. For example, the fixed‑rank random NTK regime in **Corollary 1** could be linked to random features theory or to known cases where linearization mispredicts behavior, yet this is only briefly mentioned in Appendix A.1.2. A more critical discussion would contextualize the spectral‑bias story and clarify what aspects are truly predicted by NTK and what might require non‑linear analysis.

9. **Related work on approximation and preconditioned GD is incomplete.**  
   The paper cites core NTK, INRs, PINNs, and tensor‑decomposition works, but omits several directly relevant recent papers on approximation theory under gradient flow and preconditioned gradient descent (see “Potentially Missing Related Work” below). This weakens the positioning of the contributions, especially the claim that SepPGD is the main way to control spectral bias in this family of models.

10. **Minor presentation issues.**  
    There are noticeable glitches in the reference list (e.g., the repeated “Shih, Shih, and Shih…” line on Page 13), and several typos and formatting issues (e.g., stray “^3” in complexity discussion, inconsistent use of bold vs. math italic). None of these are fatal but they do detract from the overall polish.

---

## Potentially Missing Related Work

1. **Welper, G., “Approximation Results for Gradient Descent trained Neural Networks,” 2023.**  
   This work studies approximation properties of networks trained by gradient descent, complementing purely representational universal approximation theorems. It is directly relevant to bridging Theorem 1 and the actual GD dynamics considered in the NTK analysis. It should be discussed in the approximation section (Section 2) and in Appendix A.1.2 when talking about future directions on error rates under training.

2. **Welper, G., “Approximation and Gradient Descent Training with Neural Networks,” 2024.**  
   Extends approximation theory for networks under GD, focusing on compatibility between function approximation and training. This is closely related to the paper’s goal of understanding SepNNs’ approximation and optimization together; it should be cited in Section 2 and in the discussions of NTK regimes (Section 3).

3. **Yang, Y., “Preconditioned Gradient Descent Finds Over-Parameterized Neural Networks with Sharp Generalization for Nonparametric Regression,” 2024.**  
   Proposes preconditioned GD with theoretical generalization guarantees in over‑parameterized nets. Given that SepPGD is another instance of NTK‑based preconditioning, this paper is highly relevant and should be compared in Section 4, especially regarding generalization and the effect of spectrum modification.

4. **Mahankali, A. V., HaoChen, J. Z., Dong, K., “Beyond NTK with Vanilla Gradient Descent: A Mean-Field Analysis of Neural Networks with Polynomial Width, Samples, and Time,” 2023.**  
   Analyzes training dynamics beyond the NTK regime. Since the current NTK analysis for SepNNs is strictly in the linearized regime, this work is important context and should be mentioned in Section 3 and Appendix A.1.2 when discussing the limitations of NTK and potential extensions.

5. **Ortiz-Jimenez, G., Moosavi-Dezfooli, S.-M., Frossard, P., “What can linearized neural networks actually say about generalization?,” 2021.**  
   Critically evaluates the explanatory power of NTK/linearized models for generalization. The paper’s heavy reliance on NTK to discuss spectral bias and convergence would benefit from explicitly acknowledging these limitations in Section 3 or A.1.2.

6. **Bai, Y., Lee, J. D., “Beyond Linearization: On Quadratic and Higher-Order Approximation of Wide Neural Networks,” 2019.**  
   Provides higher‑order analyses of wide nets beyond NTK. This would strengthen the discussion in Appendix A.1.2 about studying SepNNs beyond the convergent NTK regime.

7. **Bombari, S., Amani, M. H., Mondelli, M., “Memorization and Optimization in Deep Neural Networks with Minimum Over-parameterization,” 2022.**  
   Studies NTK spectrum and optimization for minimally over‑parameterized networks. Since SepNNs often use relatively small ranks in practice, this work is relevant to Section 3’s discussion of fixed‑rank regimes and spectral bias.

8. **Ji, Z., Telgarsky, M., Xian, R., “Neural tangent kernels, transportation mappings, and universal approximation,” 2019.**  
   Connects NTK to universal approximation rates. It is relevant to bridging Theorem 1 (approximation) and Section 3 (NTK dynamics), and could inform future work on rate bounds.

9. **Avidan, Y., Li, Q., Sompolinsky, H., “Connecting NTK and NNGP: A Unified Theoretical Framework for Neural Network Learning Dynamics in the Kernel Regime,” 2023.**  
   Offers a unifying view of NTK and NNGP learning dynamics. Since the paper separately discusses deterministic NTK and random NTK (fixed rank), this work should be cited in Section 3 and Appendix A.1.2.

10. **Munteanu, A., Omlor, S., “NTK with Convex Two-Layer ReLU Networks,” 2025.**  
    Analyzes NTK under a convex formulation. The techniques and conclusions could be relevant for SepNNs with ReLU factors and might be worth mentioning in the NTK section and in the discussion of potential extensions to other activations.

---

## Questions

1. **Quantitative approximation rates and rank dependence.**  
   Is it possible, even for a restricted function class like bandlimited or Sobolev functions on $[0,1]^D$, to derive an approximation error bound of the form  
   \[
   \inf_{\text{SepNN with rank }R} \|f - f_\Theta\|_\infty \le C(f,D)\,\phi(R)
   \]  
   with an explicit decay $\phi(R)$? Even a rough rate (e.g., polynomial or exponential in $R$) would make Theorem 1 much more actionable.

2. **Formal spectrum improvement for SepPGD.**  
   Can you provide a precise theorem or at least a proposition quantifying how the condition number of $K\hat S$ compares to $K$ under reasonable assumptions (e.g., $\hat K$ approximates $K$ within $\delta$ in operator norm, $S_d$ satisfy certain spectral flattening properties)? Even a simplified statement for $D=2$ in the Kronecker case would significantly strengthen Section 4.

3. **Effect of fixed rank on training dynamics.**  
   In practice, many experiments seem to use relatively modest ranks (e.g., $R=100$–$500$). Based on **Corollary 1** and the random NTK view, can you comment more concretely on when the deterministic NTK approximation is accurate for these finite ranks? Have you tried empirically checking NTK stability and spectrum vs. $R$ beyond what is shown in **Figure 1**, perhaps relating convergence speed to empirical eigenvalue decay?

4. **Baselines for spectral bias alleviation.**  
   For INRs and PINNs, have you compared SepPGD against other practical strategies to reduce spectral bias, such as frequency‑annealed Fourier features, positional encodings with adaptive bandwidth, or curriculum learning on frequencies / residuals? This would help clarify whether SepPGD is competitive against the most effective existing techniques, not just against vanilla training.

5. **Clarifying complexity claims in Table 1.**  
   Please clarify the apparent mismatch between the text (which says classical NTK‑based PGD is $O(n^D)$ in the number of samples) and **Table 1**, where the complexity entry for “Modified NTK spectrum” is given as $O(nD)$ with the condition $nD < P$. Is this a typo, or are you measuring a different quantity? A careful derivation and consistent notation would avoid confusion.

6. **Applicability beyond grid inputs.**  
   Appendix A.2 briefly discusses non‑grid inputs with an einsum‑based formulation. Could you expand (possibly in the main paper) on how the complexity and practical benefits of SepPGD change in the non‑grid case, where you lose exact Kronecker structure? Is SepPGD still clearly advantageous over standard NTK PGD, or does its benefit vanish?

---

## Flag For Ethics Review

- No ethics review needed.  

---

## Details Of Ethics Concerns

N/A.

---

## Soundness Rating

3: good.  
The core theorems (universal approximation, NTK regimes, NTK staying fixed during training) appear technically sound with detailed proofs; SepPGD’s algebraic equivalence to classical NTK PGD is convincing. However, there are no formal convergence‑rate or spectrum‑improvement guarantees for SepPGD, and some complexity claims need clarification.

---

## Presentation Rating

3: good.  
The paper is dense but generally well organized, with extensive appendices and clear notation in the main text. Figures and tables (e.g., **Figure 1**, **Figure 2**, **Table 1**, and Tables 2–8) are used effectively. Some reference glitches, typos, and minor inconsistencies in complexity expressions should be fixed.

---

## Contribution Rating

3: good.  
The combination of a general universal approximation theorem for SepNNs, an NTK characterization tailored to separable architectures, and an efficient Kronecker‑structured implementation of NTK‑based PGD for SepNNs is a solid and useful contribution. The conceptual novelty of SepPGD is moderate, and the theory could go deeper, but the work is still clearly valuable for the INR/PINN and separable‑model communities.

---

## Overall Rating

6: Marginally above the acceptance threshold. But would not mind if paper is rejected.  

The paper offers a solid mix of theory and algorithmic development for an increasingly relevant class of architectures (SepNNs), with non‑trivial use of classical tools (Stone–Weierstrass, NTK/NNGP) and a well‑engineered separable PGD that is computationally attractive and empirically effective. The main weaknesses are the lack of quantitative approximation or convergence rates, the largely heuristic treatment of spectrum improvement, and somewhat limited baselines. Overall, I view this as a good poster‑level contribution: clearly above the bar if the authors can tighten some theoretical claims and polish the presentation, but not at the level of a highlight.

---

## Reviewer Confidence

4: confident.  
I am familiar with NTK theory, kernel methods, tensor decompositions, and INRs/PINNs, and I checked the main derivations and experimental designs. Some details (e.g., fine‑grained spectrum analysis) would benefit from further author clarification, but my overall assessment is unlikely to change dramatically.