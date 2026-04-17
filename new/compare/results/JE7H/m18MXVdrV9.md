---
job_id: f996af16-174c-497c-83cc-a7ccfbaea465
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: m18MXVdrV9.pdf
paper: Information Estimation With Discrete Diffusion
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper proposes a new mutual information / KL estimator based on discrete diffusion (CTMCs) with applications to text and genomics, clearly fitting ICLR topics in generative models, probabilistic methods, and information-theoretic representation learning.

## Minimum Quality
Pass ✅.  
All core sections are present (Abstract, Introduction, Methodology, Experiments, Conclusion). The work is technically substantial, written in English, and provides extensive experiments and theory. I do not see any fundamental methodological flaw that would justify immediate desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not find any hidden prompts, non-scientific instructions, or other manipulative content in the main paper text.

---

# Expected Review Outcome:

## Summary

The paper introduces **info-sedd**, an estimator of KL divergences, mutual information, and entropy for **high-dimensional discrete data**, built on continuous-time Markov chain (CTMC)–based discrete diffusion. Using Dynkin’s formula, the authors derive an integral expression for KL (Eq. (4)) that can be approximated via parametric score functions trained with the DWDSE loss. They propose joint and conditional MI estimators (info-sedd-j / -c), show a finite-time error bound (Eq. (7)), and demonstrate strong empirical performance on synthetic MI benchmarks, text summarization, genomics (classification and motif discovery), and Ising entropy estimation.

## Strengths

1. **Conceptual contribution: KL via discrete diffusion + Dynkin formula**  
   - Section 2.2 derives an expression for \(\mathrm{KL}[p_0\|q_0]\) for discrete CTMCs (Eq. (4)), starting from time-reversed dynamics and Dynkin’s formula (Eq. (3)).  
   - The final estimator (Eq. (5)) expresses KL as an expectation over the forward process, with ratios \(p_t(x)/p_t(X_t)\) and \(q_t(x)/q_t(X_t)\) approximated by score networks.  
   - This is a clean and fairly self-contained construction that extends recent discrete diffusion modeling (Lou et al., 2024; Sun et al., 2023) toward **information estimation**, rather than just generative modeling.

2. **Single-model trick via absorbing diffusion is elegant and practically important**  
   - Section 3 shows that choosing a token-level absorbing rate matrix \(Q_t^{\text{tok}}=\sigma(t)Q^{\text{tok}}_{\text{absorb}}\) lets one recover marginal score ratios from a model trained only on the joint, yielding Eq. (6).  
   - The proof in Appendix A.3 (Eq. (10)) is straightforward but powerful: using the probability that the Y-component has jumped to \(\emptyset\), the ratio \(\frac{p_t(X_t=x, Y_t=\emptyset)}{p_t(X_t=\bar{x},Y_t=\emptyset)}\) collapses to the marginal ratio \(\frac{p_t^X(x)}{p_t^X(\bar{x})}\).  
   - This makes **info-sedd-j** much more scalable and reuse-friendly, since only one discrete diffusion model on the joint is needed to estimate both joint and marginal scores.

3. **Theoretical error analysis is nontrivial and reasonably thorough**  
   - Section 3 and Appendix E derive the bound (Eq. (7)):
     \[
       \Big|\mathbb{E}_{x\sim p}\mathcal{E}(s_\theta^p, s_\phi^q; x) - \mathrm{KL}[p\|q]\Big|
       \leq \underbrace{\bar{\sigma}(T)D|\chi|\big(1+\frac{C_2}{C_1}\big)(\epsilon_p+\epsilon_q)}_{\text{estimation error}}
       + \underbrace{(1-\vec{p}_T(\emptyset^D)) (D C_2\log|\chi|)}_{\text{truncation bias}}
     \]
   - The proof in Appendix E carefully propagates boundedness assumptions on score functions (A.1–A.2) through the functional \(\mathcal{E}\) (Eq. (39)) and bounds both the approximation error and finite-time bias.  
   - The discussion around consistency in E.3 (invoking convergence of DWDSE-trained scores and the Continuous Mapping Theorem) is sober and avoids overclaiming; it clearly states that the estimator is consistent **up to exponentially decaying truncation bias**.

4. **Strong synthetic results in the regime where neural MI estimators are known to struggle**  
   - **Table 1 (Page 6)** is a key result: for MI = 10, 20, 30, 40, 50 with matching dimension \(D\), INFO-SEDD is essentially on target (e.g., \(20.02\pm0.21\) for MI=20, \(39.11\pm0.65\) for MI=40) while GAN-DIME, HD-DIME, KL-DIME, MINE, NWJ, SMILE all either **saturate or under-/over-shoot badly** as MI and dimensionality grow.  
   - The standard deviations are small, indicating stable training.  
   - Additional tables (Tables 3–6 in Appendix C.1) show that INFO-SEDD is competitive or better in memory (Table 3), runtime (Table 4), and is robust to varying sample sizes and support size \(|\chi|\) (Tables 6–7).  
   - **Figure 6 (Page 25)** clearly illustrates faster convergence in MI estimates over training epochs for INFO-SEDD compared to other estimators.

5. **Compelling real-data applications, especially the discrete nature of the method is actually used**  
   - **Text summarization (Section 4.2):**  
     - INFO-SEDD is applied directly on token sequences using a discrete diffusion language model backbone (MDLM-small), without mapping texts to continuous embeddings.  
     - **Figure 1** (Page 7) plots MI vs \(\rho\) in the shuffling experiment. Both INFO-SEDD-C and -J follow the theoretically motivated linear trend (grey "empirical MI estimate" band), while variational baselines remain stuck in low-MI regimes due to batch-size limitations.  
     - **Table 2** (Page 8) shows correlations between MI estimates and human metrics on SUMMEVAL. INFO-SEDD-C reaches Pearson 0.74 and Kendall 0.505 with **consistency**, higher than all other MI estimators, which supports the claim that MI can serve as a meaningful signal for summarization model selection. **Figures 2 and 3** (Pages 7–8) further visualize that consistency saturates but MI continues to differentiate models beyond this ceiling.
   - **Genomics (Section 4.3):**  
     - **Figure 4** (Page 9) displays the \(\rho\)-consistency test on the HUMAN VS. WORM dataset. INFO-SEDD-C closely tracks the classifier-based MI reference curve, while competitors (HD-DIME, SMILE, GAN-DIME) deviate more, especially as \(\rho\to 1\).  
     - **Figure 5** shows a sliding-window MI profile over Arabidopsis thaliana promoter sequences. The MI peak aligns well with the biologically known TATA-box region (-39 to -26 relative to TSS), and the coloring encodes TATA-box overlap. This is a convincing example of MI-based motif discovery that leverages the ability of info-sedd-j to compute MI for arbitrary masked subsets without retraining.
   - **Ising model entropy (Appendix D.1, Figure 11):** INFO-SEDD-H matches analytical entropy per-site curves over a wide temperature range, including the low-temperature regime where KL is large and variational MI estimators typically fail.

6. **Good attention to computational aspects and scalability**  
   - Section 3 exploits the structured decomposition \(X=[X_1,\dots,X_D]\) and sparse CTMC updates with unit Hamming distance to reduce complexity of \(Q_t\) from \(|\chi|^{2D}\) to \(O(D|\chi|^2)\), then further effectively \(O(D|\chi|)\) in the estimator since only neighbors are enumerated.  
   - Tables 3 and 4 report **peak memory** and **per-epoch runtime** for the high-dimensional synthetic tasks, showing that INFO-SEDD uses significantly less memory and is faster than GAN-DIME / HD-DIME / KL-DIME at large D.  
   - The ability to plug into pretrained MDLM (text) and Caduceus (DNA) backbones without architectural surgery is practically attractive.

7. **Clarity of mathematical exposition and proofs**  
   - The derivation of Eq. (4) in Appendix A.1 is detailed and walks through the key steps: decomposing \(\mathbb{E}[\log \overleftarrow{p}_T|\overleftarrow{X}_0]\), computing \(\partial_t \log \overleftarrow{p}_t\) via \(\overleftarrow{Q}_t \overleftarrow{p}_t / \overleftarrow{p}_t\), using the explicit reverse generator (Eq. (1)), and combining terms into the function \(K(\alpha)=\alpha(\log\alpha-1)\).  
   - The proof of Eq. (6) in Appendix A.3 is particularly clean and makes the "marginal from joint with absorbing process" property transparent.  
   - The pseudo-code in Algorithms 1–3 (Appendix B) makes the estimator concrete and should be directly implementable.

## Weaknesses

1. **Some mathematical steps are opaque or rely on strong, under-discussed assumptions**  
   - The error bound in Eq. (7) (and Appendix E) hinges on boundedness assumptions A.1–A.2 on both true scores \(s^p,s^q\) and network approximations \(s^p_\theta, s^q_\phi\), with constants \(0<C_1<C_2<\infty\). In realistic high-dimensional discrete distributions (e.g., language), ratios \(\frac{p_t(\hat{x})}{p_t(x_t)}\) can be extremely small or extremely large, so the assumption \(C_1 \le \|s^p\|_\infty\) is nontrivial.  
   - The proof effectively enforces \(\min_x s^p(x)\ge C_1>0\), which translates to a **uniform lower bound on all conditional probabilities and probability ratios** considered in \(\mathcal{S}\). This seems unlikely to hold even approximately in large-vocabulary, long-sequence regimes (rare tokens, rare configurations).  
   - While the authors cite Ren et al. (2024) and Chen & Ying (2024) for similar assumptions, they do not discuss **how sensitive Eq. (7) is to violations of boundedness**, nor whether clipping or regularization is used in practice to enforce these bounds. This matters, because constants like \(C_2/C_1\) appear multiplicatively and could blow up the bound. A short empirical sanity check (e.g., observed range of scores during training on MDLM / Caduceus backbones) would make the theoretical section much more meaningful.

2. **Estimator complexity and practicality at large vocabulary / sequence length not fully unpacked**  
   - Although Section 3 notes the use of sparse \(Q\) and absorbing token-level transitions, Algorithms 1 and 2 require, for each position \(i\) where a token is \(\emptyset\), a loop over all \(n\in[1:N]\) (where \(N=|\chi|\)). This leads to per-sample cost proportional to the number of masked positions times vocabulary size.  
   - For SUMMEVAL-like setups, where the MDLM-small model has a nontrivial vocabulary (likely tens of thousands of tokens), it is not clear how this is implemented efficiently. Are they using full vocabulary enumeration or a sampled subset of neighbors in \(|\chi|\)? The text only mentions "we consider sequences which only differ in one component" but not how \(|\chi|\) is handled when large.  
   - Without this detail, it is hard to assess whether INFO-SEDD remains practical for modern LLM-scale vocabularies or only for relatively small |\(\chi\)| (e.g., 4 in DNA, 2 in Ising). A clear complexity analysis with concrete numbers for vocabulary size and runtime in the text summarization experiments would strengthen the scalability claim.

3. **Some confusion / possible typos in algorithmic pseudo-code**  
   - **Algorithm 2 (INFO-SEDD-C)** on Page 22 appears to mix variables in a way that is at least confusing and possibly incorrect:  
     - Step 2 perturbs only \(Y_t\sim p_t(\cdot|Y_0)\), as expected for a conditional MI estimator \(I(X;Y) = \mathbb{E}_X[\mathrm{KL}(p_{Y|X}\|p_Y)]\).  
     - In Step 7, however, the third term uses \(s_\theta([\vec{X}_t,\vec{Y}_0])_{[\tilde{X},\vec{Y}_0]}\), which involves \(\vec{X}_t\) and \(\vec{Y}_0\), whereas Steps 6–7 above are defined on \([\vec{X}_0,\vec{Y}_t]\). This mismatch between \(X_0, X_t, Y_0, Y_t\) in the indices is nontrivial and makes it unclear what is actually being scored.  
   - While this might be "just a typo" and the implementation in the released code might be correct, from the paper alone the conditional estimator is under-specified and somewhat hard to reconstruct from Eq. (5). Given INFO-SEDD-C is central in the strongest empirical story (summarization and genomics), a correct, internally consistent pseudo-code is important.

4. **Comparisons may be somewhat unfairly tilted toward info-sedd in discrete domains**  
   - All competing neural MI estimators (MINE, SMILE, DIME variants, MINDE) are originally designed for continuous representations and are used via the "embedding trick". The paper is upfront about this, but then draws fairly strong conclusions about superiority on discrete data.  
   - For text summarization (Section 4.2), competitors get a learned embedding table + shallow MLP or similar, while INFO-SEDD leverages a discrete diffusion language model (MDLM-small) pre-trained on OpenWebText. It is not fully clear whether the same level of pretraining is given to baselines; the text says "we use the MDLM-SMALL model as the backbone, with minimal changes... for all methods" but for MINDE and DIME estimators this is more naturally a continuous latent backbone rather than a discrete masked LM.  
   - This setup is reasonable if the question is "given a strong discrete diffusion backbone, what is the best way to estimate MI?", but it does blur the line between comparing **architectures** and **estimation principles**. At minimum, the paper should be explicit that INFO-SEDD is advantaged by being architecturally aligned with the discrete setting, whereas baselines are forced into a less natural embedding-based configuration.

5. **Heuristic “ground truth” MI curves in consistency tests are only loosely justified**  
   - In the text summarization consistency test, the "empirical MI estimate" plotted in **Figure 1** is derived by multiplying entropy rates from Takahira et al. (2016) and Cover & King (1978) with average sequence length, then assuming a linear \(I(X;Y^{\rho})\approx \rho I(X;Y)\) relationship when MI is "significantly larger than \(\log 2\)". This is a coarse heuristic; it is used as a quasi-ground-truth baseline to claim INFO-SEDD “matches the empirical derivation”.  
   - Similarly, the DNA consistency reference in **Figure 4** approximates \(I(X;Y^{\rho})\) by assuming nearly constant classifier accuracy and using binary entropy (Appendix C.4.1), which again is a heuristic model.  
   - These approximations are not invalid, but the paper could be more cautious framing them as **sanity-check bounds** rather than "empirical MI estimates". The linear-in-\(\rho\) behavior could be tested more robustly by synthetic experiments with known MI on text-like and DNA-like data, or by at least varying key parameters (e.g., entropy-rate assumptions, classifier accuracy) to show that INFO-SEDD remains consistent across these choices.

6. **Limited exploration of truncation time \(T\) and bias–variance trade-offs**  
   - Eq. (7) and the discussion in Appendix E.3 emphasize the trade-off between truncation bias \((1-\vec{p}_T(\emptyset^D))\) and the linear dependence of estimation error on \(\bar{\sigma}(T)\). However, there is essentially no **empirical** study of how changing \(T\) or \(\sigma(t)\) affects MI estimates, bias, or variance in practice.  
   - For example, on the synthetic benchmark in Table 1 or the Ising experiments, plotting MI estimates vs \(T\) (for fixed training) would demonstrate that the exponentially decaying bias is actually negligible beyond some T, which would greatly help practitioners tune the method.  
   - As it stands, the choice of \(T\) and \(\sigma(t)\) appears to be fixed to whatever Lou et al. (2024) used, without justification beyond “absorb configuration”.

7. **Related work coverage around discrete diffusion & information estimation is not fully up to date**  
   - While the paper cites Lou et al. (2024), Sun et al. (2023), Austin et al. (2021), and Franzese et al. (2023a) / Bounoua et al. (2024) on diffusion-based information estimators, it misses several **recent discrete-diffusion / information-theoretic** works that seem directly relevant (see “Potentially Missing Related Work” below).  
   - In particular, there is no discussion of discrete-bridge-based MI estimation or discrete copula diffusion, which offer alternative ways of marrying diffusion and information measures. This weakens the claims of uniqueness and prevents a sharper comparison of design choices (e.g., CTMC vs. bridge processes, Poisson diffusion).

8. **Minor but nontrivial clarity issues and typos**  
   - Eq. (5) introduces \(\vec{\hat{X}}_t\) without having been defined before (the text around uses \(\vec{X}_t\)).  
   - In Section 3, the notation for the marginal distributions in Eq. (6) is inconsistent: \(\vec{p}_t^X(x)\) vs. \(\vec{p}_t'(x')\). It would help to standardize notation and explicitly denote the marginal processes.  
   - Algorithm 3 (INFO-SEDD-H) has a term \(\frac{1}{N(s^{\theta(t)}-1)}\) that appears to be a typo for \(\frac{1}{N(e^{\overline{\sigma}(t)}-1)}\) from Appendix A.2. This is more than cosmetic, because it is the only place where the uniform-reference ratio enters the implementation.  

   These do not invalidate the method, but they do make it more difficult for a reader to reimplement INFO-SEDD purely from the paper.

## Potentially Missing Related Work

(Only listing papers that are not cited in the submission and appear directly relevant.)

1. **Jeon et al., “Information-Theoretic Discrete Diffusion”, 2025**  
   - Directly addresses information-theoretic identities in discrete diffusion models, relating mutual information and score-based training objectives. Given that this paper builds an MI estimator on top of discrete CTMC diffusion, this work should be discussed in Section 2 (Methodology) and Section 5 (Conclusion) as a complementary theoretical perspective, and in the related work discussion around diffusion-based information estimators.

2. **Bhattacharya et al., “ItDPDM: Information-Theoretic Discrete Poisson Diffusion Model”, 2025**  
   - Proposes a Poisson-based discrete diffusion for likelihood and information-theoretic objectives on discrete data. This is very close to the problem tackled here (information estimation on discrete state spaces) and should be cited in the introduction and compared in Section 2.2 as an alternative to CTMCs with absorbing states.

3. **Zabarianska et al., “Discrete Bridges for Mutual Information Estimation”, 2026**  
   - Develops MI estimation via discrete bridge processes. This is arguably the most directly comparable alternative MI estimator for discrete variables and should be included in the related work and, ideally, in experimental comparison (at least on synthetic data) if feasible. At minimum, the paper should discuss design differences (CTMC vs. bridge, training objectives, sample complexity).

4. **Liu et al., “Discrete Copula Diffusion”, 2024**  
   - Introduces a discrete copula-based diffusion framework, which is another way to construct discrete diffusion processes. This should be discussed in Section 2.1 when surveying discrete diffusion choices and how they relate to the chosen absorbing CTMC design.

5. **Tang et al., “PepTune: De Novo Generation of Therapeutic Peptides with Multi-Objective-Guided Discrete Diffusion”, 2025**  
   - While application-focused, this paper showcases discrete diffusion on peptide sequences, which is conceptually adjacent to the genomics application here. It would be appropriate to acknowledge in Section 4.3 that discrete diffusion has been used in related biological sequence modeling, and contrast their generative focus with INFO-SEDD’s information-estimation focus.

6. **Wang et al., “Remasking Discrete Diffusion Models with Inference-Time Scaling”, 2025**  
   - Addresses efficiency and scaling in discrete diffusion via remasking and inference-time tricks. Since scalability is a stated goal of INFO-SEDD, it would be useful to mention this work in Section 3 when discussing sparse rate matrices and potential future work on inference-time efficiency.

7. **Fu et al., “Discrete Curvature Graph Information Bottleneck”, 2024**  
   - Introduces an information-theoretic objective for discrete graph-structured data. While not diffusion-based, it provides another angle on discrete MI estimation that might be worth briefly citing in the introduction or related work, especially when motivating applications in structured discrete domains (beyond sequences).

8. **Wang et al., “Di2Pose: Discrete Diffusion Model for Occluded 3D Human Pose Estimation”, 2024**  
   - Uses discrete diffusion for pose estimation. This can be briefly acknowledged in the discrete diffusion overview (Section 2.1), positioning INFO-SEDD as orthogonal (information estimation rather than predictive modeling).

## Questions

1. **Boundedness of score ratios (Assumptions A.1–A.2):**  
   - In high-dimensional text or DNA settings, how realistic is it to assume that all ratios \(s^p(x_t)_{\hat{x}} = \frac{p_t(\hat{x})}{p_t(x_t)}\) lie in \([C_1,C_2]\) with finite positive \(C_1\)?  
   - Do you apply any explicit clipping or regularization to the network outputs to enforce such bounds during training? If so, please describe; if not, can you provide empirical histograms of \(s^p\) values to argue that the assumptions are approximately met in practice?

2. **INFO-SEDD-C implementation details (Algorithm 2):**  
   - The pseudo-code includes terms like \(s_\theta([\vec{X}_t,\vec{Y}_0])_{[\tilde{X},\vec{Y}_0]}\) which do not seem consistent with the rest of the algorithm that conditions on \([\vec{X}_0,\vec{Y}_t]\). Could you clarify the correct indexing and provide a clean derivation of INFO-SEDD-C starting from Eq. (5)?  
   - Is INFO-SEDD-C implemented exactly as Algorithm 2, or does the code differ? A corrected algorithm in the camera-ready would be helpful.

3. **Complexity in large-vocabulary language models:**  
   - For the SUMMEVAL experiments, what is the vocabulary size \(|\chi|\) used by MDLM-small, and do you perform full enumeration in the inner loop over \(n\in[1:N]\) in Algorithms 1–2, or do you sample neighbors?  
   - If you sample, how many neighbors per position, and how does this affect estimator bias and variance? Some quantitative runtime / FLOPs comparisons on SUMMEVAL would be useful.

4. **Sensitivity to truncation time \(T\) and \(\sigma(t)\):**  
   - How did you choose \(T\) and \(\sigma(t)\) for the different experiments, especially the high-MI synthetic ones vs. text vs. DNA vs. Ising?  
   - Could you share results of a small ablation where you vary \(T\) (keeping the score network fixed) and report MI estimates and variance, to empirically validate the truncation-bias theory in Eq. (7)?

5. **Baselines and pretraining alignment:**  
   - For text summarization, are the baseline estimators (DIME variants, SMILE, MINDE, etc.) trained from scratch on SUMMEVAL with the same MDLM-small backbone, or do they also benefit from pretraining on OpenWebText?  
   - If the backbones differ in pretraining between INFO-SEDD and baselines, could you clarify and, if possible, align them in future experiments so that the comparison isolates the estimation method rather than backbone strength?

6. **Discrete bridges / Poisson diffusion vs. CTMC absorbing approach:**  
   - How would you position your CTMC-with-absorbing-states approach relative to discrete-bridge MI estimators or Poisson-based discrete diffusion for information-theoretic quantities?  
   - Do you see any fundamental advantages of your construction (e.g., the absorbing trick of Eq. (6)) that would be hard to replicate in those frameworks?

Answers and clarifications on these points, especially around Algorithms 1–2, complexity at scale, and the role of the boundedness assumptions, could substantively increase confidence and strengthen the paper.

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A. The paper uses standard text and genomics datasets without sensitive human subject data, does not propose harmful applications, and does not appear to violate data usage norms as described.

## Soundness Rating

3: good.  
The derivations around Eq. (4) and Eq. (7) are careful and logically consistent given the assumptions, and the empirical validation is broad and matches the claimed strengths. Some assumptions (bounded scores) and implementation details (Algorithm 2, practical complexity) need better justification and clarification, but I do not see fatal methodological flaws.

## Presentation Rating

3: good.  
The paper is generally well written, with clear motivation and a structured exposition. Key equations (1), (4), (6), (7) are explained, and figures/tables (e.g., Table 1, Figures 1–5) effectively support the narrative. However, several notational inconsistencies and algorithmic typos (especially in Algorithm 2 and Algorithm 3) should be fixed to improve reproducibility.

## Contribution Rating

3: good.  
The work proposes a meaningful integration of discrete diffusion and information estimation, including a practically useful single-model absorbing trick, theoretical error analysis, and strong experiments on challenging discrete domains. While not entirely without competition from other recent discrete-diffusion / MI works, the combination of theory, method, and applications is a solid and valuable contribution to the ICLR community.

## Overall Rating

8: Accept, good paper (poster).  
Despite some clarity and assumption caveats, this is a technically solid and impactful paper that advances MI estimation for discrete data using discrete diffusion, backed by substantial experiments and nontrivial theory. It is well suited for presentation as a poster.

## Reviewer Confidence

4: confident.  
I am familiar with diffusion-based generative modeling and MI estimation work, have closely examined the main derivations and the experimental design, and I am reasonably confident in the assessment, though I have not line-by-line verified every lemma in the appendices.