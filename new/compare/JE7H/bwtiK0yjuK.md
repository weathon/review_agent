---
job_id: a7effc77-fb41-45fa-9a4d-af1cfef268d8
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: bwtiK0yjuK.pdf
paper: Change Point Localization and Inference in Dynamic Multilayer Networks
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper develops statistical methodology for change point detection in dynamic multilayer random dot product graphs with theory and experiments, which clearly falls under “learning on graphs”, “learning theory”, and general machine learning.

## Minimum Quality
Pass ✅.  
The paper is in English and has all required components (Abstract, Introduction, model/method sections, theory, experiments, and Conclusion). The methodology and theorems are nontrivial, proofs are detailed in the appendix, and experiments are substantive with multiple baselines and real data. I do not see fatal theoretical or experimental flaws that would justify desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I do not observe any attempts to manipulate automated reviewing (no hidden prompts, meta-instructions, or similar content).

---

# Expected Review Outcome:

## Summary

The paper studies offline multiple change point localization and inference in dynamic multilayer random dot product graphs (D‑MRDPGs), where nodes share latent positions but layer-specific weight matrices change over time. The authors propose a two-stage offline procedure that combines seeded binary segmentation with tensor heteroskedastic PCA for low-rank estimation, prove consistency for the number and locations of change points under a quantified SNR condition, and derive limiting distributions of a further refined estimator in both vanishing and non-vanishing jump regimes. They also provide a data-driven confidence-interval construction and demonstrate strong empirical performance on simulated multilayer networks and two real-world datasets (world agricultural trade and U.S. air transport).

## Strengths

1. **Well-motivated extension to multilayer random dot product graphs.**  
   The paper starts from MRDPGs and their dynamic version (Definitions 1–2, Model 1) and gives a clean formulation for multilayer temporal networks with shared latent positions and time-varying layer weight matrices. This setting is relevant to many ML-on-graphs applications where multiple relation types coexist.

2. **Technically substantial two-stage methodology with nontrivial tensor component.**  
   - Stage I adapts seeded binary segmentation (Definition 3 and Algorithm 1) with CUSUM statistics for tensors (Definition 4) and uses inner products between CUSUMs from two independent sequences as detection statistics.  
   - Stage II builds refined scan statistics (Definition 5) by projecting CUSUM tensors onto a low-rank estimate obtained via TH-PCA (Algorithm 2), which is nontrivial: they exploit the Tucker structure of the expectation in (2)–(4) and Assumption 1 to justify using tensor-based low-rank approximation.  
   This integration of SBS with tensor heteroskedastic PCA is technically careful and not a standard, superficial extension.

3. **Strong asymptotic guarantees with explicit rates.**  
   - Theorem 1 proves that Algorithm 1 recovers the correct number of change points and localizes each $\eta_k$ with error $|\tilde\eta_k-\eta_k|\le C\log T/\kappa_k^2$ with high probability, under SNR Assumption 2. The rate matches the single-layer optimal rates up to logs and cleanly incorporates multilayer complexity via $L$ and $m_{\max}$.  
   - Theorem 2 (and Theorem 3 in Appendix A) further derive limiting distributions of the refined estimator $\hat\eta_k$ (Equation (5)) in both vanishing and fixed-jump regimes, with argmin functionals of a two-sided Brownian motion / random walk. This is a technically demanding result, and the proofs (Appendix F) carefully control bias terms like (35)–(37) and stochastic terms using Lemma 5.

4. **Inference contribution (confidence intervals) is rare in network CP literature.**  
   Building on Theorem 2, Section 3.1 gives a concrete simulation-based CI procedure: estimate jump tensor and variance (Steps 1–2), simulate approximate limit processes (Step 3), and invert to an interval in $t$-space (Step 4). The empirical evaluation in Table 2 shows good coverage with reasonably short intervals in scenarios consistent with Model 1; this is a meaningful step beyond “point estimate only” methods.

5. **Empirical evaluation is extensive and includes challenging baselines.**  
   - Simulation scenarios cover both MRDPG-like constructions (DDM) and multilayer SBMs, with both model-conforming and model-violating changes (Scenarios 1–4). This tests robustness beyond the exact assumed model.  
   - Table 1 compares against gSeg and kerSeg using both raw-network and Frobenius-norm inputs, with multiple metrics: error in number of CPs, directed Hausdorff distances, and time-segment coverage. CPDmrdpg is consistently best or tied, especially on $d(\mathcal C|\hat{\mathcal C})$ and coverage, and avoids the spurious oversegmentation seen in the baselines.  
   - Table 2 reports CI coverage, which is rarely reported in this area.  
   - Appendices add sensitivity to threshold $c_{\tau,1}$ (Tables 5–8), ranks $r$ (Table 9), frequent and random CPs (Tables 10–11), temporal dependence (Table 12), and comparisons to specialized online multilayer and deep-learning-based methods (Table 13). This is substantially more thorough than usual.

6. **Real data analyses are thoughtful and interpretable.**  
   - For worldwide agricultural trade, Table 3 shows that CPDmrdpg yields four change years (1991, 1999, 2005, 2013) that map cleanly to WTO / geopolitical events; competitors either miss late CPs or produce clusters of very close ones. Table 4 provides tight 95% CIs for these points, which helps interpretability.  
   - The U.S. air transportation analysis (Appendix G.2, Tables 14–15) similarly finds five CPs matching regulatory and COVID-related disruptions. These case studies convincingly argue that the method captures meaningful structural shifts rather than noise.

7. **Careful handling of figures and explanations of the theoretical setup.**  
   The only figure, **img-0.jpeg**, in Appendix F illustrates the interval $(\tilde \eta_{k-1}, \tilde \eta_{k+1})$ containing three neighboring change points and their refined estimates. This figure clarifies the complex bias analysis in Step 2 of the proof of Theorem 2/3, making the splitting of intervals and the various distances between true and estimated CPs much easier to follow.

8. **Clarity of mathematical assumptions and tools.**  
   Assumptions 1–2 are clearly stated, with interpretation provided (e.g., discussion after Assumption 1 about ranks of $X$ and $Q^{s,e}$). The use of the tensor matricization $\mathcal M_s$, Tucker ranks, and TH-PCA is systematically introduced, and Lemma 5 gives a clean sub-Gaussian Bernstein-type inequality for tensor inner products that is central to many bounds.

## Weaknesses

1. **Model and rank assumptions are quite restrictive and under-discussed for practice.**  
   - Assumption 1(ii)–(iii) require that for every CUSUM/average matrix $\widetilde Q^{s,e}(t)$ and $Q^{s,e}$ the smallest nonzero singular value is bounded below by a constant and the condition number is bounded above. This enforces a uniform, strong low-rank structure on the layer-wise weight matrices across all intervals and CUSUM constructions. In high-$L$ or highly heterogeneous multilayer graphs this may be unrealistic.  
   - The practical rank choice is treated somewhat heuristically: it is set to $r_1=r_2=15$ and $r_3=L$ in simulations (Section 4.1), with a brief pointer to Wang et al. (2025). There is no data-driven rank selection or sensitivity beyond the small exploration in Table 9. Given that the SNR in Assumption 2 explicitly depends on $d^2 m_{\max} + nd + L m_{\max}$, overestimating ranks can significantly affect the theory, yet users are not given actionable guidance beyond “use a fairly large rank”.

2. **Strong spacing assumption $\Delta = \Theta(T)$ and limited treatment of frequent changes.**  
   Model 1 requires the minimal spacing $\Delta$ between consecutive change points to scale linearly with $T$, effectively bounding $K$ by a constant. While Appendix G.1 includes experiments with higher $K$ (Table 10), the theory does not cover that regime and the main text only briefly mentions that “this assumption can be relaxed.”  
   From a learning-theory perspective, this is a major limitation: many practical temporal networks may have $K$ growing with $T$ at a sublinear rate. The authors should be more explicit about what breaks when $\Delta \ll T$ (e.g., seeded interval coverage, SNR dependence), and at least sketch how alternatives like narrowest-over-threshold (mentioned in the Conclusion) could be integrated into their tensor pipeline.

3. **Limited comparison to other network-specific change-point methods, especially multilayer/tensor-based ones.**  
   - The main comparisons are to gSeg and kerSeg, which are generic high-dimensional sequence methods. There is only a brief comparison in Appendix G.1 to CPDonline (Wang et al., 2025) and AutoCPD (Li et al., 2024).  
   - Notably absent from experiments and core related-work discussion are several directly relevant methods for temporal network change detection that also exploit multi-relational or tensor structures, such as subspace tracking on dynamic heterogeneous networks or MDL-guided tensor decompositions (see “Potentially Missing Related Work”).  
   - This underplays how competitive CPDmrdpg actually is against methods specifically tailored to dynamic/multilayer networks, and makes it harder to understand when one should prefer this low-rank Tucker approach over, for example, online subspace tracking or MDL-based tensor segmentation.

4. **Interpretability and scale of CIs are somewhat questionable in real-data results.**  
   In Table 4 (agricultural trade) and Table 15 (air transport), many 95% intervals are extremely narrow (often width < 0.1 time units on a discrete yearly/monthly scale). This is a consequence of dividing the bootstrapped argmin $u_k$ by $\hat\kappa_k^2$ and using $M=T$ in the Brownian approximation. While theoretically coherent under the strong SNR and independence assumptions, in real-world networks with unmodeled nonstationarity and dependence, such ultra-tight CIs are likely overconfident. The paper does not discuss any diagnostic to assess whether the vanishing-jump asymptotics and variance estimation in Step 2 of Section 3.1 are appropriate for a given dataset.

5. **Heavy independence assumptions and relatively light treatment of dependence.**  
   - The main theory assumes temporal independence of adjacency tensors and even uses four mutually independent copies $\{A, A', B, B'\}$ in Algorithm 1 for technical convenience. In practice, the authors use odd–even splits, but the impact of this approximation on finite-sample performance is not theoretically quantified.  
   - Appendix B extends to a particular Markovian dependence with parameter $\pi$, but this is separate from the empirical dependence model used in temporal dependence experiments (Table 12) and is somewhat hand-wavy: multiple key inequalities are stated informally as “by revising the proof using Lemma 14 in Padilla et al. (2022)”. An explicit statement and proof of a dependence-robust version of Theorem 1, even under restrictive $\pi$, would significantly strengthen the paper.

6. **Some aspects of the math exposition could be clarified, especially around notation and limit objects.**  
   - In Definition 5, the refined scan statistic  
     \[
     \tilde D_{b_k}^{s_k,e_k}(t)
       = \left|\left\langle
        \tilde{\mathbf P}^{s_k,e_k}(b_k)/\|\tilde{\mathbf P}^{s_k,e_k}(b_k)\|_F,\,
        \tilde{\mathbf A'}^{s_k,e_k}(t)\right\rangle\right|
     \]  
     works only if $\|\tilde{\mathbf P}^{s_k,e_k}(b_k)\|_F$ is nonzero. In theory this follows from Assumption 2 and Lemma 6, but in finite samples a small $\kappa_k$ or mislocalized $b_k$ could lead to degeneracy. A remark on numerically handling $\|\tilde{\mathbf P}\|_F$ near zero would be useful.  
   - The limiting distributions in Theorem 2 are stated as $\kappa_k^2(\hat\eta_k - \eta_k) \Rightarrow \arg\min_{r\in\mathbb R} \mathcal P'_k(r)$ but then $\mathcal P'_k$ is defined piecewise in terms of Brownian motions on $\mathbb Z$ with sums $\sum_{i=\lceil Tr\rceil}^{-1} z_i^{(b)}$. There is a slight mismatch between the continuous $r\in\mathbb R$ and the implicit discrete scaling, and a more explicit mapping between the discrete-time random walk (Theorem 3) and its Brownian limit for Theorem 2 would help.

7. **Computational costs and scalability are under-explored.**  
   The paper states the algorithmic complexity as $O(T n^2 L r\log^2(T\vee n))$ and mentions 10 hours for 100 Monte Carlo runs with $n=100,L=4,T=200$, but there is no systematic scaling experiment. For larger graphs (e.g., $n$ in thousands, $L$ tens), repeated TH-PCA in Stage II may be expensive. Some empirical timing vs. $n,L,T$ or discussion of parallelization strategies would help practitioners assess feasibility.

8. **Figure and interval-structure intuition could be expanded.**  
   The only figure, **img-0.jpeg**, depicts $\tilde \eta_{k-1}$, $\eta_k$, and $\tilde \eta_{k+1}$ on a line, in the bias analysis of Appendix F. While it is helpful, the main text never references or explains it; readers have to infer that this is the configuration with three CPs in $(\tilde\eta_{k-1},\tilde\eta_{k+1})$. A short pointer in Section 3 (or just before Step 2 in Appendix F) explicitly describing what the ticks represent and how they relate to Equations (35)–(37) would improve clarity.

9. **Missing and under-discussed related work (beyond what is cited).**  
   The paper frames itself primarily relative to Wang et al. (2021, 2025) and generic change-point methods. More nuanced positioning versus recent temporal-network CP methods that use tensorization or structured models (e.g., separable dynamic ERGMs, MDL-guided tensor decompositions) is missing, which weakens the argument for originality and domain significance.

10. **Minor technical nits.**  
    - In the SNR Assumption 2, the term $\sqrt{nL^{1/2}+d^2 m_{\max}+nd+Lm_{\max}}$ is somewhat opaque. A short derivation or intuitive explanation of each term (e.g., which dimension of the tensor noise it corresponds to) would aid understanding.  
    - Some notation glitches occur in Appendix E/F (e.g., $m_{b_k}^{s,e}$ vs. $m_t^{s,e}$, stray typos in the last pages where some LaTeX seems partially corrupted), which impede following the long proofs.

## Potentially Missing Related Work

1. **Kei, Li, Chen (2025), “Change Point Detection on a Separable Model for Dynamic Networks”.**  
   This work proposes change-point detection for dynamic networks under a separable temporal exponential-family random graph model, which is conceptually close to this paper’s D-MRDPG in that both impose structured matrix/tensor forms on temporal networks. It should be cited and discussed in Section 2 (problem formulation and model assumptions) and in the related-work narrative (currently scattered across the Introduction and later comparisons). A brief comparison of modeling assumptions (separable ERGM vs. MRDPG / multilayer weights) and potential pros/cons for inference would be useful.

2. **Zhang, Zhang, Sun (2023), “Change Point Detection in Dynamic Heterogeneous Networks via Subspace Tracking”.**  
   This paper develops subspace-tracking-based CP detection in dynamic heterogeneous networks, which is particularly relevant given that CPDmrdpg uses TH-PCA for low-rank structure. It should be added to the related-work discussion around tensor / subspace methods and, ideally, compared experimentally (e.g., by adapting it to multilayer adjacency streams), at least in one simulation scenario.

3. **Huang, Hu, Chen (2026), “Structural Change Point Detection in Temporal Networks via MDL-Guided Tensor Decomposition”.**  
   This is directly related in both model (temporal network tensors) and method (tensor decomposition plus MDL criterion). It should be cited when motivating the tensor-based TH-PCA refinement (Section 2.2–2.3) and compared conceptually as an alternative unsupervised tensor approach. Including it as an additional baseline in the simulation suite, or explaining clearly why such a comparison is technically difficult, would strengthen the empirical story.

4. **Dong, Chen, Wang (2019), “Modeling and Change Detection for Count-Weighted Multilayer Networks”.**  
   This addresses multilayer change detection for count-weighted networks (not Bernoulli), but structurally is very close: multilayer, repeated in time, with change points. It should be mentioned early in Section 1 as prior work on dynamic multilayer networks, and in Section 4 when discussing how the current D-MRDPG framework could be extended to weighted/count edges.

5. **Wang, Chakrabarti, Sivakoff (2017), “Fast Change Point Detection on Dynamic Social Networks”.**  
   Although focused on single-layer social networks, this paper develops efficient CP detection algorithms that could be adapted to multilayer settings. It should be included in the general CP-on-networks related work (Introduction / Section 2.1) and perhaps cited in Remark 1 when discussing computational vs. statistical trade-offs of different CP algorithms on network streams.

## Questions

1. **Practical rank selection and robustness.**  
   Could the authors provide a more concrete, data-driven strategy for choosing $(r_1,r_2,r_3)$ in TH-PCA beyond the fixed choices in Section 4.1? For example, do simple scree plots of $\mathcal M_s(\tilde{\mathbf B})\mathcal M_s(\tilde{\mathbf B})^\top$ work well, or is there a principled threshold based on the theoretical $\sigma_{\min}$ lower bounds in Assumption 1? An additional small experiment varying $r$ over a larger range (e.g., up to $d^2$) would help clarify sensitivity.

2. **Behavior when $K$ grows with $T$.**  
   The theory assumes $\Delta = \Theta(T)$, but Table 10 suggests reasonable performance for $K$ up to 12 at $T=200$. Can the authors provide at least heuristic guidance or conjectured scaling of localization error when $\Delta$ shrinks like $T^\beta$ for $\beta\in(0,1)$? Does seeded binary segmentation with the current $\mathcal J$ fundamentally break down, or is it “just” the SNR that fails?

3. **Dependence and the real-world data.**  
   The agricultural and air-transport networks are almost certainly temporally dependent and subject to nonstationary external shocks. How sensitive do the CI construction and localization bounds appear empirically when the dependence parameter $\pi$ in Appendix B is misspecified or stronger than assumed? Could the authors provide a short simulation where $\pi$ is large (e.g., 0.2–0.5) to illustrate the practical degradation?

4. **Clarification of Theorem 2’s limit process.**  
   In Theorem 2, the limiting process $\mathcal P'_k(r)$ is written using Brownian motions $\mathbb B_1,\mathbb B_2$ and $r\in\mathbb R$, while the simulation in Section 3.1 uses discrete sums over Gaussian $z_i^{(b)}$ scaled by $1/\sqrt T$. Could the authors clarify whether the argmin is over $\mathbb Z$ or $\mathbb R$, and how the discrete-time construction in Step 3 of Section 3.1 approximates the continuous-limit argmin?

5. **Handling small or zero estimated jump sizes.**  
   In CI construction, the interval shrinks with $1/\hat\kappa_k^2$. What is the recommended practical treatment when $\hat\kappa_k$ is very small or zero (e.g., due to noise or extremely subtle change)? Is there a minimal $\hat\kappa_k$ threshold under which one should, say, default to a wider, conservative interval or refrain from inference at that CP?

6. **Comparisons to other tensor-based methods.**  
   Are the authors able to comment on how their TH-PCA-based refinement would compare theoretically or empirically to, say, a CPD/Tucker decomposition with MDL penalty (as in MDL-guided tensor CP detection)? Even a conceptual comparison on assumptions (heteroskedastic vs. homoskedastic noise, Tucker vs. CP rank, etc.) would be useful.

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The theoretical development is careful and mostly correct as far as I can see, with detailed proofs and appropriate concentration tools. The main caveats are the strong structural and spacing assumptions, and the somewhat informal treatment of temporal dependence in the appendix.

## Presentation Rating

3: good.  
The paper is overall clear and well organized, with precise definitions and well-explained algorithms. Some notation in the long proofs and the mapping between discrete and continuous limit processes could be sharpened, and the single figure could be better integrated.

## Contribution Rating

3: good.  
The paper addresses an important and underexplored problem (offline CP detection and inference in dynamic multilayer networks), proposes a technically substantive method, and provides nontrivial limiting-distribution results and CI procedures. The main limitations are strong assumptions and somewhat limited empirical comparison to other network-specific tensor methods.

## Overall Rating

8: Accept, good paper (poster).  
The methodological and theoretical contributions are solid and nontrivial, the empirical evaluation is comprehensive, and the problem is important for learning on dynamic multilayer graphs. While there are some strong assumptions and missing related-work comparisons, the paper is clearly above the bar for an ICLR poster.

## Reviewer Confidence

4: confident.  
I am comfortable with change point detection, random graph models, and low-rank / tensor methods, and I have carefully read the proofs and experiments. Some of the very technical details in the appendices could still conceal subtleties, but I am reasonably confident in my assessment.