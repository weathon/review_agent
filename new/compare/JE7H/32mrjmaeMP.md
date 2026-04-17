---
job_id: ee552731-ee00-4e51-a51c-135c54594061
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: 32mrjmaeMP.pdf
paper: Dataless Weight Disentanglement in Task Arithmetic via Kronecker-Factored Approximate Curvature
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper addresses curvature-based regularization for task arithmetic in deep networks, squarely within optimization, representation learning, transfer learning, and model editing, which are all core ICLR topics.

## Minimum Quality
Pass ✅.  
The paper includes Abstract, Introduction, Background/Methodology, Experiments, Results, and Conclusions. It is technically detailed, in clear English, with substantial empirical evaluation and nontrivial methodological development. I do not see any fatal methodological or theoretical flaws that would justify desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I do not see any attempts to manipulate automated reviewing (no hidden prompts, meta-instructions to reviewers, or similar content).

---

# Expected Review Outcome:

## Summary

The paper studies task arithmetic and weight disentanglement, proposing a dataless regularizer for fine-tuning task vectors based on curvature information. Under linearized fine-tuning, the authors show that representation drift between tasks can be written as a quadratic form involving the Jacobian Gram matrix, which is an instance of the generalized Gauss–Newton (GGN) matrix, and then approximate this via KFAC. They further introduce a heuristic to merge per-task KFAC factors into a single aggregated surrogate, achieving constant complexity in the number of tasks, and demonstrate strong performance on task addition and task negation for CLIP vision models and T5 in both linearized and non-linear fine-tuning regimes.

## Strengths

1. **Conceptual link between representation drift and curvature is clean and useful.**  
   Section 3.1 shows that under linearization, the representation drift penalty reduces to a quadratic form in the Jacobian Gramian,  
   \[
   \mathcal{L}^{\text{drift}}_{t\to t,t'}(\tau_{t'}) = \alpha_{t'}^2\, \tau_{t'}^\top G_t(\theta_0)\tau_{t'},
   \]
   where \(G_t\) is the data-averaged Jacobian Gram matrix (Eq. (3)). The connection of this matrix to the GGN in Eq. (5) is technically correct and well explained, and it directly enables the reuse of KFAC machinery. This is a crisp theoretical framing of “dataless representation drift regularization”.

2. **Dataless regularizer with constant-in-\(T\) complexity is practically attractive and nontrivial.**  
   The multi-task extension in Section 3.4, moving from the naïve sum of per-task quadratic forms (Eq. (7)) to the aggregated surrogate in Eq. (8), is an interesting design choice. The Kronecker “merge” heuristic  
   \[
   \sum_t \lambda_t B_t^l\otimes A_t^l \approx \left(\sum_t B_t^l\right)\otimes \left(\sum_t \lambda_t A_t^l\right)
   \]
   is backed by an explicit approximation-error analysis in Appendix C (Eqs. (9)–(18)), which makes clear how deviations of task-specific factors around their means drive the error. The experiments in Table 3 show this heuristic is very close to the idealized \(O(T)\) objective, which is a strong empirical sanity check.

3. **Strong empirical evidence on both vision and language, including task addition and negation.**  
   - **Task addition (vision):** Table 1 shows that on 8 Vision, the proposed TAK regularizer, in the linear regime, matches or surpasses τJp (which uses external data) and clearly dominates prior dataless regularization (Diagonal GGN). For example, for ViT-B/16 with \(\alpha=1\), TAK reaches 88.3 abs / 97.9 norm vs Diag GGN’s 82.9 / 93.2 and plain linear FT’s 80.2 / 88.9.  
   - **Task negation:** Table 2 shows TAK achieves far stronger forgetting of target tasks while preserving control performance at ≥95% of pretrained, beating TaLoS and τJp. For ViT-L/14, TAK drives target accuracy down to 3.5 while maintaining 72.6 on ImageNet, compared with τJp’s 3.7 / 73.0 and TaLoS’ 10.7 / 73.6.  
   - **Language tasks:** Figure 3 and Table 2 (bottom) & Table 8 show that TAK consistently improves task addition for T5-base over linear FT, non-linear FT, and attention-only FT, and is competitive with τJp despite being dataless.

4. **Robustness to scaling coefficients and compelling \(\alpha\)-sweep analyses.**  
   A central claim is that TAK makes simple \(\alpha=1\) task-vector summation competitive without cross-task validation. Figures 2, 4, 11, and 12 and Table 1 substantiate this.  
   - In **Figure 4a**, TA with KFAC (green curve) is both peak-accurate and very flat over \(\alpha\in[0.5,1.5]\), while ISO, TSV, and TIES under non-linear FT are much more sensitive.  
   - In **Figure 11** (and Figure 12 for non-linear regime), the KFAC-regularized models retain higher accuracy over a broad \(\alpha\) range compared to non-regularized baselines.  
   This robustness is a strong practical advantage for scenarios without shared validation data, and it directly follows from the theory: the quadratic regularizer discourages task vectors from affecting out-of-task regions, making the composed function less sensitive to moderate rescaling.

5. **Nice empirical story on task localization / weight disentanglement.**  
   The paper does more than just report accuracy: it directly probes disentanglement.  
   - **Figure 5** shows the distributions of \(\|\mathrm{J}_\theta f(x,\theta_0)\tau_t\|_2^2\) for inliers vs outliers. Under linear FT, the distributions heavily overlap, but under TAK they are clearly separated with out-of-task examples having scores pushed towards zero.  
   - **Figure 9** shows disentanglement error heatmaps in the \((\alpha_1,\alpha_2)\) plane: KFAC-regularized linear FT yields near-zero interference along the diagonals, much more so than unregularized linear FT or attention-only FT.  
   - **Figure 13** and **Figure 14** confirm similar task-localization behavior under non-linear FT and with compressed KFAC.  
   These visualizations support the qualitative claim that curvature regularization enforces a sort of “task-localized” Jacobian response, not just improved average accuracy.

6. **Thorough analysis of efficiency, approximation quality, and practical tradeoffs.**  
   The paper does not hand-wave around KFAC complexity; it quantifies it.  
   - **Figure 6a** and Table in Figure 6b show that MC KFAC precomputation over 8 Vision tasks takes about 4 minutes vs ~199 minutes for exact factors, and that training-time overhead is moderate and lower than τJp. Memory overhead is also quantified (+12% in linear FT, +22% in non-linear attention-only).  
   - **Figure 7a** shows how performance saturates with 128–256 examples and only 1–2 MC samples; more MC samples can even hurt.  
   - **Figure 7b** explores several compression schemes (block diagonal, SVD, pruning, quantization) and shows that block-based KFAC can reduce memory by ~87% for ViT-B/16 while losing only ~1 point accuracy, which is a very useful result for scaling.  
   These details will be highly useful for practitioners interested in actually adopting TAK, and they strengthen the paper’s empirical credibility.

7. **Clarity and positioning relative to prior work are generally good.**  
   The relation to τJp, TaLoS, attention-only FT, and prior second-order / diagonal GGN regularization is clearly discussed. The background on GGN and KFAC in Section 3.2–3.3 is crisp, with correct derivations (e.g., KFAC factorization for a dense layer using \(\mathbb{E}[B_n^l\otimes A_n^l]\approx \mathbb{E}[B_n^l]\otimes \mathbb{E}[A_n^l]\)). Related work on linearized fine-tuning and tangent-space task arithmetic is carefully discussed in Appendix G.

## Weaknesses

1. **Heuristic multi-task KFAC merging (Eq. (8)) lacks a task-arithmetic-specific justification beyond generic Frobenius bounds.**  
   The core scalability claim relies on approximating  
   \[
   G_{-t'}(\theta_0^l) \approx \sum_{t\neq t'} \lambda_t B_t^l\otimes A_t^l \approx \left(\sum_t B_t^l\right)\otimes \left(\sum_t \lambda_t A_t^l\right),
   \]
   which is mathematically convenient but quite coarse. Appendix C provides a Frobenius norm bound \(\|E\|_F \le T\sigma_A\sigma_B\), but this is a generic worst-case error and does not connect to the *actual penalty* \(\tau^\top G\tau\) or to how this affects task interference. In particular:
   - The error bound does not depend on \(\tau\), so it gives no guarantees on the quadratic form relevant for training.  
   - The assumption that A/B factors are “clustered” across tasks (hence σ small) is plausible for 8 Vision on CLIP but not empirically examined: there is no quantitative analysis of factor similarity across tasks or what happens when tasks are structurally different (e.g., mixing vision and text, or very heterogeneous vision domains).  
   - Tasks appear in tens at most; it is not clear whether this merging remains benign for hundreds or thousands of tasks where \(\sigma_A, \sigma_B\) could grow.  
   While Table 3 shows small performance gaps between naïve multi-task FT and accumulated reg., these experiments are limited to moderate T and specific domains. Some empirical analysis of factor divergence across tasks or at least a sensitivity study of the merging heuristic vs number and diversity of tasks would strengthen the claim that the O(1) surrogate is generally safe.

2. **The method’s dependence on linearization is only partially mitigated in the non-linear regime; the conceptual story there is weaker.**  
   The core theoretical derivation of the regularizer (Sec. 3.1) is exact only in the linearized setting. The paper then applies the same regularizer in non-linear FT, arguing that attention-only FT approximately enforces linear behavior. However:
   - There is no explicit quantitative measurement of how linear the actual fine-tuning is under attention-only FT or non-linear FT with TAK, e.g., measuring NTK stability or linearization error across training.  
   - Figures 2(right), 3(right), 12 show that non-linear regimes remain significantly more sensitive to \(\alpha\) and often require tuning; in some cases the gains from TAK are smaller than in the linear regime.  
   - The paper explicitly notes that the regularizer is “not theoretically exact in the non-linear regime”.  
   As a result, the conceptual tightness between objective and behavior is significantly looser outside the linearized setting. This is not a fatal flaw, but the paper could be more honest about the limits of its guarantees and provide more diagnostics on where the linearized curvature prior remains informative.

3. **Mathematical formulation of the overall training objective could be clearer about scalar factors and scaling invariances.**  
   In Eq. (3) and Eq. (6), the drift loss is \(\alpha_{t'}^2\,\tau_{t'}^\top G_t\tau_{t'}\). In Eq. (7), the overall objective is written as  
   \[
   \mathcal{L}_{\mathcal{D}_{t'}}(\tau_{t'}) + \beta\sum_{t\neq t'} \lambda_t \sum_l \tau_{t'}^{l\top}(B_t^l\otimes A_t^l)\tau_{t'}^l,
   \]
   but the explicit \(\alpha_{t'}^2\) factor disappears. This suggests the authors may be absorbing \(\alpha_{t'}\) into \(\beta\) or working with \(\alpha\) at inference only, but this is not clearly spelled out. Similarly, the discussion of robustness to \(\alpha\) scaling is qualitative. A more careful mathematical discussion of the scaling behavior of the loss with respect to \(\alpha\) and \(\tau\) would be valuable, especially because a central claim is that TAK makes performance almost invariant to \(\alpha\) (Figures 4, 11, 12). For example:
   - Under linearization, composing \(\theta = \theta_0 + \sum_t \alpha_t \tau_t\) means predictions are linear in \(\alpha\), so rescaling \(\tau_t\) and inverse-rescaling \(\alpha_t\) can be equivalent in some regimes.  
   - It would be helpful to articulate what form of invariance the regularizer is enforcing (e.g., penalizing large Jacobian responses so that moderate changes in \(\alpha\) have controlled impact).

4. **Limited diversity of benchmarks relative to very broad claims about practical deployment.**  
   While the experiments on 8 Vision (multiple CLIP backbones), 6 NLI tasks (T5-base), and additional class-incremental splits (Tab. 7) are substantial, there are still some gaps relative to the paper’s narrative about “massive datasets” and “foundation models”:
   - All backbones are medium-scale CLIP ViTs and T5-base; there are no experiments on LLaMA-style large LMs or very large vision models where KFAC’s quadratic layer scaling might become constraining.  
   - All tasks are supervised classification / NLI; it is unclear whether the approach would behave similarly for generation tasks, dense prediction, or RL settings, where representation drift might manifest differently.  
   - The number of tasks remains modest (<=10 or so). Given that the method is explicitly motivated by modular, scalable composition for many tasks and constant complexity in T, an experiment where T is substantially larger (even with smaller networks) would be informative.  
   The paper hints at these limitations in Appendix B, but the main text sometimes overgeneralizes the applicability.

5. **Some experimental comparisons could be more comprehensive and better controlled.**  
   Several baselines are strong, but there are still some places where the evaluation feels slightly skewed or under-specified:
   - In Table 1, τJp uses external data and TAK does not, which is properly noted. However, there is no ablation where TAK also uses the same external data to estimate curvature (even though the method is said to be compatible with that). This would clarify whether the main boost of τJp over TAK in language (Table 2 bottom, Table 8) is purely due to better curvature precision from more data, or due to qualitatively different regularization.  
   - For language, the non-linear regime experiments (Figure 3b) combine attention-only FT with TAK, but there is no “non-linear FT + TAK (all layers)” baseline. This would help disentangle whether we really need attention-only to approximate linear behavior, or whether KFAC alone is already regularizing non-linear FT sufficiently well.  
   - In Figure 4b and Table 4, merging methods are evaluated on top of linear FT or TAK, but there is no attempt to tune their hyperparameters under a KFAC-aware regime (e.g., using curvature information to adapt SVD thresholds). This is arguably outside the scope but important when concluding that simple TA + TAK “beats” ISO/TSV.

6. **Assumption of availability and sharability of KFAC statistics is not fully problematized.**  
   The paper argues that KFAC factors can be shared instead of data and are “privacy-preserving”. However, curvature matrices can leak information about the training distribution (e.g., through gradients or Fisher information). This is nontrivial in a federated or privacy-sensitive setting. The paper does not discuss whether KFAC statistics might be sensitive, nor whether they could be computed under differential privacy or secure aggregation. This is particularly relevant when the method is proposed specifically as a workaround for data-sharing constraints. A brief treatment (or at least a caveat) in the ethics/limitations section would be appropriate.

7. **Clarity issues and minor notation inconsistencies.**  
   Overall, the paper is well written, but there are some points where clarity could be improved:
   - In Algorithm 1 the KFAC approximation “(approximate via KFAC, Sec. 3.3)” and the “red font” comment are a bit confusing in isolation; a more self-contained description of how KFAC is precomputed (number of examples, MC samples) in the main text, not only App. E/F, would help.  
   - The symbol \(\mathcal{L}_{\mathcal{D}_t}(\tau_{t'})\) in Eq. (7) seems to have an index typo: the left side uses \(\mathcal{L}_{\mathcal{D}_t}\) but later text refers to training on \(\mathcal{D}_{t'}\). This is a minor but distracting inconsistency.  
   - In Section 3.3, the notation \(\mathbf{s}_{n,m}\in \mathbb{R}^n\) is odd; presumably this vector lives in \(\mathbb{R}^C\) since it relates to Hessian vectors in output space. Tightening these details would avoid confusion, especially for readers familiar with KFAC.

## Potentially Missing Related Work

N/A.  
The core KFAC and curvature literature, as well as recent task arithmetic, tangent-space, and model merging works, are already cited (Martens & Grosse 2015; Grosse & Martens 2016; Ritter et al. 2018; Dangel et al. 2025; Eschenhagen et al. 2023; Ortiz-Jimenez et al. 2023; Yoshida et al. 2025; Gargiulo et al. 2025; Stoica et al. 2025; etc.). I do not see obvious, directly-related missing citations that would materially affect positioning.

## Questions

1. **On the merging heuristic and its limits:**  
   - Can you provide any empirical measurements of \(\sigma_A,\sigma_B\) (as in Eq. (17)) across 8 Vision tasks, to quantify how similar KFAC factors really are?  
   - Have you experimented with synthetic settings where tasks have deliberately orthogonal statistics (e.g., different input domains) to see when the merged surrogate starts to degrade vs the naïve O(T) regularizer?

2. **Scaling to many tasks and larger models:**  
   - How does the approximation quality and memory footprint scale if T is in the hundreds and backbone is, say, a 1B+ parameter model? Can the compression strategies in Figure 7b keep KFAC practical in that regime, or do you anticipate a qualitative breakdown?  
   - Would you expect the block-based approximation to remain effective for much deeper transformers, or is there some depth/width where its accuracy collapses?

3. **Non-linear regime behavior and linearization diagnostics:**  
   - Have you measured any notion of linearization error (e.g., difference between full model and first-order Taylor expansion over training) for the different regimes (linear FT, attention-only FT, attention-only + TAK)? Showing that attention-only + TAK actually keeps the model close to its linearization would significantly strengthen your story for non-linear regime.  
   - In non-linear FT without attention-only, does TAK still help, or does the mismatch between curvature at \(\theta_0\) and the actual trajectory dominate?

4. **Security/privacy of sharing KFAC statistics:**  
   - Have you considered whether the KFAC matrices themselves might leak significant information about the training data? Are there obvious ways to make their computation differentially private (e.g., DP-SGD on curvature estimates)? It would be useful to hear your perspective, given that “privacy-preserving” is a core motivation.

5. **Ablation on using external data with TAK:**  
   - Can you run an ablation where TAK is allowed to use the same external task data as τJp during curvature estimation for T5, to see how much of the remaining gap in Table 2/8 is due to the choice of regularizer vs pure data availability? This would help clarify whether there is still headroom in the TAK framework if one relaxes the dataless constraint.

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The derivation connecting representation drift to the Jacobian Gram / GGN is correct, the use of KFAC is standard, and the empirical evidence is extensive and mostly well controlled. The main concern is the heuristic nature of the KFAC merging in Eq. (8), which is not theoretically justified beyond coarse Frobenius bounds, but empirical results and extra analyses (Table 3, Appendix C/F) alleviate this.

## Presentation Rating

3: good.  
The paper is well written, relatively easy to follow despite heavy notation, and backed by informative figures (e.g., Figures 2, 4, 5, 9, 11, 13). A few notation inconsistencies and minor typos exist but do not impede understanding.

## Contribution Rating

3: good.  
The contribution is a solid combination of a conceptual link (representation drift ↔ curvature), a practical KFAC-based regularizer tailored for task arithmetic, and an aggregation heuristic that addresses scalability. It pushes the state of the art in dataless task arithmetic and provides a fairly complete empirical and efficiency study.

## Overall Rating

8: Accept, good paper (poster).  
The work offers a clear and non-trivial contribution to task arithmetic and weight disentanglement by turning data-dependent drift penalties into a dataless KFAC-based curvature regularizer. It is technically sound, algorithmically interesting (especially the O(1) KFAC aggregation heuristic), and empirically strong across multiple settings, with convincing visualizations of disentanglement and robustness to \(\alpha\). The main limitations concern the heuristic nature of factor merging, partial mismatch between the linear theory and non-linear practice, and somewhat narrow range of task types and model scales. Nonetheless, these are incremental issues rather than fatal flaws; the paper is well worth presenting at ICLR.

## Reviewer Confidence

4: confident.  
I am familiar with task arithmetic, linearized fine-tuning, and KFAC / curvature approximations, and I carefully checked the main derivations (Eqs. (3), (5–8), Appendix C) and experimental tables/figures. Some details (e.g., factor similarity across tasks or privacy aspects) might benefit from author clarification, but they are unlikely to change my overall assessment.