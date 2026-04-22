Now I have all the information needed. Let me write the final consolidated review.

## Summary

The paper proposes TAK (Task Arithmetic with KFAC regularization), a dataless method for weight disentanglement in task arithmetic. The key insight is that the Jacobian Gram matrix used for representation drift regularization (Eq. 3) is an instance of the Generalized Gauss-Newton (GGN) matrix under squared loss, enabling the use of KFAC—a well-established curvature approximation—for efficient, data-free regularization. A merge heuristic (Eq. 8) further reduces storage and computation from O(T) to O(1) in the number of tasks.

## Strengths

- **Non-trivial theoretical connection between representation drift regularization and the GGN matrix (Sec. 3.1–3.2):** The paper shows that $G_t(\theta_0) = \frac{1}{|D_t|}\sum_{x \in D_t} J_\theta f(x,\theta_0)^\top J_\theta f(x,\theta_0)$ from Eq. (3) is exactly the GGN matrix when squared loss is used as the criterion. This equivalence is clean, correct, and enables transferring well-established curvature approximations to the weight disentanglement problem—a direction prior work (Yoshida et al., 2025; Porrello et al., 2025) did not exploit.

- **Dataless regularization achieving competitive or superior performance to data-dependent baselines:** On task addition (Table 1), TAK matches or exceeds τJp on ViT-B/32 and ViT-L/14 at α=1 (85.8 vs 85.0; 91.6 vs 90.9 absolute) while requiring no external task data. On task negation (Table 2), TAK achieves stronger forgetting (3.4 vs 6.7 target accuracy on ViT-B/32) with better control-task preservation—all without needing ImageNet data for regularization. This directly delivers the paper's central claim of a privacy-preserving, modular approach.

- **Comprehensive efficiency analysis (Fig. 6–8):** MC KFAC estimation with M=1 takes only 3.9 minutes vs. 198.7 for exact (Fig. 6b); block-based compression reduces KFAC storage from ~550 MB to ~70 MB (87% reduction) with only ~1-point accuracy drop (Fig. 7b); applying the penalty every 16 steps incurs only ~1.4 points degradation (Fig. 8). These are practically important findings.

- **Strong α-robustness in the linearized regime (Fig. 4):** TAK's accuracy varies far less with α than any competing method, meaning α=1 works well without validation-data-dependent tuning—a genuine practical advantage when held-out data is unavailable.

- **Evaluation across modalities:** The paper tests on both vision (8 Vision with three ViT variants, Table 1) and language (6 NLI tasks with T5-base, Fig. 3), in both task addition and negation, and in both linearized and non-linear regimes.

## Weaknesses

### Fatal
None.

### Major

- **The Kronecker merge heuristic (Eq. 8) lacks theoretical justification, and the O(1) scaling claim is only validated with 8 tasks.** The approximation $\sum_t \lambda_t (B_t^l \otimes A_t^l) \approx (\sum_t \lambda_t B_t^l) \otimes (\sum_t \lambda_t A_t^l)$ introduces cross-terms $B_i \otimes A_j$ for $i \neq j$ absent from the left-hand side. Table 3 shows a 0.7-point absolute accuracy gap on ViT-B/32 (86.5 vs 85.8), which the paper frames as "marginal." However, whether this approximation error grows with the number of tasks or task diversity is entirely untested. The abstract's claim of "constant complexity in the number of tasks" (line 7) and the contribution statement (line 38) both hinge on this heuristic's validity at scale, yet no experiment with more than 8 tasks is provided. If the merge degrades at larger T—exactly the regime where O(1) matters most—the core complexity advantage collapses. A quantitative analysis of the approximation error (e.g., $\|\sum_t \lambda_t(B_t \otimes A_t) - (\sum_t \lambda_t B_t) \otimes (\sum_t \lambda_t A_t)\|_F$ per layer) or an experiment with 16+ tasks would substantially strengthen this claim.

- **No variance or statistical significance reported for main results.** Tables 1–3 report single numbers. Many key comparisons are decided by sub-percentage-point differences: on ViT-B/32, TAK (85.8) vs. τJp (85.0) at α=1 is a 0.8-point gap; on ViT-B/16, τJp (98.3) vs. TAK (97.9) in normalized accuracy at α=1 is a 0.4-point gap in τJp's favor. The paper mentions "variance across seeds" in the KFAC estimation section (line 303), confirming seed-level data exists, but it is not reported for the main comparisons. Without this, the "state-of-the-art" claim cannot be rigorously assessed.

### Minor

- **The "state-of-the-art" claim is selectively supported and should be qualified.** On ViT-B/16 at α=1, τJp outperforms TAK in normalized accuracy (98.3 vs 97.9); at best-α, τJp wins on both metrics (88.6/98.7 vs 88.3/98.1). TAK also does not beat τJp on language tasks (Fig. 3 text: "leveraging data from other tasks yields additional gains"). The SOTA framing is most accurate for task negation and for the dataless category; for task addition without the dataless constraint, the evidence is mixed. The abstract should qualify this claim.

- **The "eliminating the need for held-out tuning" claim (abstract, line 7) overstates the non-linear regime.** In the linearized regime, α=1 is competitive with best-α (e.g., ViT-L/14: 91.6 vs 91.6), well supporting the claim. But in the non-linear regime with Attention-Only FT on ViT-B/32, α=1 achieves only 60.3 absolute accuracy vs. 83.1 at best-α (Table 1)—a 23-point gap. The abstract's blanket statement does not distinguish between regimes.

- **The per-task weighting scheme $\lambda_t = |D_t| / \sum_{t' \neq t} |D_{t'}|$ (line 189) is introduced without ablation or justification.** The 8 Vision datasets vary substantially in size (e.g., SVHN has ~73k training examples vs. DTD with ~1.8k), so this weighting could meaningfully affect results. An ablation against equal weighting would be informative.

- **The task localization / OOD detection claim is speculative.** Figure 5 shows that $\|J_\theta f(x,\theta_0)\tau_t\|_2^2$ is pushed toward zero for out-of-distribution inputs under TAK regularization—a nice emergent property. However, the paper claims this "provides a principled mechanism" for OOD detection (line 273) without any quantitative evaluation (no AUROC, no baseline comparison). The observation is interesting but the OOD claim is unsupported.

### Trivial
None.

## Nice-to-Haves

- Integration or preliminary experiment with PEFT methods (LoRA, adapters), given that PEFT is the dominant fine-tuning paradigm for large models.
- Per-task accuracy breakdown for the merge comparison (Table 3) to reveal whether the approximation error concentrates on specific tasks.
- Validation of the merge heuristic with more than 8 tasks to directly test whether O(1) complexity holds at scale.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **"The non-linear FT baseline at α=1 achieves only 32.0 on ViT-B/32, below the pre-trained model (48.4), making the relative improvement of TAK less informative."** (Harsh Critic, Sec. 4 Table 1 note): This observation is accurate but does not constitute a weakness of TAK. The severe interference in naive non-linear FT is a well-known problem that motivates both linearized FT and regularization-based approaches. TAK's improvement in this regime is meaningful regardless of how poorly the unregularized baseline performs.

- **"MC=1 is the default for main results but not stated upfront."** (Harsh Critic, Sec. 3.3): The paper discusses both Exact and MC variants in Section 3.3 and the KFAC estimation analysis (Fig. 7a) makes the MC=1 choice clear. This is a minor presentation issue, not a methodological concern.

- **"The 'dataless' framing could be misread from the abstract."** (Harsh Critic, Sec. 4): The abstract says "dataless approach" and the body consistently explains it means no data from *other* tasks. This is a standard usage in the privacy/modularity literature and unlikely to genuinely mislead readers.

- **"Missing integration with PEFT methods."** (Harsh Critic, Missing Experiments #4): The conclusion explicitly identifies this as future work. Criticizing its absence is scope creep—the paper's stated scope is task arithmetic with KFAC regularization, not PEFT integration.

- **"Missing related works" suggestions:** Per instructions, I do not have external sources to confirm existence of suggested related works, so these are removed.

- **Formatting/presentation nitpicks** (e.g., "Eq. 3 footnote should be more explicit in main text"): Removed as per formatting nitpick rule.

## Novel Insights

The paper's insight that the Jacobian Gram matrix used for representation drift regularization is an instance of the GGN under squared loss is a clean bridge between two literatures that have been largely separate—optimization (KFAC/GGN) and model merging (task arithmetic). This suggests that other curvature approximation advances (e.g., KFLR, eigenvalue-corrected KFAC, or sketching-based approximations) could be transplanted to the task arithmetic setting, opening a productive research direction. The empirical finding that the merge heuristic (Eq. 8) can sometimes *outperform* the exact O(T) version on larger models (Table 3: ViT-B/16, T5-base) is surprising and hints that the cross-terms introduced by the Kronecker merge may act as implicit regularization rather than pure noise—an observation the paper does not discuss but could be valuable to investigate.

## Suggestions

- Report mean ± std over at least 3 seeds for the main comparison tables (Tables 1–3), since many key differences are sub-percentage-point.
- Test the merge heuristic (Eq. 8) with 16–20 tasks to validate the O(1) complexity claim at the scale where it matters most.
- Quantify the Kronecker merge approximation error (Frobenius norm of the difference per layer) to provide theoretical intuition for when the heuristic works and when it degrades.

## Evaluation

**Originality:** The GGN connection is non-trivial and novel for the task arithmetic setting. Prior work used representation drift regularization but did not connect it to curvature approximation. The merge heuristic (Eq. 8) is pragmatic but lacks theoretical grounding.

**Importance of research question:** Weight disentanglement in task arithmetic is a practically important problem, especially for privacy-preserving and modular deployment of foundation models.

**Claims support:** The core claim of dataless regularization is well-supported. The SOTA and O(1) complexity claims overreach the evidence. The α-robustness claim holds for the linearized regime but is overstated for the non-linear regime.

**Soundness of experiments:** Comprehensive in scope (vision + language, addition + negation, linearized + non-linear) but weakened by lack of error bars and the 8-task ceiling on merge validation.

**Clarity:** Generally well-written. The theoretical derivation is concise and correct. Some claims in the abstract are stronger than the evidence warrants.

**Value to community:** High. The dataless property matters for privacy and modularity, and the theoretical connection opens new directions for importing curvature approximation tools into model merging.

## Calibration

**Anchors retrieved:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| "When is Task Vector Provably Effective?" (task vector theory) | vRvVVb0NAz | 7.50 | More theory-heavy with generalization guarantees; this paper has less depth of theory but more practical methodology |
| "Safety-Aware Subspace Merging" (model merging) | dqMqAaw7Sq | 7.00 | Practical contribution to merging; this paper has a stronger theoretical grounding and broader evaluation |
| "Meta-CL Revisited: Hessian Approximation" (curvature + regularization) | TpD2aG1h0D | 8.67 | Similar pattern of connecting curvature to regularization; much deeper theoretical unification, got oral |
| "K-FAC Parameterization" (KFAC optimization) | g8sGBSQjYk | 7.33 | KFAC in optimization; this paper applies KFAC to a new problem setting |
| "Task Arithmetic in Trust Region" (task arithmetic conflicts) | q3ztjJRQuJ | 5.75 | Applied task arithmetic; lacks the theoretical depth of this paper |
| "CABS: Conflict-Aware Sparsification" (model merging) | plflYGf23L | 4.75 | Practical but incremental merging; this paper has a clearer theoretical contribution |
| "Delta Parameter Editing" (shallow analysis) | yx8bU8T5ZN | 2.33 | Fundamentally flawed analysis; this paper is far above this level |
| "Graph-based Model Ensemble" (very weak) | 3t57X1Fvaf | 1.50 | Extremely weak; irrelevant comparison |

This paper sits above the medium-scoring task arithmetic papers (5.5–5.75 range: TATR, CABS) which lack the theoretical grounding, and below the top theory-heavy papers (8.67: Meta-CL/Hessian; 7.5: Task Vector Theory) which offer deeper unification. It is comparable to the KFAC optimization paper (7.33) and the safety-aware merging paper (7.0). The genuine theoretical contribution (GGN connection), strong practical results (dataless + competitive performance), and comprehensive evaluation place it solidly in the 6.5–7.0 range. The lack of error bars and the unvalidated O(1) scaling claim pull it slightly below 7.0, but the core contribution is solid.

MY FINAL SCORE: <pineapple>7.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>