## Summary

The paper proposes Forget-to-Focus (F2F), a two-stage protocol that first applies targeted unlearning on a "forget set" of general-domain data (with an optional "retain set" for stability), then fine-tunes on a domain-specific dataset. Through experiments on medical, mathematics, and coding benchmarks across models from 0.6B to 72B parameters, the authors demonstrate that F2F consistently outperforms standard fine-tuning, DAPT, LoRA, and CurlLoRA baselines, and provide representational geometry analysis (CKA, SVCCA, Fisher information, PCA-shift) arguing that unlearning reshapes models away from generalist features toward domain-specialized structures.

## Strengths

- **Novel repurposing of unlearning for domain adaptation:** The insight that machine unlearning—traditionally a privacy tool—can serve as a preparatory capacity-reallocation mechanism for specialization is conceptually original and directly addresses the negative transfer problem. This reframing is the paper's core contribution and is well-motivated.

- **Empirical breadth across models, scales, and domains:** The paper evaluates on 5+ model architectures (0.6B–72B), three distinct domains (coding, medical, math), and multiple fine-tuning strategies. The gains are substantial and consistent: e.g., Qwen-0.6B HumanEval pass@1 improves from 19.50 (base SFT) to 42.07 (F2F); Qwen-72B HumanEval from 70.12 to 78.50 (Table 1). Multi-seed robustness (Table 9) with small standard deviations strengthens confidence.

- **Rich mechanistic analysis beyond accuracy:** The CKA/SVCCA analysis (Figures 4–5), Fisher information profiling (Figure 7), PCA-shift analysis (Figure 6), and spectral surrogate analysis for LoRA capacity (Figure 9) collectively provide a multi-faceted view of *how* unlearning alters internal representations. The finding that F2F dampens shallow-layer Fisher sensitivity while maintaining depth-wise activity is particularly interesting.

- **Calibration improvement on safety-critical tasks:** The ECE reduction from 0.277 (base tuned) to 0.050 (F2F) on MedMCQA (Table 7, Figure 8) is a practically significant finding for deployment in medical settings. The reliability diagrams confirm this is not simply a confidence collapse.

## Weaknesses

### Major:

- **Lack of compute-matched controls undermines attribution of gains:** F2F performs two sequential training phases (unlearning + fine-tuning), while baselines like SFT perform only one. The paper does not demonstrate that F2F's gains persist when compared against baselines given equivalent total optimization steps, GPU-hours, or FLOPs. It is possible that the extra gradient updates from the unlearning phase—rather than the *unlearning mechanism itself*—drive the improvements. A compute-matched SFT baseline (e.g., SFT trained for additional epochs or with a larger effective batch size to match F2F's total compute budget) is essential to validate the core claim. The paper notes runtime for unlearning is ~0.55 GPU-hours (Section C.1), but never integrates this into a fair comparison.

- **"Stable optimization dynamics" claim contradicted by small-model instability:** The abstract and conclusion assert that F2F yields "more stable optimization dynamics." However, Table 1 shows Gemma-2B-Instruct collapsing to 0.00% pass@1 after the UnlGA+GD phase, and Table 3 shows several configurations where intermediate unlearning produces extreme degradation. While retuning recovers performance, the unlearning stage itself is unstable for smaller-capacity models. The paper should explicitly qualify the stability claim and analyze under what conditions (model capacity, forget-set quality, σ/λ settings) the method becomes volatile.

### Minor:

- **Inconsistency between default hyperparameters and ablation findings:** Section 3.4 specifies λ=1.0 (GA weight) and σ=0.5 (GD weight) as defaults, but Appendix A.10 finds λ=0.5 is optimal for accuracy improvement, with λ=1.0 "severely limiting improvement." This discrepancy is not discussed. If the best-performing configuration differs from the reported defaults, the tables may not reflect the strongest version of F2F, or the defaults need justification.

- **Theoretical proposition relies on assumptions that do not hold for LLMs:** The Proposition in Section 2 assumes orthogonal decomposition of parameter space into domain-relevant (V) and irrelevant (U) subspaces, strong convexity of L_D, and β-smoothness. The paper acknowledges using "a convex linear surrogate to clarify the mechanism," but then the Corollary claims concrete convergence rate implications for the retuning phase. The gap between the simplified model and non-convex Transformer optimization is too large for the theoretical section to provide actionable guarantees; it would be more honest to frame it as intuition-building rather than formal analysis.

- **CKA/SVCCA drift may reflect optimization trajectory length, not unlearning specifically:** The representational analysis shows F2F drifts further from the base model than standard fine-tuning does (Figure 4). However, since F2F involves additional optimization steps, greater drift is expected simply from longer training. Without a compute-matched control, the attribution of geometric shifts to *unlearning* (rather than to *additional optimization*) is unconvincing. Correlating the magnitude of CKA drift with downstream accuracy would strengthen the causal claim.

- **Missing convergence dynamics evidence for "stabler optimization" claim:** The paper repeatedly claims F2F produces "more stable optimization dynamics" and "stabler optimization," but provides no training loss curves, convergence rate comparisons, or gradient norm trajectories during the fine-tuning phase. This is a straightforward experiment that would directly support the claim.

### Trivial:

- **Calibration analysis limited to medical domain only:** The improved calibration (ECE, reliability diagrams) is demonstrated only for MedMCQA. Whether F2F improves calibration on coding or math benchmarks is untested, leaving open whether this is a general property or domain-specific.

## Nice-to-Haves

- A random/naïve forget set control (e.g., random text or noise) to validate that the gains require meaningful forget-set curation rather than just an extra training perturbation.
- Calibration analysis extended to coding and math domains.
- Systematic sweep of target-domain contamination percentage in the forget set (beyond the 200/1000 BC-Mixed split) to quantify robustness to imperfect forget-set curation.
- Parameter-efficient unlearning (applying the unlearning phase only to LoRA adapters) to reduce the computational overhead.

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Weakness: Baseline SFT performance may be under-optimized for Qwen-0.6B.** The reviewer speculated that SFT HumanEval of 31.71 is "relatively low compared to literature," but provides no concrete evidence of this, and the paper uses standard SFT recipes with the same learning rate and optimizer across methods. Without verified numbers from an external source, this is speculative.

- **Weakness: Qwen-72B uses QLoRA while 0.6B uses full fine-tuning, making comparison misleading.** These are different models evaluated independently; no cross-model comparison of absolute scores is being claimed as a methodological finding. The within-model improvements are the relevant metric.

- **Weakness: Missing related work on other domain adaptation or unlearning methods.** Per review rules, I cannot confirm the existence of specific uncited works and should not flag missing references.

- **Weakness: Reproducibility concerns about hyperparameters and implementation details.** The paper specifies learning rates, batch sizes, epoch counts, weight settings, and provides code. This level of detail is standard.

- **Weakness: Abstract should hint at trade-offs (compute vs. performance).** This is a stylistic preference, not a substantive flaw.

- **Weakness: BC-Cosine introduces dependency on MiniLM encoder and similarity threshold.** The paper explicitly describes this method and shows it performs comparably to BC-Select (Table 3). The dependency is acknowledged and is not a hidden limitation.

- **Weakness: Societal risk of reduced robustness to adversarial attacks.** This is outside the paper's stated scope of domain adaptation performance.

## Novel Insights

The most striking observation across the reviews and paper is the tension between F2F's framing as a "principled, stable" intervention and its empirical behavior as a *high-variance* perturbation that requires careful stabilization. The method's success seems to stem less from surgically removing "irrelevant" knowledge and more from creating a perturbed initialization that escapes the pretraining basin—functionally similar to warmup or learning rate restart strategies. The fact that BC-Cosine (automatic, similarity-based) matches BC-Select (manual curation) suggests the forget set's content matters less than its *directional opposition* to the target domain in embedding space, hinting that F2F may be fundamentally about constructing an optimization landscape reset rather than targeted knowledge erasure. This reframing would also explain why smaller models (Gemma-2B) collapse: they lack the parameter capacity to absorb a large perturbation and still recover via retuning.

## Suggestions

- **Add a compute-matched SFT baseline:** Train standard SFT for the same total number of gradient steps as F2F (unlearning + fine-tuning combined) and report results. This single experiment would either validate or invalidate the core attribution claim.
- **Plot fine-tuning loss curves:** Show training loss during the retuning phase for F2F vs. standard SFT to directly substantiate the "stabler optimization dynamics" claim.
- **Reconcile λ defaults with ablation findings:** Either justify why λ=1.0 was used as the default despite λ=0.5 being optimal in ablations, or re-report main tables with the best λ setting.
- **Qualify the stability claim:** Explicitly state in the main text that F2F can be unstable for low-capacity models and that the retain mechanism (σ > 0) is critical for smaller architectures.
- **Correlate CKA drift magnitude with downstream accuracy:** If models that drift more perform better, the mechanistic story is strengthened; if there is no correlation, the representational analysis is merely descriptive.