## Summary

Cost-sensitive Multi-fidelity BO (CMBO) proposes a framework that reframes hyperparameter optimization from maximizing asymptotic validation performance under a fixed budget to maximizing a user-defined utility function that trades off BO performance against computational cost. The method introduces a utility-aware EI acquisition function with a dynamically chosen optimization horizon, a probabilistic stopping criterion interpolating between regret-based and probability-of-improvement signals, and a transfer learning scheme for Prior-Fitted Networks (PFNs) based on learning curve (LC) mixup across datasets and configurations. Extensive evaluation across LCBench, TaskSet, PD1, and a collected real-world object-detection dataset shows consistent and substantial improvements over multi-fidelity BO and transfer-BO baselines.

---

## Strengths

- **Genuinely novel problem framing for multi-fidelity BO.** Existing freeze-thaw methods (DyHPO, iFBO, DPL) optimize either greedy one-step EI or asymptotic performance at a fixed horizon. Reformulating the objective as maximizing a user utility U(b, ỹ_b) over the BO trajectory—and deriving both acquisition and stopping from this objective—is a conceptually clean and practically relevant departure. The dynamic horizon selection (max over Δt in Eq. 2) is a direct consequence of this framing and is not present in prior work.

- **LC mixup preserving inter-configuration correlations.** The two-stage mixup—across datasets first with a shared λ₁ applied to all configurations, then across configurations—is a simple yet thoughtful technique. Using a single shared λ₁ in the first stage explicitly preserves the correlation structure encoded in each dataset's LC matrix L_m. Fig. 6 directly demonstrates that the mixup reduces overfitting of the PFN surrogate and translates to improved BO regret, providing concrete evidence for the mechanism rather than just end-to-end performance.

- **Coherent interpolation between two extreme stopping rules.** The BetaCDF(p_b; β, β)^γ formulation in Eq. (4) provides an interpretable one-parameter family: β→0 recovers the regret-only threshold used by baselines (δ_b = 0.2 when γ = log₂5), while β→∞ recovers a hard PI-based threshold. The smooth interpolation at β = e⁻¹ is well-motivated and tested across all three benchmarks in Fig. 7d.

- **Empirical breadth and real-world validation.** The method is tested on four distinct LC benchmarks spanning tabular classification (LCBench), diverse NLP tasks (TaskSet), large-scale vision and biology tasks (PD1), and a self-collected object detection dataset with three heterogeneous architectures. Multiple utility function shapes (linear, quadratic, square root, staircase, estimated) are evaluated, and Table 2 shows robustness across all of them with CMBO achieving rank 1.0 in every setting.

---

## Weaknesses

### Fatal
None.

### Major

- **Table 3 ablation inconsistency undermines the incremental gain narrative.** Rows 3 and 4 in Table 3 are both labeled (p_b ✓, Acq. ✓, T. ✓) yet produce very different results (e.g., 4.4 vs. 0.9 for α=2e-04). The text claims "performance improves sequentially as each component is added," implying four distinct configurations, but only three distinct checkmark patterns appear (missing one intermediate ablation row). This makes it impossible to cleanly attribute gains to the stopping criterion vs. acquisition vs. transfer learning, which is a core claim of the paper. At minimum, one row is mislabeled, and the correct labels must be provided for the ablation to be interpretable.

- **Stopping-criterion confound in baseline comparisons.** As the paper notes in footnote 2, the PI-based term in Eq. (5) depends on the utility-aware acquisition, so baselines cannot directly use the same stopping rule. This is a principled justification, but it means the cost-sensitive results in Table 1 and Fig. 5 conflate: (i) better configuration selection from the utility-aware acquisition, and (ii) better stopping from the mixed PI+regret criterion. The ablation in Table 3 shows the stopping criterion alone contributes substantially (4.4→0.9 at α=2e-04), yet the baseline comparison gives baselines only the inferior regret-only stopping. A cleaner decomposition—e.g., showing CMBO's acquisition with regret-only stopping vs. full CMBO—would make the contribution boundaries clearer. The dotted "achievable regrets without stopping" lines in Fig. 5 are helpful but insufficient to resolve this.

- **Uniform per-step cost assumption undermines the motivating scenario.** The utility U(b, ỹ_b) is defined over BO steps b and evaluated over "total epochs spent." This implicitly assumes each BO step (each epoch evaluation) has identical cost. The paper's motivating examples invoke cloud credits and Slurm allocations, where wall-clock cost is the relevant resource. The real-world object detection experiment includes ResNet-50, HRNet, and MobileNetV2 evaluated jointly, which almost certainly have different per-epoch wall-clock costs. The mismatch between the motivation (heterogeneous wall-clock costs) and the formulation (step counting) is a genuine gap that goes unacknowledged in the main text.

### Minor

- **Utility elicitation is empirically under-validated.** The Bradley-Terry preference model is demonstrated only via synthetic recovery in Fig. 2 (1,000 pairwise labels, no sensitivity to fewer/noisier labels). In the main experiments, all reported results use predefined utility functions (linear, quadratic, etc.). The single "Estimated" row in Table 2 constructs preferences from iFBO's trajectory assuming "the user wants a better tradeoff than iFBO"—this is an artificial proxy for user preference, not evidence that the end-to-end pipeline (elicitation → BO → stopping) works in practice.

- **Algorithm 1 notation inconsistency.** Line 4 reads: n* ← argmax_{n∈C} A(n), where C = {(x, t, y)} is the history of partial LC observations. At initialization C = ∅ (line 2), making this argmax undefined. Furthermore, the text in §3.1 says "we predict for all x∈X the remaining part of the LCs," implying the argmax should range over the full configuration pool X, not C. This discrepancy should be corrected for reproducibility.

- **Notation inconsistency between Eq. (2) and Eq. (5).** Eq. (2) uses ỹ_{b+Δt} (tilde-y, the best-so-far BO performance), while Eq. (5) uses ȳ_{b+Δt} (bar-y). Whether these are identical quantities should be clarified explicitly, as the distinction between the running best performance (line 10: ȳ_b = max(ȳ_{b-1}, y_{n*,t_{n*}})) and the extrapolated BO performance matters for the stopping criterion computation.

- **ESBO baseline is undefined.** ESBO appears in Tables 2 and 4 but is not described anywhere in the baselines section (§4). Its definition, source, and relationship to CMBO (it appears to be a strong baseline in Table 4) must be provided; its absence from Table 1 also suggests it is not applicable in all settings, which should be explained.

- **γ parameter is fixed without sensitivity analysis.** β is ablated in Fig. 7d across all three benchmarks, but γ is fixed at log₂5 (corresponding to δ_b = 0.2) without any analysis. As γ and β jointly determine the stopping behavior, a sensitivity test on γ is warranted.

### Tiny

- **Mixup validity for discrete/categorical hyperparameters is not discussed.** Convex combinations of configuration vectors (step 2 of the mixup) may produce invalid hyperparameter settings when some hyperparameters are categorical or integer-valued. The paper should at minimum state that this is applied only to continuous hyperparameters or discuss how categorical cases are handled.

- **Key PFN architecture details are deferred to appendices (§E, §G) that are not available in the main text.** Architecture size, tokenization of partial LCs, number of meta-training examples, and inference procedure are relevant for assessing the method's practical overhead and reproducibility.

---

## Nice-to-Haves

- **EI/cost as a baseline.** The standard cost-aware acquisition divides EI by expected evaluation cost; including it (even as a black-box surrogate variant) would clarify whether the utility formulation offers advantages beyond this simpler cost-weighting approach.

- **Utility trajectory visualization with oracle stopping point.** Showing U(b, ỹ_b) over BO steps for CMBO and baselines, with the actual stopping point marked and the oracle b* indicated, would directly demonstrate whether the stopping criterion is well-calibrated.

- **Sensitivity to utility misspecification.** A brief analysis of how CMBO degrades when the estimated utility deviates from the true utility (e.g., wrong penalty weight α) would quantify the practical risk of the elicitation approach.

- **Wall-clock time experiment.** One experiment with actual per-configuration compute time (rather than epoch counting) would validate the cost-sensitivity claim in a realistic heterogeneous-cost setting.

---

## Removed Points

*These points were flagged for removal; treat with caution.*

- **Zero variance (±0.0) as a credibility concern.** Quick-Tune† and FSBO are deterministic methods; ±0.0 is expected and not suspicious. The paper explicitly uses 30 runs only for methods with large variance.

- **β ablation limited to PD1 only.** This misreads Fig. 7d, which plots normalized regret vs. β for LCBench, TaskSet, PD1, and Average simultaneously, with asterisks marking optima for each. The ablation covers all three benchmarks.

- **Demand for theoretical guarantees.** This is an empirical systems paper in the freeze-thaw BO tradition; no prior competing method provides theoretical stopping or regret guarantees, and demanding them would impose a non-standard bar.

- **Finite configuration pool being too restrictive for "general BO."** All freeze-thaw BO methods operate in this setting (DyHPO, iFBO, DPL). The paper targets tabular HPO benchmarks where this is the standard setup; criticism for not covering continuous-space BO is scope creep.

- **Criticism about Quick-Tune† modification being unfair.** The modification removes the model-selection component to isolate the transfer learning mechanism, which makes the comparison fairer for the hyperparameter selection task—the baseline is weakened in a way that benefits it (more compute per HP eval), not the proposed method.

- **Concern about comparison to methods not yet released / not existing.** The paper cites iFBO, Quick-Tune, DPL, etc.—these are assumed to exist.

---

## Novel Insights

The acquisition function's dynamic horizon selection (Eq. 2: max over Δt) induces a principled behavioral transition from non-greedy to greedy over the course of BO—not by scheduling but as a direct consequence of cost-dominated utility. Fig. 7b visualizes this concretely: early BO steps select large Δt (look-ahead), while late steps collapse to Δt≈0 (myopic exploitation) as the cost term dominates. This is a cleaner explanation of the exploration-exploitation transition in cost-sensitive BO than ad hoc schedule designs, and the analysis in Fig. 7c shows the resulting configuration-selection concentration matches intuition. The LC mixup's cross-dataset shared λ₁ to preserve inter-configuration correlation is a subtle but nontrivial design choice that distinguishes it from naive per-curve augmentation and deserves attention from the broader PFN training community.

---

## Suggestions

1. **Fix Table 3**: Identify and correct the mislabeled row so the four ablation rows correspond to four distinct component combinations, enabling clean attribution of gains to stopping criterion, acquisition function, and transfer learning separately.

2. **Clarify Algorithm 1 initialization**: Specify how the first configuration is selected when C = ∅, and correct the domain of the argmax in line 4 from n∈C to n∈[N] (or n∈X) to match the described procedure.

3. **Address uniform-cost assumption explicitly**: Add a paragraph acknowledging that U(b, ỹ_b) treats per-step cost as homogeneous, and either justify this for the benchmarked settings or describe a simple extension to variable per-step costs (e.g., replacing step count b with cumulative wall-clock time).

4. **Define ESBO**: Add a description of the ESBO baseline in §4, clarify its relationship to CMBO, and explain why it appears only in Tables 2 and 4.

5. **Add stopping criterion decomposition**: Report CMBO with regret-only stopping (β→0) alongside baselines with the same stopping rule, to isolate acquisition quality improvements from stopping policy improvements in the cost-sensitive setting.