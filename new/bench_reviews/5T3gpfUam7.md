Now let me read the calibration papers to score properly.Now I have enough to write the final consolidated review. Let me synthesize across all reviewers.## Summary
The paper proposes **label annealing**, a method that mitigates catastrophic forgetting during LLM finetuning when original pretraining data is unavailable. The approach adds a KL divergence regularization term between the finetuned model's predictions and those of a frozen copy of the initial model, operating purely on the finetuning data distribution. The method is evaluated across four settings (math, code, instruction tuning, niche QA) on Llama 3 8B and is complemented by a theoretical analysis using overparameterized linear regression. 

---

## Strengths

- **Well-motivated practical problem.** Open-weight models (Llama, Mistral) do not release pretraining data, and the inability to replay original data is a genuine and growing constraint. The paper clearly scopes this setting and stays within it.

- **Compelling empirical results, especially in code finetuning.** Table 2 is striking: direct finetuning on code causes a catastrophic MATH collapse from 15.92 → 1.19 (likely due to loss of few-shot formatting ability), while label annealing recovers it to 17.16 with only modest HumanEval loss (51.06 vs. 54.53). The magnitude of this effect is too large to be explained by hyperparameter selection alone.

- **Breadth of evaluation.** Four distinct finetuning settings (math, code, instruction tuning, niche QA) with appropriate target/source benchmarks per setting give confidence that the results are not cherry-picked from a single favorable scenario.

- **Smooth Pareto frontier visualization.** Figures 2–3 show the full sweep over λ values, allowing practitioners to navigate the target/source tradeoff, rather than reporting only a single selected point.

- **Honest limitations section.** Table 3 in Section 5 acknowledges that RedPajama replay matches or beats label annealing in the math setting, positioning the method correctly as a practical fallback when replay is unavailable—not a universal best solution.

- **Theoretically grounded intuition.** Theorem 1 provides a clean geometric characterization of why direct finetuning discards pretraining information within the finetuning span, while label annealing preserves a convex combination of pretrained and finetuning information in that span. Even as a toy model, this adds genuine conceptual clarity over ad-hoc intuitions.

---

## Weaknesses

### Fatal
*None identified.*

### Major

- **Hyperparameter selection performed on the reported evaluation benchmarks.** Section 3.1 explicitly describes the selection protocol: filter λ (and T) to ensure target benchmark improvement, then pick the configuration with best source benchmark performance—applied identically for L2 and label annealing. The same target and source benchmarks are then directly reported in Tables 1–2 and Figures 2–3. This means the reported numbers are the outcome of test-set-coupled model selection, not unbiased generalization estimates. For tables 1–2, where only a single point per method is reported, this introduces optimistic bias. The smooth Pareto curves in Figures 2–3 (showing the full λ sweep) are less affected, but their absolute positions relative to baseline are still tuned. The paper should either report on held-out benchmarks not used for selection or, at minimum, report results across the full hyperparameter sweep rather than the best selected configuration.

- **Framing of L2 regularization as uniformly ineffective is overstated.** The abstract and Section 3 repeatedly describe L2 as providing "little to no benefit," but Table 2 shows L2 recovering MATH from 1.19 (direct) to 11.36—a substantial improvement, though still below label annealing's 17.16. Section 4.3 says L2 "admits no clean intuition why it would help," which is inconsistent with the empirical evidence in Table 2. A more careful framing would characterize label annealing as *systematically* better than L2, not as L2 being effectively useless.

- **Missing Pareto curve for L2 regularization in the alignment experiments.** Figures 2–3 compare label annealing's full sweep of λ values against the single "best" L2 point and the direct finetuning point. Without plotting L2's full tradeoff frontier, it is impossible to judge whether label annealing's Pareto curve actually dominates L2's, or whether they overlap. This is the most important missing comparison for evaluating the paper's central claim.

### Minor

- **Computational overhead not discussed.** Label annealing requires a full forward pass through a frozen copy of the model for every training step, effectively doubling activation memory and adding ~1× forward-pass compute. For an 8B model under multi-GPU training this is non-trivial overhead. The paper provides no analysis of wall-clock cost, memory footprint, or how this scales to larger models (e.g., 70B). This omission reduces the paper's practical utility.

- **Temperature parameter T is unexplored empirically.** Temperature is presented as a first-class component of the method (Eq. 2), and the paper discusses its semantic role (makes distributions more/less peaked). But no ablation over T holding λ fixed is presented. It is unclear whether T provides value over T=1 (standard KL), or whether the method is robust to T. Practical guidance is absent.

- **In-context learning (ICL) claim is unverified.** The abstract specifically mentions "forgetting certain capabilities (e.g., in-context learning ability)" and this is used to motivate the code finetuning MATH collapse ("model loses its ability to follow few-shot prompts"). No ICL evaluation is presented; the diagnosis is anecdotal, even if plausible.

- **Limited comparison with stronger continual-learning baselines.** The paper compares only against direct finetuning and L2-to-initialization. EWC (Kirkpatrick et al., 2017) is the canonical data-free weight-regularization approach and uses Fisher information to prioritize important parameters; the paper mentions EWC in passing and describes L2 as "a simplified case of EWC" but does not run it. In the weight-only setting, EWC is the natural stronger baseline to include.

- **KL direction unjustified.** Equation (2) uses KL(p_θ,T ∥ p_θ₀,T), the "forward" KL from finetuned to pretrained. Conventional knowledge distillation uses KL(teacher ∥ student), the reverse direction. The paper does not motivate this choice theoretically or empirically, nor does it test whether reversing the direction matters.

### Trivial

- **"T→∞ reduces to label smoothing" is imprecise.** As T→∞, both p_θ,T and p_θ₀,T approach the uniform distribution, so KL(p_θ,T ∥ p_θ₀,T) → KL(uniform ∥ uniform) = 0—the regularization vanishes rather than recovering standard label smoothing. The conceptual analogy (reference distribution becomes uniform, as in label smoothing) is valid, but stating this is a formal equivalence is incorrect.

---

## Nice-to-Haves

- **Test on ≥ 1 additional model or scale (e.g., Llama 3 70B).** The single-model evaluation leaves open whether findings scale. Demonstrating on a larger or different model family would strengthen generalizability claims and also reveal whether compute overhead grows prohibitively.
- **Compare or combine with LoRA.** LoRA is a widely used practical finetuning method that may naturally mitigate forgetting by restricting parameter changes. Whether label annealing is complementary to LoRA or redundant with it is a natural practitioner question.
- **More detailed discussion of replay interaction.** Table 3 shows replay + label annealing slightly outperforms replay alone. A more thorough exploration of when the combination helps and whether label annealing adds consistent value over replay across all settings would refine the method's positioning.
- **Statistical reliability.** No confidence intervals or multi-seed results are reported. For the alignment scatter plots, some error bars appear in Figure 3 but are not explained. Reporting at least two seeds for the tabular results would help readers assess whether smaller differences are reliable.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

1. **"Critical Issue #1: Structural invalidation of all core empirical claims"** (Harsh Critic). While the hyperparameter selection protocol is a real concern (kept as a Major weakness), the characterization that it "invalidates" all core claims is excessive. The methodology is standard practice in the field (model selection using dev-set metrics), and the magnitude of several results—particularly the code finetuning MATH collapse and recovery—is far too large to be explained by selection effects. Downgraded to a Major weakness with appropriate framing.

2. **"T→∞ reduces to label smoothing is too loose and stated as equivalence"** (Harsh Critic). This is a legitimate imprecision (kept as Trivial) but not a falsification of the method's design or effectiveness.

3. **"No variance / confidence intervals for Tables 1–2"** (Harsh Critic, Human Finder). Kept as a Nice-to-Have; single-run evaluation is the norm for large-scale LLM benchmarks. Removed as a stand-alone weakness.

4. **"Synthetic data quality concerns / potential biases in the GPT-4o-mini generated corpus"** (Human Finder). This is a generic concern applicable to any synthetic-data paper; it is not grounded in any specific artifact observed in the results.

5. **"Novelty of the core technique is insufficient for this venue"** (Neutral Reviewer, Human Finder). The application of KL regularization from a frozen model to open-weight LLM finetuning without pretraining data is a clearly motivated and useful contribution, even if the individual technical components are known. Novelty concerns are noted but not treated as disqualifying.

---

## Novel Insights

The most genuinely novel observation surfaced by this review cycle is the *asymmetric treatment* of finetuning data in direct finetuning versus label annealing: the linear theory in Section 4 shows that direct finetuning geometrically erases pretrained information *specifically within the span of the finetuning data*, while preserving information orthogonal to it. Label annealing instead maintains a convex interpolation in both subspaces simultaneously. This offers a clean theoretical vocabulary for understanding not just "how much" forgetting occurs but "where" in representation space it occurs—a framing that generalizes beyond the specific method proposed and could inform future analyses of fine-tuning dynamics. The code finetuning result (near-total MATH collapse driven by loss of few-shot formatting ability rather than mathematical knowledge) is also a notable empirical finding in its own right, independently of label annealing.

---

## Suggestions

1. **Report held-out benchmark results** (or clearly label current results as "selected" results) to address the hyperparameter-selection-on-eval concern.
2. **Plot L2 regularization's full Pareto curve** in Figures 2–3 with the same λ sweep used for label annealing—this is the single most impactful experiment for fairly evaluating the method's advantage.
3. **Ablate temperature T** independently from λ in at least one setting (e.g., code finetuning) to establish whether T provides value beyond T=1.
4. **Quantify computational overhead**: report wall-clock time per training step and GPU memory for direct finetuning vs. label annealing on the 8B model.
5. **Add EWC baseline** in the math or code finetuning setting to confirm that the advantage over weight-space regularization holds beyond the weakest possible baseline.
6. **Tone down L2 characterization** to reflect that L2 provides substantial but incomplete mitigation in Table 2, rather than "little to no benefit" across the board.

---

## Score and Decision

**Calibration:**

| Comparison Paper | Topic | Score | Decision |
|---|---|---|---|
| tmsqb6WpLz – "Dissecting learning and forgetting in LM finetuning" | LLM forgetting analysis | 8,5,5,5 (avg 5.75) | Accept (poster) |
| VrHiF2hsrm – "Catastrophic Forgetting via Implicit Inference" | LLM forgetting theory+empirics | 3,8,6,6 (avg 5.75) | Accept (poster) |
| YeErX16hMC – "Mitigating Forgetting in LLM SFT" | LLM forgetting mitigation | 3,8,6,5 (avg 5.5) | Reject |
| VQ7Q6qdp0P – "Fine-tuning can cripple foundation models" | Distillation-based forgetting mitigation | 3,8,5,3 (avg 4.75) | Withdrawn |

**Reasoning:** This paper is stronger than the rejected YeErX16hMC (which had theoretical errors and a less clear problem definition) and the withdrawn VQ7Q6qdp0P (which addressed a less impactful setting with CLIP and had weaker novelty framing). It is comparable to the accepted tmsqb6WpLz and VrHiF2hsrm: similar scope and quality of experimental analysis, clear problem motivation, and some theoretical backing. However, unlike tmsqb6WpLz (which provided deep mechanistic analysis) or VrHiF2hsrm (which provided a clean falsifiable theory), this paper's primary value is practical rather than conceptual. The hyperparameter-selection concern (Major weakness) and missing Pareto curve for L2 in the alignment experiments are the primary factors keeping it from a confident accept. The practical value, breadth of experiments, and the striking code finetuning result collectively support acceptance.

**Evaluation on key axes:**
- **Originality:** Moderate — KL from frozen model is a known technique; the framing and application are well-motivated but incremental.
- **Importance of research question:** High — finetuning open-weight LLMs without pretraining data is a real and growing practical challenge.
- **Claim support:** Moderate — main results are directionally robust but quantitatively optimistic due to selection protocol.
- **Soundness of experiments:** Moderate — four diverse settings, but key comparisons (L2 Pareto curve, EWC) missing.
- **Clarity of writing:** Good — well-structured, the method and results are clearly presented.
- **Value to research community:** Good — simple method, easy to implement, addresses a constraint facing every practitioner using open-weight models.

**Final Score: 6.0** — Weak accept. The paper clears the acceptance bar on practical value and experimental breadth, but not by a large margin.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>