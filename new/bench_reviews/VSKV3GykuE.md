Now let me search for calibration anchors.Now I have enough information to write the final review. Let me synthesize.---

## Summary

RAC-LoRA proposes a new parameter-efficient fine-tuning framework that chains randomized asymmetric LoRA blocks, fixing one matrix randomly per step and optimizing only the other. The key insight is that this reduces each chain step to a randomized preconditioned gradient descent update on the original smooth loss, enabling the first provably convergent LoRA-type method that actually models low-rank updates (as opposed to COLA's theory which replaces them with full-rank updates). The paper proves O(1/T) convergence to a stationary point in the non-convex setting and linear convergence under the Polyak–Łojasiewicz (PL) condition, covering GD, SGD, Random Reshuffling, and a federated learning extension.

---

## Strengths

- **Concrete counterexample motivating the framework (Section 3, Figure 1):** The 3×3 quadratic with M = Diag(10,1,...,1) cleanly and reproducibly shows that LoRA and COLA diverge at the theoretically correct step size η = 1/L, while AsymmLoRA converges to a sub-optimal stationary point. This is the strongest motivating piece of the paper.

- **First provably convergent LoRA-type method that preserves low-rank structure (Theorems 5.3 and 5.5):** Section 2.2 correctly identifies that COLA's existing theory replaces low-rank optimization with a full-rank ΔW analysis, making it irrelevant to what the algorithm actually does. RAC-LoRA's convergence proof works directly in the (B_S, Â) parameterization. Theorem 5.3 gives E[‖∇f(W̃^T)‖²] ≤ 2(f(W^0)−f*) / (λ_min^H γ T) and Theorem 5.5 gives linear convergence under PL—both with explicit dependence on rank through λ_min^H = r/n under isotropic sampling.

- **Remark after Assumption 5.1 (Section 5.2):** The derivation that isotropic sampling gives E[H] = (r/n)I via rotational invariance is unusually concrete, provides an easily satisfiable sufficient condition, and makes the convergence rate's dependence on rank directly interpretable. Setting r = n recovers standard GD.

- **Table 3 (MNIST, Section 6.2.2):** By matching trainable parameter budgets and intentionally restricting capacity (rank 1, pre-train on digits 0–4, fine-tune on 5–9), the MNIST experiment isolates the effect of chaining. RAC-LoRA (Gaussian/Zero) achieves 92.0% vs. AsymmLoRA's 62.3% at identical 133-parameter budget—a 30-point gap. This is the most unambiguous empirical evidence that the chaining procedure delivers genuine benefit.

- **Broad optimizer and setting coverage (Table 1):** Results for GD, SGD, Random Reshuffling, and Fed-RAC-LoRA under both non-convex and PL regimes provide a complete theoretical picture.

---

## Weaknesses

### Fatal
None.

### Major

- **NLP experiments are empirically unconvincing and the LoRA baseline appears misconfigured.** Table 2 is the paper's main real-world benchmark. RAC-LoRA averages 77.0 across four GLUE tasks, below both LoRA (78.5) and COLA (77.6). More critically, the authors' LoRA replication achieves 75.2 on RTE, compared to 86.6 in the LoRA* row (taken directly from Hu et al., 2021). This 11-point gap on a single task is large enough to suggest that the authors' experimental setup is not replicating the baselines correctly—different epoch counts (100 vs. task-specific 30–80), learning rate schedules, or rank settings. Because all comparisons in Table 2 (including COLA and RAC-LoRA) are run under the same potentially suboptimal configuration, the validity of the relative comparisons is compromised. The paper acknowledges underperformance candidly, but the explanation ("these tasks don't require extra capacity") is post-hoc and not validated by any diagnostic.

- **No experiments at scales where LoRA is practically deployed.** The largest model tested is RoBERTa-base (125M parameters). The paper's motivating application is LLM fine-tuning, but no experiment touches 1B+ parameter models, instruction tuning, commonsense reasoning, or code generation—the settings where LoRA is most practically relevant. Without at least one demonstration that chaining improves over single-block AsymmLoRA in an NLP setting where FPFT clearly outperforms LoRA, the core empirical premise of the paper (RAC-LoRA bridges LoRA and FPFT) has no real-world anchor beyond a contrived MNIST split.

### Minor

- **Abstract overclaims convergence guarantee without qualification.** The abstract states "We provide provable guarantees of convergence to the same solution as FPFT" without qualification. In the general non-convex setting, Theorem 5.3 only guarantees convergence to a stationary point (E[‖∇f‖²] → 0), which may be a local minimum or saddle point far from FPFT. Convergence to f* (same as FPFT) is only established under the PL condition (Theorem 5.5). The paper's contribution bullet in Section 2.3 correctly qualifies this distinction, but the abstract creates a misleading impression of the general result.

- **No explicit connection to sketch-and-project gradient descent literature.** The RAC-LoRA update W^{t+1} = W^t − γ H_B^t ∇f(W^t) (Equation 3) with a random projection matrix H_B^t is structurally identical to sketch-and-project gradient descent. The convergence rates in Theorems 5.3 and 5.5 (O(1/T) non-convex, linear under PL, with rate degraded by λ_min^H = r/n) parallel known results in that literature. The paper does not position its theoretical contribution relative to this body of work or articulate what mathematical novelty the theorems add beyond the interpretation of LoRA as randomized projection descent. If the novelty is primarily the connection itself, that remains a genuine conceptual contribution but should be framed explicitly.

- **Theory–practice gap: GD theory, AdamW practice.** All convergence theorems (Section 5.2 and Appendices C–F) are derived for GD, SGD, and RR. All neural network experiments (Tables 2 and 3) use AdamW. The paper does not discuss whether the sketch-and-project interpretation carries over to adaptive optimizers, leaving the theoretical insights disconnected from the empirical evidence.

### Trivial
None beyond what is noted above.

---

## Nice-to-Haves

- A single experiment on a model ≥1B parameters (e.g., LLaMA-7B on a commonsense benchmark) where FPFT clearly outperforms single-block LoRA—demonstrating that RAC-LoRA chains close this gap—would directly validate the paper's main claim.
- Wall-clock time and memory comparison per chain step vs. single LoRA block, to quantify the overhead of the inner optimization loop.
- Training loss curves for the GLUE experiments to diagnose whether RAC-LoRA is converging stably or to a qualitatively worse fixed point than LoRA.

---

## Removed Points

*These points are flagged as removed. Treat them with caution.*

- **Harsh Critic: "Convergence only with step size η = 1/L"** — The critic notes this is a "theoretically maximal step size that would never be used in practice." This is a fair concern, but the counterexample's purpose is theoretical: to demonstrate loss of Lipschitz smoothness under LoRA parameterization and its algorithmic consequences. The paper does not claim practical divergence at typical step sizes; it demonstrates a theoretical pathology. Removed as scope creep.

- **Harsh Critic: "Exact closed-form A^t minimizer vs. approximate GD-based solution"** — The paper uses δ-approximate solutions in the analysis and this is explicit in Theorem 5.3's statement and proof setup. The distinction between the upper-bound minimizer and the iterative GD solution is standard in proximal analysis. Removed as an existing and reasonable addressal.

- **Harsh Critic: "MNIST is ecologically invalid"** — The MNIST setup is explicitly designed to isolate the chaining effect, as acknowledged in Section 6.2.2. The paper never claims MNIST generalizes to all LLM fine-tuning settings. The use of a controlled experiment for ablation is methodologically sound. Removed as scope creep.

- **Strength Finder: "Honest discussion of GLUE experiments"** — The paper is transparent about the underperformance in Table 2. While this honesty is notable, it does not constitute an independent empirical strength and is subsumed by the major weakness about the NLP results. Removed to avoid conflating candor with contribution.

---

## Novel Insights

The most genuinely novel observation is the identification that fixing one LoRA matrix randomly per chain step (making the update asymmetric) is the precise move that converts a non-smooth joint (B, A) optimization into a smooth, randomized preconditioned gradient descent step on the original loss f(W). This "randomization restores smoothness" insight is clean, original, and explains in one equation why standard LoRA/COLA lack convergence while RAC-LoRA does not. It also provides a principled mechanism by which rank r enters convergence rates (through λ_min^H = r/n), giving a direct theoretical basis for the rank–accuracy tradeoff observed empirically across the LoRA literature.

---

## Evaluation on Key Axes

- **Originality:** Solid. The convergence-restoring randomization trick is genuinely novel within the LoRA literature; no prior work provides convergence guarantees for a method that preserves the actual low-rank update structure.
- **Importance of research question:** High. Convergence theory for LoRA is actively needed and the paper attacks a real gap.
- **Claims well-supported:** Partially. The theoretical claims are well-supported by the mathematical derivations; the practical claims (bridging LoRA and FPFT) are supported only by a contrived MNIST experiment.
- **Soundness of experiments:** Weak. The main NLP benchmark is unconvincing—an apparently misconfigured LoRA baseline, no large-scale LLM, and overall underperformance vs. simpler methods.
- **Clarity of writing:** Good. The paper is clearly structured and the algorithm and main theorems are easy to follow.
- **Value to research community:** Moderate. The theoretical framework is genuinely useful for the community; the practical payoff is not yet demonstrated.

---

## Score and Decision

**Calibration anchors:**
1. *GoLore (udtrtwkvk5)* — avg 5.25, Reject. Most similar paper: convergence failure of existing LLM optimization method + randomized fix + convergence proofs + GLUE/LLaMA experiments. GoLore had NLP fine-tuning at 125M + LLaMA2-7B curves; RAC-LoRA has RoBERTa-base + MNIST. GoLore's experiments were marginally better validated despite being similarly criticized.
2. *MAST (sPuLtU32av)* — avg 7.0, Accept. Similar framework (random sketch operators + convergence theory for fine-tuning), but MAST had better-validated experiments and stronger engineering contributions. RAC-LoRA is below MAST.
3. *LoRA-RITE (VpWki1v2P8)* — avg 8.67, Oral. Strong theory + consistent improvements on Gemma 2B/7B. Clearly above RAC-LoRA.
4. *FeDeRA (GtlRN48XYA)* — avg 3.0, Reject. Empirical-only FL+LoRA paper with no theory. RAC-LoRA is clearly above FeDeRA.
5. *BONE (RP0NPepy1m)* — avg 4.4, Reject. PEFT paper with missing theory and mixed experiments. RAC-LoRA's theory is stronger but NLP experiments are comparably weak.

**Assessment relative to anchors:** The paper sits between GoLore (5.25, rejected) and MAST (7.0, accepted). RAC-LoRA has a cleaner theoretical contribution than GoLore (full convergence analysis across all settings, clear sketch-project interpretation, MNIST ablation) but worse empirical coverage (no LLaMA-scale results, NLP results below baselines). This places it at approximately the same level as GoLore—a borderline case trending toward rejection due to insufficient empirical validation of the main practical claim.

**Final Score: 5.0 — Reject (Borderline)**

MY FINAL SCORE: <pineapple>5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>