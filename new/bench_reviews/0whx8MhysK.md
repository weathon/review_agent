Now I have a good picture of the paper. Let me run calibration searches.Now I have enough information to write the final consolidated review.

---

## Summary

This paper proposes Influence-Guided Diffusion (IGD), a training-free guided sampling framework for dataset distillation. The method connects the dataset distillation objective to the trajectory influence function, and uses pre-computed gradient checkpoints from real-data training (rather than synthetic training) to steer a pretrained diffusion model (DiT) toward generating data with higher training influence and diversity. Applied as a plug-in on top of vanilla DiT and Minimax-tuned DiT, IGD achieves state-of-the-art results on ImageNette, ImageWoof, and ImageNet-1K (60.3% top-1 at IPC=50), with consistent cross-architecture improvements.

---

## Strengths

- **Novel framing and influence-function connection (Section 3.1, Eq. 5–6):** The paper establishes a concrete derivation linking the distillation objective (Eq. 1) to the trajectory influence function (Eq. 5), providing a principled motivation for using influence as a guidance signal rather than relying on heuristic objectives. This connection is well-presented and meaningfully distinguishes IGD from prior diffusion-based distillation approaches.

- **Consistent, cross-architecture empirical improvements (Tables 1–4):** IGD improves both vanilla DiT and Minimax across all 3 IPC settings, 2 subsets (Nette/Woof), 3 test architectures in Table 1, and 4 architectures in Table 3. The margins are substantial and non-trivial: 5.8–6.6% boosts for DiT-IGD over DiT on ImageNette/Woof, and 6.9% on ImageNet-1K at IPC=50. This constitutes a strong and reproducible empirical result.

- **SOTA on ImageNet-1K (Table 2):** Minimax-IGD achieves 60.3% at IPC=50, surpassing RDED (56.5%) by 3.8% and extending to cross-architecture generalization in Table 3 with average margins of ~5% over RDED at IPC=50.

- **Well-designed ablations (Tables 5–6, Figure 2):** The ablation cleanly separates influence guidance (G_I) and deviation guidance (G_D) contributions and shows complementarity. The gradient-similarity-based checkpoint selection outperforms uniform sampling with fewer checkpoints (4 vs. 10). The early-stage guidance analysis (Figure 2b/c) clearly demonstrates the quality–influence trade-off and why partial guidance is preferable.

- **Independent validation via Wasserstein distance (Figure 3):** Figure 3 uses Wasserstein distance — not the optimization objective — as an independent distributional metric. The finding that Minimax-IGD achieves better accuracy than DiT-IGD despite a higher Wasserstein distance specifically supports the conditional-distribution hypothesis over simple distributional alignment.

---

## Weaknesses

### Fatal
None.

### Major

- **Circular equivalence argument in Section 3.2:** The paper states that replacing checkpoints $\theta_e^S$ with $\theta_e^{\mathcal{T}_c}$ "is an optimally equivalent target," where the equivalence holds "when $\mathbf{z}$ can provide the same training dynamics as $\mathcal{T}_c$." This condition is precisely what the distillation problem aims to achieve — it is satisfied only at the solution, not during the optimization. At any intermediate stage of sampling, the generated $\mathbf{z}$ does not match real training dynamics, so the substitution introduces an uncontrolled approximation error. The paper then further extends to full-dataset mini-batch checkpoints $\theta_e^{\mathcal{T}}$ rather than class-specific ones, layering a second approximation. While the practical rationale is sensible (avoid retraining, mitigate trajectory mismatch), the framing as an "optimally equivalent target" is misleading — it describes an ideal endpoint, not a bound on approximation error during sampling. The paper would be more honest calling this a practical approximation with the supporting empirical evidence rather than claiming theoretical equivalence.

- **Hyperparameter sensitivity for $k$ and $\gamma_t$ not demonstrated on ImageNet-1K:** Figure 2 (b/c) shows that accuracy can vary significantly with $k$ on ImageWoof (from ~65% to ~58% when $k$ goes from 5 to 10). The headline 60.3% result on ImageNet-1K depends on hyperparameters "empirically preset" with details deferred to the appendix, and no analogous sensitivity sweep is shown for ImageNet-1K. Given the 1000-class scale difference from the 10-class evaluation subsets, the reader cannot assess whether the headline figure is robust to small perturbations in $k$ and $\gamma_t$.

### Minor

- **Tautological corroboration via normalized influence (Figure 1):** Figure 1 presents normalized influence — computed from Eq. 7, the same objective IGD directly minimizes — as independent evidence that IGD improves training effectiveness. Reporting that IGD achieves higher influence values than vanilla DiT is analogous to reporting that gradient descent reduces its objective: it validates that the optimization succeeded, not that the objective is correctly specified. The causal claim would require an independent metric. The Wasserstein distance analysis (Figure 3) partially compensates for this, but the narrative around Figure 1 overstates what is shown.

- **Trajectory influence approximation limitations not discussed:** Equation (5) adopts the trajectory influence approximation from Pruthi et al. (2020), which relies on assumptions (short training, constant learning rate, near-quadratic loss) that do not hold in ImageNet-1K training. While the method works empirically, the known degradation of this approximation under long training on non-convex objectives is not discussed.

- **Asymmetry between influence guidance's effect on DiT vs. Minimax not explained:** Table 5 shows that $\mathcal{G}_I$ alone provides +1.3% for raw DiT but +3.4% for Minimax (IPC=50, ImageNette). The authors mention "Minimax's inherent focus on diversity" but provide no mechanistic explanation for why influence guidance is substantially more effective when applied on top of a diversity-fine-tuned backbone than on a vanilla one.

### Trivial

- The cosine similarity in Eq. (7) vs. the dot product in Eq. (6) is stated to "stabilize the magnitude of the guidance signal," but the ablation (Table 5) treats the full influence guidance block as a unit, so the specific effect of this normalization choice is not isolated.

---

## Nice-to-Haves

- A hyperparameter sensitivity sweep for $k$ on ImageNet-1K (at least 3 values) analogous to Figure 2c on ImageWoof would substantially strengthen confidence in the headline number.
- An experiment applying IGD to a second diffusion backbone (e.g., a U-Net LDM) would validate the claimed architecture-agnosticism beyond the DiT family.
- Showing failure cases (images where large $k$ produces degraded or semantically incoherent output) alongside the success cases in Figure 4 would make the overfitting discussion more concrete.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Training-free" claim elides surrogate training cost (Harsh Critic, Introduction):** REMOVED. The paper accurately claims no retraining of the *diffusion model*. The surrogate is a ConvNet-6 trained for 50 epochs — explicitly described as such in Section 4.1. The claim is not misleading.

- **Comparison to RDED involves an asymmetry (Harsh Critic, Section 4.2):** REMOVED as a weakness. The paper uses the same evaluation protocol as RDED (soft labels from ResNet-18 trained on full ImageNet), so the comparison is entirely fair. Any diversity advantage is equally available to RDED.

- **Missing coreset-selection oracle baseline (Harsh Critic):** REMOVED — this is a nice-to-have, not a core claim of the paper. IGD's scope is improving diffusion-based generation for dataset distillation, not comparing to coreset selection methods.

- **Single-GPU practicality (Strength Finder):** REMOVED as a standalone strength — it is a useful implementation detail noted in Section 4.1 but too generic to constitute a core contribution.

- **"Principled connection" strength being tautological (circular):** Partially addressed by including the circular equivalence as a Major weakness, while preserving the genuine novelty of the influence-function framing in the Strengths section.

---

## Novel Insights

The most genuinely novel observation in this work is the intersection between the dataset distillation objective and the trajectory influence function as a generative guidance signal. The finding (Figure 3) that Minimax-IGD achieves higher accuracy than DiT-IGD despite *larger* Wasserstein distance from the original distribution empirically supports the hypothesis that conditional-distribution targeting (finding a high-influence sub-distribution) can be more effective than minimizing global distributional divergence. This has implications beyond this paper: for generative-model-based dataset distillation broadly, optimizing for training-effectiveness rather than distributional fidelity may be the more productive objective.

---

## Suggestions

1. **Replace the "optimally equivalent target" language in Section 3.2** with honest characterization: "We treat the real-data checkpoints $\theta_e^{\mathcal{T}}$ as a practical approximation of the ideal $\theta_e^S$, motivated by the fact that at the optimum these converge — while acknowledging the approximation gap during sampling." This is more accurate and does not weaken the contribution.
2. **Add a $k$-sensitivity sweep on ImageNet-1K** (Appendix is acceptable), analogous to Figure 2c, to establish robustness of the 60.3% headline.
3. **Provide a mechanistic explanation** for why influence guidance benefits Minimax more than raw DiT (Table 5 asymmetry), even if informal.

---

## Score and Decision

**Calibration anchors used:**

| Paper | Path | Avg Score | Comparison |
|---|---|---|---|
| "A Defense of One-Step Learning" | CCoa6XgO8F.md | 3.8 | Low anchor: dataset distillation paper with unsupported claims, poor experiments. Paper under review is substantially stronger on all axes. |
| "Rethinking Dataset Quantization" | xajif1l65R.md | 5.0 | Medium anchor: coreset/distillation method, missing large-scale experiments. Paper under review has full ImageNet-1K evaluation and stronger results. |
| "GIFT: Unlocking Full Potential of Labels" | FoF5RaA3ug.md | 6.5 | High anchor: plug-and-play dataset distillation improvement on ImageNet-1K. Similar positioning (adds a component to existing distillation pipeline), but paper under review has a more novel core idea and larger empirical gains. |
| "Fast Direct: Black-box Guidance for Diffusion" | OmpTdjl7RV.md | 6.75 | High anchor: guided diffusion with strong results. Comparable level of novelty and empirical rigor. |
| "REPA" | DJSZGGZYVi.md | 9.0 | Exceptional oral paper; paper under review is clearly below this tier. |

The paper under review is clearly above the medium band, with comprehensive large-scale experiments, novel methodology (influence-guided diffusion), and a consistent SOTA result. It is comparable to GIFT (6.5) and Fast Direct (6.75). The main real weaknesses — the circular equivalence argument in Section 3.2 and the missing hyperparameter sensitivity on ImageNet-1K — are genuine but do not invalidate the empirical contribution, which is robustly demonstrated. Scoring consistent with the 6.5–6.75 anchor cluster.

**Originality:** Good — framing dataset distillation as guided diffusion via influence functions is novel.  
**Importance:** Moderate-high — dataset distillation for large-scale ImageNet is an active, impactful problem.  
**Claims well-supported:** Mostly yes — empirical claims are well-supported; theoretical framing has a gap.  
**Soundness of experiments:** Good — multi-architecture, multi-dataset, ablations present.  
**Clarity of writing:** Good.  
**Value to community:** Clear — consistent improvements over prior diffusion-based distillation methods.

**Final Score: 6.5 — Accept**

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>