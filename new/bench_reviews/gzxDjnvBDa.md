Now I'll run calibration searches in parallel to find anchor papers.Now I have enough information to write the full review. Let me synthesize everything.

---

## Summary

CrystalFramer proposes *dynamic frames*—atom-wise, layer-wise local coordinate systems constructed from self-attention weights rather than global structural properties—and integrates them into the Crystalformer transformer architecture for crystal property prediction. The core insight is that global/unit-cell frames misalign with the local masked view implied by message-passing weights, motivating frames that focus on *interacting* atoms specifically. The method achieves state-of-the-art MAE on the majority of tasks across JARVIS, MP, and OQMD datasets while adding only ~100K parameters to the baseline.

---

## Strengths

- **Conceptually well-motivated dynamic frame idea** (Sec. 3, Eq. 5): The argument that conventional frames are influenced by atoms with zero message-passing weight is concrete and non-trivial. This leads naturally to atom-wise, layer-wise frame construction—a clean conceptual departure from prior FA approaches (Puny et al., 2022; Duval et al., 2023; Yan et al., 2024).

- **Consistent SOTA results across three datasets of vastly different scale**: Max frames achieve best MAE on 4/5 JARVIS tasks (Table 1), 3/4 MP tasks (Table 2), and all 3 OQMD tasks (Table 3, ~817K materials), demonstrating both quality and scalability.

- **Inherent unit-cell-variation invariance** (Sec. 3, last paragraph): Dynamic frames use the full crystal structure $\tilde{P}$ rather than unit-cell representation $(P, L)$, avoiding the sensitivity to unit-cell choice that plagues PCA and lattice frames—a genuine, crystallography-specific advantage.

- **Parameter efficiency** (Table 4): CrystalFramer adds only ~100K parameters over Crystalformer (952K vs 853K total), far below iComFormer (5.0M), while achieving superior accuracy on most tasks.

- **Max frame construction avoids eigenvalue degeneration** (Sec. 3.1): By directly selecting the highest-attention-weight atom direction, max frames bypass the ~10% degeneration rate confirmed for PCA frames on symmetric crystals.

---

## Weaknesses

### Fatal
None.

### Major

- **Ablation for "dynamic" vs. "static" is methodologically impure**: The static local frames baseline uses $w_{ij(n)} = \exp(-r_{ij(n)}^2)$ distance-decay weights, while max frames use learned softmax attention weights. The comparison therefore conflates (a) *learned-attention-based vs. distance-decay-based* weighting with (b) *dynamic (updated per layer) vs. fixed* weighting. A clean control would freeze the attention weights and use them for frame construction, keeping everything else identical. As designed, one cannot attribute the gains specifically to the *learned* or *dynamic* nature of the weights rather than to the specific choice of weighting functional form. Notably, on JARVIS E hull, static local frames (0.0444) *outperform* max frames (0.0471), suggesting max frames' advantage is task-dependent even on the best-case JARVIS benchmark.

- **Weighted PCA frames—one of two proposed dynamic frame types—fail on MP**: On all four MP tasks, weighted PCA frames underperform the Crystalformer baseline (e.g., formation energy 0.0197 vs. 0.0186, bandgap 0.214 vs. 0.198, bulk modulus 0.0423 vs. 0.0377). The paper acknowledges this and defers to Appendix F, but this is not a minor failure—it implies that the "dynamic frames" concept does not generalize across both proposed instantiations. A paper arguing for a general principle should either explain *mechanistically* why one instantiation fails (e.g., eigenvalue degeneration causing gradient noise, PCA averaging suppressing the strongest-interaction signal) or narrow the contribution claim to max frames specifically.

### Minor

- **Gradient stopping through frame axes is left unexplained** (Sec. 3.1, footnote 2): Frame axis construction is non-differentiable (eigenvectors under degenerate eigenvalues, argmax in max frames), so gradients are not propagated through frames. The paper notes that "simply ignoring frame gradients gave the best results" over straight-through estimators and softmax temperature annealing. This is an empirically puzzling finding—if gradient flow through frames helps calibrate them for prediction quality, its absence should hurt; if it hurts, that itself reveals something about the training dynamics. The finding is reported but not analyzed.

- **No variance or statistical significance reporting**: Tables 1–2 report single-run point estimates. Several margins over iComFormer are small (e.g., MP shear modulus: 0.0677 vs. 0.0637 for iComFormer; JARVIS bandgap OPT: 0.117 vs. 0.122), and stochastic frame construction (random sign flips, perturbation noise) introduces run-to-run variability. Multiple-seed results are warranted to support the "state-of-the-art" claim on these close tasks.

- **Hyperparameter asymmetry with iComFormer not fully resolved** (Sec. 5.1): The paper correctly flags that iComFormer uses per-task hyperparameter tuning while CrystalFramer uses uniform per-dataset settings—but presents this as favoring iComFormer. The paper does not provide even a single CrystalFramer run with task-specific tuning to confirm the headline SOTA claim on close tasks like MP shear modulus.

### Trivial

- The angular GBF scale $s=4.0$ (vs. $s=1.0$ for distance) is stated to work empirically better but no ablation is provided. This is a minor presentation gap but does not affect the overall conclusions.

---

## Nice-to-Haves

- A clean dynamic-vs.-static ablation: train Crystalformer without frames, then freeze its attention weights and construct max frames using those frozen weights. Compare against fully dynamic max frames. This single experiment would properly test whether the dynamic (layer-wise, learned) nature of the weights specifically drives performance improvement.
- Multi-seed variance for the main results tables, at least for the tasks where CrystalFramer's margin over iComFormer is within ~2%.
- A mechanistic analysis of when and why weighted PCA frames degenerate more on MP than JARVIS—this would strengthen the conceptual framework rather than leaving one instantiation as an unexplained failure.
- Demonstration on one additional crystal transformer architecture to show the concept is not architecture-specific. The paper's explanation for restricting to Crystalformer (channel-wise sigmoid attention in Matformer/ComFormer is unsuitable) is reasonable, making this a low-priority addition rather than a flaw.

---

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Critic claim that weighted PCA frames are "incidental"**: The paper honestly reports this failure and does not hide it. However, the lack of mechanistic explanation does constitute a real gap (kept as a Major weakness), not a reason to entirely dismiss the contribution.

- **Critic claim about "not jointly optimized"**: The paper does not claim frame axes are trained via gradients from the loss. The framing is that attention weights are trained for their primary message-gating role and frame axes are derived from those weights. This is a reasonable design, not a flaw.

- **Strength Finder's "Ablation directly validates the dynamic aspect"**: This overstates the ablation. The ablation is partially informative but methodologically impure (as detailed above). Removed as a standalone strength.

- **Strength Finder's "Minimal, modular architecture change"**: Too generic—applies to any incremental paper. Removed.

- **Critic's demand for architecture-independent formulation**: The paper explicitly explains why other crystal transformers don't fit and scopes the claim accordingly. Moved to Nice-to-Have.

---

## Novel Insights

The most interesting observation across the reviews is the tension between the paper's *conceptual* and *empirical* contributions. The conceptual claim—that attention-weight-based dynamic frame selection is what drives improvement—is plausible but not cleanly established, because the ablation baseline (static local frames, which outperform both PCA and lattice frames and are competitive overall) uses a different weighting functional form. The genuinely novel and empirically validated finding is narrower: local, atom-centered angular features substantially outperform global frame approaches in crystal transformers, and attention-based selection of the primary axis (max frames) provides a further improvement over distance-decay-based local frames in most tasks. This reframing is more modest than the paper's narrative but more defensible. The complete failure of weighted PCA frames on MP is an anomaly that the community should examine—it may indicate that averaged (PCA) local structure summaries are actively harmful when the training signal is dominated by the strongest pairwise interaction direction.

---

## Calibration

**Anchor papers:**
- `/home/wg25r/review_agent/human_reviews/fxQiecl9HB.md` — Crystalformer, avg **7.25**, Accept. The very baseline this paper extends. CrystalFramer achieves stronger results but has the methodological gaps described above; likely slightly below this anchor.
- `/home/wg25r/review_agent/human_reviews/kpq3IIjUD3.md` — SLEM equivariant model for quantum operators in materials, avg **7.33**, Accept spotlight. CrystalFramer is in the same performance tier but more empirical and less theoretically grounded; below this anchor.
- `/home/wg25r/review_agent/human_reviews/5wxCQDtbMo.md` — GotenNet equivariant tensor network, avg **6.75**, Accept. Similar profile: solid empirical gains with clear engineering contribution; CrystalFramer is comparable.
- `/home/wg25r/review_agent/human_reviews/0aaaM31hLB.md` — Learning Symmetries through Loss Landscape, avg **5.25**, Reject. Medium paper with equivariance framing but weaker experiments; CrystalFramer is clearly stronger.
- `/home/wg25r/review_agent/human_reviews/zUDbPgskDS.md` — CrysToGraph, avg **3.25**, Reject. Weak crystal property prediction paper with inconsistent/limited contributions; CrystalFramer is substantially stronger.
- `/home/wg25r/review_agent/human_reviews/rcdR97P2Mp.md` — Lipschitz-continuous invariants for crystals, avg **4.50**, Reject. Theoretical but lacking empirical depth; CrystalFramer is stronger experimentally.

**Comparison**: CrystalFramer sits between GotenNet (6.75) and the medium anchor (5.25). Its empirical contributions are genuine and extensive, SOTA results span three datasets at multiple scales, and the conceptual contribution is real. The ablation impurity and weighted PCA failure are major concerns but do not invalidate the core empirical finding. Given the cluster of high anchors at 6.75–7.33 and the methodological gaps, I place CrystalFramer at **6.5**: a solid, accept-worthy paper that somewhat overclaims its conceptual advancement but delivers genuine empirical value to the community.

---

## Score and Decision

The paper presents a genuine empirical advance—local angular features derived from attention weights consistently improve crystal property prediction, achieving SOTA across three datasets with minimal overhead. The core conceptual framing slightly overstates what the ablations can prove, and the failure of one of two proposed frame types on an entire dataset is left incompletely explained. These are real limitations but do not undermine the paper's primary empirical contribution. The work is clearly above the medium-scoring anchors (5.25, 4.50) and approaches but falls short of the top-performing anchors (7.25, 7.33) due to methodological gaps.

**Final Score: 6.5 / 10 — Accept (poster)**

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>