Now let me search for calibration papers.Now let me do one more calibration search to find papers with similar contribution patterns (new dataset + new loss function + new architecture).Now I have sufficient information to write the final authoritative review.

---

## Summary
This paper addresses a fundamental scalability failure of Hamiltonian-learning models: they achieve low elementwise MAE yet produce catastrophically wrong total energies for large molecules. The authors dub this the "Scaling-Induced MAE-Applicability Divergence" (SAD) phenomenon, demonstrate it empirically and theoretically via a condition-number argument, and propose Wavefunction Alignment Loss (WALoss) — a physics-informed training objective that aligns predicted and ground-truth eigenspaces via basis transformation rather than through backpropagation through eigensolvers. They accompany WALoss with a new dataset (PubChemQH, 50K drug-like molecules with 40–100 atoms) and a modernized architecture (WANet), reducing System Energy MAE on PubChemQH from ~65,000 kcal/mol (QHNet baseline) to 47 kcal/mol.

---

## Strengths

- **PubChemQH fills a genuine and costly gap** (Section 5.1): prior benchmarks cap at ~31 atoms; a 50K dataset spanning 40–100 atoms from PubChemQC geometries, generated with 128 V100-GPUs for one month, is a substantial community resource that uniquely enables the study of large-molecule scalability.

- **The SAD phenomenon is concretely demonstrated** (Figure 1, Theorem 1, Corollary 1): the controlled Gaussian-perturbation experiment showing that identical relative Hamiltonian MAE produces energy errors of up to 10⁶ kcal/mol for large molecules but negligible errors for small ones is a clean, informative demonstration. The theoretical grounding in κ(S)/‖S‖₂ scaling with basis size B provides a principled explanation rather than a purely empirical observation.

- **WALoss avoids backpropagation through eigensolvers** (Eq. 2–3, Table 4): by projecting the predicted Hamiltonian onto ground-truth eigenvectors C\*, the method sidesteps numerical instability that makes direct eigenvalue losses impractical. Table 4 confirms this directly — naive WALoss (backprop through eigensolvers) yields 13,562 kcal/mol System Energy MAE and 5.36% C similarity vs. 47.193 kcal/mol and 48.03% for full WALoss.

- **Component-wise ablation validates each design choice** (Table 4): naive WALoss, WALoss without reweighting, and full WALoss are systematically compared, demonstrating that both the basis-change design and the occupied-orbital reweighting are necessary.

- **Extrapolation to out-of-distribution molecule sizes** (Figure 4): WANet+WALoss maintains low HOMO/LUMO MAE on elongated alkanes up to 182 atoms — roughly 3× the training-set average — particularly in the D2 region, demonstrating genuine extrapolation capability.

---

## Weaknesses

### Fatal
None.

### Major

- **System Energy MAE on QH9 is absent, leaving the paper's core "general improvement" claim unverified.** Table 2 shows that WALoss raises Hamiltonian MAE on QH9 (WANet: 0.0502 → 0.0914 for WANet+WALoss), yet System Energy MAE — the metric the paper argues matters most — is never reported for QH9. Because WALoss explicitly trades elementwise accuracy for eigenspace alignment, and because the paper's theoretical analysis predicts the tradeoff is worth it primarily for large molecules (where κ(S)/‖S‖₂ is high), the possibility that WALoss is net-negative on small molecules is a live hypothesis that the paper does not resolve. Any claim that WALoss is a universally beneficial training objective is unsupported without this number.

- **"Physical accuracy" is asserted but not quantified against any meaningful standard.** The abstract states the method achieves "physical accuracy." The best reported System Energy MAE is 47.193 kcal/mol (Table 1), which is 47× above chemical accuracy (≈1 kcal/mol) — the conventional threshold below which computed energies are meaningful for predicting reaction pathways, conformational preferences, or spectroscopic properties. The paper never defines what "physical accuracy" means in this context, does not discuss the gap to chemical accuracy, and does not identify an application where 47 kcal/mol absolute errors are acceptable. This is an overclaim that should be replaced with an honest statement such as "a 1392× improvement over QHNet in system energy prediction" — which is both true and impressive.

### Minor

- **The SCF wall-clock speed-up claim rests on a single, undescribed molecule.** Figure 3(a) reports one measurement (DFT: 392.879 s vs. WANet-augmented: 302.763 s). The relative SCF-iteration reduction (82%) is reported across the test set in Table 1, but wall-clock time is sensitive to molecule size, basis set, and hardware state. Reporting mean ± std over at least a sample of test molecules would make the efficiency claim credible.

- **WANet's inference throughput is 2.4× slower than QHNet** (Figure 3b: 0.45 k/s vs. 1.09 k/s). The paper claims efficiency advantages while training time and GPU memory are indeed better, but inference speed is not. For downstream use cases like virtual screening where many molecules must be evaluated quickly, this is a practical concern that the paper downplays.

- **The 1347× headline figure describes WALoss's contribution to WANet specifically, not the improvement over prior art.** The abstract reads as if 1347× improvement is made over a prior-art baseline. In fact it is the ratio WANet(63,579)/WANet+WALoss(47.193). The honest improvement over prior state-of-the-art (QHNet, 65,721 kcal/mol) is 65,721/47.193 ≈ 1392× — actually larger. The paper should present the comparison to QHNet as the headline figure.

- **Table 4's "WALoss without Reweighting" row shows a εavg value of 41,230** in one metric column while achieving a reasonable System Energy MAE of 55.492 kcal/mol. This order-of-magnitude anomaly in one column is unexplained and should be clarified, as it suggests a catastrophic failure mode of the unweighted variant for that specific metric.

- **Elongated alkane scalability experiment lacks a QHNet baseline.** Figure 4 compares only WANet+WALoss, WANet-without-WALoss, and the initial guess on alkane chains. Without QHNet+WALoss or QHNet-only performance on the same out-of-distribution test, it is unclear whether WALoss or WANet drives the observed extrapolation benefit.

### Trivial

- The MoE gating function uses only `z = rbf(‖r_ts‖₂)` for routing, meaning experts are assigned purely by interatomic distance. Whether the experts learn genuinely distinct behaviors or collapse is not checked and could be added to the ablation.

---

## Nice-to-Haves

- **System Energy MAE on QH9** should be the first priority — it is likely already computed and would directly address the major weakness about generality of WALoss.
- **Systematic SCF wall-clock statistics** (mean ± std over, say, 100 molecules) would validate the 18% speed-up claim.
- **Absolute energy error decomposition**: quantifying how much of the 47 kcal/mol total energy error is systematic (correctable with a global linear shift) vs. structural would clarify whether near-chemical-accuracy is achievable post-hoc.
- **Condition number visualization in main body**: Figure 6 (κ(S)/‖S‖₂ distribution) is referenced in the remark after Corollary 1 but reportedly relegated to the appendix; bringing it to the main text would strengthen the theoretical narrative.
- **Comparison of regression baselines in Section 5.3** would benefit from explicitly noting that Equiformer V2, UniMol+, and UniMol2 are general-purpose models not tuned for PubChemQH, so the gap likely overstates the structural advantage of Hamiltonian learning over property regression.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"1347× is a comparison against a self-inflicted failure, misleading about improvement over prior art"** (Harsh Critic): REMOVED. The 1347× isolates WALoss's contribution to WANet. The actual improvement over QHNet (prior art) is 1392× — even larger. The paper is not inflating the comparison against prior art; if anything it is underselling it. Table 1 reports all baseline numbers transparently so the reader can compute any comparison directly.

- **The SAD phenomenon exists for small molecules too, undermining the large-molecule narrative** (Harsh Critic): PARTIALLY REMOVED. Figure 1 (left) does show that QH9 molecules also exhibit SAD at ~10⁻² relative MAE, but the paper explicitly addresses this: "For small molecules, a 10⁻² relative MAE is sufficient for accurate system energy predictions (left panel), but this accuracy does not extend to large systems (right panel)." The narrative is about the practical severity, not the existence, of SAD — the paper's framing is accurate.

- **Theorem 1 uses a non-standard 1,1-norm** (Harsh Critic): REMOVED as not verified against the paper with sufficient external context. The paper uses ‖ΔH‖_{1,1} as a matrix norm bound and provides a proof delegated to the appendix; this is a non-standard choice but does not constitute a fatal flaw in the stated theorem, which the paper proves.

- **Claim 1 is unproved** (Harsh Critic): REMOVED as a missing-appendix-proof concern. The paper states "The proof is delegated to Appendix J.1" for Theorem 1 and Corollary 1. Claim 1 is a motivating claim clearly labeled as such; demanding a formal proof for it is scope creep for an empirical systems paper.

- **Table 1 duplicate column headers for "εocc MAE↓"** (Harsh Critic): REMOVED. This is clearly a PDF parser artifact; the two columns have different numerical values (e.g., 2067.45 vs. 1532.672 for QHNet), confirming they are distinct metrics (likely εorb and εavg) whose labels were garbled in extraction.

- **Comparison with regression models (Equiformer V2, UniMol) as incomplete** (Harsh Critic): WEAKENED to Nice-to-Haves. The comparison does favor the authors because the regression models are not tuned for this task, but the primary purpose of Section 5.3 is to show that Hamiltonian learning generalizes to multiple properties from a single model — an inherent architectural advantage that is valid regardless of tuning.

---

## Novel Insights

The paper makes one genuinely important observation that has not been systematically studied: elementwise Hamiltonian losses can fail catastrophically for large molecules even at low relative MAE, because the eigenvalue sensitivity scales as κ(S)/‖S‖₂ · ‖ΔH‖₁,₁, and κ(S)/‖S‖₂ grows with system size. The consequence — that models achieving 0.01% relative MAE on large-molecule Hamiltonians can produce energy errors of 10⁵–10⁶ kcal/mol — is a counter-intuitive, practically important finding that prior benchmark papers (which used smaller molecules) could not reveal. WALoss's design of projecting the predicted Hamiltonian onto the ground-truth eigenbasis, which circumvents backpropagation through ill-conditioned eigensolvers while directly penalizing eigenspace misalignment, is a clean and reusable methodological response to this insight.

---

## Suggestions

1. **Add System Energy MAE on QH9** as a new row/column to Table 2 — this is the single most important addition and likely requires no new experiments.
2. Replace "physical accuracy" in the abstract with a quantitative statement of improvement over QHNet and note that reaching chemical accuracy (1 kcal/mol) remains an open challenge.
3. Clarify the 1347× figure by noting it compares WANet with and without WALoss; separately state the 1392× improvement over QHNet baseline.
4. Report distribution of relative SCF iterations and wall-clock times over the test set (not just one molecule).
5. Add a brief experiment or qualitative analysis showing that MoE experts specialize to different distance regimes (e.g., expert utilization histograms), validating the MoE design choice.

---

## Score and Decision

**Calibration anchors used:**

| Paper | Path | Avg Human Score | Comparison |
|---|---|---|---|
| SLEM (Hamiltonian prediction, DFT, spotlight) | kpq3IIjUD3 | 7.33 | Broadly similar domain; SLEM addresses multiple quantum operators and diverse materials with strong transferability concern; this paper identifies a novel failure mode (SAD) and fixes it, with a larger dataset but narrower scope. |
| QO2Mol quantum chemistry dataset | o6aUi3ukdd | 2.5 | Weak comparison: pure dataset paper with no methodological novelty and limited benchmarks; this paper is far stronger. |
| Physics-informed diffusion models | tpYeermigp | 5.75 | Moderate comparison: introduces physics constraint loss for generative models; this paper is more comprehensive with clearer ablations. |
| MolSpectra dataset + SpecFormer | xJDxVDG3x2 | 6.33 | Close comparison: introduces dataset + architecture + novel training objective for molecular ML; similar profile and quality. |
| SIU dataset + bioactivity method | S8gbnkCgxZ | 7.0 | Somewhat similar profile (dataset + method); that paper's benchmarks are more thorough. |

**Assessment**: The paper sits clearly above medium anchors (5.75) and the MolSpectra-type paper (6.33) due to the genuine novelty of the SAD identification and the WALoss solution, the scale of the dataset effort, and the thoroughness of the ablation. It falls short of SLEM (7.33) and the SIU paper (7.0) due to the missing System Energy MAE on QH9 (a key missing experiment that directly bears on the paper's central generalization claim) and the overclaiming of "physical accuracy." A score of **6.0** is appropriate — this is a solid, accept-level contribution at the poster level, with real scientific value but needing clearer claims and one additional experiment.

**Originality**: High — identification and principled resolution of the SAD phenomenon is novel and important.
**Importance of research question**: High — scalable Hamiltonian learning for large molecules is a key bottleneck in computational chemistry.
**Support of claims**: Moderate — core claims on large molecules are well-supported; the generality of WALoss is not fully supported without QH9 System Energy.
**Soundness of experiments**: Good — well-ablated on PubChemQH; incomplete on QH9.
**Clarity of writing**: Adequate — the abstract overclaims ("physical accuracy," implicit framing of 1347×); the technical sections are clear.
**Value to community**: High — both PubChemQH and WALoss are reusable contributions.

**Score: 6.0 — Accept (Poster)**

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>