Now let me do calibration searches to properly score this paper.Now I have enough to write the final review.

---

## Summary

SPARK is a physics-guided quantized augmentation framework for dynamical system modeling under distribution shift and data scarcity. It builds a discrete VQ-VAE-based memory bank by incorporating boundary positional encodings and physical parameter channel attention, then augments training data by interpolating encoder embeddings with top-K memory bank entries (Eq. 7). A Fourier-enhanced graph ODE forms the prediction backbone. Experiments span five benchmarks (Prometheus, Navier-Stokes, Spherical-SWE, 3D Reaction-Diffusion, ERA5) and a cross-domain transfer study (ERA5 → SEVIR).

---

## Strengths

- **Strong and consistent OOD generalization across five benchmarks (Table 1)**: SPARK achieves the best MSE on every benchmark under both w/o OOD and w/ OOD conditions, with large margins over the strongest prior baseline NMO—~36% on Prometheus OOD (0.0308 vs 0.0483) and ~43% on ERA5 OOD (0.0321 vs 0.0563). This breadth is convincing evidence that the system as a whole is effective.

- **Transferability across multiple backbone architectures validated (Table 3)**: SPARK is shown as a plugin applied to SimVP, PredRNN, and Earthfarseer in the ERA5→SEVIR transfer experiment. It consistently reduces MSE across all three backbones at every data fraction (20%–100%), with the most pronounced gains in the data-scarce regime. This provides partial empirical support for the "universal plugin" claim in a realistic cross-domain scenario.

- **Superiority over dedicated OOD methods (Table 4)**: SPARK outperforms specialized OOD baselines (LEADS, CODA, NUWA) on three benchmarks, with LEADS failing dramatically on ERA5 (0.4233 under OOD vs. SPARK's 0.0401). This shows physics-guided augmentation is more effective than general-purpose OOD approaches on this task.

- **Energy spectrum analysis (Figure 6)**: Physical consistency is validated via energy spectra across Navier-Stokes, Spherical-SWE, and 3D Reaction-Diffusion—a domain-appropriate evaluation beyond MSE that most baselines do not employ.

- **Efficient compression via curriculum augmentation (Section 3.3)**: The gradual integration of augmented samples through curriculum learning is a practical engineering choice that stabilizes training; SPARK's training and validation losses converge within 80 epochs on sea ice (Figure 5).

---

## Weaknesses

### Fatal
None.

### Major

- **Bundled backbone and plugin in Table 1 prevents attribution of gains.** The primary results table compares "OURS + SPARK" — which packages both the novel Fourier-enhanced Graph ODE backbone *and* the SPARK augmentation plugin — against eight baselines that use entirely different architectures. There is no row in Table 1 for the authors' custom backbone *without* SPARK. The headline claim is that SPARK is a "physics-guided compression and augmentation *plugin*" that drives generalization improvement; but without a "OURS backbone-only" baseline in Table 1, it is impossible to determine what fraction of the ~26–43% improvements stems from the novel prediction architecture versus the augmentation mechanism. The transferability experiment (Table 3) partially addresses this by applying SPARK to standard architectures, but the main benchmark evaluation does not.

- **Unexplained numerical inconsistency between Table 1 and Table 4 for the same method and dataset.** "OURS + SPARK" in Table 1 reports 0.0294 / 0.0308 (w/o OOD / w/ OOD) on Prometheus, whereas "SPARK (OURS)" in Table 4 reports 0.0323 / 0.0328 on the same dataset. These are materially different numbers with no explanation of the experimental configuration difference. One plausible interpretation is that Table 4 tests SPARK as a plugin without the custom backbone (which would implicitly provide the ablation missing from Table 1), but this is never stated. Similarly on Spherical-SWE, Table 1 gives 0.0018/0.0020 while Table 4 gives 0.0022/0.0024. Unexplained inconsistencies across tables undermine reproducibility and raise questions about experimental control.

- **Missing component-level ablation of SPARK.** The paper introduces four interconnected components—boundary relative positional encoding (Eq. 1), channel attention (Eqs. 2–3), VQ-VAE discretization (Eq. 5), and memory-bank augmentation (Eq. 7)—but never isolates the contribution of each. In particular, there is no comparison between VQ-based discrete retrieval and continuous k-NN interpolation, or between boundary-inclusive encoding and a version without boundary positional encoding. Without these ablations, the specific design choices cannot be justified empirically, and it remains unclear whether the discrete memory bank is necessary or whether a simpler augmentation scheme would achieve similar gains.

### Minor

- **OOD protocol underspecified per dataset.** The paper mentions "ten different viscosities" for Navier-Stokes and references the Prometheus benchmark's original settings, but never precisely defines what constitutes the "w/ OOD" split for each of the five datasets. For ERA5 and Spherical-SWE, the OOD conditions are entirely unspecified in the main text. Since OOD generalization is the central experimental axis, this omission makes it difficult to interpret the magnitude of OOD shifts and to compare results across papers.

- **Section 4.3 (sea ice) draws strong claims from weak evidence.** The section title states "SPARK can handle challenging tasks effectively," but (a) only FNO and U-Net are compared — not the full baseline suite from Table 1; (b) the quantitative metrics (SSIM ≈ 0.95, PSNR ≈ 40 dB) are reported only for SPARK's own training curves, with no corresponding numbers for FNO or U-Net; and (c) the comparison is qualitative only via Figure 4. The section shows that SPARK converges and produces visually plausible outputs, but does not substantiate that it is the best-performing method on sea ice.

- **Negative transfer in Table 3 is unaddressed.** Without SPARK, ERA5 pretraining *hurts* SimVP at 60–100% SEVIR data (e.g., +15.79% at 100%), and PredRNN at 100% (+8.70%), and Earthfarseer at 100% (+6.25%). The paper does not discuss why pretraining on ERA5 degrades fine-tuning at high data regimes but benefits it at low data regimes. More importantly, the paper does not explain the mechanism by which SPARK consistently prevents this degradation. Without this analysis, it is unclear whether SPARK is genuinely improving transfer or primarily mitigating a domain-mismatch penalty introduced by the ERA5 pretraining choice.

- **Theorems 1 and 2 are motivational only, not specific to SPARK.** Both theorems (Eqs. 12–13) instantiate standard PAC-Bayes / information-theoretic bounds. They correctly state that *if* incorporating physical priors reduces I(θ; D | P) or KL(Q ∥ P), then the generalization bound tightens. However, neither theorem proves that SPARK's specific mechanism (VQ-VAE quantization, boundary PE, top-K augmentation) actually achieves the assumed reduction. The theoretical section provides motivation but no validation specific to the proposed method.

### Trivial

- **Figure 1 radar chart axes mix model names with a metric name.** The axes are labeled "VIT, CNO, U-Net, SSIM, NMO" — SSIM is a standard image quality metric, not a model, while all other labels are model names. The figure as labeled is incoherent and should relabel SSIM (presumably intended as another backbone or a metric axis). The intended message is clear from context, but the mislabeling is misleading.

---

## Nice-to-Haves

- Add a row "OURS (backbone only, no SPARK)" to Table 1. Even a single such row across all five datasets would resolve the attribution ambiguity for the main result.
- Apply SPARK to NMO (the best prior baseline) in Table 1 to demonstrate the "universal plugin" property on the primary evaluation. Table 3 already does this for video-prediction architectures; doing it for the strongest neural operator baseline would strongly reinforce the claim.
- Embed space visualization of the memory bank (t-SNE/PCA of VQ-VAE codebook entries colored by physical parameter values) would validate the claim that the memory bank is "physics-rich."
- Provide a precise OOD protocol definition per dataset (which parameters are held out, what range).

---

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Harsh Critic: "Augmentation formula v_i differs from standard Mixup."** The critic asks for a comparison to continuous k-NN retrieval vs. VQ discretization — this is addressed above as a missing ablation (valid concern). But the specific framing that Eq. 7 "essentially is Mixup" is a stretch; Mixup operates in input space while this operates in physics-informed VQ codebook embedding space with discretized entries. The framing overstates the similarity.

- **Harsh Critic: "Theorems logically vacuous — invalidates the theory section."** The criticism is partially valid (addressed as Minor above), but characterizing the theory section as "logically vacuous" overstates the case. Such motivational theorems establishing general sufficient conditions are widely used in applied ML and are not expected to provide algorithm-specific proofs.

- **Harsh Critic: "Table 4 discrepancy is a reproducibility concern."** We retain the discrepancy concern as Major above but remove the framing that it reflects "data fabrication" or that results "cannot be reproduced" — the numbers differ across two different experimental setups (Table 1 vs Table 4), which is a presentation transparency issue rather than an integrity concern.

- **Strength Finder: "Theoretical grounding (Theorems 1 and 2 provide information-theoretic bounds)."** Removed as a strength because the harsh critic's verified point that these are standard bounds (not specific to SPARK) is correct. The theorems are motivational scaffolding, not a genuine theoretical contribution.

---

## Novel Insights

The most genuinely novel observation from the reviews — not in the paper itself — is the potential that the numerical gap between Table 1 ("OURS + SPARK") and Table 4 ("SPARK (OURS)") may implicitly encode the ablation the paper is missing: if Table 4 corresponds to SPARK applied as a plugin to a standard backbone (without the custom Fourier-enhanced Graph ODE), then the gap (0.0294 vs. 0.0323 on Prometheus) would suggest the custom backbone accounts for roughly half the improvement over NMO, and SPARK's augmentation accounts for the other half. The paper should test and confirm this interpretation explicitly, as it would substantially strengthen both the ablation story and the "universal plugin" claim simultaneously.

---

## Suggestions

1. **Clarify Table 1 vs. Table 4 experimental setups explicitly** (which backbone is used in each) — this single clarification could resolve the most serious transparency concern.
2. **Add one ablation row in Table 1**: "OURS backbone only (no augmentation)."
3. **Discuss the negative transfer phenomenon in Table 3** with at least a hypothesis about domain mismatch and SPARK's regularization effect.
4. **Specify the OOD split precisely** in the experiment section (not just the appendix).
5. **Restrict section 4.3's claims** to match the actual evidence (qualitative advantage over FNO/U-Net on sea ice), or add quantitative baseline comparisons.

---

## Evaluation on Key Axes

- **Originality**: Moderate. The combination of VQ-VAE with physics-informed encoding for dynamical system augmentation is novel; the Fourier-enhanced Graph ODE builds on well-established components.
- **Importance of research question**: High. OOD generalization and data scarcity in physics-governed dynamical systems are genuine bottlenecks.
- **Claims well-supported**: Partially. The empirical gains are real and large, but attribution between the backbone and the plugin is not established in the main experiment.
- **Soundness of experiments**: Fair. Five benchmarks plus transfer study is commendable breadth; the critical missing experiment is the backbone-only ablation.
- **Clarity of writing**: Fair. The method is described clearly; the experimental section has transparency gaps (OOD protocol, Table inconsistency).
- **Value to the research community**: Moderate-to-high if plugin claim is substantiated; moderate if the gains are primarily from the novel backbone.

---

## Score and Decision

**Calibration anchors used:**

| Paper | Topic | Scores | Decision |
|---|---|---|---|
| AgTSjXh7vl (P-Align) | Physics-aware augmentation plugin for dynamical systems (close match) | 6/3/5/3 avg ~4.25 | Withdrawn |
| 5LvTfc4fBz (Physics-Enhanced Neural Op.) | Physics-enhanced neural operator for turbulence | 3/5/6/5/5/6 avg ~5.0 | Rejected |
| tpYeermigp (Physics-Informed Diffusion) | PDE-constrained physics-informed models | 5/6/6/6 avg ~5.75 | Accepted poster |
| 8HG2QrtXXB (HelmSim) | Physics-inspired fluid simulation, incomplete ablations | 3/5/6/6 avg ~5.0 | Rejected |
| h6Tz85BqRI (VQGraph) | VQ-VAE for graphs, incomplete ablations | 5/8/8/6/5 avg ~6.4 | Accepted |
| 4yaFQ7181M (Space-time PDE) | Physics simulation from sparse observations | 8/8/8/6/8 avg ~7.6 | Spotlight |

**Reasoning**: SPARK is most comparable to P-Align (nearly identical framing: physics-guided augmentation plugin for dynamical systems across multiple backbones), which averaged ~4.25 and was withdrawn due to attribution and ablation gaps. SPARK has stronger empirical breadth (5 benchmarks vs P-Align's 5 datasets with better OOD coverage), and the transferability experiment with three backbone types partially validates the plugin claim. However, SPARK shares P-Align's central weakness: attributing improvements to the plugin vs. the architecture is unresolved. Relative to Physics-Enhanced Neural Operators (~5.0, rejected) and HelmSim (~5.0, rejected), SPARK has broader evaluation and clearer practical motivation. The large OOD gains and cross-domain transferability push toward 5.5; the missing backbone ablation and unexplained table inconsistency pull back toward 5.0. I position SPARK at **5.0**, just below the acceptance threshold, reflecting solid empirical contributions undermined by a critical methodological transparency gap.

## Score and Decision

**5.0 — Reject (revise and resubmit)**

The paper makes a genuine empirical contribution with large, consistent OOD generalization improvements across five benchmarks and validated cross-backbone transferability. However, the core attribution claim — that the SPARK augmentation plugin drives the reported improvements — is not established in the main result table, where the novel backbone and the plugin are bundled together and never ablated. The unexplained numerical discrepancy between Table 1 and Table 4 for the same method compounds this concern. These are not superficial presentation issues; they bear directly on whether SPARK is a "universal augmentation plugin" (the paper's central claim) or primarily a novel prediction architecture with augmentation as a secondary component. The paper would be significantly strengthened by a backbone-only row in Table 1 and explicit clarification of the Table 1 vs. Table 4 experimental setup.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>