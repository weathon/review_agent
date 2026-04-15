## Summary
This paper proposes OMG, a lightweight modification for 3DGS-based inverse rendering that makes each Gaussian’s opacity depend on material properties through a learned cross-section term, motivated by the Bouguer-Beer-Lambert law. Concretely, it replaces the usual linear opacity form with \(\alpha_i(x)=1-\exp(-o_i G_i(x) f(m_i))\), and shows improved average performance when plugged into three prior inverse-rendering systems (R3DG, GaussianShader, GS-IR) on both synthetic and real datasets.

## Strengths
- **The paper identifies a real modeling gap specific to disentangled 3DGS inverse rendering.** In prior 3DGS inverse-rendering pipelines, opacity is typically optimized separately from BRDF/material parameters, whereas here the paper explicitly couples them. This is a concrete and nontrivial contribution rather than a generic “better loss” tweak.
- **The method is genuinely simple and broadly applicable.** The same core idea—material-conditioned opacity via a small MLP—is integrated into three distinct baselines, including both one-stage and two-stage pipelines. This breadth of integration is good evidence that the idea is practically reusable.
- **The Synthetic4Relight evaluation includes direct material metrics, not just image metrics.** Table 1 shows gains not only in NVS and relighting, but also in albedo and roughness estimation (\(+0.61\) dB albedo PSNR and roughness MSE dropping from 0.011 to 0.007), which is important because the claimed contribution is about inverse rendering, not merely view synthesis.
- **Average gains are consistent across all three baselines tested.** The paper reports improvements for R3DG on Synthetic4Relight, GaussianShader on Shiny Blender / Glossy Synthetic, and GS-IR on Mip-NeRF 360, suggesting the proposal is not narrowly overfit to a single codebase.

## Weaknesses
### Fatal
- None.

### Major:
- **The paper overclaims physical correctness; the core derivation is heuristic at the crucial step.** The argument in Sec. 4.1 maps 3DGS opacity to Beer-Lambert attenuation by identifying \(o_i G_i(x)\) with number density and then setting path length \(s=1\) because “each Gaussian is ‘splatted’ to a 2D plane.” That is not a derivation from the renderer so much as an engineering reinterpretation. As written, this does not justify repeated claims such as “physically correct activation function,” “derive the exact form,” or “more accurate physical properties.” What the paper convincingly supports is **physics-inspired reparameterization**, not a demonstrated physically correct opacity law for 3DGS.
- **The experiments do not isolate which part of the modification causes the gains.** The proposed method bundles together at least three changes: (i) replacing linear alpha by \(1-\exp(-t)\), (ii) adding an MLP, and (iii) coupling material into opacity with extra gradients to materials. The paper’s mechanism claim is specifically about material-opacity coupling and added gradient constraints (Sec. 4.3), but there is no ablation comparing the exponential activation alone, a non-material conditioning variant, or detached-gradient variants. Without these controls, it is hard to know whether improvements come from the claimed physical coupling or from a smoother activation / extra capacity.
- **Several empirical claims are stronger than the evidence supports.** The paper repeatedly says “universal improvement” and attributes gains on all datasets to improved material estimation. But Table 3 shows per-scene regressions on Flowers (worse on all three metrics) and slightly worse LPIPS on Treehill, so the effect is not universal scene-by-scene. Likewise, direct material-quality evidence is only shown on Synthetic4Relight; for GaussianShader and GS-IR experiments, the claim that better NVS comes from better materials is plausible but not directly demonstrated.
- **The “plug-and-play” framing is somewhat overstated.** The core module is portable, but the actual implementation is not completely invariant across baselines. Sec. 5.1 states that GaussianShader uses albedo/roughness/specular tint as MLP input, whereas GS-IR and R3DG additionally feed SH coefficients “to make the optimization of the MLP in the first stage meaningful.” That is still useful engineering, but it is more baseline-adapted than a strict drop-in module.

### Minor
- **The analysis in Sec. 4.4 only partially supports the central novelty.** The Taylor expansion correctly shows that \(1-e^{-t}\approx t\) locally, but that only justifies the activation replacement, not the material-dependent cross-section term \(f(m_i)\), which is the main new ingredient. So the analysis is suggestive rather than sufficient.
- **Claims about improved geometry/normal estimation are under-supported.** Figs. 5 and 6 suggest nicer normals, and Sec. 5.3 notes that geometry optimization is affected through \(\partial \alpha_i/\partial o_i\). But there are no quantitative normal/depth/geometry evaluations, so assertions of better geometry should be softened.
- **The physical analogy is simplified in a way the paper does not discuss much.** Eq. 10 models cross section as a scalar \(f(m)\), while the Beer-Lambert discussion in Sec. 3.3 uses frequency-dependent notation \(\sigma_\nu\). Given that the paper works in RGB inverse rendering, this simplification is understandable, but it weakens the “strictly following” physical-law phrasing.
- **The magnitude of gains is modest.** The average PSNR improvements are typically around 0.3–0.5 dB. These are real and useful, but not transformative on their own; the case for acceptance therefore depends more on the conceptual contribution than on overwhelming empirical margins.

### Trivial
- **The paper does not report computational overhead of the added MLP.** Since one attraction of 3DGS-based methods is efficiency, it would be useful to know the training/runtime cost of the modification, even if the added network is small.

## Nice-to-Haves
- Add ablations isolating: exponential activation only, material-conditioned term only, detached-gradient variant, and a simpler non-MLP parameterization.
- Report whether the learned cross-section values correlate meaningfully with different material regimes; even a small visualization would make the physical interpretation more convincing.
- Quantify runtime / memory overhead to support the practical “plug-and-play” story.
- Soften causal explanations on datasets where material ground truth is unavailable, or add auxiliary diagnostics there.
- If space permits, include one quantitative geometry/normal evaluation on a dataset with ground truth.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Shadows baked into albedo/roughness are not adequately addressed.”** Removed as a main weakness because this criticism is imported from other papers rather than directly established here. In this submission, the authors do provide direct albedo and roughness metrics on Synthetic4Relight and show improvements over baseline. It is fair to say material analysis is incomplete on other datasets, but not fair to assert unresolved shadow baking without evidence from this paper.
- **“Cross-baseline evaluation is incomplete because each baseline is only tested on its customary dataset.”** Weakened/removed as a core flaw. The paper’s claim is about compatibility with multiple baselines, and it does test three baselines across four datasets. Demanding every baseline on every dataset goes beyond what is needed to support the main point.
- **“No MLP architecture ablation” as a major issue.** Removed as a major weakness. Architecture tuning for a small auxiliary MLP is not central unless there is evidence the result is highly sensitive. It is a reasonable nice-to-have, not a substantive flaw.
- **Any criticism about missing related work.** Removed per instruction.
- **Formatting/typo/style issues.** Removed per instruction.
- **Reproducibility complaints about missing hyperparameter minutiae.** Removed per instruction.

## Novel Insights
The strongest way to read this paper is not as a rigorous physical derivation, but as a useful **structural regularization for disentangled 3DGS inverse rendering**: once opacity and BRDF are optimized as separate channels, coupling them through a shared material-dependent attenuation term can inject a missing dependency that NeRF-like shared-field models get “for free.” This perspective makes the work more compelling than the paper’s current “exact physical correctness” framing. In other words, the real contribution is less “we proved the right opacity law for 3DGS” and more “we found a simple cross-channel coupling that systematically improves inverse-rendering optimization in explicit Gaussian models.”

## Suggestions
- **Reframe the theory claims.** Replace “physically correct,” “exact form,” and “strictly follow” with “physics-inspired” or “Beer-Lambert-motivated,” unless the derivation is substantially strengthened.
- **Add a minimal but decisive ablation suite.** At least compare:
  - baseline alpha,
  - \(1-\exp(-o_i G_i)\) only,
  - \(1-\exp(-o_i G_i f(m_i))\) with gradients detached to materials,
  - full model.
  This would directly test the claimed mechanism.
- **Be more precise about empirical claims.** Say “improves average performance across three baselines” rather than “universal improvement,” and avoid attributing all gains to better material estimation where material metrics are not reported.
- **Quantify efficiency overhead.** Even a short table with training time / memory changes would strengthen the practical contribution.
- **Tone down geometry claims or support them quantitatively.** If normals are a meaningful secondary benefit, measure them.

## Score and Decision
**Novelty:** Good. The idea of making opacity explicitly material-dependent in 3DGS inverse rendering is a real conceptual contribution, especially because it targets a specific weakness of disentangled Gaussian formulations.

**Technical soundness:** Moderate. The method itself is reasonable and the empirical improvements are credible, but the theoretical framing overshoots what is actually derived.

**Empirical support:** Moderate. Multi-baseline evidence is a strong point, and Table 1 is particularly valuable because it includes material metrics. However, the missing ablations leave the mechanism under-verified.

**Significance:** Moderate. The gains are not huge, but the idea is simple, reusable, and likely to influence subsequent 3DGS inverse-rendering work.

**Clarity:** Fair to good. The paper is understandable, but some claims should be stated more carefully to avoid conflating “physically motivated” with “physically established.”

### Calibration against similar papers
I compared this submission against the following human-reviewed papers from the repository:

- **GeoSplatting** (`/home/wg25r/review_agent/human_reviews/l5VA9wHJ8u.md`, Reject, scores 5/5/6/5): another 3DGS inverse-rendering paper. Compared with GeoSplatting, this paper is narrower and less ambitious, but also cleaner in its core idea. It avoids some of GeoSplatting’s broader geometry-heavy claims, yet shares a similar weakness of overclaiming physical interpretation relative to evidence. Overall, OMG feels somewhat stronger in conceptual crispness, but not by a huge margin.
- **GI-GS** (`/home/wg25r/review_agent/human_reviews/hJIEtJlvhL.md`, Accept Poster, scores 6/8/6/8): also a 3DGS inverse-rendering paper, but with a more substantial algorithmic change around indirect illumination. GI-GS was accepted despite some evaluation gaps because the contribution was broader and the empirical/storyline impact was stronger. Relative to GI-GS, OMG is simpler and probably easier to adopt, but its empirical gains are smaller and its theory-to-claim gap is more central to the paper’s identity.
- I also inspected nearby calibration examples from related 3DGS work returned by search, such as **6DGS** (`sUvBTEYXGt.md`, Accept Poster), which helps calibrate that ICLR tends to accept clear, technically sound 3DGS modifications when the evidence is solid and the claims are proportionate.

Putting this together: this paper is **better than a weak reject** because the core idea is real, simple, and empirically useful across multiple baselines. But it falls short of a clear accept because the central theoretical framing is overstated and the ablations needed to validate the claimed mechanism are missing. I land in the **borderline reject / weak reject** regime.

**Score:** 5.8

MY FINAL SCORE: <pineapple>5.8</pineapple>
MY FINAL DECISION: <orange>Reject</orange>