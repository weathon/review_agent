## Summary

The paper introduces MARBLE, a framework for PDE modeling that extends INR-based methods by replacing global modulation with GridMix—a spatial modulation technique that represents modulation parameters as a linear mixture of shared grid-based basis functions—and augmenting training with random spatial domain subsampling. GridMix preserves local detail from grid-based representations while regularizing the modulation space to prevent overfitting to the training domain. MARBLE builds on the CORAL framework and demonstrates substantial error reductions over CORAL and other baselines on Navier-Stokes, Shallow-Water, and geometry-aware inference benchmarks.

## Strengths

- **Clear, well-motivated problem formulation.** The paper identifies a concrete limitation of global modulation (inability to capture local details) and traces the logical progression to spatial modulation overfitting on training domains, motivating each proposed component (GridMix, channel-wise extension, spatial domain augmentation) in response.

- **Strong empirical performance.** MARBLE achieves consistent and often large error reductions over CORAL (up to ~94% on Navier-Stokes with π=20%) and generally outperforms neural operator baselines including FNO, DeepONet, and MP-PDE across dynamics modeling and geometry prediction tasks.

- **Thorough ablation studies.** Tables 4a–4e systematically evaluate each component and hyperparameter, and Table 5 provides a controlled comparison against scaled CORAL baselines, demonstrating that improvements come from architectural design rather than parameter count.

- **Effective visualization of the overfitting problem.** Figure 3 clearly illustrates how vanilla spatial modulation overfits to the training domain while GridMix recovers generalization, supporting the narrative behind the method.

## Weaknesses

### Major

- **Missing critical ablation: vanilla spatial modulation vs. GridMix under controlled conditions.** The core narrative is that vanilla spatial modulation overfits to the training domain and that GridMix fixes this. However, the main ablation (Table 4a) only compares CORAL (global modulation) → CORAL+SDA → CORAL+SDA+MCGM. There is no quantitative row for vanilla spatial modulation with matched training protocol (with and without SDA). While Table 1 and Figure 3 are referenced as showing vanilla spatial modulation's failure, Table 1 is not visible in the paper body, and Figure 3 is purely qualitative (single functions). Without this ablation, the mechanistic claim that "vanilla spatial modulation overfits and GridMix solves it" is not rigorously supported. The data in Table 4a actually shows that SDA alone accounts for most of the improvement (In-t: 2.18e-3 → 4.22e-4, a ~5× gain), while GridMix adds further gains (4.22e-4 → 1.62e-4, a ~2.6× gain on top). This raises the question of whether a properly regularized vanilla spatial modulation with SDA could achieve comparable results at lower complexity.

- **No computational cost analysis.** The paper acknowledges that GridMix "may introduce additional computational overhead and memory requirements" (Section 5), but provides no measurements of training time, inference latency, or GPU memory. GridMix adds multi-channel grid basis functions per layer and interpolation operations at every forward pass, while spatial domain augmentation adds stochastic subsampling to the meta-learning procedure. For a PDE solver paper where efficiency is a key practical concern, this omission hinders practitioners from assessing the trade-off between accuracy gains and computational cost.

### Minor

- **All experiments use SIREN as the sole INR backbone.** The paper acknowledges that SIREN "may struggle with data featuring strong directional anisotropy" and that this explains MARBLE's underperformance on the Pipe dataset versus Geo-FNO/Factorized-FNO. However, no experiments with alternative backbones (e.g., WIRE, MFN, Fourier features) are provided, leaving the generality of GridMix's benefits unverified across INR architectures.

- **Spatial domain augmentation is conceptually simple and underspecified.** SDA (Eqs. 8–9) is a random subsampling of training coordinates, presented as a core contribution. While effective, it is essentially stochastic coordinate-batch training. The paper does not clarify whether X_sub is resampled per inner loop step k or once per meta-learning episode, nor whether SDA is applied during the second stage (training g_ψ). The novelty of this component is modest relative to its impact.

- **All PDE experiments are 2D.** Navier-Stokes is 2D, Shallow-Water uses a 2D spherical grid, and geometry tasks are 2D/3D surface problems. No experiments on 3D volumetric PDEs are provided, where grid-based representations face more severe memory scaling (O(N³)) and the spatial generalization problem is arguably more critical. This limits the paper's claim of "versatility" and broad applicability.

- **Underperformance on Pipe dataset.** MARBLE underperforms Geo-FNO and Factorized-FNO on Pipe (Table 3), achieving 1.03e-2 vs. 6.59e-3 and 7.33e-3 respectively. The SIREN-backbone explanation is plausible but unverified experimentally.

### Trivial

- Notation is inconsistent in places (e.g., $\mathcal{L}_{X_{tr}}$ vs. $\mathcal{L}_{\mathcal{X}_{tr}}$; "GirdMix" in Section 3 title).

## Nice-to-Haves

- **Visualize learned grid basis functions** ($\Phi_i^m$) across layers. This would directly verify whether GridMix captures meaningful global structural modes or simply acts as implicit regularization, which is the core mechanistic claim.

- **Test with alternative INR backbones** (WIRE, MFN, etc.) to demonstrate GridMix's generality beyond SIREN and potentially address the Pipe weakness.

- **Extend to higher-dimensional PDEs** (3D+time problems) where the locality-globality trade-off is more acute and memory scaling of grid representations becomes a serious concern.

- **Provide hyperparameter selection guidance** for M (number of basis functions) based on problem characteristics, beyond the empirical sensitivity analysis in Table 4d.

## Removed Points

- **Missing comparison with hash-grid baselines (Instant-NGP style).** While interesting, this requests a specific alternative architecture that is not standard in the PDE modeling literature. The paper motivates GridMix as an alternative to vanilla grid-based representations and global modulation, which are the appropriate comparisons within its scope. *Treat with caution — this could strengthen the paper but is not a required baseline for the PDE application domain.*

- **Lack of comparison with recent neural operator baselines beyond Serrano et al. (2023).** The instruction states not to flag missing related works or baselines that may not exist in the reviewed version. The paper uses published baseline numbers from a recent paper, which is standard practice. *Treat with caution — the field moves fast, but this is not a paper flaw per se.*

- **Missing classical interpolation baselines.** The paper already includes FNO+lin.int. for irregular settings. Requesting spline or RBF interpolation baselines goes beyond the paper's scope of comparing against neural PDE solvers. *Treat with caution — absolute error contextualization would be nice but is not standard for this community.*

- **Reproducibility concerns about training details.** Minor specification gaps (resampling schedule of X_sub, etc.) are routine in conference papers where supplementary details are in appendices. *Treat with caution — the core algorithm is sufficiently specified for reproduction.*

- **SDA is not novel.** While simple, SDA is not presented as a novel algorithmic contribution (it's presented as an augmentation strategy), and its effectiveness is empirically validated. The novelty claim rests on GridMix. *Treat with caution — the paper should be clearer that SDA is an engineering contribution, not a conceptual one.*

- **Insufficient theoretical grounding for GridMix.** The paper makes an analogy to spectral methods but does not provide formal analysis. However, theoretical guarantees for neural PDE solvers are not standard in this community, and the empirical evidence is substantial. *Treat with caution — formal analysis would strengthen the paper but is not required by community standards.*

## Novel Insights

The paper identifies an important and underappreciated trade-off in neural field modulation for PDE tasks: grid-based spatial modulation excels at capturing local detail but catastrophically overfits to the training domain, a problem that is largely irrelevant in vision tasks (where domains are fixed) but critical for PDE modeling (where generalization to unseen spatial regions is essential). The GridMix solution—constraining spatial modulation to the linear span of shared basis grids—is an elegant way to interpolate between global and local modulation, and the connection to spectral decomposition ideas makes the regularization effect intuitive. The most important empirical insight is that spatial domain augmentation (simple random coordinate subsampling) accounts for a substantial portion of the performance gains on sparse grids, suggesting that the generalization problem is partly an artifact of how meta-learning trains INRs on fixed domains, not just a modulation capacity issue.

## Suggestions

- Add a vanilla spatial modulation ablation to Table 4 (with and without SDA) to directly substantiate the claim that GridMix solves the overfitting problem that vanilla spatial modulation exhibits.
- Report wall-clock training time and peak GPU memory for MARBLE vs. CORAL baselines to allow practitioners to assess the accuracy-cost trade-off.
- Clarify the training protocol: specify whether spatial domain augmentation is applied only during the reconstruction/meta-learning stage or also during the forecasting stage, and whether X_sub is resampled per inner-loop step or per episode.
- Visualize the learned basis grids $\Phi_i^m$ to provide insight into what global structures GridMix captures, making the spectral analogy concrete.

## Score and Decision

Calibration against similar papers:
- **Coordinate-Aware Modulation (4UiLqimGm5.md, scores 6–8):** A strong paper on grid-based modulation for neural fields; MARBLE has similarly motivated contributions but weaker experimental rigor (missing critical ablation) and no computational cost analysis.
- **How Learnable Grids (Ge7okBGZYi.md, scores 3–8):** Concerns about limited applicability and grid design choices; MARBLE has more thorough evaluation but similar scope limitations (only 2D).
- **INR field reconstruction (kuTZMZdCPZ.md, scores 6):** Decent but not exceptional; MARBLE has stronger empirical gains but comparable scope.
- **Accurate Differential Operators (HxHrRUHMOD.md, scores 3–8):** Polarized reviews; low scores due to scalability concerns and single-architecture experiments — similar to MARBLE's SIREN-only issue.

MARBLE sits above the mid-range PDE solver papers (which scored 3–5) due to its strong empirical results and clear motivation, but below the top-scoring grid/INR papers (6–8) due to the missing vanilla spatial modulation ablation, absence of computational cost analysis, and limited backbone diversity. The core contribution (GridMix as regularized spatial modulation) is sound and well-motivated, but the claimed mechanistic explanation ("vanilla spatial modulation overfits, GridMix fixes it") is under-supported by the quantitative ablations, and SDA—arguably the most impactful single component—has limited novelty.

MY FINAL SCORE: <pineapple>6</pineapple>
MY FINAL DECISION: <orange>Accept</orange>