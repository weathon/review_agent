=== CALIBRATION EXAMPLE 84 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title accurately reflects the paper’s main contribution: a benchmark for real-world PDE/physics datasets. “REALPDEBENCH” and “complex physical systems with real-world data” are consistent with the content.
- The abstract clearly states the problem (lack of real-world data in scientific ML), the method/contribution (benchmark with paired real and simulated data), and the evaluation setup (five datasets, three tasks, nine metrics, ten baselines).
- The main claim that this is “the first benchmark for scientific ML that integrates real-world measurements with paired numerical simulations” is plausible but strong. The paper does compare against prior benchmarks and a concurrent work (REALM), but “first” should be justified very carefully because the related-work discussion is not fully exhaustive in demonstrating that no earlier benchmark had paired real/simulated measurements across any domain.
- The abstract’s claim that “pretraining with simulated data consistently improves both accuracy and convergence” is broadly supported by the reported results, but it is slightly over-generalized: the gains are not uniform across all metrics, datasets, and baselines, and some tables show only modest or mixed improvements depending on the metric.

### Introduction & Motivation
- The motivation is strong and very relevant to ICLR: a central gap in scientific ML is that most methods are validated on simulation, not measurement. The paper clearly articulates why this matters for sim-to-real transfer and deployment.
- The introduction states four contributions: data, tasks, metrics, and baselines. This is helpful and accurate at a high level.
- A key strength is that the benchmark is not just a dataset dump; it includes paired real and simulated trajectories and a defined protocol for comparing training regimes. That is a meaningful benchmark contribution.
- The introduction does somewhat over-claim in saying current scientific ML “cannot truly evaluate” real-world performance. That is too absolute; the real point is that evaluation is limited and often unrepresentative, not impossible.
- The novelty relative to existing benchmark efforts should be framed more cautiously. The paper is compelling, but its “first” claim and broad positioning would benefit from sharper differentiation from related real-data efforts and from the concurrent REALM benchmark mentioned in Sec. 2.

### Method / Approach
- The benchmark design is mostly clear: three training regimes, five datasets, nine metrics, and a standardized split protocol.
- The task definition in Sec. 3.1 is understandable, but one issue matters for reproducibility and interpretation: for simulated training, the model is trained on all \(N\) simulated samples, whereas evaluation is on real-world test data. This is fine, but the paper should be more explicit about whether validation for simulated-training is also from real-world data only, and how early stopping/model selection is done comparably across regimes.
- The “three categories of prediction tasks” are useful, but the terminology is slightly confusing because “real-world finetuning” is used in Sec. 4 to refer to the simulated-pretraining plus real-world-finetuning regime. That is manageable, but the paper should be careful to avoid ambiguity.
- The metrics are a key part of the contribution, yet some definitions are under-specified or appear inconsistent in notation. For example:
  - fRMSE is described as using a 3D FFT and grouping by frequency magnitude, but the datasets include 2D cross-sections and in some cases scalar fields. The paper should clarify exactly how the FFT is applied per dataset/channel and what dimensions are transformed.
  - FE is defined as an MAE between spectra of temporally aggregated signals, but the exact normalization and handling of multiple channels/probes are not fully specified.
  - KE is only meaningful for velocity fields, which is acknowledged later, but the paper should state more explicitly how the metric is excluded/handled on combustion in the main metric definition.
  - Update Ratio is a useful idea, but the paper defines it relative to the best RMSE of real-world training. This makes it dependent on a chosen optimization budget and stopping rule; the paper should explain how update counts are compared fairly across models with very different optimization dynamics.
- The mask-training strategy in Sec. D.1 is conceptually reasonable, but it raises an important methodological question: randomly masking simulated modalities to mimic missing measurements is a strong design choice that can materially affect the simulated-pretraining results. The paper should discuss sensitivity to mask probability and whether this is tuned per dataset.
- The combustion surrogate in Sec. D.2 is a substantial extra modeling layer: simulated inputs are converted to real-world modalities by a learned surrogate. This is important enough that its error propagation should be more directly analyzed. As written, it is not fully clear how much of the combustion performance comes from the benchmarked model versus the surrogate’s approximation quality.
- The theoretical side is not central here, and there are no major derivations to assess, but the methodology would benefit from a clearer statement of all assumptions about alignment between real and simulated trajectories, temporal synchronization, and spatial registration.

### Experiments & Results
- The experiments do test the paper’s main claims: they compare real-world training, simulated training, and simulated pretraining across multiple datasets and baselines, with real-world test evaluation.
- The baseline set is generally appropriate for a benchmark paper in this area. It spans classical DMD, CNN-based methods, neural operators, transformer-style operators, and a pretrained foundation model (DPOT). That breadth is a strength.
- The comparison to DPOT is particularly relevant for ICLR, since benchmark relevance depends on testing modern pretrained models, not only older operator learners.
- However, several experimental design issues limit how strongly the results support the conclusions:
  - There is no clear ablation isolating which part of the simulated-pretraining gain comes from pretraining itself versus the mask-training strategy or modality augmentation.
  - For combustion, the surrogate model introduces an additional learned component that is not comparable to the other datasets; the benchmark results there may not isolate model quality cleanly.
  - The paper does not report uncertainty estimates, confidence intervals, or significance tests. For a benchmark paper, especially when claiming consistent gains, this is a limitation.
  - The Update Ratio is interesting but not enough on its own to establish convergence improvements because it depends on optimization settings and stopping criteria. A curve-based or budget-matched analysis would be more convincing.
- The reported findings are mostly consistent with the tables:
  - Table 1 and Appendix Tables 2–6 show that real-world finetuning often outperforms training from scratch on real-world data.
  - The claims about frequency-domain behavior are supported by Figure 3a and Figure 6, but these are fairly narrow diagnostics.
  - The claim that U-Net and DPOT-L-FT are strong is supported in many tables, though not uniformly across all tasks.
- One important concern is that some summary claims appear stronger than the evidence. For instance, saying simulated pretraining “consistently improves both accuracy and convergence” is directionally true, but the tables show exceptions and metric-dependent variability, especially for long-horizon autoregressive settings and some frequency metrics.
- Dataset and metric choices are generally appropriate for the benchmark’s goals. The paired-data setup is valuable, and the inclusion of physics-oriented metrics is a plus. That said, the benchmark would be stronger with clearer protocol details on splits, preprocessing, and any hyperparameter tuning performed per baseline/dataset.

### Writing & Clarity
- The paper is readable overall and the high-level story is coherent, but some sections leave important technical ambiguity.
- The clearest weakness is the metric and evaluation description: the reader has to work hard to understand exactly how fRMSE, FE, KE, and MVPE are computed for each dataset and prediction mode.
- Another clarity issue is that the paper sometimes mixes benchmark description with empirical interpretation without fully separating them. For example, Sec. 4.2 and Sec. 4.3 make causal-sounding claims about why pretraining helps, but the evidence is correlational.
- The figures generally serve their purpose, especially Figure 1 (real vs simulated differences), Figure 3 (frequency error and finetuning curve), Figure 4 (trade-off plot), and Figure 5 (MVPE). Still, some of the figures and tables are hard to parse in the extracted text, and in the paper itself the main tables should be easier to consult if the benchmark is to be widely used.
- The paper would benefit from a more compact presentation of the benchmark protocol, especially because ICLR readers will care about replicability and how to use the benchmark correctly.

### Limitations & Broader Impact
- The limitations section is present and acknowledges several important constraints: limited domain coverage, lack of combustion-specific metrics, and insufficient exploration of strong OOD regimes.
- That said, several key limitations remain under-discussed:
  - The benchmark covers only five scenarios, all from fluids/combustion, so generality across other scientific domains is still untested.
  - Real-world data collection and simulation are not equally detailed across tasks, and the combustion pipeline in particular relies on a surrogate mapping from simulated to measured modalities.
  - The benchmark’s paired-data design may encourage methods that exploit benchmark-specific alignment assumptions rather than truly robust sim-to-real adaptation.
- Broader impact is modest and mostly positive, but the paper correctly notes that models trained on this benchmark should not be used directly in safety-critical contexts. Given the application domains, that caution is important.
- A missing discussion is the potential for overfitting to the benchmark’s specific measurement process. Because real-world observability is partially limited by PIV/CL techniques, methods might optimize to these sensors rather than to underlying physics.

### Overall Assessment
RealPDEBench is a valuable and timely benchmark contribution for ICLR: it addresses a real gap in scientific ML by pairing measured data with simulations, defines a useful comparison protocol, and includes a broad set of models and metrics. The strongest aspect is the dataset contribution itself, which is substantial and likely to be useful to the community. The main concerns are methodological and evidential rather than fatal: the evaluation protocol needs clearer specification in a few places, the combustion setting introduces an extra surrogate stage that complicates attribution, and the paper’s claims about consistency and “firstness” are a bit stronger than the evidence currently supports. Despite these issues, the contribution appears significant and likely stands as a meaningful benchmark paper, but it would benefit from sharper methodological transparency and more cautious claims to meet ICLR’s standards for strong empirical rigor.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces RealPDEBench, a benchmark designed to evaluate scientific ML models on real-world physical systems rather than only on simulations. It provides five paired real/simulated datasets spanning fluid dynamics and combustion, defines three training settings (simulation-only, real-only, and simulation pretraining plus real fine-tuning), and evaluates ten baselines with a mix of data-oriented and physics-oriented metrics.

### Strengths
1. **Clear and timely motivation aligned with an important ICLR need.**  
   The paper targets a major gap in scientific ML: most models are benchmarked on idealized simulations, not real measurements. This is highly relevant to ICLR’s interest in ML systems that generalize beyond toy or curated settings.

2. **Substantial benchmark construction effort.**  
   The dataset collection is nontrivial: the paper reports 736 paired trajectories across five scenarios, with real experiments and matched simulations, covering cylinder flow, controlled cylinder, FSI, foil, and combustion. The inclusion of both measured and simulated data is a meaningful contribution.

3. **Thoughtful task design for sim-to-real evaluation.**  
   The three training regimes are well motivated and useful: real-world training, simulated training, and simulated pretraining with real-world fine-tuning. This directly tests a core practical question in scientific ML: whether synthetic data can help real-world performance.

4. **Multiple evaluation dimensions beyond pointwise error.**  
   The benchmark uses nine metrics including RMSE, relative L2, R², Fourier-space error, frequency error, kinetic energy error, and mean velocity profile error. This is a strength because ICLR reviewers often look favorably on evaluations that probe both accuracy and physics consistency.

5. **Empirical evidence that sim-to-real transfer helps.**  
   The experiments consistently suggest that pretraining on simulated data improves performance and convergence on real-world data. The update-ratio analysis and finetuning curves are useful evidence that the benchmark can reveal more than just final error numbers.

6. **Good breadth of baselines, including pretrained foundation models.**  
   The comparison includes neural operators, convolutional models, transformers, a traditional method, and DPOT pretrained models. This makes the benchmark more useful for the community and increases the practical value of the results.

7. **Reproducibility-oriented presentation.**  
   The paper states that code, data, checkpoints, and scripts are released, and it describes a unified framework for loading datasets and models. For a benchmark paper, this is an important plus.

### Weaknesses
1. **The benchmark’s novelty is strong in data collection, but the methodological novelty is limited.**  
   The main contribution is a dataset/benchmark rather than a new ML method. That can still be acceptable at ICLR if the benchmark is clearly novel and impactful, but the paper’s scientific claims are largely empirical and descriptive rather than algorithmic.

2. **Some benchmark design choices are not fully justified.**  
   For example, the paper fixes evaluation on real-world test data and introduces masking/noise heuristics for simulated training, but the rationale for specific choices and their sensitivity is not deeply analyzed. ICLR reviewers would likely want stronger evidence that the benchmark protocol is robust and not overly tailored to the reported baselines.

3. **Limited coverage of out-of-distribution generalization.**  
   The paper mentions OOD support in the codebase, but the main results do not systematically explore strong distribution shift. For an ICLR benchmark paper, this is a notable omission because generalization under shift is central to modern ML evaluation.

4. **Physics coverage is narrower than the paper’s framing suggests.**  
   The benchmark is restricted to fluid dynamics and combustion. While these are important, the introduction positions the benchmark as a broader step for “complex physical systems,” yet the current scope is still domain-specific. The paper itself acknowledges this limitation.

5. **Potential confounding from modality mismatch and preprocessing.**  
   Simulated and real data differ in available channels, and the paper uses masking plus surrogate modeling for combustion to bridge these differences. This is sensible, but it also makes it harder to isolate whether gains come from better dynamics learning or from auxiliary preprocessing/modeling choices.

6. **The experimental comparison is extensive but somewhat descriptive.**  
   Many results are tabulated, but the paper offers limited deeper analysis of *why* specific methods succeed or fail on particular datasets. For instance, architectural conclusions are mostly high level (“U-Net does well because tasks resemble image processing”) and not rigorously supported.

7. **Reproducibility details are promising but not yet fully sufficient in the paper text alone.**  
   Although the paper claims code/data release, key details such as exact hyperparameters, training budgets, random seed handling, and preprocessing choices for each baseline are scattered across appendices. For a benchmark paper, ICLR would expect a very precise recipe enabling exact replication.

### Novelty & Significance
**Novelty:** Moderate to strong for a benchmark paper. The combination of real experimental measurements, paired simulations, multiple physical systems, and sim-to-real evaluation is a meaningful contribution. The paper’s novelty is primarily in the benchmark/data resource rather than in a new learning algorithm.

**Significance:** High if the datasets are truly high quality and broadly usable. Real-world scientific ML benchmarks are rare, and a benchmark that exposes the sim-to-real gap could shape future work in the field. That said, the significance depends on whether the datasets are robust, well documented, and broadly adopted beyond the authors’ own baselines.

**Clarity:** Generally clear in its high-level goals and structure, but some evaluation protocol details and dataset-specific decisions are not fully justified in the main narrative. The paper is readable, but the benchmark framing would benefit from a sharper discussion of what exactly is new relative to prior real-data efforts.

**Reproducibility:** Above average for a benchmark paper, given the stated release of code, data, and checkpoints. Still, exact reproducibility would depend on whether the released materials include precise preprocessing, calibration, and training settings for every baseline and dataset.

**ICLR significance vs acceptance bar:** This is the kind of benchmark paper that can meet ICLR’s bar if the datasets are credible and likely to influence future work. However, to be clearly competitive at ICLR, it would benefit from a more rigorous analysis of benchmark validity, stronger OOD evaluation, and more evidence that the benchmark yields insights not easily obtainable from existing simulated-data benchmarks.

### Suggestions for Improvement
1. **Add a systematic benchmark validity study.**  
   Evaluate how sensitive results are to masking probability, noise scale, train/val/test splits, and preprocessing. This would strengthen confidence that the benchmark conclusions are robust.

2. **Include stronger out-of-distribution experiments.**  
   Since the code supports OOD modes, the paper should present results on held-out Reynolds numbers, control frequencies, angles of attack, and combustion settings that are genuinely outside the training range.

3. **Provide deeper analysis of sim-to-real transfer.**  
   Compare several transfer strategies beyond simple finetuning, such as feature alignment, multi-fidelity training, domain adaptation, or uncertainty-aware selection. This would better justify the benchmark’s utility.

4. **Clarify the effect of modality mismatch.**  
   Since simulated data have extra channels, run ablations that separate the contribution of extra modalities from the contribution of additional data volume. This would make the transfer conclusions more convincing.

5. **Strengthen the methodological discussion of baseline failures and successes.**  
   Go beyond qualitative statements and analyze failure modes by dataset and metric, especially for combustion and autoregressive rollouts.

6. **Report exact training details more uniformly.**  
   Provide a compact appendix table with hyperparameters, batch sizes, learning rates, number of updates, model selection criteria, and seed counts for every baseline and dataset.

7. **Clarify benchmark scope and positioning relative to prior real-data efforts.**  
   A more explicit comparison table against REALM and other datasets should distinguish paired real/sim data, number of trajectories, modalities, resolution, parameter coverage, and evaluation tasks.

8. **Add community-facing benchmark documentation.**  
   Since this is a benchmark submission, a concise “how to use RealPDEBench” guide and canonical baselines would help adoption, which is especially important for ICLR impact.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add direct cross-domain transfer experiments: train on simulation and test on real-world without finetuning, then compare against finetuning and real-world-only training. The paper claims sim-to-real bridging, but the core evidence is mostly within-domain real-world evaluation; without this, the sim-to-real contribution is not convincing for ICLR.

2. Add stronger baselines for domain adaptation and physics transfer, not just forecasting models. At minimum include simple and standard alignment methods (e.g., feature normalization, CORAL/MMD-style adaptation, adversarial domain adaptation, teacher-student distillation) because the benchmark’s main claim is about the sim-to-real gap, and current baselines do not isolate that problem.

3. Add ablations on the simulation-specific tricks: mask-training, noise injection, surrogate combustion modality mapping, and DPOT resizing/padding. The main claims that simulated data helps and that masking/unmeasured modalities matter are unsupported without showing how much each design choice contributes.

4. Add a fair robustness study across multiple train-set sizes and parameter regimes, especially low-data and out-of-distribution splits. ICLR will expect evidence that the benchmark is useful beyond one fixed split; otherwise the reported gains may be split-specific and not generalizable.

5. Add calibration/uncertainty baselines or metrics for real-world noise sensitivity. Since the benchmark emphasizes noisy measurements and measurement/simulation mismatch, deterministic point prediction alone is not enough to justify claims about real-world deployment.

### Deeper Analysis Needed (top 3-5 only)
1. Quantify the sim-to-real gap separately by dataset, modality, and frequency band, and connect it to physical causes. The paper states that real and simulated data differ, but does not analyze which discrepancies matter most or why certain models fail, so the benchmark’s scientific value is underexplained.

2. Analyze whether simulated pretraining helps because of more data, better inductive bias, or access to extra modalities. Without disentangling these factors, the claim that simulated data “bridges” real-world learning is ambiguous and could just reflect scale or regularization effects.

3. Report statistical variability over multiple random seeds for all main results. For benchmark papers at ICLR, single-number tables are not sufficient when improvements are modest; seed variance is needed to judge whether observed gains are stable.

4. Examine failure modes under long-horizon rollout, not just aggregate autoregressive error. The current analysis shows error growth, but does not identify when trajectories diverge, whether phase drift or amplitude drift dominates, or which baselines are stable in physically meaningful ways.

5. Analyze whether ranking changes across metrics are consistent or conflicting. The benchmark uses many metrics, but the paper does not resolve when local accuracy, spectral accuracy, and physics metrics disagree, which is important for judging whether the benchmark gives actionable model comparisons.

### Visualizations & Case Studies
1. Add side-by-side real vs. simulated vs. finetuned trajectory visualizations for each dataset, with error maps over time. This would expose whether models genuinely correct sim-to-real mismatch or merely fit average patterns.

2. Show failure cases where the best RMSE model has poor physical fidelity, and vice versa. That would make the data-vs-physics metric tradeoff concrete and reveal whether the benchmark actually distinguishes scientifically useful models.

3. Add spectral/phase evolution plots for representative trajectories, especially on Controlled Cylinder and Combustion. The current figures are too aggregate; these plots would reveal whether models capture frequency, phase lock-in, and oscillatory instabilities.

4. Add qualitative comparisons of predicted vs. true mean velocity profiles at multiple horizons, not just one 10-round example. This would show whether autoregressive predictions preserve flow structure or collapse over time.

5. Add a few failure-mode case studies for the hardest setting, especially Combustion. Since this is the most novel real-world scenario, examples of what breaks there would be the most informative evidence for the benchmark’s intended challenge.

### Obvious Next Steps
1. Add a standardized sim-to-real adaptation track as a first-class task in the benchmark. Right now the benchmark documents a gap but does not fully operationalize the main problem it is motivating.

2. Add out-of-distribution parameter splits as a mandatory evaluation setting. ICLR reviewers will expect a benchmark claiming real-world generalization to include harder generalization tests, not only seen-parameter interpolation.

3. Add a unified benchmark leaderboard protocol with fixed preprocessing, training budgets, and tuning rules. Without this, the reported comparisons may be confounded by implementation choices and unequal optimization effort.

4. Add physics-aware objectives or constraints as optional baselines. Since the benchmark is meant to advance scientific ML, the next step should test whether physics-informed or hybrid methods outperform purely data-driven ones on real data.

5. Add a benchmark protocol for uncertainty estimation and decision-making under measurement noise. That is a natural extension for real-world physical systems and would materially strengthen the benchmark’s impact beyond point prediction.

# Final Consolidated Review
## Summary
This paper introduces RealPDEBench, a benchmark for scientific ML on real-world physical systems with paired numerical simulations. It collects five datasets across fluid dynamics and combustion, defines three training regimes centered on sim-to-real transfer, and evaluates a broad set of baselines with both standard and physics-oriented metrics.

## Strengths
- The dataset contribution is substantial and genuinely useful: the paper reports 736 paired real/simulated trajectories across five challenging physical scenarios, with real measurements collected via PIV or chemiluminescence and matched simulations under corresponding parameters.
- The benchmark is thoughtfully structured around a real scientific ML question: whether simulated data help on real-world prediction. The three training regimes, real-world-only evaluation, and inclusion of frequency/physics metrics make the benchmark more informative than a standard forecasting suite.

## Weaknesses
- The paper’s central sim-to-real claim is only partially isolated. Simulated pretraining is confounded by several design choices: extra simulated modalities, random masking, and for combustion an additional surrogate mapping from simulated to real modalities. As a result, it is hard to tell how much of the improvement comes from better transfer versus auxiliary preprocessing and added capacity.
- The experimental protocol is not rigorous enough for a benchmark paper claiming “consistent” gains. The paper reports single-run tables without uncertainty across seeds, no confidence intervals or significance tests, and limited ablations on masking, noise injection, or dataset size. That makes the reported improvements less convincing than the headline language suggests.
- The benchmark does not really stress the main generalization problem it motivates. Although the code supports OOD modes, the main results focus on fixed parameter-level splits and within-distribution comparisons. For a sim-to-real benchmark, stronger held-out-regime evaluation should be core, not optional.
- Several metric and evaluation definitions are still under-specified in the main text. The physics-oriented metrics are sensible, but details such as how frequency transforms are applied across datasets/channels, how update ratio is normalized fairly, and how missing metrics are handled per modality are not sufficiently explicit for easy reuse.

## Nice-to-Haves
- A compact benchmark-validity study varying mask probability, noise scale, training budget, and split choice.
- A standardized “leaderboard protocol” table with exact preprocessing, tuning rules, seed counts, and stopping criteria for each baseline.
- More direct qualitative comparisons of real vs. simulated vs. finetuned trajectories, especially on the combustion dataset.

## Novel Insights
The most interesting insight is not simply that real data differ from simulation, but that the benchmark exposes a three-way tension between data fidelity, modality completeness, and transfer utility. The paper’s results suggest that simulated data can help real-world prediction, but only when the model can exploit extra structure in simulation and when the benchmark’s sensor mismatch is bridged carefully; otherwise the gains are ambiguous. That makes RealPDEBench potentially valuable, but also means its headline conclusions are more about a particular engineered sim-to-real pipeline than a clean causal statement about simulation pretraining in general.

## Potentially Missed Related Work
- REALM (Mao et al., 2025) — a concurrent benchmark for neural surrogates on realistic multiphysics reactive flows; directly relevant for positioning and comparison.
- CFDBench / FlowBench / LagrangeBench / PDEBench — relevant prior benchmarks for fluid/PDE learning; useful for clearer differentiation in scope and protocol.
- None identified beyond the above as clearly missing for the paper’s main claims.

## Suggestions
- Add an explicit ablation disentangling: pretraining benefit, mask-training benefit, extra simulated modality benefit, and combustion surrogate benefit.
- Make OOD evaluation a first-class result section, with held-out Reynolds numbers, forcing frequencies, angles of attack, and combustion settings.
- Report multi-seed variability for the main tables and include a short protocol appendix that standardizes training budgets and preprocessing across baselines.

# Actual Human Scores
Individual reviewer scores: [10.0, 10.0, 6.0, 4.0]
Average score: 7.5
Binary outcome: Accept
