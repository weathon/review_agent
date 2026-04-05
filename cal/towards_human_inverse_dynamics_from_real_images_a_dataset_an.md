=== CALIBRATION EXAMPLE 35 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title accurately signals the paper’s main thrust: predicting human joint torques from real images, with a dataset and benchmark. That said, “human inverse dynamics from real images” is a strong claim, and the paper should be careful to distinguish visual torque estimation from full inverse dynamics in the biomechanical sense.
- The abstract does clearly state the problem, dataset, and baseline, but several claims are too broad relative to what is actually demonstrated. In particular:
  - “the first dataset tailored for the joint torque prediction from real human images” appears plausible, but the paper should substantiate novelty more carefully against prior vision-based biomechanics resources.
  - “our baseline method achieves the state-of-the-art performance on almost all the evaluation criteria” is hard to verify because the comparison is only against two methods on the same dataset, one of which is not a vision method.
- The abstract does not clearly state the evaluation protocol, the task difficulty, or the limitations of the benchmark, which matters for an ICLR submission claiming a new benchmark.

### Introduction & Motivation
- The motivation is strong: torque estimation from unconstrained human images is a genuinely interesting and underexplored problem. This fits ICLR’s interest in new ML problem settings and datasets.
- However, the gap in prior work is not cleanly articulated. The introduction mixes three separate lines of work—sEMG-based estimation, classical inverse dynamics, and imitation learning—without clearly showing why existing vision methods are insufficient or how the proposed dataset resolves a specific limitation.
- The contribution statements are clear, but the paper over-claims in several places:
  - It repeatedly implies “purely real human image-based” torque estimation, but the method still relies heavily on kinematic supervision, marker regression, and pretraining on 3D pose datasets.
  - It presents the dataset as enabling end-to-end mapping from images to dynamics, yet the benchmark and model remain tightly coupled to marker/pose intermediates rather than a truly direct image-to-torque model.
- For ICLR standards, the introduction would benefit from a sharper statement of what is algorithmically new versus what is primarily dataset/benchmark contribution.

### Method / Approach
- The overall pipeline is understandable at a high level: a ResNet backbone, a spatial probabilistic model, a soft-argmax pose estimator, a marker regressor, and a temporal Transformer for torque prediction.
- But the method description has several important reproducibility and logic gaps:
  - The formulation in Eq. (2) lists marker positions, velocities, height, and mass as inputs to VID, but later the paper says the approach is “purely visual” and that only the image is the model’s input. It is not clear how subject metadata are used at train/test time.
  - The soft-argmax formulation in Eqs. (3–6) is partially garbled, but the intended method is still clear. What is missing is a precise definition of coordinate normalization and how 3D heatmaps are parameterized from monocular images.
  - The marker regressor is under-specified. It says the regressor maps “spatial probabilistic features” to marker coordinates, but the exact dimensionality, which markers are predicted, and whether the regressor uses pose coordinates, heatmaps, or backbone features are unclear and somewhat inconsistent.
  - TorqueInferNet uses “centered prediction” over 13 frames, but the paper does not specify whether inference is causal or bidirectional, which matters for practical deployment claims.
- There are also conceptual issues:
  - Predicting joint torques from a single monocular image is an ill-posed problem without explicit motion cues; the paper resolves this by using a temporal window, but this should be stated more honestly as video-based torque estimation.
  - The method appears to depend on pose pretraining on external 3D pose datasets, but there is no discussion of domain mismatch between those datasets and the VID data.
- The loss design is simple and reasonable, but the constraint that the two weights sum to 1 is arbitrary and not justified. There is no sensitivity analysis for this choice.

### Experiments & Results
- The experiments do not fully test the paper’s claims.
  - The main result is a comparison against Dino and ImDy, but both baselines are not vision-based methods. This makes the strongest claim—“real image-based inverse dynamics”—less well supported, because the paper does not compare against any existing image-to-motion or image-to-biomechanics baseline adapted to this task.
  - Since the paper’s main contribution is a dataset and benchmark, the evaluation should more carefully isolate whether the gain comes from the dataset, the architecture, or the supervision setup.
- The baselines are only partially appropriate:
  - Comparing to ImDy makes sense as a dynamics baseline, but ImDy is trained on imitation observations, not real images.
  - Dino is a hybrid method, but the paper does not explain whether its inputs were converted fairly to match the VID setting.
- Missing ablations are substantial:
  - No ablation on the pose pretraining source.
  - No ablation on marker regressor versus using predicted joint coordinates directly.
  - No ablation on temporal window length beyond the small appendix table, and no study of whether the model actually needs the full 13-frame window.
  - No ablation on subject metadata such as height/mass, despite these being included in the formulation.
  - No comparison to a simpler image-only temporal baseline, which is especially important for validating the necessity of the spatial probabilistic design.
- Statistical rigor is limited:
  - No error bars, confidence intervals, or significance tests are reported.
  - The train/test split is a single 8:2 split; there is no cross-subject evaluation, which is especially important in biomechanics and explicitly mentioned as a future direction.
- The results support that the proposed model performs better than the selected baselines on this dataset, but they do not yet establish a strong benchmark in the ICLR sense because the evaluation is narrow and the comparison set is weak.
- Dataset and metric choices are reasonable for a first benchmark, but the metric definition is not fully transparent. mPJE is said to be normalized by body weight, yet the exact formula in Eq. (8) is not clearly presented, making it hard to reproduce or interpret the magnitude of the reported errors.

### Writing & Clarity
- The paper is understandable at a high level, but several sections are confusing in ways that matter for the contribution:
  - The dataset section mixes claims about synchronization, smoothing, marker annotation, and OpenSim export without a fully coherent description of the preprocessing pipeline.
  - The method section conflates images, markers, pose coordinates, and spatial features in a way that makes it hard to tell what is predicted from what.
  - The evaluation section introduces three “new criteria,” but these are really three views of the same metric applied at different granularities. The novelty of the benchmark protocol is therefore somewhat overstated.
- Figures and tables are useful in principle, especially the dataset comparison and the joint/action breakdowns, but the manuscript would benefit from clearer explanation of what exactly is being compared in Tables 2–5.
- The appendix helps with dataset details, but the main paper should not depend so heavily on the appendix for basic understanding of the data layout and ground-truth torque variables.

### Limitations & Broader Impact
- The paper acknowledges some future directions in the conclusion, especially cross-subject generalization and in-the-wild settings, which is good.
- However, the key limitations are under-discussed:
  - The benchmark is derived from a single source dataset with a limited number of subjects and controlled laboratory actions. This is a major limitation for claims about unconstrained environments.
  - The method depends on pose pretraining and temporal context, so it is not really a direct image-only torque estimator.
  - There is no discussion of domain shift, occlusion, clothing variation, or camera viewpoint changes, all of which are central to real-world deployment.
- Broader impact is largely absent. Because this is a biomechanics paper, the ethical risks may be modest, but there should still be discussion of privacy implications for using body images and possible misuse in surveillance or performance profiling.
- The appendix section on LLM use is not a scientific limitation discussion and does not substitute for a real statement of methodological boundaries.

### Overall Assessment
This paper addresses an interesting and timely problem and contributes a potentially valuable dataset plus a baseline benchmark for vision-based human inverse dynamics. The strongest point is the creation of VID and the fact that the authors have carefully synchronized visual, kinematic, and dynamic data. However, for ICLR’s acceptance bar, the work currently feels more like a dataset/benchmark paper with an incremental baseline than a methodologically deep learning contribution. The main concerns are limited baseline coverage, weak experimental validation of the central claims, insufficient cross-subject/generalization testing, and some conceptual ambiguity about whether the task is truly “from real images” versus from temporally windowed, pose-assisted supervision. The contribution is real, but the paper would need stronger experimental rigor and a clearer formulation to stand confidently at ICLR.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces VID, a new dataset for vision-based inverse dynamics that pairs synchronized real human images with kinematic and dynamic annotations, and proposes a baseline model (VID-Network) for predicting joint torques directly from images. The authors position this as the first benchmark for estimating human joint torques from real images, and report improved performance over two prior methods on their proposed evaluation protocol.

### Strengths
1. **Timely and potentially impactful problem setting.**  
   Predicting joint torques directly from real human images is a meaningful and underexplored direction, with clear applications in biomechanics, sports, and rehabilitation. Framing the task around unconstrained, image-based inference is aligned with ICLR’s interest in learning methods that extend beyond lab-only settings.

2. **A new dataset with multi-modal annotations.**  
   VID appears to provide a useful combination of synchronized monocular video/images, kinematics, and dynamics for 9 subjects and 63,369 frames. The paper also states that the data were manually synchronized, smoothed, and outliers were corrected, which suggests nontrivial curation effort.

3. **Clear benchmark structure.**  
   The paper defines three evaluation views—overall, joint-specific, and action-specific—which is a reasonable way to probe model behavior beyond a single aggregate score. This is helpful for future work and better than reporting only one metric.

4. **A strong baseline is provided.**  
   The proposed VID-Network is a reasonably motivated architecture that combines a pose estimator, marker regression, and temporal Transformer-based torque inference. The ablation study indicates that the auxiliary modules improve performance, which supports the design choices at least within the authors’ setup.

5. **Empirical gains on the proposed benchmark.**  
   The results show substantially lower error than the two baselines (Dino and ImDy) on the overall metric and on many joint/action categories. If the evaluation is sound, this suggests the baseline is competitive and the dataset is nontrivial.

### Weaknesses
1. **Novelty is limited relative to ICLR expectations unless the dataset is truly unique and well-validated.**  
   The main contribution is a dataset plus a baseline. ICLR can accept dataset papers, but they usually need especially strong evidence of scientific value, careful analysis, and clear reproducibility. Here, the core methodological novelty is modest: the model is largely a standard pipeline built from a CNN pose head plus a Transformer temporal regressor.

2. **Weak evidence that the task is truly “from real images” in the end-to-end sense claimed.**  
   The method still appears to rely heavily on intermediate pose/marker supervision and pretraining on large 3D pose datasets. This makes the setup closer to “image-to-pose-to-torque” than a truly direct image-based dynamics model. The paper’s claims about eliminating marker entities are also somewhat muddled because marker regressors are still used internally.

3. **Evaluation is incomplete for ICLR standards.**  
   The paper compares only against two methods and only on the newly constructed dataset. There is no analysis of robustness, no cross-subject generalization results, no zero-shot or out-of-distribution testing, and no comparison to simpler baselines that would help isolate what is actually driving performance. ICLR reviewers often expect stronger experimental breadth.

4. **Generalization concerns are substantial.**  
   The dataset has only 9 subjects and appears to come from a controlled capture environment derived from existing open-source data. The paper itself notes future work on cross-subject and in-the-wild settings, which underscores that these settings are not yet addressed. For a vision-based biomechanics paper, this is a major limitation.

5. **Reproducibility is only partially supported.**  
   The paper provides some training details, but many important specifics are missing or underspecified: exact train/validation/test split protocol, whether subject-disjoint splits were used, precise preprocessing steps for the data extraction pipeline, marker/joint definitions in operational detail, and hyperparameter settings for all modules. The dataset construction process seems labor-intensive and somewhat bespoke, which may hinder replication.

6. **Metric design and reporting could be stronger.**  
   The use of mPJE normalized by body weight is plausible, but the paper does not fully justify the metric or show alternative measures. It also does not report uncertainty, variance across runs, or statistical significance, which is important given the small subject count and the likelihood of high within-subject correlation.

7. **The benchmark may conflate task difficulty with dataset artifacts.**  
   Since the dataset is manually curated from existing sources and synchronized offline, it is unclear how much the results reflect genuine visual inverse dynamics learning versus exploiting dataset regularities, preprocessing choices, or subject-specific cues. This is especially important given the relatively small and curated nature of the dataset.

### Novelty & Significance
**Novelty: Moderate.** The dataset/benchmark contribution is the main novelty, while the model itself is incremental. The “first” claim, if accurate, is notable, but the technical advance is not especially deep by ICLR standards.

**Significance: Moderate to potentially high.** If the dataset is indeed high-quality, publicly released, and the task is well-posed, this could become a useful benchmark for a new research area. However, the current evidence is not yet strong enough to show that it will support robust scientific progress beyond a narrow curated setting.

**Clarity: Mixed.** The high-level motivation is understandable, and the paper’s structure is conventional. However, the explanation of the pipeline, data derivation, and evaluation protocol is sometimes ambiguous, and some claims feel stronger than the evidence supports.

**Reproducibility: Moderate to low.** The paper gives some implementation details and an appendix, but the dataset creation process, exact splits, and evaluation setup are not described with enough precision to make faithful reproduction straightforward.

**ICLR acceptance-bar assessment:** This is an interesting application-oriented dataset/benchmark paper, but on current evidence it likely falls below a typical ICLR acceptance bar because the methodological novelty is limited and the experimental validation does not yet establish strong generalization or broader scientific insight.

### Suggestions for Improvement
1. **Use subject-disjoint evaluation and report cross-subject generalization.**  
   This is crucial for a biomechanics dataset. A random frame split may overestimate performance because frames from the same subjects and trials can leak across train/test.

2. **Add stronger baselines.**  
   Include simpler and more interpretable baselines such as linear models, temporal CNN/LSTM baselines, pose-only vs image-only ablations, and a direct image-to-torque model without marker regression. This would clarify what each component contributes.

3. **Report uncertainty and statistical significance.**  
   Provide multiple runs, standard deviations, and significance tests, especially given the small number of subjects.

4. **Clarify the dataset construction pipeline in detail.**  
   Explain exactly how the real images, kinematics, and dynamics were aligned, how missing/outlier frames were handled, what filtering was applied, and how the torque annotations were obtained from OpenSim. A step-by-step reproducible pipeline would greatly strengthen the paper.

5. **Strengthen the claim of “from real images.”**  
   If the model depends on intermediate pose/marker supervision, discuss this honestly and position the work as image-conditioned inverse dynamics rather than a fully direct estimator. Alternatively, add a true end-to-end image baseline to justify the claim.

6. **Expand evaluation beyond the curated benchmark.**  
   Test robustness to occlusion, viewpoint changes, partial observations, and unseen subjects or actions. Even limited stress tests would make the benchmark more convincing.

7. **Release full code, splits, and annotation scripts.**  
   For a dataset paper, this is especially important. Releasing the exact train/test split, preprocessing scripts, and evaluation code would improve reproducibility and community adoption.

8. **Discuss ethical/privacy and data governance considerations.**  
   Since the dataset contains personally identifiable human motion data and anthropometrics, clarify consent, usage rights, and any privacy-preserving steps, which are increasingly important for ICLR dataset papers.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add subject-disjoint evaluation because the current 8:2 random split can leak subject-specific appearance and body-shape cues; without cross-subject testing, the claim that VID enables general torque estimation from real images is not convincing for ICLR standards.
2. Add stronger image-based baselines, not just Dino and ImDy. The paper claims a vision-only benchmark, so it should compare against a direct image-to-torque model, an image-to-pose-to-torque pipeline, and recent video/motion transformer baselines adapted to this task.
3. Add ablations that isolate each visual input and supervision signal: image-only, image+predicted pose, image+markers, image+temporal window, and with/without subject metadata (height/mass). Right now it is unclear whether the gains come from vision, kinematic priors, or easy subject-specific shortcuts.
4. Add a sanity-check baseline using ground-truth pose/markers as input to estimate the ceiling performance. Without this, it is impossible to tell whether the bottleneck is visual estimation, temporal modeling, or the torque regressor itself.
5. Add robustness tests under occlusion, truncation, and motion blur. If the method is meant for real/unconstrained images, the core claim needs evidence that performance does not collapse under realistic visual corruption.

### Deeper Analysis Needed (top 3-5 only)
1. Analyze per-subject generalization and variance, not just aggregate averages. The dataset has only 9 subjects, so a few easy subjects could dominate the reported gains; ICLR reviewers will expect evidence of stability across people, body types, and genders.
2. Quantify statistical significance and uncertainty over multiple random splits or cross-validation. A single 8:2 split is not enough to trust a 39.81% improvement, especially on a small dataset with correlated frames.
3. Analyze failure modes by action dynamics. The paper should explain why walkingTS1/2 are worse for the proposed method and whether this reflects temporal aliasing, camera viewpoint, or insufficient temporal context.
4. Analyze sensitivity to the manual synchronization, smoothing, and interpolation pipeline. Because the dataset is heavily processed, the paper must show that the results are not artifacts of the preprocessing choices.
5. Report correlation and calibration, not only mPJE. Torque estimation is useful only if predicted trajectories preserve dynamics structure, so the paper should show whether the model captures sign, phase, and peak timing of torques.

### Visualizations & Case Studies
1. Show predicted vs. ground-truth torque time series for several representative trials, including successes and failures. This would reveal whether the model tracks timing and magnitude or merely reduces average error.
2. Add qualitative comparisons across actions with varying difficulty, especially down-jump, walking transitions, and squats. The current single figure is insufficient to demonstrate where the method actually works.
3. Visualize attention or temporal saliency over frames and body regions. Without this, the claim that the network uses meaningful visual cues rather than dataset shortcuts is not credible.
4. Show error breakdown by viewpoint, subject size, and occlusion level. This would expose whether the method is robust or whether performance is tied to easy camera conditions.
5. Provide examples where the pose/marker regressor fails and how that propagates to torque prediction. This is essential to understand whether the proposed architecture is genuinely useful or just stacking error-prone modules.

### Obvious Next Steps
1. Replace the current random split with a standard cross-subject benchmark and publish the split protocol. This is the most immediate requirement for making the benchmark scientifically useful.
2. Benchmark against end-to-end video models and pose-free torque regressors. If the task is truly vision-based, the paper should establish that its architecture is competitive against modern visual sequence models.
3. Extend the benchmark to multi-view or in-the-wild footage. The paper’s stated motivation is unconstrained environments, but the current dataset appears laboratory-bound.
4. Release a documented preprocessing and synchronization pipeline with exact labels, coordinate conventions, and train/test splits. For a dataset paper, reproducibility is part of the contribution.
5. Add uncertainty estimation for torque outputs. In biomechanics, calibrated confidence is important, and ICLR reviewers will expect a stronger story than point estimates alone.

# Final Consolidated Review
## Summary
This paper introduces VID, a new dataset and benchmark for estimating human joint torques from real monocular human images, together with a baseline architecture that combines pose estimation, marker regression, and a temporal Transformer. The paper’s main value is the dataset curation effort: synchronized visual, kinematic, and dynamic annotations for 9 subjects and 63k frames are potentially useful for biomechanics and vision research. However, the methodological advance is modest, and the experimental validation does not yet convincingly establish a robust vision-based inverse dynamics benchmark.

## Strengths
- **VID is a genuinely useful dataset contribution if released as described.** The paper provides synchronized real images with kinematic and dynamic annotations, manual validation, smoothing, and outlier correction over 63,369 frames. For a task that has been poorly served by existing resources, this is the strongest part of the work.
- **The paper makes a concrete attempt to define a benchmark beyond one aggregate metric.** The three-way evaluation split into overall, joint-specific, and action-specific analyses is sensible, and the additional breakdowns do reveal some behavior differences across joints and actions.

## Weaknesses
- **The core claim is stronger than the evidence.** The paper repeatedly frames the task as “purely visual” or “from real images,” but the method relies on strong intermediate supervision, pose pretraining on external 3D pose data, marker regression, and a temporal window of frames. This is closer to image-conditioned inverse dynamics than a truly direct image-to-torque solution, and the paper does not cleanly separate what comes from vision versus what comes from kinematic priors.
- **The evaluation is too narrow to support the benchmark claim.** The main comparison is against only two baselines, neither of which is a clean modern image-based torque estimator. More importantly, the split is a single 8:2 random split; with only 9 subjects, this risks subject leakage and makes the reported improvement much less convincing than the headline number suggests.
- **Generalization is largely untested.** There is no subject-disjoint evaluation, no robustness testing, and no stress test for occlusion, viewpoint, or other real-world conditions. Since the paper motivates unconstrained deployment, this is a major gap.
- **The method is under-ablated.** The paper does not convincingly isolate the contribution of pose pretraining, temporal context, marker regression, or subject metadata such as height and mass. Without these ablations, it is hard to know whether the architecture matters or whether the gains come from privileged supervision and dataset-specific shortcuts.
- **Reproducibility is only partial.** Key parts of the dataset construction and evaluation pipeline are still underspecified, including exact split protocol, preprocessing details, and how the various labels are aligned and used at train/test time.

## Nice-to-Haves
- Add uncertainty estimates or multiple runs with variance reporting to show that the gains are stable and not split-dependent.
- Provide predicted-versus-ground-truth torque trajectories for representative samples, especially the harder walking-transition cases where the method underperforms the best baseline.
- Clarify the exact role of height and mass in the model and whether they are used only as supervision/context or as direct inputs.

## Novel Insights
The most interesting scientific point in this paper is not the architecture, but the benchmark framing: by pairing real images with synchronized dynamics labels, the authors are implicitly turning inverse dynamics into a visually grounded sequence learning problem rather than a pure biomechanics reconstruction problem. That is a meaningful direction, but the current setup still depends heavily on hidden structure from pose estimation and curated laboratory capture, so the paper’s apparent “vision-only” story is stronger rhetorically than empirically. In other words, VID may be a useful resource, but the paper has not yet shown that it supports robust learning of torque from appearance in the wild.

## Potentially Missed Related Work
- **OpenCap** — relevant because it also derives human movement dynamics from video-based capture and provides a strong point of comparison for vision-biomechanics pipelines.
- **AddBiomechanics** — relevant because it is a more recent large-scale biomechanics dataset with synchronized kinematics/dynamics and would help position VID’s dataset contribution.
- **Learning-based inverse dynamics of human motion / weakly-supervised learning of human dynamics** — relevant because these are closer dynamics-estimation baselines than the methods emphasized in the current evaluation.

## Suggestions
- Replace the random 8:2 split with a subject-disjoint benchmark and report cross-subject results.
- Add a truly image-based baseline and a pose-to-torque baseline to separate vision difficulty from torque regression difficulty.
- Include ablations for pose pretraining, marker regression, temporal window size, and subject metadata.
- Release exact splits, preprocessing code, and label alignment scripts to make VID usable as a community benchmark.

# Actual Human Scores
Individual reviewer scores: [4.0, 2.0, 4.0, 2.0]
Average score: 3.0
Binary outcome: Reject
