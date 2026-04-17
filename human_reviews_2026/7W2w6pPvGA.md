# ULTRA-360: Unconstrained Dataset for Large-scale Temporal 3D Reconstruction across Altitudes and Omnidirectional Views

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Significant progress has been made in photo-realistic scene reconstruction over recent years. Various disparate efforts have enabled capabilities such as multi-appearance or large-scale reconstruction from images acquired by consumer-grade cameras. How far away are we from digitally replicating the real world in 4D? So far, there appears to be a lack of well-designed dataset that can evaluate the holistic progress on large-scale scene reconstruction. We introduce a collection of imagery on a campus, acquired at different seasons, times of day, from multiple elevations, views, and at scale. To estimate many camera poses over such a large area and across elevations, we apply a semi-automated calibration pipeline to eliminate visual ambiguities and avoid excessive matching, then visually verify all calibration results to ensure accuracy. Finally, we benchmark various algorithms for automatic calibration and dense reconstruction on our dataset, named ULTRA-360, and demonstrate numerous potential areas to improve upon, e.g., balancing sensitivity and specificity in feature matching, densification and floaters in dense reconstruction, multi-appearance overfitting, etc. We believe ULTRA-360 can serve as the benchmark that reflect realistic challenges in an end-to-end scene-reconstruction pipeline.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper construct ULTRA-360, a video benchmark of buildings captured by different devices. The data of a building is captured in various seasons width different appearance from different elevations. The paper also presents a method to calibrate such a large number of images. Several methods are used as baselines to test on the ULTRA-360 for reconstruction.

### Strengths
The paper contributes a building images dataset, where every building is captured at different seasons, under various weather conditions, from different viewpoints. The data are curial for generating building appearance variations over time. These data are not rare.

### Weaknesses
Although the main advantage of such data is 4D evolution, the paper did not take 4D experiments. Only 3D reconstructions were evaluated.

Time evolutions of many kinds of data are important, but the dataset is only comprised of buildings. More complicated data are welcome, such as trees, whose appearance and geometric shape both change over time.

### Questions
Will these data be made publicly available? Will ground truth be available for buildings when the dataset is published?

What benefit is there in using images of buildings under different lighting conditions for 3D reconstruction? I think the images are useful for 4D reconstruction. However, I cannot find any benefits of these images for 3D reconstruction, except that they are troublesome.

I couldn't see obvious visual differences between the images in Figure E in the appendix.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
To address the limitations of datasets used in training NeRF and 3DGS, e.g., limited in scale, camera perspective diversity, the authors introduce ULTRA-360, a novel and comprehensive dataset designed to benchmark the holistic progress in large-scale, immersive 4D scene reconstruction. In addition, they conducted extensive comparisons using state-of-the-art methods on ULTRA-360 across two main tasks: camera calibration and dense reconstruction, even in some challenging settings. The observations from those experiments point the way for future research directions.

### Strengths
1. The motivation is strongly justified and clearly articulated. It effectively identifies specific, well-known gaps in existing datasets (lack of scale, elevation diversity, temporal realism) and connects these gaps to practical limitations in evaluating SOTA algorithms.
2. The ULTRA-360 dataset and its associated calibration pipeline is well-executed. The dataset's scale, multi-elevation, multi-season, and multi-camera nature is impressive. The semi-automated calibration pipeline, with its strategies for doppelganger mitigation and robust coordinate alignment, is a substantial technical contribution that adds great value to the dataset's utility and credibility.
3. The experimental section is comprehensive, evaluating a wide range of SOTA methods on clearly defined challenges (cross-elevation feature matching, scene graph optimization, large-scale dense reconstruction, multi-appearance reconstruction). The results are convincing and clearly demonstrate the shortcomings of current algorithms. The analysis is insightful, correctly attributing RoMa's performance to its DINOv2 backbone and identifying the overfitting problem in multi-appearance embeddings.

### Weaknesses
1. In experiments, some SOTA methods are not included, e.g., VGGT, Pi3.
2. The data is sourced solely from a single university campus environment, which limits scene diversity and the validation of model generalizability. 
3. The heavy reliance on manual intervention (e.g., manually defining scene graphs) in the calibration pipeline contradicts the goal of fully automated and scalable data collection, potentially introducing subjectivity. 
4. Its embodiment of "4D" is primarily confined to slow seasonal and illumination changes, lacking fast dynamics or geometric structural evolution in the scene.

### Questions
No more questions, see Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors present ULTRA-360, a campus-scale dataset that contains 37.7 k images extracted from hundreds of videos recorded over two years. Data span four seasons, three elevations (ground, 60 m, 100–120 m) and two device classes (perspective and 360° cameras). A semi-automatic COLMAP-based calibration pipeline is proposed to register all views into a single coordinate frame, followed by an empirical survey of six recent NeRF / 3D-Gaussian-splatting variants on cross-elevation and multi-appearance novel-view synthesis. The work is positioned as a benchmark that reveals “realistic challenges” in end-to-end 4D scene reconstruction.

### Strengths
1.	Scale & diversity: To date, this is the largest real-capture repository that combines ground-level 360° imagery, drone footage, dense time sampling, and seasonal variation for the same physical site.
2.	Thorough calibration effort: The authors manually verified every building-level reconstruction and performed cross-elevation alignment.

### Weaknesses
1.	Novelty is primarily dataset-oriented, with limited methodological innovation. While the paper introduces a valuable large-scale dataset, it does not propose new algorithms, loss functions, or theoretical frameworks.
2.	The calibration pipeline is an engineering assembly of existing components (COLMAP, SuperPoint/SuperGlue, RoMa) plus manual clean-up; 
3.	Feature-matching ablations omit contemporary SOTA baselines such as MASt3R[1], DUSt3R[2].
4.	The observation that “cross-elevation training reduces Gaussians” is reported but not explained; no ablation of densification schedule, initialization, or gradient statistics is given.
[1] Grounding Image Matching in 3D with MASt3R
[2] DUSt3R: Geometric 3D Vision Made Easy

### Questions
1. What are the technical novelties in the paper?
2. Conduct comparative experiments with MASt3R and DUSt3R.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces ULTRA-360, a campus-scale dataset and benchmark targeting unconstrained, temporal, and multi-elevation 3D/4D reconstruction. The data spans ~140 acres over ~2 years, covering 20 academic buildings with both ground (perspective & 360°) and aerial (60/100/120 m) imagery acquired using consumer devices (iPhone, Insta360, DJI Mini), yielding >30k calibrated images (later summarized as 37.7k frames) with PII blurred. The authors build a semi-automated calibration pipeline: (i) within-elevation calibration via carefully constrained scene graphs to mitigate “doppelgänger” ambiguities, (ii) cross-elevation registration using transitional drone sequences from ground to air, and (iii) campus-wide coordinate alignment. They then benchmark feature matching (SIFT, SP+SG, LoFTR, RoMa; plus scene-graph optimizers such as NetVLAD and Doppelgänger++) and large-scale neural rendering methods (Block-MERF, Splatfacto-W, CityGaussianV2, Scaffold-GS, Octree-GS, EVER). Key findings: RoMa is the only method reliably finding cross-elevation correspondences but is prone to false positives; Octree-GS tends to perform best with mixed elevations; cross-elevation training often hurts densification; time-based appearance embeddings reduce overfitting relative to per-image embeddings.

### Strengths
Overall, this is a timely dataset paper with a clear gap and careful engineering. I appreciate it in the following aspests: 

1.Clearly motivated and realistic benchmark for end-to-end large-scale reconstruction across elevations and seasons; combines perspective and 360° ground coverage with aerial views. 

2.Thoughtful calibration strategy addressing doppelgänger matches via side-aware sequential pairing and cross-elevation transitional sequences; sensible divide-and-conquer pipeline.   

3. Provide strong, diverse baselines & analysis covering both matching/scene-graph choices and modern 3DGS/NeRF variants, with concrete takeaways (e.g., RoMa sensitivity vs. specificity; Octree-GS advantages with multi-elevation).  

4. A temporal appearance study with a simple but insightful time-based embedding alternative that reduces view-direction overfitting compared to per-image embeddings.  

5. Privacy consideration via automated PII blurring in the released frames.

### Weaknesses
Although the topic is timely, the core contribution is largely data collection plus a semi-automated calibration pipeline with several heuristic/manual steps, which limits both novelty and technical depth for a ICLR main-track paper. The work would read stronger in a benchmarks/datasets track or at a vision/graphics venue focused on 3D reconstruction. Reproducibility/logistics also feel under-specified (clear license, canonical train/val/test splits by building/elevation/season, and standardized compute budgets/hyperparameters for baselines), which makes it hard to ensure fair future comparisons.

While realistic, ~37.7k frames over ~20 buildings is modest relative to recent city-scale datasets such as Mapillary Metropolis, StreetGaussians, DL3DV-10K, or 4D4NeRF; the paper offers little direct comparison or clear positioning against them. It also omits strong contemporary cross-view/cross-modality matchers (e.g., MASt3R/DUSt3R, LightGlue, VGGSfM/++), leaving the central “cross-elevation” claim under-tested. The authors should articulate where ULTRA-360 is uniquely diagnostic (e.g., specific failure modes it exposes) and define standardized tasks/splits (no-transition cross-elevation calibration; ground→aerial / aerial→ground NVS; seasonal generalization) with apples-to-apples baselines to justify its distinct value.

Here are some minor weakness: 

1. Manual components remain central (manual bucketing of “front/back” sides; manual verification of cross-elevation sets), which could limit scalability/reproducibility and make results sensitive to human curation.  

2. Calibration quality reporting is mostly qualitative; clearer quantitative pose/rotation accuracy and failure analysis (per building/elevation) would strengthen claims about robustness. (The paper notes remaining ambiguities even with scene-graph optimization.) 
3. Cross-elevation dependency on transitional sequences may understate the difficulty when such data are unavailable; direct ground↔aerial matching is intentionally disabled as infeasible. 
4. Evaluation scope: image-quality metrics (PSNR/SSIM/DSIM) are reported, but free-navigation artifacts (e.g., floaters) are not quantitatively assessed; geometric plausibility metrics are suggested as future work.

### Questions
1. Beyond visualizations, can you report quantitative pose errors (e.g., ATE/RE) against a reference (e.g., campus-wide aerial bundle) and per-building success rates for different scene-graph/matcher settings? This would help substantiate the semi-automated pipeline’s reliability. 

2. How sensitive is performance to the front/back bucketing and the sequential window size |i−j|≤10? Could a learned side-classifier (or GPS/compass when available) reduce manual curation? Please provide failure cases and statistics.  

3. Since you disable ground↔aerial matching (P_grd^aerial = ∅) as unreliable, what happens if only ground+high-altitude imagery exist (no transitional videos)? A “no-transition” benchmark variant and results would be very informative. 

4. You already include SIFT, SP+SG, LoFTR, RoMa. Could you add/compare LightGlue/MASt3R-SfM and report robustness vs. false positives in ULTRA-360’s doppelgänger regimes? (You discuss scene-graph choices; numbers for these newer matchers would round this out.) 

5. Consider a floater index (e.g., depth variance or 3D occupancy regularity under aerial renders of ground-only models) and a geometric plausibility score for free-navigation views. Even simple proxies would give a quantitative handle on the artifacts you highlight. 

6. You note cross-elevation training reduces Gaussian counts and likely stresses densification. Can you isolate whether this is due to optimization schedules, regularization, or scene-graph/pose noise? An ablation on densification thresholds and background/sky modeling across methods would clarify attribution.

### Soundness
3

### Presentation
3

### Contribution
2
