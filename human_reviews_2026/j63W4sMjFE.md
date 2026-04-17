# High Performance Space Debris Tracking in Complex Skylight Backgrounds with a Large-Scale Dataset

- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
With the rapid development of space exploration, space debris has attracted more attention due to its potential extreme threat, leading to the need for real-time and accurate debris tracking. However, existing methods are mainly based on traditional signal processing, which cannot effectively process the complex background and dense space debris. In this paper, we propose a deep learning-based Space Debris Tracking Network (SDT-Net) to achieve highly accurate debris tracking. SDT-Net effectively represents the feature of debris, enhancing the efficiency and stability of end-to-end model learning. To train and evaluate this model effectively, we also produce a large-scale dataset Space Debris Tracking Dataset (SDTD) by a novel observation-based data simulation scheme. SDTD contains 18,040 video sequences with a total of 62,562 frames and covers 250,000 synthetic space debris. Extensive experiments validate the effectiveness of our model and the challenging of our dataset. Furthermore, we test our model on real data from the Antarctic Station, achieving a MOTA score of 73.2%, which demonstrates its strong transferability to real-world scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces SDT-Net, a deep-learning-based tracker for space debris, and SDTD, a large-scale synthetic dataset for training/evaluation. The authors simulate 18k videos (62k frames) of linear-shaped debris on real ZTF backgrounds, then benchmark SDT-Net against MOT methods. An ablation study confirms the utility of the proposed RoI-FE module, line-source detection head, and debris-offset association mechanism.

### Strengths
1. Automated debris tracking is critical for space-safety, and data-hungry DL approaches have been stymied by the absence of large annotated corpora.
2. SDTD is two orders of magnitude larger than prior debris data and includes both synthetic and real labels; the simulation pipeline (PSF convolution, realistic motion, ZScale preprocessing) is well motivated and reproducible.

### Weaknesses
1. Recent transformer trackers (e.g., MOTR, TrackFormer) and joint detection-embedding models (e.g., FairMOT) are ignored. The authors should at least include one modern transformer baseline or justify its exclusion.
2. The performance improvement compared with OCSORT is minor. More importantly, the proposed method is trained on the debris tracking dataset, while the OCSORT is trained on the general video. It is hard to identify whether the performance improvement comes from the method or the training data.

### Questions
I am not an expert on debris tracking. I encourage the authors to clarify the Weaknesses. The proposed benchmark seems valuable.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This manuscript proposes SDT-Net, a deep learning-based network for space debris tracking, designed to achieve high-precision, real-time debris monitoring. The authors concurrently constructed a large-scale simulated dataset, SDTD, for model training and evaluation. Experimental results demonstrate that SDT-Net achieves exceptional performance in both synthetic and real-world scenarios (utilizing data from the Antarctic station), attaining 73.2% MOTA on real-world data, which validates its strong generalization capability and practical utility.

### Strengths
Motivation and Meaning: motivation is feasible
This manuscript focuses on detecting and predicting the motion trajectories of space debris to mitigate collision risks and advance the development of the aerospace industry.

Dataset：
This manuscript constructs a relatively large-scale dataset for space debris tracking, addressing a critical gap in existing research resources.

Writing Quality：
The manuscript is clearly articulated and highly readable. The methodological approach, experimental validation, and visualizations are comprehensively presented and well-supported.

### Weaknesses
Innovation: the contribution not enough for a ICLR paper
The design of RoI-FE demonstrates the distinction between SDT tasks and classical detection and tracking methods, suggesting that solutions tailored to the unique challenges of SDT may be necessary.


Methodology Section:
The explanation of the final loss function is insufficient, and there is a lack of corresponding ablative experimental analysis.

Experimental Evaluation:
The comparative experiments and ablative studies are inadequate.

### Questions
1. The comparative experiments lack comparisons with state-of-the-art methods published after 2024, and also lack comparisons with backbone methods.
2. What are the fundamental distinctions between the SDT task and classical detection and tracking methods, and what are its unique challenges?
3. RoI-FE is a straightforward feature fusion module composed of multiple stacked convolutional layers. What is the computational cost associated with this operation?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces SDT-Net, a CenterTrack-style tracker tailored to long-exposure, line-like space debris. The network adds a Region-of-Interest Feature Enhancement (RoI-FE) segmentation mask, endpoint heatmaps with a pairing embedding, and a per-endpoint offset head for frame association. The authors also build SDTD, a large synthetic-plus-real benchmark derived from ZTF backgrounds via an observation-based simulation with PSF blur and random truncation. SDTD comprises 18,040 videos / 62,562 frames / ~250k debris and is used to train/evaluate SDT-Net

### Strengths
1. Large, reproducible benchmark built from real survey backgrounds (ZTF) with PSF and truncation adds realism; the dataset scale and explicit dense-scene split are valuable to the community.

2. On SDTD, SDT-Net improves over CenterTrack/OCSORT/ByteTrack; on real Antarctic data it leads across MOTA/HOTA/DetA. Ablations isolate the gains from line-segment detection, RoI-FE, and the offset head.

3. Clear task formulation with architecture tweaks that match physics. Modeling debris as paired endpoints plus an offset field is well-motivated for line-sources; RoI-FE reduces skylight clutter before detection/association. The components (heatmap loss, CornerNet-style embedding push/pull, offset regression) are standard but effective.

### Weaknesses
1. Although the SDTD dataset is large and covers complex backgrounds, its generation process is mainly based on superimposing line sources from the background of ZTF astronomical images. The paper uses simple long-exposure line drawing with Gaussian blurring, without introducing high-fidelity physical constraints such as realistic trajectory dynamics modeling, PSF spatial variation, photosensitivity saturation, or noise field modeling. Therefore, from a technical perspective, the dataset's contribution leans more towards the scale of engineering collection and synthesis than proposing new physical fidelity or statistical generation mechanisms in simulation methods.

2. The overall architecture of SDT-Net largely follows the existing multi-object tracking paradigm: it centers on detection-regression-association, using heatmap regression endpoints, embedding matching, and offset prediction to achieve temporal correlation. The proposed "Region-of-Interest Feature Enhancement (RoIFE)" module and "offset module" are essentially lightweight integrations of existing feature enhancement and motion offset ideas, without introducing new mechanisms in the algorithm's principles or optimization objectives.

3. Train/val clips are mostly 1–4 frames, whereas test sets include sequences up to 30 frames (dense set concentrated around ~18). It’s unclear whether training on very short clips biases the tracker or underutilizes temporal cues.

### Questions
Please refer to the weaknesses

### Soundness
3

### Presentation
3

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
The paper presents SDT-Net, a deep learning framework for tracking space debris in complex skylight backgrounds, and introduces the Space Debris Tracking Dataset (SDTD), a large-scale synthetic dataset containing 18,040 video sequences with 62,562 frames and 250,000 synthetic debris instances. SDT-Net integrates feature enhancement, detection, and tracking modules to achieve high accuracy in cluttered, occluded, and dense debris environments. Evaluations on both synthetic and real Antarctic telescope data demonstrate strong performance, achieving a 73.2% MOTA score. The study highlights the potential of deep learning for real-time, transferable debris tracking and establishes SDTD as a benchmark for future research.Is this conversation helpful so far.

### Strengths
The paper proposes a deep-learning approach for space debris tracking in complex skylight backgrounds. The main contributions are:

1.The authors introduce a novel dataset, the Space Debris Tracking Dataset (SDTD), created by an observation-based simulation scheme, drawing on astronomy images (from e.g. the Zwicky Transient Facility, ZTF) and synthetically imposing debris trajectories and backgrounds. The dataset reportedly includes 18,040 video sequences (≈ 62,562 frames) and ~250,000 synthetic debris instances. 
Moonlight

2.They propose a network named SDT‑Net, which comprises a Region-of-Interest Feature Enhancement (RoIFE) module, a detection module and a tracking module (tracking by detection plus association across frames). The network is targeted at the multi-object tracking (MOT) task in astronomical / debris scenarios. 

3.They conduct experiments on their synthetic dataset and also evaluate transfer to real-world data: they claim a MOTA score (Multiple Object Tracking Accuracy) of ~73.2% (or ~70.6% in some versions) on a small real dataset collected at an Antarctic station. 

4.They argue that their dataset addresses the paucity of annotated debris-tracking data, and that SDT-Net exhibits robustness under dense debris, occlusion, and complex star-field backgrounds.

Thus the paper is an attempt to bring modern deep-MOT methods into the space-debris tracking domain, supported by a large synthetic benchmark.

### Weaknesses
While the paper makes interesting advances, there are several concerns and weaknesses that the authors should address:

1. Synthetic-to-real transfer gap / dataset realism

1) Although the dataset is large and simulation-based, synthetic data may not fully replicate the statistical characteristics of real debris tracks, noise sources, background clutter, telescope artefacts, or imaging conditions (e.g., atmospheric scintillation, streak brightness variation, non-uniform PSF, variations in exposure times, sensor noise). The authors do test on a small real dataset, but the size is tiny (36 video sequences, ~2,228 frames) and limited to one station (Antarctic). This raises questions about generalisability to other sensors, orbital regimes, debris sizes, lighting conditions, star-field densities.

2) The paper reports a single performance number on real data; more extensive evaluation across different observational setups would strengthen the claim of “strong transferability”.

2. Dataset annotation / ground-truth fidelity and bias

1) The synthetic generation process may introduce biases (e.g., debris speed, size, appearance, background variation) that favour their method, especially since the method is trained on the synthetic data. It’s unclear how well annotation errors, occlusion patterns, sensor artefacts, and false positives/negatives are handled.

2) The real data annotations (astronomy experts) are limited in quantity; the annotation criteria, inter-annotator consistency, debris definitions (what qualifies as debris vs star/artefact) may affect reproducibility.

3. Evaluation metrics and baseline comparisons

1) The paper uses MOTA as a key metric; however, MOTA alone may not capture fine issues like ID-switches, fragmentation, false alarms in cluttered star fields, long-term track survival, or tracking latency (important for real-time/operational use).

2) The baselines compared are relatively generic MOT methods (e.g., CenterTrack, OCSORT) rather than domain-specific methods tailored to astronomical debris or long-exposure streak detection. A stronger argument would include recent astronomy/space-debris tracking methods.

3) The paper claims “state-of-the-art”, but many details about run-time, sensor input frame rate, false positive/false negative rates, resource usage (GPU/CPU) are missing. For an operational system, these are important.

4. Scalability and real-time viability

1) Space debris tracking in real operational settings often demands real-time or near-real-time performance, dealing with many debris objects, variable frame rates, large fields of view, and possibly resource-constrained platforms. The paper does not sufficiently discuss latency, computational load, or memory constraints.

2) Dense debris scenarios (e.g., mega-constellations, low Earth orbit clutter) may stress the method beyond the distribution of synthetic data; how well does it scale beyond the densities in the dataset?

5. Lack of orbital/physical modelling integration

1) The method appears largely vision-based (image/video processing) without explicit incorporation of orbital dynamics, sensor geometry, debris kinematics, or space situational awareness (SSA) context (e.g., Two-Line Elements, orbital propagation). In many practical applications, combining image tracking with orbital dynamics yields more robust performance. The paper doesn’t show how their output could link to orbit prediction or catalogue maintenance.

2) Without physics-based constraints (motion models, known debris motion patterns), the tracker may fail in ambiguous scenarios (e.g., overlapping tracks, rapid acceleration, non-linear motion), and the paper does not explore these limitations in depth.

6. Generalisation to other observational platforms

1) The dataset is constructed from ZTF images (ground-based optical telescope) and the real evaluation is from a single station. It is unclear how well the method would generalise to different sensors: e.g., space-based optical imagers, radar, different exposure times, spectral bands, or telescopes with different PSFs, different background noise levels, different orbital altitudes.

2) The authors should discuss how the method would adapt to e.g., GEO, MEO, or LEO regimes, or to different sensors (infrared, radar) or daytime/nighttime imaging.

7. Benchmark release and reproducibility

1) The paper mentions that “dataset and code will be released soon”. Without immediate availability, reproducibility and community uptake may be limited. The authors should commit to making the dataset, annotations, evaluation scripts and code available under a clear license, and provide a leaderboard or standard evaluation split.

2) If synthetic only, there is a risk that future users will duplicate their simulation bias. Clear documentation of simulation parameters, debris motion models, background modelling is needed.

8. Limited real-world deployment discussion

1) The paper could benefit from a deeper discussion of how this tracking method would integrate into operational debris tracking pipelines, what the false alarm risk is, how track continuity and object correlation across multiple passes/sensors would be handled, and what the end-to-end system implications are (e.g., collision avoidance, catalogue updating).

2) It is also unclear how many frames per second, what field of view, and what detection sensitivity (size/magnitude of debris) the system supports; practical relevance to e.g., <10 cm debris tracking is not characterised.

### Questions
Here are several relevant prior works that the authors should cite, covering space-debris detection/tracking, multi-object tracking in astronomy, datasets, and physics-informed approaches:

**Space-debris detection/tracking and optical observations**

Cament, L. et al., “Space Debris Tracking with the Poisson Labeled Multi-Bernoulli Multi-target Tracking Filter”, Sensors, 21(11):3684, 2021.

“A Robust Vision-based Algorithm for Detecting and Classifying Small Orbital Debris” (NASA MSFC) – algorithm for small debris using optical detection. 
NASA Technical Reports Server

Navya, M. et al., “Deep Learning-Based Space Debris Tracking and Mitigation”, J Electrical Systems, 20(1):606-611, 2024. 

Zhou, D., Sun, G., Zhang, Z., Wu, L., “On Deep Recurrent Reinforcement Learning for Active Visual Tracking of Space Non-cooperative Objects”, arXiv:2212.14304, Dec 2022. 

Roll, D. S., Kurt, Z., Woo, W. L., “CosmosDSR – a methodology for automated detection and tracking of orbital debris using the Unscented Kalman Filter”, arXiv:2310.17158, Oct 2023. 

**Astronomical multi-object tracking / star-field object tracking**

Guan, J., Cheng, H-Y., Wu, Y-P., Tian, C., Qi, J-Y., “Multi-target tracking for star sensor based on CenterTrack deep learning model”, Scientific Reports 15:37125 (2025). 

**Space-debris modelling / simulation and environment context**

Kim, et al., “Review of Space Debris Modeling Methods and Development Trends”, Journal of Astronautical Sciences, 41(4):209-… (2024) 

ESA Space Debris Environment Report, https://sdup.esoc.esa.int/discosweb/statistics/, sdup.esoc.esa.int

**Deep learning object/tracking methods in cluttered/low SNR astronomical settings**

SDebrisNet: “SDebrisNet: A Spatial–Temporal Saliency Network for Space Debris”, Applied Sciences 13(8):4955 (2023). 

**Benchmarks/datasets for debris/satellite detection**

The authors should mention existing optical/space-object detection datasets, even if only for detection (not tracking) to position their contribution. For example, the Kaggle “Debris Detection Dataset” (optical images) – though limited. 

**Orbit/dynamics embedding into tracking**

Although not directly DL-tracking, works that link vision tracking with orbit dynamics may strengthen the discussion. For example the PINN-based tracking after collision: “Tracking an Untracked Space Debris After an Inelastic Collision Using Physics Informed Neural Network”, arXiv:2307.09938 (2023).

### Soundness
2

### Presentation
2

### Contribution
2
