# MSTformer: Multiscale Spatiotemporal Motion-aware Transformer Network for Effective AI-Generated Video Detection

- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Recent AI-generated videos (e.g., Veo3) are growing increasingly realistic and indistinguishable from real videos. Current existing detectors usually rely on artifacts present in earlier or inferior generations, resulting in poor generalization to the newly published generators. To address the challenge of newly generated videos, we propose a novel dataset, AIDetection, for the AI-generated video detection task. The proposed AIDetection dataset contains 39,298 real and 19,731 generated videos from 27 diverse sources, specifically designed to evaluate cross-generator generalization under out-of-distribution settings. For the real videos, the motion of moving objects and the background show clear distinctions. Based on this observation, in this paper, we introduce a novel Multiscale Spatiotemporal motion modeling Transformer framework (MSTformer) for the AI-generated video detection task, which learns motion-aware discriminative representations from both local and global viewpoints. Specifically, a novel multiscale spatiotemporal downsampling mechanism is designed to capture local motion discrepancies between real and generated videos. Further, to prevent the discriminative cues from being weakened, we also employ a contrastive learning mechanism implemented on multiscale spatiotemporal features, enabling the model to maintain the global discriminative ability. Extensive experiments on three benchmark datasets (i.e. AIDetection, GVF, and GenVideo) demonstrate that MSTformer achieves the superior cross-domain generalization performance. In addition, ablation studies further confirm the effectiveness of multiscale temporal modeling and contrastive learning in enhancing robustness for AI-generated video detection.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a novel multi-scale spatiotemporal motion-aware Transformer architecture (MSTFormer) for detecting high-quality AI-generated videos. The authors have constructed a new dataset, AIDetection, which includes nearly 39k videos from 27 sources, specifically designed to evaluate generalization across different generators. MSTFormer effectively captures the motion differences between real and generated videos through its motion-aware spatiotemporal downsampling module and cross-scale semantic contrastive learning module. It demonstrates outstanding performance on multiple benchmark datasets, particularly showing strong generalization capabilities in out-of-distribution (OOD) settings.

### Strengths
The method performs excellently across multiple OOD test scenarios, demonstrating its robustness on unknown generators.
Comprehensive experiments: Ablation studies and analyses of the effects of different sampling lengths and batch sizes have been conducted to verify the effectiveness of each module.
Good interpretability: Differences in motion patterns between real and generated videos are visualized using techniques such as optical flow maps.

### Weaknesses
1. MSTFormer has limited innovation, as operations for feature pooling already exist in RecoNet and MViT v2. This approach is very similar to these works.
2. The dataset proposed by the author is very small, with only 39K samples, which is significantly smaller than GenVideo and GenVidBench. This could lead to biased statistical results.
3. Table 2 presents the test results on three datasets. It can be observed that the results on the three test datasets are close, and AIDetection does not show a significant advantage.
4. The training set of AIDetection should be used for training, and the test sets of GenVideo and GenVidBench should be used for testing to determine whether the training set of this dataset offers any advantages.

### Questions
See the above

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the challenge of detecting high-quality AI-generated videos that are becoming visually indistinguishable from real ones. The authors introduce a new dataset AIDetection that covers diverse generators and real sources to evaluate OOD generalization.
Besides, the authors propose MSTformer which contains two main modules named MSTD and CSCL. Experiments show some improvements.

### Strengths
1. Writing is clear and the modules are well motivated.
2. AIDetection dataset is comprehensive for OOD evaluation in AI generated videos detection.

### Weaknesses
1. The manuscript only compares MSTformer with MViTv2 and UniFormerV2, omitting recent and more competitive AI-generated video detection methods. As a result, the evaluation appears limited, and the relative performance of MSTformer within the current state of the art remains unclear from my perspective.

2. The ablation study is somewhat incomplete. The authors mainly examine the presence or absence of the MSTD and CSCL modules, while omitting finer-grained analyses. For instance, it would be informative to investigate the impact of different 3D kernel sizes, strides, and downsampling strategies in MSTD (as mentioned around lines 217–218). Similarly, the influence of different cross-scale pair combinations in the CSCL module deserves further exploration.

3. The paper is motivated by the observation that real videos exhibit clear distinctions between object motion and background motion. However, it lacks corresponding interpretability results to support this claim. Providing interpretable analyses or visualizations would help demonstrate that the proposed method indeed captures these motion discrepancies.

4. The paper claims MSTformer is lightweight but computational cost (FLOPs, parameters, runtime) is not reported.

5. AIDetection dataset partly reuses videos from GenVideo and GVD, it's unclear whether these are excluded from training when testing on other datasets.

### Questions
I have one additional concern regarding the core hypothesis stated around line 52 — that “generated videos exhibit motion patterns inconsistent with the physical world.” While this assumption is plausible, it lacks direct empirical validation. For instance, a statistical analysis of optical flow distributions could provide supporting evidence for this claim.

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
4

### Summary
This paper tackles the challenge of detecting AI-generated videos, especially under cross-generator OOD generalization. The authors propose a two-part framework called MSTformer.They also introduce a new benchmark dataset AIDetection, containing videos from multiple commercial and closed-source generators as well as real short videos. This dataset aims to reflect realistic, diverse distributions.
Experiments on AIDetection, GVF, and GenVideo show consistent improvements across metrics (ACC, F1, AP). For instance, on AIDetection, MSTformer achieves ACC = 91.31, F1 = 91.12, and AP = 97.08; on GenVideo, ACC = 94.32, F1 = 93.50, and AP = 98.50.
The paper further reports per-generator ACCs, ablations on the two modules, and sensitivity to parameters τ and λ, number of frames, and batch size.

### Strengths
1.Clear and targeted motivation.
The authors correctly observe that modern video generators largely eliminate spatial artifacts, making motion dynamics the more reliable cue. The LK flow visualizations in Fig. 1 show that generated videos exhibit stronger foreground–background motion correlation.

2.Method design is consistent with the motivation and technically feasible.
MSTD performs spatiotemporal downsampling via 3D convolution before attention, enlarging the receptive field while preserving local temporal correlation.
CSCL enforces semantic alignment across multiple scales using supervised contrastive loss, preventing ambiguity in single-scale representations.

3.OOD-oriented evaluation setup.
The experiments explicitly test cross-generator generalization under “unknown real sources,” which is highly relevant for real-world robustness.

4.Rich experimental evidence.
Comprehensive comparisons across three datasets, per-generator breakdowns, and one-to-many OOD tests all support the claimed improvements.
Ablation studies confirm that MSTD substantially increases recall (46.56 → 66.81), while CSCL further improves F1 and ACC.

5.Implementation details are sufficiently disclosed.
Frame sampling, optimizer, training setup, and hardware are all clearly stated, aiding reproducibility.

### Weaknesses
1.Inconsistent or unclear dataset statistics.
Different sections report inconsistent counts of real/generated samples and generator sources (e.g., 39,298 vs 19,298 real). The authors should reconcile these and provide detailed splits in the appendix.

2.Misaligned evaluation protocol affects external comparability.
The paper explicitly states that, instead of evaluating each generator separately, all test samples are merged.
This change makes results not directly comparable with prior works and could blur whether the gains arise from method improvements or mixed distribution effects.
Suggestion: also report results following the original per-generator protocols (at least in the appendix).

3.Instability on advanced generators (e.g., Sora).
In Table 3, the per-generator ACC on Sora (68.60) is lower than MViTv2-S (71.74), indicating limited robustness to complex temporal motion. The authors should analyze failure cases by motion type, scene dynamics, compression, or frame rate.

4.Limited metrics and fixed-threshold evaluation.
All main results are based on ACC/Precision/Recall/F1/AP computed at a fixed threshold = 0.5.
No ROC-AUC, EER, or calibration analysis is provided.
Suggest adding AUC, EER, and PR-AUC results and discussing calibration or uncertainty under OOD settings.

5.CSCL lacks empirical justification for “cross-scale augmentation.”
Section 4.3 describes semantic consistency qualitatively but omits supporting quantitative evidence (e.g., alignment visualization, mutual information, or ablation comparing single-pair vs. multi-pair combinations).
Suggest including these analyses and reporting contrastive queue statistics in the appendix.


6.Reproducibility and openness.
The release of code and dataset is not guaranteed. The paper should clarify data licensing and provide code/configs or feature files if raw videos cannot be shared.

### Questions
.Please confirm the final statistics of AIDetection (real/generated counts, number of generator sources, and split details).
2.Can you reproduce results under the original GVF/GenVideo protocols to enable direct comparison?
3.Why does performance drop on Sora? Is it correlated with scene complexity, motion diversity, or frame rate?
4.For CSCL, what is the pair selection strategy and contrastive queue size? How sensitive is performance to these choices?
5.Could you provide ROC-AUC/EER metrics and discuss calibration or temperature scaling?
6.Have you tested robustness under varying compression levels, frame rates, and resolutions?

### Soundness
2

### Presentation
2

### Contribution
2
