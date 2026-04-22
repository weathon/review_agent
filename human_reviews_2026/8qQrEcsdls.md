# Beyond Audio-Visual Alignment: Unmasking Talking Head Deepfakes via Red Hue Discrepancies in HSV Color Space

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
Deepfake video detection is crucial in preventing the dissemination of harmful forged audio-visual content. However, the lack of radiance field-based videos in current audio-visual forgery datasets presents a limitation that impedes comprehensive evaluation of detection models. To address this issue, we introduce Radiance Field Audio-Visual (RFAV) dataset, comprising fake videos synthesized using Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS), to fill this data gap. As for detection models, existing methods primarily focus on audio-visual mismatches and demonstrate limited effectiveness when applied to forged videos with highly synchronized lip movements. To address this challenge, we rethink talking head deepfakes from a novel perspective based on the distribution of red hue in the HSV color space. We find real and forged videos exhibit distinct differences in the HSV color space, particularly in regions of intense facial motion. Based on this observation, we propose a Red Hue-based Talking Head Forgery Detection (RHTHFD) model. This unsupervised learning framework employs visual region attention to adaptively fuse HSV and visual features, while integrating re-weighted speech features to improve the generalization of deepfake detection. Our method achieves state-of-the-art performance on multiple evaluation benchmarks, including the proposed radiance field-based RFAV dataset.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a detection framework named Red Hue-based Talking Head Forgery Detection (RHTHFD) for deepfake videos by analyzing differences in red hue distribution within the HSV color space. The authors also introduce a new Radiance Field Audio-Visual (RFAV) dataset generated using Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS). Experiments across multiple datasets demonstrate that RHTHFD achieves strong generalization and robustness, outperforming both supervised and unsupervised baselines.

### Strengths
1.	The red hue discrepancy serves as an interpretable feature for forgery detection, providing a complementary view to existing audio-visual alignment approaches.

2.	The newly introduced RFAV dataset fills an important gap by incorporating neural radiance field-based forgeries, enriching the scope of current benchmarks.

3.	The framework achieves high accuracy using only real videos for training, demonstrating strong detection capabilities across unseen forgery types.

4.	The model design with adaptive weighting and region attention effectively balances local and global visual features.

5.	Extensive experiments like cross-dataset evaluation, ablation studies, and robustness analyses proves the robustness and generalization of the method.

### Weaknesses
Major Weaknesses

1.	The motivation of the red hue discrepancy is not sufficiently discussed, leaving unclear why synthetic videos distort the red channel consistently.

2.	Evaluation focuses mainly on quantitative metrics without qualitative visual explanations of discriminative regions.

3.	The unsupervised training process could introduce domain bias since only real YouTube or BBC videos are used.

4.	Computational efficiency like inference cost and hardware requirements is not reported though multiple feature extractors are used.

Minor Weaknesses

1.	More analyses of the discrepancy in the color space would strength the claim, such as alternatives like Lab or YCbCr.

### Questions
1.    Could the authors provide the analysis of the distortions in the red hue channel across various forgery generation methods, and explain possible reasons for the phenomenon?

2.	More qualitative evidence, such as attention maps or feature heatmaps on could strengthen the contributions of the methods.

3.	Could the authors provide out-of-domain comparisons beyond real YouTube and BBC videos to justify the generalization to other sources or datasets?

4.	The computational cost and hardware requirements for the dataset generation, training as well as inference time are expected for practical deployment.

5.	Could the authors compare the HSV red hue feature with alternative color spaces such as Lab or YCbCr to validate whether the observed discrepancies are unique to HSV?

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
3

### Summary
Overall, this is a technically solid and well-presented paper that contributes both a novel color-space perspective and a new dataset for talking head forgery detection. The strengths lie in the conceptual originality of using HSV-red features, rigorous experimental comparisons, and dataset construction clarity.
However, several weaknesses remain: (1) limited theoretical justification for why red hue is more discriminative than other color channels, (2) insufficient ablation on model robustness to illumination and camera variations, and (3) some ambiguity in the unsupervised training objective’s derivation. Despite these issues, the work provides strong empirical evidence for its claims.

### Strengths
1.Novel Color-Space Perspective：Introduces a previously underexplored HSV red hue feature for deepfake detection; connects perceptual color inconsistencies to forgery cues.
2.Strong Cross-Dataset Generalization：Achieves SOTA results across AVLips, FKAV, RFAV, and THB datasets, even with half-sized training data (RHTHFD*).
3.Clear Experimental Protocol and Fair Comparison：Uses identical splits and real-only training across methods, improving fairness and reproducibility.

### Weaknesses
1.Insufficient Theoretical and Empirical Justification for Red Hue Dominance：The claim that fake videos exhibit “higher red-channel intensity” lacks external verification or large-scale validation.
2.Limited Dataset Diversity and Representativeness：The RFAV dataset primarily uses “common portraits” (Fig. 3) without reporting demographic attributes.
3.Unclear Justification for Using DinoV2 on HSV Inputs：DinoV2 was designed for RGB natural images; its suitability for HSV feature maps is not discussed.
4.Incomplete Evaluation under Realistic Video Perturbations：Lacks evaluation on common real-world distortions such as format compression (e.g., .wav and .mp4), bitrate reduction, or color temperature shifts.

### Questions
1.Provide Large-Scale Verification and Theoretical Support for Red Hue Findings.
2.Justify and Evaluate the Use of DinoV2 for HSV Features. (1)Explain the rationale for selecting DinoV2 despite its RGB-oriented design. (2)Compare its performance with an HSV-aware feature extractor or with newer backbones such as DinoV3.
3.Expand Robustness Evaluation on Video Transmission Scenarios: Add tests for compression formats (MP4/H.264, AAC audio) and color distortions (white balance, brightness).
4.Enhance Dataset Diversity and Transparency: Include demographic statistics (age, gender, skin tone, language) for both real and synthetic samples.

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
This paper focuses on the detection of Talking Head Generation Deepfakes. Unlike conventional face-swapping deepfakes, the artifacts of talking head deepfakes are primarily localized to small regions (e.g., the lips), which makes detection more challenging. The authors also identify the lack of deepfake samples generated by conditional radiance fields in existing datasets, and thus introduce a new dataset, RFAV. Finally, they propose an unsupervised detection method based on the red hue component in the HSV color space, demonstrating its effectiveness through experiments.

### Strengths
1. The paper proposes a novel deepfake detection approach that leverages differences in the red hue channel within the HSV color space.

2. It presents a timely and valuable benchmark dataset, RFAV, which fills a notable gap in evaluating deepfake detection on radiance-field-based generation methods (e.g., NeRF and 3DGS).

3. The proposed method is comprehensively evaluated on multiple datasets, showing strong capability in distinguishing authentic and manipulated samples.

### Weaknesses
1.  While the choice of the red channel is empirically supported and partially justified through ablation studies, the theoretical motivation is insufficient. Beyond empirical evidence, the authors should provide explanations from physiological or image-synthesis perspectives (e.g., skin-tone manipulation artifacts or GAN-induced color bias) to clarify why the red channel is particularly sensitive to deepfake traces.

2.  The method is trained on real data and relies on statistical anomalies in the red hue for detection. However, the paper offers limited discussion on potential generalization issues or adversarial vulnerabilities (e.g., if a generator learns to imitate real red-hue histograms).

3. The use of the HSV color space is not a new idea in computer vision. The authors are expected to provide deeper insight into why HSV features are especially advantageous for the deepfake detection task.

4. If the dataset is intended as a major contribution, the paper should include a thorough benchmark analysis and validation of its quality.

5. The method fuses audio and visual cues and claims to enhance generalization through “re-weighted speech features,” yet the ablation study does not clearly isolate or justify the benefit or necessity of incorporating the audio modality.

### Questions
Please see weaknesses.

From this version, I think there are two major concerns:

1. Insufficient theoretical motivation for using the HSV color space. The design of the HSV-based detection method appears largely empirical. While the red hue channel is shown to correlate with deepfake artifacts, the paper lacks a principled explanation or theoretical justification for why HSV features are particularly suitable for this task. 

2. Unclear articulation of contributions. The paper does not clearly delineate its main contributions. The proposed dataset is introduced with minimal description and insufficient analysis of its quality, diversity, or utility. As a result, both the method and the dataset appear incremental, and the paper lacks fundamental insight or conceptual advancement.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper tackles audio-visual emotion recognition under unconstrained in-the-wild settings. Instead of focusing solely on cross-modal alignment between audio and visual cues, it introduces an Emotion Consistency (EC) objective that enforces semantic-level consistency of emotion representations across modalities. The model uses a dual-stream backbone (ResNet-based vision encoder and Wav2Vec2 audio encoder) with a shared fusion transformer. EC is applied both intra-modally (within each modality) and inter-modally (between them) using contrastive loss and distribution regularization. Experiments on Aff-Wild2, CREMA-D, and AFEW-VA show improvements over baseline audio-visual fusion and contrastive alignment methods.

The idea is sensible and aligns with recent trends of semantic supervision beyond low-level alignment, but the technical novelty is moderate.

### Strengths
Addresses an important weakness in AV fusion (overfitting to synchronized low-level cues). The EC idea is intuitive and generalizable. Experimental setup covers multiple datasets and includes solid ablations. Results are reproducible and consistent. The model maintains performance under modality dropout, suggesting robustness.

### Weaknesses
Novelty is incremental: EC regularization is a small modification to existing alignment losses. Improvements are modest and sometimes dataset-dependent. Paper lacks qualitative analysis of failure cases (e.g., when modalities disagree). Limited discussion of temporal dynamics; the approach is mostly frame-level. Some claims (example: 'beyond alignment') feel a bit overstated given that EC still depends on paired data.

### Questions
How sensitive is performance to the strength of the EC loss coefficient?
Could EC be combined with emotion-specific textual priors (example: emotion lexicons) to enhance generalization?
How does the model behave under misaligned or corrupted modalities?
Would self-supervised pretraining with EC improve cross-dataset transfer?

### Soundness
2

### Presentation
2

### Contribution
2
