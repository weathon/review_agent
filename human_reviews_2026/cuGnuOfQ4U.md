# Learning Patient-Specific Disease Dynamics With Latent Flow Matching For Longitudinal Imaging Generation

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 2

## Abstract
Understanding disease progression is a central clinical challenge with direct implications for early diagnosis and personalized treatment. While recent generative approaches have attempted to model progression, key mismatches remain: disease dynamics are inherently continuous and monotonic, yet latent representations are often scattered, lacking semantic structure, and diffusion-based models disrupt continuity through the random denoising process.
In this work, we propose treating disease dynamics as a velocity field and leveraging Flow Matching (FM) to align the temporal evolution of patient data. Unlike prior methods, our approach captures the intrinsic dynamics of disease, making progression more interpretable.
However, a key challenge remains: in latent space, Autoencoders (AEs) do not guarantee alignment across patients or correlation with clinical severity (e.g., age and disease conditions). To address this, we propose learning patient-specific latent alignment, which enforces patient trajectories to lie along a specific axis, with magnitudes increasing monotonically with disease severity. This leads to a consistent and semantically meaningful latent space.
Together, we present ∆-LFM, a framework for modeling patient-specific latent progression with flow matching. Across three longitudinal MRI benchmarks, ∆-LFM demonstrates strong empirical performance and, more importantly, establishes a new framework for interpreting and visualizing disease dynamics.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The research team introduces $\Delta$-LFM for longitudinal MRI modeling that learns patient-specific disease trajectories as continuous latent flows. They focus on the problem of patient-specific trajectory generation, which is currently a very valuable topic in the field, as most of the past models focus on population-level prediction. Specifically, to address this problem, the paper suggests a new loss: ArcRank Loss in the latent space to guide the patient-specific disease trajectories to have a stable direction and monotonic disease progression. The team also provides a temporal flow matching ($\Delta$-LFM architecture, which generalizes flow matching to clinically meaningful intervals $[0, T]$ for arbitrary future-time predictions.  Besides the novelty in the model itself, this paper also suggests a new metric, $\Delta$-RMAE, which aims to capture the residual differences between baseline and follow-up scans. 

The authors utilize three datasets: ADNI, OASIS-3, and AIBL to validate their model's performance. $\Delta$-LFM outperforms direct-prediction, deformation, and diffusion baselines (e.g., DiffuseMorph, TADM, BrLP, MambaControl) while producing interpretable patient-specific latent trajectories.

### Strengths
1. Clinical relevance and motivation. The focus on individualized disease dynamics—rather than population-level progression—is well motivated and important for neurodegenerative modeling.
2. Coherent methodology. Combining flow-matching dynamics with a longitudinal latent regularizer (ArcRank) yields interpretable and temporally smooth trajectories.
3. Elegant SVD-based alignment. Treating each patient’s latent tensor $Z_k$ as a matrix and applying SVD to extract orthogonal direction (U) and magnitude ($\sum{}$) is a neat, principled idea that mitigates the instability of cosine-norm losses under noisy or scale-varying latents.
4. Robust experimental design. Evaluation on three large AD cohorts with multiple baseline classes (direct/deformation/diffusion) is comprehensive; ablations are clear and interpretable.
5. New progression metric ($\Delta$-RMAE). Correctly highlights that PSNR/SSIM can mask small but clinically meaningful anatomical changes.
6. Interpretability. Latent trajectories align by patient and even cluster by diagnosis despite no supervision—visually persuasive evidence that ArcRank captures disease semantics.

### Weaknesses
1. $\Delta$-LFM’s formulation closely parallels latent-ODE and rectified-flow approaches. The paper should more clearly delineate the conceptual or practical advantages of its flow-matching objective over these established continuous-dynamics frameworks.
2. The model assumes straight-line latent trajectories and roughly constant velocity; though acknowledged, this remains biologically limiting. More details on how conditioning on T and patient attributes relaxes this assumption would help. 
3. $\Delta$-RMAE relies on image residuals, which are potentially confounded by registration or intensity artifacts. A sensitivity analysis to misregistration/bias correction would increase confidence.
4. The authors claim that SVD is better than cosine similarity and absolute values for magnitudes, but there is no direct comparison between the performance of these two designs. It will be better if the authors could provide a comparison between them.

### Questions
Same as weakness.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors point out a key limitation of current methods for modeling (intrinsically continuous) disease progression: the latent representations are often scattered, lacking semantic structure, and diffusion-based models disrupt continuity with random denoising process. To remedy this, the authors propose to treat the disease dynamic as a velocity field and leverage flow matching to align the temporal evolution of patient data. To ensure the latent space is organized by clinical severity, he authors propose to learn patient-specific latent alignment that enforces patient trajectories to lie along a specific axis along which clinical severity monotonically increases. The proposed method, termed $\Delta$-LFM, demonstrates stronger empirical performance and offers a way to interpret and visualize disease dynamics.

### Strengths
1. I find this topic computationally interesting, scientifically important, and clinically relevant.

2. The proposed method is well motivated and easy to follow. 

3. Very good visual presentation. 

    a. The figures are beautifully made, with carefully chosen colors (Figure 1), clear illustration of ideas (Figure 1), and proper colormaps (Figures 2 & 4). The only less nice-looking figure (Figure 3) can be easily improved by removing top and right borders and enlarging the labels and tick labels.

    b. Tables are also well designed and easy to follow. My only suggestions are to remove the vertical lines and adjust the column spacing or put a categorical name on top if you want to further distinguish the columns. For example, for Table 3, you can consider something like “image quality” above “SSIM” and “PSNR”, and “clinical structure faithfulness” above “Regional MAE” and “$\Delta$-RMAE”. Besides, you can use \cmidrule(lr){startColumn – endColumn} for the horizontal bars below each dataset name to make sure consecutive horizontal bars are not touching.

4. Quantitative results look promising. Approximately 1.0 absolute point increase in PSNR and SSIMx1E2 are shown across three large-scale datasets.

5. Ablation results are comprehensive and properly showcase the effectiveness of the proposed ArcRank loss.

### Weaknesses
1. A few comments on comparisons to existing work.

    a. The authors mentioned that the proposed method “treats the disease dynamic as a velocity field” when modeling disease progression. As a result, I believe it is expected to compare against ImageFlowNet [1], since the design philosophy is similar. The authors can stay assured that I recognize the novelty of this submission, since there are sufficient distinctions, namely (1) flow matching instead of neural differential equations and (2) the ArcRank alignment.

    b. I am also aware of a recent work [2] that uses flow matching for disease progression in longitudinal images, but comparison against it will be an unreasonable request, because it is too recent (Oct 2025) and the code is not available.

2. For methods that forecasts longitudinal images, one major concern I have is that they tend to overly reconstructing and minimally forecasting. A common shortcoming I observe in this field is the underestimation of changes in the images over time. In this paper, the authors have not sufficiently shown that the proposed method is not simply performing very good reconstruction of the input image. For example, in Table 4 in appendix, the authors showed the forecasting performance degrades as the time horizon increases, but that could also be explained by an alternative hypothesis that the prediction is just a reconstruction of the input. I would like to see how the authors can argue against my concern.

[1] ImageFlowNet: Forecasting Multiscale Image-Level Trajectories of Disease Progression with Irregularly-Sampled Longitudinal Medical Images, ICASSP 2025 Oral.

[2] Longitudinal Flow Matching for Trajectory Modeling, arXiv 2025.

### Questions
1. Please refer to the weaknesses.
2. Please also refer to the minor formatting suggestions in the strength section. These are just suggestions but not mandatory.
3. In Table 3, I believe SSIM and PSNR are swapped.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces a generative framework for modeling patient-specific disease progression from longitudinal MRI data. The authors treat the disease dynamic as as a velocity field and use ideas from flow matching to align the temporal evolution of patient data. To overcome the mismatch of representations between patient-specific trajectories and the latent space representations, they propose to learn patient-specific latent alignments. These alignments are penalized by a definition of a new loss, what is introduced here as a ArcRank loss. This loss measures the discrepancies between both representations by aligning latent trajectories across time within patients and by enforcing angular consistency and monotonic magnitude growth. This loss is optimized under a temporal flow-matching framework (that has been proposed before) to learn continuous velocity fields over arbitrary time intervals. The authors claim that the combination of this approach enables the learning of personalized disease progression as smooth latent flows. Experimental results are demonstrated on ADNI, AIBL, and OASIS-3 datasets with the image fidelity (PSNR/SSIM) and anatomical progression accuracy as metrics for evaluation.

### Strengths
While different elements (TADM, BrLP, SADM etc., which are cited here) used in this work have been previously proposed separately, one of the contributions of this paper is to combine these several ideas into a common framework. This makes this work somewhat novel.

The main novelty of the work is the introduction of the matching loss ArkRank loss. This loss enforces patient-specific trajectory alignment and temporal ordering. The trajectory alignment is achieved using angle matching of features in the latent space, while the monotonic growth of time features is achieved by temporal ordering (ranking).

Another contributions is the reformulation of flow matching by extending the sampling strategy from a fixed unit interval to an arbitrary interval. This improves the accuracy and minimizes residual errors, especially with arbitrary time-domain patient samples. 

Experimental evaluations over multiple datasets is a strength. 

The close clustering of patient identity as well as diagnosis status helps demonstrate the interpretability of the method. The authors also qualitatively demonstrate MRI progression over 9 years. 

Ablation studies are performed adequately.

### Weaknesses
There is one main novel idea in the paper, i.e. the introduction of the ArcRank loss. The rest of the ideas are incrementally novel and have been conceptually proposed before and also applied to MRI images. 

The ArcRank loss involves an SVD may be expensive to evaluate. 

An intrinsic weakness in the definition of the ArcRank loss is the reliance on two  weights \lambda_{arc} and \lambda_{rank}. The authors don't mention how these weights are imposed. Aligning the directions (first term) and maintaining the ordering (second term) is a non-trivial problem, even when the trajectories or the initial and final mappings are known. Thus the computation of the loss is itself challenging. Since the loss function is supposed to be on of the main novel contributions, it should be further analyzed. 

The ablation studies with the Ark loss, Rank loss, Ark + Rank loss don't seem to show dramatic improvement in results. Although the role of the arbitrary sampling scheme over [0, T] may be important. This is not fully discussed.

### Questions
What is pull and push in Figure 1 in the ArcRank loss panel?

Can the authors comment on the computational efficiency of the ArkRank loss?

Are the weights \lambda_{arc} and \lambda_{rank} fixed or learnt over time? How are they chosen?

Can the authors comment on the stability of the ArkRank loss? 

In the ablation study (Table 3), how will the w/ Arc loss over [0, T] and the Rank loss over [0, T] perform? This may yield more granular information when the full w/ Arc+Rank loss is evaluated over [0, T]. 

In Figure 4, why do you see more progression prediction (enlargement of ventricles) in the ADNI dataset over AIBL and OASIS? Are the cases more severe in ADNI compared to the other two datasets?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposed a latent flow matching method for disease progression prediction. The key idea is to enforce monotonic, per-patient latent progressions (same direction, growing magnitude) and then learn a temporally meaningful velocity field in latent space, so progression becomes smoother and clinically interpretable. They evaluate the method on ADNI, AIBL, OASIS-3.

### Strengths
- The paper has a clear formulation that couples patient-specific latent alignment (ArcRank) with temporal flow matching.

- The method shows consistent empirical results across three longitudinal AD MRI benchmarks.

- The paper has good ablation and visualization.

### Weaknesses
- Temporal sampling [0, T] still effectively assumes roughly uniform progression between scans; the paper acknowledges uneven progression but does not truly model accelerations/plateaus.

- Evaluation is confined to AD-style neurodegeneration datasets; claims about general utility (tumor, faster diseases, multi-organ) are speculative. It strongly weakens the conclusion.

- ArcRank depends on SVD-based decomposition per latent, which could be brittle or expensive and the paper does not compare to simpler angular/ranking surrogates on equal footing.

- The method hinges on a learned AE latent; there is no analysis of how sensitive results are to the AE capacity or to using a stronger/pretrained encoder.

- The new metric is reasonable but mostly motivated empirically; there is no user/clinician study to show that better ∆-RMAE corresponds to more actionable longitudinal reads. If a disease progression does not have user study from medical doctors, it is not convincing to clinical readers.

- Clinical conditioning is mentioned (age, sex, status) but not deeply analyzed—no ablation on which attribute matters most, or on robustness to missing/noisy metadata.

- Many important previous works are not mentioned and compared, such as [1,2].


[1] Kyung, Daeun, et al. "Towards Predicting Temporal Changes in a Patient's Chest X-ray Images based on Electronic Health Records." arXiv preprint arXiv:2409.07012 (2024).      
[2] Liang, Kaizhao, et al. "Pie: Simulating disease progression via progressive image editing." (2023).

### Questions
See weaknesses for details.

### Soundness
2

### Presentation
3

### Contribution
1
