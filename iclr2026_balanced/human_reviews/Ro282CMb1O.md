## Human Reviewer 1

### Summary
The paper introduces U-Bench, a large-scale benchmark for U-Net–style medical image segmentation. It re-implements (retrains) 100 publicly available U-Net variants across 28 datasets, training on 20 and testing zero-shot on 8 within-modality target domains. It proposed U-Score, a quantile-normalized metric that combines accuracy (IoU) with efficiency (parameters, FLOPs, FPS), and an advisor agent that recommends architectures conditioned on data characteristics (foreground scale, boundary sharpness, shape regularity) and resource constraints. Training is standardized with a single pipeline (SGD, LR=0.01, 300 epochs, batch 8, seed 41) and consistent splits (official where available; otherwise 7/3), with models largely taken from official implementations and minimally adapted I/O. Overall, the work is ambitious and potentially impactful for the community.

### Strengths
1. A comprehensive benchmark and a non-trivial effort: this paper implements 100 U-Net variants. Great effort went into developing the benchmark.
2. Training on a source dataset and testing on different datasets of the same modality and task aligns with clinical domain-shift realities and is clearly described.
3. U-Score percentile-normalizes IoU, parameters, FLOPs, and FPS using the 10th/90th quantiles, then combines them with equal-weight harmonic means, providing a clearer view of accuracy–efficiency trade-offs than IoU alone. The formulation is explicit and easy to reproduce.
4. Extensive statistical testing at scale and significant GPU resources went into building this benchmark.
5. The ranker leverages model and dataset descriptors (e.g., size, FLOPs, FPS; foreground scale, shape complexity, boundary sharpness) and reports NDCG/MAP/Spearman, making the benchmark actionable for practitioners.

### Weaknesses
1. Regarding segmentation accuracy, this paper only considers IoU while disregarding other metrics, such as boundary-aware metrics (e.g., Boundary-F1), which can be crucial. I think the authors are aware of this, but they should further discuss in the manuscript why they use IoU as the only metric.
2. Inference speed measurement details are not clear. U-Score incorporates FPS, but the measurement protocol (e.g., batch size = 1, warm-up, mixed precision, resize vs. native resolution, dataloader vs. pure inference) is not fully specified. The hardware (GPU models and count) and software (deep learning framework such as PyTorch) are given, but a precise FPS protocol would improve reproducibility.
3. No explicit reproduction-gap analysis is provided. Although the authors retrained 100 models under a unified recipe, there is no table comparing the reproduced in-domain baselines against the original papers’ reported metrics (e.g., IoU). This omission makes it hard to judge implementation fidelity and may confound historical trend claims. I suggest adding a “Δ vs. original” column or a “ΔU-Score” column for a representative subset.
4. Scaling behavior is not directly studied. Although U-Score bins parameters/FLOPs/FPS and charts decade-long trends, the paper lacks a systematic scaling-law analysis (within-family width/depth scaling) and a data-scaling study (e.g., 25/50/75% subsampling). Given today’s large-model landscape, both analyses would strengthen conclusions about compute optimality and data efficiency.
5. In the anonymous repo, the .py files under scripts/ and models/ don’t seem accessible (perhaps a hosting/LFS issue); could you please confirm whether the full training code—or at least a minimal reproducible package (training/eval scripts, one dataset dataloader+splits, and an env spec)—will be made available?
6. Minor writing/terminology issues: The manuscript inconsistently uses “U-Stone” (likely meant U-Bench) in the appendix. Please standardize names across text, figures, tables, and references.

### Questions
1. U-Score: Did you test robustness to different weights (Params/FLOPs/FPS and α) or user-specified accuracy–efficiency trade-offs (e.g., resource-constrained settings)?
2. FPS protocol: Framework (PyTorch/CUDA/cuDNN/TensorRT versions), GPU model×count, CPU model & threads (OMP/MKL), batch & input size, precision (fp32/fp16/bf16/AMP), warm-up/timed iters, dataloader workers, include I/O/post-proc?, and any torch.compile/TensorRT settings.
3. Scaling laws: Did the authors conduct within-family scaling analyses (depth/width) and dataset-size ablations to test family-specific benefits on small vs large datasets?
4. Compute accounting: Can you report per-model (and per-dataset) GPU-hours for training and FPS benchmarking—or at least median/IQR by family—and, if possible, time-to-target IoU/U-Score?

### Soundness
4

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper introduces U-Bench that evaluates 100 U-shaped medical-image segmentation models across 28 datasets, reporting IoU as the primary metric. The author also introduces U-Score (a composite of IoU, Params, FLOPs, FPS built from quantile normalization and harmonic means) and adds an XGBoost-based ‘advisor agent’ to rank models from discretized dataset/model descriptors. Training uses a single unified recipe for all models.

### Strengths
1. Much needed and important medical image segmentation benchmarks. Ambitious scale and breadth across architectures and datasets, including in-domain and zero-shot reporting.

2. Attempts to account for efficiency with U-Score instead of accuracy-only comparisons.

### Weaknesses
1. Evidence of training/evaluation failures. Several reported per-dataset results are near-zero, strongly suggesting misconfiguration, broken preprocessing, or training collapse: Polyp-PVT: IoU 0.06 on DCA; CFPNet-M: IoU 0.00 on Promise and OCT; SimpleUNet: IoU 0.03 on OCT; EMCAD: IoU 0.00 on Synapse. Such values are inconsistent with prior literature expectations and should trigger ablations (loss curves, seeds, lr/scheduler sweeps, image size/normalization checks, data-loader sanity tests). It also raises concern with the validity of experiments, training, and evaluation. 

2. One-size-fits-all training likely biases results: enforcing a single optimizer, schedule, or resolution across heterogeneous families (CNN, Transformer, Mamba, RWKV, Hybrid) can under-train some methods and over-favor others. 

3. Metrics are incomplete relative to community norms. IoU alone is emphasized, while Dice and HD(95) (standard in medical segmentation) are not consistently reported, limiting comparability and clinical relevance. 

4. U-Score is model pool-dependent. It normalizes each factor using model-pool 10th/90th percentiles; adding/removing models shifts percentiles and can reorder rankings. It uses a harmonic mean over three correlated efficiency terms (Params/FLOPs/FPS) and then another harmonic mean with accuracy, effectively double-weighting efficiency. Equal weighting is assumed without sensitivity analyses. 

5. Advisor agent analysis is thin: features are discretized, train/test split is small, and there is limited evidence of deploy-time gains (e.g., fewer trials to reach a target U-Score).

6. Inconsistent loss weights in equation 7 and text (lines: 1601-1605); inconsistent variables in text and equation 8 (lines: 1609-1614).

### Questions
1. For the near-zero IoUs (Polyp-PVT at DCA, CFPNet-M at Promise/OCT, SimpleUNet at OCT, EMCAD at Synapse), what went wrong?

2. Why enforce a single SGD, LR=0.01, 300-epoch configuration for all models instead of allowing a small, budgeted tuning per model to avoid systematic under-training?

3. Could you please benchmark models based on Dice and HD95 to align with standard reporting?

4. Please provide U-Score sensitivity analyses: alternate percentiles (e.g., 5/95), different weights between accuracy/efficiency and within efficiency, and rank-stability (Kendall-$\tau$) vs IoU-only and a Pareto-frontier view.

5. For the advisor, evaluate leave-one-modality-out settings with confidence intervals and compare to simple heuristics (e.g., top-K by IoU on a closest dataset).

### Soundness
1

### Presentation
2

### Contribution
2

### Rating
2

### Confidence
5

---

## Human Reviewer 3

### Summary
The paper introduces a large-scale benchmark of 100 U-shaped segmentation networks and evaluates across 28 datasets spanning 10 medical imaging modalities, with three main tests of statistical robustness, zero-shot generalization, and computational efficiency. Moreover, the authors propose U-Score, a metric combining accuracy with parameters/FLOPs/FPS via quantile-normalised harmonic means

### Strengths
- The idea of evaluating diverse networks proposed in the field of medical image segmentation is interesting. Usually, the number of papers and networks proposed in the field is high, and it makes it tough for the community.

- The experiments and results are very thorough and completely address diverse varieties of architecture-family, including CNN/Transformer/Mamba/RWKV/Hybrid.

- The paper highlights the gaps in the current practice of segmentation networks, where most of the recent works omit zero-shot generalisation and statistical tests.

### Weaknesses
- The paper uses equal weighting of accuracy vs. efficiency and of each efficiency component (Params/FLOPs/FPS) in the U-Score benchmark, which seems to be chosen arbitrarily. Is there a reason for this?

- Claims like "first large-scale, statistically rigorous benchmark" are strong, but comparisons (Table 1) overlook some previous works [1].

- Some of the models evaluated work on different protocols such as different input sizes, how do you ensure that resizing performed does not affect their performance?



[1] Bassi PR, Li W, Tang Y, Isensee F, Wang Z, Chen J, Chou YC, Kirchhoff Y, Rokuss MR, Huang Z, Ye J. Touchstone benchmark: Are we on the right way for evaluating ai algorithms for medical segmentation?. Advances in Neural Information Processing Systems. 2024 Dec 16;37:15184-201.

### Questions
- Can you provide ablations varying weights among accuracy, Params, FLOPs, FPS for the U-Score metric?

- The provided GitHub link does not seem to have the implementations, and the requested files are not found. Can you double check?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
5

---

## Human Reviewer 4

### Summary
The paper presents a benchmark of different families of U-shaped architectures for biomedical image segmentation. The benchmark comprises 28 datasets spanning various modalities. The authors propose a novel metric to combine performance with efficiency, known as the U-score. They analyze how the model performance and efficiency evolves with the year of publication across different domains. The study shows that in-domain performance tends to saturate while more modern models focus increasingly on efficiency gains and zero-shot performance, thanks to the progress in long-range dependency modeling families such as the transformers, mamba, or RWKV. The authors state that many of the benchmarked architectures present non-significant in-domain gains with respect to the original U-Net. The authors also describe how the different model families handle easy and complex cases to segment, and with this information, they build an XG-Boost-powered agent that, based on dataset characteristics and efficiency requirements, suggests a list with the best architectures to be implemented in each case.

### Strengths
The paper evaluates a large number of U-shaped architectures across multiple modalities, surpassing the scale of previous literature. Importantly, the authors do not only benchmark models based on model performance, but also on efficiency, which is a key factor for the clinical adoption of a segmentation model, especially in clinical setups with limited access to expensive hardware such as GPUs. The information on model performance and efficiency is distilled in a newly proposed metric, which is then evaluated for each modality and each model family across the time of publication. The authors reveal key information on the global progress in the field of image segmentation, such as the fact that many novel architectures are not providing significantly better results, or that in-domain model performance is saturating while efficiency and zero-shot capabilities are improving, thanks to recent advances in long-range dependency modeling.
The paper even goes beyond benchmarking by proposing an advisor agent that recommends the best architectures based on dataset characteristics and efficiency requirements. The authors also provide their code for reproducibility, which further strengthens the transparency of this work.

### Weaknesses
Major:
1) The criteria used to select the benchmarked architectures are not provided (citations, performance, efficiency, year of publication, etc.). Some SOTA architectures such as ResEnc U-Net (currently utilized by the self-configuring framework nnU-Net), MedNeXt, CoTr, STU-Net, or 3D-UXNet, are not part of the benchmark. Similarly, the authors do not mention any criteria to discard any architectures from the benchmark. Given that the paper aims to present a representative and comprehensive benchmark, clarifying these inclusion and exclusion criteria is critical to ensure the validity of the study.

2) All datasets are processed as 2D, but in 3D modalities such as MR or CT, the state-of-the-art is imposed by 3D models. We are not asking for a 3D benchmark, but it could at least be reported how the 2D slices from these 3D datasets were processed (for example, which slice orientations were used). 

3) We noticed that some of the architectures are 3D, for example, SwinUNETR, but the data processing is in 2D. It would be good to know how these 3D architectures are adapted to 2D data.

4) If significance testing is not only done for performance but also for efficiency metrics, multiple testing corrections should be applied.


Minor:
1) Figure 1 presents a lot of visual information, making it hard to interpret at a glance. Some subfigures 1A, 1B, and 1G appear to add limited scientific information to the reader and saturate the figure. Simplifying the figure would improve clarity.

2) The types of encoder and decoder choices in Figure 2, together with their respective colors (Pure A, Pure B, etc.), are not explained in the text, only in the appendix.

3) The acronym RWKV is never explained in the text, assuming the reader already knows about this.

4) In line 177, the authors mention that U-shaped models comprise three elements, but then they enumerate four.

5) The paragraph spanning lines 251–263 would fit better in the Results and Discussion section.

6) In Fig. 5, it is unclear whether the statistical tests are conducted on the IoU or on the U-scores.

7) Dice coefficient is a more commonly used metric in the image segmentation community than IoU, although both are related. 

8) The authors do not explicitly state whether all datasets used in the benchmark are publicly available. Although this seems likely, it would strengthen the manuscript to include a clear statement confirming dataset availability and ethical compliance.

9) At the end of line 335, is → are

10) The use of Comic Sans for figures reduces the visual professionalism. Perhaps a more standard scientific font would enhance the presentation.

11) The framework name “U-Bench” changed to “U-Stone” in the appendix.

### Questions
1) Several recent state-of-the-art architectures, such as ResEncL, MedNeXt-L, CoTr, and STUNet-L, appear to be omitted, even though many of them are evaluated in nnU-Net Revisited, which this work claims to outperform. Could the authors clarify the criteria used for selecting or excluding architectures in the benchmark, and explain the rationale for omitting these recent models?

2) The benchmark includes both 2D and 3D architectures, yet it is unclear how the authors handled the dimensionality mismatch between models and datasets. How did the authors process 2D slices from 3D modalities such as MR or CT, and how were 3D models adapted for 2D data (or vice versa)? Clarifying these preprocessing and adaptation steps would be important for understanding the fairness and comparability of the benchmark results.

3) Could the authors consider, for future work, consulting clinicians to assess whether they prioritize segmentation performance or computational efficiency? Based on such insights, it might be possible to design a U-score metric with performance and efficiency weights grounded in clinical evidence.

### Soundness
2

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3