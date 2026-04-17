# Reliable Evaluation of MRI Motion Correction: Dataset and Insights

- Decision: Accept (Poster)
- Scores: 4, 2, 6, 6

## Abstract
Correcting motion artifacts in scientific and medical imaging is important, as they significantly impact image quality. 
However, evaluating deep learning-based and classical motion correction methods remains fundamentally difficult due to the lack of accessible ground-truth target data. 
To address this challenge, we study three evaluation approaches: real-world evaluation based on reference scans, simulated motion, and reference-free evaluation, each with its merits and shortcomings. 
To enable evaluation with real-world motion artifacts, we release PMoC3D, a dataset consisting of unprocessed $\textbf{P}$aired $\textbf{Mo}$tion-$\textbf{C}$orrupted $\textbf{3D}$ brain MRI data. 
To advance evaluation quality, we introduce MoMRISim, a  feature-space metric trained for evaluating motion reconstructions. 
We assess each evaluation approach and find real-world evaluation together with MoMRISim, while not perfect, to be most reliable. 
Evaluation based on simulated motion systematically exaggerates algorithm performance, and reference-free evaluation overrates oversmoothed deep learning outputs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper attempts to address the evaluation problem for MRI motion correction algorithms. It contributes a small-scale (n=8) paired dataset (PMOC3D) and a new metric (MoMRISim). The authors use this to argue that simulation-based evaluation is misleading and reference-free metrics are biased.

### Strengths
The paper focuses on an important problem of MRI motion correction about reliable evaluation. It provides some experimental evidence pointing out the pitfalls of simulation and reference-free evaluations. The MoMRISim metric shows a high correlation with a single rater on this specific dataset.

### Weaknesses
1. The core contribution (a dataset and a metric) is weak. First, a dataset of n=8 subjects is too small to serve as a meaningful or generalizable benchmark. Second, the concept of paired scans is not novel in the medical imaging domain (e.g., in infant datasets like BCP). The work feels more like a domain-specific dataset paper than a methodological contribution suitable for ICLR.
2. The paper's entire argument is self-contradictory. The authors' central thesis is that simulated data is "unreliable". Yet, they use this "unreliable" simulated data to train their proposed metric MoMRISim.
3. The benchmark's clinical relevance is questionable. All quantitative conclusions are derived exclusively from instructed voluntary motion. The paper fails to provide any analysis of the involuntary motion samples mentioned in the dataset, thus lacking evidence that instructed motion is a valid proxy for the true clinical challenge.
4. The paper's core claims suffer from low statistical power. The conclusions are drawn from a small sample size of only eight subjects. Critically, the gold-standard PMAS used to validate the MoMRISim metric was established by a single licensed medical doctor, with no inter-rater reliability analysis. This design means the headline claim (MoMRISim's $\rho=0.92$ correlation) lacks statistical significance. The method's applicability is also severely limited, as the authors concede their reference-based approach is invalid in the mild-motion regime.

### Questions
1. Please discuss the core contradiction (Weakness #2): Why should a metric trained on "unreliable" simulated data be trusted to evaluate real data?
2. Please provide an analysis to justify why instructed motion (used in the paper) is a valid proxy for involuntary motion.
3. Please comment on how the low statistical power (n=8 subjects, 1 rater) affects the confidence in the claim that MoMRISim is the best metric.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents two main contributions. The first is a curated dataset comprising T1-weighted brain MR images from eight subjects. For each subject, scans were acquired with four distinct levels of induced motion artefacts, including a reference scan presumed to be motion-free. Each image is accompanied by a "perceived motion artefact score."

The second contribution is the proposal of two novel metrics for quantifying motion artefact severity. The first, MoMRISim, is a reference-based metric. It is an adaptation of the DreamSim approach (Fu et al., 2023) that is specifically trained on data with simulated motion severity levels instead of manually annotated images. The second is a reference-free metric, termed the VLM score, which is derived by prompting a Vision-Language Model.

The authors conduct an analysis to study and compare the behaviour of these proposed metrics against existing methods using their newly introduced dataset. They investigate the correlation between visual ratings and both reference-based and reference-free metrics, as well as the variations in performance when using synthetic motion-corrupted MRIs compared to images with real motion.

### Strengths
- **Valuable dataset**: The creation of a dataset with real MRI scans acquired at different, controlled levels of motion artefact, including a motion-free reference for each subject, is a significant strength. Such data is difficult and costly to obtain, and it provides a valuable resource for developing and rigorously evaluating motion detection and correction algorithms.
- **Novel and relevant metrics**: The introduction of two new metrics—a tailored reference-based metric (MoMRISim) and a novel reference-free metric using a Vision-Language Model (VLM score)—addresses an important need in the field. The goal of developing metrics that align with human visual perception of artefact severity is highly relevant and practical for automated quality control.

### Weaknesses
- **Poor clarity and structure**: The paper's organisation significantly hinders comprehension. The introduction lacks a foundational explanation of motion artefacts, their impact, and existing detection/correction methods. Key details, such as the derivation of the "perceived motion artefact score" (Section 2.2), are insufficiently explained in the main text. Section 3, which covers evaluation approaches, confusingly mixes descriptions of existing and proposed solutions without clear distinction. The rationale for selecting the three specific motion reconstruction methods used for evaluation is not provided, making the experimental design appear arbitrary.
- **Unclear methodological contributions**: The technical description of the two proposed metrics is unclear. For the reference-based MoMRISim metric, it is not adequately explained how the adaptation of DreamSim differs from the original work beyond the use of simulated motion data for training. For the reference-free VLM score, the prompting strategy and the specific Vision-Language Model used are not described with sufficient detail to understand or reproduce the method.
- **Ambiguous evaluation**: The design and interpretation of the results are problematic. Figures like Figure 2 are unclear; it is not evident if each data point represents a unique image or if the same subject/image is represented multiple times for different reconstruction types. This raises a critical concern that the observed correlations may be driven by the differences between reconstruction methods rather than by the metrics' ability to assess motion severity within a consistent context. This potential confound undermines the validity of the conclusions.

### Questions
- **Figure 2 interpretation**: Could you clarify what each data point in Figure 2 represents? Specifically, does a single subject/image appear multiple times for different reconstruction types, and if so, how does this affect the correlation analysis?
- **Metric details**: For the VLM score, please specify the exact prompt and the Vision-Language Model used. For MoMRISim, what are the key differences in the training protocol compared to the original DreamSim?
- **Baseline selection**: What was the rationale for choosing the three specific motion reconstruction methods used in the evaluation?
- **Motion simulation parameters** (Section 3.2): How were the number of events and the amplitude of the motion simulation parameters chosen? 

Typos : 
- In legend for Fig.1, images S7_1 and S5_1 are mentioned, albeit not actually appearing in the figure. 
- Page 7 line ~34: mention of ‘AltOpt’, but abbreviation not previously introduced in first element of bullet-point list in part 4.1
- Page 9 line ~482: missing d letter “the full complexity of real-world motion and tends to […]”

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This work aims to construct a reliable and standardized evaluation pipeline for the MRI motion correction (MoCo) problem. The main contributions include three parts: 1) Collecting PMOC3D, a real-world MRI dataset; 2) Introducing MoMRISim, a feature-based MoCo metric; 3) Benchmarking several MoCo methods (including both traditional iterative and deep learning-based approaches) on simulated and real PMOC3D datasets. Overall, this work builds a comprehensive and reliable evaluation framework for the MRI MoCo task. It is interesting work.

### Strengths
- The collected PMOC3D dataset includes paired motion-free and motion-corrupted 3D k-space raw data, covering diverse motion types. Its construction and release can enable the development of more advanced deep learning–based MoCo methods, which is a critical contribution.
- The work explores many quantitative evaluation metrics for MoCo reconstructions, including reference-based metrics (e.g., pixel-level SSIM and PSNR, feature-level DreamSim) and reference-free metrics (e.g., AES, TG, and VIM scores based on LLMs). Moreover, a new feature-based metric, MoMRISim, is introduced. These studies on evaluation metrics may provide valuable tools for assessing MRI MoCo accuracy.
- Benchmarking several recent MoCo models on both simulated and real datasets provides useful insights that could guide future development of more advanced MoCo approaches.

### Weaknesses
- The PMOC3D dataset only includes 8 subjects. While I understand that MRI data collection is costly and time-consuming, this number is still relatively small. Therefore, the dataset may be more suitable as a test set rather than for training deep learning models.
- To my knowledge, the combination of deep learning models and physics-based iterative reconstruction is currently a mainstream paradigm for MRI MoCo [1][2][3]. These methods typically involve motion trajectory estimation. However, the proposed dataset does not provide the motion trajectories of the scanned subjects, which may limit its usability for evaluating many advanced MoCo methods.
- For building PMAS, two Ph.D. students performed the classification of motion types. I believe involving clinical experts in this process would improve the reliability of the annotations.
- Regarding the proposed MoMRISim metric, the paper lacks a sufficient intuitive explanation. As a reader unfamiliar with DreamSim, I found it difficult to understand the principle and motivation behind MoMRISim (Lines 270–286).

[1] Levac, Brett, et al. "Accelerated motion correction with deep generative diffusion models." Magnetic Resonance in Medicine 92.2 (2024): 853-868.

[2] Wu, Qing, et al. "Moner: Motion Correction in Undersampled Radial MRI with Unsupervised Neural Representation." The Thirteenth International Conference on Learning Representations.

[3] Singh, Nalini M., et al. "Data consistent deep rigid MRI motion correction." Medical imaging with deep learning. PMLR, 2024.

### Questions
See Weaknesses, please.

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The manuscript investigates how to **reliably evaluate 3D MRI motion correction** algorithms when ground truth is unavailable. It introduces **PMoC3D**, a paired real and motion-corrupted brain MRI dataset with raw k-space data, and **MoMRISim**, a learned feature-space similarity metric trained through triplets representing different motion severities. The authors comprehensively compare real-world, simulated, and reference-free evaluation paradigms. The results are convincing: on PMoC3D, the proposed feature-based metric (MoMRISim) correlates best with expert judgments, while simulation-based evaluation tends to overestimate performance, and reference-free metrics (including a VLM score) may favor oversmoothed outputs. The paper provides thorough details about data acquisition, PMAS (Perceived Motion Artifact Score) construction, baselines, and implementation.

### Strengths
- **Well-scoped and clearly structured problem framing.**  
The paper provides a well-defined problem setup and a thoughtful taxonomy covering paired, simulated, and reference-free evaluations. It clearly explains why obtaining ground truth is inherently difficult in 3D MRI, which makes the motivation convincing and the study design logical.

- **PMoC3D: a valuable paired 3D dataset with raw k-space.**  
The authors present PMoC3D, a paired dataset that includes raw multi-coil k-space data, coil sensitivity maps, sampling trajectories, and timestamped motion instructions. The inclusion of raw acquisition data makes this dataset especially valuable to the community.

- **MoMRISim: a motion-aware learned metric.**  
The proposed MoMRISim metric is technically sound and well-motivated. It is trained using a triplet-learning strategy on simulated motion severities and simplifies to a cosine distance during testing, making it both practical and interpretable. It serves as an MRI-specific perceptual proxy with clear relevance.

- **Careful and realistic experimental design.**  
I find it particularly commendable that the authors conducted every simulation and validation with great attention to detail, making the setup as realistic as possible. This level of experimental care is not common and significantly increases the credibility of the findings.

- **Meaningful impact and insights for the community.**  
Overall, this work has substantial value for the MRI motion-correction field. The authors convincingly show that existing approaches perform far from ideal in real motion correction and, through diverse experiments, provide numerous insights into evaluation strategies and failure modes. The study feels genuinely impactful, and I look forward to future developments in this direction.

### Weaknesses
**Limited dataset size and missing motion trajectory data.**  
The dataset PMoC3D is an important contribution, but there are some limitations. First, the sample size is quite small, containing only eight subjects. Second, although the dataset includes rich metadata such as raw k-space, motion instructions, and timestamps, it lacks **true motion trajectories**, meaning the exact six-degree-of-freedom head poses over time. Including this information would greatly enhance the dataset’s scientific value and utility. I understand that capturing such detailed motion trajectories is technically challenging, but it could be a meaningful improvement for future versions.

### Questions
1. In line 36, the paper mentions the work of **Wu et al. (2025)**. As far as I know, that work also conducts **3D MRI motion correction** experiments and adopts a fully 3D representation rather than stacked 2D slices. Given its strong relevance, the authors should discuss this work in the section reviewing related 3D motion-correction studies.  
2. In Figure 1, the caption appears to contain a small error. The labels “S7_1” and “S5_1” do not match the figure annotations.

### Soundness
4

### Presentation
2

### Contribution
4
