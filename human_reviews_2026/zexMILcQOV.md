# MnemoDyn: Learning Resting State Dynamics from  $40$K FMRI sequences

- Decision: Accept (Poster)
- Scores: 6, 2, 6

## Abstract
We present a dynamical-systems based model for resting-state functional magnetic resonance imaging (rs-fMRI), trained on a dataset of roughly $40$K rs-fMRI sequences covering a wide variety of public and  available-by-permission datasets. While most existing proposals use transformer backbones, we utilize multi-resolution temporal modeling of the dynamics across parcellated brain regions. We show that MnemoDyn is compute efficient and generalizes very well across diverse populations and scanning protocols. When benchmarked against current state-of-the-art transformer-based approaches, MnemoDyn consistently delivers superior reconstruction quality.
Overall, we find that with such large-scale pre-training on (non-proprietary) rs-fMRI datasets, we get a highly performant model for various downstream tasks. Our results also provide evidence of the efficacy of the model on small sample size studies which has implications for neuroimaging studies at large where resting state fMRI is a commonly acquired imaging modality.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper MnemoDyn: Learning Resting-State Dynamics from 40K fMRI Sequences introduces MnemoDyn, a foundation model for resting-state fMRI that replaces attention mechanisms with operator-based dynamical system modeling. Instead of tokenized sequence learning, MnemoDyn learns multi-resolution temporal operators using wavelet parameterization and pseudo-differential kernels to efficiently capture multiscale neural dynamics. The architecture offers computational efficiency and scalability to long sequences, achieving strong reconstruction and predictive performance across datasets such as UK Biobank, HCP, and ADNI. Fine-tuned versions of MnemoDyn show improved accuracy and robustness on downstream tasks including age, sex, and disease classification. Overall, the paper argues for a biologically grounded, operator-based alternative to transformer models for neuroimaging, highlighting interpretability and compute efficiency but offering only moderate conceptual novelty and limited theoretical validation.

### Strengths
1. Proposes an elegant operator-based formulation for modeling brain dynamics, avoiding attention mechanisms.
2. Achieves state-of-the-art results on multiple large and small-scale fMRI datasets with high compute efficiency.
3. Presents a comprehensive evaluation pipeline, supporting claims of generalization and scalability.
4. Aligns with neuroscientific priors regarding multiscale temporal structure, enhancing biological plausibility.

### Weaknesses
1. Conceptual novelty is modest; builds on existing operator learning and neural ODE frameworks without substantial innovation.

2. The connection to neuroscientific principles is largely rhetorical, lacking physiological validation or interpretability analysis.

3. The paper does not sufficiently contrast MnemoDyn against simpler baselines (e.g., recurrent or convolutional models) to justify complexity.

4. Results, though impressive numerically, may be inflated by dataset overlap and lack of external reproducibility tests.

### Questions
1. How sensitive is MnemoDyn to the choice of wavelet basis and parameterization depth?

2. Could the operator-based approach generalize effectively to multimodal or task-based fMRI?

3. Does the model yield interpretable components that map to known neural circuits or physiological rhythms?

4. How does MnemoDyn’s performance scale with smaller datasets without fine-tuning?

5. Would the model retain benefits under voxel-level resolution rather than parcellated inputs?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces a continuous-time latent dynamical model that learns an evolution operator over rs-fMRI trajectories, parameterized via multi-resolution wavelet kernels and implemented with pseudo-differential operators to induce sparsity and block-diagonal computation.​

Pretrained on ~40K rs-fMRI sequences and fine-tuned with lightweight heads, MnemoDyn improves reconstruction, clinical classification (ADNI), and demographic/trait prediction (UKB, HCP-Aging) versus other approaches, such as Brain-JEPA, BrainLM, and graph CNNs, while requiring less computing power.​

### Strengths
- The paper proposes an alternative to the tokenization/positional heuristics problems, which are common to transformer-based methods.

- Efficient multiscale design: wavelet-domain pseudo-differential parameterization and CP tensor factorization exploit sparsity and low-rank structure, enabling single-GPU pretraining and potentially allowing scaling to longer sequences than conventional transformer-based methods can handle.​

- Broad and consistent gains: large improvements on ADNI NC/MCI and amyloid classification, and better age/sex/trait prediction on HCP-Aging and other cohorts (HBN, ADHD-200, NKIR), indicating robust transfer across sites/protocols.​

### Weaknesses
- Operator interpretability is under-explored: the paper motivates pseudo-differential structure and multiscale kernels but provides limited analysis of learned symbols/kernels or neurophysiological correlates of the operator.​

- Pretraining objectives comparison is shallow: masking and JEPA-style training are included, but ablations isolating what aspects of the operator parameterization versus objective drive gains are not fully disentangled.​

- The figures in the paper do not help with clarification regarding model structure, training and inference.

### Questions
- Figure 3 presents parcel reconstructions for unseen data, where the predicted signals show an almost perfect overlap with the measured fMRI time series. Since this is unseen data (ie, not seen during training), this raises concerns about how inference is performed with this model. Are these outputs generated by evolving the latent state from an initial condition (ie, first point of the dynamics)? Is this the result of solving equation 11? If the reconstruction indeed follows the ODE solution over time, it is surprising that the model can reproduce signals with noise-like structure, such as those shown for the UK Biobank example in Fig 3. More details on the inference procedure are necessary to clarify this result.

- The authors motivate their model using a continuous-time ODE formulation. However, modeling brain dynamics as a first-order (Markovian) system may be restrictive, given the nonlocal and temporally dependent nature of neuronal interactions. Could the authors elaborate on why ODEs, which assume locality in time, were chosen over formulations that can capture non-Markovian dependencies, such as integral equation (IE) or delay-based models? Prior work has shown IE formulations can yield superior numerical accuracy even for Markovian dynamics [1, 2, 3]; therefore, additional justification for the ODE assumption would be valuable.

- Since MnemoDyn is conceptually related to Fourier Neural Operator (FNO) and DeepONet—both of which learn functional mappings using spectral or integral representations—it would strengthen the work to include direct comparisons with these models. In particular, given that MnemoDyn operates in the wavelet domain (analogous to Fourier parameterization in FNO), a head-to-head evaluation would help quantify the benefits of the proposed multiresolution pseudo-differential operator design.

- Tables 2 and 3 report MnemoDyn outperforming several transformer and graph-based baselines on clinical and demographic prediction tasks. Were all baseline models fine-tuned under the same conditions and on the same datasets as MnemoDyn? Additional details about training protocols, data splits, and hyperparameter parity are necessary to assess the fairness of these comparisons.
The paper refers to MnemoDyn as a “foundation model.” Could the authors clarify the specific criteria under which this designation applies? For instance:
  - Does MnemoDyn show improved performance with scale (in model or data size)?
  - Can its representations transfer effectively to other neuroimaging modalities or tasks beyond label prediction?
  - How does it satisfy the general-purpose, reusable, and cross-domain characteristics typically associated with foundation models? 

Ref: 

[1] Vladimir Rokhlin. Rapid solution of integral equations of classical potential theory. Journal of computational physics, 60(2):187–207, 1985.

[2] Vladimir Rokhlin. Rapid solution of integral equations of scattering theory in two dimensions. Journal of Computational Physics, 86(2):414–439, 1990.

[3] Zappala, Emanuele, Antonio Henrique de Oliveira Fonseca, Josue Ortega Caro, Andrew Henry Moberly, Michael James Higley, Jessica Cardin, and David van Dijk. "Learning integral operators via neural integral equations." Nature Machine Intelligence 6, no. 9 (2024): 1046-1062.

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
This paper introduces MnemoDyn, a multiscale state-space model designed to model resting-state fMRI data. The model uses a wavelet basis to represent temporal dynamics and a structured, learnable transition matrix A to capture latent state evolution. The approach keeps the multidimensional structure of fMRI data and shows notable improvements in fine-tuning tasks compared to existing baselines.

### Strengths
•	Creative use of multiscale state-space modeling for fMRI signals.
•	Preserves spatial correlations across brain regions rather than treating them independently.
•	Demonstrates consistent fine-tuning performance gains over strong baselines.
•	The model seems more parameter-efficient than transformers, which is valuable for neuroimaging data

### Weaknesses
•	The claimed computational efficiency isn’t empirically supported—there are no runtime or scalability benchmarks.
•	Missing interpretability analysis of the learned transition matrix A (e.g., identifying key frequencies or scales).
•	Table 1 lacks R² reconstruction metrics for HCP and UK Biobank datasets, which would be a necessary metric for comparisons.
•	The trade-off between pretraining effort and downstream performance is mentioned but not shown.
•	Writing could be tightened in the methods section for clarity.

### Questions
1.	Can you include runtime and memory benchmarks to support the efficiency claims?
2.	Could you analyze the learned matrix A to show which frequencies or scales are most predictive?
3.	Please report R² reconstruction results for HCP and UK Biobank, ideally alongside fine-tuning scores.
4.	Could you visualize or quantify the trade-off between pretraining difficulty and fine-tuning accuracy?
5.	How sensitive is the model to the choice of wavelet basis or number of scales?


Suggestions:
•	The methods section could benefit from a short schematic showing how the wavelet decomposition and latent transition components interact.
•	Consider adding an ablation study (e.g., removing multiscale structure or low-rank decomposition).

### Soundness
4

### Presentation
3

### Contribution
3
