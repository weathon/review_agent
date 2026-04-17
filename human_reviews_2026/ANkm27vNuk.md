# Brain-Semantoks: Learning Semantic Tokens of Brain Dynamics with a Self-Distilled Foundation Model

- Decision: Accept (Poster)
- Scores: 8, 4, 8, 6

## Abstract
The development of foundation models for functional magnetic resonance imaging (fMRI) time series holds significant promise for predicting phenotypes related to disease and cognition. Current models, however, are often trained using a mask-and-reconstruct objective on small brain regions. This focus on low-level information leads to representations that are sensitive to noise and temporal fluctuations, necessitating extensive fine-tuning for downstream tasks. We introduce Brain-Semantoks, a self-supervised framework designed specifically to learn abstract representations of brain dynamics. Its architecture is built on two core innovations: a semantic tokenizer that aggregates noisy regional signals into robust tokens representing functional networks, and a self-distillation objective that enforces representational stability across time. We show that this objective is stabilized through a novel training curriculum, ensuring the model robustly learns meaningful features from low signal-to-noise time series. We demonstrate that learned representations enable strong performance on a variety of downstream tasks even when only using a linear probe. Furthermore, we provide comprehensive scaling analyses indicating more unlabeled data reliably results in out-of-distribution performance gains without domain adaptation.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents Brain-Semantoks, a novel fMRI foundation model that shifts the paradigm from reconstruction-based learning to abstraction-based representation learning. The authors propose a neuroscientifically grounded semantic tokenizer that aggregates voxel signals into functionally meaningful network-level tokens, coupled with a self-distillation objective across temporal views to learn stable, high-level representations of brain dynamics. To mitigate instability from noisy fMRI data, a Teacher-Guided Temporal Regularizer (TTR) is introduced to gradually guide the model toward robust convergence. Extensive experiments on multiple public datasets show that Brain-Semantoks achieves state-of-the-art linear probing performance, outperforming both supervised and existing self-supervised methods, with strong generalization across tasks without domain adaptation. Overall, the work makes a compelling case for semantically informed, abstract representation learning in neuroimaging foundation models.

### Strengths
1.  Reorients fMRI foundation modeling from masked reconstruction to semantic abstraction via a functional-network tokenizer, aligning tokenization with neuroscientific priors rather than voxel/ROI grain. The self-distillation across temporal crops plus TTR curriculum is a thoughtful adaptation of DINO/iBOT-style ideas to low-SNR fMRI. 

2. Solid ablation suite quantifies the effect of tokenizer design (structured conv, patch length), masking scheme/ratio, and TTR duration; results cohere with the hypotheses (e.g., higher masking ratios; network/slice masking reducing interpolation; moderate mask-loss weight). 

3. Clear articulation of the problem with reconstruction-centric objectives for phenotype prediction and a well-motivated methodology figure/flow (student-teacher; temporal views; semantic tokens). Related work positioning is balanced. 

4. Demonstrates SOTA linear-probe performance across diverse tasks, often surpassing supervised methods, suggesting practically valuable, broadly transferable embeddings; claims of scaling and OOD gains are timely for neuro-foundation models.

### Weaknesses
1. Heavy emphasis on linear probing; fewer results on full/partial fine-tuning (or lightweight adapters) and limited analysis of task-based fMRI limits claims about universal transfer beyond resting-state embeddings. The authors acknowledge the task-fMRI gap as future work. Including even a small task-fMRI probe or adapter-tuning study would strengthen external validity. 

2. Tokenizer depends on fixed functional networks; while neuro-plausible, robustness to alternative parcellations or data-driven grouping is only proposed as future work. A sensitivity analysis across multiple parcellations (e.g., Schaefer/Glasser granularity) would bolster generality. 

3. Cross-dataset variability (hardware/protocols/cohorts) is central; yet the paper could provide a deeper distribution-shift audit (e.g., per-site/per-scanner breakdowns, ComBat-style harmonization baselines, robustness to TR differences) to substantiate OOD claims. Authors note investigating harmful shifts as future work. 

4. While ablations are thorough, a targeted component attribution (e.g., semantic tokenizer vs. self-distillation vs. TTR via controlled swaps against strong masked-JEPA baselines) would clarify which element drives most of the gain. Current tables hint at trends but don’t isolate effect sizes against matched alternatives. 

5. The “rigorous linear-probe protocol” could use a compact table specifying classifier type, search space, train/val splits, number of repeats, and CIs across all tasks to help reproducibility reviewers quickly verify comparability. (Some of this is in appendices; mirroring a summary in the main text would help.)

### Questions
1. Can you report adapter-based or LoRA-style finetuning on a subset of downstream tasks to test whether the learned features remain robust under end-to-end adaptation and to compare against masked-JEPA/BrainLM finetuned baselines?

2. How do results vary across alternative parcellations and numbers of networks/tokens (e.g., Schaefer 100/400, Glasser 360)? You ablate 1/9/20/58 tokens—could you include cross-parcellation comparisons and per-task preferences? 

3. Please provide per-site/scanner/TR robustness analyses and/or harmonization baselines (e.g., ComBat) to substantiate OOD claims; which specific shifts most degrade transfer, and does the semantic tokenizer mitigate them? 

4. Could you include a matched JEPA/MAE-style latent reconstruction baseline with the same tokenizer and encoder to isolate the benefit of self-distillation + TTR? 

5.  Beyond cropping two long segments, did you test time-warping/jittering or frequency-domain perturbations that preserve physiological plausibility? How sensitive is performance to Tcrop? 

6. Any early evidence on task-fMRI or clinical endpoints (diagnosis/prognosis) to complement resting-state tasks, even if small-scale?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Brain‑Semantoks, a self‑supervised fMRI foundation model that shifts from reconstruction to abstraction by (1) introducing a semantic tokenizer that aggregates ROI time series into network‑level tokens and (2) pretraining via self‑distillation across long temporal views with a Teacher‑guided Temporal Regularizer (TTR) for stability on noisy fMRI signals. The model shows strong linear‑probe performance across nine downstream datasets/tasks (ABIDE, HBN, UKB, SRPBS), outperforming BrainLM and Brain‑JEPA and often rivaling or surpassing fully supervised/fine‑tuned baselines, with ablations and scaling analyses.

### Strengths
- Clear reframing toward semantic abstraction with a neuroscience‑grounded tokenizer operating at functional network granularity, reducing token length and noise while injecting inductive bias.

- The slice masking to avoid trivial interpolation is a strong regularization that forces the model to learn meaningful relationships between tokens. 

- Well‑designed curriculum via TTR that averages network tokens over time early in training, improving stability of the model during training

- Rigorous linear‑probe protocol with 10‑fold CV and standardized heads; strong results across clinical and demographic tasks

- Detailed scaling analysis for fMRI FMs under linear probing, showing power‑law like gains with pretraining size and OOD improvements without domain adaptation.

### Weaknesses
- The atlas choice is mostly arbitrary. No analysis of how results change with alternative parcellations (Schaefer, Shen, Yeo‑17) or different subcortical/cerebellar groupings; no exploration of data‑driven network discovery to justify the choice of nine functional networks.

- The geometry of learned network identity embeddings is not analyzed; it is unclear whether they capture canonical inter‑network relationships or known hierarchies.

- Precise kernel sequences and decay parameters are unclear; there is no connection to hemodynamic time constants nor visualization of learned filters or frequency responses to support the inductive bias.

- The scaling of the coding‑rate regularizer with batch size and feature dimension is not described; the influence of batch‑dependent normalization on the covariance and the regularizer is not evaluated.

- There is no evaluation of physiologically realistic, band‑limited, or acquisition‑mimicking perturbations (e.g., motion, respiratory/cardiac noise, TR/resampling artifacts) beyond simple channel/time zeroing, Gaussian noise, and amplitude scaling.

### Questions
- Functional networks: Reducing the complexity of the input data via the use of an atlas for grouping based on functional network annotation is a sensible choice. However, as briefly discussed by the authors, there are several alternatives for such atlases (eg, Schaefer, Shen or even Yeo-17 [see ref1 for other atlases]), from which Yeo-7, in addition to the two subcortical networks, is the most minimalist. How sensitive are the results to the chosen atlas and the definition of subcortical/cerebellar groupings? Have you evaluated alternative parcellations? Have you considered a data‑driven network discovery for an appropriate number of functional networks?

- Positional and network embeddings: Does the model learn network identity embeddings that capture known functional relationships? Any probing of these embeddings’ geometry vs canonical network hierarchies?

- Structured convolutions: What precise kernel sequences and decay profiles were used, and how do they relate to expected hemodynamic time constants? Can you share learned filters or frequency responses to justify the inductive bias?

- Coding‑rate regularizer: how is R scaled with batch size and feature dimension; what is the effect of batch‑dependent normalization on R?

- Augmentations: The paper uses light corruptions (channel/time zeroing, Gaussian noise, amplitude scaling), which likely result in out-of-distribution data. Have you tested physiological noise models or band‑limited perturbations that mimic TR/resampling artifacts?

References:

[ref1] https://www.lead-dbs.org/helpsupport/knowledge-base/atlasesresources/cortical-atlas-parcellations-mni-space/

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces Brain-Semantoks, a self-supervised fMRI foundation model that combines a semantic tokenizer and a student–teacher self-distillation framework to learn high-level, temporally stable representations of brain dynamics. Instead of voxel- or region-level reconstruction, the model produces recording-level embeddings based on functional brain networks. The approach shows improved downstream performance across several neuroimaging datasets and phenotypic prediction tasks, such as age, sex, and ASD classification.

### Strengths
•	Innovative use of self-distillation and semantic tokenization to learn stable, abstract representations of brain activity.
•	Clear performance improvements over existing foundation models (e.g., BrainLM, Brain-JEPA).
•	Semantic tokenizer proves particularly effective for demographic and clinical predictions (age, sex, ASD).
•	The shift from low-level voxel embeddings to network-based embeddings is conceptually strong.
•	Significant ablation studies to explore the benefit of each component into the model, which shows robust evaluations by the authors.

### Weaknesses
•	While results are solid, gains from other architectural components beyond the tokenizer are more modest.
•	The paper could discuss temporal resolution more thoroughly — would finer sampling (e.g., sub-2s TR) lead to better representations or unnecessary noise amplification?
•	The work focuses exclusively on resting-state data; some commentary on potential extension to task-based fMRI would strengthen the contribution.

### Questions
1.	Have you tested whether higher temporal resolution (beyond the 2s sampling) affects performance?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The manuscript proposes a self-supervised foundation model for fMRI time series based on a transformer architecture. It introduces a semantic tokenizer that aggregates ROI time series into network-level tokens. The model is trained using a combination of three objectives: a global loss that minimizes the Euclidean distance between global summary tokens from different views, a network token loss that aligns masked and unmasked network tokens, and a temporal regularizer that uses time-averaged representations across each network. Experiments on several neuroimaging datasets (UK Biobank, ABIDE, HBN, SRPBS, and LEMON), show that the proposed model performs better than previous self-supervised and supervised methods.

### Strengths
- The model is evaluated on multiple datasets, including UK Biobank, ABIDE, HBN, SRPBS, and LEMON, and shows consistent improvements over strong baselines such as BrainLM and Brain-JEPA.
- The paper also includes extensive ablation studies on the effects of the semantic tokenizer design, temporal regularizer duration, masking type, loss components, and masking ratio.

### Weaknesses
- A main weakness is that all experiments rely solely on resting-state fMRI data, which limits the claim of being a true foundation model.
- Tables 1 and 2 need to have a statistical comparison between the best model and other models. It is commonly done with a pairwise test. Then, p-values are usually corrected for multiple comparisons.

### Questions
- The paper does not specify how checkpoints were selected, and it is unclear whether the ablation results are based on the validation set, as we can not select hyperparameters on the test set. Please clarify your checkpoint strategy.
- It seems the authors used the wrong ICLR LaTeX template or edited it to fit more content.

### Soundness
3

### Presentation
3

### Contribution
3
