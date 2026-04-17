# Mind the State: Towards Unified, Context-Aware EEG-to-fMRI Synthesis

- Decision: Reject
- Scores: 6, 6, 2, 4

## Abstract
Functional magnetic resonance imaging (fMRI) provides dynamic measurements of human brain activity at high spatial resolution and depth, but its use is constrained by high cost, limited accessibility, and strict acquisition requirements. Synthesizing fMRI data from more accessible, non-invasive modalities such as electroencephalography (EEG) offers a promising alternative, enabling inference of deep brain activity from low-cost scalp recordings in naturalistic settings. Despite recent progress, existing EEG-to-fMRI translation methods typically require training separate models for individual brain regions and offer limited consideration of subject-level variability in brain dynamics. In this study, we propose UniEFS, a unified EEG-to-fMRI synthesis model that enables full-brain fMRI reconstruction while accommodating datasets with varying demographic and physiological contexts within a single model. UniEFS leverages a pretrained fMRI decoder to embed rich spatial priors, as well as condition-aware prompt tokens that encode subject-level and experimental metadata to handle heterogeneous datasets. We extensively evaluate our model performance on eyes-closed resting-state data and demonstrate that it can reliably reconstruct temporally resolved whole-brain fMRI activity, with strong potential to generalize to task-based fMRI in a zero-shot setting.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces UniEFS (Unified EEG-to-fMRI Synthesis), a framework for translating EEG signals into high-resolution, temporally synchronized fMRI representations. The model operates in two stages: first, an fMRI pretraining phase using masked signal modeling (f-MSM) to learn spatial-temporal structure from large unlabeled fMRI data; second, a context-aware EEG-to-fMRI mapping that aligns EEG embeddings with the pretrained fMRI latent space using transformer architectures. UniEFS incorporates context prompts encoding vigilance, demographics, and dataset information to handle variability across subjects and acquisition settings. Extensive experiments demonstrate that the model outperforms prior EEG-to-fMRI baselines in temporal and functional connectivity correlations, generalizes zero-shot to unseen task fMRI datasets, and preserves individualized connectome patterns. Overall, the paper contributes a scalable, unified, and context-adaptive model for multimodal brain activity translation, advancing cross-modal neuroimaging representation learning.

### Strengths
1. Unified, frame-wise, whole-brain EEG→fMRI synthesis with context-aware prefix prompts (dataset, age, sex, vigilance) is a fresh and relevant formulation for heterogeneous, eyes-closed resting-state data. The two-stage route—masked fMRI pretraining + domain adaptation, then latent alignment—addresses data scarcity and improves spatial priors. 

2. Solid baselines and component ablations show: (i) pretraining matters, (ii) alignment and reconstruction losses are both needed, (iii) vigilance/demographic/dataset tokens help, with quantifiable drops when removed. Zero-shot task transfer and fingerprinting analyses further speak to generalization. 

3. The method is structured and motivated (why ROI-level, why high mask ratio, why prompts), with clear training pipeline and evaluation. Figures highlighting network-wise effects of vigilance conditioning are compelling and biologically plausible. 

4. Demonstrates consistent improvements over prior EEG→fMRI baselines in temporal correlation across cortical/subcortical/cerebellar regions, with potential downstream impact for scalable, portable neuroimaging in clinics and naturalistic settings

### Weaknesses
1. The paired EEG-fMRI datasets are relatively small (22+7 subjects, 29+10 scans), which may limit claims of broad generalization; additional public cohorts or cross-site test-only evaluations would strengthen external validity. 

2.  While prompts encode context, the paper lacks deeper analyses of site/scanner/TR shifts and how much prompts vs. pretraining mitigate them (per-site performance, leave-one-site-out).

3. Leakage/overfitting risks: The domain adaptation step fine-tunes on the fMRI portion of the paired set; more guardrails (strict subject-held-out splits are used, but clarifying leakage risks and testing strict cross-dataset generalization would help). 

4. Emphasis is on TCorr and FC correlations; adding voxel/surface recon quality (where applicable), physiological plausibility checks, and per-network temporal dynamics metrics (e.g., dynamic FC) in main text would round out evaluation.

5. Statistical rigor in main text: Some results rely on means±sd; more confidence intervals, multiple-comparison controls, and explicit p-values (beyond the single paired t-test note) in the main body would improve soundness.

### Questions
1. Can you report per-dataset/per-site results and leave-one-dataset-out evaluations to quantify true OOD generalization and the specific gains attributable to dataset tokens? 

2. Beyond on/off, can you provide effect sizes for each prompt type across networks and subjects (e.g., vigilance vs. demographics), and whether prompts ever harm performance on homogeneous subsets? 

3. In Table 2, removing Lrecon devastates performance. Could you include a matched variant using only Lrecon (no Lalign) and quantify stability/learning speed to better justify the latent alignment pathway? 

4. Please expand zero-shot results with region/network-wise breakdowns and compare against personalized fine-tuning and adapter/LoRA to test whether EEG encoders or the fMRI decoder bottleneck limits transfer. 

5. Did you try HRF-aware temporal shifts/warps or frequency-domain perturbations during Stage-2 training? How sensitive is performance to EEG window length (16s) and DiFuMo granularity (P=512)? 

6. Any preliminary analyses that relate reconstructed fMRI to behavioral/clinical covariates (even post-hoc), to support applied value claims?

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
2

### Summary
This paper proposes UniEFS, a unified and context-aware framework for reconstructing full-brain fMRI activity from EEG signals. The key innovation lies in combining self-supervised fMRI pretraining with context-conditioned EEG encoding that incorporates demographic and physiological priors. The model performs frame-wise fMRI reconstruction at the ROI level and achieves strong performance in both resting-state and zero-shot task-based settings.

### Strengths
The problem considered in this paper is very interesing. Besides, the idea of injection of auxiliary information to improve the generalization across population is interesting. The experimental validation is solid and convincing.

### Weaknesses
The paper utilizes an (\ell_2) loss to enforce alignment between fMRI and EEG. In other works, similarity scores are sometimes used for this purpose. It would strengthen the paper if the authors could provide a comparison between these two approaches. Additionally, the EEG covariance matrix captures neural interactions, which could be leveraged to guide the learning process. However, this aspect is not considered in the current work.

### Questions
See the weakness above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes UniEFS, a novel transformer-based framework to synthesize full-brain fMRI from low-density EEG recordings, addressing the long-standing challenge of inferring hemodynamic brain activity from electrical scalp measurements. The authors introduce a two-stage training approach leveraging masked fMRI self-supervised pretraining and context-aware conditioning on subject demographics, vigilance state, and dataset identity to enhance EEG-to-fMRI translation. The model is evaluated on multiple resting-state datasets and outperforms previous methods across cortical, subcortical, and cerebellar regions while demonstrating promising zero-shot generalization to task fMRI.

### Strengths
- The work pushes boundaries in reconstructing deep brain and full-brain fMRI components from EEG, a problem that is highly relevant for scalable, low-cost brain imaging with broad applications.

- The proposed method integrates pretrained fMRI representations with context-aware EEG encoding, a novel and interesting approach.

- The paper is clearly written and has good presentation

### Weaknesses
- The differences reported in the ablation study are small, making it difficult to assess the significance of the contribution of the parts to the model's performance.

- The spatial correlation loss term is not explicitly derived or quantified in analytical detail.

- The proposed method seems to be mostly functional for cortical regions, while performance on sub-cortical regions is poorly explored, diminishing the actual contribution of the model as an effective mapping between EEG and fmri. 

- Zero-shot evaluation is limited to one auditory task; broader task validations would strengthen claims of generalizability.

- Some hyperparameter decisions (e.g., 5 dataset tokens) and model architecture lack extensive justification or exploration.

### Questions
- In the EEG encoder, the authors use two modules to extract EEG embeddings: a spatio-temporal module and multi-scale spectral transformers. However, the specific contributions of these modules are not clearly assessed. Could you clarify the expected role of each? Is the spatio-temporal module supposed to capture local (fine-grained) features while the multi-scale spectral transformers capture more global or scale-invariant patterns? Do these modules actually complement each other in practice, and is there evidence that both are necessary? It appears that the ablation studies do not test or justify the need for both modules. 

- Regarding context tokens, the authors mention using 5 tokens to describe the dataset, with ablation results in Table 2 showing 5 tokens outperform 1 or 10 tokens. However, it’s unclear what exactly these tokens represent or encode. Are they simple dataset identifiers, or do they capture meaningful experimental conditions or metadata? More detail on the nature and creation of these tokens would clarify their role. 

- For the baseline models, the authors state that the final projection layer was modified to map embeddings to the selected ROIs. Were these baseline models fine-tuned with the new projection layers? Demonstrating that these models were trained to convergence after modification would strengthen the comparison.

- In the main results, the authors report better prediction performance for cortical networks (somatomotor, dorsal attention, salience/ventral attention), consistent with EEG’s cortical focus. However, subcortical and cerebellar predictions have low temporal correlation (~0.25). This raises a key question: if the model performs well for cortical regions already well represented by EEG, what additional value does this EEG-to-fMRI mapping provide? What new insights does extrapolated fMRI from EEG yield beyond EEG alone? This important question remains inadequately addressed.

- Finally, the ablation studies overall seem to weaken the paper. Most differences are modest and may lack statistical significance, and except for reconstruction loss, other terms contribute marginally. Additionally, the model without EEG-fMRI alignment (i.e., no EEG info) still performs within the standard deviation of the full model. Clarification of these findings and their implications for the model’s robustness and design would be appreciated.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes UniEFS, a unified model that reconstructs fMRI signals from EEG. The method combines a pretrained fMRI decoder (learned via masked modeling) with a context-aware EEG encoder that includes demographic and vigilance tokens. The approach aims to handle cross-subject variability and achieve full-brain reconstruction from EEG using a single model.

### Strengths
•	Ambitious and well-motivated approach tackling a challenging multimodal translation problem.
•	The two-stage framework (fMRI pretraining + EEG-fMRI alignment) is conceptually sound and leverages large-scale unpaired data effectively.
•	Comprehensive experiments showing improvement over prior EEG-to-fMRI synthesis models.
•	Context conditioning via metadata (age, sex, vigilance) is innovative and biologically meaningful.

### Weaknesses
•	Potential data leakage concern: the pretrained fMRI decoder is later fine-tuned using data that overlaps with the EEG-fMRI pairs used in the alignment stage. This could inflate performance metrics. The paper should clarify whether fMRI data used for both LoRA finetuning and EEG-to-fMRI translation training step were fully disjoint from evaluation subjects. 
•	The alignment loss seems to have marginal benefit according to the ablation table. The authors should explain whether alternative alignment methods (e.g., contrastive or InfoNCE-style objectives) were tested and why simple MSE was chosen.
•	The model is trained for next-frame prediction, which assumes future fMRI frames depend only on preceding EEG. However, since the EEG window already includes pre-fMRI dynamics, it’s not clear whether the model truly predicts unseen future frames or partially reconstructs signals already reflected in the EEG.

### Questions
1.	Can you clarify whether fMRI data used for decoder pretraining or fine-tuning overlaps with EEG-fMRI training subjects? If so, this could lead to data leakage.
2.	Have you tried contrastive alignment or other objectives (e.g., cosine similarity, CCA, InfoNCE)? If tested, what were the differences in alignment quality or downstream reconstruction?
3.	The ablation shows the alignment loss doesn’t contribute much—can you elaborate on why it might be?
4.	Given the temporal nature of EEG and fMRI, did you test predicting more than one frame ahead, or is the model implicitly reconstructing known signals? Also have to try multiple timescale for forward prediction? Any of them produce a benefit?
5.     The vigilance embedding results are interesting—would be nice to see whether similar gains appear for task-based fMRI.

### Soundness
3

### Presentation
3

### Contribution
3
