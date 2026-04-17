# scPPDM: A Diffusion Model for Single-Cell Drug–Response Prediction

- Decision: Reject
- Scores: 2, 6, 2, 8

## Abstract
This paper introduces the Single-Cell Perturbation Prediction Diffusion Model (scPPDM), the first diffusion-based framework for single-cell drug-response prediction from scRNA-seq data. scPPDM couples two condition channels, pre-perturbation state and drug with dose, in a unified latent space via non-concatenative GD-Attn. During inference, factorized classifier-free guidance exposes two interpretable controls for state preservation and drug-response strength and maps dose to guidance magnitude for tunable intensity. Evaluated on the Tahoe-100M benchmark under two stringent regimes, unseen covariate combinations (UC) and unseen drugs (UD), scPPDM sets new state-of-the-art results across log fold-change recovery, $\Delta$ correlations, explained variance, and DE-overlap. Representative gains include +36.11%/+34.21% on DEG logFC–Spearman/Pearson in UD over the second-best model. This control interface enables transparent what-if analyses and dose tuning, reducing experimental burden while preserving biological specificity.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes scPPDM, a diffusion-based model for predicting single-cell drug responses. The approach operates in a frozen latent space from SCimilarity VAE and introduces two main components. First, pre-perturbation cell state and drug information are encoded separately and combined through Guided Decomposable Attention rather than concatenation. Second, during inference, factorized classifier-free guidance provides two control parameters for state preservation and drug response strength, with the latter modulated by dose through a log-sigmoid mapping. The model is evaluated on Tahoe-100M dataset under unseen covariate combinations and unseen drugs settings, showing improvements over existing baselines including chemCPA, PRnet, STATE, and scGPT across correlation, explained variance, and differential expression overlap metrics.

### Strengths
1.	The paper introduces denoising diffusion models to this problem domain, which represents a novel application even if the underlying techniques are borrowed from computer vision.
2.	The evaluation on Tahoe-100M spanning millions of cells with proper train/test splits and multiple baselines provides substantial empirical evidence, though biological validation remains absent.
3.	The ablations demonstrate that conditioning, non-concatenative injection, structure-aware drug encoding, dual-knob guidance, and dose mapping each contribute to performance improvements.
4.	The separation of baseline state and drug effects through factorized guidance provides a potentially useful interface for dose optimization and counterfactual analysis, though practical utility remains undemonstrated.
5.	The model achieves consistent improvements across multiple metrics including correlation, explained variance, and differential expression overlap on both evaluation regimes.
6.	The appendices provide thorough theoretical background and implementation details with a commitment to release code, supporting reproducibility.

### Weaknesses
1.	Causal framing is unjustified and potentially misleading. Equation 1 approximates observational probability with causal do-operator without explaining when this equivalence holds or citing the relevant identifiability conditions from Pearl's framework. This appears to be notation rather than genuine causal modeling.
2.	Critical claims lack supporting evidence. The assertion that concatenation "amplifies batch noise and harms generalization in early training" is central to the GD-Attn design but presented without any experimental support, ablation, or theoretical justification.
3.	Hold-out strategy is insufficiently described. The paper does not explain what specific cell lines or drugs are held out, whether held-out cell lines share cancer types with training data, or how similar held-out drugs are structurally to training drugs. Without this information, the difficulty of the generalization task cannot be assessed.
4.	Biological validation is entirely absent. There is no validation that predictions align with known drug mechanisms, pathway enrichment analysis, comparison to experimental dose-response curves, or demonstration of novel biological hypotheses. For a science application, correlation metrics alone are insufficient. This problem has been explored intensively. More sophiscated benchmarking should be taken rather than introducing benchmarking settings from scratch (maybe refer to A benchmark for prediction of transcriptomic responses to chemical perturbations across cell types).
5.	SCimilarity VAE fine-tuning procedure is unclear. SCimilarity was trained on healthy tissue data and requires cell type annotations. The paper does not explain how it was fine-tuned for cancer cell lines, whether cell type labels were available or synthesized, or how the latent space properties were validated post fine-tuning.
6.	Differential expression gene selection lacks statistical rigor. Genes are selected by absolute log fold change magnitude without multiple testing correction, significance thresholds, or consideration of biological effect size. The top 200 genes may include many statistically non-significant changes, inflating apparent performance.

### Questions
1.	How do you justify the causal framing in Equation 1? Are you modeling causal relationships via graph? What are the variables, genes, drugs, or both? Under what conditions does observational probability equal the causal distribution under intervention? Are you defining/How do you define an "approximate" causal equality? Is this mainly decorative? This equivalence requires specific assumptions about confounding and exchangeability that are not discussed. 
2.	What is the evidence that concatenation amplifies batch noise? Can you provide ablation experiments, training curves, or gradient analysis supporting this claim? This is central to justifying GD-Attn but currently lacks any empirical or theoretical support.
3.	What are the specifics of the hold-out strategy? Which cell lines and drugs are held out in each split? What is the structural similarity distribution between held-out and training drugs? Do held-out cell lines share cancer subtypes with training data?
4.	How was SCimilarity VAE fine-tuned on cancer cell lines? What labels were used during fine-tuning? How many epochs? What validation was performed to ensure the latent space remained meaningful? Did you compare to training a VAE from scratch on your data?
5.	Can you validate predictions against known biology? For example, do predicted responses to EGFR inhibitors show expected effects on EGFR pathway genes? Can you show pathway enrichment analysis? Do dose-response curves match pharmacological expectations?
6.	Are the top K genes statistically significant differentially expressed genes? Did you apply multiple testing correction? What percentage of your top 200 genes would pass a significance threshold like FDR less than 0.05? How does performance change if you restrict to statistically significant genes only?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces scPPDM, claimed as being the first diffusion model for single-cell drug response prediction. They work in a VAE latent space and condition on two separate channels (pre-perturbation cell state and drug encoded from Morgan fingerprints plus dose) injected via a proposed "Guided Decomposable Attention" mechanism. At inference, factorized classifier-free guidance provides two interpretable knobs for tuning state preservation versus drug effect strength, with dose mapped to guidance magnitude. On the Tahoe-100M benchmark they show improvements over baselines.

### Strengths
* I haven't seen diffusion models applied to single-cell perturbation prediction before. The dual-channel conditioning design feels like a natural but non-obvious way to decompose the problem, and the dose-to-guidance strength mapping is creative. I particularly like how they've adapted classifier-free guidance to have separate "knobs" for state preservation vs drug effect, which seems like it could be really useful
* The experimental work is solid. The Tahoe-100M benchmark is large-scale and appropriate, the UC/UD splits are properly stringent (no data leakage), and the results are consistently strong across many metrics. I appreciate the thoroughness of the ablations. 
* The paper is generally well-written and easy to follow

### Weaknesses
* The dose-to-strength mapping (Equation 16) seems pretty ad-hoc. Why sigmoid of log(1+dose) specifically? You mention linear mapping caused "over-saturation" but don't explain why. How sensitive are results to the choice of s0, α, γ? This feels like it could be a fiddly hyperparameter that needs careful tuning for each new drug.
* Computational costs are completely missing. Training on 100M cells must be expensive. How long does it take? How much memory? How does inference cost compare to baselines? I'm skeptical this is practical for most labs if it requires massive compute resources. Even just a table showing training time and GPU memory would help.
* The dose modeling feels simplistic. Real dose-response curves are often non-monotonic (hormesis, biphasic responses), but your sigmoid mapping (Eq. 16) assumes monotonicity. Have you validated that predicted responses actually follow correct dose-response shapes? Can the model handle dose interpolation (predicting intermediate doses)?

### Questions
* What are the actual computational requirements? Training time, GPU memory, inference speed compared to baselines?
* How sensitive are results to the dose mapping hyperparameters (s0, α, γ)? Also, why sigmoid of log-dose rather than other monotonic functions?
* Have you tested generalization to other perturbation types (CRISPR, genetic) or cell types (primary cells, non-cancer)? If not, what's your intuition about whether the approach would transfer?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces a diffusion based model, scPPDM,  for molecular perturbation prediction. The diffusion model is conditioned on both the initial cell state and the drug + dosage information. Furthermore, they use a guided decomposable attention layer to fuse the control state and the drug information. They evaluate their model on Tahoe dataset and on various metrics and show in most of these metric, scPPDM outperforms the baselines.

### Strengths
- The problem addressed in this paper is rather important and interesting with real world application.
- This is the first diffusion model based paper to solve molecular perturbation task. 
- The are some interesting novelty in the methodology such as GD-Attn, and the dual channel conditioning.

### Weaknesses
I would be happy to reconsider my assessment and improve my score if the authors address the questions and concerns raised below:

- While scPPDM introduces an interpretable diffusion-based conditioning framework, it relies on pseudobulked data which means it models average perturbational responses rather than single-cell variability.
However, the baselines used in this paper such as STATE and chemCPA operate on  single-cell distributions, enabling them to capture heterogeneous or partial responses. It is not clear if the authors changed these models to operate on pseudobulked data instead of single cell or are reporting single cell performance. Either way,  this discrepancy raises concerns about fairness in the comparison.
- There are some key baselines missing in this paper such as BioLORD[1], SAMS-VAE[2], and scDCA[3]. since this model is built on top of scSimilarity, in some sense this resembles the work scDCA which is built on top of scGPT. It would be interesting to compare these two models. 
- The paper was only evaluated on Tahoe datasets while prior works such as Chemcpa used sciplex3 as their main dataset. It would be useful to see how they perform on such datasets as well. 
- Although scPPDM uses a diffusion-based architecture, it is never evaluated for generative quality.
All experiments treat it as a deterministic predictor.
Without demonstrating the benefits of the diffusion process, this architectural choice appears conceptually unjustified relative to simpler supervised baselines.


[1] Piran, Z., Cohen, N., Hoshen, Y. & Nitzan, M. (2024). Disentanglement of single-cell data with BioLORD. Nature Biotechnology

[2] Bereket, M., Karaletsos, T. et al. (2023). Modelling Cellular Perturbations with the Sparse Additive Mechanism Shift Variational Autoencoder. In Proceedings of the 37th Conference on Neural Information Processing Systems (NeurIPS 2023)

[3] Maleki, S., Huetter, J.-C., Chuang, K. V., Scalia, G., Biancalani, T. (2024). Efficient Fine-Tuning of Single-Cell Foundation Models Enables Zero-Shot Molecular Perturbation Prediction.

### Questions
- Is it possible to apply this model to tasks such molecular prediction on novel cell line context? Since the dataset used in this paper (Tahoe) has 50 cell lines, it would be interesting to test this task too. 
- Could you clarify how many genes have been used for the training and how are the authors choosing these genes? Is the same set used in the test?
- scGPT is a single cell foundation model, how did the authors adopt it for the task of molecular perturbation prediction?
- Another interesting metric in this domain is Perturbation Discrimination Score (PDS) Wu et al., 2024. It would be useful if the authors could evaluate scPPDM on this metric as well.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors propose scPPDM, a diffusion model for predicting drug responses at a single-cell resolution. They use a dual-channel conditioning process, where they condition the denosing process on the control gene expression profile and the fingerprint of the SMILES representation of the drug, combined with the dosage. The method is evaluated on the recently released Tahoe-100M dataset in two different settings, unseen drug and unseen cell line.

### Strengths
- The authors represent the first application of denoising diffusion models to single-cell drug-response prediction. The paper presents several non-tirival adaptations to extend diffusion models to scRNA-seq data. This makes it a novel and important contribution to the field.
- The empirical results on the Tahoe-100M dataset are impressive. scPPDM outperforms even recent state-of-the-art methods such as STATE by a considerable margin across most metrics.
- The design choices are well explained, and several ablation experiments are included in the paper to distinguish the contribution of each of them.
- The paper is well-written, easy to follow, and the figures effectively illustrate the method.

### Weaknesses
- The empirical evaluation of scPPDM is only done on the large Tahoe-100M dataset. For most potential users, it may also be interesting to see how the method performs on smaller datasets and what kind of scaling behavior the method exhibits with respect to dataset size.

### Questions
- How does scPPDM scale with dataset size? The dataset size should probably be considered as the number of unique cell line/drug combinations.
- Does scPPDM perform well on datasets other than the  Tahoe-100M?

### Soundness
4

### Presentation
4

### Contribution
4
