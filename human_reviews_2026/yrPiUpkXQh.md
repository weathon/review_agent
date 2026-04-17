# Probabilistic Modeling of Multi-rater Medical Image Segmentation for Diversity and Personalization

- Decision: Reject
- Scores: 8, 2, 2, 6

## Abstract
Medical image segmentation is inherently influenced by data uncertainty, arising from ambiguous boundaries in medical scans and subjective variations among expert annotators. To address this challenge, previous works formulated the multi-rater medical image segmentation task, where multiple experts provide separate annotations for each image. However, existing models are typically constrained to either generating diverse segmentations that lack expert specificity or producing personalized outputs that merely replicate individual annotators. We propose Probabilistic modeling of multi-rater medical image Segmentation (ProSeg) that simultaneously enables both diversification and personalization. Extensive experiments on both the nasopharyngeal carcinoma dataset (NPC) and the lung nodule dataset (LIDC-IDRI) demonstrate that our ProSeg achieves a new state-of-the-art performance, providing segmentation results that are both diverse and expert-personalized.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper presents a probabilistic graphical model, ProSeg, for multi-rater medical image segmentation. The key design lies in introducing two latent variables to separately model expert-specific preferences and data ambiguity, allowing simultaneous generation of diverse and personalized segmentation results. The contribution is clear and well-motivated, and the method achieves consistently superior performance compared to related works on two public multi-rater datasets.

### Strengths
Clear and interesting motivation and contributions.

Better performance than other works.

An end-to-end framework to ensure efficiency.

Good paper written.

### Weaknesses
The task does not have real groundtruth to supervise the diverse segmentation outcomes. It remains unclear how the model guarantees prediction reliability and prevents hallucinated or implausible segmentations during sampling.

Given the rapid advances in autoregressive (AR) and diffusion-based generative segmentation models, it would be important to evaluate how the proposed probabilistic framework performs when integrated with or compared to these stronger generative backbones.

The current design involves two latent variables. It would be valuable to explore the scalability of the probabilistic model with additional latent factors. For example, incorporating random variables to intra-expert uncertainty.

In the current ProSeg formulation, each expert is assumed to be conditionally independent, modeled by its own latent variable. Also, how to model the relationships among different experts? Also, how about modeling this setting into a causal graph model and putting a shared latent condition?

The data ambiguity is modeled by a Gaussian distribution, and the expert priors are Dirichlet distributions. However, there is no clear motivation for adopting the two parameterized distributions. What is the advantage?

This work follows the typical setting, evaluation, datasets as in prior works. The unique contribution should be further highlighted in the main content.

The dataset should be cited with its correct reference. Wu et al., "Dataset, Challenge, and Evaluation for Tumor Segmentation Variability," ACM MM 2024.

### Questions
See the above weaknesses.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes ProSeg, a probabilistic variational framework designed to model both inter-rater diversity and annotator-specific personalization in medical image segmentation. The model introduces two latent variables, τ and Z, to capture annotator preference and data ambiguity, respectively, and optimizes a unified evidence lower bound (ELBO). 
Experiments are conducted on two datasets, LIDC-IDRI and NPC, where the method is compared with several baselines such as Probabilistic U-Net, TAB, and D-Persona.
The authors claim that ProSeg is the first unified probabilistic approach capable of simultaneously capturing diversity and personalization among multiple annotators.
While the topic is meaningful and the writing is generally clear, the paper suffers from fundamental conceptual and methodological flaws that seriously weaken its contributions and undermine the reliability of its results.

### Strengths
1. The paper addresses an important problem in uncertainty-aware medical image segmentation by attempting to unify the modeling of inter-rater diversity and annotator personalization within one probabilistic framework.

2. The probabilistic formulation and ELBO derivation are mathematically consistent, and the structure of the manuscript is clear and easy to follow. 

3. The visualizations and tables are well presented, and the authors compare with several existing multi-rater methods.

### Weaknesses
1. The use of pseudo-annotators in the LIDC-IDRI experiments is not representative. Instead of using real annotator identities, the authors artificially construct four “virtual doctors” by sorting masks according to lesion area. This invalidates the claim that the method learns human-specific annotation styles. What the model effectively learns is merely area-based morphological variation, not true inter-rater differences. As a result, the reported personalization metrics are not meaningful, and the experimental foundation of the paper becomes questionable.

2. The theoretical formulation relies on several unrealistic independence assumptions such as p(Y∣X,R,τ,Z)≈p(Y∣τ,Z) and a factorized prior over latent variables. None of these assumptions are justified empirically, nor is there ablation analysis to confirm that they are harmless. 

3. The variational family and prior selections are also rather arbitrary: the authors fix Z as a standard Gaussian and τ as a symmetric Dirichlet distribution with α₀ = 1, yet provide no heuristic or sensitivity analysis to justify these choices. It is unclear why the symmetric prior is reasonable for modeling annotator heterogeneity, how different α values would influence the learned distribution, or how the model behaves under varying numbers of raters or classes. Given that the paper explicitly aims to model individual annotator differences, such prior rigidity and the absence of sensitivity evaluation make the entire probabilistic formulation insufficiently validated.

4. The paper lacks any efficiency analysis—no training time, GPU memory, or parameters, inference time are reported. This omission makes it impossible to judge the practical scalability of the framework.

5. The experimental validation is narrow and does not demonstrate generalization. Only two relatively small datasets are used (LIDC-IDRI and NPC). There is no evaluation on other publicly available multi-rater datasets (e.g., RIGA[a], QUBIQ[b], and SUN-SEG[c]), no cross-dataset or cross-modality experiments, and no statistical significance reporting.

6. The presentation raise further concerns. The paper exhibits poor adherence to academic standards: the reference list contains numerous formatting inconsistencies, missing venues and publishers, and the authors do not provide a reproducibility statement or code prior to acceptance. These presentation and transparency issues further reduce the credibility of the work.

[a] Almazroa, Ahmed, et al. "Agreement among ophthalmologists in marking the optic disc and optic cup in fundus images." International ophthalmology 37.3 (2017): 701-717.

[b] Li, Hongwei Bran, et al. "Qubiq: Uncertainty quantification for biomedical image segmentation challenge." arXiv preprint arXiv:2405.18435 (2024).

[c] Ji, Ge-Peng, et al. "Video polyp segmentation: A deep learning perspective." Machine Intelligence Research 19.6 (2022): 531-549.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

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
The paper tackles multi‑rater medical image segmentation where uncertainty arises both from ambiguous image boundaries and inter‑observer variability. It proposes **ProSeg**, a unified probabilistic graphical model with two latent variables: Z (Gaussian prior) to capture image ambiguity and $\tau$ (Dirichlet prior) to capture annotator preference. The factorization in Fig. 2(d) and Eq. (3)–(6) leads to the training of five neural components (encoders/decoders for x, a class‑embedding $q(\tau \ | \ R)$, a classifier $p(R\ | \tau)$, and a single segmentation head $p(y_r \mid \tau,z))$, shown in Fig. 3. The objective maximizes an ELBO whose negative log‑likelihood decomposes into image reconstruction, annotator classification, and segmentation losses plus two KL terms. At inference, personalized outputs are sampled via $q(\tau | \ r)\,q(Z \mid x)$ (Eq. 10), while diverse outputs come from prior sampling $\tau^\*\sim \text{Dir}(\alpha)$ combined with $q(Z \mid x)$ (Eq. 11).

### Strengths
1. The PGM separates image ambiguity ($Z$) from annotator preference ($\tau$), clarifying how the model can produce both diverse and expert‑specific segmentations.
2. ProSeg is competitive across multiple metrics on two benchmarks (Tables 1 & 3), notably improving $D_{\text{soft}}$, $D_{\max}$, and $D_{\text{match}}$ on NPC where annotator preferences vary more.  
3.	Beyond reporting GED, the paper decomposes it into $d_{pp}$, $d_{pa}$, and $d_{aa}$ (Table 9) and provides distance‑to‑random‑experts plots (Fig. 7), making the diversity–reliability trade‑offs interpretable.
5.t‑SNE of samples from p(\tau\!\mid r) shows separable annotator clusters and dataset differences (Fig. 6).

### Weaknesses
1. Eq. (6) models a *single* image as generated by the product of multiple conditionals “since one image corresponds to multiple understandings of experts.” This is unusual: $x$ is not multi‑valued and such a factorization can overweight the reconstruction term by effectively counting the same $x$ $N$ times ( Fig. 3 & Eq. 6). A more standard choice would be a single $p(x \mid z)$.
2. Posterior for $\tau$ depends only on $R$. The variational family is $q(\tau \mid R)$ (Fig. 3; Eq. 7–9), so per‑case preference shifts of the *same* expert must be absorbed by $Z$. This design may limit true personalization when an expert’s style changes with the case; a $q(\tau \mid R,Y)$ or $q(\tau \mid R,x,Y)$ might capture this better.
3. For diverse outputs, $\tau^\* \sim \mathrm{Dir}(\alpha)$ is sampled and then *classified* into a discrete expert class $i$ before prediction (Eq. 11). This can collapse diversity to a few expert prototypes instead of leveraging the continuous $\tau$ space directly.
4. To construct the *virtual* annotator using LIDC dataset, ranking based on the area is not a good choice. This may dampen real inter‑observer variation and favor methods that learn stable annotator embeddings. A comparison without re‑assignment would strengthen claims. 
5. The baseline choices are not optimal, the simplest way to model multi-annotator uncertainty is to construct an ensemble model, which can both satisfy the diversity and personalization goal. Although multiple unets are compared, the simple method, such as ensemble model is not compared.

### Questions
1. Please refer to the weaknesses above. 

2. Could you report results without the image reconstruction term and/or with alternative priors for $Z$, including calibration analyses (e.g., ECE/Brier) to show how $Z$ tracks segmentation uncertainty?

3. What exact tests were used, at what aggregation level (slice vs. subject), and did you apply any multiple‑comparison correction across metrics/methods?

4. Given ProSeg’s lower $d_{pa}$ but higher $\mathrm{GED}$ than D‑Persona (I) on NPC, can you clarify which regime is preferable for clinical decision‑making and whether user studies support prioritizing $D_{\text{soft}}$/$d_{pa}$ over aggregate $\mathrm{GED}$?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
ProSeg is a probabilistic framework for multi-rater medical image segmentation that addresses data uncertainty stemming from ambiguous anatomical boundaries and inter-observer variability. It introduces two latent variables to explicitly model expert-specific annotation preferences and image-level boundary ambiguity. By leveraging variational inference, the model learns conditional probabilistic distributions over these latent factors, enabling the generation of segmentation outputs that are simultaneously diverse and tailored to individual annotators. Evaluated on the nasopharyngeal carcinoma (NPC) and LIDC-IDRI lung nodule datasets, ProSeg achieves state-of-the-art performance, demonstrating its ability to produce both personalized and varied segmentation results.

### Strengths
1.	This work presents with clear and meaningful motivation: addressing the critical gap between diversity and personalization in multi-rater segmentation, rooted in clinical uncertainty and inter-observer variability.
2.	The framework design is interesting. It Introduces a unified probabilistic framework with two disentangled latent variables (τ for expert preference, Z for boundary ambiguity) deduced from variational inference.
3.	Achieves state-of-the-art results on both NPC and LIDC-IDRI datasets across diversity (GED, Dsoft) and personalization (Dmax, Dmatch) metrics.

### Weaknesses
1.	Limited architectural novelty and generalizability: The model relies primarily on a standard CNN-based (U-Net) backbone and is trained on datasets where annotations are either from real radiologists (NPC) or artificially constructed virtual experts (LIDC-IDRI). This setup may not reflect the complexity and heterogeneity of real-world clinical environments, raising concerns about practical generalizability.
2.	Lack of architectural diversity in validation: The proposed framework is exclusively implemented with CNNs. To demonstrate its broader applicability, experiments with other mainstream architectures, such as ViTs, should be included to verify that the probabilistic formulation is architecture-agnostic and not tied to CNN-specific inductive biases.

### Questions
1.	What are the practical clinical applications of multi-rater medical image segmentation? Specifically, in which real-world scenarios (e.g., treatment planning, consensus building, or quality assurance) is modeling inter-observer variability essential, and how does capturing both diversity and personalization improve clinical utility?
2.	How well does ProSeg generalize to out-of-domain or unseen datasets?
Can the probabilistic framework adapt to new imaging modalities, anatomical regions, or previously unobserved expert styles without retraining or fine-tuning?

### Soundness
3

### Presentation
3

### Contribution
3
