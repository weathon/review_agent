# CLEF: Clinically-Guided Contrastive Learning for Electrocardiogram Foundation Models

- Decision: Reject
- Scores: 2, 6, 6

## Abstract
The electrocardiogram (ECG) is a key diagnostic tool in cardiovascular health. Single-lead ECG recording is integrated into both clinical-grade and consumer wearables. While self-supervised pretraining of foundation models on unlabeled ECGs improves diagnostic performance, existing approaches do not incorporate domain knowledge from clinical metadata. We introduce a novel contrastive approach that utilizes an established clinical risk score to adaptively weight negative pairs: clinically-guided contrastive learning. It aligns the similarities of ECG embeddings to clinically meaningful differences between subjects, with an explicit mechanism to handle missing metadata. Using 12-lead ECGs from 161K patients in MIMIC-IV dataset, we pretrain single-lead ECG foundation models at three scales, collectively called CLEF, using only routinely-collected metadata without requiring per-sample ECG annotations. We evaluate CLEF on $18$ clinical classification and regression tasks across $7$ held-out datasets, and benchmark against $5$ foundation model baselines and $3$ self-supervised algorithms. When pretrained on 12-lead ECG data and tested on lead-I data, CLEF outperforms self-supervised foundation model baselines: the medium-sized CLEF achieves average AUROC improvements of at least $2.6\\%$ in classification and average reductions in MAEs of at least $3.2\\%$ in regression. Comparing with existing self-supervised learning algorithms, CLEF improves the average AUROC by at least $1.8\\%$. Moreover, when pretrained only on lead-I data for classification tasks, CLEF performs comparably to the state-of-the-art ECGFounder, which has been trained in a supervised manner. Overall, CLEF allows more accurate and scalable single-lead ECG analysis, advancing remote health monitoring. We will publish our code and pretrained CLEF models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces CLEF (Clinically-Guided Contrastive Learning Framework), a self-supervised approach that integrates clinical risk scores into contrastive pretraining for ECG representation learning.
By weighting negative pairs and aligning embeddings with clinical dissimilarities, CLEF aims to encode clinically meaningful relationships within the latent space.
Extensive experiments across multiple ECG datasets demonstrate robust and consistent improvements over existing contrastive learning baselines.

### Strengths
1. Comprehensive and rigorous experimentation — The study presents extensive evaluations and ablation studies across diverse datasets and clinical tasks, clearly supporting the robustness of the proposed approach.

2. Clarity and strong organization — The methodology and motivation are well articulated with intuitive explanations and clear visualizations, making the paper easy to follow even for non-domain experts.

### Weaknesses
1. Lack of representation-level validation — Although the authors claim that CLEF adjusts contrastive representations based on clinical risk, the paper lacks direct empirical evidence demonstrating how this modification affects the underlying representation geometry. No analyses such as t-SNE or UMAP visualization, pairwise similarity distributions, or intra/inter-cluster distance comparisons are provided. All evaluations focus solely on downstream performance metrics, leaving the clinically guided nature of the learned representation as a theoretical assertion rather than an empirically verified outcome. Moreover, since CLEF introduces its clinical weighting only through additional loss terms atop a conventional contrastive backbone, potential representational shifts are likely to occur at the pairwise similarity level rather than in global embedding structure, making the claimed improvement difficult to confirm.

2. Limited clinical expressiveness due to single-lead input — The restriction to a single lead (Lead I) imposes inherent limitations on the model’s physiological depth. By excluding spatial context such as precordial lead variations and vectorcardiographic relationships, the model captures only waveform-level abstractions and neglects multi-lead correlations crucial for ischemic or regional cardiac representation. Consequently, the learned representation remains shallow in clinical richness despite its medical motivation.

### Questions
It would be valuable to visualize or quantify how the proposed clinical weighting actually reshapes the learned representation space, to provide more concrete evidence of the claimed effect.

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
4

### Summary
**Summary**

This paper proposes a novel framework called CLEF (Clinically-Guided Contrastive Learning) for electrocardiogram (ECG) foundation models. Its core innovation lies in overcoming the limitations of conventional self-supervised pretraining methods that fail to incorporate clinical domain knowledge and treat all negative pairs equally. CLEF dynamically links the weighting of negative pairs in contrastive learning to the clinical dissimilarity between subjects—measured via an established clinical risk score. This allows the model to align the ECG embedding space with clinically meaningful differences, performing stronger repulsion for pairs with high risk dissimilarity and weaker repulsion for clinically similar pairs. The framework incorporates a *negative weighting loss ($\mathcal{L}^w$)* and a *dissimilarity alignment loss ($\mathcal{L}^d$)*, along with a mechanism for missing metadata, to optimize the model for 18 downstream clinical classification and regression tasks.

### Strengths
- **Problem orientation is concrete and clinically motivated.** The work targets label scarcity and weak cross-device/population generalization in single-lead ECG, injecting domain knowledge via routinely captured clinical metadata rather than dense manual labels.

- **Method is technically neat and conceptually coherent.** Clinical dissimilarity is converted into a training signal through risk-aware negative reweighting, paired with a distance-alignment term, while missing metadata are modeled explicitly instead of being discarded.

- **Pretraining is scalable and evaluation is broad.** Models are trained on large ECG corpora and assessed across diverse datasets and tasks (classification and regression), including cross-dataset transfer, strengthening external validity.

- **Performance gains are consistent against strong baselines.** The approach surpasses competitive self-supervised and supervised methods on standard metrics, with improvements that persist across tasks, splits, and evaluation settings.

- **Robustness is carefully documented.** Multiple random seeds with confidence intervals and component-wise ablations delineate the contribution of each design choice and indicate stable training behavior.

### Weaknesses
- **Teacher–task mismatch (SCORE2 for immediate diagnostics).** SCORE2 supervises with a 10-year cardiovascular risk (prognosis) derived from demographics and vitals, yet the paper applies it to pretrain models for short-horizon diagnostic tasks (e.g., arrhythmia/rhythm/beat classification) without justifying why a long-term risk surrogate is an appropriate teacher for immediate signal decisions. A minimally necessary control is to compare against an *acute* risk or diagnosis-proximal target.  

  

- **Unproven hyperparameter choices on ECG data.** The paper defines the minimum-distance parameter \alpha and the temperature \tau, and fixes the final objective to L=L_w+L_d, but provides no ECG-domain sensitivity for \alpha, \tau, or the 1:1 weighting—pointing instead to an image-domain proxy (CIFAR-100) to justify loss composition. This leaves robustness and optimality on ECG tasks unsubstantiated.    

  

- **Counter-intuitive missing-metadata multiplier** M_{ik}**.** By construction, M_{ik}=\exp\!\big(-\frac{A-m_i}{A}\times\frac{A-m_k}{A}\big) downweights pairs with more complete metadata and increases weights as metadata go missing—an opaque choice with no principled motivation given, despite being a core mechanism.  

  

- **Missing-metadata handling shows mixed efficacy.** The ablation on Table 5 reveals that removing the missing-metadata mechanism  sometimes improve results, indicating the component is not uniformly beneficial and may be suboptimal for certain tasks/datasets. This contradicts a claim of universal benefit and warrants per-task analysis. 

  

- **Weighted-InfoNCE can suffer batch-composition dominance.** Because negatives are reweighted inside the denominator of the contrastive loss, large-\Delta r pairs can dominate gradients, risking reduced representation uniformity and batch-effect brittleness—especially with small batches or skewed risk distributions. The paper does not probe these interactions.

### Questions
1. What is the empirical correlation between SCORE2 differences and the target downstream tasks (e.g., arrhythmia classification, blood pressure prediction)? This would validate the core premise.

2. For missing metadata handling, why not use imputation or domain adaptation instead of multiplicative weighting? Have alternatives been explored?

3. Why does CLEF-L underperform CLEF-M consistently? Is this due to optimization dynamics specific to large models on ECG?

4. In Table S13, lead-specific pretraining outperforms random-lead pretraining (2% avg gain). How do you reconcile this with the claimed advantage of random lead selection?

5. For ECGFounder comparison: it uses 150 diagnostic categories during supervision. How much of the performance gap is attributable to labeled data vs. method differences?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents CLEF (Clinically-Guided Contrastive Learning), a self-supervised framework for ECG foundation models that integrates clinical risk scores into the contrastive pretraining process. The key idea is to move beyond traditional binary notions of similarity/dissimilarity and use continuous, clinically meaningful dissimilarities derived from metadata (e.g., SCORE2 cardiovascular risk) to guide the representation learning. The method introduces two loss components: a weighted contrastive loss (Lw), which adaptively weights negative pairs based on clinical risk and missing metadata, and a dissimilarity alignment loss (Ld), which aligns embedding similarity with clinical risk distance. CLEF is pretrained on 161K 12-lead ECGs from MIMIC-IV, and evaluated across 18 tasks spanning 7 datasets. The results show strong improvements over both self-supervised and foundation model baselines (up to +3.1% AUROC), and competitive performance with the supervised ECGFounder model.

### Strengths
The idea of weighting negative pairs using risk score–derived dissimilarities is elegant and well-motivated. It directly connects the latent geometry of embeddings to real-world clinical semantics, improving both interpretability and utility.
Strong empirical evaluation and broad benchmarking.
CLEF achieves strong performance even when pretrained on 12-lead data but fine-tuned on single-lead (lead I/II) tasks, suggesting it can effectively bridge clinical and consumer-grade ECG domains — a valuable property for real-world applications.
The framework (Figure 1) is well-illustrated, and the mathematical formulation of each component (Eqs. 5–7) is consistent and interpretable. The model’s connection to clinical reasoning adds conceptual clarity often missing in foundation model research.

### Weaknesses
While SCORE2 is a standard cardiovascular risk score, it is not necessarily the optimal or most generalizable measure for all datasets or populations (especially non-European cohorts). 
Since metadata such as age, gender, and blood pressure are correlated with many downstream labels, the inclusion of such information during pretraining might inadvertently leak label information. 
Although the weighting matrix W = D⊙M is central to the approach, the paper does not deeply explore sensitivity to α (Eq. 5) or the relative contribution of Lw vs Ld. This makes it difficult to assess how robust the improvement is to hyperparameter tuning.
The authors claim CLEF embeddings reflect “clinically meaningful geometry,” but there is no direct visualization or correlation analysis between latent distances and clinical outcomes. A t-SNE or UMAP projection aligned with risk scores would be valuable.
Computational cost and scalability discussion is missing.

### Questions
Could this clinically-guided weighting be applied to other biosignals (EEG, PPG, or medical imaging) with analogous metadata?
Did you test other risk scoring systems beyond SCORE2? For example, Framingham or ASCVD risk models — to verify if the method’s gains are score-agnostic.
What is the clinical interpretability of embeddings?
Have you examined whether clusters in the latent space correspond to specific cardiac pathologies or risk strata?
Does the model’s advantage persist under domain shift?
For example, does pretraining on MIMIC-IV generalize equally well to wearable ECG datasets with different sampling rates and demographic distributions?

### Soundness
4

### Presentation
4

### Contribution
4
