# Survival VAE: Robust Local Explanations via Double-Pass Risk Consistency

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 6

## Abstract
In the era of advanced machine learning, the need to explain models has grown significantly. One particular domain, survival analysis, has benefited from the rise of deep learning but has lagged behind in the development of methods for explaining risk and survival models. 
Only a few works have adapted explainable AI methods, such as LIME and SHAP, to survival analysis. Despite these efforts, explaining survival models remains challenging given the complex nature of the data used for survival predictions and the presence of censoring.
In this work, we propose a local feature identification method that inherently operates on the instance ordering induced by event and censoring times. It enables faithful, per-sample feature importance by identifying which reconstructed input features preserve consistency in predicted survival risk across a double-pass through the variational autoencoder.
Empirical results on the large multi-cohort dataset from The Cancer Genome Atlas demonstrate superior quantitative performance of our method. Qualitatively, analysis of mask weights highlights the biological relevance of the feature selection process.
This information can be used to identify new diagnostic markers and treatment targets for cancer patients.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a novel method, DP-SurVAE, for generating local feature explanations for survival analysis models. The core idea is to utilize a VAE and a novel "double-pass" training objective designed to ensure that predicted survival risk remains consistent even after applying a sparse feature mask to the reconstructed samples. The authors evaluate their method on the large-scale, high-dimensional TCGA dataset, demonstrating that features selected by DP-SurVAE achieve superior quantitative performance and show biological relevance.

### Strengths
$\textbf{Importance of the Problem:}$ The paper tackles a critical problem in medical AI: providing reliable and faithful explanations for complex survival analysis models, especially when dealing with censored data. This is an active and impactful area of research.

$\textbf{Strong Empirical Results:}$ The paper conducts a comprehensive empirical evaluation on two large-scale, high-dimensional datasets from TCGA. The results show that a downstream model trained on the feature subset selected by DP-SurVAE consistently and significantly outperforms baselines.

$\textbf{Biological Relevance Validation:}$ A major highlight of this work is the validation of the selected features for biological relevance. Through Gene Ontology (GO) enrichment analysis and correlation analysis with known cancer genes, the authors provide strong qualitative evidence for the model's explanatory power, suggesting it not only performs well in prediction but also uncovers meaningful biomarkers.

### Weaknesses
$\textbf{Lack of Theoretical Justification for the Core Loss Function:}$ This is the most significant weakness of the paper. The proposed "Double-Pass Log-Partial Likelihood" (DPLPL) loss is the central innovation. However, the paper provides no theoretical proof or deep explanation as to why this specific formulation is the "correct" or "principled" way to enforce risk consistency. This leaves the theoretical foundation of the core contribution feeling weak.

$\textbf{Excessive Model Complexity and Lack of Ablation Studies:}$ The DP-SurVAE architecture is quite complex, integrating a VAE, an attention-masking module, and a double-pass encoding flow. The authors do not sufficiently justify this complexity. For instance, it is unclear why the mask is applied to the reconstructed sample instead of directly to the original input. The paper lacks ablation studies on these alternatives, making it difficult to assess the actual contribution of each component.

$\textbf{Questions Regarding the Evaluation Method:}$ The evaluation primarily focuses on using the selected features to retrain a downstream model and comparing its predictive performance. While this reflects feature importance, it does not directly measure the faithfulness of the explanation—that is, how well the explanation represents the model's own decision-making process. Since DP-SurVAE is an end-to-end model and the baselines are post-hoc, the comparison might inherently favor DP-SurVAE.

### Questions
$\textbf{1. }$ Could you provide a more rigorous theoretical motivation for the specific form of the $L_{dplpl}$ loss function? Why is treating the original and masked samples as separate entries in an extended risk set the optimal way to enforce consistency?

$\textbf{2. }$  Theorem 1 states that the masking operation degrades risk consistency, thus motivating a training objective to correct for it. Beyond this intuitive observation, what is the primary theoretical insight provided by this theorem? Does it justify $L_{dplpl}$ as the unique or optimal solution to this problem?

$\textbf{3. }$ Have you considered simpler architectures, such as applying the mask directly to the input x and then enforcing consistency between the latent means or risk predictions of the original and masked samples via a KL divergence or L2 loss? How does such a simplified model perform compared to DP-SurVAE?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes DP-SurVAE, a method for local feature importance in survival analysis 
that combines a VAE with masking and a novel "double-pass" objective to ensure risk 
consistency when identifying important features for predicting patient survival. The method 
operates on high-dimensional gene expression data from The Cancer Genome Atlas (TCGA), 
comparing against LIME, SHAP, SurvLIME, and SurvSHAP. The authors evaluate on 37 disease 
cohorts using C-index and provide biological validation via Gene Ontology enrichment and 
cancer gene analysis. While the paper addresses an important problem, it suffers from 
significant clarity issues and vague technical exposition

### Strengths
1. **Important and timely problem**: Explainability in survival analysis is genuinely 
   underexplored. Accounting for censoring when explaining predictions is an interesting 
   problem domain.

2. **Large-scale empirical evaluation**: Testing on 37 TCGA cohorts (19 mRNA, 18 miRNA) 
   demonstrates consistency across multiple cancer types.

### Weaknesses
The paper combines existing techniques without sufficient novelty for a top-tier venue:

**VAE for feature importance is well-established**
- Using VAEs for interpretability/feature importance has been extensively studied (e.g., 
  β-VAE for disentanglement, attention-based VAE masking).
- The main contribution—masking on VAE reconstructions—is a straightforward extension of 
  existing attention-masking mechanisms (not novel).
- Reference: CompSense, IntGradCAM, attention-based VAE masking are standard techniques 
  that the paper does not clearly differentiate from.

**Double-pass might be incremental contribution**

- The "double-pass" mechanism is simply: (1) encode original, (2) mask reconstruction, 
  (3) re-encode masked version.
- This is a minor engineering contribution. It's unclear why this specific procedure is 
  necessary vs. alternative consistency enforcement mechanisms (e.g., auxiliary losses, 
  constraints on latent space).


**Comparison to SurvLIME/SurvSHAP is unfair**

- SurvLIME/SurvSHAP are post-hoc explainers for arbitrary survival models.
- DP-SurVAE is a full model that jointly learns predictions + explanations.
- These are not comparable; a fair comparison would be DP-SurVAE vs. other jointly-learned 
  interpretable survival models (e.g., neural additive models, attention-based models).
- The paper excludes CoxNAM (the only jointly-learned baseline) due to "impracticality in 
  high dimensions"—but this is exactly where interpretability is needed most.

**What exactly is novel?**

- Combining VAE + masking + Cox loss: incremental engineering.
- Double-pass consistency: engineering solution, not conceptual novelty.
- Biological validation: application validation, not methodological novelty.


**The paper relies almost exclusively on C-index for evaluation, which is fundamentally 
insufficient**

- C-index only measures ranking agreement between predicted risk and actual survival times.
- It says nothing about:
  - Calibration (are predicted probabilities accurate?)
  - Discrimination at specific time points 
  - Sensitivity/specificity tradeoffs
  - Ability to identify high-risk vs. low-risk patients
  

**Missing standard survival analysis metrics**

- **Time-dependent AUC (AUC(t)):** Evaluates ranking at specific time horizons (e.g., 1-year, 
  5-year survival). Critical for clinical applications. Missing.
- **Brier score:** Measures calibration, crucial for predicting actual survival probabilities. 
  Missing.
- **Integrated Brier Score (IBS):** Integrates calibration error over the entire follow-up 
  period. Standard in survival analysis literature. Missing.



**The method is not evaluated for its core purpose: explainability**

- The paper claims to provide "faithful, per-sample feature importance."

- **No evaluation of explanation quality:**

  - Do selected features have causal effects on survival?
  - Are explanations consistent across similar samples?
  
- Only C-index (predictive performance) is measured. Explanation quality is never evaluated.

**Theorem 1 derivation is informal and circular**

- The theorem doesn't *prove* consistency is maintained after training. It merely states that 
  "optimizing for consistency enforces consistency"—this is tautological.
- No bound on how loose δ(α) becomes with aggressive masking (e.g., α=0.99).
- No proof that training L_dplpl actually achieves |h(μ_x̂̂) - h(μ_x)| ≤ ε(α).


**Missing Ablation: Loss component contributions**
- What is the contribution of each term in L_total?

**Missing Ablation : Double-pass necessity**
- How much does the double-pass contribute vs. single-pass?

**Unclear VAE formulation**

- Section 4.1 introduces ELBO with z_x sampling, but Section 4.2 uses μ_x (the mean) for 
  risk prediction.
- Why not use z_x? Deterministic use of μ_x removes the stochasticity that the ELBO is 
  meant to regulate.
- **Current:** No justification. Appears to be a design choice, not principled.
- **Needed:** Explain why μ_x instead of z_x. Ablation comparing both.

### Questions
1. **Novelty**: What exactly is novel beyond combining VAE masking + Cox loss? How does 
   double-pass differ from simply adding a consistency auxiliary loss?

2. **Theorem 1**: Can you formally define "risk consistency"? Provide explicit bounds on 
   |h(μ_x̂̂) - h(μ_x)| after training?

3. **Extended risk set**: How does Equation (5) maintain Cox PH assumptions? Samples are 
   paired (x, x̂̂), not independent. Proof?

4. **C-index only**: Why not report AUC(t), Brier score, calibration plots? These are standard.

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
3

### Summary
This paper proposes Survival-VAE, a novel framework that combines VAEs with survival analysis to model heterogeneous hazard distributions across subpopulations, aiming to build more explainable and individualized survival models. Following standard VAEs, 
Survival-VAE encodes input covariates into a latent distribution and then decodes the latent distribution back to the input space by optimizing the ELBO objective. To model the individial risk, the authors add a learnable hazard function $h$ in the latent space that takes as input the latent features to predict risk by optimizing the partial likelihood. An additional masking objective is introduced to enable sample-specific feature selection. Finally, the authors propose a double-pass risk prediction by feeding the reconstructed input covariates to the VAE encoder and risk prediction function, which is a reminiscent of the IntroVAE.

### Strengths
- The exploration of generative models (in contrast to discriminative models) for survival prediction is interesting and sound. The proposed double-pass prediction can further regularize the learned latent distributions. 
- The paper is theoretically justified. 
- Experimental results on miRNA cohorts demonstrate the effectiveness of the proposed method compared to strong baselines.
- The paper is easy to follow.

### Weaknesses
- The main focus of this paper is on developing explainable survival models. However, tree-based survival models (such as Survival Tree [3]), which are inherently interpretable. There are also some works that combine traditional survival tree and deep neural works (see e.g. [4]). However, this family of works is neither discussed nor included in the comparative analysis. 
- The Deep Survival Machine [1] is also a generative model for survival prediction. It would be interesting to see the comparison with this method. 
- Comparison to other survival models, such as discrete-time survival models (e.g., deephit [2]) is encouraged. 
- Ablations of different loss components are missing. 
- It is also meaningful to evaluate on standard survival benchmarks, e.g., METABRIC, SUPPORT.

[1] Deep Survival Machines: Fully Parametric Survival Regression and Representation Learning for Censored Data with Competing Risk

[2] DeepHit: A Deep Learning Approach to Survival Analysis With Competing Risks

[3] Tree-structured survival analysis

[4] SurvReLU: Inherently Interpretable Survival Analysis via Deep ReLU Networks

### Questions
- While Survival-VAE learns meaningful clusters, the mapping between latent factors and clinical variables is not deeply analyzed—making interpretability somewhat superficial.
- Can the authors also provide evaluations using Integrated Brier Score (IBS).

### Soundness
3

### Presentation
3

### Contribution
3
