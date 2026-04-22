# Causality-Inspired Robustness for Nonlinear Models via Representation Learning

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Distributional robustness is a central goal of prediction algorithms due to the prevalent distribution shifts in real-world data. The prediction model aims to minimize the worst-case risk among a class of distributions, a.k.a., an uncertainty set. Causality provides a modeling framework with a rigorous robustness guarantee in the above sense, where the uncertainty set is data-driven rather than pre-specified as in traditional distributional robustness optimization. However, current causality-inspired robustness methods possess finite-radius robustness guarantees only in the linear settings, where the causal relationships among the covariates and the response are linear. In this work, we propose a nonlinear method under a causal framework by incorporating recent developments in identifiable representation learning and establish a distributional robustness guarantee. To our best knowledge, this is the first causality-inspired robustness method with such a finite-radius robustness guarantee in nonlinear settings. Empirical validation of the theoretical findings is conducted on both synthetic data and real-world single-cell data, also illustrating that finite-radius robustness is crucial.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes a two-step method for distributionally robust predictors called CIRRL, which is designed for settings in which relationships among variables can be non-linear. CIRLL can be seen as a combination of two existing methods: 
- 1) the Distributional Principal Autoencoder (DPA) by Shen & Meinshausen (2024) which learns identifiable latent representations (enhanced with regularisation term that pushes the learned representations to follow a mixture of Gaussians representing training environments) and 
- 2) the DRIG method from Shen et al. (2023) that learns a linear head on the so learned latent representations.

The authors give a theoretical result about the optimality of the so learned function mapping $X$ to $Y$ (a concatenation of the encoder from 1. and the linear head from 2.): under some assumptions (including a data generating process following a directed acyclic causal graph), it is learning the $L^2$ function achieving the lowest worst case out-of-distribution MSE. 

The effectiveness of CIRLL is tested on one synthetic dataset, the ICU dataset from MIMIC-III, and a large single-cell RNA sequencing dataset. In comparison to its baselines (ERM, IRM, VREX, ARM, DRIG) it achieves superior results in most cases, in particular for the word case MSE among test environments on the single-cell data (but not so much on the ICU data in comparison to DRIG).

### Strengths
- The paper proposes a method to extend DRIG to non-linear relationships between variables
- It provides theoretical optimality guarantees about the worst case robustness of the predictor learned with the proposed method
- Empirically, the method outperforms existing methods and its linear counterpart (DRIG) noticeably on the worst ood MSE on the single-cell data experiment, showing that the non-linear extension of DRIG is relevant and necessary in some practical applications

### Weaknesses
- The biggest weakness of the paper is the experiments section (detailed in bullets below)
- The experiments section is noticeably short in comparison to the rest of the paper. As a result, a lot of important content about experiments is missing (e.g. experimental setup incl. description of datasets used, choice and ablation of hyperparameters, ablation of loss term $L_g$). As a bare minimum, their position in the Appendix should be referenced in the main part, but ideally they are already summarised in the main part. Suggestions for shortening would be introduction and related work (I have never seen a related work section that is longer than the experiment section).
- The benefit of the proposed additional loss term $L_g$ is not clear. In its ablation in Figure 7 of the Appendix, the blue curve for $\alpha =0$ (corresponding to omitting it) is not shown for the test loss of the single-cell data and the one of its nearest neighbour ($\alpha = 0.001$) is not significantly dissimilar in performance from the ones of higher values for $\alpha$. 
- Given that out of the 2 real life datasets, the proposed method noticeably outperforms its purely linear component DRIG only on one of the two datasets, while being on par on the other, it would be interesting to have some discussion about this that tries to explain this at least or ideally an analysis that adds more depth to the fundamental differences in the data that explain this difference in relative performance.
- More guidance for practitioners would be useful - e.g. is there a way to verify if assumption 1 holds, how to choose hyperparameters etc. 

Minor: 
- I think the paper could benefit from an illustrative example of the causal graph assumed in Eqn. (2) and (3) in Section 3.1. E.g. ‘X_1 being smoking, X_2 being BMI, Y being …’. 
- At times, key sources are missing: e.g. in line 249 after ‘a vector c’, line 170 after ‘if the graph is acyclic’, line 110 after ‘finite-sample regimes’, line 65 after ‘multi-source heterogeneity’, as well as citing the authors of the datasets used in Section 4.  
- More a comment than a weakness: given that the authors mention adversarial perturbations in line 30 as example, and design a method to make predictors robust to worst case shifts, an experiment on adversarial robustness would have been very interesting! 
- Lines 310 and 321 in sum subscripts: typo $\mathbb{E} \to \mathcal{E}$
- Table 1 needs to be formatted (centering ICU and Sincle-cell data, inserting vertical lines as separators between blocks)

### Questions
- Line 208: ‘The original DPA ensures that the decoder produces reconstructions $\hat{X}$ that follow the same distribution as the original data $X$,…’ -> why is this not the case for a standard auto encoder? Why is the DPA needed?
- Line 335: what are the true latents mentioned here? $\phi^*(X)$?
- Line 45: is it the really the invariant features that remain stable or rather their relationship to each other and to the target?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a two-step method, CIRRL, to achieve causality-inspired, finite-radius distributional robustness in nonlinear settings. It first uses a modified autoencoder to learn a latent representation where a linear structural causal model holds, identifying it up to an affine transformation. It then applies a robust linear regressor (DRIG) in this latent space, providing the first method (to the authors' knowledge) with a finite-radius robustness guarantee for this nonlinear context.

### Strengths
- The paper studies a significant and well-motivated problem: extending causality-inspired, finite-radius robustness guarantees from purely linear models to the nonlinear settings common in machine learning.
  
- The paper cleverly synthesizes two advanced lines of research: identifiable representation learning and causality-inspired DRO. The combination is non-trivial and well-justified.
  
- The method is validated on synthetic data (including a misspecified case violating theoretical assumptions ) and two challenging, high-stakes real-world datasets (ICU and single-cell). The results in Table 1 and Figure 1 are consistently positive and demonstrate a clear advantage over relevant baselines, especially in worst-case performance.

### Weaknesses
- The paper includes an extensive related work section but clearly overlooks several recent studies on robustness and causality. Moreover, it claims to introduce the first causality-inspired DRO method, whereas prior works on this topic already exist, such as:
  
  - Causal Adversarial Perturbations for Individual Fairness and Robustness in Heterogeneous Data Spaces. *Proceedings of the AAAI Conference on Artificial Intelligence* (2024).
    
  - Wasserstein distributionally robust optimization through the lens of structural causal models and individual fairness. Advances in  Neural Information Processing Systems (2024).
    
  - Designing Ambiguity Sets for Distributionally Robust Optimization Using Structural Causal Optimal Transport. Proceedings of the AAAI Conference on Artificial Intelligence (2025).
    
- The reliance on the Gaussian mixture model assumption for latent variables (Lemma 2) may be restrictive in practice, though the paper notes the method works with non-Gaussian shifts (Section C.2).
  
- The main optimality guarantee (Theorem 3) rests on a chain of strong assumptions: (i) the specific latent SCM structure (Eq. 1-3), (ii) the assumptions for affine identifiability (Lemma 2, e.g., piecewise affine decoder), (iii) the technical assumption on the *test* intervention's mean (Assumption 1), and (iv) elliptical noise distributions. While the paper is transparent about these and tests robustness, the guarantee's applicability is heavily qualified.
  

## Minor

There are some typos in the text:

- "DISTRIBUTIONALLY ROBUST OPIMIZATION" (line 077) → OPTIMIZATION
  
- "respetively" (line 208, 222) → respectively
  
- "latent, variables" (line 250)→ latent variables
  
- "step;" (line 268)→ step:
  
- "of of dimension" (line 288) → of dimension
  
- "the resulting is" (line 341) → the result is
  
- "adaptive risk minizimation" (line 359) → adaptive risk minimization
  

#

### Questions
- Could the method be extended to handle non-additive interventions, which would broaden its applicability to more complex causal structures?
  
- Can you provide (a) intuition for when assumption 1 holds in practice, and (b) an empirical test or diagnostic to verify it on real data?
  
- Why wasn't Rep4Ex included as a baseline? How does CIRRL's finite-radius certificate compare to Rep4Ex's extrapolation guarantees in your experimental settings?
  
- Can you provide confidence intervals or p-values for the real-data results? The ICU test set has only 67 samples—is the CIRRL vs. DRIG difference significant?

### Soundness
3

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
The paper proposes CIRRL (Causality-Inspired Robustness via Representation Learning), a two-step method for achieving distributional robustness in nonlinear settings. The first step learns a nonlinear representation of covariates using a Distributional Principal Autoencoder that maps data to a latent space. The second step applies the DRIG framework on the learned representations to achieve finite-radius robustness guarantees. The key theoretical contribution is establishing that CIRRL achieves optimal worst-case risk among all square-integrable functions under certain assumptions. The method is validated on synthetic and real-world datasets, where CIRRL consistently outperforms existing robust learning methods in worst-case and mean (across environments), and best or second-best in median.

### Strengths
- The paper addresses a gap by extending finite-radius robustness from linear to nonlinear settings. This is an important contribution since many/most real-world settings are nonlinear.

- The two-step approach is well-motivated and elegantly combines representation learning with causality-based DRO.

- The theoretical framework is rigorous and establishes the optimality of the learned predictor (Theorem 3).

### Weaknesses
**1.** The paper’s presentation, although overall clear, could be improved. The introduction and related work sections are unnecessarily lengthy and could be more focused. The introduction sounds somewhat vague and lacks specificity about the main results and contributions. Ideally, the introduction should be more to-the-point and give a clear, specific (though high-level) summary of the main results. The related work section on the other hand, is overly elaborate. It could be moved to the appendix and condensed in the main paper to make room for an example, or further discussion on the main content.

**2.** The paper lacks concrete examples throughout to ground the abstract concepts. 

**3.** The experimental section provides limited insight into when CIRRL's advantages are most pronounced and when existing methods might suffice.

**4.** I apologize if I miss it, but I cannot find a discussion on the choice of hyperparameter $\alpha$ in $L_{RL}$ (Eq. 4). This seems to be important.

**5.** A minor point: the paper is missing a citation to [1], which provides a unifying perspective on causality, IRM, and DRO that would strengthen the paper's positioning. They cite the pieces of work that [1] puts together, but I believe citing [1] itself is appropriate here.

[1] Bühlmann, Peter. "Invariance, causality and robustness." Statistical Science 35.3 (2020): 404-426.

### Questions
**1.** Could you provide a concrete example in a tangible scenario and walk the reader through the steps of the argument, the results, the assumptions, and their implications, with the help of that example? Given the overly elaborate intro and related work, I think the paper has room for such an example.

**2.** Can you elaborate on the choice of hyperparameters?

### Soundness
3

### Presentation
2

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
This paper proposes a novel framework, CIRRL (Causality-Inspired Robustness via Representation Learning), which aims to achieve distributional robustness in nonlinear settings. The method integrates recent advances in identifiable representation learning with causality-inspired robust optimization. Specifically, the authors first learn a nonlinear latent representation that is affine-identifiable across environments, and then apply a DRIG-style finite-radius robust regression in this latent space. The paper provides theoretical guarantees for robustness and identifiability under a structural causal model (SCM) formulation and demonstrates empirical superiority on both synthetic and real-world biomedical datasets.

### Strengths
•	Clear writing and presentation: The paper is well-written and easy to follow. The motivation, intuition, and mathematical formulation of CIRRL are presented clearly, and the connections with prior works such as DRO, DRIG, and IRM are well articulated.
•	Conceptual novelty: By combining representation learning with causality-based robustness, CIRRL offers a practical way to handle nonlinear dependencies and additive perturbations. This is a meaningful step toward bridging causality and distributional robustness.
•	Empirical performance: The experiments are convincing and demonstrate that CIRRL achieves consistently strong robustness compared with existing baselines (ERM, IRM, V-REx, ARM, DRIG). The proposed approach seems practical and potentially applicable to a wide range of real-world scenarios.

### Weaknesses
•	Limited discussion on causal mechanisms. While the method is labeled “causality-inspired,” the paper does not provide a concrete definition or interpretation of the causal mechanisms involved. The SCM formulation is mainly used to justify additive perturbations, but there is little insight into mechanism-level invariance or identifiability beyond affine transformations.
•	Affine representation assumption may not generalize to complex data (e.g., images). The proposed affine identifiability assumption is questionable for high-dimensional or convolutional data. The paper claims general applicability (including to image data), yet no experiments are conducted on such domains. It is unclear whether the linear latent assumption remains valid when spatial and hierarchical dependencies are dominant.
•	Theory is internally consistent but relies on tricky assumptions. The theoretical results only show optimality of the estimator under the self-defined loss function and a series of strong assumptions (existence of a linearizing nonlinear map, elliptical noise, Gaussian mixture latent structure). These make the results mathematically elegant but potentially fragile and hard to verify empirically.
•	Empirical validation does not directly test competing uncertainty-set definitions. Although the paper emphasizes CIRRL’s advantage in modeling nonlinear relationships and finite-radius robustness, the experiments do not include direct comparisons with other uncertainty-set formulations (e.g., Wasserstein-based DRO, moment-based sets). This weakens the claim of superiority in modeling distributional uncertainty.

### Questions
1.	In Equations (2) and (3), the additive interventions δ_e and v are used to model distributional shifts. Could the authors clarify how these additive perturbations capture realistic SCM-level distribution shifts beyond mean or variance changes?
2.	How should the dimension of the latent representation Z be chosen relative to X? Is it a hyperparameter, or are there theoretical or empirical guidelines?
3.	Since CIRRL relies on a linear regression in the latent space, do the learned linear coefficients b offer any interpretability benefits? Or does the affine transformation φ(·) essentially make the model another black-box representation?

### Soundness
3

### Presentation
3

### Contribution
3
