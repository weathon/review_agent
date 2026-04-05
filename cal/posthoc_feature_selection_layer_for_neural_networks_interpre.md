=== CALIBRATION EXAMPLE 24 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title is accurate in broad terms: the paper indeed proposes a post-hoc feature selection layer for interpreting neural networks. However, it slightly overstates the generality of the method because the paper only evaluates tabular data and primarily discusses classification networks, not “neural networks” in general.
- The abstract does state the problem, method, and evaluation setup reasonably clearly. It identifies the post-hoc adaptation of FSL, frozen pre-trained models, and comparison to several attribution methods.
- The main concern is that the abstract makes several strong claims that are only partially supported by the paper’s evidence:
  - “successfully identified relevant features across the different datasets” is too broad, because Appendix C explicitly reports that on SynthA post-hoc FSL is the worst among feature-weighting methods on PIFS/PSFI and silhouette.
  - “distinct advantages over other state-of-the-art methods” is not consistently true across all metrics; in several tables FSL or gradient-based methods outperform post-hoc FSL on stability or predictive performance.
  - The abstract also emphasizes “stability” but the paper’s own results show post-hoc FSL is often less stable than the original FSL and several attribution methods.

### Introduction & Motivation
- The problem is well motivated: interpretability for deployed neural networks in high-stakes settings, especially healthcare, is a meaningful ICLR-level concern.
- The gap is identified correctly: original FSL is embedded and requires training from scratch, which limits use on already trained networks. That is a legitimate and useful motivation for a post-hoc version.
- The contribution statement is mostly accurate, but it slightly over-extends the expected impact. In particular, the claim that the method may “potentially improve predictive performance” is not established as a core contribution; in several real-world settings it is comparable rather than better, and the paper later acknowledges instability.
- The introduction could have framed more clearly what is novel relative to simply learning a mask layer on top of a frozen network. The core conceptual difference from common post-hoc masking or feature reweighting approaches is not sufficiently sharpened.

### Method / Approach
- The method is described at a high level, but reproducibility is weaker than ICLR typically expects for a method paper.
- The central formulation in Section 3.2 is conceptually simple: freeze the pretrained model, insert a trainable one-to-one feature weighting layer, and optimize only the layer weights. That is clear.
- However, there are important missing details:
  - The exact objective used for post-hoc training is not fully specified. The paper says the final output is used to compute the loss of the whole model, but it does not state whether the loss matches the original task loss, how labels are used, or whether the goal is to preserve predictions or explain labels.
  - The exact form of the FSL regularizer is not clearly presented in the main text; Equation (1) appears garbled in the parser, but the paper should still clearly define the regularization mathematically in a stable form.
  - The choice of ReLU activation and initialization at 1.0 is motivated heuristically, but the paper does not justify why this is preferable to the original FSL initialization at 1/n beyond intuition. This matters because initialization strongly affects optimization and may partly explain the reported differences.
- There are also conceptual issues:
  - If the layer multiplies inputs by nonnegative weights and the base network is frozen, the method is effectively learning a global input mask that approximates the original model. That is a valid approach, but the paper does not discuss identifiability: multiple masks can preserve predictions while encoding very different “importance” patterns.
  - The paper treats learned weights as feature importance, but does not justify why the learned mask should be interpreted causally rather than as a proxy for feature utility under a constrained retraining objective.
- Edge cases/failure modes are only partially addressed. The paper later notes multi-class difficulty, but there is no principled discussion of when the post-hoc mask will fail because the frozen model uses distributed feature interactions that cannot be faithfully recovered by a single global weight per input feature.
- For the theoretical side, there are no proofs, which is fine for this type of paper, but the method would benefit from a clearer argument for why the learned weights correspond to interpretability rather than just predictive approximation.

### Experiments & Results
- The experiments do test the broad claims, but not as thoroughly as ICLR would usually require for a method paper making interpretability claims.
- The dataset selection is reasonable in spirit: XOR and SynthA give ground truth for relevance, and the microarray/spam datasets test high-dimensional tabular settings.
- That said, several important issues limit the strength of the empirical evidence:
  - The main evaluation appears to emphasize t-SNE visualizations and silhouette scores. These are only indirect proxies for interpretability and cluster structure, not direct measures of feature attribution quality. For a feature selection paper, these should be secondary, not primary evidence.
  - The paper relies heavily on the weighted t-SNE visualization framework from prior work, but it is not fully clear how sensitive the conclusions are to t-SNE hyperparameters or the weighting scheme.
  - There are no clear ablations on the post-hoc FSL design choices that matter most: initialization at 1.0 vs 1/n, ReLU vs other activations, with/without L1, or training schedule/epochs. These are central to the method and should be ablated.
  - It is not clear whether all competing attribution methods were tuned fairly. For methods like Integrated Gradients, DeepLIFT, Gradient SHAP, and Feature Ablation, the baseline/reference choice and implementation details strongly affect results. Those details are not adequately described.
  - Statistical reporting is incomplete in places. Some tables include means and standard deviations, but there is little indication of the number of runs, confidence intervals, or whether significance tests are applied consistently across all comparisons.
- The results themselves are mixed:
  - On XOR, post-hoc FSL matches the others, which is a weak but positive sanity check.
  - On SynthA, post-hoc FSL is worse than FSL and several post-hoc baselines on both PIFS/PSFI and stability, so the paper’s claim that it is generally strong on synthetic data is not supported.
  - On real-world datasets, post-hoc FSL is often comparable on predictive performance, but the strongest stability results are not consistently in its favor; in fact, Appendix D explicitly says it is among the least stable.
  - The TabPFN + Kernel SHAP comparison is interesting, but it is only run on one dataset, with source-code compatibility constraints. That is too narrow to support broader claims.
- Some claims appear cherry-picked or at least selectively emphasized:
  - The abstract and conclusion emphasize “superior performance” and “distinct advantages,” but the detailed tables show this is metric-dependent and dataset-dependent.
  - The paper highlights silhouette improvements while downplaying stability degradation and the weaker SynthA feature-selection accuracy.
- Datasets and metrics are broadly appropriate for tabular feature selection, but the evaluation is missing a direct qualitative assessment of feature relevance on real-world datasets, e.g., known biological pathways or spam-related terms, which would have strengthened the interpretability story.

### Writing & Clarity
- The paper is understandable at a high level, but several sections are not sufficiently clear for reproducibility or for assessing the contribution.
- The method section would benefit from a precise algorithm box or pseudocode specifying optimization, loss, batching, stopping criterion, and how feature importance is extracted.
- The experiments section is hard to follow because the narrative sometimes mixes predictive performance, attribution quality, visual separability, and stability without clearly distinguishing which is primary and which is supporting.
- Figures and tables are conceptually useful, but the parser-extracted text makes them hard to inspect here. More importantly, even setting aside parsing artifacts, the paper should better explain what conclusions each figure is meant to support. For example:
  - Weighted t-SNE figures are used repeatedly, but the paper does not sufficiently justify why they are appropriate evidence of interpretability.
  - The stability plots in Appendix D are mentioned but not integrated into a clear story about when post-hoc FSL should or should not be trusted.
- The discussion around multi-class datasets is one of the clearer parts, but it arrives late. This limitation should have been stated much earlier, because it materially affects the scope of the contribution.

### Limitations & Broader Impact
- The paper does acknowledge two important limitations in Section 6.1:
  - instability on multi-class problems,
  - lack of suitability for image data because feature semantics differ across pixels.
- However, the limitations are incomplete relative to the claims made:
  - The method assumes a single global importance weight per feature, which is a major limitation for heterogeneous, context-dependent tasks even in tabular settings.
  - The paper does not discuss the risk that post-hoc weights may produce a misleading sense of explanation if they only approximate the frozen model’s behavior rather than true feature relevance.
  - There is no discussion of fairness, bias, or downstream harms, despite the introduction emphasizing healthcare and high-stakes use cases.
- A broader-impact discussion would be especially relevant because interpretability methods can be over-trusted. If the method is unstable on multi-class tasks, that is not just a technical limitation but a potential user-safety concern.
- The ethics statement is minimal and does not meaningfully engage with the interpretability risks or misuse potential.

### Overall Assessment
This paper proposes a plausible and practically useful idea: adapting the Feature Selection Layer into a post-hoc module for frozen tabular neural networks. That is a legitimate contribution, and the empirical setup is reasonably broad for a tabular interpretability study. However, the current version does not yet meet a strong ICLR acceptance bar because the method description lacks important implementation details, the evaluation leans too heavily on indirect visual metrics, ablations are missing for key design choices, and the results are more mixed than the abstract and conclusion suggest. Most importantly, the paper does not convincingly establish that post-hoc FSL is consistently better than existing post-hoc methods or even the original FSL; its strengths appear dataset- and metric-dependent, and its instability on harder settings is a real concern. The idea is worthwhile, but the evidence and framing need tightening before the contribution can be considered solid.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes a post-hoc version of the Feature Selection Layer (FSL) that is inserted in front of a frozen pretrained neural network, with only the new layer trained to produce feature weights for interpretability. The main claim is that this preserves the predictive behavior of the original model while offering feature relevance estimates competitive with established post-hoc attribution methods, especially on tabular and high-dimensional datasets.

### Strengths
1. **Clear practical motivation for post-hoc interpretability.**  
   The paper identifies a real limitation of the original embedded FSL: it must be trained jointly from scratch, which prevents use on already trained models. The proposed post-hoc adaptation directly addresses this deployment scenario, which is relevant to ICLR’s interest in usable deep learning methods.

2. **Simple and modular method design.**  
   The method is conceptually straightforward: freeze the pretrained network and train only a lightweight feature-weighting layer. This makes the approach easy to understand and potentially easy to integrate into existing pipelines.

3. **Evaluation spans multiple dataset regimes.**  
   The experiments include synthetic data with ground-truth informative features, a noisy synthetic dataset, and real-world high-dimensional tabular datasets (spam and microarray). This breadth is useful because it probes both feature recovery and practical interpretability settings.

4. **The paper compares against multiple strong post-hoc baselines.**  
   The method is evaluated against Integrated Gradients, Noise Tunnel, DeepLIFT, Gradient SHAP, Feature Ablation, and Kernel SHAP in a TabPFN experiment. This is a meaningful comparison set for attribution-style interpretability work.

5. **Uses more than one evaluation perspective.**  
   In addition to predictive metrics, the paper reports feature-selection metrics, stability measures, and visual clustering metrics such as weighted t-SNE and silhouette score. This is better than relying on a single proxy for interpretability.

### Weaknesses
1. **Novelty is limited relative to existing post-hoc attribution and feature reweighting ideas.**  
   The core idea is to freeze a pretrained network and learn a feature-wise gating layer to mimic or preserve predictions. This is a natural adaptation of an existing embedded method, but the paper does not clearly establish a strong conceptual advance beyond applying FSL post-hoc. For ICLR’s acceptance bar, the contribution currently feels incremental unless the empirical evidence convincingly shows a distinct advantage.

2. **The experimental narrative is internally inconsistent in places.**  
   The paper makes broad positive claims in the abstract and conclusion, but several results are mixed or contradictory. For example, the post-hoc method is said to achieve “distinct advantages” and “superior performance” on visual and clustering-based interpretability, yet it also underperforms FSL on important synthetic settings and is less stable on some datasets. The paper does not sufficiently reconcile these trade-offs.

3. **Stability analysis suggests the method is weaker than alternatives.**  
   On SynthA and Spam, post-hoc FSL often has lower Jaccard, Spearman, and Pearson stability than other methods, and in some cases substantially worse than the original FSL. Since interpretability methods are often expected to be robust, this is a notable weakness rather than a minor caveat.

4. **Limited evidence that learned weights correspond to faithful explanations.**  
   The paper evaluates whether weights align with known informative features and whether weighted t-SNE improves clustering, but it does not provide stronger faithfulness tests such as deletion/insertion curves, counterfactual sensitivity, or causal perturbation analyses. Improved silhouette score does not necessarily imply faithful attribution.

5. **Multi-class case handling appears weak.**  
   The paper itself acknowledges that the method is unstable for multi-class datasets and that it produces a single global importance score, which is less suitable when feature relevance is class-dependent. This is an important limitation because one of the real-world datasets is six-class breast cancer data.

6. **Baseline and implementation details are not sufficiently specified.**  
   ICLR expects reproducible methodology. The paper lacks enough detail about model architectures, training hyperparameters, optimization settings, stopping criteria, regularization values, and how exactly the pretrained baseline/FSL models were constructed. The reproducibility statement is also incomplete because the code link is not actually provided.

7. **Evaluation protocol leaves open questions about fairness of comparison.**  
   It is unclear whether all attribution methods were compared under the same architecture, same trained model, same baselines, and same preprocessing. In interpretability papers, fairness of attribution comparison matters greatly because results can be sensitive to model choice, baseline choice, and feature scaling.

8. **The paper over-relies on visualization metrics.**  
   Weighted t-SNE and silhouette are useful exploratory tools, but they are indirect proxies for explanation quality. For ICLR standards, they are not sufficient evidence on their own to support strong claims about interpretability.

### Novelty & Significance
**Novelty: modest.** The paper adapts an existing embedded feature-selection layer into a post-hoc wrapper for pretrained networks. This is a reasonable engineering extension, but it is not a major methodological leap.

**Clarity: moderate.** The overall idea is understandable, but the exposition is weakened by inconsistent claims, limited implementation details, and over-dependence on indirect metrics. The method description is clearer than the empirical argument.

**Reproducibility: partial.** Dataset sources are cited and the general procedure is described, but key training and architectural details are missing, and the code is not actually available in the paper text provided. That falls short of ICLR’s reproducibility expectations.

**Significance: moderate to low.** The idea may be practically useful for probing pretrained tabular models, but the current evidence does not establish a strong enough improvement over existing post-hoc attribution methods to clear the ICLR bar. The contribution appears more like a useful variant than a broadly impactful advance.

### Suggestions for Improvement
1. **Tighten the claim and align it with the evidence.**  
   The paper should distinguish clearly between cases where post-hoc FSL is competitive, where it is better than gradient-based methods, and where it is worse than the original embedded FSL.

2. **Add stronger faithfulness evaluations.**  
   Include deletion/insertion tests, feature perturbation studies, or retraining-based sanity checks to show that the selected features truly matter to model behavior, not just to visualization metrics.

3. **Clarify the training objective and optimization details.**  
   Report the exact loss used for the post-hoc layer, the regularization strength, learning rate, optimizer, batch size, number of epochs, early stopping, and how weights are normalized or constrained.

4. **Strengthen and standardize the baselines.**  
   Ensure all post-hoc attribution methods are evaluated on the same frozen model, same preprocessing, and same data splits. Explain baseline choice for each attribution method, especially for gradient-based methods.

5. **Address multi-class limitations more directly.**  
   If the method is intended only for binary or globally aggregated tabular tasks, state that explicitly. Otherwise, propose a class-conditional variant and test it on the Breast dataset.

6. **Report variance and statistical testing more transparently.**  
   Since the method appears unstable in some settings, provide confidence intervals, per-fold results, and a clearer comparison across runs. Explain the statistical significance tests in a way that directly supports the paper’s main claims.

7. **Provide an ablation study.**  
   Is the gain coming from the post-hoc training itself, the ReLU constraint, the L1 penalty, or the initialization at 1.0? An ablation over these design choices would materially strengthen the contribution.

8. **Include complete reproducibility artifacts.**  
   Release code, exact preprocessing scripts, random seeds, and model checkpoints if possible. For ICLR, this is especially important because interpretability results can be sensitive to implementation details.



# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add comparisons to strong tabular feature-selection baselines such as L1/logistic regression, random forest/permutation importance, mutual information, ReliefF, and Boruta on the same datasets. ICLR will expect evidence that post-hoc FSL is meaningfully better than standard tabular selectors, not just comparable to gradient attribution methods.

2. Add an explanation-specific evaluation on synthetic data with known ground truth under more than one signal structure, especially multi-class and class-conditional relevance. Right now the method is only stress-tested on one XOR setup and one synthetic binary dataset, which is too weak to support claims about general feature relevance recovery.

3. Add a direct post-hoc-vs-embedded FSL comparison controlling for the same frozen backbone, training budget, initialization, and regularization. Without an ablation showing that the post-hoc adaptation itself—not a different optimization regime or tuning—drives the reported gains, the central contribution is not convincing.

4. Add robustness experiments across multiple random seeds, class imbalance settings, and train/test splits with statistical significance testing on the final claims. The paper reports averages but does not establish that the observed advantages are stable enough to survive ICLR scrutiny for interpretability methods.

5. Add a scalability experiment on the claimed high-dimensional setting with runtime, memory, and convergence behavior as feature count grows. Since the method is proposed as lightweight and post-hoc, its practicality on 20k–50k features must be demonstrated explicitly.

### Deeper Analysis Needed (top 3-5 only)
1. Analyze why post-hoc FSL sometimes underperforms the original FSL and even some attribution methods on SynthA and stability metrics. The paper currently attributes this to dataset structure, but without a concrete analysis the reader cannot tell whether the method has a fundamental limitation or just needs tuning.

2. Analyze whether the learned weights are actually faithful explanations of the frozen model or just a surrogate that approximates the model’s outputs. The paper claims interpretability, but it never tests faithfulness via deletion/insertion curves, output perturbation, or agreement with model behavior under feature masking.

3. Analyze sensitivity to the FSL design choices: ReLU, L1 penalty, and initialization at 1.0 versus 1/n. These are core methodological choices, and without ablation the reader cannot know whether the method is robust or whether the reported behavior is an artifact of these settings.

4. Analyze class-wise importance for the multi-class Breast dataset instead of only a single global importance vector. The authors themselves note this limitation, and without class-conditional analysis the method’s usefulness on multi-class problems remains unproven.

5. Analyze whether feature rankings are consistent with domain knowledge on the microarray datasets. For ICLR-level interpretability claims, the paper needs evidence that the top-ranked genes map to known biological markers or pathways, not just better silhouette scores.

### Visualizations & Case Studies
1. Show ranked feature bar plots with confidence intervals across seeds for each dataset, especially the top-10 or top-30 features. This would reveal whether the method consistently isolates the same features or if the rankings are too unstable to trust.

2. Add deletion/insertion or masking curves for the top-ranked features versus random and baseline rankings. This would directly show whether the selected features truly drive the frozen model’s predictions.

3. Add a class-conditional feature attribution visualization for the Breast multi-class task. A global ranking can hide class-specific signals, and this visualization would expose whether the method collapses distinct class evidence into a misleading average.

4. Add a case study on a few individual examples showing input features, learned weights, and prediction changes after perturbing the top features. This would make it clear whether the method is capturing causal drivers or only producing plausible-looking scores.

### Obvious Next Steps
1. Extend the method to class-specific post-hoc FSL or a multi-head version for multi-class tasks. The current global-weight formulation is a direct mismatch to the paper’s own reported weakness on Breast.

2. Benchmark against established post-hoc explanation frameworks on tabular data, including SHAP variants, permutation-based methods, and learned surrogate explainers. The current comparison set is too narrow for an ICLR paper claiming state-of-the-art interpretability.

3. Add a rigorous ablation study of the training objective and hyperparameters, including the effect of freezing only part of the backbone. This is the most direct next step to determine whether the method is intrinsically useful or dependent on fragile implementation details.

4. Release and evaluate on more than four datasets, including at least one additional real-world benchmark from a different domain. The current empirical base is too small to support broad claims about pre-trained DNN interpretability.

# Final Consolidated Review
## Summary
This paper proposes a post-hoc adaptation of the Feature Selection Layer (FSL) for frozen pretrained neural networks on tabular data. The idea is straightforward: insert a lightweight, trainable feature-weighting layer in front of a fixed model and interpret the learned weights as feature importance, then compare this against the original embedded FSL and several post-hoc attribution methods.

## Strengths
- The motivation is practical and legitimate: the original FSL must be trained jointly from scratch, while this variant can be attached to an already trained model, making it more usable for deployed tabular networks.
- The experimental scope is reasonably broad for a tabular interpretability paper, covering synthetic data with ground-truth informative features, high-dimensional real-world datasets, and several attribution baselines. The comparison against Integrated Gradients, DeepLIFT, Gradient SHAP, Feature Ablation, Noise Tunnel, and a TabPFN/Kernel SHAP setting is a meaningful effort.
- The method is simple and modular, which makes it easy to understand and potentially easy to integrate into existing pipelines.

## Weaknesses
- The empirical story is weakly supported by the paper’s own results. The abstract and conclusion overstate the gains: post-hoc FSL is not consistently superior, and on SynthA it is clearly worse than the original FSL on feature-selection quality and stability. That matters because the paper’s central claim is about interpretability, not just one-off performance on selected datasets.
- The method is under-validated as an explanation method. The paper relies heavily on weighted t-SNE and silhouette scores, which are indirect clustering proxies, not direct faithfulness tests. For a feature-selection method, this is too thin to justify strong interpretability claims.
- Stability is a real issue, not a minor caveat. The paper itself reports that post-hoc FSL is often less stable than the original FSL and some attribution baselines, especially on SynthA and Spam. If feature rankings vary substantially across folds, the explanation is hard to trust.
- The multi-class limitation is important and not adequately addressed. The authors acknowledge that a single global weight vector is a poor fit for class-dependent relevance, and the Breast dataset results are indeed weaker and less convincing. This is a substantive limitation of the method’s scope.
- Reproducibility and methodological detail are insufficient. The paper does not clearly specify the exact training objective, optimizer settings, early stopping, hyperparameters, or ablation of core design choices such as ReLU, L1 regularization, and initialization at 1.0. Given how sensitive a learned masking layer can be, this is a serious gap.

## Nice-to-Haves
- A direct faithfulness evaluation such as deletion/insertion or feature-masking curves would strengthen the paper substantially, even though the current evaluation already suggests several weaknesses.
- A class-conditional extension for multi-class problems would be valuable, since the current global-ranking formulation is mismatched to the Breast dataset setting.
- More transparent reporting of variance across runs and clearer statistical testing would help readers judge whether the method is reliable or just occasionally competitive.

## Novel Insights
The main conceptual contribution is not a fundamentally new interpretability principle, but a useful reframing of an embedded feature-selection layer into a post-hoc wrapper for frozen networks. The best takeaway from the experiments is also the most sobering one: this reparameterization can preserve performance and sometimes produce visually cleaner cluster structure, but it does not reliably outperform established attribution methods, and its weakest point is stability when the feature relevance structure is complex or class-dependent. In other words, the method looks like a plausible engineering adaptation, but the paper does not yet show that it is a robust explanation technique rather than a learned surrogate mask with mixed faithfulness.

## Potentially Missed Related Work
- SHAP / permutation-based tabular feature importance methods — relevant as stronger tabular baselines than only gradient-style attribution methods.
- Boruta, ReliefF, mutual information, random forest importance, L1/logistic regression — relevant standard feature-selection baselines that would better situate the proposed approach against classical tabular selectors.
- None identified beyond these benchmark-style methods.

## Suggestions
- Add a direct ablation of the three core design choices: initialization at 1.0 vs 1/n, ReLU vs alternative activations, and with/without L1 regularization.
- Report the exact post-hoc training objective and all optimization details so the method can be reproduced and fairly compared.
- Include a faithfulness test on top-ranked features, not just t-SNE/silhouette, to show the learned weights actually drive the frozen model’s predictions.
- If the method is intended mainly for binary or globally homogeneous tabular tasks, say so explicitly; otherwise, develop and test a class-conditional variant.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 2.0]
Average score: 2.0
Binary outcome: Reject
