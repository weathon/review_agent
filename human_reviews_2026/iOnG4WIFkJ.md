# JUCAL: Jointly Calibrating Aleatoric and Epistemic Uncertainty in Classification Tasks

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 2, 4, 8

## Abstract
We study post-calibration uncertainty for trained ensembles of classifiers.
 Specifically, we consider both aleatoric uncertainty (i.e., label noise) and epistemic uncertainty (i.e., model uncertainty).
 Among the most popular and widely used calibration methods in classification are temperature scaling (i.e., *pool-then-calibrate*) and conformal methods.
  However, the main shortcoming of these calibration methods is that they do not balance the proportion of aleatoric and epistemic uncertainty.
  Nevertheless, not balancing epistemic and aleatoric uncertainty can lead to severe misrepresentation of predictive uncertainty, i.e., can lead to overconfident predictions in some input regions while simultaneously being underconfident in other input regions.
  To address this shortcoming, we present a simple but powerful calibration algorithm *Joint Uncertainty Calibration (JUCAL)* that jointly calibrates aleatoric and epistemic uncertainty. 
  JUCAL jointly calibrates two constants to weight and scale epistemic and aleatoric uncertainties by optimizing the *negative log-likelihood (NLL)* on the validation/calibration dataset. 
JUCAL can be applied to any trained ensemble of classifiers (e.g., transformers, CNNs, or tree-based methods), with minimal computational overhead, without requiring access to the models' internal parameters. 
We experimentally evaluate JUCAL on various text classification tasks, for ensembles of varying sizes and with different ensembling strategies.
  Our experiments show that JUCAL significantly outperforms SOTA calibration methods across all considered classification tasks, reducing NLL and predictive set size by up to 15\% and 20\%, respectively.
  Interestingly, even applying JUCAL to an ensemble of size 5 can outperform temperature-scaled ensembles of size up to 50 in terms of NLL and predictive set size, resulting in up to 10 times smaller inference costs.
  Thus, we propose JUCAL as a new go-to method for calibrating ensembles in classification.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a calibration technique *JUCAL* that considers both aleatoric uncertainty and epistemic uncertainty, i.e., the method jointly calibrates two constants to weight and scale epistemic and aleatoric uncertainties by optimizing the negative log-likelihood on the validation/calibration dataset. Experiments with fine-tuned large language models (LLMs) on six text classification tasks demonstrate the improvement of the proposed method.

### Strengths
1. The paper is relatively well-structured, and the motivation is clearly stated.

2. The initiation behind the method design is well described, together with detailed background knowledge and introductions in the appendix.

3. The proposed method shows improvements over the baselines.

4. The proposed method is a post-processing technique, can potentially be implemented on various ensemble models and classification tasks.

### Weaknesses
1. The proposed method lacks theoretical justification (limited theory).

2. JUCAL considers both aleatoric and epistemic components during calibration. But, it is not clear how it would actually influence the aleatoric, epistemic, and total (predictive) uncertainty in terms of the widely used entropy measures (i.e, the equations 5-7 in the Appendix). It would be better to include more theoretical analysis in this context, as well as the empirical evaluations. For example, how would the deep ensembles perform on out-of-distribution tasks, using the original measures vs. the calibrated ones?

3. Regarding the evaluation metrics, it would be beneficial to consider the expected calibration error (ECE) and also highlight the prediction performance, e.g, the accuracy.

4. Regarding the experiment designs, while the method itself is introduced for general classification models, its applicability on other classification tasks, like image classification e.g., [1] remains underexamined. Similarly, in the abstract, it claims that *any trained ensemble of classifiers (e.g., transformers, CNNs, or tree-based methods)*. It is not fully empirically evaluated. The baselines are limited.

[1] Roschewitz, Mélanie, et al. "Where are we with calibration under dataset shift in image classification?

### Questions
1. In lines 230-231, could the authors please indicate whether the inverse of the softmax is actually implemented or just using the logit?

2. I think in most cases, the softmax of the averaged logits is not equal to the averaged softmax of individual logits. How would this point affect your method design?

3. Minor: The readability of Figure 3 is a bit low; maybe consider increasing the font size, etc. It would be better to include the cost of the calibration in the main body. *is never significantly outperformed by any of the baselines on any evaluated metric.* (line 467) sounds like a really strong claim.

4. Would the authors discuss some limitations and future work in the main body?

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
The authors propose JUCAL, a post-hoc calibration method for ensembles of classifiers that aims to jointly calibrate aleatoric and epistemic uncertainty . The method introduces two calibration constants, `c1` and `c2`, which are optimized by minimizing the negative log-likelihood (NLL) on a validation set. `c1` acts as a temperature parameter on the logits of individual ensemble members to calibrate aleatoric uncertainty, while `c2` scales the diversity of the logits across the ensemble to calibrate epistemic uncertainty . The authors claim this simple approach leads to significant improvements in NLL, predictive set size, and other uncertainty-related metrics, even allowing smaller ensembles to outperform much larger ones .

### Strengths
*   The paper addresses the important and underexplored problem of balancing aleatoric and epistemic components during uncertainty calibration. Standard methods like temperature scaling treat predictive uncertainty as a single quantity, and the idea of disentangling these sources for more granular control is well-motivated.
*   The proposed method, JUCAL, is simple, intuitive, and computationally inexpensive. 
*   The experimental results on several text classification tasks are promising.

### Weaknesses
My main concerns are regarding the strength of the paper's core claims, the limited experimental scope, and the lack of crucial baselines.

*   **Overstated Disentanglement Claims:** The central claim of "jointly calibrating aleatoric and epistemic uncertainty" seems overstated and is not supported by sufficient experimental evidence. The primary evidence for successful disentanglement is presented in Figure 5, which shows that the estimated epistemic uncertainty (EU) decreases with more training data while aleatoric uncertainty (AU) does not . While this is a necessary sanity check, it contradicts recent literature suggests this separation is non-trivial. A recent benchmark (Mucsányi et al., NeurIPS 2024) shows a high correlation between common AU and EU estimators, which questions the practical benefit of calibrating them with separate parameters. The authors' own discussion acknowledges critiques of the mutual information-based decomposition they use (see Appendix A.2.1), yet the main paper proceeds as if the decomposition is clean. A more thorough analysis is needed to substantiate the claim that the observed performance gains stem from successful joint calibration rather than simply being a more flexible (two-parameter) fitting of the final predictive distribution. 
In particular, the authors should compare against alternative estimators for AU and EU derived from ensembles, such as those based on the Bregman decomposition or simple variance of the logits. Furthermore, comparing the AU/EU estimates from JUCAL to those from dedicated single-model methods like Evidential Deep Learning (EDL) or Deep Deterministic Uncertainty (DDU) would provide a much stronger test of whether JUCAL is truly capturing distinct uncertainty components.

*   **Missing Crucial Baselines:** The experimental evaluation is narrow, comparing only against "no calibration" and "pool-then-calibrate". The authors seem to have missed some crucial baselines that would be necessary to properly contextualize their contribution. For instance, in Section 4.1, they introduce "calibrate-then-pool" as a method that primarily adjusts aleatoric uncertainty. A comparison to this baseline is missing and would be essential for an ablation study to understand the individual contributions of `c1` and `c2`. 

*   **Limited Experimental Scope:** The experiments are confined to text classification tasks using ensembles of fine-tuned LLMs . To support the claim that JUCAL is a "new go-to method for calibrating ensembles in classification" , the evaluation must be extended to other domains, especially computer vision, and other model families like CNNs and ViTs. The dynamics of aleatoric and epistemic uncertainty can vary significantly across different data modalities and model architectures, and demonstrating robustness in these diverse settings is critical.

*   **Presentation of Results:** The presentation could be improved. The bar charts in Figure 4 are cluttered and hard to read, making it difficult to assess the relative performance improvements, especially given the small error bars . More effective visualizations, such as plots of relative improvement or tables with statistical significance tests, would improve the clarity of the results.

### Questions
See above

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes post-hoc calibration method for deep ensembles that jointly calibrates aleatoric and epistemic uncertainty using two scaling parameters c1 and c2. The method aims to address the limitation of temperature scaling and conformal methods, which do not balance the epistemic and aleatoric uncertainty. Concretely, JUCAL (1) applies temperature scaling (via c1) to each ensemble member (calibrate-then-pool style) to affect aleatoric uncertainty, and (2) linearly rescales each temperature-scaled member toward/away from the ensemble mean using c2 to adjust ensemble diversity (epistemic uncertainty) without changing the ensemble mean. JUCAL is evaluated on text classification tasks using pre-trained LLM ensembles, demonstrating improvements in NLL and prediction set size.

### Strengths
- The paper is generally well written
- JUCAL is conceptually simple, easy to implement, and can be applied to off-the-shelf ensembles without model re-training.
- The experiments compare across ensemble sizes and demonstrate consistent improvements on NLL and predictive set size.

### Weaknesses
**Lack of ablation studies on core claims**

The paper's main idea is that c1 calibrates aleatoric uncertainty while c₂ calibrates epistemic uncertainty. However, no ablation studies are provided to validate this claimed separation. Without these ablations, the interpretation of c1 and c2 as targeting specific uncertainty types remains speculative.

**Lack of support for main claims**

The conclusion states: “our approach provides a principled and impactful advance in uncertainty calibration for deep learning.“ However, no calibration metrics are reported. The paper evaluates NLL, AORAC, AOROC, and prediction set size, but omits standard calibration metrics (e.g. ECE, MCE, Brier score, class-wise calibration analysis, reliability diagrams). The absence of these metrics is especially evident given that the paper explicitly positions itself as advancing "uncertainty calibration." Without these, it's unclear if JUCAL actually improves calibration.
Moreover, describing the approach as 'principled' suggests derivation from theoretical foundations or formal guarantees. However, the paper provides neither a rigorous connection to established calibration theory nor mathematical justification for why the specific parameterization in Equation 2 should successfully decompose epistemic and aleatoric uncertainty.
	

**Limited experimental depth**

Gaps include missing ablation studies (e.g., comparing JUCAL against c1-only optimization, which is equivalent to the calibrate-then-pool approach mentioned in Section 4.1), omission of standard calibration metrics (see points above), and absence of key baselines—notably, the paper claims improvements in prediction set size yet fails to compare with conformal prediction methods designed specifically for this task with formal coverage guarantees. The evaluation is limited to pre-trained transformer ensembles on NLP tasks.

**Limitations of the method are not discussed**

For example, optimizing two parameters (c1, c2) via grid search requires sufficient calibration data to avoid overfitting, yet the paper provides no analysis of minimum sample size requirements.

### Questions
How does JUCAL relate to proper scoring rules or calibration error metrics (ECE, MCE)?

### Soundness
3

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
3

### Summary
Most existing calibration methods treat all instances uniformly, without accounting for the underlying uncertainty associated with each prediction. However, instances can differ significantly, some may exhibit high aleatoric uncertainty, while others reflect high epistemic uncertainty . Applying the same calibration to all such cases can lead to overconfidence on out-of-distribution samples and underconfidence on in-distribution samples with high aleatoric noise.
The paper introduces Joint Uncertainty Calibration (JUCAL), a post-hoc method that uses uncertainty estimates from an ensemble model to calibrate the predicted class probabilities more effectively. Specifically, it scales the aleatoric and epistemic components of the ensemble’s predictive distribution, allowing the calibration to adapt to the nature of the uncertainty in each instance.
The method is evaluated on multiple text classification benchmarks, where it consistently outperforms some standard calibration techniques across four metrics.

### Strengths
The method applies temperature scaling in a new way, taking into account ensemble differences.
The method is intuitive, simple to understand and can be applied post-hoc to already trained ensembles. This makes the method easy to use.

### Weaknesses
The experiments could include other post-hoc calibration methods as a comparison. Also, calibrate-then-pool could be added to comparison, since it is the special case of JUCAL when $c_2 = 1$.

Section 5.1 briefly mentions that ensemble members are based on pretrained architectures like GPT-2, BERT, and T5, but it remains ambiguous whether the ensembles consist of a single architecture or multiple architectures. Since the nature of the ensemble affects both uncertainty decomposition and calibration behavior, this detail is important for reproducibility and for interpreting the results.

Paper's motivation is that traditional calibration methods tend to be overconfident on out-of-distribution samples and underconfident on in-distribution samples with high aleatoric uncertainty. This claim would be better supported by explicitly evaluating JUCAL under OOD settings for example, by testing on textual datasets from different distributions than the training data (e.g., evaluating on a News classification dataset when models were trained on DBpedia). Such experiments would strengthen the argument that JUCAL provides more robust calibration across varying uncertainty regimes.

### Questions
In Table 10, both DBpedia Full and SetFit Full are shown with 25 members. Could you clarify whether Greedy-50 in these settings still uses only 25 models?

If the ensemble consists of a mix of architectures (e.g., BERT, T5, GPT-2), could you elaborate on what types of models were selected by the greedy algorithm? It would be helpful to know whether the selection favored certain architectures or maintained diversity, as this might impact both calibration behavior and generalization.

If possible, I’d be interested in seeing how the test NLL varies with different values of $c_1$ and $c_2$. Does JUCAL’s performance degrade smoothly, or is it sensitive to the specific combination of these parameters? A sensitivity analysis or ablation could help clarify how stable the calibration effect is with respect to these parameters.

### Soundness
3

### Presentation
3

### Contribution
4
