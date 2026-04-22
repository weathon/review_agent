# Transductive Learning for Out-of-Distribution Molecular Property Prediction

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Predicting molecular properties outside the training data distribution (Out-of-Distribution, OOD) is critical for accelerating drug discovery. This task requires models to extrapolate beyond known property ranges and generalize to novel chemical structures—a common failure point for standard machine learning models. While transductive analogical reasoning shows promise, prior methods are often constrained by fixed descriptors and single-anchor comparisons. To overcome these limitations, we introduce Multi-Anchor Latent Transduction (MALT) framework, which operates directly within a learned latent space. MALT can leverage embeddings from any powerful, pre-trained molecular encoder to select multiple relevant analogues of query molecule. It then integrates the query and anchor embeddings to generate a final prediction. On rigorous OOD benchmarks targeting shifts in both property values and chemical features, MALT consistently improves generalization over standard inductive baselines. Notably, our framework also matches or surpasses the in-distribution performance of these base models. These findings establish multi-anchor transduction in latent space as an effective strategy to augment existing molecular encoders, enabling robust and extrapolative predictions needed to solve challenging discovery tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a transductive method to address the out-of-distribution (OOD) problem in molecular representation learning. The authors employ a strategy that selects anchor samples from the training data to integrate features for input query molecules, thereby generating their representations. This approach leads to improved model performance under OOD data splits. The model was evaluated on multiple datasets for both classification and regression tasks, where it demonstrated superior performance.

### Strengths
1. The writing is clear, making the paper easy to follow.
2. The experimental analysis is comprehensive, thoroughly evaluating the method's effectiveness from various aspects.

### Weaknesses
1.  The presented baseline comparisons are limited. The work does not compare against any existing Test-Time Adaptation (TTA) paradigms from related work. Furthermore, for invariant molecular representation learning under the inductive setting, only one baseline is compared, which is not the most recent. There exist more up-to-date baselines, for example [1]. For the regression task, no inductive invariant learning baselines are included at all; a comparison could be made by adapting the predictive loss $ \mathcal{L}_{pred}$ from [2] for regression.

2.  Some of the definitions and formulations in the paper are confusing. Please refer to the **Questions** section for details.

3.  The proposed method is not strictly limited to molecular representation learning and could potentially be applied to broader graph OOD problems. It is a pity that the authors did not explore this with corresponding experiments.

**Reference**

[1] Aming, W. U., and Cheng Deng. "CFD: Learning Generalized Molecular Representation via Concept-Enhanced Feedback Disentanglement." *The Thirteenth International Conference on Learning Representations*.

[2] Zhuang, Xiang, et al. "Learning invariant molecular representation in latent discrete space." *Advances in Neural Information Processing Systems* 36 (2023): 78435-78452.

### Questions
1.  Equation 3 is puzzling. Even if the possible values of $\Delta x_{tr}$ are constrained to the set $\\{x_{an}-x_{j} |  x_j \in D_{X}^{tr} , y_{j} < y_{an}  \\}$,  the specific value of $\Delta x_{tr}$ used in the equation is still not fixed and can vary within this set. Could you please clarify how this value is determined?
2.  The main text describes the Transduction Module as using a fixed, rule-based algorithm for anchor selection. This appears to contradict the "Update $\mathcal{T}$" step in Algorithm 1. Could you explain the necessity and purpose of this update step?
3.  What was the policy for selecting the model checkpoint used for inference on the test sets?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a learning framework named Multi-Anchor Latent Transduction (MALT) for handling out-of-distribution (Out-of-Distribution, OOD) problems in molecular property prediction (Molecular Property Prediction, MPP). The main contributions of the paper include: (1) A model-agnostic transductive framework that can be integrated with any pre-trained molecular encoder (such as GIN); (2) A multi-anchor latent reasoning mechanism that overcomes the fragility of single-anchor methods; (3) Experimental validation on benchmarks such as MoleculeNet, DrugOOD, and Activity Cliffs, showing that MALT outperforms baselines in OOD generalization while matching or surpassing baselines on in-distribution (ID) tasks.

### Strengths
The quality of the paper is high, with a clear structure and smooth logic. The authors selected diverse benchmark datasets (MoleculeNet, DrugOOD, Activity Cliffs, and Lo-Hi), covering regression and classification tasks, as well as various types of distribution shifts (such as scaffold-based covariate shifts and label shifts). The baseline selection is reasonable. The evaluation metrics (such as AUROC, MAE, RMSE) are consistent with domain standards, and comparisons of ID and OOD performance are reported.

### Weaknesses
The novelty of the method is insufficient. Although MALT introduces a multi-anchor latent transductive mechanism and claims to overcome the limitations of previous single-anchor methods, the multi-anchor concept is not entirely original. For example, in semi-supervised learning, Halpern et al. (2016) used multiple anchors to combine features to enhance representations; in domain adaptation tasks, UniJDOT (arXiv:2503.11217, 2025) adopted multiple anchors in the feature space to align unknown samples, although applied to time series data, its dynamic anchor update and fusion mechanism is similar to MALT's multi-anchor attention fusion. Additionally, MALT is built on the basis of Bilinear Transduction (arXiv:2502.05970, 2025), which already uses single anchors in representation space for transductive extrapolation; MALT's main extension is from single-anchor to multi-anchor and operating in learned latent space, but this is more like an incremental improvement rather than a fundamental innovation. If it can be proven that the specific application in the field of molecular property prediction (such as handling activity cliffs) brings unique advantages, this will strengthen the novelty claim of the paper; otherwise, readers may view it as a combination of existing ideas rather than a pioneering framework.

### Questions
1. The paper emphasizes that multi-anchors overcome the fragility of single-anchors, but how to ensure that the selected anchors are diverse rather than redundant? For example, in activity cliff scenarios, if multiple anchors come from similar scaffolds, will it reduce generalization?
2. In Equation 5, is it possible to consider the labels of anchors?

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
3

### Summary
The paper introduces MALT, a multi-anchor latent transduction framework for robust molecular property prediction under out-of-distribution (OOD) conditions arising from covariate shift (novel structures) or label shift (novel property values). Given a query embedding, MALT retrieves the top-k nearest training embeddings from a memory bank $Z_{train}$. These anchors then serve as keys/values in a multi-head attention layer, producing z_attn. A learnable prediction head then takes $z_{query}$, $z_{attn}$, and optionally $W_{anchors}$ (query-anchor distances) as inputs before making its final prediction. Across multiple benchmarks, MALT shows consistent gains over the baselines.

### Strengths
- MALT is conceptually intuitive and can be used with most molecular encoders in a plug-and-play manner by adding a retrieval and attention module.
* Reported results show generally consistent gains on OOD scenarios in MoleculeNet, DrugOOD, and Activity Cliffs datasets.

### Weaknesses
* Covariate-shift claim depends on the retrieval quality. Cross-attention over anchors intuitively helps only if the retrieved set is informative in OOD regimes. Using only $z_{anchors}$ for K/V (default configuiration) may not directly address covariate shift when $z_{query}$ is still far too different from what is seen in the training set. Since the model can optionally include $W_{anchors}$, an ablation isolating its effects might strengthen the claim.
* The baselines do not cover recent large-scale foundation models (e.g., UniMol-style pretraining). Showing results against large-scale pretraining methods would better contextualize MALT.
* There are multiple cases in the reported tables where a MALT variant underperforms its non-MALT backbone. A systematic failure analysis and a safeguard to fall back to the base model such that the prediction quality is at least as good as the backbone would improve the usability of MALT.
* The encoder $E$ is trained end-to-end, so $E_{n}(x)$ ≠ $E_{m}(x)$ for training steps $n$ and $m$, and molecule $x$. Because $Z_{train}$ is updated only every $N$ epochs, $W_{anchors}$ may not reflect real-time latent distances, i.e., the memory bank becomes stale during training. A clarification on whether this mismatch degrades performance would be interesting.
* The predictor uses $f(z_{query}, z_{attn})$ (optionally with $W_{anchors}$), but it does not directly use the labels $\{y_i\}_{i=1}^k$ of retrieved anchors during inference. A label-aware fusion/attention could be beneficial for OOD cases.

### Questions
Please refer to weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes MALT, a multi-anchor transductive learning in latent space designed for generalizability on molecular property. MALT uses a memory
bank of embeding vectors corresponding to training samples. Given a query molecule, MALT useses multiple anchor points selector from the memory bank,
and aggregates this information with the query to make predictions. MALT achieves improvement on multiple OOD benchmarks.

### Strengths
- MALT hows good performance on OOD generalization. 

- It relies on multiple anchor points as opposed to a single anchor point, making it more robust.

- Since MALT operates in latent space, it makes the method modular and can be used with any molecular encoder.

- Shows utility by demonstrating performance on real world drug discovery benchmark.

### Weaknesses
- The effectiveness of MALT relies on the quality of the chosen encoder hence if the encoder produces poor representations, the transduction merely
  propagates this. Thus, the foundational problem of OOD generalization cant be mitigated by the method if the underlying foundation model isn't
  robust enough.

- The computation cost of maintaining a memory bank, latent embeddings for all samples in the training set, can be high. This can be prohibitive
  compared to keeping a few anchor points or even learned anchor points.

### Questions
- For extremely large databases, how will the memory bank approach scale? How will anchor selection work in such a case?

- Does anchor selection correlate with molecular properties? And in regions where very few samples exist in the molecular space, how does the anchor
  based approach hold up?

### Soundness
3

### Presentation
3

### Contribution
3
