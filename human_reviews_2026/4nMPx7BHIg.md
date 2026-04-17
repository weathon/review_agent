# Learning Based on Neurovectors for Tabular Data: A New Neural Network Approach

- Decision: Reject
- Scores: 2, 0, 2, 2

## Abstract
In this paper, we present a novel learning approach based on Neurovectors, an innovative paradigm that structures information through interconnected nodes and vector relationships for tabular data processing. Unlike traditional artificial neural networks that rely on weight adjustment through backpropagation, Neurovectors encode information by structuring data in vector spaces where energy propagation, rather than traditional weight updates, drives the learning process, enabling a more adaptable and explainable learning process. Our method generates dynamic representations of knowledge through neurovectors, thereby improving both the interpretability and efficiency of the predictive model. Experimental results using datasets from well-established repositories such as the UCI machine learning repository and Kaggle are reported both for classification and regression. To evaluate its performance, we compare our approach with standard machine learning and deep learning models, showing that Neurovectors achieve competitive accuracy while significantly reducing computational costs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper present a predictive model based on Neurovectors, which are created to model similarity between data points. Unlike typical backpropagation used in neural network, the proposed model is trained usng energy-driven process. The model is evaluated on three tabular datasets and compared with basic baselines.

### Strengths
-The authors tackle an important problem of tabular data classification, where typical neural networks are comparable to shallow models.
-The idea is interesting and the model is inspired by LLMs

### Weaknesses
I could miss some important details but I think that the model is not correct. Looking at the formula of f(\tau) on page 4, there might not exist any neurovectors from the train set which have the same value at any feature. Take, for instance, a training set composed of two 2D points (1,1) and (2,2). If we want to make prediction for point (3,3), then what is returned by f(\tau)? 

Even if the above could be corrected the method looks very similar to k-NN approach. Therefore I do not see much novelty in this method.

Finally, the evaluation is below the standards of ICLR: 3 simple datasets and only shallow (and one MLP) baselines is not sufficient. Even with strong novelty, the method has to be evaluated on more examples.

### Questions
The authors could explain points (1) from the weakness section. Moreover, they also should elaborate on the connections with k-NN.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper presents a novel supervised learning method and reports its accuracy on three small datasets. Unfortunately, the paper fails to point out that the new method is a variation of k-nearest neighbor with k=1.

### Strengths
The central idea is interesting and the experimental results are believable.

### Weaknesses
The major weakness is that the method proposed in this paper is not novel; it is a variation of k-nearest neighbor. Specifically, Equations 6 and 7 say that the predicted label of a test example is the label of the training example with maximum count(NV) score. The score of a training example is the number of its feature values that equal the value of the same feature in the test example.

The predicted label is the label of the single nearest (most similar) neighbor of the training example, where similarity is measured as the number of identical feature values.

Section 3.3 provides a method for editing the training set by upweighting examples that provide correct predictions. A conceptually similar idea is proposed by Wilson, D.L. (1972) Asymptotic properties of nearest neighbor rules using edited data. IEEE Transactions on Systems, Man, and Cybernetics, 2(3), 408-421.

Other weaknesses:

Equation 1 says that features are real-valued but then Section 3.2.1 requires exact matches, which is not sensible for real numbers.

Equations 8 and 9 are purely heuristic, so it is not justified to call the method energy-based.

The experiments are insufficient: on only three small datasets. The results in Table 2 are not impressive: the new method does not yield systematically better accuracy.

The paper claims that Python dictionary lookups have time complexity. This is true only in the average case, and the worst case is O(n).

The FLOPS discussion on page 8 is pointless because the datasets are all small. 10^10 FLOPS is less than a second on a GPU nowadays.

### Questions
No specific questions.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces a **“neurovector”** paradigm for tabular learning that avoids backpropagation and trainable weights. Each training row is stored as a neurovector made of tokens from feature–value pairs; at inference, a test instance is tokenized the same way, candidate neurovectors are retrieved by token overlap, and the prediction is taken from the **most-overlapping** candidate. Formally, for test instance (x_j) with tokens (\tau_{j,l}), let (C_j) be the set of training neurovectors that share at least one token with (x_j). Let (M(NV,j)) be the number of shared tokens between a candidate (NV \in C_j) and (x_j). The method predicts
$$
m=\arg\max_{NV\in C_j} M(NV,j), \qquad \hat{y}_j = y_m .
$$

Ties are broken by an **energy** score. For classification:
$$
E(NV)=\frac{(s(NV))^2}{u(NV)} ,
$$
where (s(NV)) is the number of past correct uses of (NV) and (u(NV)) is the total uses. For regression:
$$
E_{\mathrm{reg}}(NV)=\frac{(s(NV))^2}{u(NV)} ,\exp!\left(-\alpha,\mathrm{MAE}(NV)\right).
$$

The approach aims to be **interpretable** (explicit token overlaps) and **efficient** (no gradient steps; create-on-error storage). On several UCI/Kaggle datasets, the authors report competitive accuracy compared to standard ML/DL baselines with reduced computational cost.

### Strengths
* **Simplicity & interpretability:** Prediction follows transparent token overlaps; energies provide per-instance diagnostics.
* **Gradient-free training:** Create-on-error storage avoids backpropagation/hyperparameter sweeps, attractive for low-resource settings.
* **Clear, reproducible core:** Retrieval and tie-breaking rules are explicit; basic complexity can be reasoned about via hash lookups and candidate ordering.
* **Potential efficiency:** If storage/candidates remain small, inference could be fast and memory-light in practice.

### Weaknesses
* **Limited evaluation:** Only a few datasets; no multi-seed cross-validation; several strong tabular baselines are missing (CatBoost/LightGBM/XGBoost, TabPFN, FT-Transformer); statistical tests and average-rank analyses are absent.
* **Tokenization brittleness:** Using **exact numeric values** as tokens risks near-zero overlap; discretization/quantization schemes (or similarity metrics) are not explored.
* **Compute claims unclear:** FLOP comparisons are indirect; no **wall-clock**, **RAM footprint**, or scaling curves vs. dataset size/feature cardinality; unclear training vs. inference accounting.
* **Protocol clarity:** Split definitions and tuning budgets per model are not consistently documented; potential for selection bias.
* **Theory gap:** No bounds on error, storage growth, or retrieval accuracy; the energy function lacks principled justification.

### Questions
1. **Numerical features:** How do you handle continuous values? Please report results with **binning/quantization** (e.g., equal-width, quantile, learned discretizers) and analyze sensitivity.
2. **Evaluation protocol:** Which split strategy is final (ratios, seeds)? Please provide **mean±std over 10–30 random splits** per dataset and **significance tests**.
3. **Compute & memory:** Report **wall-clock**, **RAM** (dictionary size vs. (|\text{train}|)), and scaling of candidate set size (m) with (|\text{train}|) and feature cardinality. Add **Pareto plots** (accuracy vs. compute/memory).
4. **Baselines:** Include **CatBoost/LightGBM/XGBoost** and recent deep tabular baselines (**FT-Transformer, TabPFN**) on a **larger benchmark suite** (10–20 datasets) with **average ranks**.
5. **Ablations:** (a) exponent in (\text{success}^2/\text{use}); (b) remove energy or replace with a **learned** tie-breaker; (c) **store-all** vs. **store-on-error**; (d) robustness to **label noise** and missing values.
6. **Collision control:** How are rare categories/high-cardinality handled? Any hashing scheme or token pruning?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a novel learning paradigm for tabular data that replaces backpropagation with energy propagation in vector spaces. Instead of weight updates, the model encodes information through interconnected nodes and vector relationships, aiming for higher interpretability and computational efficiency.  Experimental results demonstrate the effectiveness of this method.

### Strengths
1. The paper presents a well-motivated idea. Transforming tabular data into vectorized or text-like representations to make them compatible with large language models (LLMs). This direction is timely and meaningful, as it moves beyond conventional tree-based models toward architectures that can leverage foundation models.
2. The paper is clearly written and well-structured, making the proposed approach easy to follow and conceptually accessible.

### Weaknesses
1. The experimental evaluation is rather limited in scope. The paper includes only a few datasets, and the results in Table 2 are not convincing. For instance, on Breast Cancer, the proposed method performs comparably to the baseline; on Absenteeism at Work, results are reported as N/A; and Red Wine Quality is a small, non-representative dataset. To substantiate the claimed advantages, additional experiments on more diverse and large-scale tabular datasets are necessary. Moreover, the paper omits comparisons with strong deep learning baselines specifically designed for tabular data, such as FT-Transformer, TabNet, or NODE. Including these methods would provide a more meaningful and fair evaluation of the proposed approach, particularly in assessing its scalability and competitiveness against modern deep architectures.
2. In the experiments, does “Gradient Boosting” refer to XGBoost, Gradient Boosting Decision Trees (GBDT), Gradient Boosting Regression Trees (GBRT), or another variant of the Gradient Boosting family? Please clarify which specific implementation or library was used, as different versions can differ substantially in optimization strategy, regularization, and performance.

### Questions
see above

### Soundness
2

### Presentation
3

### Contribution
2
