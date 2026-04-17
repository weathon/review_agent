# Jacobian Aligned Random Forests

- Decision: Accept (Poster)
- Scores: 4, 4, 2, 6

## Abstract
Axis-aligned decision trees are fast and stable but struggle on datasets with rotated
or interaction-dependent decision boundaries, where informative splits require linear combinations of features rather than single-feature thresholds. Oblique forests
address this with per-node hyperplane splits, but at added computational cost.
We propose a simple alternative: JARF, Jacobian-Aligned Random Forests. Concretely, we fit a random forest to estimate class probabilities or regression outputs,
compute finite-difference gradients with respect to each feature, form an expected
Jacobian outer product/expected gradient outer product, and use it as a single
global linear preconditioner for all inputs. This preserves the simplicity of axisaligned trees while applying a single global rotation to capture oblique boundaries
and feature interactions that would otherwise require many axis-aligned splits to
approximate. On tabular benchmarks, our preconditioned forest matches or surpasses oblique baselines while training faster. Our results suggest that supervised
preconditioning can deliver the accuracy of oblique forests while keeping the simplicity of axis-aligned trees.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors introduce JARF, a method that linearly transforms inputs before fitting a random forest. The linear transform is derived from an expected Jacobian outer product (EJOP), which has been developed and used in prior work. When applied to Random Forests, the authors find that JARF achieves performance results matching existing oblique tree baselines while reducing computation time.

### Strengths
- The authors study an interesting problem
- The authors approach is intuitive
- The paper is well-written and clear throughout

### Weaknesses
- Main weakness: My understanding of JARF is that it requires first fitting an RF as a step in computing the EJOP, followed by fitting the RF on the conditioned input data. Why is the fitting of the first RF not taken into account in the Efficiency and Compute subsection (or if it is, why is JARF's compute time not atleast 2x that of fitting RF)? This computational efficiency claim is central to the authors' message and should be taken seriously.
- The paper's novelty is somewhat limited, as the EJOP is already defined and used in prior work (although the authors to swap a kernel regression estimator for RF when estimating it)

### Questions
- It could be nice to understand how performance varies as the size of the subsampled dataset used for computing the EJOP varies.
- How did the authors select the 10 datasets they used for evaluation? It would be nice to use a standard suite of datasets (e.g. TabArena or PMLB) to avoid any possibility of biased dataset selection

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose a **supervised pre-processing step** applied to the feature matrix used as input for a random forest or other **axis-aligned predictors**.  
The goal is to enable more flexible splits in the input space, thereby improving the performance of axis-aligned classifiers.  

Their empirical evaluation shows that the proposed method performs **on par with SPORF**, and slightly better than **XGBoost (XGB)** and **Random Forest (RF)**.

### Strengths
- Interesting idea to transform the data in a supervised manner before training.  
- The method should be relatively fast to run, better on the compute-performance tradeoff than SPORF.

### Weaknesses
- The reported improvements in performance are **not particularly meaningful**.  
- Evaluating only on **10 real datasets** is not sufficient to claim generality or robustness.  
- There is **no discussion** on how to tune hyperparameters for the proposed method.

### Questions
1. The proposed method appears similar to a **one-step RFM** [1] for classification.  
   Can the authors clarify the conceptual and mathematical connection between their procedure and RFM?  
2. Does the matrix **H** have to be derived from the **same model** used for prediction?  
   If not, the authors should provide guidance on how to select and pair the transformation and prediction models.  
3. Can **JARF** be applied to **XGBoost** as well, or is it restricted to random forests?  
4. The authors mention that the *electricity*, *magic*, and *letter* datasets have **complex decision boundaries**.  
   Can they explain why these particular datasets were chosen to illustrate this property?  
5. Can the authors provide **details about the datasets** used in the experiments (e.g., number of samples, features, task type)?  
6. Can the authors provide **guidance on hyperparameter tuning** or recommendations for practical implementation?


[1] Radhakrishnan, A., Beaglehole, D., Pandit, P., & Belkin, M. (2024).  
*Mechanism for feature learning in neural networks and backpropagation-free machine learning models.*  
**Science**, 383(6690), 1461–1467.

### Soundness
2

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
4

### Summary
The paper proposes Jacobian-Aligned Random Forests (JARF): learn one global supervised linear transform $H$ from the Expected Jacobian Outer Product (EJOP) of class-probability gradients (estimated via finite differences of an RF surrogate), then train a standard axis-aligned RF on the transformed features $XH$. This aims to capture rotated/interaction directions so axis-aligned splits behave like shared oblique hyperplanes, while keeping RF’s simplicity. Experiments on 10 tabular classification datasets and controlled synthetic rotations show competitive accuracy vs oblique forests with modest overhead, plus ablations supporting the EJOP step and implementation choices.

### Strengths
1. It is a simple and one-pass method, plugging the EJOP to perform initial feature transformation. This enables direct application on RF in the subsequent step.

### Weaknesses
1. The proposed method clearly lacks novelty, which does not match the conference standard. It is mainly based on a known paradigm EJOP. The paper’s main change is estimating EJOP with a surrogate RF and finite differences. This feels incremental relative to existing supervised/oblique projection lines rather than a new learning principle.

2. The estimator uses finite differences of RF class probabilities to approximate Jacobians (Sec. 3.6), but the analysis later assumes $f\in\mathcal C^3$ with bounded third derivatives (Assumption A1), which is incompatible with piecewise-constant tree ensembles. The text informally argues that ensemble averaging “smooths” predictions, but the formal guarantees hinge on smoothness that the surrogate does not satisfy. And usually, such smoothness improvements are stated when comparing to a single decision tree. This creates a theory–practice gap in the central estimator.

3. A single matrix $\hat H$ is shared across the entire forest (Sec. 3.4–3.5). This seems to reduce the diversity of the allowed splits. I would encourage the author to make this part more flexible and check the performance.

4. There is a concern about the fairness of the experimental setting. For instance, it was stated that XGBoost is run with a “small shared grid” only; RF fixed at 200 trees; oblique baselines appear near-default. More detailed hyperparameter tuning is necessary.

5. The real-data suite is 10 classification tasks, many with d ≤ 60  and moderate n. More experiments with large-scale datasets (either n or d) would be helpful for the evaluation.

### Questions
I have no further questions.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces JARF, a method to enhance standard axis-aligned decision forests by applying a single global supervised linear preconditioner to the input features before training. This makes the forest behave like an oblique forest without changing the RF training algorithm.


The key idea is:

1. Fit a surrogate probabilistic classifier (a random forest) on the original data.
2. Estimate gradients of the class probabilities with respect to each input feature on a subsample of points.
3. Construct the Expected Jacobian Outer Product (EJOP) matrix. 
4. Use this EJOP estimate (with light regularization and normalization) as a **global linear transform** $\hat H$ and train a standard axis-aligned Random Forest on transformed features ($X$$\hat H$).

### Strengths
- The core idea is simple, clean, and easy to implement on top of existing RF code. 
- The method is well-motivated and clearly positioned between axis-aligned forests and oblique trees, leveraging prior EJOP work. 
- Experiments are solid: realistic baselines (RF, XGBoost, RotF, CCF, SPORF), multiple datasets, plus timing comparisons. 
- The mechanism analysis (alignment of oblique split normals with EJOP subspace) and ablations give good insight into why it works.

### Weaknesses
- The method heavily depends on the quality of probability estimates from the surrogate RF used to build EJOP, which is not deeply analyzed. 
- It only evaluates standard tabular classification datasets and does not explore regression or more challenging/high-dimensional settings. - There is no direct comparison to simpler global projections (e.g., PCA, LDA) used once before RF. 
- The novelty is mostly in combining known pieces (EJOP + RF + preconditioning) rather than introducing fundamentally new theory.

### Questions
- How sensitive is JARF to the choice and calibration quality of the surrogate model? Would using XGBoost or a small NN as the surrogate improve EJOP and performance?
- Are there datasets or regimes where JARF clearly underperforms CCF or SPORF, indicating that a single global transform is insufficient?
- Have you tried an EGOP-based variant for regression?
- Your experiments use 10 classic UCI/OpenML-style tabular datasets; have you evaluated JARF on larger-scale or more modern industrial tabular dataset?

### Soundness
3

### Presentation
2

### Contribution
3
