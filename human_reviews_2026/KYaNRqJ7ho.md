# Z0-Inf: Zeroth Order Approximation for Data Influence

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Influence-functions are a popular tool for measuring the impact of a training point on a model’s prediction; in particular, self-influence, which quantifies the influence of a training point on itself, has found many uses such as data selection and outlier detection. However, the use of influence functions has been severely limited in modern models such as LLMs due to their low accuracy or high computational
cost: most existing algorithms are either highly inaccurate, or require computing gradients or approximations to inverse Hessians, which can be prohibitive for large models. In this work, we introduce a highly efficient zeroth-order approximation to data influence that requires only a fraction of the time and memory footprint of previous methods and is applicable to both differentiable and non-differentiable loss functions. We demonstrate that in addition to its computational advantage, our method delivers superior accuracy in estimating self-influence and comparable or better accuracy in estimating train-test influence for fine-tuned large language models, paving the way for broader and more practical application of influence-function techniques in state-of-the-art AI systems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Z0-INF, a zeroth-order approximation for data influence that bypasses expensive gradient computations by using finite difference directional gradients across model checkpoints. This makes it much faster and less memory-intensive while being more accurate for large language models.

### Strengths
Broad applicability: As the method does not require differentiable loss functions, it can be applied to black-box or non-standard loss settings, which extends the toolbox for both theory and practice.

Substantial computational efficiency gains: Z0-INF leverages finite-difference (zeroth-order) information to dispense with gradients and Hessian inverses, yielding significant reductions in both memory and runtime.

### Weaknesses
Potential missing ablations for checkpoint interval and layer selection. The paper claims computational savings by only considering last-layer parameters; however, it does not report how performance degrades or improves with the inclusion of different subsets of model parameters. Similarly, the trade-off between number of checkpoints and estimation quality is not studied.

### Questions
To what extent does the selection of parameters (e.g., only last-layer vs. selected layers or all parameters) affect influence estimation? Is there a noticeable gain in accuracy for higher parameter inclusion, and if so, does it offset the computational gains?

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
4

### Summary
This paper introduces Z0-Inf, a fast zeroth-order method to estimate data influence without gradients or Hessians. It achieves higher accuracy and lower cost than baseline methods, demonstrating its effectiveness.

### Strengths
1. The paper presents both theoretical analysis and experimental results.

2. The proposed zeroth-order method demonstrates impressive computational efficiency and scalability.

### Weaknesses
1. The motivation could be strengthened and validated by providing comparisons with true influence.

2. The self-influence formulation appears very similar to [1].

3. The Spearman correlation results in Section 4.3 are relatively low.

4. The writing can be made more consistent. For example, standardizing equation references ("Equation (4)" vs. "6" in line 427) to improve readability.

5. Figure 3 could be improved by aligning the bars and x-axis more precisely and using clearer visual groupings to better distinguish between datasets.

6. There is a missing citation in Line 113: "Lastly ? proposes…".

```
[1] Chhabra, Anshuman, et al. "Outlier Gradient Analysis: Efficiently Identifying Detrimental Training Samples for Deep Learning Models." ICML (2025). 
```

### Questions
Please see the Weaknesses part.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes **Z0-Inf**, a zeroth-order approximation method for estimating data influence, particularly self-influence and train-test influence, in modern machine learning models, including large language models (LLMs). The key insight is that first-order gradient-based approximations (e.g., TracIn) become increasingly inaccurate and computationally expensive as model size grows. Instead, Z0-Inf bypasses gradients entirely by using finite-difference directional gradients derived from checkpoint losses, enabling both faster computation and improved correlation with the “remove-and-retrain” baseline (approximated via Subsample-and-Retrain, SSRT). The authors also introduce a variance-based estimator for self-influence that avoids high-dimensional vector operations. Empirical results on vision models and fully fine-tuned LLaMA-3 8B across reasoning and instruction-tuning datasets show that Z0-Inf achieves higher Spearman correlation with SSRT than TracIn and LoGra, while being significantly more efficient in both time and memory.

### Strengths
1. **Practical relevance**: The method directly addresses a critical bottleneck in applying influence functions to LLMs, computational intractability by eliminating the need for gradients or Hessian approximations.  
2. **Strong empirical validation on LLMs**: Unlike many prior works limited to small models, the paper evaluates on fully fine-tuned 8B-parameter LLMs, demonstrating clear advantages in both accuracy (correlation with SSRT) and efficiency.  
3. **Theoretical grounding**: The zeroth-order formulation is well-motivated by the observed breakdown of first-order approximations in large models (Fig. 1b), and the variance-based self-influence bound is a clever simplification.  
4. **Algorithmic optimizations**: Techniques like Gram matrix precomputation and offline loss caching enhance practicality without compromising the core idea.  
5. **Reproducibility**: Experimental details (datasets, models, SSRT setup, checkpoint usage) are clearly specified, and code release is promised.

### Weaknesses
1. **Limited experimental scale**: While LLM experiments are a strength, the datasets used (GSM8K, MATH-500, Dolly) are relatively small (thousands to tens of thousands of examples). It remains unclear how Z0-Inf scales to *large-scale* fine-tuning scenarios (e.g., millions of examples), which are increasingly common in LLM applications. The reliance on SSRT—which itself becomes infeasible at large data scales—further limits the scope of validation.  
2. **Train-test influence results are modest**: The gains for train-test influence are less consistent than for self-influence (e.g., TracIn slightly outperforms Z0-Inf on GSM8K), and the paper acknowledges high noise in this setting. The discussion of why zeroth-order helps here is somewhat superficial.  
3. **Checkpoint dependency**: The method still requires storing multiple model checkpoints, which may be prohibitive in extremely large-scale training (though less so than full gradients). The sensitivity to the number and spacing of checkpoints is not thoroughly explored.  
4. **Lack of downstream task evaluation**: While correlation with SSRT is a reasonable proxy, it would strengthen the paper to show concrete benefits in applications like data pruning, outlier detection, or targeted fine-tuning (as hinted in the conclusion but not demonstrated). 
5. **Limited Novelty**: This work represents only a modest advancement over TracIn. Moreover, zeroth-order methods for influence estimation have already been explored in prior literature, for example,  The Mirrored Influence Hypothesis: Efficient Data Influence Estimation by Harnessing Forward Passes

### Questions
See weakness

### Soundness
3

### Presentation
2

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
This paper proposes a **zeroth-order influence estimator (Z0-INF)** that replaces gradient-based stepwise contributions with simple **forward-loss differences between adjacent checkpoints** along the training path. By aligning influence with the **actual update direction** taken during training, the method aims to capture what truly changed the model’s behavior without using backpropagation. The authors evaluate both **self-influence** (per-example impact on the trained model) and **train→test influence** (how a training example affects a target/test example), and provide ablations on checkpoint selection and practical implementation choices. Empirically, Z0-INF delivers strong self-influence results and competitive train→test performance, while being easier to run on large models since it only requires forward passes.

### Strengths
- **Clear idea, simple to implement:** Replacing gradients with loss differences along the actual training updates is conceptually neat and engineering-friendly.
- **Optimizer- and path-aware:** By tying scores to the direction the optimizer actually moved, the estimator reflects what happened in training rather than assuming perfect first-order linearization.
- **Forward-only practicality:** No backward/Hessian; easier scaling to large models and long training runs.

### Weaknesses
- **Compare Z0-INF with Hessian-free Inner-Product (TracIn) and add a phase view.**  
  It would help readers if the paper **explicitly contrasts** Z0-INF with the Hessian-free inner-product family (TracIn-style methods) and gives a **phase-aware** view (early, non-converged vs. late, near-converged).  
  *My view (for context only):* Z0-INF is likely stronger **early** when steps are larger and curvature is strong, while Inner-Product (TracIn) may fit better **near convergence**. A related reference that may help frame this comparison is **“Revisit, extend, and enhance hessian-free influence functions”**; citing and contrasting with that line could make the differences clearer.

- **Optimizer / schedule sensitivity.**  
  Because Z0-INF follows the **actual update direction**, results can vary with **AdamW vs. SGD**, momentum, weight decay, and learning-rate schedules. Please add controlled ablations. A simple step-level diagnostic (e.g., how well each method tracks loss changes under different optimizers) would make the conclusions more convincing.  
  *Expectation:* Under **AdamW-style preconditioning**, Z0-INF may outperform Inner-Product (TracIn) more clearly; with **small-step SGD**, the gap may shrink.

- **Checkpoint density & neighborhood averaging.**  
  Influence quality likely depends on **which checkpoints** are kept and **how many neighbors** are averaged. Please include **sensitivity plots** (number/placement of early checkpoints; neighborhood size) and provide **recommended defaults** for practical use.


If these concerns are addressed, I’m willing to raise my score.

### Questions
Same with weakness

### Soundness
3

### Presentation
3

### Contribution
3
