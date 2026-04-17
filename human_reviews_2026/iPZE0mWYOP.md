# Principal Components for Neural Network Initialization: A Novel Approach to Explainability and Efficiency

- Decision: Reject
- Scores: 4, 8, 2, 4

## Abstract
Principal Component Analysis (PCA) is a commonly used tool for dimension reduction and denoising. Therefore, it is also widely used on the data prior to training a neural network. However, this approach can complicate the explanation of eXplainable Artificial Intelligence (XAI) methods for the decision of the model. In this work, we analyze the potential issues with this approach and propose Principal Components-based Initialization (PCsInit), a strategy to avoid this complexity by initializing the first layer of a neural network with the principal components. We will illustrate that when this first layer (which is initialized by the principal components) is frozen, PCsInit is equivalent to applying PCA to the input and then training on the principal components, while being simpler to explain. In addition, we propose two variants PCsInit-Act (to incorporate nonlinearity) and PCsInit-Sub (for a more scalable approach), 
and show that the proposed techniques possess desirable theoretical properties.  Moreover, as will be illustrated in the experiments, such training strategies can also allow further improvement of training via backpropagation compared to training neural networks on principal components.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Principal Components-based Initialization (PCsInit) and two extensions (PCsInit-Act and PCsInit-Sub) as new weight initialization methods for neural networks, inspired by the structure of Principal Component Analysis (PCA). The central claim is that initializing the first layer with PCA-derived principal components leads to better optimization stability, improved explainability for XAI methods (e.g., SHAP, LIME), and potentially better generalization compared to models trained directly on PCA-reduced data (PCA-NN). The authors provide theoretical derivations regarding Hessian conditioning, Lipschitz continuity, and robustness to noise, as well as experiments on several datasets (Heart, Parkinson, Micromass, HTAD, MNIST, etc.) to support their claims.

### Strengths
[1] The paper explores an original direction in connecting PCA with neural network initialization rather than data preprocessing, which could have practical implications for explainable AI.
[2] The presentation of theoretical properties (e.g., conditioning, Lipschitz constant) provides mathematical intuition for the potential advantages of PCA-based initialization.
[3] The methods are computationally lightweight, potentially easy to reproduce.

### Weaknesses
[1] The paper claims that PCsInit “improves interpretability” compared to PCA-NN, yet all empirical evidence focuses on SHAP value visualization differences. There is no quantitative or formal evaluation of interpretability, nor a clear metric to support that the method meaningfully improves explanation fidelity.
[2] The figures show SHAP values for PCA-NN and PCsInit but do not explain why one visualization is “better.” The visual difference in SHAP plots does not constitute a formal improvement in interpretability. Moreover, SHAP itself is an approximation, and no quantitative assessment is provided.
[3] The neural network architectures are only vaguely described (“five layers”) without specifying activation types, layer widths, or normalization techniques.
[4] The datasets used are relatively small and scalability to large models or complex data remains untested.
[5] The paper mentions in Section 4.2 that the first r components are retained, but does not specify how r is chosen or whether variance thresholds are used. Since the number of components directly affects initialization rank and model expressivity, an ablation varying r would clarify sensitivity and optimality.
[6] The paper omits architectural and training details crucial for reproducibility, such as activation functions, learning rate, batch size, and optimizer type.

### Questions
[1] Does freezing and unfreezing the first layer meaningfully affect convergence compared to continuous fine-tuning?
[2] Most datasets are small tabular ones (Heart, Parkinson, Micromass). What motivated their choice, and have any larger or more complex datasets been tested to confirm scalability?
[3] The SHAP and LIME plots are visually different, but the evaluation is qualitative. Could the authors provide numerical interpretability metrics to support the claim that PCsInit improves explainability?
[4] The motivation for these two variants is described conceptually but not evaluated in depth. What specific empirical improvements do they bring relative to base PCsInit?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors propose a principal components-based initialization strategy to incorporate PCA into the first layer of a neural network via initialization of the first layer in the network with the principal components.

### Strengths
- The paper is well organized and written.
- The contribution is theoretically interesting
- The proposed method named  PCsInit is detailed and reproducible.
- Experiments show the performance of PCsInit on the Heart dataset.

### Weaknesses
Comparison should be extended with more previous algorithms.

### Questions
Is it possible to design a sparse version of the PCsInit to reduce computational time?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
They propose a new machine learning framework called Principal Components–based Initialization (PCsInit), which initializes a network’s first layer with PCA loadings and adds two variants (PCsInit-Act and PCsInit-Sub) to preserve interpretability while enabling nonlinear modeling.
The approach aims to make feature attributions more transparent than PCA-preprocessing pipelines and, across real-world classification/regression benchmarks, matches or outperforms PCA-NN baselines with simpler explanations.

### Strengths
**(S1).** It removes explainability constraints common to PCA-based baselines by tying the first-layer weights directly to principal components, preserving a clear input - PC - prediction mapping.

**(S2).** It introduces a practical variants—PCsInit-Act—to better capture nonlinear patterns adding activations function after PC.

### Weaknesses
**(W1)**. The first-layer weights initialized with PCA are free to drift during training, so $W^{\text{optimal}}_1$ cannot be said to span the same subspace as the top-$r$ eigenvectors; to substantiate the interpretability claim, please either (i) prove or measure subspace closeness ( e.g., principal angles or $|W^{\top}_1 W_r|_F$ ) across epochs, or (ii) show that projection by $W^{\text{optimal}}_1$ preserves variance essentially as well as $W_r$ on data.

**(W2).** Section 3’s bounds appear to concern estimation/generalization error rather than optimization dynamics; they do not formally imply faster or more stable convergence. If efficiency is a key claim, provide optimization guarantees (e.g., convergence rate lower bounds or excess-risk upper bounds as a function of $(n, r)$ ) that specifically distinguish PCsInit from PCA-preprocessing baselines.

**(W3).** To justify the method as “explainable,” feature importance produced by PCsInit should be directionally consistent with the feature importance from other XAI methods or that it reliably identifies truly important variables. To this end, go beyond PCA-NN and experimentally compare feature-importance rankings against multiple XAI baselines, or verify on synthetic datasets with known ground truth that significant variables are accurately recovered.

**(W4).** The computational-efficiency story is under-evidenced. Please add training time vs epoch curves for PCsInit, PCA-NN, and PCsInit-Sub, include PCA/SVD setup cost; current experiments do not isolate whether PCsInit-sub actually reduces total training time for comparable accuracy.

**(W5).** Because interpretability hinges on the PCA semantics of the first layer, quantify attribution drift: compare per-PC attributions at initialization vs. at convergence, and report whether explanation faithfulness degrades as $W_1$ departs from the PCA subspace (e.g., sparsity/orthogonality metrics, input reconstruction error through $W_1$).

### Questions
**(Q1).** Could you standardize the rule for choosing the first-layer width? In this paper, $r$ is usually used as a dimension of reduced space. The text alternates between “retain n components” and “retain r% of variance,” while the appendix mentions “at least 95%”. 

**(Q2).** In Algorithm 3 you list “ReLU, He, …” as the activation after the PCA-initialized layer, but “He” is an initialization, not an activation. Which activations can be actually used in PCsInit-Act?

**(Q3).** How are $J(\cdot)$ and the Hessian matrix defined for $r>1$? In Theorem 3.1, you set $J(W)=\tfrac{1}{2}\lVert W^{\top}X - Y\rVert^{2}$, but when $r>1$ and $W_r\in\mathbb{R}^{d\times r}$ is substituted, a dimensional mismatch occurs. Therefore, it seems necessary to review the statement of Theorem 3.1 overall.

**[Minor]**
1. Expression (LaTeX) :  In p.3, “so that **n** components is retained” → “so that **$n$** components is retained” ?

2. Expression (grammar): In p.7, “the results of performance loss and accuracy **is** reported” → “the results of performance loss and accuracy **are** reported ” ?

3. Expression (grammar): In p.7,  “**performances** of the proposed techniques” → “**performance** of the proposed techniques” ?

### Soundness
2

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
The paper proposes PCsInit, which initializes the first linear layer with the top-r PCA directions, along with two variants. PCsInit-Act adds a nonlinearity after that layer; PCsInit-Sub computes PCs on a data subset to cut cost. The idea aims to keep PCA “inside” the network so explanations (e.g., SHAP/gradients) can be computed directly in input space without back-projecting from PCA features. Claimed theory covers Hessian conditioning, Lipschitz constants for the first layer, and noise propagation; experiments compare against “PCA-NN” across several tabular datasets and MNIST (with noise).

### Strengths
The paper’s main strength is a simple, practical idea with clear motivation.  Initialize the first layer with data PCs so the network “absorbs” a common preprocessing step while keeping explanations in the input space. This yields conceptual originality relative to the usual PCA before NN pipeline by unifying feature extraction and learning, and the two variants (subset PCA for scalability; post-activation for expressivity) make the recipe broadly usable. The approach is easy to implement, cheap to run, and aligns with a sensible inductive bias (early directions capture most of the variance, improving stability and noise handling) that is corroborated by consistent trends across several tabular datasets and a noisy-image toy case. The method is straightforward to reproduce, and the interpretability of SHAP/gradient attributions does not require back-projection. If validated more broadly, this could become a lightweight practice for tabular models (and potentially inspire analogous “in-network PCA”  in other modalities), offering a small but functional gain in robustness and a cleaner interpretability pipeline

### Weaknesses
The paper’s main weaknesses are (i) theoretical inconsistencies, the noise propagation and norm-preservation rely on ambiguous matrix shapes/orthogonality, and in places are improper or rely on restating standard spectral-norm/Lipschitz facts. (ii) limited and unconvincing experiments. Small tabular sets plus a toy vision case, key results relegated to the appendix, and no confidence intervals, multi-seed variance, or significance tests. (iii) weak baseline coverage—comparisons omit strong standards such as raw-input MLP/CNN with modern init+normalization, whitening/ZCA, random orthogonal/SVD inits, and representation-learning baselines.  (iv) missing ablations/sensitivity and no systematic study of the number of PCs, freeze duration, learning-rate/regularization, subset-PCA sampling, or the PCsInit-Act variant. (v) unclear scope for spatial data and flatten-then-PCA discards structure, so claims for vision models are unsupported; (vi) unmeasured interpretability, no quantitative evaluation of attribution faithfulness/stability. And (vii) reproducibility/presentation gaps, no code, typos/notation. The authors should expand the benchmarks with full tables and CIs across many seeds, add strong baselines, and include targeted ablations. Either limit claims to tabular data or provide a conv-friendly variant with CIFAR/ImageNet-style results. Evaluate attribution faithfulness/stability, and release code and standardize notation.

### Questions
1. The noise-handling and “norm preservation” claims seem to rely on mixed assumptions. Please state the exact assumptions (how inputs are standardized, what “orthonormal” means here, and the shape of the matrix) and provide a small synthetic check (e.g., Gaussian noise through your first layer) that reproduces your claims.

	2. Do you center/standardize features before computing PCs and during training? If not, why? Please clarify the full pipeline so others can reproduce identical PCs and results.

	3. Number of PCs (r): How is r chosen (fixed, variance-explained threshold, tuned on validation)? Show sensitivity curves for accuracy vs. r and discuss the bias–variance trade-off.

	4. How long is the first layer kept frozen, and how sensitive are results to this choice? Please add an ablation sweeping freeze duration and show convergence/learning-curve plots.

	5. Add standard baselines, (i) raw-input MLP/CNN with modern initialization + BN/LN, (ii) whitening/ZCA pipelines, (iii) random orthogonal/SVD initializations, (iv) a simple representation-learning baseline (e.g., autoencoder). 

	6. Run more seeds and report means ± confidence intervals and significance tests. For tabular, consider a standard suite. Include CIFAR-10/100 with convnets (or explain why the method is tabular-only).

	7. The paper argues for cleaner attributions because explanations stay in the input space. Please quantify this using standard tests (faithfulness via deletion/insertion/AOPC, stability across seeds) and show whether your method alters conclusions from SHAP/IG relative to PCA-NN and raw-input models.

	8. How are subsets sampled, and what is the accuracy vs. compute trade-off? Compare to streaming/incremental PCA and show when subset PCA breaks down (e.g., rare but important features).

	9. Flattening discards spatial structure. Either narrow the claim to tabular data or provide a conv-friendly variant (patch- or channel-wise PCA seeding) and results that demonstrate actual vision benefits.

	10. What is the overhead of computing PCs relative to training time? Please release code, add a training-details table (data prep, hyperparams, hardware), and move key quantitative results from the appendix into the main paper.

### Soundness
1

### Presentation
2

### Contribution
2
