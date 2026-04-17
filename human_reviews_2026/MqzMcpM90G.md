# BRep: Graph-structured Brain Representation Learning via Parametric High-order Dependence Measures

- Decision: Reject
- Scores: 2, 6, 2, 8, 4

## Abstract
The brain network plays an important role in diagnosing neurological disorders. Brain functional network construction often follows the hand-crafted Correlation Coefficients of blood-oxygen-level-dependent (BOLD) time series without any learnable components. At the same time, most efforts are made to the models that predict individual neurological disorders with the constructed brain network as input, such as graph neural networks. Unfortunately, the fixed brain network may lose critical information during construction and lead to difficulty in performance improvement, even with deliberately designed graph models. From this perspective, the current situation is similar to the machine learning community, i.e., hand-crafted features and learnable predictors, before the advent of representation learning.
In fact, the brain network can be regarded as a graph-structured learnable representation of the brain.  By drawing on representation learning, this paper presents the Brain Representation (BRep) learning problem. To this end, the widely used linear and nonlinear correlations are enhanced to be high-order, parametric, and learnable. The flexible brain representation makes the following predictor simple and leads the framework to possess an end-to-end characteristic. The framework is implemented by combining the parametric correlation and a TopK sparsification. Extensive evaluations demonstrate that the proposed BRep possesses superior performance, high efficiency, and interpretability. The source code is publicly available at https://anonymous.4open.science/r/BRep-demo/

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper proposes learning functional brain connectivity from fMRI time series via end-to-end backpropagation, rather than using fixed correlation measures. The model learns a parameter matrix O that defines pairwise connectivity as (x_i O)(x_j O)^T, from which an adjacency matrix is extracted via top-k sparsification. This learned brain network is then flattened and fed into an MLP for disorder classification. The authors argue this mirrors the shift from hand-crafted features to representation learning, enabling simpler predictors. Experiments on ABIDE and ADHD-200 datasets show competitive performance, and the learned connectivity patterns align with established neuroscience findings about autism.

### Strengths
- Framing  brain network construction as a representation learning problem aligns well with the spirit of deep learning.
- The paper is overall well written and the problem statement is well defined.
- The unification of linear and non-linear correlations through inner product in latent spaces is clever (section 2).
- The experimental section is thorough, including 14 baselines across GNNs, GTs, and NNs, with proper ablation studies and interpretability analysis.
- The differential connectivity analysis (Section 3.2) showing hyperconnectivity in ASD aligns with existing neuroscience literature, indicating promising interpretability and overall validating the approach.
- The approach enables simple downstream predictors (MLP) which might we valuable for high-throughput applications.

### Weaknesses
- Authors claim to learn high-order correlations, but what makes the interactions learned by the approach high-order? As far as I understand, what is learned is r_ij = (x_i O)(x_j O)^T, which is still a pairwise interaction between regions i and j. I suppose if there were multiple such layers, high-order interactions (that is, interactions between >2 regions) can be captured, but with a single layer that just doesn't happen, which makes me confused about the framing of the paper.
- Authors claim to achieve superior performance (abstract), but on ABIDE dataset, there are better methods, and on ADHD-200 dataset, the difference is within error bars.
- In my opinion, the contribution lacks noveltly sufficient for an ICLR contribution. The approach of learning adjacency matrices end-to-end and connecting it to functional connectivity is well-established [1,2]. Moreover, several included baselines (GAT, Graphormer, FBNetGen) already learn connectivity through attention or generation mechanisms. While these are compared numerically (Table 1), the paper does not acknowledge their learnable nature or explain how the proposed explicit parameterization via transformation matrix O differs from or improves upon these implicit learning mechanisms. 
- I personally believe the work would be a much stronger contribution if it focused on the learned patterns and a thorough comparison to other methods in the aspect (e.g. [1,2], or graph structure learning approaches such as classical graph VAE from Kipf et al.). It feels to me that the contribution is more suitable for a medical journal / conference where the interpretability results might be interesting to the practitioners.

[1] Zhdanov, Maksim et al. “Investigating Brain Connectivity with Graph Neural Networks and GNNExplainer.” 2022 26th International Conference on Pattern Recognition (ICPR) (2022): 5155-5161.
[2] Kan, Xuan et al. “Brain Network Transformer.” ArXiv abs/2210.06681 (2022): n. pag.

### Questions
- Would it be possible to compare learned FC patterns against other frameworks with learnable connectivity? 
- What makes method learn high-order interactions? Are not those just pair-wise that the method encodes?

### Soundness
3

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
In order to solve the problem of information loss caused by the dependence on fixed manual features (such as Pearson correlation coefficient) in the construction of brain functional networks, this paper proposes a graph-structured brain representation learning framework BRep. The core idea is to regard the brain network as a learnable graph structure representation, build a high-order, parameterized and learnable dependency metric (HDM module) by unifying and expanding linear/nonlinear correlation coefficients, and realize end-to-end training by combining TopK sparseness. On two benchmark data sets, ABIDE (autism diagnosis) and ADHD-200 (ADHD diagnosis), BRep is superior to most GNN, Graph Transformer and traditional neural network baselines in AUC and ACC, and it is interpretable (biological rationality is verified by differential brain network visualization) and universal (HDM module can improve the performance of other models). The paper also verifies the effectiveness of the key components of the framework through ablation experiments and parametric analysis, and the code is open to ensure reproducibility.

### Strengths
1. The linear (Pearson) and nonlinear (dCor, HSIC) correlation coefficients are unified, and the parameterized matrix approximates high-order dependence, which not only retains the modeling ability of complex brain connections, but also controls the computational complexity through dimension matching, giving consideration to performance and efficiency.
2. The end-to-end framework simplifies the downstream predictor (only MLP) and reduces the deployment difficulty.
3. The visualization results of differential brain network are consistent with the existing research conclusions related to autism (such as highlighting network hyperconnection), which is biologically interpretable and meets the needs of medical scenes.
4. HDM module can be flexibly integrated into existing models such as GCN and BrainNETTF, and its performance can be improved. It provides a general feature enhancement tool for brain network analysis, not limited to a single framework.

### Weaknesses
1. Only based on fMRI data verification, not extended to structural imaging (sMRI, DTI), molecular imaging (PET) and other multimodal data; Moreover, it is only aimed at the binary classification task of two kinds of neurological diseases, and lacks verification in multi-classification and regression tasks (such as disease severity assessment).
2. Although it is mentioned that HDM reduces the complexity through parameter approximation, it does not directly compare the speed/memory cost with the existing efficient high-order dependency modeling methods (such as lightweight Tensor-HSIC), and it is difficult to determine its applicability in large-scale brain networks (such as ROI > 500).
3.Differential brain network analysis only focuses on Top20 differential connections, and lacks quantitative statistical verification (such as replacement test) on why these connections are related to diseases.
4.The key hyperparameters, such as the optimal K value of TopK sparseness (80) and the module dimension of HDM (100), only verify the performance optimality through experiments, and do not explain the rationality of selection in combination with the biological characteristics of brain anatomical structure or functional partition, which reduces the domain adaptability of the method.

### Questions
Please refer to weaknesses.

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
5

### Summary
This paper proposes BRep, a framework for graph-structured brain representation learning, where the functional brain network is no longer constructed using fixed, hand-crafted correlation measures (e.g., Pearson, dCor), but instead learned end-to-end as a parametric, high-order dependence measure. The core idea is to treat the brain connectivity matrix as a learnable representation parameterized via a trainable matrix O that maps BOLD time series into a latent space where inner products define edge weights. This is combined with TopK sparsification and a simple MLP predictor, forming an end-to-end pipeline for neurological disorder classification. The authors evaluate BRep on ABIDE and ADHD-200 datasets, reporting competitive or superior performance over a wide range of GNNs, Graph Transformers, and neural baselines, along with interpretability analyses aligning with known neurobiological findings.

### Strengths
1. By replacing complex GNN/GT architectures with a learnable connectivity estimator + simple MLP, the method achieves SOTA performance with reduced architectural complexity.
2. The analogy to the pre-deep-learning era of “hand-crafted features + learnable predictors” is compelling and well-articulated.

### Weaknesses
1. The core technical contribution a ``parametric high-order dependence measure'' is mathematically equivalent to a learnable inner product in a linearly transformed space: $ r_{ij} = (\mathbf{x}_i \mathbf{O})(\mathbf{x}_j \mathbf{O})^\top$
where $\mathbf{O} \in \mathbb{R}^{D \times D}$ is a trainable matrix. This formulation is not high-order in any statistical sense (e.g., it does not involve third- or higher-order moments, cumulants, or tensor interactions). Instead, it is a classic instance of metric learning or representation learning via linear projection, widely used in contrastive learning, Mahalanobis distance learning, and even early deep learning architectures.

2. The paper conflates nonlinear correlation (e.g., dCor, HSIC) with low-order and rebrands a bilinear form as high-order, which is technically inaccurate and misleading. True high-order dependence measures, such as those based on U-statistics over $M$-tuples ($M \geq 3$) are explicitly mentioned but then abandoned due to computational cost. The proposed approximation via $\mathbf{O}\mathbf{O}^\top$ sidesteps the actual challenge and reduces the method to a simple, well-known paradigm.

3. The paper emphasizes  interpretability  and clinical relevance, yet provides no evidence of practical clinical value. The differential connectivity analysis (Fig. 3, Fig. 11) merely reproduces known ASD biomarkers (e.g., hyperconnectivity in the salience network, pSTS), which have been reported. This is post-hoc validation, not discovery.

4. No analysis of motion artifacts, scanner differences, or missing ROIs common in real clinical fMRI.

5. ABIDE contains 17 heterogeneous sites, yet the paper uses random splits without site-stratification. Performance may collapse under realistic multi-center conditions.

6. The model reports population-level ACC/AUC, but clinicians need calibrated, individualized risk scores with uncertainty quantification absent here.

7. The term ``graph-structured representation learning'' is presented as novel, but similar ideas appear in BQN (Yang et al., ICML 2025) and FBNetGen (Kan et al., MIDL 2022a).

### Questions
See above

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes BRep, a learnable high-order dependence estimator for constructing functional brain networks from BOLD fMRI time series. The key idea is to replace simple (non-)linear correlations with a learnable parametric estimator that can capture higher-order dependencies between the regions. Experiments on ABIDE and ADHD200 show gains over conventional baselines and also demonstrate improvements on GNN/GT/NN-based models as a plug-in HDM module.

### Strengths
- The motivation of the paper addresses an important issue in the field (assumptions of connectivity as simple correlations), and neatly formulates this issue as a research problem.
- The method is validated with extensive experiments, including performance benchmarks and neuroscientific interpretation.
- The method is theoretically explained throughout.

### Weaknesses
- While I do not see a major/general weakness, please refer to the Questions section for detailed comments.

### Questions
### Major
- I agree with the authors' statement (Fig. 1) that a well-defined high-order correlation connectivities require simpler predictors. Currently, the simple predictor is an MLP, which can still be optimized to learn complex relationships. Could the authors provide experimental results on linear approximators on $\mathbf{A}$? 
- For the analysis of the optimality of the dimension $D$ in Section 3.2., it would be more convincing if the optimal dimension follows the timeseries length for other data with longer timepoints. Could the authors check if the trend in Fig. 5 follows the timeseries length for data with longer timepoints, or remains optimal at $D=100$?
- Providing qualitative heatmap plots of the learned $\mathbf{A}$ matrix would be helpful to the readers.
- It seems that the trainable parameters within $\mathbf{O}$ include TopK within its backpropagating path, which is not explicitly differentiable. Please clarify.

### Minor
- Some abbreviations are defined multiple times throughout the paper.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper reframes brain functional connectivity as a **learnable, graph-structured representation** and proposes BRep, which replaces fixed correlation graphs with a **parametric high-order dependence estimator** trained end-to-end. Concretely, BOLD time series are mapped via a learnable matrix \(O\) to form \(Z=\hat{X}O\), and the adjacency is built as \(A=\sigma(\mathrm{Norm}(\mathrm{TopK}(ZZ^\top)))\). Across ABIDE and ADHD-200, BRep paired with simple MLP/BQN heads **outperforms or matches** GNN and Graph Transformer baselines; inserting the proposed HDM as a plug-in also improves existing models. The method further employs a denoising regularizer and reports sensitivity to the HDM dimension.

### Strengths
- **Conceptually clear reformulation.** Treating the connectivity graph itself as the representation, rather than a fixed input to a downstream GNN, motivates an end-to-end pipeline that simplifies the predictor without sacrificing accuracy.
- **Unified, parametric high-order dependence (HDM).** The paper connects linear/nonlinear correlations and proposes \(r^{[\mathrm{high}]}_{ij} = (x_i O)(x_j O)^\top\) as a learnable surrogate, followed by principled sparsification (TopK) and normalization.
- **Consistent empirical gains and plug-in utility.** BRep attains strong results on ABIDE/ADHD-200 and yields additional improvements when HDM is inserted into GCN/BrainNETTF, indicating broad applicability.

### Weaknesses
1. **Insufficient theoretical grounding of the HDM surrogate.**  
   The approximation \(JJ^\top \approx OO^\top\) and the use of \( (x_i O)(x_j O)^\top \) as a proxy for high-order dependence are motivated heuristically. Formal properties (e.g., bias/consistency under realistic noise models, identifiability or approximation guarantees) are not established, leaving it unclear when the surrogate faithfully captures higher-order interactions.

2. **Limited evaluation under site variability and distribution shift.**  
   ABIDE/ADHD-200 are multi-site, yet no site-held-out or scanner-held-out protocol is reported. Without cross-site/cross-dataset tests, robustness to dataset shift and potential site leakage remain unassessed.

3. **Attribution of gains among HDM, TopK, and denoising is unclear.**  
   Ablations remove normalization/denoising, but the specific contribution of HDM vs. TopK(\(ZZ^\top\)) vs. the GCN-based denoising regularizer is not disentangled. Sensitivity to \(k\), \(\lambda\) (denoising weight), and the activation \(\sigma\) needs to be characterized to support the claimed mechanisms.

4. **Scalability and compute reporting are incomplete.**  
   With \(O\in\mathbb{R}^{D\times D}\) and the construction of \(ZZ^\top\), memory/time complexity and wall-clock performance are not reported as functions of \(N\) (ROIs) and \(D\) (time length). Practical feasibility on higher-resolution atlases or longer scans is thus unclear.

5. **Generalization across variable time-series lengths is not addressed.**  
   Results suggest \(D{=}100\) performs best because \(O\) is square. However, the behavior under variable \(D\) (different TRs, windowing, missing timepoints) is not evaluated, despite being common in real datasets.

6. **Interpretability evidence lacks statistical rigor.**  
   Differential ASD–NC connectivity visualizations are suggestive but lack group-level statistical testing (e.g., permutation tests, FDR control) and site-stratified analyses, making the biological claims preliminary.

### Minor
- **Typos/wording:** “multi-layer **perception** (MLP)” → *perceptron*; “HSCI” → *HSIC*; “**caculated**” → *calculated*; “**deontes**” → *denotes*.
- **Clarity:** In Eq. (6), restate shapes for \(X,\hat{X},Z,O\) and \(A\in\mathbb{R}^{N\times N}\).  
- **Protocol:** Specify whether splits are **site-stratified** and how repeated sessions per subject (if any) are handled to avoid leakage.

### Questions
1. **Theory:** Under what assumptions is the HDM estimator \( (x_i O)(x_j O)^\top \) a consistent or bounded-bias surrogate for high-order dependence? Can you provide error bounds or identifiability conditions (e.g., under sub-Gaussian noise or bounded moments)?
2. **Distribution shift:** Do you have **site-held-out** evaluations (leave-one-site-out/leave-k-sites-out) and **cross-dataset transfer** (e.g., pretrain on ABIDE, fine-tune/evaluate on ADHD-200)? These would clarify robustness to scanner/protocol shift.
3. **Ablation attribution:** Can you isolate contributions via: (i) HDM only (no denoising; fixed \(k\)), (ii) denoising only (Pearson graph), (iii) HDM+denoising; and provide sensitivity to \(k\in\{20,40,80,160\}\), \(\lambda\), and \(\sigma\) on both datasets?
4. **Scalability:** Please report parameter counts, peak GPU memory, and wall-clock for representative \(N\in\{200,400\}\) and \(D\in\{100,200,400\}\). How is \(ZZ^\top\) batched to avoid \(O(N^2D)\) bottlenecks in practice?
5. **Variable-length handling:** If time lengths differ across subjects, do you window, resample, or mask? Can you show robustness to time-window length and stride to support the \(D{=}100\) choice beyond the square-\(O\) argument?
6. **Interpretability statistics:** For ASD–NC differentials, can you include edge-wise permutation tests with FDR control and site-stratified analyses to rule out site confounds?

### Soundness
3

### Presentation
2

### Contribution
2
