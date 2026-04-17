# Rethinking the Flow-based Gradual Domain Adaption: A Semi-Dual Transport Perspective

- Decision: Reject
- Scores: 2, 6, 6, 4

## Abstract
Gradual domain adaptation (GDA) aims to mitigate domain shift by progressively adapting models from the source domain to the target domain via intermediate domains. However, real intermediate domains are often unavailable or ineffective, necessitating the synthesis of intermediate samples. Flow-based models are recently used for this purpose by interpolating between source and target distributions, but their training typically resorts to sample-based log-likelihood estimation, which can discard useful information and thus degrade GDA performance. The key to addressing this limitation is constructing the intermediate domains via samples directly. To this end, we propose an $\underline{\text{E}}$ntropy-regularized $\underline{\text{S}}$emi-dual $\underline{\text{U}}$nbalanced $\underline{\text{O}}$ptimal $\underline{\text{T}}$ransport (E-SUOT) framework to construct intermediate domains. Specifically, we reformulate flow-based GDA as a Lagrangian dual problem and derive an equivalent objective that circumvents the needs for likelihood estimation. However, the dual problem results in the unstable min–max training procedure. To alleviate this issue, we further introduce entropy regularization to convert it into a more stable alternative optimization procedure. Based on this, we propose a novel GDA training framework and provide theoretical analysis in terms of stability and generalization. Finally, extensive experiments are conducted to demonstrate the efficacy of the E-SUOT framework.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work proposed a semi-dual formulation for gradual domain adaptation. Such a  formulation avoids the use of probability density which is hard to estimate. An additional entropy regularizer is added to ensure the uniqueness of the problem. A concrete algorithm is developed based on this formulation, and a theoretical bound is proposed.

### Strengths
1. The formulation itself is interesting as it avoids the use of pdf.
2. The use of entropy regularizer ensures the uniqueness of the solution.

### Weaknesses
1. The author mentioned that gradual domain adaptation is useful "when the source–target shift is substantial or class overlap is weak" in line 41, but I do not find any theoretical and experimental evidence to support this. In the experiments, all baseline methods are gradual adaptation methods. The comparison with one-shot methods with different "class overlap" and "source–target shift" is lacking.

1. The lack of experiments on standard datasets like Office-Home and VisDa.

2. The author mentioned that "flow-based methods still require explicit estimation of the target domain’s PDF to guide the evolution," (line 448) However, in previous works like Zhuang et al. (2024), the pdf of the target domain is never explicitly estimated, and only the samples are needed. Also, the statement in contribution 1 should be modified.

3. The algorithm seems extremely inefficient. For each time steps, $w$ is trained for several epochs, then $T$ is trained for several epochs, finally $h$ is finetune on all time steps.
   
4. The author claims that the proposed method is stable in line 76. However, no evidence is provided. I suggest adding an ablation study on this.

### Questions
1. In figure 1, why the generated samples of E-SUOT are different from the ground truth? The variance of the generated samples seems much smaller than the ground truth.
2. What is the self-training method?
3. In Thm 6. The hypotheses and the loss function need to be Lipschitz. Does this result apply to the classification problem considered in this work? How to ensure the Lipschitzness of hypotheses?
4. Line 303. "From Theorem 5, we observe that as t increases, the transported PDF ρ(x) progressively becomes similar to pT (x)." Why?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work considers the fundamental limitations of flow-based method for gradual domain adaptation, i.e., the quality of estimated PDF of target and generated intermediate domains. The key idea is to avoid the explicitly estimation of PDF by introducing the semi-dual formulation of gradient flows with entropy regularization, which ensure the stability and convergence of training. Theoretical results on optimality and generalization error are provided.

### Strengths
+ The idea of improving flow-based method with PDF-free metric (i.e., Wasserstein) seems to be interesting and sounded.

+ The theoretical results are solid to ensure the numerical property of the defined flow model and the generalization error of the adaptation process.

+ The empirical results are convincing, which show the proposed method indeed shows consistent behavior with the theoretical analysis.

### Weaknesses
+ The justifications w.r.t. the theoretical assumption and results could be improved.  

+ The empirical validations seem to be limited, e.g., the compared baselines and evaluation datasets.

### Questions
I have no major criticisms on this submission. Here are several minor points. 

**Questions**

Q1. The smooth label space assumption in assumption (A.3) could be justified more deeply. Though such an assumption could be satisfied in many scenarios, there are still some cases which the label space could change rapidly with a slight change of feature x. For example, the fine-grained recognition tasks, where the subcategories could be close in feature space while totally distinct in label space. Thus, it would also be important to clarify the feasible and infeasible scenarios of the derived results, e.g., the condition of the assumption.

Q2. There seems to be intractable terms in the generalization bound Eq. (12). Specifically, the constants in the third and fourth terms. For example, the selection of the loss function could significantly affect the upper bound, while other terms like label smoothness are even unknown. It would be highly appreciated to justify the intractable constants in detail.

Q3. It seems that some related advanced framework is not compared in empirical evaluation, e.g., diffusion-based methods that share similar innovations. Specifically, compared with the standard diffusion process, does the proposed framework admit better properties? Besides, could the proposed method achieve better empirical performance?

Q4. Will the developed flow method be sensitive to the data scale? Specifically, would it be feasible or efficient for large-scale data? Moreover, the existing empirical evaluation only considers simple and small datasets. Are there any other potential application scenarios of GDA that have larger data scales?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a semi-dual optimal transport formulation stabilized by entropy regularization for flow-based gradual domain adaptation. It aims to overcome the weakness in existing flow-based GDA methods which rely on accurate, explicit target domain probability density function estimation.

### Strengths
-	Novelty: The idea of reformulating flow-based GDA within a semi-dual OT framework to eliminate the need for explicit target PDF estimation seems novel.
-	Solid Theoretical Analysis: The theoretical analysis of the semi-dual formulation, proofs of uniqueness/convergence, and generalization bounds sound solid.
-	Improved Stability: The entropy regularization effectively addresses the inherent instability of the adversarial semi-dual formulation.
-	Reproducibility: The source code is available, facilitating its reproducibility.

### Weaknesses
-	Limited Empirical Evaluation: Experiments are conducted only on low-dimensional (8D) UMAP embeddings of relatively simple datasets (Portraits, rotated MNIST). The performance and scalability on high-dimensional, complex real-world datasets (e.g., Office-Home, DomainNet) remain unproven.
-	Hyperparameter Sensitivity: The model performance is highly sensitive to multiple key hyperparameters (batch size B, discretization step size η, simulation steps T, entropy regularization strength ε,), as shown in Figure 3. This necessitates careful tuning and could limit practical applications.
-	Computational Burden: While scaling better than some GP-based methods, the approach still requires training a sequence of neural networks (w_φ and T_θ for each intermediate step), making it computationally intensive compared to simpler baselines.
-	Lack Intuitive Illustration: The dense notations and propositions without the intuitive illustrations influences its accessibility.

### Questions
-	Why the features are mapped to low-dimensional (8D) UMAP embeddings? 
-	How does the performance and stability of E-SUOT change with the dimensionality of the input data? 
-	Are there limitations to the current neural parameterization of w_φ and T_θ in very high-dimensional spaces?
-	The motivation of using unbalanced optimal transport instead of the standard OT in this application had not been made clear. Also, how did you tune the unbalanced factors lambda_1 and lambda_2 defined in Equation A.6 for the experiments?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The method reformulates the flow-based adaptation as a Wasserstein-distance-regularized optimization problem. By deriving the semi-dual of this problem, they create an objective function that cleverly avoids PDF estimation and relies only on sample-based expectations. In general, the paper makes contribution by unifying flow‑based GDA with semi‑dual OT and by offering an entropy‑regularized, sample‑only training objective with uniqueness.

### Strengths
The paper is well written and easy to follow. The proposed solution is also theoretically sound. It involves reformulating the gradient flow as a Wasserstein-regularised problem and then moving to its semi-dual form in order to bypass density estimation. However, the novelty of this reformulation is questionable because the same steps of reformulations has previously been applied to different OT methods. However, this type of optimization and practical problem had not previously been considered,  so I am more in favour of the novelty of the method.

### Weaknesses
The proposed solution is theoretically sound well. The reformulation of the gradient flow as a Wasserstein-regularized problem and the subsequent move to its semi-dual form to bypass density estimation. But novelty of this reformulation is questioning because previously the same tricks was applied to the different OT methods. Specifically for this type of the optimization problem it was not previously considered, so I am toward more that the method is novel. 

One of the main weaknesses of the paper is that all datasets are relatively simple (Portraits; MNIST rotated 45°/60°) and the pipeline uses semi‑supervised UMAP embeddings to “preserve class discriminability” before adaptation (Sec. 4.1, p. 6). It is unclear whether all baselines receive the same UMAP features, and whether the gains persist without this strong, label‑aware pre‑processing. The absence of standard multi‑domain vision benchmarks (e.g., Office‑Home/DomainNet/VisDA) makes it hard to assess real‑world impact. Also table 1 marks methods that E‑SUOT “significantly” outperforms but does not report standard deviations or the number of seeds/runs in the main text; the testing protocol (data splits, early‑stopping criteria, target‑side tuning) is summarized at a high level only. This makes it hard to assess robustness.

The paper focuses on adapting the feature distribution ($p(x)$). The workflow described in Algorithm 2 appears to assume the labels are invariant along the transport path ($y_{t+1} \leftarrow y_t$). It is not clear how the framework would perform in scenarios with significant label shift (i.e., where the relationship $p(y|x)$ also changes between domains), which is a common for unbalanced settings.

### Questions
*Question 1*: Can you add experiments on widely used multi‑domain image benchmarks without UMAP pre‑processing, and complementary tests that introduce mass/prior mismatch (label‑shift, class imbalance, missing classes) to demonstrate the benefit of the unbalanced formulation? Please report per‑method standard deviations across multiple seeds.

*Question 2*: Your ablation study shows that the KL divergence (via its conjugate $f^*$) performs significantly better than other f-divergences like $\chi^2$ or an identity function. What is the intuition for this? 

*Question 3*: Does KL divergence have a specific property (perhaps related to its gradient flow or the stability of its conjugate function $f^*$) that makes it uniquely suited for this semi-dual transport framework?

*Question 4*: The full algorithm chains multiple transport steps ($T_{\theta,0}, T_{\theta,1}, ...$). How do errors from a non-optimal transport map $T_{\theta,t}$ at an early step $t$ propagate through the rest of the chain? It would be interesting if authors can draw a parallel to the diffusion process and importance of the earlier steps.

### Soundness
2

### Presentation
3

### Contribution
2
