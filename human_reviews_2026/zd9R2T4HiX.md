# Synthesising Counterfactual Explanations via Label-Conditional Gaussian Mixture Variational Autoencoders

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 6, 4, 2, 4

## Abstract
Counterfactual explanations (CEs) provide recourse recommendations for individuals affected by algorithmic decisions. A key challenge is generating CEs that are robust against various perturbation types (e.g. input and model perturbations) while simultaneously satisfying other desirable properties. These include plausibility, ensuring CEs reside on the data manifold, and diversity, providing multiple distinct recourse options for single inputs. Existing methods, however, mostly struggle to address these multifaceted requirements in a unified, model-agnostic manner. We address these limitations by proposing a novel generative framework. First, we introduce the Label-conditional Gaussian Mixture Variational Autoencoder (L-GMVAE), a model trained to learn a structured latent space where each class label is represented by a set of Gaussian components with diverse, prototypical centroids. Building on this, we present LAPACE (LAtent PAth Counterfactual Explanations), a model-agnostic algorithm that synthesises entire paths of CE points by interpolating from inputs' latent representations to those learned latent centroids. This approach inherently ensures robustness to input changes, as all paths for a given target class converge to the same fixed centroids. Furthermore, the generated paths provide a spectrum of recourse options, allowing users to navigate the trade-off between proximity and plausibility while also encouraging robustness against model changes. In addition, user-specified actionability constraints can also be easily incorporated via lightweight gradient optimisation through the L-GMVAE's decoder. Comprehensive experiments show that LAPACE is computationally efficient and achieves competitive performance across eight quantitative metrics.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
A novel method to generate counterfactual explanations (CEs) is proposed. By learning a label-conditional VAE with Gaussian-mixture latent distribution (LGMVAE), the proposed method enables the generation of valid, plausible, close, robust, diverse and actionable CEs. The generation process only requires a few forward passes through the decoder of the LGMVAE (and one through its encoder) along an interpolation path, from the latent input to several possible cluster centroids, which correspond to the target class. Most costs are amortized through a single offline LGMVAE training. Extensive numerical experiments are provided.

### Strengths
- The proposed method is simple and effective.
- Most desirable properties of CEs are addressed. 
- The method is clear and the paper is rather well-written.
- The numerical experiment section provides comparison against multiple baselines.

### Weaknesses
- Several questions regarding the interpolation process itself remain unexplored: I detail this in the “Questions” part of the review.
- Questions about the choice of target centroid are also unexplored. Indeed, although providing every interpolation path to every valid class centroid guarantees diverse explanations, one could wonder if smart target centroid choices can be made. For example, in Figure 1, choosing the centroid of cluster 4 would guarantee better proximity of the generated CEs than choosing that of cluster 3: such considerations could allow users to optimize for different criteria (in this example, a trade-off between diversity and proximity). 
- Typo: L/K should be K/L at line 169.

### Questions
- How to better train a LGMVAE so that interpolation paths are realistic ? Regularization techniques to obtain a Euclidean latent space (where shortest paths are indeed straight lines) could be explored, e.g., regularizing the anisotropy of the decoder’s Jacobian.
- Differently, how to produce geodesic interpolations (in the sense of prefering traversing high-density regions) with a normally-trained LGMVAE ?
- A plot showing the evolution of the likelihood of the interpolated CE along the produced paths (as a function of \tau) is lacking. Since this is not optimized for, we can expect these interpolation paths to traverse low-density regions of the latent space (and thus of the feature space, once decoded).
- The predicted class (according to the model) along this path, as a function of \tau, would also be interesting, since the validity of these decoded interpolated points as CEs is not guaranteed.

### Soundness
3

### Presentation
4

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
This work presents a novel, model-agnostic approach for generating counterfactual explanations by constructing latent paths between factual instances and class-conditioned cluster centroids. These centroids are obtained using L-GMVAE, a label-conditional Gaussian Mixture VAE that learns multiple cluster centers per class. By interpolating in the latent space, the method allows users to control the trade-off between proximity and robustness when selecting counterfactuals. Robustness is particularly relevant in the current landscape of counterfactual explanations, where stability under input perturbations and model updates remains a key challenge. Additionally, the method supports actionability constraints by applying gradient-based adjustments whenever a generated sample violates a constraint.

### Strengths
1. The paper is generally well written, and the proposed method is clearly described.
2. The solution is model-agnostic and can be applied to both differentiable and non-differentiable models.
3. The approach allows users to choose an appropriate trade-off between proximity, plausibility, and robustness.
4. The authors introduce a novel L-GMVAE architecture that is specifically designed to support the proposed counterfactual generation method.

### Weaknesses
1. Evaluating plausibility using a single metric may be insufficient. Relying solely on Local Outlier Factor may not fully capture realism; complementary metrics (e.g., Isolation Forest, reconstruction likelihood, or classifier confidence margins) could provide a more comprehensive assessment.
2. The experimental evaluation is relatively limited. Although several baselines are included, the number and diversity of datasets is small.
3. Increasing the number of diverse counterfactuals requires retraining the L-GMVAE with a different number of clusters, which introduces additional computational overhead and slows down the overall counterfactual generation pipeline.
4. The effectiveness of the method heavily depends on successful L-GMVAE training. When latent clusters do not align clearly with class structure (e.g., due to class imbalance or categorical noise), the guarantees regarding validity and robustness may break down.
5. The visualizations in Figure 4 are dense and difficult to interpret. Presenting the results in table format will improve readability.
6. The evaluation uses only 100 test samples per fold pair, which may be insufficient for more complex datasets such as Adult.
7. The description of actionability constraints lacks precision. It would be useful to provide a clearer formal definition.
8. The method performs less effectively on datasets with many categorical features. This limits applicability in common tabular domains.
9. The work lacks an ablation study evaluating how the number of clusters per class affects diversity, robustness, and plausibility. Since this parameter controls core method behavior, such analysis would strengthen the empirical grounding.
10. The experimental section does not include comparisons to recent plausibility-oriented generative counterfactual methods such as PPCEF or C-CHVAE. Including such baselines would provide a more meaningful reference for plausibility performance.

### Questions
- How could the proposed method be extended to support global or group-level explanations, rather than generating counterfactuals for individual instances?
- How does the number of clusters per class influence proximity, plausibility, and diversity of the generated counterfactuals? Is there a principled way to select this parameter?
- What is the computational overhead introduced by enforcing actionability constraints? How does this affect the latency of counterfactual generation in practice?
- Since generative models often struggle with datasets containing many categorical features, did you consider techniques such as dequantization to improve latent space learning for categorical variables?
- The paper empirically demonstrates robustness, but does the method admit a formal robustness guarantee? If so, could you provide a more explicit formulation or theoretical justification?

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
This paper proposes to provide counterfactual explanation using deep generative models (VAE with gaussian mixtures to be exact), by interpolating on the latent space between different latent clusters corresponding to different conditions/perturbations.

### Strengths
The paper is overall very easy to follow. Aside from a very minor clarity issue which I described in the next section, the methodology as well as mathematical formulation are generally laid out very clearly. The figures also gave a good demonstration of the key idea. The derivation of ELBO and it's underlying factorization are also expressed clearly.

### Weaknesses
I found the approach proposed in this work not well-justified, heuristic, and very much lack robustness if the authors decide to frame it under the counterfactual scope. The key concern is the blatant violation of the consistency assumption in causal inference, which is the fundamental backbone of the definition of counterfactual in Pearl's 3 layers of causality. To put it simply, the counterfactual needs to maintain the same exact exogenous noise as the factual, i.e. minimal changes under perturbation, i.e. anything that is not causality affected by the perturbation should remain exactly the same in the counterfactual. And there is nothing in this work's methodology that gives that kind of guarantee or at least that kind of encouragement. This aspect (usually called "exogenous noise abduction") is very important in deep counterfactual modeling, and I suggest the authors consult some literatures in counterfactual modeling [1,2,3] and especially the recent literatures [4,5] that specifically tackle this consistency issue, and made it more and more clear that VAE is not well-suited for deep counterfactual modeling [4,5]. To put it more simply in the context of this work, VAE essentially gives you these different latent clusters of points under different $c$, but you don't know which two points in two different clusters correspond to the same exogenous features. And when you do interpolation without knowing the optimal transport path, features not causally related to $c$ also change. To use MNIST for example, if you simply do interpolation from a point in a digit cluster to the center of another digit cluster, not only is the digit changed, the exogenous features (style of writing, thickness of writing, intensity of writing, etc.) could all change, which results in not minimal changes, which results in the wrong counterfactual explanation.

What the authors are doing in this work is better described as interventional (layer 2 of causality), not counterfactual (layer 3 of causality). The metrics used in experiments are also predominantly interventional inference metrics. I suggest the authors consult latest literatures in counterfactual evaluation [6] for the more valuable evaluations of counterfactuals. As for dataset, I also suggest evaluation on Morpho-MNIST instead of MNIST, which is a true perturbation dataset where the counterfactual truths can be simulated and the preservation of exogenous features can be evaluated.

Minor:

1. Clarity of the main objective: I suggest the authors parameterize Eq. (3) to make it more clear which components are parameterized by neural nets and which ones are assumed to be known priors. I found the paragraph underneath it (L195-202) to be slightly vague and not sufficient. Canonically, I would assume the learnt encoder components to be $q_\phi(c | x, y)$ and $q_\phi(z | x, c, y)$ and learnt decoder components to be $p_\theta(x | z)$, but in this case $p_\theta(z | c)$ is also parameterized and it'd be good to make it clear.

[1] Pawlowski, Nick, Daniel Coelho de Castro, and Ben Glocker. "Deep structural causal models for tractable counterfactual inference." Advances in neural information processing systems 33 (2020): 857-869.

[2] Shen, Xinwei, et al. "Weakly supervised disentangled generative causal representation learning." Journal of Machine Learning Research 23.241 (2022): 1-55.

[3] Ribeiro, Fabio De Sousa, et al. "High Fidelity Image Counterfactuals with Probabilistic Causal Models." International Conference on Machine Learning. PMLR, 2023.

[4] Wu, Yulun, Louis McConnell, and Claudia Iriondo. "Counterfactual Generative Modeling with Variational Causal Inference." The Thirteenth International Conference on Learning Representations.

[5] Ribeiro, Fabio De Sousa, Ainkaran Santhirasekaram, and Ben Glocker. "Counterfactual Identifiability via Dynamic Optimal Transport." arXiv preprint arXiv:2510.08294 (2025).

[6] Monteiro, Miguel, et al. "Measuring axiomatic soundness of counterfactual image models." The Eleventh International Conference on Learning Representations.

### Questions
N/A

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper addresses the problem of generating counterfactual explanations (CE) that remain robust to small changes in both model parameters and input data. It introduces the Label-Conditional Gaussian Mixture Variational Autoencoder (L-GMVAE), which represents each target class as a set of clusters following a gaussian mixture distribution. The centroids of these clusters act as robust prototypes for the target class. Building on this, the authors propose LAPACE (Latent Path Counterfactual Explanations), which generates CE by interpolating between an input’s latent representation and the centroids of the target class, yielding diverse and plausible counterfactual paths. The method is benchmarked against several common baselines and datasets, demonstrating improved robustness, plausibility, and diversity compared to existing approaches.

### Strengths
- The L-GMVAE formulation is a good contribution, offering a unified framework that simultaneously addresses key aspects of robustness, validity, and diversity in CE generation.

- The experiments on L-GMVAE training and evaluation are well designed, providing convincing evidence that the learned latent structure captures the underlying data distribution effectively. This acts as an important sanity check, showing that the generative model produces samples closely resembling the true data, thereby validating its suitability for CE generation.

- The empirical evaluation demonstrates that the proposed LAPACE method performs strongly across multiple datasets. Results indicate that it is competitive or outperforms existing baselines, particularly in robustness and plausibility, highlighting the effectiveness of the proposed approach.

### Weaknesses
- The qualitative MNIST experiments are rather synthetic and do not fully demonstrate the method’s generality. It would be great if the authors could include results on more challenging image or text datasets. Similarly, the authors could incorporate more high-dimensional tabular datasets as well.

- The proposed approach for accommodating actionability constraints may risk moving latent representations away from the learned data manifold, potentially generating invalid counterfactuals. Some quantitative or qualitative assessment of this would be nice.

- The argument that cluster centroids lie well within the data manifold and improve robustness to model updates feels somewhat ad hoc. While there is supporting empirical evidence, a stronger conceptual or theoretical justification would be better,

- The presentation of results in Figure 4 is overly dense and makes it difficult to interpret. Simplifying the figure, or reorganizing the metrics for better readability, would help communicate the results more effectively.

### Questions
- Line 170, there seems to be a typo, it should be uniformly assign $K/L$  cluster to each class.
- Equation (3) probably has a typo, the LHS should be $p(x|y)$ instead of $p(x)$.
- Line 249, "Additionally, they do not risk exposing existing data points given their synthetic nature." I don't follow this, please clarify.

### Soundness
3

### Presentation
3

### Contribution
2
