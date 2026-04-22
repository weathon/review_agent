# Unsupervised Representation Learning - an Invariant Risk Minimization Perspective

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 8, 2, 6

## Abstract
We propose a novel unsupervised framework for Invariant Risk Minimization (IRM), extending the concept of invariance to settings where labels are unavailable. Traditional IRM methods rely on labeled data to learn representations that are robust to distributional shifts across environments. In contrast, our approach redefines invariance through feature distribution alignment, enabling robust representation learning from unlabeled data. We introduce two methods within this framework: Principal Invariant Component Analysis (PICA), a linear method that extracts invariant directions under Gaussian assumptions, and Variational Invariant Autoencoder (VIAE), a deep generative model that separates environment-invariant and environment-dependent latent factors. Our approach is based on a novel ``unsupervised'' structural causal model and supports environment-conditioned sample-generation and intervention. Empirical evaluations on synthetic dataset, modified versions of MNIST, and CelebA demonstrate the effectiveness of our methods in capturing invariant structure, preserving relevant information, and generalizing across environments without access to labels.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes two unsupervised algorithms inspired by the invariant risk minimization (IRM) paper.  PICA (principal invariant component analysis) is in fact quite constrained and therefore less interesting than VIAE (variational invariant auto-encoder) which works by splitting the latent into an invariant component and an environment dependent component.  VIAE can also be used for environment transfer and to possibly recover IRM

### Strengths
- The notion of unsupervised invariant algorithms is compelling
- The VIAE approach seems promising (as illustrated by the environment transfer algorithm.).

### Weaknesses
- Some notations are confused (see below)
- PICA is very constrained because the difference of two empirical covariance matrices $\Sigma_1-\Sigma_2$ is likely to have a null kernel.  VIAE is far more satisfactory.
- Experiments in section 4.2.1 are insufficient. For instance, you could construct a CMNIST problem with labels that only depend on the shape of the digit, and a RevCMNIST problem using the same patterns but with labels that only depend on the color. The invariant features in the sense of IRM would be the shape for CMNST and the color for RevCMNIST. But since both datasets have the same input patterns, the unsupervised approach cannot make this difference.

### Questions
Imprecise notations: 
- line 88. $(X,e)\sim P_X^e(X)$.  If $P^e_X$ is a distribution over $X$, then it does not generate pairs $(X,e)$
- line 91. What's the relation between $P^e_X$ and $P_X$ ?
- line 98. Should there be an additional sum or an empirical average on the data for each environment.
- line 102. If the definition of $P(X,Y)$ implicitly depends on $\Phi$, why write $P(X,\Phi(X))$ in line 98?

Questions:
- What is the relation of PICA and CCA (canonical correlation analysis, Hoteling 1936)?
  Hoteling is also the inventor of PCA btw.
- Isn't PICA very constrained as the difference of two empirical convariance matrices  $\Sigma_1-\Sigma_2$ can easily have a null kernel.
- Line 286. Why not the other direction $P_e(Z_{inv},Z_e|X)=P(Z_{inv}|X) P_e(Z_e|Z_{inv},X)$ ?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes unsupervised Invariant Risk Minimization (IRM), extending the IRM framework to settings without labels by redefining invariance through feature distribution alignment across environments. The authors introduce two methods: Principal Invariant Component Analysis (PICA), a linear Gaussian approach that identifies invariant directions via null-space projections, and Variational Invariant Autoencoder (VIAE), a deep generative model that factorizes latent representations into environment-invariant (Z_inv) and environment-specific (Z_e) components. The framework is evaluated on synthetic data, modified MNIST variants (SMNIST, SCMNIST), and preliminary experiments on CelebA.

### Strengths
- The problem formulation--extending IRM to unlabeled multi-environment data--appears to be novel.
- The two-method approach (linear PICA + nonlinear VIAE) provides complementary perspectives with clear mathematical exposition.
- The fairness application demonstrates a natural use case where environment-invariant features correspond to removing sensitive attributes.

### Weaknesses
- One of the core weaknesses is that the objective is conceptually ill-defined. The paper redefines invariance as matching the marginals of the representations across environments but does not justify how this invariance serves IRM's goal of robust prediction. For instance, the learnt invariant features could be useless for any feasible downstream tasks. 
- Connecting to the previous point, there is no identifiability analysis of the learned representation; the model could learn arbitrary rotations of the invariant features [see 1] and there is no theoretical or empirical justification along those lines. I am also unsure if the learned representation would be coherent in all cases, for instance when data distribution is unbalanced across environments (as an extreme case, consider all 3's in one environment, all 1's in another). 
- The paper is missing extremely relevant prior literature on learning disentangled representations [1, 2, 3, 4] and fairness via disentanglement [11, 12, 13], as well as less critical but appropriate citations works on causal representation learning, IRM and domain generalization [5, 6, 7, 8, 9, 10], among others. Similarly, links between section 4.2 and references in domain adaptation should be fleshed out.
- Empirical justification is very limited: (1) MNIST-based simple datasets, no baselines - zero comparison to disentangled representations such as $ \beta $-VAE, or supervised IRM methods (2) No quantitative disentanglement metrics, evaluation relies on visual inspection; (3) Section 4.3's claim that 84% linear probe accuracy "validates" the approach lacks context without supervised baselines showing achievable performance.

Overall: The paper explores an interesting question but suffers from fundamental conceptual ambiguity (what is "invariance" without labels?) and insufficient empirical rigor (simplistic datasets, no baselines, no disentanglement metrics, no supervised comparisons). 

References:

[1] Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations - Locatello et al. 2019  
[2] beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework - Higgins et al., 2017  
[3] Isolating Sources of Disentanglement in Variational Autoencoders - Chen et al, 2018  
[4] Disentangling by Factorising - Kim et al, 2018  
[5] Towards Causal Representation Learning - Schölkopf et al., 2021  
[6] Weakly Supervised Disentangled Generative Causal Representation Learning - Shen et al, 2022  
[7] On Learning Invariant Representations for Domain Adaptation - Zhao et al, 2019  
[8] Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization - Sagawa et al, 2020  
[9] Learning Optimal Features via Partial Invariance - Choraria et al., 2023  
[10] Context is Environment - Gupta et al, 2023  
[11] Learning Fair Representations - Zemel et al., 2013  
[12] Flexibly Fair Representation Learning by Disentanglement - Creager et al., 2019   
[13] On the Fairness of Disentangled Representations - Locatello et al., 2019

### Questions
- Can the authors provide some grounded justification for distributional equivalence?
- Can the authors demonstrate a practical use-case, backed with results, of the presented method?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes an invariance principle for unsupervised learning, where an optimal reconstruction is sought from a latent representation that is constrained to be identically distributed along the training environments. The principle is demonstrated on Gaussian data, then a tailored VAE architecture for the problem is presented and demonstrated on MNIST variants and CelebA.
The empirical results show the model learns latent representations that hold domain invariant features, and another latent representation that holds domain dependent features.

### Strengths
Overall I liked the topic of the paper, the discussion presented and the developed methods.
I think there are some original contributions that are presented clearly, and the work could attract some interest from crowd interested in these topics.

### Weaknesses
There are some apparent weaknesses that I think the authors should take into account when revising the paper:
1. The motivation for the problem is not entirely clear. "Risk minimization" is a term mostly used in the context of a prediction problem, and I think the unsupervised setting is inherently different, hence a more suitable name for the work or solution might be something like an Invariant autoencoder/Environment-Invariant autoencoder etc. Now given this framing, it is not entirely clear what one has to gain from invariance in the unsupervised case, and what is the spurious correlation present at training time. Are we expecting the reconstruction error to be stable or min-max optimal on new environments, or some other form of robustness? The unclarity about motivation is especially apparent in section 4.2 where the authors link the problem back to supervised learning. I was not fully able to follow the claims in this section, and I think some more conceptual clarity is required either via math (as suggested in point 3 below), or a more convincing empirical problem. 
2. The idea seems quite similar to works from the domain adaptation and generalization literature, like DICA, DANN, CORAL, and Domain Separation Networks [1, 2, 3, 4]. Some of them are more closely related than others to the work under review, but it seems important that the authors refer to these works and explain the conceptual differences. Other than the presence of a label, the invariance principle where one wishes to learn a representation $\Phi(X)$ such that $P_{e}(\Phi(X)) = P_{e'}(\Phi(X))$ for each $e,e'\in{\mathcal{E}_{train}}$ is shared across these works.
3. The paper could benefit from some additions like a mathematical result demonstrating that the approach achieves some well-motivated desirable property on the linear-gaussian setting. Another interesting aspect could be to discuss the causal version of the "unsupervised" graph in Fig. 1, i.e. $Z_e \leftarrow X \rightarrow Z_{inv}$ and derive the corresponding architecture.
4. Finally, a clear drawback is that experiments are performed in rather small datasets and carefully designed problems that follow the assumptions of the method.

Small comments: It might be worthwhile enlarging Figure 1 and explaining the terms FIIF and PIIF formally. There are several places that use terms like "recover the causal structure" (line 248), "causality constraints" etc. I think it's better to keep the word invariance rather than causality, because recovering the causal structure might allude some readers to think the work tried to do causal discovery or some sort of structure learning, which is not the case. 

Overall, while I have several reservations about the paper, I gave an overall score of borderline accept and will reconsider it upon the authors' response. 

[1] Bousmalis, Konstantinos, et al. "Domain separation networks." Advances in neural information processing systems 29 (2016).
[2] Muandet, Krikamol, David Balduzzi, and Bernhard Schölkopf. "Domain generalization via invariant feature representation." International conference on machine learning. PMLR, 2013.
[3] Sun, Baochen, Jiashi Feng, and Kate Saenko. "Correlation alignment for unsupervised domain adaptation." Domain adaptation in computer vision applications. Cham: Springer International Publishing, 2017. 153-171.
[4] Sicilia, Anthony, Xingchen Zhao, and Seong Jae Hwang. "Domain adversarial neural networks for domain generalization: When it works and how to improve." Machine Learning 112.7 (2023): 2685-2721.

### Questions
What is the conceptual difference from methods mentioned in the "weaknesses" part? The presence of a label is a technical difference, but the motivation for replacing the label with a reconstruction loss seems somewhat weak.

It is mentioned that a separate encoder is trained for each environment. Is it actually a separate set of weights being trained, and if so then why? It seems more natural to train an encode that takes a one hot encoding of the environment, and will leverage the larger combined dataset to train its weights, while also possible learning similarities between the domains.

### Soundness
3

### Presentation
3

### Contribution
2
