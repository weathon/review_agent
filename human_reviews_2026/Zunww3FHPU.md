# Navigating the Latent Space Dynamics of Neural Models

- Avg Score: 6.50
- Decision: Accept (Oral)
- Scores: 6, 6, 6, 8

## Abstract
Neural networks transform high-dimensional data into compact, structured representations, often modeled as elements of a lower dimensional latent space. In this paper, we present an alternative interpretation of neural models as dynamical systems acting on the latent manifold. Specifically, we show that autoencoder models implicitly define a _latent vector field_ on the manifold, derived by iteratively applying the encoding-decoding map, without any additional training. We observe that standard training procedures introduce inductive biases that lead to the emergence of attractor points within this vector field.
Drawing on this insight, we propose to leverage the vector field as a _representation_ for the network, providing a novel tool to analyze the properties of the model and the data. This representation enables to: $(i)$ analyze the generalization and memorization regimes of neural models, even throughout training; $(ii)$ extract prior knowledge encoded in the network's parameters from the attractors, without requiring any input data; $(iii)$ identify out-of-distribution samples from their trajectories in the vector field.
We further validate our approach on vision foundation models, showcasing the applicability and effectiveness of our method in real-world scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes viewing autoencoder models as latent dynamical systems, where iterating the mapping $f=E\circ D$ defines a latent vector field and reveals attractors that capture the model’s memorization and generalization behavior.

The authors connect the local contractivity of this mapping to the emergence of attractors and use them for practical analyses such as (1) distinguishing memorization vs. generalization regimes, and (2) performing data-free probing and out-of-distribution (OOD) detection by analyzing trajectories toward these attractors.

In addition to theoretical connections and remarks, empirical results are shown using autoencoders, a diffusion-model autoencoder, and a large-scale vision model utilizing masked autoencoders.

### Strengths
- The idea of treating $E\circ D$ as a dynamical system in latent space is novel and intuitive, providing a unifying perspective on autoencoders.
- The framework is validated across various architectures, including autoencoders, a pretrained diffusion AE, and a large-scale vision model, showing that attractors can indeed be identified even in large-scale, complex models.
- Using attractors derived from noise, without requiring access to source training data, to reconstruct meaningful images is an interesting way to explore what information the model stores in its weights, opening up interpretability and compression directions.
- The paper provides clear and well-motivated definitions of concepts such as contractivity and attractors, and links them intuitively to properties of the model’s Jacobian, which helps make the overall framework more interpretable and understandable.

### Weaknesses
- As the authors acknowledged in the discussion, it remains uncertain whether the approach applies to widely used forecasting or next-token prediction models, or encoder-only architectures. Since such models dominate modern representation learning, discussing whether attractors exist or can be defined meaningfully in these settings would strengthen the impact. To examine this, would training a lightweight decoder on top of a frozen encoder (without modifying the encoder weights) help reveal similar attractor dynamics?
This could clarify whether the attractor framework extends beyond autencoder-based models.
- While the authors show that attractors inferred from noise can reconstruct the actual inputs, it is not entirely clear whether these attractors correspond to actual training examples, or how one can infer generalization capacity without explicitly comparing attractors to the training data.
- The theoretical analysis assumes local contractivity, which potentially does not hold globally. Empirically, as long as stable attractors can be identified, the proposed approach appears to remain valid. Nevertheless, it would be good to quantify how much of the latent space exhibits convergent dynamics, characterize the stability of these attractors, and report how long or how many iterations are typically required to discover them beyond the MNIST example. 
- Is the KNN analysis performed using latent embeddings of training data or the attractors identified by training data? How sensitive are the KNN results in Figure 5a to the choice of the number of neighbors K? Would similar results hold for smaller values of K or with adaptive neighborhood sizes? Also, for the proposed attractor trajectory-based scoring, is the distance computed with the nearest training attractor's trajectory or averaged across all training attractors?
- Following up on the previous item, how does the proposed attractor trajectory-based scoring compare with other standard OOD detection metrics, such as the Mahalanobis distance in latent space or the reconstruction loss (MSE) in input space?
- Similarly, would OOD detection performance remain similar if attractors were computed from Gaussian noise?
- Please provide the definition of FPR95 where it first appears. Also, the definition in L415 can be improved by stating that it uses a threshold such that 95% of ID samples are correctly classified.
- The caption of Figure 5 needs improvement; it is currently unclear which histogram corresponds to which method. Similar clarification should be added for Figure 2 for the attractor reconstructions, by specifying whether those are latent attractors or decoded outputs.

### Questions
- For a pretrained model, is it possible to generate attractors from random noise and gain intuition on whether the model is operating in the generalization or memorization regime?
- Do the reconstructions of attractors for the pretrained models carry interpretable or semantically meaningful information?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work introduced a method to interpret autoencoder neural networks as dynamical systems defined by a latent vector field on their manifold. This vector field on their manifold is derived by iteratively applying their encoding-decoding map.

The paper claims that inductive biases introduced by standard training can be seen as emerging attractor points in their latent vector fields and propose to leverage this vector fields as representations of the neural network for downstream tasks such as (i) the analysis of the neural network with respect to generalization and memorization, (ii) the extraction of knowledge encoded in the weights of the neural network, and (iii) as tool to identify out-of-distribution samples.

The paper presents three experiments. The first experiment investigates the relationship between generalization and memorization and the role of regularization on 30 convolutional AE trained on small-scale datasets such as CIFAR10, MNIST, and FashionMNIST. The second experiment aims to investigate vision foundation models and probe the recovery of information about the data encoded in the models' weights. This is done on Stable diffusion AE and vision transformer masked AEs. The third experiment aims to demonstrate the method's expressiveness to detect distribution shifts of input data from the latent trajectories of the vector field.

I like the idea and think that this paper does well in motivating the proposed methods, providing a theoretical foundation for it, and demonstrating the method's utility through the set of downstream tasks. Unfortunately, this approach is limited to encoder-decoder models, which is mentioned in the limitation section of the papers.

There is one open question that I would appreciate getting answered by the authors. There exists another method that learns a lower-dimensional manifold of neural network models using an autoencoder architecture. The embeddings on this manifold are then used for several downstream tasks, revealing the encoded information of the neural network's weights. This paper and the method I have mentioned sound similar, and I would like to make sure that they are different, as I understand it. I do provide mode details in the questions section.

### Strengths
- **(S1)**: I appreciate the paper's motivation and theoretical foundation. It provides an interesting view of (AE) neural networks and provides a novel tool for analysis.

- **(S2)**: I think that the experimental section is honestly aiming to demonstrate the method's utility with respect to different downstream tasks and different datasets. I also appreciate the details and additional results listed in the appendix.

### Weaknesses
- **(W1)**: The proposed method is limited to reconstruction-based autoencoder neural networks. The authors are aware of this as they do mention this in the imitation section.

### Questions
- **(Q1)**: As mentioned above, I would like to make sure I understand the presented approach properly and do not confuse it with another method. In [1], a lower-dimensional manifold of neural network weights is learned by an encoder-decoder setup using a reconstruction loss and a self-supervised loss. This encoder-decoder bottleneck is interpreted as the representation of a neural network, which itself can be used to reveal encoded information of the neural network. I think that this submission is different from the work mentioned, but I am not sure since the methods are similar in their terminology and ideas.

[1] Self-Supervised Representation Learning on Neural Network Weights for Model Characteristic Prediction
Schuerholt et al, NeurIPS, 2021

### Soundness
3

### Presentation
4

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
The paper presents an alternative representation of autoencoder (AE) models as dynamical systems acting on the latent manifold. The authors introduce the theoretical framework required for this interpretation and establish results linking the dynamical system induced by the iterative application of an AE to the gradients of the underlying data distribution. They also theoretically characterize the attractors of the latent vector field and link them to memorization and generalization.

This framework is then used to analyze how regularization affects the phenomena of memorization and generalization in AEs, showing that as memorization decreases and generalization increases, the attractors of the AEs evolve from latent points corresponding to training examples toward more general attractors. The paper further shows that a transition from memorization to generalization occurs during training, with an increasing number of attractors being learned and the similarity of attractors found using different data converging.

Finally, the authors extract AEs from common pre-trained models and show that noisy inputs can be used to find the attractors of the induced dynamical system, and that these attractors can serve as a dictionary helping reconstruct data points from diverse distributions (as compared with a random orthogonal basis). The trajectories of examples can also be used for OOD detection.

### Strengths
- This new perspective on AEs is simple and intuitive. It is somewhat surprising that this type of analysis has not been done sooner.
- The links to regularization, memorization, and generalization are interesting, and the proposed framework could be a useful analysis tool.
- The theoretical framework is well presented and clear, and the theoretical results appear correct.
- The paper is well written; the dynamical-systems terminology is clear and intuitive.
- The experiments using AEs extracted from pre-trained models are particularly strong, without these, the toy settings described earlier would not have been sufficient.

### Weaknesses
- The scope of the paper is somewhat limited since the theory only holds for AEs.
- While the proposed framework is well justified and interesting in its own right, its impact is difficult to gauge. There is no immediate practical impact for practitioners, nor any strikingly new finding that this framework helps uncover. However, the work has clear potential as a future analysis tool.

See the Questions section for more precise comments.

### Questions
- Theorem 1 strength and scope: The result relies on uniform contractivity and latent concentration on fixed points, which are strong assumptions rarely satisfied by general AEs. The empirical section provides motivation for approximate contractivity, but the theorem should be reframed as a local or heuristic alignment with the score, not strict proportionality.
- Banach fixed-point misstatement: The text says well-posedness “holds iff f is Lipschitz-continuous.” Banach’s theorem requires a contraction (Lipschitz constant < 1), not mere Lipschitz continuity. Please correct.
- Definition 3 typo: There appears to be an extra f after the Jacobian when defining the Lipschitz constant in Definition 3.
- Detail: The numbering of the Theorems / Propositions is inconsistent across the main paper and appendix, which is confusing. Either match the numbering exactly or link the appendix theorem/proposition in the main paper.
- Section 4.1: The claim that “OMP = PCA” when using a random orthogonal basis is incorrect. PCA involves the data-covariance eigenbasis; OMP on a random orthobasis is simply sparse projection in that basis, not PCA. Please fix the description and, ideally, compare against true PCA (top-k principal components learned from data) as a stronger baseline.
- From the definition of the trajectory score in Sec. 3.2.2, it is not directly clear whether the distance is to all training attractors at once, and which exact point-cluster distance is used (this likely affects results since different point-cluster distances capture different notions of similarity).
- OOD baselines are too weak. The proposed trajectory-distance score is only compared to K-NN (with K = 2000). Modern OOD detection for vision backbones includes MSP/energy scores, Mahalanobis, ViM, ODIN, KLM, etc. Adding these would materially strengthen the claim that trajectories convey additional signal beyond embeddings. Also, which neighbors are considered? Since it is trajectories that are analyzed, and each embedding along the trajectory may have different K-NNs, the reference points are moving if you recompute the K-NN for each point in the trajectory.
- The distinction between “aggressive regularization” (1) and “over-parameterization” (2) as two forms of memorization is very interesting and would warrant further analysis. Perhaps this framework could allow the characterization of different forms of memorization in NNs. Currently, these are disjoint and hard to compare since (1) is presented in the main text as a function of k (latent-space dimension) and (2) is presented in the appendix as a function of dataset size. Unifying these observations would be valuable.

### Soundness
3

### Presentation
3

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
This paper shows that iterating the autoencoder‑induced map $f(z)=E \circ D(z)$ implicitly defines a vector field in latent space. It then exploits the field's dynamics and attractor structure to diagnose memorization versus generalization and to detect out‑of‑distribution (OOD) inputs.

### Strengths
- It is a novel and distinctive observation that iteratively applying $f$ induces the residual vector field $V(z) = f(z)-z$, whose fixed points serve as attractors toward which nearby trajectories converge.
- The claim that this vector field is proportional to the score of the latent prior $q(z)$ is highly intriguing; it effectively generalizes the small‑noise limit result for denoising autoencoders to the latent space.
- Proposition 2 is particularly insightful: when training biases the model toward memorization, the prototype term approaches zero while the coverage term narrows, yielding a clear, interpretable criterion for judging memorization versus generalization from the proposed error decomposition.
- The paper also establishes a lower bound on the number of iterations required to converge in simple linear settings, grounding the dynamics with an interpretable complexity estimate.

### Weaknesses
- The explanation for why contraction emerges *naturally*  via initialization bias, explicit regularization, and implicit regularization would benefit from a stronger theoretical foundation or, at least, a more formal set of sufficient conditions.
- Several assumptions, e.g., smoothness of the induced latent distribution and related regularity, are stated, but the extent to which they hold for large‑scale models in practice remains unclear.

### Questions
- **Numerical validation of Theorem 2.** Can you empirically validate Theorem 2? Such evidence would help assess the plausibility of the assumptions underlying its derivation and test the theorem's robustness in realistic settings.
- **Iteration complexity under weaker assumptions.** Is it possible to analyze (or bound) the number of iterations required to reach a fixed point under assumptions weaker than those currently stated?

### Soundness
3

### Presentation
3

### Contribution
3
