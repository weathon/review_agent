# Unsupervised discovery of the shared and private geometry in multi-view data

- Avg Score: 6.00
- Decision: Reject
- Scores: 6, 6, 8, 2, 8

## Abstract
Studying complex real-world phenomena often involves data from multiple views (e.g. sensor modalities or brain regions), each capturing different
	aspects of the underlying system. Within neuroscience, there
	is growing interest in large-scale simultaneous recordings across multiple
	brain regions. Understanding the relationship between views (e.g., the neural
	activity in each region recorded) can reveal fundamental insights
	into each view and the system as a whole. However, existing methods to
	characterize such relationships lack the expressivity required to
	capture nonlinear relationships, describe only shared sources
	of variance, or discard geometric information
	that is crucial to drawing insights from data. Here, we present SPLICE: a neural network-based method that infers disentangled,
	interpretable representations of private and shared latent variables from
	paired samples of high-dimensional views. Compared to competing methods, we
demonstrate that SPLICE **1)** disentangles shared and private
representations more effectively, **2)** yields more interpretable
representations by preserving geometry, and **3)** is more robust to
incorrect a priori estimates of latent dimensionality. We propose our approach as a general-purpose
method for finding succinct and interpretable descriptions of paired data
sets in terms of disentangled shared and private latent variables.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces SPLICE, a novel framework for learning disentangled and interpretable representations of shared and private latent variables from paired observations of high-dimensional data. 

This is a fundamental problem with implications across multiple domains, from neuroscience to multimodal representation learning. The authors provide a conceptually elegant and theoretically motivated approach grounded in predictability minimization and manifold geometry preservation. They validate SPLICE on three complementary scenarios, demonstrating substantially superior performance to prior methods. 

The paper is well-structured, methodologically coherent, and convincingly analyzed, offering both theoretical insight and practical utility.

### Strengths
The use of predictability minimization for enforcing first-order independence between shared and private latents is original and intuitively appealing, addressing information leakage issues in earlier methods.

The experiments are carefully designed to illustrate disentangling, interpretability, and robustness to latent dimensionality mis-specification.
The application to real neural data adds strong credibility and relevance.

Beyond neuroscience, the framework provides a general solution for analyzing multiview high-dimensional data, potentially benefiting multiple research communities.

### Weaknesses
1. While SPLICE is presented as a general framework, the actual performance is closely tied to specific neural architectures used for encoding, decoding, and measurement. It remains unclear whether the proposed disentangling principles will generalize across architectures or scales without extensive hyperparameter tuning. A discussion or an analysis of architectural sensitivity would strengthen the claims.

2. While the approach is conceptually elegant, some arguments (e.g., the assumption that the full manifold is the cross-product of submanifolds) remain heuristic rather than theoretically justified.

3. Preserving submanifold geometry improves interpretability but may come at a computational or robustness cost, particularly for noisy or large-scale data.

### Questions
1. The proposed method relies on different neural architectures to learn different subproblems (shared, private, measurement). Given this dependence, to what extent can the theoretical ideas meaningfully guide the design of simpler or more scalable architectures?

2. Given the adversarial training component, what specific stabilization strategies were most effective in preventing mode collapse?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a method called SPLICE, which uses private and shared latent space to produce disentangled and interpretable representations. The proposed method also contains geometry identification and preservation components, which increases the fidelity of the latent space, leading to interpretable intrinsic sub-manifolds representations that preserves geometry. The proposed method is validated on controlled simulation datasets, and also is applied on neural datasets with different brain regions.

### Strengths
- The authors study the important problem of private/shared information disentanglement, which is a highly important problem in computational neuroscience.
- The proposed geometry identification and preservation method is novel, and very relevant in the neuroscience domain. The produced latent geometry can help study small-scale datasets, potentially help with scientific discoveries. 
- The experimental designs are very detailed. Key ablations such as latent dimensions are carefully studied.

### Weaknesses
- This work [1] seems highly relevant, which also proposes a private/shared latent space disentanglement method for neuroscience datasets. The authors should benchmark this work with the proposed work. The authors should also consider benchmarking with more advanced transformer architectures, such as [2, 3].

[1] Liu, Ran, Mehdi Azabou, Max Dabagia, Chi-Heng Lin, Mohammad Gheshlaghi Azar, Keith Hengen, Michal Valko, and Eva Dyer. "Drop, swap, and generate: A self-supervised approach for generating neural activity." Advances in neural information processing systems 34 (2021): 10587-10599.

[2] Liu, Ran, Mehdi Azabou, Max Dabagia, Jingyun Xiao, and Eva Dyer. "Seeing the forest and the tree: Building representations of both individual and collective dynamics with transformers." Advances in neural information processing systems 35 (2022): 2377-2391.

[3] Chau, Geeling, Christopher Wang, Sabera Talukder, Vighnesh Subramaniam, Saraswati Soedarmadji, Yisong Yue, Boris Katz, and Andrei Barbu. "Population transformer: Learning population-level representations of neural activity." ArXiv (2025): arXiv-2406.

- The presented experiments are very small scale. The authors should consider include results from larger datasets.

### Questions
NA

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
They propose a new deep learning method to reveal shared latent variables and separate latent variables across two datasets (or "views"). The architecture consists in two non-linear auto-encoders which swap parts of their latent space. In order to ensure that the private part of the latent does not leak shared information, they adapt an adversarial strategy proposed by (Schmidhuber 1992), predictability minimization, where a decoder aims at predicting the other dataset, while the global objective is to minimize the performance of that decoder.

### Strengths
- I find the method original yet simple, and well justified conceptually. I appreciate the elegant adaptation of the predictability minimization scheme to this problem.
- The 3 experiments demonstrating the quality of the method and superiority over other methods (DMVAE and non-linear CCA) are fitting and convincing. The experiments are also nicely complementary, going from toy datasets to real neuroscience problems.
- The writing is very clear for the most part (and some of my questions may have more to do with my own limited expertise than with clarity).
- The results presented in experiments 2 and 3 are quite promising from the point of view of scientific discovery in neuroscience, and I believe that this architecture has the potential to be useful in many scientific applications beyond neuroscience.

### Weaknesses
1. "Using shared latents from one view to reconstruct the other view guarantees that private information does not leak into the shared latents" => Not sure of that because the shared latent could as well contain other information discarded by the decoder. Please tone down claim if my assessment is correct.
2. See Questions about robustness of the method (may not be weaknesses if properly addressed)
3. Remark: Reading the intro I was missing many citations such as reduced-ranked regression (RRR), CCA, non-linear CCA such as Barlow Twins, CKA, SSL, equivariant SSL, but then found most of them in Discussion. It would be better to announce in Intro that a lot of references will be found in discussion, and at least cite RRR and CCA in intro (ideally applied to neuroscience datasets).

### Questions
1. On robustness (I): how many hyperparameters and how finetuned they need to be to the specific datatset? This information does not come through from reading the paper.
2. On robustness (II): "We use these measurement networks in an adversarial disentangling scheme: in predicting the opposite region’s observations as well as possible, the measurement networks try to exploit any shared information that has leaked into the private latents." => This requires careful implementation in my experience to avoid cyclic behaviors. Did you experience these cyclic behaviors? How did you mitigate them? Is your solution robust?
3. I did not understand the rational for step 2 of the method (in particular Projecting onto Submanifolds). Please explain the rational more concisely.
4. "Importantly, SPLICE confined virtually all private latent variance to two dimensions" => what regularization is responsible for that?
5. LGN - V1 experiment: unclear to me what insights could we have gained from SPLICE here?


typo:
-zˆA  sometimes has the small hat sometimes the big hat

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces a two-stage approach for unsupervised discovery of geometric structures in shared and private latent variables in multi-view data. The first stage uses an autoencoder while optimize for cross-latent (un-)predictability to induce disentanglement of shared and private latent variables. The second stage uses manifold learning technique to preserve geometric structures of data in latent space. Experiments are provided for rotated MNIST dataset, as well as synthetic and real data from neuroscience.

### Strengths
The paper is well written and easy to follow. The experiments showcase some synthetic and real scenarios where shared/private latent geometry is important but overlooked by the considered baselines, where the SPLICE outperforms. An application of the method in neural decoding is shown.

### Weaknesses
1. My main concern lies in novelty of the paper. While the applications to shared-private latent modeling in multi-view settings might be new, the cross-reconstruction framework used in Step 1 uses well-known technique for the purpose of aligning/disentangling latent representations (Schmidhuber, 1992; Chen et al., 2021). The manifold learning technique in Step 2 is from existing literature (L-ISOMAP), and isometry is also a widely adopted prior for representation learning in existing literature.


2. Missing discussions and experiments with related works:
- An advantage of SPLICE is to retain geometric structure of data, by encouraging isometry between data space and latent space. This seems similar to an existing line of work on geometric structure preservation in disentanglement, e.g. (Gropp et al., 2020, Lee et al., 2022, Uscidda et al., 2025). However, there is no discussion or comparison of SPLICE with these methods.
- Another advantage of SPLICE is to disentangle shared/private latents "without a priori knowledge of latent dimensionality" (line 81-83). Related works on this topic are missing, e.g., (Gui et al., 2025) showed that multi-modal contrastive learning adapts to intrinsic dimension of shared latent variables, or (Shrestha et al., 2025) where the authors aimed to tackle content-style learning with unknown latent dimensionality.


3. While qualitative results are provided for Experiment 2, an extensive quantitative result (e.g., with R^2 metric) was not provided.

**References**

Schmidhuber, “Learning Factorial Codes by Predictability Minimization”, Neural Computation, 1992.

Chen et al., “Exploring Simple Siamese Representation Learning”, CVPR, 2021.

Gropp et al., “Isometric Autoencoders”, arXiv:2006.09289, 2020.

Lee et al., “Regularized Autoencoders for Isometric Representation Learning”, ICLR, 2022.

Uscidda et al., “Disentangled Representation Learning with the Gromov-Monge Gap”, ICLR, 2025.

Shrestha et al., “Content-Style Learning from Unaligned Domains: Identifiability under Unknown Latent Dimensions”, ICLR, 2025.

Gui et al., “Multi-modal Contrastive Learning Adapts to Intrinsic Dimensions of Shared Latent Variables”, arXiv:2505.12473, 2025.

### Questions
1. Since SPLICE is a two-step procedure, which step is mainly responsible for the reported good performance in the experiments? What would happen if another method for disentangling shared-private latent variables is used for Step 1, together with L-ISOMAP in Step 2?
2. Did the authors try other manifold learning methods in place of L-ISOMAP for Step 2?
3. What are other potential applications for SPLICE, besides neuroscience?

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The manuscript presents SPLICE, a two-step neural network approach for unsupervised disentanglement in multi-view data, aiming to infer interpretable, non-linearly mixed shared and private latent representations while preserving their intrinsic submanifold geometry. The method first employs a crossed autoencoder and the predictability minimization objective for effective disentangling, and then refines the latents using a geometry-preserving loss derived from estimated geodesic distances along the submanifolds (calculated efficiently using landmarks). Evaluated on simulated CV data (rotated MNIST and SPRITES), simulated neural activity, and a real-world neurophysiological dataset, SPLICE yields interpretable representations, and exhibits robustness to mis-specified latent dimensionality compared to state-of-the-art disentangling and shared-only methods.

### Strengths
- Disentanglements include robustness to misspecification of latent dimensions as well as the loss regularization parameter $\lambda_{geo}$.
- Disentanglements show clearer separation compared to baselines and superior interpretability via geometry preservation.
- Rigorous experimental analysis spans well defined simulations covering rotated MNIST, SPRITES as well as simulated and real neural spiking datasets.

### Weaknesses
- Computational Efficiency:
  - How does the runtime of SPLICE compare to the other baseline approaches?
- Clarity:
  - The methodology in Step 2 - Geometry Identification and Preservation -  requires clarification.
  - Which method did you use to estimate the nearest neighbor graph?
- Hyperparameter Determination and Stability:
  - Clarify the rationale for taking multiple gradient steps for the measurement networks per autoencoder update, as mentioned in the manuscript. The hyperparameter for this is currently missing.
  - Algorithms 1 and 2 suggest frequent resetting of the measurement networks. Please discuss the convergence stability for practitioners.
  - State whether a fixed parameter set of $n_{msr}$ and $T_{restart}$ worked effectively across all datasets or if extensive tuning was required.

### Questions
see weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
