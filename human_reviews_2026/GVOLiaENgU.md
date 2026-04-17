# A Bayesian Nonparametric Framework For Learning Disentangled Representations

- Decision: Accept (Poster)
- Scores: 6, 8, 4

## Abstract
Disentangled representation learning aims to identify and organize the underlying sources of variation in observed data. However, learning disentangled representations from observational data alone without any additional supervision necessitates inductive biases to solve the fundamental identifiability problem of uniquely recovering the true latent structure and parameters of the data-generating process. Existing methods address this by imposing heuristic inductive biases that typically lack these theoretical identifiability guarantees. Additionally, these methods rely on strong regularization to impose these inductive biases, creating an inherent trade-off in which stronger regularization improves disentanglement but limits the latent capacity to represent underlying variations. To address both challenges, we propose a principled generative model with a Bayesian nonparametric hierarchical mixture prior that embeds inductive biases within a provably identifiable framework for unsupervised disentanglement. Specifically, the hierarchical mixture prior imposes the structural constraints necessary for identifiability guarantees, while the nonparametric formulation allows the latent representation to scale with infinite capacity to faithfully represent the complete set of underlying variations without violating these structural constraints. To enable tractable inference under this nonparametric hierarchical prior, we develop a structured variational inference framework with a nested variational family that both preserves the hierarchical structure of the identifiable generative model and approximates the expressiveness of the nonparametric prior. We evaluate our proposed probabilistic model on standard disentanglement benchmarks, 3DShapes and MPI3D datasets characterized by diverse source variation distributions, to demonstrate that our method consistently outperforms strong baseline models through structural biases and a unified objective function, obviating the need for auxiliary regularization constraints or careful hyperparameter tuning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a Bayesian nonparametric prior over the embedding space of latent-quantization autoencoders to determine the appropriate latent space complexity and the optimal strength of regularization constraints, and shows the effectiveness with experiments on two benchmark datasets. The proposed method produces competitive performance relative to some baseline methods.

### Strengths
- A sound theoretical derivation of the new framework for unsupervised learning of disentangled representation.

- Experiments support the proposed framework with relative superiority to the baselines.

### Weaknesses
- Relatively weak novelty on the methods for disentangled representation learning. More recent methods should be compared and discussed.

- The comparing methods should be updated with recent methods.

### Questions
- What is the main difference of the proposed method compared with the recent methods for disentangled representation learning? Most of the methods in related works are out of dated.

- Is the performance gain enough to argue as an achievement? Depending on the metrics, some of the baselines produce better performance. More in-depth discussion is required.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces Bayes-QLAE, an innovative framework for learning disentangled representations through a Bayesian non-parametric prior. Its core idea is to use a Dirichlet Process to allow the codebook for each latent dimension to adaptively adjust its capacity based on data complexity. Experiments show that the method achieves disentanglement performance comparable to strong baselines on standard benchmarks without the need for additional regularization.

### Strengths
1. Transforming the codebook size from a fixed hyperparameter into a quantity learned from data via a  Dirichlet Process is an elegant and significant advancement that directly addresses a key limitation of existing quantization methods.
2. The paper provides solid and rigorous theoretical proofs, while also achieving competitive results on standard datasets. Furthermore, ablation studies analyze the contribution of the model's various components.

### Weaknesses
1. While making the codebook size adaptive is intuitively appealing, it contradicts findings from prior work (e.g., Tripod et al.), which suggests that maintaining a smaller codebook is beneficial for disentanglement. How does the proposed method balance the tendency of the codebook to grow with the need to maintain a compact, disentangled representation? A clear discussion or analysis of this trade-off is necessary.
2. The current experimental results are limited to standard disentanglement metrics like InfoMCE and DCI. A more comprehensive evaluation is needed. For instance, a comparison of reconstruction accuracy against baseline methods would provide crucial insight into whether the gains in disentanglement come at the cost of representation fidelity.
3. The paper hypothesizes that allowing adaptive codebook sizes helps align individual factors with single dimensions. However, the experiments fail to substantiate this claim. The final learned sizes of the codebook dimensions should be reported to demonstrate this adaptivity. Furthermore, visualizing the effect of a single latent variable is essential to qualitatively assess the smoothness and interpretability of the learned representations, which is a standard practice for evaluating disentanglement.

### Questions
Please refer to Weakness section.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This work replaces the gaussian prior of latent variables with discrete codebook with a nonparametric Dirichlet Process (DP). This process can adjust representation capacity according to the data complexity. The authors propose a hierarchical prior structure to capture complex dependences of latent variables.  The proposed method is verified on 3DShapes and MPI3D to show superior disentanglement scores.

### Strengths
This work aiming at removing assumption of the capacity of each factor could be an important step to practical applications of disentanglement learning.

The proposed solution, Dirichlet Process for a Bayesian nonparametric prior on latent variables seems technically sounds.

The proposed method, Bayes-QLAE, achieves good disentanglement metrics on both 3DShapes and MPI3D.

### Weaknesses
The proposed method still lacks practical proof in real situations where some combinations of generative factors are missing.
Experiments did not verify how the DP adjusts the capacity of each factor, which is an important claim of this work.
No visualization to demonstrate the reconstruction quality.
The experimental results show that the proposed Bayes-QLAE did not surpass Tripod on 3Dshapes and MPI3D.

### Questions
What are the advantages of Bayes-QLAE compared to Tripod?
Why do we need the hierarchical Bayesian nonparametric approach? Are there any special benefits beyond disentanglement?
Can the experiments be added to demonstrate the advantage of the work for practical problems? 
Also, adding training details would be beneficial. 
I case of deep encoders with a lot of non-linearities consisting of a high-dimensional latent space, would the encoder network not be able to learn to project the data into a space where Gaussian prior assumption would be enough? 
Can the principles laid in this work for learning disentangled representation be shown on some recent architectures (e.g., disentanglement in the latent space of diffusion models)?

### Soundness
3

### Presentation
2

### Contribution
2
