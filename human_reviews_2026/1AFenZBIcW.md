# No More DeLuLu: A Kernel-Based Activation-Free Neural Networks

- Avg Score: 4.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 8

## Abstract
We introduce the ⵟ-product, a kernel operator that combines quadratic alignment with inverse-square interactions. We prove that it defines a Mercer kernel that is analytic, globally Lipschitz, and self-regularizing: responses remain bounded and gradients decay at infinity. Neural Matter Networks (NMNs), constructed as linear combinations of ⵟ-atoms, are universal approximators on compact domains without explicit nonlinear activations. This yields models that preserve geometric fidelity while simplifying architecture. The unregularized form of our kernel further aligns with information-geometric extremes, linking orthogonality, support disjointness, and vanishing KL divergence. Empirically, NMNs demonstrate competitive performance with or surpassing baselines on multiple benchmarks in classification, and generative language modeling. Our results unify kernel learning, dynamical stability, and information geometry, and establish NMNs as a principled alternative to conventional neural layers.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose an activation free kernel-based deep model. The kernel is constructed using the Yat-product, which gives us the geometric interpretability. The universality of the proposed model is investigated. In addition, since the proposed model is activation free, it is computationally efficient.

### Strengths
The proposed model is based on the Yat-product, which has good geomrtric properties and introduces nonlinearlity in a computationally efficient way. The topic is interesting and relevant to the community.

### Weaknesses
- For me, it was unclear that why the universality is guaranteed with the setting of Theorem 2.2. Using the universality of the Mercer's kernel, we may derive the universality of the shallow model in some sense. Hower, in Theorem 2.2, the authors assume $w_i\in W$. If $W$ is a singeton and $K$ is large enough, I don't think the space $\mathcal{G}$ is dense in $C(K)$. Am I missing something?
In addition, the universality of the shallow model is directly derived by the universality of the RKHS associated with the Mercer's kernel. Since the paper mainly focus on the deep model, it would be interesting to investigate what is the advantages to constructing deep models compared to shallow models. In the case of neural networks, there are some interpretations that support advantages of deep models (e.g. Cohen et al., JMLR, 2016). Do you have any ideas or insights regarding this?

- The proposed model seems to be related to deep model with kernels (e.g. Youngmin Cho and Lawrence K. Saul, NeurIPS 2009). Can the proposed model can be regarded as a special case of these models?

- The readabiity should be improved. Since some important statements are in Appendix (e.g., Theorem A.7 and A.8), it was hard for me to understand the theoretical properties of the prposed model without reading the appendix. Although the proofs are not necessality in the main text, important statements to understand the proposed model should be in main text.
In addition, some important explanations are missing. For example, what the colors and the stars in Figure 2 indicate?

**Minor comments**  
l 231: "sepaable" should be "separable"?

### Questions
Please see the weakness section.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces the "E-product", a kernel operator $k_{\mathcal{E}}(x,w)=\frac{\langle x,w\rangle^{2}}{||x-w||^{2}+\epsilon}$, designed to combine geometric alignment and spatial proximity information, thereby aiming to eliminate the need for standard activation functions in neural networks. The authors develop Neural-Matter Networks (NMNs) based on this kernel. Key theoretical results include proving that the E-product satisfies Mercer's conditions (Theorem 2.1) and that NMNs possess universal approximation capabilities (Theorem 2.2). The paper also presents empirical results on computer vision benchmarks (CIFAR-10/100, ImageNet, etc.) and a language modeling task, suggesting that NMN-based architectures ("Aether" variants) achieve performance comparable or superior to baseline models like ResNet, ViT, and GPT-2, while offering potential benefits in memory efficiency and architectural simplicity.

### Strengths
The paper introduces a novel, conceptually interesting kernel, the E-product, inspired by physical inverse-square laws, which intrinsically combines vector alignment and proximity.

### Weaknesses
The paper suffers from significant weaknesses in its theory and experiments 

### Theory 
* Flawed universality proof and mis-invoked theory from literature: A critical step in the universality proof (Theorem 2.2, Appendix A.8) relies on citing "(Steinwart & Christmann, 2008, Thm. 4.62)" to justify that the RKHS of their derived zonal kernel is dense in $C(S^{d-1})$. This constitutes a logical gap because the actual Theorem 4.62 in the cited book concerns kernels on the discrete space $\mathbb{N}_0$ and conditions under which the RKHS is *not dense* in $L_q$ spaces. The cited theorem is therefore entirely irrelevant to the paper's context (continuous zonal kernels on the sphere $S^{d-1}$ and density in $C(S^{d-1})$), providing zero support for the claim and misrepresenting the cited literature. Potential replacements from the same book, like Theorem 4.56 concerning universal kernels via an algebra condition, have different prerequisites that the paper's proof does not establish, making them unsuitable substitutes without significant proof restructuring. Thus, this key step in the universality argument lacks valid justification from the cited modern reference.
* It appears that for the Mercer kernel proof step concerning $k_2$,  authors should have cited ( Theorem 3)  from "Schoenberg, I. J. (1938), Metric Spaces and Completely Monotone Functions, Annals of Mathematics, instead of Schönberg (1948),  

### Empirical 
* **Weaker than expected vision baselines** The empirical results presented for vision tasks (Table 2) claim "competitive or surpassing" performance for the proposed "Aether" variants (Aether-ResNet, Aether-ViT) compared to baseline ResNet and ViT models. However, the reported performance of these baselines, trained from scratch by the authors, appears substantially lower than widely accepted standard benchmarks, particularly on ImageNet. For example, the ResNet-50 baseline is reported at 74.13% top-1 accuracy, whereas the original paper and subsequent works typically report ~76-77% or even higher. Even the original ResNet paper from 2015, it reports two top-1 accuracies of 77.15% and 79.26%, both much higher than 74% used in this paper. Similarly, the ViT-Small baseline is reported at 69.91%, far below the ~77-81% range expected for comparable ViT-Base models trained from scratch or transferred. This constitutes a comparison against weak baselines, which potentially inflates the perceived effectiveness of the proposed method. The claim of competitiveness is therefore unsubstantiated against the relevant state-of-the-art.
* The language modeling experiments compare AetherGPT against a standard GPT-2 baseline, claiming improved validation loss (2.29 vs 2.43) and highlighting architectural simplification via removal of activation and normalization layers. However, the evaluation fails to include a comparison with relevant and strong baselines that *also* achieve normalization-free training for Transformers, most notably ReZero (Bachlechner et al., 2021). ReZero provides a simple and effective mechanism (a single learnable parameter per block, initialized to zero) to train very deep Transformers without normalization layers, often achieving faster convergence. Without comparing AetherGPT to ReZero, the paper's claims about the practical benefits and novelty of its architectural simplification (specifically, removing normalization) are unsubstantiated relative to existing state-of-the-art techniques addressing the same goal.
* Key empirical results in the main tables (e.g., Table 2) are presented as single numbers, lacking error bars or explicit reporting of means/standard deviations over multiple runs, which makes it harder to assess the statistical significance and robustness of the findings (though the Table 2 caption implies multiple runs were done for the source data).


--- 
For completeness and transparency, here are the verbatim texts of the theorems from Steinwart & Christmann (2008) discussed in the review:

> **Theorem 4.62:**   from p. 157.
> There exists a bounded, strictly positive definite kernel $k$ on $X:=\mathbb{N}\_{0}$ with $k(\cdot,x)\in c\_{0}(X)$ for all $x\in X$ such that for all finite measures $\mu$ on $X$ with $\mu(\{x\})>0$, $x\in X$, and all $q\in[1,\infty]$, the RKHS $H$ of $k$ is **not dense** in $L\_{q}(\mu)$.

Potential alternative (but conditions are not met) 
> **Theorem 4.56** (A test for universality).** Let $X$ be a compact metric space and $k$ be a continuous kernel on $X$ with $k(x,x)>0$ for all $x\in X$. Suppose that we have an injective feature map $\Phi:X\rightarrow l\_{2}$ of $k$. We write $\Phi\_{n}:X\rightarrow\mathbb{R}$ for its n-th component, i.e., $\Phi(x)=(\Phi\_{n}(x))_{n\in\mathbb{N}}$ for $x\in X$. If $\mathcal{A}:=span\{\Phi\_{n}:n\in\mathbb{N}\}$ is an **algebra**, then $k$ is universal.

### Questions
I'm open to reading authors' responses to all of the above questions/concerns

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose an alternative to the standard 'Linear+Activation' unit in
neural networks, called a 'neural matter network' layer. This layer consists of
the sum of several 'E+-products' between weights and activations. Similar
replacements are proposed for convolutional and attention layers. The E+-product
(and thus the NMN layers) are motivated through many nice theoretical
properties, such as universal approximation, stable gradient etc.. NMN layers
are more computationally expensive (2x) than regular MLP layers, but appear to allow
removal of normalization layers. Based on the experiments, NMN layers provide marginal performance
gains.

### Strengths
The idea of using the E+-product in NNs is simple, and as far as I am aware, original.
Overall I find the idea very elegant and compelling! Figure 1, 2 and 3 do a good job of illustrating benefits of the E+-product. The results (while needing some more work, and being perhaps marginal) are promising.

Finally, the theoretical contributions appear solid, with many mathematical results to
back up claims of nice qualities of the E+-product/NMN layers (such as universal approximation, stable gradient, etc.).

### Weaknesses
I find the related work section very lacking. The related work section does not seem at all thorough enough in terms
of considering NN activation replacements in the literature, leading me to perhaps unnecessarily doubt the novelty of the method. A few examples are: KAN [1], ErfAct and Pserf
[2], and even GLU [3] and Spiking NNs [4], but I expect to see more.

- [1]: Liu, Ziming, et al. "Kan: Kolmogorov-arnold networks." arXiv preprint arXiv:2404.19756 (2024).
- [2]: Biswas, Koushik, et al. "ErfAct: Non-monotonic smooth trainable Activation Functions." arXiv preprint arXiv:2109.04386 (2021).
- [3]: Shazeer, Noam. "Glu variants improve transformer." arXiv preprint arXiv:2002.05202 (2020).
- [4]: Tavanaei, Amirhossein, et al. "Deep learning in spiking neural networks." Neural networks 111 (2019): 47-63.


Authors are also missing some related works on more modern kernelized neural
networks. For example, [4,5,6] should be added, but I again expect to see more.

- [4]: Wilson, Andrew Gordon, et al. "Deep kernel learning." Artificial intelligence and statistics. PMLR, 2016.
- [5]: Yang, Adam X., et al. "A theory of representation learning gives a deep generalisation of kernel methods." International Conference on Machine Learning. PMLR, 2023.
- [6]: Aitchison, Laurence, Adam Yang, and Sebastian W. Ober. "Deep kernel processes." International Conference on Machine Learning. PMLR, 2021.

Overall, the related work for inverse square laws is overemphasized, and the
rest is underemphasized.

Some of the motivations are slightly dubious.
The practical motivation for replacing MLP layers could be improved: for example
in the introduction, I find the motivation based on 'complexity of NN architectures' very weak. Authors take aim at normalization layers, attention mechanisms, but the NMN layers are also complex, and I'm also unsure what is meant by 'sophisticated regularization' on line 054/055. Further, one proposed motivation for E+-product is
its infinite differentiability, but there are many smooth activations to choose
from (e.g. SiLU, GELU) which offer this.

The results are promising, but lacking. Table 2 has many missing entries. For
the fineweb pretraining experiments, I would like to see more configurations
checked, like different model sizes, different architectures (e.g. Qwen3 or
Llama3), performance with/without QK-norm, etc.. Additionally, some of the
experimental details are unclear; what numerical precision was used for
experiments? One worry is that NMNs require higher precision than standard MLP
layers, which ought to be easy to address.

### Questions
I would be happy to upgrade my score if authors addressed the following major
issues:

- More comprehensive related work section, as discussed in the 'weaknesses' section above.
- More thorough experiments. Table 2 should be completed, and more
  configurations should be checked for the language pretraining experiments.
  Experimental clarifications on numerical precision would also be appreciated.


More minor suggestions/Questions/typos:
- Exactly which sophisticated regularization are you referring to on line 054/055?
- Why did you use the prefix 'Aether'?
- Why 'YAT product' and not 'E+-product' for the title of the 3rd column in Figure 1? Log1p is also a bit ugly.
- Fix legend in Figure 3 (it currently hovers on 'digit 7' column).
- Figure 7 is a screenshot and not a proper figure, and the training loss y-axis
  should be zoomed in more so that we can better distinguish the curve.
- Typo on line 231 'sepaable'.
- The authors mention invariance to flipping the sign of the weights (line 303). Could you explain
  why you think this is important/relevant in practice?
- Could you explain exactly where memory usage improvements come from with the NMN layers?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes the \textbf{ⵟ-product}, a new kernel operator meant to replace the standard “linear layer + activation” block in neural networks. The operator combines a squared dot product (for alignment) with an inverse-square distance term (for proximity), giving it both polynomial-like and RBF-like behavior in one analytic form. Networks built from these units, called \textbf{Neural Matter Networks (NMNs)}, supposedly achieve nonlinearity without any explicit activation functions or normalization layers. 
The authors prove (actually most are well known standard or follow standard pattern though) several nice properties (Mercer kernel, Lipschitz continuity, bounded gradients, and universal approximation) and show competitive results on image classification (ResNet/ViT variants) and language modeling (GPT-2 variant). Conceptually, the paper tries to unify geometric computation, kernel theory, and information geometry under a single physical-inspired operator.

### Strengths
-- A fresh attempt to rethink the basic linear+activation structure using kernel theory.

-- The paper is built on solid mathematical foundations (Mercer property, boundedness, and universal approximation), though most of these results follow established proof techniques and are extensions of well-known theoretical frameworks.

-- Demonstrates activation- and normalization-free architectures that still train stably.

-- The claim that LayerNorm and BatchNorm are unnecessary is intriguing and convincingly supported through experiments on standard architectures. This is a strong result, since these normalization layers are considered de facto essential in modern networks. Demonstrating that stable training is possible without them is valuable, as it offers new insights into understanding and simplifying neural network design.

--  Good geometric intuition: the “potential well” interpretation and prototype visualizations are compelling.

--  Interesting robustness property: invariance to sign-flipped prototypes ($w \rightarrow -w$).

### Weaknesses
-- The claimed self-regularization may not hold in very high-dimensional settings. In other words, in very high-dimensional embeddings, where all points are far apart, does the denominator saturate? How do you handle that?

-- Missing comparisons to other distance-aware kernels (RBF instead of the distance term, i.e., $\[
k_{\text{RBF-dot}^2}(x, w)
    = (x^\top w)^2\ * \exp\left(-\lambda\,\|x - w\|^2\right)
\]$
 etc).

-- The “inverse-square law” motivation is appealing but mostly heuristic; no toy examples justify why that decay is better than exponential, or comparison with the existing large scale experiments.

--  Interesting robustness property: invariance to sign-flipped prototypes ($w \rightarrow -w$). ==> But not clear how to connect this with use-cases, in what cases its useful.

### Questions
-- How does the ⵟ-product compare empirically with $\[
k_{\text{RBF-dot}^2}(x, w)
    = (x^\top w)^2 * \exp \left(-\lambda\,\|x - w\|^2\right)
\]$ kernels using the same architectures?

--  In very high-dimensional embeddings, where all points are far apart, does the denominator saturate? How do you handle that?
L 258: Comparison of decision boundaries: conventional linear model (left) shows unbounded prototype growth, while ⵟ-product method (right) learns bounded, representative prototypes that better capture class distributions. ==> in high dimensions, the denominator may become nearly constant if all points are far apart, reducing the self-regularization effect.

-- Is the $w \rightarrow -w$ invariance always desirable? Could it hurt tasks where direction matters?

-- Have you tried changing the exponent in the denominator (e.g., $1/r^p$ with $p \neq 2$) ,as why use p=2 only, may be you want to add some more insight regarding that ?

### Soundness
3

### Presentation
3

### Contribution
3
