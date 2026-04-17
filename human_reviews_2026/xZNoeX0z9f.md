# Landing with the Score: Riemannian Optimization through Denoising

- Decision: Accept (Poster)
- Scores: 8, 2, 6, 10

## Abstract
Under the \emph{data manifold hypothesis}, high-dimensional data concentrate near a low-dimensional manifold. We study Riemannian optimization when this manifold is only given implicitly through the data distribution, and standard geometric operations are unavailable. 
This formulation captures a broad class of data-driven design problems that are central to modern generative AI. Our key idea is a \emph{link function} that ties the data distribution to the geometric quantities needed for optimization: its gradient and Hessian recover the projection onto the manifold and its tangent space in the small-noise regime. This construction is directly connected to the score function in diffusion models, allowing us to leverage well-studied parameterizations, efficient training procedures, and even
pretrained score networks from the diffusion model literature to perform optimization. On top of this foundation, we develop two {efficient} inference-time algorithms for optimization over data manifolds: \emph{Denoising Landing Flow} (DLF) and \emph{Denoising Riemannian Gradient Descent} (DRGD). We provide theoretical guarantees for approximate feasibility (manifold adherence) and optimality (small Riemannian gradient norm). We demonstrate the effectiveness of our approach on finite-horizon reference tracking tasks in data-driven control, illustrating their potential for practical generative and design applications.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
In this paper, the author's introduce a new way to harness data manifolds described using score-based diffusion models. By introducing a so-called link function, whose gradient is related to the score function and thus to the projection onto the data manifold, they derive Riemannian optimization algorithms that work directly on the score-based description of the manifold. The authors analyze the expected performance of their algorithms under accuracy assumptions on the underlying score functions. Finally, they demonstrate the applicability of their method in two synthetic examples.

### Strengths
To me, the main strength of the paper is that it uses a representation of data manifold that is widely used in current machine learning (i.e. score-based models) to introduce practically usable algorithms on said manifolds. There is, of course, a lot of papers currently that try to achieve something to this end. Here, I enjoyed the clear combination of geometry and probabilistic components in the construction of the method. The resulting algorithms seem natural and actually usable in practice.

Furthermore, I appreciate the provided theoretical results, which give an insight into what to expect of the method. At the same time, the authors do not oversell the impact of their results 

Finally, I found the paper to be well written. I could read the paper in one go without any road blocks even though I am not an expert on optimization. After this, I believe to have understood the method to a point where I could now implement it myself, which is huge plus in my book. I also enjoyed the discussion of related work and clear positioning of the paper.

### Weaknesses
The primary weakness of the presented, from my perspective, is the experimental validation. It is limited to two synthetic examples, one on O(n) and one discrete data-driven control. Since the authors themselves make the point that score-based models have become ubiquitous in modern machine learning, I would have hope that one of these would also be considered here. For example, for image generation models one could try to modify a given image to mislead a given classifier. There are certainly other creative examples one could come up with. However, I understand that this requires a different expertise and computational resources, and this is not a deal-breaker for me.

### Questions
- the authors should please check again their usage of \citep and \citet. In some places, it seems to be off.
- I would maybe switch (8) and (9), so that the method is first introduced with the more accurate \pi before going to the score-based approximation.
- One reference I missed is "What regularized auto-encoders learn from the data-generating distribution"  by Guillaume Alain and Yoshua Bengio. It also analyses denoising autoencoders (although not considering an underlying manifold) and it would be nice to set it into relation with the present work.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper targets the problem of optimizing objective function on data manifolds, i.e., manifolds that are given only implicitly through the data distribution. The paper proposes to do so via Riemannian optimization, by obtaining the required quantities (Riemannian gradient and Hessian, projection onto the tangent space) through a link function associated with the data distribution which is associated with a diffusion score function.

### Strengths
The paper targets an interesting problem, which is relevant to the field of manifold learning. The approach nicely combines manifold learning with elements from recent works in diffusion models and score matching. The paper is generally clearly written.

### Weaknesses
The paper does not mention an important category of related works in manifold learning, where the data manifold is characterized via the pullback metric of the decoder of a (deep) generative model (Tosi et al, Arvanitidis et al, Syrota et al, Diepeveen et al). In such cases, the data manifold is endowed with a Riemannian pullback metric, which can directly be used for computing explicit manifold operations for Riemannian optimization. In this case, the motivation of the paper (“In this data-driven regime, methods from classical Riemannian optimization cannot be applied directly, since they rely on explicit manifold operations such as tangent-space projection, retraction, or exponential maps”) does not hold anymore, as such quantities can be computed based on the Riemannian metric. I think that these approaches must be discussed, and some of the claims of the paper revised according to this perspective. 

In addition, it is unclear to me if the proposed approach really qualifies as Riemannian optimization. In Riemannian optimization (Absil et al, Boumal, as cited in the paper), the objective function is defined on the manifold of interest and the optimization strictly takes place on the manifold, i.e., all iterate x lie on M. In contrast, in equation 10, the second term belongs to the orthogonal space to the manifold, and pushes the iterate towards the manifold, i.e., the iterates do not necessarily belong to the manifold of interest. 

The experiments provided in the paper are simple and conducted only on low-dimensional spaces. I would like to see how the approach performs on more complex and higher dimensional optimization problems. The fact that score functions and diffusion models scale well to high dimensions could allow the approach to scale well, and could be presented as a strength if shown experimentally.

All in all, the performance of the approach seems very dependent on the value of $\sigma$. In Fig. 3, only $\sigma=1e^{-5}$ leads to a good optimization, with significant differences compared to $\sigma=1e^{-4}$. It is unclear how the approach would perform for different values of $\sigma$ in more complex optimization problems.

Given that the link function seems to be available based on the data distribution, which is assumed to be given, it is unclear to me why it is necessary to additionally learn a network $s$ to represent the gradient of the link function. 


References: 
- Tosi, A., Hauberg, S., Vellido, A. & Lawrence, N. D. Metrics for Probabilistic Geometries. in UAI (2014).
- Arvanitidis, G., Hansen, L. K. & Hauberg, S. Latent Space Oddity: on the Curvature of Deep Generative Models. in International Conference on Learning Representations (ICLR) (2018).
- Syrota, S., Zainchkovskyy, Y., Xi, J., Bloem-Reddy, B. & Hauberg, S. Identifying metric structures of deep latent variable models. in ICML (2025). 
- Diepeveen, W., Batzolis, G., Shumaylov, Z. & Schönlieb, C.-B. Score-based pullback Riemannian geometry: Extracting the Data Manifold Geometry using Anisotropic Flows. in ICML (2025).

### Questions
- It is unclear to me why it is necessary to learn a network $s$. Could the gradient $\nabla \ell_{\sigma}$ and Hessian be computed solely from equation 3 instead?
- How is the network $s$ trained? Which training data are needed?
- How does the proposed approach compare to Riemannian optimization on learned data manifold endowed with a Riemannian metric?
- How is the probability measure obtained in general? How is it obtained in the problems considered in the experiments?
- How does the approach perform for more complex and higher dimensional optimization problems?
- How should $\sigma$ be chosen in practice?
- What does “exact landing” refers to in Figure 3?
- How does the proposed approach compare to gradient descent on the O(n) manifold? I know that the paper considers the manifold as unknown, but providing such a comparison would still be valuable to understand the influence of the manifold learning component.
- It would be better if figures corresponding to main results, e.g., Figure 3, would appear in the main text. If space is an issue, I would suggest to reduce the introduction (which takes 4 pages out of 9).

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper studies constrained optimization on data manifolds when the manifold is only available implicitly via samples. It smooths the data distribution with a Gaussian of scale σ and defines a “link function” $\ell_\sigma$ whose gradient and Hessian can be related to manifold operations. Theorem 1 shows that as $\sigma \to 0$, $x + \sigma^2\nabla log p_\sigma(x)$ uniformly approximates the closest-point projection $\pi(x)$ to the manifold, and $I + \sigma^2\nabla^2 log p_\sigma(x)$ approximates the Jacobian of that projection with error $O(\sigma|log \sigma|^3)$ on a tubular neighborhood. Corollary 2 then states that, on the manifold, $I + \sigma^2\nabla^2 log p_\sigma(x) \to P_{TM}$. By this process, the denoising score approximates manifold projection/retraction.

Building on this, the paper proposes two algorithms that only require inference-time access to a trained score network sσθ and its Jacobian–vector products: (i) Denoising Landing Flow (DLF), a continuous-time flow combining a Riemannian-like step with a “landing” term toward the manifold (eq. 8), and (ii) Denoising Riemannian Gradient Descent (DRGD), a discrete method that applies an approximate tangent step followed by an approximate retraction given by the Tweedie mean (eq. 12). The paper provides non‑asymptotic guarantees: for DLF, accumulation points are near the manifold and have small Riemannian gradient norm with bias proportional to the geometry-approximation error (Theorem 3); for DRGD, an average Riemannian gradient-norm bound with similar bias terms (Theorem 5). Theorem 20 shows that with exact manifold operators ($\sigma=0$) the flow reduces to classical exact Riemannian gradient flow. Experiments on illustrate the approach. Figures 1 and 2 show optimized trajectories tracking references more closely than the best sample from the training set.

### Strengths
* The link function viewsystematically connect $\nabla \log p_\sigma$ and $\nabla^2 \log p_\sigma$ to retraction and tangent projection, respectively, with explicit uniform error bounds (eq. 5). This is a useful and reusable idea. 

* Theorem 1 leverages Laplace-method estimates to obtain uniform $O(\sigma|log \sigma|^3)$ control on a tubular neighborhood; the constants are traced to curvature/tubular radius and density smoothness (Appendix D). This is technically solid and extends beyond heuristic “score is normal to the manifold” statements. 

* Both DLF and DRGD require s(x) and Jacobian–vector products s′(x)v (Remark 4), which makes them practical with modern autodiff. 

* The analysis is structured so that (a) convergence of optimization with approximate manifold operators is established assuming an $L_{\infty}$ approximation error, and (b) Theorem 1 supplies such an error by letting $\sigma$ be small. This cleanly decomposes the problem. 

* The paper clearly distinguishes constrained optimization from posterior-guided sampling and explains why small “temperatures” in the latter can drift off the true manifold—motivating the explicit feasibility machinery here (Section 1.2).

### Weaknesses
* The dependence on $\sigma \to 0$ is strong in practice. Theorem 1 and Corollary 2 fundamentally rely on $\sigma$ being small, and all optimization guarantees carry a bias term $\sigma|log \sigma|^2$. Training accurate scores precisely in the small‑$\sigma$ regime is hard, so the theoretically clean limit may be brittle in practice. The paper acknowledges this by proposing a landing flow, but still the guarantees rely on $\sigma$ shrinking (Theorem 3). 

* Uniform $L_\infty$ approximation is a strong assumption. Convergence theorems (7) require s to approximate both $\pi_\sigma$ and its Jacobian uniformly over a neighborhood. Score models are typically trained in L2 (denoising score matching). The authors do note this (Remark 6) and defer L2‑type analyses to future work.

* Comparison to classical Riemannian GD is limited. With $\sigma = 0$ and exact operators, DLF reduces to Riemannian gradient flow with exact landing (Theorem 20), which is consistent with classical results; however, for $\sigma > 0$ the paper provides average gradient‑norm bounds with additive bias rather than full complexity guarantees comparable to known rates for feasible Riemannian methods. A more direct side‑by‑side empirical and theoretical comparison to “Riemannian GD with a known retraction” would be  useful.

* The experiments demonstrate improvement over the best training sample, but comparisons to explicit‑manifold baselines (where available) are limited to an “exact landing flow”. There are no comparisons to standard Riemannian optimizers using the true retraction on the same objective, nor ablations for the effect of $\sigma$ and Jacobian accuracy in higher‑dimensional settings.

### Questions
1. The theorems assume a fixed small $\sigma$ to control the geometry error. In practice, do you anneal $\sigma$ during optimization? Or do you pick a $\sigma$ that trades off stability and bias? Some concrete empirical guidance would be helpful.
2. DRGD performs “tangent step + retraction” using s and s′. If we hypothetically replace $s$ with the exact projection $\pi$ and $s'$ with $P_{T_x M}$, do your bounds recover the standard complexity guarantees for Riemannian GD (up to constants), or is there an intrinsic loss because you work in ambient coordinates with the landing penalty? It would be helpful to position Theorem 5 against known rates. 
3. For real networks, $s$ may not be conservative. Apart from the barrier‑type argument in Appendix E.4, are there practical regularizers or architectural choices (e.g., gradient‑field parameterizations) that make s approximately integrable?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
The goal of the paper is creating a novel framework that ties theoretical Riemannian optimization tools (retraction, computing the Riemannian gradient, projection onto a tangent space, etc) to practical methods that rely solely on the data distribution (e.g., diffusion models). The proposed work is impactful and impressive in numerous ways. I highly recommend this paper to be accepted.

### Strengths
Strong points: the paper is well-written, with clearly made arguments, well-structured problem, meaningful theoretical as well as experimental results. The paper provides a comprehensive literature review and acknowledges its own limitations.

The paper supports results with rigorous mathematical proofs, as well as intuition-based explanations, which I appreciated very much. This work has many, many strengths. To name a few: this work provides a rigorous theoretical analysis of how Riemannian optimization tools can in fact approximate practical operations such as projection (for denoising purposes), and how this can be done by learning from data. This works provides a massive step forward in this literature by connecting two fields that have been struggling to converge -- theory of high-dimensional geometry and diffusion models.

The significance of this work is huge. It makes a rigorous connection between theoretical and practical fields (Riemannian optimization and diffusion models), which is very high-impact and rare to come across. Creating such impactful and meaningful work requires a vast and deep knowledge of the field, and one can dream to be at that level as a researcher!

Authors make several innovations. Impressively, authors prove that the quantities in equation 4 converge to manifold operations as $\sigma \rightarrow 0$. For example, the proposed link function provides a way to approximately project points onto the unknown manifold wihtout explicitly knowing its equation, using the score function learned from data. Authors also clearly demonstrate how to use these operations for optimization by proposing two concrete algorithms that tell us how exactly their work can be implemented. Lastly, one non-obvious strength is in experiments, where the authors show that even though theoretical results are strong when $\sigma \rightarrow 0$, experiments work even when $\sigma$ is nonzero.

### Weaknesses
The authors heavily rely on the claim that classical algorithms require knowledge of the manifold to be able to use Riemannian optimization tools, however, this claim is not necessarily true. Plenty of works, especially [1], develop provable, feasible, accurate, learnable, and inference-time efficient algorithms that create machinery to use Riemannian optimization tools without knowledge about the manifold, while having access to noisy samples only. In fact, the noise level can also be relatively high. Authors are also missing another key work [2] which provides a provable and accurate denoising method without knowing the underlying manifold. While I find the results of the paper impressive and useful, if one were to make the paper even stronger, I would recommend re-positioning / re-phrasing the argument that "classical Riemannian optimization cannot be applied directly, since they rely on explicit knowledge of the manifold."

Lastly, this work heavily relies on the fact that a diffusion model will learn the score function well. If this is not the case, the proposed framework will not be useful.

[1] Wang, Shiyu, Mariam Avagyan, Yihan Shen, Arnaud Lamy, Tingran Wang, Szabolcs Márka, Zsuzsa Márka, and John Wright. "Fast, Accurate Manifold Denoising by Tunneling Riemannian Optimization." Proceedings of the 42 nd International Conference on Machine
Learning (ICML), Vancouver, Canada. PMLR 267, 2025.

[2] Yao, Zhigang, Jiaji Su, Bingjie Li, and Shing-Tung Yau. "Manifold fitting." arXiv preprint arXiv:2304.07680 (2023).

### Questions
1. It makes sense that authors need a $C^2$-submanifold. Do you have any intuition on what this means in terms of limitations, both practical and theoretical?

2. How does the proposed approach compare to well-known baselines, such as VAEs and GANs for modeling manifolds? This would help position the work in the literature.

3. Could the proposed approach converge slowly (in terms of convergence rate) in certain scenarios?

4. Is the approach easily extendable to incorporating methods like heavyball (momentum) into the optimization process? If so, what are some potential challenges that you foresee?

5. Are there certain manifolds where the proposed framework could struggle?

### Soundness
4

### Presentation
4

### Contribution
4
