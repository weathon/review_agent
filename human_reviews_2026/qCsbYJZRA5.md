# The Spacetime of Diffusion Models: An Information Geometry Perspective

- Avg Score: 6.50
- Decision: Accept (Oral)
- Scores: 6, 4, 8, 8

## Abstract
We present a novel geometric perspective on the latent space of diffusion models. We first show that the standard pullback approach, utilizing the deterministic probability flow ODE decoder, is fundamentally flawed. It provably forces geodesics to decode as straight segments in data space, effectively ignoring any intrinsic data geometry beyond the ambient Euclidean space. Complementing this view, diffusion also admits a stochastic decoder via the reverse SDE, which enables an information geometric treatment with the Fisher-Rao metric. However, a choice of $\mathbf{x}_T$ as the latent representation collapses this metric due to memorylessness. We address this by introducing a latent spacetime $\mathbf{z}=(\mathbf{x}_t,t)$ that indexes the family of denoising distributions $p(\mathbf{x}_0 | \mathbf{x}_t)$ across all noise scales, yielding a nontrivial geometric structure. We prove these distributions form an exponential family and derive simulation-free estimators for curve lengths, enabling efficient geodesic computation. The resulting structure induces a principled Diffusion Edit Distance, where geodesics trace minimal sequences of noise and denoise edits between data. We also demonstrate benefits for transition path sampling in molecular systems, including constrained variants such as low-variance transitions and region avoidance. Code is available at: https://github.com/rafalkarczewski/spacetime-geometry.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents an information-geometric perspective on the latent space of diffusion models. The authors first show that the conventional pullback metric induced by the Probability Flow ODE (PF-ODE) is degenerate: geodesics under this metric always decode to straight lines in data space, thus failing to capture any intrinsic geometry. To address this limitation, the authors propose a new Fisher–Rao metric defined on the denoising distributions $p(x_0 | x_t)$. They further introduce a latent spacetime representation $z = (x_t, t)$, where both the sample and its diffusion timestep jointly parameterize the geometry. They prove that these denoising distributions form an exponential family, which enables a tractable estimator for geodesic energy and distance. Based on this framework, they define the Diffusion Edit Distance (DiffED) and demonstrate applications in image interpolation and transition path sampling for molecular systems, showing competitive performance with specialized baselines.

### Strengths
- The paper provides a conceptually novel and rigorous framework.
- This paper shows the inherent limitations of the pullback metric defined by the PF-ODE sampler from $x\_{T}$ to $x\_{0}$ and motivates the introduction of the information geometry using the denoising distribution $p(x\_0 | x\_{T})$.
- The transition-path sampling experiments on the Alanine Dipeptide system demonstrate that the framework has potential beyond visualization or interpolation tasks.

### Weaknesses
- The claim that “memorylessness” causes $p(x\_0 | x\_T) \approx q(x\_0​)$ deserves clearer probabilistic justification. While the forward process is memoryless, it seems like this does not directly imply conditional independence in the reverse direction. Isn't it $p(x_0 | x_{T}) \propto p(x_{T}) q(x_0)$?
- Table 1 reports the number of energy evaluations but omits the actual trajectory computation time or runtime comparison with baselines.

### Questions
- Could the proposed Diffusion Edit Distance be leveraged for practical downstream applications, such as by combining with SDEdit?
- How would the transition path differ if one were to fix a specific diffusion timestep $t$ (as in Line 169) instead of employing the current spacetime modeling? What are the conceptual or empirical benefits of jointly modeling both space and time in the proposed spacetime representation? The current discussion in Lines 175–181 mainly explains how the framework enables spacetime geodesic modeling, but it does not clearly state why this joint modeling is beneficial or what additional insights it provides.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a new geometric framework for diffusion models based on information geometry, introducing a spacetime manifold that combines both the noisy sample and the diffusion time (x_t, t). The authors show that the conditional denoising distributions, $p(x_0|x_t)$, form an exponential family, allowing them to endow the diffusion process with a Fisher–Rao metric (a principled way to measure distances between nearby denoising distributions)

Using this metric, they define spacetime geodesics (shortest paths) between data points and introduce the Diffusion Edit Distance (DiffED), a new notion of distance induced by the geometry of the diffusion process. They compute these geodesics by discretizing the trajectory in spacetime and minimizing a Fisher–Rao energy functional using gradients derived from the trained denoiser.

Empirically, the paper illustrates the framework on image diffusion models and molecular simulation tasks, showing that the resulting geodesics correspond to meaningful transition paths and that the proposed distance can characterize structure in data manifolds.

### Strengths
- The paper addresses an interesting theoretical question: how to define a meaningful geometric structure for diffusion models and connects it to information geometry.

- The idea of modeling denoising distributions as an exponential family and equipping the resulting spacetime with a Fisher–Rao metric is conceptually sound and mathematically motivated.

- The second application (transition path sampling in molecular systems) is original and can lead to generating more research in that direction (with the caveat that I'm not very familiar with this particular area of applications of diffusion models)

### Weaknesses
- The paper does not clearly describe the algorithm used to compute the Diffusion Edit Distance (DiffED). Including pseudocode or a concise algorithmic summary would greatly improve clarity and reproducibility.

- The computational cost of finding geodesics in spacetime using DiffED is unclear. Since Equation (16) must be evaluated at many noise levels, the method appears potentially expensive. How does its efficiency compare to related approaches? For instance, What’s Inside Your Diffusion Model? For example in [1]  the authors propose a similar framework for computing geodesics with diffusion models. Some discussion or comparison would help. This paper also contains some comparisons with other methods which the authors can use. 

*A Score-Based Riemannian Metric to Explore the Data Manifold, Azeglio & Di Bernardo, arXiv:2505.11128

- The paper notes that the proposed distance correlates with SSIM but not with LPIPS. Could the authors elaborate on why this occurs, given that LPIPS is generally regarded as a stronger measure of perceptual or semantic similarity than SSIM?

Minor comments: 

- Some terminology seems unnecessarily reinvented. For example, using “denoising” or “decoding distribution” instead of the standard term posterior may reduce clarity for readers familiar with Bayesian or diffusion-model terminology.

- Correlations are reported as percentages in line 281. should these not be expressed as standard correlation coefficients?

- The caption of Figure 1 describes the geodesic in spacetime as the “shortest path between two distributions.” Shouldn’t it instead be the shortest path between two images (or corresponding distributions over clean images)?

### Questions
- What is the motivation for comparing to the ODE baseline? It is not clear what specific insight this comparison is intended to convey.

- The pullback geodesic experiment seems somewhat like a strawman comparison—does it add significant insight? It may not warrant an entire page of the paper.

- Is the time discretization used to compute the geodesic linear, or is it adapted in some way along the path?

- How do the proposed results compare to stochastic interpolants? A direct comparison or discussion would be helpful for positioning this work relative to recent diffusion-based interpolation methods.

- What theoretical or empirical guarantees can you provide that the Diffusion Edit Distance (DiffED) indeed corresponds to a true geodesic under your defined Fisher–Rao metric?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The authors extend a geometric perspective on diffusion models by defining the Fisher–Rao metric not only over the latent space but across the latent spacetime $(z = (x_t, t))$. This structure allows them to define the length, or energy, of curves within the latent spacetime, and to define geodesics, or shortest path, over latents and timesteps. The authors show that geodesics from noise $(t = T)$ to denoised states $(t = 0)$ closely follow PF-ODE sampling trajectories. This also leads to a new measure of distance, the Diffusion Edit Distance, expanding upon known interpolation distances such as LPIPS.

### Strengths
The paper is clearly written, and the proposed approach is well presented.

The work introduces a novel and interesting notion of latent spacetime. While previous studies usually considered interpolation between samples as transitions within a slice of latent space fixed in timestep, this paper generalizes that perspective. It may have a broad impact on the image editing domain, effectively connecting geometric ideas with techniques such as DDIM inversion.

### Weaknesses
There are concerns regarding the computational complexity. Although the paper mentions that the method may be slower, it would be beneficial to include exact runtime comparisons and additional evaluations.


The authors demonstrate that PF-ODE sampling trajectories and geodesics are similar but do not provide any quantitative metrics. It would be interesting to see how this approach compares with LPIPS, for example by fixing the starting and final points to the same timestep.

### Questions
The paper could benefit from discussion on image editing approaches such as DDIM and Null-text inversion. 

The authors mention that their method is numerically slower and suggest potential distillation. What are their preliminary thoughts on the architecture and training objective for a separate model intended to predict DiffED? Specifically, should the distillation target focus on replicating the absolute geodesic length, or rather on the manifold’s tangent structure defined by the Fisher–Rao metric, to ensure geometrically meaningful similarity estimation?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper revisits the geometric structure underlying diffusion models and argues that meaningful geometry does not reside in the noisy latent space x_t alone, but rather in the full spacetime domain formed by pairs (x_t, t) combining state and noise level. The authors show that when considering only the latent variable x_t at a fixed noise level, the denoising distribution p(x_0 \mid x_t) becomes increasingly isotropic as t increases, causing the geometry (e.g., tangent and normal decomposition, pullback metrics) to collapse. To recover a non-trivial Riemannian structure, the paper proposes defining the metric using the Fisher–Rao information geometry over spacetime, resulting in the GIG metric that varies jointly with state and time.

Using this metric, the authors define the Diffusion Edit Distance (DiffED); the length of the shortest geodesic path in spacetime connecting two clean samples (x,0) and (y,0). This yields a principled notion of semantic distance that measures “how naturally one data point can be transformed into another” under the diffusion generative process. Moreover, the paper introduces a transition-path sampling method based on Annealed Langevin Dynamics (ALD) that follows the computed spacetime geodesic to generate actual interpolation trajectories. Empirical results in Section 6 demonstrate that these spacetime geodesics produce smooth, consistent, and semantically meaningful transformations between examples.

### Strengths
1. Clear and compelling reframing of the latent representation in diffusion models.
While prior work typically analyzes individual x_t states or only the fully-noised x_T, this paper emphasizes that the entire trajectory \{x_t\} constitutes the latent representation. Defining geometry on the spacetime manifold (x_t, t) using the Fisher–Rao metric is both conceptually straightforward and surprisingly underexplored, making the contribution feel natural yet novel.
2. Diffusion Edit Distance offers meaningful semantic comparisons.
The proposed DiffED provides a principled notion of distance based on how naturally one data point can transform into another through the diffusion process. Although currently computationally expensive, the definition itself seems robust and opens opportunities for future research in efficient approximations and downstream applications such as morphing, retrieval, and generative editing.
3. Transition-path sampling via Annealed Langevin Dynamics adds practical value.
The use of ALD to follow the spacetime geodesic demonstrates that the metric is not only theoretically motivated but also operationally useful. The method enables the generation of smooth and interpretable interpolation paths, giving the framework concrete applicability rather than remaining purely abstract.

### Weaknesses
1. High computational cost.
The method requires computing geodesics in spacetime and then sampling along those paths with ALD. This is quite expensive in practice, and the paper does not propose any way to reduce this cost. As a result, it may be difficult to use the method in large-scale or time-sensitive settings.
2. Limited variety of data types in experiments.
The experiments are mostly on datasets where the structure is relatively simple and smooth (e.g., faces, digits). It is unclear how well this approach works on more complex images (e.g., multiple objects, cluttered scenes) or on stylized data (e.g., cartoons, anime). Testing or discussing these cases would make the evaluation stronger.
3. Unclear stability of geodesic optimization.
The final interpolation path comes from optimizing an energy function, which may have multiple local minima. The paper does not analyze how sensitive the result is to initialization or parameter choices. Because of this, it is not yet clear how stable or reproducible the method is.

### Questions
1. Comparing image pairs with different visual attributes:
For two cases (i) two images with similar color tone but completely different content, and (ii) two images with identical structure/shape but very different color, which pair does the Diffusion Edit Distance assign a larger distance to? In other words, does the metric primarily reflect semantic structure, appearance, or a combination of both?

1-1. Potential analytical uses of DiffED.
Since Diffusion Edit Distance provides a meaningful notion of distance in the spacetime geometry, it seems possible to use it for analyzing model behavior (e.g., structure of learned manifolds, semantic neighborhood relationships, mode connectivity). Have the authors explored such analytical applications, or do they have ideas for experiments where DiffED could be used as a diagnostic tool?

2. Extension to video data:
Do the authors believe that the proposed spacetime geodesic framework can be extended to video sequences? If so, what would be a reasonable way to handle the temporal consistency constraint, for example, by treating time as an additional dimension alongside the diffusion timestep, or by defining geodesics directly in a trajectory space?

3. Effect of different latent representations.
The behavior of DiffED may depend on the type of latent space in which the diffusion model operates (e.g., VAE latent space, pixel space, or non-image modalities). Have the authors observed different geometric patterns or performance characteristics when applying the method to models operating in these different latent representations?

### Soundness
4

### Presentation
3

### Contribution
3
