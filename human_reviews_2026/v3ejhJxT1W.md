# Splat the Net: Radiance Fields with Splattable Neural Primitives

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 4

## Abstract
Radiance fields have emerged as a predominant representation for modeling 3D scene appearance. Neural formulations such as Neural Radiance Fields provide high expressivity but require costly ray marching for rendering, whereas primitive-based methods such as 3D Gaussian Splatting offer real-time efficiency through splatting, yet at the expense of representational power. Inspired by advances in both these directions, we introduce splattable neural primitives, a new volumetric representation that reconciles the expressivity of neural models with the efficiency of primitive-based splatting. Each primitive encodes a bounded neural density field parameterized by a shallow neural network. Our formulation admits an exact analytical solution for line integrals, enabling efficient computation of perspectively accurate splatting kernels. As a result, our representation supports integration along view rays without the need for costly ray marching. The primitives flexibly adapt to scene geometry and, being larger than prior analytic primitives, reduce the number required per scene. On novel-view synthesis benchmarks, our approach matches the quality and speed of 3D Gaussian Splatting while using 10x fewer primitives and 6x fewer parameters. These advantages arise directly from the representation itself, without reliance on complex control or adaptation frameworks.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes Splat the Net, a unified representation that integrates the expressivity of neural radiance fields with the real-time rendering efficiency of splatting-based approaches. Each primitive in the proposed framework defines a bounded neural density field represented by a shallow sinusoidal network, for which the authors derive a closed-form solution to line integrals along view rays.
This analytic construction practically allows accurate, differentiable rendering without ray marching. The approach achieves state-of-the-art visual quality on multiple benchmarks, such as NeRF Synthetic, Mip-NeRF360, and Tanks & Temples, while requiring an order of magnitude fewer primitives and parameters than 3D Gaussian Splatting.

### Strengths
1. The proposed framework is novel and well-explored. By introducing neural primitives that can be analytically integrated and splatted, the authors effectively bridge two traditionally separate families of radiance field models.
2. The resulting design reconciles neural expressivity with analytic efficiency, achieving a favorable trade-off between compactness, rendering quality, and runtime.
3. The paper provides extensive experiments across synthetic and real benchmarks, complemented by ablations and visual comparisons that clearly support the claims.

### Weaknesses
1. The reliance on per-primitive neural parameters may constrain the method’s scalability and practicality, particularly in memory-limited environments.
2. While the experimental results are comprehensive, the paper could be further strengthened by a brief discussion of scenarios where the proposed neural primitives may be less effective, which would help clarify the scope and robustness of the approach.

### Questions
Given that training is more computationally demanding than purely analytic splatting, are there potential strategies—such as improved initialization, parameter sharing, or adaptive optimization—that could help accelerate convergence?

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
4

### Summary
The paper proposes splattable neural primitives: volumetric, ellipsoid-bounded primitives whose density is represented by a shallow neural network with periodic activation. A key technical claim is a closed-form antiderivative for line integrals through the neural density, which yields a perspectively accurate splatting kernel and thus avoids ray marching while retaining neural expressivity. Empirically, on synthetic and real novel-view synthesis benchmarks, the method targets the quality/speed of 3D Gaussian Splatting (3DGS) while using ~10× fewer primitives and ~6× fewer parameters, attributing the gains to the representation itself rather than to heavy control frameworks.

### Strengths
This paper introduces a neural primitive whose volumetric density is learned via a one-hidden-layer MLP, yet remains analytically integrable along rays. This bridges the perceived “neural vs. primitive” dichotomy and is, to my knowledge, a first in making the primitive itself neural while still splatting.
The analytical formulation is clearly spelled out (ray–ellipsoid intersection, anti-derivative, front-to-back alpha compositing), and the implementation details include population control (split/duplicate/prune via weight-gradient magnitudes) and geometric regularization to avoid degenerate ellipsoids.
Ablations probe network width/frequency and the effect of regularization; comparisons to an alternative neural integration strategy (AutoInt) clarify multi-view consistency benefits.

### Weaknesses
1. While the representation is new, final image quality/speed sometimes appears comparable to strong modern 3DGS variants that incorporate compression/regularization/adaptive control (e.g., BetaGS, T-3DGS, structured/linear kernels), some of which achieve very low memory or high FPS. A more direct, apples-to-apples model-size / memory / bandwidth comparison to these compression-oriented pipelines would clarify the net practical advantage. (Table 2 partially covers this, but a focused compression study would help.)
2. Adding per-primitive neural components (albeit shallow) complicates implementation relative to purely analytic Gaussians, which map naturally to existing graphics pipelines and hardware rasterization paths. This may limit industrial adoption unless the benefits (fewer primitives, comparable speed) translate into easier deployment (e.g., on mobile/embedded) than a tuned Gaussian pipeline. A short discussion of engine integration, batching, and runtime kernels would strengthen practical significance. (Authors do note PyTorch/CUDA implementation.)
3. On the synthetic dataset, initializing primitive positions from resampled ground-truth meshes risks leaking geometry and may overstate robustness; a stronger setting would evaluate multiple non-oracle inits (sparse SfM points, random, noisy depth). The paper acknowledges slow convergence vs. Gaussians and extends training, which heightens the importance of init robustness.
4. It may be worth to consider augmenting each primitive with a small learnable feature vector as additional input to the MLP to further boost expressivity under fixed primitive counts; this could bridge to richer local modulation without exploding primitive numbers.

### Questions
How robust is training to different initializations (e.g., sparse SfM points, noisy/sparse seeds, or purely random placements/shapes)? Can you report quantitative results (PSNR/SSIM/LPIPS, convergence rate) and qualitative failure modes across several inits on both synthetic and real scenes? (This would mitigate concerns about mesh-based seeding.)
The authors extend training to 100k iterations due to slower convergence. Can you share wall-clock training time comparisons and memory bandwidth/throughput metrics vs. 3DGS?

### Soundness
3

### Presentation
2

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
This paper introduces splattable neural primitives, a hybrid radiance field representation designed to unify the expressivity of neural radiance fields with the efficiency of splatting-based rendering. Each primitive is represented as an ellipsoid-bounded neural density field, parameterized by a shallow sinusoidal network that admits a closed-form integral along view rays. This analytical formulation eliminates the need for expensive ray marching while retaining multi-view consistency. Experiments on both synthetic and real-world datasets demonstrate that the proposed approach achieves comparable or superior performance to 3D Gaussian Splatting (3DGS), requiring ten times fewer primitives and fewer parameters overall.

### Strengths
1. The proposed formulation is conceptually sound and well motivated. Representing volumetric primitives as shallow neural fields bounded by ellipsoids provides an elegant way to connect neural and analytic splatting methods under a single theoretical framework.

2. The analytical derivation of the antiderivative for the density field is mathematically consistent and efficiently implemented, offering a clear path to rendering without ray marching.

3. Experimental results indicate that the proposed neural primitives maintain strong image quality under strict memory constraints, remaining both compact and efficient.

### Weaknesses
1. The overall training process is more complex, requiring more iterations and careful convergence control due to the optimization of numerous small neural modules.

2. While the ablation studies illustrate the role of model parameters, a more detailed analysis of trade-offs between expressivity, convergence, and stability would have strengthened the paper’s argument.

3. The impact of network width or frequency choices on quality and efficiency remains underexplored in the ablation section.

### Questions
1. How does performance vary with different configurations of network width and frequency? 

2. The scalability of the method for very large scenes is insufficiently discussed, how does runtime and memory usage scale with the number of primitives in such scenes?

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
2

### Summary
This paper introduces "splattable neural primitives," a novel radiance field representation that combines the expressivity of neural networks with the rendering efficiency of primitive-based splatting. Each primitive is an ellipsoid-bounded volume with a shallow neural network defining its density field, enabling exact analytical integration along view rays. The method achieves real-time rendering performance comparable to 3D Gaussian Splatting (3DGS) while using significantly fewer primitives (10×) and parameters (6×), demonstrating strong results on synthetic and real-world novel-view synthesis benchmarks.

### Strengths
1. Novel Representation: The proposal of fundamentally neural primitives with closed-form ray integration is a conceptually clean and innovative contribution. It successfully bridges the gap between expressive neural fields and efficient splatting-based rendering, a notable advance in the field.
2. Empirical Efficiency: The method demonstrates compelling practical benefits, matching 3DGS's quality and speed while drastically reducing primitive and parameter counts. This efficiency is directly attributed to the representation's inherent expressivity, not external control mechanisms.

### Weaknesses
1. Optimization Challenges: The paper acknowledges slower convergence and difficulties in optimizing millions of neural primitives due to a complex loss landscape. This suggests the method may be less robust or more sensitive to training configurations compared to established baselines like 3DGS.
2. Limited Ablation on Real Scenes: While toy examples (e.g., Snowflake, Leaf) effectively showcase expressivity, the ablation studies on network width and regularization lack depth for complex real-world scenes. The claimed expressivity advantage is not fully quantified or visually demonstrated on challenging benchmarks.

### Questions
1. Scalability & Robustness: Given the optimization difficulties mentioned, how does the method scale to extremely large, unbounded outdoor scenes? Are there specific types of scenes or geometries where the neural primitives consistently fail or underperform?
2. Integration Cost: The paper emphasizes "exact" and "efficient" integration. What is the precise computational overhead of evaluating the analytical anti-derivative compared to a single 3D Gaussian splatting kernel? A breakdown of rendering time (integration vs. blending) would clarify the practical trade-offs.

### Soundness
2

### Presentation
2

### Contribution
2
