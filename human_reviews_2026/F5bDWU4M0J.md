# SpRePE: A Spherical Geometry-Aware Position Embedding for Vision Transformers

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Position embedding (PE) is a key mechanism that breaks the permutation symmetry of tokens in Transformer, introducing a spatial inductive bias that enables attention to model locality, distances, and directional relations. 
Spherical data arise in many scientific domains, most notably in astronomy and meteorology, where Vision Transformers is increasingly adopted for the ability to capture long-range dependencies. However, conventional PEs are designed for linear sequences and cannot faithfully capture the sphere’s non-Euclidean geometry. 
Furthermore, existing designs for encoding spherical positional information rely on additional network modules or specialized network architectures, which introduce extra parameters and computational overhead.
These limitations motivate a geometry-aware and efficient embedding scheme that fully exploits spherical structure to advance Transformer-based modeling on the sphere.
We introduce \textbf{Spherical Reflection Position Embedding (SpRePE)}, a lightweight method efficiently leveraging spherical positional information for Vision Transformer.
SpRePE encodes the absolute position on the sphere using a Householder matrix and incorporates the explicit relative position dependency into the attention formulation, achieving both high computational efficiency and high accuracy without requiring substantial additional parameters and modifications to the overall model architecture.
We evaluate SpRePE on representative tasks, including spherical image classification and global weather forecasting. SpRePE consistently outperforms well-known baselines including APE, RPE, ALiBi and RoPE.
These results indicate that SpRePE offers an efficient and broadly applicable position embedding scheme for Transformer models on the sphere.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
SpRePE introduces a spherical position embedding for Vision Transformers using Householder reflection matrices to encode positions on a sphere, applied directly to query/key vectors with RoPE-like computational efficiency. The method shows improvements over older standard position embeddings (APE, RPE, ALiBi, RoPE) on Spherical MNIST classification and ERA5 weather forecasting tasks.

### Strengths
Principled geometric foundation, using Householder reflections to respect spherical topology while maintaining the computational efficiency and drop-in simplicity of methods like RoPE.

### Weaknesses
- Experiments use relatively small-scale tasks (Spherical MNIST, downsampled ERA5 at 128×256 resolution) and lack comparison to specialized spherical architectures or more recent baselines designed for spherical data.
- The improvements over simpler baselines like RoPE are often marginal (e.g., 0.88pp on MNIST), making it unclear whether the added geometric complexity translates to meaningful real-world benefits that justify adoption.
- The paper lacks ablation studies on key design choices (auxiliary point selection, masking strategies), theoretical analysis of when spherical geometry matters most, and evaluation on diverse spherical tasks beyond weather forecasting and toy image classification.

### Questions
- When does spherical geometry actually matter?
- How should auxiliary points be selected, and how much do results depend on masking strategies, the number of 3D subspaces, or other hyperparameters that lack principled guidance?
- How does SpRePE perform at production-scale resolutions, on other spherical domains (astronomy, geology), with different architectures (Swin, hierarchical transformers), and does the computational overhead remain negligible at scale?

### Soundness
3

### Presentation
2

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
The paper introduces a positional encoding method for data defined on a sphere. The core concept builds on the RoPe scheme, where positions are encoded using complex exponentials, which are then multiplied (rather than added) to the vectors. This approach is further adapted for datapoints on a 3D-sphere. From what I gathered, the key innovation lies in using Householder reflection matrices (relative to a predefined grid of points, denoted as $n$) to encode the positional data.

### Strengths
1. The manuscript is well-written, presenting a novel positional encoding scheme by adapting RoPe for spherical data.

2. The approach is evaluated through benchmarking against several existing methods from the literature.

### Weaknesses
1. The manuscript does not provide a more formal statement for the proof in Appendix C. Although Section 3.1 discusses the result, its current form makes it unclear what the formal statement is and what its implications are (see also Question 3).

2. It would be helpful to include one or two additional experimental settings. For instance, the Spherical CNN approach by Cohet et al. (2018) offers experimental settings that could be considered for comparison.

### Questions
1. Can you explain how you are choosing the grid of $n$. There is a discussion in section 3.1 about this saying "... for every pair of $(p_1, p_2)$, any two auxiliary points $n_1$ and $n_2$ must define distinct sections.", but how is this condition enforced both for uniform and non-uniform grid?

2. It is not very clear to me the details of "Density-adaptive Mask". Can you elaborate on that?

3. In the derivation provided in Appendix C, what is is the main result and the consequence?

(I have kept my score on the lower end and will change it based on the answers of the above questions.)

4. The RoPe scheme uses complex exponetials, which can be thought of as spherical harmonics in 2D. Have you thought of extending this idea and designing something using spherical harmonics in 3D?

Typo in line 255: "Noticed" -> "Notice".

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Spherical Reflection Position Embedding (SpRePE), a geometry-aware and lightweight position embedding method for Transformers operating on spherical data. The approach leverages the Householder matrix to encode absolute positions on the sphere and incorporates explicit relative position dependencies directly into the attention formulation. Unlike previous spherical embedding methods that rely on additional modules or complex architectures, SpRePE maintains high computational efficiency with minimal parameter overhead. Experiments are conducted on spherical image classification and global weather forecasting tasks, showing improvements over existing baselines such as APE, RPE, ALiBi, and RoPE.

### Strengths
The paper targets an important and under-explored area — transformer modeling on spherical domains, which has strong relevance to scientific applications like meteorology and astronomy.

The formulation is elegant, providing a theoretically grounded yet efficient way to encode positional information on non-Euclidean manifolds.

The method is lightweight, requiring minimal architectural changes and extra parameters.

The paper is well written, and the idea is clearly explained with sound mathematical motivation.

### Weaknesses
The performance gain is quite small compared to APE — for example, in Table 3, accuracy only improves from 96.29 to 96.74, and in Table 4, from 0.9954 to 0.9965. Such marginal improvements raise concerns about the practical significance of the proposed method.

The evaluation is limited to a few specific datasets (spherical image and weather data). To better demonstrate generalization and robustness, it would be valuable to test on more widely used image or language datasets (e.g., ImageNet, COCO, or text benchmarks).

It is unclear whether SpRePE provides benefits beyond specialized spherical data — this limits its broader impact and may restrict its applicability.

### Questions
see weakness

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes SpRePE (Spherical Reflection Position Embedding), a geometry-aware and efficient positional encoding for ViTs on spherical data. It uses Householder reflections to encode absolute positions on the sphere and lets relative information emerge via attention inner products. No extra trainable parameters, and it’s drop-in to standard Transformers. Validated on Spherical MNIST and ERA5.

### Strengths
1.	Novel geometric formulation: Reflection-based, sphere-aware encoding; handles poles & longitudinal wrap-around better than planar PE.
2.	Minimal overhead: No architecture change; same complexity class as RoPE; avoids quadratic RPE.
3.	Empirical gains: Better accuracy/robustness (especially at high latitudes & long horizons).
4.	Comprehensive eval: Comparisons + ablations (masking, robustness).
5.	Clarity & reproducibility: Derivations are clean; code & settings well documented.

### Weaknesses
1.	Dataset breadth: Only Spherical MNIST & ERA5; lacks panoramic CV / point-cloud / remote-sensing detection tasks.
2.	Qualitative insight: Could add attention maps / distance heatmaps to visualize geometric effects.
3.	Ablation depth: Need to isolate the contribution of the geometric term vs. reflection itself; clarify sensitivity to auxiliary points {n_i}.
4.	Theory rigor: Proof that reflections yield correct relative encoding could be strengthened.
5.	Baselines: Missing comparisons with newer geometry-aware methods (e.g., Sphere2Vec, Heal-Swin) on same backbones.

### Questions
1.	Auxiliary points: How are {n_i} chosen (fixed grid / learned / random)? Sensitivity & stability?
2.	Masking: Tried learned or entropy-gated masks instead of cosine-latitude heuristics?
3.	Generality: Extendable to other manifolds (hyperbolic / cylindrical)? What changes?
4.	Scale: Results on larger backbones / higher-res ERP (e.g., ViT-L/16 @ 1024×2048)? Memory trade-offs?
5.	Interpretability: Any visualization of reflection effects in latent space or attention geometry?

### Soundness
3

### Presentation
3

### Contribution
3
