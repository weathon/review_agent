# Fused-Planes: Why Train a Thousand Tri-Planes When You Can Share?

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 2, 6

## Abstract
Tri-Planar NeRFs enable the application of powerful 2D vision models for 3D tasks, by representing 3D objects using 2D planar structures.
This has made them the prevailing choice to model large collections of 3D objects.
However, training Tri-Planes to model such large collections is computationally intensive and remains largely inefficient.
This is because the current approaches independently train one Tri-Plane per object, hence overlooking structural similarities in large classes of objects. 
In response to this issue, we introduce Fused-Planes, a novel object representation that improves the resource efficiency of Tri-Planes when reconstructing object classes, all while retaining the same planar structure.
Our approach explicitly captures structural similarities across objects through a latent space and a set of globally shared base planes.
Each individual Fused-Planes is then represented as a decomposition over these base planes, augmented with object-specific features.
Fused-Planes showcase state-of-the-art efficiency among planar representations, demonstrating $7.2 \times$ faster training and $3.2 \times$ lower memory footprint than Tri-Planes while maintaining rendering quality.
An ultra-lightweight variant further cuts per-object memory usage by $1875 \times$ with minimal quality loss.
Our project page can be found at https://fused-planes.github.io .

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Fused-Planes, a tri-planar scene representation for efficiently training large collections of 3D object models. Specifically, each object’s planes are decomposed into an object-specific micro-plane and a class-level macro-plane. The training is conducted in a jointly learned 3D-aware latent space, supervised by both the latent and RGB domains. Experiments on ShapeNet and Basel Faces show that Fused-Planes achieves faster training speed and lower per-object memory than standard Tri-Planes, while maintaining comparable rendering quality.

### Strengths
1. This paper tackles a clear challenge in scaling object-centric Tri-Plane models and introduces both micro- and macro-decomposition to exploit inter-object redundancy.

2. The proposed method uses a base-plane bank and a learned weighting mechanism to provide a compact and interpretable way to model class-level structures without sacrificing the planar format useful for 2D backbones.

3. The performance is impressive. The proposed method significantly reduces the training consumption, including time and space overhead, while still maintaining comparable reconstruction performance.

### Weaknesses
1. The memory of each object excludes the shared network components, including encoder, decoder, and base planes, which may lead to an unfair comparison with baselines. It would be better if the authors could provide a more balanced total-cost comparison across different values of N.

2. The method is class-specific, which means it cannot be generalized to unknown classes.

3. The evaluation is on object-centric, bounded scenes, e.g., ShapeNet categories and Basel Faces. The proposed method struggles with fine details and unbounded scenes, limiting its applicability relative to non-planar methods.

### Questions
See weaknesses

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
This paper presents a resource-efficient tri-planar representation (namely, Fused-Planes) for modelling large collections of 3D objects. The core idea is to decompose each object's representation into object-specific 'micro' planes and shared 'macro' planes (constructed from learned base planes), trained in a 3D-aware latent space. The work addresses a real problem, i.e., the computational cost of training thousands of Tri-Planes, and demonstrates impressive efficiency gains while maintaining quality (7.2× faster training, 3.2× lower memory). The combination of micro-macro decomposition with latent space training is well-motivated, and ablations demonstrate both components are necessary. Experiments and ablations are extensively conducted across multiple baselines and datasets.

### Strengths
The major strength of this paper is the significant resource savings (7.2× speed, 3.2× memory reduction vs Tri-Planes). Besides, the ultra-lightweight variant achieves 1875× memory reduction. And it maintains the rendering quality.

### Weaknesses
The method only works within a single object class. For multiple classes with large visual variations, you need multiple instances of Fused-Planes. This significantly limits practical applicability. For diverse datasets, the overhead of multiple base plane sets could negate efficiency gains.

For open surfaces and unbounded scenes, this method is still limited like other triplane methods.

Table 4, Fused-Planes (Micro)(latent space without macro planes) performs worse than Tri-Planes. This suggests the latent space itself introduces quality degradation, which is only compensated by the macro planes. This is concerning because it means the latent space is not providing a better representation per se. It's just enabling the sharing mechanism.

During inference, each Fused-Planes requires computing the weighted sum at inference (Eq. 2). What's the computational cost of this operation compared to directly loading a Tri-Plane? For applications requiring real-time rendering, is this overhead acceptable? What are the runtime/FPS at inference time across different methods?

### Questions
What happens when you train on the entire ShapeNet dataset, not just per-category? How many base plane sets would be needed? What's the break-even point where efficiency gains disappear?

At what scale (N objects) does M=50 need to increase? Is there a theoretical or empirical relationship between M, N, and object class diversity?

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
This paper introduces Fused-Planes, an efficient tri-plane–based method for reconstructing large classes of 3D objects. The approach incorporates a Micro component to capture object-specific features and a Macro component to encapsulate structural similarities shared across the object class. In addition, it leverages a 3D-aware latent space to accelerate both the rendering and training processes of Fused-Planes. Compared with the original Tri-Planes, Fused-Planes achieves 7.2× faster training and 3.2× lower memory consumption while maintaining comparable rendering quality. The authors further propose an ultra-lightweight variant that almost entirely omits the micro component, yielding a remarkable 1875× memory reduction. Experimental results demonstrate that Fused-Planes significantly outperforms existing plane-based methods such as Tri-Planes and K-Planes in terms of both training efficiency and memory scalability for large-scale multi-object reconstruction.

### Strengths
1. The paper introduces a 3D aware latent space as a form of shared representation in the object reconstruction domain, enabling the model to better capture structural similarities across object classes. According to the ablation study, employing this latent representation rather than directly optimizing in RGB space leads to faster convergence while maintaining comparable rendering quality.
2. Compared to C3 NeRF, which scales only up to around 20 scenes, the proposed approach remains scalable to thousands of objects, demonstrating superior generalization and efficiency across large datasets.
3. Under the current task setting, the method achieves significant advantages in training speed and memory efficiency over traditional KPlanes and TriPlanes approaches.

### Weaknesses
1. Although the paper claims that Fused-Planes remains scalable to thousands of objects, I do not observe convincing evidence of this property from the presented experimental results or supplementary videos.
2. In line 103, the paper states that TensoRF, 3DGS, and Instant-NGP cannot be reshaped into image-like tensors. However, to my knowledge, several recent works in the 3DGS domain, such as Animatable Gaussians, ASH, GaussianAvatar, and Reperformer, have successfully employed 2D UV unwrapping or Morton mapping strategies to project 3D point clouds into 2D grids and then apply image-based CNN architectures to learn the appearance of avatars under novel motions. I believe these 2D parameterization methods should be properly cited and discussed for completeness.
3. While I understand the limitations of Tri-Planes in representing fine details and handling unbounded scenes, the object-centric datasets used in this paper (e.g., ShapeNet, Basel Faces) appear to be extremely simple in geometry. Despite such simplicity, both Fused-Planes and Fused-Planes-ULW still produce noticeably blurry renderings. This level of visual quality makes it difficult to assess the practical value and applicability of the proposed approach.
4. To my knowledge, several tri-plane-based methods such as TeTriRF can generate highly detailed and realistic renderings of complex human data with relatively lightweight models. It is therefore unclear why the proposed method fails to achieve comparable quality even on synthetic datasets with simple geometries.
5. Furthermore, in Fig. 5, where Fused-Planes is compared with other per-scene training methods in terms of size and training time, I believe this comparison is highly unfair

### Questions
As noted in the weaknesses section, the paper should better highlight its capability to scale up to thousands of objects. It should also provide a discussion of relevant references regarding the use of 2D parameterization in 3DGS, and clarify the factors contributing to its lower rendering quality.

### Soundness
3

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
This paper proposes Fused-Planes, a shared tri-planar representation that improves the efficiency of training large 3D object collections. Instead of training independent Tri-Planes for each object, the method decomposes each object’s planes into a micro component (object-specific details) and a macro component (a weighted sum of shared base planes capturing class-level structure). The model is trained jointly within a 3D-aware latent space, which further reduces computation. Experiments on ShapeNet and Basel Faces show strong results: up to 7.2× faster training, 3.2× smaller per-object memory, and comparable or better rendering quality than Tri-Planes. An ultra-lightweight variant achieves extreme compression with minor quality loss.

### Strengths
* **Clear and practical contribution.** The paper tackles a real inefficiency in Tri-Plane training and offers an intuitive solution that effectively shares structure within a class.
* **Strong empirical gains.** Training speed, memory footprint, and quality all improve substantially. The ULW version demonstrates impressive compression with minimal degradation.
* **Elegant design.** The micro–macro split is simple yet powerful, allowing reuse of planar architectures while amortizing cost across objects.
* **Comprehensive evaluation.** The experiments, ablations, and comparisons are thorough and well-presented. The results convincingly support the claims.
* **Readable and well-organized.** The paper is clear, figures are informative, and the setup is easy to follow.

### Weaknesses
* **Single-class limitation.** The method assumes one class per model. Scaling to diverse datasets would require separate sets of base planes, reducing its flexibility.
* **Limited analysis of shared bases.** The learned base planes are not explored in depth. It is unclear what structures they capture or how weights vary across instances.
* **Restricted evaluation scope.** Experiments focus only on novel view synthesis of synthetic datasets. No tests on real or multi-class data, or on downstream tasks that might use the planar outputs.
* **Dependence on latent space.** The model relies on a pretrained VAE initialization, but the sensitivity to this setup and its cost are not fully studied.
* **Constant overhead.** While per-object cost drops sharply, the shared components introduce a large fixed memory load, which only becomes efficient at larger scales.

### Questions
1. How sensitive is performance to the number of shared base planes and the micro–macro feature split?
2. Can a single set of base planes handle multiple classes, perhaps with class-conditioned weights?
3. What do the shared base planes actually learn? Visualizing or analyzing them could offer useful insight.
4. How dependent is training on the pretrained latent codec? Could the system work with a smaller or scratch-trained encoder?
5. Could Fused-Planes be tested on real multi-view datasets or downstream 2D-compatible tasks to demonstrate broader utility?

### Soundness
3

### Presentation
3

### Contribution
3
