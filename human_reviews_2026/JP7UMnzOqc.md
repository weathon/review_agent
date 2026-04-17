# FRACTAL3DGEN: COARSE-TO-FINE 3D GENERATION VIA FRACTAL HIERARCHICAL TRANSFORMERS

- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
Autoregressive generation models have demonstrated strong performance across a wide range of applications. In the context of 3D shape generation, several approaches have attempted to adopt this paradigm by flattening the 3D representation into a serialized one-dimensional sequence and predicting it sequentially, token by token. These methods are inherently inefficient due to their strictly sequential generation process and are often inadequate in exploiting hierarchical self-similar representations during generation. Inspired by the success of FractalGen in image generation, we propose Fractal3DGen, a hierarchical fractal-autoregressive framework that leverages self-similarity across multiple scales for 3D shape generation.  Specifically, the 3D shapes, represented by sparse SDF grids, are encoded into a compact, low-resolution sparse voxel latent space for more efficiency. Our generation process then employs a divide-and-conquer strategy to efficiently capture the intrinsic hierarchical structure of 3D shapes. Furthermore, due to the spatial sparsity of 3D shapes, we introduce an adaptive pruning mechanism during generation to promptly halt the processing of empty regions at higher scales, significantly improving both training and inference efficiency while reducing memory consumption. Extensive experiments on ShapeNet demonstrate the effectiveness of Fractal3DGen. Fractal3DGen achieves the lowest average FID against all baselines, with over 15% improvement across five categories. Moreover, it provides a 1.85× speedup,  reducing the average inference time per shape from 54.1s to 29.3s compared with OctGPT. Beyond benchmark evaluations with class conditioning, we further explore text- and image-conditioned 3D generation, demonstrating the generalizability of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose Fractal3DGen, a hierarchical fractalautoregressive framework that leverages self-similarity across multiple scales for
3D shape generation. Besides, they introduce an adaptive pruning mechanism to improve the efficiency. The experimental results on ShapeNet demonstrate the effectiveness of the proposed approach.

### Strengths
1. A new auto-regressive 3D shape generation model with different generative order from OctGPT,  improving both the quality and efficiency of autoregressive models.
2. A 3D fractal pruning strategy for efficient inference. 
3. It achieves significant improvements in generation speed, quantitative metrics, and computational efficiency.

### Weaknesses
1. The comparison is only conducted on ShapeNet, while OctGPT is also evaluated on Objaverse and Synthetic Rooms dataset. I think ShapeNet is too simple and may not fully reveal the performance of the proposed approach.  
2. I think "BIOLOGICAL INSPIRATION OF FRACTAL3DGEN" of Appendix A is over-claimed. It is a new design of generative order built upon OctGPT, and the "biological inspiration" is too vague and not really reflect the contribution.

### Questions
N.A.

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
The paper presents a 3D generation method inspired by fractal formulation. The sparse 3D volume hierarchically upsamples to finer resolution, with an additional pruning framework that maintains the underlying sparsity and efficiency. They design the order of traversal for training and inference, and propose a normalized Hausdorff dimension to apply loss in different resolutions. The results demonstrate superior performance against other baselines with autoregressive generation methods. In addition to the quality of the results, the hierarchical generation is fast and efficient. Additional results include ablation on the number of layers and various application scenarios.

### Strengths
The paper builds on the fractal generation method in 2D and applies a similar technique in 3D. To transfer to the 3D domain, the representation incorporates a sparse voxel representation, which is also widely used in 3D volumetric representation. But the pruning mechanism and the loss formulation with binary masks are tailored to the representation. Additionally, the proposed normalized Hausdorff dimension provides necessary scaling for different resolutions, considering the subdivision, which nicely integrates the proposed method.

### Weaknesses
- The effect of Hausdorff dimension needs further evaluation to be perceived as a valid contribution.

- It seems like a two-layer architecture is used for most of the results (stated in Section 4.1), which might not be challenging to consider as "fractal". Fractals often imply a greater number of recursive formulas, or almost infinite layers of progression. 

- The results are evaluated in the ShapeNet dataset, which is relatively old and small in size. Also, the amount of improvement is not very significant, and even underperforms many existing works, depending on the shape categories.

### Questions
- Figure 3 is hard to understand, which should be critical to improve the clarity of the exposition.

- Why do you propose a different traversal for training (breadth-first) and inference (depth-first)?

- What are "example tokens" in Figure 4?

- Is layer and level the same thing in Section 4.3? Why not use five-level architecture if they are more accurate and faster?

### Soundness
3

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
5

### Summary
This paper introduces a coarse-to-fine fractal-autoregressive framework for 3D shape generation. Shapes are encoded as sparse SDF voxels into a latent space via VQ-VAE; a self-similar Transformer fractal generator is reused across levels to refine voxel blocks from global structure to local details. Inference uses an adaptive pruning mask to stop refining empty regions. On ShapeNet, the method achieves lower FID than prior baselines and ~1.85× faster inference than OctGPT, while preserving fine geometry. The method further supports text- and image-conditioned generation.

### Strengths
1.The paper has a conceptually elegant framework, which proposes a dynamic, recursive generation process. This "divide-and-conquer" strategy is a principled approach to managing the high dimensionality of 3D data.

2.This work achieves state-of-the-art performance on the ShapeNet benchmark,with an impressive speedup in average inference time by pruning.

3.The paper defines Normalized Hausdorff Dimension with detailed derivation to clearly evaluate the computational efficiency.

### Weaknesses
1. Ambiguity in novelty.The hierarchical structure and autoregressive fractal generators are similar in “Fractal Generative Models”,while the VQ-VAE encoder is used in OctGPT.

2.Experimental results in Table 1 only use FID as metric, some others can be introduced.Also,some baseline models like MeshGPT are not trained on all the categories.The superiority of Fractal3DGen will be more convincing if all categories are evaluated.

3.The model details in practical applications like text- and image-to-3D generation using CLIP or DINOv2 encoders are not clear.

Shapenet is a pretty easy dataset, which can be easily overfitting. Shall it be reasonable (and make sense) to work on large scale and more diverse dataset, say, objverse?

### Questions
1.Can the structure of Fractal3DGen be directly extended to point cloud or mesh generation, which preserve explicit surfaces and topology and provide stronger cues for semantic reasoning?

2.Does Fractal3DGen have the ability to generate larger and more complicated 3D contents, e.g. a table with multiple objects on it?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Inspired by the success of FractalGen in image generation, this paper proposes Fractal3DGen, a hierarchical fractal-autoregressive framework that leverages self-similarity across multiple scales for 3D shape generation. Furthermore, due to the spatial sparsity of 3D shapes, it introduces an adaptive pruning mechanism during generation to promptly halt the processing of empty regions at higher scales, significantly improving both training and inference efficiency while reducing memory consumption. Experiments on ShapeNet demonstrate the effectiveness of Fractal3DGen.

### Strengths
+ The proposed hierarchical fractal-autoregressive framework for 3D shape generation is an interesting and novel idea.
+ Experiments are conducted on ShapeNet dataset to demonstrate the effectiveness of the proposed method.

### Weaknesses
- **Technical Contribution:** Applying the fractal generative pipeline to 3D shape generation is an interesting and novel idea. However, this single contribution is not enough. The proposed pruning strategy is relatively simple, and the experimental verification is insufficient (see the following comments for details). It is required to discuss the difficulties and challenges in 3D fractal generation and propose specific solutions to address them.
- **Technical Detail:** Several implementation details of the architecture are missing or ambiguous. (1) What is the difference between the two-layer and five-layer Fractal3DGen model? (2) What is the optimal/final number of hierarchical levels used in this paper? (3) It would be better to add a Preliminary Section of FractalGen.
- **Experimental Analysis:** The experimental analysis is not thorough enough. First of all, in Table 1, compared with OctGPT and OctFusion, the proposed method still has inferior accuracy in some categories. Meanwhile, only the FID indicator is not comprehensive enough; other popular metrics for ShapNet, like 1-NNA, MMD, or ECD, should also be considered. Secondly, regarding the study of text-to-3D and image-to-3D generation in Sec. 4.4, the results are only qualitative, and only 8 samples from 2 categories are shown. This does not provide sufficient evidence to demonstrate the effectiveness of the proposed method in text-to-3D or image-to-3D generation. Thirdly, the evaluation of the efficacy of the pruning strategy is also incomplete. The paper does not compare the speed and computational cost with other methods (as emphasized in the abstract), and Fig. 6(b) also fails to illustrate the gains in model inference speed and memory overhead before and after using the pruning strategy. Fourthly, there is no discussion on the interpretability of fractal generation, which I think is a notable point. Could the author provide different visualizations for the same 3D shape that the generation of its 3D structure details from coarse to fine based on the hierarchical level from one to many?

### Questions
Please see the Weaknesses for details.

### Soundness
3

### Presentation
3

### Contribution
2
