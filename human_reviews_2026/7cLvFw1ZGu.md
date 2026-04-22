# VoxSet: Sparse Voxel Set Tokenizer for 3D Shape Generation

- Avg Score: 4.40
- Decision: Reject
- Scores: 4, 4, 2, 4, 8

## Abstract
3D tokenizers are crucial in latent 3D generative models. 
Recent sparse voxel tokenizers can reconstruct detailed shapes but produce variable-length latent tokens which necessitate a two-stage generation pipeline. 
Conversely, vector set tokenizers have fixed-length latent tokens with higher compression but struggle with reconstruction quality. 
In this work, we introduce VoxSet, a novel tokenizer that combines the strengths of both approaches. 
Our method employs sparse voxels in the outer layers to capture fine surface details and a vector set bottleneck for high compression. 
This design achieves high-quality reconstructions while maintaining a compact and fixed-length latent code for different objects, eliminating the extra generation stage required by sparse voxel methods. 
Experiments demonstrate that VoxSet achieves competitive reconstruction quality compared to sparse voxel tokenizers, while sharing the simpler training and inference pipeline of vector set-based 3D generation models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduce VoxSet, a novel tokenizer that combines the strengths of both approaches. The method employs sparse voxels
in the outer layers to capture fine surface details and a vector set bottleneck for high compression. It achieves competitive reconstruction quality compared to sparse voxel tokenizers, while sharing the simpler training and inference pipeline of vector set-based 3D generation models.

### Strengths
1. The voxel set tokenizer that combines sparse voxel representation with fixed-length vector set compression, achieving high-fidelity reconstruction within a compact latent code.
2. An efficient implementation. 
3. The author conducts experiments on large-scale dataset, demonstrating the superior performance over existing works.
4. The generative quality of the proposed approach is amazing (Figure 4).

### Weaknesses
1. The paper directly combines two SOTA 3D shape representations, which is incremental.
2. Although the author provides quantitative comparisons on reconstruction, such comparison on generation is missing. The author compares Hunyuan3D et al qualitatively but not quantitatively.
3. The performance still falls behind state-of-the-art sparse voxel tokenizers, it shows that the proposed design stills sacrifices the quality. 
4. It would be better is the author provides more experiments like text-to-shape.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes VoxSet, a novel 3D tokenizer that combines the strengths of sparse voxel and vector set representations, achieving both high-quality and compact format. Experiments demonstrate reconstruction and image-to-3D generation capabilities of VoxSet. Moreover, this paper proposes an algorithm optimization for sparse voxel conversion stages and shows its efficiency.

### Strengths
The motivation of this paper is clear, and the proposed VoxSet effectively combines the advantages of sparse voxel and vector set representations.
The experimental results demonstrate the effectiveness of VoxSet in 3D shape reconstruction and image-to-3D generation tasks, showing its potential for practical applications.

### Weaknesses
The paper lacks detailed explanations of the proposed methods, for example, the comparison between different tokenizers and the image-to-3D generation flow. This makes it difficult to fully understand the contributions and novelty of the work.
The tokenizer requires a long training time, which is not friendly for hyperparameter tuning in practice. 
The performance improvement compared to voxel-based methods is not very significant, requires further justification of the efficiency of the proposed method.

### Questions
1. The methods in Section 3.1 are several straightforward algorithm optimizations, it's better to present a pseudo code (maybe in the Appendix) to illustrate the algorithm. Additionally, provide more information about previous works and consider analyzing why they did not utilize similar optimizations to demonstrate the novelty.
2. It's better to provide comparisons in Figure 2 with the sparse voxel tokenizer and vector set tokenizer, to clarify their differences between the proposed sparse voxel tokenizer.
3. Section 3.3 is a bit brief, which doesn't clarify the image-to-3D generation process. For example, what is the role of the tokenizer in the image-to-3D generation, and where is its input from and output to?
4. As shown in Table 2, VoxSet has a similar performance and token count to the sparsecube method. How about the quantitative efficiency comparison between these two methods?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper combines the Shape2VectorSet-like Perceiver IO tokenization paradigm with sparse voxel encoder-decoder, which is stated to have enjoy the strenghs from both perspectives: (1) The latent compactness of S2VS; (2) The ability of representing details via Sparse Voxel Convolution. It represents reconstruction comparison with others to demonstrate the superb geometry compression ability. Moreover, it shows an Image-to-3D experiment to show the feasibility of generating such compact voxel latent tokens.

### Strengths
**The core argument is fair**: TRELLIS-like sparse voxel does have the need to generate sparse voxel locations first and then generate the associated latents. To tackle this, the author put a Perceiver IO-like structure within the bottoleneck of sparse voxel encoder-decoder network, enabling a fixed length compact representation. The sparse voxel encoder-decoder in-return will improve the reconstruction details compared to pure S2VS-like methods.

### Weaknesses
1. The **presentation** in the paper:

    a. Section 3.1, it uses few long paragraphs to mention the **implememtation details**. While it is appreciated to show details of implementation, it is strongly recommended to move the technical details to supplememntary. 

    b. The more important question is that if those implementations really **have something new or just engineering**? As a graphics people, I do think algorithm efficiency is one of the keys of graphics system application, but based on my evaluation, the paper is not inventing something new here, which makes the point stronger that it should move those contents to supplementary/appendix section.
 --------
&nbsp;

2. *The two-edge sword*:  
     a. It is a fact that we do need two stages of generation in TRELLIS-like sparse voxel generation task. However, it is proved that we can decode to various representation based on a unifed structral latent. On the other hand, this paper only shows the geometry generation instead of incooparating material generation etc.

     b. **More importantly**:  Although it is an important beneficial to have a more compact 3D representation, it should also be stated that is essentially beneficial for generation of 3D based on that representation. That is why it is recommended to measure rFID as well as FID in 2D image tokenization and generation task. So when we look at Figure 4., which is the sole 3D generation comparison in the paper, we can notice an *very important issue* of Image-to-3D demo of Voxel Set, **Alignment**:

          i . **Third row tiger-child-statue**: the human head and tiger orientation alignment with the input image of Voxel Set is not only incorrect but also obviously worse than Huyuan3D and Hi3DGen.
         ii. **The last row**: Some big cracks on the cat body shows on Huyuan3D-2.1 but not on Voxel Set. 

     In all, it makes very hard to clearly demonstrate it is a better representation for generation. After all, **alignment** with input images is the most important feature we want to see for an Image-to-3D task.
 --------

&nbsp;

3. Lack of Spark3D/Hitem3D comparison in reconstruction and generation: I think it is an important baseline both in recosntruction and generation and it is being mentioned in introduction but I am not sure why it is not compared somehow.

### Questions
1. As raised in the weakness session, we need more baseline comparisons. Especially, Spark3D/Hitem3D.

2.  It is crucial to dig further on the generation ability based on Voxel Set. I recommend you design more experiments to find out why the alignemt is the issue.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces VoxSet, a Sparse Voxel Set Tokenizer for 3D shape generation, designed to combine the strengths of two dominant 3D tokenization methods: sparse voxels and vector sets. This is a very interesting work.

### Strengths
The core strength is the novel hybrid design that integrates sparse voxels for high-resolution detail capture with a fixed-length vector set bottleneck for efficient compression and a simpler generation pipeline.

- The method achieves competitive reconstruction quality, outperforming vector set methods (like Dora-1.1) at the same token count and remaining competitive with state-of-the-art sparse voxel methods (like SparseFlex and SparseCube) with fewer average tokens. 

- The paper is well-structured and easy to follow.

### Weaknesses
- While competitive, the reconstruction quality (F1-score) of VoxSet still slightly lags behind the current state-of-the-art sparse voxel tokenizers like SparseCube and SparseFlex, especially at their average token counts.

- The comparison in Table 2 shows that your method performs well with 4096-32768 fixed tokens, while SparseCube and SparseFlex use variable lengths with much larger averages ($\approx 34K$ and $\approx 218K$). While impressive for compression, could you provide results for VoxSet with a higher, but still fixed, token count (e.g., $65536 \times 64$) to see if it can fully close the minor quality gap with SparseCube's $34K$ average, or if the limitation is inherent to the vector-set bottleneck? And what about the memory consumption?

- Your method uses sparse convolution layers (voxel-based) at the outer layers and a transformer (vector-set based) in the middle. Have you performed an ablation to quantify the contribution of each component to the final reconstruction quality? For example, comparing a parse-voxel-only version at resolution (before the transformer) to the full VoxSet model. 

- What about the training time and inference time for the voxset？

- For the proposed accelerated voxel-to-mesh algorithm, Figure 1 claims super improvements in the time efficiency. What about the final mesh quality? Is it comparable?
    
- For the visual comparison in figure 3, could you demonstrate the performance under the same number of tokens?
    
- I notice that the proposed method has the ability to reconstruct the thin structure in figure 3. What about the noisy input? 

- You mention downsampling the $1024^3$ input to $128^3$ using sparse average pooling. Could you elaborate on how the sparse average pooling is implemented in detail? Specifically, how is the average calculated when the surrounding voxels might be nonexistent (sparse)?

- There are similar work (Geometry Distributions) published at ICCV 2025; some discussion and comparison on final mesh quality should be evaluated.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces VoxSet, a hybrid 3D shape tokenizer that combines the advantages of sparse voxel and vector set representations for efficient 3D generative modeling. The authors argue that existing methods face a trade-off between fidelity (sparse voxels) and compactness (vector sets). VoxSet integrates sparse convolution layers to capture fine local geometry and employs a transformer-based vector-set bottleneck for fixed-length latent compression. The method maintains the reconstruction quality of sparse voxel tokenizers while simplifying the generation pipeline to a single stage. Extensive experiments on the Trellis500k dataset show competitive or superior reconstruction results to vector-set baselines and close performance to state-of-the-art sparse voxel tokenizers, with quantitative and qualitative evaluations. The authors also provide CUDA-based acceleration for voxel-mesh conversion.

### Strengths
1. Well-motivated hybrid design effectively combining sparse voxel and vector-set representations.  
2. GPU-accelerated voxel–mesh conversion (CUDA flood fill and sparse marching cubes) significantly improves preprocessing and inference speed.  
3. Simplified single-stage generative pipeline by removing variable-length latent encoding.  
4. Comprehensive experiments (quantitative, qualitative, and ablation) showing strong reconstruction with fewer latent tokens.  
5. Potential applicability to future 3D diffusion and flow-based generative models.

### Weaknesses
1. Limited conceptual novelty — mainly a pragmatic hybridization rather than a new theoretical idea.  
2. Overemphasis on implementation details over architectural or theoretical insights.  
3. Missing scalability and robustness analysis, especially for noisy or imperfect meshes.  
4. Shallow flow model evaluation and lack of strong baseline comparisons.  
5. Insufficient analysis of latent space behavior and trade-offs between token efficiency, fidelity, and complexity.

### Questions
1. How sensitive is VoxSet to voxel resolution and latent token count during training and inference? Does performance degrade gracefully with fewer tokens?  
2. Would the hybrid approach still be beneficial if the generative prior directly operated on sparse voxel structures (as in Trellis or Sparc3D)?

### Soundness
4

### Presentation
4

### Contribution
4
