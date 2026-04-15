# InstantSplamp: Fast and Generalizable Stenography Framework for Generative Gaussian Splatting

- Decision: Accept (Poster)
- Scores: 8, 3, 6, 5

## Abstract
With the rapid development of large generative models for 3D, especially the evolution from NeRF representations to more efficient Gaussian Splatting, the synthesis of 3D assets has become increasingly fast and efficient, enabling the large-scale publication and sharing of generated 3D objects. However, while existing methods can add watermarks or steganographic information to individual 3D assets, they often require time-consuming per-scene training and optimization, leading to watermarking overheads that can far exceed the time required for asset generation itself, making deployment impractical for generating large collections of 3D objects. To address this, we propose InstantSplamp a framework that seamlessly integrates the 3D steganography pipeline into large 3D generative models without introducing explicit additional time costs. Guided by visual foundation models,InstantSplamp subtly injects hidden information like copyright tags during asset generation, enabling effective embedding and recovery of watermarks within generated 3D assets while preserving original visual quality. Experiments across various potential deployment scenarios demonstrate that \model~strikes an optimal balance between rendering quality and hiding fidelity, as well as between hiding performance and speed. Compared to existing per-scene optimization techniques for 3D assets, InstantSplamp reduces their watermarking training overheads that are multiples of generation time to nearly zero, paving the way for real-world deployment at scale. Project page: https://gaussian-stego.github.io/.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The author skillfully integrates the visual foundation models into 3D steganography by leveraging cross-attention mechanisms to embed hidden information during the generation process. The proposed framework optimizes the balance between rendering quality and watermark fidelity, ensuring minimal distortion while preserving the integrity of the embedded data. The author validates the practicality of this approach through extensive experiments on various 3D assets.

### Strengths
1. The proposed InstantSplamp framework is highly innovative as it integrates watermarking directly into the 3D generation process, significantly reducing time overhead, making it practical for large-scale deployment. 
2. The methodology leverages visual foundation models and cross-attention mechanisms in a novel way to embed and recover hidden information effectively, while maintaining high rendering quality. 
3. The paper presents strong empirical validation across multiple deployment scenarios, demonstrating the method’s efficiency and generalizability with various 3D objects and modalities, including images, text, QR codes, and even video. 
4. The use of adaptive gradient harmonization to balance rendering fidelity and information hiding represents a practical and insightful solution to a common challenge in steganography, ensuring minimal visual quality degradation.

### Weaknesses
1. While Figure 1 illustrates the time efficiency improvements of the proposed method for watermarking, could you provide some quantitative experimental results to further emphasize this point?
2. The robustness testing only considers two types of corruptions (JPEG compression and Gaussian blur), which seems limited in scope. It would be valuable to include additional forms of corruption, such as noise, scaling, or cropping, for a more comprehensive evaluation. Additionally, a comparative robustness analysis with other state-of-the-art methods is missing, which would provide a clearer understanding of how InstantSplamp performs under various conditions.
3. How does the proposed method compare with other 3D watermarking approaches targeting binary messages, such as those for NeRF or other 3D representations? Specifically, it would be helpful to see a comparison of performance in embedding and recovering complex information, as well as any advantages InstantSplamp may have over these existing methods.

### Questions
See problems mentioned in Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The paper introduces an end-to-end framework for 3DGS steganography, embedding an image during Gaussian generation and recovering it from a specific rendering via a decoder network. In the hiding stage, the framework employs a cross-attention mechanism to seamlessly integrate the hidden image features into the spatial details of the intermediate Gaussian features. In the recovery stage, the decoder network extracts the hidden image exclusively from the rendering of a specific viewpoint. Additionally, an adaptive gradient harmonization technique is introduced, which functions as a masking mechanism, embedding the hidden information within certain model weights to preserve both steganographic ability and the visual quality of the renderings.

### Strengths
- The paper proposes a generalizable steganography mechanism that avoids additional time costs and modifications to the original Gaussian generation process.
- The experimental results in the paper demonstrate that the steganography capability of 3DGS surpasses that of similar methods applied in NeRFs.

### Weaknesses
- The method is similar to StegaNeRF and lacks sufficient novelty.
- The experimental baselines are too limited. Notably, an existing method, GS-Hider: Hiding Messages into 3D Gaussian Splatting, already achieves multi-scene information hiding within a 3DGS model.
- The experiments lack an analysis of steganographic capability, such as different capacity, resistance against steganalysis networks and robustness to additional distortions.

### Questions
- Does the proposed method exhibit superior steganographic capability compared to existing 3DGS steganography techniques?
- Is it possible to increase the capacity for embedding additional images within the steganographic framework?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a method, named InstantSplamp, to insert watermark information in generated 3D contents. The method is generalized, which means it does not need per-scene optimization. The time cost is extremely faster than previous methods.

### Strengths
1. The result achieved by the proposed method is much better than previous methods, especially for the hidden recovery performance.
2. The method does not need per-scene optimization, which is a generalized model and thus leads to faster speed.

### Weaknesses
1. The notations are not clear, it is hard to understand the figure 2 without notations. 
2. It's hard to understand the "AdaptiveGradientHarmonisation", the cosine similarity in Eq. (4) seems to be calculated based on all parameters. In this way, the similarity is not a vector value, so what does the mask stand for?
3. The training is only conducted on one model, i.e., LGM. This limits the application. Authors should show more results on different 3D generative models with different representations.

### Questions
See weakness

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
InstantSplamp introduces a fast, scalable framework that embeds hidden information like copyright tags into 3D generative models without additional processing time. Leveraging visual foundation models and cross-attention mechanisms, this approach integrates watermarking directly within the 3D Gaussian Splatting process. Unlike traditional methods requiring per-scene optimization, InstantSplamp minimizes overhead to nearly zero, enabling efficient large-scale deployment of watermarked 3D assets. A U-Net-based decoder recovers the hidden information, balancing visual quality with steganographic fidelity. This innovation addresses scalability challenges in 3D asset generation and protection, optimizing both watermark embedding and retrieval.

### Strengths
- It indeed doesn’t require per-scene optimization, which gives it a time advantage.

- The idea of injecting a watermark directly into the 3D generation model is good.

### Weaknesses
- In Figures 3 and 4, the 3D assets generated by your method show some artifacts in rendering, and the colors are somewhat distorted. Injecting the watermark affects the visual quality. Although it performs much better compared to StegaNeRF, the impact on visual quality due to watermark injection seems counterproductive.
- There is no 360-degree visual quality demo, and only two views are provided, which makes it hard to assess the rendering quality of the 3D assets and the quality of watermark extraction. It’s unclear whether the rendering quality of the 3D assets is 3D consistent.
- From the data in Table 1, the rendering quality of your method is not significantly better than LSB or DeepStega, and there’s no comparison with the latest method, GS-Hider.

GS-Hider: Hiding Messages into 3D Gaussian Splatting

### Questions
See Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2
