# Implicit Inversion turns CLIP into a Decoder

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 2

## Abstract
CLIP is a discriminative model trained to align images and text in a shared embedding space. Due to its multimodal structure, it serves as the backbone of many generative pipelines, where a decoder is trained to map from the shared space back to images. We show that image synthesis is nevertheless possible using CLIP alone—without a pre-trained generative decoder or CLIP tuning. Our approach optimizes a frequency-aware implicit neural representation that encourages coarse-to-fine generation by stratifying frequencies across network layers. To stabilize this inverse mapping, we introduce adversarially robust initialization, a lightweight Orthogonal Procrustes projection to align local text and image embeddings, and a blending loss that anchors outputs to natural image statistics. With CLIP frozen, this framework unlocks capabilities such as text-to-image generation, style transfer, and image reconstruction. Our findings suggest that discriminative models may hold untapped generative potential, hidden in plain sight.

Code: https://github.com/OmnAI-Lab/implicit-inversion

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces $CLIP^{-1}$, a novel framework for text-to-image synthesis that demonstrates the latent generative capabilities of a frozen, pre-trained discriminative model (CLIP) . The central claim is that a dedicated, trained generative decoder is not strictly necessary. Instead, the authors reframe image synthesis as an inversion problem: finding an image that, when encoded by CLIP's image encoder, matches a target text embedding .The paper's core contribution is a set of techniques to solve this notoriously ill-posed problem, which typically yields unstable, non-naturalistic images (adversarial examples) . The authors' solution avoids direct pixel optimization and instead optimizes the weights of a frequency-aware Implicit Neural Representation (INR), which inherently enforces smoothness and a structured, coarse-to-fine generation process .To further stabilize this inversion, the $CLIP^{-1}$ pipeline introduces three key components:An adversarially robust initialization using Adversarial Weight Perturbation (AWP) on a blurred seed image, which anchors the optimization in a stable "flat" region of the loss landscape .A lightweight, geometric solution to the modality gap using Orthogonal Procrustes Analysis to project the target text embedding onto the image embedding manifold, providing a more stable target .A blending loss that acts as a natural image prior, regularizing the output by encouraging it to remain close to the statistics of real image embeddings .Empirically, $CLIP^{-1}$ substantially outperforms other training-free inversion methods like DAS, achieving a Fréchet Inception Distance (FID) that is less than half of its competitor . The framework also demonstrates zero-shot versatility on downstream tasks, including image reconstruction, controlled text-guided editing, and neural style transfer.

### Strengths
1) The paper's core idea, unlocking the generative potential hidden within a frozen discriminative model, is highly original and significant. It moves beyond simply using CLIP as an encoder and shows it can be "run in reverse" as a generator, challenging the conventional wisdom that separates discriminative and generative models.

2) The key innovation is reframing the inversion problem from pixel space to the weight space of an Implicit Neural Representation (INR). This choice enforces smoothness, helps in preventing high-frequency artifacts and enables a structured coarse-to-fine optimization by leveraging the frequency-stratified architecture of the FINER network.

3) The paper introduces simple, effective, and computationally cheap solutions to complex problems. I like the use of Orthogonal Procrustes Analysis to geometrically correct the modality gap, as it requires no training and is demonstrably effective.

4) The combination of Adversarial Weight Perturbation (AWP) for initialization and the blending loss creates a more stable optimization process. These components successfully anchor the search and prevent it from converging to the "adversarial examples" (perceptually meaningless noise) that have plagued previous inversion attempts. 
5) The ablation study is thorough and provides a clear justification for each component of the $CLIP^{-1}$ pipeline.

### Weaknesses
1) The paper's primary weakness is its discussion of the computational cost. The method is "training-free", but it shifts the entire computational load to inference time, which requires an iterative optimization process (400 steps). The runtime analysis in the appendix compares $CLIP^{-1}$ only to DAS, another inversion-based method. The main quantitative comparison compares quality (FID/IS) against SOTA generative models like LDM and GLIDE but critically omits a comparison of inference time and VRAM usage. A single forward pass through a diffusion decoder is likely much faster than the 1+ minute optimization required by $CLIP^{-1}$. To fairly evaluate this as a generative "decoder" alternative, this trade-off must be made explicit. 
2) As the authors rightly acknowledge, a significant gap in photorealism and fine-detail fidelity remains when compared to SOTA diffusion models. The generated images, while coherent, often have a "painterly" or slightly artificial quality and lack crisp, high-frequency textures. This limits the method's immediate practical application for photorealistic generation.Initialization.
3) The full, best-performing model relies on retrieving a semantically relevant seed image from the LAION dataset to perform its AWP-based initialization. While the out-of-distribution evaluation shows that a "Plain $CLIP^{-1}$" variant (starting from random weights) can still function, the main method's output is still influenced by this seed. This could constrain the creative novelty of the generations, potentially biasing them towards common compositions found in the retrieval database.

### Questions
Following up on the main weakness: 
1) Could the authors provide a direct comparison of the inference time (time per image on a standard GPU like an A100 or RTX 4090) and peak VRAM usage for $CLIP^{-1}$ against the SOTA generative baselines listed in Table 1 (e.g., LDM-KL-8 or GLIDE)? This would be invaluable for understanding the practical trade-off between the method's training cost and its inference cost.
2) The optimization process is set to 400 steps. Have the authors experimented with this hyperparameter? How does image quality (FID) degrade as the number of steps is reduced? Is there a "sweet spot" that balances quality and speed? 
3) This question is about the robustness of the initialization strategy. The paper's AWP-trained seed image is described as a "robust anchor". We are interested in how strongly this anchor constrains the final generation.
- First, the paper shows a "Plain $CLIP^{-1}$" variant that starts from random INR weights. Could the authors provide a qualitative comparison for this variant? Does it simply converge slower, or does it suffer from more significant artifacts, confirming the necessity of the AWP anchor?
- Second, what happens in an adversarial scenario where the retrieved seed image is semantically incorrect (e.g., the prompt is "a photo of a dog" but the retrieved seed is "a photo of a car")? Is the optimization pipeline, guided by the Procrustes-corrected text embedding and the blending loss, strong enough to correct this completely wrong starting point and still generate a "dog"? Or does the robust anchor effectively trap the optimization near the (incorrect) seed, leading to a failed generation?
4) The use of Orthogonal Procrustes is a very clever and simple solution for the modality gap. Did the authors experiment with any alternative methods, such as learning a small, lightweight mapping network (e.g., a simple MLP) to bridge the gap? Was Procrustes chosen primarily for its simplicity and training-free nature, or did it also empirically outperform other simple alternatives?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates image synthesis through CLIP inversion. Rather than optimizing directly in the pixel space, they leverage implicit neural representations (INRs) to perform the inversion. To stabilize the inverse mapping, they incorporate techniques such as adversarial training, orthogonal Procrustes projection, and a blending loss.

### Strengths
- The paper provides a well-rounded review of related works across multiple relevant areas, helping readers grasp the broader context and motivation behind the proposed approach.
- It integrates several complementary techniques to tackle distinct challenges—for instance, using Adversarial Weight Perturbation (AWP) for robust INR initialization and incorporating natural image priors through blending.
- Unlike prior CLIP inversion methods that rely on direct pixel-space optimization and often produce noticeable artifacts, the proposed approach demonstrates improved visual quality in its qualitative results.
- The paper includes comprehensive ablation studies, analyzing the contribution of each introduced component both quantitatively and qualitatively.

### Weaknesses
- There are some minor issues in how certain points are presented. For instance, in the related work section, the statement “Modern image generators fall into three…” overlooks autoregressive models, which are also widely used and should be acknowledged.
- From a practical perspective, while the approach is conceptually interesting, it’s not particularly applicable in real-world scenarios. As shown in the results, the generated images still lag far behind other methods. However, the work remains valuable from an interpretability and analytical standpoint.
- The visualization and presentation could be improved. Specifically, Figures 5 and Tables 2 and 3 are rendered too small, which hinders readability.

### Questions
-

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper propose a new method of inversing CLIP encoder to achieve text-to-image geneartion using $\textnormal{CLIP}^{-1}$.
This is achieved thorugh a training-free approach, leveraging INR to achieve image embeddings similar to target text embeddings.

### Strengths
The technical contributions of this work are interesting. The authors propose techniques to: (1) enable a training-free approach that generates images without training any decoder, (2) mitigate the CLIP modality gap, and (3) achieve a robust, generalizable framework by leveraging AWP.

The comparison to existing work is solid, and the reported FID and other metrics are reasonable for this class of technique.

### Weaknesses
The primary concern with this work is its relevance and timeliness. The idea of inverting CLIP for image generation was novel and exciting a few years ago; however, the community has since shifted toward more scalable and expressive generative models.

Moreover, the outputs here are not yet high-quality enough, which is a significant shortcoming for practical use. Therefore, the work’s practical impact seems limited to me.

### Questions
How do the authors think about the scalability of this approach, given the per-image optimization loop?

Also, would this method extend to higher-resolution outputs?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
In this paper, the author introduces CLIP−1, a novel approach for text-to-image synthesis by inverting CLIP embeddings without relying on a pretrained generative decoder or fine-tuning CLIP. The method leverages Implicit Neural Representations (INRs) optimized layer-by-layer in a frequency-aware manner, enabling coarse-to-fine image generation. Overall, this inversion method sheds light on how encoder models can potentially be used as image generations and also on the inner workings of CLIP. However, I believe that there's still a way to go for such methods to be used as pure image generators.

### Strengths
- The paper proposes a decoder-free and tuning-free approach for text-to-image synthesis, which is computationally efficient. 

- The method supports zero-shot generalization to multiple tasks, including reconstruction, style transfer, and controlled edits, within a unified framework

- Extensive experiments on MS-COCO and Flickr30k demonstrate strong performance and also ablations showing improvements over other inversion methods.

### Weaknesses
- While the generated images are semantically aligned, fine-grained spatial details and textures occasionally exhibit distortions or artifacts. I understand that the authors are not competing with SoTA image generation methods, but a discussion on it would be fruitful. 

- I am a bit apprehensive on the practical applications of the method beyond just understanding the CLIP space. How far is the performance (quantitatively) from pure image generators?

- While FID, IS, and CLIP-SIM are reported, additional perceptual metrics or user studies could strengthen the evaluation of visual quality.

### Questions
Overall, the paper makes a significant contribution by repurposing a frozen discriminative model for generative tasks, showcasing its potential for image synthesis. However, limitations in handling complex prompts which suggest areas for improvement and lack of comparisons to pure image generators draw me towards a reject. The work however opens promising directions for leveraging pretrained models in generative applications. 

I urge the authors to answer questions about the practical utility of inversions with encoder models? How do they see such methods replacing pure image generators or how can such methods be used to improve the encoder capabilities itself. I believe these are unanswered questions.

### Soundness
3

### Presentation
3

### Contribution
3
