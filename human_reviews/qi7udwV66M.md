# Zero-Shot Image Compression with Diffusion-Based Posterior Sampling

- Avg Score: 4.25
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 5, 3, 3

## Abstract
Diffusion models dominate the field of image generation, however they have yet to make major breakthroughs in the field of image compression. Indeed, while pre-trained diffusion models have been successfully adapted to a wide variety of downstream tasks, 
existing work in diffusion-based image compression require task specific model training, which can be both cumbersome and limiting. This work addresses this gap by harnessing the image prior learned by existing pre-trained diffusion models for solving the task of lossy image compression. This enables the use of the wide variety of publicly-available models, and avoids the need for training or fine-tuning. Our method, PSC (Posterior Sampling-based Compression), utilizes zero-shot diffusion-based posterior samplers. It does so through a novel sequential process inspired by the active acquisition technique "Adasense" to accumulate informative measurements of the image. This strategy minimizes uncertainty in the reconstructed image and allows for construction of an image-adaptive transform coordinated between both the encoder and decoder. PSC offers a progressive compression scheme that is both practical and simple to implement. Despite minimal tuning, and a simple quantization and entropy coding, PSC achieves competitive results compared to established methods, paving the way for further exploration of pre-trained diffusion models and posterior samplers for image compression.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper transplants a compressed sensing approach to neural image compression. It turns out that this compressed sensing approach works quite well. The proposed approach is simple, as the transforms are linear. Besides, no training is required. Further, the result is also promising compared with HiFiC.

### Strengths
It is good to know that a linear transform is enough to achieve a high performance codec. Further, this paper is the first zero-shot perceptual codec for images of size 512x512 using latent diffusion models.

### Weaknesses
The first thing that confuse me is the impact of variance minimization of Adasense. It seems to me that variance minimization is important in compressed sensing. However, I am not sure about how variance minization will contribute to image codec. For example, is it possible to improve performance by chaning variance minization to entropy minization? As this target seems to be aligned with codec target.

The second thing that confuse me is the simplicity of transformation. Adopting a simple linear transform is great if the authors want to design a light weight codec. However, clearly this codec is not light weighted. As $\Pi$GDM does not limit the transformation to be linear, it becomes confusing why the authors stick to a linear transformation. Perhaps better performance can be achieved with non-linear transformation.

As this paper works on low bitrate regime, it is better to compare with [Towards image compression with perfect realism at ultra-low bitrates]. Further, as this paper works on zero-shot perceptual image compression, it is better to discuss and compare with [Idempotence and Perceptual Image Compression].

### Questions
I am curious about why the variance minimizing target works well in terms of rate-distortion performance for image compression.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper introduces a new compression algorithm leveraging pre-trained diffusion models combined with AdaSense, a sequential adaptive compressed sensing algorithm that selects the most relevant eigenvectors obtained through PCA to restore compressed data. The results are compared with classical compressors like JPEG, JPEG2000, BPG, and a GAN-based method, HiFiC.

### Strengths
The first half of the paper is well-written and well-structured.

The proposed algorithm compresses and decompresses an image, producing a result closer to the original compared to other methods. Compressing based on data importance seems interesting to me. 

The key idea behind their algorithm is between lines 184 and 191, and they achieved this progressive decompression using the AdaSense algorithm on the held matrix H.

 Indeed, the question, "How would the decoder know which transform to apply in recovering the image?" in line 198 was on my mind throughout the reading. The narrative of the paper is clear enough to answer such questions directly during the reading.

### Weaknesses
Line 466 states, "The primary limitation with our proposed method is its high computational cost." However, there is no discussion of computational time in the paper. JPEG, for example, may have lower quality but offers low latency. Timing is essential in compression/decompression algorithms, yet this information has been omitted in the paper.

The pre-trained diffusion model is a minor part of the algorithm. It’s not even the most relevant component, and yet the authors emphasize the zero-shot diffusion element in the title and abstract, while the main focus of the paper is on earlier steps, using diffusion only for the final posterior sampling. I would be interested in the authors' opinion on this emphasis.

While the mathematical description in the first half of the paper is accurate, I find the experimental section somewhat superficial.

In Fig. 2, it is unclear what the lines represent—I would expect a single point per dataset. Additionally, Fig. 2 is not well described in the text. The same applies to Fig. 4, which lacks detail on the method. It is presented as a high-level figure without adequate explanation in the text. Similarly, in Fig. 5, it’s unclear why the results degrade significantly in the columns on the right—perhaps it’s just a different seed? The same question applies to Fig. 6.

### Questions
Look at weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper proposes a zero-shot image compression method that utilizes pre-trained diffusion models to compress images without further training. The progressive encoding strategy has a high computational cost, but allows balancing distortion and perception.

### Strengths
* The main strength is that the approach is *training-free*, leveraging pre-trained diffusion models for zero-shot compression.
* The progressive encoding strategy allows balancing rate-distortion-perception based on application.
* The quality of the results are competitive with approaches that require further training or bespoke architectures.

### Weaknesses
* The main weakness is that the high computational costs associated with this approach are not well-discussed, and the high-level theoretical framework, justifying the strategy, is lacking in presentation. 
   * The results do not meet the standard of ICLR, as they are not state-of-the-art while incurring a presumably high computational cost. The performance (time/params per model) is not clearly presented, e.g. with tables and figures.
* The paper has many steps building on existing works (AdaSense) which are not well-known in the community, making it difficult to follow or understand the high-level theory underpinning this work. I found the pseudocode in part C of the supplementary material to be more informative for the general approach. In general, several parts of the methodology could be presented at a higher level, rather than reading like dependent steps of a complicated recipe.
* Discussion on some important related research is missing, such as Gao et al., NeurIPS 2022, "Flexible Neural Image Compression via Code Editing" and Frequency Aware Transformer, Li et al., (ICLR 2024).

### Questions
* Can you show a comparative table with performance data, such as the encoding/decoding time/#pretrained model params/with along with quality measures /R-D/B-D rates? If this is better than expected, I would happily change my opinion.
* How does this compare with Gao et al., NeurIPS 2022, "Flexible Neural Image Compression via Code Editing" that has a single decoder and can be adopted on existing pre-trained models and Li et al., ICLR 2024 "Frequency-Aware Transformer" (state-of-the-art in learned compression)?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper proposes a ``progressive'' scheme known as Posterior Sampling-based Compression (PSC) for image compression utilizing diffusion priors. Notably, the method operates in a zero-shot manner, meaning it requires no additional training. The image compression process, represented as \( y = Hx \), constructs the matrix \( H \) row-by-row with the assumption that an initial matrix and quantized measurements are shared between the encoder and decoder. This design, combined with the Adasense algorithm, ensures that the matrix \( H \) can be identically generated on both the encoder and decoder sides, thereby negating the need for communication of side information. The Adasense algorithm leverages a diffusion posterior sampler to draw multiple samples, subsequently computing PCA to estimate the top principal component of the posterior distribution's covariance \( p(x \mid H_{0:k}, Q(y_{0:k})) \).

In summary, while the paper introduces an interesting approach, substantial improvements are needed in terms of structure and presentation. Additionally, the technical novelty of the work appears limited, as the main components are not original contributions.

### Strengths
- The proposed method demonstrates good performance.
- The empirical results are reasonable.
- The paper extends the discussion to include latent diffusion models enhanced with text conditioning, which improves compression performance.

### Weaknesses
- The writing in the paper could be significantly improved to make the contributions clearer. The primary component of the algorithm, Adasense, is not a novel contribution of this work.
- The abstract claims the method is practical and simple, but there is no analysis or data provided to quantify the number of function evaluations required to achieve the reported results. For example, posterior sampling methods like DDRM or PiGDM typically involve at least 100 NFEs. Depending on the number of posterior samples used in line 1 of Algorithm 1, the cost could be as high as \( s \times 100 \) NFEs. 
- The paper does not provide sufficient clarity on how the active acquisition strategy helps reduce uncertainty.
- Two main components of the approach---posterior sampling methods (e.g., DDRM or PiGDM) and the Adasense algorithm---are not novel contributions of this work, limiting its originality.
-  While the paper mentions ``high computational cost,'' it does not provide any quantification or comparative analysis with other algorithms.
- The rationale for not considering non-linear measurements is unclear, as posterior samplers for non-linear inverse problems exist in the literature.

### Questions
- Why is PiGDM considered superior to DDRM for the compression task?
- What is the performance gap between the proposed method and neural compression-based methods?
- PSC is stated to be limited to linear measurements due to the posterior sampler. However, samplers such as DPS can handle both linear and non-linear inverse problems. Did the authors explore this option?
- Did the authors attempt to use more efficient generative models, such as consistency models, to reduce the number of NFEs?

### Soundness
2

### Presentation
2

### Contribution
2
