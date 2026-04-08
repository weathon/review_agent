## Human Reviewer 1

### Summary
This paper addresses the critical need for provenance in diffusion-generated images by proposing a new watermarking scheme. The authors introduce "Spherical Watermark," a lossless and encryption-free framework that embeds a binary watermark into the initial Gaussian noise by mixing it with random padding, projecting it onto a unit sphere, applying an orthogonal rotation, and scaling it with a chi-square-distributed radius. The method is theoretically proven and empirically demonstrated to be statistically indistinguishable from standard Gaussian noise, while also being computationally efficient and robust to various post-processing and adversarial attacks, outperforming prior lossless methods.

### Strengths
S1 (technical novelty): The proposed spherical mapping module is a novel technical contribution. It provides a clear mathematical pipeline to transform a structured binary vector into a vector that is statistically indistinguishable from a standard Gaussian distribution.

S2 (theoretical foundation): The "lossless" claim is strongly supported by a rigorous theoretical analysis. The paper proves that the watermarked noise distribution matches a true Gaussian prior up to third-order moments by leveraging the properties of spherical 3-designs.

S3 (strong performance): The paper shows clear improvements in efficiency and robustness over baseline methods. The method is also shown to be effectively indistinguishable from the original, non-watermarked distribution.

S4 (comprehensive experiments): The ablation studies clearly justify the design, demonstrating the necessity of both the binary embedding and spherical mapping modules for undetectability and robustness, respectively.

### Weaknesses
W1 (motivation): The paper claims that existing methods require per-image key storage or cryptographic overhead, but this method also has cryptographic overhead (Eq. 13). Hence, The "encryption-free" claim is potentially misleading.

W2 (scope of robustness): The paper does not explicitly test against attacks that are more specific to generative models, such as watermark destruction via diffusion-inversion and re-generation with different noise. The authors have acknowledged that resisting editing/forgery is a limitation but is out of scope.

W3 (missing related work): Some recent works are not discussed, such as [1,2].

- [1] Wei et al. Robust watermarking for diffusion models: A unified multi-dimensional recipe. 2025.
- [2] Wang et al. SleeperMark: Towards Robust Watermark against Fine-Tuning Text-to-image Diffusion Models. 2025.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
4

### Rating
6

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper proposes a training-free and lossless diffusion watermarking method that encodes a binary watermark message into the initial noise of a diffusion model. The method consists of two main transformation steps. First, the message is encrypted by mixing it with random padding through a structured binary embedding matrix. Second, the resulting vector is projected onto a unit sphere and then transformed by a random orthogonal rotation followed by a radius rescaling using a chi distribution. This produces a final watermark vector that closely matches the distribution of standard Gaussian noise, making it suitable as the initial noise for diffusion models and statistically indistinguishable from normal noise. The method is compatible with multiple diffusion models, does not modify model parameters, and enables fast decoding.

### Strengths
1. The paper is well motivated and logically structured. The paper addresses the problem of embedding watermarks in diffusion-generated images in a clean and motivated way. 
2. The proposed method is reasonable and effective. The approach avoids any model fine-tuning or training and leverages statistical geometry (spherical design + chi rescaling) to achieve high-quality watermarking while maintaining indistinguishability.
3. The experiments are comprehensive. The paper evaluates watermark accuracy and detectability under different attacks. The ablation study also demonstrates the effectiveness of their key designs.
4. The paper is easy to read and well-organized, with clear diagrams and concrete definitions.

### Weaknesses
1. Lack of diffusion-based attacks: While the paper evaluates robustness under post-processing and adversarial attacks, it does not include experiments on regeneration or rinse-based attacks (e.g., re-diffusion or editing using other diffusion models), which have been recently identified as strong attacks for watermark removal. I suggest the authors refer to arXiv:2401.08573 and consider incorporating some of their benchmarking strategies.

2. Storage overhead for decoding: To decode the watermark, the user must store the embedding matrix $T$ (specifically, the sparse matrix 
$R$) and the rotation matrix $C$. Since $R \in \mathbb{F}_2 ^{N l_m \times l_r}$ and $C \in \mathbb{R}_2 ^{l_x \times l_x}$, the memory cost could be significant, especially when generating high-resolution images with large latent dimensions. A naïve implementation would incur nontrivial storage. Discussion about the storage overhead should be included in the main text.

### Questions
1. Have you evaluated your method under regeneration-based or rinse attacks? If not, can you comment on the potential vulnerability under such transformations?

2. Could you elaborate on whether $T$ and $C$ are reused across images, or whether they are derived per image? What is the typical memory overhead for storing or generating them in a realistic deployment?

### Soundness
4

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 3

### Summary
This paper introduces Spherical Watermark, an encryption-free and lossless watermarking approach designed for diffusion models. It focuses on tracing and verifying the provenance and authenticity of AI-generated images, addressing recognized limitations of current watermarking techniques, such as quality degradation, detectable shifts, and key management complexity. The method embeds watermarks into the latent noise with indistinguishable Gaussian statistics, utilizing a high-entropy binary embedding and spherical mapping mechanism. The framework maintains perfect image fidelity and allows rapid watermark extraction without modifying the diffusion model. Empirical results demonstrate strong robustness against common image manipulations and adversarial attacks, competitive and often superior to prior approaches. The paper also emphasizes ethical considerations and provides theoretical analysis for transparency and reproducibility.

### Strengths
- The proposed watermarking technique preserves image quality, making watermarked and non-watermarked images visually indistinguishable. The framework is efficient, enabling fast watermark extraction with no need for per-image keys or model modifications.
- Robustness is demonstrated against a wide range of image processing operations and adversarial settings.
- The solution is well-supported with both theoretical analysis and comprehensive experiments.
- The method is deployable on mainstream diffusion architectures and easy to integrate in practice. The authors address the ethical implications of watermarking and ensure reproducibility standards.

### Weaknesses
- The method may have limitations when facing extremely sophisticated adversarial attacks specifically designed to break watermark recovery.
- There is limited discussion on extending the approach to content editing or direct forgeries, such as partial GAN-based manipulations.
- Some implementation parameters may require careful adjustment for different generative scenarios and applications.

### Questions
This is an interesting paper. So I want to know what the main motivation is for the introduction of the spherical mapping module in the context of watermark embedding. Secondly, if this method is so promising, are there any limitations for this method to scale up in practice?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
2

---

## Human Reviewer 4

### Summary
This paper introduces Spherical Watermark, a novel encryption-free and lossless watermarking framework for diffusion models designed to overcome the drawbacks of existing methods that either degrade image quality or rely on computationally expensive cryptography. The core contribution is an elegant method that embeds a binary watermark into the model's initial Gaussian noise latent vector. This is achieved first through a binary embedding module that creates a high-entropy bitstream, which is then processed by a spherical mapping module that projects it onto a unit sphere, applies an orthogonal rotation, and scales it with a chi-square distributed radius. This process yields a noise vector that is statistically indistinguishable from a standard Gaussian distribution, a claim supported by theoretical proofs. The framework's key contributions include this new mapping technique, the elimination of cryptographic overhead, and state-of-the-art performance. Experiments show that Spherical Watermark preserves high visual fidelity while offering superior robustness against attacks and a dramatic improvement in computational efficiency, with watermark extraction being orders of magnitude faster than its closest lossless competitor.

### Strengths
1.	The core methodology is novel and elegant, using spherical geometry to transform a binary watermark into statistically standard Gaussian noise, which successfully bypasses the need for complex and computationally heavy cryptographic components used in prior lossless methods.
2.	The paper is supported by a strong theoretical foundation, providing formal proofs that the watermarked noise distribution matches a true Gaussian prior up to the third-order moments by leveraging concepts like spherical 3-designs.
3.	The method demonstrates exceptional performance and efficiency, as it is extremely fast in the extraction phase (approximately four orders of magnitude faster than its closest competitor), shows superior robustness against various attacks, and maintains high fidelity and undetectability.
4.	It offers excellent scalability and capacity, handling large watermark payloads without the performance degradation seen in competing approaches, which makes it highly flexible for applications requiring the embedding of rich metadata.

### Weaknesses
1.	The comparison of computational efficiency should be included in the main paper instead of the appendix as it can effectively demonstrate the advantages of this method compared to PRC watermark.
2.	Lack of experimental comparison on newer models such as FLUX and Qwen image.
3.	Does this method rely on the accuracy of inversion? I want to know if different inversion methods will affect the accuracy of extraction, and if the sampling step size will also have an impact. In other words, when will the deviation between the latent obtained from inversion and the latent obtained from the original embedding render this method ineffective?

### Questions
See the weakness.

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
8

### Confidence
2