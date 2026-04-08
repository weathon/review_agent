## Human Reviewer 1

### Summary
This paper investigates whether modern deep watermarking is approaching the capacity–quality–robustness limit. Rather than working within the classical Gel’fand-Pinsker problem [1], the authors adopt a geometric, high-dimensional grid view to analyze the trade-off between PSNR and capacity under perfect decoding, first in the noise-free case and then under linear distortions. Empirically, a handcrafted construction comes close to the PSNR-only bound, and an expanded model (Chunky Seal) achieves ~4× capacity while maintaining PSNR and robustness comparable to VideoSeal. These results suggest that current deep watermarking systems remain far from the achievable limit in terms of capacity at a given quality/robustness level.

[1] S. Gelfand and M. Pinsker, “Coding for channel with random parameters,” Prob. of Control and Inf. Th., vol. 9, no. 1, pp. 19–31, 1980.

### Strengths
1.	The paper tackles a fundamental and important question: after roughly several years of progress in deep learning–based watermarking, are we actually approaching the limit of the quality–robustness–capacity trade-off? Rather than starting from the classical information-theoretic setting, the authors proceed from a high-dimensional grid perspective and derive, step by step, the maximum information capacity, the capacity under a PSNR constraint, and the capacity under linear distortions.

2.	Empirically, a handcrafted watermark in the noise-free setting approaches the theoretical upper bound, while under noise the Chunky Seal model achieves higher capacity yet similar PSNR and robustness to Video Seal, further indicating current limits of deep learning watermarking performance.

3.	The theoretical development is reasonable and clear: it analyzes the limitations of deep models and articulates a plausible theoretical upper limit.

### Weaknesses
1.	The related work is not sufficiently comprehensive. The paper does not adequately cite and explain existing traditional information-theoretic analyses, making it hard to evaluate the advantages of the proposed geometric high-dimensional grid approach over prior, thoroughly studied capacity analyses from the information-theoretic perspective.

2.	The current capacity analysis remains limited to linear distortions; discussion of non-linear distortions is still quite limited.

3.	The observed capacity gains via tiling are unusual, yet the paper provides little analysis of why this phenomenon occurs.

4.	In the noise-free case, the paper proposes a handcrafted method that nearly attains the theoretical optimum; however, it remains unclear how to approach the theoretical capacity under noise.

5.	In the high-capacity experiments, the paper does not compare against LISO [1], which achieves 4 bpp at ~25 dB PSNR with near-100% accuracy in the noise-free case. A study of high-capacity watermarking should analyze and compare with this method.

[1] Chen X, Kishore V, Weinberger K Q. Learning iterative neural optimizers for image steganography. ICLR 2022.

### Questions
1.	For non-linear distortions, if a theoretical analysis is not feasible, do the authors have empirical methods to predict or measure the upper-bound capacity?

2.	The finding that tiling increases capacity is quite unexpected. Do the authors have an analysis of why this occurs? Why does direct training at high capacity tend to fail?

3.	The straightforward expansion of Video Seal yields a model roughly 11× larger, yet the capacity is still only 0.0052 bpp. This does not appear to be a viable path toward the paper’s proposed theoretical upper limits. What new design ideas do the authors have for future models?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper introduces a formalization of watermarking capacity and shows how many DL approaches to watermarking do not achieve this capacity. This work then proposes a new methodology that is able to use 1024 bits encoded.

### Strengths
This paper derives a first principal approach to understanding a fundamental question in watermarking: the theoretical limits of capacity that images can hold/embed under image quality and also robustness. Current literature usually uses on the order of 100-200 bits which is sufficient for many cases but this work highlights that this is under-represented. The authors show that using up to 1024 bits in practice has little to no performance drop.

### Weaknesses
I think that the current suite of attacks are kind of basic. I would ideally like to see some more modern attacks (regeneration, rinsing, and maybe even a combination of a lot of attacks). I think that these settings can really test the robustness of the method.

### Questions
- I would be curious to understand the theoretical formulation of a combination of attacks.
- I would also like to see if there is a principled way to understand regeneration attacks in your current framework.
- (The regeneration/other tests I asked for I mostly care about for empirical validation/comprehensiveness.)

### Soundness
3

### Presentation
4

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper investigates the theoretical capacity of image watermarking and highlights the large gap between theoretical upper bounds and practical deep learning performance. The authors derive capacity bounds under PSNR constraints by interpreting PSNR as an equivalent l2-ball constraint within the image cube. By counting integer points within the cube-ball intersection, they estimate the achievable bit capacity and demonstrate that current models (e.g., VideoSeal) use only a fraction of the potential.
Empirical results show that under a simple grayscale + PSNR setup, VideoSeal fails to reliably encode 1024 bits, while linear or handcrafted embedding-decoding schemes can succeed with 1024–2048 bits. The proposed method, ChunkySeal,  demonstrate that scaling up the baseline achieves 4× higher capacity (256 → 1024 bits) while maintaining quality and robustness, but still remains far from the theoretical limits.

### Strengths
- Clear theoretical framework linking PSNR and l2 constraints : The cube–sphere intersection formulation provides intuitive and quantitative insight into capacity bounds.
- Well-controlled experiments isolating key variables : Simplified settings (single grayscale image, PSNR constraint only) help pinpoint that architectural/optimization limits (not data or format) cause the current capacity gap.
- Practical contribution (ChunkySeal) : A straightforward scaling of the embedder/extractor boosts performance, serving as a sanity check and strong baseline for future work.
- Inclusion of robustness considerations: The paper extends its theory to cover linearized transformations (LinJPEG, rotation, scaling), proposing heuristic and conservative bounds.

### Weaknesses
- Limited formal robustness analysis : The provided bounds for transformations (Bounds 10–13) are heuristic or overly conservative. Non-linear effects such as quantization and rounding are not analytically handled.
- PSNR as a potentially weak perceptual proxy : The paper’s reliance on PSNR ignores perceptual discrepancies—two images with identical PSNR can differ visually. Extensions using LPIPS or MS-SSIM would better reflect real-world perceptual constraints.
- Lack of in-depth analysis on model failure causes : While VideoSeal’s underperformance is empirically demonstrated, the architectural or optimization bottlenecks (e.g., skip connections, normalization, bandwidth limits) are not deeply dissected.
- Simplified image-space assumption : The capacity derivation assumes a BMP-like uncompressed pixel grid. Real-world formats (JPEG, PNG) involve non-linear compression steps not fully captured, even with the LinJPEG approximation.

### Questions
- Validity of PSNR and L2-ball equivalence : Have you tested whether two perturbations with equal PSNR but different visual artifacts yield consistent capacity results? Would using perceptual metrics (LPIPS, MS-SSIM) alter the theoretical limit?
- Failure analysis of VideoSeal : What specifically prevents VideoSeal from scaling beyond 1024 bits—optimization instability, insufficient representation capacity, or architectural bottlenecks? Any diagnostic results (e.g., layer-wise activation spectra) to support this?
- Completeness of Figure 2 cases : Does Figure 2 fully capture all geometric cases? What happens if the sphere’s center lies along cube edges or planes (partial overlap)? Are there discontinuities or nonlinear capacity changes in these intermediate configurations?
- Image format generalization : How would your capacity estimation adapt to real formats like JPEG (non-linear quantization) or PNG (filter-based compression)? Can LinJPEG capture these effects accurately, or are there measurable deviations?
- Theoretical limits vs. hyperparameter tuning : If a theoretical limit exists, why can’t it be reached through simple hyperparameter sweeps (e.g., reconstruction loss weight )? Is the gap due to optimization dynamics or representational constraints? A quantitative analysis (e.g., singular value decomposition of the embedding mapping) would clarify this.

Efficiency and practicality of ChunkySeal.ChunkySeal reaches higher bit capacity but at the cost of ~760M parameters. How feasible is this in deployment scenarios? Could lighter architectures (e.g., tiled embeddings, structured transforms) achieve similar performance?

### Soundness
3

### Presentation
1

### Contribution
2

### Rating
4

### Confidence
2