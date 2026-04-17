# VAE with Hyperspherical Coordinates: Improving Anomaly Detection from Hypervolume-Compressed Latent Space

- Decision: Reject
- Scores: 2, 4, 4

## Abstract
Variational autoencoders (VAE) encode data into lower-dimensional latent vectors before decoding those vectors back to data. Once trained, one can hope to detect out-of-distribution (abnormal) latent vectors, but several issues arise when the latent space is high dimensional. This includes an exponential growth of the hypervolume with the dimension, which severely affects the generative capacity of the VAE. In this paper, we draw insights from high dimensional statistics: in these regimes, the latent vectors of a standard VAE are distributed on the `equators' of a hypersphere, challenging the detection of anomalies. We propose to formulate the latent variables of a VAE using hyperspherical coordinates, which allows compressing the latent vectors towards a given direction on the hypersphere, thereby allowing for a more expressive approximate posterior. We show that this improves both the fully unsupervised and semi-supervised anomaly detection ability of the VAE, achieving the best performance on the datasets we considered, outperforming existing methods. For the unsupervised and semi-supervised modalities, respectively, these are: i) detecting unusual landscape from the Mars Rover camera and unusual Galaxies from ground based imagery (complex, real world datasets); ii) standard benchmarks like Cifar10 and subsets of ImageNet as the in-distribution (ID) class.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a method to compress the latent space of VAE for learning more distingushiable representations for AD tasks. Specially, the authors first convert the encoder output from Cartesian coordinates into hyperspherical coordinates,
then the prior is Specially designed so that samples can be moved to a zone with reduced volume. Experiments conducted on several image datasets illustrate the effectiveness of the proposed method.

### Strengths
- The experiments are conducted under both unsupervised and semi-supervised settings.
- Overall the paper is well-structured and easy to follow, except several parts of 2.
- The proposed method is well-motivated.

### Weaknesses
- Questionable novelty. The core technique appears highly similar to the method in [1], and the paper does not clearly delineate what is new. 
- Incomplete related work. Prior studies on concentration-of-measure phenomena and compression (e.g., [2]) are not discussed, nor is [2] used as a baseline.
- Weak baselines. In the fully unsupervised setting, only several AE/VAE-based baselines are considered. Many recent unsupervised AD methods are missing (e.g., [3]–[5]).
- Missing large-scale benchmark. Evaluation on a widely used benchmark such as ADBench would better substantiate the method’s effectiveness and generality.
- Conceptual conflation. The paper treats OOD detection as semi-supervised AD. These are distinct problem formulations; see [6] for definitions of weakly/semi-supervised AD. Meanwhile, the proposed method is not effective in the OOD case. 
- The usage of dimensionality reduction methods plus kNN is not novel and result in high time complexity of $\mathcal{O}(n^2)$ for inference, suppose $n$ is the number of samples in training dataset. 

[1] Ascarate, Alejandro, et al. "Improving the Generation of VAEs with High Dimensional Latent Spaces by the use of Hyperspherical Coordinates." arXiv preprint arXiv:2507.15900 (2025).

[2] Zhang, Yunhe, et al. "Deep orthogonal hypersphere compression for anomaly detection." arXiv preprint arXiv:2302.06430 (2023).

[3] Livernoche, Victor, et al. "On diffusion modeling for anomaly detection." arXiv preprint arXiv:2305.18593 (2023).

[4] Yin, Jiaxin, et al. "MCM: Masked cell modeling for anomaly detection in tabular data." The Twelfth International Conference on Learning Representations. 2024.

[5] Thimonier, Hugo, et al. "Beyond individual input for deep anomaly detection on tabular data." arXiv preprint arXiv:2305.15121 (2023).

[6] Durani, Walid, et al. "Weakly Supervised Anomaly Detection via Dual-Tailed Kernel." Forty-second International Conference on Machine Learning.

### Questions
- See weaknesses.
- The derivation in this paper is not sufficiently clear. For example, how is equation (2) obtained from equation (1)? In equation (2), how are the coefficients α and 𝛽 introduced and defined?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes a modification of the standard Variational Autoencoder (VAE) by introducing hyperspherical coordinates in the latent space and having adapted priors to these coordinates instad of Gaussian priors in euklidean space. The authors argue that this change addresses issues related to high-dimensional Gaussian distributions, such as concentration of measure and the tendency of latent vectors to cluster in equatorial regions of the hypersphere. They apply their method to anomaly detection in both unsupervised and semi-supervised settings and evaluate it on multiple datasets.

### Strengths
1. The paper addresses an interesting conceptual idea: leveraging hyperspherical geometry to restructure the latent space of VAEs.
2. The proposed approach is technically novel in how it reformulates the latent distribution.
3. The experimental results look strong on first sight.
4. The appendix is really strong and includes a lot of intuitoin and background information (which would have helped in understanding the paper).

### Weaknesses
1. Lack of statistical rigor in evaluation:
The results section does not include any indication of how many runs were performed per method, nor are any measures of statistical variation (e.g., standard deviation, confidence intervals) reported. This is a major issue.
2. Unmotivated choice of high-dimensional latent space:
The paper uses high-dimensional latent vectors (e.g., 100–256 dimensions) but provides no strong empirical or theoretical justification for this choice. It is unclear whether lower-dimensional latent spaces perform worse or whether the method’s advantages only emerge in high dimensions.
3. Limited methodological accessibility:
While the paper touches on complex geometric ideas such as volume concentration on hyperspheres and almost-orthogonality, these are presented without sufficient intuition or formal support. Notation is moved to the appendix. Readers unfamiliar with high-dimensional geometry will find it difficult to follow key parts of the method. More explanatory figures, examples, or visualizations would have helped greatly. I also do not understand what $a_{i,j}$ and $b_{i,j}$ are from the short description (even though I can make a probably correct educated guess).
4. Unclear problem significance:
The paper assumes that measure concentration and “equator clustering” are problematic in standard VAEs for anomaly detection, but it does not provide convincing evidence that this is a real or significant problem in practice. No baseline analysis is presented to demonstrate that the equator effect harms anomaly detection performance. As such, the motivation for the proposed fix remains speculative.
5. Notational problems (sometimes due to notation only being explained in the appendix): E.g. in (2) on the left hand side, you have $\varphi_k$ for a fixed $k$ and on the right hand side $k$ is a summation index. Or why does setting a mean in (6) enforce a radius without fixing a standard deviation to zero? Also, the citations are formated wrongly in the paper text.

### Questions
Does your prior have any classical description? I can (I think) understand the formulas and I can mostly follow your argument for the prior, but I am lacking a clearer understanding in terms of math.

### Soundness
2

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
This paper proposes a hyperspherical-coordinate reformulation of the VAE to mitigate high-dimensional sparsity in latent space. By compressing latent vectors toward compact regions on the hypersphere, the method enhances anomaly separability. Experiments on both unsupervised and semi-supervised settings demonstrate consistent improvements in FPR95 and AUROC over standard VAEs and related baselines.

### Strengths
1.The paper presents a well-motivated and theoretically sound reformulation of the VAE framework by introducing hyperspherical coordinates, effectively addressing the limitations of latent space concentration in high-dimensional settings.
2.Extensive experiments on both real-world and benchmark datasets show consistent and interpretable improvements in anomaly detection performance.

### Weaknesses
1.The writing and overall organization of the paper could be improved for clarity and coherence, figures and methodological explanations could be better integrated and more consistently formatted.
2.The paper would benefit from more comprehensive ablation studies to isolate the contribution of each design choice on anomaly detection performance.
3.As shown in Table 2, the proposed method does not reach state-of-the-art performance on several far-OOD datasets, suggesting that further validation or complementary analyses are needed to more convincingly demonstrate the effectiveness and generalizability of the approach.

### Questions
1.The paper would benefit from a more detailed complexity analysis, as the transformation from Cartesian to hyperspherical coordinates introduces additional computational steps and parameters.

### Soundness
2

### Presentation
2

### Contribution
2
