# Binary Neural Network for Hyperspectral pansharpening

- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
Hyperspectral pan-sharpening aims to generate high-resolution hyperspectral (HRHS) images by fusing low-resolution hyperspectral (LRHS) data with high-resolution panchromatic (PAN) images, enabling applications in mapping, surveillance, and environmental monitoring. While deep learning methods achieve strong performance, their heavy computational and memory demands limit deployment on resource-constrained satellite platforms. To address this, we explore binary neural networks (BNNs) for hyperspectral pan-sharpening. Conventional binarization, however, introduces gradient instability and severe information loss, compromising spectral–spatial fidelity. We propose the Adaptive Tan Identity Straight-Through Estimator (ATISTE), a soft binarization strategy that decouples forward approximation from gradient propagation and employs adaptive scaling to preserve consistency with full-precision features. Building on ATISTE, we design HS-BiNet, a lightweight binary CNN with residual connections and multi-scale fusion to effectively capture spectral–spatial dependencies, while avoiding computationally intensive operations such as unfolding inference and non-local self-attention, thereby ensuring suitability for real-time deployment on edge and satellite platforms. Extensive experiments show that HS-BiNet consistently outperforms binary baselines and remains competitive with, and in some cases surpasses, full-precision models, offering a practical solution for high-fidelity HRHS reconstruction.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents the first exploration of Binary Neural Networks for hyperspectral pan-sharpening, a task that fuses low-resolution hyperspectral images with high-resolution panchromatic images to produce high-resolution hyperspectral outputs. To address the gradient instability and information loss caused by conventional binarization, the authors propose a novel Adaptive Tanh-Identity Straight-Through Estimator . ATISTE decouples forward approximation from gradient propagation via a dual-path design and introduces adaptive channel-wise scaling to maintain fidelity with full-precision features. Building on ATISTE, they design HS-BiNet, a lightweight binary CNN with residual connections and multi-scale fusion modules, avoiding expensive operations like non-local attention or unfolding inference. Extensive experiments on both reduced- and full-resolution datasets show that HS-BiNet outperforms all existing binary methods.

### Strengths
1.First BNN application to hyperspectral fusion; ATISTE is a novel STE variant.
2.Theoretical analysis and experimental validation are both strong.

### Weaknesses
1.The figures and presentation are not clear enough. （The module names in Figure 3 could be arranged horizontally for better readability； Although the text mentions an "encoder," it is not clearly visible in Figure 3；The authors should explicitly indicate which convolutions are standard and which are binary within the model；For the core theoretical contribution, should the authors briefly review existing binary networks and highlight their limitations to clarify the novelty?）
2.No ablation on HS-BiNet components (e.g., Edge Injector, Multi-scale Extractor, Decoder Enhancement).
3.Only compares efficiency among binary models; lacks direct comparison with full-precision ones.
4.No mention of code or model release.

### Questions
see the weakness

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a binary neural network (BNN)-based framework for hyperspectral image (HSI) fusion, aiming to reconstruct high-resolution hyperspectral images (HRHS) from low-resolution hyperspectral inputs (LRHS) and high-resolution panchromatic images (PAN). The goal is to achieve efficient and low-cost hyperspectral pan-sharpening via network binarization, facilitating real-time deployment on satellites or edge devices.

### Strengths
Hyperspectral pan-sharpening is a valuable topic in remote sensing data processing and compression modeling, especially under energy-constrained conditions. Designing lightweight models for on-orbit or edge deployment is meaningful and timely.

Traditional binarization often leads to severe information loss and degradation in spectral-spatial fidelity. Applying binary neural networks to hyperspectral reconstruction is relatively rare and constitutes a novel direction worth exploring.

The proposed Adaptive Two-Path Straight-Through Estimator (ATISTE) decouples forward approximation and gradient propagation through adaptive scaling, addressing a key limitation in BNN training.

The HS-BiNet integrates multi-scale feature extraction, residual connections, and local fusion blocks to efficiently model spectral–spatial dependencies.

### Weaknesses
The paper claims that the proposed binary network is computationally efficient but provides no quantitative evidence. A comparison of FLOPs, parameter count, and inference time with other BNN baselines is necessary to substantiate this claim.

Figures 4 (Botswana dataset) and 5 (WDC dataset) appear visually identical, even though the two datasets originate from different sensors and regions. This strongly suggests an error in visualization or dataset usage, which seriously undermines the credibility of the experimental results.

The paper lists PSNR, CC, SSIM, SAM, and ERGAS as evaluation metrics but does not explain their physical meanings or relevance to hyperspectral fusion, affecting readability and scientific clarity.

### Questions
Are the qualitative visualizations in Figures 4 and 5 actually repeated? Please confirm whether the same results were mistakenly reused.

Please include additional visual comparisons of fusion results from different methods on the reduced-resolution WDC and Botswana datasets. Add comparisons of inference time and model parameters to demonstrate the claimed efficiency of your method. In Table 1, highlight your proposed model (HS-BiNet) in bold for better clarity.

Hyperspectral fusion aims to obtain both high spectral and spatial resolution. However, existing hyperspectral interpretation tasks (e.g., anomaly detection) often use high-resolution HSI directly. Could fusion-based HRHS images improve downstream analysis performance? For instance, in hyperspectral anomaly detection, would the fused HRHS input lead to sharper or more precise anomaly boundaries?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper applies exploration of binary neural networks for hyperspectral pan-sharpening and introduces a lightweight architecture driven by the proposed Adaptive Tan Identity Straight-Through Estimator (ATISTE).

### Strengths
This paper focuses on improving current hyperspectral pansharpening models, which rely on so heavy computational and memory demands that limit deployment on resource-constrained satellite platforms. This task is important in real scenarios.

### Weaknesses
1. The novelty of this paper is limited since it just applies the binary neural network to the hyperspectral pansharpening task. Although this paper claims to make specific modifications to the current BNN, the contribution is insufficient for a top conference. This work is more suitable for a journal.
2. The motivation of this work is also not solid since many unsupervised model-based pansharpening methods just need to implement an iterative algorithm, which is not computationally expensive.
3. The experiments are not sufficient since many unsupervised model-based hyperspectral pansharpening methods are not compared. Additionally, the experimental datasets in Table 1 and Table are not consistent, which makes it diffucult to evaluate the performance of this work.

### Questions
Please refer to the weaknesses.

### Soundness
2

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
5

### Summary
This paper is the first to apply binary neural networks to hyperspectral image fusion tasks, proposing a lightweight binary convolutional neural network architecture, HS-BiNet, and introducing a novel gradient estimator, ATISTE, to address the forward approximation error and gradient instability problems during the binarization process. The authors conducted extensive experiments on hyperspectral image fusion tasks, demonstrating that the proposed method, while maintaining a lightweight model, significantly outperforms existing binary models and even surpasses full-precision models in some metrics.

### Strengths
1. Binary neural networks are an extreme form of model compression, but they perform poorly in high-precision regression tasks (such as super-resolution and fusion). This paper proposes a targeted solution to this problem.

2. ATISTE proposes a dual-path gradient estimation strategy, decoupling forward approximation from gradient propagation.  It uses adjustable parameters $α$ and $λ$ to control approximation accuracy and gradient stability respectively, demonstrating theoretical soundness and flexibility. HS-BiNet combines multi-scale feature extraction, residual connections, and edge injection structures, adapting to the characteristics of hyperspectral data while avoiding computationally intensive operations (such as Transformers and non-local attention).

3. Systematic evaluations were conducted on multiple standard datasets (WDC, Botswana, FRI), covering both downsampled and full-resolution scenarios. Compared with numerous full-precision and binary baselines, the results show that HS-BiNet significantly outperforms existing binary methods in key metrics such as PSNR, SAM, and QNR, and even surpasses some full-precision methods. Computational efficiency analysis is provided, demonstrating the model's advantages in terms of parameter count, FLOPs, memory usage, and inference time.

### Weaknesses
1. Although ATISTE is theoretically proven to be "rational," its theoretical support, such as convergence analysis and generalization error bounds, is still insufficient. The theoretical comparison with recent STE improvement works such as ReSTE is not in-depth enough.

2. Training strategies (e.g., learning rate scheduling, data augmentation) are not described in detail in the main text, only briefly mentioned in the appendix. It is not stated whether pre-trained weights were used or how the model was initialized, which significantly affects the training stability of binary networks.

4. Binary networks still suffer from information bottlenecks under extreme compression, especially in the recovery of high-frequency details. This paper does not adequately discuss the failure cases of HS-BiNet in extreme scenarios. The performance of the binary model in terms of noise robustness and cross-sensor generalization is not analyzed.

### Questions
1. Why is it that in Table 1, only one method combining zero-shot learning and diffusion models performs well among all the full-precision methods? Theoretically, zero-shot learning is primarily used for unsupervised learning and should perform better on full-resolution tests, so it's rather strange that the PSNR and other downsampling test metrics in Table 1 show better performance.
2. In the ablation experiments concerning computational efficiency, what are the specific differences between SignSTE and HybridSTE, and how do they differ from the standard BNN framework? Furthermore, how do they compare in terms of computational efficiency to traditional convolutional or attention operators?

### Soundness
3

### Presentation
3

### Contribution
4
