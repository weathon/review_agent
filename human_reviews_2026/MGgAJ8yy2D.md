# Self-Guided Low Light Object Detection Framework

- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Object detection in low-light environments is inherently challenging due to limited contrast and heavy noise, both of which significantly degrade feature representations. In this paper, we propose a novel self-guided low-light object detection framework that effectively addresses these issues without introducing additional parameters or increasing inference time. Our method incorporates a detachable auxiliary pipeline during training, consisting of an image enhancement module and a denoising module, followed by a Fourier-domain fusion block. This pipeline improves the feature representation of the detector's backbone, enhancing its robustness under low-light conditions. Importantly, at inference time, our method incurs no additional computational cost compared to the baseline detector while achieving substantial performance improvements. Extensive experiments on widely used low-light object detection benchmarks, such as DARK FACE and ExDark, demonstrate that our method achieves state-of-the-art performance. Notably, experiments on the nuImages dataset show that our approach can outperform domain adaptation methods—especially when a large domain gap between source and target domains is inevitable in the real-world applications—highlighting its practical effectiveness. Code is available at https://github.com/gw-shin/SGLDet.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a Self-Guided Low-light Object Detection Framework with a detachable auxiliary pipeline used only during training, which generates a high-quality supervisory target through an image enhancement module, a denoising module, and Fourier-domain fusion. Experimental results show that this method achieves state-of-the-art performance on multiple benchmarks, with a significant improvement on DARK FACE.

### Strengths
1. The designed auxiliary pipeline is activated only during training and is completely detached at inference time, which allows the model to achieve substantial performance gains while maintaining the exact same inference speed.
2. The method leverages the properties of the Fourier transform and self-supervised training strategy, which is novel and sound.
3. The writing is well-organized and easy to follow.
4. Experiments show significant improvement on multiple benchmarks.

### Weaknesses
1. The paper repeatedly emphasizes "zero inference overhead", but completely omits any discussion of the cost during training-stage. The introduction of the auxiliary pipeline ($\mathcal{E}, \mathcal{D}, \mathcal{G}$), retraining of $\mathcal{E}$ and $\mathcal{D}$ on the target dataset, and multiple Fourier transforms will clearly increase training complexity and time, but this trade-off is not quantified. 
2. The total loss function uses a hyperparameter $\lambda$ to balance the main and auxiliary pipeline, but the paper lacks a sensitivity analysis for this crucial hyperparameter.

### Questions
In Section 3.3, the authors chose the serial strategy $x^{\mathcal{E}+\mathcal{D}} = \mathcal{D}(x^{\mathcal{E}})$. Why is the current serial strategy the superior choice? Why not choose a parallel fusion strategy (e.g., fusing $\mathcal{P}(\mathcal{E}(x))$ and $\mathcal{A}(\mathcal{D}(x))$, where $\mathcal{D}$ operates on the original input $x$)?

### Soundness
4

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
This paper proposes a self-guided low-light object detection framework to improve detection performance under challenging lighting. The method utilizes image enhancement and denoising, and then the outputs are fused in the Fourier domain. The fusion is used to generate a dense pixel-wise supervision signal encouraging the detector's backbone to learn more robust low-light representations. Experiments are performed on DARK FACE, ExDark, and nuImages datasets, along with ablation and qualitative analyses.

### Strengths
1. Fourier-domain fusion: The method’s technical core by combining the amplitude from a denoised image with the phase from an enhanced image using FFT/iFFT is grounded in signal processing principles and attempts to both preserve structure and suppress noise. 
2. Extensive evaluation: The experiments cover multiple datasets (DARK FACE, ExDark, nuImages) and detectors. Ablation studies isolate the contributions of each module, and Figure 4 offers qualitative insights.
3. No inference overhead: The method is attractive for applications as it adds no complexity or latency at inference.

### Weaknesses
1. While the use of Fourier fusion is motivated by separation of amplitude and phase, the theoretical justifications for this separation, particularly in the context of low-light image statistics and deep feature learning, are not well explained and discussed.
2. The mathematical details for the Fourier fusion are sometimes ambiguous, e.g., the precise computation of bi-level amplitude-phase combination per-channel, whether normalization occurs before/after fusion, whether channel alignment causes artifacts.
3. The impact of severe boundary artifacts, non-Gaussian noise, or extreme illumination imbalance on the auxiliary fused target and final detector features is not explored.

### Questions
1. Did you analyze how signal-dependent noise, common in low-light shots, is distributed between the amplitude and phase components?
2. Did you investigate whether the fusion process introduces frequency-domain artifacts, and if so, how were they handled?
3. How does the proposed method perform when significant boundary artifacts are present in the input images?

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
4

### Summary
The paper introduces a novel framework to improve object detection in low-light conditions. The core idea is to use a detachable auxiliary pipeline during the training phase to provide a self-guided supervisory signal to the main detector's backbone. This pipeline, which is removed at inference time, consists of self-supervised image enhancement and denoising modules whose outputs are combined in the Fourier domain to create a high-quality target image. The backbone is then trained on a multi-task loss, combining the detection loss with a reconstruction loss based on this generated target. The authors claim this approach improves feature representation for low-light scenes, achieving state-of-the-art results on several benchmarks (DARK FACE, ExDark, nuImages) without adding any computational cost during inference.

### Strengths
+ The manuscript is well-represented and easy-to-follow.

+ The auxiliary pipeline (enhancer, denoiser, fusion) is completely detached after training. This means the detector is identical in architecture, parameters, and speed to the baseline model, yet it performs significantly better.

+ The effectiveness of the framework is validated across three different datasets and with multiple detector architectures.

### Weaknesses
+ The paper claims that performance stems from the framework design itself. However, in Table 5, using simple modules (Gamma Correction, Gaussian Blur) yields a 70.9 mAP, while advanced modules (SCI, SDAP) are required to reach the top performance of 76.6 mAP. This indicates the framework's effectiveness is tied to the quality of the chosen enhancer and denoiser. It would be better to see more variant of enhancer/denoiser pairs and more discussion for this.

+ The model is trained with a multi-task loss combining sparse detection (L_det) and dense pixel-reconstruction (L_self). How is the sensitivity to the weighting hyperparameter $\lambda$?

+ Figure 4, Retienxformer -> Retinexformer. The authors should carefully check the manuscript to eliminate typos and grammarical errors.

### Questions
Please refer to weakness part.

### Soundness
3

### Presentation
3

### Contribution
3
