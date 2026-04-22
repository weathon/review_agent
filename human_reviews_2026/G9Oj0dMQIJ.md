# Training-free AI-generated Image Detection via Spectral Artifacts

- Avg Score: 3.33
- Decision: Reject
- Scores: 4, 2, 4

## Abstract
The rapid progress of generative models has enabled the synthesis of photorealistic images that are often indistinguishable from real photographs, raising serious concerns about misinformation and malicious use. While most existing AI-generated image (AIGI) detection methods rely on supervised training with labeled synthetic data, they struggle to generalize to unseen generators and incur substantial overhead for retraining. In this work, we propose SpAN, a simple yet effective training-free detection framework based on spectral analysis. Our key observation is that upsampling operations in generative models inevitably introduce spectral artifacts, which remain most pronounced at the axial Nyquist frequencies, even when images appear realistic. Building on this insight, we design two techniques to enhance detection reliability: (1) power calibration via azimuthal integration to mitigate bias from image-specific frequency distributions, and (2) autoencoder-based reconstruction to amplify residual artifacts and enable discrepancy-based scoring between original and reconstructed images. Extensive experiments across multiple datasets and generative models demonstrate that SpAN achieves robust and generalizable detection performance. For example, SpAN outperforms other training-free detection methods by a substantial margin (+0.241 AUROC) in the Synthbuster benchmark, which contains recent generative models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes SpAN, which is a training-free detector for AI-generated images based on spectral analysis of artifacts. The method measures the axial Nyquist power in an image and calibrates it by comparing it to the power through azimuthal integration. The image is passed through a pretrained autoencoder  and the calibrated power of the original and reconstructed image is compared. The paper showed experiments on two benchmarks and claimed that SpAN achieves higher performance compared to existing methods.

### Strengths
1. The paper introduces a training-free detection method, which is also easily adaptable and efficient to deploy for real-world scenarios.
2. The method utilizes the discrepancy between the original image and a generated image through an autoencoder, to assess the spectral characteristics relative to the original image. This is a dynamic setup, which should help in the generalization of the method.

### Weaknesses
1. The paper's methodology stems from the fact that generative models exhibit spectral artifacts, which arise from transposed convolutions of stride 2. However, there are generative models that don't utilize transposed convolutions. There is no ablation study regarding this. If this method only works for models that have transposed convolutions, then this will be a severe limitation of the methodology. 
2. Figure 3 shows that there is indeed separability between generated and original images. However there is overlap in the distributions too, making this method sensitive to specific images. I would be interested to know how this method performs when a real and generated image have very similar visuals and features. Furthermore, if an image is generated using inpainting, how would the performance be effected? I would imagine this method will be sensitive to those images. 
3. The evaluation shows performance in two benchmarks Synthbuster and GenImage.  The method shows good performance in Synthbuster, but is rarely the best model in GenImage. The paper didn't try to explore this further, and there is no mention why the performance might have degraded. The authors should benchmark on more datasets (for e.g. [1]) to show proper performance comparison. 
4. Due to the nature of the method, I would also be very interested in how often the model misclassifies real images. I would suggest to include ablation studies by taking real images from ImageNet, COCO, etc, and show the misclassification scores. 

[1] Rahman, Md Awsafur, et al. "Artifact: A large-scale dataset with artificial and factual images for generalizable and robust synthetic image detection." 2023 IEEE International Conference on Image Processing (ICIP). IEEE, 2023.

### Questions
1. How would the method perform on images from a generator that does not use explicit transposed convolution layers? Is there any experiment the authors have done regarding this? 
2. Did the authors do experiments that show methodology performance when generated images are very similar to real images, or which is generated through inpainting? 
3. Do the authors show performance on extended benchmarks to show robustness of their methodology?
4. Can the authors show performance specific to real images?

### Soundness
2

### Presentation
2

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
The paper proposes SpAN, a training-free AIGI detection method based on spectral analysis. It observes that upsampling in generative models (e.g., transposed convolutions) introduces persistent spectral artifacts at axial Nyquist frequencies, even in photorealistic images. SpAN detects these via:

- Power calibration using azimuthal integration of high-frequency power to mitigate content bias,

- Autoencoder reconstruction to amplify residual artifacts and compute discrepancy in calibrated power between original and reconstructed images.

Experiments on Synthbuster and GenImage show SpAN significantly outperforms other training-free methods.

### Strengths
- SpAN requires no labeled data, no retraining, and outperforms other training-free baselines across proprietary and open-source models.

- The method is straightforward, making it easy to reproduce.

### Weaknesses
- The authors claim to be the first to directly leverage spectral-domain information in the Fourier space as a metric for training-free AIGI detection. However, utilizing frequency information to detect generated images has been extensively studied[1, 2, 3]. I believe these works need to be highlighted in the paper, emphasizing the differences between the proposed method and those methods.

- The empirical evidence presented (e.g., Figure 3(b)) suggests that the proposed spectral features, in isolation, possess limited discriminative power. This finding raises concerns about the centrality of the paper's claimed contribution, as the method's overall efficacy appears to be heavily reliant on the subsequent autoencoder component. The role of autoencoder requires further clarification. Furthermore, using an autoencoder incurs significant additional computational overhead. 

- The paper's empirical validation is notably constrained. The omission of several key baseline detectors [1, 3, 4, 5] and the exclusion of diverse, large-scale datasets (e.g., DiffusionForensis, AIGCDetectBenchmark, DRCT2M, Chameleon) preclude a comprehensive comparative analysis, making it difficult to robustly assess the proposed method's performance and generalizability against the current state-of-the-art.

- The omission of an appendix is a notable concern. This absence prevents the authors from providing crucial supplementary information, such as experimental details.

Reference:

[1]: Thinking in frequency: Face forgery detection by mining frequency-aware clues

[2]: Frequency-aware deepfake detection: Improving generalizability through frequency space domain learning

[3]: Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection

[4]: Towards Universal Fake Image Detectors that Generalize Across Generative Models

[5]:  Manifold induced biases for zero-shot and few-shot detection of generated images

### Questions
Please see weakness.

### Soundness
2

### Presentation
1

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
This paper presents a training-free AI generated image detector, which is based on frequency analysis. To achieve robust detection with the training-free model, this paper presents power calibration and applies auto-encoder to reconstruct a testing image which reveals different distributions on the spectral artifacts. To show the effectiveness of the proposed method, this paper conducts experiments on Synthbuster and GenImage datasets. Overall, I think this paper is not novel enough to publish in ICLR. As well known, the spectrum artifacts are explored a lot and many variations are made on the frequency analysis. Furthermore, as mentioned in this paper, the key observation is  upsampling operations in generative models inevitably introduce spectral artifacts, which remain most pronounced in the frequency domain. This is revealed by many previous work, in particular in GAN-based image generation. The diffusion model adds random noises on the image, and then the up-convolution issue is not clear. I vote for marginally below the acceptance threshold for this paper.

### Strengths
1. This paper proposed a simple method for AIGI detection. Overall, this paper is easy to follow. Ablation study is also provided.

2. To improve the robustness of the pipeline, a reconstruction process is applied. The proposed method shows better results than RIGID, MINDER, AEROBLADE, and Manifold Bias.

### Weaknesses
1. The idea is not novel enough. As well known, the spectrum artifacts are explored a lot and many variations are made on the frequency analysis. Furthermore, as mentioned in this paper, the key observation is  upsampling operations in generative models inevitably introduce spectral artifacts, which remain most pronounced in the frequency domain. This is revealed by many previous work, in particular in GAN-based image generation. 

2. Cross-domain detection results will be helpful to improve this paper. As this paper mentioned that generalization is a problem for the training-based method, this paper needs to show more studies on cross-domain detection. As the different images are used to train the auto-encoder, the model still performs differently.

### Questions
Distribution gap is commonly existing in frequency domain, for example DCT, FFT domains. Why is Nyquist frequency applied in this work? What are the advantages of Nyquist frequencies compared with others?

### Soundness
2

### Presentation
3

### Contribution
2
