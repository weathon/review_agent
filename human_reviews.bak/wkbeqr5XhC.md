# LUM-ViT: Learnable Under-sampling Mask Vision Transformer for Bandwidth Limited Optical Signal Acquisition

- Decision: Accept (poster)
- Scores: 6, 6, 6

## Abstract
Bandwidth constraints during signal acquisition frequently impede real-time detection applications. Hyperspectral data is a notable example, whose vast volume compromises real-time hyperspectral detection. To tackle this hurdle, we introduce a novel approach leveraging pre-acquisition modulation to reduce the acquisition volume. This modulation process is governed by a deep learning model, utilizing prior information. Central to our approach is LUM-ViT, a Vision Transformer variant. Uniquely, LUM-ViT incorporates a learnable under-sampling mask tailored for pre-acquisition modulation. To further optimize for optical calculations, we propose a kernel-level weight binarization technique and a three-stage fine-tuning strategy. Our evaluations reveal that, by sampling a mere 10\% of the original image pixels, LUM-ViT maintains the accuracy loss within 1.8\% on the ImageNet classification task. The method sustains near-original accuracy when implemented on real-world optical hardware, demonstrating its practicality. Code will be available at [https://github.com/MaxLLF/LUM-ViT](https://github.com/MaxLLF/LUM-ViT).

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The proposed method of this paper is LUM-ViT, a learnable under-sampling mask vision transformer for bandwidth-limited optical signal acquisition. It is a novel approach that utilizes deep learning and prior information to reduce acquisition volume and optimize for optical calculations. The methodology unfolds in two primary stages: training from pre-trained models in a solely electronic domain using existing datasets, followed by inference to evaluate model performance, and assessing the real-world performance of LUM-ViT with a DMD signal acquisition system. During acquisition, target information undergoes a single instance of DMD optical modulation before capture, and then is funneled into the electronic system for further processing.

### Strengths
(i) The studied problem about under-sampling hyperspectral data acquisition, achieving data reduction from signal collection while preserving model performance. This accelerates the HSI processing in real applications such as remote sensing, object tracking, medical imaging, etc.

(ii) The idea of using a learnable mask refined during training to selectively retain essential points for downstream tasks from the patch embedding outputs, and thereby achieving under-sampling (reducing the required sampling instances) is interesting.

(iii) The performance is good. On the ImageNet-1k classification task, the proposed LUM-ViT maintains accuracy loss within 1.8% at 10% under-sampling and within 5.5% at an extreme 2% under-sampling.

(iv) This work not only conducts experiments on the synthetic data but also sets up hardware (as shown in Figure 6) to evaluate the effectiveness of the proposed method. It is a good and non-trivial exploration. The accuracy loss of LUM-ViT does not exceed 4% compared to the software environment, demonstrating its practical feasibility.

### Weaknesses
(i) The detailed formulations for DMD are missing, which is confusing. More explanations are required.

(ii) Code and pre-trained weights are not submitted. The reproducibility of this work cannot be checked. 

(iii) The multi-stage training pipeline is tedious, which makes the whole technical route unreliable. What's worse, the finetuning details about stage 3 are missing.  Other researchers cannot re-implement this complex approach. 

(iv) The writing should be further improved, especially the mathematical notations in Section 3.3. The formula for binary compression is not formal.

(v) The experiments are not sufficient and many critical comparisons are missing. For example, the backbone is ViT variants. However, the ViT is very computationally expensive because its computational complexity is quadratic to the input spatial size. This also embeds the real-time applications of HSI processing. In contrast, MST [1] or MST++ [2] are specially designed for HSI processing. They treat spectral feature maps as a token to capture the interdependencies between spectra with different wavelengths. Most importantly, they are very efficient with linear computational complexity regarding spatial resolution. Thus, it is better to add a comparison with the spectral Transformer in Figure 5.

(vi) The binarization mechanism is out of fashion. BiSCI [3] provides a specially designed binarized convolution unit BiSR-conv block to process HSI data cubes. It is also better to add a comparison with this new technique.

[1] Mask-guided Spectral-wise Transformer for Efficient Hyperspectral Image Reconstruction. In CVPR 2022.

[2] MST++: Multi-stage Spectral-wise Transformer for Efficient Spectral Reconstruction. In CVPRW 2022, NTIRE 2022 Winner in Spectral Recovery.

[3] Binarized Spectral Compressive Imaging. In NeurIPS 2023.

### Questions
The technical route and core idea in this paper is to accelerate the HSI processing, which is similar to Coded Aperture Spectral Snapshot Imaging (CASSI). Could you please analyze the differences, advantages, and disadvantages of these two systems?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a novel approach using pre-acquisition modulation with a deep learning model called LUM-ViT. Specifically, it utilizes ViT as the backbone network and a DMD signal acquisition system for patch-embedding. Moreover, a kernel-level weight binarization technique and a three-stage fine-tuning strategy is proposed for optimizing the optical calculations. With low sampling rates, LUM-ViT maintains high accuracy on ImageNet dataset.

### Strengths
1.	The idea is of the paper is interesting. The proposed method performs calculations of the patch-embedding layer instead of directly sampling the whole images. The proposed LUM-ViT is suited for both dataset and downstream tasks.
2.	The accuracy loss is low with extreme under-sampling

### Weaknesses
1.	The description of the entire system and methods is not clear and intuitive enough, and the figures are also misleading. The RGB image is used as an example in Figure 1, which does not reflect the characteristics of hyperspectral imaging. 
2.	Lack of experiments on real hyperspectral imaging. The author has emphasized hyperspectral imaging in the introduction section, but in reality, it has not been verified using real hyperspectral images. The performance of this method in real hyperspectral imaging tasks still needs to be discussed

### Questions
1.	The training phase uses images of 3 color channels while the real-world experiment uses 7 color images. What is the meaning of ‘reconfigured’? Or the author just fine tuned the LUM-VIT with 7 color samples? Can the pre trained model be used directly for images in different bands without the need for matching data?
2.	This acquisition system seems to have to work together with vit to obtain intermediate features of images. Can it reconstruct a complete hyperspectral image in a real-world environment?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a learnable under-sampling mask vision Transformer, which incorporates a learnable undersampling mask tailored for pre-acquisition modulation.

### Strengths
+ The paper is well-organized and clearly written.

+ The proposed three-stage training strategy for training LUM-ViT is effective.

### Weaknesses
- Technical details should be clear. How to achieve the learnable under-sampling mask? How is this learnable achieved? Is the learning accurate? Relevant visualization results should be provided.

- The experimental results seem insufficient. The author only conducted validation on the ImageNet-1k classification task, and other tasks should also be further explored.

-----------------------After Rebuttal---------------------------

Thank you for your feedback. The rebuttal addressed my concerns well. Considering other reviews, I have decided to increase my score.

### Questions
See the above Weaknesses part.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
