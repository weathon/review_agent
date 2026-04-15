# Med-Tuning: Parameter-Efficient Transfer Learning with Fine-Grained Feature Enhancement for Medical Volumetric Segmentation

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 6, 5

## Abstract
Deep learning-based medical volumetric segmentation methods either train the model from scratch or follow the standard ``pre-training then finetuning" paradigm. Although finetuning a pre-trained model on downstream tasks can harness its representation power, the standard full finetuning is costly in terms of computation and memory footprint. In this paper, we present the study on parameter-efficient transfer learning for medical volumetric segmentation and propose a new framework named Med-Tuning based on intra-stage feature enhancement and inter-stage feature interaction. Additionally, aiming at exploiting the intrinsic global properties of Fourier Transform for parameter-efficient transfer learning, a new adapter block namely Med-Adapter with a well-designed Fourier Transform branch is proposed for effectively and efficiently modeling the crucial global context for medical volumetric segmentation. Given a large-scale pre-trained model on 2D natural images, our method can exploit both the crucial spatial multi-scale feature and temporal correlations along slices for accurate segmentation. Extensive experiments on three benchmark datasets (including CT and MRI) show that our method can achieve better results than previous parameter-efficient transfer learning methods for segmentation task, with much less tuned parameter costs. Compared to full finetuning, our method reduces the finetuned model parameters by up to 4x, with even better segmentation performance.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes a new technique for adapting pre-trained weights from (among others) vision transformer architectures that have been initially trained on natural images to 3D medical image segmentation.

### Strengths
The idea in principal is interesting. The method explores a different approach to conventional fine-tuning by inserting specially designed adaptation layers. 
The approach in particular deals with the challenge of moving from 2D pre-trained weights to 3D problems and also discusses the additional gap that stems from networks weights that were used for global classification to pixel-wise segmentation.
The Fourier transform based module seems especially efficient and useful for larger context.

### Weaknesses
The experimental validation is not appropriate because it performs insufficient comparison to SOTA. In particular the “full” fine-tuning scheme (or model trained from scratch) are not at all 3D aware but have to solely rely on 2D operations. The proposed approach, however, gains access to inter-slice dependencies by inserting additional convolution / transformer modules that act on the depth / third dimensions.
This leaves a wrong impression about the capabilities of the proposed model as e.g. for KiTS 19 the authors claim SOTA performance, but a simple 3D nnUNet (trained from scratch) reaches a far superior Dice score of 91.2% (composite), 97.4% (Kidney) and 85.1% (Tumour)
The results for the Swin-UNet are even worse which highlights the weakness of 2D models for the tasks.
While the ablation on data efficiency is interesting, it appears that most models are trained for a surprisingly short time 0.17-1.34hours only. This indicates that the full model could not have converged, making the comparison rather unfair.

### Questions
Explain the reasoning for omitting 3D nnUNet results and short training times.
Add further ablations that demonstrate the motivation behind the Fourier module and the multi-scale concatenation of adaptation layers.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies how to adapt 2D pre-trained models on the 3D segmentation problem, particularly focusing on the medical domain. Authors claimed they intend to tackle the challenge of modality gap and task gap when using 2D models pre-trained over natural images. Authors proposed Med-Tuning accordingly. The core techniques include decomposed convolutional layers, FFT/IFFT transformation and inter-layer feature interaction. The framework was evaluated on CT and MRI datasets including BraTS 2019, BraTS 2020 and KiTS 2019, and shown to outperform several baseline methods with fewer tuned parameters. Multiple architectures and pre-trained models were involved.

### Strengths
1. Adapting 2D pre-trained models on 3D medical tasks is an important and interesting task worth studying. 
2. The overall framework is technical sound and shows good performance. 
3. Experiments are exhaustive with multiple SOTA architectures and pre-trained datasets.

### Weaknesses
1. The actual contribution of this paper is not consistent with the claim of 2D->3D adaptation. As described in the section 3.3, this paper used simple reshaping to adapt 2D pre-trained models on 3D data. This is also a weak part in the proposed framework as the solution did not consider the interaction between adjacent slices. There is already existing work investigating effective 2D->3D adaptation like [1]. 
2. Lack of technical novelty. Using the FFT module to integrate global information is not a novel idea. It has been proposed in [2]. As analyzed in ablation studies, the FFT/IFFT module is an essential component of the framework. It’s not surprising that using a powerful module as the adapter results in decent performance.  
3. The review of related work should be improved.

[1] Adapting Pre-trained Vision Transformers from 2D to 3D through Weight Inflation Improves Medical Image Segmentation.  Machine Learning for Health (ML4H) 2022
[2] Global Filter Networks for Image Classification. NeurIPS 2021.

### Questions
1. What does concentration mean in Figure 3? It is not mentioned in the figure caption or main text. 
2. How do you choose the value of $m_n+1$ and why?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this work, the authors propose a new framework called Med-Adapter for PETL (Parameter Efficient Transfer Learning). They aim to use the easily available pretrained 2D Transformer models for the 3D segmentation task. They do this via fine-tuning of adapter blocks that they introduce into the Transformer architecture. The authors aim to bridge two gaps in fine-tuning: 1) the task gap that existing pretrained models are trained for classification but we would like to fine-tune them for segmentation; and 2) the modality gap that existing pretrained models are trained on 2D images while 3D medical volumes also have a temporal (depth) component.
By conducting experiments on 3 datasets and 3 backbone architectures, the authors demonstrate that their fine-tuning method can achieve good segmentation performance in a parameter-efficient manner.

### Strengths
1) The authors solve the task gap (classification/segmentation) by introducing FFT and IFFT which are lightweight and perform similarly to attention mechanism.
2) Since they deal with 3D data, they perform appropriate reshaping in order to use the existing 2D transformer blocks. In the Med-Adapter blocks that they introduce, they perform 3D convolution but use the parameter-efficient versions such as depth convolutions and 1 x K x K etc as approximations.
3) The authors do extensive experiments to validate their method. They experiment on 3 datasets (BraTS 2019, 2020 and KiTS 2019) on 3 backbones (ViT + UPerNet and Swin-T) and compare against several PETL baselines. Their method consistently achieves better performance against the baselines with much lesser trainable parameters.
4) I appreciate the authors including experiments on several pretrained weights such as CLIP and the recent SAM.
5) The authors perform appropriate ablation studies -- on the different branches in IntraFE; different ways to fuse features in InterFI; ratio used in reshaping; decoder architectures; and reduced data setting.

### Weaknesses
1) Could the authors discuss on the applicability of their method on non-Transformer architectures such as large pretrained VGGs, ResNet50, InceptionV3 and so on? Currently the proposed method is only validated for Transformer architectures.
2) While the proposed method achieves better performance compared to baselines, the novelty of the method seems limited. This is because depth-wise convolutions as well as 1 x K x K convolutions to approximate 3D convolutions already exist in literature as ways to reduce parameter count. Furthermore using FFT and IFFT for attention and reduced parameters also exists in literature [1,2,3]. Hence the proposed method seems like an amalgamation of several existing concepts.

**References**

[1] Chi, Lu, Borui Jiang, and Yadong Mu. "Fast fourier convolution." Advances in Neural Information Processing Systems 33 (2020): 4479-4488.

[2] Yang, Yanchao, and Stefano Soatto. "Fda: Fourier domain adaptation for semantic segmentation." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.

[3] L. Bai, X. Lin, Z. Ye, D. Xue, C. Yao and M. Hui, "MsanlfNet: Semantic Segmentation Network With Multiscale Attention and Nonlocal Filters for High-Resolution Remote Sensing Images," in IEEE Geoscience and Remote Sensing Letters, vol. 19, pp. 1-5, 2022, Art no. 6512405, doi: 10.1109/LGRS.2022.3185641.

### Questions
1) Please also see the weakness above.
2) The final row in Table 3 corresponds to which method --- Table 1 or Table 9? Specifically, I cannot see in Table 1 or Table 9 where WT has a Dice value 90.05. I believe the numbers are supposed to be consistent. 
3) As standard deviation values have not been provided, please mention if the results in bold are just numerically better, or, if t-test [4] has been conducted to check if the performance improvement is statistically significant or not.

**References**

[4] Student, 1908. The probable error of a mean. Biometrika, pp.1–25.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
With the recent success of foundation models, there is an increasing interest in fine-tuning these models for various problems. So, how to do the fine-tuning is a valid question, and updating as few parameters as possible during fine-tuning is vital for limited labeled data regimes and computation costs. With this motivation, the paper proposes a plug-and-play block for parameter-efficient fine-tuning of neural networks. The proposed block can be added to any architecture and only the block's parameters (along with the decoder added for segmentation) are updated during fine-tuning. The block consists of inner blocks: intra-feature enhancement (IntraFE) and inter-feature interaction (InterFE). In IntraFE, local features are extracted using channel-wise separable convolutions while FFT is used for global features. The results of each branch in IntraFE are summed and passed through a 1x1x1 convolution. The output of IntraFE is given to the InterFE block where it is concatenated with the features from the previous stage.

The experiments are conducted in 3 different datasets for segmentation, and the results show slight improvement compared to the state-of-the-art in terms of Dice and Haussdorff metrics, with a better accuracy-efficiency tradeoff.

### Strengths
- The paper presents extensive experiments on three different medical image segmentation datasets. Additionally, it presents a quite extensive ablation study to show the contribution of each component of the proposed block.

- The accuracy/efficiency trade-off is improved significantly compared to the state-of-the-art methods.

### Weaknesses
- The experiments are performed with k-fold cross validation; however, only the average Dice score/Hausdorff distance results are presented in the tables. Especially, given that the quantitative results are very close to some of the existing methods, standard deviation results (or statistical significance test) would be helpful to interpret the results better. 

- The contribution of the branches in Intra-FE block is not very significant, according to Table 3. There is a very small difference between the first row and the third row. Standard deviation results are also required for this experiment. The most significant contribution appears to be achieved by the CM according to the table. I would expect to see Conv3-Conv5 with CM to understand the contribution of the FFT block better. If the results are not different that Conv3-Conv5-FFT-CM, then we can conclude that the FFT branch doesn't have many contributions. It is crucial to see this since FFT is proposed as one of the main contribution of the block.

- The contribution of the Inter-FE block is not significant, according to Table 4. Standard deviation results should also be added for this experiment.

- I also have concerns about the novelty of the proposed block. Channel-wise separable convolutions have already been widely used for parameter-efficient training, e.g. [1]. Also, given that the contributions of the FFT branch and InterFI block are very small, it seems that the main improvement of the accuracy comes from the channel-wise separable convolutions, which is already an existing technique. 

[1] Chollet, Xception: Deep Learning with Depthwise Separable Convolutions - https://arxiv.org/abs/1610.02357

- In the current experiments, the proposed block is added after each block. How would the results change if it is used e.g. after only a few earlier blocks?

### Questions
Please address my comment in the weaknesses section.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
