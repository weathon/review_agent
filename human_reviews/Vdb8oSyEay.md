# E$^{2}$GAN: Efficient Training of Efficient GANs for Image-to-Image Translation

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 5, 3

## Abstract
One highly promising direction for enabling flexible *real-time on-device* image editing is utilizing data distillation by leveraging large-scale text-to-image diffusion models, such as Stable Diffusion, to generate paired datasets used for training generative adversarial networks (GANs). 
This approach notably alleviates the stringent requirements typically imposed by high-end commercial GPUs for performing image editing with diffusion models. 
However, unlike text-to-image diffusion models, each distilled GAN is specialized for a specific image editing task, necessitating costly training efforts to obtain models for various concepts.
In this work, we introduce and address a novel research direction: *can the process of distilling GANs from diffusion models be made significantly more efficient?*
To achieve this goal, we propose a series of innovative techniques.
First, we develop an attention-based network architecture tailored for efficient image-to-image translation on mobile devices, which yields faster inference speeds, reduces the number of parameters, and lowers computational costs compared to existing image-to-image models.
Second, we introduce a hybrid training pipeline that efficiently adapts a pre-trained text-conditioned GAN model to different concepts while substantially reducing computational costs. 
Moreover, this approach significantly minimizes the storage requirements for each concept.
Third, we investigate the minimal amount of data necessary to train each GAN, further reducing the overall training time.
Extensive experiments demonstrate that we can efficiently empower GANs with the ability to perform real-time high-quality image editing on mobile devices with remarkable reduced training cost and storage for each concept.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper focuses on improving the inference and training efficiency of GANs for image-to-image translation. For inference efficiency, the authors design an efficient encoder-decoder backbone with fewer convolutional blocks and one additional transformer block. For training efficiency, the authors first pre-train a text-conditioned GAN on various concepts, then employ LoRA to minimize the number of trainable parameters, and finally leverage feature clustering to reduce the size of the training dataset. The proposed framework delivers good empirical performance, in accuracy, inference efficiency and training efficiency.

### Strengths
The paper is well-written and easy to follow. The authors offer a comprehensive introduction to the knowledge transfer pipeline for GANs in Section 3.1. This provides clarity for readers unfamiliar with the domain. The proposed framework is technically sound and delivers good empirical results. Furthermore, the provided implementation details are extensive, enabling researchers and practitioners in the field to replicate the findings and expand on this method.

### Weaknesses
The proposed framework appears to be "a bag of tricks":
1. Improved efficient backbone with hybrid convolution-attention architecture.
2. Base promptable image translation model.
3. Parameter-efficient fine-tuning using LoRA.
4. Training data subsampling by clustering CLIP features.

Diving deeper into these components, (1) is not a novel contribution. A hybrid convolution-attention architecture is already a common practice in the vision community and has been extensively explored in many papers, such as MobileViT [Mehta et al., ICLR 2022] and EfficientViT [Han et al., ICCV 2023]. Likewise, LoRA in (3) is widely used for parameter-efficient fine-tuning (PEFT), spanning both vision and language communities. While (2) and (4) are not conceptually new, it is still a good engineering effort to explore them in the context of GANs. Overall, it is unclear to me whether the technical contributions presented in this paper meet the publication standards of ICLR. 

Further compounding this is the observation that these four techniques appear to operate in silos, without much interdependence. Despite the framework's impressive end-to-end speedup, the individual impact of each component remains unclear. A comprehensive breakdown is crucial to elucidate the distinct contributions and effectiveness of each component to readers.

### Questions
My primary concerns are outlined in the weaknesses section. Beyond these points, I have several additional questions/comments:
* Table 1 only presents efficiency metrics for different models. Please kindly add accuracy metrics for image generation quality as well.
* The term "data distillation" may be misleading due to its resemblance to "dataset distillation". It might be more appropriate to use "distillation" or "model distillation".

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work proposes a framework to distill features from a pre-trained large diffusion-based image-to-image model into a GAN-based image-to-image model for fast on-device inference performance. The authors replace conv with attention module in the img2img GAN to achieve more efficient inference, and they also inject text condition into the GAN framework. Lastly, they empirically show such a design does not need to retrain for a new kind of edit, which greatly reduces storage and training efforts. Overall the paper is easy to follow and well-motivated.

### Strengths
– The author conducts extensive ablation study on the design choices of the architecture in terms of the number of different blocks
– Table 1 shows strong inference speed improvement in terms of both model size and FLOPs, comparing to baselines

### Weaknesses
– The author only conducts experiments on small-scale datasets, i.e. FFHQ and Flickr-Scenery with 1000 images. Also, the resolution is only 256, so it is hard to verify the claim that the base GAN can be trained on various text condition prompts on a large scale.

– Lack of technical novelty. The proposed base GAN architecture and also how to do text conditioning is very similar to GigaGAN.

### Questions
– I feel it lacks comparison to recent proposed efficient diffusion-based image2image models
– Is it practical to use the proposed method considering significant quality drops after distillation?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work is conducted towards efficient training and efficient inference for image-to-image translation via generative adversarial network (GAN). The paper constructs the base GAN model with a text conditional image generation task trained with various prompts and corresponded edited images from diffusion models, and then the pre-trained GAN model is fine-tuned with the analysis of effective partial weights for reducing the training cost and saving the storage.

### Strengths
- The efficient image editing, addressed by this work, is useful and helpful for the current mobile devices.
- It seems that the final model is light-weight and can be run fast on mobile devices.

### Weaknesses
- The novelty of this work is confusing. This work combines many existing techniques to pre-train and fine-tune a light-weight GAN model for efficient image-to-image translation, however, it seems that, not only the network architecture and loss functions, but also the transfer learning and knowledge transfer strategies, are not new and it is confusing what are the key contributions of this work that can differ from previous techniques and also are meaningful to the related study. Besides, the whole training and inference pipeline is also not clear enough, for example, how and why does this work distill GANs from diffusion models significantly efficient? Although this is the key question raised by this work, it is not easy to get the satisfactory answer after reading through the paper.
- The evaluation experiments are insufficient. The evaluation is mainly conducted on the FFHQ and Flickr-Scenery datasets compared with pix2pix and pix2pix-zero-distilled in terms of FID, and if it is towards the mobile devices, except for the efficient requirement, it is also important for the acceptable results required by the users, but it seems that this paper concerns not more about this factor and provides not enough evidence on the requirement.

### Questions
- What are the key contributions of this work that can differ from previous techniques and also are meaningful to the related study?
- What are the performance towards the real-world applications on mobile devices required by the users?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper provides an approach to enable flexible real-time image editing on mobile devices. It use Stable Diffusion to generate paired data to train Pix2pix. This approach overcomes the high computational requirements of traditional image editing with diffusion models. The research introduces more efficient methods for distilling GANs from diffusion models.  The proposed techniques include: 1) developing an attention-based network architecture for image-to-image translation on mobile devices,  2) implementing a hybrid training pipeline to adapt a pre-trained text-conditioned GAN model to different concepts efficiently,  and 3) investigating the minimum amount of data needed to train each GAN, further reducing the training time.

### Strengths
1) This paper build a connection between Stable Diffusion and GANs. It makes sense to use SD to create paired data, since there appears effective techniques to do image editing with SD.

2) This paper is easy to follow.

3) Authors conduct extensive experiments that demonstrate the proposed techniques.

### Weaknesses
I have some questions about this paper.

1) This proposed method are not supervising, since it is not easy to collect paired image from SD. In fact it is not hard to get this idea to create data.  I think this idea has less contribution for I2I translation community.

2) This paper combines more techniques into one paper. I see it is useful, but stacking a few techniques into one paper make me feel hard to  grasp the key contribution.  I could not grasp the important contribution from this paper. 

3) Also transferring knowledge from SD to Pix2pix model is not much interesting.   Is the approach  generalizable for other tasks? 

4) The shown figures are not too much interesting. Comparing the powerful editing ability of SD, using Pix2pix to do this task is not attracting. 

5) Using attention mechanism  I think is not contribution, which just follows what have done in SD.

### Questions
1) Since this paper combines a few contributions into one paper,  I could not grasp the key contribition.

2) This paper is  less interesting.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
