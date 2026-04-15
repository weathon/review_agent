# Revealing Unintentional Information Leakage in Low-Dimensional Facial Portrait Representations

- Decision: Reject
- Scores: 5, 8, 3

## Abstract
We evaluate the information that can unintentionally leak into the low dimensional output of a neural network, by reconstructing an input image from a 40- or 32-element feature vector that intends to only describe abstract attributes of a facial portrait. The reconstruction uses blackbox-access to the image encoder which generates the feature vector. Other than previous work, we leverage recent knowledge about image generation and facial similarity, implementing a method that outperforms the current state-of-the-art. Our strategy uses a pretrained StyleGAN and a new loss function that compares the perceptual similarity of portraits by mapping them into the latent space of a FaceNet embedding. Additionally, we present a new technique that fuses the output of an ensemble, to deliberately generate specific aspects of the recreated image.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces a novel face generation method from a low-dimensional (32 / 40-element) feature vector. The low dimensional feature vector is intended to only describe the attribute information, however the proposed method is able to generate a face that has high similarity with the target image from the encoded feature vector. Thus, it is able to reconstruct the unintentional private information, such as identity. The proposed method utilizes style-GAN and a new loss function based on the face perceptual similarity. In addition, a novel technique to fuse multiple layers of Style-GAN is introduced.

### Strengths
+ The proposed method tackles a difficult problem where it wants to reconstruct a target face image from a low-dimensional feature vector. The method works because it effectively designs the framework so that it limits the search space. The feature vector (f) is mapped into a generator input (n) so that the generator (G) is able to produce consistent face image. Thus, the mapping function (P) is trained so that the feature vector can be used to generate similar images as the target image. 

+ The loss functions consist of pixel-wise and Facenet embedding losses. While the pixel-wise loss is used to maintain the overall image similarity, the Facenet embedding loss is used to preserve the facial features similarity, such as shape of the eyes, lips, etc. As each block in the StyleGAN defines different features of an image, the mapping function (P) is trained using different losses for each block as mentioned in Section 3.4. Thus, Facenet embedding loss is applied to second and third block which are in charge of the facial features. 

+ Experimental results show that the proposed method overcomes the predecessor (Yang et al - 2019) qualitatively. It is able to reconstruct clear and not-blurred face image that has similarity with the target image. In addition, the experiments show that the proposed method achieves low embedding error of the extracted VGG-Face and OpenFace feature embedding. It shows that the generated face has high similarity with the target image.

### Weaknesses
- The paper only measures the overall error of the generated dataset. Figure 4 & 5 needs to add the similarity score to give a rough idea about how similar the generated image with the target image. It is unclear how the last column in the figures can be noted as similar person. Note that the paper needs to justify that the private information is leaked in the reconstructed image. 

- The paper utilizes different loss functions for each block in StyleGAN. However, there is a lack of justification of the results. The ablation study is required to justify the decision.

- The paper shows that the training of D and E on different dataset is more difficult than on the same dataset. While the quantitative shows the differences, however it is not reflected in qualitative evaluation. In addition, it leads to the question whether the OpenFace and VGG-Face embedding is suitable for measuring the similarity. It is recommended to follow [A] procedure to measure the possibility of identity leakage in the encoder. 

- As mentioned in [A], different encoder (face feature generator) might lead to different conclusion in terms of identity leakage. Thus, it is important for the authors to measure the possibility of different face attribute encoder methods. In addition, it is also important to use the more challenging face embedding encoder, like ArcFace or SphereFace to evaluate the similarity between images. 

- There are several references that are able to reconstruct face from attribute feature vector [B, C]. Those are might be used for better methods to compare.  

Additional references:
* [A] This Face Does Not Exist... But It Might Be Yours! Identity Leakage in Generative Models, WACV 2021
* [B] Attributes Aware Face Generation with Generative Adversarial Networks, ICPR 2020
* [C] Latent Vector Prototypes Guided Conditional Face Synthesis, ICIP 2022

### Questions
* What are the similarity scores of the reconstructed and target images in the figures?
* How do the authors justify that the private information is leaked in the reconstructed face image?
* How are the performance using different face attribute encoders?
* How are the performance using different face embedding, such as Arcface and SphereFace?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper investigate unintentional leakeage from a feature used for face recognition in terms of soft biometrics. The problem is approached using a StyleGAN to build the invertible networks. The method is evaluated on two scenarios in conjunction  with CelebA database.

### Strengths
1. The idea to investigate leakage in face recognition modules is very interesting. Approach and results are also welcomed 

2. Additional results  from the appendix help to understand better the problematic.

### Weaknesses
1. The paper has no "Conclusion" section. While it formally exists, it repeats the main steps of the paper. It is not clear to me what is the lesson learned. The paper is entitled "REVEALING UNINTENTIONAL INFORMATION LEAKAGE ..." and after reading I do not have a clearer view of what has been leaked and what is not

Minor comments:
 - section 2 "Thread" - "threat" ?

### Questions
I believe that the paper really needs some conclusion. Otherwise is not complete and therefore it is not clear why people would want to hear from about it. Now the paper is non-committal: there is a problem, we identify it, we analyze the limits of a solution, we propose some solution, we evaluate some solution bearing in mind the limits and this is it.

I expect that starting from the facts [presented in the paper a conclusion can be provided during the rebuttal and I will increase my score. But until then, I believe that the paper is not ready for publication.

------
Post-rebuttal:
----
I have read other reviewers opinion and author rebuttal. In the revised version/rebuttal it has been answered to the question I have asked. While it is not in line with other reviews. I see this paper contribution as being positive. I appreciate the fact that are pointing so some issues which they measure, in my view appropriately. I do not share the opinion that publicizing such aspect could reveal some backdoor for malicious intents. 

I am keeping my suggestion which is the paper is interesting and it will attract auditorium.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies the unintentional information leakage that can happen in deep encoder networks that extract latent representations with abstract attributes from face images. The paper proposes a method that is capable to reconstruct an input face image from a feature vector representation using only black box access to the image encoder. The method is based on the StyleGAN formulation, which is extended with an additional loss that compares the perceptual similarity of portraits by mapping them into the latent space of a FaceNet embedding. The purpose of this paper is to raise awareness about the relevant security issues of existing deep learning systems for face analysis.

### Strengths
+ This paper deals with an interesting and important problem that has attracted limited attention from the computer vision community. It is particularly important for reasons related to security and preservation of privacy. 

+ The proposed pipeline is intuitive and sound, building upon the formulation of the StyleGAN model.

### Weaknesses
- The technical novelty of the proposed method is relatively limited. It only describes a small extension of the loss function of the StyleGAN model. It is mostly interesting as an application of the GAN-based formulations, but I think that it lacks sufficient contributions for a paper accepted in ICLR. Other venues might be more appropriate for such paper. 

- The experimental evaluation is highly inadequate. The only quantitative evaluation is the one presented in Table 1. However, this corresponds to an internal evaluation of the proposed method, without any comparison with other SOTA methods. Closely related methods like (Yang et al., 2019) and (Zhao et al. 2021) should have been included in the quantitative comparisons. In addition, a perceptual user study should have been included in the experiments, in order to quantify the performance of the proposed method and other compared methods, in terms of whether the reconstructed faces are perceived by humans to have the same identity as the original real faces. 

- The paper has also inadequacies in terms of discussing and citing prior art. First, Some closely-related works, like (Razzhigaev et al. 2020) are only presented in Table 2 of the Appendix. However, such works should have been presented in the main paper, with discussion about their similarities and differences from the proposed method. Furthermore, the paper has not cited some closely-related works like the following:

Khosravy, M., Nakamura, K., Hirose, Y., Nitta, N. and Babaguchi, N., 2022. Model inversion attack by integration of deep generative models: Privacy-sensitive face generation from a face recognition system. IEEE Transactions on Information Forensics and Security, 17, pp.357-372.

Khosravy, M., Nakamura, K., Nitta, N. and Babaguchi, N., 2020, December. Deep face recognizer privacy attack: Model inversion initialization by a deep generative adversarial data space discriminator. In 2020 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC) (pp. 1400-1405). IEEE.

### Questions
Please see my comments regarding the weaknesses of the paper. 

In addition, there are issues with the clarity of the presentation in some parts of the paper. For example, the last two paragraphs of Section 1 are repeated twice.

### Soundness
1 poor

### Presentation
2 fair

### Contribution
3 good
