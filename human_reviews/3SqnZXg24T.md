# RetinexGAN Enables More Robust Low-Light Image Enhancement Via Retinex Decomposition Based Unsupervised Illumination Brightening

- Avg Score: 2.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 1, 3

## Abstract
Most existing image enhancement techniques rely heavily on strict supervision of paired images. Moreover, unsupervised enhancement methods also face challenges in achieving a balance between model performance and efficiency when handling real-world low-light images in unknown complex scenarios. Herein, we present a novel low-light image enhancement scheme termed \textbf{RetinexGAN} that can leverage the supervision of a limited number of low-light/normal image pairs to realize an accurate Retinex decomposition, and based on this, achieve brightening the illumination of unpaired images to reduce dependence on paired datasets and improve generalization ability. The decomposition network is learned with some newly established constraints for complete decoupling between reflectance and illumination. For the first time, we introduce the feature pyramid network (FPN) to adjust the illumination maps of other low-light images without any supervision. Under this flexible framework, a wide range of backbones can be employed to work with illumination map generator, to navigate the balance between performance and efficiency. In addition, a novel attention mechanism is integrated into the FPN for giving the adaptability towards application scenes with different environment like underwater image enhancement (UIE) and dark face detection. Extensive experiments demonstrate that our proposed scheme has a more robust performance with high efficiency facing various images from different low-light environments over state-of-the-art methods.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposed a Retinex-decomposition-based generative adversarial network for low-light image enhancement, which is trained with unpaired samples. The experiments showed the superior performance of the proposed neural network over some recent approaches.

### Strengths
+ Combination of GAN, Retinex model, and unpaired learning.
+ Interpretability of the neural network architecture from due to the use Retinex model.

### Weaknesses
- The compared methods in the expriements are somehow out of data. Only one compared method is published at or after 2022. This makes the experiemtns not convincing.
- GAN, unpaired training, and Reintex model are widely studied and utillized in low-light enhancement. The papers lacks a detailed discussion with existing methods and ideed it seems that the proposed components have no big differences from the existing ones.

### Questions
See the Weakeness part.

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
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper presents a semi-supervised illumination brightening framework for low-light image enhancement. A lightweight CNN model is used to achieve Retinex decomposition, and then a feature pyramid network is employed to brighten the illumination maps in a unsupervised manner. The proposed method has been evaluated on several datasets, and improved results have been achieved.

### Strengths
* The paper is easy to follow.
* The proposed framework achieves improved performances on both synthetic and real-world low light images.

### Weaknesses
* My main concern about this paper is its limited novelty and technical contribution. The adopted FPN network and the spatial attention mechanism are very common strategies in many research fields.
* For the experimental results, although improved qualitative results have been obtained, the visual results in Fig.1(a), Fig.6 and Fig.7 still look poor. For better comparisons, can the authors provide more comparison results with ground truth as a reference? For downstream applications, it may be more intuitive to use P-R curves for comparisons (as Zero-DCE).
* The paper’s readability is poor. A lot of technical details are missing. The writing should be improved for a better review.
* The literature survey should be more elaborate. Several newly published works are not discussed, especially the supervised methods.

### Questions
Other issue:
* The abbreviation "RetinexGAN" has been used in "Ma et al., RetinexGAN: Unsupervised low-light enhancement with two-layer convolutional decomposition networks, IEEE Access".

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
1: strong reject

### Rating Number
1

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The work introduces a decomposition network based on the Retinex theory. Subsequently, it incorporates a Generative Adversarial Learning strategy by constructing an illumination generation network based on FPN to further enhance low-light images captured in unknown scenarios. A series of related experiments demonstrate the superiority of the proposed method and the effectiveness of the various modules and constraints designed.

### Strengths
The approach of building an illumination enhancement network by introducing FPN is quite intriguing. The overall framework's process and structural details are described clearly, and the overall presentation is commendable.

### Weaknesses
1. Unclear Motivation: The motivations behind the various components in the overall framework constructed by the author seem to lack clear explanations. For instance, the initial intention of decomposition followed by enhancement, the benefits of introducing FPN, and the reasons for employing different training strategies at different stages, among other aspects, are not adequately addressed. Unfortunately, the author appears to have not provided relevant analyses for these issues. Furthermore, it is suggested that the author should provide additional evidence in their response to support their claims.
2. Lack of Novelty: Whether it's the decomposition network based on Retinex, generative adversarial networks, attention mechanisms, or the range of introduced loss functions, these are common approaches for addressing low-light image enhancement problems. Additionally, while the idea of introducing FPN is promising, the lack of a clear explanation by the author impacts the inevitability of the contribution's novelty. In other words, this work appears more like a combination of existing effective techniques, leaving room for increased novelty.
3. Unconvincing Experimental Results: Most of the visual results presented in the manuscript exhibit evident color shifts, and some even contain artifacts. In comparison to other methods, the superiority of the proposed approach in terms of qualitative results is hard to establish.
4. Need for Improvement in Experimental Settings and Presentation: From the manuscript, it can be seen that the author conducted a series of comparative experiments on the LOL dataset, but it seems that no qualitative results are provided. I am curious about the performance of the proposed method on data from the LOL dataset that contains significant noise. Furthermore, it appears that no comparisons were made with low-light image enhancement methods proposed in 2023. As far as I am aware, there have been many recent representative advances in this field, and it is hoped that the author can include comparisons with them to more comprehensively validate the effectiveness of the proposed method.

Overall, this work requires significant improvement in several aspects. The ablation experiments section also suffers from a lack of comprehensiveness. It is hoped that the author can provide sufficiently detailed responses to the above-mentioned issues.

### Questions
Please refer to Weaknesses.

### Soundness
1 poor

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The research direction of this article is low-light image enhancement, which mainly solves the image degradation problem through Retinex decomposition theory and FPN-based attention mechanism. The contribution of the article is to simultaneously provide flexibility in low-light image enhancement by leveraging lightweight cellular neural networks for data-driven Retinex decomposition and leveraging FPN with different types of pre-trained backbones for downsampling in illumination generation.

### Strengths
1. The main purpose of this article is to propose a flexible framework for enhancing low-light images, which has practical applications.
2. Motivation of this paper is clear and the main idea is easy to understand.

### Weaknesses
1. The author did mention that the work aims to find a balance between efficiency and performance, but the main body of the text primarily focuses on the analysis of performance. Furthermore, based on the experimental results regarding computational efficiency in the appendix, it seems that the proposed method may not have achieved a sufficiently prominent advantage in terms of efficiency.
2. In the second section of the method introduction, the author dedicates a significant amount of space to explaining specific network structures and the various constraints introduced, without providing an explanation for the reasons. In other words, the motivations for the various components in the proposed framework may require additional clarification from the author.
3. Compared to other methods, the results shown in Figure 1, Figure 7, and Figure 11 do not appear to be the best. In Figure 5, there are even instances of artifacts. It is suggested that the authors provide an explanation for the phenomena mentioned above.
4. In the ablation experiment section, the author only analyzed the loss function. To further validate the effectiveness of the constructed framework, it is recommended that the author include ablation experiment results for each component of the framework in the manuscript.

### Questions
The relevant questions have been raised in the weaknesses section.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
