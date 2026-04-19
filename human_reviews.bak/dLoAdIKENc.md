# Robustness of AI-Image Detectors: Fundamental Limits and Practical Attacks

- Decision: Accept (poster)
- Scores: 5, 6, 6, 6

## Abstract
In light of recent advancements in generative AI models, it has become essential to distinguish genuine content from AI-generated one to prevent the malicious usage of fake materials as authentic ones and vice versa. Various techniques have been introduced for identifying AI-generated images, with watermarking emerging as a promising approach. In this paper, we analyze the robustness of various AI-image detectors including watermarking and classifier-based deepfake detectors. For watermarking methods that introduce subtle image perturbations (i.e., low perturbation budget methods), we reveal a fundamental trade-off between the evasion error rate (i.e., the fraction of watermarked images detected as non-watermarked ones) and the spoofing error rate (i.e., the fraction of non-watermarked images detected as watermarked ones) upon an application of a diffusion purification attack. In this regime, we also empirically show that diffusion purification effectively removes watermarks with minimal changes to images. For high perturbation watermarking methods where notable changes are applied to images, the diffusion purification attack is not effective. In this case, we develop a model substitution adversarial attack that can successfully remove watermarks. Moreover, we show that watermarking methods are vulnerable to spoofing attacks where the attacker aims to have real images (potentially obscene) identified as watermarked ones, damaging the reputation of the developers. In particular, by just having black-box access to the watermarking method, we show that one can generate a watermarked noise image which can be added to the real images to have them falsely flagged as watermarked ones. Finally, we extend our theory to characterize a fundamental trade-off between the robustness and reliability of classifier-based deep fake detectors and demonstrate it through experiments. Code is available at https://github.com/mehrdadsaberi/watermark_robustness.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper analyzes the robustness of two common AI-generated image detection approaches: watermarking and classification. For watermarking, it points out that diffusion purification can effectively remove low-perturbation budget watermarks but fails to work on high-perturbation budget ones. For the latter one, the paper proposes using a model-substitution adversarial attack to remove the watermarks. It also proposes a spoofing attack against watermarking by blending a watermarked noise image with the non-watermarked target image. Finally, it demonstrates a trade-off between the robustness and reliability of classification-based deepfake detectors.

### Strengths
The paper lacks a clearly identifiable strength.

### Weaknesses
The paper is poorly written and not well-organized. For example, acronyms were defined far after their first usage. Figures are hard to follow and understand.
 
The contribution of the paper is unclear. It is common sense pointed out by several previous research that the stronger the perturbations, the more difficult for purification. Using random noise or adversarial perturbations to compromise the machine-learning-based forensics models (watermarking and deepfake detection) has been studied in the past, which is also mentioned in the related work section.
 
The experiment settings are insufficient to demonstrate the claimed problems. Besides DiffPure (Nie et al., 2022), there are other diffusion-based approaches, such as DDNM (Wang et al., 2022), and non-diffusion-based ones, which are not investigated.
 
The robustness of the proposed attack methods, which is an important property, was not evaluated. DiffPure (Nie et al., 2022), JPEG compression, or Gaussian blur can be used to mitigate such attacks, although it may slightly degrade the clean accuracy.

The paper should include a paragraph of ethics statement.

### Questions
Please refer to the comments in the weaknesses section.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper analyzes the robustness of various AI image detectors, including watermarking and classifier-based deepfake detectors, wherein watermarking is considered a promising method for identifying AI-generated images. 

The paper also evaluates the trade-off between evasion error rate and spoofing error rate in watermarking methods, introducing subtle image perturbations. Besides, it is demonstrated that a diffusion purification attack that amplifies the error rates of low perturbation budget watermarking methods, thereby revealing the fundamental limits of the robustness of image watermarking methods. For large-perturbation watermarking methods, the diffusion purification attack is ineffective. Therefore, the authors propose a model substitution adversarial attack to successfully remove watermarks.

Overall, this paper makes significant contributions to the robustness of AI image detection methods, supported by detailed theoretical proofs of the viewpoints presented. Some of the theoretical analyses and attack methods in the article are relatively complex and may require readers to have a certain level of background knowledge to fully understand.

### Strengths
1) Originality: The paper challenges the limitations of existing watermarking techniques by proposing new attack methods, driving progress in the field.

2) Quality: The quality of the paper is very high, with in-depth and rigorous theoretical analysis, reasonable experimental design, and results that fully validate the theoretical analysis.

3) Clarity: The structure of the paper is clear, the logic is tight, and the discussion is detailed and easy to understand. However, some of the complex theoretical analyses and attack methods may require readers to have a certain level of background knowledge.
Importance: By revealing the vulnerabilities of existing watermarking techniques, the paper lays the groundwork for further research and development in the field. Additionally, by proposing new attack methods, the paper challenges the limitations of existing watermarking technologies and promotes progress in the field.

### Weaknesses
1) Complexity of Theoretical Analysis: 
While the paper provides an in-depth theoretical analysis, the complexity of these analyses might pose challenges for some readers, especially those who are not familiar with diffusion models and the theoretical underpinnings of adversarial attacks. Certain sections of the paper may appear somewhat opaque to these readers. It would be beneficial if the authors could simplify the explanations or provide additional resources to aid understanding.


2) Complexity of Theoretical Analysis: 
While the paper provides an in-depth theoretical analysis, the complexity of these analyses might pose challenges for some readers, especially those who are not familiar with diffusion models and the theoretical underpinnings of adversarial attacks. Certain sections of the paper may appear somewhat opaque to these readers. It would be beneficial if the authors could simplify the explanations or provide additional resources to aid understanding.

### Questions
1) Problem Description: The experiments conducted in the article primarily utilize the ImageNet dataset, potentially limiting the generalizability of the results. 
Recommendation: In future work, conduct experiments using multiple datasets from various fields and sources to validate the effectiveness and stability of the method.

2) Problem Description: The theoretical analysis presented in the article is quite complex, which may be challenging for all readers to comprehend. 
Recommendation: Provide more intuitive explanations and examples, and simplify some of the theoretical derivations to make them more accessible and easier to understand.

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
This work investigates the resilience of AI-image detection methods, focusing on watermarking and classifier-based deepfake detectors. The authors highlight the crucial need to distinguish between authentic and AI-generated content due to the rising threat of fake materials being used as genuine ones. They reveal a fundamental trade-off in the effectiveness of watermarking techniques, showcasing the limitations of low-perturbation and high-perturbation watermarking methods. Specifically, they propose diffusion purification as a certified attack against low-perturbation watermarks and a model substitution adversarial attack against high-perturbation watermarks. Additionally, the paper emphasizes the vulnerability of watermarking methods to spoofing attacks, which can lead to the misidentification of authentic images as watermarked ones. Finally, the authors extend their analysis to classifier-based deepfake detectors, demonstrating a trade-off between reliability and robustness.

### Strengths
The strengths of this work are as follows:
1. Comprehensive analysis: The paper provides a comprehensive analysis of the robustness of AI-image detection methods, focusing on both watermarking and classifier-based deepfake detectors. This thorough investigation helps in understanding the limitations and vulnerabilities of these methods.
2. Practical attacks: The paper introduces practical attacks, such as diffusion purification and model substitution adversarial attacks, to illustrate the vulnerabilities of different watermarking methods. 
3. Clarity in trade-offs: The paper effectively highlights the trade-offs between various aspects of AI-image detection methods, such as the trade-off between evasion error rate and spoofing error rate in the case of watermarking methods. This clarity helps in understanding the challenges associated with designing robust AI-image detection systems.
4. Sound theoretical study and guidelines for designing robust watermarks: The paper offers insights into the attributes that a robust watermark should possess, including significant perturbation, resistance to naive classification, and resilience to noise from other watermarked images. These guidelines can serve as a valuable reference for researchers and developers working on improving the security and reliability of AI-image detection methods.

### Weaknesses
The experimental results are missing a key element, specifically the PSNR, SSIM, or other image quality metrics comparing the diffusion-purified or adversarially attacked images with the original images. This result is crucial as adversaries aim to eliminate watermarks while preserving high-quality images simultaneously.

### Questions
Please refer to the weakness.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this paper, the authors introduce a diffusion purification attack to break AI-image detection methods using watermarking with a low perturbation budget. For high perturbation image watermarking, they develop a model substitution adversarial attack. Besides, they successfully implemented a spoofing attack by adding a watermarked noise image with non-watermarked ones. Furthermore, they use comprehensive experiments to substantiate the trade-off between robustness and reliability of deepfake detectors.

### Strengths
A novel watermarking erasing method is proposed to break high perturbation budget watermarking like Tree Ring or StegaStamp.
This paper is well-written and easy to understand. The presentation and organization of the paper are good.

### Weaknesses
1.	The crux of this paper is theorem 1, which gives a lower bound for the sum of evasion and spoofing errors with regard to the Wasserstein distance between diffusion purification processed images. However, the empirical studies are not sufficient.
2.	The empirical studies for theorem 2 are unpractical. It is almost impossible to add noise to the interior feature maps directly in practical cases.

### Questions
1. In Fig. 6, authors adopt four low perturbation budget watermarking, including DWTDCT, DWTDCTSVD, RivaGAN, WatermarkDM, to validate theorem 1. More watermarking methods are required to be considered, like reference [1]. Many cutting-edge watermarking methods are absent in Fig. 6.
2. The experimental setup of section 4 is impractical. I suggest authors add noise to spatial images directly instead of interior feature maps. 
3. The title of the paper is “AI-image detectors”. The authors only consider fake facial images in section 4. AI-generated facial detectors are only a small part of AI-image detectors. Therefore, more AI-image detectors for arbitrary contexts require consideration.

[1] MBRS : Enhancing Robustness of DNN-based Watermarking by Mini-Batch of Real and Simulated JPEG Compression

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good
