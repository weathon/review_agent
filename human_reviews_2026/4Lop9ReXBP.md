# Model-agnostic Watermarked Image Restoration without Additional Training

- Avg Score: 2.50
- Decision: Reject
- Scores: 4, 2, 2, 2

## Abstract
Post-processing image watermarking technology, which can prove the authenticity of real images, causes quality degradation and information loss in the original image. Although various methods have been proposed to restore a watermarked image to the original image, these methods are model-dependent. In this study, we propose a model-agnostic watermarked image restoration method that requires no additional training. The proposed method first extracts a message from a watermarked image and embeds the same message into the watermarked image. Then, our method computes a watermark component as the subtraction between the watermarked image and the double watermarked image. Finally, the proposed method generates a restored image by subtracting the watermark component from the watermarked image because the watermark component has a high correlation with the subtraction between the watermarked image and the original image. Experimental results show that the proposed method obtains a restored image with higher image quality for 10 of 11 existing watermarking methods. Furthermore, we have extended the existing eight methods and added a re-watermarking function that updates an embedded watermark with another watermark.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes a model-agnostic watermarked image restoration method, which aims to address the issue that post-processing image watermarking technology causes quality degradation and information loss in the original image. Unlike existing model-dependent restoration techniques, such as reversible watermarking techniques and TrustMark RM (Bui et al., 2023a), a core advantage of this method is that it requires no additional training.

### Strengths
- **Clear writing:**  The paper is well-written and has a logical flow.
- **Clear Baseline Comparison:**  The paper evaluates 11 post-processed watermarking methods. The baseline comparison covers major strategies in watermark removal.

### Weaknesses
- **Discrepancy between White-Box Setting and Practical Usefulness:**  Although the paper labels the method as "model-agnostic", this primarily means it requires no additional training. Fundamentally, the method operates under a strict white-box setting: The user must possess the same embedder and  message m. This requirement contradicts the threat models of many generic watermark removal attacks, such as DiffAtt (Zhao et al., 2024) and Watermark Steganalysis (Yang et al., 2024), which often deal with black-box scenarios where the embedder/message is inaccessible. Furthermore, the proposed restoration method can only be applied to post-processing watermarking, and not to in-processing watermarking. Please clarify these scope boundaries more explicitly.
- **Fairness of Baseline Comparison (White-Box vs. Black-Box/Grey-Box):**  The comparison pits a white-box recovery method against generic attack methods designed for black-box or grey-box settings, specifically DiffAtt (Zhao et al., 2024) and WmStg (Yang et al., 2024). White-box methods inherently hold a significant information advantage, leading to expectedly higher image quality metrics. Please analyze this discrepancy more carefully.
- **Weak Theoretical Foundation for Watermark Component Correlation:**  The core approximation supporting the method is the high correlation assumption:  $W(I_{\text{orig}}, m) \approx \alpha \cdot W(I_{\text{wm}}, m)$. However, the watermark component W often results from complex or non-linear operations, such as deep learning models or complex frequency transformations. While the paper presents visualizations demonstrating the components' similarity, it lacks formal mathematical proof or boundary analysis to guarantee the robustness of this crucial approximate linear relationship across diverse and non-linear watermark embedders. Such theoretical analysis would strengthen this work.

### Questions
- **Inconsistencies in Reference Formatting:**  The review notes that the list of references contains a mix of different formats and types of entries, such as conference papers, arXiv preprints, and embedded GitHub URLs. Please standardize to the conference’s bibliography style.
- Other issues as noted above under Weaknesses.

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
This paper proposes a model-agnostic watermarked image restoration method that requires no additional training to address the quality degradation caused by post-processing image watermarking. The proposed approach first extracts a message from a watermarked image and re-embeds the same message into it, then computes a watermark component through subtraction between the original watermarked image and the double-watermarked version. By leveraging the high correlation between this component and the difference between the watermarked and original image, the method restores the image by subtracting the watermark component.

### Strengths
1. It is a  training-free method. Watermark restoration can be achieved without additional training, reducing computational costs and time consumption.
2. Compared to other model-agnostic watermark removal technologies, it demonstrates significantly superior performance.

### Weaknesses
1. This method is technically limited to recovering images by extracting watermark residuals, demonstrating weak innovation.
2. The introduction and related work sections of this paper are  difficult to read, containing numerous logical inconsistencies that make it hard for readers to grasp the author's intended arguments.
3. What does "information loss" mean in the abstract, and how does "Post-processing watermarking causes quality degradation in the original image and reduces data reliability" from the introduction relate to this? Specifically, what does data reliability refer to.

### Questions
1. This method must ensure that the embedded information remains identical each time. However, in practice, variable time information is often included, resulting in differing embedded data each time. How can this be resolved?
2. This method assumes that the acquired watermark image is free from distortion interference. However, I am curious about the restoration effectiveness of this method when only distorted watermark images are available.
3. Section 4.2 states that different methods require varying numbers of iterations. Why does PSNR increase for some methods as iterations progress, while it decreases for others? Does this indicate limitations in the method?
4. Watermarks like SSL and RoSteALS embedded in latent space appear to yield very low PSNR upon recovery. Does this method perform poorly on such watermarking techniques? Additionally, how does it fare on LaWa[1]?

[1] Rezaei A, Akbari M, Alvar S R, et al. Lawa: Using latent space for in-generation image watermarking[C]//European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024: 118-136.

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
3

### Summary
This paper considers the problem of removing watermarks from watermarked images, and then restoring them.

Consider a watermarking method $W$ that takes an image $I$ and a message $m$. This paper considers watermarks that compute the watermarked image as $I_wm = I_orig + W(I_orig, m)$.  The paper proposes repeating this process to produce $I_wm' = I_orig + W(I_wm, m)$ multiple times.

They also propose restoring the watermark via creating the restored image $I_res = I_wm - alpha W(I_wm, m)$ for various choices of alpha. They evaluate their method against 11 watermarking methods from 2007, 2007, 2018, 2019, 2021, 2022, 2023, 2023, 2025, 2025, and 2025.

### Strengths
They compare their method to many watermarking methods.

### Weaknesses
1. The "method" they propose is ridiculously simple: Just apply the watermark... again! And, if you want to remove it, subtract the watermark multiple times! I'm not sure this should be a published idea, much less at a major ML conference like ICLR.

2. For lacking any meaningful insights (not to mention at techniques or theory), they try to make up for it with the experiments. But only three of the 11 watermarking methods they compare to are from 2024 or later. Given the pace of generative AI, I would recommend comparing to more methods (they even cite TreeRing, but I don't *think* they compare to it): TreeRing, RingID, Gaussian Shading, PRC, WIND, and SEAL.

3. The presentation is poor. The writing is difficult to follow (even though their idea is quite simple), and the Figure 1 and Figure 2 are a) not very informative and b) quite visually unappealing.

TreeRing: https://arxiv.org/abs/2305.20030
RingID: https://arxiv.org/abs/2404.14055
Gaussian Shading: https://arxiv.org/abs/2404.04956
WIND: https://arxiv.org/html/2412.04653v3
PRC: https://arxiv.org/pdf/2410.07369?
SEAL: https://arxiv.org/abs/2503.12172

### Questions
Could you please justify the watermark removal process in Equation 5?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
In this paper, the authors propose a model-agnostic watermarking restoration method by simply calculating the difference between the watermarked image and the double-watermarked image and the substract it from the watermarked image. The author also conduct a discussion about how the proposed method develop re-watermarking and how to act as an attack method. Abundant experimental result show promising results.

### Strengths
1. This paper contains abundant experiments on so many watermarking methods, which imporves the persuasiveness of the paper.
2. The discussion part is quite interesting. Restoration process acts as an attaker could be useful and more practical in real-world.

### Weaknesses
1. The proposed scheme seems to be too simple. Only simply calculating the difference between the watermarked image and the double-watermarked image and the substract it from the watermarked image. Well, we have to say, this really make sense, but we think more theoretical analysis on this shall be provided, or in other word, WHY the proposed method works. For now, we suggest the persuasiveness is not enough. Since there are too many post-processing watermarking methods, a more comprehensive analysis help the reader to believe the proposed method is truly model-agnostic.

2. No offense to this work and any other work in this area, but in our humble opinion, the watermarking restoration methods which aims to fight against quality degradation and information loss might fall into a wrong path for real-world application. Since most of the post-processing watermarking methods are used for protection or generally speaking, they act as a unique stamp to prove something (belong to whom or whether it is generated by AI etc.). Therefore, we might raise two point: first, keeping the watermarking exist is vital for most of the time (unless you are the attacker who want to bypass the watermark and use the image illegally), and second, quality degradation and information loss might not be a vital problem (unless there are severe degradation, but in recent years, most watermarking methods achieves satisfying visual quality). Therefore, we are appreciating more on the part where the author tries to become an attacker who remove the watermark and add a new one. We think that is more relevant to real-world applications.

3. Figure 1 and 2 are not good, occupying much space but offers too little information.

### Questions
1. It is clear that the proposed method cannot be applied to watermarking methods that is not post-processing. However, in-processing methods become more popular in recent years. Try to discuss how the proposed method can be applied on them with only a minor manipualation. (We will considered changing our score based on the response, especially on this question)

### Soundness
3

### Presentation
2

### Contribution
2
