# Posterior-Mean Rectified Flow: Towards Minimum MSE Photo-Realistic Image Restoration

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 8

## Abstract
Photo-realistic image restoration algorithms are typically evaluated by distortion measures (e.g., PSNR, SSIM) and by perceptual quality measures (e.g., FID, NIQE), where the desire is to attain the lowest possible distortion without compromising on perceptual quality. To achieve this goal, current methods commonly attempt to sample from the posterior distribution, or to optimize a weighted sum of a distortion loss (e.g., MSE) and a perceptual quality loss (e.g., GAN). Unlike previous works, this paper is concerned specifically with the *optimal* estimator that minimizes the MSE under a constraint of perfect perceptual index, namely where the distribution of the reconstructed images is equal to that of the ground-truth ones. A recent theoretical result shows that such an estimator can be constructed by optimally transporting the posterior mean prediction (MMSE estimate) to the distribution of the ground-truth images. Inspired by this result, we introduce Posterior-Mean Rectified Flow (PMRF), a simple yet highly effective algorithm that approximates this optimal estimator. In particular, PMRF first predicts the posterior mean, and then transports the result to a high-quality image using a rectified flow model that approximates the desired optimal transport map. We investigate the theoretical utility of PMRF and demonstrate that it consistently outperforms previous methods on a variety of image restoration tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces posterior-mean rectified flow framework, which approximates optimal estimator under perceptual constraint. The method includes predicting posterior mean and subsequent distribution transport using rectified flow. The experiments on face image restoration tasks validate its efficacy.

### Strengths
1.	The motivation is clearly stated. The main idea that considering image restoration problem using MSE optimization under perceptual constraint is interesting and essential.
2.	The experiments on face image restoration demonstrate superior performance over previous methods.

### Weaknesses
1. The method seems like such a pipeline: Supervised restoration model trained using MSE loss + SDEdit method (with added $\sigma_s$ noise). Can authors provide further interpretations?
2. The experiments focus on specific the face image domain. The extension to other image domains (such as natural images) is unclear and unexplored.
3. The structure of the paper should be reorganized. The main body does not contain any quantitative results.

### Questions
I suggest the authors add experiments on other image domains rather than face domain since the main topic is image restoration problem.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The work introduces Posterior-Mean Rectified Flow (PMRF) for image restoration, addressing the challenge of minimizing Mean Squared Error (MSE) while maintaining high perceptual quality. PMRF operates in two stages: first predicting the posterior mean, then using a rectified flow model to transport the result to match the ground-truth image distribution. The authors provide theoretical foundations showing that PMRF can achieve better MSE than posterior sampling methods while maintaining perfect perceptual quality under certain conditions. Through extensive experiments on various image restoration tasks (including blind face restoration, denoising, super-resolution, inpainting, and colorization), they demonstrate that PMRF consistently outperforms or matches state-of-the-art methods in both distortion metrics and perceptual quality measures.

### Strengths
+ The paper provides comprehensive experiments and solid theoretical proof, supporting the effectiveness of the proposed method.
+ It is well-written, with clear and informative figures and tables that enhance the reader's understanding. 
+ The motivation behind the approach is convincing, and the experimental results substantiate the motivation to a significant extent, demonstrating the practical impact of the proposed framework.

### Weaknesses
- While the method shows strong results in blind face image restoration, its evaluation is limited to this domain. Given the theoretical generality of the proposed framework, it would be valuable to see experiments on more diverse and general image datasets to assess the broader applicability of PMRF. Extending their experimental evaluation to include general image restoration would demonstrate the versatility of their method.
- Including ablation studies on the rectified flow model, posterior mean prediction, noise levels, and other architectural choices would provide deeper insights into the importance of each component and help the reader understand which parts of the framework drive the performance improvements.

### Questions
If I understand correctly, the idea proposed by the authors should also be effective on general images, beyond the specific tasks tested. It would be interesting to know if the authors have considered conducting experiments or making attempts on general image datasets to further validate the versatility of their method. Of course, the paper is still completely acceptable without these additional experiments. However, including them would significantly enhance the impact of the work.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a novel approach to photo-realistic image restoration (PIR) that aims to minimize mean squared error (MSE) while maintaining perceptual quality. The authors introduce the Posterior-Mean Rectified Flow (PMRF) algorithm, which optimally transports the posterior mean prediction to the distribution of ground-truth images. This method contrasts with traditional approaches that often prioritize perceptual quality at the expense of distortion. By leveraging a rectified flow model, PMRF effectively approximates the optimal estimator that minimizes MSE under a perfect perceptual index constraint. Experimental results demonstrate that PMRF outperforms existing methods across several restoration tasks.

### Strengths
+ PMRF utilizes an optimal transport approach to align the posterior mean with the ground-truth distribution, enhancing the accuracy of image restoration.
+ The paper provides theoretical foundations that demonstrate PMRF’s ability to achieve lower MSE compared to traditional posterior sampling methods.
+ The authors conduct extensive experiments across multiple image restoration tasks, showcasing PMRF’s effectiveness and robustness compared to state-of-the-art techniques.

### Weaknesses
- The recent FlowIE work also employs rectified flow for image restoration. I noticed that this paper references FlowIE in the supplementary material. It would be beneficial to include a discussion of this related work and highlight the differences in the main paper.
- In Figures 1 and 4, there are some unnaturally abrupt reflective spots or areas on the forehead, inner eye corners, and cheeks. Are these artifacts caused by the Posterior-Mean Rectified Flow? Please explain the reason behind these effects.

### Questions
There are noticeable reflective spots or areas on the forehead, inner eye corners, and cheeks. Are these artifacts a result of the PMRF ? Please clarify the cause of these effects. Additionally, are there any limitations to the proposed method? Could this design be adapted for more general image restoration tasks beyond photo-realistic images?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper starts from the distortion-perception tradeoff in image restoration. It first generates the posterior mean and then transports the result to a high quality image following the ground-truth image distoration. The first stage is done by a traditional end-to-end model SwinIR trained with L1 loss, while the second stage is implemented with a rectified flow model by approximating the desired optimal transport map.

### Strengths
1, It minimizes the MSE under a perfect perceptual index constraint and achieves a smaller MSE (higher fidelity) than posterior sampling methods (e.g., diffusion-based methods directly restoraing the image from noises).

2, With such a two-stage model, it could be easy-to-train (training a L1 loss-based model is stable) and fast-to-inference (both of the two stages are few-step models).

3, This paper is well-motived, well-written, well-supported and easy-to-understand.

### Weaknesses
1, What’s the performance of the baseline model trained the L1 loss? Will it achieve the best PSNR?

2, In addition to distortion metrics, could you please the model’s ability for human face identity perservation?  

3, What are the inference speed, memory consumption and FLOPs?

4， The core idea of generating a L1 loss result and then transport it to visually-pleasing image is somewhat similar to the idea of [1] (using the normalizing flow to generate high-quality images conditional the low-quality input). What are their relations?

[1] Hierarchical Conditional Flow: A Unified Framework for Image Super-Resolution and Image Rescaling

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
3
