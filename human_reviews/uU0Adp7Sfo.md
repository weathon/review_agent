# Competitive-Collaborative GAN with Performance Guarantee

- Decision: Reject
- Scores: 5, 3, 3, 3

## Abstract
Generative Adversarial Networks (GANs) generate data based on a competition game to minimize the distribution distance between existing and new data. However, such a competition game falls short when insights about data distributions beyond their authenticity are imperative, such as in multi-modal generation and image super resolution. In recognition of the limitations inherent to the pure-competitive mechanism, we introduce CCGAN, a Collaborative-Competitive Generative Adversarial Network scheme to enable data generation with additional knowledge beyond the provided dataset distribution. For theoretically preserving the equilibrium point and numerically avoiding training collapse issue, we show the need to convert regularization term into a divergence, so that the modified GAN is well-defined in game theory. By harmonizing the competition and collaboration losses in CCGAN, we effectively reduce the degree complexity of solving the optima, facilitating the establishment of a closed-form equilibrium point. This equilibrium point serves as a guidance for training and hyper-parameter tuning, resulting in consistently high-quality generated samples. Meanwhile, the regularization breaks the mutual dependency between the generator and discriminator. This newfound independence empowers the CCGAN to explore a broader parameter space, effectively mitigating the training collapse issue. To validate the capabilities of CCGAN, we design comprehensive experiments across four publicly available datasets and systematically compare CCGAN against a range of baseline models. The experiments demonstrate the efficacy of CCGAN on generating satisfactory samples tailored to specific requirements, particularly when applied to the generation of images featuring regularly shaped objects.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
GAN uses a competition game to generate new data that mimics authentic data, but this method struggles with complex tasks like multi-modal generation and image super-resolution. To address these challenges, CCGAN is introduced, using a Collaborative-Competitive approach that allows for better data generation by balancing competition and collaboration, thereby establishing a more stable equilibrium for training and hyper-parameter tuning. 

Through testing across multiple datasets, CCGAN has proven effective in generating higher-quality images of regularly shaped objects, by exploring a wider parameter space independent of traditional generator-discriminator dependency.

### Strengths
- Promising idea support by theory proofs 
- The newly introduced method does not cause additional significant computational cost. 
- Nice application demo on multiple vision tasks.

### Weaknesses
1. Formatting is not quite right. 

2. There are existing multiplayer methods to improve the stability of GAN training, which changes the original GAN competition setting by introducing additional players. Would like to hear some words from the authors about the advantages or disadvantages compared methods in this setting. 
- https://arxiv.org/abs/2101.07524
- https://arxiv.org/abs/1907.02690
- https://arxiv.org/abs/1709.03831

3. Over claim of the visual results
- All the visual results are too small. I can’t really see the difference unless I zoom hard on a giant monitor, which is not possible from the printed version. 
- Figure 3 column 3,4: Not really better than baseline
- Figure 4b: I can’t see what the original mask is, again too small to show the details.

### Questions
1. It sounds like the proposed method should work with the original GAN method. Why not a direct comparison with it on an image generation task with FID / IS scores? 

2. “We prove that the generator and discriminator is not mutually dependent in CCGAN” Can you combine it with SOTA methods like StyleGAN3?

3. Training details are missing from the experiment section. Resolution? Sample size?

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes a new GAN model known as Collaborative-Competitive GAN (CCGAN), aimed at improving existing GANs in tasks such as multi-modal generation and image super-resolution. The authors argue that CCGAN overcomes the limitations of pure-competitive GAN models and other existing collaborative GANs. Essentially, the authors design collaborative regularization as the divergence metric to align the loss functions, reduce complexity in finding optima, and derive closed-form equilibrium points. The experiments are conducted on benchmark datasets for image-image translation, image super-resolution, and a toy dataset for image colorization. They are compared to a few baselines, including COCO-GAN, CollaGAN, and Co-GAN. Additionally, the paper validates the proposed GAN against false data injection attacks. 

The design of collaborative regularization is not entirely new; it bears resemblance to the popular "trick" of label smoothing used by the community to enhance GAN stability. There is a question about whether the original GAN loss, combined with regularization, is necessary, as it alone may perform well. However, the paper makes a contribution with its theoretical analysis.

Overall, the paper's structure is easy to follow, and the results show encouragement when compared to some existing GAN methods. Nonetheless, there are many concepts that are not well-defined, and the claims in the paper lack strong empirical support. Most of the experimental results are qualitative, and quantitative results are limited, lacking a comprehensive study on the most significant contribution of collaborative regularization. Many claims in the paper remain unsupported by experimental results. Detailed comments can be found below. I recommend rejecting the paper in its current form.

### Strengths
Inspired by the existing collaborative GAN model, the paper suggests replacing the regularization with a new one based on divergence metrics. This approach is similar to the original GAN, except the target is \lambda instead of 1. It seems to resemble the label smoothing "trick" commonly employed by the community to enhance GAN training. The primary contribution of the paper lies in its theoretical analysis, which seeks to demonstrate the convergence of the proposed method, although it remains somewhat limited in its current form.

### Weaknesses
1. The paper presents various claims together with some concepts which are not popular and without definition or explanation, some of which somewhat quite arbitrary. The authors argued that CCGAN overcomes the limitation of a pure-competitive GAN model and other existing collaborative GANs in terms of beyond the “authenticity”  of data distribution, incorporating “additional knowledge”. or “harmonizing” the loss functions, reducing “degree of complexity” in finding the optima and deriving closed-form which “serves as a guidance for training and hyper-parameter tuning”. 

2. The authors should clarify what form of “additional knowledge” does the proposed regularization add into? What is “authenticity” of the data distribution? What is “mutual dependency” and why is it helpful for GANs training to break this? In addition, could the authors conduct experiments to support claims of “harmonizing” the loss function, reducing the “degree of complexity” in finding the optima and how does the regularization “serves as a guidance for training and hyper-parameter tuning”?

3. As mentioned in the paper's introduction, it is not entirely clear to me why replacing the cross-entropy loss with the least square loss or Wasserstein loss fosters collaboration between the generator and discriminator. Could the authors provide clarification on the definition of collaboration and how it is enhanced with these losses compared to cross-entropy?

4. From mathematical perspective, it is unclear why $JSD (p_{data} || \frac{\lambda - 1}{\lambda} * p_{gen})$ is superior than $JSD (p_{data} || p_{gen})$ of original GAN? As the new JSD appears not optimal, regardless of the selected value of $\lambda$, JSD never converges to optimal values.

5. The regularization appears close to the label smoothing “trick” which is widely used in GAN training. This suggests regularization again might suffice as a GAN loss. Have the authors explored the impact on results when removing the GAN loss function in this study?
Can the author explain why the optimal D in Theorem 3 help CCGAN avoid the training collapse?

6. The experimental results are mostly the qualitative results and quite limited. For instance, only one quantitative result in Table 1 for image super resolution. The regularization is the key contribution which needs to be extensively studied but missing in the current form. Could the authors provide the ablation study to investigate the impact on $\lambda$ and how to select it, as well as how the performance changes with different values of $\lambda$? 

Typo in conclusion: presente

### Questions
See above.

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
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces CCGAN, a Collaborative-Competitive Generative Adversarial Network, to address limitations in GANs related to data distribution and additional knowledge incorporation. CCGAN harmonizes competition and collaboration losses, reducing complexity and achieving a closed-form equilibrium point for stable training. It successfully generates high-quality samples across various datasets, particularly excelling in generating images with regularly shaped objects.

### Strengths
The paper proposes to harmonize competition and collaboration losses, reducing complexity and achieving a closed-form equilibrium point for stable training. Presentation is quite clear and straightforward.

### Weaknesses
See questions below. Holds major concerns how the method will be applied and generalized.

### Questions
1. For condition generation like colorization, image translation and super-resolution, discriminator is usually not regularized, unlike eq.2. That mean, $\lambda=0$ for discriminator optimization. How to handle this situation? What's the formulation specifically for each task?

2. For Theorem 1, $\mathcal{L}_reg$ are mostly formulated as absolute error instead of squared loss. How does it affect the solution?

3. Following 1 and 2, the baselines like coco-gan, colla-gan and co-gan neither claimed benchmark on these task. it thus questionable about the experiments

4. In the abstract, it claims that "enable data generation with additional knowledge beyond the provided dataset distribution". Could it more specific? Didn't find evidence about it.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes to address the regularization problem when learning GANs. More specifically, after proving the absence of close-form equilibrium when adding regularizations, a specific regularization term has been proposed, in which the closed-form solution for the equilibrium exists. The established closed-form equilibrium is claimed to avoid the mode collapse issue when training GANs. The proposed GAN is named as the CCGAN, and experimental validations were performed regarding colorization, image-to-image generation tasks.

### Strengths
1. A closed-form equilibrium is obtained in this paper, given a specific regularization term added to the adversarial learning. 
2. Experimental results verify that the proposed CC-GAN works on several image generation tasks, including colorization, and image-to-image generation.

### Weaknesses
1. For me, the established regularization term is very ad hoc, without clarification on why it is constructed like that. As I read from the introduction, the authors seem to regularize GANs to learn partial modes or addition restrictions, the so-called learning beyond the authentic distributions. However, the established regularization term does not indicate those intensions. 
2. This paper claims that existing regularized GANs do not have closed-form equilibrium, which is proved based on the very simple Gaussian case. This is not very convincing. Even though, in my opinion, for training GANs, the way to approach the equilibrium is more important than the closed-form solution obtained by ad hoc regularization terms. It is well-known that for the vanilla GAN with closed-form equilibrium, we are not always ensured to get this equilibrium.
3. The experimental results are also confusing for me, even contradictory to the introduction. In the introduction, the authors claim that "in multi-modal generation Liu et al. (2021), the focus is on learning one or multiple modes within the data distribution.". However, in the experiments, the authors verify that the proposed CCGAN is able to learn all the modes in the synthetic datasets.
4. The verification is weak and not convincing for me. The comparing baselines are basically not designed for those experimental tasks. For example, COCO-GAN is designed for generating images, instead of image-to-image generation and super-resolution. By my understanding, the regularization in COCO-GAN is for allowing for generating by patches. If the authors wish to beat the COCO-GAN, metrics such as FIDs by generating patches should also be provided. Also for the super-resolution task, 30.43 dB is not generally a good PSNR. 
5. Why Theorem 3 proves that the CCGAN can avoid the mode collapse issue? The experimental results related to the mode collapse test are not convincing as well. The authors should present statistical results instead of illustrating several subjective results.

### Questions
Please see my weakness. Also why in Theorem 3, a new hyper-parameter \gamma appears in addition to \lambda? How \gamma avoids the mode collapse in theory?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
