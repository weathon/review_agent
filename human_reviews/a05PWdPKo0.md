# PAR-AdvGAN: Improving Adversarial Attack Capability with Progressive Auto-Regression AdvGAN

- Avg Score: 4.50
- Decision: Reject
- Scores: 5, 5, 5, 3

## Abstract
Deep neural networks have demonstrated remarkable performance across various domains. However, they are vulnerable to adversarial examples, which can lead to erroneous predictions. Generative Adversarial Networks (GANs) can leverage the generators and discriminators model to quickly produce high-quality adversarial examples. Since both modules train in a competitive and simultaneous manner, GAN-based algorithms like AdvGAN can generate adversarial examples with better transferability compared to traditional methods. However, the generation of perturbations is usually limited to a single iteration, preventing these examples from fully exploiting the potential of the methods. To tackle this issue, we introduce a novel approach named Progressive Auto-Regression AdvGAN (PAR-AdvGAN). It incorporates an auto-regressive iteration mechanism within a progressive generation network to craft adversarial examples with enhanced attack capability. We thoroughly evaluate our PAR-AdvGAN method with a large-scale experiment, demonstrating its superior performance over various state-of-the-art black-box adversarial attacks, as well as the original AdvGAN. Moreover, PAR-AdvGAN significantly accelerates the adversarial example generation, i.e., achieving the speeds of up to 335.5 frames per second on Inception-v3 model, outperforming the gradient-based transferable attack algorithms. Our code is available at: https://anonymous.4open.science/r/PAR-01BF/

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
To further enhance the performance of GAN-based attacks, the paper proposes Progressive Auto-Regression AdvGAN (PAR-AdvGAN), which incorporates an auto-regressive iteration mechanism to maximize attack effectiveness with minimal perturbation. Experimental results show that compared to the existing various attack methods, PAR-AdvGAN achieves the SOTA attack performance, significantly accelerates the generation of adversarial examples, and has good transferability performance.

### Strengths
1. The paper is written clearly and is easy to understand.
2. Extensive experiments demonstrate that, compared to some gradient-based attacks, PAR-AdvGAN exhibits good transferability performance.

### Weaknesses
1. The paper lacks some important experimental results. The evaluation of attack performance only involved a set of comparative experiments, as shown in Table 1. This does not well support the strong attack capability of the proposed method, and it is necessary to include comparative results of various attacks, such as PDG, AutoAttack, DiffAttack, etc. Especially since diffusion model is gradually surpassing GAN model in various aspects, what are the advantages of GAN-based attack compared to diffusion-based attack? Moreover, there is a lack of experiments on more datasets such as CIFAR-10 and CIFAR-100, where simpler attacks like PGD have already achieved nearly 100% ASR. It is necessary to evaluate whether PAR-AdvGAN remains the best performance on these datasets.
2. The paper overly emphasizes some advantages that are not unique to PAR-AdvGAN. For example, in the inference phase, it does not require gradient calculations to attack (lines 352-361); and the time required for attacks is much shorter than that for gradient-based attacks (Section 4.4). These are advantages of GAN-based methods in general, not unique advantages brought by the proposed method.
3. The term “perturbation rate” first appears in line 339, but without explaining. What is the difference between perturbation rate and the scale of perturbation (epsilon)? In methods like FGSM and PGD, the scale of perturbations is directly constrained by setting epsilon. Why is the “perturbation” different in Tables 2-6?
4. In Algorithm 1: The target network N is missing in the cross entropy calculation. Additionally, I suggest moving “here $x_{adv}^0 = x_0$” and “update the ...” outside of the for loop to more clearly explain their roles. And I noticed that throughout the process, epsilon is not used to constrain the scale of perturbations. Although the algorithm aims to achieve minimal perturbation by $L_d$, this does not guarantee that the final perturbations will definitely be smaller than epsilon.
5. The reason for transferability is unclear. Although section 4.3.6 provides some simple analysis of the results, it still does not explain the source of transferability. Additionally, the paper lacks ablation experiments, which are crucial to understanding the impact of each component on the attack performance and transferability performance.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The proposed PAR-AdvGAN in this paper enhances the transferability of adversarial examples generated by the baseline AdvGAN.

### Strengths
1 The writing is clear.

### Weaknesses
1 In Table 7, why isn't the baseline AdvGAN included for comparison? This is because the processing speed of our algorithm may be attributed to the baseline AdvGAN rather than to the proposed progressive autoregressive method. I suggest that the authors add AdvGAN to Table 7 and discuss how PAR-AdvGAN's speed compares specifically to AdvGAN. This would help clarify the speed improvements due to the progressive autoregressive method versus the baseline AdvGAN approach.

2 The latest transfer attack methods from the past five years are not included for comparison such as [1][2].

[1] Wang X, He K. Enhancing the transferability of adversarial attacks through variance tuning[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021: 1924-1933.
[2] Ge Z, Liu H, Xiaosen W, et al. Boosting adversarial transferability by achieving flat local maxima[J]. Advances in Neural Information Processing Systems, 2023, 36: 70141-70161.

3 What are the application scenarios for excessively fast processing speed? Discussing potential real-world applications or using cases where the high processing speed of PAR-AdvGAN would be particularly beneficial or necessary. This would help contextualize the importance of the speed improvements and strengthen the paper's motivation.

4 The font size in Figures 1 and 2 is too small.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper presents Progressive Auto-Regression AdvGAN (PAR-AdvGAN), an innovative approach that enhances the generation of adversarial examples using Generative Adversarial Networks (GANs). PAR-AdvGAN addresses this limitation by integrating an auto-regressive iteration mechanism within a progressive generation framework, resulting in adversarial examples with improved attack capabilities. The performance of PAR-AdvGAN is thoroughly evaluated through extensive experiments, showing superior results compared to various adversarial attacks and the original AdvGAN.

### Strengths
1. The author questions that the generation of perturbations is usually limited to a single iteration, thus degrading the attacking performance, which looks like an interesting issue.
2. The paper is well-written and easy to follow.
3. Extensive experiments are performed to evaluate the proposed PAR-AdvGAN, compared to other baselines.

### Weaknesses
1. The paper argues that the limitations of the previous GAN method for generating perturbations stem from its reliance on single-step sampling. While the experiments demonstrate significant improvements, I question whether this is the primary reason for the shortcomings of the previous method (probably the adversarial features are not well-learned). To further validate this assumption, what would happen if we incorporated additional time steps into the training of the GAN, such as concatenating 2 or 3 time-step images? It would be more convincing to expect further improvement.
2. This approach seems analogous to diffusion models, which are not sufficiently discussed in the paper (never mentioned even once), despite the existence of related work on adversarial attacks utilizing diffusion techniques.
3. The comparison of FPS appears somewhat unfair, as training the GAN (for 60 epochs, as mentioned in the paper) brings an additional cost that should be considered, while other gradient-based methods don't. Otherwise, the claim should only account for the time required for deployment (after training).
4. The authors raise the question RQ3 but I don't see a clear explanation why it works.
5. The proposed method PAR-AdvGAN mainly relies on the Inception and Resnet model as a source model, it would be good to explore more different architectures.

### Questions
1. Typo in line 330 "gaussian"
2. How to read Table 1? It seems the bold represents the result of the PAR-AdvGAN and not all of them are the best.
3. Overall the motivation to me is good and clear. However, a deeper insight would be appreciated to justify why the previous method fails and to explain the necessity of the Progressive Auto-Regression process.
4. The author mentioned AdvGAN++ but did not compare it in the experiment part. Also, are there other improved variants for AdvGAN?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper extends previous GAN-based adversarial sample generation works from one-step generation to multiple-step generation. The overall technical novelty is limited, since the proposed method is a trivial extension from previous AdvGAN, simply increasing the generation step from one to $n$. 

Non-trivial performance gains are shown in terms of attack success rates compared with some previous method in most cases. The exception is in Table 4, where SINI-FGSM has tiny advantage over the proposed method. Some important baseline methods (i.e. diffusion model-based attacks) are not considered in the experiments. 

Overall writing is smooth and easy to follow, with some minor issues to improve for better presentation.

### Strengths
1. Overall writing is smooth and easy to follow.

2. The proposed method to extend AdvGAN to multiple-step attack is intuitively reasonable. 

3. Non-trivial performance gains are shown in terms of attack success rates compared with some previous method in most experiments. In Table 4 SINI-FGSM has tiny advantage over the proposed method. But this is not any issue to me, since the difference is tiny enough to be ignored and also the attackers can choose the model with best generalization ability as attack-generation model and don't have to stick with the model used in Table 4.

### Weaknesses
1. The technical novelty is limited. The extension over AdvGAN is trivial. 

2. When talking about multi-step generation, usually people would assume diffusion models to do better jobs than progressive GANs. Can diffusion models perform the same job as the proposed Par-AdvGAN? There are multiple paper applying diffusion models to generate adversarial samples like this one [1,2]. It seems to me they can be adopted for the same purposed as Par-AdvGAN. I would suggest to add them as important baseline methods in the experiments, if you find it reasonable. Otherwise, please discuss why they can't be used. 

[1] Diffusion-Based Adversarial Sample Generation for Improved Stealthiness and Controllability. 

[2] SD-NAE: Generating Natural Adversarial Examples with Stable Diffusion. 

3. In Table 7, why not include AdvGAN when comparing attack latency? I assume AdvGAN should be faster than Par-AdvGAN. The fast attack speed actually comes from the intrinsic advantage of generative model-based attack methods over optimization-base method, instead of the novelty of Par-AdvGAN. 

4. Line 9 and line 18 in table 5 is confusing. Shouldn't they be put into comments instead two extra steps of the algorithm?

### Questions
Please see above.

### Soundness
3

### Presentation
3

### Contribution
2
