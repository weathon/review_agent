# How and how well do diffusion models improve adversarial robustness?

- Decision: Reject
- Scores: 5, 5, 8, 5, 5

## Abstract
Recent findings suggest that diffusion models significantly enhance empirical adversarial robustness. While some intuitive explanations have been proposed, the precise mechanisms underlying these improvements remain unclear. In this work, we systematically investigate how and how well do diffusion models improve adversarial robustness. First, we observe that diffusion models intriguingly increase—rather than decrease—the $\ell_p$ distances to clean samples. This is the opposite of what was believed previously. Second, we find that the purified images are heavily influenced by the internal randomness of diffusion models. To properly evaluate the robustness of systems with inherent randomness, we introduce the concept of fuzzy adversarial robustness, and find that empirically a substantial fraction of adversarial examples are fuzzy in nature. Finally, by leveraging a hyperspherical cap model of adversarial regions, we show that diffusion models increase robustness by dramatically compressing the image space. Our findings provide novel insights into the mechanisms behind the robustness improvements of diffusion-model-based purification and offer guidance for the development of more efficient adversarial purification systems.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper presents several observations and explanations for DiffPure, mainly summarized as follows:

- Diffusion models increase—rather than decrease—the ℓp distances to clean samples.
- The concept of fuzzy adversarial robustness is introduced.
- A hyperspherical cap model of adversarial regions is proposed.
- It is shown that diffusion models increase adversarial robustness by compressing the image space.

### Strengths
- By measuring the distance between denoised examples and adversarial images, this paper shows that DiffPure actually increases the distance between denoised images and adversarial examples, rather than decreasing it, which contradicts the previous claim that "DiffPure increases robustness by decreasing the perturbation budget." Although I believe **Carlini et al., Xiao et al., and Nie et al. have never made this claim**, I still consider this experiment insightful, as many people agree with this idea.

- I strongly appreciate the authors' experiments demonstrating that "randomness in diffusion determines the purification direction but not the randomness in the input." I believe this is insightful for diffusion researchers.

### Weaknesses
This paper contains several ambiguities, overclaims, and misunderstandings of previous work (see both Weakness and Questions sections), but these can likely be quickly addressed during the rebuttal phase. I would consider raising my score if the authors address these issues.

- I disagree with the authors' claim that "a model simply encoding every clean image as the prior mode may not generalize well" and "the results above indicate that diffusion models are ineffective in removing small perturbations." A large distance between diffusion-purified images and real images does not support this claim. Imagine an image of a panda; the direction of hair growth in pandas can vary, so even if the hair direction changes, the image remains realistic. Such subtle changes often occur after diffusion denoising, causing an increase in ℓp distance, but not due to poor generalization.

- In Fig. 2 (a) and (b), the term "correlation" (e.g., 0.2368, 0.9954 in the paper) is not clearly defined. I'm still unsure of the x and y axes or what is meant by "correlation" here (covariance? cosine similarity? normalized correlation?).

- "Some treated randomness as a bug and proposed methods to cancel its effect in evaluation (cite Carlini et al.)." Carlini never stated this. What Carlini means is that randomness may obscure gradients, making evaluations challenging, but Carlini does not claim that "randomness is a bug" or that randomness inherently negates robustness.

- The authors' definition of "fuzzy adversarial examples" is already extensively discussed in the context of certified robustness via randomized smoothing. You should cite at least [1, 2], as their definition \( g(x)_c = \text{Pr}(f(x) = c) \) closely aligns with yours. While I recognize the differences between your definition and theirs, citing these works is essential for academic integrity.

---

**References**  
[1] Cohen, Jeremy, Elan Rosenfeld, and Zico Kolter. "Certified adversarial robustness via randomized smoothing." International Conference on Machine Learning. PMLR, 2019.  
[2] Salman, Hadi, et al. "Provably robust deep learning via adversarially trained smoothed classifiers." Advances in Neural Information Processing Systems 32 (2019).

### Questions
- I'm unclear about Fig. 1(c). The x-axis is labeled as the diffusion noise level, but what is the y-axis? Is it the distance between noisy examples and the original example, or between denoised examples and the original example? Additionally, why does the image become clearer rather than blurrier as \( t \) increases from 50 to 200? Are these images noisy examples or denoised ones? I can't fully understand this figure, and as a result, I'm unable to accurately interpret lines 162-201.

- "We removed the forward process by only performing denoising and still observed that the distances to clean samples increased." Could you please include this figure? In my opinion, diffusion will produce collapsed images if the input noise level does not align with the diffusion network's noise level condition. I’m really curious to see the result of this experiment.

- "Under these assumptions, one can mathematically show that the adversarial regions should form a hyperspherical cap when considering the \( \ell_2 \) neighborhood." In addition to including the proof in the appendix, I strongly recommend that the authors provide some intuition in the main paper to help readers understand why this holds. Could the authors give further intuition on this result during the rebuttal?

- In Fig. 5(b), does the y-axis represent "variance explained" as the eigenvalue? What is meant by "explained" here?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper examined the properties of diffusion models used in defensed against adversarial attacks. The authors showed that diffusion models do not shrinkage the distance of the transformed images toward clean images. Instead, diffusion models force the images to a compressed latent space, thus enhancing adversarial robustness. Altogether, this paper provides an explanation as to why diffusion models improve empirical adversarial robustness.

### Strengths
- This paper made an interesting observation on adversarial defenses based on diffusion models, where the purified images have larger l2 distance to the clean images than the adverasrial images. This suggests that diffusion models do not increase robustness by simply removing noise.  
- The paper is well-written and motivated, providing clear empirical evidence and concrete discussion as to how diffusion models work on improving adversarial robustness.

### Weaknesses
- In Figure 1, the authors show that the l_2 distance to the original clean images increase after diffusion purification. However, the authors do not consider the effect of the sampling process of diffusion models, e.g., deterministic vs random, or more advanced sampling method [1]. It is encouraged to validate the observation on more sampling methods of diffusion models.  
- It might be good to have some practical discussions on how the observations/understanding from this paper could contribute to the development of future defenses, or even improving existing diffusion model based defenses.  

[1] Elucidating the design space of diffusion-based generative models. 2022.

### Questions
- Could the authors evaluate the one-shot denoising approach as used by [1] and validate if the claim that l2 distance between purified images and clean images decrease as compared to that between adversarial images and clean images?  
- As indicated by [1], one reason why diffusion model could provided adversarial robustness is that the denoised images can be well classified by the base classifier. Therefore, would it possible that the actual distance decrease under a perceptual based losses [2]?  

[1] Certified adversarial robustness for free. Carlini et al, 2023.  
[2] Perceptual losses for real-time style transfer and super-resolution. Johnson et al, 2016.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
As mentioned in the title, this paper aims to investigate how and how well diffusion models improve adversarial robustness. Specifically, this paper first demonstrates that diffusion models push the purified images away from clean images, which challenges the previous belief (i.e., diffusion models will push adversarial images closer to clean images).  Then, this paper shows that the randomness largely influences the robustness of diffusion models. Based on this, this paper introduces a new concept called 'fuzzy adversarial examples', which considers the probability of adversarial attack fooling the system and proposes a new robust evaluation metric called 'cumulative fuzzy robustness (CFR)' to evaluate the robustness of fuzzy adversarial examples. Lastly, this paper uses a hyperspherical cap model to show that diffusion models improve robustness by shrinking the image space.

### Strengths
1. This paper is very well-written and easy to follow.

2. This paper provides sufficient insights (as mentioned in the summary) on diffusion-based adversarial purification, which can inspire researchers in this area to develop more advanced defense methods. More importantly, this paper closes an important research gap on how diffusion-based adversarial purification actually works. I would also like to hear other reviewers' opinions on the contributions of this paper.

3. The concept of fuzzy adversarial examples is novel and intuitive. Based on this, the proposed cumulative fuzzy robustness (CFR) is a more suitable evaluation metric to examine the robustness of a system with inherent robustness (especially for diffusion-based adversarial purifications).

### Weaknesses
1. While the observation that the diffusion model further pushes adversarial examples away from clean examples is intriguing, this paper lacks a theoretical explanation of why this occurs. However, this is only a minor weakness, as including a theoretical explanation is not always possible.

2. Fuzzy robustness evaluation is only performed on DiffPure, which is a bit outdated. Authors are encouraged to evaluate more recent diffusion-based adversarial purification methods to make results more convincible (e.g., [1] [2]).

[1] DensePure: Understanding Diffusion Models Towards Adversarial Robustness, ICLR 2023.
[2] Robust Evaluation of Diffusion-Based Adversarial Purification, ICCV 2023.

### Questions
1. What is the PGD+EOT result on ImageNet? 

2. Just curious—do you think the conclusions in this paper would hold up if diffusion models are constructed in latent space instead of pixel space? And why?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This is an analytical work to investigate how diffusion-based purification (DBP) improve the adversarial robustness. With pilot experiments, this work finds that diffusion models increase the lp distances to clean samples and the purified images are heavily influenced by the internal randomness of diffusion models. Furthermore, this work introduces the concept of fuzzy adversarial robustness and hyper-spherical cap model of adversarial regions, and gives an explanation on how DBP works.

### Strengths
1.	Overall, this paper is well-written, with clear organizations and illustrations.
2.	The finds in this work are reasonable, with sufficient explanations and experimental supports.

### Weaknesses
My major concern on this work is about the contribution of this work:

1)	It is not surprising that diffusion-based purification will guide the adversarial images to a place with a larger distance to the original images. As the adversarial perturbation is usually small and the corrupted process (adding Gaussian noise) will make it hard to reverse it to the original image (information on the original image is missing). According to [1,2], the reversion process of diffusion models is to make the generated image has a higher probability to be close to the original image distribution, which does not mean this image should be closer to the original image. Below are some suggested experiments to make this work more comprehensive:

- a)	Besides traditional l_p distance, the metrics for generative models (e.g., FID score, Inception score) should also be used to evaluate the distance. In such latent space, the conclusion might be different.

- b)	Lines 171-172. I think this experiment is important and should not be hidden.

- c)	Early denoiser-based defense [3] has been shown to be ineffective while diffusion-based purification is effective. It is important to show if early denoiser-based defense has the similar behavior (enlarging the lp distance) to diffusion models. If they are similar, the findings in this work may not explain why they have different defense performance.

2)	The proposed fuzzy adversarial robustness (a new framework in evaluation) is similar to [1] while the discussion on it is missing.
3)	This work claims that the findings offer guidance for the development of more efficient adversarial purification (AP) systems, but no deeper discussions and experiments are provided. If this work can give a prototype method of more efficient AP with the findings, I could have improved my score.

Line 450: $\textbf{x}_0$， line 269: $\textbf{x}^{`}$

[1] Xiao C, Chen Z, Jin K, et al. Densepure: Understanding diffusion models towards adversarial robustness[J]. arXiv preprint arXiv:2211.00322, 2022.

[2] Chen H, Dong Y, Wang Z, et al. Robust classification via a single diffusion model[J]. arXiv preprint arXiv:2305.15241, 2023.

[3] Liao F, Liang M, Dong Y, et al. Defense against adversarial attacks using high-level representation guided denoiser[C] CVPR. 2018: 1778-1787.

[4] DiffHammer: Rethinking the Robustness of Diffusion-Based Adversarial Purification. NeurIPS, 2024

### Questions
Please see the weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper provides theoretical and empirical studies on the mechanism of diffusion-based adversarial purification methods. First, based on empirical studies on the behavior of diffusion-based purification models, it is suggested that the purification generally increases the $\ell_p$ distance of the input sample to the clean sample, and that the purification results are affected by randomness in diffusion models. Second, the concept of fuzzy robustness and the corresponding evaluation method are proposed for diffusion-based purification methods. Third, a hyperspherical cap model is introduced to depict the adversarial regions. Finally, it is argued that the robustness brought by diffusion-based purification is due to the compressing of image space.

### Strengths
- The observation of the increasing $\ell_p$ distance to the clean sample produced by diffusion-based purification is interesting.
- The definition of fuzzy adversarial robustness is meaningful and the proposed CFR curve can be a practical evaluation metric for stochastic defense methods.
- The figures clearly demonstrate the ideas of the paper.

### Weaknesses
- The description of the previous theoretical result in (Nie et al., 2022) is unfaithful.
  - In Section 3.1, it is stated that the previous explanation of the robustness of diffusion-based purification methods lies in the "shrinkage of the adversarial attack", or more specifically, the decreasing $\ell_p$ distance to the clean sample after purification.
  - Nonetheless, Theorem 3.1 in (Nie et al., 2022) only suggests that the divergence between the clean data distribution and adversarial sample distribution can be decreased by the *forward diffusion process*. It is not assumed that the purified sample obtained by the *diffusion-denoising purification process* is closer to the clean sample under $\ell_p$ distance.
  - From my perspective, while the purified sample is expected to lie on the non-adversarial manifold, its $\ell_p$ distance to the clean sample is not important as long as the semantic information is preserved.
- The hyperspherical cap model is not well supported.
  - Crossing the critical threshold along an adversarial direction can be a necessary condition for adversarial samples, but it is insufficient. Specifically, Assumption 2 is valid locally given the continuity of the model, but there can be an upper bound on radius $\hat{\gamma}(x_0, \eta)$ determined by $x_0$ and the direction $\eta$ for this assumption to hold. In other words, the sample may cross the boundary again as the radius exceeds $\hat{\gamma}(x_0, \eta)$. An example of such a decision boundary is depicted in Figure 1 of [1].
  - Therefore, unless a non-trivial uniform upper bound $\hat{\gamma}$ is derived, it cannot be claimed that crossing the critical threshold is a sufficient condition for adversarial examples within an $\ell_2$ neighborhood with a certain radius.
  - It is also assumed that "the classification boundaries are locally linear" (Lines 375-376), which is not supported by valid theoretical or empirical evidence. While the decision boundaries depicted in the 2D projection in Figure 4(b,c) appear to be linear, they are likely non-linear in the high-dimensional input space. Stronger evidence or proper references are required to claim this point.
- The causal relationship between the compression of image space and the improved robustness is not well explained.
  - Section 6 has validated that diffusion-based purification can compress the image space, but it's still not apparent how it contributes to the robustness. For example, while the magnitude of the critical threshold can be reduced due to the compression, the purified sample $f(x_0)$ may also lie closer to the decision boundary, which cannot explain the improved robustness.
  - It is better to clarify whether the compression effect is the sole contributor to the robustness of diffusion-based purification models.
- The little-o notation in Line 717 is inappropriate since two real numbers are compared here instead of two growing functions. The "considerably larger" change of logit should be better defined.

[1] Kim, Hoki, Woojin Lee, and Jaewook Lee. "Understanding catastrophic overfitting in single-step adversarial training." AAAI 2021.

### Questions
- If image space compressing is sufficient for improving the robustness, are conventional image compressing methods like JPEG effective for adversarial defense?

### Soundness
2

### Presentation
3

### Contribution
2
