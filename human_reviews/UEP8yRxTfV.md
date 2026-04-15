# Debias the Training of Diffusion Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 8, 3, 1

## Abstract
Diffusion models have demonstrated compelling generation quality by optimizing the variational lower bound through a simple denoising score matching loss. In this paper, we provide theoretical evidence that the prevailing practice of using a constant loss weight strategy in diffusion models leads to biased estimation during the training phase. Simply optimizing the denoising network to predict Gaussian noise with constant weighting may hinder precise estimations of original images. 
To address the issue, we propose an elegant and effective weighting strategy grounded in the theoretically unbiased principle. 
Moreover, we conduct a comprehensive and systematic exploration to dissect the inherent bias problem deriving from constant weighting loss from the perspectives of its existence, impact and reasons. These analyses are expected to advance our understanding and demystify the inner workings of diffusion models. Through empirical evaluation, we demonstrate that our proposed debiased estimation method significantly enhances sample quality without the reliance on complex techniques, and exhibits improved efficiency compared to the baseline method both in training and sampling processes.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper demonstrates that the use of a constant loss weight strategy in traditional diffusion models leads to biased estimation during the training phase. To remedy this, this paper proposes a weighting strategy grounded in the theoretically unbiased principle to address this problem. Furthermore, it conducts a thorough and systematic investigation to analyze the inherent bias issue resulting from constant weight loss from multiple perspectives. Finally, the effectiveness of the method proposed in this paper is confirmed through experiments.

### Strengths
1.This paper exhibits a concise and lucid narrative style in presenting the methodology, rendering the proof process accessible to a wide readership.

2.The proposed methodology unequivocally enhances the performance of diffusion models with much less training iterations and sampling steps.

3.This paper offers a comprehensive exposition of the bias issue within traditional diffusion models, thereby facilitating a deeper comprehension of diffusion models.

### Weaknesses
1.This study exhibits a deficiency in the comprehensiveness of its experiments, and it lacks validation on commonly used benchmark datasets, such as CIFAR-10 and ImageNet. Is this attributed to limitations in the scalability of the proposed methodology? I suggest to conduct more experiments to prove the scalability of the proposed method.

2.In equation (8), the coefficient $\frac{1}{2\sigma ^{2}}$ has been omitted. Though it has no effect on the overall proof, it is better to present this one.

3.This paper, while providing an exposition on biased estimations in diffusion models, places greater emphasis on comparative illustrations with experimental results, thereby rendering the theoretical evidence somewhat less robust.

4.How is the stability of the trainging process? Can you achieve consistent results in every experiment with identical initial conditions?

5.What is the difference of the weighting schedule campare to [1]? It seems the two are very similar and [1] consider more complex situation.

[1] J. Choi, J. Lee, C. Shin, S. Kim, H. Kim, and S. Yoon. Perception prioritized training of diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11472–11481, 2022.

### Questions
The same as Weaknesses.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper discusses the training weighting of the loss of diffusion models. They theoretically find that original constant weighting strategy is suboptimal, and further propose an improved training loss weight strategy. Besides, they give in-depth analyses of the sub-optimality of the constant weighting strategy from the perspective of existence, impact and reasons. The effectiveness of the proposed method is verified on several datasets.

### Strengths
1. It is an interesting idea to analyze the sub-optimality of the training loss weight. This paper theoretically reveal the inherent bias of constant weighting strategy, and propose a debiased principle on the design of the training weight.
2. The in-depth analyses of the sub-optimality of the constant weighting strategy from the perspective of existence, impact and reasons are impressive and insightful. These analyses provide valuable insights and inspirations on the opague generation process of diffusion model.
3. The experiments are solid. The proposed method gains substantial performance improvement on several datasets via simply modifying the training loss weight. The reproducibility is well guaranteed.
4. This paper is well-organized and easy to read and understand.

### Weaknesses
1. Allocating higher weight to large t will improve the overall performance. While, what is effect of allocating lower weight at small t. It seems that the MSE error is slightly higher at t=0 than the constant weighting strategy in fig. 4.
2. The visual difference between different baselines and the proposed method seems not obvious in fig. 5.
3. Minor suggestions. The “DDPM” is also used to denote constant weight in fig. 7. The authors are advised to general denotation to avoid confusion.

### Questions
I am curious about the performance of treating the original image as training target and its comparison with noise-prediction mode.

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
4 excellent

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This submission deals with the bias of using constant weights for the denoising loss of diffusion models in the training phase …it first identifies the biased generation issue as a result of constraint weighting that results in artifacts such as poor details, global inconsistency, and color shift. It then proposes a new SNR-based weighting mechanism that lifts the diffusion error to the image space and thus debiases the generation. Experiments with FFHQ, AFHQ-dog, and MetFaces show significant FID gains and sampling steps compared with constant weighting.

### Strengths
Improving the training efficiency and sampling quality of diffusion models is a timely topic

Experiments and comparison are extensive and the gains are significant

### Weaknesses
The paper misses related work in the literature that have already proposed the idea of SNR-weighting for diffusion models. Glancing through the literature, this reviewer found this related work in [1] that proposes the SNR weighting with the same justifications and derivations for sampling. The major novelties need to be clarified. In particular, the derivations from eq. 9 and 10 are already proposed in [1]. 

[1] Mardani M, Song J, Kautz J, Vahdat A. A Variational Perspective on Solving Inverse Problems with Diffusion Models. arXiv preprint arXiv:2305.04391. 2023 May 7.

### Questions
The abstract is vague and high level. The main idea which is the SNR-based weighting mechanism is not explained well. 

Do you use SNR weighting for the sampling phase as well?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
1: strong reject

### Rating Number
1

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper studies the biased estimation problem of diffusion models, by examining the flaw in the $\epsilon$-prediction. The authors also conduct several empirical analyses to support the bias effect. Empirically, the proposed objective achieves better FID scores across facial datasets.

### Strengths
- The paper examines the potential bias problem when using $\epsilon$-prediction.

- Empirically, the proposed weighting scheme outperforms previous ones.

### Weaknesses
- **No novelty**: It seems that the paper reinvents a well-established objective in the diffusion models literature -- $x_0$ prediction (see the blog https://medium.com/@zljdanceholic/three-stable-diffusion-training-losses-x0-epsilon-and-v-prediction-126de920eb73). The paper "rediscovered" the relation between $x_0$-prediction and $\epsilon$-prediction. The proposed objective in Eq.11 is actually doing $x_0$-prediction type loss: by setting $\epsilon_\theta = \frac{x_t-\hat{x}_0}{\sigma}$ and $\epsilon=\frac{x_t-x_0}{\sigma}$ one could recover the $x_0$-prediction loss.

One step further, there are already works focusing on combining the strengths of $x_0$-prediction and $\epsilon$-prediction, like the $v$-prediction [1] and the pre-conditioning techniques in EDM [2], in the past year.

- The FID score in Table 1 is way too high in the small NFE regime (NFE<100). It makes the comparison much less convincing.


[1] Progressive Distillation for Fast Sampling of Diffusion Models, Salimans et al.

[2] Elucidating the Design Space of Diffusion-Based Generative Models, Karras et al.

### Questions
N/A

### Soundness
1 poor

### Presentation
2 fair

### Contribution
1 poor
