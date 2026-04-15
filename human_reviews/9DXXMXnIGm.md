# Elucidating the design space of classifier-guided diffusion generation

- Decision: Accept (poster)
- Scores: 5, 6, 5, 6, 8

## Abstract
Guidance in conditional diffusion generation is of great importance for sample quality and controllability. 
However, existing guidance schemes are to be desired. 
On one hand, mainstream methods such as classifier guidance and classifier-free guidance both require extra training with labeled data, which is time-consuming and unable to adapt to new conditions.
On the other hand, training-free methods such as universal guidance, though more flexible, have yet to demonstrate comparable performance. 
In this work, through a comprehensive investigation into the design space, we show that it is possible to achieve significant performance improvements over existing guidance schemes by leveraging off-the-shelf classifiers in a training-free fashion, enjoying the best of both worlds. 
Employing calibration as a general guideline, we propose several pre-conditioning techniques to better exploit pretrained off-the-shelf classifiers for guiding diffusion generation. 
Extensive experiments on ImageNet validate our proposed method, showing that state-of-the-art (SOTA) diffusion models (DDPM, EDM, DiT) can be further improved (up to 20\%) using off-the-shelf classifiers with barely any extra computational cost.
With the proliferation of publicly available pretrained classifiers, our proposed approach has great potential and can be readily scaled up to text-to-image generation tasks.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper focuses on a very interesting question of guiding the diffusion modes by leveraging off-the-shelf classifiers in a training-free fashion. Specifically, the authors first provide a simple analysis to demonstrate that the off-the-shelf classification can achieve better accuracy than the fine-tuned classifier when the noise level is high, which may have been ignored in the previous works. Then, the authors turn to exploit the pre-trained classifier for guiding diffusion generation with comprehensive consideration of the detailed settings.

### Strengths
1.	In my opinion, the major contribution of this paper lies in Section 4.1, which provides an empirical analysis by evaluating the calibration of both fine-tuned and off-the-shelf classifiers.  The results reveal that fine-tuned classifiers are less calibrated than off-the-shelf ones when the noise level is high, which indicates that Off-the-shelf classifiers’ potential is far from realized.
2.	To optimize the design of the proposed methods, the authors have considered multiple aspects, including the classifier inputs, smoothing guidance, guidance direction, and guidance direction, and designed the corresponding strategies to improve the overall performance.

### Weaknesses
1.	 The paper focuses on the classic class-conditional diffusion generation, which is a simple case in the text-to-image generation. How about the performance of the proposed methods for the general case with text prompts as conditions? In particular, in section 5.4, the authors also provide a simple analysis of this case with the CLIP model. However, the results are not good.
2.	Human-level metrics should be involved for clearer comparisons. It is well-known that some quantitative metrics like FID, may be problematic in some cases. In particular, the FID metric, used in ablation analysis in section 4 and experiments in section 5, cannot measure conditional adherence which is important for conditional generation. CLIP score should also be considered to evaluate the performance.
3.	It is weird that the final images are merely the same as each other in Figure 4 with different settings of logit temperature, while the FID score varies with different settings in Table 4.

### Questions
My major commons lie in the experimental evaluation. Please provide more experimental analysis or discussions to validate the effectiveness and robustness of the proposed methods.

How about the performance in terms of CLIP score and Human-level metrics?

How about the performance for the conditional generation with the general text prompts?

Please provide more discussions about Table 4 and Figure 4.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The manuscript analyzes the limitations of current classifier guidance in terms of flexibility, calibration error as well as smoothness. Based on the analysis, the authors propose to use an off-the-shelf classifier instead of a noise-aware classifier for guidance which shows some encouraging results.

### Strengths
The problem is very interesting, the method is simple yet effective. The current approach requires finetuning classifiers as well as training diffusion models to be a joint model. This work removes this limitation and shows that the off-the-shelf classifier has the potential to outperform noise-aware models.

### Weaknesses
Although the work is very interesting, there are still some questions:

1. The comparison is not so comprehensive. Since the main baseline should be the noise-aware classifier, the work provides little comparison in Table 5. More metrics such as IS, sFID, Precision and Recall should be included. 
2. In Table 5, results for IMN64x64 as well as IMN256x256 for DDPM from Dhariwal and Nichol (2021) is missing. Whether or not the method can perform well on other resolutions.
3. Does off-the-shelf classifier guidance provide conditional information for the unconditional diffusion model? In my belief, the main objective of the guidance should be providing conditional information for the unconditional diffusion model rather than just improving the generated image quality. In Table 5, the Diffusion model on ImageNet128x128 is conditionally trained. Thus, the conditional information from an off-the-shelf classifier is not important. Results for combining off-the-shelf classifiers with unconditional models should be included.
4. How off-the-shelf models Resnet can be used with ImageNet64x64 or ImageNet128x128 although these models are trained on ImageNet224x224? Besides, the generated images are clip to be in range of [-1; 1], how does it fit with model trained with input images normalized by ImageNet datasets?
5. There is a fatal gap in the modeling part in Algorithm 1, in the sampling equation $\hat{x}_{t-1} \sim \mathcal{N}(\mu + \gamma _t g, \sigma _t)$. In the original paper by Dhariwal and Nichol (2021), the sampling resulted from the $\log(p _{ \theta }(x_t|x _{t+1}) p _{\phi}(y|x_t))$. However, given the structure of Algorithm1, this equation is no longer valid since the gradient is taken regarding $x _0 (t)$ instead of $x_t$. In order to apply the same sampling process, even when $x_0(t)$ is forward through the classifier, the gradient should be taken regarding $x_t$. 
6. I guess from the point (5), this is the main reason why the method can not be applied to DDIM?
7. Ablation study is missing, it is quite vague to understand which proposed scheme is the main course for the improvement. From my understanding, there are three main differences from normal classifier guidance which are:
* Gradients via $x_0(t)$ instead of $x_t$
* Guidance schedule
* $\tau _2$ temperature

However, discussion on the contribution of each of them is missing

8. Given three contributions as in (7), does the performance of the noise-aware classifier guidance also get improvements?
9. Besides Resnet, how are other architectures such as DenseNet, Transformers? Do they also work with this scheme?
10. CLIP guidance should be compared against the noise-aware CLIP guidance
11. The connection from 4.1, 4.2, 4.3 and 4.4, 4.5 as well as the design of the algorithm lacks some connections.

It seems that the paper is written in hurry so that the format of the paper is not really good as well as some errors in equations:
1. Equation (2) should be $\log exp(\tau f _{y(x)})$? Check format of the equation (2) also.
2. Equation (3) should be $\log exp(\tau _1 f _{y(x)})$?

### Questions
See the weaknesses. The work is interesting and potential, yet there are a number of concerns as well as writting. Will raise the score if all concerns are solved.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper presents a novel method to improve guidance in conditional diffusion generation without additional training by utilizing off-the-shelf classifiers. The authors introduce pre-conditioning techniques that enhance the performance of existing state-of-the-art diffusion models by up to 20% on ImageNet. Their approach is efficient, scalable, and leverages the widespread availability of pretrained classifiers, promising advancements especially in text-to-image generation tasks.

### Strengths
The paper adeptly integrates elements from both classifier-free and classifier-based diffusion approaches. Additionally, it employs off-the-shelf classifiers to enhance performance while maintaining efficiency. The experimental results presented confirm the effectiveness of these methods.

### Weaknesses
1. Sec 4.2 "PREDICTED DENOISED SAMPLES" seems trivial. This is already explored by the CVPR paper " Universal guidance for diffusion models". The authors should not make this part a separate subsection if this is not the author's original work.

2. Sec 4.3 "SMOOTH CLASSIFIER" seems trivial. I think there are already many works which use Softplus as activation function and explore its difference/advantage with ReLU. If I understand it correctly, the only contribution here is to replace ReLU with Softplus. The novelty point is not enough for an ICLR paper.

3. Sec 4.4 "JOINT VS CONDITIONAL DIRECTION", the author mention "we reduce the value of marginal temperature", which seems kind of manual tuning parameters. Is there some validation metric the authors use to determine the optimal temperature?

4. Sec 4.5 "GUIDANCE SCHEDULE" seems trivial. If I understand it correctly, the only contribution here is introducing a sin factor. This seems more like a trick instead of some research contribution. For it to be a research contribution, the author should first discuss what kind of guidance is good and why the author choose the sin factor here. 

5. Sec 5.3 "OFF-THE-SHELF GUIDANCE FOR DIT", the authors propose to incorporate classifier guidance g into classifier-free sampling. The idea is straightforward but the same question the authors introduce a parameter gamma_t here. How do the authors choose a proper gamma_t for a specific case?

### Questions
See Weaknesses.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this paper, the authors analyze off-the-shelf classifier guidance diffusion through multiple perspectives including calibration, smoothness, score decomposition and guidance scale scheduling. They use calibration error as a new metric to assess the performance of classifier guidance diffusion and propose several techniques to improve this task. They conduct experiments on multiple diffusion models with pre-trained ResNet classifiers and show consistent improvement with their proposed method compared to baselines.

### Strengths
This paper provides several interesting techniques to improve off-the-shelf classifier guidance diffusion. These techniques are practical, training-free and fairly easy to be incorporated into the current diffusion frameworks. Their analysis also provides interesting insight into what happens in the process of classifier guidance diffusion from multiple different perspectives.

### Weaknesses
1. I don’t really see the direct connection between Proposition 4.1 and Eq. 1. There seems to be a gap between ECE and $\|p_n-p\|$. Also to make the paper self contained, “bins”, “acc”, “conf” should be defined before mentioning.

2. It is unclear to me how Proposition 1 and ECE inform the design choices of the method. For Section 4.2, the same conclusion can be drawn with only FID. For Section 4.3, Proposition 1 only says “with smoothness $k>1$”, but it doesn’t claim better calibration/score prediction with higher smoothness. Section 4.4 and 4.5 seems to be completely irrelevant to ECE.

3. Relating to Weakness 2, I think the story line of this paper is a little bit scattered. There are many components in the story and it is difficult to tell which one is the main contribution of this paper. The components can also be better connected.

4. There are four components to the proposed method: (1) use $ \nabla_{\hat{x_0}(t)} \log p(y|\hat{x_0}(t))$ instead of $\nabla_{\hat{x_t}} \log p(y|\hat{x_t})$ (2) use Softplus activation to increase the smoothness (3) use a second temperature to adjust the “ratio” of joint and marginal guidance (4) sin factor guidance schedule. Since there is no ablation study that involves all four components conducted, it is very difficult to tell which one is actually effective. It would be great if the authors can include experiments that gradually exclude these components one-by-one to see which one is the most effective.

Minor suggestions:

1. According to the official style files provided by ICLR call for papers, the appendix should be included in the same PDF file as the main text and the bibliography.

2. Eq. 1 $k$ notation conflicts with the smoothness $k$ in Proposition 4.1.

### Questions
1. The formula for $\text{ECE}_t$ is with respect to $\hat{x_t}$ but in Section 4.2 the authors talked about using $\hat{x_0}(t)$ will provide better calibration, so which sample did the authors use when providing the ECE results for the rest of the experiments?

2. It is unclear to me how did the authors incorporate the Softplus activation function into the pre-trained classifiers, did they just replace all the activation functions in the pre-trained models? Or is there anything else that they did?

3. How did the authors calculate the marginal likelihood for CLIP guidance generation?

4. What is the recurrent step in Table 1? Is it the same as the “backward guidance” in “Universal Guidance for Diffusion Model” paper? And why is the inference time not changed significantly with the calibrated method given there is extra marginalization required for all classes?
What type of GPU did the authors use in their experiments?

I am happy to raise my score if the authors address my concerns and questions in the rebuttal.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 5

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper investigates improving sample quality in conditional diffusion generation by leveraging off-the-shelf classifiers in a training-free context. The authors propose pre-conditioning techniques based on classifier calibration, significantly enhancing diffusion models' performance with minimal computational overhead, verified through experiments on ImageNet. They introduce a novel metric, integral calibration error (ECE), to evaluate the effectiveness of classifier guidance and find that off-the-shelf classifiers outperform trained classifiers under high noise. To combat the diminishing influence of classifier guidance in later diffusion stages, a new weighing strategy is suggested, yielding better sample quality. The paper highlights the potential of their methods in text-to-image generation and points out the limitations of current guidance enhancement methods in terms of sampling efficiency.

### Strengths
- The paper convincingly establishes the research context. In particular, it is effective in demonstrating the robustness differences across time step intervals between off-the-shelf classifiers and fine-tuned classifiers using the ECE metric.
- The division of the design space for diffusion guidance appears appropriate, and the empirical process of selecting among the various options seems justified.
- The authors' exploration reveals that guidance using off-the-shelf models exhibits performance comparable to or surpassing previous methods, which required significant computational costs.
- The experiments with guidance through CLIP demonstrate the potential for extending guidance models beyond classifiers, indicating scalability in the approach.
- The theoretical explanations provided lend solid persuasive strength to the authors' design choices.

### Weaknesses
- While I rate this study highly in general, it falls short in comparing with previous research utilizing off-the-shelf models.
- Specifically, the omission of closely related prior works, [1, 2] , is significant. PPAP[2] explores guidance using a wide range of modalities of off-the-shelf models in an efficient tuning and plug-and-play manner without significant computational costs, which is directly relevant to this paper's discussion. Observations such as the varying contributions of tuning guidance models across different time steps could also reinforce the evidence between the two studies.
- Overall, I highly appreciate the paper's contribution to the completeness of the discussion on diffusion models' guidance. However, to properly highlight this paper's direct contributions, a comparison with prior studies attempting off-the-shelf guidance is crucial. Although the authors conducted experiments comparing various design choices of off-the-shelf guidance, it is not clearly presented how these relate to previous research, and there is no direct comparison of the final FID scores with prior methodologies. The lack of such comparisons has inclined my evaluation towards rejection.
- In the CLIP guidance experiments, it seems that the authors did not apply the same level of guidance as they did with off-the-shelf classifiers. The CLIP guidance experiment does not appear to reflect the authors' contributions with a new methodology for off-the-shelf guidance.

[1]: Graikos, Alexandros, et al. "Diffusion models as plug-and-play priors." Advances in Neural Information Processing Systems 35 (2022): 14715-14728.

[2]: Go, Hyojun, et al. "Towards practical plug-and-play diffusion models." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

### Questions
- Based on the various considerations and discoveries made by the authors, I am curious whether it is possible to extend off-the-shelf guidance beyond class conditions to various modalities of conditional generation. It seemed that the CLIP experiment was intended to demonstrate such a possibility; however, as mentioned in the weaknesses, it did not feel like an experiment based on a new methodology reflecting the authors' considerations and findings.

### Soundness
4 excellent

### Presentation
3 good

### Contribution
3 good
