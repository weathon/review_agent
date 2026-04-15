# Understanding and Improving Adversarial Attacks on Latent Diffusion Model

- Decision: Reject
- Scores: 5, 5, 3, 5, 3

## Abstract
Latent Diffusion Model (LDM) has emerged as a leading tool in image generation, particularly with its capability in few-shot generation. This capability also presents risks, notably in unauthorized artwork replication and misinformation generation. In response, adversarial attacks have been designed to safeguard personal images from being used as reference data. However, existing adversarial attacks are predominantly empirical, lacking a solid theoretical foundation. In this paper, we introduce a comprehensive theoretical framework for understanding adversarial attacks on LDM. Based on the framework, we propose a novel adversarial attack that exploits a unified target to guide the adversarial attack both in the forward and the reverse process of LDM. We provide empirical evidences that our method overcomes the offset problem of the optimization of adversarial attacks in existing methods. Through rigorous experiments, our findings demonstrate that our method outperforms current attacks and is able to generalize over different state-of-the-art few-shot generation pipelines based on LDM. Our method can serve as a stronger and efficient tool for people exposed to the risk of data privacy and security to protect themselves in the new era of powerful generative models.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper theoretically analyzes the adversarial attacks against latent diffusion models (LDM) for the mitigation of unauthorized usage of images. It considers three subgoals including the forward process, the reverse process, and the fine-tuning process. It proposes to use the same adversarial target for the forward and reverse processes to facilitate the attack. Experiments show it outperforms baselines and is robust against super-resolution-based defense.

### Strengths
1. This paper studies protecting images from being used by stable diffusion models without authorization, based on adversarial perturbations. This is an important research problem.

2. Different from existing empirical studies, this paper proposes a theoretical framework to help understand and improve adversarial attacks.

3. Experimental results show the proposed method can outperform existing baselines.

### Weaknesses
1. Not clear if this method needs white-box access to the subject LDM. That is, do the adversarial attackers use the same network used by infringers? Are the adversarial examples generated on one model/method still effective on different or unknown models/methods?

2. Only one defense method is evaluated. Are the adversarial samples robust to other transformations such as compression or adding Gaussian noises? Also, no adaptive defense is evaluated. If the infringers know about this adversarial attack, can they adaptively mitigate the adversarial effects?

3. From my understanding, Liang & Wu, 2023 and Salman et al., 2023 are not just "targeting the VAE" as claimed in this paper. They attacked the UNet as well.

4. Many issues in the writing. 

    4.1. On Page 2, "serve as a means to" -> "mean". 

    4.2. In Section 2.2, it says "As shown in Figure 1 adversarial attacks create an adversarial example that seems almost the same as real examples". However, Figure 1 only contains the generated images by the infringers instead of the adversarial examples as indicated by the title.

    4.3. The references to figures and tables are incorrect such as "Figure 5.3", "Table 5.2", "Table 5.3", etc. In the ablation study, the caption of the table is "Figure 5".

    4.4. Some math symbols are not defined where they first appear. For example, It would be better to mention that $\phi$ in Section 2.1 means the VAE. I suggest to use $\mathcal{E}\_{\phi}$ or $\mathcal{E}$ consistently. What does $\sqrt{\bar{\alpha\_t}}$ (the line below equation 5) mean? The text below Equation 10, for "$q\_{\phi}(v\_t | x')$ and $q\_{\phi}(v\_t | x')$", the second one should be $q\_{\phi}(v\_t | x)$. It would be better to briefly explain the N, M, and K in  Algorithm 2.

    4.5 The citation of SR in section E.2 is wrong. "Salman et al., 2023" -> "Mustafa et al. 2019".

### Questions
1. Could you explain why the equation 4 holds? Why do we need to use $q$ to express the left $p$?

2. Can the adversarial examples be effective for different or unknown models/methods? 

3. According to section E.1, the target $\mathcal{T}$ in Equation 15 and 16 is defined as $\mathcal{E}(x^{\mathcal{T}})$. I can understand for the $\mathcal{L}\_{vae}^{\mathcal{T}}$, it's meaningful to encourage $\mathcal{E}(x')$ to be close to $\mathcal{E}(x^{\mathcal{T}})$. However, for the UNet part, what's the rationale to encourage the predicted noise at each timestep to be close to $\mathcal{E}(x^{\mathcal{T}})$? Because I think the final output $z_0$ should be close to $\mathcal{E}(x^{\mathcal{T}})$, but not the intermediate predicted Gaussian noise.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper introduces a theoretical framework for understanding adversarial attacks on latent diffusion models. Based on this framework, the paper proposes a novel and efficient adversarial attack method that exploits a unified target to guide the attack process in both the forward and reverse passes of latent diffusion models.

### Strengths
1. The paper focuses on curbing the misuse of powerful LDM-based generative models for unauthorized real individual imagery editing, which is an important topic for securing privacy.

2. The theoretical foundation behind adversarial attacks on diffusion models is built, which contributes to the understanding of the behaviors of adversarial attacks.

### Weaknesses
1. More thorough examination accounting for a wider range of generative techniques could further validate the method's real-world utility and limitations.  While the proposed attack focuses on the prevalent LDM framework, its generalization to other powerful generative paradigms like SDXL, DALL-E, and Deep Floyd remains untested. 


2. A more powerful baseline of PhotoGuard, i.e. Diffusion Attack is not compared to. This comparison could help gauge the true leadership of the new method. Without including this more powerful adversarial technique, the paper's claims about the proposed attack outperforming the current approaches remains uncertain.

3. The authors assert a memory-efficient design but do not provide details to support this claim. Further explanation or experimental evaluation of memory usage compared to alternative approaches would help validate the proposed method's efficiency advantages.

### Questions
Please refer to the weakness section.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposed at a theoretical framework for adversarial attacks on Latent Diffusion Model (LDM).  The key to the theory is formulating an objective to minimize the expectation of the likelihoods conditioned on adversarial samples, of which the two terms implemented within the LDM explains the adversarial attack on the VAE and on the noise predictor, respectively. In addition, a new adversarial attack combining the two types of adversarial attacks are proposed.

### Strengths
1. Adversarial attacks on LDM is an interesting and practical problem.
2. Various experiments are conducted.

### Weaknesses
The proposed theoretical framework is not sufficiently innovative. In addition, the methodology exists many errors and the experimental verification are not sound. Specifically,

1. The key formulation of minimizing the conditional likelihood is trivial. The given theoretical proof is complicated. Actually, the likelihood equivalent to the KL divergence has been well-know. From this perspective, the proof is somehow trivial.

2. There exists many wrong equations.   
a. In Eq. (5), the left term q(v_t|x) should be equal to the integral of the right term. Similar issue in Eq. (8).  
b. In the first paragraph of Sect 3.2, q(v_t|v_{0:t-1} is mistakenly formulated.  
c. In the last paragraph of Page 4, z_{t-1} is mistakenly formulated.  
d. In Eq. (3), the sum in terms of z is mistakenly formulated given the expectation.   
e. For Eq. (3), (4), (9)…, p()|x=x’ or p()|x’ is inappropriate, which should be put as the subscript or p(|x=x’).  
f. The reformulation of z as v is unnecessary.  

3. From the empirical results (Table 1), the strategy of combining two adversarial attacks does not perform significantly better than the Eq. (16). This raises doubt about the effectiveness of the newly proposed attack method.

Minor:
Some references are wrongly denoted, e.g. Figure 5.3.

### Questions
1. The proposed theoretical framework is not sufficiently innovative. 

2. The methodology exists many errors

3. The experimental verification are not sound

### Soundness
1 poor

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work proposes a method for improving adversarial attacks on Latent Diffusion Models (LDMs). The purpose is to generate adversarial examples preventing the LDMs from generating high-quality copies of the original images in order to protect the integrity of digital content. The authors mathematically introduce a theoretical framework formulating three sub-goals that existing adversarial attacks aim to achieve. This framework exploits a unified target to guide the adversarial attack both in the forward and the reverse process of LDM. The authors implement an attack method jointly optimizing three sub-goals, demonstrating that their method outperforms current attacks generally. The experiments are focused on the attacks on training pipelines, including SDEdit and LoRA.

### Strengths
1.	The author conducted extensive mathematical derivations, providing mathematical explanations for the existing methods. 

2.	The experiments show that this method outperforms the baseline in most criteria and succeeded in attacking LoRA and SDEdit with Stable Diffusion v1.5.

### Weaknesses
1.	The only backbone model used in the experiments is Stable Diffusion v1.5. 
There are plenty of more recent LDMs, such as Stable Diffusion v2.1 [1] and DeepFloyd/IF [2]. Will this method perform well in more advanced LDMs?

2.	The pseudo-code of algorithm 1 seems redundant and demonstrates nothing. 
It literally equals to its description: “To optimize J_{ft}, we first finetune the model on x and obtain the finetuned model θ(x). Then, we minimize J_{ft} over x′ on the finetuned model θ(x).”


3.	The target image adopted by Liang & Wu (2023), visualized in “Figure E.1” (Is it mislabeled in Figure 8?), is the only target image used in the experiments. Did the authors try using different target images? Will the target image affect the effectiveness of this method?

4.	There are issues in the document layout. The labels of figures are mismatched to those in the texts. 

5.	The offset problem needs to be clarified.
The authors claim that “The result in Figure 5.3 implies that offset takes place in 30% - 55% of pixel pairs in Δ_z_t and Δ_ε_θ, which means that maximizing J_q pulls Δ_z_t to a different direction of Δ_ε_θ and interferes the maximization of J_p.” Could the authors further explain it and Figure 3 in Section 4.1?

[1] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10684–10695, 2022.

[2] Mikhail Konstantinov, Alex Shonenkov, Daria Bakshandaeva, and Ksenia Ivanova. Deepfloyd. https://github.com/deep-floyd/IF, 2023.

### Questions
1.	Will this method perform well in more advanced LDMs like Stable Diffusion v2.1 and DeepFloyd/IF?

2.	Will the target image affect the effectiveness of this method? Would the authors use other images as targets and test the effectiveness?

3.	Is that a typo in “In this tractable form, z′_t and z_t sampled from q_φ(v_t|x′) and q_φ(v_t|x′), respectively” below Equation 10?


4.	How does maximizing J_q pull Δ_z_t to a different direction of Δ_ε_θ and interfere the maximization of J_p? Could the authors further explain Figure 3 in Section 4.1?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 5

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
Since diffusion and other advanced generative models have been used to replicate artworks and create fake content, a line work proposes a defense mechanism that adds a kind of adversarial perturbation to the protected images to prevent the adversary from fine-tuning their model on the images. This work proposes a more thorough theoretical formulation of the problem compared to the prior work and relies on this formulation to build an empirically stronger attack.

### Strengths
### Significance

I believe that this paper addresses an important problem with a widespread impact on both the technical community as well as society at large. The empirical results show a convincing improvement over the prior works on two different fine-tuning methods and two datasets.

I believe that introducing the target image for the adversarial objective and hence conducting a targeted attack instead of an untargeted one have a large effect on the empirical success of the attack.

### Weaknesses
### Correctness and clarity of the theoretical results

The paper formulates an adversarial optimization problem particularly tailored for the latent diffusion models (LDM). The analysis guides the algorithm design to some degree (more on this later). However, due to the lack of clarity and various approximations being introduced without proper justification, the theoretical results become less convincing. I will compile all my questions and concerns from Section 3 and 4 in place:

1. I am not sure what the sum $\sum_z$ is over in Eq. (3). The expectation is already over $z$ so I am a bit confused about the summation. My guess is that the sum is over all the latent variables in the diffusion process (different $z$’s in different steps). Is this correct?
2. If my previous understanding is correct, my next question is why should the adversary care about the latent variables in the intermediate steps of the diffusion process instead of, say, the final step of the inverse process before the decoder?
3. Based on the text, Eq. (3) should be equivalent to $\mathbb E_{z \sim p_{\theta}(z|x)}[- \log p_\theta(z|x')]$. My question is that a slightly different formula $\mathbb E_{z \sim p_{\theta}(z|x')}[- \log p_\theta(z|x) + \log p_{\theta}(z|x')]$ also seems appropriate (swapping order in the KL-divergence). Why should we prefer one to the other?
4. Section 3.2 uses the notation $\mathcal N(\mathcal E(x), \sigma_\phi)$ instead of $\mathcal N(f_{\mathcal E}(x), \sigma_{\mathcal E})$ from Section 2.1. Do they refer to the same quantity?
5. In the last paragraph of page 4, the Monte Carlo method must be used to estimate the mean of $p_\theta(z_{t-1}|x)$, but I cannot find where the mean is actually used. It does not seem to appear in Eq. (10) or in Appendix A.1. I also have the same question for the variance of $p_\theta(z_{t-1}|x)$ mentioned in the first paragraph of page 5.
6. Related to the previous question, it is mentioned that “the variance of $z_{t-1}$ is estimated by sampling and optimizing over multiple $z_{t-1}$.” It is very unclear what “sampling” and “optimizing” refer to here.
7. I do not quite see the purpose of Proposition 1. It acts as either a definition or an assumption to me. The last sentence “one can sample $x \sim w(x)$ from $p_{\theta(x)}(x)$” is also very unclear. Is the assumption that the true distribution is exactly the same as the distribution of outputs of the fine-tuned LDM?
8. $x^{(eval)}$ is mentioned in Section 3.4 but was never defined.
9. In Eq. (11), should both of the $\theta(x)$’s be $\theta(x')$ instead? Otherwise, $x'$ has no effect on the fine-tuning process of the LDM.
10. Section 4.1 is very convoluted (see details below).

### Issues with the offset problem and Section 4.1

**Comment #1**: I do not largely understand the purpose of the “offset” problem in Section 4.1. In my understanding, most of the discussion around the offset can be concluded by simply expanding the second term on the first line of Eq. (13):

$$
\sum_{t \ge 1}\mathbb E_{z_t,z'_t} || \Delta z_t +  \frac{\beta_t}{\sqrt{1 - \bar{\alpha_t}}} \Delta \epsilon ||_2^2 
$$

$$
= \sum_{t\ge 1}\mathbb E_{z_t,z'_t} ||\Delta z_t||_2^2 + || \frac{\beta_t}{\sqrt{1 - \bar{\alpha_t}}}\Delta\epsilon ||_2^2 + \frac{2\beta_t}{\sqrt{1 - \bar{\alpha_t}}}\Delta z_t^\top\Delta\epsilon
$$

So the problem that prevents optimizing just the norm of $\Delta z_t$ and the norm of $\Delta \epsilon_\theta$ directly is the last term in the equation above (the dot product or the cosine similarity). I might be missing something here so please correct me if I’m wrong.

**Comment #2**: It is also unclear to me how the last line of Eq. (13) is reached and what approximation is used.

**Comment #3**: In theory, there is nothing preventing one from optimizing Eq. (13) as is. The issue seems to be empirical, but I cannot find the empirical results showing the failure of optimizing Eq. (13) directly and not using the target trick.

**Comment #4**: The authors “let *offset rate* be the ratio of pixels where the vector $\Delta z_t$ and $\Delta \epsilon_\theta$ have different signs.” If my understanding of the cosine similarity above is correct, this seems unnecessary and imprecise given that the cosine similarity is the exact way to quantify this.

**Comment #5**: In the first paragraph of page 7, it is mentioned that “meanwhile, since the original goal is to maximize the mode of the vector sum of…” I think instead of “mode,” it should be “magnitude” or the Euclidean norm?

### Empirical contribution

1. After inspecting the generated samples in Figure 11-15, my hypothesis is that the major factor contributing to the empirical result is the target pattern and the usage of the targeted attack. The pattern is clearly visible on the generated images when this defense is used, and this pattern hurts the similarity scores. This raises the question of whether the contribution comes from the theoretical formulation and optimization of the three objectives or the target. I would like to see an ablation study on this finding: (1) the proposed optimization + untargeted and (2) the prior attacks + targeted.
2. The choice of the target $\mathcal T$ is ambiguous. While the target pattern is shown in the Appendix, there is no justification for why such a pattern is picked over others and whether other patterns have been experimented with.

Overall, I believe that the paper can have a great empirical contribution, but it seems to be clouded by the theoretical analysis which appears much weaker to me.

### Questions
1. What are the approximations made on the fourth line of Eq. (22) and in Eq. (23)?
2. Why are MS-SSIM and CLIP-SIM used as metrics for SDEdit whereas CLIP-IQA score is used for LoRA? The authors allude to this briefly, but it still largely remains unclear to me.
3. The similarity metrics used in experiments seem to focus on the low-level textured detail rather than the style (please correct if this is not accurate). I am wondering if a better metric is the one that measures “style similarity” between the trained and the generated images. This might align better with the artwork and the copyright examples.
4. For the results reported in Table 1, how many samples or runs are they averaged over? Based on the experiment setup, 100 images are used for training the model in total for each dataset, and they are grouped in a subset of 20. So my understanding is that there are 100/20 = 5 runs where 100 images are generated in each run? Is this correct?
5. The fine-tuning hyperparameters for LoRA are mentioned in Section 5.1. Does the LoRA fine-tuning during the attack and the testing share the same hyperparameters? What happens when they are different (e.g., the adversary spends iterations during fine-tuning, etc.)? Can the proposed protection generalize?
6. Have the authors considered any “adaptive attack” where the adversary tries to remove the injected perturbation on the images (e.g., denoising, potentially via another diffusion model) before using them for fine-tuning?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
