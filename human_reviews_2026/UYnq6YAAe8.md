# Iterative Importance Fine-tuning of Diffusion Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 4, 6

## Abstract
Diffusion models are an important tool for generative modelling, serving as effective priors in applications such as imaging and protein design. A key challenge in applying diffusion models for downstream tasks is efficiently sampling from resulting posterior distributions, which can be addressed using the $h$-transform. This work introduces a self-supervised algorithm for fine-tuning diffusion models by estimating the $h$-transform, enabling amortised conditional sampling. Our method iteratively refines the $h$-transform using a synthetic dataset resampled with path-based importance weights. We demonstrate the effectiveness of this framework on class-conditional sampling, inverse problems and reward fine-tuning for text-to-image diffusion models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Oftentimes we want to use a pretrained diffusion model as a prior for conditional sampling for some downstream task. The posterior distribution in this case can be thought of as a tilted version of the pretrained diffusion model's distribution. One method of sampling from such tilted distributions involves using the Doob's h-transform. In this paper, the authors propose an iterative algorithm for estimating the h-transform. Specifically, they propose refining the estimate of the h-transform by sampling from the diffusion model using the current estimate of the h-transform, filtering samples based on how well they fit the tilted distribution, and using those samples for supervised fine-tuning. This procedure is applied iteratively many times, yielding a sequence of improved estimates for the h-transform. They present experiments applying this approach to class-conditional sampling, super-resolution, and reward-based finetuning.

### Strengths
The paper is well-written and presents a novel method for estimating the h-transform to improve conditional sampling from diffusion models, which draws on methods such as importance based fine-tuning as well as iterative fine-tuning procedures. Notably, this method does not require differentiating the score function of the base model and also is more memory efficient than other methods. The authors additionally prove that their iterative approach is a descent algorithm that minimizes the general loss function for the h-transform and that the distribution of samples at each step is closer to that of the tilted distribution.

### Weaknesses
The experiments in the paper could be improved. For class-conditional sampling, I would like to see additional results in a more difficult setting with more classes such as ImageNet. For super-resolution, I would like to see a few different things. First, I think there should be experiments with more than 2x super-resolution, such as 4x and 8x super-resolution. Additionally, I would like to see experiments on additional datasets, such as ImageNet and CelebA-HQ, which are more standard. For super-resolution, I also don't think you compare with other methods, such as DDRM [1] and SR3 [2]. For reward-based fine-tuning, you should use a larger set of prompts for evaluation. Finally, you should compare with additional baselines. SMC-based approaches also allow for sampling from the tilted distribution and you should compare with recent SMC-based methods for this problem, such as FK Steering [3]. I would also like to see a simple Best-of-N baseline for each setting.

[1] Kawar, Bahjat, Michael Elad, Stefano Ermon, and Jiaming Song. "Denoising diffusion restoration models." NeurIPS 2022. [https://proceedings.neurips.cc/paper_files/paper/2022/hash/95504595b6169131b6ed6cd72eb05616-Abstract-Conference.html](https://proceedings.neurips.cc/paper_files/paper/2022/hash/95504595b6169131b6ed6cd72eb05616-Abstract-Conference.html).

[2] Saharia, Chitwan, Jonathan Ho, William Chan, Tim Salimans, David J. Fleet, and Mohammad Norouzi. "Image super-resolution via iterative refinement." IEEE transactions on pattern analysis and machine intelligence 2022. [https://ieeexplore.ieee.org/document/9887996/](https://ieeexplore.ieee.org/document/9887996/).

[3] Singhal, Raghav, Zachary Horvitz, Ryan Teehan, Mengye Ren, Zhou Yu, Kathleen McKeown, and Rajesh Ranganath. "A General Framework for Inference-time Scaling and Steering of Diffusion Models." ICML 2025. [https://arxiv.org/abs/2501.06848](https://arxiv.org/abs/2501.06848).

### Questions
How did you choose the set of prompts you use for the reward-based fine-tuning experiments?

Did you consider applying this to non-differentiable reward functions? Since you don't need to differentiate the reward function it may be useful to evaluating in this setting.

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a new method to finetune diffusion models to achieve high rewards. The main idea is an iterative approach, where the models are continuously trained on importance-weighted samples (or with a similar rejection sampling adjustment). These reweighted samples are supposed to align with the posterior distribution, induced by the prior distribution model and the exponentiated reward. The method is compared to a range of other methods in synthetic 2D, image super-resolution, class-conditional image generation and text-to-image generation tasks.

### Strengths
The paper in general is easy to follow. The proposed idea is simple and straightforward to implement.

### Weaknesses
The main weakness of the current submission is that the efficiency and effectiveness of the proposed method remain unclear. The presentation lacks critical algorithmic details, making it difficult to understand how the method actually work. Moreover, the empirical comparisons to prior work are limited and not convincingly justified. See more details below, 

- Important algorithmic details are not included in the main paper. The choice of hyperparameter $c$ should be important for the efficiency of rejection sampler, however it is not discussed and only reference is mentioned to prior work. Similarly, while the computation of importance weights are built upon prior work, it should be explained in the main paper for the completeness of the method. The importance weights presented in Alg. 13 is not directly tractable.

- Empirical evaluation should include the algorithm efficiency or complexity aspects. For example, how many total training samples each method uses, and what is the sample acceptance rate for the proposed method. If we control for such compute factors, what is the margin of the proposed method to others?

- For each experiment subsection, too few benchmarks are considered. For example, inverse problem in sec 4.2 only includes one example, and text-to-image reward finetuning only includes 3 prompts. In addition,  I believe the paper can benefit from adding more baseline methods for comparison ,for example, for the image super-resolution task. 

- Empirical results did not include standard errors. 

- Finally, as the authors pointed out in the limitation section, such importance sampling or rejection sampling methods rely on the assumption that the proposal (the prior model) has sufficient support on the high reward region. This point is more about the assumption of the method rather than a critique.

### Questions
- Eq 16 states parameterizes the network with reward gradient conditioning. Is my understanding correct that is reward gradient conditioning is used across all experiments except the text-to-image ones? What would happen if you didn't use such conditioning for the remaining experiments? 

- Why is the Stochastic loss presented in Figure 2? Isn't the proposed method trained on $L_{FT}$ loss in eq 15?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Given a diffusion model $p_\theta(x_0)$ and a reward model $r(x_0)$, the authors propose a method to sample from the reward tilted distribution. The authors introduce a fine-tuning method that builds on the h-transform learning framework introduced in Denker et al 2024, which learns the gradient of the intermediate reward $\nabla_{x_t} \log E_{p(x_0 | x_t)}[\exp(r(x_0))]$. 

- Unlike typical fine-tuning (or equivalently variational inference) methods that make use of the reverse KL, the authors propose using the supervised, forward KL, objective in eq 7.
- However, as samples from the tilted distribution are not available, the authors use the current iteration of the fine-tuned model to generate samples and then defining an accept/reject step using the Radon-Nikodym derivative (ie the importance weight) between the tilted path measure and the model defined path measure.

### Strengths
Unlike prior methods such as RAFT that make use of importance sampling with the current iteration as the proposal generator, the authors derive a principled approach for fine-tuning. The authors propose a tractable accept/reject probability for use in the diffusion fine-tuning setting.

### Weaknesses
1. The authors should engage with existing literature in a more thorough manner. For instance, iterative cross-entropy (De Boer et al 2005), reinforced self-training (Gulcehre et al 2023), and RAFT (Dong et al 2023), all use importance-sampling and rejection sampling for fine-tuning.
2. As baselines, the authors should include a comparison to sequential Monte Carlo methods, which in the low particle regime have been shown to outperform fine-tuned models. 
3. The authors also make use of reward gradient in parameterizing the h-transform, making it unclear to what extent the proposed fine-tuning method works.

De Boer, Pieter-Tjerk, et al. "A tutorial on the cross-entropy method." Annals of operations research 134.1 (2005): 19-67.

Gulcehre, Caglar, et al. "Reinforced self-training (rest) for language modeling." arXiv preprint arXiv:2308.08998 (2023).

Dong, Hanze, et al. "Raft: Reward ranked finetuning for generative foundation model alignment." arXiv preprint arXiv:2304.06767 (2023).

### Questions
See weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work tackles the problem of fine-tuning a diffusion model $p_\theta(x_0)$ to sample from the reward-tilted distribution $p_{\mathrm{tilted}}(x) \propto p_\theta(x_0) \exp(r(x_0))$. It does so by obtaining approximate samples from $p_{\mathrm{tilted}}(x)$, and then minimizing the denoising score matching loss on these samples. The approximate samples are obtained by sampling from the model in its current iteration, and correcting them using a rejection sampling scheme with importance weights computed over trajectories. This optimization algorithm (with importance sampling) is shown to monotonically minimize the reverse KL ($KL(p_{\theta}(x) || p_{\mathrm{tilted}}(x))$ to the target distribution. The method is demonstrated by reward-tilted sampling for 2D Gaussian mixture models, class conditional sampling in MNIST, a linear-inverse problem in super-resolution, and improving prompt alignment in Stable Diffusion.

### Strengths
1. The main important contribution of the paper is the use of relaxed importance sampling (from (Hertrich & Gruhlke, 2025)) to bypass the requirement of a fine-tuning dataset (as in DEFT), which is a useful and important improvement over that algorithm
    - Theorem 6, and the empirical validation in Fig 2 is useful for demonstrating that the relaxed importance sampling method doesn’t compromise theoretical validity of the method.
2. The method has a benefit over other methods in not storing (or backpropagating through) trajectories for training
3. The experimental settings chosen are relevant and useful for the proposed method (class conditional sampling, inverse problems, and prompt-alignment)
    - The method is demonstrated on a larger scale problem in prompt alignment for Stable Diffusion
4. The work is clearly written and has a logical presentation of ideas

### Weaknesses
1) The use of the replay buffer requires further explanation or justification, since it is critical to the efficiency of the algorithm:
    - Why is it reasonable to use a replay buffer? Shouldn’t trajectories be drawn on-policy (according to current $h$ ) to have the correct importance weights computed according to eq. 14 (I am assuming the accepted samples are placed in the buffer, as described in Algorithm 1)?
        - Notably in other works (eg. in Sendera et. al., 2024), the use of a replay buffer is justified by the training loss itself not needing on-policy samples, but such a justification doesn’t seem to hold for the proposed algorithm
        - Is there any degradation in performance for longer buffer lengths, it would be helpful to experimentally demonstrate the impact of the replay buffer.
        

2) For experiments - some more comprehensive evaluations are needed to demonstrate the performance of the method:
    - For the toy GMM task - from the figure or the reported metrics in Table 4 it is unclear whether the proposed method is less biased compared to alternatives (in particular classifier guidance). A metric based on reverse KL, or a comparison to a “ground truth” baseline (such as a diffusion model trained on ground truth samples from $p_{\mathrm{tilted}}$) would better illustrate the performance
    - For the linear inverse problem - the method should be compared to baselines such as VarGrad, or Adjoint Matching, to demonstrate how the method compares to alternatives
    - For prompt alignment fine-tuning - the images  produced by Importance FT seem overly bright and saturated, or with some artefacts, compared to the other fine-tuning methods (in Fig 4, as well as 8, 9, 10). Is there any degradation in performance due to the fine-tuning?

3) The advantages for the proposed method should be more clearly outlined, over the alternatives, in particular since the metrics in Table 3 doesn’t show a clear improvement for Importance FT over the alternatives. In my opinion this would be remedied by more empirical support for two claims made in the paper:
    - The conclusion implies the training method is more stable in terms of requiring fewer tricks (eg. loss clipping). Does this hold more broadly?  Eg. does the reward improve more smoothly along training iterations, or do the gradients have smaller variance compared to other baselines? Does the reward improve at a faster rate (either in terms of wall-clock time or the number of gradient updates made)? If so, the method would be more clearly useful.
    - The method has the same memory complexity as Adjoint matching (or other online RL methods using the adjoint method) (O(B)), so the claim of importance FT offering “a more efficient and scalable training pipeline” should be supported more clearly. If there is a time-overhead in adjoint matching compared to Importance FT, its significance to the prompt alignment experiment should be demonstrated clearly.

4) Minor grammar errors include:
   - Line 41 space between citation and “.Further, the tilted …”
   - Between Line 71, 73, “stable diffusion” should be capitalized for consistency
   - Line 179 “… sampling from unnormalised probability density function**s**” (s missing)
   - Line 205 “are in particular satisfied **if** the guidance $h$ …”
   - Line 277 (ii) - Should be “ $\mathbb{P}\_{tilted}$  rather than $\mathbb{P}_{tilde}$” - also proportion sign here doesn’t make sense, it should be off by a constant term
   - Line 400-401: “samples are **shown**”
   - Line 415: “samples **do** not always”
   - Line 473-474: missing word “We present **experiments** for class conditional sampling,”
   - Line 481 quotations around “good” are inverted
   - Line 483 “and high reward samples are are”

5) For related works, the paper is missing a reference and discussions to Particle Denoising Diffusion sampling (Phillips et. al., 2024), which uses a similar optimization scheme of sampling from the current iteration of the model, correcting with importance weights (in their case SMC), and then performing supervised training (albeit targeted at the sampling problem rather than reward fine-tuning).

For now I am recommending a weak accept. The primary concern being point 3 above. If more comprehensive evidence is provided which clearly shows the advantages of the proposed method, then I will strongly consider raising my score.

A. Phillips, H.-D. Dau, M. J. Hutchinson, V. De Bortoli, G. Deligiannidis, and A. Doucet, “Particle denoising diffusion sampler,” International Conference on Machine Learning (ICML), 2024.

### Questions
1. How crucial is the relaxed importance sampling step - eg. if you resampled directly based on the importance weights  - what would the effective sample size be? Or would it be infeasible?
2. Given that rejection sampling is used - do you sample a fixed number of trajectories in the outer loop, then accept a portion depending on the weights (to put in the buffer), or do you keep sampling until a fixed number of trajectories is accepted? 
3. The claim in 
    
    > We emphasize that our importance-based fine-tuning does not require to differentiate the score function of the base-model and never considers more **than one step of the generation process at once**. In particular, it can
    be applied for very large base-models where other methods run out of memory.
    > 
    
    is inaccurate (or unclear) since the method requires running the generation process to obtain trajectories (and importance weights). Did this mean to emphasize that the method doesn’t backpropagate at the level of the trajectory?
    

4. Experimental questions:
    - For the classifier guidance used for the 2D GMM, and MNIST - was DPS used, or was a time-conditioned classifier trained?
    - The main gap in performance for adjoint matching appears to be from moving to a LoRA parameterization. Is there any understanding for why this impacts the method so much?
    - Why is DPOK memory listed as O(B) instead of O(BT)? The (on-policy) gradient update in that paper appears to require backpropagation at each step of the generation trajectory.
        - This line (431) is inconsistent with the table “Importance fine-tuning reaches a competitive
        reward with similar diversity and the same O(B) memory footprint as DPOK or RTB”

### Soundness
3

### Presentation
3

### Contribution
2
