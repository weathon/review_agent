# Learn to Guide Your Diffusion Model

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Classifier-free guidance (CFG) is a widely used technique for improving the perceptual quality of samples from conditional diffusion models. It operates by linearly combining conditional and unconditional score estimates using a *guidance weight* $\omega$. While a large, static weight can markedly improve visual results, this often comes at the cost of poorer distributional alignment.
In order to better approximate the target conditional distribution,
we instead learn *guidance weights* $\omega_{c,(s,t)}$, which are continuous functions of the conditioning $c$, the time $t$ from which we denoise, and the time $s$ towards which we denoise. 
We achieve this by minimizing the distributional mismatch between noised samples from the true conditional distribution and samples from the guided diffusion process.  We extend our framework to reward guided sampling, enabling the model to target distributions tilted by a reward function $R(x_0,c)$, defined on clean data and a conditioning $c$. We demonstrate the effectiveness of our methodology on low-dimensional toy examples and high-dimensional image settings, where we observe improvements in Fréchet inception distance (FID) for image generation. In text-to-image applications, we observe that employing a reward function given by the CLIP score leads to guidance weights that improve image-prompt alignment.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes to introduce the Maximum Mean Discrepancy (MMD) to quantitatively evaluate the distance between the CFG-guided distribution and the ground-truth one. By doing so, it is enabled to design both time-dependent and condition-dependent guidance weight, which is more flexible and controllable. To realize such a pipeline, the authors employ an auxiliary network for guidance weight prediction, and further discuss the influences under different objectives. Both qualitative and quantitative results confirm the superiority of the proposed method.

### Strengths
- The paper is well structured and easy to follow. All technical details are carefully discussed and clear.
- The motivation is intuitive and effective, alleviation the discrepancy between CFG-guided and the ground-truth distributions is capable of improving the performance of guided sampling.
- The further discussion about different objective is detailed and inspiring, encouraging future study for better training efficiency and synthesis performance.

### Weaknesses
- The whole pipeline of narrowing the distribution discrepancy by optimizing some scalars is not new. Beyond the main objective being directly motivated by prior works, similar idea of canceling mismatch (or "self-consistency") between forward diffusion and reverse denoising stages has also been proposed [1].

  [1] Towards More Accurate Diffusion Model Acceleration with A Timestep Tuner. Xia et al., CVPR 2024.

- The main concern about the proposed method is the optimality. There is no theoretical analysis about the optimality or convergence about the proposed objective. Then why the optimized guidance weight could guarantee the self-consistency? What further confirms my worries is in Appendix C.1. The most intuitive objective is the L2 norm in Eq. (23), which is consistent with the vanilla training loss of diffusion models, *i.e.*, the ELBO loss. However, as the authors themselves claim, such an objective leads to zero guidance weight. That is to say, **diffusion model converges well and both conditional and unconditional scores are accurate**, thus there is no need to employ CFG. Under this discussion, I am curious about the theoretical optimality of Eq. (20) or Eq. (21). Could the authors provide theoretical analyses under some trivial toy data to verify the correctness of the method? From my opinion, Eq. (21) is somewhat an upper bound of ELBO (*i.e.*, Eq. (23)). Given that ELBO itself is an upper bound of KL-divergence (by the native theory of diffusion models), the optimization of a looser bound may lead to meaningless results.

- What further weakens the soundness of the paper is the employment of an auxiliary network. To be honest, such a setting could lead to more flexible guidance weight given different conditions. However, what if the network fails to converge well and only predicts sub-optimal guidance weight? The accuracy issue is more severe under open-vocabulary setting, *i.e.*, text-to-image or text-to-video. Considering that the authors employ a light-weight network, such a concern is also crucial. Could the author provide some closed-form expressions or some empirical solutions avoiding the employment of an auxiliary network?

- The qualitative results of text-to-image is somewhat not convincing. The improvements in Figs. 4-7 are inconspicuous. This strengthens my concerns above.

### Questions
Beyond the Weaknesses part, I am curious about the general rule of the guidance weights. Current analyses in Fig. 3 is poor. Could the authors provide more quantitative analyses about the weights under different conditions and timesteps? For example, what if a text condition gradually becomes complex and a long caption? What about the mean or variance?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose a method for learning the guidance schedule for diffusion model generation. In particular, they train a shallow network to minimize the distributional mismatch between a frozen denoiser and the true distribution. The authors provide solid theoretical motivation for their choice of objective function, and also experimental evidence for class and text-conditional models.

### Strengths
- The authors replace a hyper-parameter (guidance strength) with a learned approach
- The objective function is well theoretically motivated

### Weaknesses
- The authors train their own diffusion models for use in evaluations. It would greatly strengthen the paper if they also showed FID and reward improvements for existing pre-trained diffusion models from prior work. For example:
     - EDM for cifar-10 (Karras et al, https://github.com/NVlabs/edm)
     - EDM-2 for ImageNet (Karras et al, https://github.com/NVlabs/edm2)
     - MicroDiffusion for text-to-image (Sehwag et al, https://github.com/SonyResearch/micro_diffusion)
- The authors compare their learned approach to simple baselines such as constant or limited-interval guidance. It would strengthen the paper if they also showed improvements over stronger un-trained baselines such as in Wang et al (https://arxiv.org/abs/2404.13040).

My score recommends reject, but I would be happy to raise my score to accept if these additional experiments are included in the paper and my questions below are addressed.

### Questions
- What do the authors mean by a high-variance objective?
- What is the training cost of training the learned guidance network as a percentage of diffusion model training cost?
- Why is the FID on COCO for the constant guidance model so high? FID 31 seems incredibly high for a 1B model. How was this model trained?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper puts forward learning a classifier-free guidance schedule that varies with both time and the input condition (class or text), enabling the sampler to better match the desired conditional distribution.
Compare with handcrafting guidance strengths, the method optimizes a self-consistency objective: if one adds noise to a real image and then denoises it with guidance, the intermediate result should match what the forward noising process would produce at that timestep.
Following this desin mechnism, this reframes guidance tuning as a distribution-matching problem and naturally extends to reward-guided generation by combining an external reward (e.g., CLIP-based) with the same self-consistency regularizer.
Imortantly, exntensive experiments on standard image datasets and text-to-image benchmarks show consistent quality gains over unguided sampling, fixed guidance, and simple time-limited guidance, with learned schedules that adapt across prompts and timesteps.

### Strengths
It reframes classifier-free guidance tuning as a distribution-matching problem, replacing hand-crafted schedules with a learned, time- and prompt-aware policy—an original and practical angle.
The self-consistency objective is elegant and low-variance, making the method easy to implement on top of existing diffusion backbones without architectural changes.
Experiments span multiple datasets and backbones and include sensible ablations, showing consistent gains over unguided, fixed-guidance, and limited-interval baselines.
The learned guidance schedules are interpretable (varying with prompt and timestep), which improves clarity and aids real-world debugging.
Overall, the approach offers a broadly applicable and deployable improvement to conditional diffusion sampling with a favorable quality-to-complexity trade-off.

### Weaknesses
1. The paper argues that enforcing self-consistency is stronger and low-variance, but the conditions under which minimizing the proposed loss actually improves conditional sampling are not formalized.

2. On text-to-image, self-consistency plus CLIP reward improves FID yet struggles to surpass strong guided baselines on CLIP score.

3. Learning a guidance network and evaluating the self-consistency loss add additional overhead.

### Questions
1. How sensitive are results to the kernel/parameters in the MMD objective versus the simpler L2 variant?

2. What are the training and inference costs attributable to guidance learning and self-consistency evaluation?

3. Since the learned weights vary strongly with prompt, do certain semantic categories systematically demand higher guidance?

4. Can practitioners easily inspect how guidance evolves with time and prompt?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a method to learn the classifier-free guidance (CFG) weight by setting it as the output of a neural network. While previous CFG approaches relied on heuristics or trial-and-error to find an optimal (often large) weight, which could lead to saturation and degraded sample quality, this work optimizes guidance weights as a continuous function of time and condition. The objective is to minimize the distributional mismatch between the generated guided samples and the noised true conditional samples, using a novel self-consistency loss. This loss successfully mitigates the high-variance problem associated with theoretical approaches, leading to stable training and demonstrated performance gains across various datasets, from toy examples to COCO T2I sampling.

### Strengths
- The idea of learning the optimal guidance weight continuously via a neural network, rather than relying on grid search for optimal guidance intervals or empirical adjustments, is highly novel.
- Despite the inherent risk of the guidance weight collapsing to a trivial solution, the successful stabilization of learning via the newly proposed self-consistency loss is a significant achievement. The simplified L2 objective (Eq. 21) also appears surprisingly straightforward to implement.

### Weaknesses
The proposed method involves a large number of hyperparameters to determine the optimal guidance weight, and additionally requires performing $m$ rounds of noising and comparison at every denoising step, which introduces significant computational overhead. Given this, I wonder whether the computational cost difference compared to existing guidance distillation methods is actually substantial. It would be valuable to include a cost comparison table or discussion.

Moreover, since the loss function in this paper formalizes the matching between the true sample distribution and the guided one, I suspect that a parameter-efficient fine-tuning (PEFT) approach such as LoRA could optimize a student model directly using the same loss formulation, without needing a separate guidance network at inference time. This would result in a more inference-efficient model, and I’m curious whether the authors have considered or experimented with this alternative.

Regarding the design of the guidance network, the paper focuses heavily on the loss formulation for learning guidance weights but provides limited discussion or justification for the network’s architecture. The rationale behind its input-output design is unclear. Based on Table 1 and Fig. 1, 3, the conditioning input seems to have a stronger influence than expected, suggesting that incorporating the currently sampled image as an additional input could further improve weight estimation. Were such variants or conditioning combinations explored?

In the COCO experiments, the guidance network uses the text encoder output as a conditioning signal, and Fig. 3 shows that the learned guidance weights vary across prompts. Does this imply that the network has learned semantic understanding of the text embeddings? If so, how heavy must the model be to capture such semantics effectively? Additionally, from the qualitative results on COCO, the outputs often resemble those with weak or no guidance, which may indicate that the diffusion model’s original objective still struggles to fully capture the conditional distribution, similar to previous findings.

Finally, in the loss formulation, the authors mention using $m=4$ particles for computing the MMD in the image domain. Even if self-consistency reduces variance, I question whether the MMD loss computed from only four samples can be sufficiently accurate. Was this number chosen primarily to reduce computational cost? If so, it would be helpful to report how performance varies with different particle counts, to justify the design choice.

### Questions
- How does this method compare to guidance Interval approaches? The guidance interval method empirically finds an optimal guidance schedule, assigning near-zero guidance weight in early steps and large weights in the final few steps. Since the proposed method can measure how well the estimated distribution matches the true one via its loss, it may be possible to evaluate the optimality of the interval-based scheduling using the same loss metric. If so, this could open new directions for systematic guidance interval optimization beyond the work of LIG.
- Once the optimal guidance weights are obtained through this method, could they be further distilled into a single model to achieve both high performance and inference efficiency?

### Soundness
3

### Presentation
3

### Contribution
4
