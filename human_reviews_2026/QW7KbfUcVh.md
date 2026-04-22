# Critic-Guided Reinforcement Unlearning in Text-to-Image Diffusion

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
Machine unlearning in text-to-image diffusion models aims to remove targeted concepts while preserving overall utility. Prior diffusion unlearning methods typically rely on supervised weight edits or global penalties; reinforcement-learning (RL) approaches, while flexible, often optimize sparse end-of-trajectory rewards, yielding high-variance updates and weak credit assignment. We present a general RL framework for diffusion unlearning that treats denoising as a sequential decision process and introduces a timestep-aware critic with noisy-step rewards. Concretely, we train a CLIP-based reward predictor on noisy latents and use its per-step signal to compute advantage estimates for policy-gradient updates of the reverse diffusion kernel. Our algorithm is simple to implement, supports off-policy reuse, and plugs into standard text-to-image backbones. Across multiple concepts, the method achieves better or comparable forgetting to strong baselines while maintaining image quality and benign prompt fidelity; ablations show that (i) per-step critics and (ii) noisy-conditioned rewards are key to stability and effectiveness. We release code and evaluation scripts to facilitate reproducibility and future research on RL-based diffusion unlearning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
CGRU is a machine unlearning method tailored for text-to-image diffusion models. Its core is to formulate the stepwise denoising process as a reinforcement learning sequential decision-making task. By introducing a timestep-aware critic, it predicts the terminal outcome and guides policy-gradient updates of the reverse diffusion kernel at each denoising step. The method achieves plug-and-play integration without modifying the core architecture of standard text-to-image backbones and supports off-policy reuse of historical training trajectories via importance weighting.

### Strengths
- Innovatively implements per-timestep criticism for individual diffusion steps, addressing the limitations of sparse end-of-trajectory rewards in prior RL-based diffusion methods.
- Plug-and-play design: Requires no modifications to the core architecture of text-to-image models and can be directly embedded into existing frameworks such as Stable Diffusion.

### Weaknesses
- Fails to evaluate concept erasure for celebrities, specific styles, or concepts fine-tuned via methods like DreamBooth.
- Exhibits a significant trade-off between Unlearning Accuracy (UA) and In-domain Retain Accuracy (IRA), with inferior overall performance compared to methods like SalUn and no substantial effectiveness improvement.
- Only validated on Stable Diffusion 1.5; its concept erasure performance on other state-of-the-art generative models remains unconfirmed.

### Questions
As demonstrated in https://arxiv.org/abs/2503.10637, during the later timesteps of generation, the model can already generate the final result through single-step diffusion. In contrast, the earlier timesteps are dominated by meaningless noise. Why is it imperative to perform scoring and training at every individual generation step? What is the fundamental difference between this approach and directly generating the final step (x₀) from timestep t (as proposed in https://arxiv.org/abs/2304.05977) followed by terminal reward scoring?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents Critic-Guided Reinforcement Unlearning CGRU, a timestep aware method for reinforcement unlearning in text-to-image diffusion models. ​The key idea is to interpret the reverse diffusion process as a policy and add a timestep aware critic trained to predict the final reward. Empirically, the method shows moderate improvements over DDPO on aesthetic-reward optimization and object unlearning, reporting 95.6% unlearning accuracy and 78% retention accuracy.​

### Strengths
- The paper is well motivated, coherent, with clean notations and algorithmic details. ​

- The paper offers a new perspective on unlearning by reframing diffusion sampling as an actor-critic RL problem, and provides a formal connection between two active areas - diffusion alignment and machine unlearning - in a unified formalism.​

- Introduction of a per-timestep critic for diffusion policy optimization is a clear algorithmic step forward, especially given the substantial instability and high variance of prior end-of-trajectory reward methods like DDPO. 

- The limited evaluation that is provided shows improved performance. In particular, Table 1 (Page 8) shows that CGRU achieves the top Unlearning Accuracy (UA = 95.55%) on the benchmark while remaining competitive in In-domain Retain Accuracy (IRA = 78.47%). 

- Fig. 1 visualization of the training dynamics is convincing, offering a direct performance comparison versus DDPO. CGRU displays consistently faster convergence and higher final rewards. Further, Fig. 2 illustrates clearly superior suppression of the “Cat” class compared to DDPO, with cleaner reward trajectory improvements.

- Ablation and architectural choices are motivated and described, adding modular value for future work.

- Detailed appendices and release of code and scripts promote reproducibility.

### Weaknesses
- Scope of evaluations: The evaluation depth is limited. While the paper tests 20 object classes, it does not cover a single concept, keeping the scope narrow. The experiments focus only on one model (Stable Diffusion 1.5) and one dataset (UnlearnCanvas), concentrating on specific objects like "Cats" and "Towers" (Appendix D, Table 4). There are no results on more abstract or safety-critical tasks like style removal or identity erasure, which are mentioned as important reasons for unlearning in the introduction.  

- Weak evidence: The paper lacks enough qualitative evidence to judge its generalization abilities. The only visual examples shown in Figure 3 are for the "Cats" class, making it unclear how the method performs on the other 19 concepts tested.  

- Lower retention performance: The In-Domain Retain Accuracy (IRA) is significantly lower than in strong baselines, indicating that CGRU gives up a lot of utility for unlearning accuracy. According to Table 1, CGRU's IRA is 78.47%, while methods like SalUn and EDiff-UN achieve 96.35% and 94.03%, respectively. The authors do not discuss this critical trade-off enough; they only state in Section 7 that "methods achieving high unlearning accuracy tend to exhibit lower retain accuracy" without any detailed analysis or solutions.

### Questions
- Can the method be demonstrated on a larger scope, as detailed in the weaknesses section?
- Can the authors provide further qualitative evidence to judge generalization abilities?
- Can the authors further discuss/analyze the tradeoff between retention and utility?
- Can the critic be reused across different concepts, or must it be retrained per target?​

### Soundness
3

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
This paper introduces Critic-Guided Reinforcement Unlearning (CGRU), a method for removing specific concepts from text-to-image diffusion models by training a aper-timestep critic that evaluates noisy intermediate latents to predict the final outcome. Further RL on top of it shows effective results on object removal which is superior to methods relying only on the sparse reward.

### Strengths
1. The timestep-aware critic addresses the high-variance problem of sparse rewards in prior RL-for-diffusion methods, leading to more stable training and better credit assignment.

2. The method achieves state-of-the-art unlearning accuracy on object removing tasks.

### Weaknesses
1. As the target of this paper is for machine unlearning, I didn’t see any specific designs for machine unlearning. The proposed critic seems to be the same as the value function in normal policy gradient for variance reduction techniques. As a result it is more like an RL for diffusion method applied to a specific domain.

2. The value function is also used in methods like DPOK and the proposed method just seems to be more fine-grained such that it is also dependent on the timestep. But there’s no ablation study on whether the value function is dependent on the timestep. 

3. While unlearning accuracy is improved, the model's retain accuracy for related, benign concepts is middling, suggesting the method might be overly aggressive.

### Questions
The machine unlearning accuracy is improved at a cost to retain accuracy. Is this an inherent trade-off in the framework, or could the reward function be designed to better preserve utility for non-target concepts?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper introduces a value function baseline into policy gradient RL training of diffusion models. They do this by fine-tuning a CLIP-based value model that takes in noisy latents and predicts the achieved reward. This should help with reducing variance in RL training and providing better credit assignment.

The paper compares their new technique, dubbed CGRU, against an old baseline DDPO, showing better performance with consistently higher aesthetic scores and better unlearning performance.

### Strengths
For many legal and safety reasons, managing proper unlearning techniques through diffusion models is important, and currently not in a perfect state. So exploring new techniques is significant for the progress of the field. 

I'm not very familiar with the most recent related work in this field, but it sounds like training value model baselines for diffusion model RL is novel (though a very straightforward application of a common RL technique).

The method is straightforward, makes sense, and is presented clearly. Additionally, the initial results look promising – CGRU seems to train more stably and with better end results than DDPO, and achieves a strong balance of unlearning to in-domain retention accuracy.

### Weaknesses
Largely, more experimental results would contribute significantly to the point of the paper. 

- RL training is notoriously unstable, so having at least 3 seeds with error bars in figures 1 and 2 would make me more confident the performance improvement is not just luck.
- Include more non-cherry-picked image grids of generated image examples for the different methods.
(these two are my largest critiques, my score would likely rise if these were addressed)

- In Table 1 I recommend running a few seeds and adding standard deviations. Furthermore I'd suggest making this a graph instead so it's clearer that there' s and IRA/UA tradeoff and your method is on the pareto frontier. It also seems slightly odd to clall CGRU's IRA "competitive" when it seem like a significant decrease compare to the other methods. Maybe this is an ok tradeoff, but I think it would be worth directly talking about how SalUn dominates on IRA and which you'd prefer just depends on where on the pareto frontier you want to be.

There are also some components that are somewhat lacking in clarity. It is unclear from the paper where specific results are taken from, why specific baselines were chosen, or exactly what experiments were run (more on these in the questions section). 

The introduction mentions “ablations show that (i) per-step critics and (ii) noisy-conditioned rewards are key to stability and effectiveness,” but this "stability" argument is never mentioned again in the paper.

### Questions
Why was DDPO chosen as the comparison example? Table 1 shows metrics for many other techniques but not for DDPO, why was DDPO the chosen comparison metric? What are the metrics for UA and IRA for DDPO? 

Did the model unlearn each of the 20 object classes? If so, why are only the results for cats shown? It would be useful to see performance on multiple different classes if all 20 were unlearned, or a chart in the appendix of a summary comparison metric for how well CGRU and DDPO unlearned each class. Within classes, more visual examples from the model generation during training would make the point stronger. 

Some questions regarding Table 1:
“We evaluate CGRU’s performance using established metrics from the UnlearnCanvas benchmark Zhang et al. (2024b)”
In that paper, the results shown for UA differ for the models, and no metrics for IRA are shown. Some are similar, like ESD (92.15% in this paper, 91.42% in the other paper), but others are pretty different, like UCE (94.31% in this paper, 75.97% in the other paper). Where are these metrics coming from? Were these techniques used to retrain the model and recompute the metrics? If not, is it a fair comparison?
The cited paper for UnlearnCanvas also cites three other metrics, SC, OC, and UP. Are these useful metrics that are worth including?

Not necessary to implement, but I wonder if you need to train a separate critic model per-dataset? Or could you train e.g. a generic aesthetics critic and apply it to all aesthetics finetuning jobs going forward?

### Soundness
3

### Presentation
3

### Contribution
2
