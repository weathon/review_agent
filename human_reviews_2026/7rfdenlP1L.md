# Understanding and Improving Hyperbolic Deep Reinforcement Learning

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 6

## Abstract
The exponential volume growth of hyperbolic geometry can embed the hierarchical relationships between states in reinforcement learning (RL) with far less distortion than Euclidean space. However, hyperbolic deep RL faces severe optimization challenges, and formal analysis of why optimization fails is lacking. We identify key factors that determine the success and failure of training hyperbolic deep RL agents. By analyzing the gradients of core operations in the Poincaré Ball and Hyperboloid models of hyperbolic geometry, we show that large-norm embeddings destabilize gradient-based training, leading to trust-region violations in proximal policy optimization (PPO). Based on these insights, we introduce Hyper++, a new hyperbolic deep RL agent that consists of three components: (1) feature regularization guaranteeing bounded norms while avoiding the curse of dimensionality from clipping; (2) a categorical value loss for stable critic training; and (3) a more optimization-friendly formulation of hyperbolic network layers. On ProcGen, we show that Hyper++ guarantees stable learning, outperforms prior hyperbolic agents, and reduces wall-clock time by approximately 30%. On Atari-5 with Double DQN, Hyper++ strongly outperforms Euclidean and hyperbolic baselines. We release our code at https://github.com/Probabilistic-and-Interactive-ML/hyper-rl.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes an extension of the Hyperbolic Deep RL (HyperRL) framework from [1]. In particular, the paper analyzes the instabilities in prior hyperRL methods and proposes several improvements to counteract them. These improvements come together in the  Hyper++ algorithm, which extends the implementation from [1] with a categorical value function, a feature-scaled RMSNorm layer, and switching from the Poincaré to the Hyperboloid model as the space for the latent features. Empirically, the proposed method is validated on the Procgen benchmark [2], and a subset of three environments from the Atari benchmark [3].


[1] Cetin, Edoardo, et al. "Hyperbolic deep reinforcement learning." arXiv preprint arXiv:2210.01542 (2022).

[2] Cobbe, Karl, et al. "Leveraging procedural generation to benchmark reinforcement learning." International conference on machine learning. PMLR, 2020.

[3] Bellemare, Marc G., et al. "The arcade learning environment: An evaluation platform for general agents." Journal of artificial intelligence research 47 (2013): 253-279.

### Strengths
- I found the overall exposition of the paper to be clear and intuitive. Clear text boxes are used to highlight the main methodology aspects, which are introduced naturally within the text.  While hyperbolic geometry for ML could be a challenging topic for some readers, I believe the paper does a good job at summarizing past results and providing analysis without overdoing the theoretical complexity.
- Each component of the proposed extensions seems logical and introduced from grounded considerations on the challenges of applying hyporbolic ML to reinforcement learning.
- The paper does a good job of introducing itself within the surrounding literature of RL and hyperbolic deep learning.

### Weaknesses
**Main:**
My main area of concern is the empirical validation of the paper. The proposed Hyper++ algorithm has only been properly examined on Procgen on top of vanilla PPO. The generality of the method is yet to be validated:
- Since its inception, several improvements upon PPO have been proposed for Procgen. It would have been nice to test if Hyper++ could also be applied to improve the performance of any more recent baselines from the last couple of years.
- The evaluation for DDQN seems to be constrained to only three handpicked environments from the whole suite. Not only is this too limited to provide evidence of generality, but it is also using a very outdated baseline in DDQN (9+ years old), which is far from state-of-the-art in sample efficiency.
Without empirical validation to potentially establish the methodology, I am unsure how much the proposed simple extension to the existing hyperRL framework would be relevant to the broader ICLR RL community.

**Other:**
1) The ablation study is only focused on three Procgen environments, rather than looking at average normalized performance across the Procgen suite, providing limited insights. Moreover, I think the ablation for RMSNorm should have added back the SpectralNorm introduced in the original HyperRL work. 
2) The authors recollect their own numbers for the HyperRL baseline, which appear consistent but slightly inferior to the original results reported in this prior work. Yet, even with these baseline numbers, performance improvements of the proposed method on Procgen appear far from consistent (improving in only 8/16 and 11/16 train/test settings, respectively).
3) The paper lacks any analysis on what is the actual effect of each of their proposed addition and their effect on the learned latent representations, which is the critical main aspect of adding hyperbolic space to RL.
4) While it is good that some code was shared with the submission, it is currently far from allowing/facilitating reproducibility for the reviewers and the public. For instance, the README is empty without installation/execution instructions. I quickly did some checks trying to install the dependencies with uv and launching the ."sh" files, but was not able to get experiments started. I think for the code to be actually useful to reviewers, a simple README with instructions should have been provided at the time of submission.

### Questions
I have raised what I believe to be the main areas for improvement of the paper in the Weaknesses Section above. I would encourage the authors to address these aspects, and have no further questions at this stage.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes HYPER++, a modified hyperbolic PPO agent with: (i) a stable critic using categorical value loss, (ii) feature regularization to control embedding norms, and (iii) improved hyperbolic layer formulations. On ProcGen benchmarks, HYPER++ achieves more stable learning, better performance, and 30% faster training than previous hyperbolic RL methods.

### Strengths
- The paper is overall well-written and combines interesting ideas
- There is a quite strong theoretical section

### Weaknesses
None of the idea alone is truly novel and only the combination of several ideas. I therefore see the paper as largely empirical in scope. There is a 30% improvement on procgen as well as small improvements in 3 Atari games. However, the experimental setup is not fully convincing:
- relatively limited scope
- the interpretation of the improvements in practice are not fully clear

### Questions
- Since each component of HYPER++ (RMSNorm, scaling, categorical loss, hyperboloid model) are not fully novel, what new insight does their combination provide beyond improved stability? In other words, is there theoretical justification for why these specific components interact synergistically, or was the design largely empirical?
- Atari-3 and only 5M training steps is a very limited setup for the Atari benchmark. How do you know that it is representative of a larger set of environments (e.g. all ATARI games) and that the first 5M steps are representative of a longer training?
- How sensitive are the results to hyperparameter choices (e.g. NN architecture, etc.)
- The ablation results for HYPER++ in Table 1 are only reported on the 3 games with the highest improvements with the proposed architecture.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper aims to stabilize the learning process of PPO agents with hyperbolic representation. Authors provide empirical (Figure 2) and theoretical evidence (Section 3.2-3.4) that the instabilities arise in unstable gradients induced by nonstationarities in RL, large euclidean feature norms and exploding conformal factors in Poincare ball formulation of the hyperbolic space.

To this end, authors make three improvements to the naive Hyper method:
1. Categorical value function (HL-Gauss) to stabilize gradients
2. RMSNorm to prevent feature norm growth
3. Hyperboloid model (in replacement of Poincare model) to remove the conformal factor

This results in a superior performance in 4 Procgen and 3 Atari environments, compared to prior work (Hyper + S-RYM) and PPO with default Euclidean space.

### Strengths
- **Rich theoretical results**. Thorough analysis of the feature norms and gradients for hyperboloid MLR seems to be a contribution by itself. It also gives enough justification to the components such as RMSNorm and Hyperboloid model, which are also verified in their ablation studies.
- **Careful implementation details**. When replacing SpectralNorm with RMSNorm, authors also point out the exponential shrinkage of the hyperbolic space after normalization and proposes to rescale the features accordingly.

### Weaknesses
- **Lack of motivation for HL-Gauss**. Although all other components seem to be well-motivated, there are no sufficient justifications or observations that lead to the addition of categorical loss. Indeed, the benefits of categorical loss is not limited to hyperbolic DRL but for most DRL methods in general [1,2,3].
- **Limited evaluation setup**. The proposed method is verified in 4 ProcGen and 3 Atari environments, which is significantly smaller compared to the prior work by [4]. I suggest expanding the benchmark size, or at least incorporate environments where prior work (PPO + S-RYM) has degraded performance compared to PPO, such as `heist`, `caveflyer` and `ninja`.

Overall, I think the results are a bit weak to confidently accept this paper. However, I may be overlooking the theoretical value of this paper, as I do not have the sufficient theoretical backgrounds for it. If the authors are able to provide more empirical results, I will positively consider raising my score.

[1] FastTD3: Simple, Fast, and Capable Reinforcement Learning for Humanoid Control., Seo et al., ArXiv'25.

[2] Bigger, Regularized, Categorical: High-Capacity Value Functions are Efficient Multi-Task Learners., Nauman et al., ArXiv'25.

[3] Hyperspherical Normalization for Scalable Deep Reinforcement Learning., Lee et al., ICML'25.

[4] Hyperbolic Deep Reinforcement Learning., Cetin et al., ArXiv'22.

### Questions
- Have you tried C51-style categorical distribution instead of HL-Gauss, as a lot of works in RL tend to adapt the former [1,2,3] instead of HL-Gauss?
- Line 60-63: I think claiming that "PPO with a hybrid Euclidean–hyperbolic encoder is the prevalent architecture in deep RL" is an overstatement unless it is supported by a number of citations.

[1] FastTD3: Simple, Fast, and Capable Reinforcement Learning for Humanoid Control., Seo et al., ArXiv'25.

[2] Bigger, Regularized, Categorical: High-Capacity Value Functions are Efficient Multi-Task Learners., Nauman et al., ArXiv'25.

[3] Hyperspherical Normalization for Scalable Deep Reinforcement Learning., Lee et al., ICML'25.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper studies why deep RL agents with hyperbolic latent spaces are hard to optimize and proposes HYPER++, a PPO-based agent that stabilizes training and improves performance on ProcGen and a small Atari subset. The analysis traces instability to (i) exploding/vanishing gradients induced by the conformal factor and exponential map in the Poincaré ball, (ii) exponential growth in the Hyperboloid exponential Jacobian for large Euclidean feature norms, and (iii) non-stationary critics. The method has three ingredients: (1) replace value regression with a categorical value loss; (2) add RMSNorm at the last Euclidean layer plus a learned feature-scaling that caps norms in an optimization-friendly way; and (3) prefer the Hyperboloid model for the final layers. Empirically, HYPER++ improves median/IQM/mean normalized test rewards on ProcGen with better update-KL/clip-fraction metrics and ~30% wall-clock reductions versus prior hyperbolic agents, and transfers to DDQN on Atari-3.

### Strengths
**Clear diagnosis of instability.** The chain-rule decomposition (Eq. 3) and analysis of the conformal-factor gradient (Eq. 4) and HNN++ MLR derivative (Eq. 5) convincingly explain Poincaré instability; the Hyperboloid exponential Jacobian (Eq. 6) highlights a distinct failure mode—large Euclidean norms—justifying feature-norm control. RMSNorm placed only at the last Euclidean layer, plus a learnable radius cap, maintains capacity without per-layer SpectralNorm overhead; the bound in Prop. 4.2 ties directly to conformal-factor control.

**Well-motivated critic choice.** Switching to a categorical value loss is compatible with distributional RL’s stability theory and practice, and the ablation (+MSE) shows degradation on noisy-reward games. 
 
**Comprehensive PPO metrics.** Reporting entropy, update-KL, clip-fraction, and gradient norms is great and aligns with best-practice PPO diagnostics. 

**Empirical evidence.** Consistent ProcGen gains across aggregation metrics and competitive Atari-3 results demonstrate utility; reductions in wall-clock (~30%) are practically meaningful.

### Weaknesses
**Breadth of algorithms.** Off-policy validation is limited to DDQN. I think modern strong baselines (e.g., SAC, DrQ-v2) would better test generality. Ablations on Euclidean + categorical critic and Euclidean + RMSNorm controls would isolate which pieces help independent of hyperbolic geometry.

**Scope of environments.** ProcGen is appropriate, but analysis claims about hierarchy/tree-like structure would be stronger with goal-conditioned or hierarchical RL tasks (MiniGrid, Crafter, options) and long-horizon domains.

**Theory tightness and assumptions.** Some proofs rely on Lipschitz constants and dimension-independent bounds; clarity on tightness and behavior under non-1-Lipschitz activations would help. A formal link from stability bounds to PPO trust-region behavior (e.g., expected update-KL control) would strengthen the story but is not necessary.

### Questions
1. **Categorical critic details.** Which distributional parameterization is used (e.g., HL-Gauss vs. C51-style fixed supports)? How are supports chosen and are they shared across tasks? Could you report critic loss variance to corroborate stability claims? 

2. **RMSNorm placement.** Did you try RMSNorm earlier in the encoder, or combining with SpectralNorm only on the last layer? A small table contrasting capacity vs. stability would help.

3. **Trust-region proxies.** Beyond update-KL and clip-fraction, do you observe changes in policy-entropy across states not in the batch (to quantify unseen-state interference)?

4. **Off-policy breadth.** Any results with SAC/DrQ (image-based), and do your findings about feature-norm control translate when using target networks and Polyak averaging?

5. **Comparisons to Euclidean + categorical.** To isolate hyperbolic benefits, could you add Euclidean + categorical critic and Euclidean + RMSNorm baselines?

### Soundness
3

### Presentation
3

### Contribution
3
