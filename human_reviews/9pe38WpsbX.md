# MuDreamer: Learning Predictive World Models without Reconstruction

- Avg Score: 4.33
- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 5

## Abstract
The DreamerV3 agent recently demonstrated state-of-the-art performance in diverse domains, learning powerful world models in latent space using a pixel reconstruction loss. However, while the reconstruction loss is essential to Dreamer's performance, it also necessitates modeling unnecessary information. Consequently, Dreamer sometimes fails to perceive crucial elements which are necessary for task-solving, significantly limiting its potential. In this paper, we present MuDreamer, a reinforcement learning agent that builds upon the DreamerV3 algorithm by learning a predictive world model without the need for reconstructing input signals. Rather than relying on pixel reconstruction, hidden representations are instead learned by predicting the environment value function and previously selected actions. Similar to predictive self-supervised methods for images, we find that the use of batch normalization is crucial to prevent learning collapse. We also study the effect of KL balancing between model posterior and prior losses on convergence speed and learning stability. We evaluate MuDreamer on the widely used DeepMind Visual Control Suite and achieves performance comparable to DreamerV3. MuDreamer also demonstrates promising results on the Atari100k benchmark. Research code will be made available publicly.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper combines the methods from DreamerV3 and MuZero, and presents a new model named MuDreamer for visual reinforcement learning. The key contribution is to introduce a new world model architecture that involves the prediction of environment rewards, value functions, continuation flags, and inverse dynamics. The proposed model showcases a comparable performance to DreamerV3 in multiple domains, including DeepMind Control Suite and the Atari100k benchmark.

### Strengths
1. This paper is well-organized and easy to follow. 
2. The proposed model is extensively evaluated on widely used visual control benchmarks. It also provides comprehensive ablation studies to explore the effectiveness of each model component.
3. The model achieves performance comparable to DreamerV3, and as claimed by the authors, it is more efficient in the training time.

### Weaknesses
1. As stated by the authors, 'MuDreamer solves tasks without the need for a reconstruction loss.' However, this seems to be in contrast with the loss function described in Eq. (3), which still involves optimizing the image decoder with a reconstruction loss. If my understanding is accurate, the distinction from DreamerV3 lies in the fact that the gradient from the reconstruction loss doesn't back-propagate to the dynamics module. In light of this, I recommend that the authors consider revising the paper's title.
2. The proposed model offers limited novelty when compared to DreamerV3. The introduction of inverse dynamics and continuation prediction loss is not a novel contribution in the field of model-based RL.
3. While MuDreamer trains faster than DreamerV3, the difference in training time is relatively modest (4 hours vs. 4 hours and 20 minutes).
4. The outperformance of the proposed model compared with DreamerV3 is observed in only 3 out of 26 games on the Atari100k, which may not be sufficient to establish its overall effectiveness.

### Questions
My main concerns are about the technical novelty and the experimental results. Please see my comments above.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduces MuDreamer, an enhanced version of the DreamerV3 algorithm. MuDreamer eliminates the requirement of reconstructing input signals by learning a predictive world model that predicts the environment value function and previously selected actions. The importance of batch normalization in preventing learning collapse is highlighted, and the impact of KL balancing on convergence speed and learning stability is examined.

### Strengths
The motivation is sound, as this paper combines the strengths of dreamerV2 and MuZero to tackle tasks from image inputs with both continuous and discrete action spaces, without the need for input signal reconstruction.

### Weaknesses
1. Value predictors in this research are inspired by MuZero, and the inclusion of action prediction is a common practice in various model-based approaches. As a result, the novelty may be relatively constrained.

2. The comparison is unfair as it only considers Dreamerv3. It would be more equitable to include more model-based methods for comparison, such as Dreamerpro[1] and denoised MDPs[2]. Dreamerpro, in particular, is a highly relevant method within the domain of Reconstruction-free model-based reinforcement learning.

[1] Dreamerpro: Reconstruction-free model-based reinforcement learning with prototypical representations. ICML 2022.

[2] Denoised mdps: Learning world models better than the world itself. ICML 2022.

3. The experimental results are not satisfactory, as it appears that DreamerV3 performs better. I am aware that MuDreamer has fewer parameters, but it is important to analyze specifically where the differences lie. Please provide an analysis and remove the corresponding parts from DreamerV3 to assess the performance. Additionally, since the authors were inspired by MuZero in several aspects, it would be beneficial to compare this approach as well.

4. Why is batch normalization used instead of other normalization techniques such as layer normalization? Can layer normalization achieve similar effects?

5. Many explanations are not sufficiently in-depth. For example, in KL balancing, why does using a slight regularization of the representations toward the prior with βrep = 0.05 solve both of these issues?

### Questions
Please refer to the weaknesses.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents an exploration of 4 modifications to DreamerV3, and presents promising results on the Control Suite and more limited results on Atari100k.

They assess 4 changes: removing the observation reconstruction loss, adding a previous action prediction head, replacing LayerNorm with batch normalization and changing the weights of the L_dyn and L_rep losses.

Overall, this is a clear and well-executed piece of work, but it has quite limited scope and the results (although promising) aren’t clearly demonstrating a strong benefit over DreamerV3 or EfficientZero.

### Strengths
1. The paper is clear, presents the scope, problem it wants to tackle and related work well.
2. The presentation of the method is clear and the modifications are easy to follow (although quite directly inspired from the Dreamer papers)
3. Results are complete and well presented, with a good coverage of Control Suite results experiments as well as Atari100k. Baselines choices are good.
4. Ablation study of 5.2 is clean and well executed once again.

### Weaknesses
1. The similarities between MuDreamer and DreamerV3 are potentially too strong to make this work significant enough in this state. The paper looks like yet another version of Dreamer, with the exact same math and text dangerously close to a copy, with only a few extra ablations and modifications.
2. Results aren’t as clear-cut as I’d like. There have been a lot of MBRL papers in recent years which explored many combinations of losses, models, actors, but it is quite hard to find which components really matter. 
   1. The Control Suite results are slightly better, but not groundbreakingly so
   2. The Atari100k results aren’t that competitive, especially compared to EfficientZero which would be the clear baseline if one would take the strict desire of not learning to reconstruct observations.
   3. Figure 5 and 6 in the Appendix demonstrates this well, where all curves are fairly similar and do not show a strong enough signal for me.
3. Despite the removal of the observation reconstruction loss, Figure 2 and others indicate that image reconstruction is still done nearly perfectly, which goes counter to the original motivation. It is unclear why that is the case, but it does feel like the model is not as different in what the latent space capture compared to DreamerV3 as it could be?

### Questions
1. Why is Figure 2 so good at reconstructing the observation?
   1. One would have expected to only capture what mattered for the task if the assumptions from the abstract/introduction were true?
   2. Are there games where you have examples of “Dreamer failing to perceive crucial elements”, which MuDreamer does capture?
   3. As it stands, it is unclear to me that the latent space is any different and more abstract than DreamerV3.
2. It would have been interesting to point to specific games where this effect should arise, and make a clear comparison between DreamerV3, MuDreamer and EfficientZero.
   1. For example, having a good score on Frostbite seemed interesting (as it does contain quite a lot of hard details to model well), but looking in the Appendix Figure 6, this seems to be more about 1 seed of DreamerV3 doing badly…
3. Did you explore using the Action predictor network directly for acting, instead of having another Actor network?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
1 poor
