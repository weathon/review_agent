# Generative Adversarial Post-Training Mitigates Reward Hacking in Live Human-AI Music Interaction

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Most applications of generative AI involve a sequential interaction in which a person inputs a prompt and waits for a response, and where reaction time and adaptivity are not important factors. In contrast, live jamming is a collaborative interaction that requires real-time coordination and adaptation without access to the other player’s future moves, while preserving diversity to sustain a creative flow. Reinforcement learning post-training enables effective adaptation through on-policy interaction, yet it often reduces output diversity by exploiting coherence-based rewards. This collapse, known as ``reward hacking'', affects many RL post-training pipelines, but is especially harmful in live jamming, where musical creativity relies on dynamic variation and mutual responsiveness. In this paper, we propose a novel adversarial training method on policy-generated trajectories to mitigate reward hacking in RL post-training for melody-to-chord accompaniment. A co-evolving discriminator separates policy trajectories from the data distribution, while the policy maximizes the discriminator output in addition to coherence rewards to prevent collapse to trivial outputs. We evaluate accompaniment quality and output diversity in simulation with both fixed test melodies and learned melody agents, and we conduct a user study with the model deployed in a real-time interactive system with expert musicians. Quantitative evaluation and user feedback demonstrate improved output diversity, harmonic coherence, adaptation speed and user agency. Our results demonstrate a simple yet effective method to mitigate reward hacking in RL post-training of generative sequence models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes GAPT (Generative Adversarial Post-Training) to solve the problem of reward hacking in human-music interaction. The paper argues that current methods for RL post training in music related scenarios hacks the reward but leads to repetitive or low diversity music samples. To solve this, the authors add a discriminator component to distinguish between policy generated vs real-data trajectories. Since GAN based training can be unstable, they also propose training techniques to stabilize the training.

### Strengths
The paper addresses an important problem in live music jamming. The proposed solution of having an adversarial loss is simple and seems to work. Moreover, the authors have good evaluation setting, they evaluate on three scenarios: fixed-melody, model-model and real-time with musicians. The paper is very well written and easy to understand.

### Weaknesses
1. The scope of experimentation is narrow. For example, the proposed algorithm is tested on only melody-chord accompaniment. Similarly, the method is limited to T<=256 frames, where each frame is a sixteenth note. There are no studies in the paper that explains what happens for greater timesteps, is the training still stable?

2. The authors use note-in-chord ratio which is a very simple metric. While the qualitative results suggest that GAPT performs better, from Table 1 and 2, we see minimal improvement compared to other baselines. For instance, Table 2 shows that Online MLE performs better than GAPT in everything except for harmony in user interaction (0.448 vs 0.467 (GAPT)). The quantitative results are not convincing enough to prove the efficacy of this method.

### Questions
1. Could you provide empirical analysis of your method's performance and training stability for sequences longer than 256 frames? This is important because real musical interactions often extend beyond this length, and understanding any degradation in performance or training stability would significantly impact the method's practical utility.

2. In Table 2, Online MLE outperforms GAPT in most metrics except for harmony in user interaction. Could you explain this discrepancy and provide additional analysis to justify why GAPT should be preferred over the simpler Online MLE approach?

3. How does this method work for other musical styles?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposed Generative Adversarial Post-Training (GAPT) for live melody→chord accompaniment. A discriminator is trained to prevent reward hacking, ensuring the realism of the accompaniment. The experiment results show that with GAPT, the model outputs more diverse content.

### Strengths
1. Promise to release training datasets, model checkpoints, and code.
2. Well written, well structured, clear problem and motivation.
3. Comprehensive experiments show the improvement of diversity, harmonic coherence, adaptation speed and user agency.

### Weaknesses
1. The chord set of the model is not specified; note-in-chord could be inflated by choosing chords with more pitch classes.
2. The authors did not benchmark against other RL post-training methods.

### Questions
1. What if the discriminator conditioned on melody D(x,y)?
2. What’s the latency budget and end-to-end timing under live use? Does GAPT change compute/latency vs RealChords?
3. In the appendix, you mentioned that for all user study sessions, you set the tempo to 80 BPM. Does that mean when participants are improvising, the bpm should be set to 80?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work explores the use of adversarial learning to mitigate the reward hacking problem in reinforcement learning for the task of melody-to-chord accompaniment generation.
Reward hacking in RL often arises because (1) human-defined rewards are incomplete and may lead the policy to converge to a local minimum, and (2) RL agents cannot distinguish out-of-distribution (OOD) samples, causing them to over-rely on the reward signal.

What makes this paper interesting is how it integrates adversarial learning into the RL framework. Unlike other methods (e.g., [1]), the authors adopt a simple yet direct strategy: they add the discriminator term −log(1−y) directly to the reward. In this way, if the discriminator considers a generated sample to be “real,” the policy receives an additional (and potentially large) reward boost.

Experimentally, the authors demonstrate strong results in both quantitative metrics and perceptual evaluations. As an amateur music enthusiast, I can clearly perceive the improvement brought by the adversarial reward.

[1] Bukharin et al., Adversarial Training of Reward Models, 2025.

### Strengths
**Technical Comments**

1. Exploring adversarial learning in the context of melody-to-chord accompaniment is quite novel. Moreover, directly integrating the discriminator’s output into the reward function is a simple yet effective strategy, as demonstrated in the paper.

2. The authors present several insightful analyses and results. For example, in Figure 4, they show that the adversarial reward encourages more diverse generation outputs and expands the representational space. This finding is particularly interesting, since—according to the authors—other components such as entropy regularization remain unchanged compared to the w/o-adversarial baseline.

**Presentation**

The writing is clear and easy to follow, and the examples used in the comparative experiments are also quite convincing when listened to.

**Experiments**

1. For the main task, the authors demonstrate significant improvements over both the baseline and the w/o-adversarial-reward setting, in terms of quantitative metrics and subjective evaluation.
2. The ablation study is also comprehensive and effectively supports the paper’s claims.

### Weaknesses
**Technical Comments**

1. Since the adversarial reward takes values in  [0,+∞) and is directly added to the original reward, does this cause any numerical instability during training? Has the author conducted experiments to verify its stability?

2. Adversarial training (GAN) itself is known to be unstable. Did the author perform repeatability experiments to show the variance across different random initializations?

3. What does the reward convergence curve look like during training? It seems that this information is not presented in the paper.

4. Could the author show how the critic value differs between the models with and without adversarial reward? (I notice the authors use PPO instead of GRPO.)
Additionally, could they run an experiment using GRPO, since GRPO normalizes rewards using the group/batch average?
Given that the adversarial reward is always positive, it might shift the overall value distribution upward, potentially affecting learning stability and normalization.

5. Is the discriminator’s training process stable? What does its learning curve look like over the course of training?


I am very interested in this paper and I genuinely like the direction. My current score mainly reflects the fact that I would like to see a clearer and more complete picture from the authors, so that the contribution becomes more solid. If the authors can provide more thorough/deep analysis in the rebuttal or revision, I would be happy to raise my score.

### Questions
see above

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
The paper proposes Generative Adversarial Post-Training (GAPT) for real-time melody-to-chord accompaniment. GAPT introduces a discriminator to avoid reward hacking in RL finetuning. Evaluation shows this simple yet effective GAPT framework works to mitigate reward hacking.

### Strengths
1. Evaluation metrics are reasonable. Note-in-chord ratio allows flexibility because of the nature of real-time accompaniment, but at the same time evaluates the quality of chords.
2. Adversarial realism reward is easy to add and conceptually orthogonal to KL/entropy regularization. The confidence-gated update is a neat stability trick.
3. The RL post training scalar reward not only include discriminative models, but also include music rules which mitigate the bias brought by discriminative model as well.
4. Appreciate the examples provided to have a straightforward evaluation in ears.

### Weaknesses
1. I would appreciate that if more ablations study would be completed on GAPT itself, especially on the scalar reward in Equation (6), applying different weights to see the behavior of models would be a helpful study to provide more insights.

### Questions
1. What could be the reason for harmonic incoherence (like in Hooktheory Example 4)? Is it because the melody has higher density of notes in beats comparing to other successful accompaniments? When GAPT underperforms harmony (e.g., slight drop on OOD), what musical patterns cause it? Maybe because of the training dataset?
2. What are end-to-end latency numbers (ms) and GPU budget during live use? How does lookahead/commit horizon trade off with perceived responsiveness?
3. Would you mind elaborating a bit more on applying GAPT outside real-time accompaniment?

### Soundness
3

### Presentation
3

### Contribution
3
