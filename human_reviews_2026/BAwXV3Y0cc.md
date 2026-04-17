# TimeRewarder: Learning Dense Reward from Passive Videos via Frame-wise Temporal Distance

- Decision: Reject
- Scores: 4, 6, 6, 2

## Abstract
Designing dense rewards is crucial for reinforcement learning (RL), yet in robotics it often demands extensive manual effort and lacks scalability. 
One promising solution is to view task progress as a dense reward signal, as it quantifies the degree to which actions advance the system toward task completion over time.
We present TimeRewarder, a simple yet effective reward learning method that derives progress estimation signals from passive videos, including robot demonstrations and human videos, by modeling temporal distances between frame pairs.
We then demonstrate how TimeRewarder can supply step-wise proxy rewards to guide reinforcement learning.
In our comprehensive experiments on ten challenging Meta-World tasks, 
we show that TimeRewarder dramatically improves RL for sparse-reward tasks,
achieving nearly perfect success in 9/10 tasks with only 200,000 interactions per task with the environment. This approach outperformed previous methods and even the manually designed environment dense reward on both the final success rate and sample efficiency.
Moreover, we show that TimeRewarder pretraining can exploit real-world human videos,
highlighting its potential as a scalable approach path to rich reward signals from diverse video sources.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a reward-learning method that turns passive videos (robot or human) into dense, step-wise rewards for RL. In detail, they learn a model to predict a frame-wise temporal distance between two observations. At RL time, the predicted temporal distance between adjacent frames serves as a progress reward, optionally combined with a sparse success signal. The authors emphasize three training choices: (i) implicit negatives via forward/backward ordering to encode regress vs. progress, (ii) exponentially weighted pair sampling to focus on short temporal gaps, and (iii) two‑hot discretization for stable training. For experiments, the method reportedly reaches near‑perfect success on 9/10 tasks with 200k interactions and, in most cases, outperforms both prior video‑based contenders and the environment’s dense reward on Meta‑World tasks.

### Strengths
- Simple objective with clear inductive bias. Framing reward learning as temporal distance prediction is intuitive and easy to implement.
- Thorough ablations on design choices. The study isolates the contributions of implicit negatives, weighted sampling, and two‑hot discretization. It is useful for practitioners.

### Weaknesses
- Originality is thinner than claimed. Temporal Distance Classification (TDC) was introduced in [1]; it discretizes time differences and trains a classifier on video frame pairs, which is very close to this paper’s core target. Related pretext tasks like Time‑Contrastive Networks and Temporal Cycle Consistency similarly extract progress‑like signals from temporal structure [2,3], and Shuffle&Learn / “arrow‑of‑time” also leverage ordering cues [4]. The paper cites TCN but does not cite TDC/TCC or clearly differentiate from them.
- Connections to recent progress‑from‑video rewards are not discussed. TimeRewarder is close in spirit to VIP (value‑implicit pretraining), GVL (VLMs as in‑context value estimators), Rank2Reward (temporal ranking for shaped reward), and HashReward (progress reward with online hashing). The paper positions against these but omits or soft‑pedals some important links and failure modes already discussed there (e.g., distribution shift and online refinement) [5–8].
- The paper gestures at potential‑based shaping $R(o_t, o_{t+1})=V(o_t) - \gamma V(o_{t+1}))$ but the learned signal is pairwise $F_\theta(o_t, o_{t+1})$, not an explicit potential $V(o). Without showing that $F_\theta$ ntegrates to a state potential (or at least is path‑independent), there is no policy‑invariance guarantee. In the absence of this, shaped rewards may induce loops or reward hacking in off‑expert states [9].

## Reference
[1] Playing Hard Exploration Games by Watching YouTube.

[2] Time-Contrastive Networks: Self-Supervised Learning from Video. 

[3] Temporal Cycle-Consistency Learning.

[4] Learning and Using the Arrow of Time.

[5] VIP: Towards Universal Visual Reward and Representation via Value-Implicit Pre-Training.

[6] Vision Language Models are In-Context Value Learners.

[7] Rank2Reward: Learning Shaped Reward Functions from Passive Video.

[8] Imitation Learning from Pixel-Level Demonstrations by HashReward.

[9] Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping.

### Questions
Please refer to the weaknesses.

### Soundness
2

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
4

### Summary
Time rewarder presents a method of deriving a reward function from expert demonstrations through predicting task progress. Their reward model predicts the relative distance between two frames, normalized from -1 to 1, measuring both forward and negative progress. Time rewarder beats out chosen baselines, including both learning from observations only (OT ADS GAIfO) and other IRL methods (VIP Rank2Reward PROGRESSOR). They also include experiments that co-train their reward with human videos, and a provide a theoretical motivation for progress rewards.

### Strengths
This paper includes strong baseline comparisons across multiple tasks. They also included ablation studies explaining each of their design decisions, such as descritization, negative sampling, and relative vs. absolute progress. It is well written and well executed. The results on cross-embodiment reward transfer are also strong.

### Weaknesses
Using task progress or temporal ordering as a reward signal is not particularly novel. Several prior works (VIP, Rank2Reward, PROGRESSOR) already use temporal structure for reward learning. They note that other ranking-based methods are difficult to optimize reliably, and that rank2reward and PROGRESSOR report poor performance on out-of-distribution states. However, for their data used to train the ranking function, they collect demonstrations “under a deliberately diverse initialization protocol.”  

Additionally, Time Rewarder uses a CLIP-pretrained ViT backbone. Prior work like RoboCLIP [1] has shown that a CLIP-pretrained representation can produce meaningful reward signals. It would help to isolate the contribution of the proposed method from that of the pretrained features. One way to do this would be a comparison using a non-contrastively pretrained backbone.

[1]  Sumedh Anand Sontakke, Jesse Zhang, Séb Arnold, et al. RoboCLIP: One Demonstration is Enough to Learn Robot Policies.

### Questions
* Given that the demonstration set is deliberately diverse, does the time rewarder require state coverage of expert demonstrations to ensure relative progress does not go out of distribution?  How does Time Rewarder perform under the standard expert demonstrations for Meta World?
* The behavior cloning policy seems to perform poorly. If the authors could explain why this occurs in their setting, it would help contrast with their method. 
* What enables TimeRewarder to outperform dense-environment rewards? 
* For co-training on the human videos, if you use both the human videos and the 100 meta-world demos, does this further improve performance compared to just the 100 demos?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes TimeRewarder, a method for learning dense rewards from action-free expert videos.
The key idea is to model task progress as a temporal distance prediction problem between frame pairs.
By self-supervisedly learning the temporal gap between two observations, the model implicitly captures how far the agent has progressed toward task completion.
The approach integrates three main components: implicit negative sampling, exponentially weighted pair sampling, and two-hot discretization.
Theoretically, the learned temporal distance corresponds to a potential-based reward-shaping term, ensuring Bellman consistency.
Experiments on ten Meta-World manipulation tasks show that TimeRewarder's rewards remain highly monotonic on unseen expert trajectories, successfully distinguish failed rollouts, and enable reinforcement learning agents to achieve top success rates.
Overall, TimeRewarder is simple, theoretically grounded, and empirically strong.

### Strengths
1. The paper presents a simple yet effective idea, treating reward learning as temporal-distance estimation, which provides a clean, self-supervised way to learn dense progress signals without requiring expert actions or environment rewards.

2. The connection between temporal distance and potential-based reward shaping gives the method solid theoretical justification and ensures consistency with RL principles.

3. Evaluations on ten Meta-World tasks are extensive, showing high sample efficiency and outperforming both prior progress-based methods and imitation baselines.

4. Through implicit negative sampling, the model can recognize and penalize regressive or failed behaviors, improving policy learning stability.

5. The ablation studies carefully isolate and validate the contribution of each design component, reinforcing the method’s architectural soundness.

### Weaknesses
1. The cross-domain generalization largely stems from the CLIP backbone's semantic representations rather than the proposed temporal-distance formulation itself. The method benefits from CLIP's object-centric and domain-invariant features, but this reliance makes it unclear how well TimeRewarder would perform with less powerful or task-specific encoders.
The paper does not analyze how different visual backbones or representations affect performance, making it hard to disentangle the contribution of TimeRewarder from that of pretrained features.

2. The core idea, learning task progress from temporal consistency, builds on prior progress-based reward learning work (e.g. VIP, PROGRESSOR, Rank2Reward) with incremental architectural refinements rather than a fundamentally new principle.

3. The method assumes that expert trajectories reflect smooth, monotonic advancement toward a goal. This assumption breaks in multi-stage or looping tasks where temporal distance no longer correlates with true progress.

4. The reward predictor is fixed during RL, which limits adaptability to out-of-distribution states encountered in exploration.

### Questions
1. In the human-to-robot transfer experiment, was one model trained per task, or a single model shared across all three tasks?

2. How much of TimeRewarder's generalization depends on the CLIP backbone? It would be helpful to include an ablation using non-pretrained or weaker encoders (e.g., a randomly initialized ViT or a task-specific CNN) to quantify the backbone's contribution.

3. The reward network is frozen during reinforcement learning. Would online fine-tuning or adaptive updating improve robustness to out-of-distribution states encountered during policy exploration?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes TimeRewarder, a method that learns a dense, step-wise reward for reinforcement learning directly from passive expert videos by predicting the temporal distance between pairs of frames. The authors propose learning a model that outputs per-step rewards along an agent’s trajectory that signal progress in successful completion of the task, combined with weighted sparse task reward. To stabilize their proposed method, the use the following:
1. *Implicit Negative Sampling*: The word sampling is misleading here. They mean to say that their temporal distance reward can be positive or negative depending on the order of the frames in the recorded trajectory
2. *Exponentially Weighted Pair Sampling*: Sample adjacent frames more frequently than distant frames
3. *Two-hot Discretization*: An auxiliary loss to stabilize training; this seems to stem from an intuition that classification losses are more numerically stable than regression losses in deep learning.

The evaluate their results in simulation (MetaWorld) and compare with some relevant competitors.

### Strengths
1. Clear motivation and problem setting
  - Addresses the practical challenge of dense reward design for manipulation from only passive videos, without action annotations.
  - Connects progress-based shaping to a directly learnable quantity—temporal distance—which is readily available from videos.
2. Simple idea to benefit from human demonstrations provided using videos

### Weaknesses
**Writing quality and clarity**
The paper’s presentation can be improved. Several sections, particularly the method description and experimental setup, lack clarity. The idea can be stated in a far simpler way than the current form. 

**Overstated empirical claims**
The paper overstates the efficacy of the proposed method. Reported improvements are not statistically significant in several plots, and standard errors overlap in almost all cases. The tone implies strong superior performance, but the evidence supports minimal gain at best. 

**Limited applicability beyond simple goal-reaching tasks**
The method appears well suited for short-horizon goal-reaching tasks  (go from point A to point B). It is unclear how it would perform in environments with loops, subgoals, or frequent revisits to the same state. Because the reward is based on framewise temporal distance, states with high visitation counts is likely to confuse the model and degrade performance.

**Weak cross-domain evidence**
The claim of strong cross-domain generalization is not sufficiently supported. The real human video experiments appear tightly controlled: object positions and scene layouts closely match the simulation, with only the robot hand replaced by a human one. This suggests limited diversity and minimal domain shift. Additional experiments with different backgrounds, object appearances, and camera viewpoints would be needed to substantiate the cross-domain claim.

### Questions
1. **Statistical significance of results:**
   In most figures (for example, Figures 3, 5, 6b, and 7), the standard error bands of different methods overlap substantially. On what basis do you claim that TimeRewarder performs better? The results as shown do not appear statistically significant.

2. **Generalization to out-of-distribution trajectories:**
   How does TimeRewarder handle out-of-distribution data? With only a few demonstrations, it seems unlikely that the model can generalize to every possible trajectory. In Meta-World, what happens if the robot begins from a different initial configuration or follows a novel trajectory? How would this affect the predicted reward? Would this be a fundamental limitation of the approach?

3. **Weighting of the success reward:**
   Why is a separate weighting factor for the success reward needed in Equation 7? How sensitive is performance to the choice of $\alpha$, and how difficult is it to tune in practice?

4. **Choice of discretization bins:**
   The two-hot discretization uses (K = 20) bins. How was this number selected, and how sensitive are the results to this value?

5. **Transfer to real robots:**
   All reported experiments appear to be conducted in simulation. How do you expect the method to extend to real-world robotic systems, given practical challenges such as lighting variation, occlusion, and partial observability?

### Soundness
1

### Presentation
1

### Contribution
2
