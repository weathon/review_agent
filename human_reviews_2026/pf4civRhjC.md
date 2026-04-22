# Modification-Considering Value Learning for Reward Hacking Mitigation in RL

- Avg Score: 2.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 2

## Abstract
Reinforcement learning agents can exploit poorly designed reward signals to achieve high apparent returns while failing to satisfy the intended objective, a failure mode known as reward hacking. We address this in standard value-based RL with Modification-Considering Value Learning (MCVL), a safeguard that treats each learning update as a decision to evaluate. When a new transition arrives, the agent forecasts two futures: one that learns from the transition and one that does not. It then scores both using its current learned return estimator, which combines predicted rewards with a value-function bootstrap, and accepts the transition only if admission does not decrease that score. We provide DDQN- and TD3-based implementations and show that MCVL prevents reward hacking across diverse environments, including AI Safety Gridworlds and a modified MuJoCo Reacher task, while continuing to improve the intended objective. To our knowledge, MCVL is the first practical implementation of an agent that evaluates its own modifications, offering a step toward robust defenses against reward hacking.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces a new method called Modification-Considering Value Learning (MCVL) to prevent "reward hacking" in reinforcement learning (RL). MCVL works by treating every potential learning update as a decision. When new data arrives, the agent forecasts two possible futures: one where it learns from this data and one where it ignores it. Only accepts the update if it doesn't lead to a worse outcome by its own present standards.

### Strengths
- Novel intuition to address the issue of reward hacking
- Good Ablation Studies: The authors thoroughly test why their method works. They show that the forecasting step is essential (a simple one-step check fails) and that their method is robust to some noise and variations in setup.

### Weaknesses
- Pretraining Requirement: The entire method relies on starting with a "good" value estimator. To do this, it must be pretrained on a "seed dataset" that is guaranteed to not contain any reward-hacking transitions. This might be easy in a simulated "Safe" environment (as they use) but could be very difficult or impossible to guarantee in a complex, real-world setting.
- Requires a Transition Model: To score the "forecasted" policies, the agent needs to run rollouts in the environment. In the paper's implementation, this uses the true environment simulator. This limits its use in real-world, model-free scenarios where a perfect simulator isn't available (though the authors suggest a learned model could be a target for future work).
- The authors note, if the pretraining data already contains and endorses a misspecified reward, MCVL will not be able to identify or fix it.
- Computational cost: Forecasting two separate futures for each potential update is computationally expensive. The authors note a 1.8x slowdown in one experiment, and this overhead would likely increase with the complexity of the task.

### Questions
My main questions concern the method's practicality. Could the authors discuss any strategies to: a) Adapt the method for settings where a perfect environment model is unavailable? b) Handle the challenge of acquiring a "guaranteed safe" seed dataset in a real-world task? c) Thorough analysis of computational cost on realistic environments.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces MCVL, a wrapper for off-policy value-based reinforcement learning algorithms that evaluates each new transition by simulating two short training branches: (i) continuing without the transition and (ii) incorporating it. The transition is accepted only if the resulting policy from the second branch achieves a higher short-horizon return, as measured by a frozen evaluator consisting of a learned reward model and critic. Both branches receive equal training budget, and their performance is estimated via k rollouts of length h. The method assumes access to a small, clean seed dataset to train the evaluator before gating begins. Experiments are conducted on AI Safety Gridworlds and continuous control environments, with ablations exploring horizon effects and computational cost triggers.

### Strengths
The accept/reject rule is easy to implement with DDQN/TD3 and isolates the marginal effect of admitting a transition via matched compute and a frozen evaluator.

Author use some compute controls in the implementation

### Weaknesses
Self-referential gate is basically a form of current-utility myopia. The accept/reject rule compares two short-horizon forecasts using the agent’s own, frozen reward model and admits the transition iff the current new one is better. This only guarantees consistency with the current surrogate, not improvement on the true objective; if the evaluator is miscalibrated, it can entrench its own errors. How could authors deal with this case? Also, RL is about exploration and exploitation, how could the author guarantee that the model could still  sufficiently explore the environment? The method explicitly relies on this evaluator and a finite horizon which does not match most of the RL practice.

The authors state the approach works provided the learned evaluator scores hacking-inducing trajectories below non-hacking ones at the scoring horizon—otherwise harmful updates are admitted. That is a strong assumption and shifts the burden from learning to reward/evaluator specification; there’s no guarantee this holds beyond the gridworlds. Also, they require a seed dataset “without hacking” so the evaluator is “meaningful.” How could the author guarantee such a clean dataset? This creates distribution-shift fragility: later sections show detection quality drops when the Full world differs visually or topologically from the seed dataset. Besides, there’s no theory showing robustness to such shifts. 

Acceptance is judged with h-step bootstrapped returns. The paper itself notes that too small h flips the decision; the fix is to increase h, but there’s no bound relating h and the regret of the gate—so theory can’t guarantee you won’t reject necessary long-term improvements or accept subtle long-term hacks.

Only tested on few toy environments; Not sure about the performance on more complex tasks.

### Questions
see weakness

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper proposes Modification-Considering Value Learning (MCVL) to mitigate reward hacking in off-policy value-based reinforcement learning algorithms. It evaluates an arriving transition by either incorporating it or ignoring it, and the transition is accepted only if the resulting policy of inclusion does not reduce the return. In a word, it works by forecast-and-score. The authors implement this idea with DDQN and TD3 on both discrete and continuous controlled environments, and the results show that with MCVL these algorithms are less likely to fall into the reward hacking regime compared to without it.

### Strengths
- It is a simple algorithm with good intuition to mitigate reward hacking, it is also easy to implement.
- The performance of base learners with MCVL is comparable to the Oracle (if available), and the ablation study demonstrates the effectiveness of forecast (as compared to a simple each-step check) and reject (as compared to penalize), which sit at the core of the proposed method.

### Weaknesses
- The paper makes strong assumptions on clean seed dataset without hacking transitions in order to obtain a reasonable value estimator. My concerns are about the practicality due to such assumptions beyond the simple, controlled simulated environments. The self-consistency local test by comparing returns is only about the consistency with the current value estimator, not necessarily the true objective. It adds another reliance on a good evaluate estimator. In other words, the value estimator has to be good enough to score hacking trajectories lower than normal ones, but such reward specifications does not hold beyond the tested simple environments.
- When horizon $h$ is too small, the hacking policy may win as authors pointed out in Appendix E where the fix is to increase $h$, but there is lack of sensitivity test nor any notions of discussion related to the bounds. Basically, the $h$-step bootstrapped return is estimated from a window but we don't know whether the proposed method won't reject (or will explore) long-term improvements.

### Questions
Same as weakness. This is an emergency review so I may have missed some points. If I have misunderstood anything, I kindly ask the authors to clarify.

### Soundness
2

### Presentation
2

### Contribution
2
