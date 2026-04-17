# Use the Online Network If You Can: Towards Fast and Stable Reinforcement Learning

- Decision: Accept (Poster)
- Scores: 4, 4, 8, 6

## Abstract
The use of target networks is a popular approach for estimating value functions in deep Reinforcement Learning (RL). While effective, the target network remains a compromise solution that preserves stability at the cost of slowly moving targets, thus delaying learning. Conversely, using the online network as a bootstrapped target is intuitively appealing, albeit well-known to lead to unstable learning. In this work, we aim to obtain the best out of both worlds by introducing a novel update rule that computes the target using the **MIN**imum estimate between the **T**arget and **O**nline network, giving rise to our method, **MINTO**. Through this simple, yet effective modification, we show that MINTO enables faster and stable value function learning, by mitigating the potential overestimation bias of using the online network for bootstrapping. Notably, MINTO can be seamlessly integrated into a wide range of value-based and actor-critic algorithms with a negligible cost. We evaluate MINTO extensively across diverse benchmarks, spanning online and offline RL, as well as discrete and continuous action spaces. Across all benchmarks, MINTO consistently improves performance, demonstrating its broad applicability and effectiveness.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes MINTO, a novel update rule for deep reinforcement learning that computes regression targets as the minimum estimate between the online and target networks. MINTO is designed to address the stability-efficiency trade-off inherent in standard RL, where target networks help stabilize learning but slow down convergence, while online networks foster efficiency but risk instability. Experiments on online, offline, discrete, and continuous RL benchmarks such as Atari games and robotic control tasks demonstrate consistent and significant improvements in sample efficiency and final performance over various baselines.

### Strengths
1. The paper is well-motivated and the minimum-operator is easy to integrate into existing RL algorithms.

2. The framework is validated on a wide set of benchmarks (Atari, MuJoCo, offline and online RL) and compared to several baselines, demonstrating consistent gains in both stability and sample efficiency.

3. The modifications require minimal changes to underlying algorithms and introduce only a tiny computational cost, making practical adoption straightforward.

### Weaknesses
1. The main evaluation is conducted against SimbaV1 and SimbaV2, which appear to be less strong baselines. This certainly raises the question whether this method can generalize to strong baselines such as Rainbow.

2. The exclusive reliance on the minimum operator can be too conservative, possibly causing underestimation and inhibiting exploration in low-noise or optimistic environments.

3. The code is not attached and will be released upon acceptance. This raises certain concerns.

### Questions
1. Have you studied the impact of potential underestimation? While you claim that it is slight, is there any theoretical bound to it?

2. The paper briefly mentions possible unexplored interactions between MINTO and exploration strategies. Can the authors clarify how MINTO affects exploratory behavior?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a new method for using the target network in RL. 
Specifically, the authors modify the target computation by taking the minimum of the target Q-network and the online Q-network. 
Albeit simple, it shows substantial improvements across several RL settings, including online RL (DQN, SAC (Simba-V1, Simba-V2)), distributional RL (IQN), and offline RL (CQL).

### Strengths
**S1. Simplicity of the method**
* The proposed method is conceptually simple and easy to implement; it only modifies the target calculation by taking the minimum between the target and online Q-values. 
* This adjustment can be integrated easily into existing RL frameworks, as demonstrated in Algorithms 1–4. 
Such simplicity makes the method practical and broadly applicable.

**S2. Substantial improvement across diverse RL domains**
* Despite its simplicity, the method demonstrates strong empirical performance. 
* It improves results across various RL domains --- online, offline, and distributional settings --- and across both continuous and discrete action spaces. 
* The reported gains over baseline methods in the DQN setting, suggest that the proposed approach effectively mitigates overestimation bias than other methods.

**S3. Clear organization and presentation**
* The paper is well structured and easy to read.
* Experimental results are clearly presented.

### Weaknesses
**W1. Comparison with other regularization methods limited in DQN setting**
* Currently, the comparison with other bias-reduction or regularization methods is limited to the DQN (discrete online RL) setting. 
This restricts the strength of the empirical claims, especially since the method is positioned as a general improvement applicable across multiple RL domains.
* Including comparisons with well-established bias-reduction methods in other settings (e.g., SAC, IQN, and CQL) will clarify whether the observed improvements generalize beyond DQN.

### Questions
**Q1. Missing comparison with Clipped Q-learning and related baselines**
* To fully support the claim, the proposed method should be experimentally compared with other bias reduction methods (e.g., methods introduced in DQN experiment). 
* Among those, I think at least **comparison with Clipped Q-learning [1] is necessary**, which is equivalent to MaxMin DQN with N=2 --- the strongest baseline reported for DQN in the paper, and also widely used.

I would like to emphasize that I believe this paper is strong and has substantial potential. However, this point is a critical concern for me. I currently lean toward a weak reject, but if the authors can provide the requested experimental comparisons (e.g., with Clipped Q-learning), I would be inclined to raise my recommendation, even significantly if the results meet the expectations.

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a technique (MINTO) to make better use of the online network when also using a target network for learning the value function. The technique simply changes the update target value to be the minimum between the target network and online network's value estimates.
Experiments on a variety of benchmarks show that this technique is beneficial and can outperform common alternatives such as double Q-learning or using the minimum betweeen two target networks.

### Strengths
The paper was easy to follow and the proposed technique is very simple to understand and implement.
The experiments are comprehensive and show that method works in a variety of settings (Q-learning algorithms, actor-critic, distributional RL, continuous-control, atari).
I appreciated testing the variants that would be most relevant such as mean or max of the online and target network estimates. 

In summary, I think the proposed technique would make a nice addition to the deep RL toolbox.

### Weaknesses
I did not identify any major weaknesses.

It could have been nice to include a more detailed analysis of why MINTO may be effective. In particular, thinking through the MINTO updates and comparing to using the target net as usual, the online network value replaces the target net value when the online network's value is lower. This means we can only use the most recent information (online network) if it is lowering the bootstrap target. 
I wonder if you could study the accuracy of the value estimates in a policy evaluation setting where we could accurately estimate true value functions. Perhaps MINTO produces better estimates overall.

### Questions
Suggestions and comments(not directly impacting the score):
- In the ablations, it is reported that taking the mean between the online and target network estimates was not helpful. Do you have any hypotheses why this woudl be the case? Using the mean between two target critics has been found to sometimes an effective alternative to the min between two critics [1] and sometimes underestimation bias has been found to be harmful.


- The proposed method reminds me of the online network trust region method proposed in [2] since they both attempt to make use of the online network safely. It could be beneficial to include it as a baseline or provide some comparisons.

- The algorithm labelled  "Maxmin DQN" in Fig.3. seems to be very similar to the clipped Double-Q learning update proposed in the TD3 paper [3] and I had not seen any mention of this.  


[1] "Scaling for compute and sample-efficient continous control" Nauman et al.

[2] "Human-level Atari 200x faster" Kapturowski et al.

[3] "Addressing function approximation error in actor-critic methods" Fujimoto et al.

### Soundness
4

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
for the bootstrapping target in deep Q-learning, uses the minimum of the online net and target net value estimates, instead of using just the target net value estimate

they call their algorithm "MINTO"

### Strengths
applies MINTO on many base algorithms (DQN, IQN, CQL, SAC) and architectures (CNN, IMPALA, SimbaV1, SimbaV2), and fairly consistently shows some improvements in returns

no additional hyperparameters

simple

a good number of fair baseline algorithms -- for ex, includes mean, max, random, and min of {online, target} nets in Fig 1, and later also includes Double DQN, FR-DQN, ScDQN, and Maxmin DQN

novel as far as I am aware

written clearly

### Weaknesses
I think it would be great if the paper were even more clear about how the hyperparameters were set. while Appendix C.3 says "All methods are run with their default hyperparameters", it's not immediately clear to me how the values in Table 1 were set. are they the default hyperparameters from any existing codebase? if not, why not, and how were they set?

5 seeds is not many seeds. (but the experiments cover so many base algorithms, architectures, codebases, and environment suites that I suspect this is not a big issue)

no tests on toy tasks, along the lines of Baird's counterexample. I think there is a non-negligible chance that MINTO will get "stuck" on some such problems, where both TD and DQN sometimes empirically require their loss to increase before it will decrease. see for ex Fig 1e here: https://openreview.net/pdf?id=j3bKnEidtT. MINTO empirically appears to work on a wide variety of other benchmarks though, so this whole bullet might be irrelevant (even including the possible case that MINTO fails on some toy tasks).

a bit verbose, for ex the paper calls MINTO "simple yet effective" 4 times (once with a comma after simple)

### Questions
"$Q$-learning offers a recursive update to approximate the state-action value function $Q$"

Should that last $Q$ be $Q^*$ instead of $Q$?

&nbsp;

does the "preliminaries" section allow for stochastic rewards? if not, is that intentional?

&nbsp;

"The foundation of this success lies primarily in Deep RL, initiated by the introduction of the Deep Q-Network (DQN) (Mnih et al., 2013), which marked the first successful application of deep neural networks in RL."

Neural Fitted Q Iteration was also a success, even though DQN could maybe be considered an even bigger success

&nbsp;

"A problem that presents a challenge and obstacle"

why both?

&nbsp;

"In deep RL, the problem of moving targets is especially evident due to the use of neural networks and the resulting uncontrolled fluctuations in the values of unseen states."

what is this contrasting against? linear RL?

&nbsp;

"For example, Gallici et al. (2025) demonstrated that cleverly using parallel environments eliminates the need for a target network"

some people might consider "eliminates the need for a target network" as too strong of a claim

&nbsp;

"This makes MellowMax orthogonal to our approach."

you might say "somewhat orthogonal" to help avoid confusion

&nbsp;

"making this method orthogonal to our approach"

likewise

&nbsp;

"known as $Q$-function"

typo, should be "known as _a_ $Q$-function" (or "_the_ $Q$-function")

&nbsp;

"(Mnih et al., 2013) introduce a series of algorithmic components, more importantly, is the introduction of the target networks."

typos

&nbsp;

"Despite the success, this results in a slow learning of value function as well as the policy due to relying on out-dated estimates"

typo, should be "learning of _the_ value function"

&nbsp;

"bellman"

typo, should be "Bellman"

&nbsp;

"can we find a practical bellman update rule that results in a stable and fast learning?"

what does "practical" specify here?

&nbsp;

"When implemented in an efficient deep learning framework such as JAX (Bradbury et al., 2018), this overhead is negligible."

I don't understand. does JAX automatically parallelize the additional forward pass with the target network's forward pass?

&nbsp;

"addressing Q1.As"

typo

&nbsp;

why is Fig 1 with 10 Atari games, but Fig 2 with 15 games?

&nbsp;

Fig 2 shows CNN and IMPALA results on 100 vs 50 million frames. why not both 100 or both 50?

&nbsp;

"into both value-based and actor–critic methods"

this is maybe slightly confusing wording because actor-critic methods are partially value-based. maybe "both actor-critic and purely value-based methods" would be slightly clearer.

&nbsp;

"Offline RL aims to learn an optimal policy from a large and static dataset"

offline RL does not need a "large" dataset to be offline RL

&nbsp;

"A central challenge in this paradigm is the distributional shift problem: the learned policy may query the Q-function on state–action pairs absent from the dataset"

offline policy evaluation does not necessarily involve a learned policy. offline policy evaluation is also offline RL, and also may query the value function on state or state-action inputs not present in the dataset

### Soundness
3

### Presentation
3

### Contribution
3
