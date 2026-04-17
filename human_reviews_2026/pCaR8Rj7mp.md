# Policy Improvement with Style-Specific Demonstrations

- Decision: Reject
- Scores: 4, 2, 6, 6

## Abstract
Proficient game agents with diverse play styles enrich the gaming experience and enhance the replay value of games. However, recent advancements in game AI based on reinforcement learning have predominantly focused on improving proficiency, whereas methods based on evolution algorithms generate agents with diverse play styles but exhibit subpar performance compared to RL methods. To address this gap, this paper proposes Mixed Proximal Policy Optimization (MPPO), a method designed to improve the proficiency of existing suboptimal agents while retaining their distinct styles. MPPO unifies loss objectives for both online and offline samples and introduces an implicit constraint to approximate demonstrator policies by adjusting the empirical distribution of samples. Empirical results across environments of varying scales demonstrate that MPPO achieves proficiency levels comparable to, or even superior to, pure online algorithms while preserving demonstrators' play styles. This work presents an effective approach for generating highly proficient and diverse game agents, ultimately contributing to more engaging gameplay experiences.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Mixed Proximal Policy Optimization (MPPO), an algorithm that aims to improve the proficiency of suboptimal game agents while preserving their original 'play styles'. The core idea is to combine online data gathered from self-play (similar to PPO) with offline demonstration data from existing, stylized agents. It uses a unified PPO-like clipped objective function to handle both data sources, weighting their influence with a hyperparameter $\beta$. The method also introduces a data storage technique where offline trajectories are saved as environment seeds and action sequences. These are then "replayed" by dedicated actors to regenerate full-episode information, including advantages. The authors provide theoretical claims for monotonic policy improvement and style preservation and demonstrate results in Blackjack, Maze Navigation, and Mahjong, arguing that MPPO achieves high proficiency while remaining closer in style to the demonstrators than pure PPO.

### Strengths
The paper tackles an important and practical problem: how to leverage existing, stylized, but suboptimal bots (e.g., from rule-based systems or QD methods) and improve their performance without them all converging to a single, style-less optimal policy.

The proposed data collection method (storing seeds and actions) is a clever approach to reducing the storage footprint of demonstration data.

The experimental results, particularly in the complex game of Mahjong, are noteworthy. The MPPO agent surpasses its suboptimal demonstrator and, in the case of Bot A, achieves a win rate comparable to the top-ranked bot on the platform, even outperforming the pure PPO baseline.

### Weaknesses
My main concerns with this paper are its theoretical assumptions, practical applicability, and novelty.

1.  **Unreasonable Theoretical Assumption:** The core of the method's theoretical justification is shaky. The paper uses the importance sampling ratio $r = \frac{\pi _{k}^{\prime}(a|s)}{\pi _{k}(a|s)}$ for the offline demonstration samples. The authors correctly state that the theoretically correct ratio should be $r=\frac{\pi _{k}^{\prime}(a|s)}{\pi _{T}(a|s)}$ (where $\pi _T$ is the teacher/demonstrator policy) but then "approximate it using the previous policy $\pi _{k}$". This assumption that $\pi _T \approx \pi _k$ is highly problematic. At the beginning of training, the student policy $\pi_k$ will be very different from the demonstrator $\pi _T$. The paper's claim of monotonic improvement, which leans on TRPO/BPPO theory, seems unsupported under this flawed approximation. The authors just hand-wave this by stating the "policy update is sufficiently small", which isn't a sufficient justification.

2.  **Impractical Replay Mechanism:** The method for replaying offline data, while storage-efficient, seems computationally expensive and impractical. It requires "LfD actors initialize the game environment using the recorded seed and feed the action sequences" to regenerate states and calculate advantages. This is not "using offline data"; it is *re-simulating* the demonstrations. This might be feasible for lightweight environments like Blackjack, but it would be a massive computational bottleneck for any large-scale modern game or complex simulation, completely negating the typical benefit of offline data (which is that it's "cheap" to learn from).

3.  **Limited Novelty:** The contribution feels insufficient. The MPPO loss function (Equation 4) is, at its core, a simple linear combination of an on-policy PPO loss and an off-policy PPO loss. This is a very straightforward combination of existing methods (PPO and principles from offline RL like BPPO). The novelty seems to rest on the (questionable) replay mechanism and the application to "style preservation," rather than a new algorithmic insight.

4.  **Missing Related Work:** The related work section is missing a critical and highly relevant body of literature: **online RL with offline datasets**. This field directly addresses the problem of combining a static offline dataset with new online interactions and RL. The paper must discuss and compare against foundational methods in this area, such as [1,2,3]. These methods have developed more principled ways to combine offline and online objectives without the strong $\pi _T \approx \pi _k$ assumption.

5.  **Questionable Metric:** The "total diversity assessment indicator" $D _{policy}$ is questionable. It's defined as the total variation distance over the *entire* state space. This is intractable. The paper mentions computing it from held-out trajectories, but this is just an empirical approximation whose quality is highly dependent on the state coverage of those trajectories. It's not clear that this metric, approximated in this way, is a "rational" or reliable measure of a high-level concept like "play style."

[1] Nair A, et al. AWAC: Accelerating online reinforcement learning with offline datasets, arXiv 2020.

[2] Ball P J, et al. Efficient Online Reinforcement Learning with Offline Data, ICML 2023.

[3] Wagenmaker A, et al. Leveraging Offline Data in Online Reinforcement Learning, ICML 2023.

### Questions
1.  Can you provide a stronger justification for the $\pi_T \approx \pi_k$ approximation? What is the effect on the monotonic improvement guarantee when this assumption is strongly violated, as it must be in the early phases of training? Is this weak approximation the reason the Mahjong agents *must* be initialized from behavior cloning checkpoints and have their policy networks frozen for 1000 iterations?

2.  Regarding the seed-and-action replay method: What is the computational overhead of this "re-simulation" step? How does it compare to the standard method of just loading $(s, a, r, s')$ tuples from a (larger) offline dataset? How can this method possibly scale to complex environments where a single environment step is computationally expensive?

3.  Why was the "online RL with offline data" line of work (e.g., [1,2,3], etc.) not discussed or compared against? These methods seem to be solving the exact same problem (mixing offline and online samples) and would be a much more relevant baseline than pure IL methods like GAIL.

4.  The $D_{policy}$ metric is an expectation over $s \in S$. How was this expectation approximated in the Blackjack and Maze experiments? How many states were used, how were they selected, and how sensitive are the $D_{policy}$ results to this sampling strategy?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work proposes MPPO, aiming to enhance the agent's proficiency while maintaining consistency with the demonstration style. Experiments conducted by the authors on Blackjack, Maze Navigation, and Mahjong aim to demonstrate that MPPO substantially improves the agent's gaming performance based on suboptimal demonstrations, while the improved agent retains its gaming style. Concurrently, the authors refined the offline data storage and replay mechanism.

### Strengths
1. Overall, this paper is well-structured and effectively conveys its central argument.
2. This approach combines environmental seeds with action sequences, effectively reducing storage space requirements.
3. In Blackjack, Maze Navigation, and Mahjong, this work conducted extensive experiments.

### Weaknesses
1. The author's investigation into strategy diversity is insufficient. Extensive work in multi-agent reinforcement learning has already focused on this aspect, such as various diversity-based PSRO algorithms, which the author fails to mention in the paper.
2. The author fails to correctly distinguish certain concepts. For instance, the author repeatedly mentions “self-play,” yet both the method and experiments are based on the single-agent setting. In fact, self-play is a multi-agent reinforcement learning algorithm designed to solve competitive games.
3. This paper contains several writing errors. For example, “Figure 2B” on page 6 should actually be “Table 2B.”

### Questions
1. Although the authors claim that the diversity measure designed in the paper overcomes the shortcomings of $W_2$, it does not appear to consider diversity from a trajectory perspective. In other words, would it be more reasonable to incorporate discounted unnormalized visitation frequencies?
2. The authors claim that the policy converges monotonically toward the demonstration policy. Does this contradict the claim that “the policy improves monotonically”?
3. Intuitively, demonstration data constitutes only 5% of the total data, making it difficult to achieve the effect of preserving the demonstration style as claimed by the authors. So why was this proportion set at 5%?
4. Based on the experimental data, the $D_{policy}$ between the MPPO and the demonstration agents remains quite large. Does this truly indicate that the algorithm has successfully preserved the demonstration style?

### Soundness
2

### Presentation
3

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
This paper proposes a mixed PPO algorithm that improves the proficiency of existing suboptimal agents while incorporating implicit constraints on the distance to an unknown teacher policy. It also delivers an interesting theorem stating that the proposed offline objective reduces the distance to the teacher policy.

### Strengths
Theorem 2 is practical and useful. The paper is easy to follow. Experiments are conducted across a range of environments and demonstrate strong performance. The proposed method is decent and useful.

### Weaknesses
Although a reference is provided for the play style distance, I do not consider it a perfect metric. It only measures the distance to an agent with a specific style, while the notion of style is likely more complex. A straightforward distance measure may not adequately capture what constitutes "style." That said, I recognize this may be an open question beyond the scope of this paper.

The approximation of the importance sampling ratio for off-policy demonstrations appears a little bit arbitrary, but it is somewhat acceptable given the good practical performance.

### Questions
1. How well does the total variational divergence in Equation (2) approximate the 2-Wasserstein distance? For instance, could the authors provide some empirical results, such as correlations or L1/L2 error, or any theoretical analysis?
2. How do the authors define the teacher policy $\pi_T$? It seems to refer to the underlying policy used to generate the demonstration samples.

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
4

### Summary
This paper aims to address the trade-off between diversity and performance. It proposes a new method called Mixed Proximal Policy Optimization (MPPO), designed to improve the proficiency of existing suboptimal agents while retaining their distinct styles. MPPO achieves this by unifying loss objectives for both online and offline samples and introducing an implicit constraint to approximate the demonstrator's policy. Empirical results across environments of varying scales, including Blackjack, Maze, and Mahjong, demonstrate that MPPO achieves proficiency levels comparable to or even superior to pure online algorithms while successfully preserving the demonstrators' play styles.

### Strengths
- Addresses the trade-off between "proficiency" and "style diversity" in game AI by proposing an effective solution (MPPO).
- MPPO uses an "implicit constraint" (filtering positive-return data) to preserve style, which ablation studies showed is superior to traditional explicit supervised loss methods (PPOfD).
- The method's robustness is demonstrated by its success in three environments with vastly different complexities (simple, medium-deterministic, and high-dimensional imperfect-information).

### Weaknesses
- The proposed MPPO method appears to be a direct, weighted combination of offline PPO (BPPO) and standard PPO. While effective, this approach might be somewhat limited in its novelty, though this is not a major concern.
- The paper mixes offline and online data for training simultaneously. It would be insightful to understand how this compares to a sequential approach (e.g., pre-training on offline data followed by fine-tuning with online interaction), which mirrors a common pre-train and fine-tune paradigm.
- The rigor of some notations and formulas in the paper could be improved. I suggest the authors check them carefully. 
    - In the preliminaries, the notations $Q$, $V$, and $A$ are policy-dependent (related to $\pi$) but are not always marked as such.
    - In Equation 4, the spacing is not well-aligned, and operators like 'min' and 'clip' should be in roman (non-italic) font.

- There are several works focusing on diversity and performance in game AI beyond QD methods. I suggest the authors cite and discuss [1-3].

[1] Learning Diverse Risk Preferences in Population-Based Self-Play. AAAI2024

[2] Discovering Diverse Multi-Agent Strategic Behavior via Reward Randomization. ICLR2021

[3] Effective diversity in population based reinforcement learning. NeurIPS 2021.

### Questions
See weakness.

### Soundness
3

### Presentation
2

### Contribution
2
