# ROTATE: Regret-driven Open-ended Training for Ad Hoc Teamwork

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
Learning to collaborate with previously unseen partners is a fundamental generalization challenge in multi-agent learning, known as Ad Hoc Teamwork (AHT). 
Existing AHT approaches often adopt a two-stage pipeline, where first, a fixed population of teammates is generated with the idea that they should be representative of the teammates that will be seen at deployment time, and second, an AHT agent is trained to collaborate well with agents in the population. 
To date, the research community has focused on designing separate algorithms for each stage. This separation has led to algorithms that generate teammates with limited coverage of possible behaviors, and that ignore whether the generated teammates are easy to learn from for the AHT agent. 
Furthermore, algorithms for training AHT agents typically treat the set of training teammates as static, thus attempting to generalize to previously unseen partner agents without assuming any control over the set of training teammates.
This paper presents a unified framework for AHT by reformulating the problem as an open-ended learning process between an AHT agent and an adversarial teammate generator. 
We introduce ROTATE, a regret-driven, open-ended training algorithm that alternates between improving the AHT agent and generating teammates that probe its deficiencies. 
Experiments across diverse two-player environments demonstrate that ROTATE significantly outperforms baselines at generalizing to an unseen set of evaluation teammates, thus establishing a new standard for robust and generalizable teamwork.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper reframes ad‑hoc teamwork as an open‑ended min–max process: a teammate generator maximizes the ego agent’s cooperative regret while the ego minimizes it, alternating over training. The key idea is state‑wise regret, optimized over SP/XP/SXP state distributions to discourage sabotaging teammates by construction. A population buffer stabilizes training under a non‑stationary teammate distribution. On six two‑agent tasks across LBF and Overcooked, the method outperforms strong baselines on 5/6 tasks, and a toy “destructive matrix game” illustrates that state‑wise regret strongly suppresses sabotage.

### Strengths
Clear problem reframing. Maximizing cross‑play with unknown partners is cast as minimizing cooperative regret, yielding a natural open‑ended training framework that aligns well with UED principles.

Objective‑level protection against sabotage. Jointly optimizing state‑wise regret on SP/XP/SXP exposes weaknesses while enforcing compatibility with a best‑response partner, which is more principled than adversarial diversity only from the initial state.

Reasonable empirical support. Diverse tasks, 9–13 “benign” evaluation partners, a normalized score (upper‑bounded by an estimated BR), and targeted ablations (trajectory vs. state‑wise regret; with/without population buffer).

### Weaknesses
1.Narrow experimental scope. Evidence is confined to two agents with full observability. The absence of results on larger multi‑agent and partially observable settings (e.g., SMAC, GRF)[1] weakens claims of scalability.

2.Lack of theory and stopping criteria. “Open‑endedness” lacks a formal definition and a practical stopping rule; there are no guarantees on coverage, convergence, or regret bounds, limiting deployability.

3.Comparative evaluation is incomplete. Empirical comparisons against LIPO[2], MACOP and rigorously budget‑matched versions of BRDiv/CoMeDi are missing; fairness requires equal interaction budgets/compute.

4.Positioning vs. prior work needs precision. The relationship to MACOP and related methods should be spelled out at the level of objective functions, “benign partner” constraints, stopping rules, and network design, to avoid conceptual conflation.

5.Evaluation and ablations need tightening (detail add‑ons).

The normalized metric depends on an estimated BR upper bound; report sensitivity to BR approximation error.

Provide weight sensitivity for SP/XP/SXP and justify the SXP term’s necessity.

Use more random seeds—open‑ended procedures can have high variance.

Clarify environmental assumptions (mid‑episode policy switching/state resets); give an approximation strategy when resets are unavailable.

Discuss computational complexity as the partner space expands, and what approximations to BR/regret are admissible with guarantees.

Ref:

[1]A survey of progress on cooperative multi-agent reinforcement learning in open environment

[2] Generating Diverse Cooperative Agents by Learning Incompatible Policies

### Questions
1.Scaling to many agents. How does the method avoid combinatorial blow‑up for 5–10 agents or more? Would centralized training with decentralized execution, hierarchical BR, population BR, or fictitious‑play‑style approximations be viable, and at what cost?

2.Multi‑modal partner distributions. If the partner distribution is genuinely multi‑modal, is a single ego policy sufficient? Would mixture‑of‑experts, latent‑variable policies, or distributionally robust objectives (e.g., CVaR) be required to capture distinct partner modes?

3.Formalizing open‑endedness. What is the precise criterion—coverage growth, novelty accumulation, or monotone regret reduction? Please provide operational metrics (coverage/novelty/regret) and a stopping rule, accompanied by evidence.

4.Human–AI collaboration. Can the method transfer to real human partners (e.g., Overcooked‑human)? Would demonstrations, preference modeling, or safety constraints be needed to bound the teammate generator’s search space?

5.Visualization and interpretability. Please include state‑level sabotage heatmaps, SXP vs. XP occupancy differences, teammate embedding visualizations, and term‑wise causal ablations to show where and why each component works.

6.Embodied multi‑agent and LLM integration[1]. Can the framework extend to embodied, partially observable, continuous‑control domains? Could LLMs serve as a teammate generator or a language‑mediated coordination channel for richer partner diversity and policy decomposition?

Ref:

[1] 	
Multi-agent embodied ai: Advances and future directions

### Soundness
3

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
The paper discuss ad hoc teamwork as an open-ended partner co-learning problem and introduces ROTATE, which optimizes a per-state cooperative regret objective while encouraging competent, non-adversarial teammates. The approach alternates between training the ego policy and generating partner policies using state distributions from self-play, cross-play, and switched-play interactions, aided by a population buffer. Experiments across cooperative benchmarks report improved generalization to unseen partners.

### Strengths
The paper foregrounds the self-sabotage failure mode in open-ended partner generation, clearly articulating why partners that deliberately depress cross-play (XP) can inflate training signals yet harm zero-shot coordination; this diagnosis sharpens evaluation design (e.g., beyond average XP) and motivates principled mitigation objectives.

### Weaknesses
- Eq. 10 employs a fixed 0.5/0.5 weighting with no analytical justification, and the experiments do not analyze this hyperparameter.

- The method section has poor readability, with unclear logic and difficult-to-follow exposition.

- Missing ZSC-side baselines, especially some open-ended methods like COLE [1] and E3T [2].

### Questions
- Relationship to [3] and [4]: The method currently appears very similar to these two paper—could the authors clarify whether, under certain conditions, it degenerates to XP-min? How do the weights in Eq. 10 relate to XP-min’s hyperparameter ?

- Role of regret: The paper lacks analysis of regret’s actual effect. Do the generated partners indeed exhibit the property of “coordinating with a BR while exposing the ego’s weaknesses without engaging in self-sabotage”? Does Eq. 10 admit any theoretically provable guarantee that suppresses self-sabotage?

Reference 

[1] Li, Yang, et al. "Cooperative open-ended learning framework for zero-shot coordination." International Conference on Machine Learning. PMLR, 2023.

[2] Yan, Xue, et al. "An efficient end-to-end training approach for zero-shot human-AI coordination." Advances in neural information processing systems 36 (2023): 2636-2658.

[3] Charakorn, Rujikorn, Poramate Manoonpong, and Nat Dilokthanakul. "Diversity is not all you need: Training a robust cooperative agent needs specialist partners." Advances in Neural Information Processing Systems 37 (2024): 56401-56423.

[4]Sarkar, Bidipta, Andy Shih, and Dorsa Sadigh. "Diverse conventions for human-AI collaboration." Advances in neural information processing systems 36 (2023): 23115-23139.

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
4

### Summary
The paper proposes ROTATE, a regret-driven open-ended training framework for ad hoc teamwork that reframes zero-shot coordination as minimizing worst-case cooperative regret, i.e., $\min_{\pi^{ego}}\max_{\pi^{-i}}\mathbb{E}[\mathrm{CR}(\pi^{ego},\pi^{-i})]$. It introduces a per-state regret objective coupled with an SXP term that maximizes payoff with a best-response partner to discourage sabotage, and alternates teammate generation with ego learning using a population buffer. On Overcooked and Level-Based Foraging, ROTATE outperforms UED and teammate-diversification baselines with unseen partners, and ablations attribute gains to the per-state objective and the buffer.

### Strengths
- Comprehensive treatment of ZSC/ad hoc teamwork: the paper unifies teammate generation and ego learning via a cooperative-regret min–max objective $ \min_{\pi^{ego}}\max_{\pi^{-i}}\mathbb{E}[\mathrm{CR}]$, making assumptions and evaluation protocol explicit.

### Weaknesses
- Clarity and exposition: the paper is difficult to follow; the core algorithmic loop (who updates when, how SP/XP/SXP are sampled/weighted, and how the BR is trained/used) is buried under notation, so the end-to-end procedure remains unclear even after multiple readings.
- Mischaracterization of the gap: (a) the claim that most ZSC/AHT methods are two-stage is outdated—recent open-ended or end-to-end approaches already move beyond fixed teammate sets (e.g., COLE [1], E3T [2], TrajeDi [3]); (b) the comparison to current work is incomplete, especially where prior methods already leverage SP/XP and mixed-play/SXP-style rollouts (e.g., CoMeDi [4]), making the incremental novelty of the proposed per-state regret $J_{\text{state}}$ hard to isolate.
- Anti-sabotage rationale under-specified: the paper asserts that coupling per-state regret with an SXP best-response term mitigates sabotage, but offers little intuitive or theoretical support (e.g., no conditions under which maximizing SXP payoff with BR implies low sabotage against arbitrary partners, no analysis of bias induced by approximate BR or sampling); more formal justification or counterexample analysis is needed.

References

[1] Li, Y., Zhang, S., Sun, J., Du, Y., Wen, Y., Wang, X., and Pan, W. 2023. Cooperative Open-ended Learning Framework for Zero-Shot Coordination. In Proceedings of the 40th International Conference on Machine Learning (ICML 2023). Proceedings of Machine Learning Research, 202:20470–20484.

[2] Yan, X., Guo, J., Lou, X., Wang, J., Zhang, H., and Du, Y. 2023. An Efficient End-to-End Training Approach for Zero-Shot Human-AI Coordination. In Proceedings of the Thirty-Seventh Conference on Neural Information Processing Systems (NeurIPS 2023).

[3] Lupu, A., Cui, B., Hu, H., and Foerster, J. 2021. Trajectory Diversity for Zero-Shot Coordination. In Proceedings of the 38th International Conference on Machine Learning (ICML 2021). Proceedings of Machine Learning Research, 139:7204–7213.

[4] Sarkar, B., Shih, A., and Sadigh, D. 2023. Diverse Conventions for Human-AI Collaboration. In Proceedings of the Thirty-Seventh Conference on Neural Information Processing Systems (NeurIPS 2023).

### Questions
- Does the combination of per-state regret on SP/XP and the SXP best-response payoff formally or intuitively guarantee reduced sabotage (i.e., lower probability of destructive actions), and under what assumptions on BR optimality and sampling?
- Can an agent maximize $J_{\text{state}}$ on SP/XP while keeping high SXP payoff yet still sabotage arbitrary non-BR partners (e.g., collusion with BR)? Is there any bound linking SXP payoff to sabotage rate against unseen partners?
- How sensitive is the anti-sabotage effect to the weighting between SP/XP and SXP and to environments without reliable state resets/cut-ins? Please provide analysis or ablations.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This tackles the problem of Ad-Hoc Teamwork through iterative training diverse teams.

The diffuclty in AHT is that your AHT agent might be robust with respect to some, apparently diverse population of teammates, but not others. The paper claims that this is due to a small traning set. Their idea is to use iterative training in order to minimise co-operative regret.

The iterative idea is obviously not new and the related work contains some algorithms which maintain a diverse set of opponents. ROTATE uses a minimax approach, similarly to e.g. Villein et al, but finds a worst-case policy,rather than a distribution, at each step. This makes me think that the authors have not looked at the related work in sufficient detail.

I would have liked the algorithm to be more precisely defined in the main paper: Eq. 7 says that they use a minimax approach, but $\Pi^{-i}$ is not defined until Section 6, and only really discussed in detail in the appendix. Since many other works use iterative training, I suppose that the real open-endedness is the generation of new partners, rather than the iterative nature of the training.

### Strengths
+ Interesting notion of regret
+ Good comparison with related work.

### Weaknesses
- The authors could have done a better job of identifying which component is more important: the notion of regret, the way the teammates are generated, etc.
- Unclear novelty.
- Lack of clarity and theoretical discussion.

### Questions
Can you explain exactly how you used the baselines? From my reading of the appendix, it seems that you only took some aspect of these approaches, and adapted them to your framework, rather than have done a direct comparison.

### Soundness
3

### Presentation
2

### Contribution
2
