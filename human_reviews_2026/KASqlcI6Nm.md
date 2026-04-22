# EUBRL: Epistemic Uncertainty Directed Bayesian Reinforcement Learning

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 8, 4

## Abstract
At the boundary between the known and the unknown, an agent inevitably confronts the dilemma of whether to explore or to exploit. Epistemic uncertainty reflects such boundaries, representing systematic uncertainty due to limited knowledge. In this paper, we propose a Bayesian reinforcement learning (RL) algorithm, $\texttt{EUBRL}$, which leverages epistemic guidance to achieve principled exploration. This guidance adaptively reduces per-step regret arising from estimation errors. We establish nearly minimax-optimal regret and sample complexity guarantees for a class of sufficiently expressive priors in infinite-horizon discounted MDPs. Empirically, we evaluate $\texttt{EUBRL}$ on tasks characterized by sparse rewards, long horizons, and stochasticity. Results demonstrate that $\texttt{EUBRL}$ achieves superior sample efficiency, scalability, and consistency.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a method for incorporating epistemic uncertainty into a standard Bayes-Adaptive-MDP-solving framework (interact, update belief-posterior, find the optimal policy with respect to the posterior, repeat). The paper includes a comprehensive theoretical analysis of the new method from the perspective of optimality.

### Strengths
Strong theoretical foundation.

Strong theoretical results.

Good sanity-check experiments.

Clear description of theoretical limitations.

### Weaknesses
Missing theoretical motivation:
  1. Why is CAI chosen to formulate the epistemic uncertainty?
  2. Why is epistemic uncertainty chosen to be formulated in CAI in this way?

Missing additional limitations - the method is most likely limited to tabular settings. Is that correct?

Presentation: multiple comments, described in more detail below.

### Questions
**Presentation:**
1. Algorithm 1 is very simple and very compact, which is excellent. However:
    1. I think it is important it is at least described in the main paper.
    2. ValueIteration(b) is never defined, described, or cited, I believe.
    3. BeliefUpdate(s,r) should also be described in more detail (at least, how the update is done from a practical perspective).
    4. It seems to me that algorithm 1 is not one of the contributions of the paper, but rather a rather-standard approach for posterior-belief/over-models based methods. Is that correct? If yes, I would rephrase the contributions. Rather than introducing a new algorith, the authors introduce a new method to incorporate uncertainty into the belief / posterior. I do not view this as weakening the contribution of the paper - merely making it easier for the reader to understand the novelty of the paper.
2. I don't think $\Epsilon_b$ is ever defined?
2. The "related work section" would work much better after preliminaries (Section 2) than where it currently is, in the end of the paper. It will also make introducing the baselines and understanding the approach much easier in my opinion.
3. I would cite at least one uncertainty survey for the Uncertainty Quantification paragraph, such as [1].
4. Section 3.1 should build on citations from the uncertaity quantification paragraphs in the related work (/preliminaries). Unless 3.1 is entirely novel (in which case - this should be emphasized stronger), the prior work it builds on should be cited.
5. I think lazy chain is incompletely explained. What is the complete action space? Are transitions det. or stoch.? Is there a difference between the left end and the right end? What does solving the problem constitute (ie what is the optimal policy), if the comulative return is zero?
6. I would also suggest adding more detailed descriptions of chain and deep sea.
7. Some \citep should be \cite (line 464).

**Experiments and baselines:**
1. I believe PSRL can be tuned to work much better if the "right" choice of prior over transition models (/rewards) is made (ie - transitions from any states are to at most two different states, rewards are det / from a normal). Since experiments are already run with EUBRL+ and there's a prior selection discussion, I think these results belong there. What do the authors think?

I'm open to increasing the score, especially since most changes I would like to see are textual. The most major comments from my perspective are the clarity that relates to Algorithm 1 and the missing theoretical motivation.

[1] Hüllermeier, Eyke, and Willem Waegeman. "Aleatoric and epistemic uncertainty in machine learning: An introduction to concepts and methods." Machine learning 110.3 (2021): 457-506.

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes EUBRL, a Bayesian RL algorithm that steers exploration using epistemic uncertainty. At a high level, the per-step reward is interpolated using a learned "probability of uncertainty," $P(U=1 \mid s, a)$, which balances exploitation (the current mean reward estimate) against exploration (an intrinsic term derived from epistemic uncertainty). The authors prove nearly minimax-optimal regret and sample-complexity bounds for infinite-horizon discounted MDPs under a class of priors. Experiments on Chain, Loop ,DeepSea, and a new LazyChain benchmark compare EUBRL to several baselines and suggest strong exploratory behavior.

### Strengths
1.To the best of my knowledge, this is the first work to convert epistemic uncertainty into an explicit guidance weight $P(U=1 \mid s, a)$. This idea is novel, naturally decouples exploration from uncertain reward estimates, and provides an adaptive interpolation weight of the reward signal.

2.Theoretical guarantees are strong.  The paper gives (i) a regret bound $\tilde{O}(\sqrt{S A T} /(1- \gamma)^{1.5}+S^2 A /(1-\gamma)^2$ ) that matches known lower bounds when $T$ is large enough, and (ii) a sample-complexity bound that matches lower bounds for small $\varepsilon$.

3.The paper formalizes a class 
$\mathcal{C}$ of decomposable/weakly-informative priors and shows nearly-minimax bounds for uniform bounded priors.

4.The experimental section covers both deterministic and stochastic settings with many random seeds, which helps demonstrate robustness.

### Weaknesses
1.The algorithmic description is quite high-level. More concrete details and examples would help reproducibility-for instance: how $\mathcal{E}(s, a)$ is computed in practice; how the on-policy estimate $P(U=1| s, a)= \mathcal{E}\_{b}/ \mathcal{E}\_{\max}$ is formed; and how $\mathcal{E}_{\max }$ is chosen.

2.Some notation and concepts in the main text need clearer definitions. For example: what is the role of $w$ in Section 3.1? How is a "prior" specified precisely (i.e., prior over what objects/parameters)? It would also help to include concrete examples illustrating Definition 1.

### Questions
1.Details in Algorithm 1: How do you compute $\mathcal{E}(s, a)$ exactly in the tabular/prior choices you study? How is $P(U=1| s, a)=\mathcal{E}\_{b}/ \mathcal{E}\_{\max}$ estimated on-policy, and how is $\mathcal{E}_{\text {max }}$ selected (global constant, rolling maximum, or theoretical bound)? Do you require assumptions on $f$ and $g$ ? How is Valuelteration(b) implemented under the updated belief?

2.Theorem 3 suggests $\eta$ is important. How do you choose $\eta$ across tasks, and how sensitive are results to it? 

3.$\widetilde{V}^t(s)$ is introduced to ensure quasi-optimism. Could you give its precise definition and a short intuition for how it drives the regret decomposition?

Suggestions:
1. If some definitions or derivations are omitted due to space, please add precise references to the appendix, so readers can quickly locate the formal statements and implementation details.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors propose the Epistemic Uncertainty directed Bayesian Reinforcement Learning (EUBRL) algorithm which achieves nearly minimax-optimal regret and sample complexity in infinite-horizon Markov Decision Processes (MDPs). EUBRL directs RL exploration by utilising probabilistic inference to model epistemic uncertainty as part of the agent’s objective (i.e., adaptively weighting the mean reward and uncertainty term based on the probability of uncertainty). Nearly minimax-optimal regret bounds and sample-complexity guarantees are established for the class of priors that are decomposable, or weakly-informative, and whose rate of epistemic uncertainty is $\mathcal{O}(1\sqrt{n})$. Empirically, EUBRL is shown to outperform frequentist (RMAX, MBIE-EB), sampling-based (PSRL, BOSS), optimism-based Bayesian (BEB, VBRB), and classical Bayesian (BEETLE, Mean-MDP) baselines across tasks with sparse rewards, long horizons and stochasticity (Chain, Loop, DeepSea, and the newly introduced LazyChain).

### Strengths
The paper is eloquently written and introduces a novel theoretical proof which, for the first time (to the best of my knowledge also), achieves nearly minimax-optimal sample complexity in infinite-horizon discounted MDPs without assuming a generative model. This result improves on He at al. 2021 which shows nearly minimax-optimal regret but doesn’t extend to sample complexity. The theoretical results are backed up by convincing empirical results (using multiple seeds, reporting standard errors etc) covering a wide range of appropriate baselines (including frequentist and Bayesian), three environments from the literature, and newly introduced fourth task which targets algorithm “myopia”. In all tasks EUBRL is shown to improve upon prior works.

Overall I think the claims in the paper are both significant and well-supported both theoretically and empirically.

### Weaknesses
Towards the goal of disentangling exploration and exploitation, the evidence could be strengthened by e,g., considering an ablation (see questions).

The accessibility of the paper to a wider audience could also benefit from adding short intuitive summaries after key lemmas in the appendices.

### Questions
Would it be possible to compare EUBRL to a variant that uses the exact same uncertainty estimates but applied as an additive bonus? I think such a result would help further empirically support the benefits of disentangling exploration and exploitation versus improved UQ.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a novel Bayesian reinforcement learning algorithm, EUBRL, to solve the explore-exploit dilemma, especially in environments with sparse rewards. The paper argues that common "optimism-based" exploration (adding an uncertainty bonus) is flawed because it can magnify errors when the agent's reward estimates are unreliable. EUBRL uses an "epistemically guided reward." This is a weighted average that exploits (uses the estimated reward) when the agent is confident, but explores (uses uncertainty itself as an intrinsic reward) when the agent is uncertain. Theoretically, they show that EUBRL achieves nearly minimax-optimal regret and more importantly sample efficiency which hasn't been done in prior work. Empirically, they show that their method leads to imporved sampled efficiency on different tasks.

### Strengths
1. The paper tackles the fundamental exploration–exploitation dilemma in reinforcement learning by introducing a novel approach termed “epistemic guidance.” The method is well-motivated, and its effectiveness is supported through both rigorous theoretical analysis and comprehensive empirical evaluation.

2. The paper argues that adding an exploration bonus to the reward estimate is a flawed way to do exploration in BAMDPs because the reward estimate can be highly uncertain and can result in a poorly specified reward. Instead they propose to weigh the two rewards (reward estimate and the exploration bonus) by the uncertainty associated with the state-action pair. If the agent is uncertain it uses the exploration bonus if the agent is certain it uses the reward estimate. The paper also proves that this method achieves nearly minimax-optimal regret in both regret and sample complexity.

### Weaknesses
1. The paper's analysis is confined to discrete state-action spaces, and it does not address the challenges of integrating its method with deep function approximation. The reliance on maintaining an explicit Bayesian posterior is computationally intractable for the high-dimensional environments where deep RL is typically applied. Therefore, it is unclear how the 'epistemically guided reward' could be effectively approximated to improve sample efficiency in practical deep RL algorithms. 

2. The paper suffers from a separation of theory from implementation. While the main text provides a compelling theoretical justification for the "epistemically guided reward," it lacks a clear, procedural description of the full algorithmic loop. To understand precisely how the belief posterior is updated after each transition and how the new policy is extracted, the reader is required to hunt for the algorithm in the appendix.

3. The adaptive weighting scheme is particularly vulnerable to the degeneracy of the uncertainty estimates. If uncertainty degenerates to zero, the agent will become fully exploitative.

4. Similar to the last point, a mis-specified prior will also impact this algorithm a lot more than standard exploration bonus algorithms.

### Questions
My questions are based on the practicality of the algorithm for more realistic applications:

Exploration bonuses have been shown to work with Deep Reinforcement learning algorithms. Do the authors think that their method can be deployed with a neural-network based policy in a model-based setting. 

1. The paper claims "superior scalability" but only demonstrates it in tabular, low-state-space environments (DeepSea). Isn't this claim misleading, as the algorithm's tabular, model-based nature makes it computationally intractable for the large-scale, high-dimensional problems where scalability is actually needed?

3. Since maintaining a full Bayesian posterior is impossible, which practical proxy for uncertainty would work with this method? For example, would you use the variance from a deep ensemble or a novelty score from a pseudo-count method?

4. Could this "guided reward" concept be adapted to improve sample efficiency in a model-free deep RL algorithm (like DQN or SAC), or is it fundamentally tied to a model-based approach?

### Soundness
2

### Presentation
2

### Contribution
2
