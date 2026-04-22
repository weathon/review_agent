# The Cost of Knowing: Hallucination Quest Game in Resource-Constrained Multi-Agent Systems

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 6, 2, 4

## Abstract
Current LLM hallucination benchmarks are predominantly static, focusing on
factuality while ignoring the computational resources consumed. This creates
a distorted view of performance, as costly mitigation strategies can obscure the
inherent capabilities of more efficient models. This limitation is especially critical
in multi-agent systems (MAS), where resource efficiency and strategic interaction
are paramount. To address this gap, we introduce MAS-HQ (Multi-Agent System
Hallucination Quest Game), a dynamic, game-theoretic framework that evaluates
MAS hallucination under strict resource constraints and direct adversarial competi-
tion. Within MAS-HQ, agents compete to produce low-hallucination summaries
while minimizing resource use. Success is measured by a multi-dimensional metric
that explicitly balances factual accuracy against resource penalties, forcing a trade-
off between quality and efficiency. We instantiate this competition with Q-Agent, a
modular agent architecture designed for strategic play, within a setting that features
partial observability to drive tactical decision-making. Our experiments reveal
the emergence of diverse winning strategies—some prioritizing high factuality,
others superior resource efficiency—and demonstrate adaptive agent behaviors
driven by the competitive dynamics. MAS-HQ establishes a principled paradigm
for benchmarking hallucination in MAS and provides crucial insights into agent
strategies under adversarial, resource-constrained conditions.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces MAS-HQ, a game-theoretic benchmark for evaluating hallucination in multi-agent LLM systems under explicit resource constraints. Two competing “Q-Agents” summarize passages drawn from a news dataset; each agent chooses to continue, review, or end, and a “vision mechanism” leaks limited state information to the opponent when costly review actions are taken. The proposed metric, Q-Score, combines a hallucination score (H-Score) with a resource-usage penalty weighted by coefficients (α,β).

### Strengths
1. The paper studies an important question: hallucination leaderboards rarely price in resource costs, which matters operationally in MAS settings.

2. The Q-Score objective and the dual-objective gameplay are easy to follow and properly reflects the central motivation of the paper.

3. The experiments feature breath and ablation that justifies the design choices.

### Weaknesses
1. The paper is not well-written. Many notations are not introduced when they are used for the first time. For example, T_{B, I}, V_{B, i}^{self}. 
2. The benefits of gamification of the inherently single-agent optimization problem needs to be further justified.
    (a). I expect there still should be benefits even without symmetry breaking. One can just run multiple agents independently and chooses the best one, i.e., Best-of-N, a strong and effective baseline. Such a baseline is arguably single-agent. Meanwhile, the symmetric breaking choices seem rather heuristic.
    (b). It is definitely beneficial to ground the empirical designs with theoretical formalism. However, the introduction of POSG merely reveals that there will be some strategic interaction among agents, i.e., when making decisions, one needs to account for others. It does not justify why such strategic interaction will boost LLM's ability compared with the case without strategic interaction. 
    (c). The formalism is not rigorous either. For example, in the main text, the belief is w.r.t other agents history, while in the appendix, it is w.r.t. the hidden state. 
3. Many SOTA, closed-sourced LLMs are not included, e.g., Gemini-2.5-Pro, GPT-5, etc. It is important to see that the methods are still useful when the base models very strong.

### Questions
See above

### Soundness
2

### Presentation
2

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
This paper introduces MAS-HQ, a novel game-theoretic framework for evaluating hallucination in multi-agent systems (MAS). The authors argue that existing benchmarks are static, focusing on factuality while ignoring the computational costs of hallucination mitigation. To address this, MAS-HQ places agents in a competitive game where they must produce low-hallucination summaries while minimizing resource consumption. Success is measured by a Q-Score that explicitly balances factual accuracy against resource penalties. The framework includes a modular Q-Agent architecture and a "vision mechanism" that creates partial observability, forcing agents to strategically weigh the benefits of actions like self-review against the risk of revealing information to an opponent.

### Strengths
The problem seems reasonably novel. The critique of static hallucination benchmarks is sharp and MAS-HQ has innovtive design owing to the game-theoretic elements. The empirical validation is reasonable as well. But I think it could have been stronger given the rigors of the conference.

### Weaknesses
The 'Q-Agent' architecture itself appears to be a relatively standard modular design, and the novelty lies more in the game it is subjected to rather than the agent design. The evaluation hinges critically on an external, pre-trained model for the hallucination H-Score, yet the potential biases or limitations of this judge are not discussed.

### Questions
Could you comment on the extent to which your strategies are emergent properties of complex reasoning, versus being more direct, programmatic optimizations of the Q-Score components, especially given the use of a hard threshold for triggering reviews?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes MAS‑HQ, a dynamic, game‑theoretic benchmark for evaluating hallucination in multi‑agent systems under explicit resource constraints. Two modular Q‑Agents (Policy, Summary, Review, Evaluation modules) compete to summarize passages; taking a costly Review action improves a passage but reveals partial state to the opponent via a vision mechanism, creating imperfect information and strategic trade‑offs. Performance is the Q‑Score: an average over passages balancing factuality against API calls, tokens, time, and review counts. Experiments show that winners emerge either by slightly higher factuality or by better resource efficiency; ablations show reviews lift H‑Scores, smaller β increases Q‑Score gains, the vision/passage‑order asymmetry is necessary, and the framework extends to three agents and to a SimpleQA setting.
It seems better to me that the paper is submitted to some benchmark tracks.

### Strengths
1. The Q‑Score explicitly prices resources, fixing a common flaw of static hallucination benchmarks that ignore cost
1. The vision mechanism ties improvement steps to information leakage, producing strategic behavior that static leaderboards miss
1. Ablation study effectively shows that removing reverse order or vision collapses the game to identical outcomes, and reviews measurably raise H‑Score across models.

### Weaknesses
1. The current Q‑Score rewards factual consistency (via H‑Score) and resource frugality; it does not directly reward coverage/completeness of the information that is in the passage. That leaves room for the degenerate strategy “say less to avoid being wrong.” 
1. H‑Score relies on a single pre‑trained discriminator from the Hallucination Leaderboard; any bias or miscalibration there propagates to the Q‑Score. Human verification or multiple evaluators would strengthen conclusions.
1. In MAS‑HQ, α and β are fixed hyperparameters chosen by the authors to weight factuality vs. resource cost in the Q‑Score. Fixed weights (α=1, β=0.01) strongly shape winners; authors themselves show that smaller β favors review‑heavy strategies (Figure 3‑right). A principled way to set their values is missing. 
1. Because some LLMs refuse certain passages, each model is evaluated on a different subset of the corpus, weakening cross‑model fairness and comparability.
1. The resource penalty is normalized against the two agents in the same match, which can make Q‑Scores non‑comparable across matches and may amplify design‑choice effects

### Questions
1. What's the rationale for selecting the fixed α and β values

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Traditional hallucination evaluations are static, focusing solely on factual correctness while ignoring the computational resources required to achieve it. In multi-agent systems, hallucination can be further amplified through interaction, making this limitation more severe. To address this, the authors propose MAS-HQ, a multi-agent, game-theoretic benchmark that studies the trade-off between hallucination and efficiency. Using text summarization as a toy competitive task, MAS-HQ places large language model (LLM) agents in a resource-constrained and adversarial environment, where they must balance factual accuracy with limited computation. This setting reflects realistic multi-agent scenarios that demand both strategic reasoning and efficiency under constraints.

### Strengths
- Recasts hallucination evaluation as a dynamic competition with explicit compute budgets. This is closer to operational constraints than static leaderboards.
- The Q-Score makes the tradeoff between factuality and resource use explicit. This encourages Pareto-aware evaluation rather than single-axis optimization.
- Very clean writing style and clear visualization.

### Weaknesses
- My main issue with the paper lies in its framing of text summarization (and SimpleQA) as a competitive game for evaluating MAS.
1. If the purpose is to evaluate hallucination in fact-based tasks such as text summarization or question answering, framing these as multi-agent competitions seems far-fetched. These tasks typically do not require multi-agent competition with multi-objective constraints, and introducing such a setup makes the tasks appear artificially or unnecessarily challenging.
2. If the purpose is instead to evaluate hallucination within multi-agent systems themselves, using text summarization merely as a toy task under competitive and resource-constrained conditions, then the framing makes more sense—since collaboration and competition with multi-objective trade-offs are common in MAS. However, in that case, the paper should have chosen tasks that are naturally multi-agent competitive, such as multi-party negotiation or bidding tasks.

In either case, the problem formulation and experimental setup feel disconnected from real-world scenarios and data distributions, potentially limiting the significance and applicability of the paper’s insights. An unrealistic setup like “competitive text summarization” may undermine the very goal of hallucination evaluation, as it becomes difficult to discern whether the hallucinations arise from the model’s inherent limitations or from artifacts introduced by the Q-Agent architecture and the competition design, which could induce information loss during the process.

- All evaluations are comparisons between Q-Agents’ internal configurations. The experiments lack comparisons with external baselines or alternative methods. For instance, how would a single-agent setup perform on the same tasks without the multi-agent competition?

-  The paper does not seem to provide new insights beyond showing that certain models perform better than others on this specific artificial task on the specific agentic architecture (Q-agent) as designed by the author.

### Questions
- As stated in Weakness #1, could the authors clarify the primary research motivation? Is the goal to understand hallucination in generation tasks (e.g., summarization and QA), or to study hallucination in multi-agent reasoning dynamics under competition and resource constraints? Additionally, why were text summarization and question answering chosen as the main evaluation tasks, given that these are not inherently competitive in nature?
- Can the author provide more baselines external to Q-Agent setup, such as how a single agent would perform?
- How is the alpha and beta weight chosen in the Q-Score equation during actual implementation. Are they important parameters that could be potentially sensitive and drastically shift agentic behavior and results? If so, can the author provide ablation on the two hyperparameters?
- One may argue that the hallucination could arise from the design of the Q-Agent architecture itself. i.e the same type of hallucination may happen in Q-Agent but not on a different agent architecture. How can the author counter this argument? Has the author tried on a different/simpler agent architecture.
- Are there additional insights/findings from the experiments beyond the emergence of diverse winning strategies and adaptive agent behaviors? As one may consider such phenomenon are expected in multi-objective and long-horizon interactive agent setup.

### Soundness
2

### Presentation
4

### Contribution
3
