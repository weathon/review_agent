# Bayesian Social Deduction with Graph-Informed Language Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4

## Abstract
Social reasoning -- inferring unobservable beliefs and intentions from partial observations of other agents -- remains a challenging task for large language models (LLMs). We evaluate the limits of current reasoning language models in the social deduction game Avalon and find that while the largest models demonstrate strong performance, they require extensive test-time inference and degrade sharply when distilled to smaller, real-time-capable variants. To address this, we introduce a hybrid reasoning framework that externalizes belief inference to a structured probabilistic model, while using an LLM for language understanding and interaction. Our approach achieves performance competitive with much larger models in agent-agent play and, notably, is the first language agent to defeat human players in a controlled study -- achieving a 67% win rate and receiving higher qualitative ratings than both reasoning baselines and human teammates. We release code, models, and a dataset to support future work on social reasoning in LLM agents.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a neuro-symbolic method for a social deduction game. The proposed GRAIL utilizes a factor graph for Bayesian inference, estimate priors with language models, and propagate belief with factor graph. It augment language model outputs with the computed results.

The proposed model, with separately trained factor function approximation, outperforms other zero-shot reasoning model baselines, and surpasses human players in both winning rate and qualitative ratings.

Further analysis shows that the method uses fewer tokens overall. The ablation study demonstrates that both the language model and the graph components are essential for its success, with the graph determining the lower bound of performance.

Combining language models with probabilistic reasoning is an interesting direction. However, the results are not entirely significant, as the model is trained on additional data, and the experiments focused only on one simplified game.

### Strengths
- The overall presentation of the paper is clear and well-structured.
- The proposed integration of language models with probabilistic inference is interesting.
- The inclusion of both agent-agent and human evaluations makes the work more convincing. The experiments conducted in the Avalon domain are thorough.

### Weaknesses
- The use of a trained neural network for factor function approximation reduces the significance of the reported performance gains and makes the comparison with other methods potentially unfair.
- The experimental scope is limited. It remains unclear whether this approach generalizes to other domains or games. The trained neural network which is a key component of the system, needs to be specifically tailored and trained for each game rather than generalizable.
- The simplified Avalon setting further limits the significance of the results. As the authors noted, the original version includes different player roles (like Merlin). The GRAIL is also only evaluated as the Good players. This simplification may bias the game toward probabilistic methods rather than language-based social deduction (as also suggested by the ablation results, where the graph component is substantially more important than the language model in GRAIL). Evaluating the method in a more realistic setting could make the experiments more meaningful and the results more significant.

### Questions
- Is there a specific reason why the experiments were not conducted in a more realistic setting, such as evaluating GRAIL from both sides, and with different player roles? Also, could you clarify why including Merlin would 'introduce deception', as mentioned in the footnote on page 3?

- My understanding is that modeling other players’ beliefs corresponds to first-order Theory of Mind, whereas modeling an agent’s own beliefs does not. Since GRAIL acts as a player with full access to its own information, is the model in this work actually modeling its own belief? Could you elaborate this?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents GRAIL, a hybrid framework for social deduction games such as Avalon. The proposed model separates linguistic understanding and probabilistic reasoning: an LLM is used to interpret player dialogues and produce coarse-grained “directional priors”, while a factor graph performs Bayesian inference via max-product belief propagation to estimate hidden player roles.
Based on the inferred beliefs, the agent makes decisions such as party proposals and votes through a simple heuristic policy. The authors evaluate GRAIL against several reasoning-based and non-reasoning LLM baselines, showing improved win rates and greater stability, especially with smaller models.

### Strengths
1. The paper is clearly structured and easy to follow, with a clean separation of sections for modeling, inference, and experiments. The figures and appendices help convey the pipeline intuitively.

2. While the individual components (factor graphs, LLM priors, heuristic decision rules) are standard, their combination into a unified social deduction agent is a novel design choice. The idea of using LLMs only for qualitative “belief direction” is interesting and differs from typical chain-of-thought reasoning.

3. The method is technically sound, and the probabilistic formulation (max-product inference and neural factor approximation) is consistent. The experimental setup on Avalon is well-defined, and the human-agent evaluation adds credibility.

### Weaknesses
1. While the hybrid structure is elegant, I am not convinced that delegating the entire reasoning process to an external Bayesian graph is the right direction for long-term progress in social reasoning. Many reasoning chains in social deduction are inherently complex and multi-step, and the need for long chain-of-thought reasoning is not a weakness but a feature of such problems. By removing these steps from the LLM and handling them purely through a pre-defined probabilistic structure, the framework may gain stability but loses the very capacity for nuanced, emergent reasoning that LLMs are increasingly capable of developing. This design feels more like an engineering shortcut than a scalable reasoning principle.

2. The “higher / lower / same” directionality of language priors collapses the rich structure of social interaction into a single scalar. However, dialogue in social deduction games often contains entangled, relational cues (e.g., when one player’s statement implicitly reveals alliances or dependencies). These correlations cannot be expressed by independent prior adjustments on each player. In other words, the linguistic and relational signals are not separable; enforcing such separation risks discarding precisely the information that makes social reasoning interesting.

3. The paper linearly increases β to strengthen priors over rounds, but this schedule appears hand-tuned. It is unclear whether the same parameterization works across different LLMs, languages, or domains. Since β directly controls the interaction strength between the language layer and the factor graph, its sensitivity and robustness deserve deeper analysis—perhaps through a controlled ablation or sensitivity plot.

4. The paper primarily compares against older baselines (ReAct, DeepSeek-R1, etc.). However, recent works have advanced structured reasoning and dynamic workflow modeling for agentic systems—see AFlow (ICLR 2025), DyFlow (NeurIPS 2025), and MaAS (ICML 2025). These models would offer more competitive and conceptually relevant baselines. Without such comparisons, it is difficult to judge whether GRAIL’s improvements arise from genuine reasoning efficiency or simply from task-specific heuristics.

5. The framework assumes that LLMs can meaningfully answer prompts such as “is player X more suspicious than before?”, yet this capability is never validated. It is plausible that these judgments are inconsistent or even random. A simple perturbation test—flipping or shuffling a portion of these qualitative outputs—would reveal whether they actually carry semantic signal. Likewise, evaluating the consistency of priors across models (e.g., GPT-4 vs Llama-70B) could strengthen the empirical grounding.

### Questions
Most of my questions are already reflected in the Weaknesses section above, and I would appreciate detailed clarifications or additional experiments addressing those points. 

Although I remain skeptical about several design choices, I am open to discussion and would be glad to reconsider my evaluation based on the authors’ responses and further evidence provided during the rebuttal phase.

### Soundness
2

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
3

### Summary
This paper uses the social deduction game Avalon to evaluate and improve LLMs' social reasoning abilities. The authors introduce a hybrid probabilisitic reasoning framework called GRAIL, which achieves competitive performance compared to strong reasoning models and can defeat human players in a controlled study. They also perform thorough analysis on the method with different model families and sizes, allowing the reader to more deeply understand the strengths and limitations of the method.

### Strengths
- Social reasoning is an important topic in AI and LLM research, which this work engages with.
- The proposed method is principled and performs well.
- This work conducts many kinds of analysis, including on resource usage and hallucination. It tests the method using different models (e.g., Llama and DeepSeek, varying sizes).
- This work conducts model vs human studies, supporting the effectiveness of the proposed method.

### Weaknesses
- This work only studies one game: Avalon. While it is not necessary here to extend GRAIL to other domains, I'd appreciate if the authors include more discussions on where they think GRAIL can also apply to (e.g., other games or social reasoning settings) and where it would face challenges.
- Why is ReCon an appropriate baseline and in fact the only non-reasoning-model baseline? The authors need to introduce ReCon more (given its current role) and argue why it makes sense here.

### Questions
How is this study different from the previous studies that evaluate Avalon gameplay with LLMs, and how is GRAIL different from previous work that applies probabilistic graphical models to social deduction games? The authors do cite relevant work, but have not explicitly articulated what they consider to be significant or novel compared to prior work. This is important for clear paper writing. I would raise my score if the authors address this point.

### Soundness
2

### Presentation
2

### Contribution
2
