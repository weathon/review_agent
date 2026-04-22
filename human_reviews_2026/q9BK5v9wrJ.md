# Investigating Language Models for Supporting Complex Group Decisions

- Avg Score: 3.00
- Decision: Reject
- Scores: 6, 2, 2, 2

## Abstract
Reaching consensus is a central challenge in group decision making as agreement needs to be balanced with diversity of perspectives. Recent AI advances have opened new possibilities for synthesizing complex information and facilitating consensus. We study group decision processes by modeling the complexity of the decision surface, defined by a set of decision problems, each with multiple options. Each solution yields a gain for every participant, and the objective of deliberation is to ensure fairness by equalizing participants’ profits. We explore multiple settings: whether gains are private, arbitrary numbers, or ordered sequences; whether the exact gain for each option is public; and whether group communication is expressed in natural language or numerically. Group coordination is facilitated by an AI agent powered by a large language model (LLM). We find that reasoning LLM models perform better than non-reasoning models and that a constraint solver (CPLEX) or a reinforcement learning agent (MCTS) improves the quality of the decision. The performance of reasoning models carries over when the participants rank order their preferences instead of assigning numeric scores. Numeric feedback leads to higher quality solutions than verbal feedback and is also better than when participants state their preference between two decisions. Our findings suggest that while LLMs show promise in facilitating consensus, there remains significant room for improvement in their ability to fully capture and reason over group consensus involving numerical outcomes.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper investigates the use of Large Language Models (LLMs) to facilitate group decision-making processes where multiple participants need to reach consensus while balancing fairness and diverse preferences. The authors model group decisions as optimization problems over a decision surface with multiple options, where each participant receives gains from different choices, and the goal is to equalize participants' total profits. 

The study explores various experimental conditions: (1) whether participant gains are private arbitrary numbers or public ordered sequences, (2) whether exact gains are known or hidden, and (3) whether communication is numeric or natural language-based. The LLM agent coordinates group decisions, optionally augmented with constraint solvers (CPLEX) or reinforcement learning (MCTS).

Key findings include: reasoning models (e.g o1-preview, o1-mini) significantly outperform non-reasoning models; integrating CPLEX or MCTS improves decision quality; numeric feedback yields better outcomes than verbal feedback; and the approach generalizes to rank-based preferences. The work demonstrates both the promise and limitations of LLMs in facilitating consensus for numerically-grounded group decisions.

### Strengths
**Originality:** This paper addresses a novel and important problem at the intersection of AI, group decision-making, and fairness. The formulation of group consensus as an optimization problem over a decision surface with profit equalization as a fairness criterion is creative. The systematic comparison of reasoning vs. non-reasoning LLMs, numeric vs. verbal feedback, and the integration of constraint solvers (CPLEX) and MCTS represents an innovative multi-faceted approach.

**Quality:** The experimental design is rigorous and systematic, exploring multiple dimensions: gain structures (arbitrary vs. ordered), visibility (public vs. hidden), and communication modalities (numeric vs. verbal). The use of both simulated and real-world scenarios (grant allocation) provides practical grounding. The authors conduct proper statistical analysis with multiple trials and compare against reasonable baselines including random and greedy strategies.

**Clarity:** The paper is generally well-written with clear motivation and problem formulation. The experimental setup is explained systematically, and results are presented with appropriate visualizations (Figures 2-5). The progression from simple to complex scenarios helps readers understand the methodology.

**Significance:** This work addresses an increasingly important problem as AI systems are deployed to facilitate human collaboration and decision-making. The findings about reasoning models' superiority, the importance of numeric feedback, and the benefits of hybrid AI approaches (LLM+CPLEX/MCTS) provide valuable insights for both researchers and practitioners working on AI-assisted group coordination.

### Weaknesses
**Scalability Concerns:** The experiments are limited to relatively small groups (3-5 participants) and decision surfaces (5-10 options). Real-world group decisions often involve larger groups and more complex decision spaces. The computational complexity of CPLEX and MCTS may become prohibitive as scale increases. The paper would benefit from analysis of how performance degrades with increasing problem size.

**Fairness Criterion Limitations:** The paper exclusively uses profit equalization as the fairness criterion. However, fairness in group decision-making is multifaceted and context-dependent. Other notions like proportional fairness, maximin fairness, or procedural fairness may be more appropriate in different contexts. The choice of fairness criterion is not well justified, and alternative fairness definitions are not explored.

**Missing Ablations:** While the paper compares different LLMs and augmentations (CPLEX, MCTS), some key ablations are missing. For example: What is the contribution of the specific prompt design? How sensitive is performance to hyperparameters (temperature, MCTS iterations, etc.)? What role does the iterative refinement play versus one-shot decision-making?

### Questions
1. **Prompt Engineering:** How sensitive are results to the specific prompts used? Have you tried alternative prompt formulations? Can you include ablation studies on prompt design?

2. **Failure Cases:** Can you characterize scenarios where the LLM-based approach fails or performs poorly? Are there specific problem structures or group dynamics that are particularly challenging?

3. **MCTS and CPLEX Integration:** How exactly are CPLEX and MCTS integrated with the LLM? Are they used to verify LLM outputs, generate candidate solutions, or something else? More algorithmic detail would be helpful.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper investigates how agents can facilitate complex group decisions. The toy problem has a facilitator who must guide a group of participants to reach a consensus on a series of choices. The goal is to select a combination of options that results in the fairest possible outcome. The experiments control for different forms of communication, reasoning vs non-reasoning models, and using external tools like MCTS. The results show that reasoning models are superior, and even better with access to tools.

### Strengths
- Studying coordination and consensus in multi-agent systems is important.
- The writing is mostly clear (except for the formulation)

### Weaknesses
The problem formulation in Section 3 is sloppy. For example:

- Minimize (high - low) is very confusing, and would have a straightforward solution where they’re equal. This minimizes the difference between the maximum and minimum total gains among all participants. The goal is to make the participants' final gains as equal as possible. Then you proceed to say: “The goal of the optimizer is to minimize the maximum divergence across the participants.” This minimizes regret at the individual level. I understand the point, but I don’t think your formalization makes it clear.
- You mention $x_{jk} \geq 0$ and $x_{jk} \in$ {0, 1}. The first term is redundant, since it’s already either 0 or 1. Moreover, the formalization of “Constraints for decision j” would be much simpler with words. You’re just saying these variables are binary and only one can be 1, or basically a one-hot encoding.

While there is value in good toy problems, the one you use is quite elementary. I don’t think we can generalize any of the findings to other realistic setups. For example, the facilitator could solve the problem and tell the participants. The participants themselves could solve the problem. In my opinion, you need a task where you have guarantees that not a single agent involved can unilaterally solve the problem.

Besides the simplicity of the task, you’re also exploring what happens in simple conditions. It’s hard to see a contribution from testing reasoning vs no reasoning, etc. We have good priors for these! If the facilitator is a model, then there should be large experiments with human participants. What characteristics of the facilitator make it better at this task? If you test the help of MCTS, that won’t generalize.

Ultimately, the results appear to be straightforward, indicating that what we’re learning is trivial. Reasoning models are better at this task, which depends on planning, but that’s exactly what these models are better at. Moreover, providing tools like MCTS also helps, which is expected.

### Questions
See weaknesses

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper studies group decisions with a stylized mixed-integer program , a.k.a. menu planning. An LLM “planner” talks to three LLM “participants” for up to five rounds to pick options. The metric this paper used is “divergence” (max gap between participants’ totals). And the Main claims are the following: 1. Reasoning models beat non-reasoning; 2. Numeric feedback beats verbal and comparative; 3.  constraint solver or a reinforcement learning agent helps.

### Strengths
1. Section 5 experiments are clearly laid out; 
2. plots make the patterns easy to see.
3. Limitations are acknowledged (LLM-only, need human studies and honesty assumptions).

### Weaknesses
1. Objective function is underspecified. Sec. 3 says “Minimize (high − low)” with constraints, but “high” and “low” are not defined semantically (are they max/min (V_i)? bounds? learned?). Add clear definitions and intuition.
2. Over-claiming without context. Statements like “reasoning models perform better” and “numeric feedback is better” are presented as broad truths, but when I got into it, the evidence comes from one stylized domain and one metric. Narrow the scope of the claims would help.
3. Stylized testbed only. The whole study uses one toy MIP “menu” template. No other domains, no real data. Claims should not generalize beyond this sandbox. Do the “reasoning > non-reasoning” and “numeric > verbal” results hold in another domain (not menu/MIP)?
4. Agents are all LLMs, and NO Humans!. Both planner and “participants” are LLMs; participants are fixed to GPT-4o-mini. Results may change with stronger or different models, and we do not learn about humans!
5. Planner models are a small set (e.g., o3-mini, Gemini-2.5-Flash, DeepSeek-R1/V3). The paper does not test stronger “thinking” baselines or ablate planning depth/rounds.

### Questions
1. Why is average divergence the only metric? Does it track social welfare or fairness well? Any correlation with Pareto distance? Can the authors try other metrics like welfare/utility, Pareto distance, envy, regret, stability, or success-rate metrics, other than the divergence?
2. Can you make more rigorous framing in places like Sec. 3 where you put “Minimize (high − low)”? What exactly are high and low in the objective? Please define and justify.
3. CPLEX gets full information and unsurprisingly wins. This is not a fair operational baseline for real settings where scores are hidden. Maybe frame it as an upper bound, and report optimality gap?
4. How many search rounds matter? You cap at five. But what is the trade-off with more rounds?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper studies how the current LLMs approach the problem of complex group decisions, which is formulated as a mixed-integer programming problem.

### Strengths
The problem setting is novel.

### Weaknesses
The technical contribution is limited.

### Questions
I am generally concerned about the motivation of this paper: it seems to evaluate the capability of LLMs on solving one particularly type of mixed-integer programming problems. However, why is this type of particular interest compared with others in terms of evaluating LLMs' capability? Also, the methodologies being evaluated with are also seems to be limited: direct prompting or MCTS. I would imagine a more reasonable way for LLMs to solve this kind of tasks is to provide a set of tool calls which can directly solve the tasks via optimization procedure.

### Soundness
2

### Presentation
2

### Contribution
2
