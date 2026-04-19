# Counterfactual Effect Decomposition in Multi-Agent Sequential Decision Making

- Decision: Reject
- Scores: 6, 6, 6, 6, 8, 5

## Abstract
We address the challenge of explaining counterfactual outcomes in multi-agent Markov decision processes. In particular, we aim to explain the total counterfactual effect of an agent's action on the outcome of a realized scenario through its influence on the environment dynamics and the agents' behavior. To achieve this, we introduce a novel *causal explanation formula* that decomposes the counterfactual effect by attributing to each agent and state variable a score reflecting their respective contributions to the effect. First, we show that the total counterfactual effect of an agent's action can be decomposed into two components: one measuring the effect that propagates through all subsequent agents' actions and another related to the effect that propagates through the state transitions. Building on recent advancements in causal contribution analysis, we further decompose these two effects as follows. For the former, we consider *agent-specific effects* -- a causal concept that quantifies the counterfactual effect of an agent's action that propagates through a subset of agents. Based on this notion, we use Shapley value to attribute the effect to individual agents. For the latter, we consider the concept of *structure-preserving interventions* and attribute the effect to state variables based on their "intrinsic'' contributions. Through extensive experimentation, we demonstrate the interpretability of our decomposition approach in a Gridworld environment with LLM-assisted agents and a sepsis management simulator.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes an approach to decomposing the counterfactual effect of the actions taken by agents comprising a multi-agent Markov decision process into the effect arising directly from the other agents' resultant behaviours, and the effect coming from the resultant environmental dynamics. This provides a way to explain counterfactual outcomes in multi-agent sequential decision making settings.

### Strengths
The paper is well-written and covers a topic that I see will be very useful in the study of multi-agent systems. Using causal inference methods to understand the importance of agent decisions in complex multi-agent systems is something that will help scientists to reason about such systems. The authors also provide a good discussion of the relation of their approach to a variety of pre-existing works and notations of counterfactual effect decomposition (causal mediation formula etc.).

### Weaknesses
1) Some of the notation could be better explained. For example (and sorry if I missed this explanation somewhere in the paper), for someone like me – where I am perhaps an example of someone who is not fully a member of this particular research subfield – it wasn't clear to me why it is necessary to write $\mathbb{E}[Y \mid \tau; M]_{M^{do(I)}}$ and why someone couldn't instead just write $\mathbb{E}[Y \mid \tau; M^{do(I)}]$. Having consulted other papers (e.g., some of the references in the current paper), it does indeed seem like the former is standard notation, but it might benefit the reader and make the paper more accessible to those not directly in this research subfield to explain what exactly the subscripts mean in these expectations and why it is necessary to both "condition" on $M$ and include a subscript $M^{do(I)}$ in some of these expectations.

2) Both experiments considered only two interacting agents simulated over relatively short time horizons, and the approach isn't demonstrated on larger-scale systems. I appreciate that this is explicitly discussed as a limitation in the Discussion section, however.

### Questions
1) Could you please clarify the notation I mentioned in Weakness 1) above? (Just to be sure I understood the paper as I think I did!)

2) To what extent does the complexity of the agents themselves contribute to the complexity of computing these scores? Some multi-agent systems consist of large populations of boundedly rational agents making decisions according to relatively simple rules, or according to "satisficing" criteria. This is the other extreme to the examples considered in this paper, which consists of a very small number of agents, but who have good domain expertise/cognitive capabilities (e.g., your LLM-assisted RL agents). Does using agents with simpler cognitive rules mean it is easier to scale up these scores to larger numbers of agents?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a causal effect attribution approach that decomposes the total counterfactual effect into an agent-specific effect and an inverse state transition effect within the framework of a multi-agent Markov Decision Process (MDP). The authors further refine this decomposition at a granular level to attribute causal effects to individual agents and states. Comprehensive simulations are provided to illustrate the proposed decomposition formula, with detailed discussions on future directions.

Overall, the paper is well-written and well-structured, with ideas that are direct and easy to follow, supplemented by clear examples. While there are minor concerns regarding the degree of innovation and scalability, this work nonetheless advances the understanding of causal effects in complex multi-agent sequential decision-making systems, which I believe will be valuable for guiding future research in decision-making.

### Strengths
1. The motivation for this work is clear, with significant relevance to causal attribution in multi-agent MDPs. 

2. The theoretical decomposition is accompanied by detailed explanations, and I appreciate the authors' efforts in providing sufficient context in Sections 4 and 5.

### Weaknesses
1. The work appears to be a direct extension of Triantafyllou et al. (2024) in decomposing the agent-specific effect and Janzing et al. (2024) in decomposing the reverse state-specific effect. The main innovation of this work, as I understand, is the extension to a multi-agent MDP system. However, it is not clear which specific challenges of the multi-agent MDP framework are addressed that go beyond these previous works. For instance, if the primary challenge arises from the sequential decision-making nature of MDPs, it would be beneficial to further emphasize how the authors handle these complexities.

2. The paper addresses the computational complexity of counterfactual effect decomposition with a large number of agents and time steps, and I appreciate the discussion in Appendix H on approximation and scalability improvements. However, the simulations in the main paper are limited to two agents. It would be informative to see how the method performs with more agents (e.g., $3$ or $5$) and how the results might differ. Additionally, has the performance of the approximation methods in Appendix H been empirically evaluated, or is it based solely on theoretical considerations? Some empirical analysis of the time-accuracy tradeoff would strengthen the discussion.

### Questions
1. In Lines 242-250, the authors discuss connections between this work and mediation analysis in causal inference, where the total effect is typically decomposed into direct and indirect effects. In the multi-agent MDP setting, however, the total effect is expressed as a difference (rather than a sum) between the ASE and SSE. Could the authors elaborate on how this decomposition aligns with classical mediation analysis? Is the similarity primarily conceptual (i.e., decomposing into two parts), or is there a more detailed mapping between single-stage mediation analysis and the multi-agent MDP framework?

2. The proposed causal explanation framework relies on counterfactual estimation under a fixed action assignment sequence, which in turn requires accurate and consistent counterfactual estimates with minimal variance. In many real-world applications involving only observational data, would it be overly challenging to apply this decomposition using off-policy evaluation methods? I am curious to hear the authors’ thoughts on the feasibility of such approaches.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes new counterfactual measures that can be used to explain the observed behaviors of multi-agent systems. First, the authors proposed the total counterfactual effect, which measures the difference in the expected reward if every agent took a different action compared to the observed trajectories (as a baseline). The authors then decomposed the total counterfactual effect into two components: one measuring the effect that propagates through all subsequent agents’ actions, called the total agent-specific effect, and another related to the effect that propagates through the state transitions, called the reverse state-specific effect. Building on the method of path analysis in structural causality, the authors then further decomposed the total agent-specific effect into a series of more refined agent-specific effects. Counterfactual measures based on the Shapley value are proposed to attribute the effect to each individual agent. Similarly, the authors also proposed more refined counterfactual measures, based on the notion of intrinsic causal contributions, to decompose the reverse state-specific effects. These proposed counterfactual measures could be seen as explanations evaluating the influence of underlying causal mechanisms on the difference in the observed outcome.

### Strengths
- The proposed measures seem reasonable. All parametric assumptions have been explicitly stated. The proposed methods seem technically sound.
- Comprehensive simulations were performed. The results support the proposed algorithms.

### Weaknesses
- The identification of the proposed counterfactual measures relies on the additional parametric assumption of weak monotonicity. This assumption is restrictive and may limit the paper's application to many practical domains.
- The paper is quite notation dense. It would be best improved by including a table summarizing all the proposed measures and necessary notations.
- The introduction of each new counterfactual measure could be more gentle. This might be challenging for readers unfamiliar with counterfactual notations and structural causality. I suggest including a detailed example following each definition of the proposed counterfactual measure.

### Questions
- Can the authors provide more detailed examples for each proposed measure? For instance, can the authors explain these measures and the decomposition in the context of Figure 1 or Figure 2a? What causal mechanisms are these metrics evaluating, and what are their implications?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a causal explanation framework for multi-agent MDPs, by decomposing the total counterfactual effect of an intervention into a part that influences the agents’ behavior and and a part that depends on environment dynamics.

### Strengths
- Exposition is clear. 
- Good use of the sepsis example, which helps contextualize the abstract math throughout the text . 
- Principled development of mathematical tools to achieve the desired effect decomposition.

### Weaknesses
- The mathematical notation are quite dense and at times a little hard to grasp. For example, Def 2.1, 3.1, 3.2 all look similar but clearly they are different. The authors included textual descriptions of these mathematical definitions (individually) which is great, but I would appreciate a paragraph comparing these definitions and highlighting the differences (for example, Y vs Y_{a_{i,t}}).

### Questions
- Percentage conversion: Theorem 3.3 suggests the "total counterfactual effect" can be decomposed into "total agent-specific effect" **minus** "reverse state-specific effect". How are the two parts converted to a percentage contribution, for example in the pie chart in Fig 1b? It might be useful to show the raw effect values as well. 
- L370: cells denoted with stars are delivery locations - where are these?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper investigates the counterfactual effect of an agent's action within a multi-agent Markov decision process (MDP) environment by decomposing it into two components: effects propagating through subsequent actions and effects propagating through subsequent states. For the first component, the study introduces agent-specific effects to assess the contribution of each agent. For the second component, the reverse state-specific effect (r-SSE), the paper applies the concept of intrinsic causal contributions (ICC) to measure the impact of each state.

This decomposition is evaluated across two experimental environments, with results providing insights into the distinct roles that agents and the environment play in shaping the overall effect. The paper discusses the implications and highlights the contributions of each factor in influencing outcomes, which is helpful in decision making process.

### Strengths
1. This paper addresses an interesting and meaningful topic by explaining the counterfactual effect in MMDP through a two-part decomposition, which enables a clear, separate bi-level evaluation of each component's contribution.

2. The paper is well-written, and the proposed causal explanation formula is concise. Modifying the state-specific effect into a reverse state-specific effect, which can be combined with agent-specific effects, offers a reasonable way to evaluate counterfactual effects.

3. The experimental results seem promising, particularly in terms of supporting causal interpretation.

### Weaknesses
1. The explanation of the decomposition of TCFE, especially for r-SSE, lacks clarity and is somewhat difficult to follow since it's a reverse version. Including a toy example or a causal graph for Theorem 3.3 would improve comprehension.

2. The decision-making process in the Gridworld experiment (Section 6.1) is not clearly conveyed. A simplified trajectory or causal graph—such as a plot similar to Figure 1(a) or Table 1 in Appendix I—would help clarify the process.

3. While the experimental results allow for reasonable causal interpretation, there is limited evidence supporting their robustness or reliability. The results are reasonable under the specific settings, but can the reported percentages be considered reliable?

### Questions
1. In Figure 1(b), what does the unannotated section of the pie chart represent?

2. Adding a causal graph to illustrate ASE and r-SSE would improve clarity and enhance understanding of the counterfactual effect decomposition. 

3. In Figures 3(a) and 3(b), why is the trust parameter allowed to be 0 in (b) but not in (a)?

4. While the identifiability of TCFE is discussed under the condition of noise monotonicity, what about the identifiability of r-SSE and ASE?

5. For the proposed decomposition of TCFE, is there a way to verify the validity of the results? Specifically, how reliable are the percentage contributions assigned to each agent or state transition? Could simulations with known ground-truth causal graphs be included to demonstrate their reliability? By comparing the estimated contributions of each agent and state with the ground truth, the causal explanation formula and subsequent algorithm would be more compelling.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 6

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper studied counterfactual effect decomposition in multi-agent MDPs under the structural causal model framework. Specifically, it first shows that the total counterfactual effect of an agent's action can be decomposed into the agent-specific effect and the reverse state-specific effect. Second, it proposes an axiomatic framework based on agent-specific effects to attribute the total effect to individual agents, and introduces a method for attributing the reverse state-specific effect to state variables based on their "intrinsic" contributions.

### Strengths
1. This paper bridges reinforcement learning with the causal reasoning community, extending several well-established concepts—such as natural direct/indirect effect decomposition, intrinsic causal contributions, and flow-based attribution methods—into multi-agent MDPs.

2. It provides insights into the distinct roles that agents and the environment play in influencing effects, demonstrated through real-world examples.

### Weaknesses
**Novelty** The paper examines the counterfactual effect of a single agent’s action. However, it would be more impactful and interesting to investigate the estimands for multiple agents' actions in a multi-agent MDP, especially under varying interaction structures. Additionally, under the single-agent action setting, Theorems 3.3 and 4.2 are straightforward extensions of the causal mediation formula (Pearl, 2001) and intrinsic causal contributions (Janzing et al., 2024).

**Experiment** I find the validity of conclusions such as the causal explanation formula (lines 411 and 412) questionable, given that both TCFE and tot-ASE are estimated quantities. I also find there is no experiment to demonstrate the estimation accuracy. It would be more insightful to first use simulations to assess whether the estimation algorithm accurately approximates the true values.

**Assumption** The interpretation of the key assumption for identification, noise monotonicity, is unclear. A discussion on the conditions under which it is valid, and when it may not hold, would clarify its applicability. Additionally, an illustration using the sepsis example would be helpful.

### Questions
Please see Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
