

{0}------------------------------------------------

# INTERPRETING EMERGENT PLANNING IN MODEL-FREE REINFORCEMENT LEARNING

Thomas Bush<sup>1†</sup>, Stephen Chung<sup>1†</sup>, Usman Anwar<sup>1†</sup>, Adrià Garriga-Alonso<sup>2</sup>, David Krueger<sup>3</sup>

<sup>1</sup>University of Cambridge, <sup>2</sup>FAR AI, <sup>3</sup>Mila, University of Montreal

28tbush@gmail.com, {mhc48, ua237, dsk30}@cam.ac.uk, adria@far.ai

<sup>†</sup>Equal contribution.

## ABSTRACT

We present the first mechanistic evidence that model-free reinforcement learning agents can learn to plan. This is achieved by applying a methodology based on concept-based interpretability to a model-free agent in Sokoban – a commonly used benchmark for studying planning. Specifically, we demonstrate that DRC, a generic model-free agent introduced by Guez et al. (2019), uses learned concept representations to internally formulate plans that both predict the long-term effects of actions on the environment and influence action selection. Our methodology involves: (1) probing for planning-relevant concepts, (2) investigating plan formation within the agent’s representations, and (3) verifying that discovered plans (in the agent’s representations) have a causal effect on the agent’s behavior through interventions. We also show that the emergence of these plans coincides with the emergence of a planning-like property: the ability to benefit from additional test-time compute. Finally, we perform a qualitative analysis of the planning algorithm learned by the agent and discover a strong resemblance to parallelized bidirectional search. Our findings advance understanding of the internal mechanisms underlying planning behavior in agents, which is important given the recent trend of emergent planning and reasoning capabilities in LLMs through RL.

## 1 INTRODUCTION

In reinforcement learning (RL), decision-time planning – that is, the capacity of selecting immediate actions to perform by predicting and evaluating the consequences of future actions – is conventionally associated with agents that possess explicit world models, like MuZero (Schrittwieser et al., 2020). This naturally raises the question: can model-free reinforcement learning agents – that is, agents which lack explicit world models – also learn to perform decision-time planning?

In prior work, Guez et al. (2019) introduced Deep Repeated ConvLSTM (DRC) agents. Despite lacking an explicit world model, DRC agents behave like they perform decision-time planning. For example, they excel at strategic domains like Sokoban, and perform better if given extra test-time compute (Guez et al., 2019; Taufeeque et al., 2024). However, this only partially answers the above question as these behaviors may not be due to internal planning but, rather, other mechanisms that generate planning-like behavior in the environments studied. In this paper, we mechanistically analyze a Sokoban-playing DRC agent and show that it is indeed internally planning. In doing so, we provide the first non-behavioral evidence that model-free RL agents can learn to internally plan.

Using concept-based interpretability (Kim et al., 2018), we provide three types of convergent evidence showing that the DRC agent has learned, and is making use of, concepts that are instrumentally useful for planning. First, we use linear probes (Alain & Bengio, 2016) to show that the agent represents specific concepts that predict the long-term effects of its actions on the environment. Then, we demonstrate that these concept representations are associated with a learned planning process by analyzing how the agent uses them to iteratively construct ‘plans’ at test-time. Finally, we demonstrate that these concept representations causally influence the agent’s behavior as would be expected if these representations were being used for planning.

To summarize, this paper makes the following contributions:

{1}------------------------------------------------

![Figure 1: Examples of the DRC agent internally forming plans to push boxes to targets. The figure is a 3x3 grid of Sokoban game states. (A) 'Agent evaluates its plan' shows 6 frames from step 0 to 5, tick 3. (B) 'Agent adapts its plan' shows 3 frames from step 0 to 2, tick 3. (C) 'Agent plans backwards from targets' shows 3 frames from step 0 to 2, tick 3. (D) 'Agent plans forwards from boxes' shows 3 frames from step 0, tick 1 to 0, tick 3. (E) 'Agent extends routes in parallel' shows 3 frames from step 0, tick 1 to 0, tick 3. Each frame shows the agent (blue square), boxes (orange), and targets (green). Purple arrows indicate planned moves.](c803f6f6e2c49429d2951832bd0f208d_img.jpg)

Figure 1: Examples of the DRC agent internally forming plans to push boxes to targets. The figure is a 3x3 grid of Sokoban game states. (A) 'Agent evaluates its plan' shows 6 frames from step 0 to 5, tick 3. (B) 'Agent adapts its plan' shows 3 frames from step 0 to 2, tick 3. (C) 'Agent plans backwards from targets' shows 3 frames from step 0 to 2, tick 3. (D) 'Agent plans forwards from boxes' shows 3 frames from step 0, tick 1 to 0, tick 3. (E) 'Agent extends routes in parallel' shows 3 frames from step 0, tick 1 to 0, tick 3. Each frame shows the agent (blue square), boxes (orange), and targets (green). Purple arrows indicate planned moves.

Figure 1: Examples of the DRC agent internally forming plans to push boxes to targets. A purple arrow on a square means that a linear probe decodes that the agent plans to push a box off of that square in the associated direction. No arrow on a square means that the probe decodes that agent does not plan to push a box off of that square. (A) The agent *evaluates* a naively-appealing route, concludes it is infeasible, and forms a longer alternate path. (B) The agent *adapts* its plan and changes the target it plans to push the left-most box to. (C) The agent extends part of its plan *backward* from a target. (D) The agent extends part of its plan *forward* from a box. (E) The agent extends many parts of its plan in *parallel*. We provide further examples in Appendices A.2.1-A.2.5.

- We design a procedure, based on concept-based interpretability, for determining if a model-free agent performs planning using a hypothesized set of concepts. This procedure involves (1) probing for planning-relevant concepts, (2) investigating plan formation in the agent’s internal representations, and (3) verifying the causal effect of plans on the agent’s behavior.
- Using this procedure, we show that, in Sokoban, a DRC agent (Guez et al., 2019) internally forms plans, and that these plans can be altered to steer the agent. We find this agent learns a planning algorithm resembling parallelized bidirectional search, which differs from commonly-used planning algorithms in RL.

This work aligns with the growing body of research demonstrating that model-free RL agents can learn to plan and even reason. For example, in this study, we show that DRC agents can learn to evaluate and revise plans. Recently, DeepSeek-R1, an LLM with reasoning capabilities primarily trained via RL, has demonstrated similar self-correction behavior in its reasoning, referred to as ‘aha moments’ (Guo et al., 2025). As such, we believe that understanding the mechanisms behind these emergent capabilities in RL agents is highly important.

## 2 BACKGROUND

### 2.1 PLANNING IN REINFORCEMENT LEARNING

*Planning* has many meanings in RL, encompassing algorithms utilizing environment models during training (Sutton, 1991) or at decision time (Silver et al., 2016; Chung et al., 2024a). In this work, we study whether an RL agent is specifically performing *decision-time* planning. Henceforth, we use ‘planning’ and ‘decision-time planning’ interchangeably. In past work, an agent is considered to be *planning* in this sense if it engages with an (explicit) world model to select actions associated with the best predicted long-term consequences (Hamrick et al., 2020; Chung et al., 2024a). An example is MuZero (Schrittwieser et al., 2020), which applies a planning algorithm called Monte Carlo Tree Search (Coulom, 2006) to a model of its environment to select actions associated with the best long-run consequences. Other similar agents are VPN (Oh et al., 2017), IBP (Pascanu et al., 2017), IZA (Racanière et al., 2017), MCTSNet (Guez et al., 2018b), and Thinker (Chung et al., 2024a) agents.

{2}------------------------------------------------

By definition, model-free RL agents lack an *explicit* world model. This makes it difficult to reuse past definitions of planning that presume that an explicit world model is available. Thus, for the purposes of this work, we provide a pragmatic characterization of planning that we use as a foundation for investigating whether the model-free agent studied in this paper performs planning.

We consider plans to be sequences of potential future actions. We characterize an agent as planning *if it selects actions to perform by considering plans that it formulates and evaluates based on predicted future consequences*. This is similar to how planning is understood in neuroscience (Mattar & Lengyel, 2022). It also mirrors model-based definitions of planning but relaxes the requirement for an explicit world model to the requirement that an agent predict consequences of future actions, regardless of the method used. We discuss our characterization further in Appendix E.1. For an agent to plan under our characterization, it must: (i) form plans, (ii) evaluate plans by predicting their consequences, and (iii) be influenced by these plans when acting.

### 2.2 SOKOBAN

Sokoban is an episodic, fully-observable, deterministic environment in which an agent moves around walls in an 8x8 grid to push four boxes onto four targets. When an agent moves up/down/left/right into a square containing a box, the box is pushed up/down/left/right. Sokoban levels let agents perform actions with irreversible, negative, long-run consequences (moving boxes so the puzzle is unsolvable). Sokoban is thus difficult – it is PSPACE-complete (Culberson, 1997) – and a common benchmark for studying planning (Racanière et al., 2017; Guez et al., 2019; Hamrick et al., 2020). We study a version of Sokoban where the agent observes a symbolic representation  $x_t \in \mathbb{R}^{8 \times 8 \times 7}$  of the environment. For ease of inspection, all figures are presented as pixel representations. Figure 2 compares these two representations. Appendix E.2 further explains this environment.

![Figure 2: Pixel and symbolic representations of a Sokoban board. (a) Pixel: A 2D grid of colored squares representing the environment state. (b) Symbolic: A 2D grid of colored squares representing the environment state, with a different color scheme.](89337b7d31e8913d0af346758554070a_img.jpg)

Figure 2: Pixel and symbolic representations of a Sokoban board. (a) Pixel: A 2D grid of colored squares representing the environment state. (b) Symbolic: A 2D grid of colored squares representing the environment state, with a different color scheme.

Figure 2: Pixel and symbolic representations of a Sokoban board.

### 2.3 DEEP REPEATED CONVLLSTM (DRC) AGENTS

Deep Repeated ConvLSTM (DRC) agents (Guez et al., 2019) are model-free agents based on ConvLSTMs that perform multiple computational ticks per time step. ConvLSTMs (Shi et al., 2015) are LSTMs (Hochreiter & Schmidhuber, 1997) that utilize 3D hidden states and convolutional connections. At each time step  $t$ , a DRC agent passes an observation  $x_t$  through a convolutional encoder to generate an encoding  $i_t \in \mathbb{R}^{H_0 \times W_0 \times G_0}$ . This is then processed by  $D$  ConvLSTM layers. At time  $t$ , the  $d$ -th ConvLSTM has a cell state  $g_t^d \in \mathbb{R}^{H_d \times W_d \times G_d}$ . Unlike standard recurrent networks which perform a single tick of recurrent computation per time step, DRC agents perform  $N$  ticks of recurrent computation per step. Guez et al. (2019) show these internal ticks improve the performance and generalization of DRC agents. Appendix E.3 provides further architectural details.

DRC agents behave in a manner that suggests they internally engage in decision-time planning. For instance, DRC agents outperform model-based agents like MuZero (Schrittwieser et al., 2020) in Sokoban (Chung et al., 2024b), and exhibit improved performance when given extra test-time compute (Taufeeque et al., 2024). This raises a question: do DRC agents genuinely learn to internally perform planning, or is their planning-like behavior merely a result of complex learned heuristics?

In this paper, we investigate whether a Sokoban-playing DRC agent internally plans. The agent we study has  $D = 3$  ConvLSTM layers and performs  $N = 3$  internal ticks per step. The agent’s encoder and ConvLSTMs have 32 channels ( $G_d = 32$ ) and utilize kernels of size 3 with a single layer of input zero padding. Thus, all cell states share Sokoban’s spatial dimensions ( $H_d = W_d = 8$ ). The agent is trained for 250 million transitions on the unfiltered Boxoban training set (Guez et al., 2018a) using a similar training setup as Guez et al. (2019) as explained in Appendix E.4. Appendix E.5 shows that, consistent with Guez et al. (2019), this agent exhibits planning-like behavior.

### 2.4 CONCEPT-BASED INTERPRETABILITY

Concept-based interpretability is an approach to explaining neural network behavior that involves identifying which concepts a network internally represents (Kim et al., 2018). A concept is generally

{3}------------------------------------------------

understood as a unit of knowledge (Schut et al., 2023). In this paper, we specifically consider ‘multi-class’ concepts, which can formally be defined as mappings from input states (or parts of input states) to some fixed classes. That is, multi-class concepts correspond to interpretable, discrete features, and map inputs to classes of that concept. For instance, a multi-class Sokoban concept might be ‘the number of empty targets’. This concept would map any observed Sokoban board  $x_t$  to a class in {ONE, TWO, THREE, FOUR} depending on the number of remaining empty targets in  $x_t$ .

We focus on concepts networks represent *linearly* (Mikolov et al., 2013). To check if a network linearly represents concepts, we use *linear probes*. These are linear classifiers trained to predict concept classes assigned to inputs using the associated network activations (Alain & Bengio, 2016). As linear classifiers, linear probes compute logits  $l_k = w_k^T g$  for each class  $k$  by projecting network activations  $g \in \mathbb{R}^d$  along a class-specific vector  $w_k \in \mathbb{R}^d$ . Belinkov (2022) explains probes further.

## 3 METHODOLOGY

### 3.1 A PROCEDURE FOR INVESTIGATING MODEL-FREE PLANNING

In Section 2.1, we characterized planning as requiring that an agent (i) formulate plans, (ii) evaluate the consequences of these plans, and (iii) be guided by these plans when selecting actions. If an agent learns to plan, we expect planning-relevant concepts to emerge in its internal representations to meet the first condition. These concepts ought to reflect the agent’s plan, and so should correspond to potential future actions, or to their likely environmental effects. Additionally, evidence of plan evaluation – such as avoiding or improving bad plans – should exist to satisfy the second condition. Lastly, to fulfill the third condition, the plan must causally influence the agent’s behavior. To determine if an agent exhibits these three properties, we follow the procedure outlined below:

1. **Probe for Concept Representations.** First, we identify a group of environment-specific concepts that could be instrumentally useful for planning. We then use linear probes to establish whether these concepts are being (linearly) represented by the agent (Section 4).
2. **Investigate Plan Formation.** Next, we focus on gathering qualitative evidence of the agent forming plans based on the planning-relevant concepts probed for in the previous step, and evidence of the agent evaluating and refining these plans (Section 5).
3. **Confirm Behavioral Dependence.** Finally, we confirm that these internal plans influence the agent’s behavior. For instance, we show that the agent can be steered to form and execute desired plans by intervening on plan representations within the network (Section 6).

### 3.2 PLANNING-RELEVANT CONCEPTS IN SOKOBAN

To apply this procedure, we must specify concepts we expect the agent to plan with. Sokoban has a grid-based structure with localized transition dynamics, i.e., the future state of a square is determined by the current state of its neighbors. This makes spatially local concepts (i.e., concepts related to individual or connected squares) more natural for planning than spatially global concepts (i.e., representations of the whole board). We thus claim that an agent that learns to plan in Sokoban may do so by encoding concepts localized to individual squares. We call these ‘square-level’ concepts. Such concepts seem natural for DRC agents as the 3D structure of ConvLSTMs allows for spatial correspondence between the Sokoban grid and agent hidden states. We focus on multi-class square-level concepts which, as explained further in Appendix E.6, map grid squares to concept classes.

We hypothesize that the agent will plan using the following square-level, multi-class concepts:

- **Agent Approach Direction** ( $C_A$ ): For a given square, this concept encodes whether the agent will move onto the square in the future. If so, it also encodes the direction from which the agent will move onto the square the next time the agent moves onto it.
- **Box Push Direction** ( $C_B$ ): For a given square, this concept encodes whether a box will be pushed off the square in the future. If so, it also encodes the direction in which the next box pushed off this square will be pushed.

Figure 3 illustrates the classes assigned to each square of a Sokoban board by these concepts over six transitions near the end of an episode. Both concepts map each grid square of the agent’s observed

{4}------------------------------------------------

![Figure 3: Examples of the classes assigned to the squares of a Sokoban board over 6 transitions. (a) Agent Approach Direction C_A: Shows a 6x6 grid with a black agent, a yellow box, and green goal squares. Arrows indicate the direction the agent is approaching each square. (b) Box Push Direction C_B: Shows the same grid with arrows indicating the direction the box is being pushed.](a0167e3dcece9dcd8a378bcd98fb9cfa_img.jpg)

(a) Agent Approach Direction  $C_A$

(b) Box Push Direction  $C_B$

Figure 3: Examples of the classes assigned to the squares of a Sokoban board over 6 transitions. (a) Agent Approach Direction C\_A: Shows a 6x6 grid with a black agent, a yellow box, and green goal squares. Arrows indicate the direction the agent is approaching each square. (b) Box Push Direction C\_B: Shows the same grid with arrows indicating the direction the box is being pushed.

Figure 3: Examples of the classes assigned to the squares of a Sokoban board over 6 transitions (from left to right) by the concepts ‘Agent Approach Direction’ ( $C_A$ ) and ‘Box Push Direction’ ( $C_B$ ). An arrow corresponds to the assignment of the associated directional class. The lack of an arrow in a square indicates the assignment of the class `NEVER`.

Sokoban board to the classes  $\{\text{UP, DOWN, LEFT, RIGHT, NEVER}\}$ . The directional classes correspond to the agent’s movement directions. If the next time the agent *steps onto a specific square*, the agent steps onto that square from the left, the concept  $C_A$  would map this square to the class `LEFT`. If the next time the agent *pushes a box off of specific square*, the box is pushed to the left, the concept  $C_B$  would map this square to the class `LEFT`. Finally, the class `NEVER` corresponds to the agent not stepping onto or pushing a box off of a square again for the remainder of the episode.

Both concepts depend on the agent’s behavior: we can only determine the classes these concepts map grid squares to *after* observing the agent’s behavior over the entire episode. Furthermore, as shown in Figure 3, the classes squares are mapped to will change at every transition. Once an agent steps onto a square, the classes assigned to that square will update to represent the agent’s *future* interactions with that square. We investigate alternate concepts in Appendices D.4 and D.5.

## 4 PROBING FOR CONCEPT REPRESENTATIONS

We now perform the first step of our analysis: determining whether the agent internally represents the concepts that we hypothesize it uses to internally form and evaluate plans.

### 4.1 EXPERIMENT DETAILS

Specifically, we use linear probes to determine if the agent represents (a)  $C_A$ , the agent’s future movement onto squares, and (b)  $C_B$ , the future directions boxes are pushed off of squares. We train linear probes that take as input the agent’s cell state activations after the final of the three computational ticks performed each step. We train separate probes for the agent’s three layers.

We hypothesize the agent will learn a spatial bijection between its cell state and the Sokoban grid. Thus, when predicting  $C_A$  and  $C_B$  at each location  $(x, y)$ , our probes receive as input cell state activations centered on  $(x, y)$ . We train both 1x1 probes (which take as input just the activations at  $(x, y)$ ) and 3x3 probes (which take as input the 3x3 patch of activations around  $(x, y)$ ). These probes have 160 and 1440 parameters, so are unlikely to overfit. We consider larger probes in Appendix D.3.

Each probe is trained using logistic regression with the AdamW optimizer, and five unique initialization seeds. The training dataset is generated by running the agent for 3000 episodes on levels from the Boxoban unfiltered training dataset (Guez et al., 2018a). We test probes on a test set of transitions generated by running the agent for 1000 episodes on levels from the Boxoban unfiltered validation dataset. Further probe training details are given in Appendix D.1. We compare the performance of all probes to baseline probes that receive the raw observation  $x_t$  as input. This comparison aims to assess the extent to which probes’ abilities to predict concept classes are due to these concepts being internally represented by the agent rather than the probes learning how to do so themselves.

{5}------------------------------------------------

![Figure 4: Two bar charts showing Macro F1 scores for predicting concepts C_A and C_B. Chart (a) for C_A shows scores for Layer 1, Layer 2, Layer 3, and Baseline for 1x1 and 3x3 probes. Chart (b) for C_B shows the same. In both cases, Layer 1 and Layer 2 1x1 probes perform best, while the Baseline 3x3 probe performs worst.](c54b3ca7603d65d4589151bc3a49d054_img.jpg)

| Concept | Probe Type | Layer | Macro F1 (approx.) |
|-|-|-|-|
| $C_A$ | 1x1 | Layer 1 | 0.8 |
|  |  | Layer 2 | 0.85 |
|  |  | Layer 3 | 0.85 |
|  |  | Baseline | 0.25 |
|  | 3x3 | Layer 1 | 0.9 |
|  |  | Layer 2 | 0.9 |
|  |  | Layer 3 | 0.9 |
|  |  | Baseline | 0.45 |
| $C_B$ | 1x1 | Layer 1 | 0.85 |
|  |  | Layer 2 | 0.95 |
|  |  | Layer 3 | 0.85 |
|  |  | Baseline | 0.25 |
|  | 3x3 | Layer 1 | 0.95 |
|  |  | Layer 2 | 0.95 |
|  |  | Layer 3 | 0.95 |
|  |  | Baseline | 0.6 |

Figure 4: Two bar charts showing Macro F1 scores for predicting concepts C\_A and C\_B. Chart (a) for C\_A shows scores for Layer 1, Layer 2, Layer 3, and Baseline for 1x1 and 3x3 probes. Chart (b) for C\_B shows the same. In both cases, Layer 1 and Layer 2 1x1 probes perform best, while the Baseline 3x3 probe performs worst.

Figure 4: Macro F1s achieved by probes when predicting  $C_A$  and  $C_B$  using the cell state at each layer, or, for the baseline probes, using the observation. Error bars show  $\pm 1$  standard deviation.

![Figure 5: Six grid world diagrams showing internal plans for concepts C_A and C_B. Each grid has colored squares (green, yellow, red, blue) and arrows (teal, purple) indicating expected moves. Some squares have no arrows, indicating no plan.](46f43cb4ffd47565e7c0ca306d461435_img.jpg)

Figure 5: Six grid world diagrams showing internal plans for concepts C\_A and C\_B. Each grid has colored squares (green, yellow, red, blue) and arrows (teal, purple) indicating expected moves. Some squares have no arrows, indicating no plan.

Figure 5: Examples of internal plans computed by the agent. An internal plan corresponds to the agent’s combined square-level representations of  $C_A$  and  $C_B$ . That is, an internal plan corresponds to the classes the agent represents these concepts as mapping squares of observed boards to. These internal plans are decoded from the agent’s final layer cell state by a 1x1 probe. Teal and purple arrows respectively indicate the agent expects to next step on to, or push a box off, a square in the associated direction. No arrow indicates the agent does not plan to step onto, or push a box off, a square again. Further examples of internal plans are given in Figures 10, 11 and 12 in Appendix A.1.

### 4.2 RESULTS

In many Sokoban boards, the agent will never move onto, nor push a box off, a large number of squares. As a result, many squares are assigned the label NEVER for both concepts in our probing datasets, leading to class imbalance. We therefore evaluate probe performance using macro F1 scores in place of accuracy. Figure 4 shows the macro F1 scores achieved by probes trained to predict the classes assigned to Sokoban squares by  $C_A$  and  $C_B$ . The probes that predict these concepts using the agent’s cell state activations vastly outperform the baseline, implying the agent linearly represents  $C_A$  and  $C_B$ . This aligns with past work finding linear concept representations in many different networks (Nanda et al., 2023; McGrath et al., 2022; Zou et al., 2023).

Figure 4 confirms that the agent represents square-level concepts at localized positions of its ConvLSTM cells as opposed to distributing representations across adjacent positions. This is evidenced by the minimal improvement in performance when moving from a 1x1 probe to a 3x3 probe, compared to the significant improvement in baseline performance. We thus focus on 1x1 probes for the remainder of this paper. Interestingly, Figure 4 also shows that while probes at layer 2 generally perform slightly better than probes at layer 1, there is little variation in performance across layers. This indicates that the concepts are represented across all layers. We thus hypothesize that the agent is engaged in iterative computation (Jastrzebowski et al., 2018), whereby it refines plans across layers.

## 5 INVESTIGATING PLAN FORMATION

In this section, we now provide qualitative evidence that the agent forms plans by searching forward from the boxes and backward from the targets, and that the agent develops, evaluates, and adapts plans in parallel. In this section, we primarily focus on descriptive explanations of how the agent forms plans and the general shape of the plans. We defer more conclusive evidence – in the form of intervening on the agent’s plan formation process to steer the agent’s behavior – to the next section.

Previously, we demonstrated that the agent encodes (at least) two planning-relevant concepts:  $C_A$  and  $C_B$ . These concepts represent predictions regarding how the agent will act when moving onto a given square in the future, and how the environment – specifically, the locations of boxes – will

{6}------------------------------------------------

be affected by these actions. We thus posit that the agent’s representations of these concepts – when looked at holistically, over the entire board – will collectively constitute a plan that the agent forms and adapts. For example, in Figure 5 we visualize the agent’s representations of  $C_B$  and  $C_A$  over entire Sokoban boards, as decoded from the agent’s cell state by a 1x1 probe in different levels. Three observations can be made from Figure 5: (a) the arrows, which indicate the direction the agent expects to move or push boxes, tend to be connected and trace a path; (b) the arrows tend to connect boxes to specific targets; (c) the arrows collectively form a plan which corresponds to solving the level. In Appendix A.1 we visualize the agent’s plan across layers, and show that, while the agent’s plans often contains flaws (like the lack of one necessary arrow in Figure 5c), they usually consist of connected paths for the agent to follow and connected routes linking boxes and targets.

A natural question then arises: how does the agent form plans? To answer this, we direct attention to Figure 1. Figure 1 visualizes the agent’s plans in terms of  $C_B$  (e.g. the routes the agent plans to push boxes) over the initial steps (A-C) and internal ticks (D-E) of episodes. As can be seen in Figure 1, the agent forms plans *iteratively*. Interestingly, the agent appears to form plans iteratively by searching *forward* from boxes – as illustrated in Figure 1(C) – and *backward* from targets – as illustrated in Figure 1(D). That the agent seems to plan via bidirectional search – which is known to be especially efficient when it is applicable (Russell & Norvig, 2010) – may explain why Guez et al. (2019) found DRC agents to rival specialized planning architectures reliant on forward search. Indeed, as shown in Figure 1(E), the agent seems to utilize a form of *parallelized* bidirectional search whereby it extends multiple plans simultaneously. Appendices A.2.3, A.2.4 and A.2.5 respectively contain further instances of the agent appearing to utilize forward, backward, and parallel search.

However, recall that, in Section 2.1, we characterized planning as requiring an agent to evaluate the plans it considers. Evidence suggestive of the agent evaluating plans can be seen in Figure 1(A)-(B). Figures 1(A)-(B), show examples in which the agent appears to (1) formulate a naive plan, (2) evaluate it, and then, upon realizing that it is infeasible or could be improved, (3) adapt its plan accordingly. For instance, in Figure 1(B), the agent changes the targets it plans to push different boxes towards. This is suggestive of the agent using an *evaluative* search algorithm when forming plans. Appendices A.2.1 and A.2.2 contain further examples of the agent seeming to evaluate plans and either plan to push a box a longer route, or change which boxes it plans to push to which targets.

Further evidence of the agent planning via an iterative search algorithm can be seen in Figure 6. For Figure 6, we forced the agent to remain stationary for 5 steps prior to acting in 1000 episodes. These 5 ‘thinking steps’ give the agent 15 internal ticks of extra test-time compute. Figure 6 reports the macro F1 when using 1x1 probes to decode  $C_A$  and  $C_B$  from the agent’s final layer cell state at each of the 15 extra internal ticks, averaged over 1000 episodes. Clearly, the macro F1 improves with the number of ticks. Since the concepts are predictions of future behavior, we can see the predictions of our probes at any tick as being the agent’s internal plan *at that tick*. We can then see the corresponding macro F1 as reflecting the quality of the agent’s plan at that tick. Figure 6 shows that, as would be expected if the agent planned via an iterative search, the agent’s plans iteratively improve when given extra compute.

Appendix A.3.1 shows test-time plan refinement occurs at all layers. Appendix A.3.2 provides evidence that it is a consequence of the agent searching deeper. Appendix C.2 shows that this ‘test-time plan refinement capability’ arises early in training.

When considered with the agent’s planning-like behavior, the above evidence indicates the agent uses its representations of  $C_A$  and  $C_B$  for search-based planning. Further evidence of this is given in Appendices A.2.6-A.2.9 which show examples of the agent planning in out-of-distribution levels, such as levels in which the agent itself is not present (Appendix A.2.6), levels with additional boxes and targets (Appendix A.2.7), and levels in which walls appear and disappear (Appendices A.2.8-A.2.9). These examples suggest the agent’s ability to adapt and generalize – benefits of model-based planning Guez et al. (2019) show DRC agents possess – relate to its representations of  $C_A$  and  $C_B$ .

![Figure 6: A line graph showing Macro F1 (Y-axis, ranging from 0.6 to 0.9) versus Internal Tick (X-axis, ranging from 0 to 15). Two lines are plotted: C_A (blue line with circles) and C_B (orange line with circles). Both lines show an increasing trend, with C_B consistently achieving a higher Macro F1 than C_A. The performance for both concepts plateaus after approximately 12 internal ticks.](84e2ac543ffc4145dc85b05a48ec62e3_img.jpg)

| Internal Tick | Macro F1 ( $C_A$ ) | Macro F1 ( $C_B$ ) |
|-|-|-|
| 0 | 0.58 | 0.58 |
| 3 | 0.72 | 0.78 |
| 6 | 0.78 | 0.83 |
| 9 | 0.81 | 0.85 |
| 12 | 0.83 | 0.86 |
| 15 | 0.84 | 0.86 |

Figure 6: A line graph showing Macro F1 (Y-axis, ranging from 0.6 to 0.9) versus Internal Tick (X-axis, ranging from 0 to 15). Two lines are plotted: C\_A (blue line with circles) and C\_B (orange line with circles). Both lines show an increasing trend, with C\_B consistently achieving a higher Macro F1 than C\_A. The performance for both concepts plateaus after approximately 12 internal ticks.

Figure 6: Macro F1 when using 1x1 probes to decode  $C_A$  and  $C_B$  from the agent’s final layer cell state at each of the additional 15 internal ticks performed by the agent when the agent is given 5 ‘thinking steps’, averaged over 1000 episodes.

{7}------------------------------------------------

|  | Layer 1 |  | Layer 2 |  | Layer 3 |  |
|-|-|-|-|-|-|-|
|  | Trained (%) | Random (%) | Trained (%) | Random (%) | Trained (%) | Random (%) |
| AS | 94.6 ( $\pm 0.5$ ) | 33.7 ( $\pm 32.7$ ) | 90.1 ( $\pm 1.9$ ) | 29.8 ( $\pm 36.8$ ) | 98.8 ( $\pm 0.0$ ) | 27.8 ( $\pm 37.9$ ) |
| BS | 56.2 ( $\pm 1.4$ ) | 31.5 ( $\pm 13.9$ ) | 72.7 ( $\pm 1.1$ ) | 30.9 ( $\pm 25.8$ ) | 80.6 ( $\pm 2.4$ ) | 4.1 ( $\pm 5.4$ ) |

Table 1: Success rates (%) when intervening on each layer using representations from trained and randomly initialized probes. AS and BS refer to ‘Agent-Shortcut’ and ‘Box-Shortcut’ interventions. Success rates are averaged over 5 interventions performed. We report  $\pm 1$  standard deviations.

## 6 INVESTIGATING THE ROLE OF PLANS

So far, we have shown that the DRC agent represents  $C_A$  and  $C_B$  (Section 4), and that it uses these representations to form internal plans (Section 5). We now conclude our analysis by showing that these representations are causally responsible for the agent’s behavior. Specifically, we: (1) use these representations to intervene on the agent to force it to form and execute specific plans, and (2) show that these representations emerge concurrently with planning-like behavior during training.

### 6.1 INTERVENING ON AGENT PLANS

First, we show we can intervene on the agent’s activations to alter its behavior over entire episodes. Our interventions involve adding concept vectors learned by probes to the agent’s activations to force it to represent concepts in specific ways. We then observe the causal effect of our interventions on the agent’s behavior. Recall that a 1x1 probe projects activations along a vector  $w_k \in \mathbb{R}^{32}$  to compute a logit for class  $k$  of some multi-class concept  $C$ . We thus encourage the agent to represent square  $(x, y)$  as class  $k$  for concept  $C$  by adding  $w_k$  to position  $(x, y)$  of the agent’s cell state  $g_{x,y}$ :

$$g'_{x,y} \leftarrow g_{x,y} + w_k \quad (1)$$

If the agent indeed uses  $C_A$  and  $C_B$  for planning, altering the agent’s square-level representations of these concepts ought to modify its internal plan and, subsequently, its long-term behavior.

We intervene in two sets of handcrafted levels: ‘Agent-Shortcut’ and ‘Box-Shortcut’ levels. These sets of levels are characterized by, in each level, there existing two plans: a short plan and a long plan. The plans are similar, but differ in lengths. The agent by default follows the optimal (short) plan. We show our interventions cause it to instead form and execute the suboptimal (long) plan.

In ‘Agent-Shortcut’ levels all boxes and targets are in one region of the board, and the agent can follow either a long or short path to this region. In these levels, we intervene using vectors learned by probes trained to predict  $C_A$  to steer the agent to plan to move along the long path. Our intervention consists of two parts. We add the vector for `NEVER` to cell state positions on the short path. We call this the ‘short-route’ intervention. We also add the vector for the direction which would lead the agent to move onto the first square of the long path to the appropriate cell state position. We call this the ‘directional’ intervention. An Agent-Shortcut intervention is illustrated in Figure 7b.

‘Box-Shortcut’ levels are specially-designed levels in which three boxes are adjacent to targets and a fourth box is not. The final box can be pushed a long or short route to a target. In these levels, we intervene using vectors learned by probes trained to predict  $C_B$  to steer the agent to push this box the long route. Our intervention again consists of two parts. We add the vector for `NEVER` to cell positions on the short route. We also add the directional representation which would encourage the agent to push the box the longer route to the box’s initial position. We again call these the ‘short-route’ and ‘directional’ interventions. A Box-Shortcut intervention is illustrated in Figure 8b.

We intervene on 200 levels of each type. We created 25 levels of each type and then generated 8 versions of each level by applying vertical reflection and  $90^\circ$ ,  $180^\circ$ , and  $270^\circ$  rotations. In all levels, we repeat the ‘short-route’ intervention every step but repeat the ‘directional’ intervention only until the agent moves onto, or pushes the box off, the corresponding square.

We perform our interventions on the agent’s cell state at each layer. An intervention is considered successful if it causes the agent to solve the level in the desired suboptimal way. As a baseline, we intervene using representations from randomly initialized probes. For comparability, we scale random

{8}------------------------------------------------

![Figure 7: Three grid world screenshots showing an agent's plan. (a) Plan without intervention: the agent is at the start, and the plan is a simple path. (b) Intervention: the agent is at the start, and the plan is modified with a white arrow pointing down and white crosses at certain positions. (c) Plan with intervention: the agent is at the start, and the plan is modified with a white arrow pointing down and white crosses at certain positions.](d4e9f8f6bf5d7853ecae9c9633900af1_img.jpg)

Figure 7: Three grid world screenshots showing an agent's plan. (a) Plan without intervention: the agent is at the start, and the plan is a simple path. (b) Intervention: the agent is at the start, and the plan is modified with a white arrow pointing down and white crosses at certain positions. (c) Plan with intervention: the agent is at the start, and the plan is modified with a white arrow pointing down and white crosses at certain positions.

Figure 7: An Agent-Shortcut intervention and its effect on the agent’s plan as formulated in terms of  $C_A$ : (a) the agent’s plan after 4 steps *without* the intervention, (b) the initial state of the level and the intervention, and (c) the agent’s plan after 4 steps *with* the intervention. The ‘short-route’ intervention adds the representation of NEVER for  $C_A$  to positions with white crosses. The ‘directional’ intervention adds the representation of DOWN for  $C_A$  to the position with the white arrow.

![Figure 8: Three grid world screenshots showing an agent's plan. (a) Plan without intervention: the agent is at the start, and the plan is a simple path. (b) Intervention: the agent is at the start, and the plan is modified with a white arrow pointing right and white crosses at certain positions. (c) Plan with intervention: the agent is at the start, and the plan is modified with a white arrow pointing right and white crosses at certain positions.](c37fe03d7cad74ad675a0eb16aa43821_img.jpg)

Figure 8: Three grid world screenshots showing an agent's plan. (a) Plan without intervention: the agent is at the start, and the plan is a simple path. (b) Intervention: the agent is at the start, and the plan is modified with a white arrow pointing right and white crosses at certain positions. (c) Plan with intervention: the agent is at the start, and the plan is modified with a white arrow pointing right and white crosses at certain positions.

Figure 8: A Box-Shortcut intervention and its effect on the agent’s plan as formulated in terms of  $C_B$ : (a) the agent’s plan after 4 steps *without* the intervention, (b) the initial state of the level and the intervention, and (c) the agent’s plan after 4 steps *with* the intervention. The ‘short-route’ intervention adds the representation of NEVER for  $C_B$  to positions with white crosses. The ‘directional’ intervention adds the representation of RIGHT for  $C_B$  to the position with the white arrow.

probe representations so that the norms of both the random and trained probes are similar. Success rates are averaged over interventions performed with five independently trained or initialized probes.

Table 1 shows intervention success rates. At all layers, Agent-Shortcut interventions are successful. While the success rate of Box-Shortcut interventions is lower, it remains high relative to the baseline of interventions using random probes. These results indicate that the agent’s representations of  $C_A$  and  $C_B$  influence its behavior in the way that would be expected if it used them for planning. Figures 7 and 8 provide examples of the effect of interventions on the agent’s internal plans. These examples suggest the agent not only behaves differently following the interventions, but does so *due to forming a different plan*. We show more examples of interventions altering the agent’s internal plans in Appendix B.1. Appendix B.2 reports success rates when using an intervention scaling factor and varying the squares intervened on. Appendix B.3 reports success rates when intervening to encourage optimal behavior in levels which the agent by default cannot solve. These extra experiments further indicate that the agent’s representations of  $C_A$  and  $C_B$  influence its behavior as expected.

![Figure 9: A scatter plot showing the relationship between the percentage of extra levels solved (y-axis, 0 to 6) and the macro F1 score of probes (x-axis, 0.2 to 0.8). Two data series are plotted: Concept C_A (blue dots) and Concept C_B (orange dots). Both series show a positive correlation, with C_B generally achieving higher F1 scores and more extra levels solved than C_A.](e8ff6e66c77a8e96203c9f8db8f0986f_img.jpg)

Figure 9: A scatter plot showing the relationship between the percentage of extra levels solved (y-axis, 0 to 6) and the macro F1 score of probes (x-axis, 0.2 to 0.8). Two data series are plotted: Concept C\_A (blue dots) and Concept C\_B (orange dots). Both series show a positive correlation, with C\_B generally achieving higher F1 scores and more extra levels solved than C\_A.

Figure 9: The relationship between the percentage of extra levels, of medium difficulty, solved when an agent is given 5 steps to ‘think’, and macro F1 score of probes when predicting  $C_A$  (blue) and  $C_B$  (orange) from the agent’s final layer cell state. Each point corresponds to these quantities calculated for a single checkpoint.

### 6.2 INVESTIGATING THE EMERGENCE OF PLANNING DURING TRAINING

Finally, we show that the emergence of the agent’s representations of  $C_A$  and  $C_B$  during training coincides with the agent beginning to exhibit planning-like behavior. This indicates that the agent indeed uses its representations of  $C_A$  and  $C_B$  for planning. Specifically, we show the emergence of these representations coincides with the emergence of the agent’s ability to benefit from extra test-

{9}------------------------------------------------

time compute (Guez et al., 2019; Taufeeque et al., 2024). In particular, we collect checkpoints every 1 million transitions for the first 50 million transitions of training. For every checkpoint, we measure two quantities: (i) the macro F1 score of 1x1 probes trained to decode the concepts  $C_A$  and  $C_B$  given the agent’s cell state (following the procedure described in Section 4.1), and (ii) the number of additional levels out of 1000 medium-difficulty levels from the Boxoban dataset (Guez et al., 2018a) the agent can solve when given extra test-time compute by forcing the agent to remain stationary for the first 5 steps of an episode. Figure 9 plots these quantities against each other and shows a strong correlation between them. This implies the agent only reliably begins to exhibit planning-like behavior – benefiting from extra test-time compute – once its final layer representations of  $C_A$  and  $C_B$  are sufficiently formed. Appendix C.3 shows that this holds for its representations of  $C_A$  and  $C_B$  at all layers. Appendix C.4 shows the agent begins to perform better with extra compute at a similar point in training as to when it can use this compute to refine its plans.

## 7 ADDITIONAL RESULTS

In the Appendix, we include interesting results that we lacked space to include in the main text. Appendices F provides evidence of DRC agents planning both without internal ticks, and with additional internal ticks. Appendix H provides evidence of a DRC agents planning in a different environment: Mini PacMan. Finally, Appendix G provides evidence of a ResNet (He et al., 2016) agent planning in Sokoban. However, the question of whether a generic agent can learn to plan in a generic environment remains unanswered.

## 8 RELATED WORK

Past work has investigated concept representations learned by game-playing agents (Schut et al., 2023; McGrath et al., 2022; Hammersborg & Strümke, 2022; 2023; Lovering et al., 2022; Mini et al., 2023) and language models (Li et al., 2023; Nanda et al., 2023; Karvonen, 2024; Ivanitskiy et al., 2024). While past work has focused primarily on whether networks internally represent specific concepts, we study concept representations for the broader purpose of determining if an agent possesses a capability - planning. An exception is work by Jenner et al. (2024), which finds evidence of look-ahead in a chess-playing agent, but does not investigate a wider capacity to ‘plan’.

Concept-based interpretability is not the only approach to interpreting agents. An alternative is attribution-based interpretability. This involves determining – usually via saliency maps – which features in an agent’s observation influence its behavior (Weitkamp et al., 2019; Iyer et al., 2018; Puri et al., 2020; Greydanus et al., 2018; Hilton et al., 2020). Attribution-based methods were not used here as they can depend on subjective interpretation (Atrey et al., 2020). Another approach, example-based interpretability, explains agent behavior by providing examples of illustrative trajectories or transitions (Rupprecht et al., 2020; Sequeira & Gervasio, 2020; Deshmukh et al., 2023; Zahavy et al., 2016). Due to not studying model internals, example-based methods were ill-suited for this paper.

Finally, this paper contributes to recent work investigating the emergence of reasoning capabilities in neural networks (Wei et al., 2022; Kojima et al., 2022; Lehnert et al., 2024; Nye et al., 2021; Wang et al., 2024). However, unlike this paper in which we provide evidence of an agent *internally* performing planning, most work thus far has focused on providing *behavioral* evidence of reasoning. An exception to this is work by Brinkmann et al. (2024) in which an algorithm learned by a transformer trained on a simple symbolic reasoning task is reverse-engineered. However, Brinkmann et al. (2024) focus on a much simpler form of reasoning than planning as considered in this paper.

## 9 FUTURE WORK

In this paper, we proposed a methodology for investigating model-free planning and used it to provide the first non-behavioral evidence of learned planning in a model-free agent. Future work may extend our investigation to other RL agents, and other environments. In particular, it would be helpful to better understand the role of different training factors, e.g., model architecture, environment dynamics in the emergence of planning.

 Rest of paper (reference and Appendix) is removed.