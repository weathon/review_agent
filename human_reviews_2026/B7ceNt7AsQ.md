# Experience-Driven Exploration for Efficient API-Free AI Agents

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 2, 6

## Abstract
Most existing software lacks accessible Application Programming Interfaces (APIs), requiring agents to operate solely through pixel-based Graphical User Interfaces (GUIs). In this API-free setting, large language model (LLM)-based agents face severe efficiency bottlenecks: limited to local visual experiences, they make myopic decisions and rely on inefficient trial-and-error, hindering both skill acquisition and long-term planning. To address these challenges, we propose KG-Agent, an experience-driven learning framework that structures an agent's raw pixel-level interactions into a persistent State-Action Knowledge Graph (SA-KG). KG-Agent overcomes inefficient exploration by linking functionally similar but visually distinct GUI states, forming a rich neighborhood of experience that enables the agent to generalize from a diverse set of historical strategies. To support long-horizon reasoning, we design a hybrid intrinsic reward mechanism based on the graph topology, combining a state value reward for exploiting known high-value pathways with a novelty reward that encourages targeted exploration. This approach decouples strategic planning from pure discovery, allowing the agent to effectively value setup actions with delayed gratification. We evaluate KG-Agent in two complex, open-ended GUI-based decision-making environments (Civilization V and Slay the Spire), demonstrating significant improvements in exploration efficiency and strategic depth over the state-of-the-art methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper focuses on gaming agents, and proposes a framework based on GUI-agents to improve their performance on games such as Civilization 5 and Slay the Spire. Specifically, the authors propose KG-agent, which utilizes a knowledge-graph like skill library to tackle long horizon games. On a high level, the proposed approach uses existing/related skills from the knowledge graph when given a new game state (represented as a screenshot) if similar states can be found, or uses the VLM model directly as GUI agents to generate new actions and add new skills to the graph. Experimental results on two games shows the effectiveness of the proposed method compared to its prior work (BottomUp).

### Strengths
- Introducing a graph structure for skill libraries is intuitive and novel.

- Experiments on two different games (Civilization V and Slay the Spire) shows the effectiveness of the proposed method, improving both game score and token cost compared to its prior work (BottomUp).

- The authors conducted numerous analysis of the method, including visualizing the evolution of the skills graph and an ablation study on the proposed method.

### Weaknesses
1. The notion of using a skill library for GUI related tasks highly resembles existing work such as [1] and [2]. However, there is no comparison against these simpler skill-library based methods.

2. The proposed method includes a lot of heuristics and modules (four tables for procedure memory, a state-action KG, a reasoning module, along with 11 equations). It was challenging to understand the real contribution of each of these items, especially when the ablation study in Table 1 only ablated two aspects: rewards (covering equation 9-11), and similarity edges in the KG (covering equation 2-5). A lot of other design choices such as having four tables in the procedure memory, important of the reasoning module, as well as the effect of various hyperparameters/equation choices were omitted in the paper.


---

References

[1] Wang, Guanzhi, et al. "Voyager: An open-ended embodied agent with large language models." arXiv preprint arXiv:2305.16291 (2023).

[2] Zheng, Boyuan, et al. "Skillweaver: Web agents can self-improve by discovering and honing skills." arXiv preprint arXiv:2504.07079 (2025).

### Questions
- Section 3.1 introduces a method using SAM to help VLMs to generate executable actions on an GUI image. I wonder if the authors have tried/are aware of VLMs such as OpenAI's operator and Claude's computer-use, which can directly output low-level actions on an GUI image without using additional models such as SAM.

- L424 and Table 3 shows that removing similarity edges from the graph impairs long-term progression/planning. What is the intuitive reason behind this? To me, it seems to indicate that the backbone VLMs themselves are bad at decision-making, and programmatic approach such as reusing skills from similar states is more robust than querying the VLMs?

- How are the hyperparameters such as theta, alpha, $C_0$, etc selected on L376-377? These values seems very specific.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes KG-Agent, an API-free GUI agent that structures pixel-level interaction history into a State–Action Knowledge Graph (SA-KG). Nodes are CLIP-encoded GUI states; edges are either (i) similarity edges linking functionally similar but visually different states or (ii) skill edges recording successful multi-step skills. A hybrid intrinsic reward combines (a) a potential-based state-value term computed from SA-KG out-edge weights and (b) a novelty bonus for first-seen states. The agent uses SAM for interactable-object discovery, template matching for grounding, and a VLM-driven loop for skill invocation/augmentation/refinement with a UCT-style selector. Experiments on Slay the Spire and Civilization V report higher in-game progression and lower token cost than baselines, plus ablations removing similarity edges and reward terms.

### Strengths
1.The paper targets an important and difficult setting where agents must learn from pixels with mouse/keyboard only.

2.Converting episodic pixel experience into a connected SA-KG is conceptually clean and practically useful for retrieval and reuse; the “neighborhood of experience” nicely mitigates myopic nearest-neighbor retrieval.

### Weaknesses
1.The paper argues that turning experience into a state–action knowledge graph (SA-KG) boosts exploration and skill reuse, but it doesn’t really explain why this should beat simpler memories (like episodic lookup or nearest-neighbor with a bit of temporal logic) or a lightweight model-based rollout. Some clean, controlled head-to-head tests would make that claim much more convincing.

2.The high-level story of how the SA-KG, edges, and hybrid reward work is clear enough, but key nuts-and-bolts are missing: what exactly is the “visual change” metric, how are node merges decided, how is edge “fitness” updated over time, what’s the failure/termination policy, and what thresholds or schedules does the UCT-style selector actually use?

3.The current ablations (dropping similarity edges or intrinsic-reward terms) don’t really hit the core of the method. What’s needed is module-wise ablation: swap the graph memory for strong non-graph baselines under the same perception/grounding, vary graph topology and merge/similarity thresholds, and check sensitivity to graph size/pruning and encoder choice.

4.The system may be more complicated than it needs to be. Using a separate, frozen CLIP for state similarity instead of the planner’s own VLM embeddings risks a representational mismatch and extra moving parts. On the action side, leaning on SAM and template matching for interactable discovery runs counter to the recent trend toward direct VLM grounding—i suggest the author should show it’s actually safer, cheaper, or more reliable at comparable quality.

### Questions
See Weaknesses Part

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces KG-Agent, a framework for API-free agents that aims to address inefficient exploration and short-sighted planning. The core contribution is a State-Action Knowledge Graph (SA-KG) that structures experience by linking functionally similar states (similarity edges) and successful action sequences (skill edges). This graph underpins a hybrid intrinsic reward system designed to promote strategic, long-horizon behavior.

### Strengths
The SA-KG is an interesting and promising data structure for agent memory. The idea of explicitly representing both state similarity and action transitions in a single graph is a notable contribution that moves beyond simple episodic memory retrieval.

The authors correctly identify two of the most critical challenges facing contemporary API-free agents: sample inefficiency ("experience siloing") and the difficulty of credit assignment for long-horizon tasks (myopia). The proposed solutions, while imperfect, are directly motivated by these well-understood problems.

Evaluating on Civilization V and Slay the Spire is commendable. These environments are significantly more challenging than typical web-browsing or application-control tasks and serve as a strong testbed for strategic reasoning.

### Weaknesses
While the core idea is interesting, the proposed architecture is a complex and brittle assembly of disparate components. The system stitches together a VLM for reasoning, CLIP for state embedding, SAM for UI segmentation, and template matching for action grounding, layering them beneath a custom knowledge graph, a four-part ad-hoc reward function, a two-stage skill invocation policy, and a VLM-based refinement loop. This design makes the agent fundamentally dependent on the quality of its frozen, pre-trained models and reliant on brittle heuristics, such as similarity thresholds and arbitrary reward formulas, for all its "learning" decisions. Consequently, the system is prone to unbounded growth in memory and computation, as its knowledge is simply the entire, ever-expanding graph and skill library.

The paper's central claim of developing a generalizable framework for API-free agents is not supported by its highly circumscribed experiments. The authors have selected turn-based games, a forgiving domain that conveniently masks the architecture's inherent limitations by not requiring real-time reasoning and featuring a mostly static UI with little dynamism. This narrow validation on a "home turf" benchmark does not provide credible evidence for the method's applicability to the broader, more challenging landscape of general computer interaction.

The paper's description of the skill acquisition process is not clearly explain in the paper. The proposed mechanism of incremental search over action sequences is combinatorially explosive, and the "aggressive pruning" strategy intended to manage this complexity is not defined with sufficient detail to be reproducible. Furthermore, the framework outsources the critical step of semantic abstraction to a VLM, which retroactively labels a sequence of operations rather than enabling the agent to learn their meaning internally. This issue is compounded by a significant inconsistency: the main text presents skill refinement as a generative process for creating novel behaviors, while the corresponding prompt in the appendix describes a far simpler function of de-duplicating existing skills.

### Questions
The knowledge graph grows with every new state and skill. Table 2 shows the number of nodes and edges increasing over rounds. What are the computational and memory implications of this growth over much longer runs (e.g., thousands of steps or dozens of episodes)? Is there a risk of the graph becoming too large to query efficiently? The authors might consider discussing potential pruning, summarization, or hierarchical abstraction strategies more in details.

The framework introduces several important hyperparameters: the merging and similarity thresholds ($θ_{merge}$, $θ_{simi}$), the reward balancing factor $\alpha$, the fitness sensitivity $C_\sigma$, and the UCT exploration constant $C$. The paper states these are held constant, which is good for reproducibility, but a brief discussion on how these were selected and how sensitive the agent's performance is to them would be valuable.

The $R_{progress}$ and $R_{semantics}$ rewards rely on a VLM's judgment. While this is a common technique, VLM evaluations can be noisy or misaligned with the true game objectives. Could the authors comment on the stability of these reward components? How much of the agent's success is dependent on the quality of these VLM-based evaluations versus the more objective graph-based rewards ($R_{state}$, $R_{novel}$)?

The paper mentions that skills are constructed through incremental increases in sequence length. For a game like Civilization V, a meaningful "skill" might span multiple turns (e.g., "move worker, build improvement"). How does the current skill definition $(\sigma = (a_1,...,a_k))$ and augmentation process handle these temporally extended, multi-turn skills? More detail on the `prompt_augment` and refinement loop would be beneficial to understand how the agent moves from atomic clicks to strategically meaningful behaviors.

The authors make a point of forgoing OCR to enhance generality. While this is a valid design choice, it could be a limitation in GUIs where critical information is presented as non-trivial small text that may be lost in the VLM input if they are resized for computation reasons. Do you have an intuition on how much the VLM is taking from its knowledge of the games and how much it is actually "observing"?

The main text (Lines 305-307) describes "skill refinement" as a generative process for creating novel skills when the agent is stuck. However, the prompt_refine provided in the Appendix (Figure 6c) is clearly designed for de-duplicating and clustering existing redundant skills. Could you please clarify the true purpose and implementation of this mechanism? Which of these two distinct functions does the agent actually perform?

### Soundness
1

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
4

### Summary
The paper builds on top of the Bottom-Up Agent framework, extending it with several new components that improve exploration and performance. It introduces a Knowledge Graph to organize experiences into a structured memory of states and actions. A hybrid intrinsic reward combines state-value estimation with novelty to balance exploration and exploitation, while enhanced visual processing (using SAM for segmentation and CLIP-like embeddings) helps the agent generalize across GUI variations. Tested on Slay the Spire and Civilization V, KG-Agent achieves better progress than Bottom-Up and other simpler baselines. The authors demonstrate how graph-structured experience representation and adaptive intrinsic rewards can make pixel-only agents more capable.

### Strengths
1. The motivation is strong. Tackling API-free GUI agents (pixel-only) is genuinely difficult, and the proposed SA-KG offers a clever middle ground between brute-force exploration and explicit symbolic reasoning. The method introduces a structured way to store and reuse experience, improving coherence in long-horizon behavior.
2. The *experience neighborhood* idea, linking functionally similar but visually distinct GUI states, effectively tackles a key weakness in vision-language control, where small visual shifts often break generalization. By grouping semantically equivalent states, the agent can transfer learned behaviors across different interface appearances.
3. The hybrid intrinsic reward, combining state-value estimation with novelty, is a neat and well-motivated idea that aligns with the goal of long-horizon planning. It encourages both exploration of new states and reinforcement of meaningful progress, leading to more stable and directed learning behavior.
4. The authors select the same two environments for evaluation as in the Bottom-Up paper and surpass this baseline with their improved method.
5. The SA-KG adds a layer of interpretability, allowing the agent’s exploration and decision patterns to be visualized and analyzed, unlike the more opaque skill library used in Bottom-Up Agent.
6. The ablations are quite useful, showing how the method’s core components individually contribute to performance.

### Weaknesses
1. **Generalizability**. I am not fully convinced how well KG-Agent could (out-of-the-box) generalizable to other application.  
    a) The method relies heavily on **SAM** and **CLIP**, but both can be unreliable for GUI perception. SAM might missegment small or dynamic interface elements. For instance, segmenting nearby buttons as one region. CLIP, meanwhile, might blur fine-grained distinctions between visually similar yet functionally different states, while its feature space is directly used to infer state value and progress. The authors could investigate whether there are any failure cases caused by these components and how frequently they occur.  
    b) **Hyperparameter sensitivity.** KG-Agent introduces many hyperparameters with no reported sensitivity analysis, and it is unclear whether the same configuration was used across environments. I am unsure how much manual tuning is required for each new domain. Key hyperparameters include:  
        i. the coefficient balancing state-value and novelty rewards  
        ii. the criterion for increasing skill length  
        iii. cosine-similarity thresholds for node merging and edge creation  
        iv. constants controlling fitness sensitivity and greediness weighting  
        v. the number of skills sampled per execution attempt  
        vi. the temperature for skill selection    
    c) **Scalability**. In other environments/applications, executing a skill may require many atomic actions. As skill length grows, the MCTS-based search becomes combinatorially expensive, and SAM/CLIP must be invoked repeatedly, compounding the computational cost. Moreover, certain games naturally introduce new interface states and objects as gameplay progresses, causing the knowledge graph to also expand continuously. Without mechanisms for pruning or abstraction, the graph could grow unbounded, making retrieval and planning increasingly slow. It’s unclear how well the method scales under long-term play or whether performance degrades as the graph becomes denser and more computationally heavy to traverse.
2. The authors do not compare KG-Agent with **CRADLE**, which is odd given that CRADLE is the state-of-the-art framework for the same API-free control problem.
3. KG-Agent leans heavily on the **Bottom-Up Agent** framework but doesn’t give it nearly enough credit. A lot of the design choices, such as using SAM for segmentation, relying on implicit visual-difference rewards, the same Civilization V and Slay the Spire setup, the same evaluation metrics, and even the “zero-prior” claim, are straight out of that work. Even if the same group authored Bottom-Up Agent, that doesn’t excuse quietly absorbing its design without references. Reading the paper, it is hard to tell what is novel and what is repackaged. The authors should be more upfront about which components are inherited and what improvements their work actually contributes, rather than blurring the line between re-implementation and innovation.
4. **Evaluation**.  
    a) The authors report evaluating a mere **three episodes** per environment. This leads to very high randomness in the results. After which, it is not clear whether they report results from the single best run or averages across the three runs. There is also no reporting of variance or confidence intervals, so we don’t know how consistent the KG-Agent really is.  
    b) Taking **Slay the Spire** as an example, which consists of 51 floors, the KG-Agent reached the 16th, as seen from Table 1 and Figure 8. This might not be very impressive, as in the lower floors, the player only has a few different types of cards to choose from, so there are not many strategic considerations to be had. Even if playing cards at random, the player could traverse the initial 10 floors, since the game is not very punishing on the lower floors. If we were to simulate random gameplay thousands of times, it is likely that in some instances the random agent gets very lucky and reaches even beyond floor 16. Of course, it’s important to note that the random agent would not act with a keyboard and mouse, but a far simpler representation. Sure, from the results we see that the KG-Agent learns to play the game, but not how **well** it plays the game. It would be interesting to include this random agent with discrete actions (not keyboard-mouse) as an extra type of baseline.

### Typos

1. Figure 1. - Trail & Error
2. Line 472 -  proposed KG-Agent to organizes

### Questions
1. Is it possible to evaluate CRADLE on Civ V and Slay the Spire? Is KG-Agent better/faster/cheaper/more sample efficient than CRADLE?
2. Were the same hyperparameters used for both games? I only see the mention of “*The same agent **architecture** is deployed in both…*”.
3. When is k increased from evaluating n-step actions to (n+1)-step actions? When all combinations are exhausted?
4. How well does KG-Agent scale with skills with a long sequence of atomic actions?

### Soundness
3

### Presentation
3

### Contribution
3
