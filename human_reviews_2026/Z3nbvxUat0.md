# Planning with Generative Cognitive Maps

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 6, 2, 2

## Abstract
Planning relies on cognitive maps  -- models that encode world structure given cognitive resource constraints. The problem of learning functional cognitive maps is shared by humans, animals and machines. However, we still lack a clear understanding of how people represent maps for planning, particularly when the goal is to support cost-efficient plans. We take inspiration from theory of compositional mental representations in cognitive science to propose GenPlan: a cognitively-grounded computational framework that models redundant structure in maps and saves planning cost through policy reuse. Our framework integrates (1) a Generative Map Module that infers generative compositional structure and (2) a Structure-Based Planner that exploits structural redundancies to reduce planning costs. We show that our framework closely aligns with human behavior, suggesting that people approximate planning by piecewise policies conditioned on world structure. We also show that our approach reduces the computational cost of planning while producing good-enough plans, and contribute a proof-of-concept implementation demonstrating how to build these principles into a working system.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies structured planning, with a focus on the synthesis of compressed cognitive maps that retain only the most planning-salient information about an environment. The work's goal is twofold: first, to develop an explanatory algorithmic account of how people might be forming simplified cognitive maps and planning with them, and second, to advocate for this approach as a generic algorithmic template for structure based planning in practice, inspired by how people plan. Maze Search Task is used as the central planning domain, a simple partially observable grid game that has been of recent interest to similar work in which the player must exit the maze. The core of the approach is a two-piece algorithmic recipe: program induction over possible cognitive maps is used to hand off a transition function to a structured planner. Together, these pieces constitute "Gen-POMCP", a generative planner. This approach is contrasted with standard POMCP (without the program induction over maps), and a variant of POMCP that incorporates a cognitive constraint (Depth Plan). A human study is then conducted to draw out the explanatory power of these different hypotheses. Behavioral metrics are deployed to compare DepthPlan with Gen-POMCP in the user study. The primary finding is that Gen-POMCP is a better predictor of human behavior than DepthPlan.

### Strengths
- The question to better understand how people abstract and plan is a deep and important one. The ingredients in the proposed theory are intuitive, and resonate with other similar methods in the literature.
- The paper was easy to read, and the core of the method and experimental design are intuitive (with a minor exception regarding the details of the method, I detail this below).
- Inclusion of a human study, and the initial results contrasting Gen-POMCP with DepthPlan, are compelling.

### Weaknesses
W1. I am broadly sympathetic to the research agenda to understand how people plan. However, one stated contribution of the work is "...an algorithmic account of how structure-based planning can be implemented in practice". There are large bodies of literature within the ICAPS, AI, and RL communities dedicated to this question. As such, I believe the present work, to be treated as a novel algorithmic account of structure-based planning, needs to be contrasted using the same language and analysis that are used to evaluate planning algorithms (or, a convincing argument should be made about why different analysis tools are needed). For example, recent work by Wen et al. explicitly proves under what conditions structured planning can be efficient (Propositions 1-3). Or, early work on planning explores how structure changes the planning problem, as in ABSTRIPs by Sacerdoti (1974). Or, more recently, Jinnai et al. (2018) examine the computational hardness of finding structures that can make planning as easy as possible, and prove both that this problem is NP-Hard in general, but that efficient approximation algorithms can exist (along with an average-case result follow-up by Ivanov et al.). In this sense, there are many known results for understanding when and why we can design efficient algorithms for structure-based planning. I struggle to see the novelty here from an AI perspective, especially since (1) the basic mechanisms used to scrutinize planning algorithms are not engaged with (computational hardness, robustness, and so on), and (2) the work is not situated conceptually or experimentally relative to the vast body of literature that studies structure-based planning algorithms. As further examples, consider work by Anand et al. (2016) that develop "on the go" abstractions to facilitate tree-based search, or Hostetler et al. (2014), who make use of state aggregation to simplify MCTS, or seminal work by Littman (1997) that proves various hardness results about different kinds of planning. Or, we can look at hierarchical structure, as in SHOP (Nau, 1999) or HTNs (Erol, 1997). 

W2. Much of the terminology and notation are under-developed or lacking in rigor and depth. For instance, planning is described as search through a tree---this is one plausible account of planning, though is still only part of the story. See, for instance, the work by Littman, who gives precise formal definitions to certain variants of planning problems. Similarly, POMDPs are defined using a standard definition of Belief MDPs, which is non-standard; the introduced object is an MDP over belief state, rather than a POMDP (with Kaelbling et al. showing how to connect the two). See the standard POMDP definition in the cited Kaelbling et al. paper in Section 3.1. I can imagine the Belief MDP definition is introduced for a reason, but it would be more effective to introduce POMDPs in full detail, then describe their translation into a Markov process over belief state.

W3. The primary contribution of the work is to offer an explanatory algorithmic account about how people might be carrying out structured planning in POMDPs. The evidence offered to support the conclusion that Gen-POMCP is a valid such explanatory account is relatively thin; a variety of competing accounts are described in Section 4, but the model predictions from these alternative theories are not contrasted with. For instance, the work by Ho et al. is described as showing "that people form cognitive maps that facilitate planning ... by selectively representing only the goal-relevant parts of the map". This sounds extremely similar to the proposal in this paper: I did not understand where the two theories come apart, and where one might have higher explanatory power than the other.

W4. Lastly, the core method is not described in adequate detail. Pseudo-code will add clarity to what precisely is being proposed by Gen-POMCP.

To summarise: I perceive the stated contributed as spread across both (1) developing new structure-based planning algorithms inspired by compositional accounts of how people represent cognitive maps, and (2) a proposed explanatory account of how people form cognitive maps and plan with them. The former is missing contact with the rigor, literature, and core analysis tools that evaluate approaches to (structure-based) planning, and the latter is missing direct comparison to other models of how people form simplified representations and plan. In light of this, I take the core claims to not yet be well-supported by evidence, and so I recommend rejection at this time. I am happy to discuss during rebuttal if I have misunderstood, or if there are other factors that can remedy these issues.

Typos and Writing Suggestions:

- " Solving there problems optimally " --> maybe,  "Solving these problems optimally "?
- "In the the remainder of this section..."

References:
- Anand, A., Noothigattu, R., & Singla, P. (2016, March). Oga-uct: On-the-go abstractions in uct. In Proceedings of the International Conference on Automated Planning and Scheduling (Vol. 26, pp. 29-37).
- Hostetler, J., Fern, A., & Dietterich, T. (2014, June). State aggregation in Monte Carlo tree search. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 28, No. 1).
- Littman, M. L. (1997). Probabilistic propositional planning: Representations and complexity. AAAI/IAAI, 748-754.
- Ivanov, A., Bagaria, A., & Konidaris, G. (2025, April). Discovering Options That Minimize Average Planning Time. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 39, No. 17, pp. 17573-17581).
- Jinnai, Y., Abel, D., Hershkowitz, D., Littman, M., & Konidaris, G. (2019, May). Finding options that minimize planning time. In International Conference on Machine Learning (pp. 3120-3129). PMLR.
- Nau, D., Cao, Y., Lotem, A., & Munoz-Avila, H. (1999, July). SHOP: Simple hierarchical ordered planner. In Proceedings of the 16th international joint conference on Artificial intelligence-Volume 2 (pp. 968-973).
- Erol, K. (1995). Hierarchical task network planning: formalization, analysis, and implementation. University of Maryland, College Park.
- Sacerdoti, E. D. (1974). Planning in a hierarchy of abstraction spaces. Artificial intelligence, 5(2), 115-135.
- Wen, Z., Precup, D., Ibrahimi, M., Barreto, A., Van Roy, B., & Singh, S. (2020). On efficiency in hierarchical reinforcement learning. Advances in Neural Information Processing Systems, 33, 6708-6718.

### Questions
Q1: What are the main criteria by which we should be evaluating a planning algorithm, and how does Gen-POMCP fair relative to these criteria? How does it situate relative to other algorithms---for instance, in the planning community, to things like Hierarchical Task Networks (HTNs) or SHOP or ABSTRIPS? Or, alternatively, is there a reason we should not be contrasting the present approach to other planning algorithms in this way?

Q2: The basic mechanism described by line 281 sounds similar to prior methods for decomposing planning into subproblems (as in HTNs). Can you spell out in more detail what is happening, and comment on how it might differ from standard hierarchical planning?

And, one minor question:

Q1. It is unclear what role partial observability plays here. Planning in a POMDP vs. an MDP are quite different---how central is partial observability to the approach?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this paper, Planning with Generative Cognitive Maps, the authors designed GenPlan, a computational framework that represents a task environment as a generative cognitive map and plans by reusing policies on repeating structures, to model human navigation behavior. GenPlan consists of a Generative Map Module (GMM) that infers compositional generative representation of the environment structure through a programmatic map generated by LLM (GPT4), and a Structure-Based Planner (SBP) that leverages the learned structure to perform navigation planning with policy reuse. 

Empirically, GenPlan reproduces planning behavior more similar to that of human participants compared with naive POMCP (giving rise to a global optimal policy) and constrained POMCP (with discounts to limit planning depth). Moreover, GenPlan significantly reduces computational planning cost: in simulations on larger structured mazes, it required far fewer rollouts to fully explore the environment than a standard planner, while still finding the goal with only a modest increase in path length. Overall, the paper’s contributions include a novel generative model of cognitive maps (GenPlan), an integrated planning algorithm (Gen-POMCP) that leverages that model, and experimental evidence suggesting that humans similarly exploit compositional structure to plan efficiently.

### Strengths
1. The work introduces a novel framework by uniting generative modeling and planning in the context of cognitive maps, and provides scientifically significant results for cognitive science. What's particularly novel about the approach is the use of a Large Language Model (LLM) to induce a programmatic map for environment representation. This allows tractable inference of structure where previous methods would require potentially intractable enumeration. The framework (GenPlan) is thus novel in how it integrates an LLM-induced programmatic structural prior with a planning algorithm – a creative synthesis of ideas from program induction, hierarchical planning, and cognitive psychology. The cognitive experiment shows a higher human similarity for GenPlan model, indicating that human deviations from optimal policies are, at least in part, due to approximating planning by piecewise policies conditioned on structure for policy reuse.
2. The paper is technically solid and provides validation on its base claims. The GenPlan framework is described with a clear formal basis, with the planning problem as a POMDP, and the model is compared against relevant baseline cognitive models, including DepthPlan and naive POMCP. The experiments are relatively well designed for model and human subjects to test the behavioral hypotheses and evaluate performance.

### Weaknesses
1. While environments of many different shapes are used for the experiments, all the environments are compositionally simple, consisting of the same top-level grid maze structure with identical sub-mazes per environment. This limits the generalizability of the hypothesis due to the limited scope of the environmental composition (2-level maze with identical substructure).
2. GenPlan's GMM steps involves using GPT4 over the entire environment for producing a programmatic map generator. This requires global knowledge about the environment as prior knowledge. In the experimental description, it wasn't clear that the human subjects were provided with the similar prior knowledge. This might hinder the validity of the comparison.

### Questions
1. Could the authors clarify how exactly the LLM is used to infer the generative map structure? How is the LLM prompted? 
2. What prior knowledge do the human subjects have for the experiments? For example, are they informed that the maze has a similar substructure? 
3. There are limited performance evaluation and ablation studies for evaluating GenPlan comprehensively. How do the parameters of GenPlan affect the results?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes GenPlan, a cognitively grounded framework for planning in structured, partially observable environments. It has two parts: (i) a Generative Map Module (GMM) that infers a small set of repeated “structural units” and a program that reconstructs the map using transformations (rotations/reflections), guided by a likelihood balancing reconstruction accuracy, description length (MDL), and unit informativeness; and (ii) a Structure-Based Planner (SBP) that plans within each inferred unit (via POMCP/MCTS) and between units using a heuristic that prefers unit-boundary cells with shorter average Manhattan distance to remaining unseen cells. 
GenPlan’s predictions align with human action choices better than a depth-limited global planner; computationally, GenPlan searches large, structured mazes with far fewer rollouts than a structure-naïve POMCP while producing good-enough paths.

### Strengths
1. Inspired by resource rationality and compositionality in cognitive science, the work proposes generative, program-based representation of cognitive maps and uses it directly for policy reuse which can both explain human behavior and produce efficient plan. 
2. The paper is clearly written and structured.

### Weaknesses
1. Though human study is provided and considered, the experiment maps are relatively toy. 
    - This is actually a simplified POMDP environment. 1) Walls are assumed known from the start; only the exit is hidden, which removes core POMDP uncertainty about topology and sensing. 2) How to handle non-deterministic maps? 
    - Although a posterior over generative maps is defined, the implementation selects only the MAP reconstruction for planning, avoiding online uncertainty resolution. How does it work if the model has to plan over a distribution of maps? 
    - All behavioral mazes are structured with 2–20 repeating units covering 80-100% of the layout, which is exactly the condition GenPlan is built to exploit. It would be much better if you consider working on more realistic scenarios, or even real-world maps. 
2. It seems like the hierarchial RL and options are completely omitted, both in discussion the connection with GenPlan, and baseline models.

### Questions
See weakness.

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
4

### Summary
This paper introduces GenPlan, a computational framework that explains how humans plan efficiently in structured environments by using generative cognitive maps that capture compositional structure through repeated patterns and symmetries. The framework consists of two key components: a Generative Map Module (GMM) that discovers programmatic representations of environments using LLM-based program synthesis, and a Structure-Based Planner (SBP) that reduces planning costs by reusing policies across structural units rather than computing global optimal solutions. The authors validate their approach through a behavioral experiment with 30 participants in a Maze Search Task, demonstrating that human planning aligns significantly better with the structure-exploiting Gen-POMCP model than with the previous state-of-the-art depth-limited planning model, and they provide simulation evidence that their approach achieves substantially lower computational cost than naive planning while maintaining good solution quality.

### Strengths
The paper makes both a scientific contribution by showing that human deviations from optimal policies reflect structure-based planning with policy reuse, and an engineering contribution by demonstrating how to implement these principles in a working system. Across all environments, Gen-POMCP predicts human behavior significantly better than DepthPlan, the previous state-of-the-art model, with people showing high consistency with structure-based planning across all environments and individuals.

### Weaknesses
Foremost, experiment recruited 30 participants from Prolific in a single maze search domain that, despite being structured, remains relatively artificial compared to real-world planning contexts. The framework makes simplifying assumptions, including using the most likely generative map rather than maintaining a distribution over maps, and assuming stable population-level weights for reconstruction accuracy and planning costs.
Also the approach's heavy reliance on LLM-based program synthesis introduces potential biases: the structural discovery depends on how the LLM is prompted and the particular priors it has learned, yet there is limited analysis of robustness to these design choices.
Lastly, actually there are line of concurrent works trying to do offline planning with the power of LLMs. They are similar to your work trying to utilize the power of LLM to plan before action, hope you can cite them and highlight your novelty.
Shinn et al. Reflexion: Language Agents with Verbal Reinforcement Learning
Zelikman et al. Parsel: Algorithmic Reasoning with Language Models by Composing Decompositions
Kim et al. How language models extrapolate outside the training data: A case study in Textualized Gridworld
Yang et al. Chain of Thought Imitation with Procedure Cloning
Yao et al. ReAct: Synergizing Reasoning and Acting in Language Models
Gu et al. Is Your LLM Secretly a World Model of the Internet? Model-Based Planning for Web Agents

### Questions
How were the free parameters in the likelihood function chosen, and how sensitive are the main results to these choices? The paper notes that planning varies between individuals and proposes that this variability arises from different representations of the same map depending on available cognitive resources, but the implementation assumes fixed population-level weights. How would the framework dynamically adjust these weights to model flexible cognitive resource allocation within individuals? Can the principles extend beyond spatial domains to other planning contexts like recipe planning or multi-step problem solving, and if so, what would constitute the "structural units"? Regarding the LLM-based approach: how much do results depend on using GPT4 specifically, and would different LLMs produce meaningfully different structural decompositions?

### Soundness
2

### Presentation
2

### Contribution
2
