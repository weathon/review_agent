# CausalPlan: Empowering Efficient LLM Multi-Agent Collaboration Through Causality-Driven Planning

- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Large language model (LLM) agents often generate causally invalid plans in collaborative tasks due to their reliance on surface-level correlations rather than grounded causal reasoning. This limitation undermines their performance in terms of coordination and planning in dynamic environments. We address this challenge with CausalPlan, a framework that integrates explicit structural causal reasoning into the LLM planning process. At the core of CausalPlan is the Structural Causal Action (SCA) model, which learns a causal graph from agent trajectories to capture how prior actions and current environment states influence future decisions. This model is then used to inform the planning process, shaping proposed LLM-generated plans through causal scoring, reweighting, and fallback to grounded alternatives when needed. By embedding this causal knowledge directly into the decision loop, CausalPlan constrains planning to intervention-consistent behaviors without requiring fine-tuning. We evaluated CausalPlan on the Overcooked-AI benchmark across five multi-agent coordination tasks and four LLMs of varying sizes: Gemma-7B, Llama-8B, Qwen-14B and Llama-70B. Experimental results show that CausalPlan consistently reduces invalid actions and improves collaboration in both AI-AI and human-AI settings, outperforming strong reinforcement learning baselines. Our findings highlight the value of causality-driven planning for deploying efficient, interpretable, and generalisable multi-agent LLM systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces CausalPlan, a planner for two-agent collaborative tasks that reduces invalid actions by learning and exploiting a sparse policy structure over decisions. Given a set of trajectories, each timestep is factorized into binary features (agent states and environment state) and a one-hot previous action of the agent being controlled. Then a per-action head is trained with a sparsity mask using NLL (alternating masks/weights), indicating how much each input feature is used when choosing that action. The masks form a matrix whose entries reveal the propensity of choosing an input feature when choosing a decision (action). At test time, the agent a) prompts an LLM to propose candidate high-level actions, b) the actions are pruned based on feasibility using external rules/grounding, and c) each remaining candidate is scored by summing the mask weights from the currently active features. The score is blended with LLM scores  via a convex combination. If there are no valid actions, a fallback regime is used where the chosen action is the top-scoring action from the learned matrix. Empirically, this cuts invalid actions and improves cooperative rewards across layouts and LLM backbones.

### Strengths
- Tackles a real failure mode of LLM agents in collaborative tasks
- Fills a real gap between pure prompting and heavy world-modelling.
- Simple training pipeline: Learns from trajectories using a standard NLL+sparsity objective
- The policy-structure matrix gives insight into which inputs support which actions.
- The method shows consistent empirical gains over strong baselines.
- The system is partner-aware (by including the partner’s state in the feature set) without requiring access to or having to learn the partner’s policy
- Inference is simply masked feature sum+convex blend of LLM action probabilities, far cheaper than replanning/multiple model rollouts.
- Deployability is practical, given that the same assumptions hold. There is minimal friction into existing agent stacks given the learned matrix; one matrix lookup, one re-weighing step.
- The appendix is comprehensive and code was provided, facilitating reproducibility.

### Weaknesses
I will combine weaknesses, remarks, and questions in one section for readability.

The comments below reflect my current reading of the paper and appendix; if I’ve misread any definitions or misinterpreted any claims, I  welcome pointers and will happily revise.

To my understanding, the paper does not build a causal model of the world, though the writing sometimes suggests it is. The learned object is a policy-structure over observed features that predicts the next action, not a dynamics model one could query with do operators or counterfactuals. The SCA takes parents $(a_{t-1}, s_t)$. This models cause-effect relationships within the agent’s decision process, not the environment’s physics or tasks dynamics

According to the definition at L229-L232, each row of $M$ corresponds to a possible next action and each column to a state of past-action features. Each entry is the learned probability that feature $j$ influences action $i$, given the learned structure. Querying $M$ sums the active parent entries to produce a "causal score". In effect, from my understanding, higher sum implies that the model has learned that the currently active features are predictive parents of that next action. This is not a causal effect estimate in the Pearl sense.

The proof’s conclusion (L812-L814) that "the causal action matrix … faithfully reflect the true cause-effect relations among states and actions" reads too strongly. What is captured is a sparse dependence over $(s_t, a_{t-1})\to a_t$, which is a property of the actor’s policy, not of the world’s causal structure.

A (hard) intervention (if atempted) would possibly be toggling columns $j$ (making a feature active/inactive) and seeing how the score changes. The paper does not define or use interventional queries over an SCM over the environment’s variables. Concretely, the method learns a decision structure, not environment causality.

Can you clarify the above in the paper?

Minor: “intervention” is used informally (L277, re-prompting) which can be confusing in a section that discusses causal effects and structural causal action models.

Furthermore, there are two distinct issues with the proof:

A) Proof-method mismatches
- Estimator mismatch; The appendix analyses ridge on fixed basis features and reads parents from the support of $W$. The method itself trains neural Bernoulli heads via NLL and learns $\eta$ jointly. These are not equivalent and, as far as I can tell, one does not imply the other.
- ANM vs classifier; The proof invokes ANM-style identifiability but the trained model is a binary classifier
- Acyclicity gap: The proof assumes a DAG. The method’s heuristic "zero the smaller of each bidirectional pair" only removes 2-cycles, leaving (3+)-cycles in the action-action portion of the graph). Take for example the following relations: $W(a \to b) = 0.5 > 0.3 = W(b \to a)$, $W(b \to c) ‎ = 0.5 > 0.3 = W(c \to b)$, $W(c \to a) = 0.5 > 0.3 = W(a \to c)$. The heuristic would remove the three arrows $b \to a$, $c \to b$, $a \to c$ but $a \to b \to c \to a$ remains. This is inconsequential when the learned matrix is used in a feedforward manner as done in the paper, but the claim (L236-238)"…ensure DAG property of a standard SCM…" is not correct.
- Observability: The proposition (L220) states $a_t$ is unobservable, but the proof in the appendix uses an observed $a_t$ to train the SCA model. Clarifying unobserved at test time, observed during training would help.

Any of the above, in my view, render the statement proven in the appendix inapplicable to the method in the main paper (apart from the observability claim but that’s a wording inconsistency).

Questions: Either restate and prove a proposition for the actual model class and estimators, align the method to the ridge/basis estimator in the appendix or clearly state that this is a motivating surrogate that does not apply to the method.


B) Standalone proof issues

When considering the proof itself in isolation:

- The specific regularizing conditions/assumptions are not clearly stated (i.e., L728: "identifiability of causal direction relies on the function class having sufficient expressiveness and satisfying certain regularity conditions (e.g., nonlinearity, invertibility)". Invertibility, as far as I can tell, is not relevant to the proof. Please state the exact assumptions used.
- Ridge regression is used and the parents are identified by the support of $W_i$ (L808) with the claim (L809): "one can recover the graph structure by examining which entries of $W_i$ are significantly nonzero, using thresholding or statistical tests.". L2 regularisation has no sparsity guarantees and typically yields dense solutions. How exactly is this step justified and implemented?
- The proof assumes causal faithfulness but the collaborator’s actions are not modelled (L899-901). If $u$ denotes the collaborator’s action, an implicit assumption is made: $a_t \perp u_{t-1} \mid (s_t, a_{t-1})$. If $s_t$ is intended to be sufficient to mediate all effects of the parent’s last action, clearly state so, otherwise sufficiency is violated.
- Eq (8) instantiates the dataset as sequences of the form $(s_t, a_{t-1}, a_t)$. Clearly state what is observable and what is not.

Furthermore:
- L244-246: Can you clarify how the actions are sampled by the LLM and how they are scored?
- In the action pruning step, the method assumes access to an external verifier of feasibility. What does this verifier look like? What happens when such a verifier is unavailable (i.e., real-world robotics)? If it is required, can you add this as an explicit assumption?
- Why does CausalPlan underperform in some settings (Table 1)? A rief discussion of failure modes would help.
- L318: A short description of the baselines in the main text (with details in the appendix) would make the experiments easier to follow.
- What is the impact of removing the previous action feature/removing partner-state features or adding the partner’s previous action?
- How well does an SCA trained on trajectories from a behaviour policy paired with a partner transfer when deployed with different partners?

### Questions
See weaknesses

### Soundness
2

### Presentation
3

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
This paper proposes CausalPlan, a framework that purports to integrate explicit structural causal reasoning into LLM-based multi-agent planning. The core contribution is a Structural Causal Action (SCA) model that learns relationships between prior actions, current states, and future actions from agent trajectories. These learned relationships are encoded in a ``Causal Action Matrix'' $M$, which is then used to reweight LLM-generated action probabilities during planning. The authors evaluate their approach on the Overcooked-AI benchmark across multiple LLM backbones and show empirical improvements in task success rates and reductions in invalid actions.

### Strengths
1. Consistent gains across very different LLM backbones and layouts, not just one setup.

2. Human-partner results are stronger than baselines and include statistical testing; several settings reach p<0.05 and none show degradation when the method is enabled. 

3. The causal backup plan is an effective recovery mechanism when the planner proposes no valid actions; ablation shows it adds measurable benefit beyond the two-prompt tweak. 

4. The framework exposes a causal action matrix and publishes heatmaps, giving a degree of interpretability about which state/action factors influence next actions. 

5. Robustness to who collects the data for the buffer; using a stronger behavior policy helps, but even weaker LLM-collected data still benefits from the causal integration. 

6. Sensitivity analysis of the γ weighting shows a broad sweet spot. 

7. Explicit DAG enforcement by zeroing the weaker direction in any bidirectional pair prevents cycles in the learned structure. 

8. Modular drop-in over ProAgent with open-source LLMs, keeping the rest of the stack intact and making replication or extension straightforward. 

9. Extends beyond Overcooked to a long-horizon single-agent benchmark (Crafter) and outperforms both a causal-prompting baseline and Dreamer-V2 at 1M steps. 

12. Prompting design separates analysis from action selection, making the action extraction unambiguous; ablation indicates the components introduced to capture causal relationships drive most of the lift.

### Weaknesses
1. The framework learns from trajectories generated by a fixed behavior policy in Overcooked-AI, which means each action is conditioned on the policy’s internal decision process. Since actions aren’t randomized or independently manipulated, the data are observational, not interventional.

2. The Structural Causal Action model optimizes a likelihood loss ( -\log P(a_t \mid s_t, a_{t-1}) ), which captures conditional correlations rather than causal effects ( P(a_t \mid s_t, \text{do}(a_{t-1})) ). Without interventions or counterfactual adjustments, the learned structure reflects co-occurrence patterns, not causal mechanisms.

3. Although Overcooked-AI’s environment is deterministic, the data collection process is not interventionally controlled. The simulator ensures that actions deterministically affect states, but the trajectories used for learning are policy-dependent rollouts, not samples from systematically applied interventions.

4. Because the same policy governs both state visitation and action choice, correlations between (s_t) and (a_t) can arise from shared dependencies on unobserved latent factors such as internal LLM reasoning or high-level strategy. The model treats these as causal links.

5. The binary feature encoding used for states and actions is a coarse abstraction of the full simulator state. Hidden variables like spatial positioning or timing can confound state–action dependencies, violating causal sufficiency.

6. The framework’s only verification is improved prediction accuracy and task performance, which measure behavioral alignment, not causal correctness. A model can be highly predictive while causally wrong.

7. The paper’s theoretical identifiability proof relies on assumptions such as additive noise, faithfulness, full observability, and acyclicity, none of which are verified in Overcooked-AI. There is no empirical evidence that these assumptions hold in practice.

8. Each entry of the causal action matrix represents a learned dependency weight, not an intervention-derived causal coefficient. The matrix is effectively a correlation matrix with sparsity regularization.

9. The observed reduction in invalid actions and improved cooperation may result from regularized prediction smoothing or bias correction, not genuine causal reasoning. The gains demonstrate utility, not causal validity.

10. Because the learned structure reflects policy-specific correlations, the matrix may not transfer to different partners, environments, or task variations, contradicting the stated goal of causal generalization.

### Questions
1. How do you distinguish causal effects from correlations when all data come from a fixed behavior policy π₍ᵦ₎? What is your formal definition of causation in this context?

2. Can you provide empirical evidence that the faithfulness, causal sufficiency, and additive noise assumptions hold in Overcooked-AI? For instance, conditional independence tests, checks for unobserved confounders, or validation of the additive noise model?

3. What would an intervention experiment look like to validate your learned causal structure? For example, could you force an agent to take actions inconsistent with M and measure the deviation in outcomes?

4. Why not compare against a model that learns P(aₜ | sₜ, aₜ₋₁) with standard neural networks (e.g., feedforward or recurrent) without causal constraints? Does the DAG structure and sparsity actually matter, or are the gains from additional learned features?

5. How does performance degrade when the partner policy changes? Does your “causal” matrix M transfer to new partners, or is it partner-specific?

6. Can you show that the learned dependencies correspond to true causal mechanisms rather than artifacts of π₍ᵦ₎? For instance, by comparing M learned from different behavior policies?

7. Have you tested whether M changes systematically under distributional shift? This would be evidence of instability inconsistent with causal invariance.

8. Why is binary feature encoding sufficient when it discards causally relevant information such as spatial distances, timing, and interaction history?

9. What is the causal graph G you claim to identify? Can you draw it explicitly (not just heatmaps of M) and verify it against ground truth or domain knowledge?

10. In Proposition 1, you assume aₜ is “unobservable” during training, but clearly you observe aₜ in the trajectory data 𝓑. Can you clarify this apparent contradiction?

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes CausalPlan, a framework designed to improve LLM-based multi-agent collaboration by incorporating explicit causal reasoning into the planning process. The method introduces a Structural Causal Action (SCA) model that learns a causal graph from offline trajectories, modeling dependencies between state factors, prior actions, and next action choices. During inference, the causal graph is used to reweight sampled candidate actions from the LLM, promoting causally consistent planning and filtering out invalid or incoherent actions. 
Experiments are conducted on the Overcooked benchmark showing consistent improvements.

### Strengths
1. The paper clearly identifies a real pain point in the LLM-planning space  LLM agents often rely on correlations and fail under causal inconsistencies  and proposes a targeted solution. The motivation aligns well with current challenges in multi-agent LLM systems.
2. The method section is well-structured and easy to follow. The paper clearly explains the proposed causal model and the way it integrates with LLM action sampling. The theoretical argument that, under standard identifiability assumptions, the causal graph and functional relationships can be uniquely recovered adds credibility and supports why the approach should work.
3. Implementation details are provided in good depth, including model architecture and prompting strategies.

### Weaknesses
1. It does not compare against the most recent LLM-agent + causal reasoning  methods. For example, CausalMACE[1] and Causal-aware LLMs[2]. 
2. All evaluations are done in the Overcooked kitchen environment. While this benchmark is standard, it is still a fairly constrained action/state space in a stylized cooperative setting. It would be helpful to see results in a more diverse or general multi-agent domain (e.g., social games, robotics simulators). Otherwise, it's unclear how easily the method generalizes to richer or more realistic scenarios.
3. The method depends on manual factorization of state/action features, and lower-level actions are ignored. This raises concerns about domain specificity and manual engineering effort. In complex environments, designing semantic factors may be non-trivial, and it’s unclear how the method scales without strong prior knowledge.


[1]https://aclanthology.org/2025.findings-emnlp.777/

[2]https://www.ijcai.org/proceedings/2025/0478.pdf

### Questions
1. The paper claims not to rely on the LLM’s causal reasoning ability, yet the pipeline still depends heavily on the LLM for analysis and candidate-action generation via a two-prompt design and knowledge library. Could the authors clarify whether the method truly disentangles causal reasoning from linguistic reasoning? To what extent could the observed gains stem from improved prompting workflow rather than causal modeling itself?
2. Each decision step requires: extracting factorized features and querying the causal matrix, is the runtime overhead significant?
3. Does the training buffer come from the same task distribution as evaluation? Are trajectories from them fully disjoint from test episodes? Could the causal structure overfit to the demonstration policy rather than reflect true task dynamics?

### Soundness
3

### Presentation
3

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
This paper identifies a key failure mode in LLM-based agents, particularly smaller models, which often generate causally invalid actions in multi-agent collaborative tasks. To address this, the authors propose **CausalPlan**, a two-phase framework. In Phase 1, "Causal Action Structure Learning," a Structural Causal Action (SCA) model is learned from a dataset of agent trajectories to capture the influence of previous actions ($a_{t-1}$) and current states ($s_t$) on the next action ($a_t$). This is stored in a Causal Action Matrix (M). In Phase 2, "Agent Planning with Causal Knowledge," this matrix M is used to guide the LLM's action selection. This is done via two modules: 1) "Causal-Aware Planning," which re-weights the LLM's output probabilities with the causal scores from M, and 2) "Causal Backup Plan," a fallback mechanism that greedily selects the highest-scoring causal action if the LLM fails to produce a valid one. Experiments on the Overcooked-AI benchmark and Crafter demonstrate that CausalPlan reduces invalid action.

### Strengths
1.	The proposed two-phase framework is intuitive and modular.
2.	The paper is easy to follow.
3.	The empirical results are extensive and show consistent performance gains across multiple LLM backbones (Gemma, Llama, Qwen) and evaluation settings (AI-AI collaboration and Human-AI collaboration), outperforming baselines on the Overcooked benchmark.

### Weaknesses
1. The paper's primary claim rests on "causality-driven planning". However, the SCA model learns a supervised mapping from $(s_t, a_{t-1})$ to $a_t$ based on data collected from a single behavior policy (MEP). It is highly questionable whether this process discovers true "causal" structure as defined by Pearl or simply learns the strong correlations and biases within that specific policy's data. The proof of identifiability (Proposition 1) relies on strong, standard assumptions (e.g., causal sufficiency, additive noise) that are difficult to justify in a complex, dynamic environment like Overcooked.

2. A major limitation, which is not adequately discussed, is that the Causal Action Matrix $M$ appears to be learned **per layout**. The heatmaps in Fig. 10 and 11 are specific to the "CR layout", and the offline training takes 3 hours per environment. This severely limits the method's scalability and flexibility, which is one of the primary advantages of using LLM-based agents. The authors provide no evidence or discussion on whether $M$ learned on one layout can generalize to another.

3. The central idea of learning an external model from trajectory data to score and refine LLM-generated plans is not novel. The paper's related work section is missing key work [1] on this specific problem.
-	ReAd [1] directly tackles the same problem of inefficient LLM grounding in multi-agent environments like Overcooked.
-	The proposed "Structural Causal Action (SCA) model" is conceptually very similar to the local advantage function used in [1]. Both frameworks learn a function from agent trajectory data (collected from a behavior policy $\pi_\beta$ here) to score the utility of the proposed plan. While this paper formulates the scorer as a causal model $P(a_t | s_t, a_{t-1})$, ReAd [1] formulates it as an RL-based advantage function, the high-level approach of using a learned, data-driven scorer to refine LLM plans is highly overlapping. The authors must discuss this and other related works to properly situate their contribution.

### Questions
1.	Could the authors please clarify the novelty of the SCA model compared to other data-driven refinement models, such as ReAd [1] ? A thorough comparison in the related work section is necessary.
2.	Can the authors provide more evidence that the SCA model is learning true causal relationships rather than just the strong policy-specific correlations from the MEP dataset? What happens if a sub-optimal or random policy is used to generate the dataset $B$?
3.	Does the Causal Action Matrix M learned for one layout (e.g., Cramped Room) have any utility when transferred to another layout (e.g., Asymmetric Advantages)? If not, doesn't this per-layout offline training requirement undermine the zero-shot generalization promise of using LLMs?

[1] Zhang, Y., Yang, S., Bai, C., Wu, F., Li, X., Wang, Z., & Li, X. (2024). Towards efficient llm grounding for embodied multi-agent collaboration. ACL 2025

### Soundness
2

### Presentation
2

### Contribution
2
