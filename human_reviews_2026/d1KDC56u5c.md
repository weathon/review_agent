# Probabilistic Modeling of Latent Agentic Substructures in Deep Neural Networks

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 4, 6

## Abstract
We develop a theory of intelligent agency grounded in probabilistic modeling for neural models. Agents are represented as outcome distributions with epistemic utility given by  log score, and compositions are defined through weighted logarithmic pooling that strictly improves every member's welfare. We prove that strict unanimity is impossible under linear pooling or in binary outcome spaces, but possible with three or more outcomes. Our framework admits recursive structure via cloning invariance, continuity, and openness, while tilt-based analysis rules out trivial duplication. Finally, we formalize an agentic alignment phenomenon in LLMs using our theory: eliciting a benevolent persona ("Luigi'') induces an antagonistic counterpart ("Waluigi''), while a manifest-then-suppress Waluigi strategy yields strictly larger first-order misalignment reduction than pure Luigi reinforcement alone. These results clarify how developing a principled mathematical framework for how subagents can coalesce into coherent higher-level entities provides novel implications for alignment in agentic AI systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper is a pure theoretical paper which studied the aggregation and decomposition of probabilistic output of AI models. It defines the notion of an agent as a distribution over outcomes, and define the notion of compositional agent by assigning each agent a utility function. The paper proves several results including existence of compositional group and impossibility result in binary-outcome and linear pool. The second part of the paper concerns decomposing one agent into several subagents. This amounts to dividing one probability distribution into several.

### Strengths
The technical contribution is solid.

### Weaknesses
The motivation of this paper is questionable.

### Questions
While this paper clearly contributes several interesing theoretic results in the field of probablistic modeling, I am general concerned about its practical applications in the field of AI. To name a few: what are examples, particularly examples about LLMs, that aggregate and decompose AI models in the way the paper is concerned? Where does the utility come from? How to choose \beta, and, why is \beta outcome-independent?

How general is theorem 6? I.e., how easy it is to find such configuration? To my understanding the configurations are actually given right? Like each LLM has its own pre-defined P_i.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a probabilistic framework for understanding neural networks as collections of latent “subagents,” each represented by a probability distribution with its own utility (the log of its predicted probabilities). Using concepts from opinion pooling, it models the overall network as a log pooling of the final activations (before softmax) of the networks. The authors prove that mutually beneficial compositions of agents—where every subagent’s welfare improves—are impossible under linear pooling or binary outcomes but possible with three or more outcomes under logarithmic pooling. They further establish stability properties showing that such compositional structures are robust to small perturbations and can be consistently decomposed or extended. Finally, they apply this framework to make claims about the “Waluigi effect” in LLMs, arguing that reinforcing a benevolent persona necessarily manifests an anti-aligned counterpart.

### Strengths
- The paper is ambitious in scope and makes connections across several subfields of probabilistic modeling, welfare economics, active inference, and alignment. 

- The setup seems novel, treating neural networks as ensembles of interacting probabilistic subagents and offers a new perspective on how a family of neural agents can be combined.

- The mathematical theorems seem sound and the mathematical presentation is relatively easy to follow.

- Although I’m not entirely convinced by the analysis, providing a theoretical framing of the Waluigi effect would alone be a strong contribution to the field of AI alignment.

### Weaknesses
- No motivation is given for why we would want to aggregate LM agents in the way described in the second paragraph of the intro. Aggregating model’s opinions in this way is very limiting since it aggregates the models on a per-token basis, not leaving room for CoT or reasoning. 

- Most of the theorems just seem like facts about log pooling in general and the connection to LM agents feels detached. I think the paper needs significant work in connecting the theorems back to reality.

- The paper makes some strong assumptions in the intro that are not well motivated.: For example: “higher logits imply higher probability of selecting an outcome, which we interpret as having greater utility for the agent” – interpreting LM agents in terms of utility in this way is not common and no citation is given for why this approach is taken. It’s reasonable to treat LM agents as decision makers based on their logprob over tokens but the connection to utility is not clear. This is an extremely important point since the entire paper relies on this assumption of treating Wi(o) = log Pi(o).

- The decomposition of agents using the pooling scheme prevents agents from having separate contexts and thus precludes separate CoTs, scratch pads, or scaffolds which are important components of modern LM agents. 

- The writing is very verbose, often leaves terminology undefined, and is difficult to follow in places.

- The paper is highly theoretical, containing no empirical results or toy-examples so the connection to real LM agents is difficult to grasp.

### Questions
- Why is additive aggregation in log-probability space the natural choice compared to any other voting rule?
- The decomposition of agents using the pooling scheme prevents agents from having separate contexts and thus precludes separate CoTs, which are important components of modern LM agents.
- You call O the observation space at the end of section 2 by omitting tokens y_{<t}. This omission in the notation adds a lot of confusion since LM agent’s context includes many tokens and makes the rest of the paper 
- Why is the definition of W left general at the beginning and then specialized to Wi(o) = log Pi(o)? Why not start with this? As far as I can tell, the general case is not used.
- Thm 9 gives a positive result that beliefs exist that allow pooling for 3 or greater agents that leads to improvement for all agents. However, why is an existence proof useful? What does this case look like? It seems that in most cases of agent’s beliefs, aggregation that leads to improvement for all would be impossible.
- I don’t understand why the term ‘recursion’ is used in section 4. Isn’t this just describing a decomposition or factoring? The term recursion is used several times but never defined.
- Section 5 relies heavily on Thm 17 and makes the analogy that a gradient step or KL-constraint keeps the model within a small perturbation of the original model. However, what guarantees that the small perturbation will remain within the epsilon ball specified by Thm 17? The rest of the analysis in this section relies on this fact and I don’t see why it should hold.
- The connection to the empirical Waluigi Effect is not entirely clear - why is decomposition into subagents via subpooling a good model for differing personas in the model?

### Soundness
2

### Presentation
2

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
This paper proposes a theoretical framework for compositional agency in neural networks, treating a model (in this case, an LLM) as a collection of interacting subagents with whose beliefs are aggregated to obtain the model’s output logit. Intuitively, these subagents reflect the many different data-generating processes (i.e. authors) that are encountered in a model’s training data, and this framework describes how these many disparate voices combine to give rise to coherent model behavior. The authors prove several structural results: impossibility of unanimous welfare improvement under linear pooling or binary outcomes; existence and openness of unanimous improvement for multi-outcome spaces; and various stability and invariance properties of compositional structures. Finally, they apply this theory to a concrete alignment phenomenon (the “Waluigi effect”), in which strengthening a benevolent persona (“Luigi”) in an LLM necessarily increases weight on an antagonistic, anti-aligned persona (“Waluigi”). The author’s framework reveals that a “manifest-then-suppress” (shattering) strategy can yield greater alignment than simply reinforcing the benevolent persona alone.

### Strengths
1. **Compelling exposition and clarity**. The paper is very well written. Definitions / theorems and intuitions are consistently paired, and the prose maintains motivation throughout.
2. **Conceptual novelty**. The move to treat neural networks as log-pooled collections of sub-agents with probabilistic utilities appears original and bridges ideas from interpretability, active inference, and economics.
3. **Concrete application to alignment**. The “Waluigi effect” analysis provides a real-world-relevant proof of concept, illustrating that the framework can generate interpretable predictions about phenomena of interest to alignment researchers.
4. **Mathematical rigor**. Results appear internally consistent and connect cleanly to known principles in information theory and game theory (from my admittedly limited knowledge of these fields).

### Weaknesses
1. **Unclear practical motivation**. The paper would benefit from a more explicit discussion of what adopting the “compositional agency” formalism enables that prior formulations (e.g., mechanistic interpretability, ensemble modeling, active inference) do not.
2. **Limited empirical or illustrative scope**. The Waluigi effect is compelling but singular; no other examples are explored, which makes it hard to judge the breadth of applicability. A bit more exposition about possible applications would strengthen the work.

### Questions
1. **Interpretability connection**. How might the subagents or welfare functions be identified in practice (e.g., could they correspond to sparse autoencoder features or attention heads)? 
3. **Broader utility**. Beyond the Waluigi effect, what other alignment or interpretability phenomena could this theory uniquely illuminate (e.g., deception, self-preservation drives, or internal goal conflicts)?

### Soundness
4

### Presentation
3

### Contribution
3
