# Learning Compact Regular Decision Processes using Priors and Cascades

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
In this work we study offline Reinforcement Learning (RL), and extend the previous work on learning Regular Decision Processes (RDPs), which are a class of non-Markovian environment, where the unknown dependency of future observations and rewards from the past interactions can be captured by some hidden finite-state automaton. We utilise the language metric introduced previously for an offline RL algorithm for RDPs, and introduce a novel algorithm to learn a significantly more compact RDP with cycles, which are crucial for scaling to larger, more complex environments. Key to our results is a novel notion of priors for automaton learning, that allows us to exploit prior domain-related knowledge, used to factor out of the state space any feature that is known a priori. We validate our approach experimentally and provide a Probably Approximately Correct (PAC) analysis of our algorithm, showing it enjoys a sample complexity polynomial in the relevant parameters.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In the context of Regular Decision Processes (RDPs), the authors formalize the notion of cascades/factorization of RDPs, followed by one instance (prior x posterior). Authors provide a new learning algorithm ADACT-L that utilizes these ideas (with PAC guarantees), and demonstrate its compactness and effectiveness in five common grid environments.

### Strengths
Originality and significance: The construct are novel in the context of RDP+RL, as far as I can tell. The idea of cascades and factorization for reusability is intuitive and neat, which stands out from the three priors in Sec 3.1, providing an umbrella framework that covers three seemingly independent POMPDP concepts.

Quality and clarity: The theoretical and empirical (albeit grid world) results clearly demonstrate the advantages of using priors, which is not surprising but still really nice to have it formalized. Clarity can be improved (see below).

### Weaknesses
Readability: This paper is really dense in terms of definitions and constructs, which I think limits its accessibility and impact to broader ICLR audience outside of the core formal language + RL folks. I recommend adding a running toy example to concretize any discussion and to motivate your construct. It is somewhat surprising to me that contribution starts on page 6, perhaps most of Sec 2 (except for automata cascade and the definition of RDP) could be folded in the appendix? Illustrative and concrete examples are much more readable than a comprehensive derivation from first principles (which is also nice but perhaps not suitable for conference publications).

### Questions
1. Could you comment on the similarity and difference between cascades and the HRM work (that you have already cited)?

Furelos-Blanco, Daniel, et al. "Hierarchies of reward machines." International Conference on Machine Learning. PMLR, 2023.

2. Could you elaborate on how cycles are learned in ADACT-L?

### Soundness
3

### Presentation
2

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
The paper considers the problem of learning Regular Decision Processes (RDPs) and proposes an algorithmic technique, AdaCT–L, which enables the learning of _compact RDPs with cycles_. In addition, the authors introduce the notion of priors for RDPs. A prior is defined as an automaton used to factor out known components of the state. For example, in a grid-based robotic domain, one may factor out positional information that is already known. This procedure allows the learner to focus on the hidden components for which domain knowledge is unavailable or difficult to model explicitly. The cascade operator divides the automaton into two parts: the _prior_ $\boldsymbol{A}_p$, representing the known, domain-grounded structure, and the _remainder_ automaton $\boldsymbol{A}_r$​, which captures the hidden dynamics that must be learned from data. This decomposition reduces the size and complexity of the learned components in the RDP.  
Finally, the authors empirically compare AdaCT–L with existing RDP learning methods on several benchmark domains.

### Strengths
- The paper is extremely well written, and results are well presented.
- The contribution is clear. The paper first formalises the notion of priors and cascade operator for the RDP setting, providing examples. Then they provide an algorithm that, given a prior $\boldsymbol{A}_p$, learns the problem-specific reminder automaton $\boldsymbol{A}_r$. The authors provide also PAC analysis on the sample complexity. 
- The proposed contribution is interesting, factoring out parts of the RDP state space with domain knowledge yields several benefits. For instance, it makes the RDP more compact and learning more efficient.

### Weaknesses
- While the theoretical framework is appealing, my concern is about practical scalability. In most realistic domains, priors may be difficult to formalize except in structured settings such as navigation or when the model of some components is available.
- The empirical evaluation is limited to small, symbolic domains. It would be valuable to extend or to add some comment, on how this framework, maybe with approximations, can be extended to more realistic or continuous applications.
- Some limitations are not discussed in depth. For instance, whether there is a way to automatize prior definition, or whether the framework can be extended to stochastic or noisy domains, since the current formulation relies on deterministic automata.

### Questions
- Can the notion of priors be used not only to inject domain knowledge but also to encode constraints or specifications on the learned automaton?
- Is there a possible way to automate priors definition rather than specifying them manually? For instance the priors can be learned from data but then re-used on different applications.
- The framework currently assumes deterministic transitions for priors. How would it extend to cases where the known domain knowledge is stochastic?
- How can approximation methods be introduced? For instance parameterization of the priors or the remainder automaton with a neural networks?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies offline RL in RDPs enviroment, where future outcomes depend on past interactions.
The paper introduces **priors** to incorporate known structural knowledge into RDP learning and cycles to enable state reuse, together yielding more compact and scalable models.
Based on these ideas, the paper proposes the ADACT–L algorithm, which learns compact cyclic RDPs with priors and provides polynomial PAC sample-complexity guarantees.
The authors conduct numerical experiments to demonstrate the advantage of their algorithms.

### Strengths
1. The concept 'prior' introduced in the paper is interesting. The authors mathematically formalize how known structural information influences decision making through the introduction of priors—a general and modular representation that can be applied across multiple environment.
2. The authors provide multiple fundamental priors and demonstrate how these can be expressed using the definitions presented in the paper. This effectively showcases the generalization capability of the concept.

### Weaknesses
1. The notation system in this paper is very confusing, which makes it difficult for reviewers to follow the logic in the main text. The authors should provide a table of symbols for clarification.
2. The authors should provide some examples to support the claim that having a prior brings advantages (e.g., reducing sample complexity). The paper does not clearly demonstrate why incorporating a prior leads to better result in theory.
3. See Questions.

### Questions
1. For the cycle definition proposed in the paper, does the author mean that previous works consider $U_t \times ao \to U_{t+1}$, while this work uses $U \times ao \to U $? If so, there is essentially no difference, since $U $ can be regarded as the intersection of all $ U_t $.
2. The sample complexity of Theorem 1 appears to scale with $1/\mu_x^2$, which means that when $\mu_x$ is sufficiently small, the result becomes vacuous. Is this reasonable? Can the authors prove that $1 / \mu_x^2$ is a lower bound for the sample complexity?

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper builds on the recent work of Deb et al. (2025) to improve the learning of Regular Decision Processes (RDPs) through the use of priors and automata cascades. While Deb et al. were limited to acyclic RDPs, the present work extends the framework to learn cyclic RDPs. Experimental results indicate that the proposed approach outperforms both Deb et al. (2025) and FlexFringe, suggesting that incorporating priors and cascades can yield more compact and expressive RDP models.

However, since Deb et al. (2025) already provided detailed comparisons with FlexFringe, the main novelty here appears to be the extension to cyclic RDPs rather than a fundamentally new comparison with FlexFringe.

### Strengths
1. The paper extends previous RDP-learning methods to handle cyclic structures, which is an important and nontrivial generalization.
2. The introduction of priors and cascades provides both theoretical interest and practical utility in constructing more compact RDPs.
3. Experiments demonstrate that the approach yields measurable improvements over earlier baselines.

### Weaknesses
### 

1. There appears to be a nontrivial amount of overlap with Deb et al. (2025) in the technical exposition, related-work discussion, and experimental descriptions. This reduces the perceived novelty of the presentation.
2. The notion of automata cascades dates back to the work of McNaughton, Eilenberg, and the Krohn–Rhodes theorem, which formalizes the idea that complex automata can be represented as cascades (or wreath products) of simpler components. The paper does not acknowledge or discuss this line of work.
3. It remains unclear whether these classical results directly or indirectly inform the proposed learning method for cyclic RDPs. Explicitly connecting the two would enrich the theoretical context.
4. A figure illustrating the cascade product and the overall learning pipeline would greatly help the readers grasp the intuition behind the approach.
5. The paper states that it compares performance with FlexFringe, but the results appear identical to those reported by Deb et al. (2025). If no new experiments were run, this should be stated explicitly.
6. The narrative does not sufficiently emphasize how priors and cascades serve as the key drivers of improvement. The presentation would benefit from clearer motivation and intuition for these components.
7. In the experiments, some details differ from Deb et al. (2025). For example, the reward for the T-Maze(c) environment is changed from 4 to 1, though the paper does not explain the reason for this modification.

### Questions
1. The concept of cascades seems closely related to the *wreath product* construction in automata theory. Am I correct in understanding that the central idea of the paper is that cyclic RDPs can be learned as cascade products of simpler acyclic automata? 
2. Are there any conceptual or theoretical connections to the Krohn–Rhodes theorem that the authors could elaborate on?
3. Could the authors clarify whether entirely new FlexFringe experiments were conducted, or whether results were re-used from Deb et al. (2025)? 
4. Can the authors provide an illustrative figure showing the cascade composition and the flow of learning between levels? An overview example would be quite helpful.

### Soundness
3

### Presentation
2

### Contribution
2
