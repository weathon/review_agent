# Training as Computation: A Resource-Bounded Theory of Continual Self-Play Learning

- Decision: Reject
- Scores: 6, 0, 6, 8

## Abstract
We study \emph{training as computation} in a continual self-play setting, where a single reasoning model proposes tasks, solves them, and updates itself using verifiable signals from an external executor--verifier interface. Rather than focusing on one-shot models, we analyze the \emph{process-level} dynamics of learning under explicit resource budgets: each generation step is capped by an output budget, and the executor/verifier operate within bounded working space. Within this framework we (i) formalize a general generator--executor--verifier--buffer loop for continual learning with self-proposed curricula; (ii) provide a \emph{process-level characterization} of expressiveness---the set of functions computable by the evolving loop up to time $t$ matches a corresponding $\mathrm{SPACE}[\cdot]$ class determined by the budgets; and (iii) show monotone capability growth under explicit, length-aware exploration schedules and curriculum learnability mechanisms, without assuming non-vanishing exploration or relying on supervised traces. Conceptually, the results separate \emph{capability universality} (as properties of the training \emph{process}) from \emph{alignment and safety} (properties of objectives and verifiers). This positions continual self-play as a principled theoretical framework for understanding data-free improvement under explicit resource budgets.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This paper looks at modern self-play loops for LLM-like models, but from a more theoretical angle.  The authors formalize a loop with four parts (generator, executor, verifier, buffer, learning) and show that, under some exploration and buffering assumptions, this loop is powerful enough to simulate arbitrary computations. They also give a resource-bounded version: if you limit length, context or buffer, you get a hierarchy of what the process can solve. Finally, they provide small synthetic experiments to illustrate the ideas.

### Strengths
1. The problem is interesting and timely. Many current LLM systems actually do this kind of self-improving loop using SFT or RL so it is good to have a clean formal view.

2. The paper is clearly written overall, and well-structured.

### Weaknesses
1. Many of the key assumptions felt quite strong to me. I am not sure whether typical LLM self-play systems used in practice satisfy these conditions.

2.  The main results are more “existence/theoretical” than “practical.” As a non-expert, it was hard for me to judge how far the current experiments are from the assumptions in the theorems.

3. The paper ends somewhat abruptly without a dedicated conclusion section. A short conclusion summarizing the main theoretical message, its practical scope, and open problems would make the contribution clearer, especially for non-expert readers.

Because I do not work in this exact theory, I cannot fully verify the technical parts of the proofs.

### Questions
1. In real LLM training, entropy often collapses[1,2,3], and we do not keep a fixed exploration floor. How sensitive are your guarantees to this? If exploration becomes more and more greedy, do the main theorems still hold in some weaker form?


2. For the verifier: could you list a couple of realistic, LLM-style self-play tasks that fully meet your verifier assumptions? That would help readers like me map the theory to practice.


Finally, this topic is somewhat outside my primary area of expertise, so I apologize in advance if I have overlooked relevant prior work or misunderstood parts of the paper.

[1] Yu, Qiying, et al. "Dapo: An open-source llm reinforcement learning system at scale." arXiv preprint arXiv:2503.14476 (2025).

[2] Wang, Shenzhi, et al. "Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for llm reasoning." arXiv preprint arXiv:2506.01939 (2025).

[3] Cui, Ganqu, et al. "The entropy mechanism of reinforcement learning for reasoning language models." arXiv preprint arXiv:2505.22617 (2025).

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This submission considers the computational universality of training processes, specifically as they relate to reasoning and language models.
The authors formalize the self-play process between generator, solver and verifier. They lay out a few assumptions on such a process so that it may eventually emulate every Turing machine.

### Strengths
- Building further understanding of the relationship between computational capabilities and training processes is important.

### Weaknesses
- The paper is poorly written, with many sections of technical discussion with errors and unclear definitions, or digressions which do not seem relevant to the points being made.
- Some missing discussion of prior work which show approximation error to Turing Machines [1] and universality of learning processes [2].
- Theoretical contributions appear incorrect or vacuous
- There are no experiments, despite claims of them in the abstract: "Empirically, a light-weight self-play prototype on synthetic program-execution and abduction/induction tasks corrob- orates the theory"

### Questions
## Detailed Comments
- On computational power of learning process: previous work has investigated whether learning algorithms can approximate Turing machines. It has also been established that meta-learning methods using gradient descent are universal, in the same sense that recurrent networks are universal [2]
- line 56: the Kolmogorov complexity of a function is not defined in this submission. Such complexity measures are usually defined on strings.
- line 59: It is not defined what it means for a function to be "solved" and, moreover, isn't this bound completely vacuous? Any string with kolmogorov complexity $<\ell$ can be found by enumerating all up strings up to $\ell$, which is exponential in $\ell$.
- line 59: $c, \epsilon_0,\gamma$ are all undefined
- line 63: It is not clear how this differs at all from the traditional space hierarchy theorem.
- Definition 1: What is the significance of using Python 3 and the additional qualifiers around the particular subset?
- Definition 4: What does it mean to "halt within bounds", what bound?
- Line 212: The definition of $\Tau_{ded}$ is not clear here, is $p$ and $i$ the set of all TMs and inputs? Induction and abduction are not used at all within the paper.
- Definition 5: the term multiset is used twice without definition
- Section 4: there is no discussion of how the agent is bounded, or how the resource constraint actually pertain to the results.
- Assumption 6: What does it mean for $u  \subset mathcalG_{\theta_t}$? My lack of understanding stems from the fact that your definition of stochastic kernels is not clear. Does it involve sampling a string from $\Sigma^*$? If that's the case, then I do not think $\epsilon_{min}$ can be independent of $|u|$ or $t$. "Ensuring theoretic universality..." What does the qualifier "theoretic" mean in this context?
- Assumption 9: Temperature is not defined yet. Moreover, $\ell$ is used here for logits when it is already used for the length of the string earlier, nor is it clear how this $\ell_i$ as a logit is related to a specific symbol $i \in \Sigma$
- Section 5: there are various points at which the theoretical results digress into practical considerations for no obvious reason.
- The discussion of resource bounds in section 6.3 are a comparative afterthough, despite it being featured in the titile.

## Minor Comments
- Definition 2: Should be specified that $M \in TM$ and that $C$ is implicitly defined by $p_m$
- Definition 3: I do not understand what the zig-zag line means in the definition of the kernel map 

## References
[1] Wei, Chen & Ma. Statistically meaningful approximation: A case study on approximating Turing machines with transformers. Neural Information Processing Systems, 2022.
[2] Finn & Levine. Meta-Learning and Universality: Deep Representations and Gradient Descent can Approximate any Learning Algorithm. International Conference on Learning Representations, 2018.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces a resource-bounded theoretical framework for continual self-play learning, where a single reasoning model generates tasks, solves them, and updates itself using verifiable signals from an executor-verifier interface. It formalizes a generator-executor-verifier-buffer loop, proves that the training process is Turing-complete at the process level (matching SPACE classes under budgets), demonstrates monotone capability growth under mild exploration and curriculum assumptions, and separates computational universality from alignment/safety concerns. Contributions include process-level universality proofs, a Coverage-Persistence-Learning (CPL) decomposition, finite-time/resource-bounded guarantees, and empirical validation via a lightweight prototype on synthetic program-execution and abduction/induction tasks, showing reliable task expansion and benefits from curriculum learnability.

### Strengths
The paper exhibits strong originality by shifting focus from architectural Turing-completeness to the expressiveness of the self-play training loop itself, introducing novel concepts like the CPL decomposition and a SPACE hierarchy tied to resource budgets—this creatively combines ideas from computability theory, RL self-play, and resource-bounded complexity in a new way, addressing limitations in prior work that emphasized equilibria or sample complexity without considering process-level universality. The quality is high, with rigorous mathematical formulations, clear proofs, and well-defined assumptions that build a coherent framework, though I appreciate the theoretical richness as someone who enjoys such articles but is not an expert in computability or formal RL theory. Clarity is excellent: the structure flows logically from motivation to model, proofs, and related work, with precise notation, diagrams (e.g., Figure 1), and appendices implied for details. Significance is notable, as it provides a principled foundation for scalable, data-free AI improvement under budgets, potentially guiding real-world self-play systems in domains like LLMs or robotics, while highlighting safety implications by decoupling capability growth from objectives/verifiers.

### Weaknesses
The paper is heavily theoretical, with the empirical evaluation limited to a "light-weight self-play prototype on synthetic program-execution and abduction/induction tasks," which lacks depth—specific details on model architecture, hyperparameters, quantitative metrics (beyond qualitative "reliable expansion" and "benefits from curriculum"), or comparisons to baselines (e.g., non-self-play RL or standard curriculum methods) are absent, making it hard to assess real-world applicability or robustness; to improve, the authors could expand this section with ablation studies or scale to non-synthetic datasets.

I'm not an expert in this field, so I'm not sure if this paper without experiments is acceptable.

### Questions
1.The empirical prototype is mentioned as corroborating the theory on synthetic tasks, but details are sparse. Could you elaborate on the specific tasks (e.g., examples of program-execution or abduction/induction problems), model sizes/architectures used, key quantitative results (e.g., task coverage over time, success rates), and any ablations (e.g., with/without curriculum learnability)?

2.The paper separates universality from alignment/safety, which is a strong point, but doesn't discuss potential risks (e.g., emergent harmful behaviors in self-proposed curricula). Could you add thoughts on integrating safety mechanisms into the verifier or buffer, perhaps with examples?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes a novel framework for studying continuous improvement achievable by a model using self-proposed curriculum learning in presence of an external verifier. The paper studies learning-process universality where a training loop is guaranteed to find a weights that can solve the given task solvable under bounded resource constraints. This is very different from architectural universality where a single weight is required to solve all the tasks. The authors define a space of functions called SPACE and prove that resource bounded self play systems can learn to sovle any task in this space. The paper further proves that the capability of the model grows monotonically as long as model maintains a minimum level of exploration ensuring it will propose and learn even the most complex tasks.

### Strengths
1. The paper introduces a clean formalization of 'learning-process universality' - shifting focus from static capacity of a fixed architecture to dynamic capability of a continual training process itself thereby bringing a new perspective to the field.
2. The authors provide theoretical results proving process-level learning can learn any functions in set SPACE defined by resource/budget constraints.
3. The authors propose a "capability ladder" that bridges the gap between abstract computation theory and resource constrains present in building real world models.

### Weaknesses
1. The paper makes an strong assumption about presence of an efficient verifier model which is not always possible in practice.
2. The theorems hinges on the idea that there is non-zero proability of sampling any task string while in real world generators based on language models give zero probability to vast space on inputs.
3. The framework does not comment much on efficient learning is a big limitation.
4. The abstract mentions an empirical study of light-weight self play prototype which seems to be missing in the paper.

### Questions
1. What role does model architecture (i.e. number of parameters) play in this setup? Is it possible for a small model to learn to solve any problem in this framework?
2. The space of input string is huge. How are difficult tasks distributed in that space and how does the model ensure that it keeps learning from the difficult task?
3. Where are the empirical results mentioned in the abstract of the paper?

### Soundness
3

### Presentation
3

### Contribution
3
