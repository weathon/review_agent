# Logic of Hypotheses: From Zero to Full Knowledge in Neurosymbolic Integration

- Decision: Reject
- Scores: 2, 6, 6, 2

## Abstract
Neurosymbolic integration (NeSy) blends neural‐network learning with symbolic reasoning. The field can be split between methods injecting hand-crafted rules into neural models, and methods inducing symbolic rules from data. We introduce Logic of Hypotheses (LoH), a novel language that unifies these strands, enabling the flexible integration of data-driven rule learning with symbolic priors and expert knowledge. LoH extends propositional logic syntax with a choice operator, which has learnable parameters and selects a subformula from a pool of options. Using fuzzy logic, formulas in LoH can be directly compiled into a differentiable computational graph, so the optimal choices can be learned via backpropagation. This framework subsumes some existing NeSy models, while adding the possibility of arbitrary degrees of knowledge specification. Moreover, the use of Gödel fuzzy logic and the recently developed Gödel trick yields models that can be discretized to hard Boolean-valued functions without any loss in performance. We provide experimental analysis on such models, showing strong results on tabular data and on the Visual Tic-Tac-Toe NeSy task, while producing interpretable decision rules.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces Logic of Hypotheses (LoH), a neurosymbolic framework that extends propositional logic with a choice operator allowing the model to select among alternative subformulas in a differentiable way. Using Gödel fuzzy semantics and a reparameterization mechanism inspired by the Gödel trick, the authors claim that LoH can unify both knowledge injection and knowledge extraction within a single formalism.

### Strengths
The manuscript is well written.

The grounding in Gödel fuzzy logic and the preservation of logical consistency is elegant

### Weaknesses
Conceptual novelty and positioning is weak

Unclear novelty of choice operator

Unclear scope of contribution

Unsupported interpretability claims

I will describe these points below.

### Questions
**Conceptual novelty and positioning**
The claimed unification of knowledge injection and knowledge extraction is not clearly distinct from what existing (neuro)symbolic systems already achieve. Every ILP-style method naturally supports both the inclusion of background knowledge and the induction of new rules, as once worked the semantics of the rules-to-learn adding rules is automatic. As a result, the conceptual advance over prior work remains ambiguous.

**Semantic grounding of the “choice” operator**
The proposed operator resembles constructs that are well established in Statistical Relational AI and Probabilistic Logic Programming, such as probabilistic facts, annotated disjunctions and random-variable choices. The paper does not discuss these connections nor explain whether LoH provides new semantics beyond these long-known interpretations of discrete uncertainty. Given that “choosing among alternatives” is central to many probabilistic logic models, this omission weakens the originality and theoretical depth of the proposal.

**Unclear scope of contribution (syntax vs. semantics)**
Connecting to the point before, it is difficult to determine whether LoH represents a genuinely new semantic model or merely a syntactic/implementation-level reformulation of existing differentiable logic frameworks. The paper does not demonstrate that LoH achieves behaviors that could not already be implemented within DeepProbLog, LTN or other differentiable logical systems by suitable parameterization. The lack of such differentiation undermines the claimed unification.

**Interpretability claims**
In the experiments, the results primarily compare predictive accuracy and do not substantiate the key interpretability or symbolic extraction claims. When used on a parallel and deep extraction of rules, such architecture does not seem to be any more interpretable than existing neural models, despite maybe the discrete nature, which is not a prerogative of the proposed method. Therefore, interpretability claims are nos]t supported by sufficient evidence.

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
This paper introduces the Logic of Hypotheses (LoH), a novel language designed to unify the two primary paradigms of neurosymbolic integration: injecting pre-defined rules and inducing rules from data. LoH syntactically extends propositional logic with a choice operator, which compactly represents a hypothesis space, i.e., a set of candidate formulas. The key mechanism involves compiling any LoH formula into a differentiable computational graph using fuzzy logic, enabling the selection of the optimal formula from this space to be learned via standard backpropagation. By uniquely leveraging Gödel fuzzy logic and the Gödel trick, the resulting model can be discretized into a hard Boolean-valued function without loss in performance. The framework demonstrates strong performance on tabular data and a visual Tic-Tac-Toe benchmark.

### Strengths
- This paper elegantly bridges the gap between knowledge injection and rule induction paradigms in neurosymbolic AI, offering a single formalism that handles a spectrum of prior knowledge specification.
- The language of Logic of Hypotheses (LoH) hinges on the choice operator, which provides flexibility for encoding hypothesis spaces, syntactic templates (CNF, DNF, Horn clauses), and partial knowledge scenarios. 
- Furthermore, LoH formulas can be compiled into a fully differentiable computation graph. This allows the model to be trained with backpropagation and stacked seamlessly on top of neural networks (like a CNN) to enable end-to-end learning from raw data (like images) to symbolic rules.

### Weaknesses
- The framework is restricted to propositional logic, limiting applicability to relational/structured domains. After all, many real-world NeSy applications require relational reasoning.
- The scalability of the paper to multi-class classification tasks is weak. The experimental results show that while LoH performs well on binary and few-class tasks, it struggles on datasets with a large number of classes.
- Although the paper spends many efforts describing LoH's flexibility in handling partial knowledge (e.g., revising rules, using templates), the main experiments in Section 7 focus on the zero-knowledge (pure rule induction) setting.
- (minor) Figure 2-4: Consider moving some to the main text; they provide valuable insights.

### Questions
- When extending to FOL as a next step, what are the biggest challenges in doing this? How would the choice operator concept be extended to handle variable quantifiers?
- The reparameterization for multi-class (Section 7.1) seems ad-hoc. Have you considered alternative approaches (e.g., one-vs-rest, hierarchical)?

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
This paper proposes a unified training framework that bridges the two main directions in NeSy: knowledge injection and rule induction. Building on Gödel fuzzy logic, the authors extend propositional logic by introducing a choice operator and assigning learnable weights to subformulas, enabling the model to learn logical structures that range from zero prior knowledge to full prior knowledge. The resulting language is differentiable while preserving discrete Boolean semantics, thus supporting supervised gradient-based learning. Experimental results on small-scale tasks, including tabular datasets and a visual tic-tac-toe setting, demonstrate promising performance and interpretable logical representations.

### Strengths
1. The introduction of the choice operator allows the model to flexibly handle scenarios ranging from zero to full knowledge and supports different search tendencies, showing strong generality.
2. Under Gödel semantics, the proposed framework achieves differentiability without losing Boolean precision, with rigorous and self-consistent theoretical derivations.
3. The paper provides both analytical and empirical comparisons between disjunctive and conjunctive compilations, clarifying their respective applicability and showing that the approach is practically feasible and adaptable to data characteristics.

### Weaknesses
1. The experimental setups are relatively simple. For instance, the tic-tac-toe task only involves nine propositional variables, and the target logic can be expressed using a few disjunctive or conjunctive clauses. The framework’s computational scalability in larger logical spaces has not been demonstrated.
2. The paper directly adopts Gödel fuzzy logic without a clear justification of its advantages compared to alternative fuzzy logics such as Łukasiewicz or Product logic.
3. Although the framework claims to handle the full spectrum from zero knowledge to full knowledge, all experiments are conducted under the zero-knowledge setting; results for partial- and full-knowledge scenarios are missing.
4. The reparameterization of weights as continuous parameters z_i(logits) lacks a theoretical analysis of feasibility, stability, or convergence during training.

### Questions
If the visual tic-tac-toe input is first processed by a CNN to produce propositional variables, what is the motivation for using visual data instead of directly symbolic inputs? It would be helpful for the authors to further clarify this design choice.

### Soundness
2

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
3

### Summary
The paper proposes (i) a new language that extends propositional logic with a choice operator, (ii) a compilation procedure to transform any LoH formula into a differentiable computational graph and (iii) a unifying viewpoint of NeSy paradigms.

### Strengths
The paper presents a very interesting method which appears to be novel

### Weaknesses
The notation seems a bit imprecise, especially in the definition of the Logic of Hypothesis language especially from the main text. Below I list some questions/imprecisions that I think need some improvement: 

1. Shouldn’t the grammar include $\top$ and $\bot$ to make propositional formulas well-formed? Otherwise the inductive definition has no base case except variables. Is this intended? If so why? and where is this specified? The authors write that the choice operator can take any finite number of formulas. What would [] mean? 

2. Connected to the above: the authors at the bottom of page 4 write that "LoH is flexible enough to encode any finite set of propositional logic formulae" how would you encode $\{\top\}$? 

3. The authors never formally define the concept of hypothesis space 

4. The LoH formulas are sometimes interpreted as a set and sometimes as a formula in R1-R5. I think this makes the definitions imprecise

5. I am really struggling to see why you introduce the choice operator if then sometimes you treat it as conjunction and sometimes as disjunction. Wouldn't it be cleaner to treat the problem in propositional logic and simply compile it from there? What is the real advantage of having this operator? 

6. I struggle to follow example 3: if $\phi$ are LoH formulas it means we can substitute $\phi$ $[a,b]$ and hence the authors seem to imply that on the ground of how you write a formula you get different interpretations. How is that possible? I think the authors defer this choice to Appendix B - however, it remains unclear from the main text. 

Regarding the experimental analysis: 

1. It is not clear how many rules this method is extracting for each dataset and how this compares with the others method cited. For example how many rules does MLLP extract and how many does this method extract? 

2. It would be very nice to include some examples of the obtained rules

3. On the tabular benchmarks it is not clear to me why one should use this method rather than the MLLP - what is the practical advantage?

4.  A big selling point of the method is that it can be stacked up on top of any neural network - why did the authors only perform experiments on such simple datasets? Did you try to use it on some dataset on which some constraints are known (e.g., CIFAR 100) and see how many you recover?

### Questions
See above

### Soundness
2

### Presentation
1

### Contribution
2
