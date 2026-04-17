# The Kinetics of Reasoning: How Chain-of-Thought Shapes Learning in Transformers?

- Decision: Reject
- Scores: 2, 4, 4, 2

## Abstract
Chain-of-thought (CoT) supervision can substantially improve transformer performance, yet the mechanisms by which models learn to follow and benefit from CoT remain poorly understood. We investigate these learning dynamics through the lens of grokking by pretraining transformers on symbolic reasoning tasks with tunable algorithmic complexity and controllable data composition to study their generalization. Models were trained under two settings: (i) producing only final answers, and (ii) emitting explicit CoT traces before answering. Our results show that while CoT generally improves task performance, its benefits depend on task complexity. To quantify these effects, we model the accuracy of the logarithmic training steps with a three-parameter logistic curve, revealing how the learning speed and shape vary with task complexity, data distribution, and the presence of CoT supervision. We also uncover a transient trace unfaithfulness phase: early in training, models often produce correct answers while skipping or contradicting CoT steps, before later aligning their reasoning traces with answers. Empirically, we (1) demonstrate that CoT accelerates generalization but does not overcome tasks with higher algorithmic complexity, such as finding list intersections; (2) introduce a kinetic modeling framework for understanding transformer learning; (3) characterize trace faithfulness as a dynamic property that emerges over training; and (4) show CoT alters internal transformer computation mechanistically.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper studies how chain-of-thought (CoT) supervision changes the way transformers learn to reason in a controlled, synthetic setup. All experiments are built on a procedurally generated knowledge base (KB) consisting of entities, attributes, and binary relations, which provides the structured symbolic world for reasoning. Specifically, they train 12-layer GPT-2–style transformers from scratch on four symbolic tasks: Comparison (given entities and their attributes, decide which one ranks highest), sorting (order a list of entities by an attribute), intersection (find the entity common to multiple sets), and composition (apply a sequence of relations to infer a new entity). Each of these tasks provides a complexity parameter and an ID/OOD split. Here, ID examples use one set of entity names during training, while OOD examples use held-out entity names but the same structural forms and sequence lengths. In addition, each task has two output formats: (1) direct answer only, and (2) CoT-guided output that includes intermediate steps and the answer. 

The authors find that models achieve perfect ID performance with both output formats, but CoT enables OOD generalisation on two tasks (sorting, composition) where non-CoT does not generalise. For intersection, both formats fail to generalise. Next, the authors attempt to quantify learning dynamics by fitting the OOD accuracy curves (in log training steps) with a 3-parameter logistic model. Those parameters are: the ceiling accuracy (the asymptotic OOD performance once learning has converged), the take-off point (the log training step where accuracy reaches half of the ceiling accuracy), and a slope parameters. They find that the extracted parameters behave systematically with task difficulty and the data ratio. They interpret these shifts as evidence that task difficulty increases the effective barrier to generalisation, while more data and CoT supervision make models generalise earlier and reach higher final OOD accuracy. 

The authors also track answer and trace accuracy during training and observe what they term the „unfaithfulness gap“: a transient unfaithfulness phase where the answers are correct but the traces incorrect, before traces and answers eventually align. Finally, the authors perform linear probing (predict answer given hidden state) and activation patching to get insights into the computations of the models. They find that the CoT models encode information about the answer much earlier in the model, suggesting that CoT supervision shapes internal computations.

### Strengths
- The experiments isolate the effect of chain-of-thought supervision by training identical transformers with and without CoT traces on the same symbolic tasks. 
- Fitting OOD accuracy over log training steps with a logistic model an interesting idea to quantify learning speed and saturation. This kinetic framing is novel.

### Weaknesses
- The paper’s “OOD” split tests whether models generalise to new entity names under the same reasoning depth, but not to longer or more complex reasoning chains. This setup mainly compares memorisation against symbol abstraction. Since CoT is theoretically expected to help with length generalisation [1], it would have been much more interesting to evaluate generalisation to increased reasoning depth.
- The observed generalisation improvements of CoT over non-CoT may reflect a shift from memorising entity-answer pairs to learning reusable sequence patterns. In other words, CoT might simply force the model to attend to the relational structure of the trace rather than to specific symbols, which naturally enables transfer to unseen entities. It would therefore be interesting to see whether CoT enables algorithmic induction or just lexical abstraction (e.g., by testing extrapolation to unseen reasoning depths). 
- The „faithfulness“ analysis appears to define trace correctness as exact match with the ground-truth trace. This would mean that even minor formatting difference (e.g. swapping order, missing comma, or semantically equivalent but syntactically different steps) count as unfaithful. Consequentially, the observed „unfaithfulness phase“ might just reflect formatting inconsistencies rather than truly incorrect reasoning. Including qualitative examples of mismatched traces or alternative metrics would help clarify this. 
- In the causal tracing experiments, the observed computational pattern in the CoT model (Fig. 7) is interpreted as evidence that they adopt the „correct computational structure“. However, the tracing plots for CoT Comparison in Fig. 6 are noisy and not clearly smooth, despite achieving near-perfect generalisation. It is therefore unclear why the pattern is evidence for algorithmic correctness and appears like a post-hoc conclusion. 

[1] W. Merrill and A. Sabharwal, ‘The Expressive Power of Transformers with Chain of Thought’, in The Twelfth International Conference on Learning Representations, 2024.

### Questions
- Have you evaluated whether CoT also improves length or compositional generalisation?
- In the faithfulness analysis, how often are traces “unfaithful” because of minor formatting errors versus semantically wrong reasoning? Could you provide qualitative examples of what constitutes an incorrect trace?
- Can your kinetic “laws” be used predictively? That is, given task complexity or data ratio, can you forecast the approximate take-off step t_0 or ceiling L? Or are these relationships purely descriptive? If predictive, showing out-of-sample fits or held-out task extrapolations would make this argument much stronger.

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
4

### Summary
This paper investigates how transformers learn under chain-of-thought (CoT) supervision by analyzing learning dynamics through the lens of grokking. The authors pretrain transformers on a suite of symbolic reasoning tasks (COMPARISON, SORTING, COMPOSITION, INTERSECTION) and compare CoT-guided training (producing intermediate reasoning traces) to direct-answer training. They propose a kinetic framework that models out-of-distribution (OOD) accuracy as a function of log training steps using a three-parameter logistic curve, capturing learning speed and saturation. The analysis reveals that CoT can accelerate generalization, especially for tasks of moderate algorithmic complexity, though both CoT and non-CoT models eventually fail on more complex reasoning tasks. The paper also identifies a transient phase of “trace unfaithfulness,” where models produce correct answers but inconsistent intermediate traces, and uses probing analyses to relate CoT to shifts in internal computation.

### Strengths
•	The idea of fitting OOD accuracy over log training steps with a three-parameter logistic model to quantify learning speed and saturation is novel and interesting.
This kinetic framing offers a new lens to analyze how CoT affects learning dynamics and the grokking transition.

### Weaknesses
•	Intermediate accuracy metric definition: I understand the intermediate accuracy to be defined as a “trace overlap” measure matching the ground-truth trace to the predicted trace. However, many tasks can be solved using different but valid traces. This metric disregards all expedient but non-identical traces, potentially overstating the “knowledge gap” or “unfaithfulness phase.” A more nuanced metric accounting for alternative valid reasoning paths would strengthen the claims.
•	Limited evaluation scope: All experiments use a single transformer architecture (12-layer GPT-2 style) and a single random seed. This narrow evaluation is insufficient to support the general claims about CoT-induced learning dynamics and kinetics. Even if trends appear robust, more empirical evidence (e.g., across seeds and architectures) is needed.
•	Inconsistent measurement axes: It is unclear why training steps are used for some comparisons while FLOPs are used for others (e.g., Figure 3). The authors should clarify whether these metrics are interchangeable or whether differences in scaling affect the observed kinetic fits. The paper reports a significant drop in OOD performance between the COMPOSITION and SORTING tasks despite similar accuracy-over-log-step curves. The connection between the kinetic fit and this performance difference is not clearly explained.
•	Readability and structure: The paper is dense and often difficult to follow. The narrative flow between theoretical framing, kinetic modeling, and empirical evidence is not always clear, making it hard to track the central argument.

### Questions
1.	Does the “intermediate accuracy” metric consider different syntactic but semantically equivalent reasoning traces? If not, how might this affect the interpretation of unfaithfulness?
2.	In Figure 3, are the indices of the ϕ labels identical to the task parameters listed in Table 3?
3.	Can you clarify the meaning and practical interpretation of the data ratio (ϕ)?
4.	Why are training steps used for comparison in some plots but FLOPs in others (e.g., Figure 9)? Are the fitted curves consistent across both scales?
5.	Could the “knowledge gap” phenomenon be partially explained by the existence of multiple correct traces that differ from the ground-truth trace?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper studies small transformers on controlled symbolic tasks and claims that CoT supervision accelerates “grokking,” fits OOD accuracy with a 3‑parameter logistic curve in log‑steps, and interprets second‑order trends via an Arrhenius‑style “barrier/temperature” analogy. Empirically, CoT helps on COMPARISON and SORTING but not INTERSECTION; “unfaithfulness” (answers correct while traces are wrong) appears mid‑training; a Mamba control fails to OOD‑generalize.

### Strengths
1. Clear, controllable testbed \& measurements. The tasks vary algorithmic difficulty $k$ and data ratio $\phi$; the paper cleanly reports zero-/few-shot OOD accuracy and shows CoT often improves sample-efficiency (incl. a FLOPs view).
2. Architecture control. A matched-size Mamba fails to OOD-generalize while transformers succeed under CoT, highlighting inductive-bias differences.

### Weaknesses
1. Limited novel understandings. That CoT accelerates learning and improves expressiveness, and that intersection-like composition is hard, all broadly echo prior CoT understanding. The main new angle is the curve-fitting/Arrhenius narrative, which remains largely phenomenological. The Arrhenius analogy is not stress-tested (no explicit estimation of $\Delta$ or $T_{\text {eff }}$; validation reduces to trending $\hat{r}$ via Eq. (6)).
2. Kinetic self-consistency is shaky. The paper fits a logistic in log-time (Eq. 4) yet explains dynamics via a linear-time logistic ODE and then maps the fitted slope back to a "rate" (Eq. 6). This mixes time scales and treats a log-time shape parameter as a linear-time rate without a principled derivation.
3. "OOD" is closer to compositional recombination than true OOD. Training always includes all atomic facts (ID and OOD) and holds out only composed $k$-hop examples for OOD; test is "compositions of known atoms," not unseen entities/attributes. This inflates how far the conclusions generalize.
4. Evaluation choices may blur compute/efficiency claims. FLOPs are proxied by token counts ("relative FLOPs"), and plots lack error bars/seed variation; some tasks use partial-credit answer scoring while "full-sequence" requires exact trace+answer, complicating cross-task comparisons and the magnitude of the "unfaithfulness" gap.

### Questions
See weaknesses

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
The paper studies learning dynamics and generalization of algorithmic tasks with LLMs that use either direct prediction or CoT on four synthetic tasks (intersection, sorting, comparison, composition). It is found that CoT generalizes better, has a phase transition, and reasoning unfaithfulness is reduced as training progresses.

### Strengths
Writing:
- The paper is written clearly and is easy to follow.

Conceptual:
- The paper (re-)discovers a few interesting facts, namely reasoning unfaithfulness, CoT helping in generalization and a grokking transition.
- The experiments cover a number of challenging (although synthetic) reasoning tasks and are performed correctly.
- The interpretations of the training curves are interesting, seem plausible and give insight into the training behavior.

### Weaknesses
Contribution:
- The paper basically does a number of experiments and reports training curves on them.
- The take home message is, I would assume, well known lore in the deep learning community, including grokking, CoT faithfulness and CoT generalizability. No new insight has been produced.
- The overall setup is highly artificial with only synthetic tasks trained on an LLM from scratch. I am not sure that these findings would translate to pre-trained LLMs that might already exhibit different behaviors.
- The learning curves that are fitted are nice, but again I do not see the significance in them for a wider audience. In order for them to play a similar role to scaling laws in LLM training one would have to train on a wider range of realistic training tasks on pre-trained LLMs. As they are, they are just post-hoc fitting of training dynamics.
- The mechanistic interpretability results feel like a random addition to the main paper topic and consist of standard application of linear probing and causal tracing.

### Questions
- What is the significance of your training findings outside the synthetic reasoning tasks for real-world pre-trained LLMs and real reasoning tasks?

### Soundness
3

### Presentation
3

### Contribution
2
