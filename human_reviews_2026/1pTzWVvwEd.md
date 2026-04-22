# Incremental Learning in Transformers for In-Context Associative Recall

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 8, 2

## Abstract
Transformers acquire in-context learning abilities in abrupt phases during training, often unfolding over multiple stages, during which certain keys circuits like induction heads emerge. In this work, we characterize the dynamics behind the emergence of such circuits during these stages. We focus on a synthetic associative recall task, where sequences are drawn from random maps between a permutation group and a vocabulary range and the model is required to complete the mapping of a permutation by retrieving it from the context. On this task, we study the trajectories of gradient flow of a simplified two-layer, attention-only transformer. Leveraging symmetries in both the transformer architecture and the data, we derive conservation laws that guide the dynamics of the parameters. These conservation laws crucially reveal how initialization —both in shape and scale— determines the order of learning as well as the timescales over which such circuits emerge revealing the implicit curriculum. Finally, we provide empirical evidence across different architectural choices, validating  our simplifications and generalizing the insights from our analysis beyond the simple setting.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper investigates how a small, two-layer attention-only transformer learns an n-gram recall in-context learning (ICL) task, where the model must retrieve the correct output by matching the final \(n=4\) tokens of a cue seen earlier in the same sequence. Building on previously proposed attention-based solutions for n-gram recall (Varré et al.), the paper constructs a minimal model that implements those solutions. This model admits a fully analytic treatment under gradient flow and captures the learning behavior observed in practice. Experiments training the model are consistent with the analytic predictions.

### Strengths
1. The motivation for introducing a minimal transformer model is clear. The proposed minimal model capture n-gram recall ICL solution.

2. The minimal transformer model is fully analytic under gradient flow and yields closed-form training dynamics. Analyzing these dynamics reveals a conservation law, initialization dependence, and phase ordering

### Weaknesses
1. The abstracted transformer architecture may be too minimal. While recent work aims to reduce complexity without losing fidelity, it remains unclear how much intuition from this abstraction transfers to real LM systems.

2. The focus is limited to n-gram ICL, a setting already known to work as described; the incremental insights appear modest.

3. Experiments are conducted in a specific setting without varying key factors such as \(N\) (the n-gram length) or architectural choices.

### Questions
1. Ablations (A1--A4): How do ablations of A1, A2, A3, and A4 affect empirical training outcomes? Does the phenomenology seen in the full simple model persist under these ablations?

2. Beyond n-gram ICL: How can the findings be extended to tasks outside n-gram recall? In particular, some classes of algorithmic ICL tasks (e.g., linear regression) have known transformer implementations (e.g., Lu et al., 2025). How can the paper’s findings be generalized to that class?

3. Timing of transitions: Does the model predict the time points of performance ``jumps''? If so, do these predictions match experimental results?

4. Initializations: Can you demonstrate how varying initialization affects the results in Figure 3?

References
[1] Varre, A., Yüce, G., Flammarion, N. “Learning In-context n-grams with Transformers: Sub-n-grams Are Near-stationary Points.” 2025

[2] Lu, Yue M., et al. "Asymptotic theory of in-context learning by linear attention." Proceedings of the National Academy of Sciences 122.28 (2025): e2502599122.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper analyses how in-context learning circuits (e.g. induction heads) emerge during training in transformers.
To investigate this, the authors design:
* a controlled in-context associative recall task where the model must retrieve the correct response to a query by matching that query to an earlier key-value pair in the same prompt, and
* a heavily simplified 2-layer attention-only transformer where most parameters are fixed and only a few scalar/logit parameters are learned.

In this setting, they derive continuous-time gradient flow dynamics and prove that learning proceeds in stages: the model sits on plateaus where a partial recall circuit exists (e.g. can match only part of the query), then undergoes sharp “jumps” to a more complete recall mechanism. They link these jumps to saddle escape and show a conserved quantity that couples parameters across layers.
They further argue that which attention head “activates” first is determined by initialization scales, implying that heads specialize sequentially rather than all at once. Small experiments on order-4 recall show plateau to jump loss curves and sequential head specialization consistent with the theory.

### Strengths
1. Mechanistic training dynamics.
The work gives a concrete dynamical story for how in-context recall circuits form over time, including why capabilities appear in sharp jumps instead of smoothly.

2. Elegant task/model co-design.
The associative recall task and the simplified transformer still capture key ingredients of induction-style retrieval (causal attention, multi-head structure), but are analysable enough to get closed-form ODEs and a conservation law.

3. Theory explains qualitative phenomena people see in practice.
The staged plateaus, sudden capability jumps, and head specialisation order are all things observed in real small transformers; here we get a mechanistic account (ordered head activation driven by initialization scale, long dwell times near saddles).

4. Potential practical implication.
The result that init scales $\beta_h$ bias which head learns first suggests we can steer emergent specialization and maybe training curricula just by tuning initialization, which is actionable.

### Weaknesses
1. Gap to realistic models.
The theoretical guarantees rely on a very stylised transformer: fixed value structure, no MLPs, one-hot tokens, continuous-time gradient flow on full-population loss, etc. It's unclear how directly the results carry over to standard GPT-style models trained with SGD on natural data. The paper argues informally that the story should generalize, but does not really show it.

2. Light empirical support.
Experiments are on a tiny recall task with tiny vocab, and are mostly qualitative (plots of plateau/jump behavior, sequential $\beta_h$  growth). There’s little quantitative matching between the theoretical predictions (e.g. plateau duration scaling, ordering of head activation by init magnitude) and actual measured numbers.

3. SGD / noise is under-discussed.
The analysis assumes noiseless gradient flow. In practice, SGD noise helps models escape saddles. If saddle escape timing is central to the staged-learning story, then we need at least a discussion (or small ablation) of how noise affects plateau lengths and ordering.

4. Accessibility / reproducibility.
Some of the most interesting claims (like the conservation law and the staged head activation sequence) depend on fairly dense math and on training details that aren’t fully specified in the experimental section. This makes it harder for a broad ICLR audience to verify or reproduce.

### Questions
1. Generalization to standard transformers:
If you allow a more realistic transformer (trainable value matrices, MLPs, standard CE loss, SGD noise), do you still see the same staged head activation and ordered specialization? Have you run even small ablations in that more general setting?

2. Effect of SGD noise:
Your analysis is in deterministic gradient flow on the population loss. In actual training, SGD noise is often viewed as the mechanism that helps leave plateaus. Do you expect noise to (a) only change the timing of the jumps, or (b) potentially reorder which head “wins” first? 

3. Quantitative match to theory:
You argue that the model dwells near partial circuits for long periods and then jumps. Can you report measured plateau durations vs. the theoretically predicted scaling (e.g. O(1/ϵ)) and show how close they are?

4. Scaling the task:
The associative recall task is “clean”: the answer always appears in the prompt, there’s exactly one correct continuation, and there’s no ambiguity. How do the dynamics change if the model has to generalize from incomplete evidence or noisy matches (i.e. more like natural text)?

5. Conservation law intuition:
The conserved quantity tying together first- and second-layer parameters is one of the most interesting parts of the paper. Can you give more geometric or mechanistic intuition for what it “means” operationally, in a way that a practitioner could try to measure in a non-simplified model?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper focuses on understanding the dynamics of emergence of mechanisms for in-context tasks. In particular, they characterize the emergence of circuits for an in-context recall task in a two-layer Transformer, and demonstrate that shape/scale of initialization can influence the timescales on which circuits are learned.

### Strengths
* The "methodological" contribution of the paper, i.e. understanding how internal mechanisms are influenced by optimization pressure, is extremely strong and very valuable, and will I think encourage interesting discussion.
* The work is theoretically and methodologically sound.
* The paper is very well written, figures are super helpful.

### Weaknesses
* I think (as with any work of this kind) there are some significant limitations in terms of realism of the setting: e.g. the paper focuses on a very small Transformer, most of the analysis is done in the vanishing scale of initialization, the synthetic task might not generalize to things we actually care about.

### Questions
See weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
In this paper, the authors consider an in-context associative recall task trained using a simplified two-layer attention-only network. The main contribution is a quantitative description of the learning dynamics based on key parameters, which include the positional bias and diagonal elements of the key-query matrices.

### Strengths
The paper makes a series of simplifications to obtain insight into the training dynamics of a transformer. The simplifications are phenomenology-driven: by considering the key parameters that make up the sub-circuit that performs associative recall, they obtain a simplified description of the training dynamics that yet captures most elements of sub-circuit formation. I believe this is a strength of the paper -- the analysis provides some insight in scenarios where an exact solution cannot be derived.

### Weaknesses
In my opinion, the paper has three major weaknesses:

First, while the authors derive a series of mathematical expressions, none of these calculations are quantitatively tested in numerical simulations (except for a qualitative one in Figure 3). This seems like a missed opportunity to connect theory to experiment, which is noteworthy here because the simulations are not computationally expensive and the authors presumably already have an implementation (as evidenced by Figure 3, 4). Moreover, none of the details of the experiments are presented, which makes it impossible to interpret Figure 3,4. It is not clear whether the experiments were performed using a standard two-layer model or the simplified model. Without these experiments, it is impossible to say whether the theory the authors have written down using the simplified model accurately captures the dynamics of a full transformer model. Are there specific predictions that the theory offers which can be tested using numerics?

Second, there is a key positivity assumption made in assumption A1 (page 5 bottom). Here the query-key product is expressed in terms of \beta^2 rather than \beta. However, there is no constraint in a full transformer that imposes positivity of the diagonal entries, so it is unclear why one should assume positivity in the simplified model. How does relaxing the positivity assumption on \beta^2 change the analysis?  
This is an important point, because if \beta^2 were replaced with \beta, one would find a saddle point close to initialization: this is apparent when one writes the expression for the ODE for \beta_1 (Theorem 3.1) in terms of \beta_1^2 rather than \beta_1. That is, the dynamics will flow to different basins depending on how \beta and \alpha are initialized. A similar phenomenon has been shown to occur in previous analyses. For example, see eqns 7-9 in https://openreview.net/forum?id=INyi7qUdjZ, who also derive the training dynamics of the induction head circuit in terms of effective parameters. In that setting, the saddle point is absent when one takes into account randomness in sampling input examples. 

Third, the authors seem to be assuming that all (k+1)! permutations are shown in the context. This is an unrealistic assumption. I understand that this simplifies the analysis, but as pointed above, the randomness in sampling an input can indeed matter. It is unclear how this assumption will impact the generality of the results.

### Questions
Please see the questions raised in the weaknesses section above. In my opinion, the first point regarding the lack of numerical tests is sufficiently serious to warrant rejection -- at this stage, the theory has no empirical grounds on which it has firm footing, though this is of course not unresolvable.

### Soundness
3

### Presentation
4

### Contribution
3
