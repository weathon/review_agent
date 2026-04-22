# Latent Planning Emerges with Scale

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 8, 4, 4

## Abstract
LLMs can perform seemingly planning-intensive tasks, like writing coherent stories or functioning code, without explicitly verbalizing a plan; however, the extent to which they implicitly plan is unknown. In this paper, we define *latent planning* as occurring when LLMs possess internal planning representations that (1) cause the generation of a specific future token or concept, and (2) shape preceding context to license said future token or concept. We study the Qwen-3 family (0.6B-14B) on simple planning tasks, finding that latent planning ability increases with scale. Models that plan possess features that represent a planned-for word like *accountant*, and cause them to output *an* rather than *a*; moreover, even the less-successful Qwen-3 4B-8B have nascent planning mechanisms. On the more complex task of completing rhyming couplets, we find that models often identify a rhyme ahead of time, but even large models seldom plan far ahead. However, we can elicit some planning that increases with scale when steering models towards planned words in prose. In sum, we offer a framework for measuring planning and mechanistic evidence of how models' planning abilities grow with scale.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
Investigates whether LLMs engage in latent (not explicitly generated) planning and shows that planning ability grows with model scale (in the Qwen-3 family).

Contributions

* Provides a causal definition of latent planning  -- planning is an internal representation that (i) causes the model to produce a specific future token for forward planning and (ii) causes generation of a context that makes that token more likely. This improves on purely observational/probing definitions.
* Provides simple agreement tasks as planning probes: On a/an, is/are, and el/la tasks, larger Qwen-3 models reliably plan ahead for the content word and use that to choose the right function word. Smaller models show nascent but incomplete mechanisms.
* Mechanistic evidence via transcoder feature circuits that identify “planning features” that represent the future word and show, through interventions, that ablating them hurts performance and boosting them helps, indicating genuine causal relevance.
* Causal-mech interpretability recipe for monitoring such emergence of forward planning and backward planning ability in  open models.

### Strengths
* Originality: Introduces a causal definition of latent planning that distinguishes between forward (goal-directed token production) and backward (context-shaping) planning. This causal approach is  novel in how it rephrames what “planning” means for decoder models and correcting an overextension in prior work that equated decodability with intent. The integration of transcoder feature circuits with causal interventions is also a novel methodological synthesis, enabling verifiable mechanistic evidence rather than speculative probing.
* Quality: The experiments are rigorous and well controlled. The progression from simple grammatical-agreement tasks to rhyming and prose-steering scenarios is also structured well logically and empirically. The use of quantitative flow analysis within feature circuits adds an added layer of interpretability
* Clarity: Definitions are explicit, figures are clear and interpretable. The argument flows naturally from conceptual motivation to empirical validation. 
* Significance: The results establish that latent planning mechanisms emerge with scale and that forward planning precedes backward planning—an interpretable scaling law that contributes to our understanding of model cognition.

### Weaknesses
* In terms of planning, the paper over-indexes on short-range linguistic dependencies. Not clear if this scales to true multi-step reasoning or action planning.
* Limited to Qwen-3 series, which hurts generalization
* Experiments provide only limited support for backward planning, and the analysis of whether the generated context “licenses” the planned token is overly qualitative. :

### Questions
* Can you add a quantitative measure of contextual dependency?
*  Do you believe these results generalize to multi-step cognitive planning or compositional reasoning? Could this be shown?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper examines whether large language models (LLMs) perform latent planning, that is, internally representing and reasoning toward future tokens without explicit plans. Using Qwen-3 models (0.6B–14B), the authors find that forward planning emerges with scale while backward planning remains limited, offering a causal framework and mechanistic evidence via transcoder feature circuits.

### Strengths
- The paper provides a clear and causally grounded definition of latent planning that distinguishes genuine planning from mere correlational predictability.

- The study offers comprehensive scaling insights, showing how planning abilities gradually emerge and strengthen as model size increases.

- It links mechanistic interpretability to AI safety, highlighting how latent planning could relate to hidden goal pursuit or “scheming”, thereby extending the work’s broader relevance.

### Weaknesses
- The chosen tasks, such as a/an prediction and rhyming couplets, are synthetic and narrowly scoped, limiting the conclusions’ applicability to real-world reasoning or planning.

- The evidence for backward planning is weak and inconclusive, raising doubts about whether full planning mechanisms have truly been demonstrated.

- The study lacks cross-model comparison, as it focuses only on the Qwen-3 family, making it unclear whether similar phenomena occur in other model architectures.

- Some of the causal claims may be overstated, since interventions could affect correlated linguistic or contextual features rather than genuine planning representations.

### Questions
- Could the observed causal effects arise from correlated features instead of true planning representations?

- How would the proposed framework generalize to complex goal-directed or multi-step reasoning tasks?

- What is the relative contribution of instruction-tuning versus model scale in the emergence of latent planning?

- How might this causal framework be applied in AI safety monitoring to detect latent scheming or hidden goal formation?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper's central hypothesis is that latent planning is an emergent capability that increases with model scale. It seeks to answer (1) whether LLMs engage in a mechanistically verifiable form of latent planning, (2) how this ability can be defined and measured, and (3) how this capability scales with model size.

The methodology first establishes a strict, two-condition causal definition of latent planning, distinguishing it from prior observational or probing-based work. For an LLM to be "latent planning," it must possess an internal representation of a future goal (a token or concept $t$) that: 
1. Forward Planning: Causes the model to eventually generate $t$ (Condition 1). 
2. Backward Planning: Causes the model to generate a preceding context that licenses $t$ (Condition 2).

To identify these causal mechanisms, the authors employ Transcoder Feature Circuits, a mechanistic interpretability technique. This method decomposes a model's dense MLP activations into sparse, monosemantic (interpretable) features and identifies the causal sub-graph (the "circuit") that explains a specific behavior.1 The study is conducted on the Qwen-3 family of open-source models, ranging from 0.6B to 14B parameters.

### Strengths
1. The paper's greatest strength is its insistence on a rigorous, two-condition causal definition of latent planning. This elevates the study from a correlational observation to a test of a mechanistic hypothesis. This strength is powerfully underscored by the refutation of probing-based methods in Appendix G, which demonstrates that high probing accuracy can be causally irrelevant.
2. The quality of the core experiment is extremely high. The a/an and is/are tasks serve as an elegant "minimal pair" testbed for planning. The causal interventions (ablation and boosting) in Section 4.4 provide "smoking gun" evidence for the discovered planning circuit. The analysis in Appendix E, which surgically separates task-solving ability from planning ability, is a brilliant and crucial piece of analysis that solidifies the paper's claims.

### Weaknesses
1. The complete failure of the methodology on the el/la task (Appendix D) is a significant weakness. The authors' explanation—that "Qwen-3 is not highly capable in language besides English and Chinese" —is an ad hoc hypothesis. This failure could alternatively imply that the "planning" mechanism found is not a general-purpose planning module at all, but a highly specific and brittle circuit for English grammatical agreement. This possibility severely undercuts the generality of the paper's claims.
2. The paper repeatedly claims that smaller models (4B-8B) have "nascent planning mechanisms"  but "fail" the task. It is unclear what this means mechanistically. Does the circuit exist, but is weak? Are some features missing? Does the model have the 'accountant' feature but lacks the causal connection to 'an' ? This "nascent" concept is central to the "emergence" narrative but remains poorly defined.

### Questions
On local planning, They say X features are described as "sensitive" and found in a "small minority." Is this evidence of a real, generalizable mechanism, or an artifact of steering on specific, polysemantic features that happen to fire on common n-grams? How could this mechanism be tested more robustly?

### Soundness
3

### Presentation
3

### Contribution
4
