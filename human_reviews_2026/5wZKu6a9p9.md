# Poisoning the Inner Prediction Logic of Graph Neural Networks for Clean-Label Backdoor Attacks

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Graph Neural Networks (GNNs) have achieved remarkable results in various tasks. Recent studies reveal that graph backdoor attacks can poison the GNN model to predict test nodes with triggers attached as the target class. However, apart from injecting triggers to training nodes, these graph backdoor attacks generally require altering the labels of trigger-attached training nodes into the target class, which is impractical in real-world scenarios. In this work, we focus on the clean-label graph backdoor attack, a realistic but understudied topic where training labels are not modifiable.
According to our preliminary analysis, existing graph backdoor attacks generally fail under the clean-label setting. Our further analysis identifies that the core failure of existing methods lies in their inability to poison the prediction logic of GNN models, leading to the triggers being deemed unimportant for prediction. Therefore, we study a novel problem of effective clean-label graph backdoor attacks by poisoning the inner prediction logic of GNN models.
We propose BA-Logic to solve the problem by coordinating a poisoned node selector and a logic-poisoning trigger generator.
Extensive experiments on real-world datasets demonstrate that our method effectively enhances the attack success rate and surpasses state-of-the-art graph backdoor attack competitors under clean-label settings. 
Our code is available at https://anonymous.4open.science/r/BA-Logic.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper focuses on the problem of clean-label backdoor attacks against Graph Neural Networks (GNNs). Addressing the limitation of existing graph backdoor attacks—where modifying training labels renders them impractical in real-world scenarios—it proposes a novel attack paradigm centered on "poisoning the inner prediction logic of GNNs" and designs the BA-LOGIC framework.

### Strengths
1. The paper establishes a solid theoretical framework to unravel the core challenge of clean-label graph backdoor attacks.

2. BA-LOGIC’s two-core component design (uncertainty-guided poisoned node selector + adaptive logic-poisoning trigger generator) addresses key practical challenges of clean-label attacks while ensuring effectiveness, and comprehensive experiments prove the feasibility of BA-LOGIC.

### Weaknesses
Lack of adaptive defenses.

### Questions
If defenders are aware of the BA-LOGIC attack, what adaptive defense strategies can be employed to mitigate its impact?

### Soundness
3

### Presentation
3

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
This paper proposes a novel clean-label graph backdoor attack technique, BA-LOGIC, focusing on poisoning the inner prediction logic of Graph Neural Networks (GNNs) without modifying training labels. The work introduces a logic-poisoning trigger generator and advanced poisoned node selection, demonstrating state-of-the-art attack success rate and robustness across datasets, models, and defense methods.

### Strengths
- The clean-label attack formulation addresses a realistic scenario where attackers cannot tamper with ground-truth labels, highlighting a meaningful threat for deployed GNNs.
- Theoretical analysis and ablations help clarify why prior techniques underperform and justify the proposed logic-poisoning approach.

### Weaknesses
- The contribution builds incrementally on established ideas in trigger design and node selection, with technical optimization at the core instead of a radically new threat model.
- Reliance on surrogate models and full node feature access may limit realistic, black-box or large-scale attack scenarios.
- Comparative defense analysis lacks depth, especially regarding future defenses specifically countering logic-poisoning.

### Questions
1. To what extent does BA-LOGIC’s attack success depend on graph properties such as high heterophily, noisy node features, or severe class imbalance—does performance deteriorate in these challenging settings?

2. Is there experimental evidence that straightforward explainability regularization or gradient masking can successfully defend against the core logic poisoning strategy, and how robust is the method to such countermeasures?

3. How practical is uncertainty-based poisoned node selection when training node labels are incomplete, partially unavailable, or highly noisy in real graph data?

4. Are there defense techniques (beyond edge pruning) that specifically target logic-poisoning triggers, and how detectable is BA-LOGIC under adaptive defenses?

5. Could collaborative or distributed defense strategies—such as multiple GNN models jointly monitoring subgraph behavior—mitigate the risk of clean-label logic-poisoning attacks?

6. How does the time and computational complexity of BA-LOGIC compare to both competing attacks and real-world defense operation requirements?

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
3

### Summary
The paper studies clean-label backdoor attacks on graph neural networks. The attacker cannot modify labels and must rely on adding small structure or feature triggers to a subset of training nodes or graphs. The authors argue that many prior backdoor attacks transfer poorly to this setting because the model relies on clean neighborhood cues instead of the trigger. The paper proposes a method that selects which samples to poison and trains triggers so that the model’s internal decision logic places high importance on the trigger. Experiments cover several datasets and backbones, with high attack success and small clean accuracy drops.

### Strengths
1. Writing is clear and easy to understand.
2. The problem setup is realistic for clean-label constraints. The objective connects well to the intended failure mode.
3. Broad empirical study across tasks, models, and several defenses. Attack transfer looks strong while keeping clean accuracy high.

### Weaknesses
1. The work investigates a new setting but the novelty is limited. Several components resemble prior methods for trigger generation or importance shaping. Please position the method more sharply against the closest graph backdoor baselines and explain what is new at the objective level and at the algorithmic level.
2. No adaptive defense is proposed. Since the objective pushes importance onto small subgraphs, an adaptive baseline that penalizes gradient concentration or uses randomized trigger-edge masking during training would be informative.
3. Theoretical assumptions may not hold on real graphs with strong feature and structure correlations.

### Questions
1. Could author help me understand what is the key difference between UGBA-C and your method in the clean-label case?
2. If labels cannot be changed, in what exact sense is this a backdoor rather than poisoning attack? Is there a formal or operational line the paper uses to separate the two?
3. Why do methods designed for clean-label settings such as ERBA and ECGBA underperform EBA-C, GTA-C, UGBA-C, or DPGBA-C in your tables? Please provide an analysis.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a method to poison a training corpus without needing to flip the label for the data with the injected trigger (so-called clean label setting) but using a surrogate model to tune perturbations to training data (injecting the suitable structural changes or trigger sub-graphs) to data.

### Strengths
- Consider the more challenging setting of clean label settings (this relaxes attacker threat model)
- Evaluation across node/graph/edge classification tasks with multiple benchmark datasets
- Evaluation with a number of defences
- Consider different combinations of surrogate architectures vs. target architectures

### Weaknesses
- Clean label backdoor attacks against Graph models and the idea of generating triggers (sub-graphs) via a surrogate is not new
- The value of the theorem to the method is unclear
- Clarify the threat model in the main paper
- Unclear if input sanitization type defences were considered

### Questions
- How is the theorem related to the poisoning method? I am unclear about the value of this other than what is already shown, i.e the injected trigger is less effective. I would honestly remove this.

- Fig 8 says BA-Logic (proposed) is lower in the scores in 1 and higher in the other - this seems to contract the statements in the paper, what am I missing? Can you show the same results for other datasets and clean models? This seems more important as the thesis for the study is based on the ineffectiveness of other clean label methods as demonstrated in the ITR measure.

- Clarify threat model (attackers exert control over training or simply poison data and publish for use by a victim?)

- Explain why existing defences do not work
 
- Did the authors include input sanitization type defences? (Would be good to categorise the defences considered to state which focus on inputs, models and outputs to remove/clean/purify models/inputs.

I will re-consider the scores after the rebuttal, generally, I remain positive about the study.

### Soundness
3

### Presentation
3

### Contribution
2
