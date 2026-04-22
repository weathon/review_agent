# Evolution of Concepts in Language Model Pre-Training

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 8, 2

## Abstract
Language models obtain extensive capabilities through pre-training. However, the pre-training dynamics remains a black box. In this work, we track linear interpretable feature evolution across pre-training snapshots using a sparse dictionary learning method called crosscoders. We find that most features begin to form around a specific point, while more complex patterns emerge in later training stages. Feature attribution analyses reveal causal connections between feature evolution and downstream performance. Our feature-level observations are highly consistent with previous findings on Transformer's two-stage learning process, which we term a statistical learning phase and a feature learning phase. Our work opens up the possibility to track fine-grained representation progress during language model learning dynamics. Our code is available at https://github.com/OpenMOSS/Language-Model-SAEs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper separates the pretraining process of large language models (LLMs) into two distinct phases: a statistical learning phase and a feature learning phase. The authors argue that downstream performance improvements are closely tied to the emergence of features during the latter phase. To quantify this, they introduce a “crosscoder” network that detects features in activations with respect to a so-called decoder norm and track the development of this norm during pretraining. The paper presents empirical results linking these phases to downstream task performance.

### Strengths
1) The idea of analyzing the emergence of features during pretraining is interesting and aligns with the current interest in the mechanistic interpretability of LLMs.

2) The paper attempts to link internal feature emergence in the model to downstream generalization, which is a valuable direction.

### Weaknesses
1) The decoder norm is part of the training objective, and the weights in the norm are (obviously) part of the model that constructs the feature.
I find this quantity borderline to abstract to serve as a metric that should verify the central concepts of this paper.

2) Figures 4–6 are central to the main claims but are difficult to interpret, as the x-axis is on a logarithmic scale representing training steps.
The authors claim that two distinct learning phases appear, but on a linear scale, this might resemble a single continuous (albeit noisy) learning process. The choice of log-scale seems to visually exaggerate the phase transition; this requires at least discussion and justification. As a remark: on a linear scale, the "statistical phase" is less than 1% of training time.

3) Figure 5 introduces an additional LLM for the interpretation of features, which adds another layer of complexity and potential confounding effects. For me, the interpretation of LLMs should ideally come without LLMs to avoid a circle.

4) Minor remark: Including a figure before the abstract is unconventional.

### Questions
1) Why did you choose a log-scale for presenting all results?

2) As all experiments use Pythia, I would like to see a discussion of the training algorithm used there. How many warm-up steps were used? When did the learning rate change?   There is a (small) chance that the seen effects may be trivial if they can be related to, for example, a learning rate change.

3) The y-axis states a "normalized decoder norm". What is normalized exactly? It appears to me that each feature reaches one at some point, which may again make the results less surprising, as each feature must either be present at the start or emerge at some point if one is eventually reached.

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
1

### Summary
This paper investigates how internal features in large language models emerge and evolve during pre-training. Using cross-snapshot crosscoders—a variant of sparse autoencoders adapted to align activations across training checkpoints—the authors track interpretable features throughout the training process of Pythia models. They identify initialization and emergent features, quantify their persistence, and show that feature complexity increases with training progress. Attribution-based analyses link these microscopic feature dynamics to downstream task improvements. The study further reveals a clear two-phase transition in pre-training, from an early statistical learning phase dominated by token-level regularities to a later feature learning phase characterized by sparse, semantically rich representations.

### Strengths
The paper presents a technically novel adaptation of crosscoders for temporal analysis, offering the first fine-grained view of feature evolution across pre-training snapshots. The experiments are extensive and carefully controlled, combining mechanistic interpretability with learning dynamics. The empirical discovery of a statistical-to-feature-learning transition provides a coherent mechanistic complement to theoretical frameworks such as the information bottleneck and singular learning theory. The work stands out for its methodological clarity, strong empirical validation, and relevance to both interpretability and representation learning research.

### Weaknesses
The study provides interesting descriptive evidence, but the causal claims remain largely correlational—attribution patching is still a heuristic rather than proof of necessity.
The analysis is limited to Pythia checkpoints and relatively simple syntactic tasks, so generalization to larger or more complex models is unclear.
Some methodological choices (e.g., decoder-norm as feature-strength proxy, selection of snapshots) lack formal justification or ablation.
The paper is also quite engineering-heavy, and clarity could improve in conveying what new conceptual insight the cross-coder adds beyond existing SAE frameworks.

### Questions
How sensitive are the results to the number or spacing of snapshots?

Could decoder-norm scaling artifacts or feature-splitting affect the interpretation of “feature emergence”?

Does attribution patching preserve causal validity when features are highly correlated?

Have the authors verified that the same cross-coder features generalize across layers or different datasets?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors study the evolution of features during LLM training using crosscoder SAEs. They first study the emergence, persistence of features in LLMs, showing that many features persist for long periods and that their emergence time varies. They then consider feature complexity, showing a moderate correlation between feature complexity and the time at which the feature's emergence peaks. After analyzing feature dynamics, they study the causal relationship between features and task performance, using attribution patching to assess causal relationships for subject-verb agreement, indirect object identification, and induction tasks. Finally, they show that the initial part of training learns statistical features of token distributions and study the changes in feature dimensionality during training.

### Strengths
The authors introduce a novel crosscoder method for studying feature dynamics in LLMs. They analysis of feature emergence and causal study is clear and thorough.

### Weaknesses
Figure 3c is a bit difficult to interpret, since the analysis references the number of training steps but the figure is labeled in terms of snapshot number. Since there is a stratified sampling approach to selecting snapshots (and since the Pythia models use log-spaced snapshots + snapshots every 1000 training steps), it's not obvious which training step corresponds to which snapshot in the figure and hard to verify the accuracy of the claims. 

The argument about feature dimensionality and compression is a bit unclear with an unclear takeaway.

Only relying on one family of models reduces the potential impact.

### Questions
How do the results in 6b and 6c compare with a similar plot produced by ablating all but/ablating $k$ random features?

Could you produce similar analysis with additional models with available checkpoints, such as Stanford CRFM GPT-2 or Olmo?

### Soundness
4

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
4

### Summary
The paper claims to explore how meaningful features emerge as language models learn.  Using crosscoders to align representations across training snapshots, the authors trace training dynamics over time. They find a clear turning point early in pretraining, where models move from simple statistical patterns to more structured, feature-based understanding. Through attribution and patching analyses, these evolving features are linked to linguistic behaviors such as agreement and induction.

### Strengths
- The paper touches on an interesting and relevant question, how internal features evolve during language model pretraining. 

- It attempts to provide a structured view using crosscoders. 

- The motivation to study model dynamics rather than static checkpoints is reasonable and aligns with growing interest in understanding training trajectories.

- The work could serve as a preliminary exploration toward more rigorous analyses of concept formation in neural networks.

### Weaknesses
- Novelty. The paper offers little substantive novelty beyond prior work on crosscoders (Lindsay et al., 2024). The method follows existing frameworks closely, with minimal theoretical or heuristic grounding for its design choices. For instance, 
    - 1) The selection of the SlimPajama dataset is not clearly justified, nor is the limited set of base models sufficient to generalize the findings.
    - 2) An ablation study on activation aggregation in Eq. (1) would help clarify whether summing activations is the optimal or even appropriate choice.

- Contribution. The contribution remains narrow, more a descriptive report than a step toward formal interpretability. The reliance on visualizations such as heatmaps and curves, without clear quantitative or linguistic interpretation, makes it hard to connect the observed patterns to interpretable model behaviors. 

- Rigorousness. Several findings, particularly those regarding early n-gram learning, largely lack of contextualization with known results from prior work (e.g., Svete & Cotterell, 2024; Nguyen, 2024, Chen, 2024). 

- Writing. The writing quality is also uneven. The paper reads more like a lab report than a cohesive scientific paper. For instance
    - 1) Lack of grounded formalization for key concepts. For example, can we define "initial features" and "emergent features" (line 202-line207) formally using the notations in the paper? 
    - 2) Some phrasing is imprecise. "the pre-training process remains largely a black box." The sentence is a bit unclear. The process of pretraining is generally gradient descent on large-scale of datasets. I guess what the authors mean here is "the pretraining dynamic" instead of "process".

Overall, the work lacks the theoretical grounding, novelty, and clarity needed to make a strong contribution. That said, it is definitely worthwhile for presenting at a workshop after revision.


``References``
- Anej Svete and Ryan Cotterell. Transformers can represent n-gram language models. arXiv preprint
arXiv:2404.14994, 2024.
- Timothy Nguyen. Understanding transformers via n-gram statistics. arXiv preprint
arXiv:2407.12034, 2024.
- Chen et al. Jet expansions of residual computation  arXiv preprint 2410.06024, 2024.

### Questions
1. Could the authors clarify the rationale behind choosing SlimPajama as the pretraining dataset? Were other datasets considered, and if so, why were they excluded?
2. In Eq. (1), how were activations aggregated across snapshots? Why was summing chosen, and how might alternative strategies (e.g., averaging, normalization, or selective sampling) affect the results?
3. Can “initial features” and “emergent features” be defined more formally within the paper’s notation? A clearer definition would make the results more interpretable and reproducible.
4. Have the authors compared their findings with recent analyses of early-stage n-gram learning (e.g., Svete & Cotterell, 2024; Nguyen, 2024, Chen et al, 2024)? What new insights, if any, extend beyond those studies?
5. How stable are the observed feature dynamics across different base models or seeds? Would larger or differently initialized models exhibit the same “turning point” behavior?

### Soundness
1

### Presentation
1

### Contribution
1
