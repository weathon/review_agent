# CoMem: Compositional Concept-Graph Memory for Vision–Language Adaptation

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Continual vision–language learning is crucial for multimodal tasks such as image–text retrieval, visual question answering, and grounded reasoning in dynamic environments, yet deployed systems must learn from non-stationary streams under strict privacy and memory budgets, where naïve finetuning forgets and harms transfer. We aim to sustain stable yet plastic capability in this setting without storing raw data, enabling reuse and recombination across domains and tasks. We present CoMem, a framework that treats compositional structure as the unit of memory and rehearsal: it incrementally organizes knowledge into a compact graph of concepts and relations and rehearses directly in feature space by conditioning practice signals on sampled subgraphs. A lightweight compositional consistency objective keeps part–whole predictions coherent, while teacher-informed, uncertainty-aware filtering limits off-manifold drift. Across cross-domain retrieval, structured concept learning, and continual multimodal VQA, CoMem achieves state-of-the-art retention and transfer alongside consistent gains on SVLC and VQACL/CLOVE under matched memory and parameter budgets. By casting structure as memory and rehearsing where learning happens (feature space), CoMem provides a privacy-friendly and testable paradigm for reliable continual adaptation without raw exemplars.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces CoMem, a continual vision–language learning framework designed for dynamic, privacy-constrained environments where models must learn from non-stationary multimodal data streams without storing raw samples. Unlike traditional fine-tuning methods that cause catastrophic forgetting, CoMem conceptualizes compositional structure as the core unit of memory and rehearsal. The model incrementally builds a compact concept–relation graph, using feature-space rehearsal conditioned on sampled subgraphs to sustain knowledge retention. A compositional consistency objective ensures coherence between parts and wholes, while teacher-informed and uncertainty-aware filtering helps balance plasticity and stability. Experiments across multiple multimodal tasks, including cross-domain retrieval, structured concept learning, and continual VQA, demonstrate that CoMem achieves state-of-the-art retention and transfer performance under matched memory and parameter budgets.

### Strengths
1. Novel Methodology: The idea of treating compositional structure as memory and conducting rehearsal directly in feature space is innovative and well-motivated for privacy-limited continual learning.

2. Comprehensive Analysis: The paper clearly articulates how the proposed consistency and filtering mechanisms work together to preserve stability and plasticity, offering solid theoretical and empirical insight.

3. Strong Experimental Validation: Results across multiple multimodal tasks demonstrate consistent improvements in both retention and transfer, with fair ablation comparisons.

### Weaknesses
1. The introduction could be better structured and written. It does not clearly establish the relevant background or smoothly motivate the authors’ claims. As a result, the logical flow is somewhat fragmented, making it difficult for readers to follow the argument and understand the setting.

2. The CoMem framework involves multiple components in its training objective. It is unclear how sensitive the overall performance is to the balance among these losses. This complexity might hinder the practical application of CoMem in new scenarios. The authors are encouraged to provide a more analysis or discussion on the stability of these hyperparameters and their effect on robustness.

3. In Table 4, increasing the number of trainable parameters does not appear to yield clear performance improvements. The authors could strengthen their claims by including experiments with smaller parameter budgets to further analyze how performance gains scale with parameter count across different methods.

### Questions
See weakness.

### Soundness
4

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
5

### Summary
This paper introduces a novel model framework, CoMem, designed to address the privacy and catastrophic forgetting challenges inherent in real-world Continual Vision-Language Learning (CVLL) deployment. CoMem achieves its goals through a structured, three-stage process. First, during Concept Induction, the model extracts attributes, entities, and relations from new data to update the Concept-Graph Memory. This memory is key, as it stores abstract prototypes and anchors instead of sensitive raw data, making the framework privacy-friendly while mitigating forgetting. Second, in Graph-Conditioned Replay, a small subgraph is selected from memory to generate synthetic features. The model then uses these features to "rehearse" past knowledge, effectively preventing catastrophic forgetting of previously learned data. Finally, during Joint Optimization, CoMem trains on a mixed batch of real data from the new task and the synthetic features from memory. A comprehensive total loss function, formulated as a weighted sum of distinct loss components, is implemented to balance the acquisition of new knowledge with the retention of previously learned information. CoMem was evaluated against state-of-the-art methods across three continual learning benchmarks: cross-domain retrieval, structured concept learning (SVLC), and continual Visual Question Answering (VQA). Across all evaluations, CoMem demonstrated superior results, achieving both the highest average performance and the lowest average forgetting rate.

### Strengths
1. State of the Art performance: The proposed model achieves state-of-the-art results, demonstrating the best retention and lowest average forgetting (AF) across multiple benchmarks.

2. Novel Approach: The proposed model achieves its results with a novel “structure-as-memory” approach to effectively solve both privacy and memory issues.

### Weaknesses
1. Reliance on Upfront Parsing: The current method relies on a "lightweight text parsing" step to extract concept and relation candidates from the text.

2. Fixed Relation Schema: The framework assumes a "fixed relation schema". This weakness, as the authors note it "may constrain coverage in open-world settings" where new, unseen types of relations might emerge.

3. How do you ensure the generated features are semantically rich and diverse enough to capture the complex, nuanced interactions within a subgraph, and not just an average or blurry representation?

### Questions
see weaknesses

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
2

### Summary
COMEM proposes an innovative framework for continual vision-language learning, whose core idea is to treat ​​compositional structure​​ as the fundamental unit of ​​memory​​ and ​​rehearsal​​, rather than storing raw data. The core methodology involves organizing knowledge by structuring the data stream into a compact graph of concepts and relations, and conducting rehearsal by directly generating replay samples in the feature space based on sampled subgraphs. This approach integrates a lightweight compositional consistency constraint and a teacher model filtering mechanism, effectively balancing stability and plasticity. Under strict privacy and memory constraints, the method achieves superior retention and transfer performance across multiple challenging tasks, including cross-domain retrieval, structured concept learning, and continual visual question answering, significantly mitigating the problem of catastrophic forgetting in continual vl learning.

### Strengths
-1. Innovative memory mechanism design and privacy friendly learning paradigm: COMEM's biggest innovation lies in using composite structures as memory units, which not only saves storage space but also captures the intrinsic connections between concepts. Due to not storing any raw image or text data, COMEM naturally meets strict privacy protection requirements. All replay operations are performed in the latent feature space, avoiding the risk of sensitive data leakage and making it particularly suitable for deployment in privacy sensitive scenarios such as healthcare and finance.

-2. COMEM's component design has great flexibility: orthogonal to parameter- efficient methods, it can be used in conjunction with adapters such as LoRA. Supporting different subgraph sampling strategies, combining compositional consistency constraints,  teacher- and uncertainty-informed filtering mechanisms to improve its anti forgetting ability.

-3. Resource friendly: Only a total anchor budget of 64K is needed to achieve good experimental results.

-4. Excellent experimental performance: Significant improvement in cross domain retrieval tasks, structured concept learning, and continuous VQA tasks, robust to different hyperparameters and ViT scales, and stable performance curves in the long-horizon learning process of 18 tasks.

### Weaknesses
-1. The core of COMEM relies on a Fixed Relation Schema, which means that the identification and organization of its concepts and relationships are carried out within a predefined framework. This method is highly effective in handling known and well structured data streams, but may limit its adaptability in fully open environments.

-2. Bias of Teacher Model: Although the Teacher Informed Filtering mechanism can train stably, it may also transfer the cognitive biases or knowledge blind spots of the teacher model itself to the student models.

-3. COMEM is a multi-component complex system, and its training involves multiple stages such as concept induction, subgraph sampling, feature generation, and multi-objective optimization. This complexity brings a high threshold for engineering implementation

-4. The appendix explores the impact of Task Orders, where COMEM's forgetting degree (AF) increases significantly when tasks appear in an adversarial order. This indicates that the stability of the model depends to some extent on the "friendliness" of the data flow. This affects its generalization in real-world scenarios.

### Questions
refer to Weakness

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a continual learning framework that builds a concept–relation graph from image–text pairs by extracting (attribute, entity, relation) triplets. It then performs subgraph-conditioned replay in the representation space, combined with teacher-guided filtering and compositional consistency losses, to mitigate forgetting. Experiments on cross-domain retrieval, structured concept matching, and continual VQA report improvements over several baselines.

### Strengths
Clear motivation for feature-level replay with a graph memory under data-retention constraints.

Use of teacher confidence/entropy gating and compositional consistency is conceptually reasonable.

Broad empirical coverage with multiple tasks and ablations indicates non-trivial engineering effort.

### Weaknesses
Method design is somehow too complex but the presentation is poor.
Figure 1 (method overview) is hard to read.
The diagram is overcrowded: font is too small, visual hierarchy is unclear, and symbols in the figure do not align cleanly with those in the text. It is difficult to grasp the training and replay flow from the overview alone.

Heavy notation but unclear definition.
The paper uses many symbols (for concepts/relations/subgraphs/generator variables, temperatures, loss weights, etc.) without clear definition (e.g., s_align in Eq.(2), three α in Eq. (3) are undefined). Some symbols appear to change meaning across sections.

A large number of hyperparameters with no systematic tuning protocol.
The method includes multiple thresholds (confidence/entropy), loss weights (distillation/consistency/contrastive), structural choices (subgraph size, anchor budget, low-rank dimension), and kernel/sampling parameters. The paper should clarify how hyperparameters are chosen, validation splits, and search budgets.

Not all captions can be parsed into complete (a, e, r) triplets (or maybe in some cases, more than one entity); handling of such cases is unspecified.

### Questions
Please see the weakness.

### Soundness
2

### Presentation
1

### Contribution
3
