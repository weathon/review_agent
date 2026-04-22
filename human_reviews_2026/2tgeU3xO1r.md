# Origins and roles of world representations in neural networks

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 4, 6

## Abstract
While neural representations have been extensively studied in large practical models, the controlled conditions that govern their emergence and their downstream role in model adaptation remain poorly understood. In this work, we develop a framework separating the underlying world, the data generation process, and the resulting model representations to answer these questions in a controlled setup. This framework further allows clearly defining expected behavioral and representational changes resulting from a world update. Specifically, we define the world as a set of city coordinates and define 7 geometric tasks which generate data to train an autoregressive language model. First, we show that different data generation processes give rise to different world representations in the model. Next, we show that multi-task training drives representational alignment between models that do not share any common tasks, providing controlled evidence for the Multitask Scaling Hypothesis, a potential explanation of the Platonic Representation Hypothesis. Finally, we study whether multi-task models can integrate new entities consistently via fine-tuning. Surprisingly, we find that some fine-tuning tasks are “divergent” and actively harm the representational integration of new entities. Overall, our framework establishes a model system to study the emergence of world representations in neural networks and their adaptability in a controlled manner.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The present paper proposes a framework for studying representations in neural networks. In particular, the authors separate input data ("world") from task ("data generation"), and then study how training on single and multiple tasks effects the emerging representations. They find that training on single tasks yields divergent representational geometries, whereas multi-task training drives alignment. Furthermore, their results indicate that representational divergence measured in single-task pretraining predicts downstream failure during finetuning.

### Strengths
The paper was overall easy to follow and read. The broader question (how to make sense of neural network representations) is of both intellectual and practical interest. From a conceptual perspective, I liked the proposed seperation between data and task. In terms of results, the paper showed very cleanly that multi-task learning leads to more aligned model internal representations.

### Weaknesses
While the authors argue that their work represents evidence for the platonic representation hypothesis, that is at best only weakly the case. True evidence would require studying representations of different model architectures, which is not done presently. 

Related work is not much discussed. I am not too familar with this line of work, but surely people have studied neural network representations in multi-task settings before. How does the present work connect to this and in which aspects does it differ?

Personally, I found most of the results not very surprising and somewhat limited in impact. While the methodology is sound and the analysis thorough, the findings largely align with my prior expectations.

The authors argue that "generalization performance correlates with the CKA values from single-task pretraining." While that is true, the relationship is fairly weak.

Limiting the analysis to seven tasks seems constraining, especially if one eventually wants to transfer the insights to realistic settings.

Minor:
* Figure labels generally very small.
* Figure 1 not referenced in the text.
* Figure 4 abbreavtions not defined.
* Figure 6 not referenced in the text.
* Figure 6 dual axis very confusing without color coding.

### Questions
What is the actual input to the transformer? I assume it is just the entire string? Is it the same for the single- and multi-task settings? This is never explicitly mentioned. How are strings tokenized? Everything on the character-level except for city ids?

How consistent in their representations are single-task models across mutliple runs? That seems like an important control condition.

Crossing fails to train alone. Why? That seems strange given that this is a fairly simple setup.

"Despite these differences, we can still linearly decode (x,y) coordinates from most tasks, as shown in the second row of Fig. 4." Where can I see this in the figure?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors present a study investigating how neural networks (transformers specifically) learn convergent representations of a single latent data manifold through different tasks, or combinations thereof. They use real world cities to define a set of latent coordinates, and come up with seven function learning tasks, mapping city tuples to outputs that the networks are trained to predict. The authors show that, when trained on single tasks, the models learn representations that tend to by similar (measured by CKA), but also show structural differences. By constraining the representation space by training the models on more tasks, the representations start to align more. This nicely demonstrates a principle often formalized as the Platonic Representation Hypothesis. Lastly, the authors analyze representations when models are fine-tuned to incorporate a novel (fictitious) city.

### Strengths
* The authors conduct simple and diagnostic experiments. The results make sense and support the conclusions made in the paper
* Nice diagnostic test of the platonic representation hypothesis in a toy setting

### Weaknesses
* The introduction meanders on very general questions related to representation learning and neural networks, not easy to see how all of these are related to the questions the paper actually studies. I would recommend making the introduction more succinct and to-the-point.

Overall, the paper provides good evidence for a simple and interesting question. While it's not very surprising that multi-task training constrains the model representations, giving rise to alignment, the evidence presented is solid so I'm happy to recommend accept.

Formatting:
* On line 383 there seems to be a reference missing (see the question mark)

### Questions
* Were other model training factors tested? For instance, regularization (L1 or L2) on the residual stream representations might speed up alignment, as constraints are put on the representations.
* Does multi-task alignment interact with model size? Were differently sized transformers trained?

### Soundness
3

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
The paper introduces a simple framework to analyze how different training objectives influence the learned representations when the underlying world model is known. To make this analysis tractable, the authors construct a synthetic setup where the “world” consists of 2D city coordinates. Data are generated using seven different geometric tasks based on these coordinates. This controlled environment allows the authors to systematically study several aspects of representation learning: First, they show that models trained on most of these tasks can learn representations that are linearly mappable to the underlying 2D coordinates, thereby capturing the correct world model. Second, they demonstrate that combining multiple tasks improves this alignment with the true world representation; for some tasks, such multi-task training is even necessary for success. Lastly, they investigate fine-tuning effects: when a general-purpose model is fine-tuned on a subset of tasks with additional data, the choice of fine-tuning task is crucial. In some cases, new “cities” embed seamlessly into the existing latent geometry, while in others they occupy a separate region of the space. This behavior also determines whether the fine-tuned model generalizes across tasks or becomes specialized to the task it was fine-tuned on

### Strengths
The paper presents a well-designed framework for systematically analyzing how training objectives affect learned representations when the underlying world model is known. The idea of generating tasks based on a shared, low-dimensional representation is elegant and enables controlled, interpretable experiments. The seven tasks are well-chosen and require different forms of geometric reasoning. The overall problem setup is clearly explained, and Figure 2 effectively illustrates the environment and task construction. By isolating the training objective from other factors such as architecture or data complexity, the findings become easier to interpret.
The visualization in Figure 3 provides an intuitive view of how the world model emerges during training, demonstrating that this emergence is not necessarily directly correlated with task performance. Overall, the proposed framework serves as a valuable analytical tool for studying and bench-marking representation learning methods.

### Weaknesses
- The study appears to rely on a single random seed for training. It would be important to evaluate whether the learned representations vary more across different tasks than across different random initializations of the same task. Without such analysis, it is difficult to assess the stability of the reported findings.
- The character-based city encoding is somewhat unconventional and insufficiently justified. It is unclear how the cities are indexed or numbered—if the numbering follows geographic order, this could inadvertently leak coordinate information into the model. Furthermore, the authors note in the appendix that city coordinates starting with “0”, “00”, or “000” fail to work and were excluded from all experiments. This exclusion raises concerns about potential implementation artifacts or biases in the input representation.
- In the fine-tuning experiments, the new coordinates are concentrated in a small region of the space, leading to clustering of new cities. This represents a special case of localized additional information, while experiments using more spatially distributed new points would help evaluate whether the conclusions generalize.
- The city manifold used in this framework is a flat 2D plane, while real-world data often lie on more complex manifolds. Extending the approach to non-Euclidean geometries, such as a spherical globe, would be an interesting next step and could test the robustness of the proposed framework.
- The paper provides only a limited explanation of what constitutes “divergent tasks.” A deeper discussion of the specific geometric reasoning required by each task and why certain tasks diverge would help clarify this concept.
- Minor comments:
   - The caption of Figure 3 mentions “top,” “middle,” and “bottom” panels, but the figure appears as a single mixed layout. 
   - The caption and description of Figure 5 refer to a 21×21 CKA matrix, while the plot shows only 7×7.

### Questions
- As the Crossing task did not succeed on its own, did it work in combination with any other tasks, or only with Distance and Perimeter? In addition, while Distance appears to perform well in early experiments, it is later described as a divergent task. Could this divergent behavior be an artifact of the fine-tuning data, where the new cities (“Atlantis”) are concentrated in a single coordinate region?
- What are the accuracies of the linear probe used for coordinate prediction across different settings—models trained on single tasks, combined tasks, and the fine-tuned variants? Including these results would help quantify how well each representation captures the true world geometry.
- How is Normalized Improvement defined? Please clarify how it is calculated. Also, it is unclear, whether the deviation from max-model is in percentage or absolute units.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper performs a controlled analysis of how world models are formed in autoregressive models. 'World representations' are not fixed by the world itself but by the data‐generation tasks: single tasks carve up internal geometry very differently; training on diverse tasks forces these geometries to align ('Platonic' convergence). 

The authors find:

-World representations emerge under autoregressive training: models first cluster nearby cities, then form a world‑aligned geometry; (x,y) becomes linearly decodable before task accuracy jumps.

-Single‑task training induces distinct geometries (e.g., distance yields a thread‑like structure, angle a 2‑D manifold)

-Multi‑task pretraining aligns representations: average CKA increases with task count (1→2→3), including between models that share no tasks

### Strengths
-The controlled setup for the experiments makes the results presented in this work convincingly support the idea of Platonic Representations presented in earlier work. By decoupling the world, data, and model, the authors control exactly what changes (tasks vs. world) and show models learn only from task outputs, never coordinates, so any alignment effects can be attributed to task diversity rather than data confounds.

-Setup is straightforward and training is possible at the level of academic resources. (i.e a 6‑layer, 128‑hidden, 4‑head Transformer with a 98‑symbol ASCII vocabulary, trained autoregressively). Fine‑tuning likewise uses manageable data.

-The addition of the Atlantis fine-tuning is a creative illustration of how downstream performance is impacted by the learned 'world representation' during pretraining. This provides a clean test of whether the 7‑task‑pretrained world manifold can absorb new entities and generalize them across tasks. Fine‑tuning on a single task with Atlantis yields a generalization matrix whose gains vary by task and correlate with pretraining CKA, directly tying the geometry learned during pretraining to downstream performance.

### Weaknesses
Since this paper focuses on analysis, most of the issues I encounter in this paper are focused on the clarity of presentation:

-Many figures are way too small (e.g. Figure 3-7), as a rule, the figure text should be sized similarly to the paper text

-Citations are missing in some parts of the paper (e.g. ? citations)

-Colors shown on World Map are not indicated with a legend.

Figure 8 is hard to tell the difference or what should be noticed in the contrast between the PCA/Linear probe subfigures for non-divergent vs divergent tasks. Are there quantitative measures that quantify how the new entities are different in the non-divergent vs divergent conditions?

If the goal is to demonstrate that the coordinates are placed in an orthogonal subspace and lie close to the origin, it would be more helpful to quantify it numerically rather than showing it visually.

Figure 8a is also hard to interpret. Without a clear way to interpret what the x-axis means, it's hard to understand what each entry in the matrix denotes and, consequently, what the vertical bands indicate.

### Questions
"This suggests that divergent tasks cause optimization to encode new entities in orthogonal subspaces rather than integrating them into the existing world manifold—explaining their failure to support cross-task generalization."
Can the authors make a statement (admittedly extrapolative) about how this is handled in the real-world by current models (e.g. LLMs) trained on data that may encode divergent tasks? Presumably, the data in the real world will not be as consistent as in the idealized setting posed in this paper. 

"we do not claim that interventions to increase single-task CKA would necessarily improve fine-tuning generalization."
What are the author's thoughts on the intervention and how that would impact generalization? Was this intervention tried?

"Even excluding models with shared tasks, we find substantially higher CKA compared to single-task models" Can these pairs be isolated better from Figure 5c? Perhaps the matrix can be structured in a way where the partial overlap entries can be localized. I think non-overlapping tasks having higher alignment is an important result because it shows the common anchor is the World Map (i.e. the 'Platonic Space')

### Soundness
4

### Presentation
2

### Contribution
3
