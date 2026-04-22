# Gestalt Reasoning Machines: Structured Perception for Neuro-Symbolic Inference

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
This paper introduces Gestalt Reasoning Machines (GRMs), a novel neuro-symbolic framework that integrates Gestalt principles to enhance reasoning models with perception capabilities similar to human cognition. 
Traditional models, which rely on large datasets and complex computations, often overlook the crucial human cognitive function of grouping, resulting in inefficiencies when dealing with abstract concepts. GRMs address this challenge by incorporating a grouping mechanism grounded in Gestalt principles, enabling the system to recognize and reason over complex visual patterns that are otherwise difficult to capture through object-level features alone. 
This grouping capability allows GRMs to identify higher-order structures and relational configurations that are essential for human-like reasoning. We demonstrate that GRMs outperform purely neural baselines by leveraging logic-based reasoning infused with perceptual grouping cues, offering a more interpretable and cognitively aligned approach. 
Our contributions include the design of GRMs and the empirical validation of their effectiveness in visual reasoning tasks that demand structured perception.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper advances automated visual abstract reasoning, particularly targeting Gestalt principles such as proximity, similarity, closure, symmetry, and continuity. Due to the (to-date) failure of monolithic neural approaches, this paper takes a neuro-symbolic approach where perceptual attributes are recognized with a neural network and underlying rules are recognized and applied in the symbolic domain (first-order logic). Instead of only considering object-level descriptions, this paper advances the neuro-symbolic approach by also considering group-level descriptions, which are recognized in the neural part and further processed by symbolic reasoning. Empirically, the paper shows that adding the group-level descriptions improves the reasoning capabilities on a synthetic Gestalt dataset (ELVIS), outperforming both neuro-symbolic approaches (w/o group-level) and neural approaches (small- and large-scale foundation models).

### Strengths
Finding the right granularity of abstraction is key in neuro-symbolic approaches. This paper demonstrates that multi-level abstractions (object- and group-level) can enhance reasoning performance. In that regard, this well written paper advances the field of neuro-symbolic learning and reasoning.

### Weaknesses
1.	Supervision of perception backbone. It is not clear how the perception backbone is pretrained. Are attributes values and/or groups provided as training labels? If yes, this limitation should be clearly stated and addressed with an end-to-end learning approach. 
2.	Weak grouping performance. As shown in Table 4, the grouping accuracy is very low. Given that the main contribution of this paper is the grouping, it should propose and validate enhancements. First, one can question if a pure neural approach (MLP) without any inductive bias is suitable for this quite involved task. Moreover, the group-level perception (Section 3.1) accumulates all embeddings from the global context into one embedding. This averaging can certainly face some capacity limit. Having a more scalable approach that allows for concatenation (e.g., a Transformer) may improve the approach. Finally, prompting foundational models (e.g., GPT-5) to perform the grouping could be considered, too. 
3.	The timing measurements are missing the neuro-symbolic baselines without group-level information (NEUMANN). Moreover, the hardware should be specified for the different methods. 
4.	Finally, the evaluation is limited to only one synthetic dataset, as stated in the conclusion. While there is a pointer to another natural dataset (Visual Genome), the practical application of the proposed system is not yet justified. It would be helpful to put the work into a practical context. In which real-world applications is grouping needed?

### Questions
I would appreciate if the rebuttal could address the weaknesses. Besides, it would be good to specify the architectural details (neural network architecture) of GRM.

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
2

### Summary
This paper introduces GRMs (Gestalt Reasoning Machines), a neuro-symbolic framework inspired by human grouping nature.                    GRMs first use pretrained perception backbones to identify group structures, then perform rule learning based on logic search.                    This rule base can emerge at test-time and be applied to produce confidence-based inference on new images.                    Experiments on ELVIS, a synthetic dataset incorporating Gestalt principles, demonstrate the advantage of GRMs over few-shot learning neural networks or object-based neuron-symbolic methods.

### Strengths
1. Introducing object grouping into the visual reasoning model is novel and intuitively reasonable. This work claims as the first neuro-symbolic framework that integrates perceptual grouping with symbolic rule learning, demonstrating solid experimental results and offering great insights.
2. The overall framework is efficient, the rule induction time is much less than InternVL3-78B or GPT-5.
3. Great presentation and clear figures make it easy for the reader to follow.

### Weaknesses
1. Lack of detail in 4.1 ‘Pretraining’. From my point of view, the performance of GRMs is largely dependent on the pretrained perception backbones. However, the paper doesn’t include sufficient details (e.g., datasets, objectives, hyperparameters) about this.
2. Unconvincing comparison between GRMs and baselines. Based on W1, actually GRMs’ perception modules are (potentially) benefited from extensive training on images in the same distribution as the evaluation. So the experimental results in the main table cannot fully support such neural-symbolic method can outperform data-driven methods, because the evaluated VLMs may not be familiar with the test images. I would wonder whether the GRMs’ performance is still superior to VLMs after they are post-trained using the same dataset.
3. Limited generalization potential: GRMs' framework relies on predefined predicates and simplified group patterns, and cannot be used in processing real-world images.

### Questions
1.  Could the authors provide more reasons/evidence to support that such a neuro-symbolic method is better than large-scale data-driven training?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work proposes a Gestalt Reasoning Machine (GRM) that fuses human-like perceptual grouping with symbolic rule learning. It operationalizes Gestalt principles within a neuro-symbolic ILP pipeline. Experiments on the ELVIS benchmark show clear accuracy gains over neural and prior neuro-symbolic systems with especially strong results under increasing visual complexity.

### Strengths
- The integration of Gestalt grouping within a neuro-symbolic ILP framework is novel and well-motivated by cognitive theory. The architecture bridges perception and reasoning in a way that is both interpretable and scalable.

- The explicit use of contextual embeddings in s_p(o_i,o_j,I) = \sigma(h_p(o_i,o_j,o^*_{ij})) shows the adventage of this grouping. 

- The comparison against GPT-5 and InternVL3 convincingly shows that structured reasoning can outperform massive data-driven systems.

- Equation in section Appendix A is not clear: how  τ = 0.99 was chosen, for instance?

### Weaknesses
- Evaluation remains restricted to synthetic Gestalt scenes.

- The contextual affinity s_p(o_i,o_j,I) = \sigma(h_p(o_i,o_j,o^*_{ij})) seems to be introduced without justification. What is the theoritical grounding? Any ablation on o^*_{ij} mean embedding?

- Regarding the rule search procedure, what are the convergence guarantees and computational complexity ?

- How sensitive the proposed freamwork to the choice of grouping thresholds s_p?

- Equation in section Appendix A is not clear: how  τ = 0.99 was chosen, for instance?

### Questions
see Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Compared to traditional models relied on large datasets, this paper introduces Gestalt Reasoning Machines (GRMs), which integrates Gestalt principles to enhance reasoning models with perception capabilities. This paper demonstrates that GRMs outperform purely neural baselines in visual reasoning tasks that need perception of higher-order structures. GRM processes an image through perceptual detection, symbolic abstraction, and rule learning to derive interpretable logical rules, which are then applied by an inference engine that prioritizes high-confidence, transparent reasoning for visual prediction.

### Strengths
1.	Novelty: This work systematically integrate Gestalt perceptual principles (proximity, similarity, closure, symmetry, continuity) into a neuro-symbolic reasoning framework, bridging low-level perception and symbolic reasoning.
2.	Empirical Validation: Comprehensive experiments including human test on the newly proposed ELVIS benchmark demonstrate clear advantages over both neural (e.g., ViT) and large multimodal models (e.g., GPT-5, InternVL3).
3.	Efficiency: GRM achieves strong accuracy with significantly lower rule induction time.

### Weaknesses
1.	Data Limitation: All experiments are conducted on the ELVIS dataset.The paper mentioned that “To our knowledge, it is the only benchmark that systematically integrates these grouping principles into a neuro-symbolic pipeline, making it uniquely suited for evaluating GRM.” While this choice is reasonable for testing Gestalt-based grouping, it also raises concerns about generalization. Since ELVIS is specifically designed around grouping-centric reasoning, GRM’s advantage may partially stem from the dataset’s alignment with its inductive bias. To fully assess its robustness and versatility, it would be valuable to evaluate GRM on visual reasoning benchmarks that do not explicitly require grouping, such as CLEVR or RAVEN, to determine whether the proposed mechanism still provides benefits in more conventional reasoning settings.
2.	Writing and Presentation: Page 8 appears quite dense. The discussion on future work in the figure 5 “Currently, our grouping mechanism uses relatively simple neural networks. Developing more robust and semantically informed grouping mechanisms is a promising avenue for future work”could be moved to the conclusion section. Doing so would improve the overall flow and logical structure of the paper.

### Questions
1.	Since ELVIS is specifically designed around Gestalt grouping principles, have you tested (or do you plan to test) GRM on visual reasoning datasets that do not require explicit grouping, such as CLEVR or RAVEN?
2.	I am a bit confused about Table 2 and Figure 7. Could the authors clarify how the accuracy improvements reported in Table 2 were calculated? Similarly, how was the time reported in Figure 7 measured?

### Soundness
3

### Presentation
2

### Contribution
3
