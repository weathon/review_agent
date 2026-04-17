# Interpretable Hierarchical Concept Reasoning through Graph Learning

- Decision: Reject
- Scores: 2, 2, 6, 4

## Abstract
Concept-Based models (CBMs) are a class of deep learning models that provide interpretability by explaining predictions through high-level concepts. These models first predict concepts and then use them to perform a downstream task. However, current CBMs offer interpretability only for the final task prediction, while the concept predictions themselves are typically made via black-box neural networks. To address this limitation, we propose Hierarchical Concept Memory Reasoner (H-CMR), a new CBM that provides interpretability for both concept and task predictions. H-CMR models relationships between concepts using a learned directed acyclic graph, where edges represent logic rules that define concepts in terms of other concepts. During inference, H-CMR employs a neural attention mechanism to select a subset of these rules, which are then applied hierarchically to predict all concepts and the final task. Experimental results demonstrate that H-CMR matches state-of-the-art performance while enabling strong human interaction through concept and model interventions. The former can significantly improve accuracy at inference time, while the latter can enhance data efficiency during training when background knowledge is available.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
Concept-based models offer interpretability by predicting human-understandable concepts, but the prediction of the concepts themselves typically relies on black-box neural networks. This paper proposes a hierarchical concept prediction framework that learns an acyclic graph of concepts, aiming to improve interpretability.

### Strengths
1. The idea of hierarchically predicting concepts is sound and well-motivated.

### Weaknesses
**Major Weaknesses**

1. **Lack of Discussion on Related Work:**
There is a line of prior work that uses logic rules as model concepts [1, 2, 3]. The proposed approach seems similar in that a neural network selects rules based on their embeddings. However, the paper does not provide a clear discussion of how it differs from these methods.

2. **Concepts Are Not a True Bottleneck:**
A major concern is that in the proposed framework, concepts no longer act as a true information bottleneck. As shown in Figure 1, there appear to be direct black-box paths from input to output that may bypass the concept layer entirely. This could undermine the interpretability claim, as the model might “cheat” by encoding all information through embeddings rather than concept structure [2, 3].

3. **Methodology Unclear and Not Well-written:**
First, it is not explained how the concept pool is constructed. It is unclear whether the model relies on human-annotated concepts (as in datasets like CUB) or learns the concepts automatically. If the former, this raises concerns about scalability and generalizability; if the latter, it is unclear how the learned concepts maintain interpretability.
Second, the motivation behind parameterizing the encoder with a Bernoulli distribution and enforcing a delta distribution for the concept embeddings is not well justified. I suspect that this choice might have been made for training stability or sparsity, but it is not clear whether it plays a crucial role in performance. If this probabilistic formulation is important, it should be supported by ablation studies; if not, the reasoning for adopting it should be clarified.
Third, the description of the decoder mechanism - particularly how the model aggregates a pool of candidate rules based on parent concepts and selects one for child concepts - is confusing. From the current explanation, I suspect that the decoder may sequentially sample rules conditioned on parent availability, but this process is not clearly described. This lack of clarity makes it difficult to understand how the hierarchical structure is actually learned or enforced.
Finally, many notations and equations in Sections 2.2.2, 2.2.3, and 3 are not properly defined, which adds further confusion and makes the paper challenging to follow.

4. **Ambiguity in Likelihood Maximization (Eq. 9):**
Equation (9) suggests maximizing the likelihood of predicted concepts $\hat{c}_i$, implying access to their ground-truth labels. However, it is unclear whether such supervision is available or how the model learns the “correct” concepts if not. 

5. **Non-Deterministic Interpretations:**
It appears that the decoder selects rules probabilistically, which raises the concern that the same input-output pair could yield different interpretations across runs.

6. **Questionable Use of "Theorem:**
The statements labeled as Theorem 5.1 and Theorem 5.2 do not appear to present substantial theoretical results. It is not evident what value these “theorems” contribute.

7. **Limited Scope and Evaluation:**
The experiments are restricted to simple vision datasets. Moreover, the paper later reveals that concept annotations are required, which significantly limits practicality. There is also no qualitative demonstration of interpretability, despite this being the paper’s main goal.

**Minor Comments**

8. In several equations, $\hat{x}$ appears where $x$ would be expected. Please clarify the distinction between the two.

**References**

[1] Deep Neural Networks Constrained by Decision Rules (AAAI 2019)

[2] Self-Explaining Deep Models with Logic Rule Reasoning (NeurIPS 2022)

[3] Toward Faithful and Human-Aligned Self-Explanation of Deep Models (npj Artificial Intelligence 2025)

### Questions
1. Clarify whether and how the model architecture enforces the concept bottleneck. Consider adding experimental evidence (e.g., ablation or information flow analysis) showing that the model genuinely relies on concepts rather than shortcutting through embeddings. (W2)

2. Provide a clearer, methodological explanation, including the presence of ground-truth labels for concepts and whether the interpretations are deterministic. (W3, W4, W5)

3. Consider rephrasing these sections as Propositions or Lemmas, or alternatively, move them to an appendix if they are primarily for clarification. Explicitly state what each result contributes to the method or guarantees about model behavior. (W6)

4. Evaluate on larger-scale datasets (e.g., ImageNet) to demonstrate scalability. Include at least one qualitative visualization or example of model interpretation. Discuss how the method could be extended to domains lacking concept annotations (e.g., via weak supervision or learned concepts). (W7)

5. Include a more detailed comparison with prior works that integrate logic reasoning into neural networks. Highlight what conceptual or methodological novelty your hierarchical approach introduces beyond embedding-based rule selection. (W1)

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Hierarchical Concept Memory Reasoner (H-CMR), a new Concept-Based Model (CBM) that provides interpretability not only for the final task but also for intermediate concept predictions, which learns a DAG of concepts and represents their relationships via neural-selected symbolic logic rules stored in a shared memory. Specifically, H-CMR contains three different modules: an encoder which predicts source concepts and a latent embedding, a decoder which hierarchically infers other concepts through symbolic rule execution, and a memory which stores learnable logic rules defining parent–child relations. Experiments show that H-CMR achieves SOTA concept accuracy and maintains universal classification capability comparable to black-box networks.

### Strengths
1. The idea is interesting. Compared to previous CBM methods, which can only explain task-level predictions, H-CMR explicitly models how concepts depend on one another via interpretable logic rules. 
2. The paper is well-written and easy to follow. 
3. The experiment results on three different datasets show the effectiveness of the proposed methods.

### Weaknesses
1. Although the idea is interesting. However, I feel suspicious about whether the proposed methods can be extended to real-world scenarios. Currently, the experiments are done on some small toy datasets, which limit the contribution of the proposed methods. In order to provide a more comprehensive view of the proposed methods, it is better to show the results of the model in more complex scenarios. For example, there are lots of different vehicle categories in ImageNet. Is it possible to use the H-CRM to extract concepts that humans can understand and explain the logic between different concepts and tell the logic between different concepts? If humans cannot understand the extracted concept, the interpretability is not valid, even if humans can understand the logic between concepts. For example, a model can tell me $C_k=C_i \vee C_j$, without any understanding of $C_k, C_i, C_j$, we can not even tell whether the logic is true or false. Also, without such understanding, model intervention is not valid because we cannot even tell how to change the logic behind the concepts. 
2. The ablation studies are not sufficient. Currently, the H-CMR claims to provide interpretable inference for both concept and task predictions, the paper does not include any empirical validation that the extracted explanations (logic rules or hierarchical DAGs) are semantically or causally correct.

### Questions
I will raise my rating if the author can provide additional evidence of the interpretability and demonstrate that it is correct.

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
The authors proposed H-CMR (Hierarchical Concept Memory Reasoner), a neuro-symbolic framework that performs interpretable concept reasoning through a directed acyclic graph of learned logical rules. 
It first predicts semantically grounded concepts from perceptual inputs and then composes these concepts hierarchically via differentiable rule modules to infer higher-level concepts and final task decisions. 
Each rule defines an explicit logical relation among parent concepts, enabling transparent reasoning across multiple abstraction levels. 
The model jointly optimizes concept prediction, rule learning, and task objectives to balance interpretability with performance. 
Experimentally, H-CMR is validated across synthetic and visual benchmarks such as MNIST-Addition, CIFAR-10, and CUB, demonstrating improved intervenability and interpretability over concept-based baselines. 
These results highlight H-CMR’s ability to achieve structured, human-understandable reasoning without sacrificing predictive accuracy.

### Strengths
- **S1. Strong integration of concept-based and neuro-symbolic reasoning**

The proposed method provides a well-structured integration of concept bottleneck modeling with neuro-symbolic reasoning through its unified architecture of concept learning and rule-based inference. 
It jointly learns concept embeddings and logical rules within a directed acyclic graph, allowing the model to represent both the semantics and dependencies among concepts in an interpretable manner.
The neural encoder captures perceptual information and grounds the concepts, while the symbolic reasoning layer operates over these learned concepts using explicit logical compositions. This design effectively connects data-driven neural representations with transparent symbolic reasoning, enabling interpretable predictions at both the concept and task levels.

- **S2. Clear formulation of hierarchical reasoning and strong theoretical grounding**

H-CMR clearly defines how hierarchical reasoning emerges through the recursive composition of learned logical rules. 
Each concept is inferred from its parent concepts via explicit rule evaluation, and these dependencies propagate across multiple levels of the directed acyclic graph (DAG), forming interpretable hierarchical structures. 
Also, the authors provided solid theoretical grounding for this formulation. 
It formalizes the model’s expressivity, acyclicity, and computational complexity, with clear statements of the underlying assumptions and guarantees.

### Weaknesses
- **W1. Computational complexity and scalability considerations**

While the proposed method offers a principled formulation for interpretable concept reasoning, its computational complexity raises concerns about practical scalability. 
Empirical results in Table 4 show that runtime and memory usage increase significantly with the number of concepts, suggesting that efficiency could become a bottleneck for large-scale or densely connected concept graphs. 
It would be valuable to explore how the practicability of the proposed method could be improved--particularly by testing it on real-world, higher-dimensional datasets or by incorporating optimization strategies such as hierarchical pruning, rule caching, or sparse evaluation. Such extensions could strengthen the framework’s applicability beyond medium-scale benchmarks while preserving its interpretability advantages.

- **W2. Need for stronger qualitative concept visualization and complementary quantitative validation**

While the proposed method provides clear rule-based reasoning and interpretable dependency graphs, it would benefit from richer qualitative visualization of the learned concepts--a practice that has become standard in concept-based interpretability research. Many concept bottleneck and concept-based models support their claims with visual examples showing how individual concepts are localized, clustered, or activated across samples. Such qualitative results help readers intuitively grasp what each learned concept represents and how concept combinations contribute to hierarchical reasoning.
Incorporating visual analyses of representative concepts or rule activations--such as heatmaps, or example patches--would make the interpretability of H-CMR more tangible and relatable. 
In parallel, pairing these visuals with quantitative measures (e.g., localization accuracy, sparsity, or activation consistency across samples) could provide stronger empirical support for the model’s interpretability claims.

This also opens several natural questions for further exploration:
- How can concept localization be structured hierarchically, reflecting parent–child relationships in the rule graph?
- Can hierarchical visualizations reveal whether higher-level concepts emerge from spatial or semantic composition of lower-level ones?
- How can visual concept maps be used to validate the internal reasoning paths of H-CMR?

Addressing these questions through both visualization and quantitative validation would strengthen the link between H-CMR’s symbolic reasoning and the perceptual evidence that grounds its concepts, making the framework more convincing and complete.

### Questions
Most of my main concerns or questions have been outlined in the Weaknesses section.

### Soundness
3

### Presentation
3

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
This paper introduces Hierarchical Concept Memory Reasoner (H-CMR), a concept-based model that combines symbolic reasoning with neural attention for interpretable concept and task prediction. The key view is to learn a directed acyclic graph (DAG) over concepts and tasks, where each node is predicted using neurally selected, symbolic logic rules. This framework enables step-by-step symbolic inference and supports both concept-level and model-level interventions.

### Strengths
1. More interpretability: The model offers clear, logic-based explanations for both concept and task predictions, going beyond typical CBMs that only explain task-level outputs.

2. Human-in-the-loop friendly: The architecture is explicitly designed to allow interventions during both inference and training, making it suitable for scenarios where expert knowledge is available.

### Weaknesses
1. Latent embeddings still required: Despite the symbolic reasoning layer, a latent embedding is still used in rule selection, which introduces some opacity.

2. Limited task diversity: Most experiments focus on standard classification datasets; it remains unclear how well the method generalizes to more complex domains.

3. Quadratic complexity: The approach has worst-case quadratic time complexity in the number of concepts, which may limit scalability.

### Questions
See above.

### Soundness
3

### Presentation
2

### Contribution
3
