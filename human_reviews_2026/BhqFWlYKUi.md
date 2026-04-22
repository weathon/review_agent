# Medical Interpretability and Knowledge Maps of Large Language Models

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 6, 2, 6

## Abstract
We present a systematic study of medical-domain interpretability in Large Language Models (LLMs). We study how the LLMs both represent and process medical knowledge through four different interpretability techniques: (1) UMAP projections of intermediate activations, (2) gradient-based saliency with respect to the model weights, (3) layer lesioning/removal and (4) activation patching. We present knowledge maps of five LLMs which show, at a coarse-resolution, where knowledge about patient's ages, medical symptoms, diseases and drugs is stored in the models. In particular for Llama3.3-70B, we find that most medical knowledge is processed in the first half of the model's layers. In addition, we find several interesting phenomena: (i) age is often encoded in a non-linear and sometimes discontinuous manner at intermediate layers in the models, (ii) the disease progression representation is non-monotonic and circular at certain layers of the model, (iii) in Llama, drugs cluster better by medical specialty rather than mechanism of action, especially for Llama and (iv) Gemma-27B and MedGemma-27B have activations that collapse at intermediate layers but recover by the final layers. These results can guide future research on fine-tuning, un-learning or de-biasing LLMs for medical tasks by suggesting at which layers in the model these techniques should be applied. We attached our source code to the paper for reproducibility.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper has focused on the medical-domain interpretability in large language models. The authors have found that existing works have only explored limited medical knowledge areas with a single explainability technique. To fill the gap in this field, this paper has explored four interpretability methods on five various LLMs. Through extensive experiments, the authors have found several conclusions.

### Strengths
+ S1. This paper is well-organized and well-written, making it easy to follow.
+ S2. Extensive experiments have been conducted.
+ S3. The code is released, making it easy to reproduce.

### Weaknesses
- W1. The motivation for the specific exploration in medical areas is insufficient. It is still unclear why the interpretability techniques cannot be well adopted in the medical area.
- W2. The authors have claimed that they focused on the medical domain, but only one medical-specific LLM is experimented with in this paper, i.e., MedGemma-27B. In my view, more related LLMs, such as Huatuo-GPT, should be considered in this paper, instead of general-purpose LLMs.
- W3. Though the authors have argued that previous works only consider one of the medical knowledge areas. However, this paper also only considers them independently, while ignoring the relationships between them. Thus, it seems that this paper has only conducted more experiments but has not addressed this issue basically.
- W4. Why are the four interpretability methods, i.e., UMAP, gradient-based saliency, layer lesioning, and activation patching, selected in this paper? What's the selection criterion?

### Questions
All my questions have been included in the weakness section.

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
3

### Summary
This paper presents an interpretability analysis of five popular LLMs, focusing on the medical knowledge encoded within their layers. The authors employ four interpretability methods: gradient-based saliency, UMAP projections, activation patching, and layer lesioning. They examine how medical knowledge is localized across patient demographics, diseases, and drug treatments.

### Strengths
- The paper explores an interesting and typically underrepresented domain within interpretability research.
- It employs multiple complementary approaches, offering a broad and consolidated set of metrics and analysis.

### Weaknesses
**[W1]** While the paper presents a substantial number of results and metrics, it falls short in translating these insights into actionable recommendations. Even in the discussion section, the authors primarily offer intuition behind their findings but acknowledge that further analysis is required. To enhance the practical relevance of this work, authors can provide a clearer argumentation or implementations regarding how their analysis could inform model development.

**[W2]** The paper does not specify key statistics or details about the data used to probe the models and conduct the mechanistic analysis (apart from the prompt templates listed in the Appendix). Given that interpretability results can be highly data-dependent, this omission makes it difficult to assess the generalizability of the findings.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes CPA (Causal Prototype Alignment), a framework aiming to enhance the medical interpretability of large language models by aligning learned prototype representations with nodes in a medical causal graph. The method consists of three components:
(1) a prototype-based representation module that learns latent medical concepts via a prototype bank,
(2) a causal alignment module that regularizes the proximity between prototypes and causal nodes extracted from medical knowledge graphs (e.g., UMLS), and
(3) a counterfactual consistency module to enforce robustness under irrelevant feature perturbations.
Experiments on clinical datasets (e.g., MIMIC-III) show improvements in classification performance (AUC/F1) over baselines such as ProtoPNet and CausalBERT. However, the empirical evidence for interpretability and causal consistency remains weak.

### Strengths
1. Addresses an important and timely problem — interpretability of medical LLMs.

2. The framework combines multiple paradigms (prototype learning, causal graphs, counterfactual consistency), showing awareness of interpretability literature.

3. Implementation appears systematic, and empirical results show consistent performance gains on standard classification metrics.

### Weaknesses
1. The paper suffers from conceptual ambiguity. The definitions of the prototype, causal node, and alignment mechanism are unclear, and the core idea is only described at a high level without being formally defined or mathematically grounded.

2. The work shows weak causal grounding. The so-called causal graph is loosely constructed from UMLS relations and textual templates, but it lacks genuine causal semantics, and no causal inference or intervention analysis is performed.

3. The paper presents misaligned evaluation. Most experimental results emphasize predictive performance metrics such as AUC and F1, rather than interpretability, and there are no objective metrics or human studies that convincingly demonstrate improvements in interpretability.

4. The paper provides a poor explanation of figures. The numerous visualizations are largely descriptive rather than analytical, and they do not clearly show how interpretability or causal reasoning emerges from the proposed model.

5. There are serious reproducibility concerns. The paper omits essential implementation details, including how the causal graph is built, how prototypes are aligned with causal nodes, and how the model’s hyperparameters are selected or tuned.

### Questions
1. What ensures that learned prototypes correspond to meaningful medical concepts rather than latent clusters?

2. How was the causal graph constructed and validated for correctness?

3. Can you provide a quantitative metric or human study to support claims of enhanced interpretability?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a systematic study of medical interpretability in large language models (LLMs), introducing “medical knowledge maps” across five open-source models. It combines four complementary interpretability methods—UMAP projections, gradient-based saliency, layer lesioning, and activation patching—to locate where medical knowledge (age, symptoms, diseases, drugs, dosages) is stored within model layers. The work is ambitious and methodologically comprehensive, providing interesting insights into layer-wise organization and representational phenomena such as non-linear age encoding and circular disease progression.

### Strengths
* The study systematically applies four complementary interpretability methods, yielding a triangulated view of where medical knowledge resides in LLM layers.
* The work addresses a relatively underexplored area by focusing interpretability specifically on medical-domain knowledge and tasks.
* The empirical scope spans five models and multiple medical subdomains, improving the breadth and potential generalizability of the findings.
* The paper presents clear visualizations and “LLM maps,” making the layer-wise organization of knowledge intuitive to interpret.
* The insights have practical implications by potentially guiding fine-tuning, unlearning, and bias-mitigation strategies for medical LLMs.

### Weaknesses
* The evaluation lacks strong external validation and limited use of ground truth, leaving some claims insufficiently anchored beyond internal metrics.
* The interpretation of non-linear and circular manifolds remains conceptually ambiguous, risking over-interpretation of representational geometry.
* The statistical analysis does not thoroughly quantify robustness across seeds, prompt perturbations, or hyperparameter choices, which weakens rigor.
* The connection to broader interpretability theory and causal abstraction frameworks is not fully developed, reducing theoretical grounding.
* The reproducibility story in the main text omits detailed settings and implementation specifics (e.g., UMAP and patching parameters), which may hinder independent replication.

### Questions
* How stable are the LLM maps across different prompt templates or random seeds?
* Did you consider potential confounding due to tokenization or vocabulary frequency in medical terms?
* Could the observed circular disease progression simply arise from UMAP distortions rather than true representational topology?
* Have you tested whether the identified “knowledge layers” align with known reasoning behaviors (e.g., zero-shot diagnosis tasks)?

### Soundness
3

### Presentation
3

### Contribution
3
