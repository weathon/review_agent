# Precise and Interpretable Editing of Code Knowledge in Large Language Models

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 8, 6, 4

## Abstract
Large Language Models (LLMs) have demonstrated outstanding capabilities in various code-related tasks, including code completion, translation, or summarization. However, these pretrained models are static, posing a challenge to incorporate new knowledge into an LLM to correct erroneous behavior. Approaches such as retraining or fine-tuning demand extensive labeled datasets and might be computationally expensive, while prompt engineering fails to change models permanently. Knowledge Editing (KE) techniques offer a more efficient alternative, enabling model updates with minimal data, even just a single example. Nevertheless, existing KE methods often manipulate parameters within the Transformer's multi-layer perceptrons (MLPs), where neuronal polysemanticity hinders both the precision and interpretability of the edits. To address these limitations, we exploit TransCoder, an MLP-like model component with a wide and sparsely activated hidden feature vector. Specifically, we introduce **TransCoder-based Precise Editing** (**TCPE**), a novel method that leverages the sparsity and monosemanticity of the TransCoder’s neurons for highly localized knowledge editing. TCPE exhibits neuron-level mechanistic interpretability characteristics, revealing the correspondence between the edited neurons and the specific code-related knowledge. Furthermore, we present KECode, a new evaluation benchmark for code-to-code translation based on functional equivalence. Using KECode, we conduct a systematic evaluation of representative KE methods in the context of code-to-code translation. Our experimental results demonstrate that TCPE outperforms existing KE methods, achieving a substantial improvement of translation accuracy of CodeLlama-7b-Instruct from 57.5% to 64.0% in a low-resource scenario of Java-to-D translation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes the TCPE model editing method based on TransCoder. This method replaces the original MLP layers with an MLP-like model component that has a wide and sparsely activated hidden feature vector, avoiding the imprecision and uninterpretable edits caused by the neuronal polysemanticity of the original MLP. The paper also introduces KECode, a new code-to-code model editing benchmark. Experiments on KECode and existing model editing benchmarks demonstrate the effectiveness of TCPE.

### Strengths
* Replacing the original MLP with a wider and sparse MLP using TransCoder is novel and interesting.
* KECode, a code-to-code model editing benchmark, is proposed, providing a valuable resource for the community.
* The effectiveness of TCPE is demonstrated on the KECode, ZsRE, and CounterFact datasets.
* An in-depth analysis of TransCoder neurons is conducted, offering valuable insights.

### Weaknesses
* TCPE lacks principled innovation, as both TransCoder and ROME-like editing are existing works.
* The evaluation of the model editing task seems limited to single-case edits, lacking assessments closer to real-world scenarios such as sequential or batch edits.
* There is no discussion of TCPE’s scalability; its performance when editing large batches or performing sequential edits remains unknown.
* TransCoders require additional training, and the training environment appears to be more demanding compared to the baselines. Moreover, it is unclear whether replacing the MLP with TransCoders affects the model’s original capabilities.

### Questions
Would the introduction of TransCoders increase inference latency?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper makes two key contributions to improve the precision and interpretability of knowledge editing in code LLMs on code translation domain.

First, it proposes TransCoder-based Precise Editing, which is a method that replaces the Transformer’s MLP layer with a sparse TransCoder module, enabling edits that target specific highly activated neurons associated with a particular piece of code knowledge. This design allows localized and interpretable edits, minimizing side effects on unrelated behavior while providing clear neuron-level insight into how knowledge is inserted.

Second, the authors introduce KECode, a new benchmark for evaluating knowledge editing in code translation. KECode consists of 600 Java-to-D translation examples paired with unit tests for functional correctness verification. Using KECode, the paper shows that TCPE significantly outperforms existing methods such as ROME, MEMIT, and PMET on correcting Java-to-D translation errors.

### Strengths
* For originality, TCPE presents a novel mechanism that leverages the sparsity and monosemanticity of TransCoder neurons to perform more precise, localized edits. 
* For clarity, the authors place strong emphasis on interpretability. Because TCPE operates at the neuron level, they can explicitly identify which neurons are edited and link them to corresponding knowledge changes. The empirical finding that “highly active neurons carry more essential information during knowledge injection” is well-supported and insightful, strengthening the paper’s interpretability claims.
* For quality, the evaluation is methodologically solid. The paper reports multiple complementary metrics, including efficacy, specificity, and reliability, as well as detailed ablation studies and granular analysis tailored to the knowledge editing context.

### Weaknesses
* For scope and generalization, this paper focuses solely on Java-to-D code translation. It is unclear whether the approach generalizes to other software engineering tasks such as code completion, bug fixing, or program repair. The KECode benchmark evaluates functional error correction, which is one specific type of knowledge editing. It would also be interesting to discuss or demonstrate applicability to, for example, inserting new API knowledge or modifying non-functional aspects of code, which surface broader SWE contexts.
* For practicality and Integration, TCPE requires modifying the model architecture by replacing MLP layers with TransCoder modules. The paper does not clarify whether this change requires retraining the new layers or fine-tuning the entire model. Additional discussion on how TCPE integrates with other architectures (e.g., MoE) would help assess its practical adoption potential.
* For baseline adaptation, while TCPE outperforms existing NLP-based editing methods, like ROME, MEMIT, etc, these baselines were originally designed for factual knowledge editing in natural language models, not code. Because code has strict syntax and execution semantics, these baselines may be disadvantaged. The paper would benefit from either a stronger code-specific baseline, or a clearer discussion of how baseline implementations were adapted to ensure fairness.

### Questions
Two questions:

* How were the specific layers {10, 19, 23} selected for replacing MLP layers with TransCoder modules? Were these layers empirically identified or based on prior interpretability insights about CodeLlama?
* When initializing a TransCoder-modified model from pretrained weights, how are the parameters loaded or transferred to the new architecture? Is there a compatibility or adaptation step between the original MLP weights and the TransCoder modules?

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
1

### Summary
This paper addresses the challenge of precise and interpretable knowledge editing (KE) in Large Language Models (LLMs) for code-related tasks. The authors argue that existing KE methods, which often target standard MLP layers, are hindered by neuronal polysemanticity, leading to imprecise edits and poor interpretability. To solve this, the authors propose TransCoder-based Precise Editing (TCPE). This method involves two key stages:
1. Architectural Modification: The standard MLP layer in a target Transformer (CodeLlama-7b-Instruct) is replaced by a "TransCoder" module—a sparse, wide, MLP-like component from prior work (Dunefsky et al., 2024) that is trained to have more monosemantic neurons.
2. Editing Mechanism: A ROME-like update is applied, but it is restricted only to the small set of "active neurons" in the TransCoder module that are relevant to the knowledge being corrected.  

For evaluation, the authors introduce KECode, a new benchmark for code-to-code translation (Java-to-D) that uses functional equivalence (i.e., unit test pass/fail) as the success metric. Their experiments show that TCPE on the modified architecture (e.g., "LTC4") outperforms baseline KE methods (ROME, MEMIT, etc.)

### Strengths
Knowledge editing is an important research direction to save computational resources by avoiding retraining.

### Weaknesses
I'm not familiar with this field, so I will give my confidence score to 1. Please lower my score weight for this paper.

### Questions
Does the method apply to more applications and datasets?

### Soundness
3

### Presentation
2

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
This paper proposes a TransCoder-based Precise Editing (TCPE) method, to edit code knowledge in large language models. It also presents a new benchmark, KECode, for code-to-code translation based on functional equivalence. Experimental results
demonstrate that TCPE outperforms existing KE methods, achieving a substantial improvement of translation accuracy of CodeLlama-7b-Instruct from 57.5% to 64.0% in a low-resource scenario of Java-to-D translation.

### Strengths
1. This paper proposes a new knowledge editing method for code LLMs. There is also a new benchmark for evaluating the performance of LLMs on code-to-code translation.
2. Experimental results show that, TCPE outperforms existing knowledge editing methods with significant margins.
3. A neuron-level interpretability mechanism is introduced to effectively indicates the connection between the edited neurons and the inserted knowledge.

### Weaknesses
1. Neither codebase nor dataset is provided to confirm the reproducibility. 
2. This paper lacks discussions of limitations and broader impact.
3. The presentation should be improved. For example, fonts in tables and figures can be larger for better reading experience.

### Questions
Would you like to enlarge the fonts in Figure 1, Figure 2, Table 3 and Table 4?

### Soundness
2

### Presentation
1

### Contribution
2
