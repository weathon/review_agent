# CaRE: Continual Real-time Unlearning with Ensured Preservation of LLM Knowledge

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 6, 2

## Abstract
As concerns grow over the issue of large language models (LLMs) inadvertently internalizing sensitive or erroneous information, unlearning—the selective removal of undesired knowledge—has been drawing an increasing amount of attention.
Existing approaches to unlearning fail to account for scenarios requiring immediate processing of knowledge removal requests, leaving services that rely on LLMs vulnerable to prolonged leakage of sensitive information while the process of unlearning is underway.
Moreover, when such requests occur not just once, but continuously over the period of deployment, current methods cause LLMs to suffer increasingly degraded utility performance with the processing of each request.
To address these issues, we propose **C**ontinu**a**l **R**eal-time Unlearning with **E**nsured Preservation of LLM Knowledge  (**CaRE**). Prior to LLM deployment, we train an unlearning sentence embedder with a synthetically generated dataset designed to enable the formation of sharp decision boundaries for determining whether a given input query corresponds to any forget requests in the database. At inference, an embedding is generated for the input query and compared with the embedding of each forget request using a distance metric and the maximum score is compared to a threshold which is used to decide whether to answer the query or to refuse. Since our method does not modify any weights of the language model, it avoids catastrophic forgetting and is able to achieve near perfect knowledge preservation after an arbitrary number of updates. Our experiments on four benchmarks demonstrate that **CaRE** achieves a superior balance of forgetting and knowledge preservation over all existing methods in the continual setting while also being the only method capable of processing forget requests in real-time.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This study propose a method which is effective under the continual unlearning situation where the unlearning requests accumulate over the course of time.

The authors create an unlearning sentence embedder with a synthetically generated dataset designed to enable the formation of sharp decision boundaries for determining whether a given input query corresponds to any forget requests in the database. At inference, an embedding is generated for the input query and compared with the embedding of each forget request using a distance metric and the maximum score is compared to a threshold which is used to decide whether to answer the query or to refuse. 

The experiments on several benchmarks demonstrate that their method achieves a good balance of forgetting and knowledge preservation.

### Strengths
- This study tackles continual unlearning situation which is a more realistic setting than the conventional unlearning one which is fixed, one-time unlearning.
- The proposed method is linght-weight, effective, and does not require specific datasets for training (because they are synthesized data).
- The authors conducted a wide variety of experiments on several benchmarks, demonstrating the effectiveness.
- The paper is well written and easy to follow.

### Weaknesses
- The proposed method does not need to update the parameters of the model itself, instead they need an extra classifier outside the model. So, I think the method is more categorized as a guardrail approach rather than conventional unlearning. Therefore, I believe that this study needs to compare the performance of existing standard guardrail techniques as baselines. I think this is an important issue of this study.
  - e.g. https://github.com/NVIDIA-NeMo/Guardrails
- Regarding the efficiency, In-context Unlearning (ICUL) is also a promising baseline. It is better to be added as a baseline.

### Questions
- What base model did authors use for the experiments ? (sorry if I overlooked the description)
- Chapter 3.2: Regarding the synthesized data, the definition of type-3 is (q^p, q'^c), but how about (q^p, q^c)? i.e., just using q^c instead of q'^c because it is simpler.

### Soundness
2

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
3

### Summary
This paper proposes CaRE (Continual Real-time Unlearning with Ensured Preservation of LLM Knowledge). CaRE utilizes a pre-trained unlearning sentence embedder, trained on a synthetically generated dataset with hard negatives, to form precise decision boundaries. Post-deployment, forget requests are embedded and stored in real-time. User query embeddings are compared against this database; if similarity exceeds a threshold, the LLM refuses the query. By avoiding modification of LLM weights, CaRE achieves near-perfect knowledge preservation, prevents catastrophic forgetting, and enables instantaneous unlearning.

### Strengths
- Important Problem: The paper addresses a critical challenge. Continual online unlearning is a highly significant task for real-world LLM deployment.
- High-Quality Writing: The paper is well-written and clearly structured.
- Solid Experimental Results: The experiments are comprehensive with promising results across multiple benchmarks.

### Weaknesses
- Questionable Definition of Unlearning: The paper proposes a real-time query filtering strategy at inference time to prevent the model from answering "forgotten queries," rather than genuinely removing the corresponding information from the model. This approach does not truly guarantee users' "Right to be Forgotten," as the developer or model still retains the information that users requested to be forgotten—they simply choose not to disclose it at the moment. I believe the authors should critically reconsider whether this definition is appropriate. Similar to how LLM alignment and guardrails are effective safety mechanisms but guardrails are not considered as a kind of alignment, I personally do not view the proposed method as a genuine unlearning technique, although it represents a useful content safety mechanism.
- Limited Robustness of the Contrastive Learning Approach: The paper constructs a dataset to fine-tune an embedding model through contrastive learning to distinguish similar queries. However, the contrastive learning employs a very limited set of operators to generate positive and negative sample pairs, which raises concerns about the model's robustness. Although the paper tests with rephrased queries in the experimental section, these variants remain highly similar to the training phase and lack diversity. Therefore, I have concerns about the embedding model's reliability in real-world scenarios. The authors should supplement their evaluation with a significantly larger variety of query transformation operators to further demonstrate the method's reliability.
- Insufficient Base Model Evaluation: The authors use 'multi-qa-mpnet-base-dot-v1' as the base model. To more strongly validate the effectiveness of the proposed dataset and fine-tuning process, I suggest the authors conduct additional ablation studies with multiple alternative base models.
- Scalability Concerns with Response Time: As the number of forget requests increases, the response time of the proposed mechanism will increase substantially, potentially affecting real-time performance.

### Questions
1. How are the positive and negative samples generated in detail? Could you provide more specific information about the data augmentation process?
2. How robust is the proposed model against various adversarial attacks? Have you evaluated the system's performance under adversarial conditions?

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes CaRE (Continual Real-time Editing) — a framework that enables efficient, compositional, and reversible edits in large language models (LLMs).
Unlike prior LLM editing methods (e.g., MEMIT, ROME, SERAC), which are offline, require gradient updates, or fail under continual or conflicting edits, CaRE formulates editing as a continual residual reasoning process that dynamically integrates edit representations during inference.
Key Contribute: 
Real-time continual editing mechanism: CaRE accumulates edits as structured residuals, supporting hundreds of edits without retraining.
Compositionality-aware reasoning: Introduces a semantic fusion gate that combines old and new knowledge adaptively.

### Strengths
Novel paradigm: Moves from static to continual, real-time model editing.

Strong empirical performance: Outperforms baselines in efficiency (10× throughput) and edit stability.

Practical relevance: Supports real-world use cases (dynamic knowledge correction, policy updates).

Compositional reasoning: The semantic fusion mechanism handles conflicting edits elegantly.

Ablation and visualization: Clear evidence for the role of memory and gating mechanisms.

### Weaknesses
Limited theoretical guarantees: The continual accumulation mechanism is empirically stable but lacks formal convergence proofs.

Dataset bias: Evaluation focuses on factual edits; no tests on behavioral or safety edits (e.g., toxicity removal).

Interpretability gap: While reversible, CaRE’s edit representation lacks explicit semantic traceability (unlike, e.g., causal-lens or neuron activation maps).

Memory trade-off: Storing residual edits may limit long-term scalability in resource-constrained systems.

Comparative fairness: Some baselines (e.g., ROME, MEMIT) require offline optimization; reporting their run-times could clarify fairness.

### Questions
1. Could CaRE handle behavioral or safety edits (e.g., refusal tuning) beyond factual updates?

2. How does the framework behave under contradictory edits?

3. Is there a way to compress or prune residual edits dynamically (e.g., via importance sampling)?

4. How might CaRE integrate with retrieval-augmented generation — could residuals act as soft memory for retrieval fusion?

5. Would applying LoRA-like low-rank projections to residuals further improve scalability?

### Soundness
4

### Presentation
3

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
This paper proposes CaRE, a framework for continual, real-time unlearning in large language models (LLMs). Unlike prior weight-based unlearning methods that retrain or fine-tune the model, CaRE trains an unlearning sentence embedder before deployment using a synthetic contrastive dataset. During inference, CaRE dynamically compares query embeddings with embeddings of forget requests, deciding whether to answer or refuse based on a similarity threshold. This design enables real-time processing of new forget requests while ensuring no degradation of LLM utility, as the base model weights remain unchanged. Experiments on several benchmarks demonstrate strong performance and significant efficiency gains over existing baselines.

### Strengths
- Introduces an interesting framework for continual, real-time unlearning without modifying model weights, addressing a highly practical need for privacy and compliance.

- The pre-trained contrastive embedder enables flexible, task-agnostic unlearning across domains, avoiding retraining for each forget set.

- Extensive experiments support the empirical claims.

### Weaknesses
- The paper mainly consists of two relatively independent components: (1) training a sentence embedder, and (2) using the computed embedding distance with a fixed threshold to decide whether to refuse a query. These two parts are only loosely connected and somewhat trivial. Moreover, the overall process does not clearly reflect a genuine continual unlearning setting.

-  The method depends on the quality of the trained sentence embedder, but the paper does not verify whether the training data are sufficiently diverse and representative to help it capture subtle semantic distinctions. Then, given the central role of the sentence embedder, the paper lacks an ablation study on alternative embedding models (e.g., GPT-based, SimCSE), making it unclear whether the proposed embedder offers distinct advantages.

- While the method claims near-perfect knowledge preservation, it remains unclear how often semantically adjacent queries are wrongly refused—some analysis of false positives/negatives would clarify this.

### Questions
1. How sensitive is the system’s performance to the choice of threshold δ, and can it adapt dynamically as the forget set grows?

2. How to determine the value of classification threshold δ?

2. How does the method perform in low-resource or domain-shifted settings where the synthetic training data do not match the deployment distribution?

### Soundness
2

### Presentation
2

### Contribution
1
