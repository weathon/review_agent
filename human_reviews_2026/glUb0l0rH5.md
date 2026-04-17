# Reversible Lifelong Model Editing via Semantic Routing-Based LoRA

- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language processing. However, the dynamic evolution of real-world knowledge necessitates continual editing of specific knowledge within LLMs. While existing model editing methods explore modular isolation or parameter-efficient strategies, they often suffer from semantic drift or knowledge forgetting during sequential editing due to the continual updating of semantic content. To address these challenges, we propose $\textbf{SoLA}$, a $\textbf{S}$emantic r$\textbf{o}$uting-based $\textbf{L}$oR$\textbf{A}$ framework for reversible lifelong model editing. In SoLA, each edit is encapsulated as an independent LoRA module which is frozen after training and a semantic routing record is established to map it to the input semantic representation, allowing dynamic activation of LoRA modules via semantic matching. This mechanism avoids semantic drift caused by clustering updating and mitigates catastrophic forgetting from parameter sharing. Importantly, SoLA supports both insertion and deletion of edits. By simply removing key from the semantic routing, specific edits can be precisely revoked, restoring the model’s original behavior. To our knowledge, this reversible rollback editing capability is the first to be achieved in existing literature. Furthermore, SoLA integrates the decision-making process into the edited layer itself, eliminating the need for auxiliary routing networks and enabling end-to-end decision-making process. Extensive experiments across three representative tasks (document classification, question answering, and hallucination correction) demonstrate that SoLA effectively learns and retains edited knowledge, achieving accurate, efficient, and reversible lifelong model editing.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses catastrophic forgetting and semantic drift in lifelong model editing. The authors propose SoLA, a framework that assigns a separate, parameter-efficient LoRA module to each knowledge edit. After training, the LoRA module and its corresponding semantic "key" (generated from the input representation) are frozen. During inference, a semantic routing mechanism matches the input "query" with stored "keys" to dynamically activate the relevant LoRA module. The key innovation is enabling reversible edits: an edit can be precisely undone by removing its "key" from the routing table, without retraining. Experiments on document classification, question answering, and hallucination correction demonstrate SoLA's effectiveness in achieving accurate, efficient, and reversible editing.

### Strengths
Strength 1: The proposed "reversible editing" mechanism is a significant and novel contribution to the field of model editing. In real-world applications requiring frequent model knowledge updates and corrections, the ability to precisely and efficiently undo erroneous or outdated edits is highly valuable.

Strength 2: Effective mitigation of catastrophic forgetting

### Weaknesses
Weakness 1: I see a scalability issue, and this is the method's primary theoretical and practical limitation. The design implies that auxiliary parameters grow linearly and indefinitely with the number of edits. Although each LoRA module is small, in a true **lifelong learning** scenario involving millions of updates, this approach incurs significant storage and management overhead. This raises concerns about long-term scalability, especially compared to Mixture-of-Experts methods that use a fixed parameter pool, representing a clear drawback. I encourage the authors to conduct a quantitative analysis of this issue to properly assess the method's performance in more challenging scenarios (e.g., 3k–100k edits).

Weakness 2: Another issue is the limitation in knowledge generalization, such as hierarchical reasoning and attribute inheritance. The proposed design with independent LoRA modules may struggle in these aspects. I encourage the authors to present relevant case studies and analysis.

### Questions
See weaknesses

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
This paper proposes a continual model editing method that enables controllable knowledge editing by storing a semantic key and a corresponding LoRA for each sample that requires editing.

### Strengths
- The method proposed in this paper allows more controllable addition and removal of knowledge during model editing.
- The paper is well-organized and structured.

### Weaknesses
- The proposed method requires adding a key and a separate LoRA for each sample to be edited, and these parameters are frozen and stored together with the model. Therefore, as the number of samples to be edited increases, the total number of model parameters will grow significantly.
- This paper only compares the number of parameters involved during editing, but lacks a comparison of the total model parameters for each method.
- It does not discuss and compare important methods in the field, such as AlphaEdit [1].
- The paper edits only 1,000 samples (as shown in Figure 3). However, methods in this field typically use a larger number of edits — for example, AlphaEdit performs edits on 3,000 samples.

[1] Fang, Junfeng, et al. "AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models." The Thirteenth International Conference on Learning Representations 2025.

### Questions
Refer to the Weaknesses section

### Soundness
3

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
3

### Summary
This work introduces SoLA, a new framework for lifelong model editing. The main goal is to fix the semantic drift and forgetting problems seen in sequential editing. The core idea is pretty neat, they encapsulate every single edit into an independent LoRA module. This module, along with a corresponding semantic key derived from the input, is frozen after training. During inference, the model just matches the input's semantic representation to this frozen key table to activate the right LoRA. This freezing is what stops the drift. They also claim this is the first method that allows for reversible editing , you just delete the key to roll back a change. The paper shows SoLA performing well on classification, QA, and hallucination tasks, beating other recent methods.

### Strengths
1. The reversible editing capability is a fantastic and novel contribution. The ability to "undo" a specific edit by just removing a key is a highly practical feature for managing model knowledge.




2. The core mechanism of freezing both the LoRA module and its semantic key after training is a simple and effective solution to the semantic drift problem that plagues methods relying on dynamic clustering.



3. The method is very parameter-efficient in terms of trainable parameters. As shown in Figure 1a, it achieves strong performance with minimal additional trainable parameters compared to baselines.

4. Integrating the decision mechanism directly into the edited layer, rather than requiring an auxiliary router network, is a clean design choice that simplifies the architecture.

### Weaknesses
1. The "one LoRA per edit" approach raises major scalability concerns. Total storage seems to grow linearly with every new edit, which feels unsustainable for a true "lifelong" system, regardless of low trainable parameter counts.

2. The semantic key matching, using just the last token's embedding, seems fragile. The paper doesn't really investigate the risk of key collisions as the number of edits scales up.

3. A key ablation seems missing. It's unclear if the performance benefit comes from the "one LoRA per edit" design or simply from freezing the routing map. Comparing against MELO but with its cluster centers frozen would have been insightful.

### Questions
1. Could you clarify the storage implications? Does "one LoRA per edit" mean storing thousands of separate LoRA modules? How can that scale?

2. What is the actual total storage overhead for SoLA after 1000 edits compared to MELO, not just the trainable parameter count?

3. I'm also wondering about the key matching reliability. What's the failure rate when different edits have similar query embeddings? And how sensitive is the model's performance to the specific matching threshold $\alpha=0.01$ you chose?

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
4

### Summary
The paper proposes SoLA, a semantic routing-based LoRA framework for reversible lifelong model editing.
Each edit is represented as an independent LoRA module associated with a key vector, and a master decision-making mechanism enables dynamic semantic activation, effectively preventing semantic drift and catastrophic forgetting.
Extensive experiments demonstrate the effectiveness of SoLA in achieving reversible and stable model editing across multiple tasks.

### Strengths
The paper is well written and easy to understand.

The proposed method is clear and straightforward to implement.

Experimental results look promising and demonstrate the effectiveness of the approach.

### Weaknesses
* The main concern for me lies in the novelty of the proposed method, for the following reasons:

**Many similar ideas have already been explored in the fields of continual learning and multi-task learning**. 
For example, a classic continual learning baseline, O-LoRA [1], also trains separate LoRA adapters for different tasks and uses a routing mechanism during inference to activate one or multiple adapters.

In your method, routing based on distance is a common practice, and other variants (e.g., routing via validation data or KL divergence of output token distributions) have already been studied. 

Overall, there seems to be **limited conceptual innovation** compared to existing approaches.

More specifically, the core components in Section 3.3 rely heavily on previous work for obtaining the key vectors, without introducing new insights (a minor suggestion: please add a citation to “Following prior work” in line 244). Similarly, the routing mechanism described in Section 3.4 also lacks novelty, as discussed above.

[1] Orthogonal Subspace Learning for Language Model Continual Learning. EMNLP 2023

* Regarding Figure 1(a), I have several questions. 

i) Since both your method and MELO use LoRA for training, why does your method show a smaller number of trainable parameters (0.1 on the x-axis) while others have larger values? 

ii) In line 255, you claim that “SoLA significantly reduces the number of trainable parameters and the overall computational cost”, but while your approach may reduce trainable parameters, it **increases memory usage**, as a separate LoRA must be stored for each task. It is also unclear how the computational cost is significantly reduced compared to prior LoRA-based tuning.

* Finally, in Section 3.4, please clarify why the *first edited layer* is chosen as the **master decision layer**. What happens if you choose the last edited layer instead? This part lacks an **ablation study** to justify the design choice.

### Questions
Please see the Weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
2
