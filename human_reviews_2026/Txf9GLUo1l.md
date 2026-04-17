# Circuit-level Steering for Personalized Knowledge Injection in Language Models

- Decision: Reject
- Scores: 4, 4, 6, 8

## Abstract
As large language models (LLMs) become central to user-facing applications, effective personalization, adapting models to individual users’ evolving facts and contexts, has become crucial. However, existing approaches struggle with mutable personal knowledge: finetuning can embed static user information but is costly and prone to catastrophic forgetting, while knowledge editing methods rely on pre-cached representations from large corpora like Wikipedia, which are unavailable or unsuitable for personal domains due to data scarcity and privacy concerns. We formalize updating the fact-level personalization with mutable knowledge as a new task, constructing synthetic Personal Knowledge Graphs (PKGs) that capture user information across time points to evaluate models' ability to incorporate updates without degrading existing knowledge. Drawing on insights from mechanistic interpretability, we discover that personal facts are encoded in localized circuits within LLMs.
We propose SPIKE (Steering for Personalized Knowledge Injection), a parameter-efficient method that combines adapter modules with steering-based activation injection, targeting identified personal knowledge circuits. This approach enables the precise integration of new user-specific facts, including previously unseen triples, while maintaining the integrity of prior knowledge. Our experiments demonstrate that SPIKE effectively balances the accuracy of incorporating new facts with the preservation of existing knowledge, offering a practical solution for continual personalization in settings where user information evolves frequently.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a method for personalized knowledge injection in LLMs through circuit-level steering. Instead of retraining or fine-tuning, the approach identifies and manipulates activation circuits associated with specific knowledge domains or personalization needs, then steers these activations at inference time to inject desired knowledge.

The main contributions are:
- Using causal tracing and attribution techniques, the method isolates latent circuits linked to factual knowledge.
- Learning targeted steering vectors that can inject user-specific facts or override incorrect knowledge while leaving unrelated knowledge unaffected.
- Benchmarking the approach on personalized QA and fact-editing tasks, showing improvements in both edit reliability (the model applies the edit when needed) and edit locality (the edit does not spill over to unrelated facts).
- Demonstrating advantages over fine-tuning, ROME, MEMIT, and other editing methods in terms of accuracy, locality, and minimal side effects.

Overall, it is a timely paper. It introduces a novel and interpretable method for personalized knowledge injection that outperforms existing editing baselines. The main areas for improvement are in scalability, broader evaluation (beyond factual edits), and addressing privacy/safety implications. With these extensions, this work could become a foundational approach to safe and controllable personalization in LLMs.

### Strengths
- The paper proposes novel framing of personalization as circuit-level steering rather than weight modification or prompt engineering, and introduces a principled way of isolating knowledge-bearing subcircuits and controlling them for personalization.
- Ablation studies provide insight into which layers and circuits are most impactful for successful edits.
- The paper offers a lightweight, reversible, and interpretable alternative to fine-tuning-based personalization, and would have broader implications for safe and controllable model editing. Personalized knowledge injection is highly relevant for user-facing LLM applications (e.g., personal assistants).

### Weaknesses
- The paper focuses on factual personalization (e.g., user-specific facts in QA). Less clear how well it generalizes to complex behavioral personalization (tone, reasoning style, ethical preferences), and has limited exploration of multi-hop reasoning edits or temporal knowledge updates.
- The paper does not explore that would it benefit from evaluation against retrieval-augmented personalization (e.g., RAG with user profiles).
The work assumes that circuits can be stably isolated and steered, but does not deeply theorize why steering works robustly or when it might fail (e.g., entangled circuits).
- Personalized knowledge injection raises privacy/security questions (e.g., if steering vectors could be extracted). This is not discussed in depth.
- The paper does not fully address how well this scales to very large models (e.g., >70B parameters).

### Questions
- Can the method handle personalization in style, reasoning patterns, or ethical preferences, or is it limited to factual knowledge?
- How does circuit-level steering compare to retrieval-based personalization in terms of accuracy, scalability, and safety?
- How practical is circuit identification for very large models or for injecting many personalized facts? Could this process be batched or approximated?
- Could malicious actors exploit steering to inject harmful knowledge into shared models? How might safeguards be designed?
- Could you provide more details of personal KG construction? How dose it update? 
- How robust is circuit steering to prompt variation or adversarial input? Could injected knowledge leak into unrelated contexts?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper focuses on the core challenge of personalized knowledge injection in large language models (LLMs), innovatively proposing to formalize the variable personal knowledge update task as a knowledge graph (PKG) temporal update problem, and based on mechanical interpretability, discovering that personal knowledge is encoded in local circuits in LLMs, and then designing a circuit level guidance framework SPIKE. The paper combines adapter modules with guided activation injection to achieve precise integration of user specific facts while maintaining the integrity of existing knowledge. The experiment covered multiple models (GPT-J, Qwen2.5-7B Instruction) and multiple datasets (PeaCoK Ex, PerInfoKG), verifying the advantages of the method in balancing the accuracy of new facts and the retention of old knowledge. The research perspective is novel and the technical route is clear, providing valuable solutions for the personalized knowledge updating field of LLMs. However, there is still room for improvement in terms of theoretical depth, experimental completeness, and practical applicability of the paper.

### Strengths
- Personalizing LLM by focusing on dynamic and privacy sensitive factual knowledge updates is a highly practical and challenging new research direction.
- The SPIKE method cleverly integrates mechanism interpretability (circuit localization), knowledge graph (structured representation), and activation steering (precise intervention), providing a new paradigm for solving personalized knowledge injection problems that does not rely on large-scale corpora.
- The experimental results (Table 2) indicate that SPIKE achieved the highest overall performance (Total Score) on both datasets, especially on the challenging PerInfoKG dataset, where it maintained high accuracy (94.38%) while also achieving extremely high locality (96.34%).

### Weaknesses
- The empirical validation relies entirely on two synthetically constructed datasets, PeaCoK-Ex and PerInfoKG. While these datasets structurally model personal knowledge graphs, they may not capture the full complexity, noise, and diversity (e.g., unstructured or implicit knowledge) of real-world personal data, potentially limiting the generalizability of the findings.
- The core methodological choice to target attention heads over FFNs is justified by a circuit analysis (using HeadMap) performed primarily on older models (GPT2-Large and GPT-J) . It is unclear how well these findings, and thus the design of SPIKE, generalize to the more complex architectures of modern LLMs (e.g., LLaMA 3, Qwen 2.5).
- The SPIKE methodology introduces a separate, complex KG-LLM Alignment Module  that requires its own training phase. This module, which includes GNN encoders and attention mechanisms , adds computational overhead for training and necessitates access to the (potentially private) KG data to learn the alignment.
- The baseline comparison in Table 2  is limited to fine-tuning variants and locate-then-edit methods (ROME, MEMIT-Merge, AlphaEdit). It omits several prominent and highly relevant classes of editing methods, such as memory-based (e.g., SERAC), in-context learning (e.g., IKE, ICE), or meta-learning (e.g., MEND, WISE), making the performance claims of SPIKE less comprehensive.

[1] Zheng C, Li L, Dong Q, et al. Can we edit factual knowledge by in-context learning?[J]. arXiv preprint arXiv:2305.12740, 2023.

[2] Qi S, Yang B, Jiang K, et al. In-context editing: Learning knowledge from self-induced distributions[J]. arXiv preprint arXiv:2406.11194, 2024.

[3] Wang P, Li Z, Zhang N, et al. Wise: Rethinking the knowledge memory for lifelong model editing of large language models, 2024[J]. 

[4] Mitchell E, Lin C, Bosselut A, et al. Memory-based model editing at scale[C]//International Conference on Machine Learning. PMLR, 2022: 15817-15831.

- The main evaluation in Table 2 is conducted on only two models: GPT-J-6B and Qwen2.5-7B-Instruct. This lacks diversity and omits widely-used, foundational model families like LLaMA and Mistral, making it difficult to assess the architectural generalizability of the SPIKE method.

- The evaluation indicators in Table 2 only involve Acc and Loc, and more challenging scenarios such as generalization and portability need to be considered for general knowledge editing scenarios(e.g., zsre、counterfact、wiki_recent).

- The RAG baseline comparison in §5.3.1 is performed assuming 100% retrieval success. This idealized setting does not reflect a true RAG system's performance; instead, it represents a "sufficient context"  scenario.

[1] Joren H, Zhang J, Ferng C S, et al. Sufficient context: A new lens on retrieval augmented generation systems[J]. arXiv preprint arXiv:2411.06037, 2024.

### Questions
- How confident are the authors that the performance of SPIKE will translate to real-world scenarios involving noisy, unstructured, or implicit personal knowledge, which is not represented in the synthetic PeaCoK-Ex and PerInfoKG datasets?
- Given that the foundational circuit analysis  was conducted on GPT-2/GPT-J, what evidence suggests that attention heads, rather than FFNs, remain the optimal intervention points for personal knowledge in newer architectures like LLaMA 3 or larger Qwen models?
- What is the computational cost (e.g., training time, parameter count) of the KG-LLM Alignment Module, and how does requiring a separate training stage on the personal KG data impact the practical deployment speed and privacy footprint of the SPIKE framework?
- How would SPIKE compare against memory-based (e.g., SERAC), in-context learning (e.g., IKE, ICE), or meta-learning (e.g., MEND, WISE) baselines, which are also designed to handle sequential updates and manage interference, yet were not included in the main comparison in Table 2?
- Can the authors provide evidence or justification that the circuit-level steering approach and its superior performance, as shown on GPT-J and Qwen2.5-7B, would generalize effectively to other popular model architectures such as LLaMA or Mistral?
- Can we design more challenging problems and scenarios than acc, such as generalization and portability?

### Soundness
2

### Presentation
2

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
The paper tackles editing/updating user-specific facts in LLMs with minimal collateral changes by modeling personal info as a knowledge graph and specifying both where to intervene in the model and how to inject the update signal. It uses circuit analysis to show personal facts are localized in attention-head circuits and that targeting heads is far more effective than FFNs. They propose SPIKE aligns KG-triple deltas with internal representations and steers only the top-k important heads, improving accuracy while preserving locality without full-model fine-tuning.

### Strengths
1. Well-scoped problem + principled solution. The paper designs SPIKE to answer the two key questions—where to edit (personal-knowledge circuits) and how to inject updates (a delta derived from the KG)—with evidence that attention-head targets beat FFNs by a wide margin under similar parameter budgets.
2. SPIKE maintains high locality while integrating new facts and generalizes to unseen triples, outperforming a strong RAG baseline in that setting.
3. The paper probes which heads matter, how many heads to steer, and whether subgraph features help; removing subgraph features or replacing important heads degrades performance, and using too few heads hurts locality.

### Weaknesses
1.  External validity is limited by synthetic setups and assumptions. All experiments use synthetic personal-KG datasets (no real user data or user studies), and the approach assumes personal information is available as a well-formed Personalized Knowledge Graph to compute update deltas—conditions that may not hold in practice and raise open deployment questions despite the ethics discussion.
2. Missing a similar work as baseline. Knowledge Graph Tuning: Real-time Large Language Model Personalization based on Human Feedback.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces SPIKE (Steering for Personalized Knowledge Injection): a circuit-aware activation-steering method to update mutable, user-specific facts inside LLMs without full fine-tuning or pre-cached representation. The workflow is: (i) identify personal-knowledge circuits (a sparse network of attention heads) via head-importance scoring; ii) learn a knowledge-graph-LLM alignment module that maps a triple/subgraph change representing a factual change into a difference vector Δ. iii) add the Δ into the selected heads’ activations at run-time to realize the update. Experiments on two synthetic personal-KG datasets (PeaCoK-Ex, PerInfoKG) show high Accuracy (alignment with updated fact) and strong Locality (effect on unrelated/unchanged fact), outperforming editing, LoRA, circuit-selective finetuning on GPT-J and Qwen2.5-7B.

### Strengths
1.	Clearly identify the problem within existing solutions and address them with a novel approach: The paper identifies a clear, practical, and important gap in existing literature: finetuning approach is computationally expensive and risk catastrophic forgetting; knowledge editing like ROME and MEMIT requires access to the precached representation of original model’s knowledge, which is not accessible due to commercial and privacy concern. The paper proposes a clever approach by using an alignment module that only requires aligning a specific fact with its updating version during training, and use it to steer activation
2.	Conduct a thorough investigation on where to apply activation changes: The authors uses the HeadMap to investigate to identify a subset of attention head only, and ablation study was conducted to compare applying activation changes elsewhere; Another significant decision the authors clearly justified is the position to apply the alignment module (targeting attention heads over FFNs), and experimental evidence is provided. The experimental methodology is sound and well thought out.
3.	Generalization Capability: The experiment on "unseen triples (factual changes)" is a key strength. It shows that the KG-LLM Alignment Module learns a generalizable function for mapping knowledge changes to activation deltas, rather than just memorizing a fixed set of edits.

### Weaknesses
1.	Synthetic & Structured Data: The paper's primary weakness is its exclusive reliance on synthetic datasets - PeaCoK-Ex, PerInfoKG. These have clear relationship-level definitions of fact such as (Patient name, medical condition, disease) and simple representation of triples. However, the real-world factual representation is far more complicated than this and maybe ambiguous: for example, The doctor ran all the tests and said I'm perfectly healthy, but I still feel exhausted and ill all the time. In this example the facts are more difficult to extract and correspond to more complex relationships. This would require more triples or more complex construct to encode, potentially requiring a more complex alignment module.
2.	Implicit Assumption: A Global "Personal Circuit": The circuit discovery appears to identify a single, global "personal knowledge circuit" for each model, averaged over the entire dataset. This assumes that all personal facts (for all users, across all relations like 'Job', 'Medical Condition', 'Location') are encoded in the same sparse subgraph. This seems biologically and computationally implausible. A more likely scenario is that circuits are sparse but content-dependent or even user-dependent. The paper does not test this assumption.
3.	Limited scope on how the Locality is measured: The paper's experiments are focused on the triples contained within their two custom-built Personal Knowledge Graphs (PKGs). Locality is specifically defined as checking the model's accuracy on facts (triples) that were not part of the update. Hence, the Locality metric only measures forgetting of other personal facts, not general knowledge. This makes the approach seem more "lightweight" than it may be in practice.

### Questions
1.	On real world representation: The method is entirely dependent on clean, structured KG triples. What is your proposed path for applying SPIKE to real-world personalization, where user facts must be extracted from unstructured, messy, and often implicit text such as emails and chats?
2.	Activations of alignment module: the paper did not explicitly mention how the activation of alignment module is done. Is it similar to a RAG? Would the activation of alignment module be accurate if the an ambiguous question is asked, let’s say what is the daily routine of Mike, which likely needs the information of this occupation (pilot, in your illustrative example)?
3.	On training of the Alignment module: it seems that the training is one-off in your experiment. How would the training take place, suppose Mike’s occupation undergoes a third change (student->pilot->CEO)? Do you need to gather all the training data again and then train the module? Is there privacy concern in gathering all data from other people, since you suggest an alignment module on each user is prohibitively expensive, so all users need to share an same alignment module?

### Soundness
3

### Presentation
3

### Contribution
4
