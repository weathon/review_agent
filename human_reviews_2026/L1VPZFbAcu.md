# WebRAGent: Retrieval-Augmented Generation for Multimodal Web Agent Planning

- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Trajectory data, capturing multimodal human actions and states, are pivotal for building autonomous GUI agents and transferring skills across tasks, encoding knowledge by compressing past experience into structured Markov sequences. Yet current methods for trajectory modeling remain fragmented, often relying on task-specific heuristics or textual signals. Progress on multimodal trajectories has been limited by the difficulty of representing visual information within long-step histories that exceed model context windows. Hence, how to effectively learn from multimodal trajectories remains a major and insufficiently addressed challenge amid ever-growing datasets. In this work, we introduce Multimodal Trajectory Retrieval, bridging the gap between universal retrieval and agent-centric trajectory modeling. We construct the Unified Agent Trajectory Dataset (UATD) from annotated demonstrations and states across diverse real-world scenarios. Building on this, we present GAE-Bench, a benchmark containing a large number of trajectory-based retrieval pairs. Our proposed GAE-Retriever, a multimodal retriever based on vision–language models that uses token selection and GradCache to optimize the contrastive objective. Over multiple web-agent datasets, it surpasses strong baselines on retrieval recall. To demonstrate potential downstream applications, we develop WebRAGent, a retrieval-augmented web agent that integrates GAE-Retriever and supports both DOM- and vision-based observations. WebRAGent proves effective on both textual and visual retrieved knowledge, achieving performance gains of over 15\% vs. non-retrieval on the Online-Mind2Web benchmark.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents WebRAGent, a retrieval‑augmented multimodal web agent designed to leverage past GUI trajectories for better decision‑making. The authors introduce the Unified Agent Trajectory Dataset (UATD) and propose the new task of Multimodal Trajectory Retrieval, releasing the benchmarks GAE‑Bench and GAE‑Bench‑lite with over 700K trajectory retrieval pairs. They develop GAE‑Retriever, a VLM2Vec‑based model using token selection and GradCache for efficient contrastive training, and integrate it into the WebRAGent framework. Experiments across five datasets show substantial recall improvements over strong multimodal baselines, and on Online‑Mind2Web, WebRAGent achieves 15–22% higher success rates than non‑retrieval models.

### Strengths
-  **Innovative dataset construction and benchmarks.**
The paper introduces a unified methodology for integrating heterogeneous GUI‑based trajectory data, resulting in the Unified Agent Trajectory Dataset (UATD) and two large‑scale benchmarks (GAE‑Bench and GAE‑Bench‑lite). This contributes valuable resources for evaluating multimodal trajectory retrieval and provides a standardized foundation for future agent studies.

-  **Novel and well‑structured web‑agent framework.**
The proposed WebRAGent framework innovatively integrates multimodal retrieval with generation in agent planning. Its modular design—combining observation, retrieval, memory, and planning—demonstrates a clear architectural innovation and effectively bridges trajectory -   

- **Comprehensive and substantial work.**
The paper presents extensive data preparation, thorough model development, and large‑scale experiments across multiple benchmarks. The amount of work is significant, covering dataset unification, retriever training, and online evaluation, showing strong technical depth and implementation effort.

### Weaknesses
- **Lack of clarity in technical details.**
Several key components are insufficiently explained, such as the reward design, data annotation procedures, and implementation specifics of dataset generation and evaluation. These omissions make it difficult to reproduce and precisely understand how the framework works. (See detailed questions in the Questions section.)
- **Unclear core contribution.**
The proposed GAE‑Retriever primarily builds upon existing methods like VLM2Vec and integrates known techniques such as Token Selection and GradCache. Since these components are not original, the novelty of the contribution is uncertain. If the main innovation lies in the integration, the authors should provide clear ablation studies or quantitative evidence demonstrating the necessity and contribution of each module.
- **Onfair performance comparison.**
The retriever is trained and evaluated on data from the same source, which naturally favors high recall scores. Comparisons with untrained or zero‑shot models are therefore not entirely fair. Moreover, the paper does not compare WebRAGent’s web‑retrieval capability against existing Web search or Web‑retrieval models, making it difficult to know whether the performance is better than the existed systems or not.
- **Formatting and presentation issues.**
Figures and tables are sometimes awkwardly placed, often disrupting the reading flow. Aligning them consistently—at the top of pages—would significantly improve the presentation quality.

### Questions
- **Reward mechanism ambiguity.**
The paper mentions the use of an “LLM‑as‑judge” strategy for rewarding but provides no implementation details. Which specific LLM model was used for judging (e.g., GPT‑4, GPT‑4‑turbo, or others)? What were the prompts, scoring criteria, and calibration procedures? Given that the reward signal directly affects policy and evaluation, this should be clarified in detail.
- **Unspecified reranking model.**
The framework claims to apply an LLM‑based reranking step after retrieval, yet there is no description of the rerank model, prompt design, or how it integrates with GAE‑Retriever. How is the reranker implemented, and what quantitative performance gain does it contribute?
- **Insufficient transparency in data construction.**
At line 377 the paper states “Data are annotated with gpt‑4o‑mini‑2024‑07‑18.” but does not explain the detailed annotation process. What prompts were used for labeling? How were data quality and potential data leakage from pretrained sources verified?
- **Lack of comparison with other web‑search models.**
WebRAGent’s web‑retrieval performance is not compared with existing Web search or Web‑retrieval systems, such as dense retrievers or LLM‑based search agents. Without such baselines, it remains unclear whether the model actually advances the state of web‑scale retrieval.
- **Computational cost of retrieval augmentation.**
The paper does not quantify how much additional latency or computation the retrieval process introduces during inference. How large is the retrieval database, and what is the average time‑to‑action compared to the non‑retrieval baseline?
- **Fairness of planning models across baselines.**
In the Planning and Action section, the authors state that WebRAGent uses GPT‑4.1 for DOM mode and OpenAI’s computer‑use‑preview model for Vision mode—both very strong, closed‑source models. Do the non‑retrieval baselines employ exactly the same planners? If not, how can we separate the performance gain attributable to retrieval augmentation from that potentially caused by stronger planning models?

### Soundness
3

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
This paper addresses the question of how to retrieve the most relevant parts of past multimodal trajectories to support planning. Part of the motivation is that storing all trajectories with multimodal contents in the context is impractical. The paper constructs a trajectory retrieval corpus called Unified Agent Trajectory Dataset (UATD) from annotated demonstrations and states across diverse real-world scenarios. Building on this, it constructs GAE-Bench, a benchmark containing a large number of trajectory-based retrieval pairs.

Further, the paper proposes GAE-Retriever, a retriever for multimodal trajectories that uses token selection and GradCache to optimize
the contrastive objective. It also introduces WebRAGent, a retrieval-augmented web agent that integrates GAERetriever. Experiments are performed on the Online-Mind2web benchmark.

### Strengths
1. The core idea of retrieving similar trajectories for reuse is interesting and intuitive.
2. The GAE-Bench benchmark introduced in this paper for trajectory retrieval is a valuable resource with several patterns of trajectory retrieval, such as text-to-state, text-to-trajectory, state-to-state, etc.
3. Empirical results show a significant boost in performance over non-retrieval baselines on the Online-M2W benchmark.

### Weaknesses
1. The Unified Agent Trajectory Dataset introduced in this paper is really not novel. Similar aggregated trajectory dataset already exists [1]
2. For the Online-Mind2web results, rather than choose a simple MLLM baseline, the authors should add trajectory retrieval to exising SOTA or close to SOTA model, e.g. SeeAct [2] or AgentE [3].
3. The authors selected a subset of 100 tasks from Online-M2W without justification for not using the original dataset.

[1] Xu, Yiheng, et al. "Aguvis: Unified pure vision agents for autonomous gui interaction." ICML'25.

[2] Zheng, Boyuan, et al. "Gpt-4v (ision) is a generalist web agent, if grounded." ICML'24.

[3] Abuelsaad, Tamer, et al. "Agent-e: From autonomous web navigation to foundational design principles in agentic systems." arXiv preprint arXiv:2407.13032 (2024).

### Questions
N.A.

### Soundness
3

### Presentation
2

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
This paper addresses a critical challenge in the development of autonomous GUI agents: how to effectively learn from and utilize vast amounts of multimodal trajectory data (states, actions, visual observations) that often exceed the context windows of current models.

The authors present a comprehensive framework to tackle this issue:

1. **Unified Agent Trajectory Dataset (UATD):** They first curate and unify five existing GUI agent benchmarks into a standardized dataset, encompassing 7,747 demonstrations and over 82,000 states.
2. **Multimodal Trajectory Retrieval Task:** They formally define a new task, "Multimodal Trajectory Retrieval," to bridge the gap between general-purpose retrieval and agent-centric modeling.
3. **GAE-Bench:** Based on this new task, they construct a large-scale benchmark (GAE-Bench) with 714,628 retrieval pairs, derived from 12 extraction patterns that capture both temporal and semantic relationships.
4. **GAE-Retriever:** They propose a multimodal retriever built on VLM2Vec, employing optimizations like token selection and GradCache to efficiently train on high-resolution image sequences and large batches.
5. **WebRAGent:** Finally, they integrate their retriever into a retrieval-augmented agent framework, WebRAGent, which demonstrates significant performance gains (15-22%) over non-retrieval baselines on the Online-Mind2Web benchmark.

### Strengths
1. **Problem Significance:** The paper correctly identifies a timely and critical problem. As trajectory datasets grow, RAG is a logical and necessary step to scale agent capabilities beyond in-context learning.
2. **Foundational Dataset Contribution:** UATD and GAE-Bench are significant contributions in their own right. The engineering effort to unify heterogeneous datasets (web, mobile, desktop) into a standardized format (Section 3.1) is substantial and highly valuable for the community.
3. **Novel Task Formulation:** The "Multimodal Trajectory Retrieval" task, with its 12 extraction patterns (Figure 1), is a key conceptual contribution. This detailed formulation is crucial for training a robust retriever that understands both temporal sequence and semantic intent.
4. **Pragmatic Model Design:** The GAE-Retriever (Section 4.2) is well-designed. Using a VLM (VLM2Vec) backbone instead of CLIP-based models is well-justified for handling arbitrary combinations of multimodal inputs (lines 92-95). The use of token selection and GradCache to tackle the practical constraints of training with multiple high-resolution images is a critical and well-thought-out optimization.
5. **Strong Empirical Validation:** The paper closes the loop by demonstrating that its retrieval model directly translates to a 15-22% success rate improvement in a downstream planning task (WebRAGent). This is a very convincing validation of the entire pipeline.

### Weaknesses
While this is an excellent paper, there are a few areas that could be clarified or strengthened:

1. **Justification of "Silver Trajectories":** A key part of GAE-Bench is the semantic retrieval task (q → τ∼, lines 233-239). The authors generate "silver trajectories" via entity substitution. The example given ("Buy a t-shirt for children on Amazon" → "Order a laser printer on eBay," lines 237-239) seems to represent a pair with very **different task flows**, even if the high-level intent ("shopping") is similar.
   - **Concern:** This could be a very noisy training signal. Does retrieving a trajectory for "buying a printer" actually help an agent "buy a t-shirt," or does it introduce confusion?
   - **Recommendation:** The authors should provide a clearer justification for this data augmentation strategy. How is this "silver" pair more helpful than a hard negative?
2. **Architectural Novelty of GAE-Retriever:** The paper calls GAE-Retriever a "novel...framework" (line 90) but also states it is "built on VLM2Vec" (line 91) and uses optimizations (GradCache) from VLM2Vec (line 309).
   - **Recommendation:** The authors should more precisely articulate the **architectural novelty** of GAE-Retriever itself, distinct from VLM2Vec. If the primary novelty lies in the *application* and *task-specific training* (i.e., being the first to successfully apply this architecture to the multimodal trajectory retrieval task), this should be stated clearly.
3. **Lack of Qualitative Analysis:** The 15-22% performance gain is impressive, but the paper does not explain *why* it works.
   - **Question:** What kind of knowledge is being retrieved? Is it high-level planning steps (e.g., "log in first, then search") or low-level interaction details (e.g., "click this specific icon")?
   - **Recommendation:** The paper would be significantly strengthened by adding a qualitative analysis section with 1-2 concrete examples. Show a task where the baseline fails and WebRAGent succeeds, and—crucially—show the *actual retrieved trajectory* that made the difference.
4. **Scope Mismatch (UATD vs. WebRAGent):** The UATD is presented as a highly general dataset unifying "web, mobile, desktop, and embodied environments" (line 69). However, the downstream validation (WebRAGent) is only performed on a web-based benchmark (Online-Mind2Web). This leaves the claims about cross-platform generalization underexplored.

### Questions
1. **On "Silver Trajectories":** (See Weakness #1) How do you ensure that the generated silver trajectories share a similar **procedural flow**, rather than just being semantically related at a high level? The t-shirt vs. printer example seems to represent very different procedures.
2. **On Token Selection:** You use a UI-connected graph in RGB space to merge similar patches (lines 299-302). What are the advantages of this over a simpler baseline like bicubic resizing/downsampling of the image? Is there a risk of merging small but critical UI elements (e.g., a checkbox) that are similar in color to their background?
3. **On Inference Latency:** What is the computational overhead (latency) introduced by the GAE-Retriever step during WebRAGent's inference? How does this trade-off against the 15-22% gain in success rate?
4. **On "Hard Tasks":** You mention larger gains on "hard tasks" (line 106). Could you provide a concrete example of a "hard task" and qualitatively explain why retrieval was so beneficial for it?

### Soundness
3

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
3

### Summary
This paper introduces WebRAGent, a framework for retrieval-augmented multimodal web agent planning. The motivation is that progress in multimodal trajectory learning is limited by the difficulty of representing rich visual information within long interaction histories that exceed a model’s context window.
To address this, the authors propose multimodal trajectory retrieval, along with:
- A benchmark for trajectory-based retrieval pairs,
- A model (GAE-Retriever) for multimodal retrieval based on a vision-language backbone, and
- A retrieval-augmented web agent integrating the retriever into agent planning.

### Strengths
- The paper presents a novel and intuitive idea, aiming to connect retrieval-augmented generation with multimodal web agent reasoning.
- It is well-written and easy to follow, with clear structure.
- The release of code and resources is great and improves reproducibility.

### Weaknesses
- The motivation for the multimodal trajectory retrieval task is weak and needs clearer justification—why is this problem important, and what real-world gap does it fill?
- The introduction of the GAE-Retriever lacks sufficient motivation and integration into the overall narrative.
- Figures 1 and 3, and Tables 1, 3, 9, and 10, are difficult to read due to poor formatting and small font sizes.
- The limitations of the proposed approach are not discussed.
- The related work section oversimplifies the historical relationship between retrieval and generation in the context of multimodal retrieval. Multimodal retrieval methods have existed long before generation-based approaches became dominant.
- The overall storyline feels disjointed: it is not entirely clear how the benchmark, retriever, and agent components connect to form a coherent research contribution.

### Questions
- Could the authors clarify how the benchmark, retriever, and agent fit together conceptually and experimentally within one unified framework?
- Please expand on the motivation behind the proposed multimodal trajectory retrieval task; why is it necessary, and what unique challenges does it address?
- It would be helpful to include a single overview figure illustrating how all components (benchmark, retriever, agent) interact within the proposed system.

### Soundness
2

### Presentation
2

### Contribution
2
