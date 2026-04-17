# Agents as Knowledge Integrator and Utilizer in Multimodal Recommendation

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
The proliferation of online multimodal content has driven the adoption of multimodal data in recommendation systems. Current studies either enhance item features with multimodal data or construct additional homogenous graphs via multimodal data. However, a significant semantic gap exists between multimodal data and recommendation tasks. This gap introduces modality-specific noise irrelevant to recommendation tasks when enhancing item features and results in homogenous graphs built on multimodal data that fail to adequately consider users' historical behaviors. Fortunately, the multimodal information understanding and contextual processing capabilities of large language models (LLMs) have emerged as a promising approach to bridging this semantic gap.

To this end, we propose AgentMMRec, a novel agent-based framework that bridges the semantic gap via two cooperative agents: an Integrator Agent that uses LLMs to infer user preferences and item properties from multimodal data and users' historical behaviors, storing knowledge in a knowledge memory; and a Utilizer Agent that refines traditional homogenous item-item graphs using there knowledge, constructs behavior- and multimodal-aware homogenous graphs, and performs knowledge-enhanced reranking in recommendation stage. Integrator Agent updates the memory based on feedback from reranking performance. Extensive experiments on real-world datasets demonstrate that AgentMMRec outperforms existing multimodal recommendation models and exhibits superior performance across various data sparsity scenarios. Additionally, AgentMMRec can enhance the performance of existing multimodal recommendation models by leveraging the constructed knowledge memory.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
AgentMMRec is an LLM-powered agent framework for multimodal recommendation that tries to close the semantic gap between raw vision/text signals and collaborative-filtering objectives. The whole loop is pre-computed or run every E epochs to keep cost acceptable.
On three Amazon subsets (Baby, Sports, Clothing) the system beats 15 baselines and shows good robustness under data sparsity and item-cold-start. Code is provided.

### Strengths
1. Strong reproducibility, containing full prompts, statistics, and implementation details are given; anonymous GitHub link provided.
2. Feeding the agent-constructed graphs or re-ranker into existing models (SMORE, MENTOR, etc.) improves them, showing the memory is model-agnostic.

### Weaknesses
1. Scalability: Context-length limit Υ=5 neighbours is tiny; authors admit Υ>20 is infeasible.
2. Only Amazon review data (text + one image) are used; no short-video, audio, or social-media modalities.
3. Data leakage risk – the same LLM that reads item text/images at test time also generates training graphs; it may trivially match items by brand/title strings, inflating cold-start numbers.

### Questions
1. What is the end-to-end training time versus the strongest baseline on the largest dataset?
2. What happens if you replace the 7B-VL model with a 0.5B distilled LM – is the framework still cost-effective?

### Soundness
3

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
The paper proposes AgentMMRec, an agent-based multimodal recommendation framework leveraging LLMs to bridge the semantic gap between multimodal data and recommendation tasks. The architecture contains two agents — an Integrator Agent to infer user preferences and item properties from multimodal data and store them in a knowledge memory, and a Utilizer Agent to refine item-item graphs and rerank recommendations using this stored knowledge. Memory updates are triggered based on reranking performance feedback via prompt templates. Extensive experiments across multiple datasets show performance gains over multimodal graph-based baselines under normal and sparse data situations.

### Strengths
- Well-organized paper with thorough explanation of the proposed framework and clear methodology steps.
- Comprehensive experimental evaluation on multiple datasets, including sparsity and cold-start scenarios.
- The compatibility analysis showing benefits when components are transferred to other baselines is interesting and well reported.

### Weaknesses
- The graph structure improvement is incremental compared to existing multimodal + graph works such as; novelty is mainly in the systemization of components rather than in core graph construction. So the main innovation in my view is  "memory update via feedback". However, it is relatively simple and heavily reliant on prompt-template-guided LLM output, without deeper algorithmic or modeling advances. Therefore, the overall innovation point is still lacking.

- Heavy reliance on LLMs for preference/property extraction may limit reproducibility and interpretability, and much of the novelty lies in engineering details rather than theoretical contributions.

### Questions
- Could the authors clarify why the proposed memory update strategy, which is prompt-controlled, is preferable over more sophisticated memory storage research or methods?
- Is there any analysis of the robustness of the knowledge memory to noisy or biased prompt outputs from LLMs?

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
This paper addresses the "semantic gap" in multimodal recommendation—the gap between raw data (images, text) and the actual recommendation task. The authors propose AgentMMRec, a novel framework where Large Language Models (LLMs) act as agents to bridge this gap. A feedback loop is included where re-ranking performance is evaluated, and if a re-ranking is suboptimal, the Integrator Agent updates the user's preferences in the knowledge memory. Experiments on three Amazon datasets show that AgentMMRec significantly outperforms a wide range of state-of-the-art multimodal recommendation baselines on standard ranking metrics.

### Strengths
- The (Integrator, Utilizer) agent framework is the paper's primary strength. It provides a clean and powerful new paradigm for using LLMs in recommendation, separating the intensive reasoning/knowledge-generation phase from the application/utilization phase.
- The paper's core premise is to replace noisy, raw multimodal features with "knowledge" (inferred preferences $P_u$ and properties $P_i$) that is explicitly generated by an LLM to be task-relevant. This is a more direct and intelligent approach than traditional feature-fusion or alignment methods.
- The "knowledge memory" is a powerful and modular concept. The compatibility analysis in Table 2, which shows that transferring the "knowledge" (via +Graph or +Rerank) boosts the performance of other SOTA models, is extremely strong evidence for the paper's claims. It proves that the generated knowledge is high-quality and generally useful.
- The framework achieves significant gains over a very strong and comprehensive set of recent baselines (including models from 2023-2025) on all datasets (Table 1).
- The method's ability to infer item properties $P_i$ directly from multimodal data makes it highly effective for item cold-start (Table 4), which is a major practical advantage.

### Weaknesses
- The framework, as-proposed, is enormously complex and computationally expensive. The "pre-building" phase requires multiple LLM calls (using a 7B VLM) for every single user and every single item to generate the initial knowledge memory and graphs (Eqs. 1-8, 12). Then, it requires more LLM calls during training (every $E$ epochs) for re-ranking and knowledge updates (Eqs. 13-14). The paper dismisses this as a "pre-building" cost, but for any real-world dataset with millions of users and items, this phase is computationally intractable. A thorough cost and latency analysis is completely missing.
- The system's performance is critically dependent on a large and complex set of nine different, hand-crafted prompt templates (Appendices D.1-D.9). This introduces a significant and fragile "hidden" hyperparameter-tuning problem. It is unclear how robust this prompt-based knowledge generation is or how well it would generalize to new domains without extensive re-engineering of the prompts.
- The paper uses a threshold $Y$ (e.g., $Y=5$ in Appendix C.3) to randomly sample the interaction histories for power users or popular items, due to LLM context limits. This is a major information bottleneck. It means the "knowledge" for the most important users and items is being generated from a tiny, random fraction of their data, which could introduce significant noise and instability into the knowledge memory.
- Given the rapid progress in the vision–language modeling (VLM) community, the paper’s experimental setup feels somewhat outdated. Several state-of-the-art and larger-scale VLMs—especially those demonstrating advanced capabilities such as visual reasoning, multi-image understanding, or visual chain-of-thought—are missing from comparison. Incorporating or at least discussing stronger open-source or proprietary baselines (you can find example from the families like OpenAI, Gemini, Claude, Qwen, Hunyuan and so on) would strengthen the evaluation and better position the proposed approach within the current research landscape.

### Questions
1. Can the authors provide a practical analysis of the computational cost? How many total LLM calls (and tokens) are required for the "pre-building" phase on the Baby dataset? What is the added latency per batch during training due to the re-ranking (Eq. 13) and knowledge update (Eq. 14) steps?
2. How do the authors envision this framework scaling to a real-world recommender system with millions of users and items? The current approach of running an LLM inference for every user and item seems computationally infeasible.
3. The paper uses a small, fixed $Y$ (e.g., $Y=5$) to sample long histories. How sensitive is the model to this sampling? Is there a risk that the inferred knowledge for power users is unstable or unrepresentative, since it's based on a very small, random sample of their interactions?
4. The "knowledge" ($P_u$, $P_i$) is stored as generated text, this text is then immediately re-encoded by $t_{\theta}(\cdot)$ to be used in graphs (Eq. 5, 10) and GNNs (Eq. 15). Why not have the Integrator Agent output a structured JSON or a set of embeddings directly? What is the benefit of generating natural language text only to immediately embed it again?
5. The feedback loop (Eq. 14) only updates user preferences ($P_u$) and only when a re-ranking fails (produces a worse NDCG). Why not also update item properties ($P_i$)? And why not learn from successful re-rankings as a form of positive reinforcement to strengthen the knowledge?

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
AgentMMRec introduces an original and ambitious concept—treating LLMs as cooperating knowledge agents for multimodal recommendation. The work shows solid empirical results and good system design, representing a step toward integrating reasoning and recommendation.

### Strengths
1. The paper is well-written, includes extensive analyses (cold-start, hyperparameter, backbone LLM, multimodal alignment), and provides open-source code.
2. The semantic gap and noise issues between modalities are critical challenges in multimodal operating systems.

### Weaknesses
1. The overall methodological novelty is limited. Although the system is well integrated, its core ideas mainly extend existing LLM-based recommendation frameworks rather than introducing fundamentally new mechanisms.
2. The model’s performance heavily depends on the feedback and reasoning quality of the underlying LLM, raising concerns about robustness and reproducibility across different backbones.
3. The approach incurs substantial computational and resource costs, while yielding only minor performance gains.
4. Experiments are conducted on small-scale datasets, leaving scalability and stability on large, real-world multimodal recommendation scenarios unverified.
5. The comparison with existing methods is incomplete. Several recent studies (particularly in 2025) have highlighted the significant impact of noise in multimodal recommendation, yet these works were not included in the comparison, making the experimental evaluation less comprehensive and representative.

### Questions
See the issues discussed in the “Weaknesses” section above.



Please further clarify the core novelty of the proposed method. Specifically:
1. What is the distinct innovation of this framework compared to existing multimodal recommendation or knowledge-enhanced graph models?
2. Why is the use of large language models (LLMs) indispensable? Could a lighter-weight semantic reasoning or knowledge extraction module achieve similar effects?
3. If LLMs mainly serve as information integrators or alignment modules, their necessity and advantage should be empirically justified, not only conceptually stated.

### Soundness
2

### Presentation
2

### Contribution
2
