# CORRECT: COndensed eRror RECognition via knowledge Transfer in multi-agent systems

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
Multi-agent systems (MAS) are increasingly capable of tackling complex real-world tasks, yet their reliance on inter-agent coordination, tool use, and long-horizon reasoning makes error recognition particularly challenging. Minor errors can propagate across agents, escalating into task failures while producing long, intertwined execution trajectories that impose significant costs for both human developers and automated systems to debug and analyze. Our key insight is that, despite surface differences in failure trajectories (e.g., logs), MAS errors often recur with similar structural patterns. This paper presents CORRECT, the first lightweight, training-free framework that leverages an online cache of distilled error schemata to recognize and transfer knowledge of failure structures across new requests. This cache-based reuse allows LLMs to perform targeted error localization at inference time, avoiding the need for expensive retraining while adapting to dynamic MAS deployments in subseconds. To support rigorous study in this domain, we also introduce CORRECT-Error, a large-scale dataset of over 2,000 annotated trajectories collected through a novel error-injection pipeline guided by real-world distributions, and further validated through human evaluation to ensure alignment with natural failure patterns. Experiments across seven diverse MAS applications show that CORRECT improves step-level error localization up to 19.8\% over existing advances while at near-zero overhead, substantially narrowing the gap between automated and human-level error recognition.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the important and challenging problem of "decisive error recognition" in Multi-Agent Systems (MAS). The authors propose CORRECT, a novel, training-free framework that leverages knowledge transfer to identify the root cause of task failures. The core idea is to distill lengthy and noisy failure trajectories into compact, reusable "error schemata." These schemata are then retrieved at inference time to guide an LLM in diagnosing new, similar failures. To facilitate research in this area, the authors also introduce CORRECT-Error, a large-scale benchmark of over 2,000 annotated failure trajectories created via a novel error-injection pipeline. Experiments conducted on the new benchmark and the existing WHO&WHEN dataset show that CORRECT improves error localization accuracy over several baselines.

### Strengths
1. The discussed topic is important and timely. The problem of error attribution in increasingly complex and long-horizon MAS is both critical for reliability and under-explored.

2. The central idea of abstracting away from noisy, full-length trajectories into condensed "error schemata" is insightful and novel. This schema-guided approach is a clever way to perform knowledge transfer without the prohibitive context length requirements or computational costs of naive ICL or fine-tuning.

3. Extensive Experimental Evaluation: The authors conduct a thorough evaluation across two major benchmarks (WHO&WHEN and their new CORRECT-Error), seven diverse sub-tasks (e.g., HotpotQA, GAIA, Math500), and a wide range of open-source and proprietary models. The ablation studies on cache size and the number of retrieved schemata provide valuable insights into the framework's behavior.

### Weaknesses
1. A major weakness of the proposed framework is its critical dependence on a highly capable and proprietary LLM (GPT-5 is mentioned) for the offline schema extraction phase. The quality of the entire system hinges on the ability of this "teacher" model to generate high-quality schemata.

2. The dynamic schema management system (expansion and distillation) is described at a high level, but practical details are sparse. The process of "replaying" candidate schemas against prior trajectories to select the best one  sounds computationally intensive, especially as the number of trajectories and schemas grows.

3. Baselines center on generic LLM-as-a-judge and naive ICL. The paper does not pit CORRECT against specialized verification pipelines which could also reduce long-context noise without relying on learned schemas.

4. The schema idea is also central to the benchmark generation (semantic matching + schema-like error patterns). While the human study is encouraging, the pipeline may implicitly bias toward the kinds of failures that schemas capture well, potentially inflating CORRECT’s advantage. The paper does not quantify how often decisive errors in natural logs deviate from the assumed, reusable “pattern” structure.

### Questions
1. How exactly are decisive-step labels produced in both WHO&WHEN (human-crafted/algorithm-generated) and CORRECT-Error? Do you perform counterfactual replay that replaces step k and re-executes to confirm flip-to-success, or rely on annotator judgment?

2. What happens when top-k retrieval deliberately includes partially mismatched schemas (e.g., same tool family but different failure signature)? Could you report accuracy as a function of retrieval noise or add a hard negative experiment?

3. Have you tried a causal-replay baseline that toggles a small set of suspect steps (e.g., tool-call deltas) to find the earliest flip-to-success point? Even if expensive, a smaller-scale study would clarify whether schemas offer gains over more explicit verification.

4. The paper claims the framework is "training-free", yet it relies heavily on LLMs for both schema generation and final diagnosis. While it avoids fine-tuning, it is not independent of trained models. Could you clarify this claim or rephrase it to more accurately reflect that the method is "fine-tuning-free" but dependent on powerful pre-trained models?

5. Could you elaborate on the computational cost and practical implementation of the "schema distillation" process? How expensive is it to replay schema candidates on prior trajectories, and how does this scale as the cache grows?

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
5

### Summary
The paper introduces CORRECT, a training-free framework that detects the earliest decisive error in long multi-agent trajectories by caching and reusing compact “error schemata.” A clustering step extracts representative schemas offline. At inference they are retrieved by semantic similarity and supplied to an LLM that pinpoints the responsible agent-step. Experiments on WHO&WHEN and CORRECT-Error show step-level accuracy gains up to 20 points over baselines while incurring no additional training cost.

### Strengths
1. The paper addresses a timely issue in the filed of MAS. The proposed idea of distilling reusable failure patterns into short templates is novel and can avoid costly re-training.
2. The empirical evaluation demonstrates strong performance gain.

### Weaknesses
1. My major concerns about the paper lie in the presentation, specifically many techical details are left out:
- The fine-tuning baseline is only named. The paper never explains how the instruction-tuning dataset was built, whether the model was trained to output just the failure-step index or a reasoning trace accompanied, and whether supervised fine-tuning or RLHF was used. Hyper-parameters are missing, even from the appendix.
- In Stage 2 of Section 4.1, the text says GPT-5 “adapts the error pattern while preserving core semantics,”. But the paper provides no prompt, no definition of “error pattern,” and no description of the source.
- Many trajectories exceed 32k tokens, but the method relies on BERT embeddings, whose context window is far shorter. The paper does not mention how embedding is done to handle the length. 

2. Algorithm 1 is under-explained. The pseudo-code drives the whole framework, but the main text never walks through its lines. Key variables (e.g., $\delta$ , $\theta_{hot}$, cache replacement policy) and their interaction with Sec. 3 concepts are left implicit. I could not map each step to Sections 3.1/3.2 without guessing. A major rewrite is needed for clarity.
3. The method invoke proprietary models at test time, the api cost and wall-clock time are not reported.
4. There is limited discussion of failure cases. Where does CORRECT mis-fire? Examples of erroneous schema matches or degenerate clusters would strengthen the paper.
5. The paper does not discuss [1], which is another concurrent work with Who&When. How the method transfers to the error recoginition dataset proposed in [1] requires further discussion.

[1] Why Do Multi-Agent LLM Systems Fail?

### Questions
1. Regarding the cache management, what are the empirical values of $\delta$ , $\theta_{hot}$, and how sensitive are results to them?
2. Have you observed cases where an irrelevant but high-similarity schema misleads the detector? How is this mitigated?

### Soundness
3

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
5

### Summary
The paper introduces CORRECT, a schema-guided framework that identifies key recurrent multi-agent system (MAS) failures without requiring additional training. It also presents CORRECT-ERROR, a large-scale benchmark that models realistic error patterns for evaluating MAS reliability. Together, these tools improve the accuracy, interpretability, and scalability of MAS deployment across diverse tasks and systems.

### Strengths
This paper presents a large-scale, human-verified benchmark specifically designed for multi-agent system (MAS) error attribution. The benchmark captures realistic and diverse error patterns observed in MAS deployments, providing a valuable resource for evaluating and comparing model robustness and interpretability. Its human validation ensures high fidelity and reliability, making it a strong contribution that can serve as a foundation for future research in this area

The authors propose a schema-guided framework for MAS error attribution that leverages a schema and schema-cache mechanism to systematically identify and interpret recurrent error patterns. The framework operates on a retrieval-augmented principle, where previously learned error schemata are reused to analyze new failures efficiently without additional training.

The paper supports its contributions through extensive experiments conducted across a wide range of tasks, models, and deployment scenarios. The empirical results demonstrate that the proposed CORRECT framework significantly outperforms existing baselines in accuracy, generalization, and computational efficiency.

### Weaknesses
While the proposed framework demonstrates strong performance across static evaluation settings, a notable limitation is the absence of experiments in a streaming or online deployment scenario. In real-world MAS applications, systems often operate in a dynamic environment where the schema cache starts empty and must adapt incrementally as new error cases emerge. Evaluating CORRECT under such a cold-start or streaming setting would provide deeper insights into its practical robustness, adaptability, and scalability over time. Without this, it remains unclear how efficiently the framework can learn and reuse schemata in continuously evolving contexts.

Another limitation is that the framework relies on user confirmation or supervision during the cache-building process. While this human-in-the-loop component ensures higher accuracy in schema construction, it also introduces additional overhead and dependency on expert input, which may limit scalability in fully automated or large-scale deployments. Future work could explore strategies for automated or semi-supervised schema validation to reduce reliance on manual confirmation while maintaining interpretability and reliability.

Also note that the idea of using human-verified cache to improve model exists in the literature, eg, EcoAssistant.

### Questions
See weakness

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
4

### Summary
The paper addresses the challenge of error recognition in MAS execution, finding the root cause of failure in a long trajectory. The paper builds upon the insight that while the specifics of the failure trajectory appear unique, they stem from recurring structurally similar patterns.

With this, the authors' propose CORRECT, a training-free framework, that builds an "error schemata" by offline analysis of failure trajectories, identifying the signature pattern and context of errors. During online inference, it retrieves the most relevant schemata from a cache, providing them to LLM-as-judge, enabling targeted and accurate error localization without fine-tuning.

The paper also introduced CORRECT-Error, a large-scale benchmark of annotated failure trajectories.

### Strengths
- The paper proposes a technique that is initialized with offline analysis of failure trajectories, but can also update the schemata/cache online, to identify new failure modes as they occur during deployment
- The paper shows evidence of human alignment with bootstrapped trajectories
- Evaluation across diverse benchmarks and models, demonstrating accuracy improvements in MAS with the intervention

### Weaknesses
- The primary baselines are naive ICL, zero-shot LLM-as-judge and finetuning. Since the proposed approach is about context intervention through a cache, could the authors comment on prompt optimization techniques like MIPRO as a baseline?
- The dataset is built by corrupting successful trajectories through error injection. Could the authors' discuss the effect of sampling bias by only selecting from task instances for which an existing MAS is able to successfully bootstrap a valid solution. What about domains with close to 100% failure rates?

### Questions
- Could the author's provide examples of schema generated by CORRECT in offline and online phases? While one of the emphasis of CORRECT is inference-time schema inference, could the authors' discuss how is the offline mode different from the application of a static human curated taxonomy like MAST?
- Could the authors' address the impact of starting from successful trajectories, what about domains with high failure rates?

### Soundness
3

### Presentation
3

### Contribution
3
