# Test-Time Mixture of World Models for Embodied Agents in Dynamic Environments

- Decision: Accept (Poster)
- Scores: 6, 2, 8, 6

## Abstract
Language model (LM)-based embodied agents are increasingly deployed in real-world settings. Yet, their adaptability remains limited in dynamic environments, where constructing accurate and flexible world models is crucial for effective reasoning and decision-making. To address this challenge, we extend the Mixture-of-Experts (MoE) paradigm to embodied agents. While conventional MoE architectures modularize knowledge into expert components with pre-trained routing, they remain rigid once deployed, making them less effective for adapting to unseen domains in dynamic environments. We therefore propose Test-time Mixture of World Models (TMoW), a framework that enhances adaptability to unseen and evolving domains. TMoW updates its routing function over world models at test time, unlike conventional MoE where the function remains fixed, enabling agents to recombine existing models and integrate new ones for continual adaptation. It achieves this through (i) multi-granular prototype-based routing, which adapts mixtures across object- to scene-level similarities, (ii) test-time refinement that aligns unseen domain features with prototypes during inference, and (iii) distilled mixture-based augmentation, which efficiently constructs new models from few-shot data and existing prototypes. We evaluate TMoW on VirtualHome, ALFWorld, and RLBench benchmarks, demonstrating strong performance in both zero-shot adaptation and few-shot expansion scenarios, and showing that it enables embodied agents to operate effectively in dynamic environments.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a framework called "Test-time Mixture of World Models (TMoW)" to improve the adaptability of LLM-based embodied agents in dynamic and unknown environments. The core idea of ​​this framework is to extend the traditional MoE by introducing three key mechanisms: (1) Multi-granular prototype-based routing, which dynamically combines multiple experts based on prototype similarity during testing; (2) Test-time prototype refinement, which enables the model to adjust the prototype corresponding to each expert based on real-time data; and (3) Distilled mixture-based model augmentation strategy, which efficiently constructs new experts in few-shot scenarios. The authors conducted detailed experiments on multiple simulation environments such as VirtualHome, ALFWorld, and RLBench, as well as on real robots. The results show that TMoW significantly outperforms existing state-of-the-art (SOTA) models in both zero-shot adaptation and few-shot expansion scenarios.

### Strengths
- This paper extracts multi-granularity prototypes from datasets divided into multiple domains and trains multiple expert adapters. During inference, the outputs of each adapter are weighted based on feature similarity. This method is logically consistent and effectively decouples the granular details of the scene.

- The paper proposes a method to enable this hybrid expert model to quickly adapt to unseen domains, greatly enhancing the model's generalization ability.

- The paper provides ample experiments, covering multiple simulation platforms and including real-machine experiments, and demonstrates significant advantages compared to multiple strong baselines.

### Weaknesses
- **Naming Convention and Terminology Clarity.** The use of the term "World Model" is highly problematic and requires stringent justification. In both Reinforcement Learning (RL) and Generative Modeling, a "world model" conventionally refers to a model that predicts the environment's state transition, formally defined as $s_{t+1} \sim P(s_t, a_t)$. However, the 'world model' in this manuscript appears to be an LoRA adapter integrated into the LLM, designed to learn domain-specific knowledge. Its function is more akin to a domain-specific expert rather than a state transition predictor. Furthermore, the overall architecture (TMoW) strongly suggests an end-to-end policy learning approach. The authors must provide rigorous proof and a compelling rationale for why these LoRA modules qualify as "world models." Failing this, the terminology should be replaced with a more accurate and representative term.

- **Efficiency and Comparison with Traditional MoE.** The claim of leveraging a Mixture-of-Experts (MoE) structure is misleading regarding computational efficiency. Traditional MoE achieves reduced inference computation through sparse gating, which activates only a subset of experts. In contrast, the routing mechanism in TMoW uses a weighted summation, activating all LoRA branches. This design choice inherently increases the computational load during inference, rather than reducing it. The TMoW approach appears to be fundamentally about enhancing performance by increasing model capacity and incorporating multi-granularity perceptive heads. The authors must clarify the trade-offs between inference efficiency and performance improvement and explicitly detail the distinctions between TMoW and conventional MoE architectures.

- **Insufficient Ablation Study on Expert Utilization.**
The ablation study focusing on granularity (object vs. scene) is noted, but this primarily equates to selectively enabling shallow or deep routers, which is expected to decrease performance. A critical component is missing: there is no experiment ablating the number of activated LoRA adapters (experts). Consequently, the individual effectiveness of different experts across domains has not been adequately validated. A crucial question remains: If a traditional sparse MoE approach is adopted—activating only a few heads with the highest routing scores—would the model's performance be adversely affected, remain stable, or perhaps even improve? This experiment is necessary to fully validate the expert-based domain specialization.

- **Clarity of Problem Definition and Input Representation.**
The introductory and problem definition sections lack essential details regarding the scene observation and its transformation into the input graph. Specifically:
    - What is the explicit definition of scene observation (e.g., is it standard RGB vision information)?
    - How is the raw observation converted into a graph structure?
    - What are the specific components and structure of this graph (nodes, edges, features)?

These fundamental aspects of the input representation must be clearly and explicitly defined in the main text to ensure the completeness and comprehensibility of the paper.

### Questions
1. The update rule for the prototype $\mathbf{p}_j^{(l)}$, specifically the update term $\Delta \mathbf{p}_j^{(l)}$, is calculated as a weighted average of other prototypes based on their similarity. While the paper includes experiments demonstrating that this method adjusts prototype vectors for unseen scenarios, this adjustment approach seems to inevitably push all prototypes towards a single average value.
- Won't this degrade the independence between prototypes of different granularities and potentially lead to expert degeneracy? Why is this specific form of update effective?
- **Recommendation**: I suggest a more in-depth discussion and a theoretical explanation for the validity of this update mechanism, particularly addressing the potential convergence issue.
---
2. The experimental results show that the refined model tends to exhibit high entropy in its routing scores. The authors also note that "success cases have higher entropy in routing scores than failure cases." Intuitively, high entropy suggests a uniform distribution of scores (i.e., every branch is activated equally), whereas a good router typically makes a confident, decisive choice (low entropy), activating only a select few experts.
-  Does this observation stem from the framework fundamentally differing from the standard MoE paradigm?

- **Recommendation**: The current analysis appears somewhat superficial, failing to sufficiently explore the relationship between entropy and other crucial factors such as task type or domain similarity. A deeper investigation is warranted

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes the Test-time Mixture of World Models (TMoW) framework to enable embodied agents to adapt dynamically to unseen domains through prototype-based routing and mixture refinement.

### Strengths
TMoW demonstrates strong performance gains over state-of-the-art baselines in both zero-shot and few-shot adaptation across multiple embodied benchmarks.

### Weaknesses
1. The motivation of this paper is inaccurate. The abstract states that "conventional MoE architectures modularize knowledge into expert components with pre-trained routing; they remain rigid once deployed, making them less effective for adapting to unseen domains in dynamic environments." However, there is already a large number of works addressing Dynamic MoE, Adapted MoE, and MoE for Continual Test-time Adaptation.
2. Home environments inherently have low dynamics, making it difficult to accurately evaluate the performance of this method.

### Questions
1. There is already a large number of works addressing Dynamic MoE, Adapted MoE, and MoE for Continual Test-time Adaptation. What are the differences between this method and existing methods?
2. Home environments inherently have low dynamics, making it difficult to accurately evaluate the performance of this method. How can this method adapt to outdoor environments?
3. The framework introduces non-trivial computational overhead during test-time refinement, which may hinder deployment in real-time or resource-constrained scenarios.
4. How does the multi-granular prototype mechanism ensure interpretability of routing decisions, especially in failure cases or ambiguous domains?
5. Why was Llama-3.2-1B chosen as the base model for TMoW, while some baselines used larger models, and how does this affect fairness in performance comparison?
6. Can the authors justify the choice of hyperparameters (e.g., refinement rate α) and whether they were tuned consistently across all environments?
7. How does the distilled augmentation strategy ensure that new world models do not overwrite or interfere with previously learned knowledge?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors introduce a new framework for adapting language model-based embodied agents to unseen environments, called test-time mixture of world models (TMoW). Through this framework, they address the limitations of vanilla LM-based embodied agents  and the classic mixture of experts paradigm, which has a rigid routing function. When adapting to new domains, existing methods require expensive retraining or knowledge distillation. In contrast, the work here proposes a dynamic test-time adaptation mechanism based on 1) multi-granular prototype-based routing using a hierarchical message-passing network, 2) test-time training of the routing function & 3) a distilled mixture-based model augmentation strategy that constructs new world models based existing ones using few-shot demonstrations.The framework and the effectiveness of the novel components are supported by a comprehensive baseline analysis and ablation studies.

### Strengths
- The paper addresses an important gap in generalization of LM-based agents for embodied tasks and offers a solid alternative to costly retraining for new domains
-  Authors conduct a thorough evaluation of the framework, on a comprehensive set of environments, tasks and baselines, showcasing significant performance improvement over state-of-the-art methods across both metrics (success rate and pending steps) and a very positive performance especially on unseen domains.
-  The authors also transfer the framework to real world scenarios and share results of zero-shot performance which are superior to existing SoTA methods.
- It was good to see that the authors include extensive ablation studies that support the effectiveness of the framework components: multi-granularity, test-time refinement & distilled mixture, where each component shows a clear performance improvement over its respective baseline.

### Weaknesses
I would've liked to see some discussion on the inference time for such an approach compared to the baselines. How much time does the test-time adaptation add to the decision making process and could it be regarded as feasible in a real-world domain from this point of view?

### Questions
*Question*: What is the rationale behind using Llama-3.2-1B for some baselines and 3B for others?

*Suggestion*: In Fig 2, for clarity, it would be good to add a small description of the top right box that links to the mixture of world models.

*Suggestion*:It would be good to reference `Appendix A` in Section 4 to point readers to more details about the the environments, tasks and action space.

*Suggestion*: Reference Table 4 in main body of section 4.2 in the ablation study.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces TMoW, a Test-time Mixture of World Models framework that enables embodied agents to adapt to dynamic environments by dynamically mixing specialized world models. Its core innovation is a multi-granular, prototype-based routing mechanism that leverages hierarchical scene features—from local objects to global contexts—to select experts. The framework supports test-time adaptation through prototype refinement and allows for continual expansion by distilling new world models from few-shot demonstrations. Evaluations on VirtualHome and ALFWorld show significant gains over strong baselines (up to 27.21%), demonstrating robust cross-domain generalization and continual learning capabilities without costly retraining.

### Strengths
S1) The paper is well-written, the proposed pipeline is simple and easy to follow.

S2) The framework enables dynamic test-time adaptation of world model mixtures through prototype refinement, allowing rapid adjustment to unseen environments without retraining.

S3) The distilled model augmentation capability supports continuous expansion of the system's knowledge base through efficient few-shot learning from existing model mixtures.

### Weaknesses
W1) The paper emphasizes that multi-granular prototypes capture features from local objects to global scenes for fine-grained routing. However, when multiple distinct domains exhibit significant feature overlap at a certain granularity level (e.g., sharing similar local objects but having vastly different global scene semantics), how does the router effectively prevent expert confusion and erroneous activation?

W2) The test-time prototype refinement relies on online environmental interaction and its performance is sensitive to the refinement rate α. In non-stationary or partially observable environments, where perceptual data can be noisy or delayed, how does this optimization process based on a local interaction sequence guarantee its convergence stability and reliability?

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
