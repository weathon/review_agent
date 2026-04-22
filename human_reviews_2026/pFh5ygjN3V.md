# M$^3$E: Continual Vision-and-Language Navigation via Mixture of Macro and Micro Experts

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 4

## Abstract
Vision-and-Language Navigation (VLN) agents have shown strong capabilities in following natural language instructions. However, they often struggle to generalize across environments due to catastrophic forgetting, which limits their practical use in real-world settings where agents must continually adapt to new domains. We argue that overcoming forgetting across environments hinges on decoupling global scene reasoning from local perceptual alignment, allowing the agent to adapt to new domains while preserving specialized capabilities.
To this end, we propose M$^3$E, the Mixture of Macro and Micro Experts, an environment-aware hierarchical MoE framework for continual VLN. Our method introduces a dual-router architecture that separates navigation into two levels of reasoning. A macro-level, scene-aware router selects strategy experts based on global environmental features (e.g., office vs. residential), while a micro-level, instance-aware router activates perception experts based on local instruction-vision alignment for step-wise decision making. To preserve knowledge across domains, we adopt a dynamic momentum update strategy that identifies expert utility in new environments and selectively updates or freezes their parameters.
We evaluate M$^3$E in a domain-incremental setting on the R2R and REVERIE datasets, where agents learn across unseen scenes without revisiting prior data. Results show that our method consistently outperforms standard fine-tuning and existing continual learning baselines in both adaptability and knowledge retention, offering a parameter-efficient solution for building generalizable embodied agents.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a new framework for continual learning in Vision-and-Language Navigation (VLN) tasks. The core problem addressed is catastrophic forgetting, where an agent's performance on previously learned environments degrades after being trained on new ones. The proposed solution is a replay-free method based on a Mixture-of-Experts (MoE) architecture integrated into an LLM-based navigation agent. 

The key innovation is a dual-router system that separates reasoning into two streams: a Macro Router for global, scene-level strategic planning using a graph neural network over a cognitive map, and a Micro Router for local, token-level semantic grounding. To consolidate knowledge without replaying old data, the authors propose a Dynamic MoE Momentum Update mechanism that selectively updates or freezes expert parameters based on their utility in the current task. 

The method is evaluated on a domain-incremental benchmark constructed from the R2R and REVERIE datasets, where it is shown to outperform standard fine-tuning and several continual learning baselines in both navigation success and knowledge retention.

### Strengths
To begin with, the paper tackles  a critical and well-recognized challenge that must be overcome to build autonomous systems capable of lifelong learning in dynamic environments. Decoupling high-level spatial reasoning from low-level instruction grounding is a very sensible approach to modularizing knowledge and seems well-suited to the hierarchical nature of navigation. And the  method's ability to mitigate forgetting without storing and replaying past trajectory data is a significant strength.

### Weaknesses
I have a few concerns that temper my enthusiasm.

If I understand well, the primary motivation of introducing continual learning to  VLN is explicitly tied to real-world agents that must continually adapt to new domains. However, the evaluation is performed exclusively in a simulated environment on an artificially constructed stream of tasks (partitioning the val-unseen set). I think this is a significant limitation, as the challenges of a simulated, sequential dataset may not fully represent the complexities of real-world continual learning. The paper would be much stronger (and from my point, it ought to contain it) if it discussed the potential challenges of Sim2Real transfer.

Another concern is that I wonder about the computational complexity of the Macro Router. It appears to construct a graph from the agent's cognitive map and perform GNN propagation at each decision step. As the agent explores a large environment, this map could grow significantly. More discussion on the computational overhead and scalability of this component seems necessary for a complete picture of the method's practicality.

### Questions
(1) Could the authors comment on the challenges they foresee in deploying your method on a physical robot?

(2) Could the authors provide some analysis of the inference-time overhead introduced by the Macro Router, especially as the size of the cognitive map increases throughout an episode?

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
3

### Summary
This paper presents M3E, a replay-free continual learning framework for Vision-and-Language Navigation (VLN).
M3E is designed to disentangle global scene reasoning from local instruction–vision grounding via a macro–micro Mixture-of-Experts (MoE) structure.
The framework further introduces a dual routing mechanism for expert activation and a differentiated momentum update to consolidate task-critical experts without relying on replay data.
Experiments conducted on R2R and REVERIE demonstrate that M3E achieves improved navigation performance and alleviates catastrophic forgetting compared to existing continual learning approaches.

### Strengths
* The paper provides a clear motivation for leveraging a macro–micro MoE strategy to enhance transferability and stability in lifelong VLN tasks.

* The proposed topology-aware propagation using GNNs and the dual-routing fusion methods are conceptually simple yet appears to be an effective way to capture structural relationships in navigation environments.

* The replay-free continual learning design aligns well with realistic online settings, making the proposed approach practically meaningful.

### Weaknesses
* The paper lacks a review and comparison with recent online test-time adaptation approaches for VLN, such as FeedTTA (ICML '25) and FSTTA (ICML '24). In particular, FeedTTA also investigates catastrophic forgetting through a gradient regularization mechanism, which is conceptually similar to the replay-free and momentum-based update proposed in this paper. Please consider positioning M3E more explicitly in relation to these methods to highlight its distinct contributions by providing quantitative comparisons if possible.

* The topology-aware, task-focused dual routing mechanism is interesting, but the individual contributions of each component remain unclear. For example, Tab.3 does not isolate the effects of the macro-level router or the momentum update in ablation studies, making it difficult to assess their independent impact.

* While the dual-router fusion mechanism is intuitively reasonable, it remains unclear how effectively the model disentangles and learns global versus local information in practice. The authors should provide analytical or visualization-based evidence (e.g., expert activation maps or attention distributions per task/environment changes) to support the claim that such separation emerges through the proposed design.

At this stage, I lean toward a borderline rejection. However, I'm willing to increase the score based upon the authors responses and further clarifications.

### Questions
Please refer to the weakness section where I also listed questions and suggestions.

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
In this paper, the authors propose M3E (Mixture of Macro and Micro Experts) for continual end-to-end vision-language navigation. For replay-free continual learning, the authors use dynamic momentum update for the MoE layers by selecting the top-performing experts, which helps them prevent catastrophic forgetting. The navigation policy consists of a visual scene encoder and a transformer decoder as a policy, with the FFNs replaced with M3E. The architecture is dual router: There is a global router captures global structural regularities of the environment and uses a GNN to build spatial relationships between different nodes in the scene graph and eventually pass it to experts; the local router performs token-wise routing over N experts.

They experiment with R2R and REVERIE benchmark in MP3D datasets. They evaluate in both seen and unseen environments. In addition to traditional VLN metrics (SR, SPL, NE, OSR), they also compute continual learning metrics (backward transfer, forward transfer). The baselines used are fine-tuned policies, L2, EWC, ER, etc, consisting of regularization-based and rehearsal-based methods. M3E performs the best, achieving the highest average SPL, reducing average navigation error significantly, and having lesser back-ward transfer error showing that it prevents catastrophic forgetting. They also perform additional experiments to train on val-unseen and show that the performance on val-seen does not reduce, while maintaining competitive performance on test-unseen, against HAMT and NaviLLM. They also perform ablations on micro/macro routers and momentum based updates and show that all of them are important for best performance.

### Strengths
1. The authors tackle a relatively under-explored problem of continual learning and catastrophic forgetting in vision and language navigation, and propose a methodical approach to solve it.
2. The idea of decoupling scene level and token level information is pretty important and useful for handling the continual learning task.
3. The approach actually helps in preventing catastrophic forgetting across different environments as shown by low BWT metric.

### Weaknesses
1. The $M^{3}E$ framework "relies on accurate cognitive mapping". This implies that the performance of the macro-router, and thus the entire system, could degrade if the underlying cognitive map representation is poor, especially since it relies on a topological graph.
2. The method uses fixed expert configurations. This may limit flexibility under extreme domain shifts, as the agent cannot dynamically add new experts if a new environment is radically different from all previous ones.
3. The authors do not discuss how the approach is applicable to settings beyond VLN - VLN is a hard problem, but there are existing approaches which perform very well on these tasks.
4. While BWT is near-zero on R2R, the method is not a perfect for preventing forgetting. In the "full val-unseen training" experiment (Table 3), $M^{3}E$ still shows some performance degradation (e.g., -4.31 SPL, -3.87 SR) on the REVERIE val-seen split, though it is much more resilient than the NaviLLM baseline.

### Questions
1. How easy is it to apply this approach to other tasks such as mobile-manipulation or table-top manipulation?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a novel continual learning benchmark on vision - language navigation task and proposes a replay - free method - Mixture of Macro and Micro Experts (M3E) approach that effectively deals with the catastrophic forgetting problem in the continual learning setting. Two core components are responsible for the superior performance of the M3E approach: A macro - router conditioned on the scene context and a micro - router conditioned on egocentric observation. By designing a mixture - of - expert frameworks with both routers, the M3E is shown to work better than most of the continual learning methods.

### Strengths
1. The continual learning settings designed for the VLN task are not only theoretically interesting but also serve as a data-efficient strategy to enhance the generalization ability of VLN approaches.
2. The proposed method (M3E) is well-justified in addressing the catastrophic forgetting problem, particularly in the aforementioned continual learning settings for VLN.

### Weaknesses
1. The proposed method (M3E) relies on meta-information (i.e., scene topology) to construct the macro-router and adheres to conventional discrete VLN settings—both factors widen the gap for its practical real-robot deployment. Discrete settings, for instance, often fail to align with the continuous perceptual and motion requirements of physical robots, while the dependence on pre-defined scene topology limits adaptability to unstructured environments.
2. No evidence is provided in this paper to demonstrate that the proposed method can scale effectively under continual learning settings. Specifically, there is a lack of analysis on whether the model’s performance on episodes in later (or more complex) scenes remains comparable to, or outperforms, those in the first few scenes—an essential indicator of scalability in continual learning.
3. As shown in Table 4, the M3E fails to address the catastrophic forgetting problem under the REVERIE benchmark. This limitation makes it challenging to verify the model’s generalization ability across different VLN tasks, especially within the continual learning framework emphasized earlier.

### Questions
(1) Although the experimental results shown in Table 4 demonstrate that the M3E method can effectively solve (or greatly alleviate) the catastrophic forgetting problem, there is a lack of results to confirm whether the task performance of M3E continues to improve as it experiences more episodes under lifelong learning settings—consistent with the paper’s continual learning framework. Therefore, I wonder if there is a specific metric that quantifies the performance trend of M3E across different numbers of experienced episodes?
(2) In Table 4, the performance metrics of M3E on the R2R and REVERIE benchmarks are inconsistent (e.g., M3E alleviates forgetting on R2R but still suffers from it on REVERIE). What are the key reasons for this inconsistency in performance metrics between the two benchmarks? Furthermore, what primary factors cause M3E to still struggle with catastrophic forgetting in the REVERIE task?
(3) Currently, this work is built on the NaviLLM framework. Given this, can the M3E approach be generalized to other navigation foundation models? If not, what technical constraints limit its generalization across different foundation models?

### Soundness
3

### Presentation
3

### Contribution
2
