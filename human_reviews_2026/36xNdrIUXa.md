# ViRL-TSC: Enhancing Reinforcement Learning with Vision-Language Models for Context-Aware Traffic Signal Control

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
In real-world urban environments, traffic signal control (TSC) must maintain stability and efficiency under highly uncertain and dynamically changing traffic conditions. Although reinforcement learning (RL) has shown strong adaptability in dynamic environments, existing methods still depend on predefined state spaces and cannot directly perceive the environment. Consequently, they fail to exploit visual semantic information, which restricts their ability to generalize to unseen or evolving traffic conditions during training. To overcome these limitations, we introduce ViRL-TSC, a unified framework that integrates RL with Vision–Language Models (VLMs). A Foundation Model-Driven Visual Reasoning Engine (FM-VRE) fuses visual inputs with structured information matrices to generate high-level multimodal semantic representations of intersections. These representations are then processed by a Foundation Model-Driven  Decision Evaluation Engine (FM-DEE), which integrates them with the RL agent’s proposed actions. The RL policy ensures efficient control in scenarios encountered during training, while the VLM leverages logical reasoning and contextual analysis to handle rare events beyond the scope of RL training. By combining RL’s task-specific policy optimization with the VLM’s rich semantic understanding, ViRL-TSC maintains high efficiency during routine operations and selectively intervenes to enhance robustness under long-tail traffic conditions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposed a unified framework that using RL with VLM called ViRL-TSC to robust traffic signal control. By combining RL's task-specific policy optimization with the VLM's rich semantic understanding, it maintains high efficiency during routine operations and selectively intervenes to enhance robustness under long-tail traffic conditions.

### Strengths
1. Proposed visual LLM integrates RL to solve traffic signal control problem. Which reduces the cost of perception deployment.

2. Proposed Foundation Model-Driven Decision Evaluation Engine (FM-DEE) that integrates visual input to percept special vehicle to increase robustness.

### Weaknesses
1. The experiments are only test in two single-intersection, the author should do the scalability experiment in large-scale intersections.

2. The idea is just using LLM and RL as a simple application in traffic signal control, it makes low novelty. The idea is similar to [1], but just change the perception module to visual. The idea is similar to [2], which is also using visual as perception, and i think the author just do some prompt engineer. The author should give some discuss about below two works and give some experiments compared to the two works.

[1] LLM-Assisted Light: Leveraging Large Language Model Capabilities for Human-Mimetic Traffic Signal Control in Complex Urban Environments. 2024 Arxiv

[2] VLMLight: Traffic Signal Control via Vision-Language Meta-Control and Dual-Branch Reasoning. 2025 NeuIPS

3. The author uses visual as input but use a structured traffic feature matrix J, i think the structured feature may percept from radar and the other non-visual sensors which is contradicts the full-visual solution mentioned by the author. The author should give more explanations.

4. There is no theoretical analysis.

5. The simulator is self-developed, not publicly available, and its results cannot be produced. The RL network, hyperparameters also not reported.

6. In real-world, the camera images we capture contain a lot of noise. For example, due to different weather conditions such as rain, snow and flog. Additionally, the captured images often have a lot of building information that is irrelevant to the lanes. Which is absolutely different from your self-developed simulator. Such information can cause the hallucination problem in your MLLM. The author should give more discussions or give a hallucination example.

### Questions
See weakness.

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
This paper proposes ViRL-TSC, a framework that integrates VLMs with RL to enhance context-aware traffic signal control in complex urban environments. The framework addresses the limitations of traditional and RL-based methods, which rely on predefined rules or static states and struggle with dynamic, real-world scenarios like emergency vehicle prioritization. ViRL-TSC uses a pre-trained RL agent for efficient policy optimization and a VLM-based reasoning engine to interpret multimodal data from cameras and traffic features, enabling high-level semantic understanding and robust decision-making. Experimental results demonstrate that the framework reduces emergency vehicle waiting times with minimal impact on regular traffic, thereby narrowing the performance gap between simulation-trained agents and real-world deployments.

### Strengths
1. **Integrated Multimodal Framework:** The paper integrate of VLMs with RL. This combination leverages RL for efficient policy optimization and VLMs for high-level, context-aware semantic understanding of complex traffic scenes from visual data.
2. **Enhanced Robustness in Rare Scenarios:** The framework improves robustness, particularly for unforeseen or rare events like emergency vehicle prioritization.
3. **Structured Reasoning for Decisions:** The incorporation of structured reasoning components, which guides the VLM to perform explicit, step-by-step reasoning. This ablation-studied approach substantially enhances decision reliability and performance over using a VLM without such guidance.

### Weaknesses
- **Limited Novelty**: The paper’s contribution appears incremental rather than groundbreaking. Prior work, such as [1], has already integrated VLMs with deep DRL model TSC, including specialized mechanisms for emergency vehicle response. The present study replicates most of this paradigm and only introduces a justified alignment mechanism between VLMs and DRL policies. Unfortunately, this alignment is also problematic (see detailed questions).
- **Limited baselines**: The experimental design omits several critical and directly comparable baselines. Recent studies have demonstrated the use of large language models (LLMs) [2–4] and VLMs [1] for adaptive and interpretable TSC, achieving strong generalization across diverse urban environments. However, these representative methods are not included for comparison. The omission prevents readers from evaluating whether the proposed method further improves interpretability and robustness.
- **Insufficient experiments**: The experimental evaluation lacks depth and scalability analysis. Most TSC studies assess their frameworks across multiple intersections (typically 12 or more) to capture complex traffic interactions and coordination challenges. In contrast, this paper’s experiments are confined to a single intersection, limiting the external validity of the findings. The absence of large-scale or heterogeneous network tests raises doubts about the method’s practicality in real-world deployments. Furthermore, key metrics such as transferability across layouts, robustness under non-stationary demand, and computational overhead are missing, further weakening the empirical evidence.

### Questions
- Existing studies [5, 6] have already established RL as the standard baseline for TSC agents. Therefore, Section 3.2, which primarily reintroduces the RL formulation, appears to be part of the background rather than the core methodological contribution.
- The paper describes a scene-level representation $V_t$ and its transformation into a visual semantic summary $\Omega_v$. Could the authors clarify this process? Specifically, is the transformation handled internally by the VLM, or is an additional language model or projection module implemented to interpret the visual embedding?
- The deep RL model outputs discrete actions, while the VLM operates through open-ended reasoning. How do the authors ensure that the alignment between the VLM’s reasoning process and the RL’s policy output is both semantically coherent and behaviorally consistent?
- The paper argues that RL models lack generalization and fail under unseen conditions. If so, what is the rationale for aligning the VLM’s decision-making with the RL’s policy output? Would this alignment not inherit RL’s generalization limitations rather than overcome them?

[1] Wang, Maonan, et al. "VLMLight: Traffic Signal Control via Vision-Language Meta-Control and Dual-Branch Reasoning." *arXiv preprint arXiv:2505.19486* (2025).

[2] Lai, Siqi, et al. "Llmlight: Large language models as traffic signal control agents." *Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1*. 2025.
[3] Yuan, Zirui, Siqi Lai, and Hao Liu. "Collmlight: Cooperative large language model agents for network-wide traffic signal control." *arXiv preprint arXiv:2503.11739* (2025).

[4] Zou, Xingchen, et al. "Traffic-r1: Reinforced llms bring human-like reasoning to traffic signal control systems." *arXiv preprint arXiv:2508.02344* (2025).

[5] Wei, Hua, et al. "Colight: Learning network-level cooperation for traffic signal control." *Proceedings of the 28th ACM international conference on information and knowledge management*. 2019.

[6] Wei, Hua, et al. "Presslight: Learning max pressure control to coordinate traffic signals in arterial network." *Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining*. 2019.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates the integration of vision–language models (VLMs) with reinforcement learning (RL) for traffic signal control, aiming to improve decision-making under dynamic and uncertain urban conditions. The authors combine visual scene understanding with traffic state information to enable more context-aware and robust signal control in real traffic scenarios. The main contributions include: (1) a unified framework that combines reinforcement learning with vision–language models to improve context-aware traffic signal control; (2) a clear reasoning pipeline that integrates visual scene information with structured traffic features to refine decision-making; and (3) experimental results showing large improvements in emergency vehicle priority with only a small impact on normal traffic. The work demonstrates the potential of using multimodal reasoning to enhance the robustness of RL-based traffic control systems.

### Strengths
The main contributions include: (1) a unified framework that combines reinforcement learning with vision–language models to improve context-aware traffic signal control; (2) a clear reasoning pipeline that integrates visual scene information with structured traffic features to refine decision-making; and (3) experimental results showing large improvements in emergency vehicle priority with only a small impact on normal traffic.

### Weaknesses
However, there are several areas that require attention and improvement:
(1) The Introduction section does not clearly highlight the key gap in current TSC methods. The authors should explicitly explain why existing RL and LLM-based approaches cannot handle visual semantic information, and provide concrete examples to strengthen the motivation.

(2) The Compared Methods section does not clearly explain the Vanilla-VLM baseline. The authors should provide a clearer description of how this baseline is built, including the model setup and the prompt used, since the current statement that it directly uses VLM outputs lacks sufficient detail to understand or reproduce.

(3) In Fig. 2, the framework diagram does not clearly illustrate how the VLM-guided refinement contributes to policy improvement. It is unclear whether the refined actions are fed back into the RL learning process or only applied at inference time. The authors should clarify the closed-loop mechanism and show how the VLM intervention leads to measurable policy improvement, ideally through an explicit feedback or training signal in the diagram.

(4) The Experiments section lacks recent and strong VLM or LLM-based TSC baselines. Adding representative methods from 2023–2025 would make the comparison more fair and clearly demonstrate the advantages of the proposed approach.

(5) The Methodology section does not sufficiently justify the need to separate FM-VRE and FM-DEE into two modules. The authors should explain why a single-stage or end-to-end design is not adequate, and an ablation study is recommended to validate the architectural choice.

### Questions
(1) The Introduction section does not clearly highlight the key gap in current TSC methods. The authors should explicitly explain why existing RL and LLM-based approaches cannot handle visual semantic information, and provide concrete examples to strengthen the motivation.

(2) The Compared Methods section does not clearly explain the Vanilla-VLM baseline. The authors should provide a clearer description of how this baseline is built, including the model setup and the prompt used, since the current statement that it directly uses VLM outputs lacks sufficient detail to understand or reproduce.

(3) In Fig. 2, the framework diagram does not clearly illustrate how the VLM-guided refinement contributes to policy improvement. It is unclear whether the refined actions are fed back into the RL learning process or only applied at inference time. The authors should clarify the closed-loop mechanism and show how the VLM intervention leads to measurable policy improvement, ideally through an explicit feedback or training signal in the diagram.

(4) The Experiments section lacks recent and strong VLM or LLM-based TSC baselines. Adding representative methods from 2023–2025 would make the comparison more fair and clearly demonstrate the advantages of the proposed approach.

(5) The Methodology section does not sufficiently justify the need to separate FM-VRE and FM-DEE into two modules. The authors should explain why a single-stage or end-to-end design is not adequate, and an ablation study is recommended to validate the architectural choice.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes to integrate RL with VLMs to improve the robustness and generalization for traffic signal control (TSC) tasks. Traditional RL-based TSC methods heavily rely on fixed state representations and struggle with unseen traffic scenarios, and the authors leverage the efficiency of a pretrained RL agent for routine traffic control, while the VLM selectively intervenes in long-tail, unseen situations. They conduct extensive experiments on two real-world intersections and demonstrate that their method can reduce the emergency vehicle waiting times.

### Strengths
1. Integrating VLMs with a pretrained RL agent for TSC is meaningful, as it combines the efficiency of RL-based methods with the generalization ability of foundation models.

2. The authors present extensive experimental results on two real-world intersections, comparing their approach with multiple baselines including rule-based, RL-based, and VLM-based methods.

3. The paper is clearly written, well organized, and easy to read.

### Weaknesses
1. While I do appreciate the idea of using foundation models to improve generalization in TSC, I am unclear whether the authors only consider emergency vehicle prioritization as the long-tail or unseen scenario. In real-world settings, short-term high traffic flow (e.g., due to local events) or traffic patterns that deviate from the regular 24-hour distribution may better represent long-tail situations. Could the authors evaluate their method under such scenarios?

2. The experiments are conducted on only two intersections, which may limit the significance and generality of the results. I understand that collecting real-world visual data from many intersections is difficult. However, since the authors have extended the SUMO simulator to a vision-enabled TSC simulator, they could evaluate their method on larger networks. For example, [1] provides datasets from Jinan (12 intersections) and Hangzhou (16 intersections). Moreover, some RL-based baselines such as CoLight [2] rely on communication and coordination among neighboring intersections, which could be relevant for comparison.

3. The decision cost is relatively high, which may hinder real-world deployment. According to line 344, a single decision step requires about three seconds for inference, and this cost could increase further if multiple intersections are controlled in parallel.


[1] A Survey on Traffic Signal Control Methods. https://traffic-signal-control.github.io/

[2] CoLight: Learning Network-level Cooperation for Traffic Signal Control.

### Questions
1. Which version of the Qwen model is used as the VLM backbone?

2. I believe Average Travel Time (ATT) and Average Waiting Time (AWT) are the two most important metrics for TSC. In Table 1, the **Vanilla-RL** (w/o VLM) seems to perform better than **ViRL-TSC**(w/ VLM) in ATT and AWT. Did I misunderstand your experimental setup? I hope the authors can clarify this.

3. Since the paper focuses on improving the generalizability of RL-based TSC, it might also be valuable to consider transferability, as smaller cities often lack sufficient data to train a good RL model. I do not expect the authors to include this in the current version, but I would like to discuss their problem formulation and experimental settings during the rebuttal phase to better understand the design choices and adjust my rating accordingly.

### Soundness
3

### Presentation
3

### Contribution
2
