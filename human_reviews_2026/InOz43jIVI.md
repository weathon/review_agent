# CoSteer: Collaborative Decoding-Time Personalization via Local Delta Steering

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Personalized text generation has become crucial for adapting language models to diverse and evolving users' personal context across cultural, temporal, and contextual dimensions. While existing methods often rely on centralized fine-tuning or static preference alignment, they struggle to achieve real-time adaptation under resource constraints inherent to personal devices. This limitation creates a dilemma: large cloud-based models lack access to localized user-specific information, while small on-device models cannot match the generation quality of their cloud counterparts. To address this dichotomy, we present    **CoSteer** 
, a novel collaborative framework that enables decoding-time personalization through localized delta steering. Our key insight lies in leveraging the logits difference between personal context-aware and -agnostic outputs from local small models as steering signals for cloud-based LLMs. Specifically, we formulate token-level optimization as an online learning problem, where local delta vectors dynamically adjust the remote LLM's logits within the on-device environment. This approach preserves privacy by transmitting only the final steered tokens rather than raw data or intermediate vectors, while maintaining cloud-based LLMs' general capabilities without fine-tuning. Through comprehensive experiments on various personalized generation tasks, we demonstrate that CoSteer effectively assists LLMs in generating personalized content by leveraging locally stored user profiles and histories, ensuring privacy preservation through on-device data processing while maintaining acceptable computational overhead. Our anonymized code and data are available at https://anonymous.4open.science/r/Costeer-4977

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces CoSteer, a collaborative framework for personalized text generation, operating within a cloud-edge paradigm. It aims to leverage the power of large language models (LLMs) in the cloud while preserving user privacy by keeping sensitive context (profiles, preferences, history) strictly on the local device. The core challenge is enabling personalization without transmitting private data to the cloud, while avoiding the quality limitations of using only a small local model (SLM).

CoSteer proposes a decoding-time steering mechanism that is tuning-free. A local SLM computes the logit difference (delta) between its outputs generated with and without access to the private user context. This delta vector, representing the personalization direction, is calculated and applied locally to steer the logits received from the cloud LLM at each decoding step. The steering process uses an online learning formulation (FTRL) with an efficient closed-form update, ensuring privacy as only the final steered token is returned to the cloud.

### Strengths
Tackles the critical challenge of balancing LLM capabilities, user privacy, and personalization on resource-constrained devices, which is crucial for real-world applications like personal assistants. While cloud-edge collaboration for personalization isn't entirely new, CoSteer introduces a specific, elegant mechanism using the SLM's logits delta combined with an online learning (FTRL) formulation for steering. This particular approach to extracting and applying the personalization signal locally is a key technical contribution.

### Weaknesses
The core concept of cloud-edge collaboration or "semi on-device" processing to balance privacy, personalization, and compute power, is an active area of research. While the specific delta steering mechanism is novel, the overall collaborative architecture might be seen as an instantiation within a known paradigm rather than a completely groundbreaking framework. The per-token communication round trip (cloud-to-device for logits, device-to-cloud for token) remains a significant practical bottleneck, especially under high network latency. The paper acknowledges this and proposes AdaCoSteer, but a direct latency comparison with vanilla cloud LLM generation is missing. Requiring the local SLM to run twice per token (with/without context) plus the FTRL optimization step could impose a non-trivial computational and energy burden on the local device, potentially exceeding the cost of running the SLM just once.

### Questions
Can you provide a more detailed breakdown of the wall-clock time per token? Specifically, measurements for: (a) Cloud LLM inference, (b) Cloud-to-local logits transmission, (c) Local SLM inference (x2), (d) Local FTRL optimization (using Eq 7), (e) Local-to-cloud token transmission? How does the total per-token latency compare quantitatively to vanilla LLM cloud inference under typical network conditions?

Could you elaborate on the intuition behind why the iterative FTRL optimization (T=20) provides better results than the single-step LightCoSteer variant (T=1)? Does iteratively refining the policy within the generation of a single token allow for better integration of the SLM delta signal?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces CoSteer, a collaborative framework for personalized text generation that aims to protect user privacy. It addresses a specific and practical scenario where a powerful cloud-based LLM needs to be personalized but cannot be given direct access to a user's sensitive local data (e.g., profiles, interaction history). The core idea is to leverage a smaller, on-device SLM which can access this private context. This local SLM computes a "delta steering" signal, which is the logit difference between its context-aware and context-agnostic outputs. This privacy-safe delta signal is then used to guide the decoding process of the remote cloud LLM, aligning its generation with the user's personal context without that context ever leaving the device. The entire process is formulated as an online optimization problem solved at decoding time.

### Strengths
- The paper identifies and tackles an interesting, intuitive, and increasingly relevant problem: how to balance the need for high-quality personalization from powerful cloud models with the critical and non-negotiable requirement of user privacy.

- The proposed CoSteer framework is well-motivated and its core mechanism is straightforward to understand. The idea of using a local model to compute a "delta" to steer a remote model is an elegant solution to this problem.

- The experimental evaluation is comprehensive, testing the framework's effectiveness across multiple personalized generation tasks and different model pairs.

- I appreciate the detailed discussion section (Section 5), which proactively explores practical challenges like robustness to noisy context and collaboration between different model architectures.

### Weaknesses
- My primary concern is the paper's limited technical novelty. The contribution is almost entirely in the problem setup and framework design.

- The core optimization algorithm, which is central to the method's implementation, appears to be adopted directly from a previous work (Zhang et al., 2025b), specifically the use of FTRL for decoding-time alignment.

- This lack of technical innovation places a very heavy burden on the novelty of the scenario itself. If this collaborative, delta-steering setup is not demonstrably practical or is perceived as niche, the overall contribution of the paper feels minor.

### Questions
- My main question is about the practical grounding of this work. Could the authors comment on or provide any existing examples—whether from public applications or industrial settings they are aware of—that currently use this specific paradigm? That is, a setting where users possess sensitive local information they cannot transmit, and where logit-based signals are used as the medium for collaborative personalization? Evidence of real-world application would significantly strengthen the paper's motivation.

- Thinking about the discussions on noise robustness (5.1) and cross-architecture collaboration (5.2), I wonder if this "Delta Steering" signal could be used for aggregation. For example, could you have multiple different local models compute their respective delta signals, which are then aggregated (e.g., through voting or averaging) to create a single, more robust steering vector that could be less susceptible to the noise or bias of any single local model?

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
This paper develops CoSteer, a framework that allows for personalization at decoding-time by using the difference in logits between a personal and general-purpose small language model to steer cloud-based LLMs. They show through experiments that this approach outperforms the small general-purpose and personal language models and the large general-purpose LLM.

### Strengths
- The approach seems like a simple way to leverage the strengths of both personalized SLMs and general LLMs.
- The problem being solved is interesting and relevant for generating good-quality personalized outputs while keeping private information local.
- The paper provides thorough experimental results in a variety of settings, including different SLM-LLM combinations and hyperparameter ablations.

### Weaknesses
While the experimental section contains many experiments, the paper should further distinguish the approach from other methods.
1. The paper cites Table 3 to explain why their approach is unique, but I am not sure why exactly the constraints from the table are required. In particular, the main reason why Linear Alignment/Context Steering differ from CoSteer is that these models are not weak-to-strong collaborative. However, LA/CS seem to have fairly comparable performance to CoSteer without weak-to-strong collaboration, so it is not clear why this is needed. 
2. The baselines are only compared to CoSteer for a single model pair (Qwen7B-1.5B). Section 4.5 would benefit from additional experiments for other pairs of models.
3. The performance of CoSteer in comparison to the SLM and LLM (Table 1 and 6) seems more mixed for Llama 8B-1B, Qwen 8B-0.6B, and Qwen 8B-32B. Could the authors elaborate why this is the case? I think it would be useful to discuss this more in the main paper.

### Questions
- The metrics are listed without standard deviations or error bars. Could the authors add these to the paper?
- There are a variety of typos throughout the paper that should be fixed, particularly misspelled words and missing spaces. Also, Section 4.3 lists the last 2 Qwen models in opposite order from the other models. Is this intentional?

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
This work introduces CoSteer, a framework that enables real-time, privacy-preserving personalization of LLMs by collaborating with small local models. It steers LLM logits using locally computed delta signals from personal-context-aware small models, achieving strong personalized generation without fine-tuning or direct data leakage, validated across diverse tasks and model scales.

### Strengths
+ The problem of enabling cloud LLM to be aware of local user data without direct data access is well-motivated.  
+ CoSteer introduces a unique collaborative decoding-time personalization framework that enables real-time adaptation using local delta steering without requiring fine-tuning or directly exposing sensitive user data.
+ Extensive experiments across multiple datasets and model scales demonstrate that CoSteer improves personalized text generation while maintaining privacy and efficiency comparable to non-personalized cloud LLM.

### Weaknesses
- The core idea builds on existing context steering methods (He et al., 2025), mainly extending them to cloud-edge collaboration, offering limited theoretical or algorithmic innovation.
- Experimental results in Table 2 show only slight improvements over strong personalized baselines.
- Most evaluated datasets are mobile-centric, while CoSteer’s target use case emphasizes cloud LLM serving with personalized requirement.
- The paper lacks a formal theoretical assessment of whether sharing SLM logit differences could indirectly expose sensitive personal information to the cloud.

### Questions
Please see Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3
