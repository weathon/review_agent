# Learning to Respond:  A Large-Scale Benchmark and Progressive Learning Framework for Trigger-Centric Online Video Understanding

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
The rapid growth of online video platforms resulted in vast amounts of streaming and surveillance content, creating an urgent demand for real-time video understanding. 
Unlike offline tasks, online video understanding emphasizes proactive responsiveness, where models must detect when sufficient evidence has appeared in the stream to answer a given question (\emph{trigger}) and respond immediately.
However, current studies provide insufficient exploration of such capabilities.
To bridge this gap, we introduce TV-Online (Trigger Video-Online), a large-scale dataset with $50K$ videos, $200K$ questions, and $500K$ time-stamped answers. 
TV-Online covers progressively complex trigger-based tasks, ranging from basic temporal grounding to asynchronous scheduling and multi-trigger reasoning. 
These tasks motivate an agent-like modeling paradigm in which the system continuously processes streaming inputs and decides at each step whether to respond or remain silent. 
We instantiate this paradigm with a streaming-oriented model that employs protocol-level tagging and structured state management to regulate frame-by-frame decisions, ensuring precise response timing and consistent handling of asynchronous, multi-turn triggers.
To endow the model with such capabilities, we adopt a progressive training strategy that leverages difficulty annotations in TV-Online and reinforcement objectives to shape responsiveness, coverage, and coherence across evolving interactions.
Finally, we introduce a unified evaluation metric that integrates semantic, temporal, and coverage dimensions to holistically assess online video understanding.
Extensive experiments demonstrate that TV-Online, together with the proposed model, training strategy, and metric, provides a comprehensive benchmark for advancing trigger-oriented online video understanding toward practical real-time video intelligence.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces TV-Online, a trigger-centric benchmark for online video understanding with ~50K videos, ~200K questions, and ~500K time-stamped answers organized into hierarchical tasks—TGQA, TST, and SQU. It proposes TOM, an agent-style streaming model with protocol-level tags, fast inference, and queue-based state management, trained via a progressive pipeline (pretrain → fine-tune → RL). A unified metric holistically scores semantic correctness, coverage completeness, and temporal latency. Experiments and ablations on TV-Online report strong overall F1 and sizable gains from ID tagging and state management.

### Strengths
**1. Trigger-centric dataset & metric.** A large, hierarchical benchmark with explicit temporal triggers and a three-axis metric (semantic, coverage, temporal) targets online responsiveness beyond offline QA. 

**2. Agent-style TOM with protocol & memory.** Special tags (<|response|>, <|silent|>, IDs), fast inference, and Trigger State Management align frame-by-frame decisions and multi-trigger coherence. 

**3. Progressive training with RL.** A staged pipeline plus rewards (coverage/precision, tag reward, hallucination penalty) improves Overall/SQU F1; ID tags alone add +11.8 SQU and +6.8 Overall.

### Weaknesses
**1. Novelty of method.** While effective, the protocol-tagging, fast-skip, and queue-based state management are structural, reading as evolutionary within streaming-LLM design rather than algorithmical improvement. Clarifying conceptual novelty beyond integration would help. 

**2. Comprehensiveness of the benchmark.** TV-Online consolidates public corpora (Shot2Story, Ego4D, Vript, DiDeMo, Charades) and synthesizes QA via LLMs; many domains (e.g., meetings with ASR lag, robotics, broadcast sports audio) are underrepresented, and audio cues are not addressed. 

**3. Evaluator reliance.** The RL stage matches predictions to answers using a binary LLM evaluator within a temporal window—raising bias/robustness questions for reward supervision. 

**4. Comparability across settings.** “Online” baselines run at 1 fps under proactive output, while “offline” models receive clips around ground-truth times; heterogeneous interfaces/context budgets complicate strict like-for-like comparisons. 

**5. Timestamp sensitivity.** The benchmark assumes accurate triggers and verifies timestamps via a sliding-window scheme; stress-tests for jitter/dropped frames would strengthen claims of temporal robustness. 

**6. Throughput realism.** Implementation samples 1 fps with 48 visual tokens on Qwen2.5-VL-3B (8×A800, ~36h); end-to-end latency and behavior at higher fps/edge hardware are unclear. 

**7. Synthesis artifacts.** The multi-stage LLM-driven data pipeline (with type-specific checks) may import templated language or coverage biases despite two-stage validation. Auditing artifacts would help.

**8. Missing refereces.** Several influential streaming/online MLLM works from ICLR 2025 and CVPR 2025 are missing from the related work.

### Questions
**1. How would synchronized audio (speech/environmental cues) reshape trigger detection, latency, and multi-turn coherence—especially for SQU—given the current vision-focused setup?**

**2. What inter-judge agreement arises between the binary LLM evaluator used in reward shaping and human raters, and can reference-free or pairwise methods reduce bias in semantic matching?** 

**3. How do TOM’s fast inference and silent policy perform beyond 1 fps, with longer videos and on-device constraints; can adaptive token budgets preserve temporal precision under tight compute?** 

**4. Under noisy triggers or overlapping events, how robust are question/answer IDs and Tag Reward to misbinding—and can uncertainty-aware tagging reduce hallucination penalties?** 

**5. Do results transfer across domains not covered by source datasets (e.g., meetings/robotics), and will future releases include diagnostics for demographic or situational bias?** 

**6. Which single component (tags, queue state, RL rewards) most drives gains across TGQA/TST/SQU, and can smaller backbones or alternative caches replicate improvements with similar metrics?**

### Soundness
2

### Presentation
2

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
The paper introduces TV-Online, a large-scale dataset for online video understanding that covers three tasks: Temporal Grounded Question Answering (TGQA), Temporal State Tracking (TST), and Sequential Query Understanding (SQU). The proposed model incorporates two key modules, Fast Inference Mechanism and Trigger State Management, to enable efficient streaming and proactive response generation. Experiments on TV-Online, along with ablation studies, demonstrate the effectiveness of the proposed approach.

### Strengths
- **Scale and diversity:** TV-Online is a large and diverse benchmark, covering multiple aspects of online video understanding.
- **Model effectiveness:** The proposed method outperforms prior online models and is supported by detailed ablation studies.

### Weaknesses
- In Section 4.1, the modeling details are insufficient. The descriptions of the Fast Inference Mechanism and Trigger State Management lack clarity, making it difficult to fully understand how the method works. For example, what constitutes a "lightweight off-the-shelf model", and how does it analyze both inputs and previous outputs? More architectural details are needed.

- Correspondingly, the training process is underexplained. How many videos and instructions are used for each of the three training stages? How does the scale of the training data influence the performance?

- I would not agree with the claim in L419–421 that TOM surpasses all online and most offline models. In TGQA, TOM significantly underperforms compared to several offline and streaming models. Have the authors trained larger (e.g., 7B) variants for fair comparison?

- While TOM outperforms existing online models on TV-Online, I would recommend evaluating it on established streaming video understanding benchmarks (e.g., OVO-Bench [1], Streaming-Bench [2]) to assess generalization.

**References**

[1] OVO-Bench: How Far is Your Video-LLMs from Real-World Online Video Understanding?, CVPR 2025

[2] StreamingBench: Assessing the Gap for MLLMs to Achieve Streaming Video Understanding, arXiv 2024

### Questions
See the weaknesses

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
4

### Summary
This paper introduces a new benchmark and framework for a critical and practical task: trigger-centric online video understanding. The authors argue that existing systems for online video analysis lack the proactive, real-time responsiveness required for many applications. To address this, they make two primary contributions. First, they introduce TV-Online, a large-scale dataset featuring 50K videos and 500K time-stamped answers, specifically designed to evaluate trigger-based tasks. The dataset is hierarchically structured into progressively complex tasks, from basic temporal grounding to multi-turn, asynchronous reasoning. Second, they propose TOM (Trigger-Oriented Model), a streaming-oriented framework that employs an agent-like paradigm. TOM uses special protocol tags for response control, a queue-based state management system for handling multi-turn context, and a progressive training strategy that culminates in reinforcement learning to refine the model's responsiveness, coverage, and coherence. The paper presents extensive experiments demonstrating that TOM significantly outperforms existing online methods and establishes a strong baseline on the new TV-Online benchmark.

### Strengths
- The paper tackles a highly relevant and forward-looking problem. Moving from passive, offline video analysis to proactive, trigger-centric online understanding is a crucial step towards building more intelligent and interactive systems. The problem formulation is clear and well-motivated.
- The proposed TV-Online dataset is a major contribution in itself. Its large scale, diverse tasks, and hierarchical structure of difficulty (from Basic to Expert) provide a much-needed, systematic way to measure progress in this area. The comparison in Table 1 effectively highlights the gap it fills compared to existing datasets.
- The proposed TOM framework is thoughtfully designed to address the specific challenges of the task. The combination of a streaming-aware architecture, explicit state management for multi-turn dialogue, and a progressive three-stage training pipeline (pre-training, fine-tuning, RL) is holistic and sound. The use of reinforcement learning with custom rewards to optimize temporal accuracy and coverage is particularly fitting.
- The experimental evaluation is comprehensive. The authors compare their model against a wide range of both offline and online baselines, demonstrating significant performance gains. The ablation studies (Tables 4, 5, 7) are particularly valuable as they clearly validate the effectiveness of each component of the proposed framework, such as the progressive training stages and the ID tagging protocol.

### Weaknesses
- While the overall framework and its application to this specific problem are novel and effective, many of the individual technical components (e.g., using special tokens for control, progressive training, RL for policy alignment) are adaptations of existing techniques in the broader field of large language and multimodal models. The main novelty lies in their synergistic combination and customization for the trigger-centric video task, rather than a fundamental algorithmic breakthrough.
- The paper's goal is to advance "real-time" video intelligence. It introduces a "Fast Inference Mechanism" that skips redundant frames based on visual similarity. However, the analysis of this mechanism is limited. A more in-depth discussion on its computational overhead, actual latency reduction (e.g., in seconds or FPS), and the potential trade-offs (e.g., how the similarity threshold affects the risk of missing subtle but important trigger events) would be crucial to substantiate the "real-time" claims.
- The data generation pipeline relies heavily on LLMs to synthesize and verify QA pairs. While this is a common and scalable approach, it can introduce artifacts or biases from the generator models. The paper mentions a two-stage validation process, but further details on the extent of human verification, inter-annotator agreement (if applicable), and strategies used to mitigate potential LLM-induced biases would strengthen the credibility and quality of the TV-Online benchmark.

### Questions
- Could you provide more details on the "Fast Inference Mechanism"? Specifically, how is the visual similarity threshold determined and tuned? Have you analyzed the performance impact (both in terms of F1-score and latency) under different threshold settings? What are the failure modes, i.e., could this mechanism cause the model to miss triggers that occur during visually static but semantically important moments?
- Regarding the dataset construction: What was the scale of human involvement in the two-stage validation process? Were there quantitative metrics (e.g., acceptance/rejection rate, human evaluation scores) used to ensure the quality and correctness of the LLM-generated questions and time-stamped answers?
- In the post-training RL stage, the final reward is an aggregation of RFβ and Rtag with weights α and γ. Could you provide some intuition or results from a sensitivity analysis on how these hyperparameters were chosen and how they affect the model's final behavior (e.g., the trade-off between being responsive and remaining silent)?

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
5

### Summary
To address the gap in online video understanding, the paper proposes TV-Online (Trigger Video-Online), a large-scale dataset consisting of 50K videos, 200K questions, and 500K time-stamped answers. This dataset encompasses progressively complex trigger-based tasks such as temporal grounding, asynchronous scheduling, and multi-trigger reasoning. To leverage this dataset, the paper introduces a streaming-oriented model (TOM) that utilizes protocol-level tagging and structured state management, enabling frame-by-frame decision-making with precise timing and consistent responses to asynchronous, multi-turn triggers.

### Strengths
- The paper introduces the TV-Online dataset, which is large in scale, consisting of 50K videos, 200K questions, and 500K time-stamped answers.

- The proposed state management mechanism is interesting .

### Weaknesses
- The paper fails to clearly explain how the TV-Online dataset is divided into training and testing sets. Furthermore, Table 1 mixes the training data with benchmark comparisons, which is highly unfair.  

- On the TV-Online evaluation set, how is the accuracy of the answers ensured, and how is the difficulty of the questions controlled?  

- The proposed TOM is only tested on the TV-Online evaluation set, which does not sufficiently validate the effectiveness. How does it perform on other benchmarks designed for proactive video understanding, such as Vispeak[1] and ProactiveVideoQA[2]?

[1] Fu S, Yang Q, Li Y M, et al. ViSpeak: Visual Instruction Feedback in Streaming Videos[J]. arXiv preprint arXiv:2503.12769, 2025.

[2] Wang Y, Meng X, Wang Y, et al. Proactivevideoqa: A comprehensive benchmark evaluating proactive interactions in video large language models[J]. arXiv preprint arXiv:2507.09313, 2025.

### Questions
See weaknesses above

### Soundness
1

### Presentation
2

### Contribution
2
