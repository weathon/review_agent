# StreamAgent: Towards Anticipatory Agents for Streaming Video Understanding

- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Real-time streaming video understanding in domains such as autonomous driving and intelligent surveillance poses challenges beyond conventional offline video processing, requiring continuous perception, proactive decision making, and responsive interaction based on dynamically evolving visual content. However, existing methods rely on alternating perception-reaction or asynchronous triggers, lacking task-driven planning and future anticipation, which limits their real-time responsiveness and proactive decision making in evolving video streams. To this end, we propose a StreamAgent that anticipates the temporal intervals and spatial regions expected to contain future task-relevant information to enable proactive and goal-driven responses. Specifically, we integrate question semantics and historical observations through prompting the anticipatory agent to anticipate the temporal progression of key events, align current observations with the expected future evidence, and subsequently adjust the perception action (e.g., attending to task-relevant regions or continuously tracking in subsequent frames). To enable efficient inference, we design a streaming KV-cache memory mechanism that constructs a hierarchical memory structure for selective recall of relevant tokens, enabling efficient semantic retrieval while reducing the overhead of storing all tokens in the traditional KV-cache. Extensive experiments on streaming and long video understanding tasks demonstrate that our method outperforms existing methods in response accuracy and real-time efficiency, highlighting its practical value for real-world streaming scenarios. The code is available in the supplementary material.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces cache compression, early response, and a memory structure to achieve lightweight response generation,  
and proposes an interactive architecture that leverages the model's agentic capability to collect fine-grained information through precise image-level control.  
Specifically, the proposed model (1) predicts the development of future events within the video stream,  
(2) determines whether the current moment provides sufficient information for response generation,  
and (3) incorporates a tool-utilization mechanism to capture visual details.

### Strengths
Technically, the implementation appears efficient, reflecting good observations on existing cache optimization and attention mechanisms.  
In addition, the paper introduces the concept of "Proactive" planning, allowing the model to anticipate and prepare for future streaming information through planning, which appears original.  
The attempt to focus on real-time streaming environments and explore application domains different from offline LMMs is also meaningful.

### Weaknesses
- The definition of the Heuristic Score F = G + λU is abstract; the detailed computation procedure for G/U, the setting of λ, and the evaluator (which model performs the scoring) are not described.
- Although the prompt-based control structure is disclosed, the connection between G/U evaluation and the planning stage remains unclear.
- The criteria for "proactive response" or the statistics on response timing (e.g., observed frame ratio, average response delay) are not reported, making it difficult to quantitatively verify how effectively the proposed mechanism works.
- It is unclear whether the tool-use capability emerged from Qwen2.5VL's pretraining or was realized through the proposed planning procedure.
- Aside from the framework's conceptual interest, the actual performance improvement compared to the Qwen2.5VL backbone is minimal or even degraded, suggesting that the ability to use tools may not have contributed significantly to answer accuracy.

### Questions
- Please clarify the procedure for computing G and U in Equation (3).
- Can we assume that tool selection directly relies on the agentic capability inherently provided by Qwen2.5VL?
- In line 11 of Algorithm 1, what is the criterion for P̂ "not requires additional information"? Is it when U = 0? Please provide details.
- Does StreamAgent operate effectively with other architectures? If the model performing planning/tools is fixed while the interacting model is replaced with a different series, does performance still improve?

### Soundness
4

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a framework for anticipatory and proactive real-time video understanding in streaming settings. The work target continuous online domains such as autonomous driving and surveillance, and introduces mechanisms for proactive, query-driven decision-making and long-term memory management to surpass limitations in current online/offline video question-answering systems.

### Strengths
* The idea introduced goes beyond perception-reaction models; it integrates tools for temporal and spatial anticipation, query-based reasoning, and iterative planning in real-time streaming video. These are highly practically applicable.
* Unlike prior reactive or binary-trigger systems (VideoLLM-online, Dispider), this work explicitly models temporal anticipation through three planning modes (Reactive, Proactive, Speculative) scored via an A*-inspired heuristic balancing immediate utility $G$ and future utility $U$. This design addresses premature response errors.

### Weaknesses
* The paper mentions "zoom in," "object tracking," and "detailed captioning" as tools, but, perhaps I missed them, never quantifies how often each tool is invoked, their individual success rates, or their failure modes. While these tool use is highlighted, the exploration and comparative study of varying tool types and their influences is relatively narrow.
* Although effective, the planning combines heuristic scoring with agentic approaches. The robustness of these heuristics as video and query complexity scale is not fully interrogated, especially for rare or ambiguous scenarios.

### Questions
* What happens when tracking fails or zoom crops the wrong region? Are there redundant tool calls?
* It seems to me that the method is highly tuned for query-driven, real-time scenarios. If so, won’t it underperform for generic or artificially complex video reasoning tasks, lacking universal adaptability?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces StreamAgent to address real-time responsiveness and proactive decision-making in evolving video streams. StreamAgent anticipates temporal intervals and spatial regions likely to contain task-relevant information, enabling proactive and goal-driven responses. By integrating question semantics and historical observations, the agent predicts the temporal progression of key events, aligns current observations with expected future evidence, and dynamically adjusts its perception and actions. 

To ensure efficiency, the authors propose a streaming KV-cache memory mechanism that constructs a hierarchical memory structure, allowing selective recall of relevant tokens. This design enables efficient semantic retrieval while reducing the computational overhead of storing all tokens traditionally required for inference.

### Strengths
- The paper is well-written and easy to follow

- The figures are intuitive.

### Weaknesses
- The proposed streaming KV-Cache heavily borrows from StreamChat[1], with relatively incremental modifications.  

- The paper utilizes Qwen-VL-3B as the planning model; however, for complex problems, such a lightweight model struggles to perform adequate planning and often suffers from hallucination issues.  

- The performance of the proposed agent is inferior to that of a single model, indicating insufficient planning and answering capabilities.  

- What tools can the agent invoke during its operation? How does tool usage differ across various tasks？

- The paper suggests attending to task-relevant regions or continuously tracking subsequent frames, but this approach may negatively impact new tasks, especially in multi-turn dialogue scenario

[1] Xiong H, Yang Z, Yu J, et al. Streaming video understanding and multi-round interaction with memory-enhanced knowledge[J]. arXiv preprint arXiv:2501.13468, 2025.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
