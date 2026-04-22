# Training-Free Multimodal Large Language Model Orchestration

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Different Multimodal Large Language Models (MLLMs) cannot be integrated into a unified multimodal input-output system directly. In previous work, training has been considered as an inevitable component due to challenges in modal alignment, Text-to-Speech efficiency and other integration issues. In this paper, we introduce Multimodal Large Language Model Orchestration (MLLM Orchestration), an effective approach for creating interactive multimodal AI systems without additional training. MLLM Orchestration leverages the inherent reasoning capabilities of large language models to coordinate specialized models through explicit workflows, enabling natural multimodal interactions while maintaining modularity, improving interpretability, and significantly enhancing computational efficiency. Our orchestration framework is built upon three key innovations: (1) a central controller LLM that analyzes user inputs and dynamically routes tasks to appropriate specialized models through carefully designed agents; (2) a parallel Text-to-Speech architecture that enables true full-duplex interaction with seamless interruption handling and natural conversational flow; and (3) a cross-modal memory integration system that maintains coherent context across modalities through intelligent information synthesis and retrieval, selectively avoiding unnecessary modality calls in certain scenarios to improve response speed. Extensive evaluations demonstrate that MLLM Orchestration achieves comprehensive multimodal capabilities without additional training, performance improvements of up to 7.8% over traditional jointly-trained approaches on standard benchmarks, reduced latency by 10.3%, and significantly enhanced interpretability through explicit orchestration processes. Our work establishes orchestration as a practical alternative to joint training for multimodal systems, offering greater efficiency, adaptability, and transparency for next-generation AI interactions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces "Multimodal Large Language Model Orchestration" (MLLM Orchestration), a novel, training-free framework for integrating multiple specialized Multimodal Large Language Models (MLLMs) into a unified, interactive system. Instead of the costly joint-training approach used by contemporary omni models, this work leverages the reasoning capabilities of a central "controller" LLM.

### Strengths
1. The training-free approach is highly modular and practical, avoiding costly retraining. 
2. It achieves strong performance on several benchmarks (e.g., MMStar, MMMU) compared to baseline models.

### Weaknesses
1. The conclusions drawn from the experimental section are not solid. Audio is a crucial modality in omni models, yet the paper appears to contain comprehensive benchmark evaluation results related to audio, only some ablation exploration.
2. There is a lack of comprehensive ablation studies to analyze the importance of various components of the proposed methods in the paper.
3. I have doubts about the scalability and robustness of the framework, e.g., can this training-free framework be applied to different models. And I think the authors need to provide some failure cases and qualitative examples for deeper analysis.

### Questions
What about the performance on audio-related benchmarks

### Soundness
2

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
The paper introduces a Training-Free Multimodal LLM Orchestration framework, where a controller LLM dynamically routes tasks to pre-trained expert models through structured control tokens. It features a cross-modal memory module for unified information sharing across modalities and a parallel TTS system enabling full-duplex speech interaction. The approach aims to achieve omni-modal behavior without joint training by leveraging reasoning-based orchestration instead of parameter updates.

### Strengths
1. Proposes an agent-based controller–expert orchestration mechanism avoiding multimodal retraining; Introduces a cross-modal memory for maintaining structured, textual context; Implements a parallel TTS pipeline for efficient and interruptible audio output.
2. Shows good extensibility to integrate new experts dynamically.

### Weaknesses
1. The paper lacks explicit information about the expert model types and their parameter sizes.
2. Some tables are excessively wide and exceed standard formatting limits.
3. The functional coverage is narrower than that of Qwen-Omni models; for example, it lacks speech understanding and does not include comprehensive image/video understanding benchmarks such as WorldSense or GeneralBench that evaluate holistic multimodal capabilities.
4. The efficiency analysis is not detailed or comprehensive enough.
5. There are no ablation studies to isolate and quantify the contribution of each module.

### Questions
1.  Would integrating newer experts outperform Qwen3-Omni?
2.  Other questions are in the “Weaknesses” part.

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
2

### Summary
This paper introduces "Multimodal Large Language Model Orchestration" (MLLM Orchestration), a training-free framework for building interactive multimodal AI systems. The core idea is to leverage the inherent reasoning capabilities of LLMs to coordinate specialized models rather than jointly training them for multiple modalities. The framework features three key innovations: (1) a central controller LLM that dynamically routes tasks to specialized models through control tokens; (2) a parallel text-to-speech architecture enabling full-duplex interaction; (3) a cross-modal memory integration system maintaining coherent context across modalities. Experiments demonstrate up to 7.8% performance improvement over traditional jointly-trained approaches and 10.3% latency reduction.

### Strengths
- Complete System Design: The framework is well-designed with three core modules (controller, memory pool, and parallel TTS) that have clear responsibilities and work together effectively.

- Comprehensive Experimental Validation: Thorough evaluation across multiple benchmarks including MME, MMBench, MMStar, and MMMU, covering general understanding, vision tasks, and temporal reasoning.

### Weaknesses
- Cross-modal to text conversion process unclear, potential information loss not addressed.

- Insufficient error analysis: which task types underperform? What are main orchestration failure causes?

- Essentially an engineering integration work lacking deep theoretical innovation.

- Why can training-free orchestration match or exceed trained methods? Lacks theoretical explanation.

### Questions
Please refer to the weaknesses.

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
This paper introduces an MLLM Orchestration framework that aims to integrate multiple specialized multimodal models into a unified interactive system without requiring joint training. The framework comprises three main components: (1) a central controller LLM that generates control tokens to parse user intent and route tasks to appropriate expert models, (2) a cross-modal memory pool that stores multimodal context in JSON format, and (3) a parallel batch TTS architecture for low-latency speech synthesis. The authors evaluate their approach across several benchmarks including MME, MMMU, MMStar, and Video-MME, claiming performance improvements over existing models like GPT-4o while maintaining modularity and interpretability.

### Strengths
- **Practical system design**: The paper presents a complete end-to-end system that addresses real challenges in building interactive multimodal AI systems. The modular architecture supports extensibility and allows for component replacement without retraining.

- **Comprehensive evaluation scope**: I appreciate the breadth of benchmarks covered, including general understanding (MME), visual reasoning (MMStar), temporal reasoning (MMMU), and video understanding (Video-MME). The evaluation across different video lengths in Figure 3 provides useful insights.

- **Clear architectural presentation**: The system workflow is well-illustrated, making it easy to understand how different components interact. The control token mechanism and dynamic prompting strategy are clearly explained.

- **Attention to interaction quality**: The parallel batch TTS design with semantic segmentation shows thoughtful consideration of user experience, achieving measurable latency reduction (10.3%) that matters for real-time interactions.

- **Transparency and interpretability**: The explicit routing decisions through control tokens provide interpretability advantages over black-box joint training approaches.

### Weaknesses
- **Experimental baseline concerns**: I am concerned about potential errors in the baseline comparisons, particularly the GPT-4o results. The MMMU score reported in Table 1 (59.20%) differs substantially from publicly reported benchmarks (~69.1%). I encourage the authors to carefully verify all baseline numbers and consider whether some results might be from different model variants (e.g., GPT-4o vs GPT-4o-mini). This verification is essential as it significantly affects the interpretation of performance gains.

- **Incomplete literature positioning**: The related work section would benefit from broader coverage. I suggest discussing foundational work on agent-based reasoning and tool use (such as ReAct-style approaches and early tool-augmented language models) to better contextualize the orchestration concept. Additionally, recent work on multi-agent systems and multimodal memory mechanisms would help readers understand how this work fits into the broader landscape. A more thorough positioning would strengthen rather than weaken the contribution by clarifying the specific advances made.

- **Limited technical depth in some components**: While the overall system is well-engineered, some components could be developed more deeply. For instance, the cross-modal memory integration primarily converts multimodal information to text stored in JSON—I wonder if more sophisticated retrieval mechanisms or hierarchical memory structures could enhance performance. The control token design, while functional, would benefit from more analysis about optimal token vocabularies or learned routing strategies.

- **Missing ablation studies**: I would have appreciated ablation experiments to understand the individual contributions of the three main components (controller, memory pool, parallel TTS). This would help isolate which design choices most significantly impact performance and where future improvements might be most valuable.

- **Computational cost analysis**: While the paper reports a single "Time" metric, I believe a more detailed analysis of computational overhead would be valuable. Orchestrating multiple models potentially increases inference costs and latency—understanding these trade-offs would help practitioners assess the approach's applicability to their scenarios.

### Questions
1. **Regarding baseline verification**: Could you please clarify the source of the GPT-4o baseline results, particularly the 59.20% MMMU score in Table 1? This differs from commonly reported benchmarks. Were these results obtained from your own evaluations using specific API versions, or sourced from published reports? If possible, could you verify whether GPT-4o-mini results might have been inadvertently used?

2. **On data inconsistencies**: I noticed VITA's Video-MME score appears as 46.40% in Table 1 but 59.20% in Table 2. Could you explain this discrepancy? Understanding this would help clarify the experimental setup.

3. **Regarding orchestration overhead**: How does the computational cost and latency of your orchestration approach compare to single-model baselines? While you report time metrics, a breakdown showing the overhead introduced by routing decisions and multiple model calls would be informative.

4. **About memory pool design**: Have you explored more sophisticated memory retrieval mechanisms beyond JSON storage and lookup? For instance, could semantic search or hierarchical memory structures improve performance on complex multi-turn interactions?

5. **On component contributions**: Could you provide ablation results showing the independent contribution of each major component (controller, memory pool, TTS)? This would help the community understand which design choices are most impactful.

### Soundness
2

### Presentation
3

### Contribution
2
