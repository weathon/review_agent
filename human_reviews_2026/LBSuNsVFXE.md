# Detect What You Need: Chain-of-Causal Reasoning for 3D Intent Grounding

- Avg Score: 5.33
- Decision: Reject
- Scores: 4, 6, 6

## Abstract
Accurately matching human intentions in 3D space is an important goal of artificial intelligence. Recently, 3D Intension Grounding (3D-IG) is proposed, aiming to localize target 3D objects that match the given natural language intent. 
Compared with traditional visual grounding with a clear goal, the intent is abstract and difficult to understand, posing enormous challenges in detecting corresponding 3D objects.
To this end, towards this task, the model is required to infer the functional attributes of objects from the captured non-descriptive intent and then precisely align attributes to object features.
During this process, existing methods rely on implicit matching, which often suffers from logical gaps. As a result, they fail to establish a clear and interpretable causal reasoning between intention and object, ultimately lowering the robustness and generalizability of the model.
To tackle these challenges, we propose a new method, i.e., Chain-of-Causal Reasoning, which performs intent parsing and grounding along the causal chain. 
Specifically, the method decomposes complex intentions step by step along the causal chain into functional requirements, explicitly prioritizing and clarifying latent needs, thereby forming a causal chain from abstract intentions to object attributes and enhancing the accuracy of intent understanding.
Based on this causal chain, we construct an explicit causal graph to establish clear logical relationships between functional requirements and object attributes. Finally, a causal–visual feature alignment mechanism is introduced, which aligns causal features with the geometric–semantic features of 3D point clouds, enabling bidirectional verification between semantic reasoning and visual evidence.
Extensive experiments in 3D Intention Grounding and 3D Visual Grounding tasks demonstrate that our method effectively enhances intent understanding and improves object localization.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes Chain-of-Causal Reasoning for 3D intention grounding: a pipeline that parses a free-form intent into stepwise functional requirements, builds an explicit causal graph linking these requirements to object attributes, and aligns “causal” features with 3D point-cloud features via cross-attention for bidirectional verification. On the Intent3D benchmark, CoCR outperforms prior expert, foundation, and LLM-based baselines.

### Strengths
1. The paper clearly identifies the logical gap in verb/object matching and replaces it with an interpretable causal chain and graph, which is nicely illustrated and algorithmically specified.
2. The functional requirements establish connections between intents and objects, narrowing the gaps.
3. Causal pruning strengthens the true intent and functional requirement correspondences.

### Weaknesses
1. MA2TransVG utilizes multiple attributes to connect intents and target objects. Chain-of-Causal Reasoning actually constructs the causal graph between intent and object category. Please explain the differences between these two architectures.
2. Intent Causal Parsing relies on a fine-tuned T5-small with prompt templates. This external parsing stage could propagate errors or domain biases and may be sensitive to out-of-distribution intents or languages not seen in fine-tuning.

Minor:
This paper needs further proofreading, e.g., L302 is the same as L299.

### Questions
1. What is the difference between the functional requirements and object attributes? These two concepts are similar, and the definition of functional requirements is vague.
2. Are the functional requirements explicit? If explicit, please explain the way to generate these function requirements.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper studies the 3D Intent Grounding task. The motivation is that previous methods rely on implicit reasoning on this task. To address this, the authors propose a Chain-of-Causal Reasoning framework that performs intent parsing and grounding along a causal chain. The method first decomposes complex intents into functional requirements (Intent Causal Parsing), then constructs an explicit causal graph linking these requirements to object attributes, and finally introduces a Causal–Visual Feature Alignment module to align causal features with 3D geometric–semantic features for bidirectional verification. Experiments on 3D Intent Grounding and 3D Visual Grounding tasks demonstrate that this approach enhances intent understanding and improves localization accuracy.

### Strengths
1. It's novel to use causal reasoning to interpret abstract human intents, making the model’s decision process more explainable.

2. It's intuitive that the causal graph design explicitly connects functional requirements to visual object attributes, which helps reduce logical gaps and improve robustness. And it's also intuitive that the proposed causal–visual feature alignment provides a principled way to verify both reasoning and visual evidence.

3. Performance is good on both 3D-IG and 3D-VG.

### Weaknesses
1. I’m curious how sensitive / robust the method is to parser quality. It would be good to see a statistic analysis and bad case demo with corresponding limitation analysis.

2. The tables don't compare with and cite the recent 3D-LLM works, including Robin3D (ICCV'25) and Chat-Scene (NeurIPS'24). In the current MLLM era, the motivation to purely build such a specialist model on 3D-VG/3D-IG and discard large-scale data for synergy effects seems like a step backward. Is it faster? stronger? or more promising (why)?

3. The paper doesn’t evaluate on ScanRefer. ScanRefer is a more standard 3D-VG benchmark than Nr3D/Sr3D, because the language part is annotated by human, while Nr3D/Sr3D use templates to construct language.

### Questions
Please refer to my questions raised toward each weakness.

### Soundness
3

### Presentation
3

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
This paper proposes a Chain-of-Causal Reasoning (CoCR) framework for 3D Intention Grounding, which aims to progressively infer functional requirements from abstract natural language intentions and localize the corresponding 3D objects. The model consists of three key modules: Intent Causal Parsing (decomposing the intention into functional factors), Intent Causal Grounding (building the causal reasoning graph), and Causal–Visual Feature Alignment (aligning causal and visual representations). Experiments on the Intent3D, Nr3D, and Sr3D benchmarks show that CoCR achieves higher Top-1 Accuracy and AP scores than previous state-of-the-art methods such as IntentNet and PQ3D.

### Strengths
1. The paper is the first to introduce a chain-of-causal reasoning mechanism into the 3D intention grounding task, effectively bridging the logical gap left by semantic matching approaches and significantly improving interpretability.

2. The framework forms a complete cross-modal reasoning pipeline, from language intent parsing to functional attribute mapping and visual alignment, with a well-structured and systematic design.

3. The method is extensively validated on multiple benchmarks (Intent3D, Nr3D, and Sr3D) and provides clear visualizations, demonstrating strong generalization to complex and diverse intention descriptions.

### Weaknesses
1. The proposed CoCR framework appears closely related to existing causal graph modeling approaches. The paper should clarify the distinct contributions and theoretical differences more explicitly.

2. The introduction of multi-stage causal reasoning and feature alignment likely increases training and inference costs, but the paper does not provide a quantitative analysis of computational efficiency or deployment feasibility.

3. The Intent Causal Parsing module currently relies on a fine-tuned T5-small model. The paper does not evaluate the correctness or consistency of the parsing results. It would be valuable to analyze how parsing quality affects downstream performance, and whether using larger modern language models (e.g., GPT-5, Qwen3) could yield more accurate and semantically rich decompositions.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
