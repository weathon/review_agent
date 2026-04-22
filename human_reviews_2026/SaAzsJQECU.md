# LLM-I: LLMs are Naturally Interleaved Multimodal Creators

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 2, 4, 8

## Abstract
We propose LLM-Interleaved (**LLM-I**), a flexible and dynamic framework that reframes interleaved image-text generation as a tool-use problem. LLM-I is designed to overcome the "one-tool" bottleneck of current unified models, which are limited to synthetic imagery and struggle with tasks requiring factual grounding or programmatic precision. Our framework empowers a central LLM or MLLM agent to intelligently orchestrate a diverse toolkit of specialized visual tools, including online image search, diffusion-based generation, code execution, and image editing. The agent is trained to select and apply these tools proficiently via a Reinforcement Learning (RL) framework that features a hybrid reward system combining rule-based logic with judgments from LLM and MLLM evaluators. Trained on a diverse new dataset using four different model backbones, LLM-I demonstrates state-of-the-art performance, outperforming existing methods by a large margin across four benchmarks. We also introduce a novel test-time scaling strategy that provides further performance gains.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces LLM-Interleaved, a framework that reframes interleaved image-text generation as a tool-use problem, empowering an LLM/MLLM agent to orchestrate diverse visual tools (e.g., search, diffusion, code) to overcome factual grounding and precision limitations. Trained via a Reinforcement Learning framework with a hybrid reward system, LLM-I achieves state-of-the-art performance across four benchmarks and is further enhanced by a novel test-time scaling strategy.

### Strengths
Current unified models are often limited to synthetic imagery and struggle with tasks requiring factual grounding or programmatic precision. LLM-I addresses these limitations by enabling the LLM agent to intelligently select and apply tools like online image search, diffusion-based generation, code execution, and image editing. This allows for the creation of more accurate and diverse visual content.

LLM-I reframes interleaved image-text generation as a tool-use problem, allowing a central Large Language Model (LLM) or Multimodal Large Language Model (MLLM) to orchestrate a diverse toolkit of specialized visual tools. This modular approach is more flexible than unified models, which often suffer from a "one-tool" bottleneck.

### Weaknesses
1. Limited Benchmark Size: The newly proposed LLMI-Benchmark consists of only 30 samples. This small dataset size is insufficient to draw convincing conclusions about the model's performance on this specific benchmark.

2. Marginal Performance Gains: The performance advantage of LLM-I is not consistently significant across Tables 1, 2, and 3. For instance, in Table 1, the GPT-4o + DALL-E 3 baseline shows competitive results, outperforming MLLM-I-7B and -32B. Similarly, ISG also demonstrates strong performance on image and structural metrics, surpassing the MLLM-I variants.

3. The proposed test-time scaling strategy appears to be computationally intensive. However, the paper only offers a theoretical analysis without quantitative experimental results (e.g., latency, resource usage) to assess its practical overhead.

Minor Typos:
Line 082: "Thourgh" should be "through".
Caption of Table 2: "LLM-Bench" should likely be "LLMI-Bench".

### Questions
The results surprisingly indicate that LLM-I variants often outperform MLLM-I variants (e.g., 4B LLM-I is nearly on par with 32B MLLM-I). Could the authors provide an analysis or hypothesis for this phenomenon?

Tool call checking seems critical for generation quality, yet it is only mentioned as part of the test-time scaling strategy. Why was this verification step not included as a standard part of the core LLM-I framework?

The authors state that existing benchmarks "do not necessitate deep reasoning." Could the authors elaborate on why "deep reasoning" is a crucial aspect to emphasize specifically for interleaved generation tasks? Besides, regarding the cooking example in Figure 10 (HowTo task), images are necessary and instructional, not decorative as the authors say. Images in HowTo task can effectively guide a user who is unfamiliar with the process like cooking. I suggest using a more representative example to state the issue.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduce the LLM-I framework, which restructures image–text interleaved generation through “tool scheduling.” A central LLM/MLLM orchestrates four complementary tools: online image retrieval, diffusion-based generation, code-driven visualization, and image editing. It is trained with reinforcement learning that combines rule-based and LLM/MLLM-evaluated hybrid rewards.

### Strengths
* The integration of four complementary tools covers a wide spectrum of use cases: real photographs are obtained via search, data charts are produced by executing code, and creative visuals are synthesized through diffusion models.

* A dedicated dataset of 4,000 samples was constructed in which the model is required to “select tools implicitly” (i.e., without explicitly naming them), thereby strengthening its reasoning capabilities.

### Weaknesses
* Tool scheduling  and test-time scaling markedly increase latency, making the framework ill-suited for real-time, low-latency interactions.

* There is no diagnostic analysis of tool failures, such as code-execution errors, or editing results that fall short of expectations, so it remains unclear which tools are the weak links.

### Questions
* The reinforcement learning reward only judges the tool selection and output matching, while ignoring the limitations of the tool (e.g., low-resolution search results, diffusion distortion). As a result, the correct choice is penalized due to the tool's shortcomings, thus breeding error aversion. How to solve this problem?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces LLM-Interleaved (LLM-I), a tool-augmented agent for interleaved text–image generation. A central LLM/MLLM is fine-tuned with RL on a tool-oriented synthetic dataset using a hybrid reward—rule-based checks for image count and tag format, plus LLM/MLLM judges. At inference, a multi-stage test-time scaling pipeline performs candidate sampling, tool-call validation, selection, tool-specific enhancement, and polishing. Evaluations on OpenING, ISG, an in-domain set, and the new LLMI-Bench show strong results; LLMI-Bench additionally includes human ratings and a tool-invocation accuracy metric.

### Strengths
1.Clear tool-calling protocol with a structured placeholder tag enabling single-pass orchestration.

2.Explicit hybrid reward with rule-based gating; equations and components are well documented.

3.TTS pipeline is described step-by-step and shows stagewise gains.

4.Competitive results across OpenING/ISG/LLMI-Bench, with tool and reward ablations.

### Weaknesses
1.Limited novelty: The contribution reads as a recipe (tool use + hybrid rewards + TTS) rather than a new learning principle, integrating known ideas: inline tool tags, judge-based RL with rule constraints, and TTS reranking/polishing.

2.Evaluation setup: The in-domain evaluation reuses training-time metrics (rule/LLM/MLLM judges).

3.Metric interpretation: “Tool Acc = 100” reflects invocation success, not task correctness.

4.Data scale: The RL dataset is small (~4k) and LLMI-Bench is tiny (30 items).

### Questions
None.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a framework called LLM-Interleaved (LLM-I), which uses Reinforcement Learning (RL) to train an LLM/MLLM agent, enabling it to intelligently 'orchestrate' and use multiple visual tools (like image search, generation, editing, etc.) to solve complex interleaved image-text generation tasks.

### Strengths
## Strengths:

- This paper describes a framework for building and training a multimodal agent, centered around an LLM/VLM, which orchestrates vision tools to achieve interleaved image-text generation. I believe this is a valuable exploration for expanding the boundaries of agent capabilities. I am very optimistic about the direction of multimodal agents, and this work represents a solid exploration in this area.

- The experiments are comprehensive, incorporating both model-based scores and rigorous human evaluations. LLM-I demonstrates superior performance on both in-domain and OOD benchmarks.

### Weaknesses
## Weaknesses:

- I have concerns regarding the choice of the reward signal. Using two large-scale LLMs/VLMs as scorers during the RL process incurs significant computational overhead. The authors should provide a runtime analysis, including the number of GPUs used and the total execution time.

- Generative reward models are susceptible to "reward hacking." Although the performance results appear fine, I would like to know if the authors considered this risk when designing the LLM-I system.

- If I recall correctly, the OpenING dataset contains 5,400 samples. Why does Line 288 state that there are only 2,000 samples?

Apart from these points, I have no other major concerns. Overall, this is an excellent piece of work.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
4
