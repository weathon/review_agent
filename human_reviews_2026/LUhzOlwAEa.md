# TIR-Bench: A Comprehensive Benchmark for Agentic Thinking-with-Images Reasoning

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
The frontier of visual reasoning is shifting toward models like OpenAI o3, which can intelligently create and operate tools to transform images for problem-solving, also known as thinking-\textit{with}-images in chain-of-thought. Yet existing benchmarks fail to fully capture this advanced capability. Even Visual Search, the most common benchmark for current thinking-\textit{with}-images methods, tests only basic operations such as localization and cropping, offering little insight into more complex, dynamic, and tool-dependent reasoning.
We introduce \textbf{TIR-Bench}, a comprehensive benchmark for evaluating agentic thinking-with-images across 13 diverse tasks, each requiring novel tool use for image processing and manipulation in chain-of-thought. We evaluate 22 multimodal large language models (MLLMs), from leading open-sourced and proprietary models to those with explicit tool-use augmentation. Results show that TIR-Bench is universally challenging, and strong performance requires genuine thinking-with-images capabilities. Finally, we present a pilot study comparing direct versus agentic fine-tuning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces TIR-Bench, a novel and comprehensive benchmark designed to evaluate the agentic thinking-with-images capabilities of MLLMs. The authors argue that existing benchmarks focus on static image understanding and fail to assess the ability of models like O3 to actively use tools to transform and manipulate images during problem-solving. TIR-Bench comprises 13 diverse tasks that are intentionally designed to be unsolvable without tool use. The evaluation of 22 models demonstrates that the benchmark is highly challenging (46% accuracy for the best model, O3-TU) and that the availability of tool-use capabilities is the key differentiator for performance.

### Strengths
1. The paper identifies a significant gap in current multimodal evaluation: the lack of assessment for agentic or tool-assisted image reasoning, which is a key emergent capability of frontier models.
2. The 13 tasks in TIR-Bench are cleverly designed with a common trait: they cannot be solved by static observation alone. This strongly compels models to adopt a "think-tool-observe" loop, aligning perfectly with the benchmark's evaluation goals.
3. The experiments, especially the comparison between tool-enabled and non-tool-enabled models, clearly demonstrate that tool use is essential for these tasks and quantify the significant room for improvement, even for SOTA models.

### Weaknesses
1. The paper's core evaluation is on "tool use," but this is primarily equated to a "Code Interpreter" in the experiments. This makes the evaluation more of a test of a model's "Python CV coding ability" rather than a test of general-purpose "tool calling" capabilities.
2. In the experimental setup, proprietary SOTA models (O3-TU, O4-mini-TU) are granted access to a code interpreter, while open-source models are not. This is an unfair comparison, as the poor performance of open-source models might stem from the lack of a robust execution sandbox, not from inferior reasoning.
3. While the paper states many samples are newly created, a portion of the data is sourced from existing public datasets. It is difficult to guarantee that large models like O3 have not seen these or highly similar samples during pre-training.
4. The best model (O3-TU) scores only 46%, meaning it fails most of the time. The paper mentions some failures but lacks a systematic analysis of why. On which tasks do models fail most? What are the common failure modes?
5. Some tasks (e.g., Word Search, Low-Light VQA) feel overly synthetic, designed specifically to force the use of a particular tool (e.g., pixel comparison, image enhancement). While this serves the benchmark's purpose, it is debatable whether these tasks represent the diverse, unpredictable tool-use scenarios MLLMs will face in the real world.

### Questions
1. Many tasks in TIR-Bench seem better suited for traditional Computer Vision algorithms than for MLLMs. While models can solve them by writing code to call CV libraries, it's debatable whether this tests the MLLM's "reasoning" or its ability to recall and execute CV library APIs.
2. For a benchmark claiming to evaluate a "thinking process," the paper relies solely on final-answer accuracy. This obscures crucial details: did the model solve the problem efficiently, or via extensive, ineffective trial-and-error? Metrics for the quality, efficiency, or plausibility of the reasoning path are missing.
3. The pilot study in Sec 4.5 is conducted on only one task. The finding that "Tool-Use SFT" outperforms "Direct SFT" is insightful, but this conclusion can hardly be generalized to the other 12 complex tasks in TIR-Bench based on this single experiment.
4. See weakness.

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
4

### Summary
The authors present TIR-Bench, a benchmark for code/tool-assisted visual reasoning on images. They identify that previous benchmarks only require simple tools at most. They curate the benchmark for tasks that require complex code/tool-assisted reasoning, with a focus on dynamic or agentic approaches, with a combination of new problems and those from existing datasets. The results suggest all models have difficulty with these tasks.

### Strengths
- The paper addresses a critical need for new benchmarks in this area.
- It identifies more difficult tasks than previous datasets suitable for new code-execution based approaches.
- It highlights issues with current state-of-the-art models. 
- Table 2 presents a wide selection of model evaluations. 
- Many qualitative examples from the benchmark are shown.
- Preliminary experiments on SFT are presented.

### Weaknesses
- The data seems to need more verification; annotation by one student without at least one more pair of eyes checking may not be reliable.
- Synthetic and hand-annotated data are mixed here, when they seem to evaluate markedly different capabilities. Perhaps they should be switched? 
- The provenance of the data is unclear, posing potential issues for usage. 
- The figures showing model responses with interleaved thinking/code, like 4, 5, 24, 25 are difficult to parse.

### Questions
- The function calling and code comparison is interesting, but is the primary difference actually the presence of predefined tools rather than whether function calling or full code generation is used? This should be clarified more.

### Soundness
2

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
The paper presents TIR-Bench, a new benchmark designed to evaluate the agentic visual reasoning ability of multimodal large language models (MLLMs). Such a thinking-with-images process requires the models to actively manipulate images as part of the reasoning process rather than static analysis. The benchmark consists of 13 tasks, such as rotation, maze solving, and jigsaw puzzles, and the dataset has 1,215 examples. The authors evaluate 22 MLLMs, including open-source, proprietary, and tool-using models. Results show that the TIR-Bench is universally challenging, with 46% accuracy from o3-TU as the performance. It reveals that non-agentic models perform substantially poorly, highlighting the necessity of tool-based reasoning for complex visual tasks. Finally, the authors also study function calling behaviors and run a pilot fine-tuning comparison on rotated-OCR, finding that agentic/tool-use SFT outperforms direct SFT.

### Strengths
- The task design is comprehensive, covering 13 diverse tasks that span a broad range of visual reasoning abilities. Each task forces models to manipulate images rather than passively describing them, enabling a faithful evaluation of thinking-with-images reasoning.
- The benchmark is sufficiently challenging and ensures it is a useful, long-term suite for measuring MLLMs’ reasoning ability.
- A central contribution of TIR-Bench is that it explicitly investigates how models interact with images, which is crucial to understand the model’s agent use behavior and assess their ability for visual reasoning tasks.

### Weaknesses
- TIR-Bench is designed to evaluate models’ tool-based visual reasoning ability, but many evaluated models (e.g., LLaVA, Qwen2.5-VL, InternVL) cannot execute code or invoke tools. Their inclusion mainly serves as a static baseline rather than a true test of “thinking-with-images.”
- 1215 examples are divided into 13 tasks, making several tasks have modest data sizes. It raises concerns about the benchmark robustness given the data scale.
- Using GPT-4o to extract final answers can introduce parsing bias. However, the authors do not verify GPT-4o’s extraction accuracy. The influence of the extraction process should be excluded.
- The agentic vs direct SFT pilot uses Qwen-2.5-VL-7B only for a single task. Broader models and tasks are needed to generalize the conclusion.
- Some models can use external tools during reasoning, while others do not have access. This discrepancy makes it difficult to interpret the model’s intrinsic reasoning capability and limitations. The authors should clarify what external tools are accessible for each model.
- While the paper defines thinking with images as a process in which models manipulate images through tool use, the results in Section 4.4 indicate that models do not always invoke tools unless explicitly guided by the prompt. This raises questions about whether the poorly performing models fail due to insufficient reasoning ability or simply because they were not prompted to use available tools.

### Questions
- Could the authors clarify the rationale for including non-agentic models for tool-using ability, given that they can not use tools or write code and execute?
- Could the authors justify whether the data scale is sufficient to get statistically meaningful and reliable conclusions?
- Could the authors validate the effectiveness of using GPT-4o for answer extraction?
- Could the authors provide the fine-tuning comparison on more models and different tasks?
- For tasks that inherently benefit from external tools, please report the tools the models use.
- Could the authors provide the exact prompts used during benchmarking and clarify to what extent the models are instructed or encouraged to use tools?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper aims to better evaluate the ability to think with images in the chain of thought. Current work focuses primarily on localization and cropping capabilities. To promote and inspire broader capabilities, this paper proposes a more challenging benchmark involving thinking with images. The proposed benchmark includes 13 diverse tasks, encompassing image understanding tasks such as color setting, image selection, and maze solving. The authors evaluated open-source and proprietary models in experiments, and further evaluated the proprietary model, which can utilize image processing tools.

### Strengths
1. The paper elevates thinking-with-images from simple localization and cropping capabilities to more complex and diverse ones.

2. The authors set up a tool-using scenario, experimenting with state-of-the-art models o4-mini and o3 on the proposed benchmark, achieving the expected performance improvements.

### Weaknesses
1. Table 2 only evaluates a limited number of open-source models, covering only Illava, Qwen2.5-VL, and InternVL3. Many large-scale open-source multimodal models have not been extensively evaluated.

2. The authors did not attempt to configure tool-using on open-source models with larger parameters, such as those with over 32 bytes of parameters. Performing such evaluations on the proposed benchmark would provide a clearer understanding of the actual capabilities of the open-source models.

### Questions
1. Refer to the issues raised in the weakness section.
2. Figure 3 is unclear and difficult to interpret. The authors should provide a more detailed explanation.

### Soundness
3

### Presentation
3

### Contribution
2
