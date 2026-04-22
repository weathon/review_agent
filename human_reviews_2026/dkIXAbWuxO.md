# Earth-Agent: Unlocking the Full Landscape of Earth Observation with Agents

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 8

## Abstract
Earth observation (EO) is essential for understanding the evolving states of the Earth system.  Although recent MLLMs have advanced EO research, they still lack the capability to tackle complex tasks that require multi-step reasoning and the use of domain-specific tools. Agent-based methods offer a promising direction, but current attempts remain in their infancy, confined to RGB perception, shallow reasoning, and lacking systematic evaluation protocols.
To overcome these limitations, we introduce Earth-Agent, the first agentic framework that unifies RGB and spectral EO data within an MCP-based tool ecosystem, enabling cross-modal, multi-step, and quantitative spatiotemporal reasoning beyond pretrained MLLMs. Earth-Agent supports complex scientific tasks such as geophysical parameter retrieval and quantitative spatiotemporal analysis by dynamically invoking expert tools and models across modalities. To support comprehensive evaluation, we further propose Earth-Bench, a benchmark of 248 expert-curated tasks with 13,729 images, spanning spectrum, products and RGB modalities, and equipped with a dual-level evaluation protocol that assesses both reasoning trajectories and final outcomes. We conduct comprehensive experiments varying different LLM backbones, comparisons with general agent frameworks, and comparisons with MLLMs on remote sensing benchmarks, demonstrating both the effectiveness and potential of Earth-Agent. Earth-Agent establishes a new paradigm for EO analysis, moving the field toward scientifically grounded, next-generation applications of LLMs in Earth observation. More information about Earth-Agent can be found at https://github.com/opendatalab/Earth-Agent

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Earth-Agent extends multimodal LLMs for Earth observation (EO) by enabling spectral data understanding, multi-image reasoning, and scientific tool use. It employs a ReAct-style LLM agent that dynamically invokes 104 EO expert tools via a Model-Context Protocol for spatiotemporal analysis and workflow planning. The authors introduce Earth-Bench, a 248-task benchmark (13,729 images) spanning RGB, spectral, and geospatial data, evaluating both final accuracy and reasoning quality. Experiments using backbones like GPT-5, Kimi K2, and DeepSeek-V3.1 show Earth-Agent dramatically outperforms prior MLLMs and generic agents on EO reasoning and traditional benchmarks, establishing a new agentic paradigm for scientific Earth analysis.

### Strengths
- Earth-Agent enables multimodal, multi-step EO reasoning by dynamically invoking 100+ expert tools, which is far beyond prior static MLLMs.
- Earth-Bench with 248 tasks and 13k+ images rigorously evaluates both reasoning and results, setting a new standard for EO agents.
- Consistently outperforms general agents and prior MLLMs across modalities and EO benchmarks.
- Extensive comparisons and ablations across LLM backbones validate tool-use effectiveness. Also identify and discuss failure modes like tool/file hallucination, offering clear improvement directions.

### Weaknesses
- This zero-shot agent sometimes makes mistakes that a more specialized model might avoid, like tool hallucination and file hallucination. These errors suggest the LLM doesn’t fully understand the tool API or environment state at times. A weakness of the paper is that it does not attempt any intermediate fine-tuning or training to address these issues.
- The approach relies on a single “expert” solution path for each task. It’s unclear how the evaluation handles an agent that finds a correct answer via a different valid approach. The authors do measure partial matches, which helps, but the paper doesn’t fully discuss if an agent could be penalized for creative but correct reasoning that deviates from the expert trajectory. This could be a minor weakness if the evaluation isn’t robust to alternative solutions. Clarification would be helpful to ensure the benchmark is fair to different reasoning styles.

### Questions
- The evaluation uses expert-curated solution trajectories for step-by-step scoring. How does Earth-Bench handle cases where the agent’s reasoning differs from the expert’s? For instance, if an agent uses a slightly different sequence of tools or a valid alternative formula to reach the correct answer, do the metrics give it credit? Is the benchmark tolerant to creative solutions? A discussion on how variability in reasoning is treated would be useful for users of Earth-Bench.
- The identified tool and file hallucinations indicate the agent sometimes loses grounding in the environment. What strategies do the authors think could reduce these errors? For example, would fine-tuning the LLM on a corpus of tool-use transcripts or adding a verification step where the agent double-checks the existence of a tool/file before calling it help? During development, did the authors try any prompting techniques to address this?
- The domain of tool-using LLM agents is evolving (e.g., SciToolAgent for scientific workflows, etc.). Could the authors contrast Earth-Agent with such systems? Is the main difference primarily the focus on Earth science modalities, or are there methodological innovations that set it apart?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Earth-Bench, an expert-certified benchmark accompanied by an agentic framework for evaluation. It includes 248 questions, making it the largest Earth Observation (EO) benchmark to date in the LLM-agent domain. Using their Earth-Agent framework, GPT-5 successfully resolves around 60% of the questions, and the overall ranking aligns well with current expectations of LLM capabilities.

However, the paper lacks clarity regarding the dataset curation process, e.g., whether each question was reviewed by multiple experts. Additionally, several inconsistencies and issues within the benchmark data, as well as insufficient detail about the true number of unique questions (i.e., whether it is genuinely 248 questions or just 124 formulated in two ways), raise concerns about the benchmark’s reliability and robustness.

### Strengths
1. Overall Evaluation:
This paper explores a wide range of tools and evaluates how effectively current LLMs, acting as agents, can leverage these tools for question answering. The experiments are comprehensive, and the results are clearly presented and well-analyzed.

2. Contribution of Toolset:
The predefined set of 104 tools covers a broad spectrum of Earth Observation (EO) tasks and represents a valuable contribution toward advancing tool-augmented agents in the EO domain.

### Weaknesses
While this paper seems to propose a genetic framework and a comprehensive benchmark, I found several issues that compromise my ratings for this work. Specifically about the **rigor and the reliability of the dataset**. Below are two of the main weaknesses I observed

**1. Unclear data annotation pipeline**

The authors stated, “From these data sources, a team of domain experts ...,” but no further information is provided. For example, what is the background of these domain experts? Is there a cross-reviewing process? What happens if inconsistencies arise? Was an LLM used during the curation process? How did the experts formulate the questions?

Without a clear description of the question validation process, it is difficult to ensure that the benchmark is trustworthy and reliable.

**2. Questions about the benchmark dataset**

I reviewed some of the benchmark data provided, and found several questions. 

**- Question Predictability:**
On Page 37, the question involves Landsat 8, which has a 16-day revisit cycle. Thus, between June and September, only about 8 +- 1 observations are available. Thus, even without reading the question carefully, one could guess any option less than 8, which again raises concerns about how informative the task is. (The options provided are (1) 5, (2) 10, (3) 12, and (4) 18 days. )

**- Incorrect Formula Range:**
On Page 43, the Normalized Difference Turbidity Index (NDTI) is reported with values ranging from −666 to 100. However, NDTI is typically computed as (red − green)/(red + green), meaning its range should be [−1, 1]. For example, Option B claims: “The average NDTI increased from −666.47 to −61.17,” which is mathematically implausible and impossible. Feel free to correct me if the authors use different metrics.

**- Methodological Justification:**
On Page 51, the method count_skeleton_contours is used to estimate the number of houses, but no reference or justification is provided. A citation or methodological rationale is necessary to validate this choice.

**- Question Design Issues:**
For example, on Page 53 (Line 2808), only four images are provided, but the answer choices are (A) 3, (B) 8, (C) 10, (D) 16). It seems trivial to select “A” without even understanding the question, which undermines the reliability of the benchmark. A similar issue appears on Page 63 (Line 3356).

**- Self-Conflicting Content:**
The questions on Pages 55 and 56 appear inconsistent. The questions ask about the total area. However, the first question asks about 
sports infrastructure including both baseball and tennis courts, while the second focuses only on baseball courts, yet the answers are both 1500m*2, which is confusing to me. The two questions should have different options and the current version makes the intended reasoning unclear.

**- Ambiguity in Measurement Definition:**
On Page 65, the task asks for the distance between the two farthest planes in the image, but it is ambiguous how this distance is defined. By contrast, Page 58 explicitly specifies centroid-based distance, suggesting that the two questions target different notions but are treated as equivalent in the benchmark.

I understand that some technical details may have been omitted for brevity, and I would appreciate clarification from the authors regarding these issues and any potential misunderstandings.

### Questions
**Clarification of Anti-Planning and Instruction-Following**

Could the authors clarify whether all 248 questions each have both an auto-planning and an instruction-following version, or whether 124 are auto-planning and the other 124 are instruction-following? If it is the latter, then claiming to have 248 unique questions seems somewhat misleading, as there would actually be only 124 distinct questions expressed in two different formulations.

**Dataset Composition**

How many products or sensors are included in the dataset? So far, I can identify Landsat and MODIS (ASTER), and potentially some high-resolution imagery (Page 52) with unspecified sources. It would be helpful if the authors could describe the data sources, products, and collections in more detail, along with their distribution.

**Figure 7 Inconsistency**

In Figure 7, there appear to be more than 300 error counts, even though the dataset only contains 248 questions. Could the authors clarify how these counts are computed and why they exceed the total number of questions?

**Evaluation Without Tool Use**

How do the authors implement and evaluate model performance without tool use? For RGB-based questions, how do the models perceive or interpret the images if they are not allowed to invoke perception modules?

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper has two main accomplishments: First it proposes an agentic framework for earth observation integrating 104 tools and models. Secondly they prepare a benchmark (Earth-Bench) of 248 expert-curated questions with 13,729 image.  They then evaluate the agent on the benchmark, comparing it to openai operator and Manus. They also demonstrate that the model outperforms remote sensing foundation models.

### Strengths
The sterngth of this work are the results: both with the agentic system design, the benchmark design, the demonstration of SOTA results across a range of existing remote sensing benchmarks . The agentic system uses a wide set of specialized tools that were reasonably selected and the authors evaluated this system across a range of LLMs. They also constructed a smaller benchmark (earthbench-lite) to enable comparison against generalized agents and show that their agent outperforms consistently

### Weaknesses
When reading the paper I would have liked to learn more about the benchmark itself. How it was constructed, what are the types of question in it, and so forth. Some information is in the appendix but I feel more detailed analysis would be helpful. 

Also the authors have a long tool list and they also built a benchmark. We need to understand the relationship of the tool list to how the benchmark was constructed: at an extreme limit you could have included tools that were specifically chosen to solve individual sets of questions and thus push benchmark performance way up but in a way that is not extensible to larger sets of questions not included. I'd feel better about this if the authors could give a rationalization for their toolset that is independent of their benchmark.  Some solace is given since they outperform previous benchmarks but as they themselves point out the benchmarks are not really comparable. 

One benchmark that might be more useful to compare against is this one: https://openreview.net/pdf?id=oaYShIy3Xe.  I'd be curious how well this system does against that benchmark

### Questions
Please respond to the weaknesses I highlighted:
1. Explain more about the benchmark and why you made the choices you did when constructed it
2. Explain the relationship of your toolset to your benchmarks and what you did to make sure you weren't cherry picking tools for benchmark questions
3. Explain how your agent does on the cited benchmark above.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces Earth-Agent, a novel agent-based framework for Earth Observation (EO) analysis using multimodal large language models (MLLMs). Unlike previous EO models that rely mainly on RGB data and limited reasoning, Earth-Agent integrates RGB and spectral EO data and leverages MCP tools for cross-modal, multi-step, and quantitative spatiotemporal reasoning. It can dynamically employ expert tools for tasks such as geophysical parameter retrieval and spatiotemporal analysis. To evaluate its performance, the authors propose Earth-Bench, a benchmark comprising 248 expert-designed tasks and 13,729 images, offering dual-level evaluation on both reasoning and outcomes. Experiments across various LLM backbones show that Earth-Agent significantly outperforms existing MLLMs and general agent frameworks.

### Strengths
- Solving earth observation tasks using an agentic solution is quite novel and timely. Bringing a tool-use angle from agent solutions to improve performance seems very innovative.
- The proposed benchmark can be useful for the research community in terms of evaluating agentic systems.
- Experiments are thorough and comprehensive. Findings are supported by enough experiments.
- Paper is well-written and easy to follow.

### Weaknesses
- In any agentic system, a comparison of latency with a single-agent or single-model baseline is expected. Although the performance gains are significant, it is also important to account for the computational cost, including the number of LLM or tool calls and the total end-to-end response time. Such comparisons are important to demonstrate the overall efficiency and effectiveness of the proposed agent framework. Please add a discussion around it in the paper.

- A discussion of performance trends w.r.t. to the number of tools used for a given task is important to understand how the proposed agentic system behaves as task complexity increases and the number of tool calls grows. It would also help evaluate the system’s latency and scalability under such conditions. I would like to see such ablation studies included in the paper to better demonstrate the effectiveness of the proposed method.

### Questions
- One aspect that is unclear is how the proposed method handles cases where a tool produces an incorrect output. The framework does not appear to inherently address such failure cases. For example, if an error occurs during the first tool call, it may propagate and amplify throughout the subsequent reasoning chain. Did the authors encounter or observe such cases in their experiments?

- Throughout the paper, the authors repeatedly state that the dataset is expert-curated. However, the paper does not specify who these experts are or provide details about their backgrounds. It would be helpful to clarify whether the curators come from computer science with remote-sensing expertise, or if they are domain specialists primarily from the remote-sensing or Earth observation community.

### Soundness
3

### Presentation
4

### Contribution
4
