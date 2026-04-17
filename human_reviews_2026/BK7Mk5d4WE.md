# LongHorizonUI: A Unified Framework for Robust long-horizon Task Automation of GUI Agent

- Decision: Accept (Poster)
- Scores: 4, 2, 6, 6

## Abstract
Although agents based on multimodal large language models (MLLMs) demonstrate proficiency in general short-term graphical user interface (GUI) tasks, their robustness remains a significant challenge for handling complex long-horizon tasks in dynamic environments . In response, the LongHorizonUI framework is proposed to improve the sustained reliability of agents in long-horizon GUI tasks. To overcome core limitations, we establish a comprehensive long-horizon benchmark, LongGUIBench, covering multiple categories of games and complex general applications, with long-horizon tasks defined as requiring more than 15 steps for rigorous evaluation of long-horizon reasoning capabilities. Based on this, a Multimodal Enhanced Perceiver is designed to incorporate element detection and text recognition models, assigning unique indices to interface elements, thereby reinforcing state representation. Furthermore, a Deep Reflection Decider engine is introduced, incorporating a structured multi-level feedback validation mechanism to enable progressive reasoning and ensure accurate action execution with predictable trajectories. Finally, we introduce a Compensatory Action Executor that combines multiple degradation compensation operations with a process rollback strategy based on execution progress monitoring to ensure operational effectiveness in long-horizon task logic. Experimental results demonstrate that LongHorizonUI achieves substantial long-horizon modeling improvements on LongGUIBench while retaining competitive performance on diverse public benchmarks. The code and models will be publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a framework named LongHorizonUI, aiming to enhance the reliability of MLLM-based agents in long-horizon tasks. LongHorizonUI introduces three components: a Multimodal Enhanced Perceiver that integrates detection and recognition, and assigns indices to reinforce state; a DeepReflection Decider that leverages multi-level feedback for iterative reasoning and precise execution; and a Compensatory Action Executor that monitors progress, and performs compensation or rollback to maintain long-horizon robustness. Experiments show that LongHorizonUI significantly improves long-horizon performance on LongGUIBench while remaining competitive across various public benchmarks.

### Strengths
1. The reliability of MLLM-based agents in long-horizon tasks is one of the main challenges in GUI Agent application. Developing reliable GUI agents for long-horizon tasks is of great importance.
2. Dataset Contribution: This paper introduces LongGUIBench, a novel benchmark for long-horizon GUI interaction. It encompasses diverse complex tasks across multiple application domains.

### Weaknesses
1. A more concise analysis of the abrupt success rate drop in long-horizon tasks is needed: disentangle root causes (e.g., recurring critical-step errors vs. failures to recover from erroneous states) and explicitly link these to how the proposed MEP/DRD/CAE modules address key long-horizon challenges.
2. In Section 2.4, please clarify the feasibility of rollback in GUI tasks: specify applicable task scopes, technical details of state snapshot storage/restoration.
3. In Section 3.3, Table 3 has inconsistencies: (1) AndroidControl/GUI-Odyssey are Android-focused, so explain the inclusion of "Desktop/Web" columns and cite relevant references; (2) the "GUI-R1-7B baseline" mentioned in the navigation capability text is missing from the table—add it or align the text with the table.
4. Typo errors: standardize inconsistent bolding in Table 1 and Table 3, and adjust Figure 2b to resolve legend overlap.

### Questions
1. Highlight the novelty of this paper compared to Mobile-Agent-V3 and D-Artemis.
2. Provide an ablation study for the Deep Reflection Decider module.
3. In Section 3.3, add quantitative step-length analysis for AndroidControl and GUI-Odyssey (similar to Figure 2b).

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work targets long-horizon GUI agents, especially tasks exceeding 15 steps. The authors introduce the LongGUIBench benchmark and the LongHorizonUI framework, which integrates three carefully designed modules—Multimodal Enhanced Perceiver, Deep Reflection Decider, and Compensatory Action Executor. Experiments demonstrate improved long-horizon performance on GUI tasks.

### Strengths
This is an important topic. GUI agents serve both as a crucial platform for algorithm evaluation and as a practical application of VLMs. The long-horizon GUI agent setting remains underexplored, and strong long-horizon planning is particularly valuable.
The contributions seem compelling: the authors introduce a new framework and benchmark that address this gap.

### Weaknesses
However, although the authors propose both a benchmark and a framework, I think there is still room for improvement before acceptance.

The paper’s writing is not very strong. There are many typos, and the names and function descriptions of the components are confusing.

**Typos**:

* Figure 2 caption: “1015 steps” → likely “10–15 steps”
* Quotation marks: should be ` ', instead of ' ' (Lines 187–188)
* Line 262: “actioninstruction” → “action instruction”
* Line 263: “semanticphysical” → “semantic physical”
I present the weaknesses of the proposed framework’s components as follows.

**Multimodal Enhanced Perceiver.** It is not clear why the perceiver is “enhanced.” Identifying icons and text descriptions seems standard. The paper gives few details about the enhanced detector (Line 199). Is it trained from a VLM, or does it incorporate additional enhanced functions? It is also unclear how the confidence score is obtained—does it come from hidden states, or from an uncertainty-estimation module?

In Eq. 1, what do the authors mean by multiplying the indicator function value by the detected text?

There is also a lack of detail about high-priority areas. What is the motivation? Where are these areas defined? What is the “repair” function? How does it operate, and what does the template library look like? Why are both the template library and the detectors/OCR necessary?

For the **Deep Reflection Decider**, based on the name and the function description in Lines 226–232, I am confused about whether this component only performs reflection or also includes decision-making. I am not sure whether a new name would help, such as “Actor with Deep Reflection.”
In addition, what is “multi-level” and what is the “triple closed loop”? Figure 3 appears to show five steps for this component, but the text also describes a two-step process (pre-execution / post-execution) and three bullet points in Lines 236–247. Could you please modify Figure 4 to align these descriptions?

How exactly does this component act? From the current description, the agent outputs only an action; there does not seem to be actual post-execution reflection or validation, like how to update the agent's knowledge or how to revise.

The authors also mention a keyword extractor in Line 253, but provide no description.

There are also no details about the brief revision step

**Compensating Action Actuator**

What is the name of the component in Section 2.4: Compensating Action Actuator (section title) or Compensating Action Executor (abstract)?

What is a device-aware scaling matrix?

Rollback may be impossible in some cases. It is unclear how the proposed framework addresses this. What happens if the compensating action actuator makes a change to the screenshot but fails? Some changes can be rolled back, but others cannot, such as clicking “next page” or accidentally sending an email.

Regarding the agent’s grounding capability, the framework introduces many components and is quite complex, but the improvements may be limited: LongHorizonUI vs. InfoGUI-R1: 90.4 vs. 87.5 on ScreenSpot.

In conclusion, this work lacks sufficient detail for the writing.


For the **framework design**, in addition to the above, I still have concerns:

- The framework seems to complex.
- While the framework seems focused on avoiding error propagation—which is important—long-horizon planning, subgoal proposal, validation, and memory may be even more critical for long-horizon tasks. It is not clear to me: how does the framework contribute to these aspects?

### Questions
How does the model perform beyond 18 steps?

What exactly are agent decision mechanisms, and why do they lack long-horizon generalization due to dataset diversity? Would it be clearer to explain the notion of dataset diversity, or simply state that the datasets lack long-horizon trajectories? The term “data diversity” may be confusing here, since the tasks appear to be limited to Mobile GUIs, which may not generalize to Windows.

Likewise, what are MLLM-based mechanisms? Do they refer to a multi-agent framework for GUI tasks? Which properties confer superior sequence-modeling capability? Please also clarify dynamic interface shifts. Concrete examples comparing different methods—their challenges and how your approach addresses them—would help readers understand why your method is suitable and efficient for long-horizon GUI tasks.

Finally, what is meant by action-instruction uncertainty? Is it tied to grounding capability? And do free-format outputs denote high-level instructions or the agents’ textual step goals?

### Soundness
2

### Presentation
1

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
The paper primarily addresses the performance degradation of GUI agents in long-horizon tasks (over 15 steps), proposing a unified framework named LongHorizonUI along with a corresponding long-task benchmark, LongGUIBench.

### Strengths
- The problem is clearly defined and grounded in reality: it directly targets the degradation issue in long-horizon GUI automation tasks (those exceeding 15 steps).
- Empirical results are persuasive.

### Weaknesses
- During data parsing and alignment, an MLLM was used for semantic annotation and structured extraction (page 3), which may introduce biases or “distribution leakage” similar to those in the evaluated models. Did the author perform any manual verification?
- The proposed method is built upon Gemini as the base model, which poses certain challenges for reproducibility. Have the authors attempted experiments based on open-source models?

### Questions
1. I’m a bit confused here — Section 3.2 mentions *“415 trajectories”* (page 7), whereas Section 2.1 and several descriptions in the appendix together account for only *371 scenes* (pages 3 and 17).
2. Have the authors experimented with other benchmarks, such as AndroidWorld?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper targets the challenge of **cumulative error and contextual drift** in MLLM-based GUI agents, which leads to a dramatic performance drop as task complexity and step count increase. The authors propose a unified, highly robust framework named **LongHorizonUI**, structured around three key, complementary modules:

1.  **Multimodal Enhanced Perceiver (MEP)**: Responsible for temporally consistent state representation. It integrates icon detection and Optical Character Recognition (OCR) outputs, assigning **unique, stable indices** to UI elements to disambiguate and enhance subsequent processing.
2.  **Deep Reflection Decider (DRD)**: Implements structured, multi-level **closed-loop reasoning** via constrained, formatted prompts. This mechanism enforces explicit validation of **historical coherence, goal relevance, and action justification**, ensuring highly precise decision-making along non-trivial trajectories. It includes a **pre-execution reflection step** to proactively validate action rationality.
3.  **Compensatory Action Executor (CAE)**: Provides a powerful layer of fault tolerance. It utilizes a multi-level **fallback strategy**—leveraging element indices, relative layout priors, and absolute coordinates—to continuously monitor execution. Upon detecting performance degradation, the CAE applies targeted compensation or triggers **rollback procedures** to maintain robustness throughout the entire long-horizon task.

To facilitate rigorous evaluation, the paper introduces **LongGUIBench**, a novel benchmark comprising complex general applications and gaming scenarios that require **over 15 steps** (averaging 22.1 steps). Experiments clearly demonstrate that **LongHorizonUI** substantially improves long-horizon performance on **LongGUIBench** while maintaining competitive performance across diverse public benchmarks.

### Strengths
The paper addresses the **most significant limitation** of MLLM-based GUI agents: their brittleness and poor performance in complex, multi-step scenarios. The solution provided is highly relevant to real-world deployment.  It presents **LongGUIBench** is a **major asset** to the field. By focusing on **tasks requiring over 15 steps** and including both general and gaming applications, it provides a far more realistic and challenging testbed for long-horizon capabilities than existing short-horizon benchmarks. The high-quality data collection methodology further assures its utility.

### Weaknesses
1. We note the paper's emphasis on achieving long-horizon GUI automation. However, the definition and evaluation of "long-horizon" in this work need stronger justification, as recent concurrent works already include evaluation on tasks with comparable or even longer step sequences in highly dynamic environments. This diminishes the claimed novelty of LongGUIBench purely on the basis of task length and requires a clearer delineation of the unique challenges posed by this new benchmark beyond just step count.
2. The use of Google's **Gemini-2.5 Pro** as the reasoning backbone is critical, yet the discussion lacks depth regarding its specific impact. A key missing element is an analysis of how the framework's performance and robustness are affected by switching to **faster, potentially less capable MLLMs** (e.g., open-source alternatives or smaller commercial models). This is essential for understanding the framework's practical cost and deployment flexibility.
2.   The paper acknowledges the inherent **latency** in MLLM-dependent pipelines but fails to provide **specific quantitative data** on runtime efficiency, inference time per step, or resource consumption in the main results. In long-horizon automation, accumulated latency significantly affects user experience and real-world applicability. This omission limits a comprehensive evaluation of the framework's practicality.
3.  While **LongGUIBench** is excellent, its limitations should be discussed more explicitly. Despite the inclusion of diverse scenarios, the task count (371 total) is still small relative to real-world complexity. The paper should address its coverage of extreme challenges such as **multi-lingual interfaces, highly dynamic content loading (e.g., infinite scrolling),** or **inter-application contextual reasoning**. In Osworld and WinArena benchmarks, whether its performance promising?
4.   The introduction contrasts this work with existing online Reinforcement Learning (RL) methods, highlighting their challenges with expanding action spaces and cumulative errors. Given that **LongHorizonUI** features internal feedback (DRD) and error correction (CAE), a **more detailed, fundamental comparison** of the two long-horizon paradigms (RL vs. Structured Reflection/Compensation) would significantly benefit the reader in positioning this work.
5.   The framework heavily relies on strictly defined JSON Schema and detailed prompt templates (Figures 9, 11, 12). However, the paper lacks detailed discussion on the prompt design process, a **sensitivity analysis** showing how minor prompt variations affect the DRD's reflection quality, and the methods used to ensure the MLLM adheres consistently to the desired structure. Enhancing transparency here is crucial for reproducibility.

### Questions
1.  Please provide **end-to-end inference latency data** for **LongHorizonUI**. What are the average inference time per step and its impact on the total task completion time, especially for the longest tasks in **LongGUIBench**?
2.  Quantify the failure rate of the pre-execution reflection mechanism in the **Deep Reflection Decider (DRD)**—i.e., the rejection rate of actions (when $\phi(s_t, a | G_t, T) = 0$). What are the **most common root causes** for these rejections (e.g., element absence, action semantic mismatch)?
3.  Could you elaborate on the precise implementation of the **rollback mechanism** in the **Compensatory Action Executor (CAE)** (Algorithm 1, line 14)? Does "last committed snapshot $(s_{t-1}, p_{t-1})$" always revert to the immediately preceding successful step, or is there a strategy to revert to a much **earlier, highly stable state**? What is the associated cost of a rollback?
4.  For the General-High and Game-High tasks, what specifically defines these "**High-Level**" instructions, and how does the **DRD** facilitate the breakdown of these abstract goals into a sequence of "Low-Level" atomic operations?
5.  Provide more details on the **fallback template matcher** in the **Multimodal Enhanced Perceiver (MEP)** (lines 207-208). How is the template library **T** constructed, what types of critical elements does it include, and what are the specific **activation criteria** (e.g., low-confidence score, or no element detection in a high-priority area)?
6.  Regarding future extensions of **LongGUIBench**, are there plans to incorporate challenges related to **multi-lingual GUIs**, **complex data transfer across applications**, or scenarios requiring external knowledge retrieval?
7. The paper also has several minor and typos.

### Soundness
3

### Presentation
3

### Contribution
2
