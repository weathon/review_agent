# GUI-360° : A Comprehensive Dataset and Benchmark for Computer-Using Agents

- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
We introduce GUI-360°, a large-scale, comprehensive dataset and benchmark suite designed to advance computer-using agents (CUAs). CUAs present unique challenges and is constrained by three persistent gaps: a scarcity of real-world CUA tasks, the lack of automated collection-and-annotation pipelines for multi-modal trajectories, and the absence of a unified benchmark that jointly evaluates GUI grounding, screen parsing, and action prediction.

GUI-360° addresses these gaps with a largely automated pipeline for query sourcing, environment-template construction, task instantiation, batched execution, and LLM-driven quality filtering. The released corpus contains over 1.2M executed action steps across thousands of trajectories in popular Windows office applications, and includes full-resolution screenshots, accessibility metadata when available, instantiated goals, intermediate reasoning traces, and both successful and failed action trajectories. The dataset supports three canonical tasks, GUI grounding, screen parsing, and action prediction, and a hybrid GUI+API action space that reflects modern agent designs. Benchmarking state-of-the-art vision--language models on GUI-360° reveals substantial out-of-the-box shortcomings in grounding and action prediction; supervised fine-tuning yield significant gains.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces GUI-360°, a large-scale dataset and benchmark for computer use agents (CUAs) on desktop environments. The authors develop an automated pipeline that sources real-world user queries from help content, forums, and search logs, then uses LLM-driven task instantiation and a multi-agent execution system to collect trajectories across Word, Excel, and PowerPoint applications. The resulting dataset contains over 1.2M executed action steps across 17,189 trajectories (13,750 training, 3,439 benchmark), with 17.7M annotated UI elements, full-resolution screenshots, accessibility metadata, and reasoning traces. GUI-360° supports three tasks, GUI grounding, screen parsing, and action prediction, with a hybrid GUI+API action space. Comprehensive benchmarking reveals significant limitations in current models.

### Strengths
S1: The work introduces a largely automated collection pipeline that minimizes human annotation costs.

S2: It supports three complementary tasks (GUI grounding, screen parsing, action prediction) from a single data collection process.

S3: The combination of GUI operations and application-specific APIs reflects modern agent design practices and provides flexibility in action execution strategies.

### Weaknesses
W1: The paper exclusively evaluates models on GUI-360°-Bench, which is derived from the same data collection pipeline as the training set. This is not convincing as evidence of broader applicability. The authors should validate their approach on external benchmarks (e.g., ScreenSpot-Pro[1] and UI-Vision[2]) to demonstrate that models trained on GUI-360° generalize beyond the specific characteristics of their data-collection methodology. 

W2: Restricting to three Office applications significantly limits the generalizability of findings. Desktop environments are heterogeneous, and Office applications may not represent the full spectrum of GUI complexity (e.g., missing creative software, development tools, web browsers). Or, it would be good if the authors could include evidence of the generalization ability of the provided training data.  

W3: The 26.09% overall success rate for trajectory collection raises concerns about task selection bias and coverage. The paper does not adequately analyze why 74% of tasks fail or whether failed tasks represent systematically different characteristics.


[1] Li et al. ScreenSpot-Pro: GUI Grounding for Professional High-Resolution Computer Use. 

[2] Nayak et al. UI-Vision: A Desktop-centric GUI Benchmark for Visual Perception and Interaction.

### Questions
Q1: Can you provide a detailed analysis of failure modes in the 74% of unsuccessful trajectory collections? Are failures due to model limitations, task ambiguity, environment issues, or other factors?

Q2: Per W2, could authors include evidence that suggests that models trained on Office applications will generalize to other desktop software? Have you conducted any cross-application transfer experiments?

Q3: What is human-level performance on the provided benchmark? This would provide important context for interpreting model results.

The reviewer is willing to raise the score if the authors address most, if not all, of the questions above in the Weakness and Questions sections.

### Soundness
3

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
5

### Summary
This paper introduces **GUI-360°**, a large-scale dataset and benchmark suite for **computer-using agents (CUAs)** that operate within real **desktop environments**. The authors identify major limitations in existing work—most notably, the lack of realistic task data, automated data-collection pipelines, and unified evaluation standards that jointly assess GUI understanding, screen parsing, and action prediction.

To address these challenges, the paper proposes a **three-stage automated pipeline**:

1. **Real-world query acquisition** from authentic user sources (help documents, forums, search logs).
2. **Automated execution and trajectory collection** via a hierarchical multi-agent system called **TrajAgent**, which interacts with real desktop applications—**Microsoft Word, Excel, and PowerPoint**—while recording screenshots, accessibility (a11y) metadata, and reasoning traces.
3. **Automated evaluation and post-processing**, where LLM-based judges verify the execution quality and filter noisy samples.

The resulting dataset comprises **over 1.2 million action steps** and is organized into a **three-task benchmark suite** covering:

* **GUI Grounding** (locating target UI elements),
* **Screen Parsing** (extracting and labeling interactive components), and
* **Action Prediction** (step-wise operation generation).

Experimental results on both proprietary and open-source VLMs (e.g., GPT-4o/4.1, GPT-5, Qwen-2.5-VL, OmniParser, GUI-Actor) demonstrate that fine-tuning on GUI-360° significantly boosts model performance across all three tasks. While the benchmark currently focuses on **office-suite applications (Word, PowerPoint, Excel)**, the framework and dataset design establish a strong foundation for broader research on **desktop-level intelligent agents**.

### Strengths
* **Engineering originality & significance.** The paper proposes a *nearly fully automated* data construction pipeline—spanning **query acquisition → environment templating → task instantiation → batched execution → quality filtering**—which is a creative and practical integration of existing ideas into a scalable, end-to-end system. This design removes key bottlenecks in prior work (manual collection/annotation) and is therefore **original** in its full-stack automation and **significant** for enabling broader data generation.

* **Data quality, clarity, and breadth of supervision.** The dataset includes **screenshots, a11y metadata, intermediate reasoning traces, and both successful and failed trajectories**, enabling supervision for **three tasks**—grounding, screen parsing, and action prediction—at **large scale (>1.2M steps)**. The multimodal/structured annotations enhance **quality** (richer supervision, failure analysis) and **clarity** (well-defined inputs/outputs across tasks), while the size and composition make it **significant** as a community resource.

### Weaknesses
### **Major Point 1 — Method / Pipeline**

* **Limited methodological novelty.** The pipeline follows a now-standard pattern—mining real-world queries, mapping them to environment templates, auto-executing with an agent, and filtering via an LLM-as-a-judge. The paper does not clearly isolate what is *algorithmically new* beyond engineering integration.

* **Narrow domain scope and heavy a11y reliance limit generality.** The data and tools are confined to **Word/Excel/PowerPoint on Windows**, and perception hinges on **UI Automation (a11y)** to extract elements/bboxes. Many desktop apps (browsers with canvas-heavy UIs, email clients, media/creative tools, Electron/native hybrids) expose partial/noisy a11y trees; other OSes differ further. As written, it remains unclear whether the pipeline can generalize beyond the office suite.

### **Major Point 2 — Experiments**

The core goal of CUAs is **end-to-end task completion in real, interactive environments**. This paper’s experiments focus on **static self-benchmarking without external validation**, offering no evidence that the proposed data **improves downstream CUA capability** in live, end-to-end settings.

* **Self-benchmark only; no out-of-distribution tests.** Both **grounding** and **action prediction** are evaluated solely on the in-house **GUI-360°-Bench**, drawn from the same source/templates as training—i.e., **same-distribution self-benchmarking**. There is **no external validation** (e.g., **ScreenSpot**, **ScreenSpot-Pro**, **OSWorld-G** for grounding; **AgentNetBench**, **UI-Vision**, **Mind2Web** for action/agent behavior), so generalization claims remain unsupported. This also raises the risk of **template or near-duplicate overlap** inflating in-distribution performance.

* **Lack of dynamic, end-to-end validation.** There is **no evidence in live, interactive environments** that the SFT models (**Qwen-2.5 7B-SFT**, **UI-TARS-1.5 7B-SFT**) improve task completion. Without evaluations on **OSWorld (e.g., OSWorld-Verified)** or **Windows Agent Arena**, it remains unclear whether static gains translate to **real task success**.

### Questions
### **Main Point 1**: Missing Related Work (please cite)
Please cite and briefly position against the following closely related pipelines:
* **AgentTrek: Agent Trajectory Synthesis via Guiding Replay with Web Tutorials.** Uses **real-world tasks**, synthesizes trajectories **via agents** in live environments while **logging multiple modalities** (e.g., screenshots/DOM/AX), and **verifies** trajectories with an **LLM judge**.
* **AgentSynth: Scalable Task Generation for Generalist Computer-Use Agents.** Adopts a **multi-agent framework** with a **planner + multiple executors** for scalable task generation/execution, closely mirroring your **MAgent/EAgents** orchestration idea.
* **Explorer: Scaling Exploration-driven Web Trajectory Synthesis for Multimodal Web Agents.**

### **Main Point 2**: Pipeline extensibility beyond Office

Could you add a brief **plan and rationale** for extending the pipeline beyond Word/Excel/PowerPoint—e.g., how you would adapt templates, perception (when **a11y** is partial/absent). A short roadmap (what changes, expected blockers, and minimal proof-of-concept evaluations, including one **interactive E2E** check) would directly address the generalization concern.

### **Main Point 3**: External validation & end-to-end evaluation

Please add **external validation** and a minimal **end-to-end online evaluation**. A lightweight plan could suffice: (i) report zero-shot/SFT results on an **external grounding suite** (e.g., ScreenSpot / ScreenSpot-Pro / OSWorld-G), and (ii) eval the finetuned model on the libreoffice domain jn OSWorld or WindowsAgentArena benchmark

### Soundness
2

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
This paper introduces GUI-360°, a large-scale dataset for computer-using agents on desktop environments. The authors developed an automated pipeline that collects real-world user queries from help content, forums, and search engines. Experiments show that current vision-language models struggle with these tasks when used directly.

### Strengths
GUI-360° addresses a key gap in desktop automation research. Previous datasets were small-scale with manual annotations or focused on web/mobile platforms. This work provides the desktop dataset using real user queries from forums and documentation, ensuring tasks reflect genuine needs.

The TrajAgent framework solves the expensive manual annotation problem. Instead of requiring human demonstrations, coordinated AI agents automatically execute tasks, capture screenshots, record accessibility data, and validate results. This dramatically reduces costs while enabling larger-scale collection.

The hybrid action space combines traditional GUI interactions with application APIs, reflecting how modern automation tools actually work. This approach allows agents to leverage structured interfaces when available.

### Weaknesses
The TrajAgent framework shows promise but needs clearer technical explanation. How does the MasterAgent actually break down complex tasks? What happens when agents need to coordinate or when an ExecutionAgent fails?

The error handling and load balancing mechanisms deserve more attention. Specific examples of how the system prevents cascading failures would strengthen the contribution and help readers understand practical deployment challenges.

The task instantiation approach is clever but raises implementation questions. How does the system ensure instantiated tasks remain faithful to original user intent while being executable?

The EvaAgent validation shows good results, but the 14% disagreement cases might be informative. What task types cause validation errors? Do certain applications prove more challenging for automated assessment?

How do validation criteria adapt across domains? Excel tasks likely need different success metrics than PowerPoint presentations. More discussion of this adaptation would be useful.

### Questions
Please refer to my identified weak points for more details.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces GUI-360°, a large-scale dataset and benchmark for desktop Computer-Using Agents. It addresses key gaps in the field by providing a novel, automated pipeline for collecting real-world tasks. The pipeline sources user queries from the web, instantiates them in template environments, and uses a multi-agent LLM system to execute them. The resulting dataset includes over 1.2M action steps, counting both successful and failed in Word, Excel, and PowerPoint, complete with screenshots, accessibility metadata, and reasoning traces. The authors establish a three-part benchmark: Grounding, Parsing, Action Prediction and show that while SOTA VLMs struggle, fine-tuning on GUI-360° yields significant gains, especially when accessibility data is provided.

### Strengths
- Data Collection Methodology: The pipeline is a key strength with: 

a. Real-world Query Sourcing: Using real-world queries from forums and search logs, not just synthetic data, increases relevance.

b. Hybrid Action Space: Combining GUI actions with API calls is practical and reflects modern agent design.

c. Inclusion of Failure Cases: The large corpus of failed trajectories (1M+ steps) is highly valuable for future work on RL or learning from negative examples.

- Comprehensive Benchmark centering around GUI-360, covering grounding, parsing and planning.

### Weaknesses
- In Table 1, SS and SS-Pro are benchmarks that are usually merely considered as datasets. The authors should compare GUI-360 with other dataset works that are primarily designed for training, e.g., UGround, GUIEnv (GUICourse), SeeClick, and Aria-UI. Also, seems GUIEnv and Aria-UI are relevant works that lack discussion.
- Limited Breadth. The dataset is limited to three MS Office applications on Windows.
- LLM-as-a-Judge Validation for Benchmark Creation. The benchmark's "success" labels are AI-created by GPT-4.1, with no large-scale human review. The paper notes this judge only had 86% agreement with humans on a 100-sample spot-check. Thus, it is hard to judge whether the benchmark is as high-quality as existing ones like SS and SS-Pro. Also, the benchmark ground truth itself is also created by an AI agent. In general, this pipeline for benchmark synthesization raises concerns about the reliability, preventing the large-scale utilization of such benchmark.
- The Dataset. The authors only showed how the SFT model with the train dataset performs on the aforementioned benchmark, which is OOD, and relatively unreliable. It is unclear that with the train dataset for SFT, how the agent model can perform on other benchmarks, especially the online ones (WindowsAgentArena, OSWorld, AndroidWorld, etc.)

### Questions
Please see weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
