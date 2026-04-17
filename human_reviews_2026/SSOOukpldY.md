# OS-MAP: How Far Can Computer-Using Agents Go in Breadth and Depth?

- Decision: Reject
- Scores: 4, 10, 4, 4

## Abstract
Computer-using agents have shown strong potential to boost human productivity and enable new application forms across platforms. While recent advances have led to usable applications, existing benchmarks fail to account for the internal task heterogeneity and the corresponding agent capabilities, as well as their alignment with actual user demands—hindering both targeted capability development and the reliable transition of research progress into practical deployment. To bridge the gap, we present OS-Map, a benchmark for daily computer-using automation, consisting of 15 applications and 416 realistic tasks. To enable fine-grained analysis of required capabilities and alignment with real-world scenarios, OS-Map evaluates agents along two dimensions: automation level across a five-level taxonomy, and generalization scope across a demand hierarchy. This design captures varying levels of required agent autonomy and generalization, forming a performance–generalization evaluation matrix for structured and comprehensive assessment. Experiments show that even the strongest agents struggle with higher-level tasks involving perception, reasoning, and coordination—highlighting the need for deeper understanding of current strengths and limitations to drive the future progress in computer-using agents research and deployment. All code, environments, baselines, and data are publicly available at https://anonymous.4open.science/r/OSMap-C2F5/.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces OS-MAP, a new benchmark for evaluating computer-using agents. The authors argue that existing benchmarks fail to capture the true diversity of real-world tasks and treat them as a "flat" collection, which prevents the measuring of critical agent capabilities like autonomy and generalization, or aligning tasks with actual user demands.

The primary contribution is an OS agent evaluation benchmark consisting of 416 tasks across 15 applications, and the novel two-dimensional framework used to organize it. This framework evaluates agents along two axes: Automation Level and Generalization Scope. 

In experiments section, the authors evaluate a range of agents and showing all current agents perform poorly, achieving only an 11.5% overall success rate and near-zero performance on higher-level automation tasks (L3/L4).

### Strengths
This paper makes a valuable contribution to existing agent benchmarks, such as OSWorld, by enriching the set of tasks and carefully reorganizing and refining the benchmark categories. These additions broaden the benchmark’s coverage and help establish a more comprehensive evaluation framework for agent capabilities.

The experimental evaluation is solid and thorough. The authors provide both code and data, demonstrating strong reproducibility and completeness of the work.

Overall, the paper’s core ideas are clearly articulated, and the presentation is well structured, making the contributions easy to understand.

### Weaknesses
While the paper presents meaningful contributions, there are a few areas where it could be strengthened.

First, Appendix C contains important details about task curation, but these details currently sit outside the main narrative. Since task design is a core component of the benchmark, I would encourage the authors to move at least some of this content into the main paper to better motivate and justify the task set.

The proposed difficulty-level formulation also feels only partially novel. Similar ideas have been discussed in prior work, such as VideoWebArena (2024), which also proposes difficulty-based categorization.

Regarding Figure 1, the mapping of “proactivity” and “general” as higher-value areas is not entirely convincing. Basic execution tasks are equally important for evaluating agent capabilities. These axes might be better positioned as indicating “more challenging regions” rather than inherently “more valuable.”

I also found the proposed automation-level taxonomy unconvincing. Although I see the motivation, the categories don’t fully capture the range of challenges GUI agents face. Beyond task ambiguity, issues such as long-horizon execution, UI complexity, tool availability also significantly affect difficulty. For example, in Figure 3, I would not consider “download and use today’s Bing wallpaper” and “download and rotate the Bing wallpaper every day” to represent distinct automation levels. The distinction feels minor. The authors may want to reconsider this taxonomy, especially rethinking about adapting from SAE’s driving automation taxonomy as OS agent and driving are naturally different.

### Questions
see above

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
10

### Rating Number
10

### Confidence
3

### Summary
This paper introduces OS-MAP, a benchmark with 416 real-world computer-use tasks on 15 Ubuntu applications. Different from existing benchmarks like OSWorld, this benchmark's captures two orthogonal dimensions: automation level and generalization scope. The paper then performs a detailed set of experiments on a set of models and agents and found them to significantly underperform human.

### Strengths
- This paper is extremely well written
- It provides a very comprehensive benchmark, covering a good range of variety on multiple dimensions
- The benchmark enables detailed analysis of agent performance and failure cases
- It conducts a very thorough evaluation of popular models and agents and found interesting results.

### Weaknesses
None. I really like this paper

### Questions
None. It's very clearly written.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a systematic benchmark for evaluating computer-operated intelligent agents—OS-MAP—aimed at comprehensively characterizing the automation capabilities and generalization levels of agents in real desktop environments. The authors construct a high-fidelity interactive environment based on a virtual machine, incorporating 416 tasks covering 15 common application scenarios, and propose a two-dimensional evaluation framework: five levels of automation and three levels of generalization. Under this benchmark, the authors systematically evaluate several mainstream models, and the results show that the overall success rate of the most powerful agents is currently far lower than that of human agents. The paper further reveals the key bottlenecks of agents through hierarchical failure analysis, providing a clear development path and a scalable evaluation framework for future research.

### Strengths
1. **Unified capability framework.** The paper proposes an original two-axis evaluation scheme—automation level × generalization scope —that provides a structured view of agent competence and offers a clear roadmap for future development.

2. **High-fidelity and reproducible benchmark.** OS-MAP builds on a virtualized desktop environment that spans realistic domains such as office work, study, and system management. The setup ensures controlled reproducibility while remaining extensible to new applications.

3. **Comprehensive experimentation and analysis.** The study benchmarks both proprietary and open-source models, presenting quantitative comparisons and qualitative failure analyses. The results convincingly highlight the main obstacles faced by current CUAs in higher-level (L3/L4) tasks, such as long-horizon planning and environmental adaptation.

### Weaknesses
1. **Limited evaluation metrics.** The benchmark currently uses binary task-success scores (0/1), ignoring partial progress, efficiency, or robustness. Incorporating richer quantitative indicators—e.g., partial completion ratio, average steps per success, or recovery rate—would provide a more nuanced evaluation.

2. **High manual cost of task construction.** Although OS-MAP follows a standardized six-stage pipeline, creating and validating new tasks still requires substantial human effort and reverse engineering. This limits scalability if the benchmark is to expand beyond 400 + tasks.

3. **Single-round task execution.** Current tasks evaluate one-shot command execution rather than continuous interaction or lifelong learning.As a result, the benchmark cannot yet assess persistent collaboration or long-term adaptation capabilities.

4. **Bias toward GUI-level operation tasks.** Most tasks emphasize interface manipulation rather than semantic-level reasoning or information synthesis. Extending OS-MAP to include tasks that require multi-step reasoning or cross-application content understanding would strengthen its coverage.

### Questions
1. **L5 proactive tasks.** The paper defines a “Proactive Companion” (L5) level but does not include any concrete L5 tasks or evaluation criteria. Do the authors plan to incorporate proactive or context-aware tasks in future versions of OS-MAP?

1. **Task-generation automation.** Given the manual workload, have the authors explored using LLMs or GUI-recording tools to automatically generate task descriptions and initialization scripts?

1. **Cross-platform generalization.** All current experiments are conducted on Ubuntu. Have the authors evaluated portability to other operating systems? A cross-platform benchmark would improve external validity and demonstrate the robustness of the environment design.

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
3

### Summary
This paper introduces OS-MAP, a new benchmark designed to evaluate computer-using agents along two principal dimensions: breadth (generalization scope across user demands) and depth (automation level). The authors argue that existing benchmarks treat tasks as a flat collection, failing to capture the heterogeneity of real-world computer use and hindering progress towards practical deployment.

### Strengths
- The core contribution—the two-dimensional evaluation matrix of Automation Level vs. Generalization Scope—is a step forward for agent evaluation. It provides a structured, principled way to move beyond simple aggregate success rates. 

- The "Generalization Scope" dimension is based on a well-researched user demand hierarchy, adapting mobile and desktop usage statistics. This demand-driven approach to task curation ensures that the benchmark's relevance is tied to practical utility, a crucial aspect often overlooked by benchmarks organized solely around applications or synthetic task templates.

-The paper evaluates a diverse set of modern agents, providing a comprehensive snapshot of the current state-of-the-art. 

- The authors are releasing the benchmark, environment, and baselines. I took a look at the code and think it is valuable for commnuaity.

### Weaknesses
- The Conflation of Task Complexity and Agent Autonomy in the "L-Levels" is some what misleading. The framework's "Automation Levels" are inspired by the SAE levels for driving, but there's a crucial difference. In driving, the core task ("go from point A to point B safely") is constant. The levels define the division of labor and the operational design domain. In OS-MAP, the tasks themselves become fundamentally more complex at higher L-levels. L1 involves atomic actions, while L4 involves multi-step, multi-app workflows. This conflates two different things: 

  - Task Complexity: The inherent difficulty, length, and abstractness of the user's goal.

  - Agent Autonomy: The agent's ability to operate without human intervention or guidance.

- The "Ladder" Fallacy: Are Automation Levels Truly Sequential?
The framework implies a progression: an agent must master execution (L1) before planning (L2), and planning before adaptation (L3). This "ladder" model doesn't capture the reality that these are often parallel, and sometimes orthogonal, skills. 

- I do think the benefit of this benchmark but I does not agree on the eval framework.

### Questions
1. Could you elaborate on the process for assigning tasks to the automation levels L1-L4? Was there a formal rubric, and did you measure inter-annotator agreement to ensure consistency in the labeling? For example, the "Rotate Wallpapers" task is labeled L4, which involves downloading, setting a wallpaper, and configuring a cron job. What specific aspects make this L4 (Global Conductor) rather than a complex L3 (Adaptive Agent) task?

2. How do you envision quantitatively measuring an agent's performance on the S-axis (Generalization Scope)? Could you propose a metric based on the OS-MAP results to classify an agent as S1, S2, or S3?

### Soundness
3

### Presentation
3

### Contribution
2
