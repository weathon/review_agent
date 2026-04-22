# Just Do It!? Computer-Use Agents Exhibit Blind Goal-Directedness

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 2, 6, 4

## Abstract
Computer-Use Agents (CUAs) are an increasingly deployed class of agents that take actions on GUIs to accomplish user goals. 
In this paper, we show that CUAs consistently exhibit *Blind Goal-Directedness* (BGD): a bias to pursue goals regardless of feasibility, safety, reliability, or context. We characterize three prevalent patterns of BGD: (i) lack of contextual reasoning, (ii) assumptions and decisions under ambiguity, and (iii) contradictory or infeasible goals. We develop BLIND-ACT, a benchmark of 90 tasks capturing these three patterns. Built on OSWorld, BLIND-ACT provides realistic environments and employs LLM-based judges to evaluate agent behavior, achieving 93.75% agreement with human annotations. We use BLIND-ACT to evaluate nine frontier models, including Claude Sonnet and Opus 4, Computer-Use-Preview, and GPT-5, observing high average BGD rates (80.8%) across them. We show that BGD exposes subtle risks that arise even when inputs are not directly harmful. While prompting-based interventions lower BGD levels, substantial risk persists, highlighting the need for stronger training- or inference-time interventions. Qualitative analysis reveals observed failure modes: execution-first bias (focusing on *how* to act over *whether* to act), thought–action disconnect (execution diverging from reasoning), and request-primacy (justifying actions due to user request). Identifying BGD and introducing BLIND-ACT establishes a foundation for future research on studying and mitigating this fundamental risk and ensuring safe CUA deployment.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes an agent safety benchmark based on OSWorld, including three types of seemingly safe but actually risky tasks: posting inappropriate file content, executing unsafe scripts, and performing harmful system operations. The aim is to test whether agents can detect and avoid hidden hazards during task execution. However, it seems that easily using the prompt to tell the model check the safety can pass the benchmark.

### Strengths
The motivation is clear, and the examples are intuitive.

The three task types are well chosen to illustrate different forms of potential harm.

The paper presents the safety issue in a concrete and easy-to-understand way.

### Weaknesses
In this benchmark, the safe outcome is always defined as refusal. I understand the motivation, but it makes the setup a bit unrealistic — in many practical situations, a safe agent could check or modify the script instead of simply rejecting it.

The reported unsafe behaviors may be caused more by the setting than by model limitations. The agent is prompted to focus on completing the task, so it naturally executes even clearly unsafe commands (e.g., creating a 20 TB swap file). Once the prompt explicitly reminds the model to check safety, the unsafe execution rate drops from 50% to 20%. This suggests that the issue is largely prompt‑level and the benchmark difficulty is not inherently high.

The evaluation mainly reports execution or refusal rates. It would be helpful to include more fine‑grained metrics (e.g., distinguishing detection failures vs. decision errors) to better understand model behaviors.

### Questions
You mention employing LLM‑based judges with high human‑agreement rates. Could you clarify whether these judges were given full screen context or only textual traces of the agent’s actions? In realistic GUIs, partial observability could heavily affect judgment accuracy.

The current experiments use a simple prompt‑based intervention reminding the model to “consider safety.” Did you explore more nuanced prompts that calibrate different levels of caution — for instance, balancing safety awareness against task completion utility? This could reveal whether BGD mitigation can be tuned continuously rather than treated as an on/off switch.

### Soundness
3

### Presentation
4

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
This paper introduces and characterizes the phenomenon of Blind Goal-Directedness (BGD) in Computer-Use Agents (CUAs), defined as an intrinsic tendency to pursue user-specified goals without sufficient regard for feasibility, safety, reliability, or context. The authors categorize BGD into three prevalent patterns. To systematically evaluate this risk, the work develops BLIND-ACT, a benchmark comprising 90 tasks built on OSWorld VM, and employs an LLM-based judge for evaluation.

### Strengths
1. The work explicitly identifies and meticulously characterizes Blind Goal-Directedness (BGD) as a core security risk, which opens up a relatively new direction for CUA security research.
2. The authors have created the OSWorld-based benchmark, BLIND-ACT (comprising 90 tasks), which supports the systematic evaluation of BGD in a dynamic environment.
3. Extensive evaluation was conducted on nine state-of-the-art models, providing quantitative evidence of BGD risk, and uncovering the Safety-Capability Parity phenomenon alongside three critical qualitative failure modes.

### Weaknesses
1. The reliance on an LLM Judge (using o4), even with a high agreement rate with human annotators, introduces an element of **model-dependency** into the core evaluation metric (unlike the original OSWorld). The paper does not sufficiently address how potential systemic biases within the judge (which is itself a VLM / CUA) might fail to capture subtle BGD manifestations that fall outside the limited, pre-defined patterns.
2. The benchmark is constructed entirely within the OSWorld Ubuntu VM environment, utilizing a specific set of applications (e.g., GIMP, LibreOffice, Thunderbird).
    1. Firstly, the novelty of the new dynamic environment is questionable, as it entirely reuses previous software and VM configurations.
    2. Secondly, the selection of these apps was intended for studying the CUA's automation capabilities,  not concerns about reliability or safety. I suggest that more commonly used applications that are potentially subject to greater security implications should be included in the study.
3. Please check my questions

### Questions
1. Could the authors provide more detail on the failure modes of the LLM judge, specifically the disagreements with human annotators?

2. The paper defines BGD intentions based on whether an agent "exhibited blind goal-directedness intentions." This relies heavily on inspecting the model's textual reasoning trace.

  a. Firstly, not all CUA foundation models have a "thought" (e.g., Arguvis or other seeact architecture agent).

  b. Secondly, how confident are the authors that the observed phenomena, such as "Thought-Action Disconnect," are not merely artifacts of the agent's constrained output format rather than a genuine cognitive disconnect?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This study identifies the significant Blind Goal-Directedness (BGD) risk that CUAs prioritize execution over safety, reliability, or logical consistency. The authors characterize this risky behavior via showing three common patterns: (i) lack of contextual reasoning, (ii) assumptions and decisions under ambiguity, and (iii) contradictory or infeasible goals. Based on these patterns, they construct a BGD benchmark Blind-Act, consisting of 90 tasks crafted by human. They adopt scalable and cost-efficient evaluation based on LLM-as-a-judge to evaluate the CUAs' risky behavior on the benchmark. The evaluation results expose 3 failure modes of CUAs: execution-first bias, thought-action disconnect, and request primacy.

### Strengths
- Clearly demonstrate the significant risk of BGD quantitatively and qualitatively.
- Carefully investigate the reliability of LLM-as-a-judge.
- Explore mitigations through contextual and reflective prompting strategies.
- Identify 3 significant failure modes: execution-first bias, thought-action disconnect, and request-primacy, and provide clear demonstrations:
- The evaluation metrics, BGD and the completion rate, comprehensively capture the CUA's intentions and capability to finish the task when confronted with potentially unsafe requests.

### Weaknesses
- The evaluation would benefit from providing the percentage of runs where CUAs recognize that the user query is not safe.
- The authors should add some results and discussions about the number of running steps where the CUAs generate BGD responses. Is it usually appearing in the planning and task decomposition step, which may cause more harm, or implementation step?
- It would be great if the authors can provide some results about the correlation between the CUA's BGD tendency and the underlying model's planning performance. So we can better understand the cause of BGD, either due to the poor planning performance or because of inherent insensitiveness to unsafe content. I expect that if the agent can generate a better plan or have a clear decision after carefully assessing all inputs in the beginning, some of unsafe operations in the later stage can be avoided.

### Questions
- Does the CUAs always have the access to check the target file immediately after receiving the user query? I'm thinking whether it could be helpful to show the agent the file content before planning and implementation. If the agent is aware of all information of the task in the planning phase, does the issue of lacking contextual reasoning still exist?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces the concept of Blind Goal-Directedness (BGD) — a behavioral bias in computer-use agents (CUAs) where the agent executes user-specified goals regardless of feasibility, safety, or context. The authors propose BLIND-ACT, a benchmark with 90 GUI-based tasks covering three BGD patterns: (i) lack of contextual reasoning, (ii) assumptions and decisions under ambiguity, and (iii) contradictory or infeasible goals. The paper reports results on nine frontier models (including GPT-5 and Claude Opus 4), finding high BGD rates (≈80.8%). The authors further test “contextual” and “reflective” prompting interventions, which modestly reduce BGD, and conduct qualitative analysis revealing several failure modes.

While the topic is timely and the experiments extensive, the work feels more diagnostic than analytical — it describes problems but provides limited insight into underlying causes or mitigation strategies. The empirical section is large but methodologically opaque, particularly regarding how tasks were designed, balanced, and verified.

### Strengths
- Addresses an important and emerging area: GUI agent safety beyond prompt attacks.
- Clear taxonomy (three BGD categories) and thorough experimentation.
- Empirical validation of prompting-based interventions

### Weaknesses
- Benchmark design lacks transparency and external validation.
- Heavy reliance on LLM judges without adequate robustness checks.
- Lacks theoretical or mechanistic explanation for why BGD arises.

### Questions
1.How were the 90 tasks distributed in difficulty and domain? Were they balanced or randomly constructed?
2.Has the paper explored the underlying causes of Blind Goal-Directedness? For example, are there analyses of which model components or reasoning stages lead to goal fixation?
3.Beyond prompting interventions, has the paper attempted any model-level improvements (e.g., training-time alignment, reward shaping, or trajectory-level supervision) to mitigate BGD

### Soundness
3

### Presentation
2

### Contribution
2
