# AgentVQA: A Unified Benchmark for Agentic Visual Understanding

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
Vision-language models (VLMs) can perform a broad range of tasks across diverse settings. Yet their performance in agentic contexts remains poorly understood. Existing benchmarks are domain-specific, making comprehensive evaluation difficult, and they often require compute-expensive online simulators. To address this gap, we introduce AgentVQA, a benchmark for systematically evaluating agentic capabilities in VLMs. AgentVQA offers three key advantages: (1) $\textit{Comprehensive}$ – it consists of 14 datasets spanning five critical agentic domains: Web Agents, Robotics, Egocentric Videos, Games, and Spatial Understanding. (2) $\textit{Standardized}$ – we reformulate diverse tasks, like trajectory-based web navigation and gameplay, into a unified multiple-choice question (MCQ) format. We balance the sample distribution across multiple domains, data formats, and semantic categories. (3) $\textit{Challenging}$ – our data processing pipeline generates hard negative options in MCQs, which are then manually reviewed for correctness. Among all the models we evaluate, the best achieves a mere $\sim$60\% accuracy. Furthermore, our ablation studies highlight key error modes where current VLMs can be improved.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces AgentVQA, a benchmark designed to evaluate the “agentic” capabilities of VLMs across five domains. It aggregates 14 existing datasets into a unified format, generating “hard negatives” to increase task difficulty. The authors evaluate 15 VLMs, report that top models achieve around 60% accuracy, and show that rankings on AgentVQA differ from general-purpose VQA benchmarks. They claim this indicates the benchmark better reflects agentic reasoning and decision-making ability. However, the paper primarily focuses on dataset aggregation and evaluation without introducing new methods or clear theoretical grounding.

### Strengths
1. The paper shows inconsistent ranking correlation with prior VQA/agentic benchmarks, which is a faithful diagnostic.
2. The paper labels common failure modes by domain, giving readers a quick, if high-level, picture of where models stumble.
3. The paper writing is clear and easy to follow.

### Weaknesses
1. **Undefined core concept**. The paper introduces “Agentic VLM” as its central theme but never defines what constitutes agentic capability in vision-language models. Without a clear conceptual framework distinguishing agentic from non-agentic reasoning, the entire benchmark lacks grounding.
2. **Flawed motivation**. The claimed limitation of prior benchmarks (“cannot evaluate across tasks”) is not substantiated. The authors fail to justify why combining existing datasets under one umbrella provides deeper insight than treating each as a subtask. Without theoretical or empirical reasoning showing why this integration is necessary, the benchmark is trying to solve a problem that is never convincingly established.
3. **Lack of technical contribution**. The work contributes no new algorithmic method, model, or evaluation paradigm. Its data aggregation and annotation process follows existing standards (e.g., MCQ conversion, hard-negative sampling). Such engineering consolidation does not represent a research innovation expected of an ICLR contribution.
4. **Uninterpreted rank correlation**. The reported low rank correlation with existing benchmarks (Sec. 4.3) is presented as a result, but its meaning is not analyzed. It is unclear whether this divergence indicates that AgentVQA captures a new ability, or simply that it is noisier or inconsistent.
5. **Predictable findings**. The analyses (Sec. 4.4) merely confirm known limitations of VLMs (e.g., strong on descriptive tasks, weak on planning). These results replicate observations from prior benchmarks without introducing new insight or diagnostic understanding of agentic behavior.
6. **No implication or discussion**. The paper lacks discussion on how its findings inform future research or model design. It never articulates how AgentVQA could guide the development of agentic VLMs or embodied AI systems. Consequently, the benchmark’s practical and scientific value is unclear.

### Questions
Why were these particular 14 datasets chosen, and how do they collectively represent the spectrum of agentic capabilities?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces AgentVQA, a unified benchmark designed to evaluate the agentic capabilities of general-purpose vision-language models (VLMs). The benchmark curates data from 14 existing datasets into unified multiple-choice questions spanning five domains, including web agents, egocentric videos, robotics, games, and spatial understanding. These questions include both visual understanding questions as well as questions involve actions, such as selecting, evaluating, or explaining actions. The authors comprehensively evaluate a range of closed- and open-source VLMs and conduct detailed analyses of their behaviors. Results show that model rankings on AgentVQA have low correlation with existing benchmarks, suggesting that it could potentially capture distinct aspects of agentic visual understanding not covered by existing benchmarks.

### Strengths
1. The paper tackles a crucial and under-explored question. A unified benchmark could be valuable for advancing research on general-purpose agentic AI with VLMs, especially if it can accurately profile models' performance in actual agentic AI tasks.
2. The benchmark integrates multiple datasets across diverse domains and evaluates a wide range of closed- and open-source VLMs. It shows the significant effort from authors in data curation, standardization, and large-scale evaluation.
3. The paper provides detailed evaluations and analysis, especially error analyses, that yield insights into model's behavior and performance.

### Weaknesses
1. While the benchmark spans multiple datasets and task types, it remains unclear whether it adequately captures the full range of capabilities required for agentic AI. Although the authors provide partial justification and a taxonomy in the appendix, a clearer and more principled taxonomy of agentic skills is still missing.
2. The finding that model rankings are uncorrelated with existing benchmarks is interesting, but this alone does not validate that AgentVQA accurately measures agentic capabilities. Ideally, the benchmark should demonstrate correlation with models’ actual performance in interactive environments. 
3. The robotics portion of the benchmark focuses mainly on perception and omits other essential robot capabilities such as task planning, motion planning, or low-level control. 
4. Only a subset of annotations was manually verified. To establish confidence in the automated annotation process, the authors should report the accuracy of the automatically generated labels within the verified subset.

### Questions
1. The benchmark assumes that each multiple-choice question has a single correct answer. Does this assumption hold across such a diverse set of tasks? For instance, spatial relation questions may have multiple valid answers (e.g., both “in front of” and “left of”), and action-selection tasks may have several equally optimal actions.
2. Figure 3 shows that the proportions of different question types are uneven (e.g., tap, world modeling, and functional reasoning occupy noticeably larger shares). How was this distribution determined? Does it reflect the task distribution in real agentic tasks?
3. Although the questions are reformulated into a unified multiple-choice format, is it meaningful to directly compare VQA accuracy across domains? Since negative options are synthesized and may vary in quality or difficulty across domains, the resulting accuracies could reflect domain-specific distractor design rather than true differences in agentic capability. Cross-domain comparisons would make sense to me if based on success rates in completing actual agentic tasks.

### Soundness
2

### Presentation
2

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
This paper introduces AgentVQA, a unified benchmark designed to evaluate agentic visual question answering capabilities in embodied agents. The benchmark reformulates multiple existing datasets across different domains into a standardized multiple-choice format. This allows consistent evaluation of a wide range of vision-language models (VLMs) under a common framework. It also introduces hard negative options to make the task more challenging. The authors provide a comprehensive comparison across various models and point out several error modes in existing VLMs which can be utilized for future improvement.

### Strengths
1. Clear presentation and well-written paper. The paper is clearly structured and easy to follow. The motivation, dataset construction process, and experimental setup are all described in a logical and organized manner, making it easy to read and follow.
2. Comprehensive baseline coverage. The experiments include a wide range of baselines with different model sizes, covering both open-source and closed-source models, as well as instruction-tuned and reasoning-focused variants. This provides a fairly complete picture of current model capabilities within the proposed setting.
3. Convenient and efficient agentic evaluation. The multiple-choice reformulation, while simplifying the original tasks, offers a standardized and more efficient way to evaluate embodied reasoning. It allows for easier comparison across models and faster benchmarking, which could be useful for large-scale or frequent evaluations.

### Weaknesses
1. Limited contributions. The paper tries to bring together a number of embodied and vision-language tasks under a single benchmark, but the motivation for doing so isn’t entirely convincing. These tasks rely on very different capabilities, so combining them feels somewhat arbitrary rather than conceptually unified. Also, many of the included tasks are still challenging on their own, so the main difficulty doesn’t seem to come from the new benchmark design (e.g., adding hard negatives). The experimental results also don’t offer much new insight beyond listing model performances.
2. Questionable motivation for multiple-choice reformulation. Converting all tasks into multiple-choice questions makes evaluation more consistent and helps with negative generation, but it moves quite far away from how these tasks appear in realistic settings. For agent-based tasks, actions and reasoning are open-ended, so forcing everything into an MCQ format feels unnatural. For example, turning a GUI grounding or navigation problem into a set of options doesn’t reflect how the agent would actually perform the task. It’s not clear how progress on this reformulated benchmark would translate to better real-world performance.
3. Lack of analysis on the effects of transformation. The paper would benefit from a deeper look at how these standardization steps—especially the MCQ conversion and hard negative creation—affect the original benchmarks. For instance, comparing model behavior or accuracy between the original and transformed settings would help show whether the new format preserves the task’s difficulty or changes it. Right now, the results mainly show rankings of existing models, but don’t make it clear what unique strengths or insights this benchmark provides compared to the original ones.

### Questions
1. The multiple-choice setup may introduce non-negligible variance, since simply changing the order of options can sometimes affect model performance. In addition, the MCQ format inherently allows for a certain level of random guessing, which could inflate accuracy and obscure meaningful differences between models. Have you considered these issues in your benchmark design or evaluation protocol? 
2. It’s unclear how performance on your benchmark relates to the original versions of the tasks. Are the model rankings consistent between your unified benchmark and the source benchmarks? If they are aligned, what is the additional value of introducing this reformulation? If they diverge, how should we interpret the difference—does it suggest your benchmark captures something new, or does it distort the original task difficulty? Some analysis or discussion on this point would help clarify the practical meaning of the reported results.

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
3

### Summary
This paper introduces AgentVQA, a unified offline benchmark to evaluate agentic capabilities in VLMs. The benchmark aggregates 14 datasets across five domains, and converting them into a standardized multiple‑choice format with hard negatives generated by a VLM-assisted pipeline and then manually verified on subsets. In total, AgentVQA contains 13,400 MCQs drawn from 18,400 images and 2,000 videos, organized into 25 sub‑task categories (e.g., tap/press/typing, spatial navigation, reward modeling). Evaluation of 15 open‑ and closed‑source VLMs shows the best model (GPT-5 thinking-high) reaches only ~60% overall accuracy, with clear domain variance. The paper further reports several ablation studies and findings, including (1) an ablation showing that action history provides substantial benefits for web agent tasks, (2) a sample‑size robustness study justifying the choice of ~1,000 examples per source dataset, (3) identification “thinking loops” as a failure mode in some reasoning models, and (4) mode analysis revealing domain-specific patterns.

### Strengths
1. Comprehensive Benchmark Design: AgentVQA introduces a unified, cross-domain benchmark specifically targeting agentic visual reasoning. By aggregating 14 datasets across five domains (Web Agents, Robotics, Egocentric Videos, Games, and Spatial Understanding) into a standardized multiple-choice format, it systematically evaluates skills for agentic tasks and addresses the fragmentation of existing domain-specific benchmarks.

2. Rigorous Evaluation Methodology: The benchmark employs a VLM-assisted and human-verified hard-negative generation pipeline that produces challenging distractors testing both fine-grained perception and high-level reasoning. The paper provides thorough evaluation of 15 major VLMs under a unified protocol, with detailed per-domain and per-category breakdowns, comprehensive ablation studies (e.g., effect of action history, reasoning tokens), and error-mode analyses revealing model-specific failure patterns (e.g., grounding vs. reasoning failures).

### Weaknesses
1. Unclear Definition of "Agentic" Tasks: The paper frames AgentVQA as a benchmark specifically designed to evaluate agentic capabilities of VLMs, yet many sub-categories (e.g., Object Grounding, Object Counting, Entity Detection, Object Sizing) are standard VLM tasks that do not clearly require agentic reasoning. The authors should more clearly define what distinguishes an "agentic" task from conventional visual understanding and explain how each category specifically tests decision-making, planning, or interactive reasoning capabilities.

2. Limited Connection to Online Agent Performance: While the authors motivate offline evaluation well, MCQs may inadequately capture real agentic interaction challenges such as handling latency, exploration, tool use, and error recovery. The paper acknowledges this tension but does not empirically quantify the correlation between success on AgentVQA MCQs and actual online agent performance, leaving unclear how predictive this benchmark is of real-world agentic capabilities.

### Questions
1. Hard-Negative Generation Details: The author should include more detail on the Hard-negative generation pipelines, for example, What fraction of negatives are near‑miss vs. semantic? 

2. The author should also include more details in the evaluation procedure. For each model, how many items were skipped due to non‑conforming outputs? Please provide per‑model skip counts and whether excluding them changes rankings

### Soundness
3

### Presentation
3

### Contribution
2
