# A3: Android Agent Arena For Mobile GUI Agents

- Avg Score: 3.60
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 4, 6, 4

## Abstract
The advancement of Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) has catalyzed the development of autonomous AI agents. Mobile graphic user interface (GUI) agents, designed to perform tasks on mobile devices, represent a promising application of this technology. However, a significant gap persists in mobile GUI agent evaluation, where existing benchmarks predominantly rely on either static frame assessments such as AndroidControl or offline static apps such as AndroidWorld and thus fail to capture agent performance in dynamic, real-world online mobile applications. To address this gap, we present Android Agent Arena (A3), a novel evaluation system for mobile GUI agents. Unlike existing dynamic evaluation systems, A3 introduces a benchmark of 100 tasks derived from 20 widely-used, online apps across 20 distinct categories from the Google Play Store, ensuring evaluation comprehension. A3 also presents a novel "essential-state" based evaluation method that leverages MLLMs (either commercial or open-source models) as reward models to progressively verify task completion and process achievement. This automated evaluation approach significantly reduces the reliance on manual labor and coding expertise compared with traditional evaluation methods such as in AndroidWorld. Furthermore, A3 includes a toolkit and an evaluator to streamline Android device interaction and facilitate data collection from both human and agent demonstrations. The complete A3 system, including the benchmark and pipeline, will be publicly released to provide a robust foundation for future research and development in mobile GUI agents.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces **Android Agent Arena (A3)** — a new benchmark and evaluation framework for **mobile GUI agents**. It proposes an *essential-state* evaluation metric, where large multimodal models (MLLMs) such as Gemini-2.5-pro or the fine-tuned *A3RM* model act as evaluators to assess whether critical intermediate “essential states” are achieved during task execution. The benchmark includes 100 tasks across 20 online apps, and the authors release a full toolchain for agent-device interaction and trajectory evaluation.

### Strengths
1. The goal of handling dynamic, real-world app environments is timely and relevant.
2. The paper provides an open-source evaluation toolkit and reward model that could be useful for replication.

### Weaknesses
1. **Unconvincing evaluation results:**
    
    The A3 results **contradict** established model rankings from prior benchmarks. For example, *UI-TARS-1.5* and *Qwen2.5-VL* perform nearly the same here, despite large known differences in their reasoning and visual grounding capabilities. Meanwhile, *InfiGUI-R1* — a less data-rich model — inexplicably outperforms all others. These inconsistencies suggest that **A3’s evaluation method may not reflect real model competence**, but noise from MLLM judgment or benchmark artifacts.
    
2. **High inconsistency with the authoritative benchmark AndroidWorld:**
    
    The results on A3 are **highly inconsistent** with those on the authoritative benchmark *AndroidWorld*. Even though *AndroidWorld* uses what the authors describe as a “static” evaluation approach, its **rule-based validation produces convincing and reproducible results** that reliably reflect model capability. For example, *UI-Venus* achieves **49%[1]**, *GUI-Owl* **66.4%**[2]**,** and *UI-TARS-1.5* **64.2% [3]** on AndroidWorld — a ranking that aligns well with their expected strengths. However, in this paper, the relative order of these models changes dramatically. This discrepancy **casts serious doubt on the credibility and validity of A3’s evaluation methodology**, suggesting that its MLLM-based judging process may not correlate with genuine agent competence
    
3. **No human validation:**
    
    The claim that A3RM is “human-aligned” is unsubstantiated without **comparison with human evaluators.** Since A3RM is fine-tuned partially using Gemini’s evaluations, comparing A3RM against Gemini and finding alignment could simply indicate **self-consistency**, not real-world validity. 
    
4. **Speculative analysis without ablation:**
    
    The statement that “the capabilities of the base model appear to be a dominant factor” is **not experimentally verified**. Without controlled experiments isolating model size, architecture, and training data, such claims are anecdotal.
    

[1] Gu, Zhangxuan, Zhengwen Zeng, Zhenyu Xu, et al. “UI-Venus Technical Report: Building High-Performance UI Agents with RFT.” arXiv:2508.10833. Preprint, arXiv, August 14, 2025. https://doi.org/10.48550/arXiv.2508.10833.

[2] Ye, Jiabo, Xi Zhang, Haiyang Xu, et al. “Mobile-Agent-v3: Fundamental Agents for GUI Automation.” arXiv:2508.15144. Preprint, arXiv, August 21, 2025. https://doi.org/10.48550/arXiv.2508.15144.

[3] UI-TARS-1.5-7B https://seed-tars.com/1.5/

### Questions
1. Can the authors **quantitatively validate** MLLM evaluation accuracy against human annotators on a subset of tasks?
2. How do they explain the **inverted ranking** of models relative to AndroidWorld? Is there any evidence that A3 better reflects human judgment rather than being more random?
3. Could the authors report **inter-rater agreement metrics** (e.g., Cohen’s κ) between A3RM, Gemini, and humans?

### Soundness
1

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
3

### Summary
This work presents a novel benchmark on GUI agents for mobile device control, addressing the drawbacks of existing prior benchmarks. The proposed benchmark features the use of online apps, unlike other benchmarks that have focused mostly on static apps, and a novel evaluation system that can benefit from scalability and dense signals. The authors also rigorously investigate the effectiveness of the proposed evaluation system. By evaluating many state-of-the-art GUI agents on the benchmark, the authors reveal that significant challenges exist towards effective in-the-wild agents.

### Strengths
This work presents several strengths, which I detail below:
1. Evaluation system: The authors present a novel mechanism to evaluate agents in dynamic, online applications. The introduced evaluation method has two features: incorporation of critical states, use of MLLM, which I believe is highly beneficial in many cases. By using critical states, the users/agents can get more fine-grained signals upon success. Also, using MLLM allows the evaluation design to be scalable without necessitating human experts.
2. Comprehensive results: Results across diverse agents are presented, which potentially provide a meaningful reference point for future work.

### Weaknesses
Above all, while this work presents noticeable strengths, the main weakness lies in the ambiguity of the contribution compared to existing work. I list specific weaknesses/questions/suggestions regarding this work.
1. Motivation: While the authors emphasize that this work allows evaluation of the online apps, the reason behind why we need to evaluate online apps is not presented thoroughly. I am not fully convinced whether such a feature is highly demanded by the community. Gemini-2.5-pro, achieving around 55 task success rates, also reveals that this benchmark does not provide enough headroom for improvements. More elaboration on what the users can benefit from by evaluating this benchmark (as well as online apps) should be provided.
2. Specification of essential state: The detailed explanations of the process for defining essential states for each task are missing. What criteria are used to define the number and content of essential states for each task?
3. Use of essential-states for RL: While it is out of scope as a benchmark, as it is currently illustrated, the dense signal of essential-state evaluation can be useful for RL. I do think that investigating AR3M to RL and comparing with existing reward mechanisms can be a valuable study for the community, as adopting RL for GUI agents is gaining interest [1,2].
4. More analysis on the agents' behaviors: Although the authors present analysis of the agents' behavior in Appendix 5.2, the observations are already known to the community (very similar observations are already present in the work presented in the related work; for example, the lack of ability to accurately ground the intended action is widely studied [3,4]). As a reader, I want to know more about failure modes strongly correlated to the focus of this paper (e.g., the dynamic nature of online apps).

---

References

[1] “Digi-q: Learning q-value functions for training device-control agents.”

[2] “UI-TARS: Pioneering Automated GUI Interaction with Native Agents”

[3] “GPT-4V(ision) is a Generalist Web Agent, if Grounded”

[4] “GUI-Bee: Align GUI Action Grounding to Novel Environments via Autonomous Exploration”

### Questions
(Questions/suggestions are included in the above section)

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a new benchmark and evaluation framework for testing mobile GUI agents in realistic, dynamic environments. Unlike prior benchmarks that rely on static frames or offline apps, A3 includes 100 tasks across various real-world online applications from the Google Play Store, reflecting real-world user interactions. It proposes a novel “essential-state” evaluation method that uses commercial or open-source multimodal LLMs (such as Gemini or A3RM) as automated evaluators to verify task progress and completion with reduced manual effort.

### Strengths
- Evaluates existing GUI agents on a broader and more practical set of real-world applications compared to prior benchmarks.
- Proposes the essential-state evaluation technique, enabling more accurate assessment of intermediate task success where traditional functional-based approaches often fail to capture partial progress.

### Weaknesses
- Using closed-source and widely used commercial applications is not, by itself, a novel contribution. Previous benchmarks such as B-MoCA and AndroidLab have also incorporated commercial apps, and similar extensions could be made by simply adding new tasks.
- The paper does not address cases where MLLM-based success detection may be inaccurate. Furthermore, even if deterministic settings are enforced, MLLMs inherently exhibit stochasticity, meaning success judgments could vary across evaluations. It would strengthen the paper to discuss mitigation strategies for this issue.
- While the authors claim that essential-state evaluation overcomes the limitations of functional-based methods, the paper would benefit from a case study clearly illustrating scenarios where functional-based methods fail but essential-state evaluation succeeds.
- The experiments compare only zero-shot performance of publicly available GUI agents. It would be interesting to see results from BC (behavioral cloning) or RL fine-tuning on A3 tasks to evaluate learning improvements.
- The method for defining essential states is described ambiguously, and the full list of essential states for all benchmark tasks is not made publicly available.

### Questions
- Are there any copyright or licensing concerns related to using private company applications in the benchmark?
- Is there a generalizable rule or guideline for defining “essential states”? When adding new tasks to the benchmark, do these states have to be manually specified by humans using heuristics?

### Soundness
3

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
The paper proposes A3, a dynamic benchmark for mobile GUI agents built on 100 human-designed tasks from 20 popular, online Android apps spanning 20 categories (news, travel, shopping, fitness, notes, etc.). Unlike prior dynamic benchmarks that rely on open-source/offline apps (AndroidWorld, AndroidLab), A3 deliberately targets real, changing, store apps that users actually use, and therefore cannot be instrumented for function-based checking. To make such apps evaluable, the authors introduce an essential-state evaluation: each task is decomposed into several key milestones (e.g. “search X”, “open first item”, “answer author”), and a MLLM-based judge (Gemini or a distilled model A3RM) checks, via a sliding window over the trajectory, whether each essential state has been achieved. This yields both a normal SR and a finer ESAR (Essential-State Achieved Rate), which exposes partial progress on long tasks. They also release an execution toolkit (AITK) and an evaluator module. On eight mobile agents, SR is low (∼30% for the best open models; only T3A+Gemini gets ~58%), but ESAR is much higher, showing current agents often “get halfway” but fail late.

### Strengths
- Ecological validity: using real, online apps that update and may show ads/popups — exactly the scenarios offline/open-source benchmarks cannot cover. This is the main novelty claim. 
- Semantically aware evaluation: essential-state + sliding-window MLLM judging is a clearer and more reliable formulation than “judge the whole multi-step trajectory in one go,” which the authors note SPA-bench’s MLLM-style judging does only at ~80% accuracy. A3 breaks the task so the judge reasons locally. 
- Cost-aware judging: commercial MLLMs work best, but the paper actually trains A3RM (MiMo-VL-7B finetuned) to get close to Gemini-level essential-state accuracy and make the benchmark practical.

### Weaknesses
- MLLM-as-judge brittleness remains: although essential-state simplifies judging, Sec. A.5 still shows hallucination and semantic misalignment (judge thinks “send” worked, or confuses “title” vs “cost”). This means reproducibility across model versions / prompts is still an issue.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Android Agent Arena (A3), a benchmark system for evaluating mobile GUI agents. The main contributions include: (1) A benchmark of 100 tasks spanning 20 popular online mobile applications across diverse categories from the Google Play Store, (2) A novel "essential-state" evaluation methodology that decomposes tasks into verifiable intermediate milestones and uses MLLMs as automated judges, (3) A3RM, a fine-tuned 7B reward model that provides cost-effective evaluation compared to commercial APIs, and (4) An open-source pipeline (AITK) for agent execution and data collection. The benchmark addresses limitations of prior work by including dynamic, online applications (e.g., shopping, news, travel) that were previously untestable due to reliance on open-source apps. Evaluation of 8 mobile GUI agents reveals that A3 poses significant challenges, with the best agent achieving only 29% success rate under commercial MLLM evaluation.

### Strengths
1. The inclusion of online, commercial apps from diverse categories (shopping, travel, news) fills a critical gap in existing benchmarks that rely on offline/open-source apps.

2. The essential-state decomposition provides interpretable, fine-grained metrics (ESAR) beyond binary success rates, enabling better diagnosis of agent capabilities.

3. The paper's strongest point is its focus on "in-the-wild" evaluation. By using 20 popular, dynamic, online apps , A3 presents a more realistic and challenging environment than benchmarks restricted to offline or open-source apps , which cannot test robustness to dynamic content or real-world UI.

### Weaknesses
1. Despite improvements, A3RM still has ~10% error rate (89.9% F1). The paper documents persistent failure modes (visual hallucination, semantic misinterpretation) but doesn't adequately address whether this error rate is acceptable for benchmark validity. More critically, there's no analysis of whether these errors systematically favor certain agents.

2. Essential states are manually created without inter-annotator agreement metrics or user studies validating that they appropriately decompose tasks. The claim that they provide "uniform granularity" lacks empirical support.

3. The paper claims A3 is superior to SPA-bench because A3 allows for "reliable and programmatic resets" while SPA-bench suffers from "severe environmental instability and difficult resets". This is a critical claim, as SPA-bench also uses online apps. However, this claim is never substantiated. The paper provides no details on its "programmatic reset" mechanism for dynamic apps nor any direct comparison to prove A3 is more stable. Without this, A3's novelty over SPA-bench is unclear.

4. The entire framework hinges on the quality of the manually defined "essential-states". The process for defining these is opaque. The paper states they were created to have "uniform granularity", but this seems subjective. The work would be much stronger if it detailed the rubric or principles for defining these states and how consistency was maintained across 100 tasks.

5. While API costs are mentioned, there's no full cost-benefit analysis comparing manual evaluation, function-based evaluation, and MLLM-based evaluation across dimensions like accuracy, development time, and maintenance burden.

6. A3RM is trained on Gemini labels and validated against Gemini, potentially inheriting and amplifying Gemini's biases rather than providing an independent evaluation standard.

### Questions
1. How were essential states validated? What was the inter-annotator agreement? Have you conducted user studies to verify that the essential-state decompositions align with human perception of task progress?

2. What is your plan for maintaining the benchmark as apps update? Will you version-lock apps, continuously update tasks, or periodically release new benchmark versions?

3. Given the ~10% error rate for the MLLM judges, how confident can we be in the agent rankings in Table 4? Have you considered quantifying this uncertainty, perhaps by reporting a confidence interval on the SR and ESAR scores based on the evaluator's known F1 score?

4. How sensitive are results to the granularity of essential-state decomposition? What happens if you use coarser or finer decompositions?

5. What is human performance on this benchmark? This would provide crucial context for interpreting agent scores.

6. Have you measured how frequently the included apps update in ways that break tasks? What percentage of tasks remain valid after 3/6/12 months?

7. You state A3RM is "stricter" than Gemini , but it also uses a smaller context window (2 frames vs. 4). How do you disentangle these two factors? Is it possible A3RM is not "stricter" but simply less accurate due to its limited context, causing it to miss valid achievements that Gemini catches?

### Soundness
2

### Presentation
3

### Contribution
2
