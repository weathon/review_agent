# SMAN-Bench: A Cross-System Benchmark for Mobile Agents under Single- and Multi-path, Ambiguous, and Noisy Tasks

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
VLM-based mobile agents are increasingly popular due to their capabilities to interact with smartphone GUIs and XML-structured texts and to complete daily tasks. However, existing online benchmarks fail to obtain stable critical reward signals under dynamic environmental changes, and neglect the influence of noise components and interactive instructions. Offline benchmarks evaluate the agents through single-path trajectories, which stand in contrast to the inherently multi-solution characteristics of GUI tasks. To address these limitations, we introduce SMAN-Bench, a benchmark designed to evaluate agents under Single-path, Multi-path, Ambiguous, and Noisy task settings. We employ a slot-based instruction generation method to match templates with GUI trajectories from an existing, graph-structured, unlabeled mobile corpus. SMAN-Bench includes a common task split, with offline multi-path evaluation to assess the agent’s ability to obtain step rewards during task execution. It contains a noisy split based on pop-ups and ad apps, and a contaminated split named AITZ-Noise to simulate a realistic noisy environment. Furthermore, an ambiguous instruction split with preset Q&A interactions is released to evaluate the agent’s proactive interaction capabilities. Our evaluation covers mobile agent frameworks like AppAgent-v1, Mobile-Agent-v2, and Mobile-Agent-E, and includes both open-source and closed-source mobile fundamental models, as well as several multimodal reasoning models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper makes a meaningful benchmarking contribution to the field of mobile GUI agents. SMAN-Bench addresses key shortcomings in existing benchmarks by introducing evaluations under multi-path, noisy, and ambiguous conditions, which better reflect real-world user interactions on mobile devices. The experimental analysis is extensive and compares numerous models and agent frameworks, providing valuable insights into current system limitations.

### Strengths
1. Slot-based automatic instruction generation from unlabeled mobile GUI corpora is interesting.

2. The paper is well-written and easy to follow.

3. The paper conducted extensive evaluation across many mobile agent systems.

### Weaknesses
1. The noise simulation includes ads, popups, and redirections, but does not cover other common mobile noise sources such as keyboard overlays, partial screen scrolls, background notifications

2. The paper claims to support multi-path evaluation, but it is not clearly explained how all valid alternative paths are identified, annotated, or guaranteed to be complete. If the benchmark only includes a limited subset of possible valid paths, models that take an alternative (but logically valid) route may be unfairly penalized.

3. While they report success rate (SR) and step accuracy (Step.Acc), the paper does not provide deeper agent behavior analysis (e.g., confusion navigation, hesitancy, repeated loops, hallucinated interactions).

### Questions
N/A

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
2

### Summary
This paper constructs a large dataset of evaluating mobile agents with the combination of both dynamical and static evaluation strategies. The benchmark also contains common usage noises such as ads jumping. The authors claim they use DFS and BFS for searching trajectories, while not clear how to make sure the trajectory is reasonable and how to annotate the dataset, since the instruction are trajectories should be highly aligned. The evaluations are extensive on multiple LLMs.

### Strengths
The articulation of current benchmark limitations is clear. The proposed mixed evaluation is reasonable. The constructed benchmark is quite diverse, including multiple diverse apps. The evaluation on multiple LLMs both with and without noises are appreciated.

### Weaknesses
The dataset synthesis process is unclear, especially if using rule-based or algorithm-based methods, how to make sure the trajectory is reasonable. Is human annotation needed to construct the benchmark? As for the innovation part of the benchmark creation, could authors clarify the difference from previous benchmarks?

### Questions
How to make sure the benchmark is reliable and consistent to common user using?

### Soundness
3

### Presentation
2

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
This paper presents SMAN-Bench, a comprehensive benchmark for evaluating vision-language model (VLM)–based mobile agents across diverse and realistic settings. Built on the Mobile3M graph corpus, SMAN-Bench introduces an offline multi-path evaluation method that assesses agents across Single-path, Multi-path, Ambiguous, and Noisy task modes, providing both reproducibility and flexibility.

The benchmark features a slot-based instruction generation method (GIAS) for scalable annotation, and includes cross-system coverage across Android, iOS, HarmonyOS, and HyperOS. It also incorporates robustness and interaction evaluation, testing agents under ad noise and ambiguous instructions. Extensive experiments with open- and closed-source VLMs under multiple agent frameworks demonstrate that SMAN-Bench captures realistic challenges and reveals performance differences across reasoning, robustness, and multi-path capabilities.

### Strengths
1. **Novel and well-motivated benchmark design**: The offline multi-path evaluation successfully bridges the gap between deterministic offline datasets and unstable online evaluations. By rewarding progress toward “key nodes” within the graph rather than requiring exact path matches, the benchmark achieves both reproducibility and multi-solution coverage.
2. **Comprehensive task diversity and realism**: SMAN-Bench extends evaluation beyond task completion to robustness and interaction. The Noisy split (ads, pop-ups, redirects) and Ambiguous split (Q&A clarifications) emulate real-world conditions that are rarely tested in prior benchmarks such as Mobile-Agent-Bench or SPA-Bench.
3. **Scalable annotation pipeline (GIAS)**: The GIAS method systematically extracts action intents and fills template slots to produce thousands of instructions without manual labeling. This approach reduces annotation cost and links one instruction to multiple trajectories, supporting multi-path evaluation.
4. **Thorough experimentation**: The authors evaluate more than twenty models under three frameworks and analyze success rate (SR), step accuracy, and step efficiency (SE). Results reveal that multi-path evaluation correlates better with true capabilities than single-path evaluation, and ambiguous instructions enhance reasoning VLMs such as LLaMA and GPT-4o.

### Weaknesses
1. **Clarity and writing quality**: The paper is dense and often unclear about implementation details. For instance, the “offline simulator” and the definition of “key nodes” are spread across sections without a cohesive overview. Figures 1 and 2 contain too much information without step-wise guidance. A clearer explanation of evaluation loops and agent–graph interaction would aid reproducibility.
2. **Evaluation protocol ambiguities**: It is unclear how actions not present in the Mobile3M graph are handled (for example, unseen coordinates or slot values). Does the simulator terminate, discard the action, or return an error? Similarly, the definition of the “predefined query pool” is not quantitatively specified, and the interaction between step limits and noisy tasks remains unclear.
3. **Limited comparison with related graph-based benchmarks**: The paper focuses its comparison on Mobile-Agent-Bench and SPA-Bench but omits recent graph-structured evaluators such as ColorBench [1], WebGraphEval [2], CRAB [3], and OmniBench [4]. A discussion contrasting SMAN-Bench’s key-node reward with these graph-centric methods would clarify its unique contribution.
4. **Under-explored cross-system analysis**: Although SMAN-Bench covers four OS ecosystems, the results mainly aggregate performance without analyzing per-OS differences or transfer difficulties. Understanding how models generalize from Android to iOS or HarmonyOS would strengthen the “cross-system” claim.
5. **Slot-value and coverage uncertainty**: While the paper mentions a predefined query pool, it is unclear how slot values such as song titles or colors are bounded. If an instruction contains a slot not present in the dataset, the paper does not explain how correctness is judged. 

### **References**

[1] ColorBench: Benchmarking Mobile Agents with Graph-Structured Framework for Complex Long-Horizon Tasks. https://arxiv.org/abs/2510.14621v1

[2] WebGraphEval: Multi-Turn Trajectory Evaluation for Web Agents using Graph Representation. https://arxiv.org/abs/2510.19205v1

[3] CRAB: Cross-Environment Agent Benchmark for Multimodal Language Model Agents. https://arxiv.org/abs/2407.01511

[4] OmniBench: A Scalable Multi-Dimensional Benchmark for Essential Virtual Agent Capabilities. https://arxiv.org/abs/2506.08933

### Questions
1. **Offline simulation mechanics**: When an agent issues an unrecorded action, does the simulator terminate or ignore it? Are any mobile emulators used during evaluation?
2. **Slot template coverage**: Are slot values limited to those observed in Mobile3M, or can instructions include unseen values?
3. **Cross-system alignment**: How are UI pages and key nodes aligned across Android, iOS, HarmonyOS, and HyperOS? Is the alignment manual or automatic?

### Soundness
3

### Presentation
2

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
The paper presents SMAN-Bench, a large, cross-system benchmark for mobile GUI agents that explicitly targets four real-world difficulties: Single-path bias, Multi-path execution, Ambiguous instructions, and Noisy screens (ads/pop-ups). It is built on top of the large graph-like mobile interaction corpus Mobile3M (49 popular apps, tens of millions of transitions), from which the authors automatically construct 12,856 bilingual instructions via a slot-based generator (GIAS). Each instruction can map to one or multiple valid trajectories, enabling both single-path and multi-path evaluation. Two special subsets—Noisy (ads, interstitials) and Ambiguous (underspecified user goals requiring clarification)—are added to stress-test current agents. Experiments on a wide range of mobile agents (AppAgent, Mobile-Agent variants, recent GUI-R1/G1-style models, and MLLM-based agents) show consistent drops on noisy/ambiguous settings and significant gains when multi-path scoring is enabled, indicating overfitting to a single gold path in prior work.

### Strengths
- Multi-path as first-class citizen. Unlike benchmarks that match only one golden trajectory, SMAN-Bench merges multiple observed paths into a graph and scores hitting key nodes; this is much closer to how real mobile tasks have many valid sequences. 
- Noisy & ambiguous subsets. Explicitly injecting ads/pop-ups and instruction under-specification is valuable, because current agents often break exactly there and existing benchmarks seldom isolate this factor. 
- Scale and coverage. 12k+ bilingual tasks, 49 apps, 15 categories, and cross-system (Android, iOS, HarmonyOS/HyperOS) make it a broadly useful offline benchmark. 
- Good diagnostics. By reporting both single-path and multi-path scores, the benchmark can tell whether an agent failed because it “did the wrong thing” or because it “did a right thing that wasn’t the gold path.”

### Weaknesses
- Instruction naturalness. The GIAS is template/slot-based; although it gives nice coverage, some instructions may read slightly synthetic.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
3
