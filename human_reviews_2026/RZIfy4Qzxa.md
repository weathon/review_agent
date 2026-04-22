# MedSentry: Understanding and Mitigating Safety Risks in Medical LLM Multi-Agent Systems

- Avg Score: 4.50
- Decision: Reject
- Scores: 8, 2, 4, 4

## Abstract
As large language models are increasingly adopted in healthcare, ensuring their safety is critical, particularly in collaborative multi-agent settings. This paper develops an end-to-end attack–defense evaluation workflow to systematically analyze how four representative multi-agent topologies (Layers, SharedPool, Centralized, and Decentralized) behave under attacks from ``dark-personality'' agents. To support the evaluation, we curate MedSentry, a data resource containing 5,000 adversarial medical prompts that span 25 threat topics and 100 subtopics. Our study reveals critical differences in how these architectures handle information contamination and maintain robust decision-making, exposing their underlying vulnerability mechanisms. For example, SharedPool is highly susceptible due to open information sharing, whereas Decentralized exhibits stronger resilience owing to inherent redundancy and isolation. To mitigate these risks, we propose a personality-scale detection and correction mechanism that identifies and rehabilitates malicious agents, restoring safety to near-baseline levels. Taken together, MedSentry provides a rigorous evaluation framework alongside actionable defense strategies, offering guidance for the design of safer LLM-based multi-agent systems in medical contexts.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper targets the critical challenge of safety in medical multi agent systems and introduces MedSentry, an evaluation and adversarial framework that aggregates 5,000 adversarial medical prompts spanning 25 topics and 100 subtopics. The system injects malicious agents with dark personality traits, compares the resilience of four mainstream topologies, and quantifies safety using AMA ethical guidelines. The pipeline is clearly specified and forms a closed loop, the scenarios mirror collaborative clinical workflows, and both data and protocols are reproducible and extensible, which gives the overall design strong engineering rigor and research value. Experiments show that SharedPool is most susceptible to information contamination, Decentralized is the most robust, and Layers and Centralized fall in between. The proposed PCDC enforcement agent combines personality scale screening, behavioral verification, and topology aware isolation, effectively restoring safety scores after attack toward the baseline. The paper delivers systematic benchmarking and mechanistic insights and provides actionable tools and clear directions for improving the safety of medical multi agent systems.

### Strengths
- The topic is timely and fills a clear gap, focusing on the intersection of LLM safety and medical multi agent systems, with high importance and strong real-world relevance.
- The benchmark is rigorously constructed with clinical expert input, includes fine grained and stealthy adversarial prompts, and is more effective than existing datasets at eliciting unsafe behavior.
- The architectural comparison is insightful, with systematic experiments indicating the resilience of decentralized designs and the vulnerability of shared pool settings, which offers actionable guidance for system selection.
- The mitigation is practical and deployable. PCDC is lightweight, interpretable, and effective, integrating personality assessment with behavioral verification, which facilitates engineering deployment and compliance auditing.
- The experimental scope is comprehensive, with detailed comparisons to existing benchmarks across system structures and with attack and defense evaluations for each basic topology. The study further examines agent count, dialogue rounds, and token level safety trends and vulnerabilities, which strengthen robustness and generalizability. The appendix provides ample details and extended studies that corroborate the main text.

### Weaknesses
- In Figure 1, the Claude logo shows slight tearing artifacts and alignment instability, which should be refined.
- "Attack Obfuscation and Purification" relies on the same original model used to generate or obfuscate data and does not evaluate cross-model purification or generalizability.
- In the proposed defense, what is the rationale for monitoring only the initial and follow-up turns rather than auditing all remaining turns? How would expanding coverage to every turn influence safety outcomes?
- Consider adding an outlook on real-world application scenarios for the four topologies in medical settings, either in the Introduction or the Appendix.

### Questions
- How do you expect the proposed defenses to generalize to datasets beyond MedSentry?
- Could the Token Usage section (Appendix) include a comparison between defense-enabled and defense-disabled settings, covering per-round latency and total token consumption, to help assess deployment costs?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces an end-to-end attack–defense evaluation framework for assessing the safety of large language models in collaborative multi-agent healthcare settings. Focusing on four representative topologies—Layers, SharedPool, Centralized, and Decentralized—the authors analyze their vulnerability to attacks from "dark-personality" agents using a newly curated dataset, MedSentry, comprising 5,000 adversarial medical prompts across 25 threat topics. The study uncovers key differences in each topology's resilience, with SharedPool being highly vulnerable due to open information sharing, while Decentralized shows greater robustness. To enhance safety, the paper proposes a personality-scale detection and correction mechanism that effectively mitigates adversarial influence. Overall, this work offers a systematic framework and practical strategies for designing safer LLM-based multi-agent systems in medical applications.

### Strengths
1. Investigating the resilience of various multi-agent architectures to adversarial prompts is a valuable and relevant direction, offering insights into the robustness and design trade-offs of safety-critical LLM systems.

### Weaknesses
1. Existing safety benchmarks, like MedSafetyBench, in the medical domain already address various risks, and it is unclear how the proposed benchmark distinguishes itself, aside from the inclusion of a malicious agent.
2. The evaluated scenarios lack practical relevance, as the likelihood of inserting a malicious agent into real-world medical systems is very low. 
3. The study does not employ advanced attack methods like [1, 2], resulting in prompts that are not sufficiently adversarial to meaningfully challenge the evaluated systems.


[1] Liu, X., Xu, N., Chen, M. and Xiao, C., 2023. Autodan: Generating stealthy jailbreak prompts on aligned large language models. arXiv preprint arXiv:2310.04451.

[2] Andriushchenko, M., Croce, F. and Flammarion, N., 2024. Jailbreaking leading safety-aligned llms with simple adaptive attacks. arXiv preprint arXiv:2404.02151.

### Questions
1. The margins of the figures and tables have been adjusted in a way that impairs the readability of the paper. Clearer formatting is needed to improve visual clarity and overall presentation.

### Soundness
2

### Presentation
2

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
The paper develops a framework for understanding and mitigating safety risks in medical LLM multi-agent systems. It constructs a dataset of 5k adversarial medical prompts covering diverse topics, and tests how different architectures respond to internal “dark-personality” agents. To counter threats, the authors propose a mechanism combining an enforcement agent to monitor. Overall, MedSentry provides an end-to-end attack–defense evaluation pipeline and actionable defense strategies for building safer medical LLM multi-agent systems.

### Strengths
- The paper tackles an important and underexplored problem, evaluating the safety of medical LLM agents under multi-agent setting.

- The benchmark design is comprehensive, encompassing multiple medical stages and diverse risk categories. This makes it representative.

- Extensive experiments are done to analyze a wide range of factors that could possibly affect the risk of the multi-agent system.

### Weaknesses
- The description of the attack instruction process in *Coarse-Grained Data Generation* is vague. It is unclear how the process is iterative and how "topics and subtopics are substituted" across iterations. More explicit examples or algorithmic details would improve clarity.
- The evaluation design may introduce confounding factors across different topologies. As mentioned in Lines 209–210, the evaluator agent receives different amounts of information under different setups. This raises the concern that score differences may partly stem from unequal access to information rather than genuine behavioral differences. Moreover some evaluation criteria listed in Appendix C.3 may not be observable from a single-turn response (e.g., the final summary from layer topology), which could further bias topology-specific scores. What if there is no enough information for the evaluator to apply a certain criteria? Will it assign a high value or it just leave it as zero?
- The comparison in Table 2 seems werid. Since MedSentry and MedSafetyBench use different models, their scores are not directly comparable. Therefore, the conclusion that “MedSentry possesses greater threat potential and concealment than MedSafetyBench” is not fully supported.
- A minor problem, I don't know the reason for using LCS metric. Its purpose and interpretability in the context of medical safety evaluation are unclear, and it is not intuitive how this metric captures meaningful behavioral or safety-related differences between models.

### Questions
See above

### Soundness
2

### Presentation
2

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
This paper introduces MedSentry, a benchmark with 5,000 adversarial medical prompts across 25 categories and 100 subtopics, to evaluate safety risks in medical LLM multi-agent systems. The authors compare four topologies (Layers, SharedPool, Centralized, Decentralized) under attacks from "dark-personality" agents and propose PCDC (Personality-scale Detection and Correction) as a defense. Results show SharedPool is most vulnerable while Decentralized is most resilient. PCDC reportedly restores safety to near-baseline levels.

### Strengths
1. The paper focuses on multi-agent systems, which are increasingly deployed in healthcare but remain underexplored from a security perspective. This is a valuable contribution distinct from single-model medical AI benchmarks.  

2. The paper is a systematic investigation of topological vulnerabilities in medical multi-agent systems. The finding that topology choice significantly impacts safety is actionable for system designers.  

3. Benchmark scale and organization appear comprehensive in the paper. With 5,000 prompts across 25 categories and 100 subtopics, MedSentry provides reasonable coverage of medical adversarial scenarios. The two-level taxonomy structure allows for granular analysis of attack patterns across medical subdomains.

### Weaknesses
1. The paper introduces "dark-personality" adversarial agents without a reasonable threat model. Dark Triad and other psychometrics are human constructs with questionable applicability to AI agents. This fundamental conceptual issue undermines the entire PCDC and evaluation framework. 

2. The paper needs component-wise analysis to understand what drives the results. The paper does not do ablation studies on components of PCDC. Regarding "the defense pipeline as an integrated end to end process" is not a reasonable excuse for not doing ablation studies.

3. All experiments conducted in controlled simulation environments with no evidence of behavior in actual clinical scenarios. Clinician validation of generated outputs are not validation of system integrated in real-world clinical scenarios.

### Questions
1. You thoroughly measure safety across topologies, but do safer architectures sacrifice task performance on medical tasks? What is the tradeoff between safety and performance? This is important for practical deployment decisions.

2. What is PCDC's false positive rate? how often are benign agents incorrectly flagged as malicious?

### Soundness
2

### Presentation
2

### Contribution
3
