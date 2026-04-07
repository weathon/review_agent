=== CALIBRATION EXAMPLE 23 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title accurately reflects the paper's focus. The abstract clearly states the motivation (engineering MAS require rigor and are vulnerable), the gap (first systematic study in engineering contexts), the methodology (controlled experiments on representative problems), and the key findings (sensitivity to task type, error subtlety, communication order, and design choices that improve resilience). The claims are supported by the paper’s content.

### Introduction & Motivation
The introduction effectively motivates the problem: LLM-based MAS are increasingly used in engineering, where errors can have serious consequences, and adversarial vulnerabilities in such systems are understudied. It contextualizes the work within existing literature on MAS security and identifies a specific gap: the lack of systematic analysis of how prompting and communication structure jointly affect robustness in engineering tasks. The contributions are implied but could be more explicitly enumerated for clarity.

### Method / Approach
The method is clearly described and reproducible. The hierarchical two-agent setup (leader and misleading advisor) with a turn-based, synchronous communication protocol is well-defined. The baseline problem (Darcy-Weisbach pressure loss) is appropriate. The use of a fixed adversarial strategy (agent instructed to propose f=25/Re) provides a controlled testbed. However, several assumptions and limitations should be noted:
1. **Adversarial Model Simplistic:** The misleading agent is always explicitly adversarial and follows a fixed, known error pattern. Real-world adversaries might use more subtle, adaptive, or stealthy strategies (e.g., occasional errors, plausible but incorrect reasoning). The paper partially addresses this by testing different misleading prompts (Appendix A) but does not explore more sophisticated attacks.
2. **Limited Generalizability:** The core experiments focus on one primary engineering problem. While other tasks are introduced (beam, math, graphs), the extensive prompt and configuration variations are not systematically replicated across all tasks, making it unclear if findings generalize beyond fluid dynamics.
3. **Model Dependence:** Experiments primarily use GPT-4o mini; comparisons with GPT-4o and o3 mini are included but brief. The results may be highly model-specific, and robustness trends might differ for other LLM families or open-source models.
4. **Interaction Protocol Fixed:** The hierarchical, turn-based protocol with a rethinking phase is a specific design choice. Other MAS architectures (e.g., decentralized, voting-based) may exhibit different robustness properties.

### Experiments & Results
The experimental design is thorough, with 30 trials per condition and appropriate statistical tests (Fisher’s exact, Mann-Whitney U). Results are presented clearly across four dimensions:
1. **Prompt Influence:** Shows that explicit warnings, non-concise styles, and authoritative tones significantly improve rejection rates. This is valuable for practitioners.
2. **Task Influence:** Demonstrates that task complexity and the subtlety of the incorrect solution strongly affect susceptibility. For example, rounding errors (division task) are harder to detect than gross formula errors.
3. **Number/Order of Advisors:** Confirms a "first mover effect" and shows that adding more agents does not necessarily improve robustness and can reduce decision efficiency.
4. **Personalization:** Indicates that framing agents as experts or giving them names amplifies the first mover effect, increasing perceived credibility.

**Key Concerns:**
- **Multiple Testing:** With many comparisons, the risk of false positives increases. The paper reports p-values without correction for multiple testing, which could inflate significance claims. A brief discussion or adjustment would strengthen the analysis.
- **Correctness Under Non-Misled Conditions:** In some experiments (e.g., beam deflection), even when the leader rejects the misleading advice, the final answer is often incorrect (low correctness rates). This suggests that the leader’s own reasoning can be flawed, a nuance not deeply analyzed. The paper primarily focuses on whether the leader accepts the adversarial suggestion, but overall solution quality is also critical for engineering applications.
- **Limited Exploration of Attack Strategies:** Figure 8 shows that different misleading prompts yield varying success rates, but this is not integrated into the main analysis. A more systematic study of adversarial strategies (e.g., persuasive tactics, gaslighting) would enhance the contribution.

### Writing & Clarity
The paper is well-structured and clearly written. Figures and tables are informative. The appendix provides comprehensive details (prompts, conversations, full results), aiding reproducibility. Minor issues due to PDF parsing (e.g., broken equation in Section 3.1) do not impede understanding.

### Limitations & Broader Impact
The discussion acknowledges key limitations: the combinatorial explosion of prompt variations, non-linear agent behaviors, and the need for further research. The ethics statement appropriately notes the use of synthetic tasks and the goal of improving safety. However, the limitations section could be expanded to address:
- **Adversarial Model:** The study assumes a simple, always-misleading adversary. In practice, adversaries might be intermittent, target specific reasoning steps, or collude.
- **Task Scope:** The engineering tasks are relatively simple and well-defined. Real engineering problems involve more uncertainty, incomplete information, and multi-disciplinary constraints.
- **Societal Impact:** While briefly mentioned, a deeper discussion of potential negative impacts (e.g., if adversarial MAS are deployed in safety-critical systems without robustness measures) would be valuable.

## Overall Assessment
This paper presents a timely and systematic empirical investigation of adversarial robustness in LLM-based multi-agent systems for engineering problems. The experimental design is rigorous, with controlled variations across prompts, tasks, agent configurations, and personalization. The findings—particularly the sensitivity to prompt wording, the first mover effect, and the influence of task complexity—provide actionable insights for designing more resilient MAS. The work fills a noted gap in the literature by focusing on engineering contexts, where errors carry significant consequences.

The main weaknesses are the simplified adversarial model, limited task and model generalizability, and some statistical concerns. However, within its defined scope, the study is thorough and well-executed. The paper meets ICLR’s standards for empirical research, offering valuable insights for the community. It is suitable for acceptance, provided the authors address the noted limitations more explicitly in the final version.

# Neutral Reviewer
## Balanced Review

### Summary
This paper presents a systematic empirical investigation into the adversarial robustness of LLM-based Multi-Agent Systems (MAS) when applied to engineering problem-solving tasks. The study introduces a controlled setup where a "leader" agent collaborates with one or more "advisor" agents, some of which are designed to be misleading. By varying system prompts, task types (e.g., pipe pressure loss, beam deflection), agent numbers/orders, and personalization, the authors quantify how these factors influence the system's susceptibility to errors and its ability to reject misleading advice. The core finding is that robustness is highly sensitive to specific design choices, such as the explicitness of warnings in prompts, the "first mover" speaking order, and the complexity of the task and the injected error.

### Strengths
1.  **Clear Motivation and Relevance:** The paper effectively argues that engineering applications, which demand formal rigor and numerical accuracy, present a distinct and critical domain for evaluating MAS security, justifying the need for a domain-specific study beyond generic linguistic tasks.
2.  **Comprehensive and Systematic Evaluation:** The experimental design is thorough, exploring a wide matrix of variables: multiple leader/advisor prompt formulations (detailed in Appendix D), four distinct problem types, various numbers and orders of agents (supportive vs. misleading), and agent personalization. This allows for isolating key influential factors.
3.  **Statistical Rigor and Reproducibility:** The authors employ a sound methodology, using ≥30 trials per condition, calculating statistical significance (e.g., Fisher's Exact Test), and providing convergence analysis for sample size (TV distance). The paper includes detailed prompts, example conversations in the appendix, and a clear reproducibility statement, facilitating replication.
4.  **Actionable Insights:** The results yield concrete, practical findings. For instance, explicit warnings in the leader's prompt (e.g., "advisor’s suggestions have been proven wrong") significantly improve robustness; non-concise leader styles foster independent verification; and the first agent to speak exerts disproportionate influence ("first mover effect"), which is amplified if that agent is framed as an expert.

### Weaknesses
1.  **Limited Adversarial Model and Scope:** The study primarily investigates a single, relatively simple adversarial model: an agent that persistently advocates for a specific, incorrect formula or numerical answer. It does not explore more sophisticated, adaptive, or stealthy adversarial strategies (e.g., agents that selectively lie, contradict correct reasoning, or exploit logical fallacies) which may be more representative of real-world threats.
2.  **Model and Task Simplicity:** The core experiments rely on a single LLM (GPT-4o mini). While supplementary tests with GPT-4o and o1 show different robustness, a broader analysis across model families (e.g., open-source models) is lacking. Furthermore, the engineering problems, while representative, are simplified and closed-form; the study does not test complex, open-ended engineering design or analysis workflows where error propagation might be more severe and nuanced.
3.  **Lack of Mechanistic Explanation/Theoretical Grounding:** The work is heavily empirical. While it identifies correlational patterns (e.g., non-concise prompts help), it does not provide a deeper theoretical analysis or mechanistic explanation for *why* certain prompt phrasings or orders have the observed effects. The discussion remains at the level of observed trends rather than underlying principles of reasoning or persuasion within LLM-based MAS.
4.  **Incomplete Integration with Related Work:** The related work section adequately surveys the field but could better articulate how this paper's findings on *engineering* tasks specifically contrast with or extend prior knowledge about robustness in generic MAS. The claim of being the "first systematic study" in engineering contexts is supported, but the novelty of the individual findings (e.g., the importance of speaking order) relative to prior work like Ju et al. (2024) could be clarified.

### Novelty & Significance
**Novelty:** The paper successfully identifies and addresses a niche that is both timely and underexplored: the security of LLM-based MAS in numerical/formal engineering domains. While the vulnerability of MAS to misaligned agents is a known concept, the systematic, quantitative exploration of how this vulnerability manifests differently across engineering tasks and is modulated by prompt engineering and system design is a novel contribution.

**Significance:** The significance is **moderate**. The findings provide valuable, actionable guidelines for practitioners designing MAS for technical domains (e.g., emphasizing prompt warnings and managing speaking order). However, the core vulnerability identified—that a single misaligned agent can often corrupt consensus—is not itself a new discovery. The paper does not propose a novel defense mechanism or uncover a fundamentally new class of attacks; rather, it provides a detailed empirical map of an existing vulnerability in a new context. For ICLR, which often prioritizes foundational advances, the incremental nature of the empirical insights may be a limiting factor.

### Suggestions for Improvement
1.  **Deepen the Adversarial Analysis:** Future work should test more advanced adversarial strategies, such as agents that introduce subtle logical errors, contradict intermediate reasoning steps, or dynamically adjust their arguments based on the leader's stance. This would provide a more comprehensive threat model.
2.  **Expand Model and Task Diversity:** To strengthen generalizability, the experiments should be replicated across a wider range of LLMs (including larger, smaller, and open-source models). The task suite should be expanded to include more complex, multi-step engineering problems or simulations where agents must interact with an external tool or environment.
3.  **Develop a Conceptual Framework:** The paper would be significantly strengthened by moving beyond correlations to propose a framework or hypothesis for the cognitive or mechanistic processes within LLM agents that lead to observed robustness behaviors (e.g., how verbosity encourages internal verification, or how authority framing biases trust). This could connect the empirical results to broader theories of reasoning in LLMs.
4.  **Discuss Limitations and Real-World Implications More Thoroughly:** The discussion should more explicitly address the gap between the controlled lab setting and real-world deployment. For example, how would the findings translate to a system with 10+ agents, or where the "correct" answer is not known a priori? A dedicated subsection on limitations and future challenges would add depth.
5.  **Enhance the Narrative Around Defenses:** While the paper identifies design choices that improve robustness, it could more explicitly frame these as preliminary defensive recommendations and discuss their potential downsides (e.g., reduced efficiency from non-concise prompts, as noted) and how they might be systematically integrated into a robust MAS design paradigm.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Compare to established adversarial robustness baselines for MAS.** The paper lacks comparison to standard mitigation techniques (e.g., consensus voting, agent reputation systems, or filtering mechanisms). Without this, the claimed importance of prompt design is not contextualized within existing literature, weakening the contribution.
2. **Evaluate with a diverse set of LLM foundations.** The core findings rely almost exclusively on GPT-4o mini. To support general claims about LLM-based MAS, experiments must include other prominent models (e.g., Claude, Llama, Gemma) to show findings are not model-specific artifacts.
3. **Test more sophisticated, adaptive adversarial strategies.** The misleading agent uses a static, pre-defined error. To substantiate claims about error subtlety and complexity, experiments should include dynamic adversaries that adapt their arguments or collude, which is a realistic threat.
4. **Include more complex, multi-step engineering tasks.** The current tasks are largely single-step calculations. To validate claims about structural complexity impacting robustness, the study needs tasks with interdependent steps (e.g., design optimization, system diagnostics) common in real engineering.
5. **Ablate the communication protocol itself.** The fixed turn-based scheme with rethinking may heavily influence results. Experiments with different protocols (e.g., debate, simultaneous reporting, voting) are needed to see if the identified factors (like order) are protocol-dependent.

### Deeper Analysis Needed (top 3-5 only)
1. **Analyze the reasoning traces to explain *why* prompt variations work.** The paper reports that "non-concise" or "authoritative" styles improve rejection rates but does not analyze the underlying reasoning processes (e.g., increased chain-of-thought, more independent verification). Without this, the findings are correlational, not causal.
2. **Conduct a detailed error propagation analysis.** The paper tracks final outcomes but does not trace how the erroneous information spreads or is contested within the dialogue. Understanding the persuasive dynamics (e.g., which arguments successfully rebut errors) is critical for the claims about communication order and error subtlety.
3. **Perform rigorous statistical correction for multiple comparisons.** With dozens of experiments and significance tests, the risk of false positives is high. The paper must address multiple testing (e.g., via Bonferroni or FDR correction) and report effect sizes to ensure robust statistical claims.
4. **Disentangle the "first mover effect" from argument quality.** The observed order effect could be confounded by the content of the first message. An analysis comparing scenarios where the first agent presents a strong vs. weak argument (independent of correctness) is needed to isolate the pure order effect.
5. **Analyze the interaction between the leader's initial solution confidence and robustness.** The leader's pre-advisor solution stance (e.g., fully derived vs. tentative) likely moderates susceptibility. This analysis is missing but directly affects the interpretation of prompt and task results.

### Visualizations & Case Studies
1. **Provide systematic qualitative analysis of conversation failures and successes.** The appendix shows examples, but a structured analysis categorizing failure modes (e.g., leader accepts flawed reasoning, fails to detect numerical trickery) and success patterns across tasks would concretely show how the method works or fails in practice.
2. **Visualize the decision evolution over conversation turns.** Plots showing the leader's proposed answer or confidence score across iterations would reveal critical junctures where robustness breaks down or is reinforced, directly illustrating the claimed dynamics of error propagation.
3. **Create a visual mapping between task complexity metrics and robustness scores.** A scatter plot linking quantitative complexity measures (e.g., steps, equation complexity, conceptual distractors) to misleading rates across all tasks would powerfully support the claim that complexity drives vulnerability.

### Obvious Next Steps
1. **Propose and evaluate a synthesized defense strategy based on the findings.** The paper identifies influential factors but stops short of combining them into a concrete, actionable defense (e.g., an optimized prompt template or agent ordering protocol). This is a necessary step to translate insights into practice.
2. **Extend the evaluation to a wider range of engineering domains.** The tasks are limited to fluid mechanics, beams, basic math, and graphs. To substantiate the "engineering" focus, domains like thermodynamics, circuit design, or control systems should be included to test generalizability.
3. **Investigate the robustness-efficiency trade-off quantitatively.** The paper notes longer discussions reduce efficiency but does not model this trade-off. A systematic analysis proposing how to balance rejection rate against conversation length (e.g., via early stopping rules) is a logical next step missing from the discussion.
4. **Explore adversarial training or inference-time hardening for the leader agent.** Given the sensitivity to prompts, a direct next step is to use the findings to robustify the leader via tailored prompt optimization or iterative self-correction mechanisms, which should be piloted.

# Final Consolidated Review
## Summary
This paper presents the first systematic empirical study of adversarial robustness in LLM-based Multi-Agent Systems (MAS) for engineering problem-solving. Using a controlled setup with a leader agent and one or more misleading/supportive advisors, the authors investigate how system performance is affected by variations in prompts, task types, agent numbers/order, and personalization. The core finding is that robustness is highly sensitive to specific design choices, with explicit warnings, non-concise leader styles, and speaking order ("first mover effect") being key factors.

## Strengths
- **Systematic and Comprehensive Experimental Design:** The paper methodically explores a wide matrix of influential factors—including multiple prompt components, four distinct engineering/math tasks, various numbers and orders of agents, and personalization—enabling the isolation of key variables affecting robustness. This breadth is beyond most studies in the area.
- **Statistical Rigor and Reproducibility:** The authors employ a sound methodology with ≥30 trials per condition, statistical significance testing (Fisher's Exact, Mann-Whitney U), and a convergence analysis for sample size (Total Variation Distance). Detailed prompts, example conversations, and a reproducibility statement are provided, facilitating replication.
- **Actionable, Domain-Specific Insights:** The study yields concrete, practical findings for designing more resilient engineering MAS. For example, it quantifies how explicit warnings (e.g., "advisor’s suggestions have been proven wrong") and non-concise leader styles significantly improve rejection rates, and demonstrates how the "first mover effect" is amplified when the first speaker is framed as an expert.

## Weaknesses
- **Simplified and Static Adversarial Model:** The misleading agent consistently advocates for a single, pre-defined incorrect formula (e.g., f=25/Re). The study does not explore more realistic, adaptive, or stealthy adversarial strategies (e.g., agents that introduce subtle logical errors, collude, or dynamically adjust arguments), limiting the comprehensiveness of the threat model evaluated.
- **Limited Model and Task Generalization:** The core findings are primarily based on one LLM (GPT-4o mini). While supplementary tests with GPT-4o and o3-mini are included, a broader analysis across diverse model families (including open-source models) is lacking. Furthermore, the engineering tasks, while representative, are relatively simple and closed-form; the study does not test complex, open-ended engineering workflows where error propagation could be more nuanced.
- **Risk of Inflated Statistical Significance:** With dozens of experimental conditions and pairwise significance tests reported, the risk of false positives due to multiple testing is not addressed (e.g., via p-value correction). This weakens the confidence in some of the claimed significant differences.
- **Incomplete Analysis of Solution Quality:** The paper focuses on whether the leader accepts the adversarial suggestion. However, in several experiments (notably beam deflection), the "correctness rate" for non-misled decisions is low, indicating the leader's own reasoning can be flawed. This nuance—critical for engineering applications—is not deeply analyzed.

## Nice-to-Haves
- A deeper qualitative analysis of conversation traces to explain *why* certain prompt variations (e.g., non-concise style) improve robustness, moving from correlation toward mechanistic understanding.
- Exploration of how the identified robust design choices (e.g., prompt warnings, agent order) could be synthesized into a concrete, optimized defense strategy or protocol.
- Inclusion of a more complex, multi-step engineering task to better validate claims about structural complexity impacting error propagation.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Weakness: "Lack of comparison to established adversarial robustness baselines for MAS (e.g., consensus voting)."** — The paper's contribution is a foundational empirical study identifying vulnerabilities and influential factors, not proposing or benchmarking defenses. Demanding comparison to mitigation techniques is scope creep.
- **Weakness: "Need to ablate the communication protocol itself."** — The paper explicitly studies a specific, well-defined hierarchical turn-based protocol with a rethinking phase. Evaluating other protocols is a different research question, not a flaw in this study's setup.
- **Weakness: "Requires a theoretical grounding or mechanistic explanation."** — The paper is a thorough empirical investigation. While deeper analysis would be valuable, demanding theoretical proofs for an empirical systems paper imposes an arbitrary rigor requirement not standard for this type of work.
- **Weakness: "The adversarial model is too simplistic."** (as phrased in Review 1) — Weakened and rephrased above. The paper does test different misleading advisor prompts (Fig. 8, App. D), partially addressing strategy variation, but the core model remains static.
- **Strength/Weakness about writing style, importance of topic, or extensiveness of experiments** — These are generic and removed. Strengths and weaknesses must be specific to this paper's contributions.

## Novel Insights
The paper provides a novel, data-rich mapping of how adversarial vulnerability in LLM-based MAS manifests specifically in engineering contexts, where errors have formal consequences. It demonstrates that the interplay between task complexity (e.g., confusable numerical variations) and error subtlety is a critical driver of susceptibility, a nuance less prominent in generic linguistic MAS studies. Furthermore, it quantifies how agent personalization (e.g., naming, expert framing) amplifies the "first mover effect," revealing that perceived credibility, not just sequence, modulates persuasion in technical collaboration.

## Suggestions
- Address the multiple testing issue in the statistical analysis, for example by applying a correction method (e.g., FDR) or at least discussing it as a limitation in the relevant results section.
- Expand the discussion of the "correctness rate" findings, especially for non-misled decisions. Analyzing why leaders sometimes arrive at incorrect solutions even when rejecting the adversary would strengthen the paper's assessment of overall system trustworthiness.
- In the limitations section, more explicitly discuss the gap between the static adversarial model used and potential real-world adaptive adversaries, framing it as a direction for future work.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 2.0, 0.0]
Average score: 1.5
Binary outcome: Reject
