=== CALIBRATION EXAMPLE 18 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title and abstract accurately reflect the paper’s scope. The claim of being the “first systematic study” of adversarial robustness for LLM-based MAS in engineering contexts is plausible given the cited literature, though it could be nuanced, as some prior work touches on related themes. The abstract’s summary of findings (sensitivity to task type, error subtlety, communication order) is supported by the results. However, the abstract overstates the actionable insights somewhat; many findings (e.g., non-concise prompts help, first speaker matters) are intuitive and lack a deeper mechanistic explanation that would make them broadly actionable.

### Introduction & Motivation
The introduction effectively motivates the problem: engineering workflows require formal rigor, and adversarial perturbations in MAS could lead to unsafe outcomes. It situates the work within existing literature on MAS security and identifies a clear gap: a lack of comprehensive analysis of how prompting and communication structure jointly affect robustness in engineering contexts. The contributions, while implied, could be stated more explicitly as distinct research questions or hypotheses.

### Related Work
The related work section is comprehensive, covering adversarial attacks on MAS, security surveys, and engineering applications. It correctly notes that most prior engineering MAS work focuses on functionality, not security. However, it could better synthesize how this paper goes beyond prior adversarial studies (e.g., Ju et al. 2024, Huang et al. 2025a) by focusing on engineering tasks with numerical/formal demands, rather than general misinformation spread.

### Method
The methodology is clearly described but has several limitations that affect the paper’s contribution:

1. **Experimental Setup**: The MAS architecture is extremely simplified: a two-agent hierarchy (leader and misleading advisor) with a turn-based, synchronous protocol. While later experiments add more advisors, the interaction pattern remains elementary. Real-world engineering MAS often involve more complex coordination, asynchronous communication, and richer role definitions. This simplicity limits the external validity of the findings.

2. **Adversarial Model**: The adversarial strategy is fixed and unsophisticated. The misleading agent always injects the same semantic error (e.g., \(f = 25/Re\) instead of \(64/Re\)). There is no exploration of adaptive adversaries, multi-step attacks, or more subtle manipulations (e.g., misleading reasoning steps rather than just incorrect formulas). This reduces the study’s relevance to realistic threat models.

3. **Task Selection**: The four problem classes are reasonable but narrowly chosen. The paper does not justify why these specific problems are “representative” of engineering as a whole. More importantly, the complexity of each task is discussed qualitatively; a more rigorous characterization (e.g., number of reasoning steps, required domain knowledge) would strengthen the analysis.

4. **Statistical Rigor**: The use of 30 trials per condition, justified by a TVD convergence analysis, is adequate. Statistical tests (Fisher’s exact, Mann-Whitney U) are appropriate. However, the paper lacks a multivariate analysis that could disentangle the relative importance of factors (e.g., prompt style vs. task complexity). The results are presented as a series of univariate comparisons, making it hard to assess interactions.

5. **Reproducibility**: The appendices provide detailed prompts and configurations, which is excellent. The commitment to release code is commendable.

### Experiments & Results
The results are extensive but largely descriptive and sometimes unsurprising:

- **Prompt Influence (Sec 4.1)**: The findings that explicit warnings and non-concise styles improve rejection rates are intuitive. The authors hypothesize that non-concise leaders “solve the problem on their own first,” but this is not verified (e.g., by analyzing reasoning steps). The analysis is correlational; it does not establish causality or underlying mechanisms.

- **Task Influence (Sec 4.2)**: The observation that more complex or subtle errors lead to higher misleading rates is expected. The paper does not provide a clear definition or measure of “complexity” or “subtlety,” making it difficult to generalize. For instance, the division task’s high misleading rate is attributed to rounding-error confusion, but this is a post-hoc interpretation without systematic validation.

- **Number and Order of Advisors (Sec 4.3)**: The “first mover effect” replicates prior findings (Ju et al. 2024) in an engineering context, which is valuable. The counterintuitive result that two misleading agents (MM) outperform one (M) is interesting, but the proposed explanation (“support each other too obviously”) is speculative and not tested. The experiments with multiple agents show that robustness does not monotonically improve with more agents, which is an important insight, but the analysis stops short of explaining why certain configurations (e.g., SMM) are more robust.

- **Names and Authority (Sec 4.4)**: The finding that personalization (expertise, names) amplifies the first mover effect is novel and relevant for design. However, the effect is only demonstrated on two configurations (SMM and MSM); broader validation across other agent orders would strengthen the claim.

- **Missing Ablations**: The paper does not ablate key components of the interaction protocol. For example, what is the impact of the “rethinking phase”? Does it actually improve robustness, or is it just a redundant step? Also, the leader’s initial solution (before advisor input) is not analyzed; understanding how often the leader starts with a correct solution would help interpret susceptibility.

### Writing & Clarity
The paper is well-organized and easy to follow. Figures are clear, and the appendix provides necessary detail. Some sections, like the results, are dense with bar plots but could benefit from more synthesis (e.g., a summary table of key factors). The writing is generally clear, though a few passages are repetitive (e.g., reiterating the first mover effect).

### Limitations & Broader Impact
The limitations section acknowledges the combinatorial challenge of prompt variations and the nonlinearity of agent behavior. However, it misses several critical limitations:
- The simplified MAS architecture and fixed adversarial strategy limit generalizability.
- The study uses only one LLM (GPT-4o mini variants) for most experiments; robustness findings may be highly model-dependent, as hinted by the GPT-4o and o3 mini results in Appendix A.
- The tasks are synthetic and lack real-world engineering constraints (e.g., noisy data, incomplete specifications).
- The paper does not discuss potential negative societal impacts, such as how adversaries might exploit these vulnerabilities in safety-critical systems.

The broader impact statement is appropriate but brief.

### Overall Assessment
This paper provides a thorough empirical exploration of adversarial robustness in LLM-based multi-agent systems for engineering tasks. It identifies several intuitive but important factors: prompt design, task complexity, agent order, and personalization. The replication of the “first mover effect” in an engineering context and the novel finding about personalization amplifying this effect are valuable contributions. However, the work is primarily descriptive and lacks a deeper analytical or theoretical foundation. The experimental setup is simplistic, and the findings, while extensive, are largely incremental relative to prior MAS adversarial robustness literature. For ICLR, which typically expects stronger theoretical insights, novel methodologies, or significant empirical advances, the paper falls short. It reads more like a well-executed application study suitable for a domain-specific workshop or a conference with a broader scope. The paper could be strengthened by a more rigorous analysis of why certain factors matter (e.g., modeling persuasion dynamics) and by exploring more realistic adversarial scenarios and MAS architectures. In its current form, it is unlikely to meet the high acceptance bar of ICLR.

# Neutral Reviewer
## Balanced Review

### Summary
This paper presents the first systematic study of adversarial robustness in LLM-based Multi-Agent Systems (MAS) applied to engineering problem-solving. Through controlled experiments on tasks like pipe pressure loss and beam deflection, the authors investigate how misleading agents propagate errors, analyzing the impact of prompt design, task complexity, agent number/order, and personalization. Key findings show robustness is highly sensitive to these factors, with a strong "first mover" effect and task-dependent vulnerability.

### Strengths
1.  **Well-Structured, Comprehensive Experimental Design:** The study systematically varies numerous factors (leader/advisor prompts, task types, agent numbers/orders, personalization) across a coherent baseline (Darcy-Weisbach). The use of statistical tests (Fisher's Exact, Mann-Whitney U) and 30 trials per condition adds rigor. Evidence: Detailed methodology (Sec 3, App B), full results table (App E), and convergence analysis (Fig 9).
2.  **Actionable, Domain-Specific Insights:** The paper moves beyond generic MAS security to provide concrete, practical findings for engineering applications. Evidence: Clear results showing how explicit warnings, non-concise leader styles, and placing supportive agents first can drastically improve rejection rates (Figs 3, 5, 6).
3.  **High Reproducibility:** The authors provide extensive details on prompts (App D), model parameters, evaluation metrics, and statistical methods, and commit to releasing code. This aligns well with ICLR's reproducibility standards. Evidence: Detailed appendices (B, D, E) and a clear reproducibility statement.

### Weaknesses
1.  **Limited Theoretical Depth and Mechanistic Explanation:** The work is heavily empirical. While it identifies patterns (e.g., "first mover effect"), it offers limited analysis of *why* these vulnerabilities exist from a reasoning or mechanistic perspective within the LLMs. Evidence: The discussion (Sec 5) summarizes findings but lacks a deeper cognitive or architectural analysis of the failure modes.
2.  **Simplistic Adversarial Model and Agent Architecture:** The "misleading" agent is constrained to proposing a single, specific wrong formula or answer. Real-world adversaries could be more adaptive or sophisticated. Furthermore, the MAS architecture is a simple, fixed-turn hierarchy, not exploring more complex coordination protocols. Evidence: The advisor's goal is narrowly defined (e.g., "pretend f=25/Re", Sec 3.1, App D), and the interaction scheme is basic (Fig 1).
3.  **Narrow Scope of Engineering Tasks:** The engineering problems, while representative, are relatively simple, closed-form calculations. The study does not address more open-ended, iterative, or safety-critical engineering design or analysis workflows where error consequences and adversarial influence could be more complex. Evidence: Tasks are textbook-style problems (pipe flow, beam deflection, basic math).

### Novelty & Significance
**Novelty:** Moderate. The application to systematically study adversarial robustness in *engineering-focused* LLM-MAS is novel and timely. However, the core concepts of adversarial prompts, error propagation, and order effects in MAS have been explored in other contexts (as cited in Related Work).
**Significance:** The practical significance is high for the safe deployment of LLM-MAS in technical domains. The paper successfully highlights that robustness is not automatic and provides empirical guidelines for designing more resilient systems. The theoretical significance is more limited due to the primarily empirical and applied focus.

### Suggestions for Improvement
1.  **Deepen the Analysis of Failure Modes:** Go beyond reporting success/misled rates. Analyze conversation transcripts to categorize *how* the leader is persuaded (e.g., flawed reasoning, deference to "expertise," computational error) and how this varies with task complexity. This would strengthen the contribution by providing explanatory insights.
2.  **Explore More Advanced Adversaries and Architectures:** Test adaptive adversaries that react to the leader's reasoning or employ more subtle semantic distortions. Also, experiment with decentralized debate or voting mechanisms beyond the hierarchical leader-advisor model to see if robustness properties change fundamentally.
3.  **Expand Task Complexity and Realism:** Include engineering tasks with greater ambiguity, multi-step reasoning, or integration of external tools (code, simulators). This would better assess robustness in realistic engineering workflows and reveal vulnerabilities that simple calculation tasks might mask.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1.  **Test across multiple, diverse LLM families.** The core findings are based almost entirely on GPT-4o mini, with only a few data points from GPT-4o and o3-mini. To support claims about general vulnerabilities and mitigation strategies in "LLM-based MAS," the paper must demonstrate that the observed phenomena (e.g., first-mover effect, prompt sensitivity) are not artifacts of a single model family. Experiments should include at least one model from another major provider (e.g., Claude, Gemini) and a capable open-source model (e.g., Llama 3, Mixtral).
2.  **Compare proposed prompt-based mitigations against established baseline defenses.** The paper identifies prompt variations that improve resilience but does not benchmark them against known techniques from the literature it cites (e.g., consensus mechanisms, randomized smoothing, hierarchical coordination). Without this comparison, the claim that the study provides "actionable insights" is weak; it's unclear if these prompt tweaks are better, worse, or complementary to existing methods.
3.  **Evaluate against more sophisticated, realistic adversarial strategies.** The misleading agent uses a simplistic, static error (e.g., `f=25/Re`). To credibly assess "adversarial robustness," the study must test against adaptive adversaries that use correct reasoning with a single seeded mistake, multi-step persuasion, or attacks that exploit social dynamics (e.g., flattery, appeals to authority). The current setup primarily tests error detection, not adversarial robustness.
4.  **Include a no-adversary baseline for all multi-agent configurations.** The paper shows that adding more agents can reduce efficiency (more "no decision" outcomes). However, it does not systematically report the **correct solution rate** for configurations with *only* supportive agents (e.g., "SS", "SSS"). This is critical to assess the trade-off: does a prompt that increases rejection of bad advice also harm performance when advice is good? Without this, the "design choices" cannot be properly evaluated.

### Deeper Analysis Needed (top 3-5 only)
1.  **Analyze the *mechanisms* of failure and success in agent reasoning.** The paper reports outcome rates (misled/rejected) but provides only anecdotal conversation snippets. A systematic qualitative analysis or quantitative probing is needed to answer: *Why* is a non-concise prompt better? Does the leader actually perform independent verification? When misled, does the leader fail on physics knowledge, arithmetic, or defer to perceived authority? This analysis is essential to move from correlations (prompt X improves rate) to understanding.
2.  **Quantify error propagation through the conversation chain.** The claim about "error propagation" is central but not measured. The analysis should track how the initial error evolves: Does the leader reiterate it, compound it with new mistakes, or partially correct it? This requires parsing intermediate reasoning steps, not just final decisions, to understand the dynamics of vulnerability.
3.  **Formalize the relationship between task complexity and robustness.** The paper notes tasks with "easily confusable numerical variations" are more vulnerable, but this is only loosely demonstrated. A deeper analysis should define and measure complexity metrics (e.g., number of reasoning steps, ambiguity of key parameters, similarity of plausible wrong answers) and correlate them with misleading rates across tasks. This would transform an observation into a testable hypothesis.

### Visualizations & Case Studies
1.  **Visualize the decision trajectory for key experiment categories.** Create diagrams or flowcharts for conversations in critical conditions (e.g., "MSM" vs "SMM", authoritative vs. concise leader). These should map the sequence of claims, calculations, and challenges, highlighting the point where the leader accepts or rejects the error. This would make the "first-mover effect" and persuasion process concrete and scrutable.
2.  **Present side-by-side case study comparisons of critical failures.** Select representative examples where the same task and agent setup led to both a "misled" and a "rejected" outcome. Annotate these conversations extensively to contrast the pivotal reasoning differences. This would directly expose the failure modes the method is meant to address.

### Obvious Next Steps
1.  **Scale problem complexity within a single domain.** The paper jumps between disparate tasks (pipe flow, beams, graphs). A more controlled approach would have been to vary the complexity *within* the pipe flow problem (e.g., add fittings, consider turbulent flow, include unit conversions) to directly test how robustness degrades with increasing task demands in a cohesive engineering context.
2.  **Implement and test a simple, automated defense based on the main finding.** Given the strong "first-mover effect," a clear next step within the paper's scope would be to implement a simple mitigation like randomizing speaking order or requiring independent solution proposals before discussion, and then test its efficacy across tasks. This would elevate the work from characterization to intervention.
3.  **Report the computational cost and latency implications of robust configurations.** The paper notes that non-concise prompts and more agents lead to longer conversations. For engineering applications where these systems might be deployed, efficiency matters. The paper should report average token counts or latency for different configurations, as a practical constraint on the proposed "actionable insights."

# Final Consolidated Review
## Summary
This paper presents a systematic, empirical study of adversarial robustness in LLM-based Multi-Agent Systems (MAS) applied to engineering problem-solving. Using a controlled, hierarchical setup with one leader and one or more misleading/supportive advisors, it quantifies how factors like prompt wording, task complexity, agent communication order, and advisor personalization affect the system's susceptibility to propagated errors across four representative engineering and math tasks.

## Strengths
- **Comprehensive and statistically rigorous experimental design:** The study systematically varies a large number of factors (leader/advisor prompts, task types, agent numbers/orders, personalization) against a coherent baseline, uses appropriate statistical tests, and runs a sufficient number of trials (≥30) per condition, supported by a convergence analysis. This provides a solid empirical foundation.
- **Provides actionable, domain-specific design insights:** The work moves beyond generic MAS security to deliver concrete, practical findings for engineering contexts, such as the critical importance of explicit warnings in the leader's prompt, the robustness benefit of a non-concise leader style, and the strong "first-mover effect" where the first speaking agent disproportionately influences the outcome.

## Weaknesses
- **Simplistic and non-adaptive adversarial model:** The "misleading" agent is constrained to injecting a single, pre-defined factual error (e.g., `f=25/Re`). This does not test against more realistic, adaptive adversaries that could employ multi-step persuasion, correct reasoning with a seeded mistake, or dynamic strategies, limiting the study's relevance to broader threat models.
- **Lacks mechanistic explanation for observed vulnerabilities:** The paper is primarily correlational, identifying which factors improve or degrade robustness but not analyzing *why*. It does not systematically examine the reasoning traces to understand failure modes (e.g., whether the leader defers to perceived authority, fails at arithmetic, or misses a physics principle), missing an opportunity to move from empirical patterns to explanatory insight.
- **Claims of "representative" engineering tasks are not well-justified:** While four task classes are used, they are relatively simple, closed-form calculations. The paper does not argue for why these specific tasks are representative of the broader, often iterative and ambiguous, workflows in real-world engineering, limiting the generalizability of its conclusions.

## Nice-to-Haves
- Testing the core phenomena (e.g., first-mover effect, prompt sensitivity) across a more diverse set of LLM families (beyond GPT variants) would strengthen claims about general vulnerabilities in "LLM-based MAS."
- A direct comparison of the proposed prompt-based mitigations against established MAS defense baselines (e.g., consensus mechanisms, randomized smoothing) would better contextualize the utility of the insights.

## Novel Insights
The paper's primary novel contribution is the systematic application of adversarial robustness analysis to the specific domain of engineering-focused LLM-MAS, confirming that vulnerabilities differ from purely linguistic tasks. Within this scope, it provides the novel empirical finding that personalizing agents (assigning expert roles or names) significantly amplifies the "first-mover effect," making the system more vulnerable when a misleading agent speaks first but more robust when a supportive agent does. Beyond this domain-specific application and amplification effect, the core concepts of order sensitivity and prompt engineering are extensions of prior MAS security work.

## Suggestions
- Perform a deeper, systematic analysis of the conversation transcripts to categorize the mechanisms of failure and success (e.g., independent verification, deference to authority, arithmetic error) rather than relying on anecdotal examples. This would transform correlational findings into explanatory insight.
- To address the simplistic adversary concern, design a follow-up experiment with at least one more adaptive adversarial strategy (e.g., an agent that introduces an error in an intermediate calculation step rather than just the final formula) to test the limits of the identified robust configurations.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 2.0, 0.0]
Average score: 1.5
Binary outcome: Reject
