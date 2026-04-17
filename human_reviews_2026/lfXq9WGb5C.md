# Do Role-Playing Agents Practice What They Preach? Belief-Behavior Alignment in LLM-Based Simulations of Human Trust

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2, 4

## Abstract
As large language models (LLMs) are increasingly studied as role-playing agents to generate synthetic data for human behavioral research, ensuring that their outputs remain coherent with their assigned roles has become a critical concern. In this paper, we investigate how consistently LLM-based role-playing agents' stated beliefs about the behavior of the people they are asked to role-play ("what they say") correspond to their actual behavior during role-play ("how they act"). Specifically, we establish an evaluation framework to rigorously measure how well beliefs obtained by prompting the model can predict simulation outcomes in advance. Using an augmented version of the GenAgents persona bank and the Trust Game (a standard economic game used to quantify players' trust and reciprocity), we introduce a belief-behavior consistency metric to systematically investigate how it is affected by factors such as: (1) the types of beliefs we elicit from LLMs, like expected outcomes of simulations versus task-relevant attributes of individual characters LLMs are asked to simulate; (2) when and how we present LLMs with relevant information about Trust Game; and (3) how far into the future we ask the model to forecast its actions. We also explore how feasible it is to impose a researcher's own theoretical priors in the event that the originally elicited beliefs are misaligned with research objectives. Our results reveal systematic inconsistencies between LLMs' stated (or imposed) beliefs and the outcomes of their role-playing simulation, at both an individual- and population-level. Specifically, we find that, even when models appear to encode plausible beliefs, they may fail to apply them in a consistent way. These findings highlight the need to identify how and when LLMs’ stated beliefs align with their simulated behavior, allowing researchers to use LLM-based agents appropriately in behavioral studies.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents a pre-hoc diagnostic for assessing whether role-playing LLMs’ stated beliefs align with their actual behavior in a standardized Trust Game. It elicits internal beliefs via targeted prompts, runs matched simulations, and quantifies alignment at both the population and individual levels, revealing several notable model-specific findings.

### Strengths
1. The paper advances a pre-hoc paradigm for assessing belief–behavior consistency in LLM role play by pairing explicit belief elicitation with a standardized Trust Game and dual-granularity evaluation (population and individual). This reframing is interesting, removes key limitations of post-hoc validation.

2. Evaluation metric is objective and clear. Pre-hoc belief consistency can be promising in guidance for designing reliable agent studies.

3. The experiments reveal actionable insights, including that providing task context alone does not reliably improve alignment, self-conditioning enhances consistency while external priors degrade it, etc.

### Weaknesses
> External Validity:

The study is confined to a single Trust Game with fixed opponents. Given the breadth of role-play consistency (e.g. some trustworthy or tool-using LLM agent scenarios), this setting may be too narrow and biased.

> Supporting Experiment:

The paper states that pre-hoc is preferable, but does not quantify in what respects or by how much. No comparison with previous researches as in Table 1 are provided.

> Robustness

No repeated experiments seem to be ran. Uncertainty like mean and standard deviation need to be reported for further validity.

> No comparison to LLM-as-Judge.

This paper claims that using fully objective metric as one of its main merits, yet no comparison between these metrics (Spearman correlation, etc) and LLM grader scores are provided.

### Questions
Please refer to the weaknesses section.

### Soundness
2

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
3

### Summary
This paper investigates the "belief-behavior consistency" of LLM-based role-playing agents and proposes a "pre-hoc" evaluation framework to measure whether an agent’s simulated actions ("how they act") align with its stated beliefs ("what they say") about its assigned role. The motivation is to develop a diagnostic tool that can detect simulation flaws before expensive large-scale synthetic data generation, addressing the limitations of existing post-hoc evaluation methods. Using the Trust Game as a testbed and an augmented persona dataset, the authors test consistency at two levels. At the population level, they compare the LLM’s elicited beliefs (e.g., the statistical correlation between “Age” and “trust”) with the actual simulated outcomes and examine whether consistency improves when the model is “reminded” of its own beliefs (self-conditioning) or guided by “imposed priors” (researcher-defined beliefs). At the individual level, they test whether a single agent can accurately forecast its own future actions over multiple game rounds against fixed opponent strategies. The key findings reveal that LLM agents exhibit systematic inconsistencies; providing task context during belief elicitation does not help; self-conditioning improves consistency for Llama models but fails catastrophically for Gemma; attempts to impose external priors undermine consistency, indicating poor controllability; and individual-level forecasting accuracy declines over longer time horizons.

### Strengths
1. Addresses a Critical Problem: The paper tackles a fundamental and timely question for the agent-based modeling community—whether LLM agents are reliable. The distinction between post-hoc validation (which is costly) and the proposed pre-hoc diagnostic framework is an important contribution.

2. Novel and Comprehensive Framework: The use of belief-elicitation as a diagnostic tool for internal consistency is novel. The two-level analysis—evaluating both population-level statistical correlations and individual-level forecasting—offers a rigorous and comprehensive methodological perspective.

3. Important (and Honest) Negative Results: The paper’s main findings are negative but highly valuable: the agents are inconsistent. The failure of imposed priors to control behavior is a critical, non-trivial discovery that challenges the current paradigm of treating LLMs as controllable simulators.

4. Strong Experimental Testbed: The Trust Game is an excellent experimental choice because it provides a quantifiable behavioral output (dollars sent). The augmentation of the persona bank with Big Five personality traits is also well-motivated, as these traits correlate with trust and enable richer attribute-level testing.

### Weaknesses
1. Failure to Analyze Inconsistency: While the paper identifies systematic inconsistencies, it lacks a deep technical explanation. It remains unclear whether the “belief” (a text generation) and the “action” (another text generation) originate from distinct parts of the model’s distribution. The failure of linkage between these two processes—central to the paper’s premise—is observed but not explained.

2. Under-Analyzed Controllability Failure: The most significant finding is the failure of imposed priors, which suggests a deeper issue in agent controllability. However, the paper does not explore why this happens. It could be that persona attributes (e.g., “Age: 65+”) or internal biases overpower explicit instructions, but this haypothesis is not empirically examined.

3. Unstable “Ground Truth” (The Belief Itself): The framework assumes that elicited beliefs are stable reference points, but the paper shows that beliefs vary depending on elicitation method (e.g., CTX+TR vs. CTX+$). If the belief is unstable, the observed inconsistency may reflect differences between elicitation contexts rather than genuine misalignment between belief and behavior.

4. Unvalidated Persona Embodiment: The study assumes that labeling a persona with traits such as “Conscientiousness: High” makes the model embody that trait. However, there is no evidence supporting this assumption. The analysis effectively correlates a textual label—not an internalized trait—with behavioral output, undermining the validity of the results.

5. Conflicting and Uninterpretable Metrics: The two population-level metrics—Spearman correlation (ρ) for ranking and absolute effect-size discrepancy (|Δη²|) for magnitude—sometimes yield contradictory results. For instance, one model may show high correlation but poor effect-size alignment. The authors’ interpretation that these metrics have “complementary strengths” feels like an attempt to rationalize inconsistency rather than clarify it.

6. Unjustified Change in Methodology: While the population-level analysis uses standard prompting, the individual-level analysis suddenly introduces the ReAct framework, creating a methodological confound. Without controlling for this change, it is impossible to determine whether forecasting degradation stems from time horizon effects or the complexity of ReAct prompting.

### Questions
1. You demonstrate that the elicitation strategy (e.g., CTX+TR vs. CTX+$) fundamentally alters the elicited belief. If the belief is not a stable latent construct but merely an artifact of prompt formulation, how can it serve as a reliable ground truth for evaluating consistency?

2. Your most critical finding is the failure of imposed priors, which challenges the premise of using LLM agents for counterfactual simulation. Why do you think this occurs? Do you have evidence that pretrained biases (e.g., about age) override in-context instructions?

3. You analyze Big Five personality traits such as “Conscientiousness.” What evidence supports that the agents genuinely embodied these traits beyond having the textual label? Without behavioral verification, how can this analysis be valid?

4. Why introduce the ReAct framework for individual-level analysis when it was not used at the population level? Could the observed degradation in forecasting accuracy simply result from this methodological shift rather than genuine long-term inconsistency?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper studies whether LLM “role-playing” agents act in ways consistent with their stated beliefs. Using a Trust Game setup, the authors elicit beliefs about how persona attributes affect behavior, then compare those beliefs to simulated actions at (i) the population level (attribute–behavior correlations) and (ii) the individual level (self-forecast vs. subsequent actions). They introduce metrics (rank-ordering via Spearman ρ and effect-size via η²) and probe design choices (context during elicitation, target construct vs. dollar predictions, self-conditioning vs. imposed priors). Core findings: providing task context during belief elicitation does not systematically improve agreement, self-conditioning helps some models but not others, imposed priors often reduce agreement, and individual-level self-forecasts degrade with longer horizons (rounds).

### Strengths
1. Two-level evaluation: clean split between population-level correlations and individual-level forecasting.

2. Clear metricization: explicit definitions of rank-consistency and effect-size discrepancy, grounded in ANOVA η² and Spearman.

3. Topical and timely: addresses an important reliability question for LLM-based social simulations that are increasingly used to inform research and policy.

### Weaknesses
1. The three elicitation formats target different constructs and prompt shapes alter outputs by themselves. Treating abstract trust ratings and concrete dollar predictions as one belief concept blends measurement artifacts with true belief signals, which undermines the validity of belief–behavior consistency claims.


2. Rank correlations are computed over very few discrete levels and are therefore highly unstable. Sample sizes are modest relative to the number of attributes, and the paper does not report confidence intervals, bootstraps, or any control for multiple comparisons. Extreme correlation values are likely to be sampling noise rather than robust effects.


3. The empirical scope is too narrow to support general claims. Environments are limited to a single stylized trust game with scripted counterparts, ablations are sparse, and sensitivity analyses on decoding, temperature, prompt format, and parsing choices are largely missing. Model diversity is especially limited: the study covers only a few base/instruction models and omits contemporary reasoning-oriented systems and alternative families and sizes that could materially change outcomes. Without broader tasks, stronger ablations, multiple seeds, and a more representative model suite, the conclusions about belief–behavior consistency in LLM agents remain under-substantiated.


4. The use of one way ANOVA eta squared on bounded integer outcomes with likely heteroskedastic and non normal residuals is not well justified. Ordinal or nonparametric measures would better match the data generating process. As is, effect size comparisons may reflect modeling mismatch rather than substantive differences.


5. Imposed priors are described only by target correlations with original beliefs, with construction details left unclear. Self conditioning also changes prompt length and structure, introducing confounds. The content of illustrative priors risks embedding questionable group statements, raising concerns about realism and ethics in the intervention setup.

### Questions
The same as the weaknesses

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
This paper studies a very specific and under-examined problem in LLM-based agent simulation: belief–behavior inconsistency — cases where an LLM agent can state what a persona “would” do in a social game, but then does something else when it actually role-plays that persona. The authors instantiate this question in the Trust Game, extend GENAGENTS-style personas with extra attributes, and propose an evaluation protocol that (i) first elicits the model’s belief about expected actions for a given group/persona, (ii) then runs the same model as a role-playing agent in the exact same game, and (iii) compares the two via rank correlation and effect-size deviation. They further test multi-turn settings and simple “self-conditioning” (feeding elicited beliefs back to the agent) to see whether consistency can be improved. From this angle, the problem is interesting and timely: a lot of current “LLM-as-population” work implicitly assumes belief→behavior alignment, but very few papers make this assumption measurable.

### Strengths
1. The paper makes inconsistency itself the object of study. Prior social/agent-simulation work mostly evaluates post-hoc realism of large-scale generated interactions; here the authors propose an ex-ante check: for the same model and same persona, do stated beliefs match realized actions? 

2. The proposed protocol is more structured than most existing role-play demos: they do population-level (group ordering + effect size) and individual-level (multi-round forecasting vs. actual action) evaluation, and they quantify both with explicit metrics, not just qualitative examples. 

3. The empirical findings are non-trivial: (i) adding more game context to the belief prompt does not guarantee better consistency, (ii) giving the model its own elicited beliefs back sometimes helps but is far from perfect, and (iii) trying to override the model with researcher-imposed priors can actually reduce consistency.

### Weaknesses
1. Prompt-only persona elicitation. All persona/belief acquisition is done via prompting the same model, with small prompt variants. That means part of the observed “inconsistency” could simply be prompt sensitivity rather than a deeper gap between representation and behavior. The paper acknowledges variations in context, but does not run a proper prompt-ablation / multi-prompt robustness study, nor does it try a non-prompt conditioning method (e.g. retrieved persona memory, lightweight adapter). This weakens the causal claim that “LLM role-playing agents are inherently inconsistent.”

2. Single environment. All results are in the Trust Game — a very clean but very narrow dyadic setting with numeric actions. It is not fully clear whether the proposed belief–behavior metrics would transfer to other social tasks (e.g. ultimatum, public goods, persuasion/dialogue). At the moment, it is safest to interpret the benchmark as “state-of-the-art for belief–behavior consistency in trust-game-style role play,” not more broadly.

3. The paper argues that “belief = behavior” is the right target for role-playing agents, but it never shows how consistent real people are on the exact same trust-game setting. Because humans are known to have attitude–behavior gaps, we don’t know whether the inconsistencies they observe in LLMs are actually worse than, similar to, or even better than human behavior. Without that baseline, the results are harder to interpret.

### Questions
see weaknesses

### Soundness
3

### Presentation
3

### Contribution
2
