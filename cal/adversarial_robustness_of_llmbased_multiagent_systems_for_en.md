=== CALIBRATION EXAMPLE 25 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title is accurate and appropriately scoped: the paper studies adversarial robustness of LLM-based multi-agent systems on engineering-style tasks, not generic MAS robustness.
- The abstract clearly states the problem, the experimental setup, and the main qualitative findings. It also identifies the task types used.
- The strongest claim in the abstract is that this is the “first systematic study” of adversarial robustness of LLM-based MAS in engineering contexts. That may be plausible, but it is a literature claim that is hard to verify from the paper itself; ICLR reviewers would expect a more cautious phrasing unless the novelty is rigorously established.
- The abstract promises “actionable insights” and “design choices… significantly improve resilience,” which is supported by the reported prompt and order effects, though the paper should better separate statistical significance from practical significance.

### Introduction & Motivation
- The motivation is reasonable and timely: engineering tasks require numerical and formal correctness, so misleading agent behavior can have higher stakes than in generic conversational benchmarks.
- The gap in prior work is identified in a useful way: prior MAS security papers focus on generic collaboration attacks, while this paper asks how adversarial influence behaves in engineering problem solving.
- The contribution statement is broadly accurate, but somewhat diffuse. The introduction lists task types, prompting, communication order, and error injection, but it does not clearly distinguish the paper’s main scientific claims from the specific experimental factors.
- The introduction slightly overreaches in claiming broader safety relevance from synthetic tasks. The relevance is real, but the paper should be more careful that these are proxy experiments, not validated engineering-deployment studies.
- For ICLR standards, the problem is interesting enough, but the framing would be stronger if it more explicitly articulated what new general principle is learned beyond “prompting and order matter.”

### Method / Approach
- The basic setup is understandable: a leader agent interacts with misleading and sometimes supportive advisors, and the outcome is judged by whether the leader adopts the wrong solution. The baseline pressure-loss task and the adversarial friction-factor prompt are described concretely.
- However, the method is only partially reproducible as presented. The paper mixes many experimental axes, but the exact protocol for each condition is not always explicit:
  - how many turns each interaction had in each trial,
  - how “preliminary decision” and “rethinking” were triggered,
  - whether the same random seeds / same task instances were reused,
  - how prompts were composed when multiple advisors were present,
  - and how outcomes were classified if the leader gave a partially correct but numerically off answer.
- There is also some ambiguity in the controlled-adversary design. The misleading advisor is instructed to mislead, but the paper does not discuss whether some results are driven by prompt compliance rather than adversarial robustness in a stronger sense.
- A key methodological concern is the heavy reliance on a single model family for the main study, GPT-4o mini, with some additional model variations in the appendix. That is fine for a focused benchmark, but the conclusions about MAS robustness are still model-specific.
- The paper asserts significance-testing and confidence intervals, but the statistical treatment is not fully transparent. For instance, the paper reports Fisher’s exact test for multiple binary outcomes, but there is no discussion of multiple-comparison correction across the many conditions in Tables E.6–E.15. Given the large number of hypothesis tests, this matters.
- There are also some apparent conceptual inconsistencies:
  - In Section 3.1, the “baseline configuration” is described as a two-agent hierarchical MAS for the pressure-loss task, but later many tasks and advisor counts are studied. The method section should more cleanly separate the base protocol from the full experimental grid.
  - The “correctness” metric appears secondary, but in engineering contexts it is important; the paper should define how it handles near-correct numeric answers or alternative valid derivations.
- For theoretical claims, there are no formal proofs, so the concern is not correctness of derivation but clarity of experimental design. The current description is adequate for a prototype study, but not yet at the level of methodological rigor ICLR typically expects for a strong empirical paper.

### Experiments & Results
- The experiments do test the main claims: prompt design, task type, number/order of advisors, and personalization all affect whether the leader is misled.
- The baselines are reasonable within the paper’s scope, but the comparison set is incomplete from an ICLR perspective. The paper mostly compares within its own design space; there are no strong external baselines for MAS robustness methods, such as alternative coordination rules, debate-style aggregation, majority voting variants, or a non-agent single-model control on the same tasks.
- Several results are interesting and specific:
  - Table E.6 shows large sensitivity to leader prompt variants, with “Authoritative” reaching 0% misleading rate and “No warning” reaching 100%.
  - Table E.11 and E.12 show task-dependent vulnerability, especially for the division and “misleading axis” beam settings.
  - Table E.14 and E.15 suggest a strong first-mover / role-framing effect.
- But the experimental support is weaker than it first appears because the paper often draws broad conclusions from very small absolute counts. With 30 trials per condition, a change from 0 to 3 or 4 trials can look dramatic in percentage terms, and many conclusions are based on single-condition comparisons.
- The paper does not report confidence intervals in the main result tables, only p-values in the appendix tables. For an ICLR paper, uncertainty quantification should be more visible in the main text.
- There is no clear ablation disentangling prompt length/style from semantic content. For example, in Section 4.1 and Table E.6, “not concise,” “authoritative,” and “collaborative” may improve robustness because they change reasoning depth, but the paper does not isolate whether this comes from longer answers, more self-checking, or simple prompt tone.
- Likewise, the first-mover effect in Section 4.3 is plausible, but it is not isolated from ordering confounds: the first speaker also gets to frame the problem and set the context. A stronger design would compare order while holding content and verbosity fixed.
- The “personalization” results in Section 4.4 are suggestive, but the interpretation is not fully grounded. The claim that names and expertise “amplify credibility” is reasonable, yet the paper does not measure perceived credibility directly.
- The appendix includes additional model experiments showing GPT-4o and o3-mini are much more robust than GPT-4o mini. This is useful, but it also raises an important issue: many main conclusions may depend strongly on model capability rather than just MAS structure. That should be discussed more centrally.
- Overall, the results support a moderate conclusion: adversarial robustness in MAS is sensitive to prompt and interaction design. They do not yet support a strong general theory of engineering MAS robustness.

### Writing & Clarity
- The paper is mostly understandable, but some sections are confusing in ways that matter scientifically:
  - The method section is split awkwardly between agent design and the baseline problem setup, making the protocol harder to track.
  - Some arguments in Sections 4.2 and 4.3 rely on interpretive speculation (“first mover effect,” “leader keeps searching for one that does not exist”) without clear evidence from the conversations.
  - The relationship between the main text and appendix tables is not always easy to follow; the main narrative often references statistical results that are only fully visible in the appendix.
- Figures and tables appear informative overall, especially Figure 3–6 and Tables E.5–E.15, but the paper would benefit from more direct in-text interpretation of the main figures rather than relying on broad verbal summaries.
- One clarity issue that affects understanding: several experimental names are overloaded or similar, especially in the advisor prompt variations and the multi-advisor configurations. The paper would be easier to parse if it explicitly grouped experiments by the underlying hypothesis each tests.

### Limitations & Broader Impact
- The paper does acknowledge some limitations, especially that prompt-variation combinatorics are large and that the behavior may be non-linear.
- However, the most important limitations are underdeveloped:
  - The benchmark is synthetic and narrow, centered on a small set of engineering-math tasks.
  - The misleading advice is often very stylized and may not reflect realistic adversarial settings.
  - The study does not examine distribution shift, tool use, or integration with external solvers, which are common in engineering workflows.
  - The same model family is used for most main experiments, so generalization is uncertain.
- A major limitation is that “engineering problems” here are mostly toy problems or textbook problems, not tasks with actual design constraints, simulation, or safety-critical feedback loops. The paper should say that more explicitly.
- On broader impact, the paper is appropriately cautious and does not appear harmful. Still, since it studies how to mislead agent systems, it would be good to discuss defensive implications more concretely, such as whether the findings suggest safer routing, verification, or adversarial-agent detection mechanisms.
- The ethics statement is fine, but I would have liked a stronger statement about the risk of overgeneralizing from synthetic settings to real engineering deployment.

### Overall Assessment
This is a relevant and timely empirical paper with a clear thesis: LLM-based multi-agent systems can be surprisingly brittle under adversarial influence, and robustness depends on prompt design, task structure, and agent ordering. The experiments are generally coherent and several results are genuinely interesting, especially the strong prompt sensitivity and first-mover effects. That said, at an ICLR acceptance bar, the paper currently feels more like a well-motivated exploratory benchmark study than a fully rigorous or broadly generalizable systems paper. The main concerns are limited external baselines, modest sample sizes per condition, incomplete methodological detail, and somewhat overextended generalizations from synthetic tasks. The contribution is promising and likely useful, but the paper would need stronger experimental rigor and a sharper articulation of the general principle learned to be a strong ICLR acceptance.

# Neutral Reviewer
## Balanced Review

### Summary
This paper studies adversarial robustness of LLM-based multi-agent systems (MAS) on a set of engineering and math problems, using controlled misleading agents and variations in prompts, task structure, agent count/order, and model settings. Its main claim is that robustness is highly context-dependent: task complexity, the subtlety of the injected error, and the communication order among agents strongly affect whether the system is misled.

### Strengths
1. **Timely and relevant problem setting for ICLR.**  
   The paper addresses robustness of LLM agentic systems, a topic of clear interest to ICLR, and focuses on engineering tasks where numerical and formal correctness matter more than in generic dialogue settings.

2. **Systematic experimental axis exploration.**  
   The study varies multiple factors in a structured way: leader prompt design, advisor prompt design, task type, number/order of agents, personalization/naming, and leader model parameters. This breadth supports the paper’s central thesis that robustness depends on system design choices, not just model quality.

3. **Concrete evidence of large robustness variation.**  
   The reported results show wide swings in misleading rate, from near 0% to 100%, across conditions. For example, leader prompt changes such as “No warning” versus “Authoritative” dramatically alter rejection/misleading behavior, and the number/order of agents also changes outcomes substantially.

4. **Engineering-specific examples make the setting intuitive.**  
   The paper uses recognizable problems such as Darcy-Weisbach pressure loss, cantilever beam deflection, graph traversal, and simple math, which helps ground the discussion in a domain where incorrect reasoning can have practical consequences.

5. **Includes statistical testing and reproducibility details.**  
   The appendix reports per-condition tables, statistical tests (Fisher’s exact test, Mann-Whitney U), and trial counts, which is better than many purely qualitative MAS papers.

### Weaknesses
1. **Contribution is largely empirical and somewhat incremental relative to existing MAS robustness work.**  
   The paper claims to be the “first systematic study” in engineering contexts, but the experimental design closely follows known ideas from prior MAS robustness and prompt-attack literature: varying adversarial prompts, agent order, number of agents, and model choice. The novelty is mainly domain transfer, which may be useful, but is not by itself a strong ICLR-level methodological contribution unless the engineering setting reveals fundamentally new behavior.

2. **Limited methodological rigor in causal interpretation.**  
   Many conclusions are phrased causally or mechanistically—for example, that non-conciseness helps because the leader “solves the problem on their own first,” or that names amplify the first-mover effect. However, the paper mostly reports correlations from prompt ablations and does not provide deeper controlled analysis to justify these explanations.

3. **Evaluation protocol may conflate correctness, agreement, and persuasion.**  
   “Misled” is defined as the leader’s final decision matching the misleading advisor’s answer, which is a narrow operationalization. In several tasks, the advisor’s answer may be close, partially correct, or merely stylistically persuasive, and agreement with the advisor may not cleanly capture adversarial success. Conversely, rejecting the advisor does not necessarily imply robust reasoning if the leader reaches the correct answer for the wrong reason.

4. **Task suite is useful but small and somewhat hand-crafted.**  
   The benchmark consists of a few toy-sized engineering and math problems, many of which are specifically constructed around a single misleading assumption or numerical confusion. This limits claims about “engineering MAS” more broadly, especially for realistic workflows involving longer chains of computation, tool use, or external verification.

5. **Generalization is limited by dependence on one main base model and a fixed interaction template.**  
   Most experiments are run with GPT-4o mini in a specific hierarchical leader-advisor protocol. Although additional model variants are reported, the study does not deeply explore whether the findings persist under different MAS architectures, stronger verification mechanisms, asynchronous communication, or tool-augmented reasoning.

6. **Some results are hard to interpret without stronger controls.**  
   The paper observes that more agents can reduce efficiency and that some all-supportive configurations perform worse than mixed ones, but the study does not fully disentangle whether this is due to cognitive overload, prompt conflict, or artifacts of the specific conversation protocol. This weakens the explanatory power of the findings.

7. **Clarity and presentation are uneven.**  
   The core idea is understandable, but the manuscript sometimes overstates conclusions relative to the evidence and includes speculative interpretations without sufficient support. For an ICLR submission, reviewers would likely expect more polished framing of hypotheses, stronger ablations, and a cleaner articulation of what is newly learned scientifically.

### Novelty & Significance
**Novelty:** Moderate. The paper’s novelty lies primarily in moving adversarial MAS robustness evaluation into engineering-oriented tasks and showing that robustness patterns differ across task types and interaction structures. However, the experimental methods and high-level claims are close to existing adversarial MAS literature, so the paper is not strongly novel methodologically.

**Significance:** Moderate. The topic is important for ICLR, especially as agentic LLM systems are increasingly used in technical workflows. The finding that prompt wording, communication order, and task formulation can dramatically affect robustness is practically relevant, but the evidence currently supports incremental insights more than a major conceptual advance.

**Clarity:** Moderate. The paper is generally understandable and the tables are informative, but the narrative would benefit from sharper hypothesis framing and more disciplined interpretation of results.

**Reproducibility:** Fair to good. The paper provides prompts, model settings, trial counts, and statistical procedures, which is positive. However, full reproducibility still depends on API model versions and exact conversation handling, and the paper would be stronger with released code, seeds, raw transcripts, and a more explicit experimental specification.

**ICLR acceptance bar assessment:** The paper is relevant and empirically grounded, but ICLR typically expects either strong methodological novelty, a clearly new scientific insight, or especially convincing experimental rigor. As written, this feels closer to a solid application/benchmarking study than a clearly above-bar ICLR contribution.

### Suggestions for Improvement
1. **Strengthen the core scientific claim with sharper hypotheses and controlled tests.**  
   Turn the main observations into explicit hypotheses, e.g. about first-mover effects, prompt framing, or complexity of misleading statements, and design targeted experiments to isolate each factor.

2. **Add stronger baselines and alternative MAS architectures.**  
   Compare the current hierarchical leader-advisor setup against debate, vote-based, tool-verified, and self-consistency-style MAS protocols. This would show whether the reported vulnerabilities are general or protocol-specific.

3. **Broaden the benchmark with more realistic engineering tasks.**  
   Include tasks that require multi-step derivations, unit checking, tool use, or external calculation verification. That would make the conclusions more compelling for engineering applications.

4. **Evaluate robustness beyond exact agreement with the adversary.**  
   Introduce richer outcome metrics: numerical error, semantic correctness, calibration, partial credit, or whether the leader independently derives the correct intermediate steps. This would make the notion of “misled” more precise.

5. **Reduce speculative explanation unless supported by targeted evidence.**  
   If claiming that non-conciseness helps because the leader reasons longer, or that names amplify credibility, test those mechanisms directly with conversation-length controls or randomized prompt manipulations.

6. **Report confidence intervals and effect sizes more systematically.**  
   Since many conditions use only 30 trials, exact p-values alone are not enough. Provide confidence intervals, odds ratios, and possibly multiple-comparison correction to support the many pairwise comparisons.

7. **Release a complete reproducibility package.**  
   Provide code, prompts, raw transcripts, parsing scripts, and exact API/model identifiers. For an ICLR audience, this would materially increase confidence in the empirical claims.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add strong baselines against standard multi-agent and single-agent setups on the same engineering tasks. Without comparing to a single GPT-4o/o3 solver, debate-style MAS, majority vote, and a no-adversary control, the claim that “multi-agent collaboration” has distinctive adversarial robustness properties is not convincing for ICLR standards.

2. Evaluate more than one engineering benchmark family with real-world structure and harder instances. The current tasks are mostly toy or hand-constructed arithmetic/physics prompts; without datasets like MATH/engineering word problems, symbolic mechanics sets, or benchmarked code/physics tasks, the paper does not support a general claim about “engineering MAS.”

3. Test stronger and more realistic adversaries, not only fixed prompt-injected wrong formulas. Add adversaries that are adaptive, context-aware, and partially correct, plus non-prompt perturbations such as numerical noise, tool-output corruption, or malicious but plausible derivations; otherwise the robustness story is too narrow to generalize.

4. Include ablations that isolate the mechanism behind the reported gains from prompt wording and agent order. The paper claims effects from warnings, authority, and first-mover order, but does not separate “more reasoning,” “more verbosity,” and “more caution” from genuine robustness improvements.

5. Compare against simple defenses such as verification prompts, answer checking, self-consistency, or external symbolic validation. Without these baselines, the paper cannot show its prompt/ordering recommendations are better than obvious engineering safeguards.

### Deeper Analysis Needed (top 3-5 only)
1. Quantify whether the leader is actually reasoning correctly or just rejecting more often. The paper reports misleading/rejection rates, but ICLR reviewers will want calibration-like analysis of false negatives, false positives, and correctness conditional on reaching a decision.

2. Analyze statistical robustness beyond per-condition p-values. With many conditions and only ~30 trials each, the paper needs confidence intervals, multiple-comparison correction, and effect sizes; otherwise many “significant” differences may be unstable.

3. Explain why some prompt changes produce dramatic swings while similar ones do not. The paper currently offers post-hoc intuition, but not a mechanism-level analysis of which tokens or prompt features change agent behavior, which is needed to trust the conclusions.

4. Separate task difficulty from adversarial susceptibility. Right now the paper conflates “harder task,” “harder wrong answer,” and “more misleading”; a controlled difficulty analysis is needed to show vulnerability is due to adversarial influence rather than task complexity alone.

5. Analyze failure modes when the system makes no decision. High no-decision rates are treated as neutral, but for safety-critical engineering they are a failure mode; the paper should show when abstention is appropriate versus when it reflects deadlock or over-caution.

### Visualizations & Case Studies
1. Add full conversation traces for representative successes and failures across each task type, not just pipe flow. This is necessary to show whether the leader detects the adversarial flaw, defers to the wrong authority, or simply mirrors the first agent.

2. Add a confusion-style breakdown of outcomes by task and adversary type. A matrix showing reject/mislead/no-decision and final correctness would expose whether some settings merely shift errors into abstention.

3. Show sensitivity plots for the number of agents, order, and prompt strength. The current bars hide whether effects are smooth, thresholded, or noisy; ICLR reviewers will expect a clearer picture of how fragile the method is.

4. Visualize how error propagates through the dialogue over turns. A turn-by-turn plot of confidence, agreement, and decision changes would reveal whether robustness comes from genuine verification or just earlier termination.

### Obvious Next Steps
1. Evaluate the method on standard reproducible benchmarks with fixed ground truth and stronger baselines. This is the most direct next step needed to make the claims publishable at ICLR.

2. Test whether the proposed prompt/order heuristics transfer across models and domains. The paper hints at this with a few model variants, but does not establish cross-model robustness or domain transfer.

3. Add a defense-oriented experiment where the system actively verifies advisor claims using symbolic or external computation. For engineering problems, this is the obvious practical extension and the most relevant one for trustworthiness.

4. Study adversarial agents that exploit uncertainty, partial truth, and numerical near-misses rather than a single obvious wrong formula. That would show whether the observed robustness survives more realistic attacks.

# Final Consolidated Review
## Summary
This paper studies adversarial robustness in LLM-based multi-agent systems on a small suite of engineering and math tasks, using a leader-advisor setup where one advisor is deliberately misleading. The main finding is that robustness is highly sensitive to prompt wording, task formulation, agent order/count, and model choice; in some settings the leader is almost always fooled, while in others it reliably rejects the bad advice.

## Strengths
- The paper tackles a timely and relevant problem: whether multi-agent LLM workflows remain reliable on engineering-style tasks where numerical correctness matters. The study is clearly motivated and the engineering examples make the issue concrete.
- The experimental sweep is broad for a single paper: it varies leader prompts, advisor prompts, task type, agent order/count, and model settings. The tables do show large swings in misleading rate, which supports the central claim that these systems are fragile and design-sensitive.
- The appendix includes detailed prompts, many per-condition results, and statistical tests. That is better than typical anecdotal MAS papers and makes the core experiments at least inspectable.

## Weaknesses
- The main limitation is that the study is still a small, synthetic benchmark with hand-crafted toy problems. The “engineering” tasks are textbook-style prompts, often built around a single misleading formula or numerical confusion, so the paper does not yet justify broad claims about real engineering workflows or safety-critical deployment.
- The causal interpretation is too speculative relative to the evidence. Claims like “non-concise leaders reason more on their own,” “names amplify credibility,” or “first mover effect” are plausible, but the paper does not isolate these mechanisms with controlled ablations; much of this is post-hoc narrative built on correlation.
- The evaluation protocol is narrow and somewhat brittle. “Misled” is defined as matching the advisor’s wrong answer, which conflates persuasion, agreement, and correctness, and the paper does not analyze richer outcomes like partial correctness, numerical error magnitude, or whether the final answer is correct for the right reasons.
- Statistical support is weaker than the volume of tables suggests. Each condition uses only about 30 trials, yet the paper reports many pairwise tests without any correction for multiple comparisons. Several dramatic percentage changes are based on very small absolute count differences, so some of the apparent effects may be unstable.
- There is no strong external baseline. The paper mostly compares variations inside its own leader-advisor protocol, but does not benchmark against standard alternatives such as a single-model solver, debate/voting-style MAS, or simple verification defenses. That makes it hard to tell whether the proposed observations are specific to this setup or broadly useful.

## Nice-to-Haves
- Add stronger baselines and simple verification defenses, such as self-consistency, answer checking, or symbolic validation, to place the results in context.
- Report confidence intervals and effect sizes more prominently in the main paper, not only in the appendix.
- Include more representative failure traces beyond the pipe-flow example, especially for the task/order settings that appear most fragile.

## Novel Insights
The most interesting insight is that adversarial robustness in MAS is not a monotonic function of “more agents” or “better collaboration”; the interaction structure itself can dominate behavior. The paper suggests a strong first-mover bias, and also indicates that seemingly superficial prompt changes can radically alter whether the leader critically verifies advice or simply inherits it. Another useful observation is that task structure matters: the system is far more vulnerable when the wrong answer is numerically or structurally close to the right one, which is exactly the kind of failure mode that matters in engineering settings.

## Potentially Missed Related Work
- **Multiagent collaboration attack / debate-style attacks** — relevant because the paper studies misleading agents in MAS and should be positioned more explicitly against prior adversarial collaboration work.
- **On the resilience of LLM-based multi-agent collaboration with faulty agents** — directly relevant as prior work on faulty/adversarial agents in MAS.
- **Assessing and enhancing the robustness of LLM-based multiagent systems through chaos engineering** — relevant as a robustness-focused MAS study with a related experimental mindset.
- **Agents Under Siege** — relevant for prompt attacks on pragmatic multi-agent systems.
- **Randomized smoothing / verification-style defenses for MAS** — relevant as possible robustness baselines the paper does not compare against.

## Suggestions
- Add a compact baseline section comparing the current protocol against at least one single-agent solver, one simple multi-agent aggregation method, and one verification-based defense on the same tasks.
- Strengthen the causal claims by designing controlled ablations that separately vary verbosity, role framing, prompt length, and order while holding content fixed.
- Expand the benchmark with at least a few harder or more realistic engineering tasks where correctness can be checked mechanically, not just by matching a prewritten analytical answer.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 2.0, 0.0]
Average score: 1.5
Binary outcome: Reject
