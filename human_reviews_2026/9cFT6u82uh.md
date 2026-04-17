# Analytica: Soft Propositional Reasoning for Robust and Scalable LLM-Driven Analysis

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Large language model (LLM) agents are increasingly tasked with complex real-world analysis (e.g., in financial forecasting, scientific discovery), yet their reasoning suffers from stochastic instability and lacks a verifiable, compositional structure. To address this, we introduce **Analytica**, a novel agent architecture built on the principle of **Soft Propositional Reasoning (SPR)**. SPR reframes complex analysis as a structured process of estimating the soft truth values of different outcome propositions, allowing us to formally model and minimize the estimation error in terms of its bias and variance.
Analytica operationalizes this through a parallel, divide-and-conquer framework that systematically reduces both sources of error. To reduce bias, problems are first decomposed into a tree of subpropositions, and tool-equipped LLM *grounder agents* are employed —including a novel Jupyter Notebook agent for data-driven analysis—that help to validate and score facts. To reduce variance, Analytica recursively synthesizes these grounded leaves using robust linear models that average out stochastic noise with superior efficiency, scalability, and enable interactive "what-if" scenario analysis.
Our theoretical and empirical results on economic, financial, and political forecasting tasks show that Analytica improves 15.84\% accuracy on average over diverse base models, achieving 71.06\% accuracy with the lowest variance of 6.02\% when working with a Deep Research grounder. Our Jupyter Notebook grounder shows strong cost-effectiveness that achieves a close 70.11\% accuracy with 90.35\% less cost and 52.85\% less time. Analytica also exhibits highly noise-resilient and stable performance growth as the analysis depth increases, with a near-linear time complexity, as well as good adaptivity to open-weight LLMs and scientific domains.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a method of structural reasoning and forecasting called Analytica, based on soft propositional reasoning. The method breaks the task up into a tree of sub-propositions with an Analyzer, and probabilistically analyzes and answers leaf queries with a Grounder (Web search, deep research, Jupyter Notebook). The query results are then aggregated with a synthesizer, both linearly and with a more complicated LLM logic process. This approach aims to minimize both bias and variance, with bias minimization motivated by the power of grounders on more simple tasks and the heuristic assumption that more atomic propositions carry less bias than their more complex counterparts. Variance is managed through the linear synthesis of probabilities with (assumed) minimal covariance throughout the reasoning tree. Finally this paper empirically tests their approach, showing both cost, time and accuracy improvement over existing baselines.

### Strengths
This paper presents a well-motivated and clearly framed approach to an important problem: how to structure LLM-driven reasoning to improve both accuracy and computational efficiency.

Originality: While the individual components - task decomposition and result aggregation - draw on prior ideas, the integration into a unified “Soft Propositional Reasoning” framework is novel and conceptually appealing. The work offers a formulation of structured reasoning that can operate with black-box LLMs, extending applicability to settings where model internals are inaccessible. This synthesis of theoretical framing and modular architecture has utility within the current reasoning-agent literature.

Quality: The paper demonstrates empirical validation, testing multiple variants of the core algorithm and comparing them to strong baselines. The experiments are diverse and convincingly show that the proposed method improves both accuracy and computational efficiency, and supporting the authors’ claims about bias–variance tradeoffs.

Clarity: The introduction and framing are particularly strong, offering a clear motivation and well-scoped research question. Figure 1 effectively summarizes the system’s architecture and serves as a helpful reference throughout the paper. Although some experimental details could be clearer, the paper overall maintains a coherent narrative and logically connects earlier sections.

Significance: The ability to coordinate structured reasoning over black-box models has broad implications for scaling LLM applications in analysis, forecasting, and decision support. The framework’s emphasis on cost–accuracy tradeoffs and robustness under modular substitution makes it practically valuable, especially as computational efficiency becomes increasingly central in large-scale LLM deployment.

### Weaknesses
Clarity and Readability of Figures:
- Figure 6 contains overlapping text, which makes key labels difficult to read.
- Figure 2 is visually dense and difficult to interpret; the motivation for including each column and how they relate to the overall analysis should be made explicit. I also find there are too many variations to analyze coherently, a more compact version may be easier to understand. 
- More generally, the empirical section would benefit from clearer organization — e.g., separating results by contribution type (efficiency, accuracy, stability) — and simplifying figure layouts to highlight the main findings.

Novelty and Framing of Contributions:
- The individual components of the framework—decomposing tasks into subpropositions and aggregating subresults—are not individually novel, having clear precedents in prior work such as Tree of Thoughts (Long, J et al. 2023) and Question Decomposition Tree for Answering Complex Questions over Knowledge Bases (Huang et al., 2023).
- The paper could better emphasize what is genuinely new about their integration or about the soft propositional reasoning perspective, possibly by formalizing how Analytica’s synthesis differs from prior decomposition or ensemble reasoning methods.

Theoretical Rigor:
- The theoretical sections (particularly §4.2–4.4) rely heavily on heuristic reasoning presented as if more formally derived. For example, the claim that bias decreases with decomposition depth is asserted but not proven, and “smaller” propositions are not formally defined.
- These sections could be improved by (a) explicitly labeling heuristic arguments as such and emphasizing empirical adherence or (b) adding formal assumptions or bounds where possible

Empirical Structure:
- Although the results are compelling, the presentation could better connect each experiment (and corresponding figure) to a specific hypothesis derived from the theory (e.g., variance reduction, robustness to model scale) or a contribution highlighted earlier in the paper.

### Questions
With clarification of the following questions and addressing of the outlined weaknesses, I would be willing to increase my score. 

- Could the authors clarify whether Analytica primarily performs reasoning (logical inference) or prediction (forecasting)? These are conceptually distinct, and understanding which is central to the framework would clarify its intended contribution.

- Who is the target user or application domain for Analytica? A discussion of the framework’s practical utility and deployment scenarios would help contextualize its significance.

- Several steps in the theoretical sections appear heuristic but are presented as if formally justified. Could the authors explicitly identify which parts are heuristic assumptions versus those supported by formal derivation or proof?

- The approach seems to assume independence among subclaims, which may not hold in many reasoning tasks. Do the authors have proposed methods for modeling or mitigating dependencies between subclaims?

- Has Analytica been evaluated with smaller or weaker LLMs, particularly for the decomposition stage, even if grounding and synthesis use stronger models? This would clarify whether the framework depends on scale or transfers across model capacities.

- The Jupyter Notebook–based grounding method is described briefly yet achieves the strongest results. Could the authors provide a more detailed example or description in the main text to illustrate how this component operates?

- Was this approach evaluated on non-predictive/financial market tasks? Was the dataset compiled by the authors or another source, how did the authors ensure it is representative of the tasks the paper claims to perform well on?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a hybrid reasoning framework that combines propositional logic with statistical estimation to tackle real-world forecasting tasks. events. It decomposes queries into a tree of soft-valued propositions, grounds leaves with data tools (including a Jupyter agent), and combines beliefs via a learned linear model. On 736 finance/politics tasks it attains ≈71 % accuracy, outperforming chain-of-thought and deep-research baselines while scaling near-linearly with tree size. Contributions are the SPR formalism, modular architecture, and reproducible cost/robustness analyses—incremental but solid.

### Strengths
The design of SPR is welcoming. SPR departs from pure text-based prompting, providing a middle ground between symbolic and neural reasoning. Sec. 3 Eq. (1) decomposes MSE into bias/variance, offering a clear objective.

It offers modular architecture with strong ablations. Table 2 compares three synthesis rules; Linear achieves best accuracy/variance trade-off. Sec. 5.3 noise-injection robustness confirms Linear rule stability (Fig. 5).

Table 1 reports 54× node growth vs. only 12× wall-clock increase, supporting near-linear parallel scaling, which provides good evidence for scalability.

### Weaknesses
All experiments use o3-2025-04-16 only (Sec. 5.1); no evidence for GPT-4, Claude, or open-weight models, which undermines generality claim of "diverse base models" (Line 026).

Abstract claims “domain-agnostic” without a cross-domain Table. Dataset restricted to finance/politics (Sec. 5.1); no medical, legal, or scientific forecasting tasks. The domain transfer was not demonstrated.

Synthesis-rule statistical comparison underpowered. McNemar test (Fig. 10) uses 100 tasks; confidence intervals not shown. Simple-Logic rule failure attributed to noise amplification without formal test for interaction.

Potential data-leakage in financial data, also lack of adversarial robustness check.

### Questions
Suggested experiment: 

replicate Table 2 with at least two openly available models.

run tasks from DiscoveryBench to verify transfer.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors introduce an agentic reasoning pipeline called Analytica. The framework uses soft propositional reasoning (SFR) to structure the LLM reasoning process by estimating the soft truth values of different potential outcomes, by initially decomposing the question into facts that are evaluated using the agents. Their main contributions include improved reasoning accuracy, as well as computational efficiency through their jupyter notebook grounder.

### Strengths
-	Promising results, particularly on reasoning-tuned LLMs (o3 and 04), with similar trends in general purpose LLMs
-	The number of tasks used to evaluate the framework is large and diverse, highlighting generalizability of the framework. 
-	Baselines used to compare analytica to are sufficient and relevant to the reasoning task explored by the authors.

### Weaknesses
-	Some figures, such as Fig. 6 are unclear and somewhat cluttered. Readability and understanding of the figures should be improved. 
-	High dependency on problem decomposition quality. This could affect the whole pipeline negatively. Some more discussion needs to be added regarding this point.

### Questions
Are there any interesting failure cases? Does this pipeline fail in specific scenarios? Would be good to analyze.

### Soundness
3

### Presentation
2

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
The paper presents an LLM-agent architecture, Analytica, for propositional reasoning. The framework consists of three major components: 1) an Analyzer that establishes a tree structure to decompose the reasoning process; 2) a Grounder that verifies leaf propositions and assigns soft truth values; and 3) a Synthesizer that recursively aggregates children's estimations to their parents to obtain the final estimation of the proposition. The authors provide theoretical analysis demonstrating that the method achieves lower variance and bias in performance. In the experiments, the authors show that the proposed method exhibits better performance in terms of both accuracy and efficiency.

### Strengths
- The paper's writing and presentation are of high quality, with clear explanations and details about the proposed framework.

- The proposed Soft Propositional Reasoning is an interesting and innovative solution to the proposition reasoning problem.

- The paper presents a practical, modular architecture with strong tooling. The Jupyter Notebook grounder is a well thought-out contribution, and the framework is applicable to various grounder backends.

### Weaknesses
- The framework relies heavily on the grounder’s ability to estimate and assign numerical probabilities, which is inherently difficult to calibrate. While the tree structure enables systematic problem decomposition, it also introduces the potential for error accumulation across multiple grounders.

- The theoretical analysis relies strongly on linear decomposability and i.i.d. error assumptions, which may not hold in real-world applications.

- The evaluation tasks—mainly financial market forecasting—are somewhat ill-posed. It is difficult to justify that the model can fully reason about market movements using only searchable information. Moreover, such forecasting problems often lack a clear linear structure, which challenges the assumptions underlying the proposed framework.

- The framework is highly sensitive to prompt design. A sensitivity analysis examining how performance varies under prompt perturbations would strengthen the paper’s robustness claims.

- The baseline comparisons understate stronger alternatives. Tree-, Graph-, and Forest-of-Thought baselines are evaluated only with the Basic Search grounder, whereas Analytica is paired with more powerful grounders such as Deep Research and Jupyter Notebook, making the comparison uneven.

### Questions
Please see the above section.

### Soundness
2

### Presentation
4

### Contribution
3
