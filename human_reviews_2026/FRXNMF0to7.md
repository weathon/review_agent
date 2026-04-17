# The Personality Illusion: Revealing Dissociation Between Self-Reports & Behavior in LLMs

- Decision: Reject
- Scores: 6, 6, 0

## Abstract
Personality traits have long been studied as predictors of human behavior.
Recent advances in Large Language Models (LLMs) suggest similar patterns may emerge in artificial systems, with advanced LLMs displaying consistent behavioral tendencies resembling human traits like agreeableness and self-regulation.
Understanding these patterns is crucial, yet prior work primarily relied on simplified self-reports and heuristic prompting, with little behavioral validation.
In this study, we systematically characterize LLM personality across three dimensions: *(1)* the dynamic emergence and evolution of trait profiles throughout training stages; *(2)* the predictive validity of self-reported traits in behavioral tasks; and *(3)* the impact of targeted interventions, such as persona injection, on both self-reports and behavior. 
Our findings reveal that instructional alignment (e.g., RLHF, instruction tuning) significantly stabilizes trait expression and strengthens trait correlations in ways that mirror human data.
However, these *self-reported traits do not reliably predict behavior*, and *observed associations often diverge from human patterns*.
While persona injection successfully steers self-reports in the intended direction, it exerts little or inconsistent effect on actual behavior. 
By distinguishing surface-level trait expression from behavioral consistency, our findings challenge assumptions about LLM personality and underscore the need for deeper evaluation in alignment and interpretability.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates whether LLM “personality” measured via psychometric self-reports (Big-Five, Self-Regulation) corresponds to behavior on psychologically grounded tasks, and whether persona prompting can control either. The authors run three studies:

- **RQ1 (Origin):** Across pre-training → instruction-tuning/RLHF, self-reported traits become more “human-like”: openness/agreeableness/self-regulation rise, neuroticism falls; trait variability drops and trait–trait associations strengthen.
- **RQ2 (Manifestation):** Stable self-reports do not reliably predict behavior. Directional alignment between trait coefficients and human-expected effects is ~chance (≈45–62% across traits; ≈45–57% across tasks), with only a few large models showing modest, sometimes significant alignment (e.g., Qwen-235B).
- **RQ3 (Control):** Persona injection robustly shifts self-reports, but has little/inconsistent impact on behaviors (sycophancy, risk-taking). Thus, persona effects are largely superficial.

Overall, the paper argues for a linguistic–behavioral dissociation: current alignment and prompting stabilize surface-level self-descriptions but do not ground consistent behavior, pointing toward the need for behaviorally grounded alignment objectives.

### Strengths
- Good clarity and logical flow. The overall structure is well organized; RQ1–RQ3 form a coherent progression that makes the argument easy to follow from conceptual framing to empirical validation. Each research question builds naturally on the previous one, and the results are presented in a clear and interpretable way.
- This paper frames a careful integration of psychometrics with behavior. The work explicitly separates self-reported trait structure from downstream behavior, a distinction often blurred in prior LLM “personality” papers, and demonstrates stronger trait coherence post-alignment while simultaneously showing weak behavioral predictiveness.
- This paper clear negative result with practical implications. The near-chance alignment between self-reports and behavior (with limited exceptions) is crisply quantified, challenging common assumptions that questionnaire scores port to action policy.

### Weaknesses
- RQ3 only probes two behaviors (sycophancy, risk-taking) and two personas; the null transfer might be task-specific. Moreover, persona construction uses keyword-based prompts; training-time interventions or activation-steering baselines are not compared. This constrains the generality of the “control is superficial” conclusion.
- Limited coverage of imbuement mechanisms. “Persona injection” is the only controllability method tested; the paper does not assess the other personality-imbuement approaches. Hence, the conclusion “traits don’t control behavior” may be specific to prompt-only personas, not to personality imbuement writ large.

### Questions
- Persona persistence and breadth: In RQ3, can authors extend their analysis to all five behavioral tasks and examine the multi-turn persistence or decay of persona effects?
- Incorporating activation-level interventions (e.g., persona vectors or latent-space steering) might serve as a stronger and more interpretable diagnostic? This would also reinforce the paper’s claim that while LLMs exhibit clear shifts in self-reported personality traits, these shifts do not consistently manifest in behavior. Activation-level analyses would thus provide a principled, mechanistic check to strengthen the causal link between persona representation and behavioral expression.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper reports on a study of LLM personalities focusing on the emergence of traits through training, the consistency of behaviors and self-reported traits, and the impact of targeted interventions such as persona injection. The authors find that post-training stabilizes trait expression and strengthens relationship between traits and behaviors in a manner that correlates with human behavioral data. However, these self-reported traits do not reliably predict behavior which often diverges from human patterns. The authors argue for deeper evaluations of LLM traits and behaviors.

### Strengths
This is a comprehensive study exploring multiple facets of the emergence and manifestation of LLM personalities. It's main strength is in its novel approach to assessing trait stability and trait coherence. The authors reviewed the psychometric literature and proposed links between personality traits and behaviors. Using these empirical results, the authors examined LLMs performance on self-reported traits and behavioral tasks.

### Weaknesses
The statistical methods employed are not well motivated and in some cases the choices seem ill-advised. For example, Levene's test is not advised for repeated measure experiments, where something like Mauchy's test of sphericity would be expected.
Further, predicting training stage from personality traits seems to confuse the explanans with the explanandum. As the authors discuss in the introduction, "We find that instructional alignment plays a pivotal role in shaping LLM traits" (p.2, 074). 
Comparing LLM behavior with humans begs the question, whether LLMs are expected to show similar behavior to humans. This is far from given and risks anthropomorphizing LLMs further.

### Questions
p.2 146 "To assess the internal consistency of model trait expression, we analyze trait stability under repeated prompting with the same input
across multiple generations" -> How are you analyzing trait stability under repeated prompting? Levene's test can give you a sense of the overall (i.e., between-group) variance, but doesn't assess the variability (i.e., between-subject) across repeated measures.
Homogeneity of variance notwithstanding, it would strengthen the results to include the relative contributions of random and fixed effects to the overall variance explained, and the variance within and between runs per model.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The paper addresses three core questions:

1. When and how do human-like traits emerge and evolve across LLM training?
2. Do self-reported traits predict performance in real-world–inspired tasks?
3. How do interventions like persona injection modulate trait profiles and behavior?

The authors present sequence of the experiments on a set of models. The questions are interesting but the statistical significance of the reported results is lacking.

### Strengths
The direction is extremely interesting and really improtant.

### Weaknesses
The research questions 2 and 3 are very interesting, however the fact that alignment happens in post-training is widely-known and it is surprising to see it as a component of a research question for ICLR. 

The documented differences between self-reported properties, properties assessed with a test and observed behaviour could be interesting, but the problem is that "human studies" statements are lacking in quality and statistical soundness. Those studies could not be used as credible source of baseline for the research questions that authors try to addess.

For example, Sections H and I list papers that are used as human-reference for the association of the behaviour with certain test results. If I randomly select some of the papers listed there by the authors, I can easily see that statistical soundness of those is lacking. 

For example, A. Ispas and C. Ispas use a sample of two groups of college students (each group of 30 people, each has 27 female and 3 male participants). Making conclusions on the overall properties of human nature based on such experience seems impossible. Another paper by Sinclair et al. reports results about "91 White undergraduate students (34 men, 57 women) at the University of Virginia who received partial credit toward a course requirement for completion of the experiment." 

Most of these papers reports some results on very small select groups of people that not even slightly represent the breadth of human experience that LLM is exposed to during the pretraining. The authors also provide no internal logic for the selection of the papers that form human alignment "baseline".



The paper seems to be written in a hurry. Many figures have captions that are not legible. There are multiple papers that are cited twice as two different sources, see Binz and Schulz "Using cognitive psychology to understand gpt-3.", Bodroža et al. "Using cognitive psychology to understand gpt-3.", Li et al. "Big5-chat", Rahwan et al. "machine behaviour", Serapio-Gercia et al. "Personality traits in large language models",  Tseng et al. "Two tales of persona in llms", Zheng et al. " Personas in system prompts". While Cao and M. Kosinski "Large language models know how the personality of public figures is perceived by the general public" and Jiang et al. "Personallm: Investigating the ability of large language models to express personality traits." are cited THREE TIMES as different sources.

At the same time several works that could enhance the literature section are missing:

Sorokovikova A, Rezagholi S, Fedorova N, Yamshchikov IP. LLMs Simulate Big5 Personality Traits: Further Evidence. InProceedings of the 1st Workshop on Personalization of Generative AI Systems (PERSONALIZE 2024) 2024 Mar (pp. 83-87).

Pan X, Gao D, Xie Y, Chen Y, Wei Z, Li Y, Ding B, Wen JR, Zhou J. Very large-scale multi-agent simulation in agentscope. arXiv preprint arXiv:2407.17789. 2024 Jul 25.

Dong W, Zhao Y, Sun Z, Liu Y, Peng Z, Zheng J, Zhang Z, Zhang Z, Wu J, Wang R, Xu S. Humanizing llms: A survey of psychological measurements with tools, datasets, and human-agent applications. arXiv preprint arXiv:2505.00049. 2025 Apr 30.

Wang S, Li R, Chen X, Yuan Y, Wong DF, Yang M. Exploring the impact of personality traits on llm bias and toxicity. arXiv preprint arXiv:2502.12566. 2025 Feb 18.

Tshimula JM, Nkashama DJ, Muabila JT, Galekwa RM, Kanda H, Dialufuma MV, Didier MM, Kalonji K, Mundele S, Lenye PK, Basele TW. Psychological Profiling in Cybersecurity: A Look at LLMs and Psycholinguistic Features. InInternational Conference on Web Information Systems Engineering 2024 Dec 2 (pp. 378-393). Singapore: Springer Nature Singapore.

### Questions
Why don't you select meta-review papers for the traits for which there are meta-reviews that show stable association across various demographics? That could make the scope of your experimental work much smaller, but clearly would provide some statistical credibility for the human "baseline" and thus to the conclusions you make.

### Soundness
1

### Presentation
1

### Contribution
1
