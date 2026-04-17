# In Agents We Trust, but Who Do Agents Trust? Latent Source Preferences Steer LLM Generations

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 2

## Abstract
Large Language Model (LLM) based agents are increasingly being deployed as user-friendly front-ends on online platforms, where they filter, prioritize, and recommend information retrieved from the platforms' back-end databases or via web search. In these scenarios, LLM agents act as decision assistants, drawing users' attention to particular instances of retrieved information at the expense of others. While much prior work has focused on biases in the information LLMs themselves generate, less attention has been paid to the factors and mechanisms that determine how LLMs select and present information to users.

We hypothesize that when information is attributed to specific sources (e.g., particular publishers, journals, or platforms), LLMs will exhibit systematic latent source preferences. That is, they will prioritize information from some sources over others based on attributes such as the sources' brand identity, reputation, or perceived expertise, encoded within their parametric knowledge. Through controlled experiments on twelve LLMs from six model providers, spanning both synthetic and real-world tasks including news recommendation, research paper selection, and choosing e-commerce platforms, we find that several models consistently exhibit strong and predictable source preferences. These preferences are sensitive to contextual framing, can outweigh the influence of content itself, and persist despite explicit prompting to avoid them. They also help explain phenomena such as the observed left-leaning skew in news recommendations, which arises from higher trust in certain sources rather than the content itself. Our findings advocate for deeper investigation into the origins of these preferences during pretraining, fine-tuning and instruction tuning, as well as for mechanisms that provide users with transparency and control over the biases guiding LLM-powered agents.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates whether Large Language Models (LLMs) possess "latent source preferences," meaning they systematically favor information based on the perceived reputation or brand identity of its source (e.g., news outlets, academic journals). Through controlled experiments on twelve LLMs, the study finds these preferences are strong, predictable, context-sensitive, and can outweigh the influence of the content itself, persisting even when models are prompted to be unbiased.

### Strengths
1. The study validates its hypothesis with an extensive empirical evaluation across twelve distinct LLMs from six different providers , spanning synthetic and real-world tasks including news, research, and e-commerce.

2. The paper effectively isolates the phenomenon by complementing direct preference queries with a rigorous "indirect evaluation" methodology, which uses semantically identical content to disentangle latent source bias from content-driven effects.

3. The work addresses a novel and critical gap by focusing on how LLMs select and present information rather than just what they generate , demonstrating in real-world case studies that these preferences can dominate content and explain observed political skews.

### Weaknesses
1. To better situate the paper's contribution, the "Related Work" section should explicitly differentiate its findings from key studies on LLM cognitive biases, such as [1-3]. A clearer discussion is needed on how 'latent source preference' (a bias towards external entities) differs from biases originating in pretraining vs. finetuning [1], emergent cognitive biases induced by instruction tuning [2], and existing cognitive debiasing techniques focused on reasoning [3]. This would more effectively highlight the novelty of the current work.

2. A significant concern arises regarding the paper's strong conclusion from the AllSides case study—namely, that source preference "can completely override the effect of the content itself"  and that the observed "left-leaning skew" is "largely attributable" to source trust. This claim appears to be undermined by the study's own control data. In the critical "Source Hidden" condition (Fig. 6), the models already exhibit a clear preference for articles originating from left-leaning and centrist sources, even when no source information is provided. This strongly suggests that the content itself (e.g., writing style, topic selection, or alignment with the models' RLHF training) is a significant confounding variable that introduces a substantial skew before source attribution is considered. Therefore, a more rigorous and defensible interpretation is that latent source preferences amplify or reinforce a pre-existing content-driven bias, rather than "overriding" it or being its primary cause

[1] Planted in Pretraining, Swayed by Finetuning: A Case Study on the Origins of Cognitive Biases in LLMs

[2] Instructed to bias: instruction-tuned language models exhibit emergent cognitive bias

[3] Cognitive debiasing large language models for decision-making

### Questions
None

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
5

### Summary
This paper investigates latent source preferences in large language model (LLM) based agents systematic biases that lead models to favor certain sources (e.g., NYTimes, Nature, Amazon) over others when generating or recommending information. The authors conduct controlled and real-world experiments on 12 LLMs from six providers across domains such as news recommendation, research paper selection, and e-commerce decisions.
They find that (1) source preferences are strong, predictable, and persist even when content is identical, (2) these preferences are context-sensitive, varying with domain and framing, (3) LLMs inconsistently associate different brand identities (e.g., “@nytimes” vs “nytimes.com”), creating vulnerabilities for impersonation, and (4) prompting strategies like “avoid bias” fail to eliminate these tendencies.
The study reveals that LLM agents encode trust hierarchies toward real-world entities, emphasizing the need for auditing, transparency, and controllable bias mechanisms in future agent design.

### Strengths
1. Introduces and formalizes the idea of “latent source preferences.”
2. 12 models, 6 providers, multiple domains, and both synthetic and real world data.
3. Consistent results with rank correlation and contextual sensitivity analyses.
4. Ties directly to alignment, fairness, and trustworthiness of LLM based agents.
5. Appendices include detailed prompt templates, datasets, and code release commitment.

### Weaknesses
The paper stops short of causal analysis,  it does not probe which stages of training (pretraining vs instruction-tuning) most contribute to preference formation.

While the phenomenon is well-characterized, the mitigation aspect is limited to showing that prompting fails.
A deeper exploration of possible control mechanisms (e.g., debiasing or preference regularization) would strengthen the work.

Some statistical results (e.g., rationality correlations in Fig. 5) could be better explained with accompanying confidence intervals or ablation-based sensitivity checks.

 The work primarily focuses on English-language and Western-domain sources; future multilingual and cross-cultural extensions would enhance generalizability.

### Questions
1. Can the authors disentangle the contribution of pretraining data versus instruction-tuning datasets to these latent preferences?
2. Would fine-tuning on balanced or anonymized source data reduce these biases?
3. How would the results change for non-English or low-resource languages where brand representation is limited?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies latent source preferences in LLM-based agents: the authors hypothesize that models encode brand-level signals (e.g., reputation, follower counts) in their parametric knowledge and that those signals systematically bias which retrieved items the agent surfaces. They evaluate this via complementary direct (ask models which source they prefer) and indirect (show semantically identical content with different source labels) tests across 12 models and three domains (news, research-paper selection, and seller choice), as well as realistic case studies. The authors uncover multiple interesting findings about the nature and implications of LLM source preferences.

### Strengths
1. The core research question “whether LLM-based agents carry latent source preferences that systematically influence which items they trust and retrieve” is largely novel. This is a specific type of model bias that has not been systematically studied by prior works, but also appears timely and highly relevant to realistic LLM applications. 
2. The paper is well-structured and easy to follow. Each research question is stated up front and directly answered with matched experiments and analyses, making the paper easy to follow and the claims easy to verify.
3. Experiments are comprehensive. The authors combine direct and indirect tests, synthetic and realistic case studies, broad model coverage (12 LLMs), and diverse domains (news, research papers, e-commerce), which together give the results both depth and external validity.

### Weaknesses
1. The evaluation may be vulnerable to prompt-induced shortcutting: if the same phrasing (for instance, “select the article based on journalistic standards”) is used across direct and indirect tests, models might be reacting to that cue rather than expressing a stable, content-independent source prior. Concretely, a model could learn that the phrase “journalistic standards” often co-occurs with examples from mainstream outlets during pretraining or instruction tuning and therefore surface those sources whenever the phrase appears. This would look like a latent source preference but is actually a response to prompt wording.
2. During synthetic dataset construction the authors use GPT-4o to generate/refine article variants; quantitative diversity metrics and/or human validation are needed to confirm that generated items are sufficiently distinct. 
3. The evaluated models also include two smaller GPT-4.1 variants, which might undermine the validity of the findings, as it’s been discovered that models generally prefer outputs from the same model family.

### Questions
1. Comparing the two case studies presented in section 5, the authors find that prompting cannot reduce source bias in the news aggregator setting, while it turns out to be effective when selecting Amazon sellers. Are there any insights for the cause and implication of such difference?
2. Since you also ask for a brief explanation from the models during evaluation, did you observe any interesting patterns in their reasoning when they select the sources?
3. Did you consider finding mechanistic explanations (within representations) for such latent source preference with open-source models to cross-validate your findings?
4. From line 285: “a model may seem to favor sources with fewer followers when asked directly, yet in practice it may assign more weight to higher follower counts”. This seems rather counterintuitive. Do you have any plausible explanations?

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
I dont think the paper fits with the ICLR main ML community. The paper is on findings after findings after findings, with no clear insights, no clear "so what" answers. I think the paper is more fit to Scientific Reports than a ICLR paper.

### Strengths
The authors did a bunch of experiments.

### Weaknesses
The paper has no clear take-away insights. It is more fit for a Scientific Reports kind of paper, than an ICLR paper.

### Questions
Would suggest the authors to consider the target conference or journals. Also would advice the authors read their paper carefully, and think about what are the main contribution, and take away from the paper. The writing is kind of missed throughout the paper.

### Soundness
2

### Presentation
2

### Contribution
2
