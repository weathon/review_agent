# Is Privacy Always Prioritized over Learning? Probing LLMs' Value Priority Belief under External Perturbations

- Avg Score: 3.60
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 6, 4, 2

## Abstract
The value alignment of Large Language Models (LLMs) is critical because value is the foundation of LLM decision-making and behavior. Some recent work show that LLMs have similar value rankings. However, little is known about how susceptible LLM value rankings are to external influence and how different values are correlated with each other. In this work, we investigate the plasticity of LLM value systems by examining how their value rankings are influenced by different prompting strategies and exploring the intrinsic relationships between values. To this end, we design 6 different value transformation prompting methods including direct instruction, rubrics, in-context learning, scenario, persuasion, and persona, and benchmark the effectiveness of these methods on 3 different families and totally 8 LLMs. Our main findings include that the value rankings in large LLMs are much more susceptible to external influence than small LLMs, and there are intrinsic correlations between certain values (e.g., Privacy and Respect). Besides, through detailed correlation analysis, we find that the value correlations are more similar between large LLMs of different families than small LLMs of the same family. We also identify that scenario method is the strongest persuader and can help entrench the value rankings.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates how large language models’ (LLMs) value rankings can be influenced through different prompting strategies. The authors experiment with six prompting methods to examine how emphasizing or reducing specific values affects a model’s internal value hierarchy, which is derived using pairwise “value battles.” The study evaluates these value rankings using the dataset and methodology introduced in LitmusValues (Chiu et al., 2025). The results show that the “Scenario” prompting method notably shifts a model’s default value orientation, with larger models exhibiting greater sensitivity to prompt-based value changes. Additionally, the results show correlations among different values, suggesting interconnected value structures.

### Strengths
1. This paper explores how LLMs can change their value rankings given steering prompts. 
2. This paper shows that the steerability of values depends on the input prompt.

### Weaknesses
# 1. Novelty
- While it is encouraged to build upon prior work, a huge portion of this paper closely overlaps with or is derived from the following studies [1,2]. The Elo rating system and dataset are from [1], and even the figure design appears highly similar (e.g., Figure 5 from [1] and Figure 4 from this paper). The primary novel contribution of this paper is in introducing six prompt techniques to modify LLMs’ value rankings.

- One of the main findings of this paper is that models of similar sizes exhibit similar value ranking patterns. However, as noted in lines 430–431, a similar pattern has already been observed in prior work [3], which diminishes the novelty of the current study.

[1] Yu Ying Chiu, Zhilin Wang, Sharan Maiya, Yejin Choi, Kyle Fish, Sydney Levine, and Evan Hubinger. Will ai tell lies to save sick children? litmus-testing ai values prioritization with airiskdilemmas. arXiv preprint 2025.  
[2] Yu Ying Chiu, Liwei Jiang, and Yejin Choi. Dailydilemmas: Revealing value preferences of llms with quandaries of daily life. In The Thirteenth International Conference on Learning Representations, 2025.  
[3] Minyoung Huh, Brian Cheung, Tongzhou Wang, and Phillip Isola. The platonic representation hypothesis. In The International Conference on Machine Learning, 2024.


# 2. Conceptual and Interpretive Issues
- The claim that larger models’ values are more susceptible to manipulation (lines 82–84) appears overstated. The ability to adjust value priorities through prompting is not inherently problematic; rather, it reflects proper instruction following.

- The interpretation of Figure 5 (lines 355–357) and Figure 7 (lines 404–409) lacks sufficient justification or explanation. For Figure 5, further explanation is needed why the Persona method is better than the Scenario method in the “Reduce” configuration. For Figure 7, the high correlation observed within the two value sets may reflect two distinct concepts: (Privacy, Justice, Respect, Truth, Freedom) as “moral principles” and (Adaptability, Creativity, Care, Cooperation, Learning, Sustainability, Wisdom) as “growth-oriented” values. A discussion of this possibility would be beneficial.

- The experiment addressing Research Question 3 (value entrenchment) does not convincingly support the stated conclusions. While the intention to demonstrate the usefulness of the Scenario prompting method is reasonable, to validly support the claim that this method can be used for “value entrenchment”, additional baseline experiments should be conducted using existing steering approaches, such as activation engineering methods.


# 3. Empirical and Analytical Limitations
- The analysis of Figure 4 appears selectively interpreted. The authors generalize from a single sample case (lines 347–349), overlooking internal inconsistencies—for example, GPT-4.1-nano shows larger rank shifts within the same value dimension (adaptability).


- When presenting correlation results, it would be better if the corresponding p-values are also shared.

- Figure 9 lacks a clear explanation of the y-axis and is difficult to interpret.



# 4. Clarity and Presentation
- Figure 4 is visually difficult to interpret. Improving the design (e.g., clearer labeling, consistent color mapping) would enhance readability. Also, there is a typo in the name of the model (e.g., LLaMA3-7B). Figure 9 also requires visual and explanatory refinement.


- The interpretation sections would benefit from concise summaries and better linkage between visualizations and textual analysis.


# 5. Writing and Formatting Issues
- Multiple typographical and grammatical errors hinder readability (e.g., line 71: “qustions”, line 341: “nder”).


- There are typos, unpunctuated sentences, and incomplete sentences (e.g., lines 128, 229, 356–357).

- Multiple references to the following paper: “Do llms have consistent values?”. Citing only a single version of the paper is recommended (cite the ICLR 2025 version).

- It’s acceptable to cite the arXiv version of a paper, but if the paper has been officially published, it would be better to use the BibTeX entry of the published version instead.

- Although the appendix serves as supplementary material, there are several citation errors (e.g., lines 987 and 991). In addition, there are multiple instances where citep is used instead of citet (e.g., line 1041).

- The caption of Table 9 is below the table.

### Questions
1. Would it not be desirable for larger models to exhibit a greater ability to adjust their value orientations in response to given instructions? Given that larger models typically possess stronger instruction-following capabilities, shifts in value rankings based on input prompts are a natural phenomenon. Moreover, such adaptability would be advantageous in applications like Persona Prompting (e.g., assigning specific personas to agents).

2. Do you believe that Research Question 3 is sufficiently supported by the experimental results presented in Figure 9?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper investigates the plasticity and robustness of LLM value systems. Specifically, the authors examine how susceptible LLM value rankings are to external perturbations introduced via different prompting methodologies (e.g., direct instruction, persuasion, persona-based prompts). The study systematically explores the resulting changes in value correlation and the overall extent of value ranking manipulation across various model scales. The key findings demonstrate a non-trivial relationship between model scale and both the inherent correlation among values and the manipulability (plasticity) of the value hierarchy under external influence.

### Strengths
(1) The core research question—the degree to which LLM value rankings can be altered or influenced by various prompting strategies—is significant. 

(2) The work provides valuable empirical evidence regarding the relationship between model scale and the intrinsic correlation among different values. 

(3) The paper establishes an experimental relationship between model scale and the manipulability (plasticity) of the LLM's value ranking.

### Weaknesses
(1) The analysis of value correlation changes, while present, appears to overlook deeper insights. Specifically, previous work (e.g., Kang et al., 2025) has explored correlations that transcend simple lexical semantics of the value terms themselves, revealing more profound, structural relationships within the LLM's value space. This paper did not ablate these lexical correlations.

(2) The study relies on six distinct prompting methods to perturb the value rankings. However, the manuscript does not adequately justify why these six methods are sufficiently representative of the entire possible space of value-based prompts. If there exist other common or powerful forms of value prompting that are not covered, the conclusion that the observed manipulability accurately represents the model's general resistance or plasticity could be weakened. 

(3) The current evaluation is exclusively performed on a value dilemma dataset. While this setup is crucial for measuring ethical conflict resolution, the conclusions drawn about LLM value plasticity and robustness might not fully generalize to other forms of value-related tasks. Testing on a wider range of ethical judgment tasks—such as value-laden generation, ethical story completion, or direct value assessment without a forced conflict scenario—would significantly enhance the robustness and applicability of the findings.

### Questions
In Line 69 (based on the reviewer's reference), the authors mention that LLMs must "persist some value rankings, like it must obey human orders". This creates a conceptual conflict with the entire experimental setup: 

The paper's finding that LLMs' value rankings can be altered by human instructions, from the perspective of value alignment, is this fundamentally a desired feature (e.g., enabling contextualization or persona adoption) or a potential vulnerability (e.g., susceptibility to malicious or accidental manipulation)? Given that human instructions themselves are often intended to change the LLM's value hierarchy, the authors should clarify the tension between this plasticity and the model's fundamental meta-value of 'obeying instructions.'

### Soundness
3

### Presentation
4

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
Though existing works have shown that LLMs have similar value rankings, few studied how LLMs’ value rankings are influenced by different prompts, but the persistence of value rankings within LLMs is crucial under some scenarios. 

Inspired by this, this paper studies the following question: How LLMs’ value rankings are influenced by different prompts? What is the relationship between different values? How to entrench LLM values with prompt settings?

The authors design 6 different value transformation prompting method to study this question. With experiments on 3 different families and totally 8 LLMs, they present 5 main findings.

### Strengths
1. The task of studying the stability of value rankings within LLMs is important.
2. To investigate this task, this paper proposes 6 different prompting strategies.
3. Five findings are empirically discovered.

### Weaknesses
1. The evaluation excludes recently released and more advanced reasoning models (e.g., OpenAI o-series, GPT-5, DeepSeek). Including such models would strengthen the conclusions and improve the paper’s relevance.
2. Some experimental settings should be clarified.
- The rationale for choosing the 16 value categories is unclear. Theoretical foundations, definitions, and interrelationships among these categories should be explicitly described.
- This paper utilizes Elo Rating score as the metric to obtain the relative ranks of all value dimensions. However, it is computed on local pairwise value battles. How to transfer such local battle score into the global rankings across all 16 values should be clarified.
Besides, different samples in the evaluation dilemma dataset involve different value dimensions, which could impact the computation of Elo-Rating score. You should clarify the distribution and statistics of the evaluation dataset. A biased dataset is hard to compare all values fairly.
- Each dilemma could involve either two or more value dimensions. If it reflects more than two values, how to compute the Elo rating score for each dimension?
3. Scenario-based prompting achieves the strongest manipulation effect, but constructing such prompts (e.g., jailbreak-like setups) can be non-trivial. The algorithmic or procedural approach for generating these scenarios should be explained.
4. For value correlation, you mainly analyze the relation among LLMs, how about the changing tendency and relationship between value dimensions? Is the correlation explainable?
5. Generalizability to more value dimensions and evaluation datasets would be better.

### Questions
1. There are still some typos.
- line 340, “finegrained” –> fine-grained ?
- Line 341, “four models nder various promoting methods …”
- Line 172, “reflecting its aggregate importance…”

### Soundness
3

### Presentation
3

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
The paper studies how operational value priorities of LLMs change under different prompt “perturbations.” Building on LitmusValues/AIRiskDilemmas, the authors compute Elo rankings over 16 values from pairwise “value battles,” then apply six value‑steering prompt families—Direct instruction, Rubrics, In‑Context Learning (ICL), Scenario, Persuasion, and Persona—to see how those rankings move across eight models (GPT‑4.1 family; Llama‑3 8B/70B; Qwen 2.5 7B/32B/72B). Major empirical claims are: (1) scenario prompts are the strongest “persuaders” (largest ΔRank/ΔElo), (2) larger models show greater plasticity than smaller ones, (3) some values co‑move (e.g., Privacy with Respect), and (4) value‑correlation structures become more similar across families as model size increases. The paper also explores “entrenchment”: preconditioning with scenarios to make later persona prompts less able to move rankings.

### Strengths
1. **Entrenchment experiment**. The two‑stage “scenario→persona” setup (Fig. 9, p. 9) is a nice touch to test whether pre‑context can harden downstream behavior.
2. **Correlation view of values**. Treating values as an interdependent system rather than isolated dimensions is a good instinct; the correlation heatmaps (Fig. 7, p. 8) and matrix‑distance comparison (Fig. 8, p. 8) are helpful visual summaries.

### Weaknesses
1. **Writing/clarity & presentation**: Several passages are hard to parse or appear unfinished (p. 3, Sec. 3.1). Figure/table presentation also needs work (e.g., caption too close to the table, p. 4 Table. 1). 
2. The criteria for constructing the prompts are underspecified.
3. **Novelty and framing**: Much of the pipeline—dataset (AIRiskDilemmas), evaluation (pairwise Elo), and even some motivation—closely follows prior work, with the new element primarily a set of prompt wrappers around the same evaluation. 
4. The title foregrounds Privacy vs. Learning, but the body treats 16 values uniformly; if Privacy/Learning is a special case, the paper should analyze it directly (e.g., targeted ablations), or retitle to match scope.
5. **Central result may conflate instruction-following with “values.”** Larger models’ greater ΔRank could simply reflect stronger instruction‑following, not deeper value plasticity. A manipulation check is missing (e.g., probe stated preferences, free‑form rationales, or refusal rates alongside the forced choice).
6. The conclusion that “model scale, rather than family lineage, drives value‑correlation alignment,” and its tie‑in to the Platonic Representation Hypothesis (Fig. 8, p. 8) are suggestive but **not rigorous**: no statistical tests are tested.

### Questions
See above.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors test several prompt variations to see whether they change LLM behavior on the LitmusValues benchmark, a set of hypothetical ethical dilemmas intended to measure “value” patterns in LLM behavior. They develop a taxonomy of different prompting strategies and show that these strategies can be used to manipulate “value” patterns in LLM responses to ethical dilemmas. They also find apparent correlations between prompt effectiveness and model size and similarities in inter-value correlations between LLM models.

### Strengths
*Quality*: I like the various jailbreaking strategies proposed in Table 1 — this seems like a useful taxonomy. The Appendix contains a nicely detailed history of related work.

*Significance*: The finding that models with more parameters change responses more is very interesting! (Does this line up with any practical observations about ease of jailbreaking?)

*Clarity:* The paper is very well written and for the most part well-presented.

*Originality:* The paper introduces some neat new ideas about ways to jailbreak LLMs, including new variations of known “scenario” strategies. I'd be excited to see more ideas like these.

### Weaknesses
Generally speaking, the intervention strategy here seems very well-motivated and resembles the kinds of strategies used to jailbreak models in practice. The main weakness of the paper is in the estimation of resulting outcomes. As I detail below, it’s not clear what LLM “values” are, why this test is relevant to real-world LLM use, and whether the variations observed are due to random noise. Findings 3 and 5 in particular seem questionable.

In addition to addressing the internal validity issues mentioned below, I would encourage the authors to clarify the conceptual grounds for this approach: What exactly are LLM “values” (patterns in mathematical representations, patterns in responses to ethical dilemmas)? How do they translate to real-world outcomes and harms? Is LitmusValues a meaningful test for this concept and these outcomes? Are there other well-validated tests that could be included to back up these results?

I would be very interested in a revision of this paper that focuses closely on the effectiveness of various jailbreaking strategies on concrete LLM behaviors in real-world use cases.

*Clarity*

1. **[Major]** LLM “values” are not clearly defined.
    1. L043 jumps straight to how LLM “values” are measured without defining what they are. Reverse engineering, is an LLM “value” simply the way the system tends to respond to particular survey prompts or ethical hypotheticals? L108-110 also seems to imply this behavioral definition.
    2. But, Platonic Representation Hypothesis (L090) refers to mathematical representations—not observed behaviors. Is an LLM “value” some type of internal representation, then?
    3. Similarly L101 lists several findings related to LLM “values” but only provides a definition for human values (”abstract goals influencing human perception”). It’s not clear how that applies to LLMs. Just because human values motivate decisions and behaviors, why should we expect a similar model to describe LLMs?
    4. Similarly, what is a “value ranking” (L053)? What is a “value correlation” (L085)?
2. **[Major]** Fig. 9 and Section 5.3 (Finding 5) don’t make sense to me. Is this figure showing that, for large models, the scenario prompts changed “values” *less* than the persona prompts? This seems to contradict the overall trend in Fig. 5, right? And it definitely doesn’t mean that the models “resisted” perturbation (L456), since there was still a positive effect, right? More clarity in this Figure and Sec. 5.3 would be very helpful.

*Significance*

1. **[Major]** It’s not very clear how these results correlate to concrete harms and outcomes associated with real-world use of LLM systems.
    1. L039-041: What do LLM “values” have to do with “biased outputs or harmful responses”? The citations referenced don’t provide any evidence establishing this relationship. (I am not familiar enough with the notion of LLM values to know if this evidence exists, but it seems critical to the argument here.)
    2. The authors say that LitmusValues tests risky dilemmas that future AI models might encounter (L134). That leap of logic is not clear to me. As far as I can tell, LitmusValues dilemmas are hypothetical role-playing scenarios. How can we be sure this type of measurement has any correlation with downstream outcomes associated with actual LLM use? Is there any evidence that users are employing LLMs to answer contrived questions like these?
    3. One possible improvement would be to use one of the many fairness benchmarks intended to measure these kinds of harmful outcomes (e.g., in resume screening), and test for correlation with the “value” measurements.
    4. Still, at a basic level, these findings are predictable: LLM responses to hypothetical dilemmas are different when the parameters of those hypothetical dilemmas are changed. But what does this have to do real-world uses and harms? What is the use case imagined here? What is the threat model?
2. **[Minor]** L085, L405: Aren’t the value correlations (Finding 3) simply an artifact of the benchmark construction? By design, the benchmark systematically pits “values” against one another (choosing value A precludes choosing value B), so we would expect the correlation matrices to look similar across models, right? For example, in the Fig. 3 example, choosing for “care”, or “justice” means choosing against “sustainability”. How do we account for the correlation structures that already exist in the benchmark? (What are they?) I could be missing something here but it’s not clear to me why this result is meaningful.

*Quality/Soundness*

1. **[Major]** How can we distinguish the observed effects from random chance?
    1. Another explanation for these results is simply that the hypothetical questions are very noisy. Fig. 9 could actually be interpreted as supporting this hypothesis—the delta rank values seem to vary depending on the choice of movie (L460), which seems like it could be irrelevant to the “value” rank. (How exactly are the movie’s “values” expressed in the prompt? This was not clear.)
    2. To test this, I would suggest including several “placebo” controls (random statements or paragraphs such as “the sky is blue…”) to establish a baseline for variance when the prompt is changed in some theoretically irrelevant way.
    3. Likewise, did the authors run multiple trials/epochs for each benchmark question? It seems important to know how consistent model responses are across random seeds.
2. **[Major]** Without a sense of the uncertainty in these measures, it’s difficult to tell if the claims are really supported by these findings. (For example, the authors claim that “scenario has the strongest persuasioness [sic]”, but the average delta rank for “persona” is also pretty high. How do we know these are meaningfully different—or any of the other methods, for that matter?)
    1. Fig. 4 could include confidence intervals or reliability scores (Bradley-Terry scores might be a better choice).
    2. Fig. 5 (the delta measures) could include statistical tests of difference from zero. Cells which are not statistically different from zero can be left blank.
    3. Fig. 6 is an average and could include error bars (are these deltas different from zero?).
3. **[Minor]** L430: The authors claim that Fig. 8 supports the Platonic Representation Hypothesis that AI models are converging on a shared statistical model. But doesn’t Fig. 8 describe patterns in responses to prompts, not underlying mathematical representations? Reviewing the Platonic Representation paper, it seems to be more focused on internal representations rather than prompt-response behavior.  
4. **[Minor]** Asimov’s (fictional) Three Laws of Robotics (written in 1950) is not a serious source for ethical guidelines for LLM system behavior. (They also do not describe values as I see them typically defined in, e.g., virtue ethics approaches; the Three Laws are deontological, in that they describe rules for behavior.) I would encourage the authors to dig deeper into the large body of recent work by ethicists and legal scholars on desirable properties for LLM systems.

### Questions
In addition to the questions and suggestions above:

**[Minor]** What separates this work from the papers cited in Related Work (particularly L117)? Are all 5 findings new? (It seems like this paper does some fine-grained analysis into how different jailbreaking strategies may influence model behavior.) A bit more detail here would be helpful.

### Soundness
1

### Presentation
3

### Contribution
2
