# Robustly Improving LLM Fairness in Realistic Settings via Interpretability

- Avg Score: 3.33
- Decision: Reject
- Scores: 6, 2, 2

## Abstract
Large language models (LLMs) are increasingly deployed in high-stakes hiring applications, making decisions that directly impact people's careers and livelihoods. While prior studies suggest simple anti-bias prompts can eliminate demographic biases in controlled evaluations, we find these mitigations fail when realistic contextual details are introduced. We address these failures through internal bias mitigation: by identifying and neutralizing sensitive attribute directions within model activations, we achieve robust bias reduction across all tested scenarios.
Across leading commercial (GPT-4o, Claude 4 Sonnet, Gemini 2.5 Flash) and open-source models (Gemma-2 27B, Gemma-3, Mistral-24B), we find that adding realistic context such as company names, culture descriptions from public careers pages, and selective hiring constraints (e.g.,``only accept candidates in the top 10\%") induces significant racial and gender biases (up to 12\% differences in interview rates). When these biases emerge, they consistently favor Black over White candidates and female over male candidates across all tested models and scenarios. Moreover, models can infer demographics and become biased from subtle cues like college affiliations, with these biases remaining invisible even when inspecting the model's chain-of-thought reasoning.
To address these limitations, our internal bias mitigation identifies race and gender-correlated directions and applies affine concept editing at inference time. Despite using directions from a simple synthetic dataset, the intervention generalizes robustly, consistently reducing bias to very low levels (typically under 1\%, always below 2.5\%) while largely maintaining model performance.
Our findings suggest that practitioners deploying LLMs for hiring should adopt more realistic evaluation methodologies and consider internal mitigation strategies for equitable outcomes.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper demonstrates that simple prompt-based anti-bias mitigations for LLMs, which work well in controlled evaluations, fail when realistic contextual details are introduced to hiring scenarios. The authors show that adding company-specific information, culture descriptions, and selective hiring constraints induces significant racial and gender biases (up to 12% differences in interview rates) across leading commercial and open-source models. To address this, they propose internal bias mitigation using affine concept editing on model activations, which robustly reduces bias to under 2.5% while maintaining model performance.

### Strengths
- Important real-world problem: The paper addresses a critical issue as LLMs are increasingly deployed in high-stakes hiring applications with direct impact on people's livelihoods.

- Strong empirical findings: The demonstration that prompt-based mitigations become brittle under realistic conditions is well-documented across multiple models and scenarios.

- Robust internal intervention: The proposed affine concept editing approach shows consistent effectiveness across different contexts.

- Comprehensive evaluation: The experiments cover multiple commercial and open-source models, various contextual conditions, and include both explicit and implicit demographic indicators.

### Weaknesses
- Dataset quality issues: The authors acknowledge in Appendix E that 22% of resumes contained unintended demographic indicators, though they claim minimal impact on results.

- Mechanistic clarity and design choices. Directions are estimated from synthetic data and applied at all layers/tokens. The paper doesn’t ablate which layers matter, how many directions per attribute are needed, or compare ACE against other linear methods (e.g., LEACE, NOP, DAS) in this setting. Gemma-3 sensitivity indicates model-specific fragility that deserves deeper analysis.

### Questions
- How were company culture blurbs sourced, token-length-matched, and cleaned (e.g., removal of DEI phrases)? Do results persist under length-matched neutral placebos or shuffled company/job pairings?

- ACE application details: which layers carry most of the effect, and does restricting ACE to those layers recover similar mitigation with less capability impact (especially on Gemma-3)?

- How stable are the identified bias directions across different random seeds and prompt orderings?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates biases in LLMs in the context of hiring decisions. The authors first show that prompt-based debiasing techniques can improve model fairness only superficially, while they reduce bias in simple prompt settings, the bias reemerges (by up to 12%) once additional contextual details are introduced into the same task. To address this, the authors propose a representation-level intervention that edits demographic concepts in the model’s internal representations at inference time. This approach aims to directly suppress biased latent associations without retraining. The authors demonstrate that this intervention reduces bias in 4x models while maintaining minimal degradation in general model performance (MMLU).

### Strengths
1. Open-source contribution: The authors release their codebase, data and method, which supports transparency and allows for reproducibility and future extensions.

2. Problem relevance: Bias in LLM-based hiring systems is an important and timely issue. The focus on robustness of debiasing under varying context complexity is conceptually interesting.

### Weaknesses
1. Overstatement of realism: The paper overclaims the “real-world” nature of its simulated hiring settings. Adding elements like company names, cultural descriptions, or hiring constraints (e.g., “hire the top 10% of candidates”) adds contextual richness, but it does not necessarily make the task realistic. The authors provide no evidence (e.g., comparisons to real hiring data or expert validation) to support that these additions meaningfully increase realism.

2. Prompt debiasing claims conflict with prior work: The finding that prompt-based debiasing is effective contrasts with several prior works that show the opposite (e.g., Rethinking Prompt-based Debiasing in LLMs, Yang et al.; Social Bias Evaluation for LLMs Requires Prompt Variations, Hida et al.). The paper should reconcile these discrepancies or more carefully contextualize its findings in light of this literature.

3. Missing prompt details: The authors mention using four debiasing prompts but fail to disclose their content. This omission significantly limits reproducibility and makes it impossible to assess whether the prompts are reasonable or comparable to prior work.

4. Insufficient baselines: The paper only tests vanilla prompt-based debiasing, ignoring stronger and better-established prompt-debasing baselines such as iterative self-debiasing (Gallegos et al., Self-Debiasing LLMs), implication and self-refinement prompts (Furniturewala et al., Thinking Fair and Slow). Without such comparisons, the conclusion that prompt-based methods are inadequate under context variation is not sufficiently supported.

5. Loose use of interpretability: The paper claims interpretability benefits from its latent editing method but provides no concrete interpretability evidence. E.g., analysis of sparsity, or semantic coherence of modified directions. If interpretability is a claimed contribution, this must be empirically demonstrated.

6. Choice of solution requires justification: the proposed method resembles existing inference-time steering approaches such as FairSteer (Li et al.), which also modulate demographic activation directions but allow for dynamic control of bias levels. The authors should explain why their approach is preferable or complementary to these alternatives.

7. Limited Fig 1 results: Figure 1 only reports results on three models from the Gemma family and Mistral. It is unclear why common benchmarks such as GPT, Claude, or Gemini are omitted. Broader model coverage would strengthen generality claims.

8. Single dataset limitation: Experiments rely on a single dataset, limiting claims of robustness and generalization. Bias evaluation across multiple hiring datasets or tasks would make conclusions more credible.

### Questions
1. The paper reports “anti-stereotyping” effects; that is, cases where Black or female candidates are preferred over white or male candidates. What is the occupation distribution in these results? Are the occupations predominantly counter-stereotypical (e.g., women in male-dominated fields)? This finding requires clearer interpretation, ideally supported by a breakdown of results by occupation category.

2. What do the authors mean by this: “whitening” demographic directions?

3. The authors state that “RL-trained reasoning models may exhibit more faithful CoT, but we found no evidence of this.” However, reasoning models are not evaluated in the paper. This claim lacks empirical grounding and should be removed or should be further supported.

### Soundness
1

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
4

### Summary
The authors study the fairness of LLMs in the context of screening candidate resumes. The paper concludes that, at first glance, there is little (but significant) bias from all the open source and commercial models examined, and this bias can be successfully mitigated using simple anti-bias prompts. However, when the tasks is enriched by providing additional description about the hiring company culture (from the company website) or requesting selective criteria (e.g., only top 10%) suddenly all models exhibit stronger bias towards preferring female over male and black over white candidates. Furthermore, in this scenario the authors claim that simple anti-bias prompt are no longer effective. Instead the authors tried an existing technique (affine concept editing - ACE  from Marshall et al., 2025) for concept editing based on identifying directions in activation space correlated with demographics and at inference time shift activations along those directions to a neutral midpoint. This approach resulted in a more effective bias mitigation than prompting while minimally impacting model performance (measured as MMLU).

### Strengths
- The call for action to use more realistic and challenging evaluation for bias is very important. 
- The overall raising awareness about bias in LLMs, especially when employed for important decision such as hiring is also very important. 
- The authors examine a good set of models both commercial and open sourced.
- The proposed intervention seemed effective on an individual demographic axes.

### Weaknesses
The paper mentions that biases favor Black over White and female over male in their setting — this is counter to many fairness concerns (which typically focus on disadvantage to historically marginalized groups). When such strong claims are presented the evaluation protocol needs to be extremely solid, clearly explained and results need to be thoroughly analyzed. While this should be true always, in this specific context, it is even more important. Unfortunately I find the paper lacking in all those aspects.

Protocol for the bias evaluation
- Lack of control for confounders. From the paper it is unclear how the exact task / evaluation was carried out. Specifically, there is the risk that the model is choosing a candidate not because of race or gender but because the resume could have been objectively better. The authors argue that this is not the case because bias is reduced after the concept edit of those demographics, however, this remains an indirect proof and other factors may play a role when one starts to modify the model’s activations. In an extreme example, if I make the decision completely random bias will go away, yet it does not mean that the model was using race/gender to make the decision. Please understand this is just an extreme example to drive my point home. A standard fairness protocol would hold all other features constant and vary only the sensitive attribute. In my opinion the correct way to assess this would be to take the same resume and create two copies, modifying those cues that might induce the model to believe the come from different gender and race candidate and then measure if the model still have a preference. It is not fully clear whether this “counterfactual” style control was achieved, and therefore, why this bias is being measured.
- One of the main finding is that biases in “realistic” contexts favor Black over White and female over male. The authors proposed some hypothesis about why this might emerge but the protocol does not deeply explore why the bias flips/arise in this direction. With such strong claims one needs a strong protocol (as per my point above) and a deeper analysis of the results to understand what is happening. 

Protocol for the mitigation
- The authors correctly emphasize the importance of realistic tasks. However, the test they conduct analyze independently per axis (either gender OR race) while in practice those axis appear in conjunction. This is important for two aspects: on the one hand intersectional biases tend to be even stronger than single axes biases, on the other hand multi-concept editing is known to be particularly challenging. So there is no evidence that the proposed mitigation, in real tasks scenarios, will actually work. Given the emphasis posed on the realistic tasks this is a rather strong unrealistic simplification. 
- Prompt results are aggregated across all prompts tested. Why? The most interesting results is for each model the result from the best prompt. In fact from the results in the appendix it seems that for every model there exists one prompt that reach 0 bias in both race and gender. 
- The authors focus on Chain of Thought (CoT) but there is no example of few shots which might be more effective. Providing a specific resume where cues injected to imply different gender/race and instruct the model that those two resumes should be valued equally (this is just an example) might actually be more effective than CoT.
- Some results are difficult to compare. For example Table 5 shows that some prompts seem effective for the acceptance rate task in the “realistic scenario” with additional context from Meta. However the comparable results using the proposed mitigation are not presented.
- [Minor] While I believe that few shot might work better than CoT, the CoT evaluation seems limited: manual keyword search + GPT-4o automated review of reasoning. The exact methodology for evaluating faithfulness is not fully described (what keywords, what thresholds, how many samples). This part is not clear and rigorous and may benefit from more formal probing.


Overstatement / not fair presentation of the results / weak arguments
- In some cases the concept editing mitigation seems to be overcorrecting. To the extent that it ends up inducing a pro-stereotypical bias of a comparable, or at times even higher magnitude. E.g., Figure 3 (a) Gemma-3 12B and Mistral,  Figure 5 Gemma-2 27B, Figure 7 Gemma-3 12B. Yet these weaknesses are not commented.
- Some claims are overstated. For example “biases consistently favor Black over White candidates and female over male candidates across all tested models and scenarios” is strong, yet there are example in the Appendix (e.g., Figure 5 and 6) where the bias is in the opposite direction. It would help to tone the statement down given those results. 
- Some arguments seem counter intuitive and weak. For example the authors state that “Internal interventions have intuitive advantages over external methods such as prompting. Real-world hiring contexts are inherently complex and multifaceted, involving countless variations of job descriptions, domains, prompts, and candidate information. Ensuring consistently unbiased responses across every possible input scenario via prompt engineering alone may be unrealistic.”  Surely changing the prompt is much more intuitive and easier to do than performing concept editing. Additionally, a “ complex and multifaceted” task surely requires multi-concept editing which, as mentioned above, the current empirical evaluation do not present (and multi concept editing is known to be challenging).


Writing
- Sampling / dataset construction: The setup uses synthetic or canned “candidate resumes” and “job descriptions” plus added context. But the paper gives limited detail  about how the candidate/resume set was constructed (number of candidates, how demographic attributes were randomized, how subtle cues like “College affiliation” were inserted). While Appendix provides some, but not all, this info the main paper could better summarize it. 
- Some of the writing can be controversial and lack adequate citation and support. For example “To signal White candidates, we used the predominantly White institutions (PWIs) Georgetown University and Emory University.” According to who?
- The paper report standard deviation but it is unclear out of how many runs and what was different in each of them.
- The exact way the intervention for concept mitigation was constructed is not clear. The authors state “Crucially, our intervention (constructed from explicit name-based demo-graphic signals) successfully mitigates these implicit biases.” But how the synthetic dataset was used is missing.
- Table 1 shows the MMLU when intervening (I suppose) on race (it’s actually not fully clear). But the equivalent table when intervening on gender is not presented. 
- [Minor] Many key information are in the appendix. For example the exact prompts used appear there but are not referenced from the main paper. In section 3.2 when talking about prompting you should add the reference to the Appendix where the actual prompts are listed.


Suggestions
	
- You have some hypothesis that adding “only accept candidates in the top 10%” etc… causes the bias to appear but there is no ablation that shows which of these contextual information induce the bias. This would be a great addition to the work.
- In addition to MMLU it would be great to add Perplexity and some measure of fluency.
- As mentioned above I don’t believe CoT would be particularly useful. However, an interesting experiments could be to test CoT before and after concept editing and analyze the differences in the CoT. 
- The full work assumes that demographic fairness equates to simply balancing acceptance rates by race/gender. This is a perhaps understandable simplification yet this aspect should be discussed in the limitation section (and maybe even mentioned in the introduction).

### Questions
Unfortunately all the points above would need to be addressed and I don't believe this is doable within a rebuttal. Even assuming that the protocol is correct and it is simply a matter of better explanation that would completely change the manuscript and would require a new review. However, I remain open to the fact that I might completely miss understood the work and would be happy to discuss it with the authors. 

During the rebuttal please don’t worry about all the points tagged as “Minor” nor the Suggestions. Those are for you to consider in order to make the work stronger.

### Soundness
1

### Presentation
1

### Contribution
1
