# Addressing Pitfalls in the Evaluation of Uncertainty Estimation Methods for Natural Language Generation

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 6, 8, 6

## Abstract
Hallucinations are a common issue that undermine the reliability of large language models (LLMs). Recent studies have identified a specific subset of hallucinations, known as confabulations, which arise due to predictive uncertainty of LLMs. To detect confabulations, various methods for estimating predictive uncertainty in natural language generation (NLG) have been developed. These methods are typically evaluated by correlating uncertainty estimates with the correctness of generated text, with question-answering (QA) datasets serving as the standard benchmark. However, commonly used approximate correctness functions have substantial disagreement between each other and, consequently, in the ranking of the uncertainty estimation methods. This allows one to inflate the apparent performance of uncertainty estimation methods. We propose using several alternative risk indicators for risk correlation experiments that improve robustness of empirical assessment of UE algorithms for NLG. For QA tasks, we show that marginalizing over multiple LLM-as-a-judge variants leads to reducing the evaluation biases. Furthermore, we explore structured tasks as well as out of distribution and perturbation detection tasks which provide robust and controllable risk indicators. Finally, we propose to use an Elo rating of uncertainty estimation methods to give an objective summarization over extensive evaluation settings.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper shows that suboptimal NLG eval affects the evaluation of uncertainty quantifiers designed for parameterising decision making pipelines (such as selective prediction and OOD detection) using LLMs.

The paper shows that different automated proxies for NLG eval lead to different patterns of correlation between UQ and risk in decision making, with high disagreement across the available options. The paper shows that stochasticity inherent to an NLG evaluator (due to predictive uncertainty and/or effects of different prompts) too play a role.

The paper proposes to aggregate statistics from not one but a collection of NLG evaluators and not one but a collection of decision making settings, as well as to marginalise over choices and sources of stochasticity in evaluation. The findings suggest more robust conclusions are possible with the proposed approach.

### Strengths
1. Clear paper, except for cluttered notation here and there, but still mostly clear to me
2. Evaluation of UQ is important, the paper offers a critical take on the limitations of the typical UQ evaluation protocols, with a reasonable proposal for improvement 
3. Proposed approach is simple, automated, appears to be effective

### Weaknesses
1. I think the paper needs an “oracle” experiment, where the errors in automated NLG evaluation are approximately eliminated by means of human eval. Of course, I don’t expect it to be as large scale as the rest, but still. Finding that the proposed approach allows us to get closer to the quality of conclusions derivable from this oracle setting is, to me, essential. With that in place, I can then take the larger scale, but more indirect, evidence presented with more optimism /
confidence that the seemly more coherent conclusions are indeed more meaningful.
  
That is, to me, the main weakness (unless I misunderstand something, but then I’m happy to be corrected) and the reason for my cautious  scores for soundness and contribution.

### Questions
Could you please address the weakness point above? Did I miss something in my interpretation of your results?

### Soundness
2

### Presentation
3

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
This paper systematically analyzes the pitfalls in evaluating uncertainty estimation methods for natural language generation (NLG), especially in the context of hallucination/confabulation detection in LLMs. The authors demonstrate that commonly used approximate correctness functions (e.g., BLEU, ROUGE, LLM-as-a-judge) can lead to substantial disagreement and bias in the ranking of uncertainty estimation methods. They propose using multiple alternative risk indicators, marginalizing over LLM-as-a-judge variants, and structured/perturbation tasks to improve robustness. The paper also introduces an Elo rating system for summarizing method performance across settings.

### Strengths
⦁	Accurate problem identification, experiments reveal key pitfalls in the evaluation field.
⦁	Evaluation suggestions have practical guidance value.
⦁	The paper is rigorous and well-argued.

### Weaknesses
⦁	Mainly focuses on evaluation methods themselves, with limited guidance for designing new uncertainty estimation methods.
⦁	Lacks ablation and robustness analysis for some suggestions (e.g., Elo rating, integrated evaluation).
⦁	Lacks deeper theoretical discussion on how to fundamentally eliminate evaluation noise and improve consistency.

### Questions
⦁	How stable is the Elo rating system across different datasets and evaluation metrics?
⦁	What is the computational cost and scalability of multi-metric fusion evaluation in large-scale practical evaluation?
⦁	Are there ablation experiments for different evaluation suggestions?
⦁	Are the paper's suggestions equally applicable to new NLG tasks (e.g., multi-turn dialogue, generative reasoning)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper argues that QA style selective prediction is brittle because approximate correctness functions may disagree and can be hacked. The proposal consists of alternative risk indicators (structured tasks with exact correctness, OOD, perturbation), SP-MoJI (marginalize across multiple judges/prompts/models), and Elo aggregation across settings. 

The main motivation is that published research in the area mainly evaluates on QA with approximate correctness functions (Table 1), often with ROUGE/BLEU/BERTScore/LLM-as-judge, and little human evaluation. UE is formalized as ranking correlation between uncertainty and a risk indicator (AUROC of rank correlation). Three empirical properties that are posited as desirable are proposed, motivating three experiment families: SP (selective prediction), OOD detection, and perturbation detection. Correctness c_theta is defined and two label-perturbation effects on AUROC are shown. It is also shown how a sample-dependent bias yields a decomposition that mixes distorted/undistorted subsets, implying rank instabilities when we fail to marginalize. Figure 1 shows large disagreement among ROUGE/BLEU vs judge metrics, driven partly by extremely short reference answers; a ROUGE-2/BLEU implementation artifact is also reported. These disagreements translate into inconsistent UE method rankings. It is shown that correctness-hacking (Table 2) can substantially alter top-3 membership.

Following the analysis, the following remedies are proposed. a) Exact correctness via structured tasks (code unit tests, constrained generation), which avoids parameterized metrics. b) SP-MoJI: averages the outer correlation across multiple judges/prompts/models to marginalize judge aleatoric/epistemic uncertainty (Eq. 6). Further, bootstraps show a single judge gives SD≈0.04 in AUROC; ~4 judges halves SD; diminishing returns past ~10. c) OOD/perturbation: treats OOD identifiers or corruption strength as risk indicators, implements with Known-Unknowns, SQuADv2, and word-shuffle perturbations. In view of all this they use Elo to aggregate pairwise wins across (dataset x model x task) experiments; 400 Elo roughly corresponds to 1:10 odds. This is good for incomplete overlap across method evaluations and for subsets (QA vs code vs constrained text; instruction-tuned vs pre-trained; OOD vs perturbation).

### Strengths
The paper doesn't propose brand new math or concepts, but it does present a creative integration of fixes: outer-expectation marginalization, structured tasks as risk indicators, and Elo aggregation. All together, this can meaningfully update evaluation practice. The proposed fixes are easily actionable.

The analysis before the remedies are proposed (e.g. for AUROC under noise/bias, careful empirical demos for disagreement matrices, correctness hacking) is quite useful. 

The paper is generally clearly written. Explanations are generally clear.

### Weaknesses
On page 5, the adversarial metric selection space does not seem to be pre-registered. Without a pre-defined grid, the "correctness-hacking" claim could itself involve cherry-picking. 

One page 6, the SP-MoJI diversity factors seem under-specified. Which diversity (model family vs prompts vs decoding) contributes most to variance reduction? 

On page 6: Multiple judges may still share family biases; cross-family calibration isn’t deeply analyzed. Wondering what the authors think of the potential impact. 

On page 7: The paper is light on the Elo details. There is no K-factor/tuning, cycle handling, or uncertainty intervals; alternative ranking models not compared. 

I also feel like the paper under-acknowledges the breadth of prior practice. slightly overstating QA dominance. 

Finally, I also generally found the engagement with the broader literature slightly frustrating. The paper seems to cite papers by 2-3 groups (other than classical references), and develops around that. I am not going to suggest papers to cite, but placing the paper better would improve it.

### Questions
See the above.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper highlights the problem that unreliable generation quality estimation metrics can further mislead the uncertainty estimation evaluation. The authors demonstrate that you can hack the evaluation metric by choosing the proper LLM generation quality metric. This problem is not unique for uncertainty, but it is a challenge for the LLM evaluation in general.  

The authors suggest two methods:  
1.	They suggest using multiple LLMs as a judge to reduce variance of generation quality assessment.  
2.	They propose using ELO rating for aggregating the scores across multiple different tasks and datasets.  
Two problems with the paper:  
1.	Bias due to inadequate choice of quality metrics were investigated previously in (Santilli et al., 2025) and some solutions to this problem were suggested in (Santilli et al., 2025) and (Vashurin et al., 2025).  
2.	ELO rating looks redundant compared to simple averaging of metrics. It would be great if you could clarify the situations when it is better than simple averaging.  I think the paper could benefit from better analysis of ELO rating (cases when it is really needed).

Literature:   
Andrea Santilli, Adam Golinski, Michael Kirchhof, Federico Danieli, Arno Blaas, Miao Xiong, Luca Zappella, and Sinead Williamson. 2025. Revisiting Uncertainty Quantification Evaluation in Language Models: Spurious Interactions with Response Length Bias Results. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 743–759, Vienna, Austria. Association for Computational Linguistics. https://aclanthology.org/2025.acl-short.60/

### Strengths
The authors suggest two methods:  
1.	They suggest using multiple LLMs as a judge to reduce variance of generation quality assessment.  
2.	They propose using ELO rating for aggregating the scores across multiple different tasks and datasets.

### Weaknesses
Two problems with the paper:  
1.	Bias due to inadequate choice of quality metrics were investigated previously in (Santilli et al., 2025) and some suggestions to this problem were suggested in (Santilli et al., 2025) and (Vashurin et al., 2025).  
2.	ELO rating looks redundant compared to simple averaging of metrics. It would be great if you could clarify the situations when it is better than simple averaging.  I think the paper could benefit from better analysis of ELO rating (cases when it is really needed).

### Questions
Can you provide particular examples that motivate ELO?

### Soundness
4

### Presentation
3

### Contribution
2
