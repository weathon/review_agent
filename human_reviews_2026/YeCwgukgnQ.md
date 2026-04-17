# Uncertainty-Aware LLM Probing

- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Probing the hidden states of LLMs has been broadly studied in the literature. It is known to be an efficient and effective method to elicit linguistic knowledge and even higher-level behaviors from the models; for example, factual accuracy and uncertainty. However, most evaluations focus on specific short-form question answering scenarios. Further, the applicability of probes has been recently questioned, particularly in out-of-distribution (OOD) scenarios. Given their benefits, efficiency, and the variety of use cases, we study whether and when common uncertainty estimation techniques make probes more reliable. Moreover, we design a novel, robust gradient-based score, tailored to binary classification. We study various established uncertainty estimation techniques with a focus on probes for LLM correctness. Our evaluation covers many different LLMs and a variety of task domains with smaller distribution shifts and OOD scenarios. Most importantly, we find that, in LLM classifiers such as guardian models (i.e., a type of LLM judge to evaluate prompt safety), uncertainty estimators may considerably improve probe applicability. Generally, they, however, only work in domain. We specifically find that none of the methods studied works reliably under smaller distribution shifts. Overall, our score consistently ranks among the best performers.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper investigates whether uncertainty quantification methods can improve LLM probe reliability under distribution shifts. The authors evaluate several established UQ techniques and propose two gradient-based scores: JDGrad and ADGrad. Experiments across guardian models and QA benchmarks with multiple LLMs show that most methods perform reasonably in-domain, with deep ensembles and ADGrad ranking best, while JDGrad provides useful OOD detection signals.

### Strengths
- The paper identifies a genuine issue in the probing literature, where probes fail under distribution shifts. The motivation for exploring uncertainty quantification methods is reasonable.
- The authors test multiple UQ methods across diverse settings and several LLMs, providing a useful empirical comparison of existing approaches.
- The paper is generally well-structured with clear sections, and the writing is accessible, making it easy to follow.

### Weaknesses
I do have some concerns with the paper that I believe should be addressed:

- First, there is a misalignment between the motivation and the contribution of this paper. The paper's title and motivation focus on making probes reliable under distribution shifts. However, I found the main contribution of this paper, JDGrad's OOD detection capability, only identifies when probes might fail, without actually fixing the reliability issue. I believe the core problem the authors should address is making probes work reliably under slight distribution shifts, not just detecting OOD samples after the fact.
- The authors prove in lines 204-207 that $s_{ADGrad}(h) = |1 - 2f_\theta(h)|$, which reveals that ADGrad is simply a transformation of the probe's output probability rather than a new UQ method. More puzzlingly, Fig. 7 shows that "calibrated probe probability" performs best, which should be nearly equivalent to ADGrad. Additionally, the paper does not explain the existence of these discrepancies or offer a theoretical justification for the proposed transformation's benefits.
- I found that an Unfair computational comparison exists. Deep ensembles utilize 50 models, whereas the gradient-based methods are post-hoc computations on a single probe; the paper does not adequately discuss this significant difference in computational cost.
- While the paper claims strong results on guardian models, these tasks have highly constrained characteristics (e.g., fixed prompt templates, binary classification), and Table 1's bottom half shows that improvements on conventional QA tasks are marginal at best. This raises serious concerns about whether the proposed methods generalize beyond the cherry-picked scenarios where they work well.

### Questions
Please see the weaknesses I've outlined.

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
The authors examine how effective uncertainty quantification methods are at quantifying the uncertainty of LM probes, and design their own gradient-based quantification method. They find that no method is robust to domain shifts, but that their method can detect OOD effectively.

### Strengths
1. The method detects OOD data more effectively than other uncertainty quantification methods.
2. Examining the uncertainty of LLM probes is an interesting idea that I have not previously seen explicitly examined. 
3. The thoroughness of doing a human analysis of failures demonstrates diligence on the part of the authors, and is a good reminder to take label noise into account, which may be under-discussed in many UQ methods.

### Weaknesses
1. The writing overall tends to be unclear and not well-organized. Many sentences lack specificity or need to be better defined. (ex. Line 58 “turns out to be sub-optimal”—how? Lines 75-77, “Based on evaluations on uncertainty quantification in traditional ML, we would expect the uncertainty methods to work reliably here”—which evaluations, and why?)
2. Some of the results would benefit from further analysis, as inconclusive results are reported without investigation into the cause. For instance, lines 313-314, “More precisely, we sometimes observed very high performance, but very low one at other times”, does not present any hypothesis for why this was the case.  
3. The performance benefit on ID data does not appear to be consistent and significant. It only outperforms all other methods in “average rank” with 2/5 models. Figure 2 also suggests to me that these metrics all achieve similar scores, which does not present a compelling case for using ADGrad over other established methods.
4. While OOD data detection is useful, it is not immediately obvious why this should be compared to uncertainty quantification methods as baselines and not baselines for OOD data detection. This seems like a separate application.

### Questions
Why report the average rank of the scores and not simply average over all categories? This seems somewhat arbitrary.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Authors study uncertainty quantification of probes, specifically looking at guardian models. They argue that gradient based methods are the only ones that work at all under distribution shift, and improve on them to also be competitive in terms of uncertainty quantification. This improvements is a normalization of the gradients, which avoids variations in scale which could happen in data shifts. Use cases are provided.

### Strengths
Litterature review is well explained. Math is quite straightforward, useful, and seems to come for a relevant question. Multiple datasets, baselines, and models are used to empirically validate hypothesis. Extensive results are provided in Appendix to guide future works.

### Weaknesses
A reminder of the basics of how Guardian model works would be relevant, they are quite central to the empirical part.

Some dataset choice seems quantitative, all datasets, and the differences between them could be discussed more. Do the fail cases correspond to stronger distributions shifts? are they different to the ones observed by other methods? 

The discussion of results was hard to follow for me, require multiple read throughs - perhaps it would help to clarify the setting, and group results in subparagraphs instead of one block.

How usable is the proposed method out of distribution? Is the take home that this new gradient based method is the best OOD, but that it still performs too chaotically to be trusted with critical use cases? 

(Minor) There are many unclear sentences, this makes understanding the narrative harder:
e.g.
Line 148/149 "Moreover, give" should probably be "Moreover, we give"?
line 149/150 is BNN for Bayesian Neural Network?
line 157 "get away with the noise" might be "remove some noise"?
line 371 underperfored should be underperformed
etc...

(Minor)
Reproducibility could be improved by providing hyperparameter and prompts.

### Questions
see weaknesses

### Soundness
3

### Presentation
1

### Contribution
3
