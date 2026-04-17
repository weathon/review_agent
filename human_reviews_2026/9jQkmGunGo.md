# Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity

- Decision: Reject
- Scores: 4, 2, 6

## Abstract
Post-training alignment often reduces LLM diversity, leading to a phenomenon known as mode collapse. Unlike prior work that attributes this effect to algorithmic limitations, we identify a fundamental, pervasive data-level driver: typicality bias in preference data, whereby annotators systematically favor familiar text as a result of well-established findings in cognitive psychology. We formalize this bias theoretically, verify it on preference datasets empirically, and show that it plays a central role in mode collapse. Motivated by this analysis, we introduce Verbalized Sampling (VS), a simple, training-free prompting strategy to circumvent mode collapse. VS prompts the model to verbalize a probability distribution over a set of responses (e.g., "Generate 5 jokes about coffee and their corresponding probabilities"), which relieves the pressure to produce a single "typical" answer. Experiments show that VS significantly improves performance across creative writing (poems, stories, jokes), social dialogue simulation, synthetic data generation, and open-ended QA, without sacrificing safety and factual accuracy. For instance, in creative writing, VS increases diversity by 1.6-2.1x compared to direct prompting. We further observe an emergent trend that more capable models benefit more from VS. In sum, our work provides a new data-centric perspective on mode collapse and a practical inference-time remedy that helps unlock pre-trained generative diversity.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a simple method, called verbalized sampling, to improve LLM diveristy. It's simple—get a model to generate responses _with probabilities over the answers_, and then pick from that according to some metric. They show that this improves diveristy at a given quality level. The submitted version has some issues that I'd like to see resolved, but overall I think this could be a nice paper, if the issues are resolved.

### Strengths
- Nice to have clearly defined definitions in Table 1, which I found to be very clear. Nice.
- I found Figure 2 to be well presented and clear, and I was happy to see the diversity and quality score here.
- Method is very simple, which I think is great, and its great to study the probability threshold and scaling.
- I think diversity with fixed quality is important, especially for synthetic data scaling. Indeed, I found the setting in section 7 compelling and useful.

### Weaknesses
- the analysis in section 3.1, using a linear reward model is interesting but I believe it may have a flaw. The probability under the base reference model does _not_ just capture typically, but will capture many features, such as correctness, etc. Therefore, i'd like to see additional analysis for 3.1. in particular, what % of time in preference data is the more typical response, using BOTH a response with higher log-likelihood under pretrained model AND a simple prompted model preferred. You could use a simply logistic regression feature approach, similar to the Sharma et al Sycophancy paper. You could do this for example on some other PM datasets too, which would be helpful. 
- Moreover, you make claims about what _humans_ prefer, but in fact, _many_ humans are involved in the RLHF process, and it's unclear that they will all think the same text is typical. E.g., compare US and British english speakers. And the analysis is about the reward model scores, _not_ what humans actually prefer. Therefore, you should do the analysis I suggest above to isolate these questions. There are thus two questions: (i) what do humans prefer; (ii) what do PMs prefer. Different questions, worth studying individually, and you should not conflate the results of each type of analysis.
- there are some extra baselines I'd like to see:
    - I'd like to see the effect of the method compared to a pretrained model directly. How much of the diversity is actually recovered?
    - i'd like to see an input seeding example. e.g., "pull a random document from this set of 100" and then do the task. This is environmental randomness.
    - it is exactly right to look at diveristy, quality parteo curves. I suggest making this the main figure, and including it in the main text for Fig 2, **across all types** of task.
    - I'd like to see an additional quality check, which uses e.g., LLM as a judge and computes win-rates against a reference set, rather than just using rubrics.
- The writing is technically imprecise at times. small notes, as examples of this are below. In general, I'd like to see the strength of claims toned down and precision improved.
    - intro line 65: please crispy define typicality.
    - intro line 95: why is it principled? please explain crisply in text.
    - line 206 "to recover the diversity level". does this actually rcover it?
- I want to understand why the verbalization of probability distribution is helpful here? Do you have intuition around this? Can you explain this in the text?

### Questions
also, for Table 2, do you have a human assessment of quality for VS-standard vs e.g., direct?

(see weaknesses for most questoins)

### Soundness
2

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
3

### Summary
The paper argued that one important cause of the mode collapse is the biased preference data. The authors investigated this “typicality bias” that once quantified is shown to be detrimental to model output diversity. 
The authors propose a new prompt based method to bypass mode collapse, specifically by prompting models to list different response candidates in JSON while also listing their corresponding probability

### Strengths
- Overall, a solid and refreshing investigation into typicality bias, which identifies a key potentially undesired property in LLM preference data, and I believe this has great potential to unlock more LLM diversity.
- For each configuration of verbalized sampling, the experiment is thorough with good coverage of models, domains, varying levels of interactiveness, metrics, and human annotations.

### Weaknesses
- The motivation of the paper and the method feel mismatched — reading the paper feels like reading two. Typicality bias was thoroughly investigated and validated, but instead of continuing investigating any de-typicalized reward pairs, the authors used an unconvincing (see next point) reward-irrelevant method instead. The connection of typicality bias and verbalized sampling is at best, as the extensive appendix tried to prove under strong assumption, orthogonal. Line 203 to 207 (roughly) is not a coherent explanation either.
- Unsatisfactory baselines. The method used in the paper claims that requesting the distribution in the prompt is the key to improve observed diversity. However, strictly comparable list-based counterparts to "VS-*" methods probability request is subtly missing. To break it down Table 1:

| VS (k>1)          | List Level (k>1)  | Instance Level (k==1) |
|-------------|-------------|----------------|
| `VS-Standard`  | `Sequence`  | `Direct`       |
| `VS-CoT`        | **null** | `CoT`          |
| `VS-Multi` | **null** | `Multi-Turn`*  |

*`Multi-Turn` is an unfaithful and unacceptable misnomer. Unlike `VS-Multi` it only samples one more candidate per turn / call. So it is de facto instance level.

The missing comparable counterparts, which could be named `Sequence-CoT` and `Sequence-Multi`, are essential in evaluating the proposed method’s efficacy. Prior works and this one have shown that list does increase diversity over instance-level.

### Questions
- Line 178 claims "Having confirmed typicality bias". This claim lacks a description and content of the domains and types of the data. No description other than the dataset name is mentioned, yet the authors did not specify any conditions where the conclusion "human raters are biased towards responses more typical for the base model" are based on. So what are the domains of HelpSteer? How generalized can your claim be? Description about Eq1 lacks such context.
- For `α > 0 means that, holding the true utility fixed, higher typicality bias increases the reward`, it seems that some styles might just be universally preferred by users that the base learned to prefer already, which might drive up `α`. What could be a baseline `α` if not 0?
- For Creative writing evals, why `text-embedding-3-small` and `Claude 3.7`? Some explanation can be helpful.
- Double-check line 243. Not very readable.

### Soundness
1

### Presentation
4

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
* The paper identifies **typicality bias** in human preference data as a key cause of **mode collapse** in RLHF-aligned large language models.
* It shows theoretically and empirically that this bias sharpens the model’s distribution, reducing output diversity.
* The authors propose **Verbalized Sampling (VS)**, a simple, inference-time prompting method that restores diversity without retraining, achieving higher variety while maintaining quality.

### Strengths
1. **Novel framing**
   The paper introduces a fresh and compelling perspective by identifying *data-level human preference bias* (typicality bias) as a root cause of mode collapse. This shifts the discussion from algorithmic issues in RLHF to psychological factors in human annotations, offering a new concept for understanding mode collapse.

2. **Strong empirical evidence**
   The paper validates the typicality-bias hypothesis on real preference datasets such as HELPSTEER, showing consistent and statistically significant results across multiple base models. This empirical grounding gives credibility to the theoretical claims.

3. **Practical mitigation method**
   The proposed *Verbalized Sampling (VS)* is simple, training-free, and effective. It improves output diversity through prompt-level control at inference time, without modifying model weights. The method is orthogonal to existing decoding strategies and easy to adopt in practice.

4. **Strong and comprehensive empirical validation**
The experiments are comprehensive and convincing. The paper tests its hypothesis across multiple datasets, diverse task categories, and several subtasks.

### Weaknesses
1. **Limited validation of the core hypothesis**
    - The central assumption—that *human annotators prefer more typical responses*—is not directly validated.
    - The authors infer this pattern indirectly through reward model correlations. A direct analysis of whether humans explicitly favor more typical responses would make the hypothesis stronger.
2. **Inference cost and deployment feasibility**
    - While *Verbalized Sampling (VS)* effectively increases diversity, it requires generating *k* responses and verbalizing probabilities, leading to roughly *k×* higher inference cost. This makes the approach difficult to apply in large-scale LLM deployments despite its conceptual simplicity.
3. **Lack of qualitative analysis**
    - The paper would benefit from more qualitative examples to illustrate how VS improves diversity without hurting quality. Current evaluations are largely quantitative, leaving uncertainty about how these improvements appear in actual outputs.
    - In particular, there should be a qualitative comparison between generating responses individually (one-by-one) and generating multiple responses simultaneously (e.g., 5 at once). Although the paper reports quantitative gains, it remains unclear whether the perceived quality or linguistic characteristics of the responses differ between these settings. Providing human evaluations or illustrative examples would clarify whether VS changes not just diversity scores but the actual quality of the outputs.
4. **Missing ablation studies**
    - Figure 14 could also analyze how varying the number of generated candidates in a direct setting appears, and variants like VS-CoT or VS-Multi should also be compared in Figure 14. This would clarify which components of VS contribute most to the observed gains.
5. **Fairness issue in VS-Multi comparison**
- The comparison involving VS-Multi appears potentially unfair, as it effectively produces a larger number of responses (e.g., *5×k* candidates) than other methods. This difference may inherently boost diversity and performance, so comparisons should control for the total number of generated outputs across settings.

### Questions
- Q1 What is the expected inference cost for each of the proposed prompt settings?

- Q2 Considering cost, diversity, and quality, which approach appears to be the most feasible overall?

- Q3 How was the distribution of model-generated probabilities?
Did the distribution differ across different VS variants, and is there any observed relationship between those probabilities and the quality of generated outputs?

### Soundness
3

### Presentation
4

### Contribution
3
