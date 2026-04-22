# Verbosity Tradeoffs and the Impact of Scale on the Faithfulness of LLM Self-Explanations

- Avg Score: 3.33
- Decision: Reject
- Scores: 2, 4, 4

## Abstract
When asked to explain their decisions, large language models (LLMs) can often give explanations which sound plausible to humans. But are these explanations faithful, i.e. do they convey the factors actually responsible for the decision? In this work, we analyse counterfactual faithfulness across 75 models from 13 families. We analyze the tradeoff between conciseness and comprehensiveness, how correlational faithfulness metrics assess this tradeoff, and the extent to which metrics can be gamed. This analysis motivates two new metrics: the phi-CCT, a simplified variant of the Correlational Counterfactual Test (CCT) that avoids the need for token probabilities while explaining most of the variance of the original test; and F-AUROC, which eliminates sensitivity to imbalanced intervention distributions and captures a model’s ability to produce explanations with different levels of detail. Our findings reveal a clear scaling trend: larger and more capable models are consistently more faithful on all metrics we consider. We release our code for reproducibility.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper revisits the Correlational Counterfactual Test (CCT) for measuring LLM self-explanation faithfulness and shows that it can produce misleading results due to verbosity biases in model explanations. The authors systematically analyze this issue and propose phi-CCT to fix these issues. Experiments are conducted across several benchmarks (e-SNLI, ECQA, ComVE) with very numerous models.

### Strengths
* S1. The paper is written well, clear, and easy to follow.

* S2. It covers the widest range of LLM models and families I've seen in a faithfulness paper. Unfortunately, this wide variety of models is tested just by one family of faithfulness tests, which is a bummer, see W1.

* S3. The proposed phi-CCT metric is a sensible and well-motivated refinement of CT and CCT, even if the methodological novelty is relatively modest. It is somewhat unfortunate that CCT with its susceptibility to gaming has become an “established” baseline primarily by virtue of publication rather than robustness, necessitating yet another paper to correct its shortcomings.

### Weaknesses
* **W1. Narrow focus:** The paper’s evaluation remains narrowly focused on correlational faithfulness metrics (namely CT, CCT and now phi-CCT), without situating results against other faithfulness metrics. Prior work (e.g., [1] and [2]) has shown that different faithfulness metrics can yield strikingly divergent results even on the same models and data, raising the key question of which metric is more reliable, or at least how this new one differs conceptually and empirically. It is somewhat disappointing that the paper remains confined to testing many, many models on just the counterfactual editing family of tests, attempting to refine CCT rather than engaging with the broader concerns of faithfulness evaluation.


* **W2. Improves already flawed tests:** This work improves upon work with fundamental limitations, without addressing or fixing these fundamental limitations. Three concerns:

*Concern 1:* The proposed phi-CCT still inherits conceptual weaknesses from the original CT test, which has been criticized for its lack of correlation with other faithfulness measures [1]. This bears the question of whether CT and its evolutions actually measure the right thing.

*Concern 2:* My extra concern with CT (and all subsequent evolutions of it), is that it works under unreliable assumptions: namely, that if a (random) input change leads to an output change which is not mentioned in the explanation, the model must be unfaithful. Given that neural networks, including LLMs, are inherently sensitive to adversarial perturbations, such an assumption risks labeling all models as unfaithful under some perturbation. The use of random rather than targeted interventions further complicates interpretation: while the authors say they follow Atanasova et al. in applying random edits, this is misleading, because Atanasova et al. explicitly contrasted random and targeted edits and observed important variation between them. 

*Concern 3:* CT and evolutions (incl. phi-CCT) need to check whether the edit (the inserted word) was mentioned in the model’s explanation. And they do this check using a simple string-matching approach, which is flawed as it cannot detect synonyms, hypernyms, or negated mentions, and may falsely trigger on irrelevant references (e.g., “this detail is irrelevant”). This triggers the question about the reliability of all CT frameworks.

Overall, it is unclear why continued effort is invested in refining a metric whose foundational premise remains so problematic.

* **W3. No sanity check of the proposed test:** While the paper introduces a new faithfulness measure, it does not include any sanity checks or validation experiments to demonstrate that the metric measures faithfulness. Recent causal evaluation frameworks [2] provide principled ways to test whether a faithfulness metric truly captures faithfulness. Applying such tests would be essential to establish that phi-CCT measures faithfulness rather than surface correlations or artifacts.

References:
 - [1] Parcalabescu and Frank, 2024 cited by the paper
- [2] “A Causal Lens for Evaluating Faithfulness Metrics” (Zaman et al., 2025).
- [3] Atanasova et al. cited by the paper

### Questions
see weaknesses, which are all a big question mark about why focus so much on CT?

L 304 "commonmon sense.See A" missing whitespace

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper conducts a large-scale empirical evaluation of LLMs to identify how faithful they are to variations in the prompt through counterfactual testing. They propose a slight variation on existing metrics to account for the full confusion matrix of results and not allow the metric to be trivially gam-able by the LLM being overly verbose in its CoT.

The takeaway message is that as LLMs get bigger, they are more faithful.

### Strengths
The empirical evaluation is a primary strength. Analyzing many LLMs and prompt regimes provides a breadth of evidence that is, to my knowledge, the broadest faithfulness study to date. This makes the findings, especially the scaling trends, highly credible.

By demonstrating that phi-CCT is a strong proxy for CCT, it is a nice practical metric for researchers.

### Weaknesses
The finding that faithfulness scales with model size is interesting, but I wonder how sound the evaluation really is. Other studies such as Matton et al. [1] have actually found the opposite, and their faithfulness metric was more robust than the author's proposal here in my opinion, as it involved multiple concept edits per instances and an aggregation of this. 

I wonder if the proposal to essentially insert a word into the prompt, check for an output mention of it is really measuring the type of faithfulness that is really important, you would have to check if the model is actually saying it is using that word/concept in its decision. 

For example, if you insert a word into the prompt, the classification doesn't change, but it mentions the word in such as way as to say that it is ignoring it and not using it in the classification, then I believe this metric would give an incorrect outcome right? Or do I misunderstand?




[1] Matton, K., Ness, R., Guttag, J. and Kiciman, E., Walk the Talk? Measuring the Faithfulness of Large Language Model Explanations. In The Thirteenth International Conference on Learning Representations.

### Questions
Can you elaborate on my points about Matton et al. and the potential weakness of your metric because it does not consider if the LLM is actually stating it's using that word/concept in its classification?

I'm also curious why you think your results are contradictory to theirs?

If you can address these points, I will reconsider my score, many thanks.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The goal of the paper is to evaluate the faithfulness of LLM self-explanations. The paper builds on the prior work that proposes the Counterfactual Tests (CT) and Correlational Counterfactual Tests (CCT). The idea of both tests is to insert some tokens in the input and monitor the effect on the output of the model and its explanations. The paper formalizes the two metrics using a common terminology. The paper then uses this formalism to define the notion of alpha gemeability of faithfulness metrics and defines when the metric would be a bad metric. The paper then proposes a new faithfulness test called phi CCT test, which is a simple variation of the CCT test. While the original CCT test depends on probabilities, the proposed test consists of binary indication of the model decision being changed. The metric is then further built upon to measure faithfulness of self-explanations.

### Strengths
1. The paper is generally quite well-written and the formalism is easy to follow. For instance, the _C and _D subscript terminology when defining the interventions is very well executed. 
2. The framing of the contributed w.r.t the related work is clear. While the key improvement that the paper proposes is a relatively small one (moving from probabilities to discrete labels), it seems quite intuitive. 
3. The formal treatment of various metrics is very helpful. It provides clarity on how the various tests relate to each other and also highlight the flaws of CCT for instruction tuned models that respond in natural language. The insight around probabilities not being good indicators in free form generation is also a useful one.
4. The experimental section is large in scale and shows clear, actionable trends in terms of model sizes.

### Weaknesses
1. It is not clear from the paper exactly what explainability method is being evaluated and what the implications are. The paper makes some intervention on the inputs by inserting some new words, but that could generate out of distribution sentences that do not mean much in the specific domain while sounding linguistically natural. Did the paper manually characterize the kind of differences the interventions made to the inputs?
2. The paper should also spend some time discussing how these purely computational metrics correlate with human preferences. If the faithful metric value is high, did the humans also find the corresponding explanations useful for some downstream task, e.g., finding bugs in the model behavior, discovering bias, or offering actionable recourse? These are usually the key desiderata behind generating explanations. See for instance [Doshi-Velez and Kim](https://arxiv.org/pdf/1702.08608) and [Wachter et al.](https://arxiv.org/pdf/1711.00399).
3. While the paper is generally well-written, the prompting part in Section 4.1 can use much more detail. For instance, it would be great if the paper discussed the example prompts for prediction and self-explanation and the choices that were made to arrive at them. Were the same prompts used for all models? Could it be that some models could have benefited from prompt tuning?
4. It would also have been great to study the effect of chain-of-thought prompting on the faithfulness of the explanations.
5. In general, it's not clear if the faithfulness test is compatible for all kinds of self-explanations or just the ones considered here? For instance, are feature attribution based, or counterfactual explanations from Madsen et al. are also covered?

### Questions
1. How did we extract the model decisions from the free form generations? How effective was this strategy?
2. Line 371: How often did the model follow the instruction on explanation length?

### Soundness
2

### Presentation
3

### Contribution
2
