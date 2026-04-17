# What's In My Human Feedback? Learning Interpretable Descriptions of Preference Data

- Decision: Accept (Oral)
- Scores: 6, 8, 8, 4

## Abstract
Human feedback can alter language models in unpredictable and undesirable ways, as practitioners lack a clear understanding of what feedback data encodes. While prior work studies preferences over certain attributes (e.g., length or sycophancy), automatically extracting relevant features without pre-specifying hypotheses remains challenging. We introduce *What's In My Human Feedback?* (WIMHF), a method to explain feedback data using sparse autoencoders. WIMHF characterizes both (1) the preferences a dataset is capable of measuring and (2) the preferences that the annotators actually express. Across 7 datasets, WIMHF identifies a small number of human-interpretable features that account for the majority of the preference prediction signal achieved by black-box models. These features reveal a wide diversity in what humans prefer, and the role of dataset-level context: for example, users on Reddit prefer informality and jokes, while annotators in HH-RLHF and PRISM disprefer them. WIMHF also surfaces potentially unsafe preferences, such as that LMArena users tend to vote against refusals, often in favor of toxic content. The learned features enable effective *data curation*: re-labeling the harmful examples in Arena yields large safety gains (+37%) with no cost to general performance. They also allow fine-grained *personalization*: on the Community Alignment dataset, we learn annotator-specific weights over subjective features that improve preference prediction. WIMHF provides a human-centered analysis method for practitioners to better understand and use preference data.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces WIMHF, a method that uses sparse autoencoders to turn human preference data into a small set of clear, natural-language features. These features explain much of the preference signal and reveal both what a dataset can measure and what annotators actually prefer, surfacing harmful biases like preferring non-refusals in Chatbot Arena. The authors show practical gains: flipping labels on unsafe Arena pairs markedly improves safety without hurting overall performance, and the features act as interpretable knobs for controlled personalization. Overall, WIMHF offers a simple, data-driven way to audit and steer preference datasets for alignment.

### Strengths
1. The paper learns sparse, natural-language features from preference pairs using a sparse autoencoder, so it can explain both what a dataset can measure and what annotators actually prefer. These features capture a large share of the preference signal (about two-thirds of a black-box model’s gain), align with annotator explanations, and beat a prior baseline.

2. WIMHF is applied to seven widely used feedback datasets, revealing which preferences are measurable and which are realized. It also shows how data collection choices affect measurable preferences, e.g., diversity differences between multi-model high-temperature sampling and prompting a single model.

3. The method leads to actionable safety improvements. IT flags unsafe preferences in Chatbot Arena, such as judges disfavoring refusals and favoring sexual/toxic outputs. Flipping labels on the most affected pairs boosts RewardBench2 safety accuracy from 8.9% to 46.2% without hurting non-safety performance, turning the analysis into a simple, effective curation step.

4. The paper identifies subjective features at the annotator level and shows that learning annotator-specific weights improves held-out AUC. It further demonstrates that actively sampling high-value feature examples is more sample-efficient than random sampling, giving a simple recipe for controlled personalization.

### Weaknesses
1. WIMHF uses LLMs to write feature descriptions and to judge “fidelity,” and also relies on an LLM judge for cross-dataset checks. This may introduce circularity and judge bias. Consider adding human audits, multiple independent judges, and agreement tests to stress-test these steps.

2. The method drops prompts because including them did not help simple prediction, but many preferences are conditional on the prompt. This can mislead downstream use. Maybe add prompt-aware features, counterfactual tests, and prompt-conditioned regressions to separate context-specific from global preferences.

3. Interpretable features recover only ~⅔ of a reward model’s AUC gain on average, and much less on HH-RLHF, suggesting important signals are still missing.

4. The label-flipping demo uses one small reward model and one main safety benchmark, then infers large leaderboard shifts. It would be better to validate across model scales/family and include one more safety benchmark.

5. The pipeline removes non-English/very long items and even drops rows where both responses are marked “subjective” by an LLM. These choices could suppress hard cases and alter subjectivity estimates. Please report ablations on each filter (especially the subjective filter) and consider including multilingual analyses to test robustness.

6. Personalization is promising but thinly evaluated. Gains are modest and shown mainly on one dominant feature for high-volume annotators. Extend to multi-feature personalization, low-data users would be better.

### Questions
1. How robust are the learned features to the choice of LLM used for description and fidelity checks?

2. In the evaluation and Table 1, why don’t the colors (preferred vs. dispreferred) align with the “∆win” values? Do the colors indicate statistical significance? This could be made clearer.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
How does preference optimization change language models, exactly? This paper trains SAEs contrastively between preferred and dispreferred pairs of responses to extract features underlying human preference. A language model is prompted with corresponding exemplars to generate human-readable summaries of these differences. The paper presents these summaries for several conversational datasets, also showing that human preference may differ or even oppose preferences collected from other sources or in other settings. Finally, the authors rewrite toxic examples identified from these preferences to show how they may be put to practical use.

### Strengths
This paper represents a rare case of an actual practical application of methods from Mechanistic Interpretability to the broader sphere of influence of LLMs. I can list several strengths here:

1. The overview of related work is both comprehensive and educational. 
2. The figures look nice, and the paper is well written. I had no trouble following the writing, or understanding the examples.
3. The measurement of the variation in human preferences collected from different sources shows that people from different groups might prefer contradictory values—while not entirely surprising, this insight was unexpected for me.

I commend the authors on a well-written paper.

### Weaknesses
I only found two minor weaknesses in the paper.

1. At this moment, there is no intrinsic evaluation of how good the feature extraction pipeline is—specifically, you have shown good precision of extracted features, but not good recall. I don’t expect many changes for the review period, but I would bring up a potentially helpful experiment: why not create synthetic datasets where two responses differ precisely in some trait like “Likes to include anecdotes about pirates,” or other clearly identifiable traits, and see how many your pipeline can recover?
2. While the results on rewriting toxic examples are interesting, I have some concerns about whether such an approach can work more broadly (i.e., are there other examples of how identified traits can be used to modify or filter data?)

### Questions
I have one question for the authors. I had previously attempted to train SAEs contrastively to extract features at some point, but I recall that in my experiments, I found that the exemplars repeatedly led to the judge model (I believe it was GPT-4o at that point) outputting cookie-cutter explanations like “It is more helpful and comprehensive.” How did you manage to get precise explanations for the features? Did it require any prompt engineering? Or do you think using BatchTopK-SAEs allowed the extraction of more precise features where prompt engineering wasn’t required?

### Soundness
3

### Presentation
2

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
This work focuses on the problem of interpreting human feedback, specifically finding explanations without pre-specifying hypotheses. Unlike previous approaches to this problem, this work uses sparse autoencoders (SAEs) on top of response embeddings to find features that human annotators may have followed. The authors refer to this method as What's In My Human Feedback (WIMHF). The authors demonstrate their method's effectiveness across a diverse set of datasets, including Chatbot Arena, Community Alignement, HH-RLHF, and more. They also illustrate how their method can be used for a number of downstream applications, such as improving safety of models, excluding misaligned preferences from evaluations, and personalisation.

### Strengths
This is a well written paper, that exhibits a number of notable strengths:

1. **Introduces a novel approach to feedback interpretation problem.** The problem of understanding human feedback remains very timely and important, given the importance of such datasets across training and evaluation pipelines. The approach introduced by the authors improves on prior work tackling the same problem (Inverse Constitutional AI, ICAI) in multiple ways:
    - Adds notion of measurable preferences in addition to realised preferences: Beyond just considering explaining the preference expressed by human annotators, this work also provides information about what can be preferred in a given dataset of prompt + response pairs. (Caveats apply as discussed in weaknesses)
    - Improves efficiency: using embedding models with a small SAE model is far less costly than the full-scale LLMs in prior work.
    - According to the experimental results, improves number of relevant features found over prior work, demonstrated with a number of interesting new insights.
2. **Convincingly demonstrates utility in extensive experiments.**  The results illustrate that their method is applicable across a wide range of datasets and use-cases, e.g. recovering known preferences and finding conflicting preferences across datasets. The experimental details are fully provided and the analysis thorough. The scope of these experimental results extends beyond prior work and thereby helps inform future work in this direction.
3. **Well written and executed.** Overall the paper is mostly clear and straightforward to follow, though some experimental discussions are very dense and rely heavily on the appendix. The experiments are well motivated and executed.

### Weaknesses
The work has a few weaknesses to consider:

1. **Correction for length seemingly arbitrary.** The approach currently corrects for length. There are a number of other well-known biases with human/ai annotators (e.g. position bias, preference for structured outputs, formatting, emojis etc.). How come you chose to correct for length only? This decision appears arbitrary and adds complexity to the method (Step 3 specifically). Shouldn't the length be automatically detected as one of the features? At least the decision to correct for length but not other features should be explicitly justified.
2. **Missing discussion of important limitations.** The proposed method (and prior approaches for the feedback data interpretation problem) have important limitations that are currently not explicitly discussed as far as I can see. In particular, the following limitations are not discussed but are important for downstream users:
	1. **Measurable (and realized) preferences are not exhaustive.**  There are multiple factors that may prevent the method from finding all theoretically avaialble measurable preferences: (1) the current experiments fix the number of features extracted by the SAE (in Step 1) to a small number, (2) the embedding models may not be able to capture very complex features of the responses, (3) the feature may not be extractable from embedding differences, and (4) the measurable preferences may depend on the prompt (which is not used in current experiments). These limits should at least be acknowledged.
	2. **Natural language preferences descriptions are not unique.**  The work fails to acknowledge that the natural language descriptions in Step 2 of features are not unique. An SAE feature may be explained by many different natural language descriptions. In particular, if two natural language features consistently correlate (e.g. using bold and italics formatting), the interpretation of an SAE feature is fundamentally ambiguous. Providing results re-running the pipeline and comparing the resulting features would be useful to estimate the impact of this and the previous limitation.
	3. **Correlation $\neq$ causation in realized preferences.**  Due to the previous point, as is common in interpretability work, the method can only show that human preferences correlate with the discovered realized preferences, rather than that human preferences *are* these realized preferences. Using the same hypothetical example as before, the annotators may only care about bold text but the method may declare that they care about italics text, and vice versa. This distinction is subtle but overconfidence can mislead users of the approach into incorrect beliefs about their annotators. The authors explicitly state that annotators "prefer" or "disprefer" (e.g. L295) certain things, which may be misleading without acknowledging this limitation. *Note that whilst this limitation is important to acknowledge, it does not strongly negatively impact the utility of the method (e.g. the improving the safety of trained models certainly still holds). Overall, the results should simply be viewed with awareness of the limitations.*
3. **No tie predictions.**  With the binary logistic regression as described in L165, tie votes are currently not considered. Nevertheless understanding the nature of ties, what leads to ties, also seems to be an interesting aspect of pairwise feedback -- that may be relevant to model developers trying to avoid or understand ties.
4. **Measurable preferences not directly comparable across datasets.** The measurable preferences rely on an SAE trained per-dataset: a feature is deemed measurable if the SAE learns it, otherwise it is not. This leads to dataset-specific features. Across datasets, these features may be similar but are not fully directly comparable/equivalent (not the same SAE). A direct comparison would make the results in Section 4.2 quite a bit stronger. This limited comparability with respect to measurable preferences appears to be a fundamental part of the approach in its current form.
5. **Limited comparison to prior work.** Currently the comparison to prior work originally introducing the feedback interpretation problem (ICAI) is limited to comparing the number of statistically significant features generated by each (and related analysis). Whilst this comparison is relevant, the authors omit a direct comparison of the overall combined reconstruction capabilities of the features generated by the two approaches, across different datasets. Such an analysis would be insightful and help make it easier to compare future approaches on this problem. Further, in the current comparison the prior work seems to be negatively affected by assumptions about the feature generation problem, assuming non-redundancy and no correlation with length, which are not necessarily objectives directly considered in the prior work. Thus, both uncorrected and length-corrected results would be interesting to understand the differences.
6. **Incorrect claim in L483:** This claim about no prior work being interpretable and data-driven does not appear to be true - ICAI seems to be a prior method that is both. (*"Unlike prior methods to understanding feedback data, WIMHF’s approach is both interpretable and data-driven, enabling discovery of new hypotheses without pre-specifying attributes to measure."*)

Overall, I consider this work to be of solid quality with high impact and, if the weaknesses are adequately addressed, may be suitable for spotlight or other recognition.

*Minor (no impact on score, no need to respond):*
1. In L164: would be good to have a number for this equation

### Questions
1. Generally, for each limitation/weakness above, I would be curious whether you agree and, if, how you are planning to address the concerns.
1. For the logistic regression in L164, the label $y$ is assumed to be binary, either response A or response B, correct? This could be clarified.
	- How would you adapt the proposed approach to allow for predicting tie votes (e.g. both good, both bad)?
2. How does the method ensure that features $z_i$ are not similar? In Appendix A.1 this is briefly discussed, but I would be curious how larger values of $M$ impact the results and lead to duplicates and redundancy?
3. How consistent are features across different training runs of the SAEs under the same settings?
4. Would you be able to extend on the rationale for excluding queries with objective correct answers? A lot of important observed biases are also applicable or only observable across such queries (e.g. preferring confident answers over truthful answers).
5. L301 onwards: Why don't you apply multiple SAEs across datasets? I don't understand why you can't transfer the SAEs directly: trained on dataset A and then used to understand the realised preferences on dataset B. Is there something fundamental preventing such a transfer?
6. Are you planning to release the code?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces What's In My Human Feedback (WIMHF), a novel method to automatically discover and interpret the preferences encoded in human feedback datasets. The technique operates by training a Sparse Autoencoder (SAE) on the differences between text embeddings of preference responses. This process identifies a small set of key features that differentiate the responses. These features are then translated into human-readable descriptions using an LLM. The authors demonstrate that WIMHF can successfully surface important, and sometimes harmful, preferences within data, such as a bias towards unsafe content. This allows for targeted data curation and a more transparent understanding of how preference data shapes model behavior.

### Strengths
+ The paper tackles the important and challenging task of understanding the implicit biases and preferences within human feedback data, distinguishing between what is "measurable" in the data and what is a "realized" human preference.

+ A key strength is the demonstrated ability of WIMHF to automatically detect misaligned or unsafe preferences in widely-used datasets, providing a practical tool for improving model safety.

### Weaknesses
- The paper does not provide sufficient ablation studies to justify each component of its methodology. It is unclear if every step, particularly the use of a Sparse Autoencoder (SAE), is necessary or if a simpler approach could achieve similar results.

- The experimental setup is often described with insufficient detail, making it difficult for readers to fully understand the settings and reproduce the results.

- Please see further questions.

### Questions
Could a simpler baseline, such as using an LLM to directly explain the preference between two responses and then clustering those explanations, achieve similar results? Is the SAE a necessary step?

How is the difference in text embeddings (e_delta) calculated? Is it the embedding of the chosen response minus the rejected one, or another way?

How do the results vary when using different text embedding models? Could the choice of the OpenAI text-embedding-3-small model introduce bias?

The features are described in natural language by an LLM. How can you be sure that the LLM's description is the most accurate or salient interpretation of that feature, and not just one of several possibilities?

The qualitative validation relies on experts rating LLM-generated feature descriptions. How does this validate the underlying features discovered by the SAE, rather than just the descriptive capability of the LLM?

For datasets generated by a limited number of LLMs (like Community Alignment), could the lack of response diversity artificially limit the types of features WIMHF can discover?

Could you clarify the rationale for filtering both subjective queries and objective ones (like math and coding) from the datasets?

Figure 4 indicates that the SAE method loses more information for datasets like HH-RLHF and Reddit. What properties of these datasets make their preferences harder to explain with interpretable features?

Could you provide more detail on how the "win-rate" metric is computed and on what specific datasets or splits it is evaluated?

### Soundness
2

### Presentation
2

### Contribution
3
