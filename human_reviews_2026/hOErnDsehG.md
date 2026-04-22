# SelfReflect: Can LLMs Communicate Their Internal Answer Distribution?

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 8

## Abstract
The common approach to communicate a large language model's (LLM) uncertainty is to add a percentage number or a hedging word to its response. But is this all we can do? Instead of generating a single answer and then hedging it, an LLM that is fully transparent to the user needs to be able to reflect on its internal belief distribution and output a summary of all options it deems possible, and how likely they are. To test whether LLMs possess this capability, we develop the SelfReflect metric, an information-theoretic distance between a given summary and a distribution over answers. In interventional and human studies, we find that SelfReflect indicates even slight deviations, yielding a fine measure of faithfulness between a summary string and an LLM's actual internal distribution over answers. With SelfReflect, we make a resounding negative observation: modern LLMs are, across the board, incapable of revealing what they are uncertain about, neither through reasoning, nor chains-of-thoughts, nor explicit finetuning. However, we do find that LLMs are able to generate faithful summaries of their uncertainties if we help them by sampling multiple outputs and feeding them back into the context. This simple approach shines a light at the universal way of communicating LLM uncertainties whose future development the SelfReflect score enables. To support the development of this universal form of LLM uncertainties, we publish the code that implements our metric for arbitrary LLMs under https://github.com/apple/ml-selfreflect .

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors introduce SelfReflect, a novel metric to measure the faithfulness of LLMs expressing their uncertainty about answers. Specifically, they examine a certain type of expression of uncertainty, which entails an explicit natural language "description" of a distribution. The SelfReflect metric is a method to measure the consistency between such a natural language "distribution summary" and the real output distribution of the LLM. They do this by alternating putting the distribution summary or some real samples of the distribution in the context and seeing how the distribution of a masked word changes in these two scenarios. The logic: if conditioning on the distribution summary and the "real distribution" (represented by a set of samples) leads to similar masked word probabilities, then they contain the same information. **This point initially confused me: there is the judge LLM (which is giving the masked word probabilities) and then there is the LLM being measured (which gives the distribution summary and real distribution). These are not the same LLM.**

The findings are that the metric is generally more effective than the baselines they show, especially when it comes to nuanced judgements (judging distribution summaries that are subtly wrong in small ways). They also validate that it correlates well with human judgements.

Having established the effectiveness of the metric, they then proceed to measure various LLMs via this metric to see if they can generate effective distribution summaries in a single inference call, and conclude that *they cannot*. Neither simple prompting, chain-of-thought reasoning, reasoning models (like DeepSeek-R1), nor finetuning successfully get the LLMs to generate faithful summaries. They state that LLMs are effective at writing good distribution summary strings given samples from the distribution (the "Sample & Summarize" approach), but not at actually "introspecting" and generating the distribution summaries based on their own internal state.

### Strengths
I think the problem being presented (existing methods like providing a single number representing the certainty of one answer not being rich enough to understand the LLMs internal state) is interesting and well-motivated. 

The metric is validated extensively. I especially appreciate the human study.

The theoretical connection to predictive sufficiency is compelling, the proofs look rigorous.

The authors do seem to have tested the negative result extensively, trying many different approaches to have the models be able to introspect. They establish that "model introspection" is an open and very difficult problem.

Results are comprehensive and reproducible, writing quality in the paper is clear and comprehensive.

### Weaknesses
I am not sure that the authors effectively justify why a natural language summary of the answer distribution is better than just teaching the model to generate structured output representing is answer distribution: something like [("Paris", 0.75), ("Marseille", 0.15), ("Toulouse", 0.10)].

The example the authors provide in their diagram (Figure 3) is interesting, but it's not at all clear to me what happens when you extrapolate to N=50. What do the summary strings actually look like? Is it an enumeration of 30 different answers in e.g. high variance cases? Is this a useful thing to look at as an end-user? It would be interesting to incorporate some notion of "utility" for an end-user.

What is the "compression strategy" the LLMs are using to generate these distribution summaries? I'm assuming some answers get tossed. Which ones, and why, and are we losing important distributional information this way? It's not clear to me. Like I understand your metric hypothetically should avoid losing important distributional information. But I still really would like to see what the final distribution summaries look like at this level. Appendix E is somewhat informative but not sufficient: it demonstrates the metric, but not what the actual distribution summaries generated from the LLMs look like. 

I think a qualitative analysis of some examples of the summaries generated would be extremely useful.

The paper is extremely dense and long, with the appendices. Some important details are buried in the appendices, it's so long I may have missed other details, not sure.

I think the paper may be overreaching in its claims, specifically regarding the second contribution: examining if LLMs are capable of expressing distributional summaries without access to samples from the distribution, and if they can be trained to do so. "With SelfReflect, we make a resounding negative observation: modern LLMs are, across the board, incapable of revealing what they are uncertain about, neither through reasoning, nor chains-of-thoughts, nor explicit finetuning." -- I think this is slight overclaiming, they are not able to estimate their answer distribution, which is one specific way they can make the user aware of their uncertainty.

### Questions
I would want to see an Appendix with some qualitative analysis of some examples from the N=50 case. What do these summaries look like?

I would want to see some discussion of if these summaries are actually helpful for an end-user. There is no rigorous discussion of whether distribution summaries in natural language are actually useful, and what the benefit is compared to other ways to summarize the distribution of answers from an LLM.

Can an analysis of summary length vs N be done?

To be honest it feels like especially the sample&summarize method is effectively a form of implicit clustering. Like what I must infer is that the summary is essentially grouping different samples together and assigning some probabilities based on the frequency of certain samples over others. This is explicitly what you're doing in Appendix G it seems? Why is the connection to clustering not made explicit in the paper?

"The negative result shows LLMs can't introspect" why do you think this might be? Is it a training data issue, architectural limitation, or something about how RLHF works? Do you have hypotheses for what would enable introspection?"

How does your Sample & Summarize approach compare to existing methods like Semantic Entropy (Farquhar et al. 2024), which also clusters sampled answers? What's the advantage of natural language output over numerical entropy?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The authors introduce SelfReflect, an information-theoretic metric that measures how well a summary string matches an LLM’s true internal answer distribution. The SelfReflect metric measures how faithfully such a summary represents the model’s true internal answer distribution using masked-token prediction and Wasserstein distance. Extensive experiments show that current LLMs cannot genuinely articulate their internal uncertainties.

### Strengths
1. The experimental results in Section 4 including Distinguishing Good, Bad, and Almost-Good Summaries, Multiple-choice QA and Alignment with Human Judgments are solid and sufficient.

2. The explanation of why CoT fail on reflecting the LLM’s internal confidence is very interesting.

### Weaknesses
1. To be honest, I am not quite familiar with this area but would like to ask a very general question. Why do you think the topic, whether a LLM can accurately express its own confidence through natural language, is important?

2. Based on 1, do we even have a promising metric to reflect LLM’s internal probability distribution yet? If not, the topic of investigating whether a LLM can accurately express its internal probability distribution may not be reliable.

3. For Eq2, I am quite confused why the left hand side is equivalent to the right hand side? Why is the conditional probability measured by mutual information?

4. Could you please explain why we should use 1-Wasserstein distance to measure the difference between distributions instead of JS/KL divergence?

### Questions
Please check the Weaknesses. I'm not quite familiar with the background knowledge of this paper, so I'm willing to modify my rating based on the comments from other reviewers and ACs.

### Soundness
3

### Presentation
2

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
The paper introduces a new setting of uncertainty estimation where the model is required to provide a comprehensive summary of all of its possible answers. 
In order to measure the quality of the output with respect to the summary, a SelfReflect metric is introduced, which uses sufficient statistics arguments to measure the summarisation quality of the provided answer given a variety of answers sampled from a model.
The authors further investigate the quality of the resulting metric on syntetic data.
The metric is then used to determine the extent to which the model's 'internal knowledge' is expressed in its outputs for multiple models and decoding strategies.
The authors anticipate that their findings would improve the ability to communicate the model's uncertainty to the end user as well as help several other uncertainty related tasks in NLG.

### Strengths
1. The paper is clearly written, experiments are reasonably documented in the appendix. 
2. Theoretical Justficiation of SelfReflect metric appears to be sound, the proofs with sufficient statistics provided in the Appendix A are convincing.
3. Visible good effort to connect the theoretical propositions to experimental evaluation. Authors provide ablations of the SelfReflect metric as well as consider a broad range of LM decoding paradigms, including reasoning. 
4. Human feedback studies improve the 'value' of the paper.

### Weaknesses
1. Minor (little to no impact on my score):
    1. Antropomorphising the langauge models: while subjective and stylistic, I view it as a minor negative aspect. I.e. line 077: "its internal beliefs", line 82: "making LLMs aware of their internal uncertainties", Line 478: "make LLMs honestly describe", etc.
    2. Line 15: "all options it deems possible" - for a language model all options (i.e. every combinatoric token sequence) are technically "possible" unless -inf is allowed in the logits somehow. A strange way to put it unless some thresholding is in place.
    3. Line 483: "Extracting all output possibilities is also a core necessity for conformal approaches", not sure if this is factually accurate considering that what is meant by all possibilities here is extracting them as a single output string.
2. Major:
    1. Emprical benefits are unclear. The Tab. 1 presents comparison between several metrics as well as ablations of the SelfReflect. It can be seen that in Good vs Bad summaries the LM Judge is only second to the unablated SelfReflect, and even then 0.01% away from not achieving 95% significance. Tab.1 could be presented as a heatmap as well, depicting the performance for all combinations of summaries incorporating *all pairs* for a more wholesome picture. 
    2. Some experiments may have unanticipated caveats:
        1. The data for the Tab. 1 was reportedly created by using Qwen2.5 7B for answer generation and Gemini 2.0 Flash for summary creation. Given the tiny margins for the main Good vs Bad summaries experiment, one may want to ablate this by answers and summaries coming from different models / generated by other means. The almost-good summary generation appears to be even less robust, since it is created starting from an already model generated summary. The other pairings also seem somewhat arbitrary. 
        2. Fig 4: The certainty / uncertainty of the answers / summaries is ill defined. It therefore hard to pass judgement based on that.
        3. Tables 4 and 5: Reasoning model output lengths appears to be strange: it is hardly twice the length of normal outputs. RLVR tuned models from Tab.5 are not trained for summarization and fact retrieval, but for facing math and logic problems. 
    3. Although I do not have an in-depth familiarity with state of the art summarization literature, the paper fails to compare SelfReflect to summarization specific metrics. For example something on the lines of [1] (Which is also a PMI of sorts). 
    4. The second part of the contribution, the analysis on the expression of internal knowledge via different decoding schemes lacks clear placement: it is unclear how exactly we can benefit from this knowledge. The guidance in the Outlook is inspecific. 

### References:
1. Jung, J. et al. Information-Theoretic Distillation for Reference-less Summarization. Preprint at https://doi.org/10.48550/arXiv.2403.13780 (2024).

### Questions
1. Would masking on sentence level be a valid approximation of SelfReflect?
2. Could there be an alternative to using Wasserstein distance?
3. As this work largely addresses on fact recollection by the model, what impact could introducing RAG have on the performances presented?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
**Summary**: This paper investigates the problem of communicating uncertainty of the distribution over answers to a question, advocating for the generation of all possible answers and their likelihood when generating an answer to a question (instead of generating the most likely answer and a confidence score). It then proceeds with (1) proposing and validating a metric–**SelfReflect**– to quantify the faithfulness of a summary to a LLMs internal distribution over answer, (2) empirically investigating whether through various prompting techniques LLMs are able to produce such descriptive summaries of their internal distribution.

**Main contributions**:

- Novel paradigm for uncertainty quantification, requiring an answer that is descriptive of the LLM’s internal uncertainty.
- Proposal of a theoretically grounded metric, SelfReflect, that measures faithfulness between a descriptive answer (dubbed *summary*) and the LLM’s internal distribution over answers. 
- Extensive experimentation with 16 popular LLMs across 3 datasets to assess their abilities in generating descriptive summaries of their internal uncertainty either through prompting, finding them to be inapt.

### Strengths
- Novel contributions for uncertainty quantification in LLMs, offering a fresh perspective on uncertainty quantification and designing new set of metrics and evaluations;
- Proposal of a new metric that is grounded in information theory concepts, such as mutual information.
- Experimental design allows for the controlled assessment of the capabilities of the proposed metric, comparing it against numerous baselines including LLM-as-a-judge approaches.
- Benchmark 10+ models in 3+ datasets (TriviaQA, NaturalQuestions, SimpleQA, MMLU) and 5 different prompting strategies and 2 fine-tuning approaches to assess model’s abilities in generating descriptive summaries.
- Various appendices supporting various analyses and arguments in the paper – e.g., providing a convergence analysis of SelfReflect metric.
- Interesting findings, including the model’s inability (even after SFT or DPO) to provide descriptive summaries of its own internal uncertainty.
- Well-written and well-organized.

### Weaknesses
W1. **Controlled experiments in section 4 concern different models**, raising questions about the generalization of the results (see Questions). 
W2. Some sections are a bit confusing or not clearly explained (see Questions for details)

### Questions
**Questions**: 

1. If I understand correctly, according to _Section 3.1 Summaries as Predictive Sufficient Statistics_, an ideal summary should capture the relative frequency and diversity of the answers. This aligns well with the example in Figure 1 for which the answer distribution concerns confident-sounding answers. However, I’m curious to understand how the definition of an _ideal summary_ would account for cases  where some of the answers have uncertainty in themselves, e.g., “It’s likely to be Paris” or “It’s possible that the answer is Toulouse”?
2. The paper mentions the computational requirements of the SelfReflect metric in lines 203-208 and convergence of the SelfReflect score under various computational budgets in Appendix B (Figures 6-10). This convergence analysis is based on the SelfReflect score itself. However, I was interested in knowing how exactly Self-Reflect compares to other baselines such as LLM-as-a-judge in some controlled experiment similar to study 1 in Section 4.1. Can the authors comment on whether there’s a minimum number of samples to ensure the faithfulness of SelfReflect (i.e., if we have less than N sampling budget, we’re better off using other baseline metric?)
3. Concerning the methodology, I’d like to clarify how is the b sentence selected? For example, in Figure 3, is b selected as the most frequent generation, or does SelfReflect eventually iterate over different sampled sequences $a^{(i)}$?
4. Line 293 mentions “fine-grained quality differences” when discussing the superior behavior of the SelfReflect metric on the Good vs Almost Good setting. I could not find the details of how almost good summaries are generated. How do you ensure that these summaries differ by only fine-grained differences and what kind of differences are these? Was there any human validation performed to ensure this is the case?
5. Line 298 states “matched only twice by its own SR-P(True) ablation”. Should it be “three times” instead of “twice”, since performance values between the SelfReflect and SR-P(True) ablation are also overlapping in the “Verbalized vs or-concatenated” setting?
6. Could the authors motivate the selection of different models across controlled experiments in Section 4?
7. Section 5 includes experiments with numerous prompting and fine-tuning strategies. I was wondering if an iterative self-critiquing approach would improve upon simple prompting or “sample and summarize”?
8. In Section 5.2 the authors suggest that despite fine-tuning and providing examples of descriptive summaries to the models, they seem to be unable to perform the task and that “model memorizes individual summaries rather than learning a general mechanism”. Could this be due to data quality issues (since the data is generated synthetically) and/or hyperparameter tuning?




**Writing style**:

- Line 40: More of a subjective preference but expressions “We believe we can do better than this.” seem to be unnecessary and a bit informal for a conference style paper. I’d suggest replacing it with something more formal.
- Line 351: Please provide a citation for the Krippendorf’s alpha. This can be for a particular implementation or a relevant paper.


** Clarity**:
- The experimental design for study 2 is unclear for me. Here are a few terms and expressions that were ambiguous: 
     - line 314: “true ratio of answers”: what is the true ratio of answers computed over? Is it over the whole test set or based on the normalized logits for each question?
     - line 319: “compute the true Wasserstein distance between the distribution described in the summary and that of the test-set answers” → Could you clarify what the test-set answers distribution is in the context of a MMLU and how that is computed in a per question vs whole dataset basis? Why is the reference-metric a reference metric? What’s the intuition behind it?

**Typo**:
- Line 374: “decoding” → “decoding pass”
- Line 448: “as example case” → “As an example”
- Line 378: “Developing” → “developing”
- Table 2 caption: “Mean $\pm 95\%$.” → “Mean $\pm 95\%$ CI.”

### Soundness
3

### Presentation
3

### Contribution
4
