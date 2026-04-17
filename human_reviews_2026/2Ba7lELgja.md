# Measuring AI "Slop" In Text

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
AI "slop" is an increasingly popular term used to describe low-quality AI-generated text, but there is currently no agreed upon definition of this term nor a means to measure its occurrence. In this work, we develop a taxonomy of "slop" through interviews with experts in NLP, writing, and philosophy, and propose a set of interpretable dimensions for its assessment in text. Through span-level annotation, we find that binary "slop" judgments are (somewhat) subjective, but such determinations nonetheless correlate with latent dimensions such as coherence and relevance. Our framework can be used to evaluate AI-generated text in both detection and binary preference tasks, potentially offering new insights into the linguistic and stylistic factors that contribute to quality judgments.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
In this paper, the authors aim to define and measure what they call AI slop for AI-generated text. They first design a taxonomy of slop from expert editors and operationalize it into several annotation dimensions covering information utility, information quality, and style-related issues. They then apply this taxonomy to news and QA dataset, and collect span-level and document-level annotations from 3 annotators to study how often these slop phenomena occur and how consistently humans can identify them. Finally, they explore whether slop can be detected or scored automatically, and report that current automatic methods only barely reproduce human judgments.

### Strengths
1. The paper focuses on an important and increasingly visible issue. Social media and the internet are increasingly flooded with AI-generated text, making it crucial to develop accurate methods for measuring and detecting low-quality AI content.

2. Even though it’s subjective, the three dimensions (Information Utility, Information Quality, Style Quality) give a structured way to talk about  why a piece of text feels like “AI slop” rather than relying on vague and binary judgments.

3. The paper is generally well written and easy to follow.

### Weaknesses
1. The core empirical claim (that the proposed taxonomy explains human judgments of “slop”) relies on annotations from only three final annotators. With such a small pool, rater-specific preferences and biases cannot be averaged out, and the already low cohen's kappa values (down to −0.15) become very difficult to interpret. The paper cites Marchal et al. to argue that “multi-label is hard,” but that is precisely the scenario where more annotators are needed to stabilize labels and reduce idiosyncratic noise. In fact, with such a limited rater pool, even adding a single additional annotator (e.g., moving from 3 to 4) could significantly shift agreement scores, especially considering the annotation task is very subjective.

2. If I understand correctly, the slop label is produced by the same annotators who provide the span level slop codes. As a result, the regression simply uses annotators’ own fine-grained judgments to predict their own overall slop decisions. This setup is somewhat confusing and circular, because it tests whether annotators agree with themselves rather than validating the taxonomy against an independent signal. Of course the p-values for predictors  should be singifiantly as it just check the internal consistency between annotators' span-level subjective codes and their binary slop decisions. Moreover, even under this weird design, the agreement scores remain low, which further raises concerns about the stability and reliability of the proposed construct.

3. While conceptually interesting, the proposed taxonomy does not presently yield a practical or scalable method for detecting AI slop. All attempted automatic approaches, including traditional machine learning models, logistic regression, and LLMs-as-judges, either barely outperform the base rate or fail to meaningfully agree with human judgments at all (LLM cohen's kappa essentially zero).  For a venue like ICLR, this shows a substantial gap between the stated goal of a measurable framework and the actionable evidence of model signal.

### Questions
see above weakness

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
The paper defines and measures “AI slop”, generic, low-quality AI text, by proposing a taxonomy spanning information utility (density, relevance), information quality (factuality, bias), and style quality (structure, coherence, fluency/verbosity/tone). Using expert-guided, span-level annotations on news and QA corpora, the authors show that all seven factors predict human “slop” judgments, but their importance shifts by domain (e.g., relevance/density in news; factuality/structure in QA). Inter-annotator agreement is modest overall yet strongest for factuality, bias, and structure. Automated proxies (repetition, compression, readability) and a reward model provide only limited signal, and current LLM-as-judge approaches perform poorly at both binary slop detection and span extraction.

### Strengths
- The paper proposes a three-axis taxonomy (Information Utility, Information Quality, Style Quality) and argues that increases in these issues raise the odds a text is judged “slop”.  It establishes a foundational framework for future work by standardizing terminology and guiding annotation and metric design.
- Logistic regressions show code salience shifts by domain: utility/style and bias matter more in news; factuality/structure dominate in QA.
- Linear models over standard metrics achieve only modest AUPRC; several salient codes lack reliable automatic measures. LLM-as-judge underperforms.

### Weaknesses
- Linear models over available metrics achieve only modest AUPRC (0.52–0.55), underscoring that current proxies don’t capture “slop” robustly.
- Annotations are confined to two English domains (news and MS MARCO QA), leaving external validity for other genres (e.g., creative, technical) and languages unclear.
- Final labeling relies on three professional copy-editors whose preferences differ; authors note divergences attributable to editing style.

### Questions
- How exactly should annotators (or practitioners) decide the overall yes/no “slop” label before doing spans. 
- For practitioners, do you recommend a decision rule that combines the seven codes into a yes/no label (e.g., weighted sum, top-k issues), or is the binary label strictly a human judgment for now?

### Soundness
3

### Presentation
3

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
This paper focuses on establishing a definition and measurement for "AI slop", which is a term for low-quality AI-generated text. The authors first develop a taxonomy of "slop" by interviewing 19 experts from fields including NLP, writing, and philosophy. This taxonomy organizes "slop" into roughly three main themes: Information Utility, Information Quality and Style Quality. The authors then validate this framework by collecting span-level "slop" annotations from expert writers on two datasets: news articles and question-answering passages.

The paper finds that binary "slop" judgments are somewhat subjective and these judgments strongly correlate with the latent dimensions in the taxonomy. The strongest predictors of a text being labeled "slop" are issues with Relevance, Density, and Tone. The authors also find that these predictors vary by domain: news articles are penalized for style and utility issues, while QA passages are penalized more for Factuality and Structure. Finally, the authors find that automatic detection of "slop" is difficult and LLMs perform poorly at both binary "slop" classification and span extraction.

### Strengths
- The paper is well-written and focuses on a timely problem, the lack of a formal definition and measurement for low-quality AI-generated content.
- Directly interviewing experts and codifying their responses to develop the "slop" taxonomy is an appropriate choice here, as they are the most reliable informants of what slop actually is. 
- The finding that "slop" characteristics are domain-dependent is interesting, and adds nuance to the measurement of slop. The paper shows that predictors for "slop" in news articles differ from those in QA.
- The finding that current LLMs are unreliable judges of "slop" has important implications as it shows a limitation of using LLMs for subjective text evaluation.

### Weaknesses
- The paper argues that measuring “slop” can “help characterize and improve LLM writing” but doesn't actually show empirical evidence for this. It could have been interesting to show proof that slop measurements can help us train models that are less "sloppy".

- There is very limited discussion of how the measurement of slop interacts with prior work in NLP that measures properties of text such as diversity, novelty, complexity etc. For instance, the authors could have used existing automatic metrics for these properties (like the ones listed in Table 6) to measure slop.

- There is limited evaluation of the validity of the proposed taxonomy. The codification done based on interviews seems comprehensive but is not further validated and can hence seem a bit ad-hoc at times. Is the taxonomy sufficient? If not, how can it be improved?

- “We select 3 annotators from our pilot study based on annotation quality and availability for the remainder of the datasets.” - This statement is a bit vague about the background and expertise of these 3 annotators. Since the majority of the annotation was done by these 3 annotators, I think providing more context about them would be helpful. Finally, just relying on 3 annotators weakens the conclusions of the paper significantly.

- I think the main paper should describe precisely how span-level agreement between annotators is measured since it is an important part of the main methodology. Span boundaries are bound to have low agreement between annotators, so how does the paper deal with this? Please also describe it in your author response.

- The proposed taxonomy relies heavily on codes like Relevance, Coherence, and Factuality, which the authors mention lack reliable automatic measures. The fine-tuned Qwen-7B model for span extraction also shows weak performance. This makes the framework difficult to use for large-scale automatic evaluation.

- The paper shows that binary "slop" judgments are subjective, which is supported by the low inter-annotator agreement on the binary label. While this is a finding, it also makes it difficult to create a scalable measure for a "slop" measurement.

### Questions
- The agreement for the binary "slop" label was low but the span-level location agreement was moderate to high. This suggest that annotators agree where problems exist but disagree on whether those problems are sufficient to label the whole document "slop". Could you discuss this? Does this suggest "slop" is better measured as a count of span-level issues rather than a binary document label?
- The annotation guide says that "Not all AI-generated text is 'sloppy' and human writing can be sloppy too". Did the annotators label the human-written source texts from the news dataset? If so, were human-written "slop" spans identified, and did their characteristics differ from the AI-generated "slop" spans?
- Why do you think LLMs like GPT-5 perform poorly on this task? These models are optimized for human preferences but they seem unable to detect these "slop" characteristics. Is it a blind spot in current RLHF practices or a more fundamental limitation of LLM-as-judges for subjective style evaluation?

### Soundness
3

### Presentation
4

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
The paper develops a taxonomy of types of AI "slops" in text from 19 domain experts, conducts span-level annotations on 150 LLM-generated news articles and 100 MS MARCO passages with professionals. The paper also analyzes agreement and identifies what features of the taxonomy are the strong predictors of the AI slops. Lastly, the paper conducts testing of whether machine learning models, LLM-as-judges and rewarding models can identify the AI slops.

### Strengths
- Highly timely research problem in LLM generation field. The paper's scope that covers both quantiative and qualitative analyses (e.g. taxonomy development, span-level annotations by domain experts, and analytics) provide several insights of the linguistic and stylistic factors in AI-generated textual contents. 

- Glad to see honest reports on negative results in Section 5. The paper carefully shows that standard metrics, LLM judges, and even finetuned models fall short, thus avoiding overclaiming of their findings.

### Weaknesses
- The taxonomy development of AI slops is primarily based on two English domains and information-focused texts (News articles and web search queries); the axes of the taxonomy may not be applied to other languages (e.g., Spanish, French, or Chinese) or genres (e.g., creative writing). Also, the demographics of experts who participated in the survey and taxonomy development processes are not clearly mentioned (ie, is English your first language?). Please report the experts' language backgrounds (e.g., whether English is their first language) so reviewers can judge how general the taxonomy axes are.

- Concern about annotation reliability. Multi-label span annotation is a highly complicated task, and several axes show low agreement (some < 0.3 in Appendix Table 12), indicating notable disagreement or sparse overlap among annotators.

### Questions
- For LLM judges, did the authors analyze how o4-mini's rationale of span-level annotations line up with the expert-labeled spans and codes? If you have some relevant data, a small qualitative study would help: (1) overlap between LLM rationale spans and human-annotated spans (using token F1) or (2) axe alignment (which "slop" categories the LLM cites most vs humans), etc. 

- In Section 5, the authors mentioned that the domain of datasets also affects the evaluation of LLM-generated texts. However, within the News set, I think the magazine/source type and article topic (e.g. original titles) can also shape the slop style in the LLM-generated texts. Have you checked (1) the per-outlet(source) prevalence of each "slop" axis, (2) topic-level effects (science vs politics vs lifestyle) etc?

### Soundness
3

### Presentation
3

### Contribution
3
