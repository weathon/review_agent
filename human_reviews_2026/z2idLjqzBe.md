# Death of the Novel(ty): Beyond N-Gram Novelty as a Metric for Textual Creativity

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 8

## Abstract
$N$-gram novelty is widely used to evaluate language models' ability to generate text outside of their training data. More recently, it has also been adopted as a metric for measuring textual creativity. However, theoretical work on creativity suggests that this approach may be inadequate, as it does not account for creativity's dual nature: novelty (how original the text is) and appropriateness (how sensical and pragmatic it is). We investigate the relationship between this notion of creativity and $n$-gram novelty through 8,618 expert writer annotations of novelty, pragmaticality, and sensicality via \emph{close reading} of human- and AI-generated text. We find that while $n$-gram novelty is positively associated with expert writer-judged creativity, approximately 91% of top-quartile $n$-gram novel expressions are not judged as creative, cautioning against relying on $n$-gram novelty alone. Furthermore, unlike in human-written text, higher $n$-gram novelty in open-source LLMs correlates with lower pragmaticality. In an exploratory study with frontier closed-source models, we additionally confirm that they are less likely to produce creative expressions than humans. Using our dataset, we test whether zero-shot, few-shot, and finetuned models are able to identify expressions perceived as novel by experts (a positive aspect of writing) or non-pragmatic (a negative aspect). Overall, frontier LLMs exhibit performance much higher than random but leave room for improvement, especially struggling to identify non-pragmatic expressions. We further find that LLM-as-a-Judge novelty ratings align with expert writer preferences in an out-of-distribution dataset, more so than an n-gram based metric.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper challenges the use of n-gram novelty as a sufficient metric for evaluating textual creativity in language models. The authors argue that true creativity requires both novelty (how original the text is) and appropriateness (how sensical and pragmatic it is), based on established creativity theory. Through a study with 26 expert writers annotating 7542 expressions from human and AI-generated text, they find that 91% of top-quartile expressions by n-gram novelty are not judged as creative, and that higher n-gram novelty in open-source LLMs in facts correlates with lower pragmaticality. Furthermore, they found that LLMs are less likely to produce creative expressions than humans and even struggle to identify creative expressions (a positive aspect of writing) and non-pragmatic ones (a negative aspect).

### Strengths
1. The paper is well motivated, well written, and addresses an important research question on measuring creativity.
2. The authors conduct high-quality expert annotations, recruited 26 professional writers from top MFA programs with rigorous training, achieving good inter-rater agreement (κ_free = 0.78 for novelty, 0.72 for pragmaticality)
3. The finding is somewhat surprising: 91% of top-quartile expressions by n-gram novelty are not judged as creative by experts, highlighting the risk of over-relying on n-gram based metrics as a proxy for creativity.

### Weaknesses
1. The scale of the study is relatively small. For example, the human text samples include only 50 passages of roughly 400 words each.

2. The domain is limited to creative fiction from The New Yorker, findings may or may not generalizes to other domains such as technical writing, journalism, or poetry, where measuring creativity is also very important. 

3. The authors rely on Infinigram and open-source OLMo/OLMo-2 corpora to compute n-gram novelty. However, the training data of frontier models (e.g., GPT-5, Claude 4.1) are not publicly available, raising the question of whether the observed limitations of n-gram novelty metric comes from the n-gram metric itself or from an imprecise estimation of n-gram novelty due to incomplete corpus coverage.

### Questions
What do you think it's a scalable method to measure creativity beyond n-gram metrics and expert annotation?

### Soundness
3

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
3

### Summary
This paper declared death of n-gram as a measure of creativity, with detailed study into rare n-gram in both human & AI generated passages and painstakingly annotated dataset. Findings are cool: rare n-grams are overwhelmingly not creative. The author continue to investigate if AI can replace human in such annotations.

### Strengths
- This is a well-motivated and complete investigation into whether n-gram can be a good metric.
- The author conducted rigorous annotations on n-gram quality and creativity, with the annotation guideline elaborate and well defined. Annotations of the samples are really well thought and detailed. Statistical analysis are thorough.
- The author investigated various interesting research quesitons that flows well with the ideas, including if LLM can replace human in close reading and if experts are needed to create the best annotations.

### Weaknesses
- The scope of the work is clear but limited to New Yorker passages and n=26.
- This result is intuitive but expected, once one understand the context. While the work is solid, its insights are not very novel by iclr standards.
- The paper's writing can benefit from placing some details (e.g. python packages used) in appendix and devote more space to more analysis.
- LLM as a judge's F1 performance is overly low. Some more insights would be helpful. The current statement "it suggests it may be stemming from failure to recognize non-pragmatic expressions" is unconvincing. 
- Similarly, finetuned pragmatic or novelty identification models perform poorly, with error bars swing to as bad as x = 0. A further invesitgation on why, or finetuning larger model might be needed.

### Questions
Please see weakness.

### Soundness
3

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
3

### Summary
This paper re-examines n-gram novelty metrics which have been previously used as proxies for creativity. Specifically, they collect a pool of annotators to measure the creativity of llm-generated text and human gold text via close reading, finding that while n-gram novelty correlates with creativity, it also corresponds to lower pragmaticality compare to human texts. Their findings suggest defining better metrics for creativity that incorporate multiple factors.

### Strengths
* The paper is well-written and easy to follow
* A large amount of effort went into collecting a diverse set of annotations across many annotators and passages. The definition and task for annotators are clearly defined. In general, Section 2 is strong. The experiments are designed well and a lot of thought was put into them.
* The findings about n-gram novelty relation to creativity (human judgement) are interesting and novel. There is a lot of analysis done that goes into specific details around this behavior which are interesting, like the decreasing pragmatism. 
* A variety of models are tested

### Weaknesses
* Minor: Formatting on line 2 is odd " (Handa et al., 2025) (Anthropic) and (Chatterji et al., 2025) (OpenAI)"
* The main study comparing human and text generation is only based only on one source - The New Yorker. While this source is high-quality, it would be nice to compare LLM/Human texts in other domains to see how well n-gram novelty findings hold.
* Minor: The findings could be stronger by including more models to analyze (in addition to the task mentioned above). 
* A conceptual limitation is that the paper assumes creative expressions must be both sensical and pragmatically appropriate. While this is reasonable for narrative prose, some creative language intentionally violates pragmatic norms (e.g., surreal or poetic writing). Thus, the finding that high n-gram novelty often corresponds to non-pragmatic text may not be a fundamental flaw of the metric on its own.

### Questions
* Given the findings, what is an alternative metric for creativity?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper tests whether n-gram novelty is predictive of creativity. The taxonamize creativity into sensibility and pragmatically and build an expert-annotated dataset to study this (~8000 samples are annotated). They conclude n-gram novelty is not a reliable predictor of creativity and worse of higher n-gram novelty is also correlated with loss of sensicality.

### Strengths
- The study between the n-gram novelty and creativity judgements from experts engenders a useful conclusion.  Sensibility vs pragmatically distinction is also very interesting; 
- The resulting corpus is substantial, providing scope for running large scale analysis for future work.

### Weaknesses
- Deviation from layman perception of high quality creativity versus an expert perception of high quality creativity: Since the definitions of creativity are expert curated the claims in the paper seem very plausible; However, it is unclear if the authors expect these conclusions to hold with less expert audiences who perceive creative texts through less varying rubrics (some of which may include allegedly spurious factors like novel n-grams)

- Conditional dependence of sensicality/pragmaticality on the length of the snippet: It is unclear if the decision of the in-context sensicality or pragmaticality can vary if additional context is provided to the annotators. I understand that this is non-trivial to ablate due to the substantial variation in the induced cognitive burden to annotators but an ablation for this even a small control experiment that maintains the stability of the judgement with varying lengths over the snippet will help confirm the legitimacy,  of especially the sensible judgements.

### Questions
- In Ln 217, you mention that ‘Instances that were labeled as not sensical were removed to focus on cases where expressions do not make sense in the context’; Will this not alter the logical flow of the entire text ? 
- You mention in Appendix D, that the pragmaticality scores consistently know a significant negative effect with the crowd preferences - Couldn’t this be indicative of the pragmaticality definition being ill-suited/divergent from what a naive-user/lay crowd prefers ?

### Soundness
3

### Presentation
4

### Contribution
3
