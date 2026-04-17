# FiRE: Fine-Grained Ranking Evaluation for Machine Translation

- Decision: Reject
- Scores: 8, 2, 4, 4

## Abstract
Developing reliable machine translation (MT) systems hinges on our ability to distinguish superior translations from inferior ones—but existing evaluation paradigms, whether limited to coarse overall rankings or misaligned with human preferences, fail to deliver interpretable, fine‑grained feedback in reference‑free settings. We present a Fine-Grained Ranking Evaluation method (FiRE) that leverages off‑the‑shelf large language models to perform criterion‑driven pairwise comparison across three complementary dimensions—faithfulness, fluency, and consistency of style—rather than producing a single holistic judgment. To enable rigorous meta‑evaluation of evaluation paradigms in the absence of any suitable testbed, we construct the first human‑annotated, reference‑free benchmark for fine-grained ranking evaluation, achieving substantial inter‑annotator agreement. Through meta‑evaluation on this benchmark, FiRE demonstrably outperforms leading regression‑based and error‑analysis metrics in aligning with human comparative judgments, while providing more informative insights into translation quality. Finally, our examination of LLM evaluator biases (position and self-enhancement) and their handling of tied cases offers guidance for more nuanced MT evaluation.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper proposes fine-grained pair-wise ranking evaluation of machine translation using off-the-shelf LLMs. The evaluation is done on three complimentary dimensions: faithfulness, fluency, and consistency of style. The paper also provides the first human-annotated fine-grained machine translation evaluation benchmark.

### Strengths
1. Fine-grained human-annotated evaluation data was collected for meta-evaluation. The human annotations have high inter-rater agreement.
2. The paper provides comparison against strong baselines.
3. The paper studies the issue of position bias, showing strong position bias of LLMs and the need for position bias mitigation.
4. The paper also examines the LLM's preference for their own generations.
5. Performance on easy vs hard examples are shown, which validates that when human annotators find an example difficult the LLMs also struggle.

### Weaknesses
1. The paper only focuses on high resource languages and the author's collected data. It is understandable that collecting data for low resource languages is difficult. However as mentioned on table 10, existing MQM annotations could probably be directly mapped to the three evaluation criteria under consideration. An analysis on these MQM datasets would provide an independent verification of the proposed approach on independent datasets.
2. For position bias experiments, results have not been reported for each of the three evaluation criteria. Whether LLMs show more or less position bias on these different criteria would be an interesting research question.

### Questions
Was there any cases where the LLMs did not follow the prompt instruction and generated nonsensical answers? If so, how was it handled?

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
3

### Summary
The title promises a fine grained ranking evaluation. While reading the paper it becomes clear that this only means to split the feedback into ‘faithfulness’, ‘fluency’, and ‘consistency of style’. The authors defined style and fluency as a different class, but I would even argue that both are part of the same broader topic which I would have called fluency (maybe naturalness?). This is leaving only two data points per compared translation pair instead of one.
This is much less fine grained than MQM which gives way more detailed information about each sentence and isn’t limited to simply ranking. MQM scores can easily be converted into a ranking, this leaves only the price as an advantage of this method.

### Strengths
Releases a ranking benchmark which distinguishes between faithfulness, fluency, and consistency of style.

They analyzed the bias of the LLM evaluators.

The data will be released.

### Weaknesses
While having three scores is more fine grained than one, it still provides much less information compared to the MQM granularity.

The authors used language pairs where no existing WMT results are public which would have been easy to compare to.

### Questions
Why did you use the comparably small NLLB-200-1.3B model when the results were so weak that you had to downsample it? Using the 3.3B model should not have been any issue from the hardware perspective. If you could run the Qwen2-72B model using the NLLB-MoE-52B should also have been possible.

Why didn’t you use a language pair supported by WMT? That way you could have used submitted results and compared directly to MQM results or other metrics evaluated at WMT. You can find the ende, enes, and jazh data here: https://github.com/google/wmt-mqm-human-evaluation/tree/main/generalMT2024
It’s obviously too late now to change it since you already collected the human rating, but it seems like an odd choice to build a dataset which can not be directly compared to existing data.

Why did you separate ‘consistency of style’ from ‘fluency’, but no subcategories for the ‘faithfulness’ part? I would say that changing the style also breaks the ‘fluency’ (in a broader sense not as defined in the paper). MQM provides a lot more categories, also for ‘faithfulness’ (by MQM called Accuracy).

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
the paper presents a new LLM-based evaluation method, FIRE, by ranking two translations of the same source
three ranking criteria are distinguished: adequacy, fluency and style, as well as overall quality
the rankings are performed by language models on English-Chinese and Russian-Chinese translations and then compared with rankings of human evaluators
the FIRE results are more similar to human judgements than KIWI, xCOMET, metricX and MT-ranker

### Strengths
The evaluation on three different criteria provides more insights than only an overall scores

The method is clearly explained

### Weaknesses
the data samples apparently consist only of isolated sentences, so that the context is not taken into account
several recent studies have shown that the evaluation should be done on a paragraph level, not on isolated sentences

some parts are not fullly clear (see questions)

### Questions
034  why is BLEU (and other similar metrics) called a regression-based metric? What is the regression there? 

Maybe the idea is that they are reference-based?  
so "similarity-based" or "overlap-based" would be a good description

based on a single score


053: one source sentence: meaning that the evaluation is on isolated sentences, without context


180: which systems are NMT (encoder-decoder) and which are LLMs (decoder only)? 

258: why report only DeepSeek-R1 results and not of other used models? 

268: aligns with the majority vote of human annotations: 
Why not compare each data point with each data point? 
Overall result (majority vote) might be similar even though the actual annotations are quite different

395: a figure discussed in the main text should not be in appendix

the organisation of a paper should be in a way that the reader does not need to look into Appendix at all

421: position bias has already been discussed in 4.1

### Soundness
2

### Presentation
2

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
This paper proposes FiRE, a machine translation evaluation framework which focuses on reference free pairwise ranking across broad categories (faithfulness, fluency and style consistency). A benchmark is created and various evaluation methods are analyzed against that benchmark.

### Strengths
* The proposed framework is clear and conceptually intuitive and simple
* The benchmark can be a good contribution for subsequent evaluations of MT systems
* Experimentation and ablations appear reasonably thorough

### Weaknesses
I have several doubts/questions about the methodology. I'm listing here the main questions but see also the questions below.
* on the benchmark creation, is there a mechanism for quality control? I didn't find enough details on the human annotators and any guardrails for quality.
* I believe Section 4.5 may be one of the main claims of the paper, i.e. that the proposed framework is better than prior ones. However, that section is very small and lacks details for each row presented in Table 3.

Given the amount of uncertainties and questions, I'm currently leaning towards a weak reject and would encourage the authors to provide more details.

### Questions
204-205: “our 3-annotator 3-class setting (which typically yields lower κ values) shows comparably substantial inter-annotator reliability (κ = 0.57 − 0.81).”: do you have an intuition or explantation about this?

Table 1: why are there different number of pairs across categories and why not only retain pairs with all 4 categories?

244-246: “To derive the overall pairwise judgment, we further aggregate all errors across the three criteria into a single composite error-based score.”: what is the aggregation method?

201: “three annotators evaluate the pairwise comparisons”: I believe it is important to provide more details on the annotators, the guidelines given to them and quality control process for the data collection. Otherwise, it puts into question the validity of the benchmark created.

Was position consistency also checked for human evaluators? If not, it could be valuable information.

------------- Typos/Presentation

Figure 1: using gloss for the Chinese text would be good to better understand the explanation.

116:  “semi-automate this process using PLMs”: introduce acronym

184-185: “ and three closed-source systems (GPT4o1 , DeepL, LanMT)”: indicate the dates for all closed source systems, not only gpt-4o

306: “The aggregation procedure is described in Appendix D.”: In my opinion the procedure should be described in the main body of the paper, not the appendix as this is an important detail.

### Soundness
2

### Presentation
2

### Contribution
3
