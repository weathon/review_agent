# Camellia: Benchmarking Cultural Biases in LLMs for Asian Languages

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
As Large Language Models (LLMs) gain stronger multilingual capabilities, their ability to handle culturally diverse entities becomes crucial. Prior work has shown that LLMs often favor Western-associated entities in Arabic, raising concerns about cultural fairness. Due to the lack of multilingual benchmarks, it remains unclear if such biases also manifest in different non-Western languages. In this paper, we introduce Camellia, a benchmark for measuring entity-centric cultural biases in nine Asian languages spanning six distinct Asian cultures. Camellia includes 19,530 entities manually annotated for association with the specific Asian or Western culture, as well as 2,173 naturally occurring masked contexts for entities derived from social media posts. Using Camellia, we evaluate cultural biases in four recent multilingual LLM families across various tasks such as cultural context adaptation, sentiment association, and entity extractive QA. Our analyses show a struggle by LLMs at cultural adaptation in all Asian languages, with performance differing across models developed in regions with varying access to culturally-relevant data. We further observe that different LLM families hold their distinct biases, differing in how they associate cultures with particular sentiments. Lastly, we find that LLMs struggle with context understanding in Asian languages, creating performance gaps between cultures in entity extraction.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces the Camellia benchmark for evaluating cultural bias surrounding entities from 9 Asian languages against a collection of ‘Western’ entities. Based on the concepts and approaches from [Naous et al. 2024], the benchmark consists of manually annotated entities from different categories that each come with masked context derived from Tweets. The benchmark consists of three tasks: Context adaption, sentiment association, and entity extraction. The paper then uses the benchmark to evaluate four different models showing biased model behavior across all three tasks

### Strengths
The paper looks at the interesting problem of quantifying entity bias of LLMs. Here, the paper identifies a set of understudied Asian cultures for which a benchmark is built, thus enabling the evaluation of entity bias w.r.t. these cultures. 

The authors provide, at least partial, human annotation and validation, thus creating a more trustworthy resource than the often fully automatically generated benchmarks from related work.

The writing is clear and engaging.

### Weaknesses
**W1)** Methodologically, the paper is mostly taking the work in [Nous et al, 2024] and applying it to a new set of languages. Dataset creation and annotation is of course valuable work, especially when involving human annotation. However, this limits the methodological novelty of the work.

**(W2)** While defining and naming cultural groups is a challenging task, the authors often lack arguments for their choices and do not discuss them in conjunction with related work (where they provide nearly no reference for any of their choices). This results in several issues:

- The paper’s motivation revolves around the discrimination of an overgeneralized ‘Western culture’ group (line 013, based on the findings in  [Naous et al. 2024]). The authors' findings are more nuanced, however, showing better and worse performance for either region, depending on the setting. It is not clear, why this Asian-Western definition was chosen over, e.g., ‘high- vs. low-resource’ or ‘dataset frequency’.

- Similarly, why is the baseline group chosen as ‘Western’ and not, e.g., ‘U.S. American’ to compare at similar levels of granularity (i.e. country level)? It is likely that this experimental choice introduces bias that is not accounted for or discussed (e.g. low-resource languages in both Europe and Asia being underrepresented).
 
- The authors losely define "Western" as “North America and Europe” (line 103), missing possible other "Western"-dominated cultures like Australia and including Mexico, which is often more associated with Latin America, but on the North American continent. While arguments could be made for these choices (e.g. Mexico's cultural closeness to the Southern part of the US), the authors do not provide this argumentation.

- The notion of a benchmark for ‘Asian languages’ also implies a more diverse a selection of languages than the ones considered which are limited to 9 Asian languages from East and South Asia. 

**(W3)** Given the complexity of the topic, the authors need to be careful how they phrase their statements. E.g. a bolded statement like "LLMs still lack a robust ability to grasp implicit contexts in most non-English languages" (line 373) implies that a broad set of non-English languages have been analyzed. However, the authors only study their subset of Asian languages in this experiment, missing most of the rest of the world (incl. Europe, South America, Africa and parts of Asia) and thus making the "non-English languages" statemtent problematic. Similarly, sentences like ”Our analyses show a struggle by LLMs at cultural adaptation in all Asian languages,” (line 022) can be prone to overinterpretatation of the results provided (assuming a broad coverage of Asian languages, which is not the case. I urge the authors to be careful in their phrasing, using e.g. "all studied Asian languages" instead and avoiding overgeneralizations like "non-English languages".

**(W3)** The paragraph starting from line 203 (“Contexts for Evaluating Cultural Adaption.”) is lacking a precise definition of what constitutes a ‘cultural context’ and what criteria were used to annotate them. This creates a challenge for interpreting the results.

**(W4)** Human verification is lacking for some dataset creation tasks and some tasks would need human quality control assessments.
- The authors use double annotation for one aspect (cultural origin of an entity), the authors do not perform this (or at least do not report it) for the other manual annotation steps. This is especially relevant for the sentences that are only appropriate in one cultural context vs. all contexts (Camelia-Grounded/Neutral) and for the sentiment task (line 209). Inter-annotator agreement would be important here to understand the quality and difficulty of these annotations, as these are at the core of the following experiments. 
- Less crucial, but double annotation/inter-annotator agreement would also be interesting for other manual tasks like selection of culturally relevant entities, translations/transliterations into English or translations of Western languages into low-resource languages. 
- The authors do not evaluate some of their automated steps (e.g. the automatic parallelization of Entities in Indian Languages via English).

**(W5)** Figure 2 shows that the entities from each ‘culture’ are not distributed equally. In 5/6 categories the ‘Western’ group has more entitites than the majority of the other ‘cultures’.

**(W6)** Section 2 reports a straightforward approach to creating the dataset for multiple languages. Section 4 then presents exceptions to this process (e.g. using governmental reports and name generators for Chinese, Korean and Japanese). While I very much appreciate that the authors consider these language nuances, this should have been part of the dataset creation description.

**Minor aspects**
- It would have been nice to include a short discussion in the related work about cultural representation in general and entities just being one dimension (cf. also https://aclanthology.org/2025.tacl-1.31/ ).
- Cultural evaluation is a rapidly growing area and the authors list a lot of relevant works. Fitting to their entity-focused approach, the following recent works could also be relevant: https://aclanthology.org/2024.acl-long.345/  https://arxiv.org/abs/2505.21693 https://aclanthology.org/2025.naacl-long.402/
- The authors report averages over several runs in Figure 5 but miss any form of variance. 
- In Eq. 1, I would write the function that is defined before the equation, i.e. CBS_D … = Equation 1
- Would it make sense to report baseline values for the CBS from other studies instead of just stating 'the CBS is expected to be low’ in line 267? Without any substantiation this claim does not seem intuitive.

### Questions
1. Would it make sense to study and refer to secondary literature for the definition of the cultural categories of the paper? What would that change in your setting? (see W2)

2. How is cultural context defined? (see W3)

3. What are the results of a human verification and quality control assessment? (see W4)

4. Is the inbalance of the entities an issue? (see W5)

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
2

### Summary
This paper benchmarks cultural biases in LLMs with a focus on Asian languages and cultures, by evaluating models (four open multilingual model families) via context preference, sentiment association and extractive QA tasks.
It largely follows CAMel (Naous et al. 2024) that did similar work for Arabic and Arab culture.
Data is extracted from Wikidata, web crawls, and from X.  
They find that most LLMs struggle with distinguishing between Asian and Western entities in Asian culture contexts (i.e. scoring Western entities highly). Trends of sentiment across models and cultures appear to diverge, there is high variance based on the individual models and languages/cultures. For context extraction, there is still headroom compared to English performance. The paper also provides a discussion on entity-based research in the multilingual space.

Overall a nice and valuable contribution on an important topic (closing gaps in multilingual LLM evaluation), however, in my opinion not novel enough in terms of the methodology for ICLR, and fairly shallow in relating findings to prior work. This might be better placed at an *ACL venue with an explicit focus on language expansion.

### Strengths
- Important topic: while most LLMs are multilingual, their evaluations are missing for a lot of languages and cultures. Expanding the language coverage of evaluations is helpful to understand where today's models are not sufficiently multilingual yet.
- Necessary care for detail on the data curation side.
- Well structured and written, with a particularly nice overview figure on the first page.

### Weaknesses
- Context of other works could be made more clear: Which findings align with related works on similar cultural benchmarking? These papers are listed in Related Work, but not really taken into account in any discussion. 
- Lack of novelty in methodology: the paper largely folllows CAMel, and does not clearly indicate where an if it diverges or innovates. The paper refers to it in various places but it is not clear which aspects of the methodology or insights are new (beyond the obvious language/region expansion).
- Some open questions in experiment design, evaluation and data validation, see below.

### Questions
- Is using X as a source truly indicative of representative native speaker discussions? It seems like X is blocked in some countries (e.g. China, Iran, North Korea, Myanmar, and Russia). This includes relevant regions for this study (Asia). The data sourced from X is claimed to contain “natural discussions by native speakers” - how is this ensured/validated? The paper mentions manual inspection of retrieved tweets - how and by whom is this done?
- The data is web-sourced, so why are models that are trained on web sources not better?
- The language support is not considered in the evaluation. Some models explicitly state language support, which should be taken into account in the evaluation. E.g. Aya Expanse 32B does not support Urdu - the bias is low but probably the uncertainty is very high in general.
- How can the differences between the results of the different benchmarking formats be explained?
- How were data filters adapted to the languages of study? (Or made sure that they’re appropriate)
- How long are contexts?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Camellia, a benchmark for measuring entity-centric cultural biases in large language models (LLMs) across six Asian cultures and nine Asian languages, including Chinese, Japanese, Korean, Vietnamese, Urdu, Hindi, Malayalam, Marathi, and Gujarati. The benchmark consists of manually annotated cultural entities and naturally occurring masked contexts from social media. Using Camellia, the authors evaluate four multilingual LLM families (Llama, Qwen, Aya, Gemma) across three tasks: cultural context adaptation, sentiment association, and extractive QA. Results highlight cultural biases.

### Strengths
The paper addresses a timely problem of how LLMs treat culturally grounded entities. Camellia is large-scale, contains manually curated annotations, and covers diverse Asian cultures, filling a gap in multilingual fairness resources. The benchmark design is thoughtful and includes English translations to enable cross-lingual comparisons. Experiments are systematic and reveal insights: 
(1) models often fail at cultural adaptation, 

(2) different model families show opposite sentiment biases, and 

(3) QA performance gaps shrink when switching to English. 

The discussion of data-collection challenges is detailed and transparent. Overall, the benchmark has strong potential for community impact.

### Weaknesses
Some methodological details could be more deeply analyzed. While cultural bias is measured by comparing Asian vs Western entities, Western entities are parallel across languages, potentially oversimplifying cultural grounding. Benchmark construction depends heavily on social-media contexts which raises domain-coverage concerns and may not represent broader text ecosystems. Several analyses rely on probability-based metrics (e.g., CBS) without exploring sensitivity to tokenization granularity. The authors speculate that training-data provenance explains results; however, little direct evidence supports this. The work would benefit from stronger causal attribution and more diagnostic error analysis.

The paper covers a few "asian" languages, but could do a better job of exploring how language relatedness and differences affect results. E.g., is this a typologically diverse language sample, and how do typological differences affect the results.

### Questions
Data Representativeness: How representative is your data in a broader cultural discourse? Did you test whether cultural-adaptation findings transfer to news or conversational corpora?

Tokenization Effects: Since CBS depends on token probabilities, how sensitive are results to tokenization differences across languages?

Typological Diversity:  What is the typological diversity of your language sample, and how do language differences affect your results?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
Camellia is a multilingual benchmark to measure entity-centric cultural biases in LLMs across 9 Asian languages covering 6 Asian cultures. It contains 19,530 manually annotated cultural entities (Asian- vs Western-associated) and 2,173 naturally occurring masked contexts sourced from native-speaker posts on X/Twitter, organized for three evaluations: cultural context adaptation, sentiment association, and extractive QA. The benchmark also provides English translations of entities/contexts.

Evaluation setup: The authors test four recent multilingual LLM families (Llama, Qwen, Aya, Gemma) using prompting (no task-specific fine-tuning) and log-prob–based scoring for masked fills. A central metric, the Cultural Bias Score (CBS), counts the fraction of pairwise cases where a Western entity receives higher likelihood than the culturally appropriate Asian entity in a given context (lower is better)

### Strengths
**s1: Originality** 
Builds on Arabic-only prior work by scaling to 6 Asian cultures / 9 languages with English parallels; introduces an entity-centric design plus a thoughtful task triad (context adaptation, sentiment, extractive QA) and a clear, reproducible CBS metric.

**s2: Quality**
Uses native, naturally sourced contexts; good scale (~19.5k entities, ~2.2k contexts); high inter-annotator agreement; and unified prompting/decoding across strong LLM families for fair, controlled comparisons.

### Weaknesses
**w1: Flow/structure**
The write-up feels a bit scattered; a short “where we fit vs prior work” in the intro and a 3–4 bullet “Key Contributions” box would make the novelty and takeaways pop. (Style preference, but it really helps readability.)

**w2: One scoreboard (nice to have, not required)**
A single at-a-glance table ranking models across tasks/languages would make comparisons easier, but it’s optional.

**w3: Simple control with English-only models**
Since English parallels exist, running strong monolingual English models would help separate language issues from cultural knowledge.

### Questions
**Release package**
Will you release the prompts, and scripts necessary to reproduce the main tables? 

**Overall**
well-constructed dataset and a useful benchmark. Good coverage (6 cultures/9 languages), and a clear evaluation protocol make this a valuable resource. The paper is generally clear, and the analyses are informative. My main suggestion to strengthen the work is to test a broader set of models (including strong monolingual English models on the English parallels) to sharpen conclusions and improve comparability.

### Soundness
3

### Presentation
3

### Contribution
2
