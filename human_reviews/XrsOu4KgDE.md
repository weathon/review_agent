# Attributing Culture-Conditioned Generations to Pretraining Corpora

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 8, 8, 8, 6, 5

## Abstract
In open-ended generative tasks like narrative writing or dialogue, large language models often exhibit cultural biases, showing limited knowledge and generating templated outputs for less prevalent cultures. Recent works show that these biases may stem from uneven cultural representation in pretraining corpora. This work investigates how pretraining leads to biased culture-conditioned generations
by analyzing how models associate entities with cultures based on pretraining data patterns. We propose the MEMOED framework (MEMOrization from prEtraining Document) to determine whether a generation for a culture arises from memorization. Using MEMOED on culture-conditioned generations about food and clothing for 110 cultures, we find that high-frequency cultures in pretraining data yield more generations with memorized symbols, while some low-frequency cultures produce none. Additionally, the model favors generating entities with extraordinarily high frequency regardless of the conditioned culture, reflecting biases toward frequent pretraining terms irrespective of relevance. We hope that the MEMOED framework and our insights will inspire more works on attributing model performance on pretraining data.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper introduces a novel framework called MEMOED, designed to analyze how pretraining data contributes to cultural biases in large language models (LLMs). The framework distinguishes between knowledge generated through memorization and generalization. By focusing on cultural topics such as food and clothing across 110 different cultures, the authors demonstrate that models tend to overmemorize symbols from highly represented cultures, while underperforming in generating culture-specific symbols for less represented ones. Through a detailed analysis of the OLMo-7B model, the paper offers a systematic method to trace how pretraining data influences model outputs, highlighting the limitations of current LLMs in producing diverse, culturally accurate generations and stressing the need for improved pretraining procedures to address these biases.

(Note: My review has been revised by an LLM for improved grammar.)

### Strengths
- The paper makes a significant contribution by addressing cultural biases in LLMs, and the MEMOED framework provides a valuable tool for tracking these biases.
- The study covers a broad scope, examining 110 cultures, which enhances the depth of the analysis.
- The concept of overmemorization introduced by the authors is intriguing and may have broader implications beyond cultural biases, potentially applying to other LLM phenomena.
- The paper opens up the possibility of examining cultural biases in multilingual LLMs across different languages, which could be an interesting direction for future research.

### Weaknesses
While the methodology appears sound, my only concern is the limited scope of the study, which focuses solely on the OLMo-7B model. As a result, the findings might be seen as a case study specific to this model. It would strengthen the paper if the authors included analyses for at least one additional LLM to broaden the generalizability of their findings.

### Questions
- Do the authors have any plans to propose methods for mitigating the cultural biases identified in this work?
- Comment: I recommend adjusting the notation for subscripts, such as $d_{TOK}$. It currently appears a bit unnatural, and applying italics to the subscript, such as d_{\textit{TOK}}, would enhance clarity.

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
4

### Summary
This paper introduces a novel symbol attribution framework to determine whether symbols in LLM generations, conditioned on a culture, result from memorization of pretraining data. The authors' thorough analysis shows that high-frequency symbols are easily memorized but independent of any culture regardless of their correctness. Additionally, by showing the imbalance between the memorization of high-frequency and low-frequency cultural symbols, this paper underscores the need for improved pretraining data and methods to mitigate cultural biases.

### Strengths
**S1**. This paper introduces a novel symbol attribution framework to determine whether the symbols in LLM generations, conditioned on a culture, result from memorization of pretraining data.

**S2**. The authors provide a nuanced categorization of symbols based on their memorization/generalization levels and a thorough analysis of their relationship with the pretraining data.

**S3**. Their findings demonstrate how LLMs fail to represent cultures that are low-frequency in the pretraining data, calling for improved pretraining data and methods.

### Weaknesses
**W1**. The study is limited to only one pretraining corpus and one LLM, which is understandable given the scarcity of open resources.

**W2**. This study relies on searching for symbols in culture-conditioned generations within the pretraining data and provides a relational analysis. However, it does not guarantee that the selected training documents are causally decisive for the symbols in question. Incorporating influence functions [1] could provide insights into causal relationships. While the computational cost might be an issue, they could be applied to a subset of the dataset or specific experiments.

**W3**. Lines 427-430 require further explanation. What do these correlations imply?

**W4**.  While this study highlights the existing problems with underrepresented cultures in pretraining corpora from a new perspective, it fails to address or propose potential directions for solving these issues. Without this, the paper remains another verification of known problems, which is still valuable but not particularly groundbreaking. The authors should discuss how their findings could inform improved pretraining data/methods or mitigation strategies that do not require changes in pretraining.

**References**
1. Studying Large Language Model Generalization with Influence Functions, Grosse et al., 2023

### Questions
**Q1**. In Figure 1, the top-down order does not match the numerical order. Why are memorized symbols shown at the bottom?

**Q2**. In Figure 3, what does "overgeneralization" refer to? It is not mentioned in the text. Do you mean "overmemorization" instead? The same applies to the caption of Table 3.

**Q3**. How are culture-referring n-grams defined for the Document-Signal to Noise Ratio?

**Q4**. Why do the memorization classification criteria differ for cases where n(C_G) > 5 and n(C_G) < 5?

**Q5**. What do the bold texts represent in the "topic modeling keywords" column of Table 3?


**Typos**:

- Line 345: "none-memorized" should be "non-memorized."
- Lines 106-107: "for less prevalent symbols" -> "for less prevalent cultures."
- There is inconsistent use of "memorisation" and "memorization" throughout the text. It would be better to use one consistently.

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
3

### Summary
This paper describes MEMOED (MEMOrization from pretraining document), a framework designed to classify cultural symbols in LLM-generated text as either memorized or generalized.

### Strengths
1. Novel framework and the new problem of analyzing cultural memorization: The paper develops a systematic approach to determine if cultural symbols generated by an LLM are due to memorized data or generalization. This is a novel problem and the approach is sound and elegant.
2. Good Analysis across Cultures: The study uses data for 110 cultures on topics like food and clothing - the analysis is interesting and the conclusions are interesting as well.

### Weaknesses
Reliance on a Single Model: The analysis focuses solely on the OLMo-7B model and its pretraining dataset, Dolma. Its is unclear from the analysis how the conclusions would vary on other models or models of other sizes.

It is not clear to me how the definitions of what constitutes memorization (e.g. training document classification)  might change the analysis?

### Questions
Is it possible to do this analysis on several OLMo models?

There are some typos: and and (320)

### Soundness
4

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
3

### Summary
About culture-conditioned generations out of LMs, dealing with the issue that biased generations can be driven by pretraining corpus statistics.

They introduce a “symbol attribution framework” to determine if culture-conditioned symbols were memorized in training. They characterize symbols as independent, memorized, or generalized depending on whether they appear in a “culture’s generations” broadly, without a specific culture association, appear primarily in a small set, or if they appear broadly across cultures without presence in the pretraining corpora. 

They use document SNR, minimum token distance, and minimum sentence distance as metrics.
- document snr is the log probability ratio of counts of culture-referent n-grams to others
- minimum token distance is the length of the shortest span of tokens between one referring to the culture and one to the symbol
- minimum sentence distance uses sentences instead of tokens in the above

They use these functions to construct a heuristic for whether a training document shows the memorizable relationship. They then characterize concepts as being overmemorized, and compare the presence of these statistics in the training data & outputs as a predictor to the agreement of human annotators that these relationships are reflective.

They claim that “traceable generalization”, ie., concepts that are not closely related according to their metrics in training data are nonetheless successfully generated, are not correlated to “memorized” concepts for a culture. There’s one or the other, for example “Mexico” contains specifically memorized queries, while Trinidad has none.

**Edit**: I have responded to the authors rebuttal, and modified my "weaknesses" section. I think the technical contributions are sound, so I have bumped my soundness up to 4 (even though I still am a bit troubled by the scope of the experiments). However, I feel that the presentation of this work is severely flawed, particularly wrt how a reader has to piece together what the experimental methodology was while reading the results. So, I am keeping my presentation score at 2 (though I contemplated dropping it to 1). I will keep my overall score at weak accept---I don't think this paper is ready but if it were to get accepted, interested researchers would be able to make their way through it. I would strongly recommend that the authors consider edits for clarity that address my complaints here in the CR if it does get accepted.

### Strengths
Interesting and useful topic to address.

Mildly interesting results; though I am a little unsure about the claims (see weaknesses).

Approach may generalize, not only to memorization of cultural relationships but also to memorization of other facts/information conditioned on context.

### Weaknesses
Presentation of relatively shallow experiments, ~limited technical novelty~, and limited scale of experiments.

~I’m not fully convinced about these definitions that are used to characterize the memorization classes; how do we know that having these statistics over some threshold means that a concept is “memorized” vs just being consistently generated?~

**EDIT:** I understand the paper better after the authors' explanations and edits. I change my mind regarding the technical novelty (which is an unfair complaint to even have in the first place even if it were accurate)

That being said, I stand by my complaint about the small scale of experiments: while many symbols are generated, and a basically comprehensive set of countries are tested, **only two prompts are used to elicit these outputs.**

Over all, my biggest complaint about the paper didn't make it in to my review, but it's the **poor presentation of the method**. I believe it is a serious problem that key details of the experiments such as "how many prompts? how many cultures? how were the symbols extracted from the outputs?" are not clearly lain out before the results. Additionally, the presentation of the methods suffers from a lot of superfluous mathematical notation that clouds clarity, with symbols that once again aren't introduced until *after* an equation is read, requiring considerable backtracking.

See my response to the reviewers.

### Questions
This review is a little low confidence; I'm open to changing my mind.

Please clarify any misunderstandings I have, and elaborate on my concern about the definitions?

Why were the classes of concept chosen?

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper introduces a framework, called MEMOED, to determine whether culture related entities, refered to as symbols, are resulted from memorization or generalization of LLMs based on the pretraining data. This study defines three categories of symbols, i) independent symbols, ii) culture-specific symbols, referred to as memorized symbols, and iii) symbols generalized across certain cultures, referred to as generalized symbols. Various experiments are conducted using OLMO on 110 cultures to understand how OLMO's performance is affected by memorization and generalization.

### Strengths
* This paper selects a good research problem, which is to understand how the frequencies of certain culture related concepts or entities in the pre-training data influence the model performance, in particular from the perspective of generalization and memorization.
* The high-level ideas to discuss about independencies, memorization and generalization are reasonable.
* The dataset covers over 100 cultures.

### Weaknesses
* The definitions of the following concepts and their justification are unclear unclear to me.
    * What is a symbol? Do they cover all linguistic variations of the same entity or concept?
    * How culture is defined? Why it is represented as a combination of country and natonality. The literature in social science has already defined culture. There could be more than one cultures in a country. Would a representation of country and nationality be overly simplied?
    * How to justifiy the definition of memorization through Equation (1)? Why it makes sense?
    * How generalization is defined and why?
* It lacks of justification of the formula for r(D, Q) in Page 5, as well as the measure for memorization. Why log ratio is preferred over the standard techniques, e.g. statistical dependencies? There are often various ways to convey an entity or a concept. How are linguistic variations captured with this measure? As this measure is used together with the contribution score and z-score to determine if a symbol is memorized. There is no empirical evidence or theoretical justification showing that this measure indeed meets the expectation.

### Questions
* There could be an alternative way to convey symbol and culture overmemorization, if the purpose is to show that certain entities occur more often in model outputs that those observed in the pre-training data.
* How do you ensure the quality of annotations using culture experts?
* How symbols are collected? Is there a systematic way to sample data from the 110 cultures?

### Soundness
2

### Presentation
3

### Contribution
2
