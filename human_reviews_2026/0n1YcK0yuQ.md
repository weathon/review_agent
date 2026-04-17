# One Model, Many Morals: Uncovering Cross-Linguistic Misalignments in Computational Moral Reasoning

- Decision: Reject
- Scores: 6, 2, 2, 4

## Abstract
Large Language Models (LLMs) are increasingly deployed in multilingual and multicultural environments where moral reasoning is essential for generating ethically appropriate responses. Yet, the dominant pretraining of LLMs on English-language data raises critical concerns about their ability to generalize judgments across diverse linguistic and cultural contexts. In this work, we systematically investigate how language mediates moral decision-making in LLMs. We translate two established moral reasoning benchmarks into five culturally and typologically diverse languages, enabling multilingual zero-shot evaluation. Our analysis reveals significant inconsistencies in LLMs' moral judgments across languages, often reflecting cultural misalignment. Through a combination of carefully constructed research questions, we uncover the underlying drivers of these disparities, ranging from disagreements to reasoning strategies employed by LLMs. Finally, through a case study, we link the role of pretraining data in shaping an LLM's moral compass. Through this work, we distill our insights into a structured typology of moral reasoning errors that calls for more culturally aware AI.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors study how moral reasoning in LLMs changes depending on the language used to prompt the model. To do this, they translate the MoralExceptQA and ETHICS datasets into 5 languages and compare the behaviors of 7 models on these datasets. They demonstrate that LLMs reason differently and come to different conclusions depending on the language, often in ways which seem intuitively linked to the culture associated with the language (e.g. English induces more utilitarian reasoning than Hindi). The results are interesting and novel, and the authors go into a lot of depth analyzing different properties of the models’ moral reasoning. The only major issues with this paper are issues with data visualization (Fig. 3 in particular is difficult to interpret), and a handful of contradictory-seeming results which make me question the accuracy of a subset of the methods used.

### Strengths
1. The depth and breadth of investigation is fantastic. The authors asked a lot of follow-up questions (e.g. which reasoning steps do models rely on more in different languages) and then found clever ways to answer them.  
2. The question is novel, interesting, and relevant. It is unambiguously true that most research on moral reasoning in LLMs is done in English, and this does not capture the full picture of how these models are used in the real world.  
3. The evaluations are quite comprehensive, using a large range of models from many different model families.  
4. The authors thought critically about potential issues with their methods and generally addressed them in the paper. E.g. I was concerned about potential subtle semantic shifts that happen during dataset translation, but the authors already investigated this in an appendix.  
5. The findings are quite interesting, matching the intuitions one would have about how language could affect moral reasoning (e.g. European languages inducing utilitarianism and Asian languages inducing virtue ethics).

### Weaknesses
1. The data visualization in some figures is awful and urgently needs fixing. Fig. 3 in particular is very hard to interpret, it has no indication of what the colors mean (presumably the darker the brown, the more disagreement there is; this should be made explicit), and the color spectrum is so small that it’s hard to notice the differences (e.g. in the MEQ subfigure, all I can really see is that every language agrees with itself and disagrees with all the others). The extremely short figure caption doesn’t help. Additionally, Fig. 2 probably shouldn’t be a line plot given the nature of the data; this graph choice makes it quite hard to read.  
2. Some of the claims seem to be contradicted by the data (unless I’m missing something). In particular, they claim that “Virtue scenarios draw the clearest lines, with Hindi and Urdu forming a distinct cluster” (line 156). Figure 3 seems to suggest that Hindi and Urdu actually strongly disagree with each other on ET-Virtue.  
3. There’s an unexplained, seemingly contradictory difference between figures 4(b) and 5(a). The former suggests that Urdu and Hindi are in the same cluster, the latter suggests they’re diametrically opposed in their moral foundations. This makes it seem like the authors get very different answers for the same underlying question depending on the method used, which makes me question the reliability of the method; I could be missing something here though. Can the authors explain this apparent discrepancy? Similarly, Fig. 5(b) (and its description, lines 360–361) says that “English shows stronger Sanctity than Hindi or Urdu). Again, this seems to contradict Fig. 4(b) where “Divine Command Theory” is strongly present in Hindi and Urdu and nearly completely absent in English.  
4. Some of the central methods used in this paper aren’t explained at all, the authors only give a reference to the paper which introduced the method. In particular, eMFD and OlmoTrace are central to the paper’s analysis, and should be briefly explained in a background section (Sec. 2 would be a good place to put this if the section title is changed to “Background”).  
5. While the authors use a solid range of models, I wish they also used at least one much larger model, e.g. GPT-5 or Claude 4.5. In its current form, I’m uncertain as to how much the findings generalize to frontier models (as it seems at least plausible that very large models form abstractions that are more unified across languages).  
6. Typo at the end of line 063: should be “ LLMs’ ” not “ LLM’s ”

### Questions
See weaknesses, in particular points 2 and 3

### Soundness
3

### Presentation
2

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
The paper aims to evaluate the cross-linguistic moral reasoning abilities of LLMs, noting that these models are often pretrained primarily on English data, which can introduce cultural misalignment when deployed in multilingual settings. To address this, English-centric moral reasoning benchmarks (MoralExceptQA and ETHICS) were translated into Chinese, German, Hindi, Spanish, and Urdu to conduct zero-shot evaluations. The findings reveal significant inconsistencies and disparities in LLMs' moral judgments across languages, with performance generally favoring English and showing poor reliability in low-resource languages like Hindi and Urdu. The study further analyzes the reasoning patterns, ethical frameworks invoked, and the influence of pretraining data sources, concluding with a FAULT typology of moral reasoning errors that calls for more culturally aware AI development.

### Strengths
1. Creation of a multilingual dataset,  obtained by  translating two established, English-centric moral reasoning benchmarks (MoralExceptQA and ETHICS) into five geographically, typologically, and culturally diverse languages (i.e.,  Chinese, German, Hindi, Spanish, and Urdu), in addition to English.

2. Investigation of    how language mediates moral reasoning in LLMs through four     research questions     probing ethical preferences, reasoning strategies, moral framing, and the influence of pretraining data.

### Weaknesses
The reviewer's concerns are about methodological and interpretive challenges related to data resources, model selection, and the difficulty of disentangling linguistic and cultural variables. 

1.     RQ1 establishes that LLMs diverge across languages, favoring English and struggling noticeably with low-resource languages such as Hindi and Urdu. However, the analysis in the sources does not clearly decouple the causes of this divergence. It's indeed  unclear whether the divergence and poor performance in low-resource languages are primarily due to a linguistic bias (e.g., tokenization issues, limited data resources) or to cultural biases embedded in the training data.
2.      The claim at lines 153–154, stating that model differences in consistency when handling low-resource or culturally distinct languages are "likely due to limited training data and varying cultural norms", serves as a crucial hypothesis for the entire work. this statement is too vague because it relies on generalized assumptions about data scarcity and cultural variation without immediate supporting evidence to quantify their impact on the models at that point in the study. While subsequent RQs (RQ2, RQ3, RQ4) are dedicated to probing these drivers (moral values, ethical frameworks, and pretraining data), the claim itself is highly speculative without this deeper investigation, making the initial conclusion regarding low-resource languages insufficiently grounded.
3.    The choice of models, particularly the largest one tested, introduces a potential bias:  the largest model tested, , OLMo2-Instruct (32B), is not a declared multilingual model.  Relying on a predominantly English-focused, high-capacity model risks skewing the results toward English language norms and performance ceilings, despite the study's goal of cross-lingual evaluation.
4.   The methodology for analyzing moral values relies on the translation process on eMFD that may be unreliable. Because the eMFD provides probabilistic scores for individual words (moral foundations terms) rather than contextual sentences or phrases, the translation process (using SeamlessM4T) is likely to be unreliable due to the lack of context to properly translate each word. 
5.  Presentation issue. it would be much better to describe all methdology before presenting and analyzing results.

### Questions
See above weaknesses, consider addressing the discussed points and providing clarifications on them.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper aims to systematically investigate how language mediates moral reasoning in large language models (LLMs), as translation from one language to another does not transfer morality. The authors translated six moral question-answering (QA) tasks into five languages and evaluated seven modern LLMs on them. The paper reveals disparities in final task scores (RQ1) and the moral concepts and reasoning phases employed in reasoning traces (RQ2) across languages, models, and tasks. Furthermore, the authors demonstrate how moral concepts in reasoning traces contribute to the final answer (RQ3). Finally, via a case study (RQ4), they show how pre-training data domains correlate with moral concepts in moral reasoning traces.

### Strengths
The authors provide a generally well-motivated rationale for examining intercultural variation in the moral reasoning of large language models (LLMs). Given the widespread adoption, global use, and increasing trust in LLMs, efforts to better understand their moral reasoning are valuable to the research community. The paper presents an interesting initial investigation into how moral reasoning traces differ across languages. Further, the multilingual dataset and evaluation setup could be valuable for follow-up research.

### Weaknesses
- The related work section lacks depth and fails to connect to key prior studies (e.g., Haemmerl et al., ACL 2022; Abdulhai et al., EMNLP 2023; Agarwal et al., LREC 2024; Jin et al., ICLR 2025), making it unclear how the paper builds on or differs from them.
- Language selection is insufficiently justified—why these five languages, and are they representative of typological and cultural diversity?
- Definition of “moral reasoning” is vague; it is described as “higher-level reasoning” without referencing psychological or computational frameworks.
- Model selection criteria and rationale are not discussed.
- RQ1: The claim of English dominance is overstated; several languages (e.g., German, Spanish in ET-JUS, ET-UTI) perform comparably. Relationship between F1 and compliance rates is unclear, and “low-resource” is not defined or quantified.
- RQ2: The fifteen-stage taxonomy derived via an LLM-judge is not sufficiently motivated or validated.
- RQ3: The reason for training a separate regressor on UniMoral is unclear, and large variations in coefficients (Fig. 5) are unexplained.
- RQ4: Semantic similarity as a measure of memorization lacks theoretical justification and citation; the new source-text taxonomy appears ad-hoc.
- Interpretation of results remains largely qualitative; statistical significance or confidence intervals are absent.
- Some references are preprints although published versions exist; minor terminological inconsistencies (e.g., “Chinese” vs. “Mandarin”).


### References
-	Speaking Multiple Languages Affects the Moral Bias of Language Models (Haemmerl et. al, ACL 2022)
-	Moral Foundations of Large Language Models (Abdulhai et. al, EMNLP 2023)
-	Ethical Reasoning and Moral Value Alignment of LLMs Depend on
the Language we Prompt them in (Agarwal et. al, LREC 2024)
-	Language Model Alignment in Multilingual Trolley Problems (Jin et. al, ICLR 2025)

### Questions
1.	Figures 2–3: What do the symbols and color scales represent? How should they be interpreted relative to performance and disagreement?
2.	How was “low-resource” status determined quantitatively?
3.	Why were exactly these five languages chosen—based on typological diversity, population, or dataset coverage?
4.	What ensures that the LLM-generated taxonomy of reasoning stages (15 phases) is valid and non-arbitrary?
5.	How was semantic similarity chosen as the measure for memorization in RQ4? Are alternative approaches (e.g., n-gram overlap, retrieval-based alignment) considered?
6.	In Figure 1, shouldn’t the AI’s answer be “yes” in both examples or is the mismatch intentional to illustrate misalignment?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper explores how LLMs perform moral reasoning in multiple languages. The paper translates two moral reasoning benchmarks: MoralExceptQA and ETHICS into five non-English languages (Chinese, German, Hindi, Spanish, Urdu), and evaluates several LLMs in a multilingual zero-shot setup. It uncovers inconsistencies in moral judgments across languages, often correlated with linguistic and cultural differences. Through quantitative and qualitative analyses, it analyses moral reasoning failures and links these moral divergences to model pretraining data sources using a case study with OLMOtrace.

### Strengths
1. The exploration of moral reasoning in various languages is relevant and useful.
2. The analysis in the paper is sound.
3. The use of OLMoTrace to link model outputs to training data is also useful.

### Weaknesses
1. While the paper is useful and has a place in the literature, there is no conceptual novelty or new innovation in this paper. A relevant problem has been identified and everything is solid in its own. But not particularly unique or exciting.

2. The use of translation is a major bottleneck in this paper. While the paper has manual quality checks and semantic shift analysis, this is still a limitation.

3. The two benchmarks are inherently centred on the western hemisphere. Translating them into non-Western languages may bring Western moral assumptions. This can be better discussed in the paper.

### Questions
See weaknesses above.

### Soundness
2

### Presentation
3

### Contribution
2
