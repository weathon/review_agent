# Examining Alignment of Large Language Models through Representative Heuristics: the case of political stereotypes

- Decision: Accept (Poster)
- Scores: 6, 6, 8

## Abstract
Examining the alignment of large language models (LLMs) has become increasingly important, e.g., when LLMs fail to operate as intended. This study examines the alignment of LLMs with human values for the domain of politics.  Prior research has shown that LLM-generated outputs can include political leanings and mimic the stances of political parties on various issues. However, the extent and conditions under which LLMs deviate from empirical positions are insufficiently examined. To address this gap, we analyze the factors that contribute to LLMs' deviations from empirical positions on political issues, aiming to quantify these deviations and identify the conditions that cause them. 

Drawing on findings from cognitive science about representativeness heuristics, i.e., situations where humans lean on representative attributes of a target group in a way that leads to exaggerated beliefs, we scrutinize LLM responses through this heuristics' lens. We conduct experiments to determine how LLMs inflate predictions about political parties, which results in stereotyping. We find that while LLMs can mimic certain political parties' positions, they often exaggerate these positions more than human survey respondents do. Also, LLMs tend to overemphasize representativeness more than humans. This study highlights the susceptibility of LLMs to representativeness heuristics, suggesting a potential vulnerability of LLMs that facilitates political stereotyping. We also test prompt-based mitigation strategies, finding that strategies that can mitigate representative heuristics in humans are also effective in reducing the influence of representativeness on LLM-generated responses.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper explores the alignment of large language models (LLMs) with human intentions, focusing specifically on their susceptibility to political stereotypes. It investigates how LLMs deviate from empirical political positions, often exaggerating these positions compared to human respondents, which suggests vulnerability to representativeness heuristics. Experiments demonstrate that prompt-based mitigation strategies can reduce these tendencies, providing insights into better aligning LLMs with human values and reducing biased behavior.

### Strengths
1. The paper brings an underexplored perspective to understand and mitigate bias in LLMs by introducing representativeness heuristics from cognitive science in the context of political stereotypes. 

2. It proposes a systematic quantification of the conditions under which LLMs deviate from empirical political positions, assessing the extent of bias and misalignment. 

3. The mitigating strategies via prompt provide a simple yet practical solution to reduce stereotypes.

### Weaknesses
1. Presentation of the paper needs improvement. Some figures and tables are too small to read (i.e. Figures 3 and 4, Tables 1, 3, 7, and 8, etc.). The figure size is not consistent. The color denoted different methods in Figure 2 are hard to distinguish. There are some repeated definitions or sentences, such as the re-definition of kappa in the paragraph of **Prompt Style Mitigation Analysis**. 

2. Lack of analysis of prompt style mitigating strategies’ results, such as which strategies make LLMs more aligned to human preferences, why baseline LLMs perform better in some tasks, etc. 

3. The **potential effectiveness of political representative heuristics on downstream tasks** is unclear. The connection between stereotypes that this paper identifies and quantifies to fake news should be more clearly explained. The behavior of LLMs in fake news detection could be affected by the pre-training corpus.

### Questions
None

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
4

### Summary
This paper examines the alignment of LLMs through representative heuristics using political stereotypes as a reference context. The authors unveil that although LLMs can mimic certain political parties' positions on specific topics, they do so in a more exaggerated manner compared to humans. Finally, this work proposes some prompt-based mitigation strategies aimed at limiting such exaggerations.

### Strengths
- The findings of this work are valuable, as the unveiling of exaggerated positions compared to humans (despite being limited on the political context) is key to better comprehending how we should interact with these systems, and whether interventions are needed to align them more with human values and perspectives.
- The manuscript is well written, the methodology is properly formalized, non-ambiguous, and easy to follow. All methodological aspects are well supported by reference literature.
- The choice for diverse LLM families is valuable as sheds light on the different "behaviors" they might exhibit based on varying training data and alignment approaches.
- The proposed intervention techniques turn out to be reasonably effective in mitigating the exaggerated intrinsic behaviors.
- The Appendix of the manuscript complements the main content with additional relevant information for the proper understanding of the work.

### Weaknesses
- Focusing just on a single context (i.e., political) and scenario (the US one) is the weakest point to me, as it limits the generalizability of the unveiled patterns.
- Despite being valuable, the results would require more emphasis on the conditions underlying certain behaviors (as stated throughout the manuscript), as it will further help this work unveil the roots of the unveiled exaggerations.
- The results presentation contrasts with the methodology, as it has room for improvement in both the figures/tables presentation (some of them are hard to read) and discussion.

### Questions
- Adding more up-to-date models would be useful to also grasp potential "developments" into the unveiled positions; similarly, considering some open models might improve matching certain behaviors with specific approaches (thanks to potentially greater transparency in training data and alignment techniques).
- As the authors mentioned refusals, I wonder how they handled them and on what occasions they occurred. Shedding light on the latter point would further unveil the roots of certain exaggerated positions.
- Related to the previous point, did the models experience hallucinations? If yes, how were they handled?
- As a minor remark, Section 11 might contain some typos on the followed Ethics Policy.

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
This paper focuses on the challenges and limitations of using LLMs to simulate human behaviour. In particular, it discusses how LLMs measure stereotypical behaviour w.r.t. groups of individuals self-identified as either Democrats or Republicans. The authors use GPT-3.5, GPT-4, Gemini Pro, and Llama2 models to estimate to what extent the beliefs generated by LLMs are representative of aggregated empirical opinions specified by individuals belonging to either party (the authors use two existing datasets, ANES and MFQ, for their analysis). Results show that for ANES, LLMs tend to inflate responses for Republicans, and deflate responses for Democrats. The same is true for Democrats on MFQ (the results for Republicans are inconsistent). Overall, the results show that beliefs are consistently exaggerated by LLMs as compared to the empirical means derived from human surveys.

### Strengths
* The paper discusses an interesting topic by analyzing to what extent LLM responses are representative of human responses in the context of political opinions. The provided results are useful to inform future work aiming to better understand how LLMs can be used in that context.
* The paper’s analysis is overall extensive and thorough, even though I have recommendations on improving the paper's structure (see weaknesses). 
* I appreciate the Limitations specified in Section 10 of the paper.

### Weaknesses
* The paper uses excessive formalism to introduce the proposed method and several crucial details are moved into the Appendix. To improve readability and presentation of the obtained findings, I’d recommend to move parts of Section 3 into the Appendix instead, and add more details on the empirical setup to the main manuscript. 
* The presentation could be improved. Citations should be surrounded with parentheses if used passively as this improves readability. Some citations in Section 5.2 are incorrectly ordered. The results in Figure 2 could be presented more clearly, for example by disentangling the plots between Democrats and Republicans. I find some of the Tables (e.g., Table 1 and 3) too full and overwhelming.

### Questions
On the prompt sensitivity check in Appendix F, do you have an understanding of how this changes when adjusting the temperature values? Or, more generally, how much variation in the obtained results would you expect as the temperature values provided in Appendix D change?

### Soundness
4

### Presentation
3

### Contribution
3
