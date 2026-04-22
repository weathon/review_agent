# What Is The Political Content in LLMs' Pre- and Post-Training Data?

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2, 4

## Abstract
Large language models (LLMs) are known to generate politically biased text, yet how such biases arise remains unclear. A crucial step toward answering this question is the analysis of training data, whose political content remains largely underexplored in current LLM research. To address this gap, we present in this paper an analysis of the pre- and post-training corpora of open-source models released together with their complete dataset. From these corpora, we draw large random samples, automatically annotate documents for political orientation, and analyze their source domains and content. We then assess how political content in the training data correlates with models' stance on specific policy issues. Our analysis shows that left-leaning documents predominate across datasets, with pre-training corpora containing significantly more politically engaged content than post-training data. We also find that left- and right-leaning documents frame similar topics through distinct arguments and that the predominant stance in the training data strongly correlates with models' political biases when evaluated on policy issues. Finally, pre-training corpora subjected to different filtering and curation procedures exhibit broadly similar characteristics with respect to their political content. These findings underscore the need to integrate political content analysis into future data curation pipelines as well as in-depth documentation of filtering strategies for transparency.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper investigates the political bias present in open-sourced LLM training datasets and correlates that to bias responses given by the OLMO2 model trained on these datasets. The paper first uses an LLM-based method to identify the political bias present in samples of the datasets. The paper then uses a combination of models and methods to analyze the differences in source domains and topics of the identified biased data. Finally, the paper prompts the OLMO2 model for its political viewpoints on topics and correlates these stances with the topical stances identified in the training datasets. The paper generally finds that the model inherits the political biases of the datasets that it is trained on.

### Strengths
The paper is attacking a significant problem, is clearly written, and has reasonably good validity for its testing procedures. As the paper points out, LLMs and LLM-driven technologies like AI Agents are increasingly being relied upon for truthful information, including in the political sphere. Furthermore, the paper presents a significant finding: even if the politically biased parts of the training data are in the minority overall, they appear to carry through in the generations produced by the LLM. The paper is also generally thorough in validating any claims it makes. For example, the paper tests its LLM-based classification of political bias on other benchmark bias datasets as well as the protocol established in Goet (2019). Another example comes in prompting the model for its political opinions on topics. The paper uses a number of different phrasings in the prompt and different proxies for the implicit political stance to really see if the political opinions of the model is stable. I also appreciate the thoroughness of the analysis of the differences between the sources of right-leaning and left-leaning documents in the datasets. Overall, the paper is generally thorough in its evaluation and has a significant result in finding a correlation between the biases present in the dataset and an LLM trained on that dataset.

### Weaknesses
The paper does lack some novelty and grounding in other works that have investigated political bias in LLMs as well as concepts like stance or bias labeling by LLM. For the latter, authors like Rozado in “The Political Preferences of LLMs” and Fujimoto and Takemoto “Revisiting the Political Biases of ChatGPT” used political preference tests as a way to measure the bias of models. For stance labeling by LLM, there are a number of techniques that have been developed for this particular task, including with testing on politically sensitive stance datasets. Some examples include  Zhang et al. “A Logically Consistent Chain-of-Thought Approach for Stance Detection”, Ziems et al. “Can Large Language Models Transform Computational Social Science?”, and Cruickshank and Ng “Prompting and Fine-Tuning Open-Sourced Large Language Models for Stance Classification,” to name a few. The work would have better methodological grounding and validity (and likely see better scores against the benchmark testing that they do) if they adopted some already established approaches. Finally, there are other works that have found similar results to the significant result that this paper did, albeit by different methods. Ng et al. In “Examining the Influence of Political Bias on Large Language Model Performance in Stance Classification” found a statistically significant difference in the performance of LLMs across various politically oriented stance classification tasks and that this difference primarily manifests at the dataset level. Overall, I think the work is lacking somewhat in novelty and could benefit from adopting established approaches for tasks like bias or stance labeling by LLM.

### Questions
- Do these results extend to other forms of bias? I believe other works have also indicated that the training data gives rise to gender bias or regional biases in LLMs.

- Where is the correlation coefficient coming from in the last paragraph of section 6? Is it just the correlation of the 8 topical stance valences between the model’s outputs and the accumulation of the stance labeling done on the datasets?

### Soundness
3

### Presentation
3

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
This paper investigates how political biases in large language models originate from their training data. Using OLMO2, the largest open-source LLM with full pre- and post-training datasets, the authors sample documents, classify them as left-, right-, or neutral-leaning using a validated classifier, and analyze content, sources, and framing. Results show that left-leaning documents vastly outnumber right-leaning ones across datasets, and pre-training data contain more politically engaged material than post-training data. Moreover, models’ policy stances correlate strongly (r = 0.90) with the political leanings in their training corpora, indicating that political bias largely emerges during pre-training.

### Strengths
1. First systematic data-level study connecting political content in both pre- and post-training corpora to model behavior.

2. Uses OLMO2’s transparent datasets and a rigorous sampling plus validation pipeline.

3. Presents interpretable topic and framing analysis revealing nuanced ideological contrasts.

4. Strong empirical evidence (r=0.90) linking data stance to model bias, with clear implications for fairness and transparency.

### Weaknesses
1. The political classification model relies on heuristic assumptions about “neutral” data, potentially oversimplifying the ideological spectrum.

2. Focus limited to one model (OLMO2), so generalization to other LLMs is unclear.

3. The causal direction—whether bias truly originates from pre-training or is reinforced later—is not conclusively disentangled.

4. Some sections (e.g., topic clustering) depend heavily on automated summarization, which might introduce interpretive bias.

### Questions
1. How robust is the left-right classifier across cultural or linguistic contexts beyond U.S.-centric framing?

2. Could the observed correlation (r=0.90) arise partly from model alignment procedures rather than purely pre-training data?

3. How do you ensure that automated topic summaries or GPT-generated cluster labels do not reinforce the same ideological bias you are measuring?

4. Have you considered quantifying how much each training stage amplifies or attenuates the bias magnitude?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper conducts a comprehensive analysis of the political content of pre- and post-training data used in LLMs, focusing on the open-source OLMO2 model. They show that left-leaning documents predominate across datasets, with pre-training corpora containing significantly more politically engaged content than post-training data. Moreover, the author provides a dataset including training documents automatically annotated as left-leaning, right-leaning, or neutral.

### Strengths
1.Systematically analyzes the political content in the dataset used to train the OLMO2 model.
2. Provides a dataset containing both left- and right-leaning documents.
3. First work to focus on political bias in training data rather than model outputs.

### Weaknesses
1. The paper mainly provides a comprehensive analysis of the political issues in the training corpus but does not examine the influence of LLM, particularly regarding its mechanism, or propose any methods to address this issue. 
2. All analysis is only based on the training corpus from the OLMO2 model, but the paper only analyzes data not including the model's response; it lacks a comprehensive cross-model or cross-corpus comparison, which may introduce bias. 
3. No sufficient analysis of the classifier model and no human-in-the-loop evaluation for generated content

### Questions
1. Did the authors do any human evaluations to make sure that the political-leaning classifications in the corpus were correct? If not, how was the automatic classifier's reliability checked?
2. What do the authors think about ways to reduce bias during the data-curation stage?

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
This paper presents a thorough analysis of political content in pre- and post-training corpora of OLMO2. The authors find that left-leaning documents are more popular in all the analyzed datasets, and the predominant stance in the training data is also highly correlated with the models' political biases when evaluated on policy issues.

### Strengths
1. A thorough analysis of political content in LLMs' training datasets
2. The findings reveal how political content in the training data is potentially be converted into models' political bias.

### Weaknesses
1. One of the key weaknesses of this paper is the lack of direct implications for model training. Political content might also be connected to the quality of the training data. For example, the New York Times data is left-leaning but has also been recognized as a unique dataset with very high quality. The trade-off between data quality and political content is underexplored in the current draft.

2. While I really like the analysis of models' political biases in different training stages, it would be better if the author could conduct some experiments to further strengthen the causal relationships here. For example, try to change the data distribution of the DPO data and fine-tune the model in different settings. This will allow you to make stronger claims about the connection between data and model biases.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3
