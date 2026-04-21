# Future Language Modeling from Temporal Document History

- Avg Score: 7.33
- Decision: Accept (poster)
- Scores: 8, 8, 6

## Abstract
Predicting the future is of great interest across many aspects of human activity.  Businesses are interested in future trends, traders are interested in future stock prices, and companies are highly interested in future technological breakthroughs.  While there are many automated systems for predicting future numerical data, such as weather, stock prices, and demand for products, there is relatively little work in automatically predicting textual data.  Humans are interested in textual data predictions because it is a natural format for our consumption, and experts routinely make predictions in a textual format (Christensen et al., 2004; Tetlock & Gardner, 2015; Frick, 2015). However, there has been relatively little formalization of this general problem in the machine learning or natural language processing communities.  To address this gap, we introduce the task of future language modeling: probabilistic modeling of texts in the future based on a temporal history of texts.  To our knowledge, our work is the first work to formalize the task of predicting the future in this way.  We show that it is indeed possible to build future language models that improve upon strong non-temporal language model baselines, opening the door to working on this important, and widely applicable problem.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies and introduces the task of future language modeling. This practically involves generation of text that would be written at a future time, which is not present in the pre-training of the langauge model used. This is possible through extrapolating usage from historical time intervals.

The paper introduces three methods that enhance the regular transformer-based architecture. These rely on LSTMs to model the historical frequency and predict its change with time.

A new data set and evaluation setup is proposed, where the task is to predict future abstracts from the ACL conference. Results show the proposed methods improve over all baselines.

### Strengths
To the best of my knowledge, the application and task are novel and interesting to study.

Three architectures are introduced, each building on each other. The effects of each component are studied and give us a sense of what is working and what is needed to improve performance.

### Weaknesses
More motivation and applications for this task could be provided or speculated.

The paper uses GPT-2 as the base model for all experiments. It would be better for testing robustness to potentially include a few more base models, perhaps with larger sizes or an encoder-decoder architecture for diversity.

Additional experiments and evaluation on downstream tasks can be performed using conditional generation or prompting on existing data sets with temporal dimensions e.g. the temporal NER data set (Rijhwani & Preotiuc-Pietro, ACL 2019; Luu et al NAACL 2022) or generating hashtags in future tweets using just the tweet text (similar to Preotiuc-Pietro & Cohn, EMNLP 2013). These would avoid the issues associated with evaluating generations.

Another interesting experiment to conduct would be to study prediction more into the future and quantify the expected degradation as the time window increases.

### Questions
Please expand on what stopword list was used, as that plays an important role in content perplexity and meteor scores.

### Soundness
3 good

### Presentation
3 good

### Contribution
4 excellent

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This study pioneers the task of future language modeling, aimed at generating predictive text based on historical documents. It innovates by integrating temporal biases into a pre-trained GPT-2 model, guiding the generation of future-oriented text. The models, trained on NLP paper abstracts from the ACL Anthology spanning 2009-2021, surpass traditional non-temporal models in both automated (Perplexity, METEOR) and human evaluations.

### Strengths
Originality:
- First to formalize future textual data prediction using temporal information.
Develops novel methods for measuring temporal dynamics in language modeling.

Quality:
- Presents a thorough structure, comparing three new models against multiple baselines.
- Demonstrates model effectiveness through careful data handling, especially distinguishing between content and non-content words during evaluation.

Clarity:
- Clearly articulates research motivations, background literature, methodology, and findings.

Significance:
- Offers significant research outcomes with implications for various applications.
- Discusses potential future applications and necessary adaptations.

### Weaknesses
To improve readability, you can align the organization of tables and figures more closely with their corresponding text.

### Questions
none

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors proposed a language model that can capture the temporal pattern of text generation. They proposed 3 model variations to attack this problem and conducted experiments over a dataset of ACL paper abstracts. Their most advanced model, the doubly contextualized model, can outperform the GPT-2 based baselines and other variations in both automatic evaluations and human evaluations.

### Strengths
1. Modeling of temporal patterns in LLM has not captured much attention from the community. Yet it is an important problem to look into.
2. The paper is very easy to follow. The authors did a good job of describing their ideas and approaches in simple yet accurate terms and notations.
3. The proposed models look reasonable and are proven to be effective in generating future text based on historical documents.

### Weaknesses
1. The proposed models are relatively simple and don't leverage the power of the most advanced LLM. Some problems the authors tried to solve, such as the gating problem in Sec 3.4 look like sth that would not be an issue to GPT-3 or other recently developed LLM as they are very effective in generating readable and coherent text. Finetuning a more powerful LLM with the latest text data seems to be a very effective way to model temporal patterns.
2. Some details and questions from the experiment were not well discussed. For example, how many raters participated in the human evaluation, what are their agreements, and how subjective are their ratings? Besides, the results in Table 4. are worth more analysis and discussion. Why do the baselines not perform well in Problem and Method? Intuitively, they should be good at generating coherent and readable content.

### Questions
NO

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
