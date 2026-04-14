# Aya in Action: An Investigation of its Abilities in Aspect-Based Sentiment Analysis, Hate Speech Detection, Irony Detection, and Question-Answering

- Decision: Reject
- Scores: 5, 6, 5, 1

## Abstract
While resource-rich languages such as English and Mandarin drive considerable advancements, low-resource languages face challenges due to the scarcity of substantial digital and annotated linguistic resources. Within this context, 
in 2024, Aya was introduced, a multilingual generative language model supporting 101 languages, over half of which are lower-resourced. This study aims to assess Aya's performance in tasks such as Aspect-Based Sentiment Analysis, Hate Speech Detection, Irony Detection, and Question-Answering, using a few-shot methodology in Brazilian Portuguese. The objective is to evaluate Aya's effectiveness in these tasks without fine-tuning the pre-trained model, thereby exploring its potential to improve the quality and accuracy of outputs in various natural language understanding tasks.
Results indicate that while Aya performs well in certain tasks like Question-Answering, where it surpassed Portuguese-specific models with an Exact Match score of 58.79%, it struggles in others. For the Hate Speech Detection task, Aya's F1-score of 0.64 was significantly lower than the 0.94 achieved by the Sabiá-7B model. Additionally, the model's performance on the Aspect-Based Sentiment Analysis task improved considerably when neutral examples were excluded, but its handling of complex slang and context-dependent features in other tasks remained challenging. These results suggest that multilingual models like Aya can perform competitively in some contexts but may require further tuning to match the effectiveness of models specifically trained for Portuguese.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper assesses Aya’s performance in four tasks: Aspect-Based Sentiment Analysis, Hate Speech Detection, Irony Detection, and question answering in Portuguese language. The authors also compared Aya's performance with other language models. However, it is not understandable which one is from prior studies in Table 2. It is also unclear whether the Sabia-7B model is studied in this study or previous study.

### Strengths
The paper uses state-of-the-art multilingual LLMs (Aya), that have been known for their capabilities for low-resource languages. Moreover, the study uses four different tasks (SA, HS, ID, and QA) ranging from classification to text generation/QA.

### Weaknesses
The main weakness of this paper is written poorly. The related work and the theoretical background sections are too long. The claim about the examples for few-shot learning could be partially correct but not properly correct. The details of prompting are missing in the paper. The performances are not well discussed in the paper.
I believe the paper will be in good shape if the content is trimmed to a short paper rather than a long full paper.

### Questions
**Comments:**
1. How do you ensure the examples of few-shot learning are representative and diverse? What is the measure for representative for example do you consider the classes or the related input text? 
2. L259-261, Do all the chosen questions begin with "What", “Where”, “Who” and “When” represents the dataset in a general way?
3. L268-L269, "we selectively remove instances from the dataset and include them alongside each test example during inference." -- Why do you remove instances from the dataset? The removed instances belong to which splits are not discussed.
4. In Table 1, few-shot examples are 11 (SA), 10 (HS), 20 (ID), and 4(QA). Do you use the same examples for all inference data?
5. "In total, they mention nine different aspects, including four examples with negative polarity, four with positive polarity, and three that are neutral." -- Who mentioned? I believe there should be a citation.
6. There are nine different aspects in the dataset, does every aspect represent only one sentiment? If not, how does the example represent the other classes that are not selected? 
7. Equations 1-6 are well known to the community, the information should be redundant.
8. L421-422, why neutral is harder than positive and negative is not discussed properly.
9. Comparing the results of two classes (Positive and Negative) with three classes (Positive, neutral, and negative) is not a good idea. Given that the model can easily differentiate two classes where prediction of multi-class is harder.
10. Figure 2, the sum of the confusion matrix for columns is not 100. 
11. The authors stated that the model predicted exactly the same answer as the ground truth. However, it would be interesting to see some examples of those answers.
12. SQUAD v1 is an old dataset and there is a possibility of adding this dataset to Aya's training data. It would be interesting to see the performance on some other QA datasets.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This study aims to assess the performance of Aya, a multilingual generative model trained on a wide range of low resource languages and a variety of downstream tasks like Aspect-Based Sentiment Analysis, Hate Speech Detection, Irony Detection, and Question-Answering. The objective is to evaluate Aya's effectiveness in these tasks but only on the pre-trained model without any finetuning. This would reveal its potential to improve the quality and accuracy of outputs in various natural language understanding tasks. Instead, this work employs a few-shot methodology to evaluate the model's effectiveness as this approach is particularly better suited in abscence of extensive labelled data in low resource languages. Results indicate that while Aya performs well in certain tasks like Question-Answering for languages like Portuguese, for other tasks like Hate Speech Detection the performances were significantly underwhelming. These results suggest that multilingual models like Aya can perform competitively in some contexts but may require further tuning to match the effectiveness of models specifically trained for Portuguese.

### Strengths
This paper addresses an important problem that aims to mitigate technological inequities towards low resource language.
The methodology used in this paper is reasonable and easy to understand. In addition, the paper also makes a thorough inquiry of related research before carrying out their work.
The experiment section provides adequate amount of evaluation to properly assess the performance of the model for a number of language tasks in a low resource language like Portuguese.

### Weaknesses
This work is low in novelty despite addressing an important problem.  Training a generative model for a low resource language like Portuguese is certainly important work. While the authors employ few shot learning as a logical workaround for the issue of low training data, their evaluation reveals the relative limitation of this approach after a point. The authors could further look into some technical innovations in this regard to improve performance of the Aya model in portuguese for tasks like Hate Speech or Irony Detection.

### Questions
Other than Portuguese have the authors considered evaluating their model for any other low resource language? That would provide a more comprehensive idea of the performance variance of the Aya across languages.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper presents an evaluation of Aya, a multilingual language model, on four tasks including Aspect-Based Sentiment Analysis, Hate Speech Detection, Irony Detection, and Question-Answering, highlighting its strengths and limitations, particularly when compared to transformer-based models for the Portuguese language.

### Strengths
- This paper is centered around a very interesting and highly relevant topic.

- Clearly scoped tasks and objectives.

### Weaknesses
There are a few points that must be addressed:

Throughout the text it feels like the authors use aggressive speech, hate speech, and offensive speech interchangeably. For example, in Figure 1 we see offensive vs non-offensive. The authors should pay attention to this aspect: clearly define the specific type of abusive phenomena they are focusing on, and use that terminology consistently throughout their work – please look into the nuances surrounding the overlapping abusive phenomena (the distinction between hate speech, abusive language, and offensive language). See for example the work of Poletto et al.:

*Poletto F, Basile V, Sanguinetti M, Bosco C, Patti V. Resources and benchmark corpora for hate speech detection: a systematic review. Language Resources and Evaluation. 2021 Jun;55:477-523.*

I would have liked for the authors to spend a little bit more space detailing the methodology. The authors should provide the criteria used for selecting the examples, as well as the exact prompts used for all the tasks (not just QA):

-  How did the authors ensure that the examples were representative and diverse? 
- What was the exact input for the Aya model? The authors provide the prompt for the QA, but not for the other tasks. Does that mean that for the other, only the examples were used? I am asking this because I am surprised by the fact that for hate speech *‘the generation was more efficient when using the labels as numbers, instead of the actual labels’* (cf. lines 309-311). For example, I just asked Aya if it is familiar with the hate speech definition provided by the OHCHR, and the answer was positive. How would the generation have changed if including this type of information? Did the authors provide any task-specific instructions to the model beyond the few-shot examples?

A more in-depth error analysis would have been interesting to have. The authors could consider a subset of the misclassified data and construct aggregate statistics over modes of failure which would then inform us how prevalent each of the kinds of mistake are. This would be useful for future research, as it would become possible to prioritize on which kind of misclassification to work on next.
In regards to the ABSA example provided, I don’t agree that *‘hotel’* has a neutral sentiment – it seems to be conflict (i.e., both positive and negative) or, we could say that the entity hotel has a positive sentiment towards the attributes location and service, but negative towards the attribute that would incorporate size/room description.

Was there any hyperparameter tuning performed for the transformer-based models? Interesting results for the Albertina models on the ID task. 

Suggestions:

- abbreviations should be presented once, at their first occurrence, and always used afterwards (except for the abstract)
- I believe the paragraph starting on line 089 is actually a continuation of the previous one and does not require splitting
- line 124: an -> a
- line 161: a -> an

### Questions
Please see the above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
1

### Rating Number
1

### Confidence
4

### Summary
This paper evaluates the performance of the multilingual Aya model across tasks such as Aspect-Based Sentiment Analysis (ABSA), Hate Speech Detection (HS), Irony Detection (ID), and Question-Answering (QA) in Brazilian Portuguese. Through a few-shot learning approach, Aya demonstrates competitive results in QA, surpassing some Portuguese-specific models, though it underperforms in tasks involving nuanced or slang-heavy language like HS. The study highlights Aya's potential in low-resource contexts while indicating the need for further tuning for certain language-specific tasks to match or exceed specialized models​

### Strengths
1. The paper is well-structured with a clear methodology.

2. It offers some insights into Aya's performance in multilingual contexts and addresses challenges faced by low-resource languages.

### Weaknesses
1. This paper only conducts an evaluation of the multilingual large language model on a specific language, Brazilian Portuguese, even though Aya supports 101 languages. The tasks are limited to Aspect-Based Sentiment Analysis (ABSA), Hate Speech Detection (HS), Irony Detection (ID), and Question-Answering (QA). Many other NLP tasks could be studied, such as reading comprehension, syntax parsing, named entity recognition, and event extraction. The evaluation scope and language focus are limited, reducing the paper's contribution.

2. Describing the equations for precision and recall in such detail seems unnecessary and only increases the document length without adding value.

3. This paper lacks inspiring conclusions from the experiments. It only presents main results and a confusion matrix, without providing in-depth analysis through fine-grained evaluation or insights into the working principles of the Aya model.

### Questions
See Above

### Soundness
2

### Presentation
3

### Contribution
1
