# Extrapolating Large Language Models to Non-English by Aligning Languages

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 3, 3

## Abstract
Existing large language models show disparate capability across different languages, due to the imbalance in the training data. Their performances on English tasks are often stronger than on tasks of other languages. In this paper, we empower pre-trained LLMs on non-English languages by building semantic alignment across languages. We start from targeting individual languages by performing cross-lingual instruction-tuning (CoIT) on LLaMA, i.e. tuning it with translation task data and cross-lingual general task data to obtain cross-lingual models (x-LLaMAs), and formulate underlying scaling laws to investigate the advantages of using scalable translation data. Then we perform multilingual instruction-tuning (MuIT) with mixed resources to build multilingual m-LLaMA. We also illustrate how we leverage the scaling laws to optimize data allocation in a resource-constrained setting. Experiment results on cross-lingual benchmarks XQUAD and MLQA show that x-LLaMAs surpass the English instruction-tuned counterpart (Alpaca) by an average of 27.83% across six non-English languages. Evaluation results on translation dataset Flores-101 show that x-LLaMAs outperform previous LLaMA-based models by an average of 18.89%. Encouragingly, m-LLaMA achieves comparable performance to x-LLaMAs on individual languages and demonstrates the ability to follow multilingual instructions. Further analysis on response content and representation space reveals the alignment of the multilingual semantic space within the middle layers of m-LLaMA.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper propose to enhance the low-resource language ability of  LLM with a multi-task tuning framework which combines the translation tasks and cross-lingual general task. Specifically two tuning method is introduced: cross-lingual instruction tuning and multilingual instruction tuning.  
Experiment shows that the proposed x-LLaMA models achieves significant improvement on non-English languages,
and the language alignment is improvement measured by the improvement in machine translation results.

### Strengths
The paper is clearly written and easy to understand. 
The proposed CoIT and MuIT method is solid in improving the cross-lingual ability of pre-trained language models. 
And the formulated scaling laws to optimize the data allocation of data-constrained MuIT is novel to me.
And I like the experiment design and analysis: 1. Extensive experiment on 6 languages and 3 kinds of dataset to show the improvement of cross-lingual ability.  2. Sophisticated experiment on machine translation task, with various evaluation metrics.

### Weaknesses
I have two concerns about this paper:
1. There is not enough related work comparison: in Table 2, the proposed work is only compared to other LLaMA-based model tuned without multi-lingual dataset, so the improvement sees very significant, but I think it also should be compared with some other cross-lingual instruction tuning methods like "Few-shot Learning with Multilingual Generative Language Models", with the same multi-lingual training data to really reflect the merits of this work.
2. The author argues that the CoIT could improve the semantic alignment.  I think only using translation performance to quantify the improvement in alignment is not enough, since the evaluation metrics like Comet and BLEU doesn't always correlate with better semantic quality. So maybe add the distance between the representations of positive/negative samples is worth trying.

### Questions
See weakness for detail:
1. Is there a better related work comparison to justify this work?
2. Could the improvement in semantic alignment could be better measure?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a strategy to fine-tune a cross-lingual LLM (x-LLaMa) using cross-lingual instruction datasets and create multilingual LLMs (m-LLaMa) based on mixtures of training resources, including parallel corpora. The objective is to enhance the semantic alignment across multiple languages, thus improving the performance of a pre-trained LLM in non-English languages. Experimental results show that x-LLaMa and m-LLaMa achieve better performance on non-English tasks and translation tasks than previous LLaMa-based models. Ablation studies were conducted to demonstrate the effectiveness of the proposed strategy, including the use of different scales of data for tuning the models. Both the results and analyses reveal interesting findings for LLMs.

### Strengths
1.	This paper systematically presents a study demonstrating the use of multi-lingual instruction datasets along with multi-lingual parallel datasets to fine-tune pre-trained LLMs for non-English languages, achieving comparable or even better performance than previous models.
2.	The findings and conclusions drawn from this study provide valuable insights to the community, indicating that using translation data can enhance language abilities beyond English.
3.	Overall, the paper is well-organized and well-written, making it easy to follow.

### Weaknesses
1.	One of the contributions of the paper is the proposal of the scaling law to quantify the relationship between translation performance and the size of the training data. However, there is a lack of insightful discussions and analyses regarding this aspect. For instance, Eq. (2) considers the language similarity between English and the target language, but unfortunately, the computational details are not provided or explained, which makes it difficult to replicate the work.
2.	Another concern is that the authors only verify their proposal using the LLaMa-7B model. It remains unclear if the conclusion still holds for other LLMs, such as Bloom.

### Questions
1.	How is the similarity of language computed, and how are the values of α and β estimated for the scaling law?
2.	In Table 4, it is unclear how the optimal allocation of data is obtained. It would be helpful to provide further elaboration on this.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper effectively achieves cross-lingual transfer by employing fine-tuning techniques on LLMs using a parallel corpus. The authors introduce two distinct forms of training data, specifically a parallel corpus and a translation corpus, which are employed to fine-tune the model. Subsequently, the model is evaluated on the XQuAD and MLQA datasets. The experimental results demonstrate the effectiveness of their approach in tackling these tasks.

### Strengths
1. The collected parallel corpus in different languages is invaluable and can greatly contribute to future research.

2. The performance of the tested tasks and languages has shown remarkable improvement.

### Weaknesses
1. The method employed lacks novelty, as it primarily involves the collection of a parallel corpus and subsequent fine-tuning of the model.

2. The method lacks testing on high-resource languages closely related to English, such as French or German, as well as on low-resource languages like Thai or Swahili.

3. The method has not been sufficiently tested on multilingual NLP tasks, including reasoning tasks such as MGSM and XCOPA, as well as NLG tasks like XLSum.

4. The tested results on Flores are not convincing, especially when considering that the model has primarily been fine-tuned for translation tasks.  Additionally, for the translation training data, please refer to the following question 3.

### Questions
1. Is the translator engine effectively aligned with human-annotated data in terms of quality? Furthermore, have any techniques been implemented to ensure the removal or filtration of irrelevant or meaningless data from the training process?

2. In the CoIT example depicted in Figure 1, it is notable that the Chinese parallel corpus still includes English phrases such as 'Instruction:' and 'Input:'. It would be beneficial to analyze to assess the significance of these prompts and their impact on the overall performance. 

3. Regarding the two types of training data, given the availability of a parallel corpus, it is worth investigating whether the translation corpus truly has a significant impact. Conducting an ablation analysis could provide valuable insights in this regard. Furthermore, expanding on this point would contribute to a more comprehensive understanding of the overall training process.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
In this paper, the authors propose a method for multilingual instruction tuning. The approach relies on jointly tuning parallel corpora and translated instruction-tuning data. The model is compared with other public multilingual instruction-tuned models. However, due to many lacunae in the experimental setup (described later) the paper cannot establish that the proposed approach is better than previously proposed approaches. The paper also introduces a multilingual benchmark (MI-EVAL) which is an automatically translated version of ALPACA (the automatic translation is a major limitation of this dataset).

### Strengths
* The paper studies joint finetuning on parallel corpora and translated instruction data. Though this setting has been studied in previous work (Parrot), this paper makes an attempt to study various possibilities. 
* Ablation studies comparing monolingual pre-training, translation data instruction tuning, and multilingual task instruction tuning. These establish that tuning on translation data and translated instruction data is useful. However, the analysis still has some unanswered questions which are mentioned in the Weaknesses section.

### Weaknesses
* The proposed model is compared primarily with Bayling/Chinese Alpaca. Chinese Alpaca is trained only on Chinese. Hence, its results in Chinese are better than the proposed model, whereas it underperforms on other languages. On the other hand, Bayling is finetuned on 4 European languages, while this paper evaluates other languages. This evaluation setup is not a fair comparison to establish that the proposed approach is better than the previous work. This does not clearly answer any research question. Why not compare with truly multilingual models like Bactrian-X (or finetune alternative models to study various experimental configurations as described next).
* To understand how/if the proposed approach is indeed better, the following additional ablations would help. Is finetuning on English IFT data (Aplaca-En) plus translation data (En-Zh) sufficient to achieve crosslingual transfer? Is cross-lingual instruction tuning necessary? 
* To establish the generalization of these results, the ablation results should be reported on all languages considered, not just Chinese. 
* The research questions the paper seeks to answer are not presented with clarity. How do the ablations answer all those questions?  In its current form, the paper proposed a model but does not clearly articulate how this model improves over current work. The experimental setup also doesn’t lend itself to answering the research questions clearly as mentioned earlier. 
* With only limited exposure to foreign languages (via instruction tuning), how can the model achieve enough proficiency to generate fluent target languages? 
* It is known that a high-quality parallel corpus is needed for finetuning. It is also known that WikiMatrix is noisy. Given that there are many high-quality corpora available, why should they not be used for translation data finetuning? 
* The MI-Eval has been created via automatic translation. How good is the translation quality, and what is its impact on evaluation? Results on an error-prone translated dataset with no quality evaluation are not sufficient to measure model capabilities. Moreover, ChatGPT has been used for evaluation. How good is ChatGPT for the evaluation of open-ended tasks for non-English languages? There is no evidence of the efficacy of this evaluation methodology.

### Questions
* The paper mentions that the performance on English tasks is not impacted, and an example is provided. It would be good to report performance on English QA benchmarks before and after finetuning with cross-lingual data. 
* MI-EVAL: The English side looks like a replication of ALPACA. Why was ALPACA not directly used prior to translation?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor
