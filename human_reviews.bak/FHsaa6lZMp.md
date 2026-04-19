# Fine-Grained Machine-Generated Text Detection

- Decision: Reject
- Scores: 5, 3, 5, 5

## Abstract
Machine-Generated Text (MGT) detection identifies whether a given text is human-written or machine-generated. However, this can result in detectors that would flag paraphrased or translated text as machine-generated. Fine-grained classification that separates the different types of machine text is valuable in real-world applications, as different types of MGT convey distinct implications. For example, machine-generated articles are more likely to contain misinformation, whereas paraphrased and translated texts may improve understanding of human-written text. Despite this benefit, existing studies consider this a binary classification task, either overlooking machine-paraphrased and machine-translated text entirely or simply grouping all machine-processed text into one category.  To address this shortcoming, this paper provides an in-depth study of fine-grained MGT detection, categorizing input text into four classes: human-written, machine-generated, machine-paraphrased, and machine-translated. A key challenge is the performance drop on out-of-domain texts due to the variability in text generators, especially for translated or paraphrased text. We introduce a RoBERTa-based Mixture of Detectors (RoBERTa-MoD), which leverages multiple domain-optimized detectors for more robust and generalized performance. We offer theoretical proof that our method outperforms a single detector, and experimental findings demonstrate a 5--9\% improvement in mean Average Precision (mAP) over prior work on six diverse datasets: GoodNews, VisualNews, WikiText, Essay, WP, and Reuters. Our code and data will be publicly released upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper investigates fine-grained MGT detection; the authors propose to categorize an input text into four classes: human-written, machine-generated, machine-paraphrased, and machine-translated. They introduce a data preparation process to generate articles across different fine-grained categories, enabling the automatic creation of training and evaluation data for the task. The paper introduces a RoBERTa-based Mixture of Detectors (RoBERTa-MoD) for fine-grained MGT detection, which leverages multiple domain-optimized detectors for more robust and generalized performance. The paper presents theoretical proof that the method outperforms a single detector, and experimental findings show an improvement in mAP over prior work on six various datasets: GoodNews, VisualNews, WikiText, Essay, WP, and Reuters.

### Strengths
— The idea of separating into three categories is intuitive and transferred from the actual use cases. However, it makes the task more complicated as the generation style is similar for paraphrasing/translating/generation using the same models.

— It is beneficial for nature to theoretically assert if some models and approaches are worse or better without extensive model training.

— Reproducibility statements, limitations, and ethical considerations are included. Clear contribution. The code and data will be publicly released upon acceptance.

### Weaknesses
Some questions for reproducibility of the research.

— Please include information about the strategy of round-trip translation and paraphrasing. The rationale behind round-trip translation is not discussed. Was the translation performed for each language using all the models? 
Are the authors certain that the prompt "Paraphrase/Translate the following article: x." was used consistently across all models? If so, how does LLAMA determine which language to translate? What are the specific languages used? 
How do authors gather paraphrases and translations? For each article, there should be a minimum of four different translations or paraphrases. Have the authors confirmed that removing the first sentence from the text does not alter the meaning of the translation? 

— For the training data, have the authors saved the model's proportion distribution? No statistics for the training corpora. How many examples were used for each class? 
Section 3.1 needs more details.

— Line 372  `All methods were fine-tuned on data from Llama-3 (Touvron et al., 2023) and Qwen-1.5 (Bai et al., 2023) and then evaluated on all LLMs.`
Which data? How much data in which format? Maybe the authors mean: "evaluated on the data from all the LLMs"? I guess the methodology was to check in out-of-domain evaluation so that the data formed with different models and not evaluated by the same models in an LM-as_judge manner. The formulations are confusing. It is not clear from the texts what data you trained detectors, fine-tuned which exact models, and with what models you evaluated what.

`Llama-3 and Qwen-1.5 are in-domain generators for training the detector, and StableLM-2, ChatGLM-3, and Qwen-2.5 are out-of-domain generators to evaluate the model’s generalization ability.`
Am I right that, based on the data from models StableLM-2, ChatGLM-3, and Qwen-2.5, were the detectors not trained?

`However, in the same dataset, human-written articles in the training and testing sets may follow similar data distributions`
There is no information on whether they may or may not.

— Line 325: `LLM-DetectAIve is directly trained on fine-grained MGT data, which can be considered as a fine-tuned RoBERTa.`
Why should it be considered? No explanations/justifications

— Figure 5. It seems like the Qualitative Results are based on one example. Paraphrasing can change factual information as well as translation. The paper needs some quantitative metrics to catch it.

### Questions
— The translated and paraphrased texts, if created using LLMs, can also contain misinformation and factual errors. It depends on the LLM; the percentage of errors is much rarer than that of machine generation, but it still needs to be checked. 

— "As discussed in the Introduction," add the link to the Introduction section.

— Line 307: Set the spaces in cite like in:  `GoodNews(B`

— The paper would improve from the footnotes to direct links for the open datasets and a clear explanation of the steps with data processing.

— Table 1 would improve if the information about the model's size was added.

— It's a bit strange not to see the results section.

— The results would be interesting to check on the different lengths of the output. We could see the correlation with length.

— Factual error: LLM-DetectAIve distinguishes four categories: (i) human-written, (ii) machine-generated, (iii) machine-written, then machine-humanized, and (iv) human-written, then machine-polished. The idea is different from paraphrasing.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper addressing the detecting of machine generated text proposed to extend the traditional binary classification (machine or human generated) to a four-classes problem: human-authored, machine-generated, machine-translated or machine-paraphrased. The authors propose a multi-class classification model uses 4 Roberta-models and a gating mechanism. The whole network is trained in two stages

### Strengths
* An interesting twist to an important problem

* An analysis that combines both theoretical insights and empirical ablations

### Weaknesses
* Motivation: the idea behind a 4-class problem (instead of the more traditional 2-class) is motivated by the fact that machine-modified text (translated or corrected) might not be as harmful as text that is generated by a model from scratch. While this is a somehow appealing explanation, it would be interesting to see if that difference actually has an empirical impact. Sect 4.5 makes this attempts, by applying the proposed method on existing benchmarks, obtaining good scores (although not always outperforming the current state-of-the-art). More interesting to me would be an analysis on how `Binoculars` performs on the new datasets (considered as binary problem). Another interesting aspect would be to verify if the more fine-grained classification actually results in a better binary classification.

### Questions
* Could you address the weakness mentioned above?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper presents an in-depth study on fine-grained Machine-Generated Text (MGT) detection, which can classify text into four categories: human-written, machine-generated, machine-paraphrased, and machine-translated. The authors note that existing detectors struggle with out-of-domain text, particularly for translated or paraphrased text. To address this, they propose a RoBERTa-based Mixture of Detectors (RoBERTa-MoD) which uses multiple detectors optimized for different domains to improve performance. The authors provide a theoretical proof that their method outperforms a single detector and experiments show a 5-9% improvement in mean Average Precision (mAP) over previous work on six diverse datasets. The authors also introduce a data preparation process to generate articles across different fine-grained categories, enabling automatic creation of training and evaluation data for the task.

### Strengths
1) It presents an in-depth study on fine-grained Machine-Generated Text (MGT) detection, a topic that has been overlooked in previous studies. By classifying text into four categories (human-written, machine-generated, machine-paraphrased, and machine-translated), the research contributes significantly to the field. 
2) The paper introduces the RoBERTa-based Mixture of Detectors (RoBERTa-MoD), a novel method that uses multiple domain-optimized detectors for more robust and generalized performance. This method addresses the performance drop on out-of-domain texts, a key challenge in MGT detection. 
3) The paper's quality is evident in the theoretical proof provided for the method's superiority over a single detector. Additionally, the research is significant as it achieved a 5-9% improvement in mean Average Precision (mAP) over prior work on six diverse datasets.

### Weaknesses
1) The concept of employing a mixture of experts (MoE) for a task is not unusual, considering its extensive application in various tasks such as general LLM, summarization, and machine translation. While the application of MoE in text detection is relatively new, it is not a groundbreaking concept in terms of its fundamental idea. 
2) The study lacks an ablation analysis on MoD. Questions such as the influence of the number of detectors on the results, the effect of different corpus (domains) on detection, and the likelihood of the router specializing to specific detectors given an input article, remain unanswered. 
3) The paper doesn't delve deep into the confusion between classes, for instance, between translate and human/paraphrase. As the paper is centered on the fine-grained detection of machine-generated text, such analyses about the challenges of fine-grained detection are anticipated and would offer valuable insights to the readers. 
4) The experimental results may not have been compared in a fair manner. However, due to the lack of clear descriptions of the settings for the baselines, I will reserve my judgment until the rebuttal period, during which I expect a response.

### Questions
LN053: what is the special of your method compared to Abassy et al., 2024 given that they both do fine-grained MGT detection?

LN091: have you considered GPT models for the detection? What is the underlying reason for choosing RoBERTa? 

LN320: how do you calculate the AUROC for the 4-class classification problem?

Table 1: the experiments are limited to generations from open-source models. Have you considered latest closed-source models like GPT4, Gemini, and Claude3?

Table 1: In my view, a RoBERTa-single model trained on the same data is a demanded baseline to demonstrate the advantage of the MoE design.

Table 1: Are models like ChatGPT-D, RoBERTa-MPU, and LLM-DetectAIve trained on the same data as the proposed method?

LN427: How are your zero-shot experiments designed? When we say zero-shot, we refer to a method without any task specific training. How could your MoD method do the zero-shot given that it requires a joint training of the router and the detectors?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper presents novel task of fine-grained MTD detection, where the detector should be able to predict 4 labels: humn-generated, machine-generated, machine-translated and machine-paraphrased. Novel architecture for this task is roposed, refered as MoD (Mixture of Detectors), consisting several detectors for individual domains, and the trainable router. 

Theoretical results are presented demonstrating the theoretical benefits of the proposed architecture over the individual detector.

The evaluation is done on several popular MTD datasets for various generator models. Besides, the OOD evaluation is presented. The method outperformes  the most of the considered baselines by a large margin

### Strengths
- The paper proposes novel important tasks of fine-grained MTD. The presented analysis demonstrates that indeed Machine-Generated, Machine-Translated and Machine-Paraphrased texts have unique features and different usage scenarios, and it is important to distinguish them
- Novel architecture of MTD is proposed
- The evaluation is done of large amount of data domains and generator models, and includes OOD setup. The proposed method outperforms most of the considered baselines by a large margin.

### Weaknesses
- The theoretical framework in Sec 3.3 is not quite clear (see Questions)

- The benefit of the proposed method over standard Mixture-of-Expert is not clear
 
- In section 3.1, the statistics of the generated data is absent (see Q4)

- The most of the baseline models are Roberta-based classifiers. Comparing to the proposed method, they have k times less parameters, where k is the number of individual detectors. Only two baselines do not belong to this class: RoBERTa-MoS and Binoculars; comparing to them, the proposed method have marginal to no improvement

- No comparison to Roberta-MoS in OOD setup    

- In Sec. 4.3, the Qualitative result paragraph describes the properties of the dataset rather then the classifier results. There is no information about the typical errors, or any other qualitative description of the results of the proposed detector.

### Questions
Q1. Please explain the notation used in the Definition and Theorems in Sec. 3.3
- What is Patch? Does it correspond to a set of tokens, or a subset of features (e.g. coordinates) in the text embeddings, or smth else?
- What is feature vector $v_k$? The index indicates the whole cluster, but it is used for the description of the individual data point 
- What does the notation $y\alpha v_k$ mean? Is it a vector multiplied by 2 scalars $y$ and $\alpha$ ?

In general, could you please provide the example of the considered setup in the Machine-Generated Text Detector domain, defining patches, clusters, data features, distraction features and the noise.

Q2. How many detectors were used for each dataset? Does this number correspond to the theoretical bound from Theorem 2?

Q3. How are ChatGPT-D and Roberta-MPU atapted to fine-grained MTD setup? Are they fine-tuned on the same dataset as MoD? 

Q4. Please describe the statistics of the train/test dataset used in Tables 1 - 3 (its fine-grained part). For Generator models, please specify which version of each model is used (i.e. number of parameters)

### Soundness
2

### Presentation
2

### Contribution
3
