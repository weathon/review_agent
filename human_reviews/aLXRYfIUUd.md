# SCALE: Scaling up the Complexity for Advanced Language Model Evaluation

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 5, 6

## Abstract
Recent strides in Large Language Models (LLMs) have saturated many NLP benchmarks (even professional domain-specific ones), emphasizing the need for more challenging ones to properly assess LLM capabilities. In this work, we introduce a novel NLP benchmark that poses challenges to current LLMs across four key dimensions: processing long documents (up to 50K tokens), using domain-specific knowledge (embodied in legal texts), multilingual understanding (covering five languages), and multitasking (comprising legal document-to-document Information Retrieval, Court View Generation, Leading Decision Summarization, Citation Extraction, and eight challenging Text Classification tasks). Our benchmark contains diverse legal NLP datasets from the Swiss legal system, allowing for a comprehensive study of the underlying non-English, inherently multilingual, federal legal system. Despite recent advances, efficient processing of long documents for intense review/analysis tasks remains an open challenge for LLMs. In addition, comprehensive, domain-specific benchmarks requiring high expertise to develop are rare, as are multilingual benchmarks. This scarcity underscores our contribution's value, considering that most public models are trained predominantly on English corpora, while other languages remain understudied, particularly for practical domain-specific NLP tasks. Our benchmark allows for testing and advancing the state-of-the-art LLMs. As part of our study, we evaluate several pre-trained multilingual language models on our benchmark to establish strong baselines as a point of reference. Despite the large size of our datasets (tens to hundreds of thousands of examples), existing publicly available models struggle with most tasks, even after extensive in-domain pre-training. We publish all resources (benchmark suite, pre-trained models, code) under a fully permissive open CC BY-SA license.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work introduces a benchmark called SCALE, which consists of seven legal tasks sourced from the Swiss legal system.  On this benchmark, the authors evaluate a wide range of LLMs, including black-box models, open-source models, and the one tuned via in-domain data.

### Strengths
1. the dataset is unique and definitely interesting to the LLM community and people in related domains.
2. it includes multiple distinct and challenging tasks
3. The experiments are extensive and cover many recent models

### Weaknesses
1. The title is misleading, as a dataset for a specialized domain, the title should state its scope clearly. I think it is of great importance for a serious research paper to have appropriate title.
2. Related to the above scope issue, in the abstract (as well as the main body of the paper), the authors state that we need more challenging tasks for LLM, then why legal tasks specifically? To accomplish the goal of proposing more challenging tasks, why not use data from domains like finance, medical, etc?
3. This work includes extensive experiments, but it could be better to include more analysis, discussion and takeaways

### Questions
Citation format in the first sentence of section 5 is incorrect.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes to introduce a benchmark of model performance across classification, text generation & information retrieval tasks on legal datasets. The datasets consist of long documents that are multilingual in nature. The models benchmarked in the study include LLMs and some (smaller)  models such as mT5 and XLM-R that were fine-tuned on domain-specific data. The results demonstrate the variability in performance that LLMs yield for different tasks.

### Strengths
The paper is clearly written and highlights the variability in performance of LLMs across tasks, especially when applied to domains that they may not have been exposed to during training. It is interesting to see, for example, that smaller fine-tuned models (XLM-R or RoBERTa based models) outperform off-the-shelf LLMs for text classification tasks across all the legal datasets used in this paper.
Additionally, the experiments were detailed and clearly explained. 
Moreover, the inclusion of the observant ethics statement is highly commendable.

### Weaknesses
Although construction of a benchmark for better evaluation of LLMs along specific dimensions is of importance, I was unable to determine what the novelty of this work is, w.r.t. other already existing NLP benchmarks. Benchmarks for LLM evaluation already exist both for the legal domain [e.g. LEXTREME (https://arxiv.org/pdf/2301.13126.pdf), LexGLUE (https://aclanthology.org/2022.acl-long.297.pdf), LegalBench (https://arxiv.org/pdf/2308.11462v1.pdf) etc.] and otherwise [BigBench (https://arxiv.org/pdf/2206.04615.pdf), HELM (https://arxiv.org/pdf/2211.09110.pdf), etc]. The current work expands on these benchmarks in terms of including legal datasets specific to the Swiss legal system, which do not meet the standards of an ICLR paper, in my opinion.
Further, if the focus is on evaluating LLMs, it would be important to include more LLMs in the zero-shot & one-shot settings. These could include Falcon, Flan-T5 XXL, Alpaca, Vicuna etc. This would allow for a wider coverage of the behavior of LLMs on the tasks at hand.

### Questions
It would be great if you could highlight the key, novel contributions of the paper in comparison to the already existing benchmarks for LLMs.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This study introduces a comprehensive evaluation dataset for large language models (LLMs) focusing on the legal domain. The dataset is sourced from Swiss legal documents and comprises seven multilingual datasets covering four key dimensions: long documents, specificity to the Swiss legal domain, multilinguality, and multitasking. The authors further conduct in-domain pretraining to develop Legal-Swiss-RoBERT and Legal-Swiss-LF models specifically tailored for this domain. The proposed evaluation includes seven tasks (LAP, CP, IR, CVG, JP, CP, and LDS) forming a testbed to assess the performance of existing LLMs in the legal domain.

### Strengths
1. This study introduces an evaluation dataset specifically designed to assess the performance of large language models (LLMs) in the legal domain. The dataset emphasizes 4 challenging dimensions that pose difficulties for LLMs, thereby providing a comprehensive and rigorous evaluation for LLMs operating within legal fields.

2. The quality of the research is substantiated by rigorous experimental design. The authors conducted experiments to examine and analyzed the performance of existing pre-trained language models (LMs) in legal fields. Furthermore, the authors showcased the significance and worth of their collected dataset by fine-tuning LMs on it. The multilingual nature of the dataset adds an additional layer of complexity to the evaluation of language models.

### Weaknesses
About the experimental setup: it is advisable to expand the inclusion of additional existing large language models (LLMs) in the experiments. Given the lengthy nature of legal documents, it would be beneficial to evaluate the performance of LLMs specifically pre-trained for handling long contexts. It is recommended to conduct more comparisons with LLMs specifically designed to handle long contexts. Furthermore, it appears inequitable to compare models with input length restrictions imposed by fixed-sized tokens. It is also desirable to extend the proposed method to encompass a wider range of LLMs.

### Questions
1. Please address the weaknesses above. 

2. In text classification tasks, the models are provided with facts and considerations explicitly written by legal professionals such as lawyers or judges. This simplification reduces the evaluation complexity of language models (LMs). What are the underlying reasons for adopting this simplified approach and reducing the complexity in LM evaluation?

3. In Table 3, the majority of models exhibit superior performance on the "-C" datasets compared to the "-F" datasets. However, BLOOM, Legal-Swiss-RoBERTa_{Large}, and Legal-Swiss-LF_{Base} demonstrate relatively poorer performance. What could be the potential reasons behind the clearly weaker performance of these models?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces a new multilingual legal NLP benchmark dataset, named SCALE. The characteristics of this dataset are multilingual (German, French, Italian, Romansh, and English), long documents, and multitasking. The origin of SCALE is Swiss legal documents, and the authors arrange the raw data into several text classification and generation tasks. The authors also show the performance of large language models and it shows that the benchmark is still challenging for the NLP community.

### Strengths
The primary strength of this paper lies in proposing a challenging and extensible benchmark dataset. It is helpful for AI and NLP researchers who are interested in the tasks.

### Weaknesses
The benchmark is quite interesting and sound. But I have some curiosity which I mention in the Questions. I hope to listen to the author's responses.

### Questions
- For each task, what is human (legal experts and non-experts) performance, especially NLG tasks? Other benchmark tasks show the human performance for the tasks that can be the upper bound or challenge for AI models.
- How each task helps in the legal domain. For example, if you have an AI model that is good at law area prediction, how would it help?
- Presentation: it would be better to enlarge the font size of captions. Now it is hard to read.

### Soundness
4 excellent

### Presentation
3 good

### Contribution
3 good
