# Retrieving Texts by Abstract Descriptions

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 5, 5, 5

## Abstract
While instruction-tuned Large Language Models (LLMs) excel at extracting information from text, they are not suitable for locating texts conforming to a given description in a large document collection (semantic retrieval). Similarity search over embedding vectors does allow to perform retrieval by query, but the similarity reflected in the embedding is ill-defined and non-consistent, and is sub-optimal for many use cases. What, then, is a good query representation for effective retrieval?

We identify the well defined and consistent task of retrieving sentences based on abstract descriptions of their content. We demonstrate the inadequacy of current text embeddings and propose an alternative model that significantly improves when used in standard nearest neighbor search. The model is trained using positive and negative pairs sourced through prompting a LLM. While it is easy to source the training material from an LLM, the retrieval task cannot be performed by the LLM directly. This demonstrates that data from LLMs can be used not only for distilling more efficient specialized models than the original LLM, but also for creating new capabilities not immediately possible using the original model.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper addresses the challenge of locating texts in a large document collection based on abstract descriptions of their content. The authors argue that current text embeddings and semantic search solutions are inadequate for this task, as they lack a well-defined notion of similarity. They propose a new model that significantly improves retrieval by utilizing a consistent and well-defined similarity based on abstract descriptions. The model is trained using positive and negative pairs sourced through prompting a LLM. The authors highlight the limitations of existing search techniques, including keyword-based retrieval, dense similarity retrieval, QA-trained dense retrieval, and query-trained dense retrieval. They emphasize the need for a specific type of similarity, referred to as description-based similarity, which captures the relation between abstract descriptions and concrete instances within documents. They demonstrate the effectiveness of their proposed model in retrieving relevant texts based on abstract descriptions and suggest that their approach can enhance knowledge discovery in various data-intensive domains, including legal, medical, and scientific research. Overall, the paper emphasizes the importance of a well-defined similarity measure for effective semantic search and presents a novel approach that leverages the strengths of LLMs to achieve a retrieval task that is not feasible using traditional text generation capabilities.

### Strengths
They evaluate the effectiveness of their proposed "Abstract-sim" model in sentence retrieval based on abstract descriptions, comparing it with several baseline models. The evaluation includes both human and automatic evaluations. For the human evaluation, the researchers conducted a crowd-sourced evaluation of retrieval performance for 201 random descriptions, comparing their model with several strong sentence encoder models. The results of the human evaluation indicate that the "Abstract-sim" model outperforms the baselines significantly, with an average of close to 4 out of 5 sentences deemed relevant for the query, while the baseline models had significantly lower performance, ranging between 1.61 to 2.2 sentences. The automatic evaluation was carried out to assess the model's robustness to misleading results. The authors generated a dataset of valid and invalid descriptions, and their model demonstrated superior performance in terms of precision at various retrieval points, with the largest disparity observed at precision@1. Their modell achieved a precision@1 score of 85%, compared to 7~3% for the strongest baseline model. The paper emphasizes the potential of leveraging large language models for generating tailored training datasets, despite their limitations in direct retrieval tasks. Their results indicate that the proposed model, trained on a dataset specifically tailored to the task, performs significantly better than standard sentence-similarity models.

### Weaknesses
lack of comparisons with state of the art retrieval models and neural rankings. 

Guo, Jiafeng, et al. "A deep look into neural ranking models for information retrieval." Information Processing & Management 57.6 (2020): 102067.

Mitra, Bhaskar, and Nick Craswell. "Neural models for information retrieval." arXiv preprint arXiv:1705.01509 (2017).

### Questions
I think the baselines for their algorithms are pretty simple

### Soundness
2 fair

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
This paper introduces a novel retrieval task designed to fine sentences that exemplify the ``instance-of'' property related to a given query. To achieve this, the paper constructs a dataset using a large language model and utilizes this dataset to develop a dense retrieval model. Experimental results from a manually constructed dataset demonstrate that the proposed dense retriever outperforms baseline models.

### Strengths
The paper effectively delineates the problem at hand by highlighting its distinction from existing research. This clear exposition aids readers in comprehending the subject matter. Furthermore, the employment of crowd-workers to curate a new retrieval dataset is commendable, as it promises to significantly benefit future search research.

### Weaknesses
The main concerns regarding this paper are:

- While the paper emphasizes that the retrieval based on description-based similarity is different from the existing retrieval, the description-based similarity also belongs to the similarity between texts. Existing methods measure this text similarity through learning from query-document (sentence) pairs, while the proposed method learns from query (description)-sentence pairs. Thus, from this viewpoint, the proposed approach seems to address a specific instance of text-to-text similarity rather than introducing a fundamentally new form of similarity-based search.

- The retrieval in this paper appears specialized in a specific domain and not universally applicable. It would be better to explain in detail the actual application that requires this proposed retrieval.

- While the paper innovates by introducing a new dataset, the retriever itself lacks novelty. Essentially, it is the same as the existing methods that train encoders, previously used in dense retrieval, and subsequently use nearest-neighbor search techniques.

- Recent dense retrieval research has seen the emergence of diverse encoders and similarity techniques, such as Colbert and PLAID. It's necessary for this paper to evaluate the efficacy of its proposed method by incorporating a variety of encoders and similarity metrics in the experiments.

### Questions
Q1: Generating data with GPT often leads to the issue of hallucination. How was this tackled in this study?

### Soundness
2 fair

### Presentation
3 good

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
This paper posits that the similarity reflected in embeddings is often ill-defined and inconsistent, which can be suboptimal for various practical use cases. To address this issue, the paper adopts a novel approach. It leverages off-the-shelf large language models to generate multiple descriptions for a given document. Subsequently, it conducts sentence retrieval tasks based on these descriptions to enhance the retrieval task's ability to capture abstract semantic information. As no suitable public dataset is available for the new retrieval settings, the authors introduce a new dataset for training and evaluating their model. The results on this proposed dataset indicate that the trained model outperforms the baselines in both human evaluations and automatic assessments.

### Strengths
1. This paper offers a fresh perspective on the traditional retrieval task, highlighting the limitations of term-based and vector-based matching approaches. It introduces a novel description-based matching approach and enumerates its advantages over these traditional methods.

2. To validate the effectiveness of the proposed method, the authors construct a new dataset based on descriptions using Wiki data. They employ off-the-shelf large language models for extensive data collection and annotation, underscoring the rigor and comprehensiveness of their approach.

3. The paper is exceptionally well-written, ensuring that it is easily accessible and comprehensible for readers, making it a valuable contribution to the field.

### Weaknesses
1. I totally agree that the term-based and vector-based retrieval frameworks are not perfect and may lead to some problem in practice. However, I wonder that is it really a new of the proposed description-based framework, because as mentioned in the paper, author just modify the dataset and change the meaning of relevance. Moreover, the model used in the paper is also vector-based method.


2. Using large language model to generate training dataset is risky in two folds. Firstly, it may not cover all the aspects of a given document form the generated descriptions, so that it may missing some information of the document. Second, it may also contain duplicate aspects of one document so that after model training, some aspects will be strength or biased by the data.

3. It may not a fair comparison between proposed method and baselines. As mentioned in Weakness 1, the meaning of relevance is changed. The proposed method train and evaluate on the same data distribution is evidently better than the model test on OOD distribution.

### Questions
1. It is an interesting paper that expand the view of relevance. However, the major concern is that the formulation of description-based framework is also weak and lack some theoretical support, which is the same as the vector-based one. I think how to formulate the description-based relevance is the vital problem in the next version.
2. How to make sure abstract description is what we actually need in practical search? Some statistic study may be involved as former evidence.
3. The comparison of proposed method and other baselines should be fairer. Furthermore, some strong dense retrieval baselines should also be involved in the experiments.
 - ANCE: Approximate nearest neighbor negative contrastive learning for dense text retrieval
 - BERM: BERM: Training the Balanced and Extractable Representation for Matching to Improve Generalization Ability of Dense Retrieval
 - TAS-B: Efficiently teaching an effective dense retriever with balanced topic aware sampling.
 - Contriever: Unsupervised dense information retrieval with contrastive learning.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The authors in this paper propose a different search approach by retrieving sentences based on abstract descriptions of their content.  The authors demonstrate the shortcomings of the current methods of text embeddings and propose a metho to improve them. The authors created a dataset using LLMs to capture the notion for similarity and use the same to train an encoder whose representations are better than the state-of-the-art. Specifically, the authors used GPT-3 to generate positive and misleading descriptions for sentences from the English Wikipedia dataset. The authors utilize a pretrained sentence embedding model and fine-tune it with contrastive learning to train their model for the task of aligning sentences with their descriptions. They use two encoders – one as a sentence encoder and the other as a description encoder. Limitations of the approach were not discussed in the paper.

### Strengths
1. The authors propose a novel approach to generate abstracts instead of the regular search methods. 
2. The authors have used both human evaluation and automatic evaluation to evaluate the proposed model.

### Weaknesses
1. In the abstract, did you mean “inconsistent” instead of “non-consistent”? In the Introduction, "This make the” --> “This makes the”, "well defined” --> “well-defined”. There are several such grammatical errors, and it would benefit the authors to run the text through any of the free grammar tools available. Also, the authors can recheck the camel cases of sentences and sub-headers (full stop or no full stop?).
2. What are the different use cases of the proposed description-based search in documents? The authors can discuss some different case studies or use cases to convince readers.

### Questions
-

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 5

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper defines a new task that retrieves text based on abstract descriptions. The specific kind of similarity between text and abstract description are defined and hand curated examples were used in the instructions to LLM to generate training data. The proposed method works better than other sentence/text retrievers trained with the general definition of sentence similarity on the test data designed for this task.

### Strengths
The text and abstract description similarity is a very interesting type of similarity and would value the information retrieval field. I think the strength of the paper is to design the prompts to gather the text/description pairs that satisfy the definition.

### Weaknesses
The paper proposed an interesting new task. I'm confident it will be useful for some application or existing retrieval applications. However, the paper didn't explore what will benefit from this new task as much.

Also, I would think it is pretty straight forward to see that the proposed method would outperform a general purpose retrieval or sentence similarity model. Those method are not trained or finetuned using the same training data, which defines the relationship of sentence and its abstraction.

### Questions
Questions:
How is precision @k decreases when k is increasing, especially for the proposed method?

Suggestions:
I think this is a interesting task with value, but I think it is worth to explore what end task would be benefit from this new task, or how this task post challenges to existing retrieval models, if any.

Minor typos:
  - page 8. Settings. "invalid-recall@k" is missing the @ sign.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
