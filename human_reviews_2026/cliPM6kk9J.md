# QAProt: Enabling Sequence-to-Text Protein Function Learning with a Comprehensive QA Corpus

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
Inferring protein function from sequence is a grand challenge in genomics, yet progress is bottlenecked by the narrow, template-driven datasets available for training. These datasets, derived from structured databases, fail to leverage the rich diversity of knowledge in scientific literature. To address this gap, we introduce **QAProt**, a large-scale corpus with over 987,000 free-form question-answer pairs mined directly from PubMed abstracts, capturing broader topical and linguistic variability than existing resources. To ensure high fidelity, we developed a rigorous multi-LLM cleaning pipeline that yields a 13 times reduction in estimated hallucination rates. Our analyses reveal that current protein LLMs exhibit a performance collapse when tested on the realistic distribution of taxa and functions found in QAProt, highlighting the complementary nature of our literature-derived data distribution. A single epoch of fine-tuning on our dataset yields remarkable improvements, including an 86% performance gain on previously unseen protein domains. QAProt is a complementary new resource that enables the development of more powerful, generalizable models for protein science. Dataset available anonymously at https://huggingface.co/conferenceacc/QAProt.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors introduce QAProt, a large dataset of questions about specific proteins derived from PubMed abstracts. They describe a multi-stage pipeline using LLMs to generate questions, match them with protein sequences, and perform quality control. They show that existing specialized language models for protein Q&A fail on the resulting questions, but also that the same models are substantially improved by minimal finetuning thereon.

### Strengths
The dataset is large, diverse, and well-motivated; work like this could help to bridge the divide between existing multimodal language models, which for reasons of data availability are currently limited to text, images, video, and audio, and biological modalities.

The authors seem attentive to data quality, and I appreciate the steps taken to remove spurious questions.

### Weaknesses
There are a few major weaknesses that need to be resolved before I can recommend accepting this manuscript:

1. Based on the examples provided in the paper and my (albeit brief) examination of the data files, there seems to be a mismatch between the author's motivations and the final product; QAProt is not a true "sequence-to-text" dataset. For example, it would be a stretch to say that the answers to the questions in Figure 5 (mostly about phenotypic outcomes) can be derived from the corresponding sequence at all. Likewise, many other questions in the set mostly appear to test knowledge of the biomedical literature and not protein sequence understanding. It's not even clear to me based on the manuscript that the language models under evaluation are presented with the protein sequence at all. I think the dataset needs another split that excludes questions not answerable from the corresponding sequence alone.
2. The dataset needs deduplicating. I personally found tens of copies of several questions (especially basic ones like "What is the function of [x protein]?"), together comprising a large fraction of the corpus.
3. It does not appear that the authors have taken enough care to minimize leakage between train and validation splits; as far as I can tell, the choice of which proteins to hold out did not take sequence homology into account.
4. While substantial improvements are observed when the baseline models are fine-tuned on QAProt, it's unclear based on the results in the paper how much of that can simply be attributed to format adaptation. It would be good to include additional baselines where e.g. questions from the original training domains of these models are reformatted in the style of QAProt.

### Questions
See above.

### Soundness
2

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
4

### Summary
The authors used LMs to extract question-answer pairs about proteins' structure+function from the abstracts of scientific papers. This new dataset is highly diverse and can be used for training and benchmarking predictive models. The paper introduces the extraction method, describes the dataset, and presents some benchmarking results.

### Strengths
The paper provides a new, valuable resource to the community. 

The general direction of extracting semi-structured data from papers is a powerful framework that modern LMs have enabled. This work is a good example of the framework.

There are a number of well-executed technical details, such as how hallucinations are mitigated.

The paper is frank about weaknesses of the work and what could be improved.

### Weaknesses
The primary contribution of the paper is a new, interesting dataset. However, I didn't find the exposition on the composition of this dataset adequate. The paper should include some examples from it, a distribution of the categories depicted as colors in Fig 2A, and a discussion of question difficulty. 

I found the evaluation setup confusing. Is the bleu metric adequate for measuring accuracy of this sort of question answering? I'm concerned that the various models, particularly ones trained on templated data, can output things that are semantically correct but that have high bleu from the target text. I would have found it to be more reliable if you had formulated the questions as multiple choice or fill-in-the blank. Is there a precedent for using bleu in modern QA papers? I would have trusted an LM autorater more.

### Questions
See my above question about bleu. Can you provide an argument that this is an adequate metric, particularly when the models tend to provide templated outputs? 

I found Table 5 confusing, since it doesn't provide any comparisons. How should I be interpreting this result? 

I feel that the experiments are convolving two things: (1) predicting information about proteins and (2) formulating free-text responses. What if you used a model that gives structured outputs, such as ProTrek, and then had an off-the-shelf LM use this structured output + the question to formulate a free-text response?

 There is a large range of types of questions, yet there is no analysis of models' performance based on the question type. Some questions, such as mutation effect prediction, are likely significantly more difficult than others. Can you do some analysis where you present per-question-type metrics?

I found this comment far too informal: "The results show that
QAProt clearly outperforms the others in both semantic/topic coverage and lexical richness(Figure 2)." Is there a way to make this more rigorous?

It was unclear to me how the new data is qualitatively different from prior datasets. I understand the point about templating, but this is a superficial detail that concerns how concepts are presented, not the underlying information. In what sense is your data, when ignoring templating, qualitatively different from prior datasets? If you had used a few source datasets, such as Brenda + Uniprot, could you have obtained a similar diversity of facts?

"Next, we retain abstracts that specifically discuss proteins and genes using UniProt
API. Given an abstract, we iterate over the words in the abstract, and for every word, we make an API call to UniProt to check whether the word is a protein name or not."
This seems highly inefficient. Why not feed the abstract into an LM to extract a few candidate names and then call the API just on these?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this paper, the authors introduce QAProt, a diverse dataset specifically designed to benchmark protein language models in the context of functional annotation. By sourcing questions from PubMed, the dataset aims to address a broader range of protein-related queries, beyond the narrow focus of traditional datasets like UniProt. The authors demonstrate that fine-tuning on QAProt for at least one epoch improves performance on the task of generating protein-to-text descriptions.

### Strengths
* The paper presents a detailed experimental setup, offering quantitative evaluations across various protein language models. The authors also provide insight into the LLM filtering pipeline and the prompting strategiesused to mitigate potential biases in the evaluation process. They show a clear concern for evaluation bias and the influence of question formulation during the fine-tuning process, which adds transparency to the methodology.
 
* The proposition of QAProt as a protein benchmark that incorporates data from PubMed abstracts, rather than relying solely on curated sources like UniProt, is a contribution. This approach presents an opportunity to move beyond the mainstream Gene Ontology (GO)annotations and expand the scope of functional annotation for proteins.

### Weaknesses
* While the authors provide valuable experimental results, the paper can be difficult to follow at times, particularly when referencing tables. There are instances where the discussion jumps between tables without clear transitions or explicit mention of table numbers in the text. The current referencing style may confuse readers and make it challenging to track the progression of the argument.
 
* A notable limitation is the lack of human expert evaluation. While the authors employ various filtration techniques and model-based assessments, protein language models are still prone to hallucination or generating biologically incorrect information. The absence of statistical validation or human expert assessments, from biologists or domain experts, raises concerns about the biological accuracy and real-world applicability of the model's predictions.
 
* The experimental setup and comparisons between models could benefit from more depth. While the paper outlines the evaluations and presents the results, the analysis of these results remains relatively superficial.

### Questions
* Does the paper include any human expert evaluations to assess the biological relevance and real-world applicability of the model’s predictions? Given the complexity of protein function annotation, would human expert assessments provide valuable qualitative insights that are missing from the current quantitative evaluations (e.g., accuracy, recall)?
 
* How does the paper ensure that the proteins used to generate functional questions are sufficiently biologically similar or relevant to one another? The paper does not provide a clear methodology for confirming protein similarity when generating these questions, which raises concerns about the biological accuracy of the generated queries. Can the authors provide more transparency about the process used to group or pair proteins for this task?
 
* How does the paper handle the train-test data split, particularly with regard to sequence similarity between proteins in the training and test sets? Is there any consideration of data leakage, such as if test proteins are too similar to training proteins, which could artificially inflate performance metrics? Could the authors clarify the methods used to ensure that training and test proteins are sufficiently dissimilar, or explain how similarity is managed in this context?
 
* Are the clusters formed biologically interpretable? Do proteins within each cluster share meaningful functional or structural similarities? How do the authors assess whether the clustering method truly captures biologically relevant structures? What evaluation metrics were used to assess the quality of the protein question clusters formed based on embeddings?
 
* The paper references Table 4 when discussing the results, but the results are actually presented in Table 8 in the appendix. Could the authors clarify the reference to Table 4 and ensure consistency in the presentation of the results to avoid confusion?
 
* Given the distribution shift between the training data of the baseline models (such as those trained on UniProt/SwissProt) and the more diverse scope of the QAProt dataset, should it not be expected that the protein LLMs would exhibit reduced performance on QAProt, as indicated in Table 4? Additionally, the paper does not mention whether any sequence similarity between the proteins in the original test sets and those in the QAProt test set was considered to mitigate the distribution shift and ensure a fairer comparison. Was any effort made to match sequence similarity or relevance between the test sets to reduce the gap and make the comparison more equitable?

### Soundness
2

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
4

### Summary
The authors propose a new benchmark QAProt for understanding the relationship between protein sequence and function, an important challenge in biology. Unlike existing datasets the authors mine more open-ended question-answer pairs from the literature thus capturing a broader definition of function than that can be found in structured schemas. 

The authors first gather a set of abstracts that discuss protein/gene names. They then use an LLM to generate question-answer pairs from the abstract. They then apply LLMs to filter out examples with hallucinations. 

The authors then benchmark several models with various automatic metrics in both the zero shot and fine-tuned setting, showing that while zero shot performance is poor, finetuning on the data helps significantly.

### Strengths
Connecting protein sequence to function is an important problem and mining data from the literature presents a very promising approach for dataset collection. Resources such as this one would be very valuable to the community and help move beyond structured schemas.

### Weaknesses
-It would be great to have a more rigorous human evaluation as BLEU etc is often not well correlated with human judgement. 

-Would be great to have some understanding of inter-annotator agreement with humans as above and/or for the LLM-as-judge models. 

-Would also be great to have more error analysis (e.g. examples that contain hallucinations but pass the cleaning step and/or model errors on the benchmark)

-There is some concern that since the question/answer pairs were mined from the literature (up to 2020), that they are included in the training data of many LLM models and thus models could potentially perform well at this task due to data contamination. Would be curious to see what the authors thoughts are about this.

### Questions
Please see above.

### Soundness
3

### Presentation
3

### Contribution
2
