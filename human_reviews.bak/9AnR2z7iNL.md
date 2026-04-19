# DivKnowQA: Verifying the Reasoning Ability of LLM Through Open-Domain Question Answering Over Knowledge Base and Text

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 5

## Abstract
Open-domain complex question answering often breaks down a multi-hop question into single-hop questions, leveraging external knowledge for solutions. Current practices show a pronounced preference for unstructured texts, such as Wikipedia, often overlooking the potential of structured knowledge sources, such as WikiData. Additionally, while existing research has employed external tools to enhance the Large Language Model(LLM)’s capabilities, many tests have been conducted in artificial or toy scenarios. We argue that open-domain complex question answering presents a realistic and intricate challenge for LLM, necessitating the integration of external tools, including retrieval systems and knowledge base engines. In this paper, we present a new benchmark DIVKNOWQA to assess the LLMs’ reasoning skills and tool compatibility. Comprising 940 human-annotated intricate questions, DIVKNOWQA mandates both structured and unstructured knowledge for comprehensive answers. The subpar performance of prevailing SOTA methods, such as DSP and REACT, on our benchmark demonstrates its challenge. Moreover, we introduce our method DETLLM, which incorporates a symbolic language generation tool and a retrieval toolbox, pioneering a new approach to address this challenge. Our data and code will be released

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces DivKnowQA, a benchmark for open-domain question answering consisting of 940 human-annotated questions. These questions are sourced from external knowledge repositories like WikiData,  requiring models to use external information and engage in multi-step reasoning to provide answers. The dataset encompasses a wide range of domains, making it challenging for models to leverage external structured data effectively. In addition to presenting this dataset, the paper introduces a method called Diverse Retrieval Toolbox  Augmented LLM, abbreviated as DetLLM. DetLLM employs a textual passage retriever and a symbolic generation tool to supply external knowledge to the model, resulting in superior performance compared to other baseline methods.

### Strengths
As outlined in the summary, this paper makes a dual contribution. First, it introduces a dataset called DivKnowQA, which stands out for its diversity in question types spanning various domains. These questions necessitate multi-hop reasoning and typically require external knowledge to answer.

Additionally, alongside the dataset, the paper introduces DetLLM, a method based on a retrieval system and symbolic textual generation tools. This method has proven to be effective in harnessing structured knowledge to enhance the performance of Language Model Models (LLMs), outperforming the baseline methods.

### Weaknesses
1. Dataset Quality: A significant limitation of this paper pertains to the quality of the proposed dataset. DivKnowQA comprises only 940  question-answer pairs, a relatively small quantity even for benchmarking purposes. Moreover, because of the scarcity, the diversity of the questions is in doubt. Lastly, although these pairs undergo human verification, they originate from machine-generated sources, thus the questions may not be natural.  The authors could enhance the paper by including a table that compares the size of DivKnowQA with similar datasets, helping practitioners determine whether this dataset suits their needs.
2. Novelty of the Construction Process: The methodology employed to create  DivKnowQA bears a striking resemblance to that used for HotpotQA,  another dataset built on knowledge graphs and designed for multi-hop reasoning. Moreover, the analysis of the dataset in this paper is similar to that in HotpotQA, e.g. Figure 2 of HotpotQA and Figure 2 of DivKnowQA. Given that HotpotQA is substantially larger and was extensively studied, it is important for the authors to elaborate on the distinctive aspects that set DivKnowQA apart from HotpotQA.
3. Models being tested: If DivKnowQA is supposed to be a benchmark, then at least more than 1 LLM should be used. In this paper, only ChatGPT 3.5 is tested.

### Questions
Some notes in the paper draft are not removed for the submitted version. E.g. in page 7, "*here insert the dataset name*" should be replaced with DivKnowQA.  In page 8, "referred  to as *approach*" should be replaced with "referred to as *DetLLM*".

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this paper authors present a benchmark dataset that is Human verified that focuses on abilities of LLMs to use both the structured knowledge and textual sources. Benchmark is created keep in mind the importance is given to both structure source and textual source. It describes the process of dataset creation in detail. It also presents a baseline prompting  methods for LLMs that can aid them in solving this benchmark showing better results over existing LLM prompting methods.

### Strengths
Benchmark dataset that is carefully curated to test LLMs ability to use both structured and textual data. 
Detailed analysis of the benchmark and process used in arriving it.

### Weaknesses
Evaluation seems to be somewhat flawed to me. Some of the baselines use only one of the two sources and then the proposed prompting method uses both of them together.
Is there way authors could have provided information from both sources to the baseline methods to be fair? Given the benchmark needs both sources to be used for answering, its not fair to cripple some of the baseline methods not providing all the data to generate answer. 
Several details of how entity linking is performed etc are missing.

### Questions
Can you provide an example walk through of how DETLLM solves a question from the benchmark to make things clear on the overall process followed compared baselines. 
Do you have results for both linearised KB triples and text provided to baseline methods by devising some prompt?

### Soundness
3 good

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
This paper introduces a new QA dataset that jointly exploit open-domain question answering over structured KBs and unstructured text. The paper argues that even though there has been some work in developing QA datasets that exploit KBs and text, but more focus have been given to the textual component part. For example, no datasets come with ground-truth annotations of structured query language (such as SPARQL).  Such structured languages can be especially helpful for aggregation queries such as “How many European cities have a population > x?” as these are harder to answer from text but structured languages have operators and conditional statements which make it very easy.

The process of creating a multi-hop question begins with using an entity-linked single-hop question in the NaturalQuestions dataset along with its entity-linked answer. This is followed by gathering one-hop triples in the Wikidata knowledge graph where the question and answer entities are either subjects or objects. To ensure that KB reasoning is equally an important part of answering as text, they retain KB triples such that the corresponding entity is not present in Wikidata. Next, a single-hop question is generated from the triples which is followed by combining two single-hop questions (one from NQ and one from KB triple) into a multi-hop question. The method heavily relies on GPT-3.5 to both generate the question from KB as well as combining two question to from a multi-hop question. Apart from fact-based entity centric questions, yes/no as well as aggregate questions are generated (e.g. count-based questions). To ensure high quality of the dataset, each question is verified by two human experts and the resulting dataset consists of 940 questions. 

The paper also contributes a model that employs question decomposition, retrieval from diverse knowledge sources to answer the question. GPT-3.5 is employed heavily in most steps.

### Strengths
**Originality**

* A dataset of multi-hop questions created by composing two simpler questions from different sources have been employed before. However, this paper does a decent job to ensure one of the entity is not popular enough to have a Wikipedia/Wikidata entry. Therefore the dataset contribution is original enough.

**Quality**

* The human verification is likely to ensure that the dataset is of reasonably high quality.

**Clarity**

* I found the dataset creation steps to be clearly understandable. The motivation of having such a dataset was clear. However, I found section 3 which explains the contributed model to miss many details (more in the weakness/question section). Similarly I found the experiment section to be very hastily written missing key details and leaving several important questions unanswered (more later). Therefore I believe the paper will significantly benefit from a round of re-writing.

**Significance**

* Even though the dataset is small (only 940 questions), the dataset has potential to be integrated into a larger benchmark for testing the reasoning ability of LLMs. However, I am really unclear about the effectiveness of the proposed model. This combined with several missing details make me question the significance of the paper in its current state.

### Weaknesses
* Even though I think the decision to include KB triples where the subject/object entity is not present in Wikidata to ensure equal reasoning importance to both KB and text is reasonable, but even though an entity is not captured in Wikidata, that doesnt necessarily mean that fact is not present in web-text. Since LLMs are trained on web-text, what is the guarantee that an LLM would require KB fact to answer the question? It is possible that the knowledge corresponding to the KB triplet is present in the LLM because it picked that up from a text snippet (not in Wikipedia) during pre-training. 

* I think the model section (section 3) is missing many important details that severely hinder readability as well as trying to understand the limitations of the model. For example (and this is only a subset of missing information):

  * What are the instructions for query decomposition given to LLM?
  * How does the retriever work? 
  * Similarly what is the model that generates SPARQL queries that are executed on the KB? 
  * Does the same retriever work for both text and linearized KB triples?
* What is sparse retriever for text and KB categories in sec 4.2?
* Apart from the model details, the result sections are also very unclear. For example, 
  * It is very unclear to me why ChatGPT which does not use any knowledge is the most competitive model in Table 3? I found an explanation of this missing in the paper.
  * Why does the text + kb + sparql model perform a little worse than the text + kb model? I found an explanation of this missing in the paper
  * Overall, it is unclear if the proposed model in the paper works better than ChatGPT even though the underlying LLM of the proposed model is also ChatGPT

### Questions
I have listed several questions in the Weakness section which needs to be addressed to make the paper better.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
