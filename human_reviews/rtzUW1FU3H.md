# Can long-context large language models understand long contexts?

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 3, 8

## Abstract
Large language models (LLMs) have received significant attention by achieving remarkable performance across various NLP tasks. However, the fixed context window length of the transformer architecture makes them incapable of memorizing and understanding extremely long inputs. There are tremendous works in designing effective and advanced techniques to enlarge LLMs' context window size, which call for high demands on developing high-quality benchmark datasets to evaluate LLMs' long context understanding ability. There are some existing datasets for this purpose. However, they face the problems of (1) shorter text length compared to modern LLMs' context window length, (2) out-of-date documents that may already been included in the training corpus of modern LLMs, and (3) most of the tasks are short dependency tasks---there are few questions that really need LLMs to collect information across the whole document (which we call. Most importantly, they hardly consider assessments on long dependency modeling and understanding across segments, which are particularly challenging and valuable for improving LLM long context. In this paper, we present LooGLE, a Long Context Generic Language Evaluation benchmark for LLM long context understanding. It contains up-to-date documents (all after 2022), over 24k tokens per document, and 6k newly generated questions from diverse domains and categories. Specifically, we recruited a group of human labelers to read 145 long documents in our benchmark, and asked them to compose about 1.1k QA pairs satisfying our long dependency requirements. These 1.1k high-quality QA pairs are each cross-validated 3 times by 2 labelers, aiming to provide the currently most accurate evaluation of LLMs' ability on long dependency questions. Upon a comprehensive evaluation of 8 state-of-the-art LLMs on LooGLE, we find that: (1) Commercial models generally outperform open-sourced models. (2) LLMs are more skilled at short dependency tasks like short QA and cloze but still struggle on performing real long dependency tasks. (3) In-context learning and chain of thoughts only bring incremental improvement for long context understanding. (4) Retrieval-based techniques significantly contribute to improvement on short QA whereas many techniques for extending context window length through optimized transformer architecture or positional encoding can hardly resolve long context understanding.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper presents a new benchmark called LooGLE for long-context LLMs, with inputs longer than 24k tokes. Some/most examples are human-annotated, and some/most were cross-validated by multiple annotators.
Further, some/most of the documents in the benchmark were published after 2022, which is expected to be after the knowledge cutoff of LLMs such as GPT-3.5 and GPT-4, forcing them to rely only on their in-context learning abilities rather than their prior knowledge.

Such benchmarks are always useful and needed in the community, especially following the growing interest in long-context LLMs.

### Strengths
* The authors made lots of efforts in collecting and curating the data.
* Such benchmarks are always useful and needed in the community, especially following the growing interest in long-context LLMs.

### Weaknesses
- Following the promises in the abstract about the human-annotation and the cross-annotator validation, I was very disappointed to see that a large part of the benchmark's ground-truth output was generated using GPT 3.5 / 4:
>We utilize the powerful language processing and understanding capability of GPT-3.5-turbo to help generating short QA pairs from the original text.

>we employ GPT-3.5-turbo to generate factual summaries align with the source segment using with constraints

If this is indeed the case, this is disappointing, and the benchmark may be biased toward "questions that are easy for ChatGPT to answer".

- The text is very unclear in many cases. For example when describing the statistics of the benchmark, the paper says:
>Extra-long realistic documents. It contains 778 latest gathered and extremely long documents
with an average of 16.4k words. There are over 6000 test instances without distribution bias for a
more generalized assessment, many of which are exceeding 100k words.

So are there 778 examples or 6000 examples? If "many of which exceed 100k words", how many of them? what's the average? Are these two datasets? If not, why are these numbers reported separately?

- The results in Section 4.3.1 are very confusing and unclear. For example:
>In Table 3, it can be noticed that LlamaIndex obtains from the perspective of GPT4 evalution. Instead
of memorizing a shortcut of original input with a limited context window, retrieval-based context
compression technique augments the LLM by incorporating external memory, allowing relevant
information to be retrieved using a specific query.

I am not sure what such paragraphs are trying to say. What does it mean that "LlamaIndex obtains from the perspective of GPT4 evalution"? What do the authors exactly mean by "memorizing a shortcut"? Who is memorizing a shortcut?
- Measuring "GPT4 score" on GPT4's outputs is mostly meaningless. It would be better to just completely remove this column, or use another LLM that is not evaluated.
- Applicability: the paper does not mention anything about its implementation, its ease of use, its availability. As always with benchmarks, the devil is in the details, and the authors have not included the data itself, which makes it hard to really evaluate its quality.
- Presentation is poor: for example: 
    - the text in Figure 1 is tiny, not allowing to actually understand the overview of the new benchmark. 
The entire left part of the figure contains barely any information.
I would prefer an organized and readable list of tasks and data statistics.

    - The text in Table 1 tiny
    - The text in Table 2 is tiny. Further, it would be helpful if these statistics would include the max/min instead of category, or a more illustrative figure of the characteristics of the examples, as in Figure 1 in the [SCROLLS paper](https://arxiv.org/pdf/2201.03533.pdf)
    - The text in Figure 3 is tiny. Further, the colors are very similar, and I cannot distinguish between the different models and cannot understand anything from this figure.

### Questions
### Questions
1. Section 3.3.1 says that "we directly use the abstract of each paper as the reference for generating summaries" - so, the ground-truth summaries where **generated**? are the Abstracts **used** in any part of the process other than for evaluation?
2. Are the authors going to release the test sets, or keep them "hidden"?
3. Are there training/test spits, or is everything "test"?

### Comments
1. The comparison in Table 1 on "which tasks are included in each benchmark" shows that LooGLE contains many tasks that other prior benchmarks do not. However, it is a bit unfair, because these prior benchmarks contain tasks that are not contained in LooGLE, but these are not mentioned. For example, Scrolls, mentioned in the first line, does contain QA (mentioned with "X") and NLI (not mentioned at all).

### Summary
I appreciate the authors's efforts, but as much as good benchmarks are needed in the community, unfinished benchmarks can do harm and drive research in the wrong direction.
I cannot evaluate the benchmark itself since it was not released, but the paper still feels a bit unclear and unfinished, which makes me worry that the benchmark is too.
Thus, I currently vote for rejection, and hope that the authors would polish both the paper and the benchmark and release them when they are in a more polished state.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
3 good

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents a new benchmark for long context understanding, consisting of 24k tokens in average. Firstly, this has the advantage to be more challenging than former benchmarks that have shorter texts compared with current LLMs' context window length (reaching up to 32k tokens). Secondly, it only contains newly created documents (after 2022) which are thus not present in most LLMs' pretraining data, preventing data leakage and enabling fairer evaluation. Experiments on current state-of-the-art LLMs reveal challenges in long-context understanding.

### Strengths
- A curated dataset, thoughtfully designed with great efforts to prevent data leakage and ensure long dependency.
- Great efforts in assessing current state-of-the-art LLMs' long dependency capabilities such as information retrieval, reading comprehension and reasoning, computation and timeline reordering.

### Weaknesses
- see questions.
- the paper is sprinkled with typos (refer to questions for a few).

### Questions
- Is the benchmark english-only ? If so, this needs to be mentioned.
- How will the dataset be released ? For instance ZeroSCROLLS only released the inputs and evaluate through a system of leaderboard, will LooGLE be released the same way ? I'm concerned that revealing input/gold outputs pairs would lead to data leakage for future models.
- Concerning data collection, were the sourced documents (after 2022) subjected to any machine-generated verification ? I am concerned that ChatGPT-like texts might compromise the fairness of the evaluation.
- Are the open-source models instruction-tuned ? Commercial closed source models like GPT3.5 rely on RLHF or some instruction tuning techniques that enable them to better follow instructions. If the considered open-source baselines are not instruction-tuned, the comparison might be unfair since the prompt used for evaluation is the same and is instruction-based.
- Is it fair to have GPT4 both as baseline and evaluator ? I am also concerned about the creation of the dataset of short QAs (generated by GPT3.5). Is it fair to evaluate a model that was used to create the dataset ?
- In section 4.2, you mention human evaluation (3) but I cannot find any human evaluation both in the paper and the appendix.
- Section 4.3.2 is confusing, the experiments are based on the recent work from Liu et al. 2023, that accessing information in the middle of the document is more challenging for LLMs. If I understood correctly, you suggest concatenating the head and the tail of the inputs and give it to the LLMs as input. This would mean discarding the whole "middle" and leaving the beginning and the end. This puts an immediate limitation on information that can be retrieved by the LLMs. Also did you only use the same model (GPT4-32k) but with various context input length or different variants of GPT4 ? What is the last entry in Table 5 ? (GPT4). For long summarization, arxiv abstracts are highly biased towards the beginning of the article so it is expected that increasing context would result in a higher divergence between the generated summary (which will contain more and more details) and the (gold) abstract.
 
**Typo**
- page 2: "Mannually designed both short and long dependency tasks"
- page 3: "COmputation."
- page 7: "Retrival"
- page 8: "GPT4 evalution"
- table 4: "Performence"
- appendix page 3: "Dispcrepancy"
- not really typos but it is uninformative to report results of the order of "e-300" since at this point there is nothing to really compare.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
LLMs have shown impressive performance in various NLP tasks. However, the fixed context window length of the transformer architecture limits their ability to understand extremely long inputs. Existing datasets for evaluating LLMs' long context understanding have limitations such as shorter text lengths, outdated documents, and a focus on short dependency tasks. The paper introduces "LooGLE"  to evaluate LLMs' ability to understand long contexts. Upon evaluating 8 state-of-the-art LLMs on LooGLE, the authors found:
1. Commercial models generally outperform open-sourced models.
2. LLMs excel at short dependency tasks but struggle with real long dependency tasks.
3. Retrieval-based techniques significantly improve performance on short QA tasks, but many techniques for extending context window length struggle with long context understanding.

### Strengths
- **Up-to-date Documents**: LooGLE contains documents published after 2022, ensuring that modern LLMs have not been pretrained on these documents.
- **Diverse Tasks**: LooGLE includes both short and long dependency tasks, providing a evaluation of 8 LLMs' capabilities.

### Weaknesses
**Lack of Experimental Data to Support Some Claims**:

- The paper states that "by employing scaling techniques like positional interpolation, parallelization, and finetuning on longer texts, open-sourced models have shown improvement in handling longer inputs compared to previous versions." However, the article does not provide performance data of previous version models. This omission makes it challenging to ascertain the improvements is brought by modifying position embeddings or instruction-tuning. For example, the comparision between vicuna-2k and vicuna-16k is a better case to validate this claim.
- The claim that "GPT4-32k performs better than GPT-8k" is not consistently supported by the provided metrics. The results between the two models vary across different indicators (automatic metrics v.s. gpt4 score). A more in-depth explanation and analysis are needed to support this claim, including understanding the differences in various metrics. It's unclear why, on Long dependency tasks, the longer window 32k model performs worse than the 8k model. 

**Absence of Human Evaluation**: The paper mentions conducting human evaluations, but there's no presentation of the related data. GPT-evals may have some preference in generation length, human evaluation can be a better reference.

### Questions
- Details of Llamaindex is not clear: 7B or 13B, chat model or regular model?
- Why not use the chat version of Llama2, which is considered to be skilled at instruction-following and could be potentially better at downstream tasks.
- Why LlamaIndex is much better than any other open-sourced models? According to your results in Fig3, retrieval+open-sourced model is better than the long-context version of the same base model, and this conclusion is contradict to the conclusion from other long-context benchmarks (Table1).
- Writing format: it's better to list url as the footnote. It is supposed to leave a black between the main text and citation brackets.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors present a new dataset, called LooGLE, which aims at evaluation of LLMs on long context. Their dataset has documents with longer length compared to previous benchmarks and it is more up to date (2022+). The proposed dataset has task with long dependencies and the authors have made sure that for some of the tasks, the answer needs to be collected from multiple segments of the documents. The evaluate both commercial and open-souse models on the new dataset and provide some insights.

### Strengths
- The paper addresses a very important area, i.e., long context evaluation of LLMs
- Based on the description, the collection method, and evaluation results, the proposed new dataset seems to be of high quality.
- Authors provide extensive evaluation on different commercial and open-sourced models.
- The paper is well written and easy to follow.

### Weaknesses
- The paper presents Human Evaluations as one of the evaluation technique but never present human evaluation results.
- There are several automatic scores have been presented in Tables 3 through 5 and sometimes. These scores not always in agreement; not all the scores are better for the winning model. I found this confusing especially when some conclusions are drawn in the text.

### Questions
- For the LlamaIndex, what retriever and chunk size are used?

### Soundness
4 excellent

### Presentation
3 good

### Contribution
4 excellent
