# Unearthing Large Scale Domain-Specific Knowledge from Public Corpora

- Avg Score: 5.00
- Decision: Reject
- Scores: 5, 3, 6, 6

## Abstract
Large language models (LLMs) have demonstrated remarkable potential in various tasks, however, there remains a significant lack of open-source models and data for specific domains. Previous work has primarily focused on manually specifying resources and collecting high-quality data for specific domains, which is extremely time-consuming and labor-intensive. To address this limitation, we introduce large models into the data collection pipeline to guide the generation of domain-specific information and retrieve relevant data from Common Crawl (CC), a large public corpus. We refer to this approach as Retrieve-from-CC. It not only collects data related to domain-specific knowledge but also mines the data containing potential reasoning procedures from the public corpus. By applying this method, we have collected a knowledge domain-related dataset named Retrieve-Pile, which covers four main domains, including the sciences, humanities, and other categories. Through the analysis of Retrieve-Pile, Retrieve-from-CC can effectively retrieve relevant data from the covered knowledge domains and significantly improve the performance in tests of mathematical and knowledge-related reasoning abilities.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This work presents an automated pipeline to collect domain-specific data from public corpora by leveraging large language models (LLMs) for query expansion and BM25 for efficient retrieval. The resulting dataset spans multiple domains, including STEM, humanities, social science, and Misc. While the method emphasises scalability and cost-effectiveness over manual curation, it also faces challenges in ensuring data quality and distinctiveness compared to existing human-curated datasets. Experimental results indicate that models trained on the proposed dataset are improved in several reasoning benchmarks.

### Strengths
- This work addresses a well-motivated goal: developing an accurate and scalable approach to extract training data from the evolving web, which is essential for keeping LLMs up-to-date.

- Experimental results show clear improvements in LLM performance on the listed benchmarks when further trained on the proposed dataset.

- The authors provide a detailed analysis of the proposed dataset, including multiple evaluation factors. Notably, the data leakage analysis between the pre-training dataset and the evaluation benchmarks is a valuable addition, helping to ensure the integrity of the results.

### Weaknesses
### About Novelty

- The authors have not clearly demonstrated how the proposed dataset differs from or adds unique value compared to existing high-quality, human-curated datasets, including the ones shown in Table 1.

- Additionally, the proposed pipeline lacks distinctive features that would make this automated construction process stand out as an innovative or superior alternative. For example, since this work focuses on domain-specific knowledge, it can be beneficial to leverage knowledge bases  or other kinds of structured data to help improve the relevance and accuracy of the data points.

### About Methodology
- While the authors emphasise scalability, fully relying on LLMs to refine and expand queries is both computationally expensive and prone to errors.

- Although BM25 offers efficiency and scalability in the retrieval phase, it does not guarantee high accuracy in the retrieved `(query, answer)` pairs. Even if standard dense retrieval techniques were employed, achieving consistently high accuracy would remain challenging.

- Consequently, errors introduced during both the query generation and retrieval phases could propagate, potentially compromising the overall quality of the final dataset.

### About Evaluation

- Given that LLM pre-training typically involves a broad set of evaluation benchmarks, this work lacks an analysis of potential "forgetting" on benchmarks (e.g., HELM, GLUE, LLM leaderboard) not included in the listed experiments.

- When comparing with other pre-training data, if you aim to demonstrate that this automatically generated dataset holds value even against human-curated datasets, it would be essential to directly compare their performance. Ideally, this comparison would show that the proposed dataset lags only by a small margin. Alternatively, comparing it with well-established synthetic data generation methods (also discussed in Suggestion 1) would also help substantiate the dataset's value and highlight areas for potential improvement.

### Questions
### Suggestions

- Given the fully automated nature of the proposed pipeline, the resulting dataset is more like a synthetic dataset. This raises a different research question that how to generate high-quality synthetic datasets effectively and how this method compares to existing synthetic data generation approaches.

- I recommend that the authors invest effort in refining the paper's writing, as several language issues currently affect the fluency of reading.

- I recommend ensuring consistency in terminology throughout the paper. For instance, terms like 'Retrieve-Pile' and 'Knowledge Pile' appear to refer to the same dataset, which could cause confusion.

### Soundness
2

### Presentation
1

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
The authors propose Retrieve-from-CC, a data collection pipeline consisting of a a) query generation process and b) a document retrieval process based on the generated queries. They argue that their proposed method makes it possible to automate the data collection process for high quality domain-specific data. 
The so created dataset is then evaluated with regard to its composition (sources and domains) and its data quality (quantified as QuRating).
Finally, the authors use their newly composed dataset to fine-tune two LLMs (based on Llama2 and Mistral) and to train a Llama model from scratch. These models are then evaluated with regard to their performance on various benchmark datasets on mathematical and knowledge oriented language understanding tasks.
Their experiments indicate that their newly collected dataset of domain-specific high quality data can be used for fine-tuning LLMs and improve their performance on tasks that require specific knowledge. They additionally show that there is little to no dataset contamination when comparing the downstream task data with their own data.

### Strengths
+ extensive experiments, showing that their proposed data is helpful for training “smaller” (i.e. 7B Parameter) LLMs on domain specific task
+ automating the process of curating high-quality domain specific data
+ addressing the issue of data contamination 

The authors propose a method of automating the process of curating high-quality data in specific domains. Furthermore, they showcase how their data can be used to fine-tune LLMs and improve their performance on benchmark tasks within the respective domain.

### Weaknesses
1) writing and language throughout the paper needs improvement. This makes the present work difficult to follow at times:

l. 67 “Otherwise, […]” -> additionally?
l. 70 “[…] continuing learning […]” -> unclear if they talk about fine-tuning
l. 82 “[…] for the quality and statistical of Retrieve-Pile […]” -> statistics?
l. 82 “We statistic […]” -> usage as a verb?

Paragraph starting at l. 179: Is question evolution the same as question extension from the previous paragraph?

l. 237 “Otherwise, we discuss about the different when improving different […]”

Unfortunately, these are only some highlighted examples where the quality of presentation is lacking.

2) exposure of results:

Table 4: It would be beneficial to mention the reported metrics. If not here, then at the description of the benchmark datasets.

This issue is persistent in most of the evaluation section. The authors report an increase of performance in “points” on multiple occasions, which frankly speaking could mean anything.

In the case of data quality, a little more in-depth explanation of the QuRating would be helpful in understanding the results. As this is by now not a wide-spread metric, it would be beneficial to explain how the values are obtained and what exactly they mean.

Overall, the authors could improve the exposition of results by providing a more detailed explanation of the used metrics, as this is an important bit of information for the reader.

3) focus:
Overall, the focus of the paper is not very well defined. First, the authors introduce a dataset collection method and provide a detailed overview of the collected dataset. For the present work to be a resource paper, the proposed Query Bootstrapping methods is not explained in sufficient detail. The paragraphs on “Question Extension”, “Thought Generation” and “Query post processing” are rather vague.

The other side of the spectrum would be a paper on an empirical study. For this to be the case, the evaluation section would require to be more detailed (regarding reported metrics).
In addition to that, following one of the author’s main argument (automating the process of collecting high-quality domain-specific data), it would be nice to see how LLMs that are trained on their data perform vs. LLMs that are trained on hand-crafted datasets.

My overall impression is that the authors should focus on either the resource side of their work, or the empirical side. In its current state, the lack of focus in combination with a (at times) poor representation makes the paper appear inconsistent and at times hard to follow.

### Questions
My suggestion for improvement would be to include more details about the Query Bootstrapping method and the metrics reported in the empirical section. I am confident that the language issues could be easily resolved as well (maybe with the help of an LLM even).
My main issue, however, is the unclear focus of the paper. For it to be a good resource/method paper, the data collection process should be described in more detail. For it to be a good empirical study paper, the experiments should reflect the argument of automating the dataset collection process vs. manual creation of a dataset. Unfortunately, this would require major revisions and potentially additional work. (The result however, might be two good papers (one with focus on the data, and another one with focus on the empirical evaluation), as the underlying questions are relevant and interesting)

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper provides a method called Retrieve-from-CC to curate domain specific data to train large language models. It uses a two phased approach where initial query keywords are given by humans and a LLM is used to generate queries which are fed to retriever (BM25) to gather data that is relevant for a specific domain. Authors publish a benchmark, Retrieve-Pile, that is covering four domains that includes sciences, humanities, Social Sciences and Miscellaneous.  Paper shows that using this data set helps in improving the performance on some of the mathematical benchmarks along with standard language benchmarks.

### Strengths
Peper described a method for collecting domain specific training data without manual intervention, which is beneficial for makings LLMs perform better for domains of interest and making them more suitable for practical applications for a domain of interest. Method being automatic and showing improvements over the base LLMs is promising and can be beneficial in gathering large data that LLMs need for their training. Empirical results show that data generated results in significant improvements using it in LLM training. Quality metric also show that curated data is of good quality. Data pile created shows improvements during per-training as well as further training existing open models, which shows that generated data is of good quality.

### Weaknesses
I see some places Knowledge-Pile is used without it being talked about anywhere. I guess its a typo instead of Retrieve-Pile it is used in evaluation section, tables, figures etc. This needs to be corrected. Lot of places I see space missing after Retrieve-Pile and other typos needs to be corrected as well.

### Questions
Which LLM is used for query generation module? Did you see any difference in the queries generated with various LLMs ?
Assumption is that LLMs are not great at domain specific tasks, how does that impact your automatic query generation. Did you analyze the quality of queries generated for the domains? 
Is it the same model that is used for query generation or a bigger LLM is used ?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the problem of bootstrapping domain knowledge from general public corpora in order to reduce cost and time for manual data collection of domain-specific corpora. Using manually defined seed the presented approach, Retrieve-from-CC, first identifies seed examples in a large input corpus using BM25. For every retrieved record a LLM generates questions and responses are augmented by Chain of Thought sequences. After a quality filtering step the approach outputs a domain-specific dataset.

### Strengths
Generating high quality datasets for augmenting the capabilities of LLMs into less covered domains is high relevant for the open-source as well as professional community. Data is the key for LLM success and this paper aims to present contribute for this matter.

### Weaknesses
*Approach*:
- seeds: A downside of the approach is that it needs good seeds as input. At the same time, general-domain knowledge bases (dbpedia, yago) cover almost all domains to some degree. While the data goes beyond simple key phrases, the domains are probably only covered partially. The authors should consider leveraging this partial knowledge for generating input data for the bootstrapping phase. This drops the manual input requirement and might improve the dataset further.

*Evaluation*:
- My biggest criticism of the paper is that the author didn't compare against domain-specific LLMs. The question: "How does a LLM trained over a corpus generated with Retrieve-from-CC compare against a domain-specific LLMs?" is highly relevant for this paper and is not answered. For instance, one could compare against LLMs from the [Open Medical-LLM Leaderboard](https://huggingface.co/spaces/openlifescienceai/open_medical_llm_leaderboard). There are probably other such domain-specific resources. 


* Misc: Typo generted in Figure 2

### Questions
Beyond the tasks you evaluated were there any performance changes after you further trained the LLM with Retrieve-Pile?

### Soundness
2

### Presentation
3

### Contribution
3
