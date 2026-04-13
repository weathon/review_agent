## Human Reviewer 1

### Summary
This paper introduces GFMBench, a framework for standardizing and automating benchmarking of Genomic Foundation Models (GFMs), which integrates four large-scale benchmarks containing 42 million genomic sequences across 75 datasets. The framework addresses several challenges in genomic modeling by providing unified protocols, consistent metrics, and automated evaluation pipelines that work across both DNA and RNA tasks from multiple species. GFMBench is released as open-source software with user-friendly interfaces, built-in tools for genomic tasks, and includes a public leaderboard and online hub for sharing benchmark results and resources with the research community.

### Strengths
The paper integrates 4 large genomics benchmarks from other groups into a unified framework. They also create a way to run a pipeline with these benchmarks with GFMs.

### Weaknesses
The paper exhibits several significant limitations. First, the technical innovation appears limited in scope. While the work integrates multiple existing benchmarks into a unified framework, it does not create new benchmarking methodologies or metrics. The automation and standardization approaches described largely follow established machine learning practices, without introducing fundamental methodological advances in benchmarking or evaluation techniques.

The work's focus is predominantly engineering-oriented rather than research-driven. The main contributions center on developing practical tools and interfaces for the genomics community. While valuable from a practical perspective, the emphasis on standardization over novel scientific insights raises questions about its suitability for a top-tier machine learning conference. Furthermore, similar benchmark aggregation platforms already exist in other machine learning domains, making this approach less innovative.

### Questions
What specific technical innovations in benchmarking does GFMBench introduce?

What novel metrics or evaluation approaches are introduced?

What specific reproducibility issues does it solve that other frameworks don't?

Why were these 4 benchmarks and then GFMs selected among others?

In Table 4, why is GFMBench listed as a model?

### Soundness
2

### Presentation
2

### Contribution
1

### Rating
1

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper introduces GFMBench, a framework for genomic foundation model (GFM)
evaluation. The proposed benchmark covers a diverse set of genomic tasks and datasets, and
provides methods for standardised metrics and evaluation, thus 
enabling fair comparison of different GFMs. The papers include an evaluation of
multiple GFMs on the benchmark of standardised metrics, which allows the 
comparison of the models. GFMBench is released as open-source software and
it includes an online leaderboard.

### Strengths
Generally, the paper is well-written in good English and easy to follow.
The motivation for the paper is clear and the problem is well defined.
Overall, the authors make a significant contribution to the field of genomics and
foundational genomics models. Providing a unifying benchmark for evaluation
is a significant contribution and will very much contribute positively to the
development of the progress of GFMs. The authors will release a webpage with a
leaderboard for model comparison, which is really positive. The authors
will also release the code for GFMBench.

### Weaknesses
There are some issues regarding the clarity of the contribution of 
the paper and the reported results. The paper also presents some claims
that are not backed up or briefly supported. The issues are listed below:

Major issues:

- Data scarcity is mentioned to be an issue for genomics datasets in the paper.
While it is true that diversity is a problem in the access to genomic data it is 
important to note that there are indeed many large-scale datasets available.
The main concern regarding this issue is that the authors claim that
GFMBench tackles this problem, however, as far as I'm concerned, the benchmark
does not contribute any novel data, but it is rather a compilation of
existing datasets. The claim that GFMBench mitigates "issues of data scarcity"
does not seem truly accurate.

- On page 5 it is claimed that "GFMBench" tackles the problem of 
"benchmark standardization". While I understand the contribution of common
metrics and implementation, I don't see how this is achieved in the
"hyperparameter settings". How is standardization achieved in this regard?

- In Line 486, it is said that "OmniGenome consistently performs well across various genomic benchmarks",
however, Table 4 results do not seem to match this claim (correct me if I'm wrong).

- Through the paper, the authors praise the performance of "OmniGenome" GFM,
as it shows good performance, especially on RGB and PGB tasks.
Is OmniGenome also a contribution to the paper? If so, it would be great
to have more details on it. If not, it would be great to provide more insights
into the other models and why they perform better or worse in some tasks. 
Similarly, since the contribution of the paper is a benchmark, it would be 
more significant to provide some insight into the difficulty of the tasks
and why performance varies across them, instead of commenting only on OmniGenome.
I acknowledge that the paper highlights repeated times that the major performance
of OmniGenome is due to the fact that it is trained on "secondary structure" tasks,
However, this fact is repeated multiple times in the paper while no comment
is provided on the other models or tasks at hand and why "secondary structure"
improves performance on the tasks.

### Questions
A list of questions, minor issues and suggestions is included here:


- The abstract mentions "the absence of open-source software for diverse genomics".
This doesn't sound true to me. Genomics is especially rich in open-source
tools (e.g. see the collection of tools on the Galaxy platform 
"The Galaxy platform for accessible, reproducible and collaborative biomedical analyses").

- On page 2 "lack of comprehensive and diverse datasets necessary for robust training and testing of GFMs.".
It would be great to support these claims with related work or references.

The first paragraph of page 4 is repeated twice.

- In Table 7 caption: "in Agro-NT".
Please include a citation to a tool, work, dataset or reference when mentioning
it for the first time. This happens also later in the text, e.g. "ViennaRNA", 
"Archive2", "Stralign", etc.

- In Table 5 caption: "These benchmark datasets are held out or not included in the pretraining database".
It is unclear to me what this means. Would be thankful if the authors 
could clarify what this means.

- What experiment is Appendix C referring to? Please clarify what these
hyperparameters are for.

- On page 6, "We provide a tutorial for AutoBench in ...".
To make the paper more consistent, it would be better to include the
tutorial URL in a footnote too.

- I understand that AutoBench is a standardisation suite for benchmarking, but
I believe that the paper could benefit from a more detailed explanation of it
and its contribution. For instance, how is it an "automated" benchmarking solution?

- Genome data augmentation is mentioned as a contribution, however,
no technical details are provided on how this is achieved. This is indeed
a subject of interest, and it would be great to have more
details.

- While I think that this is not a trivial issue, the paper does not mention
how are long sequences treated with the models. Is this because this is not an issue
for the model's evaluation? Otherwise, how is this managed? Is this handled or
should be handled in the GFMBench framework?

- Why does "GFMBench" appear as a model in Table 4?

- There are some spelling errors, e.g. "pertaining" in Line 1080.
Please double-check for spelling errors.

- "blastn" in Line 1066 is usually written in uppercase.

### Soundness
3

### Presentation
4

### Contribution
3

### Rating
5

### Confidence
3

---

## Human Reviewer 3

### Summary
In this paper, the authors present an integrated platform, GFMBench, for benchmarking genomic foundation models. The authors synthesized and standardized four existing benchmarks and created a unified software framework to run GFMs and calculate evaluation metrics in a streamlined manner. In general, I consider this work lacks novelty and does not contribute sufficiently to the field, which is my primary reason for recommending rejection. While there are engineering efforts to aggregate and standardize existing benchmarks, the contribution does not extend much beyond that.

One could argue that such work helps facilitate the benchmarking process, and I appreciate the effort the authors put into building the software. However, I do not think this type of contribution is suitable for a venue like ICLR. I acknowledge that others might hold a different opinion, so I am setting my confidence level to 3.

### Strengths
1. There are interesting ideas in this work that could be expanded and delved into further. For example, the authors attempt to create a unified benchmark for DNA and RNA tasks, which raises the potentially interesting topic of studying the cross-modality transferability of GFMs. This aspect could be expanded and investigated more deeply to provide valuable insights.
2. Having a unified platform like this can enhance reproducibility in GFM development and evaluation.
3. The platform includes user-friendly features such as an API, an online hub, and a leaderboard.

### Weaknesses
1. As mentioned in the summary, this work generally lacks novelty.
2. One of the four main contributions claimed by the authors, Data scarcity and bias, is vaguely stated and not particularly valid. First, the discussion seems to conflate training data and evaluation data. This work addresses only the evaluation side, so the writing should focus on that. Second, since the data sources are four well-established benchmark datasets, there is no additional contribution to resolving the data scarcity issue, as no new dataset is introduced.
3. The authors did not discuss or justify the choice of the four benchmark datasets. There is no comprehensive review of existing benchmarks, which I think is conventional for such work. For example, the benchmarks from NT (https://www.biorxiv.org/content/10.1101/2023.01.11.523679v4), BEND (https://openreview.net/forum?id=uKB4cFNQFg), and Tang et al. (https://www.biorxiv.org/content/10.1101/2024.02.29.582810v2.full.pdf) are well-established and recognized benchmarks in the field but none were mentioned in the manuscript.
4. The manuscript contains many claims that lack supporting evidence and concrete examples. For instance: 
a. The authors mention metric variability but do not show empirical evidence. How would using the original metrics from each benchmark affect the results? 
b. The authors claim that this work helps test models on underrepresented sequences and tasks, but there is no further explanation or evidence on why or how this work addresses this issue.
5. The writing could be improved for conciseness and clarity. And there are some overly repetitive statements throughout the article.

### Questions
There are two repeated paragraphs in lines 154-167, as well as some other typos in the manuscript.

### Soundness
2

### Presentation
2

### Contribution
1

### Rating
3

### Confidence
3

---

## Human Reviewer 4

### Summary
This paper introduces a meta-benchmark for transfer learning that integrates tasks from four existing benchmark sets. They revise metrics used in these benchmarks to account for class imbalance. The authors implemented their evaluation protocol in a Python package they claim is easily extendable and flag this as their primary contribution.

Orthogonally, the paper presents benchmarking results for several candidate foundation models for genomics. In addition to DNA language models, they include models trained on RNA. The authors highlight the strong performance of the RNA language model OmniGenome.

### Strengths
1. The authors address the problem of evaluating genomic foundation models, which is an important problem.
2. They appear to have developed a comprehensive, well-documented framework for benchmarking.
3. They convincingly demonstrated that current language models trained on DNA sequences do not learn representations of RNA sequences as effectively as those trained purely on RNA sequences.
4. They provide strong evidence that supervising models on annotations of RNA structure provides these models with important information.

### Weaknesses
1. The paper would be stronger if the evaluations were more comprehensive. The authors could have included newer benchmarks—like BEND (ICLR 2024) or those included in the recent preprint by Tang and Koo (https://doi.org/10.1101/2024.02.29.582810) —that address important limitations of the previous benchmarks.
2. It would also benefit from including additional language model baselines, like PlantCaduceus or GPN.
3. It would also be worthwhile to evaluate models other than language models, since these have been shown to outperform language models on transfer learning tasks.
4. I am not convinced of their claim that “OmniGenome generalizes well to DNA-based tasks, likely due to shared sequence motifs and structural similarities between RNA and DNA.” I do not think this explanation is true for noncoding elements. It is more likely that their benchmarking needs to be more thorough. From their comparisons, it is also unclear whether OmniGenome generalizes well—only that it generalizes comparably to language models trained on DNA.

### Questions
Does OmniGenome utilize structural information in its input?

### Soundness
2

### Presentation
1

### Contribution
2

### Rating
3

### Confidence
3