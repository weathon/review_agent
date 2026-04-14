# Diverse Genomic Embedding Benchmark for Functional Evaluation Across the Tree of Life

- Decision: Reject
- Scores: 6, 5, 5, 6

## Abstract
Biological foundation models hold significant promise for deciphering complex biological functions. However, evaluating their performance on functional tasks remains challenging due to the lack of standardized benchmarks encompassing diverse sequences and functions. Existing functional annotations are often scarce, biased, and susceptible to train-test leakage, hindering robust evaluation. Furthermore, biological functions manifest at multiple scales, from individual residues to large genomic segments. To address these limitations, we introduce the Diverse Genomic Embedding Benchmark (DGEB), inspired by natural language embedding benchmarks. DGEB comprises six embedding tasks across 18 expert curated datasets, spanning sequences from all domains of life and encompassing both nucleic acid and amino acid modalities. Notably, four datasets enable direct comparison between models trained on different modalities. Benchmarking protein and genomic language models (pLMs and gLMs) on DGEB reveals performance saturation with model scaling on numerous tasks, especially on those with underrepresented sequences (e.g. Archaea). This highlights the limitations of existing modeling objectives and training data distributions for capturing diverse biological functions. DGEB is available as an open-source package with a public leaderboard at 
\url{URL hidden for anonymity}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a new benchmarking suite to evaluate the ability of protein and genomic language models to accurately represent cellular function. The authors claim that most existing benchmarks focus on structural properties instead of function, and highlight several challenges with evaluating accurate functional representation. The authors then define six categories of tasks in their benchmarking suite, and aggregate 18 datasets falling into these categories that capture aspects of cellular function. Tasks involve predicting properties of individual sequences as well as pair-wise or multi-sequence interactions. The authors then benchmark several protein and genomic foundation models on the previous tasks, and highlight comparative performance as well as the presence of scaling law behaviour.

NOTE: Raised score from 5->6 in response to rebuttals.

### Strengths
- Selection of datasets capture meaningful aspects of protein function and phylogeny, and cover a wide range of prediction tasks. Organization of datasets into hierarchy based on prediction task and relationship conceivably would help extension of the dataset by integration of future datasets.
- Authors highlight the importance of controlling data leakage that might occur through sequence homology, and construct train / test splits that avoid this using several sequence identity thresholds. This contributes towards the rigour of models that would be benchmarked on DGEM, and raises the current standard in the genomics space. An additional contribution could be to include code to regenerate splits based on a random seed; this would be useful for generating CIs. 
- Benchmarking of existing foundation models on the selected tasks reveal interesting empirical evidence on the presence of scaling laws and performance gaps between existing AA and NA models.

### Weaknesses
- Although the selected tasks / datasets do represent aspects of biological function, I am concerned that the overall claims of the paper are too large in its current state. Many aspects of biological function that could be represented by a foundation model embedding are not tested for in this dataset, with most of the proposed tasks evaluating enzymatic function or evolutionary similarity between proteins. In particular, attention to tasks specific to NA sequences that could be used to evaluate gLMs are relatively lacking in comparison to protein tasks. For example, the classification of sequence into annotation classes as defined by the Gene Ontology resource is commonly used, or classification of RNA splice isoform function or tissue-specific expression could be a task related to function that evaluates gLMs. I believe the paper should either seek to restrict the breadth of their claims, or incorporate other commonly used measures of biological function into their dataset.
- As a benchmarking paper, I believe the authors are well positioned to further increase the relevance of their work by inclusion of baselines that are not derived from a foundation model. It has been shown in other biological foundation models that pre-trained embeddings often perform worse than using a linear model on a set of naively engineered features, and addition of results similar to this could contextualize the performance of the benchmarked foundation models. Similarly, it would also be extremely relevant if performance of an ab-initio supervised model on the provided tasks are shown. This would again contextualize the actual benefit gained from the often costly foundation model pre-training.
- The contribution of the paper could be strengthened by benchmarking more gLM models. While the overall conclusions of the AA vs NA experiment are believable given the known weaknesses of applying NLP objectives on genomic data, including a more diverse set of benchmarked gLMs could strengthen the claim.

### Questions
- An anonymized version of the code/data repository does not seem available, meaning it is difficult to assess some core claims regarding DGEB's implementation (extensibility, reproducibility, etc.). Is there any way to provide a snapshot of the implementation via OpenReview or some other anonymized github repository?

- The colour scheme for Figure 3's Model Type is not the most colourblind friendly, especially the colours for NT / ProtTrans / ESM3. I would appreciate it if the figure could be adjusted for easier reading. As a suggestion, maybe the gLMs could be outlined in black as an additional dimension for visual clarity?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper presents the Diverse Genomic Embedding Benchmark (DGEB), which is focused on evaluating protein and genomic language models on a set of six embedding tasks (inspired from MTEB) across 18 datasets.

### Strengths
- The authors introduce tasks that investigate a diverse range of species, many of which seem underexplored in the existing protein language model (pLM) and genomic language model (gLM) literature.
- The tasks appear to be biologically and evolutionarily relevant.

### Weaknesses
- This paper deals with two different classes of models - AA models and NA models, which generally have very different underlying assumptions and draw from separate corresponding bodies of work. This dual focus weakens the depth of analysis, as attempting to tackle both classes leads to a more superficial investigation. A more focused approach, concentrating on a single class—such as AA models—would result in a stronger, more detailed exploration, particularly since most tasks appear to be better suited to this domain.
- The dataset used in the paper appears ill-suited for NA models, as the tasks primarily target protein sequences rather than regulatory or non-coding regions of the genome. This limits the applicability of the dataset to AA models, reducing the relevance and transferability for broader genomic applications.
- The DGEB datasets are relatively small and designed primarily for assessing zero-shot performance, where no training data is provided. This narrow scope restricts the benchmark’s utility to specific use cases, such as evaluating pre-trained models, rather than facilitating a comprehensive analysis of model performance across varied settings.
- The benchmarking of models could be expanded to include a wider range of architectures. A deeper exploration of model performance would enhance the analysis, providing more meaningful comparisons and insights into the strengths and limitations of different approaches.
- For each of the datasets, it would be helpful to provide more detailed explanations of the intended learning objectives and their relevance. This additional context would clarify the specific goals of the tasks and highlight their importance within the broader scope of the study.

### Questions
- What does the performance on these tasks/datasets reveal about the models tested? As presented in Appendix G, it seems that no single model performs best across all tasks. Therefore, a way to interpret these results seems necessary for this benchmark. 
- Given the large number of protein language models, what is the justification for benchmarking the models that were picked (over others such as AlphaFold, pLM-BLAST, etc)? 
- What measures are being taken to prevent data leakage in the tasks that have training data?
- For the convergent evolution classification dataset, how do you determine/enforce enzymes having different evolutionary histories?
- Given that phylogenies can only be inferred, how robust is the phylogeny construction to perturbations in parameters/underlying construction algorithm?

### Soundness
3

### Presentation
2

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
The authors propose a novel benchmark, diverges genomic embedding benchmark (DGEB) which proposes embedding evaluation across a variety of tasks across a diverse set of species. The authors posit that previous evaluations consisting primarily of DMS data and biophysical property evaluation is insufficient. Instead they propose an evaluation across different modalities consisting of RNA, and Protein allowing investigation of embedding qualities across different modalities. The authors have an extensive focus on investigating interacting subunits of specific protein families across different phylogenetic branches. Using this dataset the authors investigate capabilities of genomic and and protein language models. They assess impacts of scaling and comparing across modality performance.

### Strengths
- The authors perform homology deduplication when splitting the dataset based on sequence similarity. This is an important step to avoid information leakage. 
- The authors are able to perform evaluation across different modalities benchmarking capabilities of current genomic language models against pLMs.
- The authors expand evaluation beyond eukaryotes and extensive assess representation qualities of sequences from archea and bacteria.

### Weaknesses
- The authors don't propose simple baselines such as supervised models or naive approaches for any of the tasks. 
- One of the gaps identified is function takes place across diverse scales. There is no SNP evaluation and most of the evaluation consists of assessing homology relationships between sequences across different species or within.
- Overall I think a double blind venue isn't the right setting for evaluation of dataset papers since it requires assessment of the codebase. In addition it requires deep domain expertise in the subject area which is impossible to assess in a double blind setting. Out of the 4 design principles of DGEB it is impossible to assess 3 consisting of simplicity, extensibility and reproducibility. I think the quality of this work would be best assessed in a journal or a datasets and benchmarks track.
- The authors aren't very precise with their vocabulary:
	- "For BiGene Mining, we curated functionally analogous sequences found in two phylogenetically distant taxa (e.g. Bacteria and Archaea) or interacting paralog pairs in sets of orthologous sequences. "
	- Paralogs are homologous sequences related through duplication events. Orthologs are sequences related through speciation events. When the authors describe interacting genes do they mean physically interacting? If so what kind of evidence is used to support this claim?

### Questions
- "this metric cannot be used to determine how well a model can abstract evolutionary and functional relationships between non-homologous proteins."
	- I don't really understand what the authors mean by this sentence. Typically the goal of a benchmark is to assess function from sequence why would we want to abstract it?
- "These are important properties, they are too coarse in scope to evaluate whether a model has learned biologically meaningful functional information."
	- Is there evidence to support this claim?
- The authors argue that there is currently a eukaryotic bias in the evaluations available. I wonder if it's such a bad thing since most of the applications that we are interested in have to do with perturbing eukaryotic organisms? 
- Another question that is important to mention in any sequence based evaluation benchmark  is that biotechnology is continuously improving and some of the measurement methodologies currently are flawed. What is the role of batch effects in constructing these datasets? How will these evaluation datasets continue to improve overtime as our ability to measure cell properties keeps improving?

### Soundness
2

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
4

### Summary
The authors present the Diverse Genomic Embedding Benchmark (DGEB), an embedding benchmark inspired by benchmarks for natural language embeddings. DGEB includes six different tasks (BiGene mining, classification, pair classification, evolutionary distance similarity, clustering, and retrieval) across 18 expert-curated datasets selected for phylogenetic diversity, difficulty, and variety of biological significance. Notably, DGEB works for any datasets that generate embeddings, including both nucleotide and amino acid variants, causal and masked models. Special attention is paid to including non-representative species and ensuring that train-test leakage is minimized. The package is open-sourced and designed to be easy to use, including a public leaderboard to determine the models with the best embeddings performance; it can also be extended by adding further datasets to the HuggingFace hub. The authors start off by benchmarking ESM-2, ESM-3, ProGen2, ProtTrans, Nucleotide Transformer, and Evo, and do not identify a single best model: they do find, however, that many tasks show performance saturation at low accuracies, suggesting there is something inadequate in the way that current models are trained.

### Strengths
* **Benchmark design.** The benchmarks in the paper are, in general, carefully thought out and sensible. Here are some particular highlights:
  * **Task diversity.** The paper covers a variety of biologically important benchmark tasks (rather than, for instance, just focusing on classification). 
  * **Phylogenetic diversity.** The authors take great care to expand the benchmark tasks to include a diverse set of sequences from across the tree of life, allowing us to evaluate the extent to which genomic models generalize to underrepresented clades.
  * **Leakage considerations.** By deduplicating sequences in the benchmark datasets, and enforcing a 10% nucleotide identity cutoff between train and test sets, the authors minimize the risk of train-test leakage in datasets. This is a crucial, biologically-informed modification to conventional NLP embeddings benchmark that needs to be enforced in genomic contexts.
* **Modularity and usability.** The DGEB is designed to be easy to run on any model that generates embeddings, has a community leaderboard, and can even be extended with new benchmarks on the Huggingface Hub.
* **Comparison between AA and NA modalities.** This benchmark dataset allows us to compare and evaluate NA and AA modalities on four datasets. To the best of my knowledge, this is the first benchmark to facilitate such a comparison.

### Weaknesses
* **Evaluating base models**. While it seems reasonable to assume that more capable models will have geometries that more closely match various problems, this seems far from guaranteed. It could also be the case that more capable models fine-tune better, or even that more capable models have embeddings that more closely match the geometry of individual problems *in a specific subspace* (in which case one would need to learn the correct projection to make use of their embeddings). I realize addressing these possibilities is out of scope, but a more explicit discussion of these limitations is warranted.
* **Distance-based evaluations**. Many of the benchmark tasks make assumptions about the metric structure of the embedding space. There is some arbitrariness/inconsistency in what distance function is used where; in practice, the correct distance function is likely to depend on the specifics of the model architecture (e.g. Euclidean distances are not consistent in transformer models that use LayerNorm). Here are the different distance functions employed:
  * BiGene mining uses cosine similarity to a query gene
  * Evolutionary distance similarity uses a "kitchen sink" approach, comparing cosine, Euclidean, and Manhattan distances of embeddings to ground-truth distances (defined as the branches of a phylogenetic tree)
  * Multiclass, multi-label classification uses $k$-nearest neighbors classifiers, which assume a distance function (which distance function is used here? Presumably Euclidean?) 
  * Pair classification uses cosine similarities, as well as a bunch of other distance metrics
    * Side note: it's a bit unclear exactly how the pair classification works. It seems that several distance functions are tried, but cosine similarity is privileged as the main metric---am I correct in my assessment?
  * Clustering uses $k$-means, which presumably uses Euclidean distance to assess the distance to each centroid
  * Retrieval uses cosine similarity
* **Fundamental limitation of using Euclidean embeddings**. For evolutionary distances, Euclidean space will be unable to accommodate the exponential growth of neighborhoods (this has motivated the development of hyperbolic embeddings for evolutionary distance estimation). The theoretical minimum distortion of phylogenetic distances is nonzero for Euclidean embeddings (and depends inversely on dimensionality)—thus, a benchmark like this is inherently limited. To this end, the authors could do a few things:
  * Acknowledge the fundamental limitations of evaluating evolutionary distances with Euclidean embeddings, and argue for the usefulness of their benchmark as a proxy/benchmark in spite of this.
  * Characterize the effect of the embedding dimension on evolutionary distance estimation accuracy.
* **Layer choice.** Another limitation of their method is the arbitrary choice of final/middle-layer embeddings. It is not clear which layer is chosen as "mid" when the size of the model fluctuates; I also wonder how influential this choice of "mid" layer is on model performance (e.g. does choosing layer 8 versus 9 make a substantial difference? How does this interact with model choice)
* **Results presentation.** The presentation of the results across many subfigures, rather than a single benchmark table, is somewhat confusing. I recommend cleaning up the table in Appendix G and including it as the main result. (This may be more of a matter of personal preference)
* **Justification.** The authors have done a great job of motivating their choice of datasets and benchmark tasks. In light of the above comments, I would like to see the authors justify the specific design choices that go into the *architectures* they use as well, and comment on the limitations of these approaches.

### Questions
* Are there any regression tasks that can be included in this benchmark? What would a reasonable, general framework for training and evaluating a simple regression model on top of these embeddings look like?
* Same question, but for link prediction.
* Is it possible to measure/quantify, or at least demonstrate, the effect of not enforcing the phylogenetic train-test split?

### Soundness
4

### Presentation
3

### Contribution
3
