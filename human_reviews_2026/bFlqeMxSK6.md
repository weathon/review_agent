# The Human Genomics Long-Range Benchmark: Advancing DNA Language Models

- Avg Score: 6.00
- Decision: Reject
- Scores: 6, 4, 8

## Abstract
The advent of language models (LMs) in genomics necessitates benchmarks that can assess models’ capabilities and limitations. In contrast to protein models, DNA LMs can be used to study non-coding regions of the genome and must account for unique challenges, especially interactions across long sequence lengths. However, existing benchmarks for DNA LMs are defined over short sequence datasets and can involve tasks that are not considered to be biologically meaningful. Here, we present the Human Genomics Long-Range Benchmark (LRB), which focuses on biologically meaningful tasks and supports long-range contexts. We complement our benchmark with fine-tuning recipes that meaningfully improve performance. We evaluate DNA LMs across nine compiled human genome tasks and observe that they achieve competitive performance relative to supervised baselines on several tasks (e.g., genome annotation), but there remains a significant gap in domains, such as variant effect and gene expression prediction. Additionally, we introduce a visualization tool to examine model performance split by genomic properties.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors presented a new benchmark for evaluating DNA language models focused on the study of context sizes. The authors compiled a diverse set of long-range and short-range downstream tasks for DNA LMs such as variant effect prediction, gene expression, regulatory element and chromatin feature predictions. The tasks and datasets are clearly described. And it comes with some user-friendly features such as variable context size versions and visualization tools. Overall, I find it to be one of the better DNA LM benchmarks with a broad range of tasks and specific aims. I therefore recommend acceptance.

### Strengths
1. The authors selected a diverse and biologically meaningful array of downstream tasks.
2. The benchmark focuses on the study of context sizes, which is an important topic in DNA LM development and application. This benchmark can be expected to provide valuable insights to the field.
3. Details of the datasets are thoroughly documented.
4. The visualization tool looks helpful.

### Weaknesses
1. DART-Eval is another good DNA LM benchmark, but not mentioned in the paper. The authors should discuss how they are different and in which aspects LRB is potentially more useful.
2. More results on Evo-2 would be beneficial, since it has shown significant improvement in performance compared with previous DNA LMs. Would it be feasible for the authors to evaluate the transfer learning capability of it (i.e. fine-tuning), even with the 1B model?
3. It would be better to highlight the difference between ClinVar and OMIM benchmarks (missense vs. non-coding) in the task names, etc.

### Questions
1. In Table 3, Enformer is not a good baseline for the ClinVar task, since this set only contains missense variants, and Enformer is a model focused on the non-coding genome. 
2. In Table 4, why is DeepSEA chosen over Enformer for the last two tasks?

### Soundness
3

### Presentation
4

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
The paper presents a benchmark for assessing the quality of genomic language models, focusing on long-range dependencies. The datasets can be extracted from the benchmark at different context lenths, lowering the barier for conducting experiments at different context-length scales. Compared to earlier benchmarks, the paper also investigates more elaborate fine-tuning techniques, demonstrating improvements over earlier protocols. Finally, an analysis and visualization tool is included. Based on the benchmarks, the authors draw some conclusions on the state of the field.

### Strengths
**Quality** and **Clarity**. The authors motivate the problem well, making an effort to describe earlier work, and how it differs from the current contribution. The nine prediction tasks are presented at a reasonable level of detail, explicitly stating the biologically relevance and the long-range dependencies of each task.

**Significance** The paper addresses an important problem. The last years have seen a considerable increase in genomic language models, but it still remains unclear how well they perform compared supervised task-specific models. Comparing genomic language models head-to-head is also difficult - hampered by the fact that papers often report results on different experiments. It is therefore important to establish meaningful benchmarks to move the field forward.

**Originality** The paper introduces tasks that are believed to depend on long-range genomic interactions, which have been insufficiently addressed in previous benchmarks.

### Weaknesses
The main weakness of the paper is that it is seems fairly incremental. Several benchmarks for genomic language models already exist, and the current paper claims to distinguish itself by being biologically meaningful and focusing on long-range dependencies, but the BEND benchmark made similar claims when it came out. The other difference - partial fine-tuning vs full model fine-tuning - are interesting, but such observations would not normally be considered sufficient for a paper in a top machine learning conference.

As far as I can see, most of the tasks are copied from earlier benchmarks, and simply provided here with longer context lengths. Since this benchmark focuses on long-range effects, one would expect the authors to demonstrate that long range effects are critical in each of the nine sets, but they only seem to do so for the Bulk RNA task (figure 3).

More generally, we are seeing multiple new benchmarks for DNA language models coming out, and it is unclear to me whether they keep providing value to the community. One could argue that it would be more fruitful if a community-wide effort was made to consolidate to a set of agreed-upon tasks, rather than making incremental updates to existing ones. For example, another long-range benchmark called DNALongBench was recently proposed with similar benchmarks (but with some metrics that more directly addressed long-range dependencies).

### Questions
### Questions
The Bulk RNA task in Figure 3 shows clear context length dependencies. Do the other eight tasks show similar dependencies? And if not, does this mean that the long range effects stated in the "Long-Range" sections for the individual prediction tasks are not as critical as expected - or that current models cannot pick them up?

Table 4. Why was Evo2 not included here?

line 391. *"For DNA LMs to be useful for these tasks, they must also find a way to model and learn evolutionary pressures and conservation."*. The concluding statement was not quite clear to me. Why can't we just use the GPN-MSA model (or the more recent GPN-Star).

lin 451. *"most DNA LMs saw an increase in performance with increasing context"*. Does the increase in context window also result in more fine-tuned parameters, and if so, could this be a potential reason for the improved performance?

### Minor comments

line 056: the authors state *"Allowing users to select arbitrary sequence length inputs for any given dataset enables us for the first time to understand empirically the importance of long-range inputs for our proposed tasks."*. As far as I remember the BEND dataset also offered the possibility of extracting arbitrary flanking regions, so the "first time" in this sentence should perhaps we rephrased.

line 148. *"GTEx"*. Has this term been introduced?

line 213. Is it intentional that the "Pathogenic Clinvar" section has no "Long-Range" subsection?

line 230. *"are can advance"*. Remove "are".

line 236. *"Outputs are RPKM normalized"*. Has "RPKM" been introduced?

line 318. It would be helpful if the authors could make it clearer why GPN-MSA is considered a baseline, rather than a language models on par with the others. Is the use of MSAs incompatible with particular downstream tasks?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper presents the Human Genomics Long-Range Benchmark (LRB), which includes nine curated tasks to test how well DNA foundation models can handle long-range sequence reasoning. The benchmark combines data from well-known public sources such as ENCODE, FANTOM5, GTEx, and SCREEN, covering tasks like regulatory element detection, chromatin state classification, and gene expression prediction. One special but simple feature is that users can download DNA sequences of any length, because each task is defined by genomic coordinates instead of fixed-size sequence windows. This design allows fair comparison between models with different input sizes (for example, 2 kb vs. 131 kb) and makes it easier to study how model performance changes with longer contexts—something that older benchmarks could not support.

The authors also standardize all data to the GRCh38 reference genome, provide a visualization notebook and fine-tuning scripts, and show that full-parameter fine-tuning often gives better results than parameter-efficient tuning methods. Their experiments find that current DNA language models do well on local annotation tasks but still perform worse than supervised models like Enformer on gene expression prediction and alignment-based models on zero-shot variant effect tasks.

In summary, the paper does not propose a new model but makes an important contribution through a clear, flexible, and easy-to-use benchmark. The LRB transforms a simple data design idea into a useful and reproducible platform for studying long-range reasoning in human genomics.

### Strengths
1. Clear problem focus.
The paper clearly explains that most existing DNA language model benchmarks use short sequences and do not capture deeper biological meaning. LRB instead focuses on long-range genomic effects, which are more realistic and biologically important.

2. Well-defined tasks and data. (curated / repurposed from public available datasets)
The benchmark includes a diverse set of human-centered tasks with clear goals and detailed definitions. Each task is designed to test specific biological hypotheses related to long-range regulation.

3. Practical methodological insight.
The study shows that full fine-tuning improves model performance compared to head-only and PEFT, and recommends a clear procedure for others to follow. (however more in-depth discussion of why this is the case will further strengthen the paper more)

4. Helpful tools.
The provided visualization notebook and flexible sequence downloader make it easier to analyze errors, test different context lengths, and better understand model behavior.

5. Transparent evaluation.
The paper clearly reports where current DNA language models succeed and where they fall short: especially on long-range gene expression and zero-shot variant effect tasks, which highlighting the remaining performance gap with models like Enformer and alignment-based approaches. This helps set realistic benchmarks for future work.

### Weaknesses
1. Limited hyper-parameter tuning.
The authors pointed out the hyperparameter search is minimal, which can affect the ranking between models. Since the benchmark includes architectures of very different sizes and training dynamics, some performance differences might reflect suboptimal tuning rather than true capability gaps. (But this does not affect the quality of the benchmark itself, so a minor weakness)

2. Compute limitations.
Large models such as Evo2 are only tested in zero-shot mode on a subset of tasks. This limits how confidently we can generalize the results to those DNA models, and makes it difficult to evaluate whether fine-tuning scales consistently with model size. (again, not a weakness on the benchmark, but on the authors provided example results using the benchmark)

3. Task scope bias.
Although LRB is designed for long-range reasoning, most tasks involve bulk measurements or are centered on transcription start sites (TSS). This framing naturally benefits models like Enformer, which were trained with similar positional biases, and may underrepresent more complex regulatory dependencies such as distal enhancer–promoter interactions. 

4. Single-species focus.
The benchmark only includes human genomic data, without any cross-species or multi-genome comparisons. 

-- note on point 3 & 4 above --
My background is not biomedicine and my knowledge in this domain is limited. I cannot confidently judge how well the selected tasks covers or not covers all the bases.

5. Mixed training paradigms.
The evaluation compares self-supervised DNA LMs (trained with masked modeling) to supervised architectures like Enformer. While informative, this comparison mixes objectives and data regimes. A clearer discussion of how these paradigms differ and whether the gap reflects modeling power or training signal would strengthen the conclusions.

### Questions
1. On fine-tuning strategy:
a. The insight on fine-tuning strategy is intriguing, the reasoning given in the paper makes sense but is very brief (may be limited by the page limit). Have you analyzed which layers or representations benefit most from full fine-tuning compared to PEFT? (e.g. start by enabling LoRA  only on the top few transformer blocks, then progressively include deeper layers (e.g., top 2 → top 4 → top 8 → all layers).
b. Could adapter-style methods (e.g., LoRA or prefix tuning) recover similar gains if tuned for longer or with task-specific regularization?

2. On context-length scaling:
a. How do performance trends behave beyond 131 kb? Is there an observed saturation point where longer context no longer improves performance?
b. When users extract longer sequence windows (e.g., extending from 2 kb to 131 kb around a labeled locus), how does the benchmark ensure that these expanded regions do not overlap with labeled sites from the validation or test sets?

3. On task coverage and bias:
Each selected task is clearly described, but many are promoter-centered or TSS-centered, which mainly capture local regulatory signals. How do the authors ensure that these tasks are representative of broader long-range genomic phenomena? Have you considered including tasks such as enhancer–promoter interaction or 3D chromatin contact prediction, which also reflect truly long-range dependencies? My understanding of genomics is limited, so I am unsure whether the current task set fully covers the important biological contexts.

### Soundness
3

### Presentation
3

### Contribution
3
