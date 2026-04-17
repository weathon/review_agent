# GENATATORs: ab initio Gene Annotation With DNA Language Models

- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Inference of gene structure and location from genome sequences - known as de novo gene annotation - is a fundamental task in biological research. However, sequence grammar encoding gene structure is complex and poorly understood, often requiring costly transcriptomic data for accurate gene annotation. In this work, we revisit standard evaluation protocols, showing that commonly used per-token and per-sequence metrics fail to capture the challenges of real-world gene annotation. We introduce and theoretically justify new biologically grounded interval level metrics, along with benchmarking datasets that better capture annotation quality. We show that pretrained DNA language model (DNA LM) embeddings do not capture the features necessary for precise gene segmentation, and that task specific fine-tuning remains essential. We comprehensively evaluate the impact of model architecture, training strategy, receptive field size, dataset composition, and data augmentations on gene segmentation performance. We show that fine-tuned DNA LMs outperform existing annotation tools, generalizing across species separated by hundreds of millions of years from those seen during training, and providing segmentation of previously intractable non-coding transcripts and untranslated regions of protein-coding genes. Our results thus provide a foundation for new biological applications centered on accurate and scalable gene annotation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents GENATATORs, an ab initio gene annotation approach built by fine-tuning long-context DNA language models to segment genomic DNA including mRNAs and lncRNAs. The authors argue that token-level metrics are insufficient and introduce interval-level and gene-level criteria to better reflect biological correctness and study architecture and training choices. Results show strong improvements on lncRNA and UTR segmentation and competitive overall gene-level scores.

### Strengths
1. GENATATORs tackles a de novo annotation with long-context DNA LMs and emphasizes interval and gene-level metrics aligned with biological correctness.

2. Solid ablations show clear contributions from long context, multi-species training, and RC augmentation.

3. Strong performance on lncRNA and UTR where HMM-style tools underperform and potential for first-pass annotation in poorly annotated species.

### Weaknesses
1. Coverage of recent DNA-LM foundations remains incomplete like Evo.

2. The writing style is lacking a smooth logical narrative, and the reading experience is not friendly enough.

3. Beyond stating frameshift risk, quantify downstream protein-level impact when boundaries shift by 1–3 bp and compare to token-level metrics to demonstrate practical delta.

4. Keep claims strictly as validation. For NMD, report total counts of qualifying genes, baseline, and statistical significance.

### Questions
Please see Weaknesses

### Soundness
1

### Presentation
1

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
This paper introduces a family of fine-tuned DNA language models on the gene annotation task. The authors critique existing evaluation metrics, and propose a new interval-level metric to show that fine-tuned DNA language models outperform classical HMM-based models and SegmentNT. The models also generalize to evolutionarily distant species, indicating cross-species generalization.

### Strengths
1. The paper aims to tackle a fundamental task for genomics, and the ability to annotate non-coding genes and UTRs shows real progress compared to previous works.

2. The paper designs novel and rigorous evaluation metrics, the interval and gene-level metrics seem capable of capturing biologically meaningful aspects of gene segmentation performance.

3. The models demonstrate strong cross-species generalization, showing the potential of fine-tuned DNA language models to capture evolutionarily conserved sequence patterns across diverse genomes.

### Weaknesses
1. Even though the authors show that increasing the input context improves model performance, the paper lacks an analysis of performance across different gene lengths, which would clarify whether the models exhibit bias toward genes of certain lengths or whether the longer input context specifically helps in learning longer or more complex genes.

2. In the cross-species generalization task, it seems that the best models are different for different species. The paper lacks an analysis of this phenomenon. In general, it would be better for the authors to provide guidance for readers on how to choose which type(s) of DNA-LM to use in different scenarios.

3. The work’s main novelty resides in the proposed evaluation metrics and systematic benchmarking; however, the modeling methodology largely follows existing DNA LM fine-tuning frameworks, making the technical contribution relatively weak.

### Questions
1. Is there any specific reason to choose chr 8, 20, 21 for validation/test?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents strategies for large scale gene annotation. The authors introduce a new interval-based evaluation metric, and provide theoretical justification that this more reliably captures annotation quality. They proceed by fine-tuning various pre-trained DNA language models, investigating the role of modeling design choices such as architecture and training procedures. Finally, they show that fine-tuned models outperform existing gene annotation tools, in particular in an out-of-domain setting.

### Strengths
**Originality** The authors present a novel segmentation scoring procedure, which appears to better capture DNA annotation quality than current metrics. 

**Quality** The introduced metric appears well thought through, and the fine-tuning benchmarks are comprehensive, allowing them to draw  conclusions on best practices in the field. 

**Clarity** The paper is well-written and easy to follow. 

**Significance** The paper presents a convincing case for the potential of DNA LMs for gene annotation, which is likely to have impact on the community.

### Weaknesses
The paper seems to want to do several things at once: The title suggests a new prediction method (GENATATORS), but the focus in the paper itself is split primarily between discussion of a new segmentation metric, and a fine-tuning benchmark study. Any of these would be interesting, but maybe not all for the ICLR audience. The authors should consider whether they can give the paper a clearer focus, and whether ICLR is the right venue for it (compared to e.g. a Bioinformatics journal)

Since the segmentation metric is central to the paper, it would be helpful if the authors could make it clear how the presented method differs from that presented in Scalzitti et al 2020, which they cite as inspiration. It would also be helpful to the reader if the score could be contrasted clearly to the gene-level score used in Tiberius. If there is sufficient history here, it might even make sense to dedicate a pargraph in Related Work to this.

I have some concerns regarding the claim made on line 376, related to Fig 1. The authors state. *"Specifically, the GENA-based GENATATOR marginally outperforms Tiberius, while the Caduceus-based variant performs slightly below it."*. As far as I remember (correct me if I'm wrong), Tiberius is not designed to predict lncRNA, so it seems unfair to compare it against GENA that does, and then lumping together the results in both categories. As far as I can see, on the mRNA case, Tiberius clearly outperforms GENA, and it is therefore not correct that GENA "performs on par with the current -state-of-the-art model".

In my opinion, another weakness of the paper is the almost aggresive stance towards earlier work in this field. The BEND benchmark is descibed as "not biologically rigorous", and Tiberius is claimed to "consistently fail" for lncRNA, although it was not trained on this task (as far as I remember). In science, we continuously seek to improve over earlier work, but it should be sufficient to highlight own merits rather than describe earlier work in derogatory terms.

### Questions
## Questions

line 164. *"To account for this ambiguity, we use a gene level rule that accepts a prediction as correct when the predicted interval set exactly matches the interval set of any annotated isoform of the target gene."*. Is this generally reliable? How complete is our ground truth annotation of isoforms?

line 224. *"none of the models produced embeddings containing sufficient information for accurate gene segmentation"*. I was a bit puzzled about this conclusion. As far as I remember, reasonable performances have been reported for gene annotation from fixed embeddings in the past - e.g. in BEND. You state *"These observations are consistent with recent findings reported in the Nucleotide Transformer embeddings"*. Which results are you referring to here?

line 236. *"Together, these results indicate that pretraining alone is insufficient to encode the features required for precise gene segmentation and that task-specific fine-tuning remains essential for achieving high segmentation accuracy."*. Are you sure that your results warrant this strong conclusion? It might be valid for the simplistic linear mapping that you use here, but earlier results show that better performance can be obtained by more structured decoding strategies, such as the one presented here: https://arxiv.org/abs/2505.03377. It is also not unlikely that future DNA LMs will incorporate gene structure in the pretraining, and the statement you make should therefore at least be qualified to *current* language models.

line 269. *"We also found that using multiple isoforms per gene slightly reduced accuracy, confirming that the single-isoform strategy remains preferable"*. Is it a general recommendation that we should ignore multiple isoform information. Does it not instead suggest that the model you are using is not rich enough?


## Minor details

The references to figures seems unnecessarily elaborate ("Appendix A Figure A1 B"). For clarity, consider shortening them.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The manuscript introduces GENATATORs, a method for ab initio gene annotation using DNA language models (LMs), designed to segment gene structures such as exons, introns, and UTRs in mRNA and lncRNAs. The method aims to address challenges in accurately annotating gene structures and identifying non-coding genes (e.g., lncRNAs) that are often missed by traditional tools. GENATATOR achieves this by training DNA LMs on long genomic sequences (up to 250 kbp) and leveraging novel biologically grounded evaluation metrics that measure gene segmentation accuracy at the interval and gene levels. The method is evaluated on cross-species benchmarks, comparing its performance to existing tools like AUGUSTUS and SegmentNT, showing superior gene recovery and generalization to new species. GENATATOR excels in segmenting mRNA and lncRNAs, but its limited biological completeness (e.g., lack of cis-regulatory elements and alternative splicing prediction) and reliance on a single isoform per gene are key drawbacks.

### Strengths
### Cross-Species Generalization

A major strength of GENATATOR is its ability to generalize across species, with demonstrated strong performance on unseen species, including plants and animals. This cross-species generalization is particularly important for genome annotation, as it can be applied to less-characterized species without the need for species-specific tuning.

### Weaknesses
### Limited Biological Completeness

While GENATATOR performs well in annotating mRNA and lncRNA, it is biologically incomplete because it fails to predict key regulatory regions such as promoters, enhancers, and silencers. These cis-regulatory elements are essential for understanding gene regulation, as they control when and how genes are turned on or off in different cell types. Additionally, small non-coding RNAs (such as miRNAs, snoRNAs, and snRNAs) are not included in the predictions, which limits the model’s utility in complete genome annotation. This is a major drawback for using GENATATOR in comprehensive genome-wide annotation tasks where understanding gene regulation is critical.

### Alternative Splicing and Isoform Prediction

The single-isoform assumption is another significant limitation. GENATATOR currently predicts only one canonical isoform for each gene, ignoring alternative splicing and isoform diversity, which are essential for understanding gene function and cell-type-specific gene regulation. Many genes undergo alternative splicing to produce different protein isoforms or regulatory RNAs, and GENATATOR’s static model fails to capture this complexity. Incorporating the ability to predict multiple isoforms for each gene would make the model far more biologically accurate and useful in functional genomics.

### Static Gene Model Assumption

GENATATOR assumes a fixed, static gene structure, which does not account for dynamic gene expression across different cell types or tissues. Gene regulation is highly tissue-specific, and many genes are expressed differently depending on the cell type. By not accounting for these cell-type-specific variations, GENATATOR fails to model the functional complexity of genes, which is crucial for understanding disease mechanisms and gene regulation in different contexts.

### Questions
See Weaknesses. 

In addition, I'm curious to learn more about the runtime performance of the proposed method versus existing methods like SegmentNT.

### Soundness
3

### Presentation
3

### Contribution
2
