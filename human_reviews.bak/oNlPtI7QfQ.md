# Embed-Search-Align: DNA Sequence Alignment using Transformer models

- Decision: Reject
- Scores: 3, 3, 6

## Abstract
DNA sequence alignment involves assigning short DNA reads to the most probable locations on an extensive reference genome. This process is crucial for various genomic analyses, including variant calling, transcriptomics, and epigenomics. Conventional methods, refined over decades, tackle this challenge in two steps: genome indexing followed by efficient search to locate likely positions for given reads. Building on the success of Large Language Models (LLM) in encoding text into embeddings, where the distance metric captures semantic similarity, recent efforts have explored whether the same Transformer architecture can produce numerical representations for DNA sequences. Such models have shown early promise in tasks involving classification of short DNA sequences, such as the detection of coding- vs non-coding regions, as well as the identification of enhancer and promoter sequences. Performance at sequence classification tasks does not, however, translate to sequence alignment, where it is necessary to conduct a genome-wide search to successfully align every read. We address this open problem by framing it as an ``Embed-Search-Align'' task. In this framework, a novel encoder model DNA-ESA generates representations of reads and fragments of the reference, which are projected into a shared vector space where the read-fragment distance is used as a surrogate for alignment. In particular, DNA-ESA introduces: (1) Contrastive loss for self-supervised training of DNA sequence representations, facilitating rich sequence-level embeddings, and (2) a DNA vector store to enable search across fragments on a global scale. DNA-ESA is $>97\\%$ accurate when aligning $250$-length reads onto a human reference genome of $3$ gigabases (single-haploid), far exceeds the performance of $6$ recent DNA-Transformer model baselines and shows task transfer across chromosomes and species.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces an embedding for DNA sequences based on transformer models. This embedding allows a distance between two DNS fragments of potentially differing lengths to be computed. Using this distance, an alignment algorithm is constructed under the usual seed-align framework, with the embedding distance taking the place of the seeding step. The embedding quality is assessed against other transformer models on synthetic reads.

### Strengths
The problem being tacked is an important one, and the proposed architecture does appear to have some benefits over existing transformer models. The ability to handle varying fragment lengths makes the method very flexible.

### Weaknesses
The experimental validation is the weak point of this paper. There are claims of efficiency yet no experimental evidence of any resource requirements and scaling such as time and memory. Furthermore, comparitive experiments are against existing transformer models and hence show that the embedding is superior to existing ones for alignment, but not that this is a good aligner overall. There are no comparisons against standard alignment algorithms (though they are referenced). There are no experiments on real data, or evaluation of the effect on downstream variant calling.

Generalisation to other species is very interesting, however the chosen species are all closely related.

The constraint in equation 4 cannot strictly hold on real data due to homology and repeat regions.

### Questions
- Have you evaluated the method on real data?
- What are the runtime resource requirements?
- Does the model generalise to less related organisms? Presumably there will be some dependence on degree of homology.

### Soundness
1 poor

### Presentation
3 good

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
This manuscript tries to solve an alignment problem. The input are a set of short DNA sequences (reads) and a long reference sequence. The task is to find the location of each short read on the reference sequence. This manuscript proposes to solve this matching problem by projecting both short reads and reference (sub-)sequence to a shared embedding space such that reads originated from the same reference sequence region are close to each other in the embedding space. For a new read, its embedding is first calculated using the NN, after that it searches for the nearest K reference fragments in the embedding space. Exact alignment using classical Smith-waterman algorithm is between the given read and each of the K nearest reference fragments to identify the final mapping position of the read.

Regarding implementation, the authors first split the reference sequence (length = 3Gb) into fragments (length = 1250bp). Then each fragment and their substrings (reads) are assigned the same label, which (a batch of several fragments and their substrings) is then fed into a transformer with a contrastive loss. The trained network can then project a given DNA sequence to the embedding space (dimension = 384). 

Although I think the idea is somewhat interesting, the authors may want to address the following points in the next version.
1. Provide support to the hypothesis that a fragment and its subsequences are close in the embedding space. Figure 1 only contain the fragments, but it should also contain random subsequences. 
2. Make the notations more readable, e.g. in Eq. 2, q is used in SA(r,R). Eq. 6, really difficult to guess the meanings.
3. Comparison with classic aligner such as BWA or bowtie. I will expect a >99.9% mapping rate of BWA or bowtie for the used data.
4. Expand the comparison dimensions. Only Recall is used, could add precision, mapping rate etc.
5. Compare on real data. The simulated data is too clean. Phred 30 is already 99.9% identical to the original sequence, that is 250/1000=0.25 bp mutation per read. It will be nice to see how the method works on real data where the error rate is higher and the read length varies.
6. Provide details of the transformer. Do you do paddings to short reads?
7. What happens to negative samples, i.e. a random read that is not similar to any reference fragment? How do you decide the boundaries of a reference fragment in the embedding space?
8. Reorder the figures so it flows with the text.
9. Appendix A, add legend for the lines
10. Appendix B, B.3 the complexity is G, not log(G) as you have to compare with all reference fragments.

### Strengths
NA

### Weaknesses
NA

### Questions
NA

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper presents EMBED-SEARCH-ALIGN, a method for DNA sequence alignment using transformer models.  Unlike traditional methods use algorithmic solutions like indexing and efficient searching to align reads. This paper proposes a different paradigm using transformer models to learn representations of DNA reads and reference genome fragments, and performing alignment based on similarity of these embeddings.

The main contributions of the proposed architecture are as  follows. First, the authors use a DNA sequence encoder DNA-ESA that is trained using self-supervision and contrastive loss to produce DNA sequence embeddings optimized for alignment. Next, the authors use a DNA vector store to enable efficient nearest neighbor search across the entire reference genome for each read. Finally, the authors formulate the sequence alignment as an "embed-search-align" task using the encoder and vector store.

In the experiments, the authors demonstrate that DNA-ESA can align 250 bp reads to the human reference genome with over 97% accuracy, exceeding several transformer baseline models. The approach also demonstrates ability to generalize - a model trained only on Chr 2 can still align reads from other chromosomes and even other species.

The paper argues this approach mitigates limitations of prior works on transformer models for genomics and provides sequence representations suitable for alignment. It also enables "flat search" over reads and reference fragments of different lengths. Future work is discussed to improve computational efficiency.

### Strengths
+ The authors present a novel approach to align sequence reads which can provide new possibilities for DNA sequence representation and search.
+ The proposed DNA-ESA encoder learns effective sequence embeddings for alignment, and outperforms several baseline transformer models designed for specific genomics tasks.
+ The approach is promising and demonstrates ability to generalize to new sequences not seen during training, like different chromosomes and even new species. Furthermore, formulating alignment as embed-search-align could enable new capabilities like "flat search" over reads and reference fragments of different lengths.

### Weaknesses
- I felt that the paper is a very dense read for the general ML audience at ICLR for folks who do not have DNA sequencing background, and it will be great to make the paper more accessible.
- The embedding approach currently shows promising results on simulated data, but needs more evaluation on real sequencing data.
- The performance for short reads is worse than long reads, given that short reads are more commonly used, this may affect how this system can be actually used.
- Limited demonstration of applications in downstream genomic tasks.

### Questions
1) Can the authors comment on how this paper is a good fit for ICLR, and the steps the author may take to make this paper more accessible to the ML audience?
2) How does the model perform on real world sequencing data?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
