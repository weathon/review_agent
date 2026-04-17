# Binary Node Clustering via Contrastive Learning for Haplotype Phasing in de novo Genome Assembly

- Decision: Reject
- Scores: 2, 4, 6

## Abstract
Accurate haplotype phasing is essential for high-quality genome assembly, yet de novo phasing without parental data for complex genomes remains a challenge. We formulate phasing as a binary, overlapping node clustering problem on unitig graphs where nodes represent contiguous, non-branching DNA sequence fragments and different edge types capture sequence overlaps as well as Hi-C proximity information. To solve this problem, we design a contrastive learning framework with custom objective functions and train a graph-transformer-based model termed grapHiC to distinguish nodes with paternal, maternal, or homozygous haplotypes. We show that grapHiC significantly outperforms other node clustering methods on genome-sized datasets and that grapHiC’s predictions can successfully guide de novo genome assembly, producing well-phased assemblies across diverse human genome assembly graphs using the DipGNNome assembler. Our code, trained model, and dataset are available at https://anonymous.4open.science/r/graphic_iclr-688D/ (repository anonymized for peer review).

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces a graph-transformer model, grahpHiC for de novo haplotype phasing in de novo genome assembly. The model assigns values to nodes corresponding to untigs, and subsequent node clustering is used to reconstruct genomes for each haplotype. The graph-transformer model is trained using simulated data with the proposed loss function, supervised contrastive pair (SBP) loss to learn the node values. This is further regularized with an auxiliary loss to improve convergence. Results on experimental data show that global SBP loss usually yields the most accurate phasing; but D-grapHiC usually trails Hifaism in performance when assembling genomes.

### Strengths
The key contributions in this work are:

1.	The formulation of the SBP loss function
2.	The graph-transformer model to learn the values/embeddings for each untig node

These are simple, yet possibly useful contributions to the space of phased de novo genome assembly methods. Additionally, the application of graph transformers to the alignment-free setting appears to be a novel contribution.

I would also like to acknowledge the effort made by the authors to provide the reader with some biological background in the introduction and related work sections; this is particularly important for the manuscript to be self-contained and understandable by a reader lacking the requisite background in computational biology.

### Weaknesses
There are several clear weaknesses in this paper in its current form. Firstly, the paper initially positions graphic as a de novo genome assembler, but it is effectively only learning untig values to be assembled by some other method (DipGNNome in this paper). This has to be clarified and raises other potential questions, such as whether alternative clustering methods could be effective at assembling genomes after learning the node values.

Moreover, the current landscape of haplotype phasing and de novo genome assembly is somewhat misrepresented. Examples of this are listed below.

1.	Hifaism is not mentioned as a method for phased de novo genome assembly despite being used as a benchmark later.
2.	In the paragraph starting at line 173, the authors state that reference-based haplotype phasing methods have been evaluated on much smaller datasets than is typical of real eukaryotic genomes. However, CAECSeq and XHap (Consul et al, 2023) both demonstrate haplotype assembly on real chromosomal data.

This work also raises a conceptual question – the loss function necessitates the presence of ground-truth labels as it is a supervised loss function. These labels are obtained by virtue of the training data being simulated. With a motivation for the proposed work being that reference-based haplotype assembly methods are biased towards the chosen reference, it is not obvious that the use of simulated data when training grapHiC does not implicitly also introduce a bias towards the reference from which the training data is generated. 

Finally, the shortfall of performance of grapHiC vs. Hifaism raises the question of whether the use of machine learning (graph transformer) brings about any tangible benefit in tackling the problem of haplotype-phased de novo genome assembly.

### Questions
**Questions:**

1.	ONT reads typically have much higher rates than HiFi reads. How does the performance of the proposed approach vary when ONT reads are used in place of HiFi reads?
2.	What are heterographs (line 106)?
3.	Why is the acronym of “Supervised Contrastive Pair” loss taken to be “SBP”? Should it not be “SCP”?
4.	On line 156, the authors state the this is “the first method capable of phasing raw untig graphs”. As per my understanding, most de novo genome assemblers entail the construction of assembly graphs (k-mers or untigs). How is the described setting different from that of these other works?
5.	Why were other haplotype-phased de novo genome assemblers, such as Hifaism, Falcon-Unzip and HiCanu omitted when describing related work?
6.	Is the random sampling used in the global and local SBP losses fixed across epochs?
7.	What are the parameters used to simulate reads using PBSim3? Also, PBSim3 does not generate HiFi reads; rather, it generates CLR reads that have to be passed to PacBio CSS to generate HiFi reads.

**Suggestions:**

1.	The summary of contributions (lines 155-160) is fairly repetitive as points 2 and 3 are accomplished as part of the first point.
2.	The model architecture in Section 5 would be better described through a pictorial representation. This would improve the readability of the manuscript.

### Soundness
2

### Presentation
2

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
This paper proposes a graph-based deep learning framework for haplotype phasing in de novo genome assembly. The method, named grapHiC, operates on assembly or unitig graphs, where nodes represent sequence fragments and edges encode sequence overlap.
The main idea is to cast phasing as a binary node clustering problem and train a contrastive graph neural network that separates nodes belonging to different haplotypes while preserving local contiguity. Experimental results on both simulated and real sequencing datasets show that grapHiC achieves high N50 and low switch error rates.

### Strengths
The paper addresses an important and technically challenging bioinformatics problem, reference-free haplotype phasing, using modern graph representation learning. The presentation is clear and well-organized, making the ideas accessible to both ML and bioinformatics audiences. The binary node clustering formulation is intuitive and makes a clear connection between genome assembly graphs and graph learning objectives.

### Weaknesses
The conceptual novelty is somewhat limited: while grapHiC differs from prior graph-based phasing methods such as GAEseq and NeurHap by operating in a reference-free de novo setting, the underlying modeling approach remains similar. 

No deeper theoretical grounding or interpretability of the framework is offered.

The experimental evaluation is limited in scope. Table 1 compares grapHiC to generic graph-clustering algorithms (Spectral, Louvain, etc.), which are not meaningful phasing baselines; this shows improvement over standard community detection rather than true domain-specific state-of-the-art methods. Table 2 is also ambiguous: if I understand its use here correctly, hifiasm is used to generate the assembly graphs and to produce the reference phasing statistics, so the reported "comparable performance" might simply mean that grapHiC can reproduce hifiasm’s own phasing on those same graphs (i.e., not that it matches hifiasm as an independent de novo phasing method).

### Questions
The paper should include comparisons to domain-relevant de novo phasing tools (e.g., DipGNNome and a few others) to better establish novelty.

The role of hifiasm in the pipeline should be explained in more details (e.g., is it a pre-processing step or a competing phasing baseline).

It would be helpful if there is a deeper discussion about how the proposed loss fundamentally differs from standard contrastive or cross-entropy losses used in graph partitioning.

How much does performance depend on the Hi-C linkage density or noise level?

Scalability metrics (runtime, memory) and sensitivity to graph size or read coverage should be reported.

A discussion of potential generalization of the approach to polyploid or metagenomic assembly graphs would be beneficial.

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
The authors introduce a new algorithm for genome phasing, i.e., splitting a genome into haplotypes. The key advantage of their approach is that it does not rely on a reference sequence. Instead, it leverages a Hi-C graph using a graph transformer to perform the phasing based on the idea of an enrichment of contacts on the same haplotype . The method is applied to the unitig graph generated by the hifiasm assembler.

### Strengths
1. High novelty (use of Hi-C + graph transformer + bespoke loss functions), and great application of machine learning to the key problem of phasing in genomics (without a reference genome)
2. Strong results in benchmarking
3. Good ablations and baselines against traditional algorithmic approaches for clustering

### Weaknesses
1. There is a degree of circularity in the approach. On the one hand, the method is intended for use in settings where no reference genome is available; on the other hand, it is trained and benchmarked using labels derived from reference genomes. While the results are still strong, I would have liked to see an example where the model is trained on one species and then applied to reads and Hi-C data from a different species.
2. I might have missed it, but the main body of the manuscript doesn't include in the benchmarking reference-only phasing baselines

### Questions
1. Can the authors attempt to produce an assembly for a different species using the model trained on human data?
2. Please add a baseline from phasing using only a reference genome

### Soundness
3

### Presentation
2

### Contribution
3
