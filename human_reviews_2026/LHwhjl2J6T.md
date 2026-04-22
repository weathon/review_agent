# TadABench-1M: A Large-Scale Wet-Lab Protein Benchmark For Rigorous OOD Evaluation

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Existing benchmarks for biological language models (BLMs) inadequately capture the challenges of real-world applications, often lacking realistic out-of-distribution (OOD) scenarios, evolutionary depth, and consistency in measurement. To address this, we introduce TadABench-1M, a new benchmark based on a wet-lab dataset of over one million variants of the therapeutically relevant TadA enzyme, purpose-built to embody these three essential attributes. Generated across 31 rounds of wet-lab evolution, it offers unparalleled evolutionary depth and naturally presents a stringent OOD challenge. To ensure measurement consistency across this extensive campaign, we developed Seq2Graph, a scalable graph-based algorithm that systematically unifies multi-batch experimental data. Our high-fidelity benchmark highlights a critical finding: while state-of-the-art BLMs excel on a standard random split of the data (Spearman’s ρ ≈ 0.8), they fail dramatically on a realistic temporal prediction task (ρ ≈ 0.1). This stark performance gap validates the importance of our benchmark’s design principles and suggests that evolutionary depth is critical for building models with realistic utility.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes TadABench-1M, a benchmark based on a wet-lab dataset of over one million variants of the therapeutically relevant TadA enzyme, aiming to focus on realistic out-of-distribution cases, evolutionary depth, and consistency in measurement. This paper also introduces Seq2Graph, a scalable graph-based algorithm that systematically unifies multi-batch experimental data. A key finding in this work is that while state-of-the-art biological language models excel on a standard random split of the data, they fail dramatically on a realistic temporal prediction task.

### Strengths
- The temporal split of the dataset in this benchmark establishes a realistic evaluation setting that mimics a real-world engineering campaign, and also reveals a significant generalization gap.
- This work demonstrates that sequence diversity and evolutionary depth are more critical for OOD generalization than raw data volume

### Weaknesses
- The benchmark is designed around TadA enzyme and it’s evolution. This setting may restrict the generalization to other protein families. Also, it’s not clear whether the findings in this benchmark can be transferred to other proteins.
- This paper mainly focuses on biological language models, such as ESM, Evo, etc. Is it possible that the findings in this work only apply to language-based methods, it would be interesting to see structure-aware methods also included in the benchmark.

### Questions
Please refer to weakness section

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
The paper has three main contributions:
1) An experimental dataset on TadA protein variants
2) Seq2Graph an algorithm to unify the directed evolution measurements 
3) Computational validations that test the generative capabilities of existing models on this dataset

### Strengths
1) The paper is clearly written.

2) Details of the Seq2Graph algorithm are discussed.

3) Developing a benchmark for the OOD problem in protein engineering is timely and valuable for the community.

4) Experimental efforts are substantial.

### Weaknesses
1) Besides the experimentally collected data, the only algorithmic contribution is the development of the Seq2Graph; however, there is no evaluation of the algorithm itself. The validation section focuses on using the dataset as a benchmark, but the algorithm itself is not tested or evaluated. As I describe in the section below, some questionable choices are made in its development.

2) The authors highlight that “bridging the OOD gap requires exploring functionally diverse regions of the sequence space, rather than simply increasing density” (line 431) and suggest that “evolutionary depth is critical for building models with realistic utility” (line 24). However, it is unclear how TadABench-1M contributes to advancing the development of said models. The paper would be strengthened by developing one of these models on the dataset and demonstrating its utility.

3) The dataset is limited to one protein, which is TadA. While even a dataset on a single protein remains valuable, it is not clear if all the claims generalize to a large set of proteins, especially when compared to Proteingym and Proteinbench.

### Questions
1) There are several choices made in the development of Seq2Graph that are not necessarily justified/empirically validated: 

a) Why does Seq2Graph create a directed edge between two sequences only? Why not identify the k nearest neighbors? In the example of Figure 3, if the sequence D has a \Delta rad = 975, it would seem unreasonable not to connect A and D. Picking the top-1 neighbor seems to be an arbitrary choice, not explained in the paper, as picking the top-k would not violate either the reliability or sparsity as discussed in the paper.

b) The process of removing cycles and creating DAGs is reasonable from the perspective of avoiding conflicts; however, it will discard valuable information by removing edges. It seems like averaging might be another good choice.

c) Most importantly, given that many different algorithm design choices for Seq2Graph could have been made, what is a fair evaluation procedure and metric for the algorithm alone? The paper fails to assess Seq2Graph as a standalone algorithm. What would happen if we made a wrong choice in Seq2Graph? What would go wrong? Is there a baseline processing algorithm to compare Seq2Graph with?

2) In Figure 5, I struggle to see how diversity and round-based strategies are better than density-based strategies. For instance, the density Spearman correlation in Evo2-7B and Evo2-40B is almost always higher than the diversity and round correlations, making the claim that smaller diversity datasets perform better than large, dense datasets appear incorrect.

3) It is unclear whether the results from this benchmark are generalizable across different proteins/genomes or whether this is just a phenomenon found in the TadA protein. This makes it difficult to assess the generalizability of the results.

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces TadABench-1M, a benchmark built on 31 rounds of wet-lab data of over one million TadA enzyme variants, addressing existing BLM benchmark flaws with OOD scenarios, evolutionary depth, and the Seq2Graph algorithm for consistent fitness labels. It finds SOTA BLMs perform well on random splits but collapse on temporal splits, proving evolutionary depth is a key for OOD generalization.

### Strengths
- This study develops a realistic, well-validated benchmark for BLM evaluation. It builds over one million TadA enzyme variants from 31 rounds of wet-lab evolution, naturally incorporating OOD challenges (via temporal splits) and evolutionary depth (up to 25 mutations per variant). 
- The experiments are comprehensive, and the results are convincing. It contrasts random (i.i.d.) and temporal (real-world) data splits, showing SOTA BLMs perform well on random splits but fail on temporal splits.

### Weaknesses
- The paper compares several language models and tuning strategies. It would be better to compare these baselines with diffusion language models, such as EvoDiff, DPLM, etc.
- The results are convincing, but not very surpurising (that protein language models memorize more than generalize). These language models have similar behaviors. I think it would be better to present something different, i.e., which component may help models generalize better, and how does this benchmark helps guide building new models.

### Questions
- How long does it take to build such a benchmark?

### Soundness
3

### Presentation
4

### Contribution
3
