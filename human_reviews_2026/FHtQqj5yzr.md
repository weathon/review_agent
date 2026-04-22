# UncertainGen: Uncertainty-Aware Representations of DNA Sequences for Metagenomic Binning

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 6, 2, 2

## Abstract
Metagenomic binning aims to cluster DNA fragments from mixed microbial samples into their respective genomes, a critical step for downstream analyses of microbial communities. Existing methods rely on deterministic representations, such as k-mer profiles or embeddings from large language models, which fail to capture the uncertainty inherent in DNA sequences arising from inter-species DNA sharing and from fragments with highly similar representations. We present the first probabilistic embedding approach, UncertainGen, for metagenomic binning, representing each DNA fragment as a probability distribution in latent space. Our approach naturally models sequence-level uncertainty, and we provide theoretical guarantees on embedding distinguishability. This probabilistic embedding framework expands the feasible latent space by introducing a data-adaptive metric, which in turn enables more flexible separation of bins/clusters. Experiments on real metagenomic datasets demonstrate the improvements over deterministic k-mer and LLM-based embeddings for the binning task by offering a scalable and lightweight solution for large-scale metagenomic analysis.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Metagenomics binning is an important step for analyzing microbial communities. It can benefit from a probabilistic clustering approach owing to domain-specific caveats such as inter-species DNA sharing. However existing data-driven approaches (e.g. kmer profiles and foundation model embeddings) produce deterministic embeddings.

To address this, the authors propose a probabilistic formulation of contrastive learning, where an encoder produces a mean and variance of a latent representation (similar to that of an inference network in classical VAE literature), and a non-Euclidian similarity objective is optimized over (paired) fragmented sequences. Experiments are conducted over established binning benchmarks, showing that a small, trained-from-scratch MLP can be competitive against a much larger, pretrained foundation model (DNABERT-S).

### Strengths
- Metagenomics binning is a well-motivated scientific problem with important applications, and it's good to see efforts designed at improving this task. 
- I really appreciate the authors' clarity in presenting their main theoretical results, which very clearly demonstrate the impossibility of deterministic embedding in producing the correct clustering, and how a probabilistic approach could fare better.
- Experimental results are good. A small MLP trained from scratch can match and/or outperform a much larger DNABERT-S finetuned from a capable foundation model (DNABERT-2)

### Weaknesses
- I do find the application area to be too siloed in metagenomics applications. While the probabilistic embedding framework is general, the authors should discuss how their approach can be beneficial for the broader scientific ML community, and / or the theoretical contribution is significant / contains deep technical innovations, discussed next.
- Overall, while the theoretical results are neat and clearly presented, they are primarily derived from existing techniques / known results in probability. 
- On the empirical side, it is good that a small network can outperform specialized foundation models (DNABERT-S), but it does make the scale of experiments a bit lacking. For example, the authors can finetune DNABERT-S with their objective and evaluate whether this objective can bring meaningful improvements from a strong base model. 
- On the other hand, recent works (e.g. [1]) have shown that carefully trained supervised baselines can outperform specialized foundation models in the genomics domain, and it'd be great if the authors can conduct an experiment on their simple MLP, but with the standard contrastive learning objective, and evaluate the model performance on binning benchmarks. This should give us a concrete idea of their objective's benefits.

Overall, I like this work but find it to be a bit too niche for ICLR. 

[1] https://arxiv.org/abs/2411.02796

### Questions
NA. See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces UncertainGen, which is a probabilistic embedding approach for metagenomic binning.
The work models sequence-level uncertainty and provides theoretical guarantees on embedding distinguishability.
The paper also provides experimental results on real metagenomic datasets to demonstrate the improvements over deterministic k-mer and LLM-based embeddings for the binning task.

### Strengths
(i) Novelity: The paper notes that many DNA sequences can appear in multiple genomes, and proposes a probabilistic embedding approach to solve this.

(ii) Theoretical guarantee: The paper provides a theoretical guarantee beyond the experimental results.

(iii) Efficiency: The framework remains lightweight compared to large genomic models.

### Weaknesses
(i) Performance: From Fig. 4, I think DNABERT-S still performs better than UncertainGen, so what is the meaning of UncertainGen?

(ii) Theory: The theoretical part is a little hard to understand.

### Questions
(i) What is the meaning of UncertainGen, if it does not achieve better performance compared with DNABERT-S?

(ii) Could you please provide some informal version of the theoretical results to help the audience to understand the theory?

(iii) The existing metagenomic binning method has other input features [1]. Can this method be applied to a setting where we have other features? If so, it would be better if the method could achieve the SOTA.

[1] SemiBin2: self-supervised contrastive learning leads to better MAGs for short- and long-read sequencing.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces UncertainGen, a probabilistic embedding method for metagenomic binning, which represents each DNA fragment as a distribution in latent space to capture sequence-level uncertainty. Evaluations are conducted to demonstrate the performance of this work.

### Strengths
* a) The idea of exploring the uncertainty of DNA sequences for the metagenomic binning task is interesting.

### Weaknesses
**(a) Unclear method definition**  
The paper does not clearly explain the core proposed approach. In particular, while a number of lemmas, definitions, and theorems are presented, their connection to the method is unclear. It is ambiguous whether the proposed framework is intended as a loss term with constraints, an optimization objective to be maximized, or something else entirely. A more direct and structured description of the method workflow is needed to make the contribution comprehensible.  

**(b) Insufficient model details**  
The backbone network architectures used for the experiments are not described in sufficient detail. For reproducibility and better understanding of the results, the paper should clearly specify the model structures and any architectural choices that may influence performance.  

**(c) Insufficient comparisons**  
The experimental evaluation does not appear to include enough comparisons with relevant prior work. Without comprehensive benchmarks against the state of the art, it is difficult to assess the true effectiveness and novelty of the proposed approach.

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents UncertainGen, the first probabilistic embedding framework for metagenome binning. Different from traditional deterministic embeddings (k-mer profiles or LLM-based), UncertainGen represents each DNA segment as a probabilistic distribution in a latent space to capture the uncertainty of cross-species DNA sharing and highly similar sequences.The method adopts a data-adaptive metric that expands the latent space and improves the separability of bins. The authors provide theoretical guarantees on embedding discriminability and empirically demonstrate that UncertainGen outperforms deterministic embeddings on real metagenomic datasets, providing a scalable and lightweight solution for large-scale metagenomic analysis.

### Strengths
- This paper proposes a probabilistic embedding framework for DNA fragments, which is technically interesting because it can naturally capture the uncertainty in sequences that may belong to multiple genomes or have highly similar k-mer profiles.
- It is well-written and clearly explains the approach, linking theoretical guarantees on embedding distinguishability to practical benefits. 
- The experimental setup is comprehensive and convincing, demonstrating that UncertainGen outperforms selected baselines on real metagenomic datasets while remaining scalable and interpretable.

### Weaknesses
- The novelty and technical contribution of this paper are limited, as its main contribution is modeling uncertainty in DNA sequences using Gaussian distributions for fragments. However, prior work such as UnitigBIN (ICLR 2024) has already introduced the idea of representing DNA fragments as distributions, reducing the originality of the current approach.
- The main drawback of this paper is that, although it claims to propose a new solution for metagenomic binning, it does not compare against established binning tools such as VAMB, SemiBin2, MetaBAT2, or MaxBin2. Moreover, the paper does not follow the standard evaluation protocols used in metagenomic binning studies - for instance, it defines high-quality bins as clusters with F1 > 0.9, whereas the community standard relies on CheckM to assess completeness and contamination. 
- In addition, the related work section overlooks several key metagenomic binning studies and reflects a limited understanding of the field. Finally, the motivation for introducing probabilistic embeddings in this context is not sufficiently justified or biologically grounded.

### Questions
- Could the authors clarify why commonly used metagenomic binning tools (e.g., VAMB, SemiBin2, MetaBAT2, MaxBin2) were not included as baselines for comparison?
- It would be helpful to understand why the evaluation does not follow the standard metagenomic binning pipeline, which typically assesses high-quality bins using CheckM for completeness and contamination.
- Could the authors further elaborate on the motivation for their approach and expand the related work discussion to more clearly position this study within the broader metagenomic binning literature?

### Soundness
2

### Presentation
2

### Contribution
1
