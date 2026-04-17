# DNAMotifTokenizer: Towards Biologically Informed Tokenization of Genomic Sequences

- Decision: Reject
- Scores: 6, 2, 4, 4

## Abstract
DNA language models have advanced genomics, but their downstream performance varies widely due to differences in tokenization, pretraining data, and architecture. We argue that a major bottleneck lies in tokenizing sparse and unevenly distributed DNA sequence motifs, which are critical for accurate and interpretable models. To investigate, we systematically benchmark k-mer and Byte-Pair Encoding (BPE) tokenizers under controlled pretraining, evaluating across multiple downstream tasks from five datasets. We find that tokenizer choice induces task-specific trade-offs, and that vocabulary size and training data strongly influence the biological knowledge captured. Notably, BPE tokenizers achieve strong performance when trained on smaller but biologically significant data. Building on these insights, we introduce DNAMotifTokenizer, which directly incorporates domain knowledge of DNA sequence motifs into the tokenization process.  DNAMotifTokenizer consistently outperforms BPE across diverse benchmarks, demonstrating that knowledge-infused tokenization is crucial for learning powerful, interpretable, and generalizable genomic representations.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this paper, the authors proposed DNAMotifTokenizer, a PWM-driven tokenizer that trims low-information flanks, handles reverse complements, and segments sequences via greedy trie matching. The tokenizer integrates seamlessly with BERT-style masked language modeling using motif-aware masking and an end-to-end, cache-friendly pretraining pipeline compatible with DNABERT. The goal is to replace purely k-mer tokenization with a more interpretable, biologically grounded alternative that preserves coordinate-level traceability. The given results suggest consistent gains on most of the existing benchmarks.

### Strengths
1. The author proposed a novel and interesting method to integrate the motif information to the tokenization, instead of eliminating such info as previous method did. It could bring more biologically useful information and help the language model investigate deeper in existing data.

2. The authors provide the implementation of the core idea, which aligns well with the paper.

3. The given results show that the proposed DNAMotifTokenizer can bring general improvement to most tasks.

4. Apart from normal content, the authors also took a deeper look into the existing BPE tokenizer for a further discussion and comparison with the proposed one, which improves the rationality.

### Weaknesses
1. Method: Heavy reliance on heuristics (length cap, flank-trimming thresholds, greedy matching) and PWM quality. Such components may bias learning and miss unknown motifs.

2. Experiment: Although the overall performance of the proposed method is very good, it shows obvious bad results on DART-EVAL. For example, in Table 4, the accuracy of the proposed DNAMotifTokenizer is much lower than the SOTA method. The authors are suggested to give the explanations of why this happens.

### Questions
How is the proposed DNAMotifTokenizer on more cross-species tasks? And what about more modern model architectures (e.g., llama, mamba), instead of BERTs only.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes DNAMotifTokenizer, a motif-aware tokenization scheme for genomic sequences that integrates curated TF-motif priors and cCRE annotations to build a vocabulary of motif tokens. The work evaluates across several public genomics benchmarks (e.g., Genomic Benchmarks, GUE, NT-benchmarks, DART-Eval).

### Strengths
1. Brings explicit biological priors (TF motifs, cRE) into the tokenization step, offering a more interpretable alternative to opaque subword units.
2. Provides multiple ablations on vocabulary size, segmentation strategies, and qualitative motif coverage

### Weaknesses
1. The paper’s own results indicate k-mer consistency better than BPE, and DNAMotifTokenizer’s average on NT-benchmarks remains notably below k-mer. This undercuts the central narrative that knowledge-injected tokenization improves fundamental understanding. 

2. Converting PWMs via a fixed 0.5 threshold and trimming wildcard ends discards degenerate bases and positional uncertainty, which are biologically meaningful. This can fragment genuine motif families and reduce robustness to natural variation.

3. When multiple motifs overlap, the default random choice introduces non-determinism. The paper lacks multi-seed repeats and dispersion metrics to judge stability.

### Questions
Please see Weaknesses.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper benchmarks DNA tokenization strategies under a controlled pretraining setup and proposes DNAMotifTokenizer, which injects motif knowledge into the vocabulary and uses a greedy, locally flexible matching procedure. The results suggest that larger BPE vocabularies are not necessarily better, and that training the tokenizer on biologically informative subsets (e.g., motifs or cCRE regions) can perform comparably to training on the entire genome.

### Strengths
1. The proposed tokenizer is conceptually simple and biologically informed, with clear pseudocode that enhances reproducibility.
2. The experimental design of this study is rigorous as it meticulously isolates the impact of tokenization by matching computational FLOPs, model architecture, and fine-tuning pipelines across all comparisons (GUE, SCREEN, DART-Eval, Genomic Benchmarks, NT Benchmarks).

### Weaknesses
1. The technical presentation of this work requires further efforts.  The paper presents both a benchmark and a new method in a 9-page paper. The direct result is that the benchmark is not comprehensive and the analysis of the method is not enough. 
2. The experimental results show small absolute gains, and the variance is not reported, e.g. some improvements are ≤ 0.0005 in absolute terms.
3. The 0-2 bp offset and random tie-breaking are reasonable, but their stability and computational complexity are not fully characterized. More discussion are needed.
4. The figure captions are insufficiently detailed, lacking explanations for the individual subpanels (a, b, c, etc.), and the absence of descriptive legends hinders the interpretation of key elements.

### Questions
1. What's the algorithm's sensitivity to parameters like the 0-2 bp offset and the tie-breaking mechanism? The performance gains appear modest relative to the added complexity compared to standard BPE. Please discuss the specific scenarios where this complexity is justified.
·What is the primary intended contribution of this paper — a new method or a benchmark? The current structure does not fully align with either goal: as a benchmark paper, the experimental section of the main text is not enough; as a method paper, the narrative structure is not very reasonable.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a DNA sequence tokenizer called DNAMotifTokenizer, which is based on transcription factor binding sites (TF motifs). By directly incorporating biological motifs into the vocabulary, it aims to enhance the interpretability and task performance of DNA language models. The authors conducted extensive experiments under multiple benchmarks (GUE, DART-Eval, NT) to systematically evaluate the impact of different tokenization strategies on model performance.

### Strengths
- The core contribution is the hard-coding of biological prior knowledge (transcription factor motifs, TF motifs) into the vocabulary as "tokens," demonstrating performance gains across multiple benchmarks through experiments.
- Focuses on the core issue of DNA language models—the impact of tokenization strategies—and conducts systematic and reproducible comparative experiments.
- The introduction of biological priors (motifs, cCREs) enhances interpretability, showing consistent gains across multiple tasks.
- The experimental design is rigorous, and the training budget and parameter scale are well-controlled, making the results reliable.

### Weaknesses
- The tokenization relies entirely on external databases (JASPAR, ENCODE), making the approach essentially "manual knowledge injection," which cannot adapt to unknown regions or new species.

- Limited Innovation: Using motifs as a vocabulary is an engineering improvement that is insufficient for a theoretical breakthrough. There is inadequate biological interpretative analysis, as the paper does not quantify the impact of motif tokens on the model's internal representations.

- About Generalizability: The use of a traditional BERT architecture and short sequence inputs restricts the model's generalizability.

Improvement Suggestions:
- The core contribution is the hard-coding of biological prior knowledge (transcription factor motifs, TF motifs) into the vocabulary as "tokens," demonstrating performance gains across multiple benchmarks through experiments.
- Focuses on the core issue of DNA language models—the impact of tokenization strategies—and conducts systematic and reproducible comparative experiments.
- The introduction of biological priors (motifs, cCREs) enhances interpretability, showing consistent gains across multiple tasks.
- The experimental design is rigorous, and the training budget and parameter scale are well-controlled, making the results reliable.
- Introduce a learnable motif discovery module to enable the model to have adaptive tokenization capabilities.
- Test generalizability on unknown regions or artificially mutated data.
- Provide interpretability metrics such as motif recovery rates and functional region enrichment.
- Explore the scalability of the method in long sequence modeling or generation tasks.

### Questions
please refer to the weaknesses part

### Soundness
3

### Presentation
3

### Contribution
3
