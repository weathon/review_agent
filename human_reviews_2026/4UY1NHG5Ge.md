# Tokenization to Transfer: Do Genomic Foundation Models Learn Good Representations?

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 4, 4

## Abstract
The success of Large Language Models has inspired the development of Genomic Foundation Models (GFMs) through similar pretraining techniques. However, the relationship between pretraining performance and effectiveness in downstream genomic tasks remains unclear. Additionally, the high computational cost of pretraining raises questions about its cost-efficiency. To assess the usefulness of pretraining in genomics, we evaluated seven different GFMs across 52 diverse genomic tasks, comparing them to their counterparts with randomly initialized weights. Across benchmarks, we find that randomly initialized models provide surprisingly strong baselines and tokenizer and architecture choices strongly shape both these baselines and the gains from pretraining. Specifically, character‑token models often match or exceed the performance of larger pretrained k‑mer or BPE models, whereas subword models appear to benefit from pretraining. We also find that the evaluated GFMs fail to capture clinically relevant genetic mutations, with embeddings and log‑likelihood ratios showing limited sensitivity to annotated variants. For the tasks we study, these results suggest that current NLP‑style pretraining strategies provide modest, tokenizer‑gated improvements over strong random baselines and motivate more biologically informed tokenization and variant‑aware objectives. Our code is available at https://github.com/m42-health/gfm-random-eval.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper evaluates the effectiveness of pretraining in Genomic Foundation Models (GFMs). The authors benchmark seven GFMs across 52 genomic tasks, comparing them with randomly initialized counterparts. Results show that randomly initialized models often match or even outperform pretrained GFMs in both fine-tuning and feature extraction settings. The study concludes that current research lacks a clear understanding of how pretraining contributes to model performance and offers new insights into how to achieve more efficient genomic modeling.

### Strengths
- Meaningful Perspective: The paper focuses on evaluating the effectiveness of pretraining in GFMs and raises new research questions and perspectives for genomic modeling, providing valuable insights for the community.

- Comprehensive Evaluation: The study benchmarks 7 representative GFMs across 52 diverse genomic tasks, covering structural, functional, and regulatory dimensions.

- Extensive Experimental Setup: The authors conduct detailed comparisons of fine-tuning techniques, including full fine-tuning and LoRA. Also perform extensive hyperparameter searches over learning rate, batch size, and other factors with nearly 10,000 fine-tuning experiments.

### Weaknesses
- Potentially Unfair Comparison Setup: Randomly initialized models received embedding dimension optimization, but similar architectural optimization was not performed for pretrained models. What would happen if the pre-trained model also used a character tokenizer and a larger embedding dimension?
- Analysis of Pretraining: The observation that a character-level tokenizer benefits randomly initialized models may indicate design issues in the pretrained models rather than reflecting the true effect of pretraining on genomic models.
- Task Distribution: The paper evaluates 52 tasks, but it is unclear what criteria were used to select them. Are there specific types of tasks like long-range dependency tasks where pretrained models might perform better?
- Generative Tasks: All seven models evaluated in the paper are classification models, and the 52 tasks focus primarily on classification, with no assessment of generative tasks. Although the authors explain their reasons for not using Evo, recent large-scale genomic models such as Evo2[1] and GenomeOcean[2] are designed for generative purposes. Do the conclusions drawn in this study also hold for generative models, or are they only valid when generative models are applied to classification tasks?
- Quantized Model: In this paper, all models are tested under full precision. However, quantized genomic foundation models (GFMs) like GERM [3] also exist. Does the same conclusion hold for these models?

[1] Genome modeling and design across all domains of life with Evo 2.

[2] GenomeOcean: An Efficient Genome Foundation Model Trained on Large-Scale Metagenomic Assemblies.

[3] Fast and Low-Cost Genomic Foundation Models via Outlier Removal.

### Questions
See the weaknesses section.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper challenges the prevailing assumption that large-scale unsupervised pretraining is inherently beneficial for Genomic Foundation Models (GFMs). The authors conduct an extensive empirical study comparing seven different GFMs against their randomly initialized counterparts across 52 diverse genomic tasks, spanning finetuning and feature extraction. The core finding is that randomly initialized models, when properly tuned, often match or even exceed the performance of their billion-scale pretrained counterparts.

Furthermore, the paper introduces a critical set of analyses on genomic variation, demonstrating that current GFMs are largely insensitive to subtle, clinically relevant mutations like SNPs. The models perform at near-random chance on variant classification tasks and produce nearly identical embeddings for reference and mutated sequences. The authors conclude that current pretraining strategies, largely adapted from NLP, are insufficient for genomics and that the field must rethink its approach to tokenization, pretraining objectives, and evaluation.

### Strengths
1. Large-Scale, Rigorous Evaluation: The sheer scale of the study (7 models, 52 tasks) provides very strong evidence. The authors' commitment to a rigorous hyperparameter search for all models (random and pretrained) makes their comparison fair and robust.

2. Novel and Critical Analysis: The genomic variation analysis (Sec 3.3) is a major strength. By showing that GFMs are insensitive to SNPs and fail on ClinVar data, the authors expose a critical blind spot in current models and evaluation methods. This finding alone is a significant contribution.

3. Actionable Insights: The paper provides clear hypotheses for these failures (k-mer tokenization obscuring SNPs, high masking rates) and points toward concrete areas for improvement (e.g., character-level tokenizers, as used in their own Mistral model).

### Weaknesses
1. Limited Scope (Generative Tasks): The paper's claims are based entirely on classification and feature extraction tasks. The authors briefly concede in the discussion and conclusion that pretraining might still be valuable for generative tasks. This is an important limitation, and the bold title ("Pretraining Does Not Promise Performance") might be a slight overstatement. The weakness is minor, as the paper's scope is already large, but it should be stated more prominently.

2. "What" over "Why": The paper excels at showing that pretraining fails but is less definitive on why. The discussion of tokenization and masking rates is insightful but largely correlational. The paper would be strengthened by even a single targeted ablation, e.g., pretraining two identical small models, one with k-mer and one with character-level tokenization, to prove that the tokenizer is the key factor.

### Questions
1. The authors convincingly argue that k-mer/BPE tokenizers are a major issue, especially for the genomic variation tasks. Their own Mistral model, which uses a character tokenizer and performs well, seems to support this. Could the paper's central finding be more narrowly (and perhaps more accurately) stated as "Current k-mer-based pretraining strategies are ineffective" rather than a blanket statement about all pretraining?

2. Following on the previous point, the feature extraction results (Table 2, Fig. 3) suggest that a randomly-initialized model with a character tokenizer and an optimized embedding dimension is a top performer. Is this also true for the finetuning tasks? A comparison of "best random (char-tokenizer)" vs. "best pretrained" would be very illuminating.

3. The authors attribute their different findings from prior work (e.g., Dalla-Torre et al., 2024) to their more rigorous hyperparameter search. As a sanity check, were they able to reproduce the original paper's results (i.e., showing a benefit for pretraining) by using the original fixed hyperparameters? This would definitively confirm that the HPO sweep is the key methodological difference.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This submission evaluates seven Genomic Foundation Models (GFMs) on 52 tasks spanning NT Benchmark, GUE, and Genomic Benchmarks, comparing pre-trained checkpoints against randomly initialized counterparts under fine-tuning, frozen‑feature (“feature extraction”), and genomic variation settings. The headline result is that models trained from scratch often match or surpass their pre-trained versions; moreover, embeddings from multiple GFMs are notably insensitive to clinically relevant variants, with cosine similarity remaining ~0.9–0.999 even after many SNPs are introduced. The paper argues that (i) existing NLP‑style pre-training is a poor investment for regulatory genomics classification and (ii) tokenization/architectural choices (e.g., character tokenization, larger embedding dimensions) dominate performance.

### Strengths
- **(S1)** Breadth and scale of evaluation with reproducibility intent. The study spans 7 models and 52 tasks across three popular benchmarks, with nearly 10k fine-tuning runs and LR sweeps, and reports model‑wise subgroup results. Datasets and checkpoints are standard, and code is linked (anonymized) in the paper, which, taken together, improves reproducibility. Meanwhile, the Figure 1 summary is easy to parse and consistently shows a limited advantage for pre-training when judged against a strong random baseline. The authors also empirically prefer full fine-tuning over LoRA, reducing a common confound in cross‑model comparisons.

- **(S2)** Variant-sensitivity diagnostics reveal a critical gap. The mutation-sensitivity experiments and ClinVar log-likelihood–ratio AUROCs near 0.5 jointly show that many GFMs’ representations are remarkably insensitive to clinically relevant point mutations, echoing long-standing concerns that subword/k‑mer tokenization blurs SNV granularity.

- **(S3)** Positioning within ongoing benchmarking discussions. The paper’s thesis aligns with emerging benchmark evidence that SFT matters and that supervised long-range models can still dominate on gene expression, underscoring that what we benchmark matters as much as how.

### Weaknesses
- **(W1)** “Pre-training vs. random” comparisons are confounded by architecture/tokenizer changes. The feature-extraction claim that random models can beat pre-trained hinges on changing the tokenizer (char rather than original k‑mer/BPE) and increasing the embedding size for the random arm (Table 2 and Figure 3), whereas pre-trained arms keep their native tokenizers/widths. This violates ceteris paribus, conflating the benefit of model design with the absence of pre-training. A fair test needs identical architectures/tokenizers/widths with/without pre-training. As is, the headline conclusion is directionally plausible but not causally isolated.

- **(W2)** Variant sensitivity method is too blunt to support strong claims. The mutation‑sensitivity analysis relies on global pooling and cosine similarity of full‑sequence embeddings, while high similarities that sometimes increase with more mutations likely reflect pooling/normalization effects rather than true biological blindness. Moreover, the LLR analysis lacks detail for encoder-only masked LMs (pseudo-likelihood vs. left-to-right), complicating an apples-to-apples comparison with decoders. Token-level distances, attribution at mutated loci, and clearly specified pseudo-LLR for encoders are necessary.

- **(W3)** Task coverage underweights where long-range biology is known to matter. The selected tasks span mainstream benchmarks, but the suite lacks gene-expression regression (e.g., bulk RNA/CAGE) and enhancer–gene linkage tasks for which long-range inductive biases and explicit supervised training (e.g., Enformer) remain strong baselines and stress tests.

- **(W4)** Positioning more concurrent architecture findings could be sharper. The paper’s narrative at times attributes the observed wins to the absence of pretraining, when the architecture/inductive bias could be dominant. Independent recent results indicate that simple, well-tuned CNNs can outperform SSM/Transformer DNA models on many tasks without pretraining (e.g., ConvNova), suggesting that architectural priors and receptive-field design can rival or exceed pretraining gains. This alternative explanation deserves explicit treatment in the Discussion.

### Questions
- **(Q1)** Can the authors provide apple-to-apple ablations where architecture, tokenizer, embedding size, positional encoding, and training schedule are identical, and the only difference is with vs. without pre-training? This would directly test the causal value of pre-training beyond the confounds noted in Table 2 and Figure 3.

- **(Q2)** How do the findings change in label-scarce regimes (e.g., 1%, 5%, and 10% of labels)? Please include learning curves and area under the data curve statistics to test whether pre-training is more helpful at low data.

- **(Q3)** For mutation sensitivity, could the authors further provide in-silico mutagenesis with base-resolution attribution (e.g., Grad-CAM) and compare to CADD/Enformer scores and to eQTL/sQTL ground truths (AUPRC, per-gene AUROC)?

### Soundness
2

### Presentation
3

### Contribution
2
