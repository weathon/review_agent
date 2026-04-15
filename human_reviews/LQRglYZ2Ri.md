# LOL-EVE: Predicting Promoter Variant  Effects from Evolutionary Sequences

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 3, 3

## Abstract
Genetic studies reveal extensive disease-associated variation across the human genome, predominantly in noncoding regions, such as promoters. Quantifying the impact of these variants on disease risk is crucial to our understanding of the underlying disease mechanisms and advancing personalized medicine. However, current computational methods struggle to capture variant effects, particularly those of insertions and deletions (indels), which can significantly disrupt gene expression. To address this challenge, we present LOL-EVE (Language Of Life across EVolutionary Effects), a conditional autoregressive transformer model trained on 14.6 million diverse mammalian promoter sequences. Leveraging evolutionary information and proximal genetic context, LOL-EVE predicts indel variant effects in human promoter regions. We introduce three new benchmarks for indel variant effect prediction in promoter regions, comprising the identification of causal eQTLs, prioritization of rare variants in the human population, and understanding disruptions of transcription factor binding sites. We find that LOL-EVE achieves state-of-the-art performance on these tasks, demonstrating the potential of region-specific large genomic language models and offering a powerful tool for prioritizing potentially causal non-coding variants in disease studies.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper presents a conditional autoregressive transformer model trained on a large dataset of mammalian promoter sequences, which can be used to predict the effects of genetic variants in promoter sequences. The authors benchmark the model on three indel (insertion/deletion) variant effect prediction tasks, in contrast to the single nucleotide variant tasks that models are more commonly benchmarked on. They outperform a non-deep learning conservation baseline as well as previous genomic deep learning models trained genome-wide rather than on only promoter sequences.

### Strengths
- The paper is very clear, well written and well motivated.
- The paper is highly original and significant. While a number of different DNA language models - which are cited by the paper - have been proposed, they generally take the approach of training on genome-wide sequences. Therefore, the approach presented here to train a focused model on promoter sequences, as well as conditioning on clade, species, and gene, is quite novel.
- The focus on indel prediction tasks is also novel, and an important understudied area, as most prior studies only benchmark on single nucleotide variants. Therefore, I anticipate that the tasks proposed here will be a useful resource for future studies.

### Weaknesses
The main weakness of the paper is a lack of comparison to a larger set of baselines, and to tasks that have been studied in previous work, which limit the significance of the work. For example, GPN-MSA (Benegas et al., 2024a) was shown to have some capacity to predict regulatory variant effects, and would be a good model to compare to. In addition, a supervised baseline such as Enformer (Avsec et al., 2021), which was shown in Kao et al. 2024 to outperform HyenaDNA and Nucleotide Transformer for eQTL prediction, would also be a relevant comparison to add. Specifically, showing the performance of these two additional baseline models on the tasks/metrics already evaluated in the paper would be informative. In addition, while the focus on indels is novel and interesting, since the model can also be used to score single-nucleotide variants, it would add value to show performance on SNV prediction tasks, such as the SNV eQTL classification task from the Enformer paper (Avsec et al., 2021).

### Questions
- What would the effect be of a more strict PIP cutoff (such as 0.1 or 0.05) for the non-causal set in the eQTL task?
- The description of the TFBS disruption task is a bit unclear as written - are the TFBS knockouts from experimental data, or are the TFBS sequences simply altered in silico? More specific details about the type of data used in this evaluation (i.e. positive and negative sequences) would help clarify the task.

Minor:
- since the causal eQTLs referred to in the abstract and in section 4.2 are determined via finemapping rather than experimentally validated, they should be referred to as putative causal eQTLs.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
The authors present a mammalian promoter autoregressive DNA language model for insertion or deletion (indel) variant effect prediction (VEP). The proposed approach is novel in terms of data curation as well as the choice of control tags to condition the model on species and gene information. This paper introduces new relevant benchmarks to assess promoter indel VEP capabilities, where the proposed model outperforms the set of baselines evaluated.

### Strengths
- The chosen problem, indel VEP, is an impactful and understudied problem in human genetics. The authors do a very good job motivating this problem and the proposed autoregressive language model.
- The proposed approach is substantially different from current approaches, particularly in data curation and conditioning features, which the authors do a good job motivating. It is refreshing to see new ideas tailored to the biological setting that are not just trying out the latest architecture from natural language processing.
- The authors present interesting new benchmarks covering the spectrum of rare and common variants.
- The model achieves a relatively good performance compared to the chosen baselines.

### Weaknesses
- Several key baselines are lacking that are crucial to contextualize the performance of the model. First, I expect a comparison to CADD v1.7 (which should also be described in the background section), an established genome-wide deleteriousness predictor that can handle indels. The CADD website includes a web server to score variants. Second, I expect a comparison to at least one current-generation sequence-to-activity model (e.g. Enformer, Borzoi, Sei). Sei might be easier since it also has a web server to score variants, but if the authors want to keep their SOTA claim they should include Enformer or Borzoi. Note 1: not beating these models would not hurt my recommendation for acceptance, as I believe the ideas in the paper would still be valuable. Note 2: the suggested change to the frequency-based task below would reduce the sample size and facilitate running additional models.
- The authors do a good job motivating the conditioning features (species, clade, gene) but do not present any empirical result, which I expect for recommending acceptance. A fast ablation experiment would be to exclude these features (similar to how “control tag dropout” is done during training) at inference time when running VEP, one at a time, to assess their contribution.
- The authors claim “The superior performance of LOL-EVE over other models can be attributed to several factors that address limitations in existing approaches.”. Given the lack of ablation studies, I would be more cautious (e.g. “may be attributed”) and highlight that further studies are necessary to understand the contributions of the different components such as tokenization, region selection, species selection, etc.
- The TFBS disruption is relevant and interesting but the design choices seem rather arbitrary and unnecessarily complicated. A more natural task would be to distinguish variants that affect the binding of any TF vs. those that don’t affect any TFBS, across all human promoters (JASPAR has such information pre-computed: https://jaspar.elixir.no/genome-tracks/). Even if the authors still want to include the current version, I think this simpler formulation of the task should be included regardless. Moving on to the current formulation, I agree that promoters of low-variance genes will be in expectation more constrained than high-variance genes, but that is also true of highly expressed vs. low expressed genes, or highly conserved CDS vs. not (according to multi-species alignment), or highly constrained CDS vs. not (based on population data, see e.g. GeneBayes). Also, in this formulation the metric should be further clarified, e.g. what exactly is the random baseline? This sentence in particular was hard to understand: “A group counts as correctly predicted when the score is lower. ”. What does it mean that a group is correctly predicted? Wouldn’t this statement apply to the TF rather than the group? The score is lower in what compared to what?
- For frequency-based analysis, the authors compare common (MAF > 5%) vs. low-frequency (MAF < 5%) variants. It is common in the field to contrast common vs. singletons, which is expected to yield stronger signals. See e.g. Borzoi, SpliceAI, GPN, GPN-MSA. This should be reported since it would not require any extra computation, just subsetting the current set. If there’s any limitation with indel calling for singletons, it should be mentioned. An alternative set could be common (MAF > 5%) vs. rare variants (MAF < 0.1%).
- For eQTL analysis, the authors compare PIP > 0.95 vs. PIP < 0.95. It is common in the field to compare PIP > 0.9 vs PIP < 0.01 (see e.g. Enformer, Borzoi). Doing this analysis should not require any extra computation since it’s just subsetting the current sets. Also, the actual, unnormalized AUPRC should also be reported in Table 1 (there’s plenty of space). I suspect this would reveal that none of the models are doing well in this task (which is admittedly difficult) in absolute terms. To this point, the phrase “DNABERT-2 surprisingly shows a negative effect size, suggesting potential limitations in its ability to prioritize causal indel variants in this context.” can be misleading. None of the models seem to be doing well on this task, which again, is very difficult.
- It could be worth discussing the fact that some clades are over-represented in the Zoonomia alignment, and its implications.

Minor comments:
- Ref. to Fig. A2 is incorrectly formatted.
- Typo “Causal varant prioritization” in p. 2
-  ”promoter annotations” p. 5, quote incorrectly formatted.

### Questions
Please see the suggestions above.  Additional experiments which would make the paper stronger, but might take too much time:
- Do you recommend your model only for indels or also SNVs? Additional experiments on SNVs would be insightful.
- An ablation study where you train the model on entire genomes (or random regions) instead of promoters would be insightful, although expensive to run.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper presents a generative model that facilitates to score
presence vs absence of indel variants in particular. The generative
model is autoregressive in nature (derived from CTRL). The
probability to generate the next character in a promoter sequence
depends on the already generated subsequence, the clade and the
species to which the sequence is assigned, as well as the gene that
the promoter supports (gene coded as ESM2 embedding of the
corresponding protein sequence). To the best of my understanding,
the achievement of this paper is to provide a (transformer) decoder-only
type approach to computation of probabilities on gaps / insertions
appearing in viable DNA.

### Strengths
The paper appears to present a professionally executed study. Choice of data and techniques involved in the approach are of high quality, reflecting recent approved methods. The novelty consists in presenting a transformer decoder-only type of approach by which to compute likelihoods for gaps appearing in DNA. The presentation of the contents has improved considerably, and the paper is now easy to follow and understand, also supported by lovely illustrations.

### Weaknesses
There are fairness issues with respect to selection of benchmarking methods and the way they are used. The authors evaluate other methods in a way that is neither recommended by the authors of the respective papers, nor considered in subsequent literature contributions (relating to HyenaDNA and Caduceus). The authors also miss to evaluate the likely strongest competitor (GPN).

### Questions
After ample exchange with the authors, there are no questions remaining. The authors responded to my input, although very critical, always in a fair and forthcoming manner, for which I would like to thank.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
LOL-EVE is an autoregressive language model trained on mammalian promoter sequences that operates on a single base pair resolution. The authors focus on promoter indels and construct a benchmark containing three tasks to evaluate model performance: (1) distinguishing common and rare indels, (2) prioritizing fine-mapped eQTLs, and (3) distinguishing TFBS mutation scores between low-variability and high-variability genes. The authors compare LOL-EVE to phyloP and other commonly used genomic language models and find that their method outperforms the rest on these three tasks.

### Strengths
- Instead of training a genome-wide language model which has been shown to have reduced performance in understanding any one genomic element, the authors train only on proximal promoter sequences, which likely have a well-conserved, cell-type agnostic cis-regulatory grammar. This is the ideal setting to use self-supervised language models.
- The authors use an autoregressive model, with individual base pairs as tokens, which is relatively uncommon in the field but allows them to easily make predictions on indels as opposed to just SNVs.
- The authors extracted homologous promoter sequences from whole-genome alignments and then cleverly validated that aligned sequences in other species were likely promoters by using Sei promoter scores.
- The authors developed a biologically-meaningful benchmark for a setting in which benchmark datasets have not yet been constructed.

### Weaknesses
1. While the tasks are well-motivated, they may inadvertently emphasize gene-specific rather than variant-specific properties. 
     - Frequency-based indel prioritization task: Models are asked to distinguish rare promoter indels from common ones. However, if a gene is under selective pressure, the number of common promoter indels will decrease. Accordingly, a model might achieve high performance by recognizing gene-level constraints instead of specific properties of each indel variant. To avoid this, metrics should be calculated separately for each gene and then averaged.
     - Causal eQTL prioritization: The negative set consists of non-finemapped eQTLs that may not be matched to the same genes as the positive set. This again allows models to potentially rely on gene-related features rather than focusing on the distinct attributes of each variant. To avoid this, the negative set should be matched to be from the same genes as the positive set.
     - TFBS disruption: Performance is evaluated by comparing indel scores between low-variability and high-variability genes. Low-variability genes are generally under greater evolutionary constraints, which means that models might perform well by understanding these gene constraints instead of learning about the specific effects of the variants within the TFBS. This task seems to be deliberately designed to test understanding of gene constraint, so it doesn't need to be changed. 

A model that learns how much variability there is in a promoter across species (which LOL-EVE likely does) might understand selective pressures on the gene and do well on these tasks without really understanding a good causal cis-regulatory grammar. 

2. The authors should compare their method to sequence-to-activity models, like Enformer. While the authors claim that functional genomics data is not often available to train sequence-to-activity models, this data is readily available for humans at least. They should also compare to simpler non-genomic language model baselines, like CADD, fathmm-indel, and deltaSVM. Even simple baselines like GC content of the surrounding base pairs or length of the indel might be competitive. 

3. The authors should report performance on promoter SNVs even if the primary goal of developing the autoregressive model was to handle indels. Even if LOL-EVE does not outperform existing methods on SNVs, knowing SNV results would help readers better understand the limitations and strengths of this model.

### Questions
- phyloP scores were extracted for the single base pair position of the indel, presumably corresponding to the position of the leftmost element of the sequence. However, this doesn't capture the constraints of other base pairs removed in a deletion. How do performance metrics change if all reference positions in a deletion are incorporated? phyloP base pair statistics are also noisy, so it might be worth averaging scores in a 10 or 25 base pair region surrounding the indel. 
- A paper using ESM for scoring protein indels (https://www.nature.com/articles/s41588-023-01465-0#Sec8) suggests that length-normalizing pseudolikelihoods is not the best way to score indels in masked language models. How do your results change if you get rid of the length-normalization?
- I don't understand how you identified deletions in TFBS for the TFBS disruption task? Did these deletions need to be occurring in gnomAD? Did the deletions span the entire TFBS?

### Soundness
2

### Presentation
3

### Contribution
2
