# A New Paradigm for Genome-wide DNA Methylation Prediction Without Methylation Input

- Decision: Accept (Poster)
- Scores: 2, 6, 6, 4

## Abstract
DNA methylation (DNAm) is a key epigenetic modification that regulates gene expression and is pivotal in development and disease. However, profiling DNAm at genome scale is challenging: of $\textasciitilde$28 million CpG sites in the human genome, only about 1–3\% are typically assayed in common datasets due to technological limitations and cost. Recent deep learning approaches, including masking-based generative Transformer models, have shown promise in capturing DNAm–gene expression relationships, but they rely on partially observed DNAm values for unmeasured CpGs and cannot be applied to completely unmeasured samples. To overcome this barrier, we introduce MethylProphet, a gene-guided, context-aware Transformer model for whole-genome DNAm inference without any measured DNAm input. MethylProphet compresses comprehensive gene expression profiles ($\textasciitilde$25K genes) through an efficient bottleneck multilayer perceptron, and encodes local CpG sequence context with a specialized DNA tokenizer. These representations are integrated by a Transformer encoder to predict site-specific methylation levels. Trained on large-scale pan-tissue whole-genome bisulfite sequencing data from ENCODE (1.6 billion CpG–sample pairs, $\textasciitilde$322 billion tokens), MethylProphet demonstrates strong performance in hold-out evaluations, accurately inferring DNAm at unmeasured CpGs and generalizing to unseen samples. Furthermore, application to TCGA pan-cancer data (chromosome 1, 9,194 samples; $\textasciitilde$450 million training pairs, 91 billion tokens) highlights its potential for pan-cancer whole-genome methylome imputation. MethylProphet offers a powerful and scalable foundation model for epigenetics, providing high-resolution methylation landscape reconstruction and advancing both biological research and precision medicine.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents a transformer-based architecture to predict DNA methylation. The key innovation is represented by the use of "paired" gene expression profiles as sources of additional context, which enables prediction at the single CpG site even in the absence of information on methylation (but with expression measurements). The proposed approach, called MethylProphet, is tested on data from major consortia (ENCODE and TGCA) demonstrating a good level of accuracy.

### Strengths
- The paper is well written, the task is well motivated and the approach is described in sufficient detail as to be clear whet the authors are doing and why.
- The topic is of interest, although with caveats (see weaknesses).
- The idea of using gene expression as context is novel and certainly of relevance to the task.

### Weaknesses
I have many reservations about the paper, here are the main ones:
- Gene expression and methylation are well known to be correlated, so in a sense adding expression as context is a natural idea. The question is whether adding expression makes the task still worthwhile. Much of the interest in methylation stems from its mechanistic role in regulating expression, as well as its biomarker function. If you already have expression measurements, the need to know much about methylation might be questionable.
- Methylation is poorly understood in general but most CpGs are either unmethylated in islands associated with active genes, or methylated in the ocean or near repressed genes. This already suggests that knowledge of expression could provide a simple model with accuracy on a comparable scale (if in island and gene on, then 0, otherwise 1).
- The comparisons are very limited, and the Fig 5 comparison against a method that does not use expression shows a fairly limited improvement despite the extra data.

### Questions
- Methylation arrays often provide an aggregated reading for nearby CpGs, how do you deal with that?
- How does the model perform on special CpGs, e.g. the ones used in clocks?
- How do you integrate coverage information in your loss function?

### Soundness
2

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
4

### Summary
The paper introduces a Transformer-based model for predicting whole-genome DNA methylation profiles from gene expression and DNA sequence alone, without requiring measured methylation values. Trained on billion-scale ENCODE and TCGA datasets, it achieves strong performance in zero-shot methylation prediction across both unmeasured CpG sites and unseen samples.

### Strengths
1. This paper addresses a key limitation of imputation-based methods for DNAm prediction, allowing application to entirely new samples.

2. The paper proposes a new formulation for the DNAm prediction task, which is intuitive and well-motivated.

3. The authors conduct a comprehensive evaluation which includes multiple validation splits to test model’s generalization ability.

4. The manuscript is well-written and well-structured, supported by informative results.

### Weaknesses
1. The paper does not clearly explain why this task should be feasible, especially why gene expression is used as an input modality.

2. It would be better to include an analysis of which patterns of CpGs are predictable and which are not.

3. Given the two types of input data, it is hard to know the contribution from each modality. The paper should include an ablation study comparing the model to sequence-only and expression-only baselines.

### Questions
1. How sensitive is the model's performance to the 1kb sequence window size?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces MethylProphet, a proof-of-concept deep learning model that demonstrates the feasibility of predicting genome-wide DNA methylation using only gene expression and genomic context, without requiring any methylation data as input.

### Strengths
1. The methodological exposition is characterized by its clarity and high quality.
2. The objective of integrating multi-modal information is both well-defined and effective.

### Weaknesses
The primary issue with this paper is that while its core innovation is the mDNA-free prediction, it overlooks existing methods capable of predicting methylation sites from either gene expression alone or DNA sequence alone. Therefore, it is essential to establish a comparative baseline to demonstrate the synergistic advantage of fusing both modalities (gene expression and DNA sequence) over relying on a single modality.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents MethylProphet, a Transformer model trained to prediction DNA methylation from gene expression, DNA sequence, and genomic annotations (such as CpG island annotation). Their approach is novel because it allows for whole genome DNA methylation inference for samples without any measure DNA methylation sites.

Their results show that their model can highly accurately predict methylation status across CpG sites in the genome for unseen samples. However, the performance for predicting methylation status at individual CpG sites across samples is only moderate, even for CpG site seen during training.

Overall, the results presented are interesting and well executed, but the paper lacks sufficient baselines and ablations to prove that their model beats simple imputation techniques.

### Strengths
- The paper is generally well written, and the problem and approach are clearly presented.
- The paper seeks to solve a problem that has not yet been tackled by other models in the field — whole genome methylation inference — and demonstrates that their approach is predictive.
- The authors validate their results on two independent datasets and multiple different methylation sequencing assays. Their results are quite robust across datasets and assays
- The validation splits and evaluation metrics are well designed.

### Weaknesses
In my opinion, the main weakness of this paper is a lack of sufficient baselines and ablations.
- There is only a comparison to one baseline “the CNN-based attention model in Levy-Jurgenson et al. (2019b)” and only on one of their two datasets. Moreover, from the details given in the appendix, it seems that MethylProphet and the baseline are not trained with the exact same information. In particular, it seems that MethylProphet is trained with additional genomic annotations that the CNN is not. For a fair comparison, MethylProphet should be trained with the exact same features/data as the baseline
- The point is taken that a main innovation of the model is that it can perform whole genome methylation inference, which cannot be done by many of the prior models in this space. However, it would still be worthwhile to see more comprehensive baselines against popular models in the field — such as CpGPT and DeepCpG — on the in-distribution generalization task. The results presented in Figure 5 are only against one baseline model and on one chromosome.
- For the MAC-PCC metric, it would be valuable to include another simple baseline of the mean methylation status across training samples
- There are minimal ablations performed on the components of the methylProphet model. In particular, CpG island context is highly indicative of methylation status, and an ablation should be performed to understand how much of the models performance can be attributed to this annotation alone.

Minor: 
Some figures and methods could be updated for clarity. 

    - For example, in table 4 MAS-PCC and MAC-CC are not defined (although they are defined later in the text)
    - In Figure 4, axes should be labeled with “samples” and “CpGs”. In addition, moving figure 4 earlier in the paper would be helpful

### Questions
- I do not see any results showing generalization performance between datasets - was this tested?

### Soundness
3

### Presentation
3

### Contribution
2
