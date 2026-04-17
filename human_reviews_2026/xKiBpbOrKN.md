# Mutation2Text:  A Unified Protein and Text Language Model for Explaining Mutation Effects

- Decision: Reject
- Scores: 2, 2, 6, 6

## Abstract
Understanding the functional consequences of protein mutations is crucial for diagnosing and preventing diseases like cancer. However, existing protein language models (PLMs) are limited by their black-box nature, inability to process full-length proteins without truncation, primary focus on single-nucleotide variants, and lack of directional interpretation (e.g., distinguishing loss-of-function from gain-of-function). We introduce $Mutation2Text$, a multimodal generative PLM designed to generate human-understandable, rationale-based explanations for diverse mutations, including substitutions, insertions, deletions, and frameshifts. Mutation2Text employs a gated cross-attention mechanism to explicitly contrast wild-type and mutated proteins and uses a Perceiver Resampler for length-invariant encoding. We constructed Mutation2TextQA, the largest mutation interpretation dataset to date, comprising millions of question-answer pairs with substantial lexical and semantic diversity, mined from published literature, facilitating robust generalization across mutation contexts. Mutation2Text training follows a progressive three-stage approach: (1) foundational protein function grounding with UniProt annotations; (2) comprehensive biological understanding using 10 million biomedical literature-derived QA pairs; and (3) mutation-focused reasoning leveraging Mutation2TextQA. Mutation2Text consistently outperforms baselines on pathogenicity prediction, functional classification, mutation annotation, and open-ended mutation QA tasks. Our analysis reveals a novel use of LLMs for variant interpretation, providing transparent and rationale-driven predictions to enhance clinical interpretability.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces Mutation2Text, a multi-modal generative model to offer human-understandable explanations for clinical variants. With a two-branch architecture that encodes wild-type and mutated sequences and delta features, the model could handle a diverse set of mutations including substitutions, indels, and frameshifts. The paper also constructs Mutation2TextQA, a large-scale dataset with millions of question-answer pairs for protein mutation interpretation. Experiment results show that Mutation2Text achieves state-of-the-art results in mutation explanation, pathogenicity prediction, and disease association of clinical variants. The authors also notice a gap between mutation representation learning and generating descriptive texts.

### Strengths
- The work presents an intuitive approach that effectively addresses the limitations of pLMs and prior works in handling indels and frameshifts.
- The collected Mutation2TextQA dataset, upon public release, will enrich the scale, diversity, and label quality of existing mutation benchmarks, which is evidenced by the qualitative analysis in Section 5. The annotation and processing procedure of the dataset is carefully designed.
- The observed discrepancy between latent knowledge and textual output is interesting, suggesting future research directions to address this issue.

### Weaknesses
- The main contribution of the paper is unclear. To my knowledge, the work distinguishes itself from existing studies by attempting to address the limitations of existing pLMs and mutation models in handling indels and frameshifts for clinical variants. However, the paper lacks a thorough discussion on the technical challenges and justification for the Mutation2Text architecture over previous multi-modal models [1] and specialized models [2, 3]. Moreover, analysis of the distribution of different mutation types and the model performance is lacking.
- Some parts of the paper are inconsistent and ambiguous. For example:
  - In the delta-feature path, the mutation metadata is incorporated. Details of how to obtain, encode, and fuse such metadata are missing.
  - In Section 2.3, the authors introduce two datasets (Mutation2TextQA and ClinVarQA), but their relationships are unclear.
  - In Section 3, two tasks, i.e., pathogenicity prediction and disease prediction, are introduced. However, their definitions are not formally presented, and their differences and the correspondence to different subsets of the dataset are unknown.
  - The results of baselines introduced in Lines 287-292 are not reported in Table 2.
  - In Section 3.4 and Table 5, the experiment settings and ablation details are missing, leading to confusion. The meaning of "w/ delta embedding" is unclear, and the results are inconsistent with Table 2. 
- The experiments are not carried out with sufficient depth. The evaluation is constrained on overlapping-based metrics and BERT similarity, and I'm concerned if the model merely mimics the output formats of the distilled DeepSeek outputs. Expert evaluation [1] or LLM-as-a-judge [4] should be incorporated. On Mutation2TextQA, stronger baselines like few-shot prompting with closed-scoured LLMs like GPT-5 or Gemini-2.5 and supervised models are lacking. Besides, in Section 6, the results should be compared with specialized classification models such as ClinPred [5] and MetaRNN [6]. 

Refs.

[1] MutaPLM: Protein language modeling for mutation explanation and engineering.

[2] Genome-wide prediction of disease variant effects with a deep protein language model

[3] Accurate proteome-wide missense variant effect prediction with AlphaMissense

[4] A Survey on LLM-as-a-Judge

[5] Clinpred: prediction tool to identify disease-relevant nonsynonymous single-nucleotide variants

[6] MetaRNN: differentiating rare pathogenic and rare benign missense SNVs and InDels using deep learning

### Questions
My major concerns have been listed in the Weaknesses. Here are some additional questions:
- In Lines 52-53, the authors argue that pLMs are fundamentally ill-suited for modeling disease-causing variants. However, I do not agree with this statement as some pLMs show promising zero-shot classification results on the ClinVar benchmark (See Table A18 in [1]).
- It is claimed that Mutation2Text _contrasts_ wild-type and mutant sequences. However, no contrastive objective [2] is introduced, and the cross-attention block behaves more like a fusion module.
- The authors mentioned that Mutation2Text could handle full-length sequences. I'm curious about how the model handles extremely long sequences that exceed the context length of ESM-3. Additionally, is there a possible connection between model performance and the protein length?
- The model adopts ESM-3 to initialize the pLM. However, the performance of ESM-3 (1.4B Open) on mutation benchmarks is inferior to ESM-2 (650M) [3], making the choice questionable. Besides, I'm curious if introducing protein structure with ESM-3 could be beneficial, since the structures dominate protein functions.
- The QA pairs in the dataset are generated by the DeepSeek model. Does it produce similar QA pairs from the same document or hallucinate? If yes, how do the authors conduct deduplication and filtering?
- The Mutation2Text is jointly trained on MutaDescribe, ClinVar-QA, and Mutation2Text-QA, raising data leakage concerns. Could the improvements over MutaPLM on the medium and hard sets of MutaDescribe arise from similar wild-type sequences in the other two datasets?
- In Table 3, Mutation2Text achieves the best BLEU scores but significantly worse BERT scores. Is there a possible explanation for this result?

Refs.

[1] ProteinGym: Large-Scale Benchmarks for Protein Fitness Prediction and Design

[2] Representation Learning with Contrastive Predictive Coding

[3] Simulating 500 million years of evolution with a language model

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors are interested in predicting the effects of clinical pathogenic variants. They note that state-of-the-art models only model similarity to sequences in nature, and, in principle, may fail to identify important classes of mutations as pathogenic.

To address this, the authors compile a set of mutation -> pathology text data form Clinvar and pubmed (processed using an LLM) and train a model on this data. They show their particular fusion method does better than naive methods in fitting this data.

### Strengths
* They seek to generate a large dataset of question-answer pairs.

### Weaknesses
* I have concerns to do with training on test (see questions below).
* I also have concerns to do with the training data.
* They did not test the sensitivity of their results to their data templates, or LLMs used to generate the data.

### Questions
* [This work](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=8Gm8COsAAAAJ&sortby=pubdate&citation_for_view=8Gm8COsAAAAJ:0EnyYjriUFMC) outlines the challenges of releaseing a variant effect predictor, including the risks of circular logic. For example, many ClinVar labels come from experiments in literature, and training on those experiments essentially results in training on test. Could you  more carefully describe how you avoid this?
* Since you use a pretrained LLM, how do you know this model didn't see the material you are testing on?
* It seems Mutation2Text was trained on your QA datasets. Is it then fair to use BLEU scores to compare its outputs with the outputs of other models? PErhaps it just learned the formatting of your data?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper constructs a Mutation2TextQA dataset along with a Mutation2Text protein language model to interpret effects of protein sequence mutation on its function. By using a gated cross-attention mechanism as well as a Flamingo-style perceiver, Mutation2Text is able to produce not only embeddings of the sequence but also delta-embeddings owing to the mutation. The model achieves excellent performance on downstream tasks with the internal embeddings, but also observes a loss of information when using LLM decoder to generate the final answer.

### Strengths
* The paper addresses protein mutation effect prediction problem, which is a very important yet challenging topic of proteomics and biology. 
* The novel Mutation2Text architecture design with the gated cross attention and perceiver helps avoiding truncation of long protein sequences as well as improves delta-mutation interpretability. 
* A particular observation of "loss in translation" provides important insights for similar multimodal research topics.

### Weaknesses
* The augmentation of Mutation2TextQA dataset with LLM lacks validation. Teacher LLMs can hallucinate, especially for complex scientific topics like biology. 
* The paper does not explain in more details the "loss in translation" effect. It is a very important discovery, but it would be better to have further exploration of the mechanism behind, and more importantly how can we solve it. 
* The architecture design of Mutation2Text may be a bit redundant to me. The intuition of a standalone mutation feature extraction module is not very well explained with the existence of the gated cross attention module.

### Questions
Please address my three major concerns in the "Weakness" section first. I will reassess after the rebuttal. Other miscellaneous questions are as follows: 

* What's the intuition behind the "wildtype context" in the Mutation2Text model architecture? Have you tested the performance of the model without such context? 
* What's the "learned gap embeddings" in [Appendix D, ESM-based Difference Features]? How is it learned and what's the intuition behind this design? Maybe I missed it, but I couldn't find any explanation.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces Mutation2Text, a model that combines protein and language modeling to explain how mutations affect protein function. It takes both the wild-type and mutated sequences, compares them through a gated attention mechanism, compresses long proteins into a small set of representations using a Perceiver module, and maps everything into the language model’s space with a simple MLP. The model also adds a special token to represent the difference between the two sequences. The authors build a new dataset called Mutation2TextQA, which links PubMed papers and ClinVar records to create question–answer pairs about mutation effects. They test the model on several tasks, including open-ended mutation explanation, disease association, and pathogenicity prediction. Mutation2Text sets new benchmarks on text-based evaluation metrics and highlights an important finding: while the internal protein embeddings predict pathogenicity very well (AUC 0.96), that precision partly disappears when the model explains the result in natural language (AUC 0.85).

### Strengths
The paper presents a clear and creative approach to explaining protein mutations by directly contrasting wild-type and mutated sequences. It introduces a special delta token and a Perceiver module to handle long proteins efficiently and builds a large QA dataset, Mutation2TextQA, grounded in biomedical literature. The system design is coherent and shows consistent improvements on text-based tasks like MutaDescribe and ClinVarQA. The work is clearly written and highlights an important finding—the “lost-in-translation” gap between strong internal representations and less precise language explanations—offering a solid direction for future research.

### Weaknesses
Most of the evaluation still depends on BLEU, ROUGE, and BERTScore, which don’t really capture how factually correct the biomedical explanations are. It would be stronger if the authors added an evidence-based check, like matching each claim to the right PubMed sentence, and included some expert review on a sample of the outputs.

The comparison on pathogenicity is a bit indirect. The paper shows how internal embeddings outperform text output, but it doesn’t compare directly with existing pathogenicity predictors on the same ClinVar split. A clear, side-by-side ROC or AUC comparison would make the results much more convincing.

The paper also claims to handle many kinds of variants, including SNVs, indels, and long proteins, but it never breaks that down. Showing results by variant type and protein length would make the method’s generality easier to judge.

### Questions
You mention that performance drops when moving from embeddings to text, but which kinds of mutations actually lose interpretability? It would be helpful to see a breakdown by mutation type or protein length to identify where the explanations fail. Have you checked how the correlation of mutation effect predictions changes after adding natural language—for example, comparing embedding-based and text-based results on real mutation data to quantify how much signal is lost? Since the evaluation currently relies only on automatic scores like BLEU and ROUGE, could you also include an expert review or use a biomedical model to verify whether the explanations are factually correct?

In addition, could you provide a more direct comparison of pathogenicity performance on the same ClinVar split using identical train and test sets as strong baselines, and report calibration results such as ECE for the embedding-level predictors? It would also strengthen the paper to show robustness across variant types (SNV, indel, frameshift, multi-site) and protein length ranges. Finally, for the Mutation2TextQA dataset, please add more information about data quality control—such as error breakdowns, inter-annotator agreement on a checked subset, and details on how the homology-based split was constructed to prevent any data leakage.

### Soundness
3

### Presentation
3

### Contribution
2
