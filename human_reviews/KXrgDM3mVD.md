# Distilling Structural Representations into Protein Sequence Models

- Decision: Accept (Poster)
- Scores: 6, 3, 6

## Abstract
Protein language (or sequence) models, like the popular ESM2, are now widely used tools for extracting evolution-based protein representations and have achieved significant success on core downstream biological tasks.
A major open problem is how to obtain representations that best capture both the sequence evolutionary history and the atomic structural properties of proteins in general. 
We introduce **I**mplicit **S**equence **M**odel, a sequence-only input model with structurally-enriched representations that outperforms state-of-the-art sequence models on several well-studied benchmarks including mutation stability assessment and structure prediction. 
Our key innovations are a microenvironment-based Autoencoder for generating structure tokens and a self-supervised training objective that distills these tokens into ESM2's pre-trained model. 
Notably, we make ISM's structure-enriched weights easily accessible for any application using the ESM2 framework.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Though AlphaFold2 and its successors have revolutionized protein structure prediction---along with other tasks that require rich embeddings of protein residues---the computational expense of the multiple sequence alignments (MSAs) these models require as input has spurred interest in so-called "single-sequence" models: language models conditioned only on the sequence of the protein in question, and no other structural or co-evolutionary information. So far, these have lagged behind their coevolutionary counterparts on most benchmarks. In an attempt to narrow the gap, the authors propose Implicit Structure Modeling (ISM), which finetunes protein language models with an additional structural loss term for quantized structure tokens. The resulting models can be used as a drop-in replacement for popular single-sequence models but perform better at tasks that require knowledge of structure.

### Strengths
The paper is clearly written (with some exceptions---see below), well-motivated, and is fairly comprehensive, including results for most important baselines as well as ablations and analysis. The method is interesting, and it's nice that it's compatible with ESM software.

### Weaknesses
While the ISM model compares favorably to others, like SaProt and ESM2, I have two gripes: 1. the difference in performance compared to e.g. SaProt seems fairly marginal across the board, with the strongest improvements in short- and medium-range contact prediction and one of the two thermostability benchmarks (but not the other), and 2.  given the results of the ablations in Table 4, most of the heavy lifting seems to be done by MutRank, not the novel autoencoder. I think I'll need to see a few things before I consider raising my score:

1. Full results of the MutRank model in Tables 1 and 2.
2. Some kind of confidence intervals for the results in Table 4b. It's unclear to me that the small gap there isn't just noise or the result of "ensembling" two different but closely related methods. What if you just trained two independent MR models and ensembled them, for example?
3. More details about how exactly MutRank is incorporated into the model.

There are a few other small details missing as well. Given that the ISM model in Table 1 descends from SoloSeq, you should include SoloSeq results there as well. I think it would also be useful here to include SOTA results (e.g. AlphaFold3), just to illustrate how close ISM is to closing the gap.

Random bits and bobs (no bearing on score):
- The citation for the sentence "MutRank adds a self-supervised..." is wrong.
- There's a typo in Table 1 (Strucure)

### Questions
Why do the authors use a custom structure autoencoder and not, say, Evoformer embeddings?
Could you give more details about the PDB dataset used? E.g. what is the cutoff?

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposed a protein sequence model enhanced by structure distillation.

### Strengths
- The main idea of this paper—integrating protein structure information into a protein sequence model—is clear and direct. However, this concept is not entirely novel and has been explored in previous works, such as “DistilProtBert.”

### Weaknesses
- The pretraining task for the Atomic Autoencoder lead to data leakage, raising concerns about its effectiveness.
- Most design choices lack justification. For instance, it is unclear why a two-stage training process was chosen, why multiple loss functions are used, or why ablation studies are absent.
- The performance improvement appears minimal. For example, in comparison to ESM-2, the structure prediction accuracy shows only a slight increase (lddt score: 82 vs. 84).
- The overall contribution of this paper is unclear. The main idea is not novel, the design of the Atomic Autoencoder lacks clarity, and the proposed models offer only marginal performance gains.

### Questions
As noted in Weakness 2, most questions arise from the design choices. What is the rationale behind these specific design decisions, and to what extent do they contribute to performance improvement?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces a novel training strategy for protein language model, which distills structural information into protein sequence models. The main contribution of this paper is the Implicit Strucutre Model (ISM). It utilizes a micorenrironment-based Autoencoder for generating structure tokens and designs a training loss to achieve the self-supervised traning objective of distilling structure tokens. The method demonstrates comparative performence on stractural benchmarks and downstream finetuning tasks.

### Strengths
1. The microenvironment-based autoencoder enables the generation of structure tokens with implicit structure information;
2. The authors conducted comprehensive evaluations on various tasks and compared with SOTA methods;
3. The method is compatible with any framework built using ESM2.

### Weaknesses
1. The performence improvement presented in Table 2 and 3 seems limited. For instance, it seems that MutateEverything (AF) demonstrates better results than ISM even with fewer pretrain data (PDB only); moreover, in table 3, ISM does not demonstrate superior performance than ESM2 and SaProt.
2. The poteintial impact on the community of protein studies is not clearly presented. The authors could better articulate the potential impact, such as providing examples of how ISM could enable new applications or analyses in protein research that were not previously possible with existing models.

### Questions
Protein LMs such as AlphaFold and ESMFold take protein sequences as inputs and have the ability to generate structure information. It may indicate that these models also captures structure information implicitly without introducing an "ISM like" module and finetuning strategy. Thus, it makes me confused on the contribution of the method for the community of computational biology.

The authors may directly compare ISM's performance to AlphaFold and ESMFold on structure-related tasks, explain how ISM's approach differs from or improves upon the implicit structure learning in those models.

It would be benifit if the authors could clarify what unique capabilities or advantages ISM provides compared to existing protein language models.

### Soundness
3

### Presentation
3

### Contribution
2
