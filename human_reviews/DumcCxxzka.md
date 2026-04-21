# RNAinformer: Generative RNA Design with Tertiary Interactions

- Avg Score: 5.20
- Decision: Reject
- Scores: 5, 6, 5, 5, 5

## Abstract
The function of an RNA molecule depends on its structure and a strong structure-to-function relationship is already achieved on the secondary structure level of RNA. Therefore, the secondary structure based design of RNAs is one of the major challenges in computational biology. A common approach of RNA design is inverse RNA folding. However, existing RNA design approaches cannot invert all folding algorithms because they cannot represent all types of base interactions. In this work, we propose RNAinformer, a novel generative transformer based approach to the inverse RNA folding problem. Leveraging axial-attention, we directly model the secondary structure input represented as an adjacency matrix in a 2D latent space, which allows us to invert all existing secondary structure prediction algorithms. Consequently, RNAinformer is the first model capable of designing RNAs from secondary structures with all base interactions, including non-canonical base pairs and tertiary interactions like pseudoknots and base multiplets. We demonstrate RNAinformer’s state-of-the-art performance across different RNA design benchmarks and showcase its novelty by inverting different RNA secondary structure prediction algorithms.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This manuscript explores a more general RNA inverse folding problem, beyond the canonical base-pairing constraint. The authors propose a Transformer-based learning model, leveraging the adjacency matrix to capture more intricate nucleotide tertiary interactions and map them to sequences. Comprehensive experiments are conducted to validate the effectiveness of this model.

### Strengths
1.	This paper addresses the limitations of traditional dot-bracket secondary structure inverse folding by exploring adjacent-matrix based methods to represent more complex structural motifs. This approach aligns more closely with the practical requirements of functional RNA design.
2.	In the absence of prior research on this topic, this paper constructs extensive training data and diverse test tasks, demonstrating the capacities of the proposed model in multiple aspects.
3.	The overall writing structure of this paper is clear and logically organized, making it easy to follow.

### Weaknesses
1.	The related work in this paper is insufficient. It overlooks a class of RNA inverse folding methods based on tertiary structure, such as RDesign [1] and RiboDiffusion [2], which directly use the RNA tertiary structure backbone as model input and implicitly model structural interactions like pseudoknots. These methods are highly relevant to the topic of the paper.

2.	The benchmarking in this paper has several flaws:
    (1)	The folding-back algorithm is overly simplistic and lacks diversity. Tertiary structure prediction methods should be included to assess whether the designed sequences meet expected tertiary interactions. Additionally, the errors caused by the folding-back algorithm are not adequately explained.
    (2)	The model's performance under novel structures, i.e., samples with significant differences from the training set, is not analyzed.
(3)	There are relatively few baseline methods.  For adjacency matrices, graph neural network-based methods and ResNet-based methods are commonly used to represent RNA data and could be easily adapted to the current task.
(4)	Although it is reasonable to use synthetic data for proof of concept, using it as the core test set is far from the actual application scenario and may introduce more unexpected errors.
(5)	There is a lack of comparison on public benchmarks, such as Eterna100, which is crucial to demonstrate the basic capabilities of the model.

3.	The paper lacks case study results to demonstrate the correctness, rationality, and novelty of the designed sequences.

4.	For the machine learning community, the method in this paper adopts a well-studied Transformer structure and classical loss function, making it difficult to claim a technical contribution. For the biology community, the paper lacks rigorous and comprehensive evaluation to demonstrate its advantages and reliability. Overall, this manuscript does not yet meet the standard for publication.

[1] RDesign: Hierarchical Data-efficient Representation Learning for Tertiary Structure-based RNA Design. The Twelfth International Conference on Learning Representations. 2024.

[2] RiboDiffusion: tertiary structure-based RNA inverse folding with generative diffusion models. Bioinformatics 40. Supplement_1 (2024): i347-i356.

### Questions
Please refer to the Weakness section for more details.
1. Can some GNN-based baseline methods be added?
2. Can benchmarking be done on some public test sets, such as Eterna100 and Openknot?
3. Can the model design performance on relatively novel structures be tested on natural RNA of CASP15?
4. Can the designed sequence be folded using the tertiary structure prediction model?

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
The authors present RNAInformer, an RNA secondary structure inverse folding method. The authors identify that there has been strong recent progress in RNA secondary structure prediction, and use this as the motivation for the work. The authors additionally identify that the vocabulary of base-base interactions in RNA secondary structure is much larger than what the community typically models and account for this in their method through an axial attention mechanism

### Strengths
Overall I think the work is quite carefully considered: from the problem statement to the dataset setup and splits, and the evaluation. The empirical results are strong, across the different design tasks and the authors demonstrate the ability to incorporate additional property constraints (GC content) convincingly. The paper is laid out clearly and well-written.

### Weaknesses
The predominant weakness in this work is the modest technical novelty in the proposed architecture. This reviewer is of the opinion that sound and well-executed applied work (such as this) has a place in venues such as ICLR but the judgment on this rests with the AC.

### Questions
* While the hyperparameters are described, there is little discussion of how they were obtained; how much sweeping did the authors perform?

* Do the authors give any consideration to non-canonical/modified nucleotides?

* Could the authors share some reasoning why the model trained on synthetic data performs better than the finetuned model?

* The pairwise hamming distance for diversity is designed to account for base pair flips?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The authors propose RNAinformer, a new method for solving the problem of inverse RNA design. This problem considers finding the nucleotide sequence of RNAs that will result in the desired structure once they fold. Authors approach this problem in a more general setting by considering pseudoknots and base multiplets. They train a conditional transformer model to generate RNA sequences conditioned on the desired GC content and possibly additional constraints. They also introduce a new pipeline for generating synthetic data that can be used for generative models.

### Strengths
Authors have invested time to design a pipeline for generating synthetic data that can become useful for designing learning-based methods since the experimentally verified sequences are more costly to generate.

Authors consider the problem in the general form where different types of base pairing (e.g., non-canonical base-pairs and pseudoknots) exist, and also allow additional constraints (e.g., GC content) to be specified. The problem is highly important and good solutions can be very impactful.

### Weaknesses
There are many recent methods for solving the problem of inverse RNA design which have not been used for comparisons in this paper. The comparisons are limited to only a few of the prior works (mostly only 2). Some of these prior methods have similarly used conditional autoregressive formulation similar to this work. The paper in [1] targets an even more general problem by inverse RNA tertiary structure which solves the RNA inverse problem as a subproblem. Without thorough comparisons it is hard to ensure the merits of the newly proposed approach. 


1. Tan, C., Zhang, Y., Gao, Z., Hu, B., Li, S., Liu, Z., & Li, S. Z. (2024). RDesign: Hierarchical Data-efficient Representation Learning for Tertiary Structure-based RNA Design. In The Twelfth International Conference on Learning Representations.

2. Rubio-Largo, Á., Lozano-García, N., Granado-Criado, J. M., & Vega-Rodríguez, M. A. (2023). Solving the RNA inverse folding problem through target structure decomposition and Multiobjective Evolutionary Computation. Applied Soft Computing, 110779.

3. Merleau, N. S., & Smerlak, M. (2022). aRNAque: an evolutionary algorithm for inverse pseudoknotted RNA folding inspired by Lévy flights. BMC bioinformatics, 23(1), 335.


There is also a similarity between the protein inverse folding problem and RNA inverse folding and since the former has been studied more, most models can still be applicable to the RNAs (instead of 20 enzymes we will have 4 nucleotides). Therefore, works in RNA inverse folding still use these methods as baselines. For example the following works are used as baselines for RNA inverse problems as well.

4. Ingraham, J., Garg, V., Barzilay, R., & Jaakkola, T. (2019). Generative models for graph-based protein design. Advances in neural information processing systems, 32.

5. Gao, Z., Tan, C., & Li, S. Z.(2023) PiFold: Toward effective and efficient protein inverse folding. In The Eleventh International Conference on Learning Representations.


For many of the comparisons, such as the ones on nested structures in which the proposed method achieves significantly worse results than the baseline methods (SAMFEO), the authors justify the presented results by highlighting the diversity of their generated outcomes. However, the first goal is to derive valid sequences. If neither of the sequences for a task is not valid what would be the benefit of diversity?

That being said, even for most other prior methods, we can still tune the diversity. For example you can change the temperature value in the Boltzmann sampling part of SAMFEO to get a less skewed distribution that leads to a higher diversity. However, it is important to have a skewed enough distribution to make sure we have a high recall among a small number of candidates.

The method authors use for measuring the diversity of the valid sequences is computing the mean of pairwise hamming distances between the sequences. This may lead to an overestimation of the diversity because if two RNA sequences have just an indel (unpaired nucleotide) compared to each other, it considers all the subsequence bases as different. I think in this case something similar to the Needleman-Wunsch algorithm would be a better measure of dissimilarity.

It is also not very clear why diversity is an important factor here because unlike image generation and text generation, for RNA design it is more important to get the correct design among the few samples. For example, when looking for a specific RNA to bind to the snoRNA of a new virus to inhibit its activity, why should one care about the diversity of the proposed solutions rather than just having the top few solutions that are more likely to work. It is more important to have the expected RNA among the top few generated solutions to also decrease the costs of downstream in-vivo experiments.

Some results such as the one presented in Figure 4 might portray the amount of non-canonical base-pairs (NC) as a measure that has to be maximized. However, although these non-canonical base-pairs are present in folded RNAs, this amount has to be the correct amount. If these values are maximized by the model but does not lead to valid sequences then why should it be considered as a good thing?

For experimentally verified 3D structures, which can be considered the gold standard set for evaluation, authors only report the results of their method and do not compare it to any other baseline. From table 17, the ratio of the valid sequences seems to be very low and it would be important to see if prior methods do not outperform RNAinformer on this dataset.

For the experiments on RNA design with pseudoknots authors only compare their method to antaRNA which is a relatively old method (2015) while many newer methods with superior results have been published since then.

The length of the supported sequences is mentioned to be capped at 200 in the experiments. That is relatively low to what has been done in recently proposed methods (e.g., 900 in SAMFEO and 500 nucleotides in RDESIGN -- see [1] above). This is an important shortcoming because it is well-known that with the increased length of RNA the problem becomes more difficult. Authors have mentioned this in their Limitations section, but they mention that it is enough for current benchmarks while there are other benchmarks that contain RNAs with more nucleotides and they are simply not used in the presented experiments.

While authors propose a new pipeline for designing their synthetic datasets, it would have been better to use the datasets used by prior methods (e.g., Eterna100 or the one curated in [1]) to show the performance of the presented method in comparison to the baseline methods, and then mention the motivation behind using a new pipeline for generating a dataset.

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
### A method for 2d structure based RNA inverse folding enabling - for the first time - arbitrary interaction types (e.g. pseudo-knots, non-canonical base pairs, ...)

This paper introduces a secondary structure based RNA inverse folding model (RNAinformer) that is capable of designing RNA sequences from secondary structures with arbitrary interaction types (e.g. non-canonical base pairs, pseudo knots, base multiplets) that were not representable in previous 2D based inverse folding methods. This improvement of being able to represent these arbitrary interaction types is achieved by working with the more expressive adjacency matrix representation instead of dot-bracket representations of the secondary structure.

The RNAinformer model is based on an auto-regressive encoder-decoder transformer. The secondary structure (in the form of an adjacency matrix) is encoded via axial attention (similar to the RNAformer structure prediction model) and finally pooled from a 2d to a 1d vector that is passed to the decoder for decoding into an RNA sequence. RNAinformer also supports constrained generation based on masked sequences, which are embedded into a 2d representation by the encoder if provided, or desired GC content (linearly embedded and added to the embedding).

The authors also make an interesting, strong claim that training on synthetic data only improves performance over training with experimental data, that -- if true -- would be of significant interest to the community.

### Strengths
1. The paper is well written, clear and has a nice flow.

2. The provided codebase looks well structured & documented upon a first spot check. 

3. I'm a big fan of your spider plots showing a variety of metrics of interest (valid sequences, diversity, solved ,...)

4. The performance of the model and its enhanced design capabilities (GC content, masking, leveraging structure beyond what can be represented in simple dot-brackets) are promising and of interest to the community. If the authors can demonstrate that these also hold when using non-synthetic training data or further support their claim on the synthetic training data being superior to currently available experimental training data this would be of much interest to the community. (c.f. also weaknesses).

### Weaknesses
1. A link to the 3D inverse folding literature is currently missing: In the related work section I would have expected to see a mention of the (deep learning based) inverse folding efforts based on 3D structure (e.g. Rosetta, gRNADe, etc.). How does secondary structure-based inverse folding perform compared to 3D based inverse folding, in cases where 3D structure is available? This may represent a route of further testing and strengthening the hypothesis that the essential information for rna structure-function relations are encoded in 2D connectivity patterns (base pairs, psuedo-knots, base multiplets, ...).

2. I am somewhat concerned by the purely synthetic data based training strategy, since synthetic data are created with the same method that is used for evaluation. The authors argue that this allows side-stepping the data gap in gold standard secondary structure data, which is limited in the PDB. However, ultimately any secondary structure prediction method will have been trained on some level of structural data, partially exhibiting the biases and limitations of the datasets that the authors discussed. In addition to those, the models might also have certain model-specific biases that the inverse folding model may then pick up. Since the evaluation is also done by the same prediction model (e.g. RNAfold in 5.1) this risks reinforcing those model-specific biases. While the authors provide an experiment in the appendix to address this, I believe this point deserves further discussion and should be featured more central stage (at the very least including table 20, and possibly a test also for some of the other tasks specifically). This point is quite interesting and -- if true and well supported -- could be of significant interest to the community beyond the method in this paper alone.
Some analyses I would like to see in this regard: 
- a comparison of training on synthetic data from model 1, but an evaluation with another, independent structure prediction model 2.
- an analysis of the sequence recovery (an imperfect metric, I know, but still somewhat informative to exclude cases of the inverse folding model overfitting on quirks of the structure prediction model). If you were to go all out, an evolutionary inspired recovery may be even more informative (c.f. https://openreview.net/forum?id=y5L8W0KRUX&referrer=%5Bthe%20profile%20of%20Chengyue%20Gong%5D(%2Fprofile%3Fid%3D~Chengyue_Gong1) in the protein world

### Questions
- In Table 1, it'd be helpful if the authors could highlight what % means (here % solved), as well as the n=... that was used for this estimate and the topk from which the answer was picked. From the description, it is not quite clear to me how those % were obtained.

- Were all benchmarked methods re-trained on the same synthetic data?  (for each of the tasks)

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposes RNAinformer, a novel generative transformer based approach to the inverse RNA folding problem. Leveraging axial-attention, they model the secondary structure input represented as an adjacency matrix in a 2D latent space, which allows us to invert all existing secondary structure prediction algorithms. The authors claim that RNAinformer is the first model capable of designing RNAs from secondary structures with all base interactions, including non-canonical base pairs and tertiary interactions like pseudoknots and base multiplets.

### Strengths
This paper studies RNA inverse folding (from secondary structures), which is an important problem in biology

### Weaknesses
* The methodology is a standard transformer and lacks innovation

### Questions
* Have you tried finetuning RNAinformer using the data from PDB? You said secondary structures derived from PDB are the golden standard but the dataset size is small. That's why you used synthetic data to pre-train. I wonder if it can improve your model performance with additional finetuning.

### Soundness
2

### Presentation
2

### Contribution
2
