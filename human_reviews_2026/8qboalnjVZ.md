# Lightweight Midas Touch: A Paired-learning Framework for Flexible Design of Antimicrobial Peptides

- Avg Score: 3.33
- Decision: Reject
- Scores: 6, 2, 2

## Abstract
Antimicrobial peptides (AMPs) are promising potential therapeutic agents against drug-resistant pathogens, owing to their broad-spectrum activity and minimal risk of resistance. However, prevailing AMP generative paradigms privilege only the dyad of model and data: small-scale AMP data and the inductive bias of bespoke models jointly circumscribe the generated candidates. This tenacious coupling between training data and model parameters imposes a constraint on the exploration of AMP diversity. To alleviate this, we introduce a paired-learning framework, lightweight Midas touch (LMT), which has four components: (1) a primer set, (2) a mapping table, (3) a target set, and (4) a converter. During training, a mapping table explicitly associates the primers with the targets, and a lightweight converter is trained to internalize this mapping correspondence. During inference, the candidate distribution is steered by re-specifying the primers, avoiding the need to re-design the converter. By liberating the primer set and mapping table from the data-model dyad, the framework dissolves their erstwhile tight coupling and affords flexible control over generation. Comprehensive evaluations demonstrate that the proposed framework yields AMP candidates of markedly enriched diversity while attaining accuracy. Consequently, by delivering diverse and novel AMP candidates, LMT constitutes a potent strength in the global campaign against antimicrobial resistance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Lightweight Midas Touch (LMT), a paired-learning framework for antimicrobial peptide (AMP) design. Unlike traditional AMP generative models that tightly couple datasets and model parameters, LMT introduces two new components—a primer set and a mapping table—in addition to the target dataset and generative model. This design aims to decouple data from model biases, enabling flexible and controllable generation of diverse peptide candidates. The authors validate LMT against several state-of-the-art baselines (HydrAMP, PepDiffusion, AMPDesigner) using physicochemical and diversity metrics, reporting superior diversity and competitive accuracy. They also explore different configurations of the mapping table and primer set to demonstrate controllability and flexibility.

### Strengths
The paper presents a conceptually novel framework that generalizes several existing paradigms (e.g., diffusion and language models) under a unified formalism. The inclusion of a mapping table as a tunable control mechanism is elegant and well-motivated. The experimental evaluation is broad, covering diversity, accuracy, and controllability. Compared with existing models, the method achieves a good trade-off between peptide diversity and predictive performance. The theoretical discussion (e.g., Lemma 1 on entropy bounds) provides an interesting analytical perspective on diversity control. Overall, the paper offers a well-written and ambitious attempt to rethink controllable generative peptide design.

### Weaknesses
While conceptually intriguing, the practical novelty and biological significance of LMT are somewhat limited. The “paired-learning” idea largely formalizes an existing notion of conditioning or data pairing, and its empirical benefits are modest given the small dataset and limited validation (no wet-lab or structural confirmation). The theoretical analysis, though elegant, is superficial. Lemma 1 provides little practical insight beyond intuitive entropy control. Benchmarking lacks rigorous baselines such as protein-specific diffusion or autoregressive sequence models trained under similar data constraints. Moreover, the evaluation relies solely on computational predictors, which are known to produce false positives. Reproducibility is also questionable: the mapping table construction, random seeds, and parameter details are underspecified.

### Questions
1. The paired-learning framework resembles conditional generation or control through prompts in diffusion or transformer-based models. Authors should better clarify what new learning capability LMT introduces.
2. The results rely entirely on in silico AMP predictors (LSTM, BERT, ATT) with known bias; there is no experimental validation or external benchmark dataset.
3. The mapping table’s generation and training process are described conceptually but lack mathematical or algorithmic specificity—how is it optimized or regularized?
4. Theoretical analysis (e.g., Lemma 1) reads as a reformulation of intuitive properties, without empirical verification of information-theoretic claims.
5. While LMT yields “diverse” peptides, the biological or structural meaning of diversity (e.g., sequence motifs, charge distribution) is not examined.
6. Models like ProtGPT2 or ESM-IF could provide stronger baselines for peptide generation; excluding them weakens the comparative claims.
7. Important implementation details (mapping-table sampling, hyperparameters, training stability) are missing, making replication difficult.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a “paired-learning framework” (LMT) for antimicrobial peptide (AMP) generation. LMT comprises four components: a primer set, a target set (AMP dataset), a mapping table, and a lightweight converter (a two-layer Transformer). During training, primers and target sequences are paired via the mapping table to learn a mapping from primers to targets. At inference, changing the primer set is used to controllably steer the output distribution without redesigning the model architecture.

### Strengths
1. By making the primer and the mapping table explicit and manipulable, the framework provides a simple, implementable control channel. A small model can achieve distribution steering to some extent (e.g., sequence length distribution follows constraints imposed on inference primers).
2. In certain settings, the reported diversity metrics are significantly higher than baselines, and in some cases this coincides with higher predictive accuracy (e.g., LMTtmp intersection accuracy 0.904 vs. AMPDesigner 0.817).

### Weaknesses
1. In some indirect “data augmentation” settings (pairing multiple primers with the same target), accuracy increases but uniqueness decreases (redundancy rises). This suggests potential overfitting to the target distribution at the expense of novelty. The paper does not offer a systematic approach to balance control and diversity (e.g., penalties, deduplication strategies, sampling temperature, property coverage objectives).
2. Methodologically, the framework is essentially a seq2seq model that learns to map arbitrary primers to targets via forced pairing. With random mapping tables and semantically unrelated primers, training resembles memorizing a random code-to-target mapping. The mechanism enabling generalization to unseen primers and principled controllable generation is unclear and appears largely empirical (e.g., length controllability). Relative to established conditional generation, prompt-driven language models, or conditional diffusion, the novelty is more in presentation than in the core method.
3. Fairness of baseline comparison is questionable. Were all three baselines retrained on APD3 with equivalent settings? Many generative methods are originally trained on larger or different datasets; direct comparison may be unfair. Although the paper emphasizes using the same dataset, retraining details, ablations, and released code are missing, which hinders reproducibility and verification.
4. Key implementation details for the converter are insufficient: is it encoder–decoder or decoder-only? Loss, optimizer, learning rate, regularization, tokenization, positional encoding, sampling strategy, etc. are not specified beyond “default,” compromising reproducibility.
5. Reported “accuracy” relies entirely on the intersection of three deep learning predictors, and the authors acknowledge that these predictors yield only 74.37% intersection accuracy on APD3, implying substantial false positives/negatives. Predictor-based evaluation risks circular validation bias with the training distribution and does not convincingly demonstrate discovery of superior, truly effective AMPs.
6. No open-source code is provided, making reproduction difficult.

### Questions
Improve fairness and transparency in baseline comparisons. Retrain baselines under the same data and training budget with fully documented, reproducible protocols, and release code and configurations.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces Lightweight Midas Touch (LMT), a paired-learning framework that addresses challenges in antimicrobial peptide (AMP) generation by decoupling the traditionally tight coupling between training data and generative models. LMT augments this architecture with four key constituents: a primer set, a target dataset, an explicit mapping table, and a lightweight transformer-based converter. The framework operates on the principle that by liberating the primer set and mapping table from data-model constraints, practitioners gain modular control over generation through two mechanisms: (1) during training, different mapping tables produce distinct converters from the same target dataset, enabling exploration of diverse generative pathways, and (2) during inference, customized primers steer candidate generation without model retraining. Comprehensive evaluations on the APD3 database demonstrate that LMT achieves markedly superior diversity while maintaining competitive accuracy, effectively dissolving the inherent diversity-accuracy trade-off that plagues existing approaches.

### Strengths
1. This paper proposes a new pipeline for peptide generation, which consists of the primer set, the target set, the converter, and the mapping table. This converter can provide a more interpretable mapping between the primer and target peptide.
2. It assesses the model performance from multiple aspects to evaluate its effectiveness in generating diverse and accurate AMP. It also conducts a detailed study on the impact of the mapping table and test primers.

### Weaknesses
1. Some claims lack enough evidence. For example, Lines 110-118 mention the functional fixation challenge, which says that an unconstrained generated model cannot support conditional generation. However, VAE-based methods, such as HydrAMP, support both unconstrained generation and conditional generation. For unconstrained generation, they sample random noise as the input. For conditional generation for attributes, they can shift the latent space to meet the attribute. For conditional generation for a given input peptide, they can optimize the input by encoding it to an embedding with the VAE encoder and then shifting the latent representation. 
2. While the introduction describes three challenges, it is not clear to me how these challenges are solved by the proposed method. LMT is still limited by the small AMP databases because it uses the AMPs as the target set. Although the converter can learn different mappings from the primer set to the target set, the target distribution is the same. 
3. There is no guarantee that there is a learnable mapping for the converter, since the primer is randomly initialized and the mapping is stochastic. The mapping to learn may not reflect any meaningful underlying biological rules to build peptides. If that is the case, even if the mapping is learned, it cannot create meaningful peptides on a new prime set. 
4. Figures and Tables
	1. Notation in Table 1 should be consistent with the main context. For example, Lines 357-358 say that the two models are Lt and LMT_ran, while in Table 1 they are LMT_tmp and LMP_ran. 
5. Experimental results do not support the effectiveness of the proposed method in generating diverse and accurate peptides at the same time. 
	1. In Table 1, the two primer sets have different performance in terms of diversity and accuracy. While one has a high diversity, its accuracy is low. The other has low diversity and high accuracy. There is no single primer that can outperform the other baselines in both diversity and accuracy. The sensitivity also shows that it is nontrivial to choose the right primer set, limiting the effectiveness and generalization of the proposed method
	2. In Table 2, the performance of different mapping tables further shows the sensitivity of the mapping, and more importantly, all these results have a low accuracy compared with baselines in Table 1. Similar problems happen in Table 3 as well, but for diversity. 
6. A comprehensive discussion of related work is missing. Some related work is briefly mentioned in the introduction, but lacks a more detailed discussion. The related variants in the appendix are too general (diffusion model and language model). 
	1. PepVAE: Variational Autoencoder Framework for Antimicrobial Peptide Generation and Activity Prediction
	2. PepCVAE: Semi-Supervised Targeted Design of Antimicrobial Peptide Sequences

### Questions
Based on the experiments and analysis, is there an empirical way to choose the test primers and mapping that can get a good balance between the diversity and accuracy?

### Soundness
2

### Presentation
1

### Contribution
2
