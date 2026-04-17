# Evolutionary Profiles for Protein Fitness Prediction

- Decision: Reject
- Scores: 4, 8, 4, 4

## Abstract
Predicting the fitness impact of mutations is central to protein engineering but constrained by limited assays relative to the size of sequence space. Protein language models (pLMs) trained with masked language modeling (MLM) exhibit strong zero-shot fitness prediction; we provide a unifying view by interpreting natural evolution as implicit reward maximization and MLM as inverse reinforcement learning (IRL), in which extant sequences act as expert demonstrations and pLM log-odds serve as fitness estimates. Building on this perspective, we introduce EvoIF, a lightweight model that integrates two complementary sources of evolutionary signal: (i) within-family profiles from retrieved homologs and (ii) cross-family structural–evolutionary constraints distilled from inverse folding logits. EvoIF fuses sequence–structure representations with these profiles via a compact transition block, yielding calibrated probabilities for log-odds scoring. On ProteinGym (217 mutational assays; >2.5M mutants), EvoIF and its MSA-enabled variant achieve state-of-the-art or competitive performance while using only 0.15% of the training data and fewer parameters than recent large models. Ablations confirm that within-family and cross-family profiles are complementary, improving robustness across function types, MSA depths, taxa, and mutation depths.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces EvoIF, which takes within- and cross-family evolutionary profiles and structural information as inputs to model protein fitness. It reaches state-of-the-art with 76M params and 30k training set. The main contribution of the paper includes: 
1. Unified IRL interpretation of pLMs: The paper reinterprets MLM in pLMs as a form of inverse reinforcement learning, where log-likelihoods implicitly approximate evolutionary rewards.
2. Structure-sequence-evolutionary profiles. EvoIF combines a GVP encoder to process protein structure and and two complementary evolutionary profiles (within- and cross-family) through a lightweight transition block.

### Strengths
There are several strengths of this paper:
1. The paper provides a unified theoretical framing by interpreting MLM as IRL, where plms implicitly estimate evolutionary “rewards.” While I'm not expert in RL and IRL so I will look for other reviewers' opinions about this.
2. The authors construct within-family and cross-family profiles and capture distinct and complementary signals. The authors also use ablation study to prove the effectiveness of different profiles.

### Weaknesses
However, there are several issues with these paers:
1. As I said in strengths, I will look for other reviewers' opinions about regarding MLM and IRL.
2. The comparison of parameter and training dataset size isn't fair enough. Since EvoIF requires homologs retrieval, run plm and inverse folding model. The authors should also compare the run time instead of comparing the parameter and dataset only.
3. Limited novelty in model architecture. The structure embedding, profiles fusion module are incremental combinations of existing tools. I know there is not a lot of things could be done with the model architecture during the rebuttal stage, but the authors can think about this problem.

There are some over-claims in this paper:
1. The authors claim that the inverse folding captures the cross-family evolutionary profile. While none of previous inverse folding paper have explained this as "cross-family evolutionary" information. These information is more likely to be biophysical prior. Claiming it as cross-family evolutionary information may mislead the audience.
2. The claim "Existing models have not fully considered the comprehensive modeling of protein evolutionary information" is not true. At least LM-Design[1] already utilized evolutionary information from protein language model to assist protein structure to sequence design.

[1] Zheng, Z., Deng, Y., Xue, D., Zhou, Y., Ye, F., & Gu, Q. (2023, July). Structure-informed language models are protein designers. In International conference on machine learning (pp. 42317-42338). PMLR.

### Questions
1. A comparison of run time will be beneficial in evaluating different methods.
2. Reduce the over-claims or make stronger proof of the claim.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors propose EvoIF, a framework for protein fitness prediction that combines three pathways that are combined using a fusion module to calculate mutation preferences. The authors also theoretically parallelize log-odd ratios to inverse RL objectives to justify their use and correlation results as a zero-shot predictor for fitness. Results for a variant of EvoIF using MSA achieve state-of-the-art performance for fitness prediction in ProteinGym.

### Strengths
1. The parallel between protein evolution and RL is interesting.
2. The proposed fusion module and encoder are simple but well-thought-out.
3. The proposed model achieves state-of-the-art performance in ProteinGym.

### Weaknesses
1. The method seems dependent on structure availability and homolog retrieval.
2. The technical novelty of the proposed method is limited.
3. Code is not available.

### Questions
The paper is well-written, well-thought-out, and the results are convincing. My initial recommendation is acceptance. My detailed comments are as follows.

Comments:

1. (lines 117-118) The affirmation of “using only 0.15% of the training data and fewer model parameters” is strange, given that ESM-2 with 650M parameters is an important part of EvoIF architecture to calculate node embeddings. This is also related to the number of parameters column in Table 1 that might need changes.
2. From my understanding, the proposed method is used as a zero-shot predictor using Equation 3. Does using Equation 3 is enough and generalize to proteins with different functions? Does the fitness need to be normalized in a specific manner? It might be important to discuss this fact, as I guess that this normalization is performed as a preprocessing step in ProteinGym.
3. What is the difference between your encoder and the encoder architecture proposed in Zhang et al?
4. In line 172 the authors mention that “fitness landscapes are assumed time-invariant”, which is a valid assumption. How do the authors think Equation 6 would change, assuming dynamic environments or time-dependence for protein evolution?
5. How long does it take to perform the inference for one protein with EvoIF-MSA? The bottleneck in this case seems to be the homolog retrieval and MSA calculation.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
EvoIF examines zero-shot protein fitness prediction by interpreting the success of MLM loss trained models for fitness prediction through the lens of reinforcement learning, whereby evolution is a MDP, and homologs serve as expert in-context demonstrations. Evolution is then simply reward maximization, and MLM as a method to recover the latent reward (i.e. fitness). This provides a conceptual justification for using pLM log-odds as fitness estimators.

Citing the success of using MSAs for sequence modeling, the authors use a GVP-based encoder to integrate sequence-structure representations. Using Foldseek-retrieved structure homologs, intra-family information is provided. Cross-family profiles are derived from the logits of ProteinMPNN. This is fused with a transition block. Most of the model is frozen for computation efficiency and only the lightweight GNN encoder and fusion layers are trained on CATH. Evaluations are done using ProteinGym and achieve SOTA or near-SOTA using 0.15% of the training data, and fewer model parameters than most recent large models.

### Strengths
* Well-written paper with a nice set of motivating issues. This not only proposes a new method but also pinpoints some thoughtful insights on the current state of protein fitness prediction.
* Comprehensive baselines.
* Training efficiency is a great plus, given that only a light fusion module is needed, rather than having to actually scale up the PLM to get that last bit of performance boost.
* Thoughtful design to integrate evolutionary information.
* Choice to study challenging generalization regimes is a strong result
* Ablation studies in Table 2 are helpful.

### Weaknesses
* Using IF logits from ProteinMPNN means we’re bottlenecked by how good this model is. I suspect this won’t change much, but if there were additional results that also use ESM-IF or Caliby, it would be stronger evidence that using these logits are indeed due to the notion of capturing an evolutionary prior, rather than something specific to the ProteinMPNN model.
* The MaxEnt IRL formalization, though interesting, is a bit disconnected from the rest of the methods and results in my opinion. The leap from maximizing local conditional probabilities (MLM) to inferring the global reward function (IRL) is non-trivial.
* Using Foldseek should probably add a lot of latency during inference, even if training is accelerated. It might be useful to see an empirical comparison of inference throughput/latency compared to baselines, in addition to the performance, e.g. plotted on either side of the XY axes.
* It’s a bit misleading that EvoIF-MSA, the proposed new SOTA, is in fact an ensemble with another model. Highlighting that in the table, for example, as “EvoIF + GEMME (ensemble)” might be clearer than “EvoIF-MSA”. GEMME itself appears to be performing pretty well according to Table 1.
* Figure 2 would be less misleading if y-axes always started at 0. As it stands, I think it makes the gap between methods appear larger than it actually is.
* Nitpick: eq. 9 suggests that P represent logits, but the text refers to them as probabilities.

### Questions
* From the results of the ablations — it seems there isn’t a strong difference between the combinations, even when we remove both the inverse folding and structure profiles? Could authors provide some additional intuition on that?
* To clarify, when neither inverse folding nor structure is provided, this simply reduces back to the PLM case, right? Results there look fairly strong to me. What are the authors' intuitions on what the main advantages are of this work given the strong baseline?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes EvoIF, a compact sequence–structure model for protein fitness prediction that fuses two forms of evolutionary signal:
a within-family homology profile summarizing amino-acid preferences from close homologs, and
a cross-family inverse-folding profile distilled from a structure-conditioned generative model (ProteinMPNN-like).
A lightweight geometric encoder (GVP) integrates these profiles with structural context, and the fused distributions are trained with a masked-token objective on ~31 k CATH structures. The model achieves strong zero-shot performance on the ProteinGym benchmark, rivaling much larger sequence- or structure-based models while using orders-of-magnitude less compute. The authors interpret their approach through an inverse-reinforcement-learning (IRL) lens—treating natural sequences as “expert demonstrations” of evolutionary preference.

### Strengths
Clear motivation and framing:
The IRL interpretation is elegant and clarifies why likelihood-based models correlate with evolutionary fitness.

Model efficiency:
The architecture is concise (~76 M params) yet competitive with multi-billion-parameter baselines; the design is practical for labs without massive GPUs.

Complementary evolutionary signals:
The ablation results show that the within-family and cross-family profiles each add value.

### Weaknesses
Conceptual originality is modest:
The components—structure-conditioned language modeling, profile fusion, and retrieval—are all known; the novelty lies mainly in the specific fusion design. The IRL view is intellectually appealing but not concretely exploited in the loss or architecture.

Limited methodological detail:
Key implementation aspects (how profiles are normalized, exact fusion math, masking schedule) are insufficiently described for full reproducibility. It's not clear what the loss is that is being used to train the actual model that results are reported for. It's not clear what equation 9 is. Is this what the model predicts?Is  this how they come up with the true label used to supervise the model during training? Where is the loss used to train the model along with y and y_hat? Also there is not figure for the model architecture and how they incorporate IF data with ESM2 and combine this using a GVP-GNN. 

Evaluation scope:
Although ProteinGym is comprehensive, it is not high quality benchmark since the data is often fitness data collected in a multiplex fashion. The authors should also provide benchmarks on higher quality datasets like ∆∆G benchmarks where each measurement was done with purified protein variants. 

Over-claimed theoretical contribution:
The IRL framing of MLM is presented as a deep conceptual advance, However, MLM/BERT style training on protein sequences and structure has been used for 5-6 years in the literature.

### Questions
This method seems similar to the EvoRank SSL training objective, which fuses structure input with evolution self-supervision. In the EvoRank paper, they report ~50% performance improvements over baseline methods on ∆∆G tasks. I think their model was only 2M or 4M parameters. Can you compare against this method on ∆∆G datasets? 

Homology Retrieval: 
Using FoldSeek, rather than MMSeqs2, to search for homologs is great for finding remote homologs (seq sim <35%)  but these remote homologs are likely to not be within the same protein family but rather in the same structure family. Using FoldSeek gives more diverse profiles that take advantage of structurally similar proteins but not evolutionarily related proteins--a protein family. The authors provide no guarantee or analysis that the retrieved homologs are within the same protein family and have similar evolutionary constraints.  Can the author run ablations where they change the minimum sequence similarity (0%, 20%, 30%, 40%, 50% sequence similarity) threshold to ensure they are not using cross-family structural homologs when generating the profiles? 

Can you create a figure of the architecture and how you are mixing information from different sources and from different models?

### Soundness
3

### Presentation
3

### Contribution
3
