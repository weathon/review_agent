# Ensuring Physicochemical Fidelity of Generated Polymers with PoGE

- Decision: Reject
- Scores: 4, 2, 2, 2

## Abstract
Recent advances in machine learning have accelerated progress in chemistry, enabling new capabilities in molecular design, property prediction, and materials discovery. A critical challenge in materials science is designing polymers with targeted macroscopic properties. However, prior generative models often fail to produce chemically valid polymer structures, hindering progress toward this goal. We introduce PoGE (Polymer Generation and Evaluation), a framework comprising two complementary components: a physics-informed evaluation suite for polymer generative models, and an unconditional transformer-based generative model adapted to polymer representations. Building upon and extending established molecule-centric benchmarks, our evaluation quantifies the alignment between the generated and experimental property distributions using the Wasserstein distance. The generative model is trained on a hybrid corpus of synthetic and experimental polymer representations and enforces polymer-specific validity constraints (“p-validity”) beyond the standard small-molecule validity. PoGE achieves high p-validity and significantly improved agreement with experimental property distributions compared to prior methods, even without explicit property conditioning during generation. By releasing a comprehensive benchmark, a high-quality pre-training corpus, and the trained model, PoGE establishes a foundation for conditional polymer generation tasks (e.g., on-demand reverse design), enabling targeted property optimization and accelerating reproducible, domain-aware polymer discovery.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents PoGE, a framework that integrates an unconditional transformer-based generative model with a physics-informed evaluation suite.

For the unconditional generative model, this paper adopts a scaled-down variant of GPT-2, pretrained on a large volume of typically lower-quality p-SMILES and subsequently fine-tuned on a smaller set of high-quality p-SMILES examples.

For the evaluation suite, this paper proposes a set of metrics specifically designed for polymers, thereby overcoming the limitations of directly applying evaluation criteria developed for drug-like molecules.

Through establishing such a foundational platform, this paper paves the way for more effective AI-driven materials design in the polymer domain.

### Strengths
1. This paper proposes a physics-driven evaluation framework specifically designed for polymers, thereby overcoming the limitations of directly applying evaluation criteria developed for drug-like molecules.

2. This paper proposes an unconditional generative model, which achieves high p-validity and significantly improved agreement with experimental data compared to prior methods.

### Weaknesses
1. Fundamentally, the unconditional generative model proposed in this paper is little more than a straightforward application of GPT-2 through pretraining and fine‑tuning on p-SMILES, offering limited methodological novelty.

2. Moreover, since both the PI1M dataset and PolyTAO corpora have already been used as part of the proposed model’s pretraining corpus, and the proposed model is further fine‑tuned on the higher‑quality PolyInfo dataset, comparisons against PI1M and PolyTAO are inherently unfair.

3. For the evaluation metrics, some aspects are inaccurate or not entirely appropriate:
    * The p‑valid metric requires that each bond at the endpoints of the polymer repeat unit be of the same type. However, this is chemically unreasonable since repeat units can be asymmetric, branched, or linked in non‑head‑to‑tail patterns. Therefore, this rule introduces bias and may misjudge structurally valid polymers as invalid.
    * The novelty metric considers only PolyInfo data as the reference dataset. However,  this is not appropriate since the PI1M dataset and PolyTAO corpora are also used as training data. This rule lead to an overestimation of novelty, as the model’s generated samples might overlap with training data not accounted for in the evaluation.

4. More datasets and methods should be discussed and compared in this work, such as the PolyOne dataset introduced in [1] and the SMiPoly generator proposed in [2].

[1] Kuenneth, Christopher, and Rampi Ramprasad. "polyBERT: a chemical language model to enable fully machine-driven ultrafast polymer informatics." Nature Communications 14.1 (2023): 4099.

[2] Ohno, Mitsuru, et al. "SMiPoly: generation of a synthesizable polymer virtual library using rule-based polymerization reactions." Journal of Chemical Information and Modeling 63.17 (2023): 5539-5548.

### Questions
1. Except for the novelty metric, I would like to know what reference datasets are used for the other evaluation metrics, such as SNN and various descriptor‑based metrics. Are these metrics also computed using only PolyInfo data as the reference?

2. In Table 1, why don't you report the p-valid, Unique, and Novel metrics on the PI1M dataset?

3. In Tables 1 and 2, does the “PolyTAO” dataset correspond to the same PolyTAO corpus used for pretraining? 

4. In addition, this paper employs the PI1M dataset, the PolyTAO corpus, and the PolyInfo dataset as the training corpus, while promising that they will be released. However, as the PolyInfo dataset does not generally permit the acquisition of large portions of its data, I would like to know whether the authors have obtained official authorization for such usage.

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
5

### Summary
The paper presents PoGE, a polymer generation framework based on a small GPT-2 model trained on polymer SMILES (p-SMILES) representations.It introduces a p-validity definition specific to polymers and uses Wasserstein distance between distributions of molecular descriptors (molar mass, TPSA, rotatable bond fraction, aromatic fraction, etc.) for generated versus experimental polymers.
The goal is to ensure that generated polymers are not only syntactically valid but also physically plausible.

### Strengths
1. The work focuses on polymers, an important but underexplored domain.

2. The work highlights the lack of polymer-specific validity metrics in current molecular generation benchmarks.

3. The work is well written and easy to follow.

### Weaknesses
## Limited Novelty: 

1. The only methodological novelty of evaluating distributions via Wasserstein distance is very straightforward statistical measure and not new.

## Overstated claims

2. The paper uses terms like “physics-driven” and “physically grounded” evaluation, yet all evidence comes from simple 1D descriptor distributions. The evaluation does not extend to polymer ensembles with experimentally verified properties or synthesis constraints.

3. The p-valid is interesting but not comprehensive. But a valid polymer may have more than 2 (e.g., 4) indicators (*) for the polymerization positions.

## Generation

4. PoGE performs unconditional sampling only, without showing how its learned distributions could guide goal-oriented polymer design.

### Questions
1. Why does the work focus on unconditional generation instead of conditional generation, which is more useful for polymers?

2. Have the authors tested their generated polymers in simulations or lab experiments? What makes the dataset different from previous ones in terms of usefulness and properties?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The work introduces PoGE (Polymer Generation and Evaluation, a unified framework designed to improve the physicochemical fidelity of machine learning-generated polymer structures. PoGE presents two innovations: 1) a GPT-2-based generative model for unconditional polymer design, and 2) a physics-informed evaluation suite for evaluating polymers according to domain-specific property distribution metrics. The authors additionally present an unconditional dataset of polymers curated via PolyTAO, taking advantage of conditional generation with broad property coverage to curate a dataset of 1 million generated polymers. Empirical results show that PoGE achieves superior chemical validity, uniqueness, and alignment with experimental descriptor distributions relative to prior frameworks. The work establishes the first reproducible, domain-aware platform for conditional and inverse design tasks in polymer informatics.

### Strengths
1. PoGE compares favorably to baseline models on most metrics, including validity and lower Wasserstein distances to the true data with respect to various physicochemical properties. 
2. PoGE introduces a new, expanded dataset and benchmark by combining the PI1M dataset, the PolyInfo dataset, approximately 1 million structures generated by PolyTAO. The work is able to harness PolyTAO for unconditional generation by marginalizing the conditional model over the empirical PolyInfo distributions of 15 property types.
3. The authors supplement the work with property density comparison across different components of the complete dataset.

### Weaknesses
1. Although it is mentioned as a potential application of the work, PoGE does not seem to showcase experiments in property-conditioned polymer design. 
2. The machine-learning novelty of the work is rather limited. The authors appear to directly use GPT-2 with little to no architectural modifications.
3. The selection of hyperparameters to optimize with Optuna (top_p, top_k, and temperature) is never explicitly motivated in the paper. 
4. In section 3.4, the authors argue that chosen properties avoid redundancy and multicollinearity. In addition, as described in section A.2.1, descriptor values for conditional input to PolyTAO are sampled independently. This is a strong assumption and insufficiently justified, with no correlation analyses performed between individual properties. 

Minor:
1. In table 1, it may be best to indicate for each metric whether lower is better or higher is better.
2. The paper omits some details on experimental results; for instance, the number of generated samples per run.

### Questions
As a sanity check, was the PolyInfo dataset ever screened for p-validity? That is, is the entire PolyInfo dataset p-valid?

### Soundness
3

### Presentation
2

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
The authors trained GPT-2 to generate polymers. They then evaluate the distribution of the generated polymers, and show that there is a good match with existing datasets.

### Strengths
The paper is relatively clear. The resulting dataset could be useful.

### Weaknesses
I think the primary weakness of this work is that its main contributions seem to be incremental, and somewhat speculative. Inverse design is clearly important, and they authors claim that this will facilitate incremental design, it seems reasonable to think that it might, but there should be stronger evidence or at least plausible or cited ways to get to inverse design from an unconditional generator or dataset.

The paper takes a number of datasets, themselves generated, such as PI1M (RNN) and also one real-world dataset (PolyInfo). It then trains/fine-tunes on those to create what is claimed to be a better generator. The results in Table 1 are suggestive, but the ones that I believe are most important, Uniqueness, Novelty, and IntDiv, show mixed results. Validity and p-valid are useful to know as a performance standpoint, but modest differences do not seem crucial to me. The reason is: Suppose there is Method A that can generate 100 million polymers in 50 minutes, and Method B that can generate 50 million an hour. Raw outputs of Method A are only 75% p-valid, but raw outputs of Method B are 100% p-valid. But suppose the outputs of Method A can be filtered for p-validity in 10 minutes. Method A can now effectively generate 75 million, 100% p-valid polymers an hour, while Method B can only generate 50 million p-valid polymers an hour. Of course, it may be the case that the polymers generated by Method B may be higher quality, but that is a different metric.

There is a claim that the authors used BPE because it reduces the number of tokens, and mention that RNN struggle over 5-7 tokens. PoGE, however, is a transformer architecture.

Comparing the unconditional distributions as a metric seems weak. Why is this a good metric to use, if our ultimate goal is inverse design? Distributions of the conditional generation would be more convincing.

### Questions
Why are there any invalid SMILES in PI1M?

BPE has no awareness of p-SMILES/SMILES syntax, so might find byte pairs that are not very semantically meaningful. Would a custom tokenizer that was aware of p-SMILES syntax do better?

Novelty for PoGE was only tested against PolyInfo data? What about the PI1M data? That was also in the training set for PoGE.

Was Uniqueness measured using canonicalized SMILES?

### Soundness
2

### Presentation
3

### Contribution
1
