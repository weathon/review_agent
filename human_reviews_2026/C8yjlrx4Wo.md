# SemanticDPP: Efficient Uncertainty Quantification in LLMs

- Avg Score: 3.33
- Decision: Reject
- Scores: 2, 2, 6

## Abstract
Accurately quantifying uncertainty is crucial for ensuring the reliability of model outputs and enabling informed downstream decision-making. However, the output space of large language models (LLMs) is so large that traditional methods break down in this setting; yet, LLMs have been shown to respond to prompts confidently with confabulations, fabrications, ``hallucinations’’, and other erroneous information. While methods designed specifically for LLMs exist, there is a gap and need for approaches which accurately quantify uncertainty efficiently. In this work, we introduce SemanticDPP, an efficient method for quantifying the uncertainty corresponding to the semantic variation in outputs of a large language model based on internal activations. Building upon prior work, we show that this uncertainty is highly indicative of whether or not a model will answer correctly in question answering (QA) tasks---which provides a useful signal to determine whether to abstain from responding to a given question to prevent incorrect answers.
SemanticDPP uses determinantal point processes (DPPs) to learn a model over semantic (dis)similarity in model embeddings, which we use in a fully unsupervised manner to identify semantically distinct sets of responses.  We additionally present an extension, SemanticDPP-C, which yields a soft clustering of the sets of semantically distinct responses. Our extensive empirical investigation examines the behavior of our methods on two frontier open-sourced models of different capacities (that grant access to model internals), Gemma2 9B and Gemma3 27B, on a broad range of widely used QA benchmarks. SemanticDPP enables fast uncertainty quantification while it matches or exceeds the selective prediction (hallucination detection) performance of state-of-the-art baselines.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a method for uncertainty estimation building on the Semantic Entropy work. The key contribution is to reduce inference-time costs by fitting a DDP (determinantal point process) to allow the entropy prediction based on just the model embeddings instead of the pairwise clustering in Semantic Entropy. The training process is an additional computational cost but could be (easily) amortized over inference calls.
They propose two different variants: SemanticDDP, which directly uses the learned DDP to estimate uncertainty from a set of model response embeddings and SemanticDDP-C, which uses the DDP to cluster samples responses into semantically coherent clusters.
The authors evaluate the method in six different benchmarks and with two different models: Gemma2 9B and Gemma3 27B.

### Strengths
- The paper identifies the bidirectional clustering step in Semantic Entropy as a speed bottleneck and alleviate it via replacing this step via their DDP method, while reporting even better results on the evaluated benchmarks. The usage of a DDP to this end is to the best of my knowledge novel.
- They discuss and raise an issue in benchmark datasets TriviaQA and NQ open, where answers at the benchmark creation time might have been true but has since changed.

### Weaknesses
See the questions. If the open questions and weaknesses raised there are appropriately addressed, I will raise my score.

nit: the caption formatting for many tables, where the caption has almost no vertical space to the table, is not very appealing.

nit: l.425 missing space between "SemanticDDPand"

### Questions
Q1. In Table 3, the scores in the columns do not add up to the average score -- e.g. for BioASQ `(0.82 + 0.56 + 0.6 + 0.63 + 0.53 + 0.86) / 6 = 0.667 != 0.658 (reported in paper)` or for TruthfulQA `(0.82 + 0.58 + 0.60 + 0.63 + 0.55 + 0.65) / 6 = 0.638 != 0.655 (reported in paper)`. How was the average score calculated?
  - this does change the interpretation of results, as the actual gap between fitting on BioASQ and TruthfulQA is now substantially larger, suggesting less transfer than was claimed in the paper

Q2. Why are the P(True) and Token Average Likelihood baselines not reported for Gemma3 37B in Table 3? Internal/implicit representations of uncertainty may scale in quality with model size and thus make the baselines more effective. Reporting these baselines is crucial to certain the method's effectiveness as model size scales.

Q3. What is the difference between Figure 2 and Table 1 & 2? It seems to be the exact same data minus Figure 2 not including the P(True) and Token Average Likelihoods baselines. Am I missing something? Also Table 1 and 2 are not referred to anywhere in the paper. 

Q4. The computational overhead reported in your experiments for Semantic Entropy is really prohibitive, whereas the original paper (https://arxiv.org/pdf/2302.09664) claims the overhead is small in practice. Where does this discrepancy come from?

Q5. re: Table 4 -- what is the cumulative wall clock time of the actual LLM calls for the test split? This information better contextualizes how relevant the speedup in runtime of your reported methods is.

Q6. Which hyper parameters are used for the baselines, e.g. the "Linear probe" (Kossen et al., 2024)? Since the hyper parameters of the proposed method are tuned, are the baseline hyper-parameters also tuned (and how)?

### Soundness
1

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
4

### Summary
The paper presents a new method for uncertainty quantification with LLMs. The idea is to train a function that maps the embedding of an input (e.g. in the last layer of the model) to a semantic space. The uncertainty of a new input is assessed by generating multiple outputs for a given input, mapping these to the semantic space with the learned function, and in the semantic space using a fitted kernel of a determinantal point process to quantify the variability/uncertainty of the model for the provided input.

### Strengths
I did not find any mistakes in the setup or experiments. The paper is relatively clearly written. The idea of training a model to predict uncertainties has a promise to improve efficiency. The method is tested with many datasets.

### Weaknesses
The novelty is somewhat limited. There are previous methods that train a model to predict the uncertainty from the model embedding, although the details differ. If there are differences in performance, it would be interesting to understand what exactly explains these. In general, I feel the conceptual difference to the closest methods would have warranted clearer discussion.

One claimed strength of the method is that it is faster, i.e., does not require the n^2 LLM queries to assess the semantic similarities between model outputs at test time. However, also the presented method requires n generated outputs. Generating the n outputs is in practice computationally much heavier than the pairwise comparisons (which are a simpler task that can be solved with a smaller model), and hence the main computational bottleneck of the previous methods remains (and this is not discussed clearly).

One weakness of the method is that the model that predicts the uncertainties from model embeddings has to be trained, and for the best performance this would ideally be done for each new dataset from different domains. This is not needed by the comparison methods (at least not all of them). I did not see many details of how much training is needed, and how the preformance of the method depends on the amount of training.

Limited evaluation: only two models are considered whereas recent works, e.g. Kossen et al., 2024; Nikitin et al., 2024, included 6 models (and basic/instruction-tuned variants). Also, recent relevant baselines are not included (e.g. Qiu and Miikkulainen 2024; Nikitin et al. 2024). Evaluation metrics are also more limited: earlier work used two metrics: AUROC for the classification accuracy of identifying incorrect answers and AUARC which measures accuracy of responding to questions after removing questions whose uncertainty exceeds given threshold (averaged over thresholds). It seems that here only the latter is reported?

### Questions
See weaknesses.

### Soundness
3

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
5

### Summary
This manuscript presents a new uncertainty quantification metric for LLMs. The main idea is to use determinantal point processes (DPPs) to learn a model that can quantify the semantic similarities of response embeddings. Its effectiveness is measured using several QA benchmarks under a selective prediction use case.

### Strengths
1.	Utilizing DPPs to quantify the uncertainty of LLMs is new and interesting, with solid mathematical foundations.
2.	The proposed mechanisms are complement to several existing approaches, having the potential to be combined with other methods to further improve the performance.
3.	The writing of the paper is transparent with regards to the limitations of the work.

### Weaknesses
1.	The returned uncertainty metric is prompt-wise, not response-wise. This may limit the use case, i.e., it cannot be used to decide which response is more trustworthy if we have multiple responses for the same prompt.
2.	The benefit in computation time is not clearly justified. The proposed approach still needs to sample multiple responses for each prompt. This cost may dominate other computation overheads.
3.	The empirical comparisons are only with prompt-wise uncertainty metrics in abstention setup.

### Questions
1.	It is stated that semantic entropy “is hampered by the prohibitive cost of requiring a second LM to compare pairs of sampled responses.” However, the NLI model used in semantic entropy is only 1.5B, which is significantly smaller than most other SOTA LLMs. The cost in running this NLI model should be negligible compared with the sampling cost of the original LLMs. Can you make this point clearer?
2.	When stating “where Y_i denotes the subsets corresponding to each prompt i”, do you mean each “Y_i” include multiple subsets of responses, and each subset contains responses with equivalent semantic meanings while different subsets represent distinct semantic meanings? Please make this description clearer.
3.	Regarding SemanticDPP-C, what is the use case for this clustering algorithm? Is it better to have more fine-grained or even continuous measurement of semantic similarities, like in Semantic Density [1]?
4.	Can you elaborate more on how the proposed approach is complement to Semantic Density [1] and how they can be combined together?
5.	Regarding the dataset issues, have you tried cleaning the dataset by removing the questions with out-dated answers? What are the resulting performances?
6. The results reported in Section 5.4 (Table 4) are confusing. What operations are included in this runtime for SemanticDPP? SemanticDPP still needs to sample multiple responses, right? Why this sampling time is not included in this total runtime?

[1] Xin Qiu, Risto Miikkulainen. Semantic density: Uncertainty quantification for large language models through confidence measurement in semantic space, Advances in Neural Information Processing Systems (NeurIPS), 2024

### Soundness
3

### Presentation
2

### Contribution
2
