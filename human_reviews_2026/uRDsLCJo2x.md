# Learning Compact Representations of LLM Abilities via Item Response Theory

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 4, 4, 6

## Abstract
Recent years have witnessed a surge in the number of large language models (LLMs), yet efficiently managing and utilizing these vast  resources remains a significant challenge. In this work, we explore how to learn compact representations of LLM abilities that can facilitate downstream tasks, such as model routing and performance prediction on new benchmarks. We frame this problem as estimating the probability that a given model will correctly answer a specific query. Inspired by the item response theory (IRT) in psychometrics, we model this probability as a function of three key factors:  (i) the model’s multi-skill ability vector $\theta$, (ii) the query’s discrimination vector $\alpha$ that separates models of differing skills, and (iii) the query’s difficulty scalar $\beta$. To learn these parameters jointly, we introduce a Mixture-of-Experts (MoE) network that couples model- and query-level embeddings. Extensive experiments demonstrate that our approach leads to state-of-the-art performance in both model routing and benchmark accuracy prediction. Moreover, analysis validates that the learned parameters encode meaningful, interpretable information about model capabilities and query characteristics.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces a neural network *IrtNet* that is used to estimate an LLM's ability and characteristics of an LLM query. The model is then fit on a portion of embedLLM data for model routing, benchmark prediction, ablation and interpretability experiments.

### Strengths
* The authors perform an ablation study demonstrating that their MOE architecture provides improvements over a vanilla NN.
* The OOD prediction experiment demonstrates that IrtNet offers some robustness compared to EmbedLLM (though this should also be compared to other scaling laws see weaknesses)
* The difficulty characteristic and LLM embeddings seem meaningful

### Weaknesses
* Insufficient routing evaluation: In the routing literature, a router is generally evaluated on how it performs at selecting models with a performance-cost trade off in mind. The authors opt to test routing capability with performance prediction only. This leaves out a crucial piece of routing, and makes this task hard to distinguish from bench mark prediction. The authors should test their router on a dataset with cost and accuracy (see CARROT or Routerbench if embedLLM lacks this) and provide Area-Under the (Routing) Curve results.
* Lack of Scaling law baselines in benchmark prediction: Prior observational scaling laws ([3], [4]) can be used for benchmark prediction from training data. The authors cite these works but compare their scaling law to none of these, which is a major weakness in the experimental design. Furthermore, ([3], [4]) are substantially more interpretable, so performance gains really need to be significant.
* Lack of clarity in evidence for interpretability: The authors present a t-SNE projection of the query embeddings produced by IrtNet, showing that clustering by benchmark is present. Prior to passing through IrtNet, queries are fed through a pre-trained embedder. I suspect these raw embeddings would exhibit the same clustering behavior, so its unclear to me that IrtNet adds much.

### Questions
* What do the t-SNE projections of the raw embeddings look like?
* How does benchmark prediction performance compare to Sloth and other observational scaling laws?
* How does the router perform in temrs of Area Under the (routing) Curve?
* Is possible to verify the meaningfulness of the distinguishability characteristic $\alpha_q$?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a novel framework for learning compact representations of LLM abilities using Item Response Theory (IRT). The authors introduce IrtNet, a (MoE) network designed to jointly learn query characteristics such as difficulty and discrimination. The proposed method demonstrates strong performance on downstream tasks, including model routing and benchmark accuracy prediction.

### Strengths
The grounding of the capability modeling process in Item Response Theory provides a novel perspective by incorporating the difficulty and discrimination of queries.

The framework's effectiveness is demonstrated by its strong performance on downstream tasks like model routing and benchmark accuracy prediction.

### Weaknesses
The motivation for employing both Item Response Theory and the MoE network architecture is not fully elaborated. The paper would benefit from further discussion and ablation studies comparing the proposed item response function with simpler alternatives, such as plain logistic regression.

The applicability of the framework may be limited, as the evaluation is restricted to models from the EmbedLLM dataset. The paper does not explore whether this method can be generalized to more recent and capable models. Furthermore, the specific models used in the model routing task should be explicitly stated.

The primary novelty compared to EmbedLLM's encoder-decoder architecture is the introduction of the IRT formalism and the IrtNet architecture. However, the paper does not sufficiently justify that the source of the advantages of this new formulation over the previous approach lies in the IRT formalism or the MoE architecture.

### Questions
The formulation used in the paper appears to be a bit different from the standard 2PL model, where the discrimination parameter typically interacts multiplicatively with both the ability and difficulty parameters. Could the authors clarify the rationale behind their proposed formulation?

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
3

### Summary
This paper proposes IRTNet, a novel framework for learning compact and interpretable representations of large language model (LLM) abilities, inspired by Item Response Theory (IRT). This new framework can estimate the probability that a specific model will answer a specific query correctly. With a MoE network for parameter coupling, this model learns model's ability embedding, query discrimination vector and difficulty scalar jointly. The experiments shows the outperformance of the new method in downstream tasks, such as model routing and benchmark prediction.

### Strengths
1. The methodology is described clearly and easy for the readers to follow.
2. The application of IRT model in LLM embedding learning is a novel idea.
3. The comprehensive experiments cover most downstream tasks with LLM representations, and the outperformance support the efficacy of IrtNet.

### Weaknesses
1. Compared to EmbedLLM, which also tries to construct compact representations of LLMs' abilities, the new method proposed in this paper applies IRT to modeling and introduces ab MoE-based method. Although the experiments show IrtNet's outperformance, the motivation for designing is not fully illustrated in the paper. I think a paragraph to analyze why IRT and query embeddings can achieve the gain will be very insightful.
2. For validating difficulty in Section 5.2, the results is still on the level of benchmark. I think the difficulty parameter is actually on the level of query. It would be very interesting to show some query-level results. Actually there is a previous benchmark named Easy2Hard-Bench [1]. They also applied IRT method on the model's answer correctness on different queries, but their goal is to estimate the difficulty of the queries. Their published datasets include GSM8K, which is also in the datasets here. You could compare the beta parameters in IrtNet and the difficulty scores from Easy2Hard-Bench.
3. You could also try to do some human evaluations to compare the difficulty estimated by IrtNet and human's judgement, which is also applied in Easy2Hard-Bench [1]. Although it may not be relevant to the main storyline in the paper, this comparison would provide another perspective to understand LLM's abilities via human perspective.

I would like to adjust my score if the authors can address these questions.

[1] Ding, Mucong, et al. "Easy2Hard-Bench: Standardized Difficulty Labels for Profiling LLM Performance and Generalization." Advances in Neural Information Processing Systems 37 (2024): 44323-44365.

### Questions
1. The results of benchmarking prediction shown in Figure 3 indicates that both IrtNet and EmbedLLM tend to overestimate the LLMs' performance on benchmarking. Could you please accordingly provide some insights?
2. There are four variants of IRT model (1PL-4PL). What the authors used in IrtNet is more like to 2PL. Could you please explain how you decide to use this?
3. Does IrtNet have some potential limitations? It would be good to discuss in the paper.

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
3

### Summary
The paper proposes IrtNet, a framework that learns compact, interpretable representations of LLM abilities for two downstream tasks: model routing and benchmark prediction. It casts the probability that a model answers a query correctly as a 2-parameter IRT response. Query characteristics are produced from a dense MoE over query embeddings; model embedding is learned jointly, end-to-end, by minimizing BCE on model–query correctness labels. Experiments on 10 benchmarks (35k queries, 112 models) report state-of-the-art routing accuracy and data-efficient benchmark prediction; the authors also present interpretability analyses (difficulty correlates with empirical hardness; embeddings cluster by family/specialization).

### Strengths
1. Uses a dense MoE to map query embeddings to IRT parameters, arguing better stability vs sparse MoE and improved discrimination estimates—novel architectural choice for this setting.
2. Strong empirical results: IrtNet achieves the best micro/macro routing accuracy across 10 datasets;
3. Addresses a practical need: scalable, cost-aware LLM orchestration via interpretable ability representations; potentially impactful for evaluation, routing, and system design at scale.

### Weaknesses
1. While routing baselines are strong, the paper doesn’t compare with cascaded inference or test-time adaptation strategies that may implicitly capture difficulty (e.g., multi-try reasoning depth) and could be competitive on cost/quality trade-offs. Please position IrtNet against those classes more explicitly and, if feasible, add at least one cascade baseline.
2. OOD experiment is not very promising. They are also not added to the main manuscript but are present in the appendix.
3. Although the paper targets routing at scale, it lacks end-to-end latency/throughput analyses.
4. The method relies on a fixed text embedding model (all-mpnet-base-v2) to seed the MoE. There’s no sensitivity analysis to the embedding choice/domain shift.

### Questions
Check weakness above

1. MoE design choices
Why dense MoE over sparse MoE—besides stability? Did you experiment with different expert counts and routing strategies? An expert-count sweep and a sparse-vs-dense comparison (equal parameters) would contextualize the ablation.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents IrtNet, a neural adaptation of item response theory that embeds large language models and benchmark queries into shared compact representations for predicting correctness, routing tasks, and benchmarking. By combining model embeddings with query-specific discrimination vectors and difficulty scalars produced by a dense mixture-of-experts layer, the method models the probability of an LLM answering a query correctly and demonstrates state-of-the-art results on the EmbedLLM dataset for model routing accuracy, benchmark score prediction, and data-efficient correctness estimation.

### Strengths
The formulation tightly couples psychometric intuition with neural modeling, yielding a simple response function while learning expressive query parameters via the MoE. Experiments cover key downstream tasks, routing, in-distribution prediction at varying data scales, and leave-one-benchmark-out generalization, and show consistent gains over strong baselines such as EmbedLLM and Avengers-Pro. The analysis section goes beyond raw metrics, validating that learned difficulty correlates with empirical benchmark hardness, visualizing discrimination vectors that cluster by dataset, and probing LLM embeddings for family and specialization structure. These pieces collectively support both effectiveness and interpretability claims.

### Weaknesses
The evaluation is restricted to the EmbedLLM dataset with majority-voted correctness labels, so transfer to free-form generation, human preference judgments, or settings with label noise remains untested. Training details such as computation cost, convergence behavior, or sensitivity to the representation dimension and number of experts are sparse, making reproducibility and deployment considerations less clear. Although the architecture is inspired by IRT, comparisons to more recent multi-dimensional or causal IRT baselines are missing, and ablations only test removing the MoE rather than alternative query encoders or different priors on difficulty. The routing evaluation assumes access to oracle correctness labels for supervision, but practical systems may suffer from biased or delayed feedback—this is not discussed.

### Questions
How robust is IrtNet to inconsistent or adversarially noisy correctness annotations when learning the model and query embeddings? In routing scenarios with partial feedback, could the authors clarify how the system would update θ and α without exhaustive supervision? What are the computational requirements for training and inference relative to baselines, and do they scale gracefully as the number of models or benchmarks grows? Finally, have the authors explored adapting the framework to non-binary outcomes, such as partial credit or calibrated probability scores, to better align with emerging evaluation datasets?

### Soundness
3

### Presentation
3

### Contribution
3
