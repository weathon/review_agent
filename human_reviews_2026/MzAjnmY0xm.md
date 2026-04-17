# A Geometric Lens on LLM Abilities through Joint Embedding Item Response Theory

- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Standard LLM evaluation practices compress diverse abilities into single scores, obscuring their inherently multidimensional nature. We present **JE-IRT**, a geometric item-response framework that embeds both LLMs and questions in a shared space. For question embeddings, the *direction* encodes semantics and the *norm* encodes difficulty, while correctness on each question is determined by the geometric interaction between the model and question embeddings. This geometry replaces a global ranking of LLMs with topical specialization and enables smooth variation across related questions. Building on this framework, our experimental results reveal that out-of-distribution behavior can be explained through directional alignment, and that larger norms consistently indicate harder questions. Moreover, JE-IRT naturally supports generalization: once the space is learned, new LLMs are added by fitting a single embedding. The learned space further reveals an LLM-internal taxonomy that only partially aligns with human-defined subject categories. JE-IRT thus establishes a unified and interpretable geometric lens that connects LLM abilities with the structure of questions, offering a distinctive perspective on model evaluation and generalization.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes JE-IRT, a joint embedding variant of item response theory that maps both large language models and benchmark questions into a shared geometric space where direction encodes topical alignment and norm captures difficulty. Trained on the EmbedLLM correctness dataset across 112 models and 10 benchmarks, the method predicts item-level correctness via the projection of model embeddings onto question directions, achieving better accuracy than scalar-ability baselines while remaining parameter-efficient. Extensive analyses show that the learned geometry reflects interpretable structure, supports adding new LLMs via lightweight fine-tuning, and reveals that model-induced subject clusters only partially match human-defined categories.

### Strengths
The paper grounds its framework in clear theoretical motivations, demonstrating via empirical evidence that traditional 2PL IRT assumptions fail for contemporary LLM evaluations. JE-IRT produces competitive accuracy with lower-dimensional embeddings, and its inductive bias yields consistent gains across models and benchmarks, suggesting broad applicability. The authors validate interpretability claims by correlating embedding norms with question difficulty, assessing directional alignment for OOD behavior, and analyzing clustering structure, providing richer insight than aggregate leaderboard scores. Scalability experiments that onboard both held-out and newer LLMs using only a single embedding underscore the practicality of the approach.

### Weaknesses
The evaluation hinges entirely on the EmbedLLM dataset and a small set of base encoders, leaving open whether the learned geometry transfers to other evaluation suites or task formats beyond multiple-choice correctness. While the framework is positioned as interpretable, the paper provides limited qualitative case studies illustrating how practitioners would act on the embeddings, and the taxonomy analysis stops short of human validation. Computational costs for training and updating the embeddings are not reported, making it hard to judge feasibility for organizations with many models or rapidly evolving benchmarks. The method also assumes clean correctness labels and does not explore robustness to annotation noise or adversarial inputs.

### Questions
How sensitive is JE-IRT to the choice of frozen base encoder, and could domain-specific encoders or instruction-tuned representations further improve semantic alignment without retraining question embeddings? When adding new models, does the single-embedding fine-tuning ever require regularization to prevent overfitting to scarce observations, especially for highly specialized systems? Can the authors share qualitative examples where the learned geometry surfaces actionable routing or evaluation insights that human-designed subject tags would miss? Finally, how would the framework accommodate non-binary evaluation signals such as partial credit, calibrated probabilities, or human preference judgments?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a geometric item-response framework, JE-IRT, to evaluate the multi-dimensional ability of LLMs by incorporating semantics and difficulties of questions.

### Strengths
* Presents a geometric IRT framework, JE-IRT, embedding both models and questions into a shared space, with their dot product determining the probability of a correct response. Provides interpretable question vector attributes with direction capturing semantics and norm capturing difficulty.
* Clear motivation showing a total order of LLM performance using scalar abilities does not hold, and the need to move to multidimensional interactions.
* Scalable to evaluating new LLMs through only fine-tuning their embedding instead of full retraining.
* Interpretable geometric validation showing directional alignment predicts out-of-distribution drops and norms relate to difficulty. Clustering the embeddings reveals subject divisions that are different from human-defined categories.
* JE-IRT outperforms baselines in the prediction accuracy of LLMs on benchmarks.
* The methodology is clearly presented and well written with descriptive figures.

### Weaknesses
* Compare novelty and prediction performance with MIRT baseline: JE-IRT is similar to Multidimensional-IRT, an existing IRT framework (Multidimensional item response theory models, Mark D Reckase, 2009). MIRT also includes a dot product between student (model) and item (question) vectors. While the authors provide a geometric interpretation to this formulation, the novelty compared to MIRT should be clearly explained. Further, recent applications of IRT in NLP already use embeddings of question content and model representations (see the IRT-Router paper, which the authors cite).
* Compare against other IRT models on predictive performance. Since the authors extend IRT to JE-IRT, the performance of existing IRT models like 1-PL, 2-PL, and MIRT should be compared against.
* Design choices of JE-IRT should be clarified with intuition and theoretical justification. For ex, why divide by the norm of the question embedding to get model abilities (eq 2), why use the norm of the question embedding as difficulty, etc.
* The authors motivate why a scalar-based total ordering of model abilities doesn’t hold. Similarly, using the norm of the question embedding to get scalar-abilities would lead to a total ordering of question difficulties. I assume there exist harder questions that can be solved by weaker models and vice versa. Can the authors verify this? If yes, shouldn’t a similar multidimensional interaction between model vectors and question vectors be used to obtain question difficulties? Is this already done in MIRT to get difficulty (I’m not sure)?

### Questions
* The authors cluster the question embeddings and show that it is not well aligned with human-defined subjects. The authors state that items that cluster together require similar abilities from the LLM. I wasn’t able to find the justification for this claim.
* The paper’s main goal is LLM evaluation. The authors cite Fluid Language Model Benchmarking, a recent work using IRT-based LLM evaluation, which focuses on efficiency and includes evaluation metrics such as efficiency (small number of questions for LLM evaluation), validity (generalization to other benchmarks), variance, etc. While the authors show that new LLMs can be added efficiently, is JE-IRT sample efficient? Can the Fisher information idea from Fluid Language Model Benchmarking be applied to JE-IRT? Apart from LLM response correctness prediction, can results on recovering the LLM ordering at a per-benchmark level be shown?
* Generalization to some new datasets (piqa, mmlu, gsm8k) shows significant drops in performance. How can this be mitigated? What fraction of questions from these datasets would reduce the drop? For example, adding 10% of Q from gsm8k leads to similar performance as in-distribution.
* Since the L2 norm is not differentiable everywhere, the authors should clarify how backprop is performed during training (sub-gradients, etc).

### Soundness
2

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
3

### Summary
This paper proposes JE-IRT (Joint Embedding Item Response Theory), a novel framework for evaluating large language models (LLMs) through a geometric generalization of item response theory (IRT). Instead of assigning each model and question a scalar ability and difficulty as in traditional IRT, JE-IRT embeds both in a shared vector space. In this space:

1. The direction of a question embedding captures its semantics (topic/specialization).
2. The norm encodes its difficulty.
3. The dot product projection between model and question embeddings determines the probability of a correct answer.

JE-IRT explains key empirical patterns observed in LLM evaluation: (i) no universal ordering of LLMs across tasks, and (ii) smooth variation of performance across related questions. The method is trained on the EmbedLLM correctness dataset (112 models, 10 benchmarks), and evaluated for scalability, interpretability, and generalization. The learned geometry reveals that embedding norms correlate with question difficulty, while directional alignment predicts out-of-distribution (OOD) generalization. Furthermore, clustering in the learned space shows that LLMs’ internal organization of topics partially diverges from human-defined subject boundaries.

### Strengths
1. The paper introduces a geometric formulation of IRT, where model–question interactions are represented through inner products in a shared space rather than scalar parameters. This is conceptually elegant and unifies item difficulty, semantic similarity, and model ability under a single geometric lens.
2. The theoretical formulation is well-motivated and supported by two formal propositions.

### Weaknesses
1. JE-IRT’s applicability to non-multiple-choice or free-form generation tasks remains unclear; the binary correctness assumption is acknowledged as a limitation but not empirically addressed.

### Questions
1. Have you explored whether specific semantic axes (e.g., reasoning, factual recall, arithmetic) emerge in the learned space? Visual examples could greatly enhance the interpretability claims.
2. Could JE-IRT be extended to handle graded or probabilistic outputs (e.g., continuous scores or multiple-choice logits)? Would replacing binary targets with model confidence distributions improve fit and interpretability?
3. Could the JE-IRT space support continual updates as new benchmarks or question types are added, without retraining the adapter network?
4. The cosine alignment between benchmark means correlates with OOD accuracy drops. Could this be formalized as a predictive diagnostic tool for estimating when a model will fail on a new dataset?

### Soundness
3

### Presentation
3

### Contribution
3
