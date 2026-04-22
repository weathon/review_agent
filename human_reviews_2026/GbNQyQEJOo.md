# Combating Data Laundering in LLM Training

- Avg Score: 4.67
- Decision: Reject
- Scores: 6, 4, 4

## Abstract
Data rights owners can detect unauthorized data use in large language model (LLM) training by querying with proprietary samples. 
Often, superior performance (e.g., higher confidence or lower loss) on a sample relative to the untrained data implies it was part of the training corpus, as LLMs tend to perform better on data they have seen during training.
However, this detection becomes fragile under data laundering, a practice of transforming the stylistic form of proprietary data, while preserving critical information to obfuscate data provenance.
When an LLM is trained exclusively on such laundered variants, it no longer performs better on originals, erasing the signals that standard detections rely on.
We counter this by inferring the unknown laundering transformation from black-box access to the target LLM and, via an auxiliary LLM, synthesizing queries that mimic the laundered data, even if rights owners have only the originals.
As the search space of finding true laundering transformations is infinite, we abstract such a process into a high-level transformation goal (e.g., "lyrical rewriting") and concrete details (e.g., "with vivid imagery"), and introduce synthesis data reversion (SDR) that instantiates this abstraction. 
SDR first identifies the most probable goal that synthesis should step into to narrow the search; it then iteratively refines details, such that synthesized queries gradually elicit stronger detection signals from target LLM.
Evaluated on the MIMIR benchmark against diverse laundering practices and target LLM families (Pythia, Llama2, and Falcon), SDR consistently strengthens data misuse detection, providing a practical countermeasure to data laundering.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper addresses the challenge of detecting unauthorized data use in large language model (LLM) training when proprietary data has been "laundered" through semantic-preserving transformations (e.g., stylistic rewriting) to evade detection. The authors propose Synthesis Data Reversion (SDR), a two-stage method that infers the unknown laundering transformation from black-box access to the target LLM using an auxiliary LLM. SDR abstracts transformations into a high-level goal (e.g., "lyrical rewriting") from a taxonomy of 23 linguistic registers and refines it with iterative details to synthesize "training-like" queries. Evaluations on the MIMIR benchmark across LLM families (Pythia, Llama-2, Falcon) show SDR consistently improves detection metrics under various laundering scenarios.

### Strengths
1. It is timely and important to investigate the data laundering in LLM training
2. Strong detection performance compared with baseline methods.

### Weaknesses
1. The proposed method Involves multiple iterative queries to both target and auxiliary LLMs, potentially costly for large datasets or real-world audits, with no detailed cost analysis provided.
2. It would be better if the authors could report more results via TPR@1%, considering such an audit task requires much confidence.
3. There is another concern about the assumption, the authors assumes laundering follows a promptable goal-details structure executable by an auxiliary LLM, however, real-world laundering might be more opaque or non-prompt-based (e.g., human-edited).

### Questions
1. What is the specific computational cost of the proposed method?
2. Can the proposed method effectively address real-world laundering (e.g., human-edited)?

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
3

### Summary
The paper tackles data laundering in LLM training, where copyrighted texts are stylistically transformed to evade provenance checks. It introduces Synthesis Data Reversion (SDR), a black box method that first infers a high level laundering goal (e.g., lyrical rewrite) and then refines concrete stylistic details, using an auxiliary LLM to generate probes that match the laundered style. These probes restore detection gaps so the target model again reveals training exposure. On the MIMIR benchmark across diverse laundering practices and model families, SDR consistently boosts misuse detection.

### Strengths
1. Clearly shows that standard post hoc provenance tests collapse when models are trained on laundered surrogates rather than originals, making detection with losses or calibrated confidence ineffective. The setup and failure case are well motivated and demonstrated.
2. Reframe detection as reversing the unknown laundering transform, then reuse off-the-shelf detectors. The two-stage SDR pipeline uses a goal, then details abstraction over registers to search a compact prompt space with only black box access. Algorithms 1 and 2 are clear and practical.
3. In the experiment section, SDR consistently boosts several detectors across prompts, datasets, and model families and works with different auxiliary LLMs.

### Weaknesses
1. Most results rely on the laundering produced by GPT-style rewriting under predefined prompts. And the GPT may introduce a new bias. How about adding a third-party laundering pipeline, which could strengthen the whole paper.
2. SDR needs repeated calls to an auxiliary model to build templates and iterate prompts. The experiment section only includes a limited introduction of query budgets, latency, and sensitivity to n, m, l, and K. A more comprehensive sensitivity study would help strengthen the paper.
3. Results compare SDR plus standard detectors to the detectors alone. Since the contribution is a laundering-aware search, comparisons to other data-centric countermeasures or prompt search strategies would help to improve.
4. The goal identification stage assumes the laundering transformation belongs to one of the 23 predefined registers. Though this is noted in Appendix I, many realistic transformations, like pseudo-translation or hybrid creative styles, etc, may limit SDR’s recall and generalization. Moreover, the method is not tested on unseen or mixed registers, leaving its robustness to out-of-distribution transformations uncertain.

### Questions
Please see the weakness section. I will raise the score if the author addresses the questions clearly

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
This paper studies the task of data contamination (or membership inference) detection, particularly in scenarios where the training data have been laundered (e.g., undergone some register transfer) prior to model training. Previous work on data contamination or membership inference has primarily focused on detecting unauthorized data use on the exact same data, without considering potential laundering transformations.

As I understand it, the paper proposes to “reverse the laundering process”, i.e., first by identifying the most likely laundering register (i.e., the stylistic or structural form into which the original data may have been transformed), and then by recovering finer-grained data details through iterative prompting of an auxiliary LLM. Experimental results show that existing data contamination detection methods experience a substantial performance drop on laundered/synthetic samples, while the proposed reverse process helps recover several detection metrics.

### Strengths
- The paper investigates an important and interesting topic that has not been extensively explored in the existing literature.

- The proposed method demonstrates several intriguing empirical results.

### Weaknesses
- The threat model (and the corresponding protocol) is somewhat unclear. Several entities (such as the target model $M_t$, the auxiliary model $M_a$, and the datasets $D_{pro}$ and $D_{held}$) seem to have implicit assumptions, but these are not clearly stated. For example, is $D_{held}$ guaranteed to contain only non-member samples (i.e., data that were never used in training $M_t$)? Is the proprietary data $D_{pro}$ fully or partially assumed to have been used during training? These are important clarifications, especially for a work claiming to detect “unauthorized data usage”, which typically requires strict, verifiable integrity assumptions to be meaningful in practice. In addition, the data setup for evaluation remains somewhat vague (See detailed questions below). 

- The method description is somewhat confusing. Stronger intuition and clearer formulation of the design ideas would help. For example, in Algorithm 2, the pseudo-code suggests that a single “system prompt” is always returned, whereas the text description implies that it may instead correspond to sample-level prompts. It might be clearer to use explicit sample indices or a more formalized notation to distinguish between them.

### Questions
- How are member versus non-member samples defined when computing AUC, ACC, or TPR in Tables 2–5? Are they randomly split from the same MIMIR dataset? If so, how do you ensure that the non-member data were indeed not used during the pretraining of the target model (given that many large models are trained on massive web-scale corpora that may overlap with MIMIR?) Some explicit discussion of data provenance control and efforts made to ensure disjoint membership would be very helpful. Also, what exactly does the detector observe, the laundered data samples, original samples, or both?

- Related to the above concern about the unclear threat model, it remains unclear what exactly is meant by “unauthorized training data detection on $M_t$” in the construction of $Perf_r$ (see Algorithm 1, line 20, and Algorithm 2, line 11). Why is it considered a reasonable setting to compute this directly on the target model that is itself under test? Would this not constitute a form of data leakage or evaluation exposure?  In this sense, a clear and explicit description of the different data subsets (e.g., the member data, non-member data, held-out data, and potentially shadow data) and their respective roles in training and evaluation is essential, but currently missing. This clarification is particularly critical for a submission claiming to address the detection of unauthorized training data.

- Algorithm 1, Line 22: variable naming inconsistency — $perf_r$ should be in uppercase ($Perf_r$) for consistency.

- Algorithm 1 explicitly specifies the auxiliary model as GPT-5, but Algorithm 2 does not mention which model is used.

### Soundness
2

### Presentation
1

### Contribution
2
