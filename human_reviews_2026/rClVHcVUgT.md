# PATEin: A Privacy-Preserving Framework for Knowledge Integration via Adaptive Teacher Selection in C-LLMs

- Decision: Reject
- Scores: 6, 2, 4, 6

## Abstract
In-context learning (ICL) enables task adaptation without modifying model parameters, making it well-suited for commercial large language models (C-LLMs) with closed-source constraints. However, ICL prompts often contain sensitive information, raising significant privacy concerns. Most existing privacy-preserving methods for ICL require access to model parameters, making them incompatible with C-LLMs. Recent methods based on teacher ensembles with differentially private aggregation have shown promise but face two fundamental challenges: ensemble inconsistency and limited knowledge integration. We propose PATEin, a novel privacy-preserving knowledge transfer framework that dynamically selects the optimal individual teacher model for labeling, thereby mitigating the loss of individual knowledge. Furthermore, it introduces a supervised teacher strategy that selectively incorporates high-consistency voting, effectively integrating individual and ensemble knowledge. Experiments on various C-LLMs (e.g., GPT-3.5-turbo, GPT-4o-mini, Claude-3.5-haiku, DeepSeek-v3) demonstrate that PATEin significantly improves labeling accuracy, reduces computational overhead, and consistently outperforms existing baseline methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents PATEin, a privacy-preserving knowledge transfer framework for commercial large language models (C-LLMs), extending the PATE (Private Aggregation of Teacher Ensembles) paradigm. Unlike previous ensemble-based approaches (e.g., PromptPATE), PATEin addresses two major challenges—ensemble inconsistency and loss of individual teacher knowledge—by introducing adaptive teacher selection and a supervised high-consistency voting mechanism. The framework combines individual and ensemble-level knowledge via similarity-based teacher matching and dynamic aggregation, while preserving differential privacy through a Confident-GNMax mechanism. Experiments on multiple datasets (AGNews, SST-2, DBPedia, TREC) and commercial LLMs (GPT-3.5, GPT-4o-mini, Claude-3.5, DeepSeek-v3) demonstrate that PATEin improves labeling accuracy and cost-efficiency under equivalent privacy guarantees.

### Strengths
1.The paper clearly identifies the limitations of ensemble-only methods in privacy-preserving in-context learning and proposes an elegant adaptive teacher selection strategy that preserves both individual and collective knowledge.
2.Experiments cover multiple datasets and commercial LLMs, demonstrating robustness and practical applicability. The inclusion of ablation studies (teacher count, voting threshold) provides useful insights into hyperparameter sensitivity.
3.The written is well-organized, with logical progression from problem definition to algorithmic design and empirical validation. Figures and tables are informative and support the main claims.

### Weaknesses
1.The paper claims that adaptive teacher selection mitigates ensemble inconsistency, but the mechanism lacks formal analysis. No theoretical results (e.g., bounds on privacy–utility trade-off or optimality of teacher selection) are provided.
2.The teacher selection relies on cosine similarity between embeddings (Doc2Vec and text-embedding-3-small), but this approach may not capture deeper task semantics or label-level consistency.
3.While the paper mentions the use of the Confident-GNMax mechanism and claims (ε, δ)-DP compliance, the derivation is deferred to the appendix without concrete parameter values or sensitivity analysis.
4.Although the threshold and teacher count are analyzed, other key factors—such as the influence of noise scale (σ), ensemble size diversity, or supervision strength—are unexplored.
5.The paper emphasizes labeling quality but gives little detail on how student models benefit in downstream fine-tuning or real-world applications beyond token cost.

### Questions
The paper selects the “optimal” individual teacher model based on cosine similarity between embeddings (Doc2Vec and text-embedding-3-small).However, it remains unclear why text similarity correlates with labeling accuracy.Could the authors provide quantitative evidence (e.g., correlation between similarity and correctness rate) or compare against random teacher selection?


The paper mentions the Confident-GNMax mechanism to ensure (ε,δ)(\varepsilon, \delta)(ε,δ)-DP but does not report concrete privacy parameters or the chosen noise scale σ\sigmaσ.How are these parameters determined, and how do they affect the privacy–utility trade-off?A more explicit description of the privacy accounting process would help evaluate the strength of the privacy guarantees.


Figure 4 qualitatively shows complementarity between individual and ensemble teachers, but it is unclear how much this integration contributes to the final performance.Could the authors add an ablation study comparing three variants: (1) individual-only, (2) ensemble-only, and (3) hybrid (PATEin)?


Current experiments focus mainly on text classification with small to medium C-LLMs.Can PATEin scale to larger models (e.g., 14B+) or to complex reasoning and dialogue tasks?Any preliminary results or discussion would help clarify the framework’s applicability to broader LLM scenarios.

### Soundness
3

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
The paper studies privacy-preserving in-context learning. It proposes PATEin, a framework that combines teacher selection and selective ensemble voting to improve labeling accuracy and reduce query cost under claimed differential privacy guarantees.

### Strengths
- The motivation is clear.
- The code and datasets are released.

### Weaknesses
- The privacy analysis of PATEin is incorrect. PATEin first selects the “most similar” teacher based on comparisons between public inputs and each teacher’s private training data, but this selection process is not differentially private, i.e., changing one private record could change which teacher is chosen. Then, when the top two teachers agree, PATEin outputs that label directly without adding noise, which completely violates differential privacy because the output depends deterministically on private data. Therefore, the experimental comparison between PATEin and PromptPATE is not meaningful.

- The novelty of PATEin is limited. The paper works in the same problem setting as PromptPATE. The only new elements are teacher selection and selective ensemble voting. These are incremental extensions to PromptPATE.

- The problem formulation is not clear. I recommend adding a Problem Formulation section. This section should clearly define in-context learning, threat model, and differential privacy formulation.

- Experiments are limited to simple text classification benchmarks. These benchmarks are too limited for evaluating modern LLM methods. Prior work [1] on privacy-preserving in-context learning includes more complex tasks like summarization and question answering.

[1] Wu, Tong, et al. "Privacy-Preserving In-Context Learning for Large Language Models." The Twelfth International Conference on Learning Representations.

- The paper lacks ablation studies on different embedding models and noise levels. There is no comparison between using only individual, only ensemble, or combined knowledge.

### Questions
See Weaknesses.

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
4

### Summary
The paper proposes PATEin, a variant of the PromptPATE framework that integrates Private Aggregation of Teacher Ensembles (PATE) into an in-context learning (ICL) setting for closed-source commercial LLMs.
Instead of using all teachers in an ensemble, the method selects a subset of relevant teachers, based on text similarity between the query and each teacher’s example data, to reduce computational cost and improve label quality.
The authors claim this improves ensemble consistency and integrates individual teacher knowledge while maintaining privacy through Gaussian noise aggregation.

While the topic of private label aggregation in C-LLMs is timely and relevant, the novelty is limited, as the proposed “adaptive teacher selection” essentially amounts to pre-filtering which teachers vote in PATE, using standard text-similarity metrics.
The work lacks a clear privacy analysis of the selection step and overstates its conceptual contribution.
Presentation is somewhat confusing, with the abstract and introduction suggesting a new “knowledge integration framework” rather than a PATE variant with heuristic teacher filtering.

### Strengths
The direction of combining PATE and in-context learning is important and continues a promising research line (PromptPATE, Duan et al. 2023).

Addressing the high cost of multiple LLM API calls is practically relevant; exploring teacher pre-selection for efficiency is reasonable.

The paper includes experimental evaluation on multiple C-LLMs (GPT-3.5, GPT-4o-mini, Claude-3.5-Haiku, DeepSeek-v3).

### Weaknesses
### Limited Novelty and Conceptual Depth

The method retains the same overall PATE structure: partition data, generate teacher prompts, vote with added Gaussian noise, and aggregate.
The new component, a similarity-based filtering of teachers, is a heuristic efficiency improvement, not a conceptual extension of PATE.

The privacy guarantees remain those of PATE, and the overall privacy protection still depends on the support size (number of participating teachers) and the maximum agreement among them.

### Lack of Clarity about Privacy Implications

The abstract (lines 20–21) claims “dynamic selection of the optimal individual teacher model”—but such selection of one or very few relevant teachers is counter to privacy. Selective querying itself could leak information about which teachers are similar to a query.

The paper does not discuss how this adaptive selection interacts with the differential-privacy accounting.

Statements like “selects the optimal teacher model for labeling” may give a misleading impression that privacy is preserved automatically, when in fact the selection must be handled carefully to avoid additional leakage.

### Overstatement of Contribution

The claim (lines 107–110) that this is “the first privacy-preserving knowledge integration framework tailored to C-LLM settings” is an oversell. The actual novelty is only in compute efficiency not utility for privacy.
The proposed “integration of ensemble and individual knowledge” is effectively conditional voting based on teacher confidence.

### Presentation and Readability

It takes several pages to understand what the actual algorithmic change is.
The introduction repeatedly discusses “knowledge integration” and “ensemble consistency” before stating that the novelty is teacher pre-filtering based on text similarity. That the only potential gain is compute effciency (not accuracy for privacy). Heavy terminology such as “supervised teacher strategy” and “optimal individual teacher model" distracts.  A use of a single best-match teacher model, as implied in this sentence,  is highly NOT privacy preserving and this leaves the reader pondering.

Some empirical sections (Figs. 2–4) are difficult to interpret and could be summarized more compactly.

### Technical Oversights

The argument in lines 196–200 incorrectly suggests that PATE requires a majority agreement for the noisy argmax to function.
In fact, differential privacy mechanisms (e.g., DP selection or GNMax) do not rely on majority consensus; the probability of selection depends on the noise scale.

There is no formal privacy accounting for the similarity-based filtering or dynamic teacher selection. Coceptually it can be viewed as part of the voting, but this requires some care.

## Overall Assessment

The paper explores a potentially useful engineering variation on PromptPATE (for classification tasks) but lacks sufficient novelty or theoretical analysis to constitute a research advance worthy of ICLR standards.

### Questions
NA

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper targets privacy-preserving knowledge transfer for commercial large language models with inaccessible parameters, addressing the high cost and ensemble inconsistency of existing PATE methods. The authors propose PATEin, an adaptive framework that first queries the top two teachers identified by embedding similarity. If these two teachers agree, their label is used, avoiding a costly full ensemble query. If they disagree, the system falls back to a standard, differentially private ensemble aggregation. Experiments show PATEin outperforms baselines like PromptPATE in labeling accuracy across multiple C-LLMs and datasets, while significantly reducing API token cost (up to 22x), thus making private transfer more practical.

### Strengths
The paper is well-structured. The abstract and introduction clearly state the problem, challenges, and solution. This setting has practical value as it aims to use adaptive queries to reduce the API cost in using PromptPATE with closed LLMs.

### Weaknesses
1.  The method relies on embedding similarity to find the optimal teachers. As a result, the robustness and effectiveness of the proposed method heavily rely on the selection of the embedding model.
2. PATEin requires building a similarity matrix between all teachers and all public data, which is potentially computationally expensive. I encourage the authors to be upfront about this potential bottleneck.

### Questions
1. The effect of cost saving depends on the fallback rate. How does this rate vary across datasets?
2. Table 4 seems to suggest a very high fallback rate under an adversarial teacher. Does this suggest the adaptive query method is not robust?
3. How scalable is the teacher selection step to even larger public datasets?
4. Could the distribution shift between public and private data influence the teacher fallback?

### Soundness
3

### Presentation
3

### Contribution
2
