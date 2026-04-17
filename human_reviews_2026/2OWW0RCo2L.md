# Fast or Better? Balancing Accuracy and Cost in Retrieval-Augmented Generation with Flexible User Control

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 2

## Abstract
Retrieval-Augmented Generation (RAG) has emerged as a powerful approach to mitigate large language model (LLM) hallucinations by incorporating external knowledge retrieval. However, existing RAG frameworks often apply retrieval indiscriminately, leading to inefficiencies---over-retrieving when unnecessary or failing to retrieve iteratively when required for complex reasoning. Although recent retrieval strategies can adaptively navigate among alternative retrieval strategies, they make their selection based solely on query complexity and incorporate no mechanism for prioritizing speed over accuracy or vice versa. This lack of user-defined control makes their use infeasible for diverse user application needs. In this paper, we introduce a novel user-controllable RAG framework that enables dynamic adjustment of the accuracy-cost trade-off. Our approach leverages two classifiers: one trained to prioritize accuracy and another to prioritize retrieval efficiency. Via an interpretable control parameter $\alpha$, users can seamlessly navigate between minimal-cost retrieval and high-accuracy retrieval depending on their specific requirements. We empirically demonstrate that our approach effectively balances accuracy, retrieval cost, and user controllability \footnote{Code is available at anonymous github \url{https://anonymous.4open.science/r/Flare-RAG-Anonymous-D6A2/}.}, making it a practical and adaptable solution for real-world applications.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces **Flare-Aug**, a user-controllable Retrieval-Augmented Generation (RAG) framework that balances accuracy and computational cost through an interpretable parameter $\alpha$. Flare-Aug enables users to dynamically adjust retrieval strategies according to their specific needs. The framework employs two classifiers: a Cost-Optimized Classifier (trained to select the cheapest correct retrieval strategy for a specific LLM) and a Reliability-Optimized Classifier (trained on dataset-level labels for LLM-agnostic stability). Users control the interpolation between these classifiers via $\alpha \in [0, 1]$. Evaluated on 6 QA datasets with Flan-T5 and GPT-4o, Flare-Aug demonstrates good accuracy-cost trade-offs compared to Adaptive-RAG and static baselines.

### Strengths
**Novel user controllability**: enabling explicit, interpretable control over accuracy-cost trade-offs via a single parameter

### Weaknesses
1. **LLM Usage Disclosure**.
   - The manuscript contains extensive use of em dashes (—) appearing 19 times, which may indicate significant LLM involvement in paper writing. 
   - The conference guidelines explicitly state: "Not disclosing significant LLM usage can lead to desk rejection." The authors should clarify the extent of LLM usage in preparing this manuscript to comply with submission policies.

2. **Evaluation Metric Issues**
   - Metric Definition: The accuracy metric is vaguely defined as "whether the predicted answer contains the ground-truth answer" (lines 328-329). 
   - This should specify either Exact Match (EM) or token-level F1 score following standard QA evaluation metrics.

3. **Reproducibility Concerns**
   - The authors randomly sample 500 queries from each dataset rather than using standard test splits. This deviates from established benchmarks and prevents reproducibility.
   - The paper should either use official test sets or provide exact query IDs for the sampled subsets.

4. **Insufficient Baseline Comparisons**
   - The baselines are limited to Adaptive-RAG and static strategies, ignoring recent agentic RAG systems that autonomously determine retrieval necessity and stopping conditions. 
   - The paper should compare against agentic systems: AutoRAG, Search-o1, Search-R1, R1-Searcher, ReasonRAG, etc. Meanwhile, modern adaptive methods with planning capabilities should be compared.
   - These systems naturally handle the search decision-making that Flare-Aug addresses, making them critical comparisons.

5. **Outdated Model Choices**. Authors use Flan-T5 (XL/XXL) released in 2022, as answer generation models. This raises concerns about generalizability. The method should be validated on modern LLMs (e.g.,Llama-3,  Qwen2.5,  Qwen3) widely used in 2024-2025.

6. **Weak Baseline Hypothesis**: In Figure 3, multi-step retrieval sometimes underperforms Flare-Aug, suggesting the answer model may be too weak to leverage retrieved information effectively. This questions whether observed gains stem from the proposed method or model limitations.

7. **Classifier Training Overhead**. While authors claim computational efficiency (~640 seconds), they understate the data preparation burden:
   - Query sampling across datasets
   - Running LLM inference with all three strategies for labeling
   - Manual verification of correctness

   Moreover, the classifier supports only 3 decision classes. In contrast, modern LLMs possess strong built-in planning capabilities and can perform retrieval decisions in a training-free manner. The added complexity of maintaining separate classifiers seems unnecessary given advances in LLM reasoning.

8. **Retrieval Strategy Limitations**
   - BM25 Dependency: The exclusive use of BM25 is outdated. The claim that "BM25 continues to outperform many dense retrievers" (citing Ram et al., 2023) does not hold in 2025.
   - Modern embedding models (BGE-M3, Qwen2.5-Embedding, Voyage-3) offer substantially superior retrieval quality. The method should be evaluated with state-of-the-art retrievers.


9. **Multi-Step Retrieval Details Missing**. The paper lacks crucial implementation details:
   - Query decomposition: How are sub-queries generated for multi-step retrieval?
   - Knowledge gap identification: What determines missing information?
   - Stopping criteria: When does multi-step retrieval terminate?

   Without these details, the multi-step baseline cannot be reproduced, and its fairness as a comparison is questionable.

8. **Minor Issues**
   - Line 67-68: "Flare-Agu" should be "Flare-Aug"
   - Figure quality: Figures 3-5 could benefit from larger fonts and axis rescalng for readability

### Questions
Recommendations for Revision
- Add disclosure of any LLM usage in manuscript preparation
- Clarify evaluation metrics (EM vs. F1) and use standard test sets
- Include modern baselines: Self-RAG, agentic RAG systems with full results
- Validate on current LLMs: Qwen2.5, Llama-3.1, Qwen3
- Provide multi-step retrieval details: query decomposition, stopping criteria, implementation
- Compare with modern retrievers: BGE-M3, Qwen3-embedding-0.6B or similar dense models
- Justify classifier necessity given modern LLM planning capabilities

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces Flare-Aug, a user-controllable RAG framework featuring two externally trained routing modules, a Cost-Optimized Classifier (LLM-specific, selecting the least costly strategy that still yields a correct answer) and a Reliability-Optimized Classifier (LLM-agnostic, trained on single-hop vs. multi-hop supervision). Experiments on a mixed subsets of SQuAD, NQ, TriviaQA, MuSiQue, HotpotQA, and 2WikiMultiHopQA demonstrate smooth, monotonic accuracy–cost trade-offs.

### Strengths
- The single user-exposed alpha to navigate accuracy–cost trade-offs is intuitive; the author also shows accuracy and retrieval steps increasing monotonically with alpha choices.
- The cost-oriented router is LLM-specific; the reliability router is LLM-agnostic, which is a reasonable decomposition for deployment.
- Experiments across single-hop and multi-hop QA senarios show its effectiveness and strengthes.

### Weaknesses
- This work interpolates model parameters of two independently trained classifiers with a linear combination (ref. formula in Line 303-304), but there is no justification that a linear blend in parameter space yields calibrated probabilities or coherent decision boundaries, especially their training objectives and label spaces are different. This control design seems ad-hoc rather than theoretically grounded.
- A follow up question based on the first concerns: Why linear parameter interpolation rather than other strategies like score-space ensembling or a learned gate? Is there any calibration or decision-quality checks across alpha?
- The LLM-agnostic classifier is labeled entirely by dataset identity, i.e., single-hop --> single-step and multi-hop --> multi-step. It is noisy and leverages dataset bias rather than ground-truth retrieval needs. This weak supervision risks over-retrieval on many "easy" queries and under-retrieval on "hard single-hop" ones.
- Only BM25 is used for retrieval, however, dense/hybrid retrievers are also common settings. Switching to these settings may change the trade-off surface.
- Dataset scale is relatively small, which limits statistical power and stress testing.

### Questions
Please refer to weaknesses

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes Flare-Aug, a user-controllable RAG framework that enables dynamic accuracy-cost trade-offs via a tunable parameter α. The system trains two classifiers—Cost-Optimized (minimizes retrieval steps) and Reliability-Optimized (ensures answer quality)—then interpolates between them based on user preferences. Evaluated on 6 QA datasets with 4 LLMs, the method achieves competitive performance while providing flexibility lacking in prior work like Adaptive-RAG.

### Strengths
1. Addresses Real User Need
Identifies genuine limitation in existing adaptive RAG: inability to adjust retrieval strategy based on application constraints. Medical diagnosis vs. customer chatbots have fundamentally different accuracy-latency requirements.2. Simple, Interpretable Design
The α parameter provides intuitive control: α=0 prioritizes speed, α=1 prioritizes accuracy. Monotonic relationships (Figures 3-5) make system behavior predictable and easy to tune.3. Solid Experimental Methodology

Tests 4 LLMs (Flan-T5-XL/XXL, GPT-4o/mini) showing generalizability
Mixed dataset (3 single-hop + 3 multi-hop) simulates query diversity
Extensive ablations on classifier size, training epochs, α values
Low training cost (~640 seconds for both classifiers)
4. Practical Deployment Guidance
Offers concrete α-tuning strategies (incremental adjustment, validation-based estimation) with empirical validation that validation set trends transfer to test set (Appendix C.2).

### Weaknesses
1. Limited Technical Novelty
Core contribution is linear classifier interpolation: W = (1-α)W_cost + αW_reliability. This is standard multi-objective optimization used widely in ensemble methods and multi-task learning—not a novel algorithmic insight.2. Weak Theoretical Foundation
Reliability Classifier assumes single-hop datasets → single-step retrieval, multi-hop datasets → multi-step retrieval. This dataset-level labeling:

Conflates dataset construction bias with actual query complexity
Remains empirically unvalidated (acknowledged in A.4 but dismissed)
May misclassify queries (e.g., simple questions in multi-hop datasets forced into expensive retrieval)
3. Oversimplified Cost Model
Uses only retrieval step count as cost proxy, ignoring:

LLM inference costs (GPT-4 >> Flan-T5 per token)
Retrieval latency variations (BM25 vs. dense retrievers)
Context length impact on GPU memory
Real deployment costs = API fees + wall-clock time
4. Catastrophic Failure with Unanswerable Queries
Table 2 shows adding "unanswerable" class causes performance collapse. Cost Classifier loses ability to discriminate retrieval strategies, instead overfitting to answerable vs. unanswerable classification. No solution provided.5. Unfair Baseline Comparison
Claims Adaptive-RAG lacks flexibility, but:

Adaptive-RAG can retrain for different cost-accuracy preferences
Flare-Aug also requires validation tuning of α (not truly "online")
Real difference: one-time training + parameter vs. multiple training runs—a practical but not paradigmatic advantage
6. Strong Baseline Buried in Appendix
Table 5 shows direct prompting (asking LLM to decide retrieval strategy) achieves Acc=0.381, Steps=0.19—competitive with full system at lower complexity. Relegating this to appendix rather than engaging substantively raises questions about necessity of dual classifiers.

### Questions
1. Have you tested on ANY out-of-distribution data? (e.g., customer service logs, medical records, code repositories, legal documents).

In particular:
Academic Query (your test set):
"Who invented the telephone?" 
→ Single-step retrieval works perfectly

Real-World Query (day 1 of deployment):
"When will my order #12345 ship?"
→ Your classifier predicts single-step BM25 retrieval
→ Actually needs SQL database query (no text retrieval!)
→ System fails completely

2. Table 5 shows direct prompting (Acc=0.381, Steps=0.19) nearly matches your method (Acc=0.388, Steps=1.3) at 85% lower cost and zero training. Why not simply prompt GPT-4: "Given this query and user priority (speed/accuracy), should I retrieve?" This naturally handles distribution shift without dataset-specific classifiers.

3. Your Cost Classifier labels are LLM-specific (training on Flan-T5-XL's successes/failures), but users may deploy on different LLMs. When a company switches from Flan-T5 to GPT-4, doesn't this require complete retraining, negating the "reusability" claim?

### Soundness
2

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
5

### Summary
This work proposes a parametric classifier that determines the retrieval strategy for each individual query in RAG (Retrieval-Augmented Generation). The retrieval strategy for a query can be one of the three choices: no retrieval, single-step retrieval, and multi-step retrieval. The classifier was trained a T5 model. In experiments, a RAG system with the BM 25 retrieval model was setup for evaluation. The answering models in the experiments include Flan-T5-XL, Flan-T5-XXL, GPT-4o-mini, and GPT-4o. Experimental results show the addition of the classifier improves the RAG performance for the smaller models, while the more capable LLM, GPT-4o, performs slightly better with multi-step retrieval. However, the addition of the classifier can greatly reduce the retrieval steps.

### Strengths
* Multi-step RAG is a practical and challenging issue. 

* The proposed method can be generally applied to many existing RAG systems and LLMs.

### Weaknesses
* The training data and the test data were sampled from the same datasets. It is not clear if the test data were held-out from the training data. 

* In addition, it is also unclear how well the proposed classifier is applied to the test data from a new domain. 

* The addition of the classifier, which is a T5 model, adds computational cost and latency to the RAG pipeline. However, this paper did not provide an analysis of this overhead. A comparison between GPT-4o + the proposed classifier and GPT-4o + multi-step retrieval in terms of runtime could be added to clarify the benefit of the proposed method.

### Questions
* Did you hold out the test data from the training data? 

* In addition to the T5 model, did you consider an even lightweight backbone model for the classifier?

### Soundness
1

### Presentation
2

### Contribution
2
