# Private-RAG: Answering Multiple Queries with LLMs while Keeping Your Data Private

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 2, 6, 6

## Abstract
Retrieval-augmented generation (RAG) enhances large language models (LLMs) by retrieving documents from an external corpus at inference time. When this corpus contains sensitive information, however, unprotected RAG systems are at risk of leaking private information. Prior work has introduced differential privacy (DP) guarantees for RAG, but only in single-query settings, which fall short of realistic usage. In this paper, we study the more practical multi-query setting and propose two DP-RAG algorithms. The first, MURAG, leverages an individual privacy filter so that the accumulated privacy loss only depends on how frequently each document is retrieved rather than the total number of queries. The second, MURAG-Ada, further improves utility by privately releasing query-specific thresholds, enabling more precise selection of relevant documents. Our experiments across multiple LLMs and datasets demonstrate that the proposed methods scale to hundreds of queries within a practical DP budget ($\varepsilon\approx10$), while preserving meaningful utility.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper draws the attention that in a real-world, single-query DP RAG cannot meet the privacy requirements in multiple round query settings as the privacy budget will accumulate. A large privacy budget is needed to reach acceptable utility. To encounter the challenge, the authors proposed a revised framework using a privacy filter for each document. Experiments were conducted to support the main argument.

### Strengths
1. The paper identified current challenge on privacy RAG where sufficiently large $\epsilon$ is needed to achieve reasonable utility, which is an important conclusion. The illustration of motivation is important to following research.
2. The formulation of the research question is clear.
3. Abundant experiments are conducted to study the effectiveness of multiple-round query.

### Weaknesses
1. In Konga et al., the guarantee is stated in $(\epsilon,\delta)$-DP, but this paper reports RDP. Please clarify the conversion pathway and assumptions. In particular, walk readers through how the moments accountant (or equivalent) maps the mechanism’s $(\epsilon,\delta)$-DP bounds to $(\alpha,\epsilon_\alpha)$-RDP, and specify where Appendix C/D supply the exact orders $\alpha$, composition steps, and the final back-conversion (if any). Please see more in my question.
2. The tuning of $\gamma$ appears critical, especially because a document is excluded once its budget is exhausted. Please analyze utility--privacy trade-offs across $\gamma$, provide guidance for selection (e.g., via validation curves or heuristics), and discuss robustness (variance across runs/datasets).
3. The current evaluation relies on privacy budgets and membership-inference attacks (MIA), which are indirect. For stronger external validity, include empirical extraction tests: issue adversarial queries and measure whether private content (documents/strings) is retrieved or generated. Reporting concrete leakage rates alongside MIA would materially strengthen the privacy claims.

Minor:
- Typo: `R'enyi` --> `Rényi`.
- Since Algorithm 5 is referenced in the main text, consider moving it from the appendix to the main body to improve readability.

### Questions
1. What does notation $S\Delta\tilde{S} $mean?

2. How is the utility guaranteed if a document is excluded. Say one document contains a lot of privacy information and it is frequently visited, which is common in the real-world, how would the following queries be correct if it is excluded?

 think these should be included in the main paper. My confusion was not resolved by reading the main paper so I turned to check the Appendix. You start with a single query DP and replace the limited domain with an exponential mechanism in the private token generation step. How do you calculate the sensitivity? Exponential mechanism is $\epsilon$-DP and so as the conclusion in D.1. Then how is it converted to RDP in D.2. I think (b)<$\epsilon$ in D.2 is based on D.1. But how is the conclusion in D.1 derived? Corollary 3.3 is based on $\epsilon,\alpha$-DP, how do you make it to the $\epsilon$-DP? What value of $\alpha$ do you select for RDP.

3. Why other baselines are not presented in Figure 3 (Right).

### Soundness
2

### Presentation
2

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
This paper proposes two multi-query differentially private RAG algorithms (MURAG and MURAG-ADA) that improve privacy budget efficiency through per-document budget management and threshold-based screening mechanisms. The main contribution lies in engineering design optimization, extending single-query DP-RAG to multi-query scenarios, though technical innovation is limited.

### Strengths
1. The paper addresses the relevant problem of privacy protection in multi-query RAG systems.
2. The experimental evaluation covers multiple LLM models and includes privacy attack assessments.

### Weaknesses
1. The evaluation focuses primarily on privacy budget consumption while providing insufficient analysis of system complexity costs introduced by per-document budget management. Maintaining individual states for large numbers of documents may introduce significant computational overhead and storage requirements.
2. The technical contribution is mainly at the engineering design level with relatively limited core algorithmic innovation. While per-document budget management and threshold screening are effective, they primarily represent reasonable combinations of existing techniques.
3. The experimental scale is relatively limited, with the largest model being 7B parameters and a modest number of test questions, which may not adequately validate the method's performance in large-scale real-world deployments.
4. The threshold setting strategies have some limitations. Fixed threshold selection lacks clear guiding principles, while adaptive thresholds require additional privacy budget consumption, and their net benefits need further verification.
5. The practical application demand for multi-query differential privacy RAG may not be as widespread as described in the paper, as most RAG systems likely employ alternative privacy protection strategies or handle non-sensitive data.

### Questions
1. What would be the total system costs of per-document state management in real systems with millions of documents?
2. Could you provide a more comprehensive cost-benefit analysis that includes factors such as system complexity?
3. How can service quality be maintained when popular documents exhaust their privacy budgets?
4. Are there more principled methods for threshold parameter selection?

### Soundness
2

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
4

### Summary
The paper addresses privacy risks in retrieval-augmented generation when many queries are issued over the same private corpus. It proposes PRIVATE-RAG, which introduces two methods, MURAG and MURAG-ADA, that apply differential privacy at the document level so privacy loss depends on how often each document is retrieved instead of the total number of queries. Experiments on several LLMs and datasets show that the approach maintains good utility while supporting hundreds of queries within a practical privacy budget.

### Strengths
Strengths: 
1. The paper addresses a relevant and timely problem: how to make RAG systems private when handling many user queries.
2. The idea of tracking privacy at the document level rather than per query is simple but effective, and it clearly reduces the cost of privacy accounting.
3. The two proposed variants are well motivated for different query patterns.
4. The experiments cover a good range of datasets and models, showing solid performance under realistic privacy budgets.

### Weaknesses
1. Practical limitation: The proposed method is well-motivated and theoretically sound. However, in realistic deployments there are often scenarios where a single query—or a group of related queries—has high semantic overlap with a large portion of the corpus. In such settings, the thresholding mechanism would struggle to filter out many documents, so most entries would still be “touched” and thus consume privacy budget. This situation can substantially reduce the efficiency advantage of per-document accounting when the retrieval space is broadly relevant.
2. Presentation clarity: The method section is technically solid but somewhat dense; adding an overall flow diagram that shows how retrieval, thresholding, and privacy accounting connect would make the process much easier to follow.

### Questions
During adaptive thresholding in MURAG-ADA, the algorithm discretizes similarity scores into bins [a_i,a_i+1). How are the number of bins and boundaries chosen in practice, are they global or per-query, and how does this affect privacy and retrieval precision?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This paper addresses a critical gap in differentially private (DP) retrieval-augmented generation (RAG): existing DP-RAG methods are limited to single-query settings and become impractical under composition when handling multiple queries. To overcome this, the authors propose MURAG and its adaptive variant MURAG-ADA, two novel multi-query DP-RAG algorithms that leverage per-document individual Rényi privacy filters to track privacy loss based on document retrieval frequency rather than total query count. MURAG uses a fixed relevance threshold to select documents, while MURAG-ADA privately releases query-specific thresholds via noisy prefix sums to better handle correlated queries. The methods are evaluated across three LLMs (OPT-1.3B, Pythia-1.4B, Mistral-7B) and four datasets—including independent (Natural Questions, TriviaQA), correlated (MQuAKE), and privacy-sensitive (ChatDoctor) benchmarks. Results show that both methods achieve meaningful utility under a realistic total privacy budget (ε ≈ 10) for hundreds of queries, significantly outperforming naive composition baselines and synthetic-data alternatives, while also defending against state-of-the-art multi-query membership inference attacks.

### Strengths
1. Practical and Timely Problem: The paper tackles a real-world limitation of current DP-RAG systems—scalability to multiple queries—making DP-RAG viable for production deployments in sensitive domains like healthcare or legal services.

2. Comprehensive Evaluation: The experiments span diverse datasets, LLMs, query correlation regimes, and include both utility metrics and robustness against a strong multi-query membership inference attack (Interrogation Attack). The results consistently validate the claims.

### Weaknesses
I must candidly note that I am not deeply familiar with the technical nuances of differential privacy (DP), particularly advanced topics such as Rényi DP filters and individual privacy accounting. Consequently, I may have missed potential weaknesses in the paper’s privacy analysis, algorithmic design, or theoretical claims. I recommend that the reviewers with stronger expertise in DP carefully examine the correctness and novelty of the privacy guarantees and the implementation details of the proposed mechanisms.

### Questions
see the weakness

### Soundness
3

### Presentation
3

### Contribution
3
