# vCache: Verified Semantic Prompt Caching

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Semantic caches return cached responses for semantically similar prompts to reduce LLM inference latency and cost. They embed cached prompts and store them alongside their response in a vector database. Embedding similarity metrics assign a numerical score to quantify the similarity between a request and its nearest neighbor prompt from the cache. Existing systems use the same static similarity threshold across all requests to determine whether two prompts can share similar responses. However, we observe that static thresholds do not give formal correctness guarantees, result in unexpected error rates, and lead to suboptimal cache hit rates. This paper proposes vCache, the first verified semantic cache with user-defined error rate guarantees for predictable performance. It employs an online learning algorithm to estimate an optimal threshold for each cached prompt, enabling reliable cache responses without additional training. Our experiments show that vCache consistently meets the specified error bounds while outperforming state-of-the-art static-threshold and fine-tuned embedding baselines with up to 12.5$\times$ higher cache hit and 26$\times$ lower error rates. We release the vCache implementation and four benchmarks to support future research.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes using dynamic thresholds for different prompts in semantic caching, thereby providing an error rate guarantee. Based on a sigmoid parametric model and observed samples, the proposed online learning algorithm continually estimates the optimal thresholds for cached prompts. Experimental results demonstrate its effectiveness in controlling error rates.

### Strengths
- The paper is well structured and easy to follow.
- Guaranteed error rate is a crucial issue for semantic caching, and the proposed online-learning-based dynamic threshold method provides a reasonable solution.

### Weaknesses
- Although vCache can guarantee the error rate of the semantic cache, it heavily relies on the observation of correctness (Algorithm 1, Line 8). In the experimental section, the authors propose using exact matching for short prompts and LLM inference for long prompts to determine correctness. The additional LLM inference introduces extra cost for the semantic cache, which undermines its practical value. Moreover, the LLM could make mistakes in judgment, making the guarantee unreliable.
- Why is the actual error rate in Figure 4 much lower than $\delta$? According to the guarantee, they should not differ by such a large margin.
- Despite the controlled error rate, the risk of private data leakage still exists, which limits its practical adoption in industry.
- Since LMArena contains many testing prompts, the high hit rate shown in Figure 4 might result from these testing cases. Can the authors provide details about the hit prompts for the three benchmarks?

### Questions
See weaknesses

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes vCache, a caching system that optimizes GPT-Cache and dynamically returned the cached responses subjected to the pre-defined error rate. Compared to the previous works that relies on a static threshold for all incoming queries and all cached responses, this paper adopts online learning to estimate a threshold for each cached response. To help evaluate the performance of GPT-cache style caching systems, the author generates three benchmarks to help evaluate the performance. Abundant empirical experiments on different models show the effectiveness of this method.

### Strengths
- This paper tackles on an important topic of GPT-cache and provides unique perspectiives on how to optimize this problem. Static threshold is a fundmental problem that prevents GPT-cache for better selecting cached responses.
- The evaluation is holistic and abundant. The author also presents large-scale benchmarks for helping research in this field.
- Author also presents good theoretical guarantees for showing the effectiveness of this method,

### Weaknesses
- Dataset Generation neglects no match scenario. I checked the appendix about how the data is generated. For instance, for the SemCacheLMArena, 1 to 23 similar prompts will be generated. In reality, there could be many prompts where no similar answer in the cache can be fetched directly. I would suggest adding many prompts where no similar one variants are included should be helpful.
- Though the experiments regarding error rate is abundant, more analysis of latencies and throughputs should be added.

### Questions
- For creating the dataset SemCacheLMArena, how to ensure the sampled prompts are not really distinct? 
- Where do you define? I guess it's time per response?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes vCache, a novel semantic caching system for LLMs that provides user-defined error rate guarantees while learning embedding-specific thresholds online. The key innovation is replacing static global thresholds with dynamic, per-embedding thresholds estimated through an online learning algorithm that models the probability of correctness using sigmoid functions. The method is evaluated on three benchmarks across different embedding models and LLMs, demonstrating superior performance compared to static threshold baselines.

### Strengths
- The paper addresses a practical challenge of the semantic prompt caching 
- The proposed method is well-motivated

### Weaknesses
- The writing is a bit repetitive and could be streamlined

### Questions
Thank you for your submission. I appreciate the motivation behind providing formal correctness guarantees for semantic caching systems, which is indeed a significant limitation of existing approaches. The experimental validation across multiple benchmarks and the introduction of new evaluation datasets are valuable contributions. However, several aspects of the paper require clarification and the technical approach raises some concerns:

- The sigmoid modeling assumption (Equation 9) is quite strong but not well justified. Why should the relationship between similarity and correctness follow a sigmoid specifically? Have you experimented with other parametric families or non-parametric approaches?
- The confidence band computation for parameters t and γ (mentioned in Section 4.2 and relegated to Appendix C) is crucial for the guarantees but insufficiently explained in the main text. How sensitive are the guarantees to the choice of confidence level ε?
- Algorithm 2 shows that τ is computed by minimizing over ε ∈ [0,1], but this seems computationally expensive for online inference. What is the actual computational overhead of this optimization step?
- The paper claims vCache "consistently meets the specified error bounds" but Figure 4 shows the actual error rate is noticeably below the specified δ. This suggests the method might be overly conservative, potentially sacrificing cache hit rate for unnecessary safety margins. Can you quantify this conservatism?
- The comparison with GPTCache is somewhat unfair since GPTCache doesn't attempt to provide error guarantees. A more relevant baseline would be other adaptive thresholding methods from the retrieval literature adapted to this setting.
- The evaluation focuses on relatively simple benchmarks (classification, search queries). How does vCache perform on more complex scenarios like multi-turn conversations or reasoning tasks where semantic similarity becomes more nuanced?
- The i.i.d. assumption for incoming prompts is quite restrictive in practice. Real-world query distributions often exhibit temporal correlations, user-specific patterns, and concept drift. How robust is vCache to violations of this assumption?
- Figure 3's motivation is compelling, but the connection to the proposed solution could be clearer. It shows the problem but doesn't intuitively explain why sigmoid modeling would solve it.

### Soundness
3

### Presentation
3

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
This paper proposes vCache, a semantic LLM caching method that guarantees a user-defined error rate, based on the previous work of GPTCache, a semantic LLM caching method to optimize LLM cost and latency. This work contributes to the domain of LLM caching by introducing per-embedding dynamic threshold, which was improved from the static threshold of GPTCache.

### Strengths
1.	This study algorithmically improves from the existing threshold-based retrieval methods such as GPTCache using a novel approach. It theoretically guarantees a desired error rate and demonstrates improved performance across several metrics by introducing a verified semantic cache.
2.	As an online learning algorithm, the proposed method achieves strong results without fine-tuning the embedding model, thus requiring no additional training.
3.	To validate the effectiveness of the proposed methodology and foster future research, the authors have constructed and publicly released three new benchmark datasets (SemCacheClassification, SemCacheLMArena, and SemCacheSearchQueries) that reflect real-world caching scenarios.

### Weaknesses
1.	Since vCache is still a semantic caching technique, there is an insufficient evaluation of its effectiveness concerning the capability and performance across the underlying embedding model (e.g., BERT). A comparative analysis using various embedding models is needed to substantiate the "Model-Agnostic" claim, which currently lacks sufficient analysis.
2.	While the paper discusses the trade-off between accuracy and cost, the benchmark results show a trade-off between Cache Hit Rate and Error Rate when compared to GPTCache. For instance, in Appendix D.5, the best-case for GPTCache (GS) shows a 5.2% error rate with a 67% hit rate, whereas vCache (LD3) achieves a 2.0% error rate with a 54% hit rate. Although vCache has the distinct advantage of reliably guaranteeing a user-defined error rate (e.g., 2.0%), the lack of in-depth analysis on this trade-off makes it difficult to conclude its superiority across all scenarios. Combined with Weakness 1, this property raises questions about whether the improvement of vCache against GPTCache is marginal or not.
3.	As mentioned in the limitations section, the evaluation relies on an LLM-as-a-judge for benchmarks except SemCacheClassification, which has a clearly defined correctness criterion.

### Questions
•	Were comparative experiments conducted with different embedding models? An analysis of the relationship between the performance of the embedding model and the performance gains from vCache would better substantiate the "Model-Agnostic" claim. Beyond the GteLarge and E5-large models presented, were experiments with other BERT models—such as multilingual variants or those employing improved techniques—considered?
•	When a new embedding is added to the cache, how many observations are required for vCache to learn a stable threshold? I am curious about the analysis of performance degradation during the initial learning phase (the cold-start problem).

### Soundness
3

### Presentation
2

### Contribution
3
