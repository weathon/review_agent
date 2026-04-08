## Human Reviewer 1

### Summary
This work investigated why benign relearning happens when using gradient based heuristic for unlearning LLM. Different from prior explanations, this work observed that syntactic similarity is the primary driver of why relearning works instead of topical relevance. The paper analyzes the BLUR dataset and showed that the relearn success rate is mainly due to the syntactic similarity, evaluated by the cosine similarity between relearn set and target set. The paper further proposes a new way of unlearning by using GPT to rewrite the forget set so that the forget set is syntactically different from the target set, which effectively limits the power of relearning.

### Strengths
- The paper is well written.
- The paper provides a new explanation of the success of relearning in the context of LLM unlearning, which is important for understanding in this community.

### Weaknesses
- One thing that has been significantly underlooked in this paper is **between which two sets** should syntactic similarity be looked at. There are three sets: unlearning set $D_{forget}$, eval set $D_{target}$, relearn set $D_{relearn}$. Section 5 characterizes the syntactic similarity between $D_{target}$ and $D_{relearn}$, but there are other dimensions: syntactic similarity between $D_{forget}$ and $D_{relearn}$ and syntactic similarity between $D_{forget}$ and $D_{target}$. In the TOFU case, an implicit assumption is $D_{target}$ and $D_{forget}$ high overlaps. Therefore, $D_{relearn}^{syntactic}$ is syntactically different enough from both $D_{target}$ and $D_{forget}$. However, such assumption might not be true for e.g. WMDP, where $D_{forget}$ is pub-med articles and $D_{target}$ are some expert drafted questions, not necessarily about the verbatims in the articles themselves. From the current analysis, what is missing is, **whether the success of relearning is due to syntactic similarity between $D_{target}$ and $D_{relearn}$ or $D_{forget}$ and $D_{relearn}$, or more complex among all three sets**.
- It is also important to make clear separation between knowledge unlearn (such as wmdp, where $D_{target}$ and $D_{forget}$ can usually be very syntactically different) and verbatim unlearn (such as tofu, where $D_{target}$ and $D_{forget}$ are potentially closer, but also try cases where you paraphrase $D_{target}$ itself) and investigate the above question under both cases.
- The robust unlearning part is less convincing to me. In practice, designing unlearn set given target set is unfair. The ideology should be a defender build an unlearn set with the purpose of defending against model outputting sensitive information, not against a set of fixed queries. Moreover, one can always rewrite the target set with GPT as well. 
- Figure 8 is also less convincing. For TOFU, the original relearn set is a subset of the unlearn set. Now that if the forget set changes, shouldn't the relearn set also changes? Otherwise, this is not an apple-to-apple comparison as the adversarial in this case has less information.

### Questions
- Have the authors explore systematic analysis on knowledge unlearning task such as WMDP?
- I like this paper and think this paper provides good observations. The most important thing is a clearer and more systematic investigation to the weakness 1 above. That's the major issue I think the authors should address. The defense part in its current form is also not convincing as it is too rough and did not consider stronger adversarial. **I am willing to raise my score if both points have been further explored and explained in the rebuttal.**

### Soundness
2

### Presentation
4

### Contribution
3

### Rating
4

### Confidence
5

---

## Human Reviewer 2

### Summary
This paper delves into a key and perplexing phenomenon in the field of LLM unlearning—benign relearning.
The core argument of this paper is that the primary driver of benign relearning is not, as previously thought, topical relevance, but rather syntactic similarity. The authors first use rigorous experimental design to identify confounding variables in the evaluation methods of previous benchmarks and then correct these experiments, finding that the role of topic relevance is overestimated.

To verify their core hypothesis, the authors construct two relearning datasets: "topically related but syntactically different" and "topically unrelated but syntactically similar." Experimental results show that syntactically similar data triggers information retrieval more effectively. The paper further analyzes the underlying mechanism from the perspectives of representation and gradients, revealing that syntactically similar data and the forgetting target are highly aligned within the model, and that the standard forgetting process suppresses "templates" rather than "keywords," thus leaving structural "backdoors."

Based on this finding, the paper proposes a simple yet effective solution—syntactic diversification. This method enriches the syntactic structure of the forgetting set through paraphrasing before forgetting. Experiments demonstrate that this method not only effectively inhibits benign relearning but also accelerates the forgetting process and significantly reduces the impairment to the model's generalizability.

### Strengths
1. The problem is clearly and significantly addressed. The authors challenge mainstream understanding in the field (topic relevance-driven) and propose a novel and insightful perspective (syntactic similarity-driven), crucial for understanding the failure of forgetting mechanisms.

2. The experimental design is rigorous and comprehensive, ensuring fair and rigorous evaluation. Evaluation confounding in BLUR is identified and eliminated, the number of steps is standardized, and the optimal result is chosen, making the conclusions more reliable.

3. The defense methods are simple and practical. Syntactic diversification requires no modification to the optimizer or model structure, yielding significant results (slower and weaker relapses, fewer forgetting steps, and better utility), and demonstrating robustness.

4. The writing is clear and logically coherent: the paper's structure is clear and easy for readers to understand.

### Weaknesses
1. Limitations of the syntactic similarity metric. The paper uses normalized Levenshtein distance as a measure of syntactic similarity. While this is a simple and effective character-level metric, it may fail to capture more abstract and deeper syntactic structures (such as the structural similarity of parse trees). It is suggested to explore using more sophisticated syntactic analysis tools to measure syntactic similarity, which could reveal more subtle mechanisms.

2. Cost of the proposed solution. The proposed solution relies on the robust GPT-4o model to generate syntactic variants. It is suggested to conduct a short ablation experiment to explore the effects of using smaller, more cost-effective open-source models (such as Llama-3-8B itself) or simpler methods for syntactic diversification.

### Questions
see weakness

### Soundness
3

### Presentation
4

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper studies why unlearning comes back after benign fine-tuning (“benign relearning”) in LLMs. Contrary to the common belief that topic overlap (e.g., Harry-Potter-related text) is the main culprit, the authors argue and empirically show that syntactic similarity (shared surface forms/templates) is the real driver: fine-tuning on sentences with the same structure (even about different entities) reactivates the forgotten content. They formalize and test this on BLUR/TOFU-style setups, diagnose evaluation confounds in prior relearning studies, and propose a simple mitigation, syntactic diversification (paraphrasing forget queries into varied structures before unlearning). This diversification both reduces relearning, speeds up forgetting, and improves utility trade-offs.

### Strengths
1. The paper shows syntactic similarity (in the query), not topicality, is the consistent relearning driver across methods (GA, NPO, SCRUB) and datasets. The paper also identifies evaluation confounds (dataset size -> step budget i.e., non-monotonic training trajectories) that can make topicality look stronger than it is, then re-evaluates with a step-standardized protocol. This corrects the narrative and is a valuable insight for the community.

2. The analysis provided  with the Heatmaps, relearning vs unlearning steps, demonstrating the and analyses show syntactically similar relearn sets recover forgotten names more strongly than topically relevant ones (measured via keyword presence / ROUGE-L to base). Representation- and gradient-similarity analyses support the mechanism. Overall syntactic overlap aligns hidden states and gradients with the target set after unlearning, explaining recovery.

3. Syntactic diversification (multi-paraphrase forget queries via GPT-4o) reduces template rigidity, balances loss between templates and keywords, delays/attenuates relearning, requires fewer unlearning steps, and improves utility (e.g., Real Authors, World Facts and Retain split). That’s a rare win-win (better robustness with less utility damage).

### Weaknesses
1. The approach relies on GPT-4o paraphrasing. What is the cost/latency at unlearning time for large forget sets, and does quality vary by domain or language? A scaling/cost analysis and a cheaper in-house paraphrase baseline would help adoption.

2. Keyword-based relearn success rate captures name reappearance but may miss partial leakage or paraphrastic leakage. Similarly, ROUGE-L to base captures surface similarity but not factual equivalence. Including embedding-based and judge-LM evaluations would strengthen claims.

3. The overall message in the paper (templates get suppressed more than keywords during unlearning) is compelling, but stronger causal tests (e.g., controlled template injection/removal, counterfactual templates, layerwise intervention) would be needed to further substantiate this analysis.

### Questions
1. Can you report results with richer syntactic similarity measures (e.g., tree edit distance, template mining) and show which correlates best with relearning?

2. Can you add judge-LM / embedding-based leakage metrics to complement keyword/ROUGE and analyze disagreements, if not then can you help explain why such an analysis could not be performed?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3