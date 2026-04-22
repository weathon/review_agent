# Log Probability Tracking of LLM APIs

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 2, 6, 8

## Abstract
When using an LLM through an API provider, users expect the served model to remain consistent over time, a property crucial for the reliability of downstream applications and the reproducibility of research. Existing audit methods are too costly to apply at regular time intervals to the wide range of available LLM APIs. This means that model updates are left largely unmonitored in practice. In this work, we show that while LLM log probabilities (logprobs) are usually non-deterministic, they can still be used as the basis for cost-effective continuous monitoring of LLM APIs. We apply a simple statistical test based on the average value of each token logprob, requesting only a single token of output. This is enough to detect changes as small as one step of fine-tuning, making this approach more sensitive than existing methods while being 1,000x cheaper. We introduce the TinyChange benchmark as a way to measure the sensitivity of audit methods in the context of small, realistic model changes.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work addresses the problem of detecting changes in the model behind a blackbox LLM API. The authors propose detecting changes by monitoring the logprobs output by the API, which is possible for the ~23% of APIs that return logprobs. The key challenge is that logprobs can differ from call-to-call due to unintentional non-determinism. To address this, the authors propose logprob tracking (LT), which performs a permutation test on the difference between logrpobs. The authors find that this method achieves better detection AUC than baselines that do not use logprobs, while being less-costly to serve.

### Strengths
- The method is very simple to implement, when logprobs are available.
- The result that the detection AUC is not affected by prompt length is somewhat interesting.
- The authors release a benchmark for evaluating methods like this one.
- I like the plots showing how logprobs evolve over time.

### Weaknesses
- There is limited technical novelty in the methodology. Checking for differences in logprobs is the de facto approach for checking the correctness of language model implementations and APIs (*e.g.* from the VLLM tests https://github.com/vllm-project/vllm/blob/66a168a197ba214a5b70a74fa2e713c9eeb3251a/tests/models/utils.py#L90). It is well known that this is a more sensitive test than simply checking for text equality. That this is more effective than just checking text outputs is not a surprising result.
- I’m concerned about the significance of the problem and the practical utility of the method given the practices of frontier model providers today:
    - Most LLM APIs today (77%), do not return log-probs. Frontier model providers like OpenAI and Anthropic do not provide log-probs in responses. Given that the most popular API endpoints due not provide log-probs in response, it is unclear the practical utility of the method for most practitioners.
    - It’s unclear the degree to which the claim in the first sentence is true: “users of LLM APIs (developers, researchers, regulators) generally rely on the assumption that calling the same API endpoint will consistently serve the same model.” Most frontier LLMs are updated quite regularly (sometimes weekly or monthly), and most users are unaffected by the fact that the model is being updated. What should the user do if the model is updated? Frontier API providers do not provide the ability to use the old API.

### Questions
The method is not effective at detecting changes with LoRA fine-tuning. Why do you think this is?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper prsents Logprob Tracking (LT), a technique for LLM drift detection under resource constraints. The key idea is to sample the log probability of the first generated token, and then perform a permutation test. This allows LT to detect minor model updates with 1000x lower cost than existing methods. The authors also propose a benchmark to test small model modification and conduct experiments on controllable model drifts and real-world API drifts.

### Strengths
- Addresses an important and underexplored reproducibility problem: LLM APIs are increasingly widely applied while continously updated, and monitoring these behavior shift is an important topic.

- Simple, cost-efficient, and elegant approach: The proposed technique is easy to understand and implement: it basically tests if the distribution of the log probability of the first generated token has changed or not, using a permutation test. This is neat, and also cost-efficient, as it only requires the first token of a given query.


- New benchmark (TinyChange) for fine-grained change detection: This benchmark offers a systemtical way to generate model drift with different magnitudes.

### Weaknesses
- Depends on APIs exposing logprobs: As the authors notice too, only a small fraction of existing API providers (~23% in openrouter) offers logprob access.

- Detections w/o directions: The proposed method only detects whether a change occurred, not what changed or how the change looked like. In particular, it is unclear if the change leads to better responses to user queries, or what kind of biases or skills were introduced or forgotten in the model update. In practice, this is often more important than just detecting a change.


- Limited evaluation and analysis of real-world APIs: Section 3.2.3 briefly mentions drift detection of real-world APIs and the model providers' responses. A more detailed analysis on real-world API drift would improve the paper a lot and make it more relevant to practical senarios.

### Questions
- How robust is LT when the logprob sampling temperature or top-k cutoff changes across time?

- Could LT be extended to detect which type of change (e.g., quantization vs. fine-tuning) occurred?

- Have the authors considered multi-token extensions or dynamic prompting to improve evasion resistance?

- Can TinyChange be used to evaluate fingerprinting methods as well, given the conceptual overlap?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces Logprob Tracking (LT), a cost-effective and highly sensitive method for continuously monitoring LLM APIs for unintended or undisclosed changes.  Users rely on LLM APIs remaining consistent, yet providers frequently implement model modifications that go unmonitored. LT addresses this by exploiting the log probabilities of the first output token, which provide a significantly richer information source than the generated tokens alone. To overcome the non-deterministic nature of log probabilities in production environments, LT employs a simple two-sample permutation test based on the average absolute distance between per-token mean log probabilities. Experiments on the TinyChange benchmark show LT can detect changes as small as a single fine-tuning step, while achieving cost reductions of up to 1,000 times compared to baselines like MET and MMLU-ALG.

### Strengths
LT drastically reduces the cost of continuous monitoring, achieving sensitivity gains at a cost that is up to three orders of magnitude cheaper than competing state-of-the-art methods

LT provides substantially higher discriminative power and sensitivity than existing approaches. It reliably detects small modifications such as a single step of fine-tuning and demonstrates detection performance for weight pruning at an amplitude 512 times smaller than MET.

The authors use permutation tests on the mean absolute difference of per-token average log probabilities to handle the inherent non-determinism observed in production APIs.  This is a nice addition that addresses a key limitation in monitoring production systems.  Because the permutation test only requires minimally acquired data from a single output token from a 1-token input prompt, the overall cost of monitoring is drastically reduced.

### Weaknesses
The entire methodology is contingent on the API provider supporting and returning log probabilities. Data presented in the paper indicates that only 23% of reachable endpoints on OpenRouter support this.  This limits the applicability of the approach.

LLM providers can obstruct LT by requiring minimum output token lengths 

 The reliance on log probabilities for only the first output token might miss certain modifications such as adjusting the generation-length parameter.

### Questions
What mechanisms, beyond binary detection, can be integrated into the LT framework to help auditors diagnose the likely source or severity of the detected change (e.g., distinguishing an infrastructure update from a behavioral fine-tuning step)?

How does LT perform against variants created in the TinyChange benchmark where only the EOS token log probability bias is subtly modified, rather than the core model weights?

### Soundness
3

### Presentation
4

### Contribution
4
