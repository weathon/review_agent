# Selective Deferred Routing: Enabling Cost-Efficient Collaboration between Local SLMs and Remote LLMs

- Avg Score: 3.33
- Decision: Reject
- Scores: 4, 2, 4

## Abstract
The rapid advancement of large language models (LLMs) has led to remarkable performance across diverse domains such as question answering, creative writing, programming, etc., making them indispensable assistants in daily life and work. Currently, LLM services are primarily accessed in two ways: (i) paid access to cloud-hosted LLMs, which are powerful but introduce nontrivial cost; and (ii) deployment of small language models (SLMs) on personal devices or small clusters, which, while less powerful, are sufficient for handling relatively simple tasks. To achieve a balanced trade-off between monetary cost and task performance, we propose Selective Deferred Routing, a paradigm that enables cost-efficient collaboration between local SLMs and remote LLMs. In this framework, a user request is first processed by the local SLM, which not only generates a preliminary response but also provides rich semantic representations of the request. A lightweight decider module then leverages this information to either adopt the initial response or route the request in a single step to the most suitable remote LLM for a higher-quality response. Extensive experiments on 5 LLMs and 3 datasets demonstrate that our approach consistently outperforms existing multi-LLM collaboration methods across different cost–performance trade-off preferences.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces Selective Deferred Routing (SDR), a cost-efficient paradigm that optimizes collaboration between local small language models (SLMs) and remote large language models (LLMs). The approach leverages local models to generate preliminary responses and utilizes a decider module to route requests to the most suitable remote LLM for enhanced output. The paper demonstrates that SDR consistently outperforms existing multi-LLM collaboration methods by improving the cost–performance trade-offs across a variety of tasks and datasets, with extensive experimental evaluations involving five LLMs and three datasets.

### Strengths
1. The idea of combining local SLMs and remote LLMs in a selective and cost-efficient manner is novel. It addresses the challenge of balancing monetary cost and task performance, a significant concern in practical applications of LLMs.

2. The proposed method is well-structured, with clear descriptions of the decider module and the associated scoring model, backed by a strong theoretical foundation. The experiments demonstrate the effectiveness of SDR, showing consistent improvements over baseline methods in the single-remote and multi-remote scenarios.

3. The paper is generally well-written, with clear explanations of the methodology, theoretical formulations, and experimental setups. The figures, such as the cost-performance curves and AUC metrics, effectively illustrate the improvements achieved by SDR

4. The approach can significantly reduce the operational costs of LLMs by optimizing the trade-off between local and remote model performance. The flexibility of the method for various user preferences makes it highly applicable in real-world systems that deploy LLMs.

### Weaknesses
1. While SDR offers an interesting method for cost optimization, it shares conceptual similarities with existing research, particularly Immediate Routing and Model Cascades. The paper could have further highlighted how SDR uniquely overcomes the limitations of these existing methods, such as the sequential inefficiency in Model Cascades or the oversimplification of Immediate Routing.

2. The experimental settings section (pages 8–9) provides a good overview of model configurations, but the hyperparameter tuning details, especially for the decider module and thresholds, are vague. Without a detailed explanation of how these parameters were optimized, it's unclear how much the performance gains are attributable to fine-tuning versus the intrinsic qualities of SDR. Also, it would be beneficial to provide open-source code to ensure full reproducibility.

3. Although SDR performs well in multi-remote scenarios, the scalability of the model with multiple remote LLMs might pose practical issues, particularly regarding memory and computation. The parallel running of multiple scoring models during deployment, though lightweight, may still lead to increased computational overhead in large-scale systems.

### Questions
1. The paper discusses the use of a threshold parameter (α) for routing decisions (Equations 2 and 3). Can you elaborate more on how these thresholds are dynamically adjusted in real-world use cases, and whether the model adapts to changing task characteristics over time?

2. The experiments focus on single-task scenarios, but many real-world applications of LLMs require multi-task performance. How well does the Selective Deferred Routing approach perform when tasks with vastly different characteristics (e.g., creative writing vs. technical problem-solving) are combined?

3. The paper mentions cost-effective collaboration, but what happens in scenarios with limited cloud resources or heavy latency on the edge devices? Is SDR robust enough to handle extremely resource-constrained environments, and how does it compare to other edge-cloud hybrid models?

4.  Given the lightweight nature of the decider module, what are the potential latency concerns when scaling SDR to a production environment with high-frequency queries? Are there any optimizations or architectural changes that can further reduce response time in such scenarios?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper introduces Selective Deferred Routing (SDR), a two-stage collaboration scheme: a local SLM first answers and emits hidden-state features; a lightweight “decider” then either accepts the local answer or routes once to a remote LLM, aiming to optimize a cost–performance tradeoff. The core contribution is to cast binary selective routing as an AUC-style objective over a cost–performance curve, leading to a ranking-consistent training loss (via a Binary-Gap surrogate) for the decider initialized from a single SLM transformer layer. Experiments over 3 datasets and 5 LLMs show favorable tradeoffs against routing/cascade baselines, plus a simple rule to extend to multi-remote settings.

### Strengths
- Clear formalization of selective routing with an AUC objective that directly targets cost–performance trade-offs; the gap-ordering optimality condition is crisp and intuitive.

- Lightweight decider design that reuses a single Transformer layer from the local SLM; practical and latency-friendly.

- Empirical results cover five LLMs / three datasets, reporting normalized AUC curves (single-remote) and actual USD costs (multi-remote).

### Weaknesses
- Latency measurements and on-device memory/compute overheads for running the decider in parallel with the SLM are discussed qualitatively, but not benchmarked. 

- Label generation cost is under-specified: training the decider requires evaluating both local and remote outputs per query to estimate gaps/BG labels; the offline token cost could rival the savings at deployment time, but is not quantified. 

- Baselines omit some recent routers/cascades in the multi-LLM literature beyond those cited; fairness of API defaults (temperatures, system prompts) across providers is not detailed. 

- Generalization across local models is unclear: the decider is initialized from a specific SLM layer; portability to different SLMs or quantization variants is not evaluated.

### Questions
- What is the total offline cost (tokens × price) to collect BG labels per dataset and per remote LLM, and after how many live queries does SDR break even?

- How sensitive is SDR to changing the local SLM (e.g., different size/architecture, quantization levels)? Can the decider trained on one SLM transfer without re-labeling? 

- For the multi-remote case, why is averaging the scores the right aggregation? Have you compared to learned policies (e.g., budget-constrained bandits or cost-aware ranking) or provided a regret bound?

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
This paper proposes Selective Deferred Routing, a paradigm that enables cost-efficient collaboration between local SLMs and remote LLMs. In this framework, a user request is first processed by the local SLM, which not only generates a preliminary response but also provides rich semantic representations of the request. A lightweight decider module then leverages this information to either adopt the initial response or route the request in a single step to the most suitable remote LLM for a higher-quality response.

### Strengths
1. Multi-LLM Routing is important and on time.

2. Selective Deferred Routing balances cost and accuracy.

3. Extensive experiments on 5 LLMs and 3 datasets are provided.

### Weaknesses
1. The latency is not reported. Judge after SLM finishes generation can take very long. After SLM generates a few tokens, it might be sufficient to start routing.

2. Fine tuning a BERT before SLM and LLMs may have lower latency and cost.

3. In GSM8K Figure 4, the performance is close to FrugalGPT. Code is also not provided.

### Questions
In GSM8K Figure 4, why using Llama-4-Maverick as the LLM? How about GPT and Deepseek?

### Soundness
2

### Presentation
2

### Contribution
2
