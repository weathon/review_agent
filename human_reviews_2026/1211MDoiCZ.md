# Routesplain: Towards Faithful and Intervenable Routing for Software-related Tasks

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 8, 2, 2

## Abstract
LLMs now tackle a wide range of software-related tasks, yet we show that their performance varies markedly both across and within these tasks. Routing user queries to the appropriate LLMs can therefore help improve response quality while reducing cost. Prior work, however, has focused mainly on general-purpose LLM routing via black-box models. We introduce Routesplain, the first LLM router for software-related tasks, including multilingual code generation and repair, input/output prediction, and computer science QA. Unlike existing routing approaches, Routesplain first extracts human-interpretable concepts from each query (e.g., task, domain, reasoning complexity) and only routes based on these concepts, thereby providing intelligible, faithful rationales. We evaluate Routesplain on 16 state-of-the-art LLMs across eight software-related tasks; Routesplain outperforms individual models both in terms of accuracy and cost, and equals or surpasses all black-box baselines, with concept-level intervention highlighting avenues for further router improvements.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces Routesplain, the first LLM router for software-related tasks that can adaptively select a model that can balance cost and effectiveness. It leverages concept bottleneck models to achieve interpretable routing without sacrificing performance (compared with black-box counterparts).

### Strengths
1. Timely problem. The paper tackles a timely problem. Intelligently routing to use a cost-effective model can save a lot of money without sacrificing much performance.
2. Strong empirical evidence as motivation. In section 3, the paper performs an extensive study to show the inter-task and intra-task performance variance among models, highlighting the necessity of routing for software tasks.
3. Novel approach. The use of the concept bottleneck model (CBM) is novel and appropriate for the routing problem.
4. Solid evaluation. The authors compare Routesplain against baselines, including a black-box MLP router (that is not interpretable). Evaluation shows that Routesplain achieves better interpretability without losing performance.
5. Strong ablation and interesting findings. The paper performs ablation studies and delivered several interesting findings. For instance, Table 2 identifies the most salient features for the routing task (complexity, programming language). And the intervention study pinpoints the primary bottleneck for the framework is predicting query complexity.

### Weaknesses
1. Ambiguous or imprecise definition about “complexity”. The paper defines “complexity” as the fraction of models that failed on a given input. This definition is kind of circular (i.e., a query is complex because the strong models are required, and the strong models are required because it is complex). 
2. Scalability of the concept space. The set of concepts is manually defined and tied directly to the labels available in the evaluation datasets. It remains unclear whether manual labeling and concept-based learning can scale to new tasks.

### Questions
1. Could you elaborate on your choice for the complexity definition? It’s reasonable to use the “complexity” metric, but more measurements can be used as well, e.g., existing code complexity metrics from software engineering perspective like cyclomatic complexity. Did you consider or experiment with complexity metrics like that? Do you believe the failure-rate-based label is learnable in a generalizable way?
2. What’s the cost of updating this framework, e.g., when adding a model to the pool, changing a model’s pricing, or updating the tasks? Although the retraining only takes 1 minute, what about the cost of data generation?
3. The intervention results in Table 2 show that fixing "Programming Languages" or "Libraries" yields no accuracy benefit, even though the concept classifier is not perfect for them (Table 3). You state the second-stage classifier "absorbs" these errors. This is an interesting finding. Does this imply that, for routing, a perfectly accurate concept classifier isn't actually necessary for non-complexity concepts? Does the model learn, for example, that "Rust" and "Go" are "similarly hard" and thus map them to a similar set of suitable models, making a misclassification between them less harmful?
4. The right of Figure 5 clearly shows the model assignment shifting as $\lambda$ increases. Does this $\lambda$ value need to be set manually, or do you envision a system where the user can provide a "budget" or "priority" (e.g., "fastest," "cheapest," "highest quality") that dynamically selects the appropriate $\lambda$ for the query?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Routesplain, a routing system for large language models on software-related tasks. It selects the most suitable model by analyzing concepts extracted from queries, making the routing decisions both interpretable and intervenable.

### Strengths
- This paper successfully migrates the routing approach to the programming language scenario.
- The evaluation comprehensively covers 8 software tasks and 16 mainstream LLMs.
- It assesses performance variations of different models across these 8 tasks.

### Weaknesses
- The paper claims that the system is “interpretable” and “intervenable,” but these claims are largely conceptual. The so-called “interpretable” provided by the system appears to consist merely of a list of concept labels used in its decision-making. Furthermore, the described interventions do not meaningfully improve model performance and primarily serve to correct misclassifications made by the concept classifier. If the core purpose of the intervention is simply to rectify the system’s own errors, it remains unclear whether this should be considered a feature or a design flaw.
- The paper dedicates excessive discussion to LLM performance metrics, which comes at the expense of a clear exposition of its core methodological contributions. Given the rapid evolution of LLM performance, much of the reported data will quickly lose scientific relevance. The paper should shift its emphasis from these metrics to methodological contributions and technical novelty.
- Comparisons with black-box routing baselines remain superficial and lack in-depth mechanistic analysis. The paper does not clearly demonstrate in which specific aspects concept-based routing outperforms black-box methods in decision-making. 
- Unlike directly querying a single fixed model, Routesplain introduces additional steps of concept classification and model selection, which incur significant latency for each query.  These steps add significant overhead for each query, which may be unacceptable in latency-sensitive real-world applications.
- The paper assumes users can simultaneously access and manage APIs for 16 top-tier LLMs, which is impractical in terms of both cost and control for most users. This severely limits the method’s generalizability and real-world applicability.
- The authors fail to clearly define the scenarios in which Routesplain is indispensable. For many common tasks, simply selecting a high-cost-performance general-purpose model (e.g., LLaMA 4 Scout) is already near-optimal. The marginal performance gains from introducing a routing system do not appear sufficient to justify the added complexity and cost.
- Key terms such as “interpretable” and “intervenable” are used without proper citations or with inappropriate references that do not strongly support the claims. 
- The paper uses non-standard references such as “left of Figure 2” or “right of Figure 3,” which are inappropriate. Standard notation (e.g., Figure 2a, Figure 3b) should be used.

### Questions
- For code-related tasks, a straightforward approach to improving reliability is Pass@k (generating k outputs per problem). Please clarify why constructing such a complex routing system offers advantages over this direct and effective method.
- Given that many existing models already achieve strong performance across a broad spectrum of tasks, for common applications, selecting a top-tier model (e.g., LLaMA 4 Scout) is often near-optimal. The authors should clarify whether the marginal performance gains achieved through routing substantively justify the additional complexity and computational overhead introduced by the system.
- The authors should provide comparative experiments demonstrating whether Routesplain offers tangible benefits over simpler prompt engineering strategies (e.g., explicitly specifying programming language and task type in prompts). 
- Since the paper focuses on the code domain, why were specialized code models such as Code Llama, DeepSeek-Coder, or StarCoder 2 not considered? Would including these models provide a fairer reflection of the state-of-the-art in this domain?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work propose a routing framework to send queries to experts for software-related task. The proposed method, RouteSplain, trains a concept router that map the input query to concepts via natural language, and then to a selected model in a pre-defined pool. The results showed that RouteSplain achieved a pareto front with a black-box MLP router, while offering more explainability.

### Strengths
- Routing queries to appropriate models to maximize performance and minimze cost is an important research problem.
- The proposed method achieved encouraging results, outperforming individual models.

### Weaknesses
- Suboptimal router formulation: a major drawback of RouteSplain is that while the router understands the queries, it does not understand the strengths and weaknesses of each component models. Thus, the trained router in RouteSplain is just a more advanced version of a count-based solution that counts how many times a model is selected for each combination of concept, and retrieve that during inference. This might explain why most routing strategies have similar result curves in Figure 5 Left. A more desirable solutions should also take into account each component experts during inference, before providing a routing score. 

- RouteSplain has limited extensibility and out-of-distribution performance. Training and evaluation of RouteSplain follows the traditional in-domain setting, where a dataset is splitted into train-val-test sets, and the results are only reported in the test split. This strategy is quite limited in LLM evaluation. Can the trained RouteSplain generate to new benchmarks that share some similar software tasks such as CodeMMLU [A]. Despite being discussed in Section 6, I think that these aspects need to be investigated more thoroughly.

[A] Manh, Dung Nguyen, et al. "Codemmlu: A multi-task benchmark for assessing code understanding capabilities of codellms." ICLR (2025).

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
