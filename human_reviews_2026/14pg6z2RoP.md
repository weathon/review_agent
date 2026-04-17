# Unraveling Indirect In-Context Learning Using Influence Functions

- Decision: Reject
- Scores: 2, 4, 6, 6

## Abstract
In this work, we introduce a novel paradigm for generalized In-Context Learning (ICL), termed Indirect In-Context Learning. In Indirect ICL, we explore demonstration selection strategies tailored for two distinct real-world scenarios: Mixture of Tasks and Noisy ICL. We systematically evaluate the effectiveness of Influence Functions (IFs) as a selection tool for these settings, highlighting the potential of IFs to better capture the informativeness of examples within the demonstration pool. For the Mixture of Tasks setting, demonstrations are drawn from 28 diverse tasks, including MMLU, BigBench, StrategyQA, and CommonsenseQA. We demonstrate that combining BertScore-Recall (BSR) with an IF surrogate model can further improve performance, leading to average absolute accuracy gains in 3-shot and 5-shot setups when compared to traditional ICL metrics. In the Noisy ICL setting, we examine scenarios where demonstrations might be mislabeled or have adversarial noise. Our experiments show that reweighting traditional ICL selectors (BSR and Cosine Similarity) with IF-based selectors boosts accuracy on noisy GLUE benchmarks. For the adversarial sub-setting, we show the utility of using IFs for task-agnostic demonstration selection for backdoor attack mitigation. Showing a reduction in Attack Success Rate compared to task-aware methods. In sum, we propose a robust framework for demonstration selection that generalizes beyond traditional ICL, offering valuable insights into the role of IFs for Indirect ICL.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces a new paradigm for generalized in-context learning, called indirect in-context learning (Indirect ICL), which addresses scenarios where the standard assumption of having a clean, task-aligned demonstration pool is violated. The paper considers two main tasks to demonstrate the effectiveness of the proposed paradigm: 1) Mixture of Tasks (MoT), where the pool contains demonstrations for many tasks, among which only a fraction is relevant to the test tasks, 2) Noisy ICL, where the pool contains corrupted demonstrations.

The paper proposes using influence functions (IFs) to capture the "inductive bias" of task demonstrations with the end-task. There are some empirical results showing the superiority of the proposed method compared to a few constructed baseline methods.

### Strengths
- The paper studies an interesting and important problem of task demonstration selection for ICL, including task misalignment and noisy or corrupted task demonstrations in the pool.
- The paper offers some interesting insights via experiments, e.g., that pruning hurts rather than helps performance and that DataInf is better than TracIn as an IF method.

### Weaknesses
- The proposed method requires training the model to obtain the IF of the demonstrations. As such, a fair experimental comparison should involve training-based baseline methods, for example, SFT on the ICL data.
- Apart from that, the experiments are more like an ablation of the proposed method, missing performance comparison with baseline methods, including several ICL selection methods, for example, [1,2]
- Missing analysis: It is quite counterintuitive to grasp why a surrogate model outperforms running IF on the LLM itself. The authors should explain more about this point.
- Missing (actual) computational cost comparison: On top of that, what is the speedup gained by utilizing the surrogate model? How is the speed compared to other training-free baselines? While I appreciate that the paper includes a complexity analysis, it does not reflect the real computation incurred during deployment.
- Question on novelty: the idea of using IF for ICL has been explored previously, for example, in [1,3]. Notably, [1] constructs a "surrogate model" in the form of a linear regression on the hidden states of the LLM itself and can compute the IF of demonstrations quickly. Given the similarity, I hope the authors can provide an explanation comparing the proposed method and [1,3], and also include some experiments comparing the methods.



[1] Zhou et al., DETAIL: Task Demonstration Attribution for Interpretable In-context Learning, NeurIPS 2024.

[2] Peng et al., Revisiting Demonstration Selection Strategies in In-Context Learning, EMNLP 2024.

[3] M.S. et al., In-Context Learning Demonstration Selection via Influence Analysis, arXiv 2402.11750.

### Questions
- What is a precise definition of indirect in-context learning? I'm looking for either a definitive language description or a math formulation. I can't seem to find it in the paper.

The other questions are in the weaknesses.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a generalized paradigm for In-Context Learning (ICL) termed Indirect In-Context Learning, which addresses demonstration selection for two challenging, real-world scenarios: a Mixture of Tasks setting and a Noisy ICL setting. The authors propose using Influence Functions (IFs) as a tool to select informative demonstrations from a pool that is either task-heterogeneous or contains label noise. They demonstrate that combining IFs from a surrogate model with traditional semantic similarity metrics like BERTScore-Recall can lead to modest performance gains in the MoT setting. For the noisy setting, they show that IF-based pruning or averaging can improve accuracy on mislabeled data and reduce the success rate of backdoor attacks.

### Strengths
- Originality in Problem Formulation: The paper identifies and formalizes a relevant and under-explored problem—ICL when direct, clean task demonstrations are unavailable. The MoT and Noisy ICL settings are practical and important.

- Thorough Empirical Evaluation: The authors conduct extensive experiments across multiple LLMs, datasets, and settings (MoT, noisy, adversarial), which adds credibility to their findings.

- Clarity and Scope: The paper is clearly written, and the exploration of both surrogate-model and pretrained-gradient approaches to IFs is a valuable comparison.

- Actionable Insights: The finding that a simple combination of semantic similarity and IF-based re-ranking is effective is a practical takeaway.

### Weaknesses
- Limited Empirical Improvement: The performance gains are consistently small, often within a few percentage points, and their statistical significance is not established. This undermines the claim that IFs provide a substantial benefit.

- Incremental Technical Contribution: The methodology is an application of existing tools rather than a novel algorithmic or theoretical contribution. The two-stage selection process is a simple ensemble of existing techniques.

- Scalability and Cost Concerns: While computational complexity is analyzed, the practical cost of fine-tuning surrogate models or computing gradients from very large models (e.g., Llama-3-70B) for every task is non-trivial and not sufficiently justified by the modest gains achieved.

- Lack of Ablation on IF Variants: The paper tests multiple IF methods but provides limited analysis explaining why certain methods (e.g., DataInf) outperform others (e.g., TracIn) beyond the lack of second-order information.

### Questions
Please refer to Weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a new paradigm of Indirect In-Context Learning (Indirect ICL), which is mainly oriented to two types of more realistic problem settings: Mixture of Tasks (MoT) and Noisy ICL. The core idea is to utilize Influence Functions (IFs) to select/reorder candidate examples and combine them with traditional semantic similarity selection metrics (e.g., BertScore-Recall, cosine similarity) to improve ICL performance in the context of mismatch of task distribution or noisy labels. The authors demonstrate the effectiveness of the approach on multiple datasets and models of different scales.

### Strengths
1. A new form of in-context learning is proposed to extend the existing problem to realistic scenarios with task mismatch and noisy labels. 
2. The empirical coverage is extensive, and the measurements in this paper cover multiple LLMs and multi-task sets.

### Weaknesses
1. The definition of adaptive scenarios is not clear. The paper proposes Indirect ICL as a new task setting, but it is not clear under what “adaptive scenarios” this setting is more meaningful. For example, MoT (Mixture of Tasks) and Noisy ICL are defined, but the paper lacks explanations of the motivation, boundaries of use, and typical applications of these contexts in real systems. 
2. Lack of performance on closed-source models. One of the core values of in-context learning is “cross-model generalization”, which is especially important on closed-source models (e.g., GPT-4, Claude, etc.). However, the experiments in this paper are based exclusively on open-source models, and there is no report on the performance or feasibility of migration to closed-source models. 
3. Inadequate evaluation of computational complexity and scalability. The paper discusses the computational complexity of the influence function in the method section, but does not quantify its growing trend with real data sizes. Does the overhead of influence function estimation grow linearly and quadratically as the set of candidate samples increases (e.g., from thousands to millions)? The experimental table only shows results for the fixed-size task and does not reflect this scalability analysis. 
4. Ethodological design and motivation are inadequately articulated. The authors present two example selection approaches, but do not explicitly state whether both are for different problems (e.g., filtering noise vs. weight balancing) or two approaches for a single problem. 
5. IF theoretical assumptions do not match the LLM. The main text recognizes that the classical derivation of IF relies on assumptions such as second-order derivability and strong convexity, which are not satisfied by the deeply nonconvex LLM. The authors rely on experimental validity, but the theoretical shortcomings remain.

### Questions
Please examine the weaknesses.

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
3

### Summary
This paper "introduces" (although I don't think I was able to find a formal definition) a task called Indirect In-Context Learning, where in-context demonstrations may come from heterogeneous tasks or contain noise, and analyzes the role of influence functions (IFs) as example selectors in this setting. It analyses both surrogate models and pretrained gradient variants (DataInf, TracIn, LiSSA). Authors show that, when combined with BERTScore or cosine similarity, IFs show very marginal gains on task mixtures and improvements under noisy GLUE prompts and adversarial robustness.

### Strengths
- Indirect ICL is (I think, since it's not formally introduced in the paper; I'm considering it as a mixture of mixtures of tasks and noisy supervision) a task that reflects many real-world use cases of LLMs; this paper is timely and potentially very useful to many practitioners in the field
- Using surrogate DataInf with BERTScore yields the best average accuracy, improving over standard ICL baselines on LLama2, Zephyr, and Mistral
- IF-based averaging also improves accuracy on noisy MRPC/QQP/etc. and improves robustness in backdoor scenarios

### Weaknesses
- There are some sample selection baselines in recent literature (e.g. MoICL, from ACL'25: https://arxiv.org/abs/2411.02830) that have not been compared to the proposed approach
- Are the reported gains statistically significant? At times they seem fairly small
- Surrogate approach still requires fine-tuning on the candidate pool -- what's the computational footprint of the approach in that case?
- Influence scores rely on labeled validation pairs; what's the process for obtaining those in the "unknown task" setting?

### Questions
- What's the computational cost/runtime for building surrogate-based IF tables on the task mixture?
- Could you check the statistical significance of your results?

### Soundness
3

### Presentation
3

### Contribution
3
