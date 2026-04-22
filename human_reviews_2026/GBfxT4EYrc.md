# Mitigating Strategy-Selection Bias in Reasoning for More Effective Test-Time Scaling

- Avg Score: 4.40
- Decision: Reject
- Scores: 4, 8, 2, 2, 6

## Abstract
Test-time scaling (TTS) has been shown to improve the performance of large language models (LLMs) by sampling and aggregating diverse reasoning paths. However, existing research has overlooked a critical issue: selection bias of reasoning strategies during scaling. 
Specifically, when generating reasoning processes, LLMs tend to follow certain strategies (e.g., algebraic solutions for math problems) while neglecting other valid alternatives (e.g., geometric solutions), resulting in insufficient exploration of the solution space. To further understand the impact of this bias, we present a theoretical analysis that reveals when it undermines the effectiveness of test-time scaling. Motivated by this theoretical insight, we introduce TTS-Uniform, a framework designed to mitigate the selection bias of reasoning strategies. It (i) identifies potential strategies, (ii) uniformly allocates the sampling budget across them, and (iii) filters out unstable strategies prior to aggregation. Experimental results show that TTS-Uniform significantly enhances scaling effectiveness across multiple mainstream LLMs and benchmark datasets.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors first identify a strategy-selection bias issue in current test-time scaling paradigm, where the reasoning model tend to focus on a certain strategies while ignoring other valid alternatives, leading to insufficient explorations in the entire reasoning space. Then, the authors find that results obtained under low-complexity strategies are more reliable. Based on this insight, the authors propose TTS-Uniform, an effective strategy that first extracts relevant strategies for the given problem, then enables the model reason under different strategies with same reasoning budget, and finally aggregates the final answer after filtering results under high-complexity strategies based on the answer entropy metric.

### Strengths
(S1) The paper is generally well-written, and the structure is clear.

(S2) The paper provides some new insights on more effective test-time scaling. The strategy-bias issue is reasonable and can affect the test-time scaling performance under current settings.

(S3) The analysis is well supported by the empirical validations.

(S4) The experimental results validate the effectiveness of the proposed method.

### Weaknesses
I have some concerns and questions:

(W1) The experimental models are very limited. The authors only use GPT-4o/4.1-mini for the preliminary and main experiments. This makes me wonder if the experimental phenomena are only present in these two models. I suggest that the authors conduct more validation experiments on a wider range of models, especially open-source models (such as Qwen, LLaMA, etc.). Additionally, the experiments should be conducted on larger test sets to ensure the reliability of the results, as the sample quantity in AIME24 and AIME25 are quite insufficient.

(W2)  I am very confused about the definition and use of Assumption 3.2. Assumption 3.2 defines a higher-complexity strategy as one that leads to more errors, yet the conclusion drawn later (Thm. 3.5) is that results obtained using a higher-complexity strategy may be more unreliable. In my view, this is circular reasoning and not a valid analysis. I hope the authors can convince me on this point. Also, Moreover, I can't understand why entropy/uncertainty can be used to represent the degree of complexity. The logic behind this is not clear.

(W3) Regarding the analysis in Line 455-457, does this mean that as the model’s capability improves, the strategy-selection bias issue will gradually be alleviated? If so, then it seems that this issue is not something that can be discussed for the long term, as it will not persist as the model evolves.

### Questions
(1) For the visualization results in Figure 1, are Strategy I and II defined based on the embedding clustering results, or are they derived from the o3 model's predictions? It looks like the former, but I would like to know the true strategy category for each point to assess the accuracy of the embedding clustering.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
TTS: Test-time scaling. Using LLM to run multiple inferences for solving a problem & then using some strategy to combine the answers to produce the most accurate possible reasoning & answer


Paper points out that the problem solving strategies that LLMs use can be either complex (using more reasoning tokens) or simple. Simpler strategies lead to more accurate response, & vice-versa. However, LLMs tend to pick one type of strategy over the others - & sometimes, they largely prefer complex ones (leading to poorer quality in those problems)

The authors first study this existing bias experimentally, and then propose a method to side-step these biases by multi-turn prompting. They first get the LLM to generate all possible strategies, then they group these into clusters, & filter out highly complex solutions and then prompt the LLM to follow the simpler strategy.

### Investigating the bias itself (Section #2):
1. Pick mathematical problems (AIME dataset) - generate 500 different reasoning traces. Paths analyzed using a (i) sentence-encoder -> clustering & (ii) clustering paths using large LLMs
2. They found that models preferred one type of reasoning paths (dominant strategy) consistently much more than the combined proportion of all other strategies,

### Complex strategy => more errors (Section #3)

- This is proved mathematically with the assumption that there’s a constant rate of error at each token generation -> more tokens result in more potential errors.


---

## Process

1. For coarse-grained strategies: Extract strategies - prompt the LLM to generate all possible “abstract solution” strategies.
2. For fine-grained: Generate independent reasoning solutions from the LLM
    - Out of these fine-grained strategies, a reasoning tree is created.
    - Equivalent “steps” of reasoning (like an op: “square both sides of the equation”) are merged on different reasoning pathways
    - Only the reasoning paths with shared sub-chains are extracted - & treated as matching solution strategies.

3. The extracted strategies are now separately used as a part of the prompt & appended to the question, to push the LLM to reason on that specific pathway
4. This previous step results in ‘N’ reasoned solutions if we started with ‘N’ strategy clusters/types
5. They filter out the highest complexity solutions using the entropy metric.

### Strengths
1. Seeking simpler strategies is a novel idea for complex LLM reasoning
2. Exhaustive research and details presented in the paper. Eg:  low cost (“coarse-grained”) and expensive (“fine-grained”) setups presented together
3. Rigorous mathematical treatment for all aspects of the problem & solution formulations
4. Great overall results across different introduced techniques on the datasets tested on
5. Impressive attention to detail, especially experimental verification of hypothesis in the Appendix “A.4”, and coverage of other necessary points in the Appendix. This allows for easy independent reproducibility of the results

### Weaknesses
1. High computational overhead. Strategy discovery requires ‘N’ separate LLM calls, followed by another M calls for the ‘M’ simplest strategies found.
2. Overall result can be highly dependent on the model’s creativity (temperature setting). We are also pushing the model to come up with all potential strategies - that is unreliable & not guaranteed to return really novel & quick techniques if there’s training data bias
    - Eg: As mentioned in the paper, text-only LLMs will likely have high bias towards algebraic solutions instead of geometric ones.
3. Needs some clarity in the main paper around how the merging of strategy steps is done (details in the Questions/Comments section)

### Questions
1. How do you combine steps in different reasoning pathways (“identical reasoning steps”)? (Sec. 4.1.2)
    - ie, how is the semantic/operation matching done for different steps in the set of strategies?
    - Is it clustering with an LLM, or do you have a well-defined algorithmic system to ensure accuracy?
2. Any hypothesis on why coarse-grained method sometimes works better than fine-grained?
3. I wonder if you experimented with other ways of filtering out the high-complexity answers? I’m not sure the entropy metric (with the way it’s formulated) is the best way. For example, some formulation involving the logprobs of the token sequence might be more intuitive?
4. What temperature setting was used in the trials? Did you run trials with different temperatures?
5. Re. Appendix “A.4 EXPERIMENTAL VALIDATION OF THEOREM 3.5”: can you please share more details on the experimental setup involved (unless I missed it?)

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper shows that chain-of-thought (CoT) reasoning for large language models under test-time scaling (TTS) can suffer from strategy-selection bias, which limits multi-path exploration and reduces diversity. Theoretical analysis indicates that favoring low-complexity policies can be beneficial, whereas favoring high-complexity policies degrades TTS performance. To address this, the authors propose **TTS-Uniform**: (i) extract a set of candidate strategies; (ii) allocate the sampling budget uniformly across them; (iii) filter high-complexity policies using the entropy of the answer distribution; and (iv) aggregate the final predictions via majority voting. Experiments with GPT-family models on reasoning datasets show consistent improvements over baseline methods.

### Strengths
1. The paper highlights that strategy bias in the selection of reasoning paths during multi-path sampling is a key factor constraining path diversity.

2. It presents a fairly comprehensive theoretical analysis that characterizes how low- and high-complexity strategies affect model performance under test-time scaling (TTS).

### Weaknesses
1. **The approach introduces substantial additional overhead.** Both the coarse-grained and fine-grained variants depend on external (often stronger) models to extract strategies for every query (Q), which is impractical for real-world deployment.
2. **There is a gap between theory and method.** The theoretical analysis relies on the “Minimum Token Requirement of the Strategy” to derive conclusions about complexity and TTS performance, whereas the implementation uses entropy as a surrogate (owing to redundancy in CoT traces). Entropy is not within the prior theoretical framework, which considerably weakens the method’s theoretical underpinnings.
3. The entire empirical study is conducted on two closed-source GPT-series models, without using commonly adopted TTS research backbones such as the Qwen family. This raises concerns about the confidence and credibility of the experimental conclusions.
4. The experimental coverage is limited to a small set of mathematical reasoning datasets; key benchmarks like MATH are missing, and no general-purpose reasoning datasets are included, making it difficult to assess the method’s generalization.

### Questions
1. Multipath sampling faces a fundamental trade off between diversity and accuracy. For a given query Q with several plausible strategies, does distributing samples across them dilute the confidence assigned to the correct reasoning path? More extensive hyperparameter ablation studies are suggested to strengthen the empirical credibility of the results.
2. If concept level strategies are extracted solely through prompting, how can we ensure redundancy or strong correlation among the strategies? Does this practice shift policy bias from the target model to the external extractor model, with the latter’s bias merely being more favorable for solving the task rather than truly mitigating bias?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Sampling-and-aggregation–based test-time scaling (TTS) methods suffer from substantial strategy-selection bias, which hampers efficient exploration of the solution space. This paper analyzes the impact of this bias on TTS effectiveness and introduces TTS-Uniform, a framework that (i) explicitly extracts candidate strategies, (ii) allocates the sampling budget uniformly across strategies, and (iii) filters potentially unstable or overly complex strategies via answer-entropy. Experiments on AQuA and AIME-2024/2025, using two OpenAI small models, show consistent improvements over standard TTS baselines.

### Strengths
1. **Clear and intuitive problem formulation.** The paper formalizes “strategies” via equivalence classes and empirically demonstrates strategy-selection bias through clustering and visualizations, offering an intuitive lens for analyzing TTS.
2. **Simple yet effective approach.** Uniform budget allocation combined with answer-entropy–based pruning is straightforward to implement and yields more pronounced improvements on weaker models.
3. **Insightful empirical analysis.** The finding that weaker models tend to exhibit stronger bias and consequently benefit more which is coherently explained using distribution-level comparisons of expected error rates; the argument is simple but illuminating.

### Weaknesses
1. **Strong yet insufficiently validated core assumption.** The method treats the *minimum token requirement* as a complexity measure and assumes a monotone positive correlation with error rate; however, the evidence provided is largely intuitive.

2. **Overly general theoretical result.** Under monotonicity, *Theorem 3.5* effectively reduces to a first-order stochastic dominance (FOSD) statement. The analysis does not formally connect majority-vote aggregation or entropy-based pruning to the resulting Acc@k. Recommended:

   * Derive lower/upper bounds for majority-vote accuracy under with/without bias and dependent/independent error assumptions when using uniform sampling + entropy pruning;
   * Characterize how intra-strategy versus inter-strategy error correlation modulates voting gains.

3. **Questionable use of entropy as a complexity proxy.** High answer entropy may stem from task ambiguity or a larger output space rather than intrinsically “complex” strategies. Please add (i) statistical experiments, (ii) comparisons against other proxy choices (e.g., minimal provable path-length estimates, step consistency), and (iii) an expanded discussion of the hyperparameter n.

4. **Methodological detail gaps.** (i) Uniform-F requires explicit specification of the node-similarity metric, merge threshold, and denoising rules, as well as their effects on the number of strategies (m) and coverage; (ii) provide a sensitivity/cost analysis for hyperparameters—total budget (b), pre-sampling (t, i.e., strategy-set size), and the number of discarded strategies (n).

5. **Limited experimental breadth and diagnostics.** (i) Baselines: include recent representative TTS methods (e.g., *DeepConf*, etc.) to strengthen the claims; (ii) Benchmarks: extend beyond a small set of math reasoning tasks to HMMT/AMC and code/symbolic reasoning to assess generality; (iii) Diagnosing strategy-selection bias: beyond the cluster distribution in Fig. 2, report relationships to complexity and accuracy, and run rigorous ablations to isolate true gains from strategy diversity.

### Questions
1. To what extent does the assumed complexity–error monotonicity generalize across models and tasks? Please provide quantitative evidence across model families and multiple benchmarks.
2. How is the entropy-pruning threshold determined?
3. For Uniform-F, what are the explicit rules for selection, expansion, simulation, and backpropagation?
4. How robust are the bias estimates to the choice of evaluator model and the target model under evaluation? Do the conclusions change under alternative evaluators or targets?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses a critical but underexplored issue in test-time scaling (TTS) for large language models (LLMs): strategy-selection bias. The authors formalize this bias, show how it undermines TTS effectiveness, and propose a novel framework, TTS-Uniform, to mitigate it. TTS-Uniform extracts potential reasoning strategies, uniformly allocates sampling budgets across them, and filters out unstable strategies based on entropy. The method is evaluated on AQuA and AIME datasets with GPT-4o-mini and GPT-4.1-mini, showing consistent improvements over baselines like Self-Consistency.

### Strengths
1. Novel and Timely Problem Formulation:

The paper is the first to formally define and analyze strategy-selection bias in CoT reasoning. This is a significant contribution, as it highlights a fundamental limitation of current TTS methods that assume uniform strategy exploration, which is often violated in practice.

2. Strong Theoretical Foundation:

The authors provide a rigorous theoretical analysis of how strategy bias affects TTS performance. They introduce complexity-based strategy partitioning and use stochastic dominance to prove when biased sampling helps or hurts. This bridges the gap between empirical observations and theoretical understanding.

3. Practical and Effective Framework (TTS-Uniform):

The proposed TTS-Uniform framework is intuitive, modular, and empirically validated. It improves performance across multiple models and datasets, especially for weaker LLMs. The entropy-based filtering is a clever and lightweight way to estimate strategy stability without external supervision.

### Weaknesses
1. Limited Scope of Strategy Extraction:

While the paper proposes both coarse- and fine-grained strategy extraction, the methods still rely on LLM-generated strategies, which may inherit the same biases the paper aims to mitigate. There is no discussion of failure cases where the LLM fails to identify valid or diverse strategies.

2. Evaluation Limited to Math Reasoning Tasks:

The empirical evaluation is restricted to mathematical reasoning benchmarks (AQuA, AIME). While these are challenging, they may not fully represent the diversity of reasoning tasks where strategy bias could occur (e.g., commonsense, symbolic, or multimodal reasoning). Broader evaluation would strengthen generalizability claims.

3. Lack of Ablation on Budget Allocation:

The paper assumes uniform budget allocation across strategies is optimal, but does not explore adaptive or dynamic allocation strategies. It would be insightful to compare uniform allocation with alternatives (e.g., uncertainty-weighted or performance-based allocation) to validate this design choice.

### Questions
see Weakness.

### Soundness
3

### Presentation
3

### Contribution
3
