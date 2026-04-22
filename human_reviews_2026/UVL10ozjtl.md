# Diversity-aware Training for Test-time Scaling

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 2, 6, 2

## Abstract
Test-time scaling for large language models (LLMs) is a recognized effective approach to improving performance. However, when increasing test-time computation, the performance gains grow progressively smaller. This is largely due to the tendency of independent reasoning attempts to collapse into similar incorrect solutions. Existing approaches to enhancing reasoning diversity mainly focus on token-level diversity, which fail to capture reasoning-level diversity and introduce hallucinations. To this end, we introduce RePrism, a novel framework designed to act like a Reasoning Prism, guiding models to explore a spectrum of distinct and valid reasoning paths from a single input. First, we construct training data where each prompt is associated with multiple diverse yet correct answers. Additionally, we introduce noise embeddings into special tokens as implicit diversity signals, teaching the model to recognize these embeddings as indicators of diverse reasoning paths. We validate RePrism on 9 challenging benchmarks across Math, Code, and Agent tasks, where it increases the models' pass@N accuracy by up to 6.4\%, 1.1\%, and 0.5\% on Math, Code, and Agent tasks, respectively. Moreover, we demonstrate that the reasoning diversity instilled by RePrism provides a superior foundation for Reinforcement Learning (RL). RePrism not only furnishes a richer exploration space that leads to enhanced performance gain from RL, but also prevents the collapse of reasoning diversity during RL training.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces REPRISM, a training framework designed to improve the test-time sampling efficiency of LLMs by addressing the issue of diminishing returns caused by a lack of diversity in generated solutions. The method trains models to produce a diverse set of correct reasoning paths using two core components: (1) a specialized training dataset, curated with submodular optimization, that contains multiple diverse and correct solutions for each prompt, and (2) the injection of Gaussian noise into special token embeddings to act as an implicit control signal for diversity. The authors validate REPRISM across 9 math, code, and agent benchmarks, demonstrating minor improvements in pass@k accuracy.

### Strengths
1. Extensive empirical studies are done for an in-depth understanding of the proposed method.
2. Figures/tables are clear and well designed

### Weaknesses
This paper has numerous weaknesses. Notably, (1) the empirical results are concerning, and (2) the main contribution of this paper is unclear as having more diverse data (quality), and more data (quantity) is already well known in the literature. Noise injection has also been studied extensively.

1. Empirical gains stated in the abstract are minimal, and could very well be from noise

> We validate REPRISM on 9 challenging benchmarks across Math, Code, and Agent tasks, where it increases the models’ pass@N accuracy by up to 6.4%, 1.1%, and 0.5% on Math, Code, and Agent tasks, respectively

2. Significant concern on empirical results. When we deploy a model in the real-world pass@K only works on queries (x) with verifiable answer (y). Therefore, real-world performance is most closely proxied by pass@1. Consider pass@1 results in fig. 1 where the proposed method commonly performs worse than regular SFT. To show that it can be deployed in the real-world for any potential query from a user, it would have been  better to show self-consistency (majority voting) results over pass@k

3. Experiments might be incomplete. Why is Fig 1’s first curve missing data points at x = 128, 256. Why does the last curve (Agent) have no post-RL results? 

4. Empirical gains could be from an increased train data set size in terms of total tokens. Authors should provide clear comparison experiments.

5. Are the authors sure that there are no additional baselines? Their only baseline is SFT

6. Figure descriptions are commonly unclear. e.g.

Fig. 1 it is more clear to have said comparison of SFT and RePrism as displayed in the order in the image

> Figure 1: (Left) Comparison of REPRISM and SFT

and some figures/tables would help to have more description. 

7. Figure 1 is missing information: no x-axis label is provided. 

8. Typos should be fixed. E.g. Fig. 1 agerage → average

9. Writing needs revision. Why does the paper sometime use pass@k and pass@N, arent these referring to the same thing? Let me know if i am mistaken.

10. Writing is overselling the objective empirical results. E.g.

> Beyond improving test-time sampling, we find that the diversity fostered by REPRISM serves as a powerful catalyst for RL.

>  exceptional foundation for RL

The empirical gains are too minor to call it a “powerful catalyst” and “exceptional foundation”

11. Human evaluations do not have IRB approval or other information related to how it was set-up. This can raise ethical concerns like lack of appropriate compensation.

12. Hyperparameters (line 241) seem arbitrary. Why pick these? Are results sensitive to hyperparameters? Because this can be a huge headache for practitioners to tune this hyperparameter.

13. The experiments only use a single LLM (Llama-3.1-8B) making generalization to other model families unclear. This is concerning as there are other highly popular open-src model families like Qwen series

### Questions
1. There are a lot of noise injection related literature that improves performance like NEFTune: Noisy Embeddings Improve Instruction Finetuning. Are these not relevant baselines?

2. What is the cost overhead of this method, and how does its overheads compare to existing methods? This seems to be a critical discussion point that i can not find (or have missed)

could be from data curation, or compute, or other economic costs. This should be discussed to get a clear view of the contribution.

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
5

### Summary
The paper propose REPRISM, a novel algorithm used on LLM training with SFT and RL to enchance the generation diversity of LLMs by adding noise on embedding space. The algorithm is tested on Llama on math, coding and agent task to show the proposed algorithm can provide a benefit compared to the baselines compared.

### Strengths
- The problem of increasing generation diversity in LLM is important.

### Weaknesses
- The clarity of the paper can be improved. While the paper include many mathematical notations, a lot of them lack a definition and is unclear what is exactly referring to in the context. Please refer to my questions for some examples.
- The idea is not completely novel, and the related works are greatly missing. Similar idea of manupilating latent space has been explored [1, 2], improving diversity in LLM training without prompt have been explored [3]. 
- The experiments are on Llama only. More recent models need to be tested.

[1]. Geiping J, McLeish S, Jain N, et al. Scaling up test-time compute with latent reasoning: A recurrent depth approach[J]. arXiv preprint arXiv:2502.05171, 2025.

[2]. Zhang Z, He X, Yan W, et al. Soft thinking: Unlocking the reasoning potential of llms in continuous concept space[J]. arXiv preprint arXiv:2505.15778, 2025.

[3]. Chen W, Zhang Z, Liu G, et al. Flaming-hot Initiation with Regular Execution Sampling for Large Language Models[C]//Findings of the Association for Computational Linguistics: NAACL 2025. 2025: 7118-7127.

### Questions
The current paper needs significant improvement in all dimensions mentioned in the weakness section. Especially, additional experiments on more models and compared to different baselines are needed.

Specifically for clarity:
1. What is L? iIs it the number of propmts per batch? Or length of prefix?
2. Is the noise sequence $n$ pre-defined and fixed throughout different prompts?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes REPRISM, a training-time framework to improve test-time scaling of LLMs by increasing reasoning-level diversity. It (1) constructs per-prompt training sets with multiple diverse-but-correct solutions via submodular sampling, (2) injects Gaussian noise into selected special-token embeddings as an implicit diversity signal during training and inference, and (3) provides theoretical analysis of stability and the training objective decomposition. Experiments on 9 benchmarks show improved sampling efficiency / pass@N gains and more robust RL fine-tuning behaviour.

### Strengths
1. Novel integration of submodular optimization for diverse data sampling and noise embeddings for implicit diversity control;
2. Comprehensive evaluation across 9 benchmarks in math, code, and agent tasks, with consistent improvements in Pass@N;
3. Theoretical analysis showing noise injection acts as regularization and enables controlled diversity;
4. Demonstrates enhanced RL performance and prevents diversity collapse.

### Weaknesses
1. Insufficient Reproducibility: Key details are missing, including the random seed, data partitioning, specific rules for selecting special tokens, and explicit settings for the number of sampling trajectories m per prompt (the appendix only lists some hyperparameters).
2. Insufficient Ablation Experiments: Although several variants are reported, they do not adequately demonstrate the contributions of different components across various datasets and sampling budgets k.
3. Insufficient Robustness Analysis: While pass@1 decreases in some cases, there is a lack of detailed analysis regarding the failure scenarios.
4. Strong Dependence on Feature Representations: The approach relies on n-gram/AST features, which may limit sampling quality if these features fail to sufficiently capture the diversity of reasoning.
5. Limited Manual Annotation Scale: Table 1 includes annotations for only 32 queries, indicating a relatively small sample size.

### Questions
1. Candidate set construction: for each prompt you mention collecting up to m trajectories. What is the typical m used per dataset / per prompt in experiments, and how sensitive are results to m?
2. Special tokens / noise schedule: which tokens are considered “special tokens” and how were their positions chosen? Is σ=0.001 fixed for all experiments — did you sweep σ and report sensitivity? 
3. Reproducibility: authors state code is in supplementary—will the code include the full data-sampling / feature extraction (n-gram, AST) pipeline, LLM prompts, and random seeds? Appendix references but please clarify. 
4. Negative impact cases: can you provide example instances where REPRISM reduces pass@1 and analyze why (e.g., overly promoting exploratory but low-probability valid solutions)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work presents two complementary strategies for enhancing inference performance in mathematical reasoning, code generation, and agent-based decision tasks. The first leverages submodular optimization to promote diversity among generated outputs, while the second improves model robustness by injecting noise into the input embeddings during training. Empirical results demonstrate consistent gains in accuracy.

### Strengths
1. The method employs a principled submodular approach to select a diverse set of trajectories.
2. Injecting noise into the input embeddings appears to yield practical performance improvements.

### Weaknesses
1. There is no significant improvement in pass@N performance for math and code. Standard deviation numbers are not given in those experiments.
2. While n-gram–based diversity metrics are standard for identifying redundant text, they do not necessarily reflect the deeper logical diversity present across trajectories.
3. The inherently sequential nature of submodular optimization may limit the scalability of this approach. This may limit potential gains on more complex tasks.
4. Injecting noise into embeddings also requires applying noise at inference, which may be undesirable. A more suitable approach may be needed.

### Questions
1. What is the ceiling performance when you scale up the inference compute / your sampling budget is around infinite?
2. While n-gram diversity may not capture the underlying logical structure of mathematical reasoning, can you provide intuition for why a bag-of-n-grams representation is still a reasonable proxy for diversity among solution trajectories?
3. What is the baseline in Table 1? I.e. randomly sample 4 outputs w/o using submodularity?
4. Can you provide the algorithmic complexity of sumbodular optimization? Assuming you are selecting K trajectories, each with N bag-of-words?
5. It is not entirely clear why perturbing the embeddings during training improves performance. Is the effect primarily due to the regularization it introduces?
6. If perturbing the embeddings during training improves performance, then based on the theory in (6), would applying explicit regularization achieve a similar effect?

### Soundness
2

### Presentation
2

### Contribution
1
