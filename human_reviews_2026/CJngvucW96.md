# Fine-Tuned In-Context Learners

- Avg Score: 4.67
- Decision: Reject
- Scores: 6, 6, 2

## Abstract
When adapting large language models (LLMs) to a specific downstream task,
two primary approaches are commonly employed: (1) prompt engineering with
in-context few-shot learning, leveraging the model’s inherent generalization abil-
ities, and (2) fine-tuning on task-specific data, directly optimizing the model’s
parameters. While prompt-based methods excel in few-shot scenarios, their effec-
tiveness often plateaus as more data becomes available. Conversely, fine-tuning
scales well with data but may underperform when training examples are scarce.
We investigate a unified approach that bridges these two paradigms by incorpo-
rating in-context learning directly into the fine-tuning process. Specifically, we
fine-tune the model on task-specific data augmented with in-context examples,
mimicking the structure of k-shot prompts. This approach, while requiring per-
task fine-tuning, combines the sample efficiency of in-context learning with the
performance gains of fine-tuning, leading to a method that consistently matches
and often significantly exceeds both these baselines. With an emphasis on practi-
cality, we introduce a hyperparameter optimization strategy based on prequential
evaluation, which is effective in data-limited scenarios and eliminates the need for
expensive cross-validation. We conduct an extensive empirical study to investi-
gate the sample efficiency of fine-tuning, in-context learning, and the proposed
unified approach across a diverse range of downstream tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes ICL+FT, a unified adaptation method that fine-tunes LLMs with k-shot in-context (x, y) examples and then uses the in-context pairs again at inference. It selects hyperparameters via prequential next-step evaluation, eliminating the need for a held-out dev set. Across Gemma-2 model sizes and several benchmarks (e.g., BBH), ICL+FT consistently matches or outperforms ICL-only and FT-only baselines.

### Strengths
1. The paper is well structured and easy to follow.
2. Using prequential evaluation to bypass cross-validation significantly reduces the cost of hyperparameter selection.
3. Results span multiple datasets, including 23 BBH tasks, an NLP task suite, Parity-20, and FLoRes. This provides strong evidence that ICL+FT delivers gains over the ICL-only and FT-only baselines.

### Weaknesses
1. Section 3 states that prequential selection is computationally efficient, but the paper lacks explicit runtime or FLOP comparisons against CT-only, FT-only, and a simple hold-out cross-validation baseline. Concrete wall-clock measurements would help substantiate the efficiency claim.
2. Beyond ICL-only and FT-only, comparisons to other adaptation methods like prefix tuning, prompt tuning, and context tuning are missing. These would help clarify whether ICL+FT is a state-of-the-art method in performance and/or efficiency.

### Questions
1. How sensitive is ICL+FT to the order of training examples in Algorithm 1? Do different permutations lead to significantly different hyperparameters or final performance? How does this sensitivity compare to standard ICL-only?
2. Have you evaluated ICL+FT on models outside of the Gemma family? For example, Llama and Qwen are also open-source model families with varying model sizes.

### Soundness
4

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
4

### Summary
The authors consider fine-tuning in a low-data regime. They introduce a method to fine-tune on $k$ in-context examples concatenated together with each training example and additionally design a hyperparameter selection method which is better adapted to low-data contexts. They study both their training approach and hyperparameter selection method in a variety of low-data tasks, such as Big Bench Hard and low-resource translation, among others. They compare their fine-tuning method against both fine-tuning alone as well as in-context learning alone. To study their hyperparameter selection method, they compare its performance against hyperparameter selection using an i.i.d. evaluation set as well as against using a fixed global set of hyperparameters.

### Strengths
The prequential hyperparameter selection algorithm is is original and novel. There is some prior work on fine-tuning with few-shot examples (some cited as well as other recent work [1]), but the contribution on this front is still novel. The paper is well-written and figures are clear.

[1] Lu, Jack, Ryan Teehan, Zhenbang Yang, and Mengye Ren. "Context Tuning for In-Context Optimization." https://arxiv.org/abs/2507.04221

### Weaknesses
The claim, "We emphasize that the prequential training and evaluation protocol described in Section ?? does not necessitate a separate held-out set, allowing practitioners to utilize all data points for training" is too strong and not justified by the paper. At best, the paper seems to indicate that, in the low-data regime, we do not need a separate test set for hyperparameter tuning specifically. If we want to assess overfitting and generalization, we would still need a held-out test set. 

Some comparisons with baselines do not seem precisely 1-1.

Other comments:

There is a broken section reference on page 5 in the Big Bench Hard paragraph and a broken citation on page 14.

### Questions
Can you explain this point: "Note that globally-chosen hyper-parameters introduce information leakage as a large number of test-set examples are used to chose these"? Wouldn't the evaluation set still be held-out?

Does your FT baseline also take multiple gradient steps per example? 

How do you account for the fact that the ICL+FT setting has seen some datapoints multiple times (because they later appear as ICL examples for future training steps)? Do your baselines see each example the same number of times as your method does?

### Soundness
4

### Presentation
3

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
The paper proposes a unified approach that combines in-context learning (ICL) and fine-tuning (FT) to improve task adaptation in large language models. Instead of using these methods separately, the model is fine-tuned on prompts that already include k-shot examples, and during inference it again receives k in-context examples. Technically, the ICL+FT method forms training sequences consisting of k demonstrations followed by the target query, and updates model parameters using the likelihood over all answer tokens. For hyperparameter tuning, the authors introduce a prequential evaluation scheme, which incrementally trains and evaluates the model without requiring a separate validation set. Experiments on various benchmarks show that ICL+FT typically matches or sometimes outperforms both ICL-only and FT-only baselines across Gemma 2 based models (2B, 9B, 27B).

### Strengths
- Clear and simple idea that is easy to implement.
- Broad empirical performance outperformed other baselines across tasks, model sizes, and data budgets.
- Useful ablations on number of in context examples, instruction prompting, and LoRA.

### Weaknesses
- **Conceptual novelty is limited and close to MetaICL style training.** Prior work on MetaICL and related meta learning frameworks already trains on k shot episodic inputs so that models learn to use in context examples. This paper differs mainly in scope, since it targets a single downstream task with task specific fine tuning rather than cross task generalization without parameter updates. The core learning signal of using in prompt examples is therefore very similar. 
- **Limited and Unbalanced Efficiency Claims:** While the paper suggests both data and computation efficiency, this claim appears overstated. The approach still requires per-task fine-tuning and utilizes k in-context examples during inference, resulting in cumulative rather than reduced compute cost. Therefore, although data efficiency may be plausible, there is no clear evidence of computational efficiency.
- **Ambiguity in training details and sensitivity.** The algorithm samples k context examples from previously seen data, but the selection strategy and order effects aren’t analyzed. (performance of ICL is known to highly sensitive to those factors.) There is no report of variance across different context selection policies or data orderings, which can be substantial for in context methods.

### Questions
- Is there a reason why the title shown on OpenReview and the title in the PDF are different?

### Soundness
3

### Presentation
3

### Contribution
1
