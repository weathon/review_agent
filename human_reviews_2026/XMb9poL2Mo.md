# Decoupling Task-Solving and Output Formatting in LLM Generation

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 4

## Abstract
Large language models (LLMs) are increasingly adept at following instructions containing task descriptions to solve complex problems, such as mathematical reasoning and automatic evaluation (LLM-as-a-Judge). However, as prompts grow more complex, models often struggle to adhere to all instructions. This difficulty is especially common when instructive prompts intertwine reasoning directives—specifying what the model should solve—with rigid formatting requirements that dictate how the solution must be presented. The entanglement creates competing goals for the model, suggesting that more explicit separation of these two aspects could lead to improved performance. To this front, we introduce Deco-G, a decoding framework that explicitly decouples format adherence from task solving. Deco-G handles format compliance with a separate tractable probabilistic model (TPM), while prompts LLMs with only task instructions. At each decoding step, Deco-G combines next token probabilities from the LLM with the TPM calculated format compliance likelihood to form the output probability. To make this approach both practical and scalable for modern instruction-tuned LLMs, we introduce three key innovations: instruction-aware distillation, a flexible trie-building algorithm, and HMM state pruning for computational efficiency. We demonstrate the effectiveness of Deco-G across a wide range of tasks with diverse format requirements, including mathematical reasoning, LLM-as-a-judge, and event argument extraction. Overall, our approach yields 1.0\% to 6.0\% relative gain over regular prompting practice with guaranteed format compliance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates the problem of performance degradation in large language models (LLMs) when they are prompted to both perform a task (e.g., mathematical reasoning) and format their output simultaneously. To address this, the authors propose Deco-G, a framework based on a Hidden Markov Model (HMM) that estimates a model’s adherence to the required output format and adjusts its output distribution accordingly based on this feedback. The problem investigated is generally interesting. However, several concerns still remain that are summarized in the weaknessess section.

### Strengths
- The problem is interesting to discuss as model format consistency is a very important aspect when dealing with LLMs.
- The problem is well presented, and the paper is generally easy to follow.

### Weaknesses
- **Significance of the issue:** First, the paper does not clearly establish how significant the issue of formatting-induced degradation is in practice. For example, as seen in the evaluation by the authors, many models in the 7B range achieve near-perfect formatting accuracy (e.g., 99.7% compliance in LLM as a judge task by the Qwen model with formatting typically very close to 100 as seen in Table I). This suggests that newer and larger models may have largely overcome this formatting limitation. A consistent benchmarking of perhaps smaller and bigger models would make the picture clearer on that front (especially recent models).

- **Evaluation:** The evaluation by the authors focus on single prompt answer and following by the model. One intutiive way to deal with such formatting/accuracy problem is to allow the model first to provide an unrestricted (e.g., no formatting prompt) answer to the question. Then, through a second prompt, use the provided answer and the history to format the answer in the desired way. Therefore, such multi-steps prompting should also be evaluated against Deco-G given that the latter will require separate training samples and training to get good graps of the relationship between the formatting the distribution of the tokens. 

- **Reasoning models:** It will be good to also evaluate reasoning models on the considered tasks to see if the reasoning traces by the models mitigates the behavior of accuracy/formatting issue that standard instruction-tuned LLMs such as Llama go through. 

Minor typo: equation 2 is repeated twice at the top of page 5. 

Code: It would be beneficial for the paper also to provide the code implementation

### Questions
Please see the Weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents Deco-G, a framework for ensuring structured output of LLM generations. It differs from existing work by using an HMM trained on LLM generations to predict a control signal during decoding that is multiplied with the LLM's token probabilities. The framework employs HMM hidden state pruning as an optimization to reduce the inference latency.

### Strengths
- The paper presents a solution to the observation in recent works that enforcing output formatting decreases LLM capabilities. This is a problem that has great impact as enforcing output format (especially JSON formatting) is the common and recommended usage of LLMs.
- The work proposes a trie-algorithm and HMM state pruning optimization that improves the overhead from constrained generation (though the degree to how much is unclear).

### Weaknesses
- The evaluation does not present inference time results. This is valuable information when comparing the tradeoff of different decoding techniques when enforcing constraints. It is also unclear how much the HMM state pruning optimization changes in the latency. While the Appendix has some FLOPS comparison, the concrete latency numbers is still valuable metric.
- The evaluation also does not compare against using Ctrl-G to enforce format constraints. This seems like an important baseline. There's some evaluation of using Ctrl-G in Appendix C but this should be a baseline across all benchmarks in the main paper evaluation. Additionally, the paper makes the claim that "[...] using text sampled unconditionally from a model [...] fails to capture the task-oriented behavior that emerges after instruction fine-tuning". A more rigorous and thorough evaluation of the HMM training methodologies should be presented to back this claim, rather than a one-off evaluation on GSM8K. 
- The description of how the work improves over Ctrl-G is not clear. The description in Sec 2.3 is vague. It may be more beneficial to the paper to highlight the improvements Deco-G makes over Ctrl-G in Section 3, highlighting concretely what each aspect of Deco-G is from Ctrl-G and which aspect is novel. 
- It is not clear how the trie-algorithm composes with the the DFA. An example would be helpful in making the contribution easier to digest. Examples of constraints that are possible with Deco-G (because of the trie-algorithm) and not possible with Ctrl-G would also be beneficial.

### Questions
- What are the inference latency of each method on the benchmarks?
- What is the significance of the trie-algorithm? What formats can Deco-G enforce that other methods cannot because of it? Is there a computation cost to using a trie on formats that can be enforced on all methods?
- How does Deco-G compare to Ctrl-G on enforcing format constraints on the evaluation benchmarks?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes DECO-G, a decoding framework that integrates learned structural format models with LLMs to improve outputs that must follow explicit schemas (e.g., JSON). The core idea is to train instruction-conditioned Hidden Markov Models (HMMs) and flexible tries/DFA-based validators that capture structural 'regularities' in model outputs. While HMMs are used in prior work, this method trains them on instruction-resonse pairs. During inference, similar to existing methods,  DECO-G combines these learned structural priors with token-level probabilities from the LLM to guide decoding — preferring sequences that are both semantically likely and structurally valid. Experiments have been presented on GSM8K for mathematical reasoning, ACE05 for event extraction and the use in an LLM as a judge for scoring summaries of news articles demonstrating better performance than using Outlines.

### Strengths
- Clear motivation: forcing a model to handle both reasoning and strict format at once can degrade reasoning.
 - Instructions conditioned HMM training seems like a useful idea for the task

### Weaknesses
- It is unclear how well custom or non-JSON formats can be learned. The mechanism for discovering pivot and wildcard boundaries is under-described, making it difficult to assess how the model generalizes beyond standard JSON structures.

- Evaluation focuses exclusively on JSON outputs; no experiments demonstrate performance on alternative structured formats such as XML, Markdown, or list-style instructions.

- The use of the SNI dataset suggests that the diversity of instruction types and output schemas may be somewhat limited

- Key implementation details are missing, which limits reproducibility and clarity.

### Questions
1. It is not clear how the HMM is explicitly encouraged to model structure rather than surface token sequences. If the model is learning from JSON-like data with arbitrary keys and values, some preprocessing—such as parsing with a JSON parser—would presumably be required to expose structural boundaries (e.g., braces, delimiters, key–value pairs). In that case, the induced HMM and DFA might simply encode the constructs of JSON rather than learning structure in a generalizable way. How would the approach extend to formats that are not easily tokenized, such as lists, Markdown tables, XML, or other instruction-defined layouts? There are also some format following benchmarks; a clarification of why they were not considered would be helpful as well. 

2. Conversely, if no parser is used, it is unclear how the system segments raw text into pivots and wildcards without leaking task-specific information into the HMM. In that setting, wouldn’t supervised fine-tuning (SFT) on the same data serve as a natural and potentially stronger baseline?

Perhaps I am misunderstanding a few implementation details—I'd be happy to see the authors clarify these points and revise my score accordingly.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a decoding framework called DECO-G. The aim is to solve the problem that LLMs performed less well when asked to solve a certain problem (e.g., a math problem) and at the same time have format constraints. The latter is important for extracting the answer easily pragmatically. The key idea is to decouple these tasks. The framework uses a so-called traceable probabilistic model for handling formatting, whereas the input to the LLM only focuses on the task instructions. The work is based on controllable text generation (GeLaTO and CtrlG), and focuses on the problem of combining instruction-tuned LLMs with output templates. Contributions of the paper include (i) the idea of instruction-aware distillation, (ii) extending the decoding format constraints from DFA to trie, and (iii) the technique of HMM hidden state pruning. 

The approach is tested on three different kinds of tasks (math reasoning, LLM-as-a-judge, and event argument extraction).

### Strengths
- The paper describes an interesting problem of getting good accuracy while having output constraints.

- The paper is quite easy to read and well written

- The evaluation setup with the three areas of math reasoning, LLM-as-a-judge, and event argument extraction is reasonable

### Weaknesses
- It is not clear that the evaluation is comparing with the right baseline. The method compares free generation (named NL) with -S extensions using Outlines by Willard & Lauf. However, the method of outlines has not been discussed in the paper before, and the closely related methods (GeLaTo and CtrlG) have not been used as baselines.

- The technical part describing the new ideas and main contributions is only briefly described in section 3.

- Only JSON is evaluated as an output constraint. Hence, it is hard to know how generalizable the output constraint restrictions are.

- The related work section does not seem to capture new related developments, e.g., "Syntactic and Semantic Control of Large Language Models via Sequential Monte Carlo" from ICLR 2025, "Fast Controlled Generation from Language Models with Adaptive Weighted Rejection Sampling" from COLM 2025.

### Questions
- Why is the work not compared to GeLaTo and CtrlG in the experimental evaluation?

### Soundness
3

### Presentation
3

### Contribution
2
