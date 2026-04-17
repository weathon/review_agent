# Expanding Computation Spaces of LLMs at Inference Time

- Decision: Reject
- Scores: 2, 2, 2, 2

## Abstract
Chain-of-thought (CoT) rationale enables language models to use additional task-related text for problem-solving, benefiting not only from detailed reasoning steps but also from the expanded computational space of longer inputs. Prior work has trained filler or special tokens to serve as additional computation spaces. In this study, we investigate whether language models can leverage artificially inserted sequences of filler tokens solely at inference. We first identify effective token types, numbers, and insertion locations, then examine at what stage of training models begin to exploit the expanded computation space, and finally analyze dynamics within these spaces via attention maps. Experiments on models ranging from 1.7B to 32B across open-domain QA and math tasks show that appropriate token types and counts vary, but placing filler tokens directly before the final `Answer:' token is most effective. Smaller models benefit most, up to 12.372 percentage points in SmolLM2-1.7B-Instruct, indicating that these spaces act as additional computational capacity rather than redundant input. Attention maps reveal that expanded spaces often continue the original attention mechanism and sometimes focus on questions or answer options, suggesting meaningful computation for problem-solving.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates whether large language models can leverage artificially inserted filler tokens—without any additional training—to enhance reasoning performance. Building on the intuition that chain-of-thought (CoT) prompting benefits not only from explicit reasoning steps but also from the larger “computational space” of longer inputs, the authors systematically study what kinds, amounts, and insertion locations of filler tokens are effective. Through experiments across multiple open-domain QA and math benchmarks using models from 1.7B to 32B parameters, they find that inserting familiar filler characters (e.g., spaces, newlines, dashes) directly before the final Answer: token can significantly improve accuracy, especially in smaller models (up to +12.37 percentage points). Attention-map analyses reveal that these expanded spaces participate in meaningful computation, continuing or redirecting attention toward relevant parts of the problem.

### Strengths
1. The experiments cover two categories of tasks—QA (MMLU, ARC) and mathematics (GSM8K, MATH-500)—using a range of instruction-tuned models from small to large (including Qwen2.5-32B-Instruct). Results are reported in a zero-shot setting averaged over three random seeds. For multiple-choice questions, the answer is extracted by applying a softmax over the final-layer logits of options A–D, and for math problems, the numerical answer is extracted following existing evaluation rules. The evaluation interface is fully transparent.

2. A comprehensive factor analysis is conducted: systematic ablations are performed on token type (space, newline, tab, period, <pad>, dash), quantity, and insertion position, revealing several consistent experimental patterns.

3. A preliminary set of actionable guidelines is provided, summarizing multiple practical insights with direct reference value for deployment.

### Weaknesses
1. I am confused and skeptical about the motivation of the paper. If the goal is to alleviate the time consumption caused by the lack of parallelism in CoT reasoning, then according to the experimental results reported in the paper, this method seems to save only about the generation time of 256 tokens, which appears to bring minimal benefit for long-text reasoning. If the goal is to propose an alternative to CoT, the paper lacks direct comparative experiments on the answer quality obtained by CoT.

2. The authors should add a “same token budget” comparison with CoT: the paper does not take CoT prompts with the same number of tokens as a baseline for direct comparison, making it difficult to answer the key question of whether ECS performs better (or worse) than short CoT under the same token or time cost. The current results mostly verify that “merely adding positions can bring gains,” but if ECS does not show clear advantages over CoT in some aspects, its practical value will be greatly discounted.

3. The paper has not yet summarized general, actionable insights that are independent of specific tasks.

### Questions
Could the decline in accuracy with an increasing number of tokens in the ECS be related to the inherent difficulty of the problem itself? Perhaps the additional tokens cause the model to overthink, leading it to second-guess its own reasoning.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates the impact of adding filler tokens across types, counts, and positions. It also examines when during training this phenomenon emerges. Attention-map analysis shows that these tokens focus on relevant parts of the input and help the model arrive at the answer.

### Strengths
- The paper examines the impact of expanding computation space by adding filler tokens at inference time.

- It reports observations across several factors, including model type and size, as well as token types and counts.

### Weaknesses
The overall patterns of results and explanations are not consistent across model types and sizes, as well as token types and counts.

- The explanations are sometimes inconsistent. For example, for Qwen2.5-14B vs. 32B, the paper argues that larger models benefit from filler tokens, yet for SmolLM (1.7B) it claims the additional tokens are “effectively utilized for reasoning,” suggesting a different rationale for small models. In line with this concern, more experiments are needed to substantiate the claims. 

- The paper argues that additional tokens compensate for limited parameters in small models, but it evaluates only SmolLM as the “small model.” More small models should be tested to support generalization. 

- Several reported patterns do not generalize well. For instance, the paper claims that adding more than 64 <pad> tokens degrades performance, attributing this to its rarity during training. However, repetition of other tokens may also be rare in typical training data. 

- According to Llama’s explanation of <pad> vs. <eos>, the training dataset influences these patterns. However, the impact of filler tokens in Qwen2.5-14B and 32B does not align, even though they are from the same model family.

- The checkpoint analysis shows no clear trends. Pretraining (PT) checkpoints do not improve with filler tokens, while instruction-tuning (IT) checkpoints likewise fail to show a consistent pattern of improvement across the training process. 

- The attention map analysis hardly explains why the improvements occurred. Strong attention alone does not necessarily translate into better performance. Even so, the paper should present counterexamples (e.g., cases where the model answers incorrectly) and, where possible, quantify these patterns.


Typos:
- Line 119: The citation(s) should be in parentheses “( … )”. 
- Equation (2): Should Z_{1:M} be Z_{T:M}?

### Questions
Please see the Weaknesses section.

### Soundness
1

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
4

### Summary
This paper investigates whether LLMs can leverage manually inserted filler tokens (such as <pad>, \t, \n ...) at inference time to enhance their reasoning performance. The authors conduct experiments on 6 types of filler tokens on varying sizes of LLMs, revealing even filler tokens can lead to performance variations, especially for small size LLMs.

### Strengths
1. The writing and structure of this paper are clear and easy to understand.
2. The research question is interesting. It would be attractive if we find that this approach can consistently boost the performance of LLMs through simple prompt manipulation.

### Weaknesses
1. The main weakness of this paper is that the performance of LLMs after injecting filler tokens varies. Some LLMs perform better on MMLU and ARC after injecting filler tokens, while others do not. The authors neither explain nor analyze this phenomenon (Section 4.4 reads more like a case study than a rigorous analysis), nor do they make efforts to avoid performance degradation in a training-free manner. As a result, the proposed approach appears to add random manipulation to the input prompt rather than serving as a convincing, beneficial algorithm that can consistently enhance LLM performance.
2. Although the authors show that the performance of OLMo-2 can be enhanced by filler-token fine-tuning in Section 4.3, it requires large-scale pretraining and instruction tuning, while the performance gain is marginal (~1% in Figure 5) compared to the heavy computational cost. This also somewhat deviates from the main goal of the paper, which is enhancing LLM performance by expanding computation space at inference time. Furthermore, this validation is only conducted on one LLM with one filler token, making it unclear whether the training can generalize.
3. Typo: in Line 218, GSM8K contains 1319 test set examples, not 1819.

### Questions
1. What is the training corpus in Section 4.3?

### Soundness
2

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
The work explore the computational spaces in inference time to see the impact of the inserted tokens (space, enter, tab, period, etc) and the number of such inserted tokens. The author additionally attempted to show the attention distributions to show how the LLM innerworkings get impacted by such insertions.

### Strengths
Rich experiments 
- models from SmolLM to Gemma to Llama to Qwen to OLMo. 
- good dataset selection - ARC and MMLU - that have different domain QAs.

Meaningful results
- token types (e.g., pad vs space), numbers while inferencing
- attention map (fig. 6) shows meaningful results with the high scores with filling tokens/ question-context pairs (though it might have been cherry-picked results)
- interesting that the enter token plays a meaningful role in inferring better responses in Instruct prompts settings (fig. 5) regardless of the number of the tokens.

### Weaknesses
The number of tokens
- Table 1 - seems not informative; isn't this obvious that the performance decreases if the number of tokens increase extremely a lot (e.g., 2048 or more), blocking models to use the meaningful tokens (given that current LLMs struggle with long context reasoning, this is what we can expect). Performances with the numbers below this seem to be similar (very negligible changes) c with those of the baselines (without additional tokens at all). 
- Only meaningful changes are on SmolLM (1.7B) but not sure this can be a general pattern across models.
- the OLMo performances with increased number of steps (figs. 4 and 5) don't seem to show meaningful results

The filler tokens (+ numbers)
- Fig 2 - this shows some complementary results for Table 1 in regards to the performance changes from number of tokens. One weak point is that they don't seem to have meaningful patterns that share across datasets or models? despite interesting performance changes with certain tokens such as space. and those results were pretty known findings, not limited to this work [A]. so it would have been more interesting if there were more meaningful contributions.



[A] The Butterfly Effect of Altering Prompts: How Small Changes and Jailbreaks Affect Large Language Model Performance, ACL findings 2024

### Questions
my main concern for this work is that the contribution seems weak - the number of tokens, and the type of tokens do not seem to show general patterns that adds meaningful findings to the community. Would the authors can clarify them better in case I missed something or undermined the points?

I've also expected the work clarifies how those inserted tokens expand the computational search space for the models to infer responses better - but I couldn't anything regarding this. Could you point me toward those relevant information?

### Soundness
2

### Presentation
3

### Contribution
2
