# Improving Tool Calling Accuracy for Large Language Models

- Avg Score: 1.00
- Decision: Reject
- Scores: 0, 2, 2, 0

## Abstract
We introduce a novel method for improving LLM tool calling accuracy. Our approach uses a template-based generation instead of existing schema-constrained generation. Experiments on different datasets and LLM models demonstrate that our method improves F1 scores for tool names and parameters on most tests.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
Considering that the paper is clearly incomplete, I believe it should be strongly rejected.

### Strengths
Provides limited empirical comparison across models and datasets.

### Weaknesses
1. The paper is incomplete and lacks depth; it reads like an early-stage draft.
2. The idea—replacing structured JSON with a natural-language template—is trivial and does not constitute a substantive research contribution.

### Questions
None

### Soundness
1

### Presentation
1

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
This paper proposes a template-based approach to tool calling for LLMs: instead of emitting JSON or another schema-constrained format, the model produces natural-language templates which are then parsed into structured calls. Experiments on API-Bank, ToolACE, and When2Call with several models suggest gains on many F1 metrics.

### Strengths
Schema brittleness is a real source of failure in function calling. Demonstrating fewer schema violations via NL templates is a useful angle.

### Weaknesses
1. The core technique appears to be a prompt swap followed by regex parsing. There’s little discussion of design choices, alternative templating schemes, or robust parsing. Relative to the breadth of tooling literature, the contribution as framed feels incremental.
---
2. This paper’s writing quality is poor. It contains almost no discussion of related work, gives only a cursory treatment of the core method, devotes much of the main text to unstructured result listings with visible whitespace, and does not provide implementation details.
---
3.This paper asserts that templates align better with NL pretraining, but does not provide mechanistic or error-mode evidence beyond counts such as which violations decline.

### Questions
API-Bank L3 or ToolACE multi-call dialogs stress plan quality. Do templates help with tool selection across turns, or only with single-shot formatting?

### Soundness
3

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a template-based generation method to improve LLM tool-calling accuracy, replacing conventional schema-constrained outputs with natural language templates. The approach is evaluated across three datasets and four LLMs . Results show consistent F1 improvements for tool names/parameters in most tests. The method leverages LLMs' natural language alignment, reducing schema violations and enhancing contextual understanding. However, performance varies with model architecture.

### Strengths
(1)Addresses a limitation of schema-constrained tool calling (misalignment with LLMs' natural language training).

(2)Template-based generation is simple yet effective, offering a practical alternative to rigid schemas.

(3)Tests diverse models and uses multiple datasets with varying complexity. Includes statistical significance testing.

### Weaknesses
(1)The paper argues that template-based generation is superior to schema-constrained generation, based on the assumption that the template-based format is more closely aligned with natural language. However, these assumptions and views lack both in-depth theoretical analysis and rigorous experimental verification.

(2)Calculating the Macro F1 score separately for the tool name and tool parameters in the experiment seems inappropriate, as incorrect names or incorrect parameters can lead to failure. Only correct names and parameters can lead to task success. Furthermore, the 0.9 threshold set for semantic similarity is arbitrary, and different embedding models can also affect this threshold.

(3)The paper's logic and presentation lack rigorousness, and many principles, concepts, and terminology lack prior explanation and clarification. In some areas, the description is incomplete and unclear.

### Questions
(1)Why is template-based generation better than schema-constrained generation? The paper concludes that "Current LLMs are predominantly pretrained and fine-tuned for schema-constrained generation (e.g., JSON output)."(line 383) Based on this, schema-constrained generation should be better.

(2)What's the rationale for setting the semantic similarity threshold at 0.9?

(3)Are there better metrics for measuring tool calling accuracy? The F1 score doesn't directly reflect the success or failure of tool calling.

(4) (Table 2 and Table 3) Why does GPT5 perform comparable to or better than other models on the API-Bank L1 and When2Call datasets, regardless of whether it's a schema-constrained or template-based method, but perform worse than almost all other models on the API-Bank L2 and ToolAce datasets?

(5)Why not compare with other tool calling studies? After all, simply relying on improvements to prompt format and style isn't necessarily an effective approach.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper proposes to generate tool calls from models by using a natural language template rather than a json-like schema. It is shown to be better than using a json-like schema in some scenarios.

### Strengths
N/A.

### Weaknesses
1. Lack of comparison with chat templates that models are trained in. LLMs nowadays are trained to use tools following a certain template (see https://huggingface.co/docs/transformers/main/en/chat_extras). This can either be achieved using chat templates for open source LLMs or using tool call APIs for closed LLMs (https://platform.openai.com/docs/guides/function-calling). Since models are trained this way, it is only fair to compare with them in this setting, instead of directly putting the tool schemas in the prompt.

2. Lack of comparison with constrained decoding methods. Lots of the errors spotted by the authors such as schema violation, incorrect tool name, incorrect parameter name, can be avoided with constrained decoding, because it limits the vocabulary to only the valid tokens during a tool call. Constrained decoding/structured output has been implemented by major closed LLMs (https://platform.openai.com/docs/guides/structured-outputs) and open-source inference engines such as sglang (https://docs.sglang.ai/advanced_features/structured_outputs.html). Hence, a major part of the problem that this paper is solving no longer exists.

3. Marginal improvements over the baseline. In Table 2 and Table 3, many scenarios show worse results for template-based tool calling than schema-based. I'm not sure if this method is only working for certain scenarios, as there is no motivation or justification behind its effectiveness.

### Questions
see weakness.

### Soundness
1

### Presentation
1

### Contribution
1
