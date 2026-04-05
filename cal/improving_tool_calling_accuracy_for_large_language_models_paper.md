# IMPROVING TOOL CALLING ACCURACY FOR LARGE LAN## GUAGE MODELS


**Anonymous authors**
Paper under double-blind review


ABSTRACT


We introduce a novel method for improving LLM tool calling accuracy. Our approach
uses a template-based generation instead of existing schema-constrained generation. Experiments on different datasets and LLM models demonstrate that our method improves
F1 scores for tool names and parameters on most tests.


**Keywords:** Large Language Models, Tool Calling


1 INTRODUCTION


Currently, the most common approach to tool calling with large language models (LLMs) is to have the
model generate a schema-constrained output, such as JSON (Qu et al., 2025). However, the majority of
the data used to train LLMs is based on natural language. Even for code-generation LLMs, a significant
portion of the training data must be natural language; otherwise, the model may struggle to correctly handle requests expressed in natural language (Brown et al., 2020; Zhao et al., 2023; Petty et al., 2025). The
schema-constrained data is very different from natural language. We propose a new method for tool calling based on template-generated text, which is closer to natural language. We evaluated this new approach
through experiments on different datasets including API-Bank (Li et al., 2023), ToolACE (Liu et al., 2025)
and When2Call (Ross et al., 2025), using various LLMs such as GPT-4o (OpenAI et al., 2024), GPT-5 (OpenAI, 2025a), Mistral (MistralAI, 2024) and DeepSeek-Coder (Guo et al., 2024). This new approach shows
accuracy improvements compared to existing schema-constrained generation in most tests. All code used in
this paper is submitted as supplementary material and will be made publicly available after acceptance.


2 APPROACH


The schema-constrained generation approach asks LLMs to produce structured outputs that specify
which tools or APIs to invoke and with what arguments, such as JSON-based strings. For example,
to call the tool “ToolName” with parameter “param1” set to “value1” and “param2” set to “value2”,
the string could be: _{_ "name":"ToolName", "parameters":[ _{_ "param1":"value1" _}_,
_{_ "param2":"value2" _}_ ] _}_ . In this work, we explore a different approach, where the model is asked to
generate natural language-based strings following a template. For example: Call "ToolName" with
following parameters: "param1" as "value1", "param2" as "value2". These
strings are parsed using regular expressions and converted into a constrained tool-calling schema, enabling
a direct and consistent comparison of the two approaches against the same ground truth.


We used three public datasets: API-Bank, ToolACE and When2Call. API-Bank is organized into three
difficulty levels. Our analysis focuses on Level 1 and Level 2 instances, as Level 3 instances involve multistep planning, which is beyond the scope of this work. Level 1 instances provide all tools upfront, while
Level 2 introduces tools dynamically during the conversation, thereby increasing task complexity.


1


ToolACE is a benchmark for evaluating LLMs on structured tool usage. In this dataset, a single request
may require two or more tool calls. The released ToolACE data consists of a single dataset. We randomly
sampled 1,000 instances from the original dataset to create a test set, leaving the remaining instances as the
training set. Each full conversation in an instance involves multiple aspects, such as rationale augmentation
and replanning, which are beyond the scope of this work. Therefore, we focus only on the first two turns of
tool calling.


When2Call is a benchmark developed by NVIDIA to evaluate LLMs’ decision-making in tool invocation,
particularly for identifying requests that do not require a tool call.


For each dataset, we evaluated four LLMs under both methods: GPT-4o, GPT-5, Mistral-7B-Instruct-v0.3,
and DeepSeek-Coder-7B-Instruct-v1.5. GPT-5 is designed with enhanced reasoning capabilities, making it
suitable for testing how reasoning processes influence our proposed approach, which has not been explicitly
optimized in existing reasoning-oriented models. GPT-4o serves as a state-of-the-art general-purpose LLM,
widely adopted in both academic and industrial applications. Mistral provides a strong and widely used
research baseline. Finally, DeepSeek-Coder, which is primarily trained on code (with only 13% natural
language data), enables us to examine how our natural-language-based approach performs in models not
predominantly trained on natural language.


GPT-4o and GPT-5 are tested using prompts provided directly in the datasets. For GPT-5, reasoning effort
is set to minimal according to the guide for using GPT-5 (OpenAI, 2025b). Mistral and DeepSeek-Coder
are fine-tuned with the training data released alongside their corresponding datasets, and both models are
fine-tuned for five epochs with early stopping.


With each dataset and model, we tested the two approaches as follows. Detailed example prompts are
provided in Appendix B.


1. Schema-constrained generation: This is the original method used in all datasets.
A typical prompt is: "Generate an API request in the format of

[ApiName(key1=’value1’, key2=’value2’, ...)]".


2. Template-based generation: The model is asked to generate a template-based output. For
example: "Generate in the format like: Call the ‘API Name‘ API with
following parameters: ‘parameter1 name‘ as ‘parameter1 value‘,
...". Other instructions in the prompt are kept the same as in schema-constrained generation.


3 RESULTS


We evaluate performance using the macro-averaged F1 score for both tool names and parameters. Tool name
F1 is computed based on correctly predicted tool names, while parameter F1 is computed based on correctly
predicted parameters and their values. Absolute deltas and p-values are also reported. P-values are computed
using a paired permutation test with 9,999 random permutations.


The parameter values generated by the LLM may not exactly match the ground truth in wording but can still
convey the same semantic meaning. For instance, the ground truth may specify “create meeting”, while the
model produces “meeting creation”. To account for such lexical variations, we compute the cosine similarity
between the embeddings of the raw text values. Values with a similarity above 0.9 are considered equivalent;
otherwise, they are treated as different.


An overview of the dataset statistics is provided in Table 1.


2


Table 1: Dataset statistics.


Avg. # tools
Dataset # Instances # Expected tool calling # Expected parameter
provided in prompt


API-Bank L1 399 399 906 2.2
API-Bank L2 135 135 267 1.9
ToolAce 1000 1850 3242 3.4
When2Call 3652 1295 2970 3.0


The test results are reported in Table 2 for tool name F1 scores and in Table 3 for parameter F1 scores.


Table 2: Tool name F1. The improved template-based values are bolded for clarity.
Delta values in bold indicate statistically significant changes ( _p ≤_ 0 _._ 05).


Model Dataset Schema-constrained Template-based Delta p-value


GPT-4o API-Bank L1 0.9724 **0.9800** 0.0075 0.54
API-Bank L2 0.6148 **0.7185** **0.1037** 0.00
ToolAce 0.6055 **0.6714** **0.0659** 0.00
When2Call 0.2952 **0.3091** **0.0140** 0.00


GPT-5 API-Bank L1 0.9850 0.9148 **-0.0702** 0.00
API-Bank L2 0.5630 **0.6296** 0.0667 0.11
ToolAce 0.5445 **0.7650** **0.2205** 0.00
When2Call 0.3209 0.2771 **-0.0438** 0.00


Mistral API-Bank L1 0.8195 **0.9474** **0.1278** 0.00
API-Bank L2 0.8444 **0.8815** 0.0370 0.26
ToolAce 0.9308 **0.9482** **0.0175** 0.00
When2Call 0.0871 **0.1380** **0.0509** 0.00


DeepSeek API-Bank L1 0.9599 0.9323 **-0.0276** 0.00
API-Bank L2 0.9111 0.8963 -0.0148 0.51
ToolAce 0.9439 **0.9476** 0.0037 0.47
When2Call 0.0931 **0.1791** **0.0860** 0.00


3


Table 3: Paramter F1. The improved template-based values are bolded for clarity.
Delta values in bold indicate statistically significant changes ( _p ≤_ 0 _._ 05).


Model Dataset Schema-constrained Template-based Delta p-value


GPT-4o API-Bank L1 0.9198 0.8979 **-0.0219** 0.04
API-Bank L2 0.3743 **0.4930** **0.1187** 0.00
ToolAce 0.5099 **0.5498** **0.0399** 0.04
When2Call 0.2608 **0.2689** **0.0081** 0.01


GPT-5 API-Bank L1 0.9321 0.8618 **-0.0703** 0.00
API-Bank L2 0.2778 **0.3438** 0.0660 0.08
ToolAce 0.4247 **0.6127** **0.1879** 0.00
When2Call 0.2564 0.2248 **-0.0316** 0.00


Mistral API-Bank L1 0.7138 **0.8681** **0.1543** 0.00
API-Bank L2 0.5306 **0.5880** 0.0574 0.06
ToolAce 0.7942 **0.8020** 0.0077 0.25
When2Call 0.0754 **0.1185** **0.0432** 0.00


DeepSeek API-Bank L1 0.9046 0.8823 **-0.0223** 0.03
API-Bank L2 0.6283 0.5957 -0.0327 0.07
ToolAce 0.8131 **0.8150** 0.0019 0.77
When2Call 0.0838 **0.1585** **0.0748** 0.00


For GPT-4o and Mistral, template-based generation consistently improves performance, with 15 of 16
F1 scores showing gains, 11 of which are statistically significant. For GPT-5, performance varies across
datasets: strong gains are observed on ToolACE, though some statistically significant losses also occur.
Further details are provided in Section 4.5.


For DeepSeek-Coder, four F1 scores improve, two of which are statistically significant. DeepSeek-Coder is
specifically tuned for code generation and contains only 13% natural language data in its training corpus. We
hypothesize that generation tasks closer to a model’s training data are more likely to yield improved results.
Since natural language–based generation is different from DeepSeek-Coder’s primary training domain, it
is unsurprising that its performance with template-based generation is lower compared to other models.
Nevertheless, the observed performance losses are marginal.


4 RESULT ANALYSIS


For our analysis of the two approaches, we selected representative datasets exhibiting statistically significant
performance differences between template-based and schema-constrained generation. Except for GPT-5,
discussed in Section 4.5, losses associated with template-based generation are generally marginal ( _<_ 3%).
To highlight the most substantive differences, our analysis focuses on datasets where template-based generation yields a clear and substantial advantage.


For the purpose of analysis, we categorize the model outputs into the following mutually exclusive groups:


1. Reasoning Failure: The model produces incorrect logical inferences. For example, it may refuse to
answer by claiming that insufficient information is available when, in fact, the provided context is
sufficient. Or it may request additional information for tool calls that are irrelevant or incapable of
resolving the task.


4


2. Schema Violation: The model output cannot be parsed by the format specified in the prompt.


3. Incorrect Tool Name: The tool names are incorrect.


4. Incorrect Parameter Name: The parameter names are incorrect.


5. Incorrect Parameter Value: The parameter values are incorrect.

6. Correct: The tool names, parameter names, and parameter values are all predicted correctly.


4.1 MISTRAL ON API-BANK LEVEL 1


Template-based generation improves upon schema-constrained generation in categories “Schema Violation”
and “Incorrect Tool Name”, as shown in Table 4.


Regarding “Schema Violation”, schema-constrained generation fails to correctly generate some
complex structures, such as lists of dictionaries. For example, a typical error is an unmatched bracket: "[ _{_ ’name’: ’blood ~~p~~ ressure’, ’value’: ’120/80’ _}_, _{_ ’name’:
’heart ~~r~~ ate’, ’value’: ’80’] _}_ ]".


In cases of “Incorrect Tool Name”, schema-constrained generation sometimes fails to interpret
conversations where the user’s request changes during the dialogue. For example, the initial request at the beginning of the conversation may differ from the final request at the
end, as illustrated in Appendix B.1. Schema-constrained generation produces "API-Request:

[DeleteAccount(token=’foo’)]", while template-based generation outputs the correct result
"API-Request: [GetUserToken(username=’foo’, password=’bar’)]".


Table 4: Mistral on API-Bank L1.


Type Schema-constrained generation Template-based generation


Reasoning Failure 0 0
Schema Violation 14 0
Incorrect Tool Name 58 21
Incorrect Parameter name 21 12
Incorrect Parameter value 37 27
Correct 269 339


4.2 GPT-4O ON API-BANK LEVEL 2


Template-based generation improves upon schema-constrained generation in the category “Incorrect Tool
Name”, as shown in Table 5.


When examining instances from API-Bank Level 2, we observe that the ground truth tool
name is sometimes provided within the conversation rather than in the initial instructions.
Consider the case illustrated in Appendix B.2: template-based generation correctly outputs
"API-Request: [SymptomSearch(symptom=’rash’)]", while schema-constrained generation produces "API-Request: [ToolSearcher(keywords=’rash’)]", reflecting a misunderstanding of the conversation. Therefore, higher accuracy in tool prediction may indicate an enhanced ability
to interpret and utilize conversational context.


5


Table 5: GPT-4o on API-Bank L2.


Type Schema-constrained generation Template-based generation


Reasoning Failure 0 0
Schema Violation 0 2
Incorrect Tool Name 52 36
Incorrect Parameter Name 1 1
Incorrect Parameter Value 34 32
Correct 48 64


4.3 GPT-5 ON TOOLACE


Template-based generation shows the largest improvement over schema-constrained generation in the
“Schema Violation” category, as shown in Table 6. Schema-constrained generation frequently fails due
to punctuation errors, such as missing quotation marks required by the instructions. Further discussion of
GPT-5 can be found in Section 4.5.


Table 6: GPT-5 on ToolAce.


Type Schema-constrained generation Template-based generation


Reasoning Failure 39 35
Schema Violation 394 165
Incorrect Tool Name 23 41
Incorrect Parameter Name 26 35
Incorrect Parameter Value 73 127
Correct 445 597


4.4 DEEPSEEK ON WHEN2CALL


Template-based generation improves upon schema-constrained generation in the category “Reasoning Failure”, as shown in Table 7. Compared to template-based generation, schema-constrained generation exhibits
more frequent failures in contextual understanding, such as unnecessarily requesting additional information
even when the provided context is sufficient.


Table 7: DeepSeek on When2Call.


Type Schema-constrained generation Template-based generation


Reasoning Failure 2313 2137
Schema Violation 2 6
Incorrect Tool Name 1 17
Incorrect Parameter Name 17 56
Incorrect Parameter Value 27 69
Correct 1292 1367


6


4.5 DIFFERENT REASONING EFFORT ON GPT-5


For template-based generation, GPT-5 demonstrates strong gains on two datasets, while exhibiting reduced
performance on the other two. Some metrics are even lower than those of GPT-4o. To gain deeper insights,
we tested GPT-5 with varying levels of reasoning effort across different datasets, as shown in Figure 1.


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
|Schema-constrained tool name F1<br>Template-based tool name F1|Schema-constrained tool name F1<br>Template-based tool name F1|Schema-constrained tool name F1<br>Template-based tool name F1|Schema-constrained tool name F1<br>Template-based tool name F1|Schema-constrained tool name F1<br>Template-based tool name F1|
|Schema-constrained tool name F1<br>Template-based tool name F1|Schema-constrained tool name F1<br>Template-based tool name F1|Schema-constrained tool name F1<br>Template-based tool name F1|Schema-constrained tool name F1<br>Template-based tool name F1|Schema-constrained tool name F1<br>Template-based tool name F1|
|<br>Schema-constrained parameter F1<br>Template-based paramter F1|<br>Schema-constrained parameter F1<br>Template-based paramter F1|<br>Schema-constrained parameter F1<br>Template-based paramter F1|<br>Schema-constrained parameter F1<br>Template-based paramter F1||
|<br>Schema-constrained parameter F1<br>Template-based paramter F1|<br>Schema-constrained parameter F1<br>Template-based paramter F1||||


|Col1|Col2|Col3|
|---|---|---|
||||
||||
|Schema-constrained tool name F1<br>Template-based tool name F1|Schema-constrained tool name F1<br>Template-based tool name F1|Schema-constrained tool name F1<br>Template-based tool name F1|
|Schema-constrained tool name F1<br>Template-based tool name F1|Schema-constrained tool name F1<br>Template-based tool name F1|Schema-constrained tool name F1<br>Template-based tool name F1|
||<br>Schema-constrained parameter F1<br>Template-based paramter F1|<br>Schema-constrained parameter F1<br>Template-based paramter F1|
||||


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
|Schema-constrained tool name F1<br>|Schema-constrained tool name F1<br>|Schema-constrained tool name F1<br>|Schema-constrained tool name F1<br>|Schema-constrained tool name F1<br>|
|Schema-constrained tool name F1<br>|Schema-constrained tool name F1<br>|Schema-constrained tool name F1<br>|Schema-constrained tool name F1<br>|Schema-constrained tool name F1<br>|
|Template-based tool name F1<br>Schema-constrained parameter F1<br>Template-based paramter F1|Template-based tool name F1<br>Schema-constrained parameter F1<br>Template-based paramter F1|Template-based tool name F1<br>Schema-constrained parameter F1<br>Template-based paramter F1|Template-based tool name F1<br>Schema-constrained parameter F1<br>Template-based paramter F1||
|Template-based tool name F1<br>Schema-constrained parameter F1<br>Template-based paramter F1|Template-based tool name F1<br>Schema-constrained parameter F1<br>Template-based paramter F1||||


|Schema-constrained tool name F1<br>Template-based tool name F1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
|Schema-constrained tool name F1<br>Template-based tool name F1|Schema-constrained tool name F1<br>Template-based tool name F1|Schema-constrained tool name F1<br>Template-based tool name F1|Schema-constrained tool name F1<br>Template-based tool name F1|Schema-constrained tool name F1<br>Template-based tool name F1|
|Schema-constrained parameter F1<br>Template-based paramter F1|Schema-constrained parameter F1<br>Template-based paramter F1|Schema-constrained parameter F1<br>Template-based paramter F1|Schema-constrained parameter F1<br>Template-based paramter F1|Schema-constrained parameter F1<br>Template-based paramter F1|
||||||
||||||
||||||


Figure 1: Impact of reasoning effort on F1 score: (a) API-Bank Level 1, (b) API-Bank Level 2, (c) ToolACE,
and (d) When2Call. X-axis is reasoning effort, y-axis is F1 score.


In most cases, increased reasoning effort enhances or preserves performance for schema-constrained generation, suggesting that the model benefits from additional reasoning. By contrast, template-based generation
shows inconsistent trends, which may indicate that the model’s reasoning process is not well aligned with


7


1


0 _._ 8


0 _._ 6


0 _._ 4


0 _._ 2


1


0 _._ 8


0 _._ 6


0 _._ 4


0 _._ 2


0
minimal low medium high


(a) API-Bank L1.


1


0 _._ 8


0 _._ 6


0 _._ 4


0 _._ 8


0 _._ 6


0 _._ 4


0 _._ 2


0
minimal low medium high


(b) API-Bank L2.


1


0 _._ 2


0
minimal low medium high


(c) ToolACE.


0
minimal low medium high


(d) When2Call.


template-based patterns. Intermediate reasoning steps could amplify small deviations, leading to instability
in the final output.


We report the total number of instances in the “Reasoning Failure” and “Schema Violation” categories under
different levels of reasoning effort in Table 8, as these categories are expected to be heavily influenced by
reasoning effort. Consistent with expectations, schema-constrained generation shows a reduction or a stable
number of instances as reasoning effort increases, whereas the effect on template-based generation appears
more variable.


Another observation is that, for ToolACE, template-based generation exhibits fewer “Schema Violations”
than schema-constrained generation, as shown in Table 6. In contrast, for other datasets, template-based generation exhibits more “Schema Violations”, with a common pattern being the omission of required backtick
marks in the prompt. For example, the expected output is "Call the ‘SymptomSearch‘ API with
following parameters: ‘symptom‘ as ‘rash‘", but the actual output is "Call the
SymptomSearch API with following parameters: symptom as rash". This inconsistency also suggests a possible lack of optimization for handling such structural patterns.


Table 8: Total “Reasoning Failure” and “Schema Violation” instances at different reasoning effort.


Minimal Low Medium High


API-Bank L1 Schema-constrained 0 1 0 0
Template-based 27 75 49 25


API-Bank L2 Schema-constrained 0 0 0 0
Template-based 2 32 21 6


ToolAce Schema-constrained 433 258 135 94
Template-based 200 145 104 111


When2Call Schema-constrained 976 698 702 705
Template-based 1085 1095 1077 1035


4.6 KEY OBSERVATIONS


These findings demonstrate that template-based generation generally improves contextual understanding
and schema generation accuracy for non-reasoning models. For reasoning models, however, performance
depends strongly on the training of the reasoning process, and template-based generation may fail to achieve
consistent gains over schema-constrained generation when reasoning steps are insufficiently optimized. For
code-centric models with limited natural-language training, the gains from template-based generation tend
to be smaller, though the impact on overall performance remains limited.


5 CONCLUSION AND FUTURE WORK


This work introduces a new approach for LLMs to perform tool calling. We compare the existing schemaconstrained generation with our new template-based generation. Experiments across multiple datasets and
models demonstrate that template-based generation can improve LLM performance in tool calling, achieving
higher accuracy than schema-constrained generation in the majority of tests.


Our central hypothesis is that this performance gain stems from the template-based format’s closer alignment with natural language, the native domain of most pretrained LLMs. By framing tool calls in a more


8


descriptive and less rigid structure, this approach likely mitigates schema violation errors and leverages the
models’ inherent linguistic capabilities more effectively.


Despite these strong results, the full potential of template-based generation remains largely untapped. Our
primary limitations, which also highlight key directions for future research, are as follows:


1. Evaluation under suboptimal conditions: Our method was evaluated against baselines that have
a significant incumbent advantage. Current LLMs are predominantly pretrained and fine-tuned
for schema-constrained generation (e.g., JSON output). This inherent bias likely disadvantages our
novel template-based approach, suggesting that its observed improvements may be an underestimation of its true capability. Future work should explore pretraining models specifically on templateformatted data to create a more equitable evaluation and unlock further performance gains.


2. Limited scope of evaluation: Due to resource constraints, our investigation was confined to a specific set of models, datasets, and a single prompt structure. A broader study involving more diverse open-source and proprietary models, larger-scale datasets, and systematic prompt engineering
would be crucial to fully generalize our findings. Furthermore, potential overlaps between existing evaluation datasets and models’ pretraining data may have artificially inflated the performance
of the schema-constrained baseline an advantage that does not extend to our novel template-based
prompts.


3. Need for deeper mechanistic understanding: We hypothesize that the observed benefits of templatebased generation arise from its structural similarity to the natural language that constitutes the majority of an LLM’s pretraining data. A systematic investigation of this “natural language alignment”
hypothesis presents a promising direction for future work.


In summary, template-based generation represents a promising approach to improving tool calling accuracy
and suggests new opportunities for advancing model training and optimization strategies.


6 REPRODUCIBILITY STATEMENT


All code used in this paper is submitted as supplementary material and will be made publicly available after
acceptance.


REFERENCES


Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss,
Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens
Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess,
Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei.
Language models are few-shot learners. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell,
Maria-Florina Balcan, and Hsuan-Tien Lin (eds.), _Advances_ _in_ _Neural_ _Information_ _Processing_ _Systems_
_33:_ _Annual_ _Conference_ _on_ _Neural_ _Information_ _Processing_ _Systems_ _2020,_ _NeurIPS_ _2020,_ _December_
_6-12,_ _2020,_ _virtual_, 2020. URL [https://proceedings.neurips.cc/paper/2020/hash/](https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html)
[1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html.](https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html)


Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Y. Wu,
Y.K. Li, Fuli Luo, Yingfei Xiong, and Wenfeng Liang. Deepseek-coder: When the large language model
meets programming – the rise of code intelligence, 2024. [URL https://arxiv.org/abs/2401.](https://arxiv.org/abs/2401.14196)
[14196.](https://arxiv.org/abs/2401.14196)


9


Minghao Li, Yingxiu Zhao, Bowen Yu, Feifan Song, Hangyu Li, Haiyang Yu, Zhoujun Li, Fei Huang,
and Yongbin Li. Api-bank: A comprehensive benchmark for tool-augmented llms. In Houda Bouamor,
Juan Pino, and Kalika Bali (eds.), _Proceedings of the 2023 Conference on Empirical Methods in Natural_
_Language Processing, EMNLP 2023, Singapore, December 6-10, 2023_, pp. 3102–3116. Association for
Computational Linguistics, 2023. doi: 10.18653/V1/2023.EMNLP-MAIN.187. [URL https://doi.](https://doi.org/10.18653/v1/2023.emnlp-main.187)
[org/10.18653/v1/2023.emnlp-main.187.](https://doi.org/10.18653/v1/2023.emnlp-main.187)


Weiwen Liu, Xu Huang, Xingshan Zeng, Xinlong Hao, Shuai Yu, Dexun Li, Shuai Wang, Weinan Gan,
Zhengying Liu, Yuanqing Yu, Zezhong Wang, Yuxian Wang, Wu Ning, Yutai Hou, Bin Wang, Chuhan
Wu, Xinzhi Wang, Yong Liu, Yasheng Wang, Duyu Tang, Dandan Tu, Lifeng Shang, Xin Jiang, Ruiming
Tang, Defu Lian, Qun Liu, and Enhong Chen. Toolace: Winning the points of LLM function calling. In
_The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-_
_28, 2025_ . OpenReview.net, 2025. [URL https://openreview.net/forum?id=8EB8k6DdCU.](https://openreview.net/forum?id=8EB8k6DdCU)


MistralAI. Mistral-7b-instruct-v0.3, 2024. URL [https://huggingface.co/](https://huggingface.co/Mistral-7B-Instruct-v0.3)
[Mistral-7B-Instruct-v0.3.](https://huggingface.co/Mistral-7B-Instruct-v0.3) Model Card on Hugging Face.


OpenAI. Introducing gpt-5, 2025a. [URL https://openai.com/index/introducing-gpt-5.](https://openai.com/index/introducing-gpt-5)


OpenAI. Using gpt-5, 2025b. URL [https://platform.openai.com/docs/guides/](https://platform.openai.com/docs/guides/latest-model#minimal-reasoning-effort)
[latest-model#minimal-reasoning-effort.](https://platform.openai.com/docs/guides/latest-model#minimal-reasoning-effort)


OpenAI, Aaron Hurst, Adam Lerer, Adam P. Goucher, and ... Gpt-4o system card, 2024. URL [https:](https://arxiv.org/abs/2410.21276)
[//arxiv.org/abs/2410.21276.](https://arxiv.org/abs/2410.21276) arXiv preprint.


Jackson Petty, Sjoerd van Steenkiste, and Tal Linzen. How does code pretraining affect language model task
performance? _Trans. Mach. Learn. Res._, 2025, 2025. [URL https://openreview.net/forum?](https://openreview.net/forum?id=pxxmUKKgel)
[id=pxxmUKKgel.](https://openreview.net/forum?id=pxxmUKKgel)


Changle Qu, Sunhao Dai, Xiaochi Wei, Hengyi Cai, Shuaiqiang Wang, Dawei Yin, Jun Xu, and Ji-rong
Wen. Tool learning with large language models: a survey. _Frontiers of Computer Science_, 19(8), January
[2025. ISSN 2095-2236. doi: 10.1007/s11704-024-40678-2. URL http://dx.doi.org/10.1007/](http://dx.doi.org/10.1007/s11704-024-40678-2)
[s11704-024-40678-2.](http://dx.doi.org/10.1007/s11704-024-40678-2)


Hayley Ross, Ameya Sunil Mahabaleshwarkar, and Yoshi Suhara. When2Call: When (not) to call tools.
In _Proceedings_ _of_ _the_ _2025_ _Conference_ _of_ _the_ _Nations_ _of_ _the_ _Americas_ _Chapter_ _of_ _the_ _Association_ _for_
_Computational_ _Linguistics:_ _Human_ _Language_ _Technologies_ _(Volume_ _1:_ _Long_ _Papers)_, pp. 3391–3409,
Albuquerque, New Mexico, April 2025. Association for Computational Linguistics. ISBN 979-8-89176189-6. [URL https://aclanthology.org/2025.naacl-long.174/.](https://aclanthology.org/2025.naacl-long.174/)


Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen
Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang,
Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, and Ji-Rong Wen. A survey
of large language models. _CoRR_, abs/2303.18223, 2023. doi: 10.48550/ARXIV.2303.18223. URL
[https://doi.org/10.48550/arXiv.2303.18223.](https://doi.org/10.48550/arXiv.2303.18223)


A THE USE OF LARGE LANGUAGE MODELS


Large Language Models are used to aid or polish writing, like correcting grammar errors and make sentences more fluent. The typical prompt used is like: "Fix grammar errors and make this
more fluent to use in a paper".


10


B PROMPT EXAMPLES


Below is the full prompt used in our experiments. The schema-constrained generation prompt is taken
directly from the original dataset. For When2Call, we added extra instructions to the prompts in both approaches to help the model produce the correct output in cases requiring more information or that it could
not handle.


B.1 API-BANK LEVEL 1


11


B.2 API-BANK LEVEL 2


12


API-Request: [SymptomSearch(symptom=’rash’)]


B.3 TOOLACE


13


14


15


B.4 WHEN2CALL


16


17


18


19