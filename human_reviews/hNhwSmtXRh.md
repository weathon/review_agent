# Lemur: Harmonizing Natural Language and Code for Language Agents

- Avg Score: 7.00
- Decision: Accept (spotlight)
- Scores: 8, 6, 6, 8

## Abstract
We introduce Lemur and Lemur-Chat, openly accessible language models optimized
for both natural language and coding capabilities to serve as the backbone
of versatile language agents. The evolution from language chat models to
functional language agents demands that models not only master human interaction,
reasoning, and planning but also ensure grounding in the relevant environments.
This calls for a harmonious blend of language and coding capabilities
in the models. Lemur and Lemur-Chat are proposed to address this necessity,
demonstrating balanced proficiencies in both domains, unlike existing
open-source models that tend to specialize in either. Through meticulous pretraining
using a code-intensive corpus and instruction fine-tuning on text and code
data, our models achieve state-of-the-art averaged performance across diverse
text and coding benchmarks. Comprehensive experiments demonstrate Lemur’s
superiority over existing open-source models and its proficiency across various
agent tasks involving human communication, tool usage, and interaction under
fully- and partially- observable environments. The harmonization between natural
and programming languages enables Lemur-Chat to significantly narrow the
gap with proprietary models on agent abilities, providing key insights into developing
advanced open-source agents adept at reasoning, planning, and operating
seamlessly across environments. Our model and code have been open-sourced at
https://github.com/OpenLemur/Lemur.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces Lemur and Lemur-Chat, openly-accessible large language models that harmonize natural language with code capabilities. The Lemur models are trained on the basis of Llama-2, with a code-centric pre-training stage with a code-to-text ratio of 10:1 for code-text harmonization, and a supervised instruction fine-tuning stage. The authors conduct systematic and comprehensive evaluations of Lemur models on diverse benchmarks, consisting of fundamental code/language benchmarks and pratical scenarios that connect LLMs to environments. The paper categorizes the capabilities of LLM agents in four aspects: agument with tools, self-debug, following feedback, and exploring environments. Over extensive benchmarks, the experimental results demonstrate harmonized capabilties between natural language and codes, and show that the Lemur models consistently outperform their counterparts on a wide range of tasks.

### Strengths
- The idea of harmonizing the natural language and coding capabilities of LLMs is nice. With carefully designed code-to-text ratio and the selection of training data, the resulting Lemur models achieve a harmonious blend of language and coding capabilities.
- The resulting Lemur models achieve competitive performance on language-coding tasks against gpt-3.5-turbo. The open-sourced Lemur models will be useful for the research community, and would be foundation models to develop agents.
- The experiments are solid and evaluations are systematically organized. The Lemur models are evaluated in a clear and comprehensive evaluation process. The evaluation consists of the evaluations in each domain of code or language, and diverse code-language tasks that are grouped into 4 types of skills, establishing a good evaluation procedure for language-code LLM agents.
- The paper is clear and concise with well-structured evaluations.

### Weaknesses
- As mentioned in Introduction, the paper has offered valuable insights on synergy, but it is unclear what the insights exactly are. I would suggest clearly presenting the insights instead of letting readers find where is the insights across the paper.
- Minor: In Figure 2, the capitalizations are not consistent. (Use->,  run->); Section 4.5: mapp -> map; Section 4.5: intermm.

### Questions
- Why is a large proportion of the pre-training data is in Python?
- Is the harmonization controlled by the text-to-code ratio? How did you come up with the idea of setting a ratio of 10:1?

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
4 excellent

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes Lemur and Lemur-Chat language models, emphasizing their combined proficiency in both natural language understanding and coding capabilities. These models are designed to bridge the gap between understanding human interactions and manipulating code, aiming to serve as versatile language agents.

what contributions does it make:
1.The proposed models Lemur and Lemur-Chat narrow the gap with proprietary models in terms of agent abilities, leveraging its harmonization of both natural language and programming languages.
2.Provide comprehensive evaluations of language and coding abilities.
3.These models are open-source, providing a valuable resource for the community and potentially contributing to the development of advanced open-source agents that can be seamlessly reasoned, planned, and run in a variety of environments.

### Strengths
1.It improves the coding ability while maintaining the reasoning ability of Llama-2.
2.The Lemur is pre-trained and fine-tuned using a rich dataset that includes text and code, ensuring a balance of performance across a variety of text and coding benchmarks.
3.The model showcases proficiency in agent tasks, encompassing human communication, tool usage, and interaction across observable environments.

### Weaknesses
1.It seems that pre-training takes the responsibility to gain the coding ability, and the supervised fine-tuning takes the responsibility to gain the natural language ability, while it is vague how the proposed model balance these two abilities. 
2.As shown in Tables 4, 5, and 7, the performance of the proposed model Lemur-70B-Chat falls short when compared to GPT-4 and this discrepancy in performance lacks an explanatory or discussion.
3.Table 3 lists three baseline models—StarCoder-15B, StarCoderPlus-15B, and WizardCoder-15B—without corresponding explanations or references in the provided context.
4.Table7 does not have references and analysis.

### Questions
1.GPT4 also integrates text and code capabilities, what are the advantages of this paper?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents Lemur and Lemur-Chat, large language models that exhibit balanced proficiency in both language and coding. The paper further trains LLAMA on a corpus with a code-to-text ratio of 10:1 and fine-tunes the model on four instruction datasets. The paper evaluates these two models across a broad spectrum of tasks, which includes text benchmarks (such as MMLU, BBH, etc.) and code benchmarks (such as HumanEval, MBPP, MultiPL-E, etc.). Moreover, the paper demonstrates that these models perform exceptionally well in language agent scenarios, such as augmenting with tools, self-debugging with environment feedback, adhering to natural language feedback, and exploring in partially observable environments.

### Strengths
This article validates the effectiveness of the Lemur model on a large number of benchmarks and verifies the importance of balanced language and coding capabilities for language agent scenarios.

### Weaknesses
1) The technical contribution of this article is quite limited, it merely continues training the LLAMA model on a mixture of text and code data and instruction tuning on four datasets.
2) When comparing performance on the code benchmark, the authors use a large 70B model, but the code-specific models they compare with are mostly 15-30B in size, which makes the comparison somewhat unequal.

### Questions
Why you use 10:1 text-to-code ratio in your pretraining data?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes two models Lemur and Lemur-Chat by training on a combined data of natural language and programming languages. Comprehensive experiments show that the proposed models show superior performance on 12 agent benchmarks.

### Strengths
**Originality:** This paper proposes a novel way of training LLMs with code + text data to design language agents. 

**Quality:** There are detailed studies included in the paper about how training LLMs can be beneficial to solve both the language and agent tasks. 

**Clarity:** The paper is well-written and easy to follow.

### Weaknesses
**Ambiguous Motivation:** I am fully not convinced with the sentence "for the construction of language agents, it is imperative for language models to possess harmonized capabilities in both natural language and programming languages." Its unclear how programming languages correlate with language understanding. In fact in the context of linguistics (morphology, syntax and semantics), programming languages might not satisfy any of them. I believe the authors should provide more context for it. Although the experimental results show that Lemur-Chat outperforms on majority of the datasets, correlation does not imply causation.

### Questions
1. Is there any reason to choosing scripting languages?
2. Is the performance replicable for base models other than Llama?
3. How much does the size of Llama matter for experiments? Can the same pipeline be replicated for smaller Llama versions?

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
4 excellent
