# ResPrompt: Residual Connection Prompting Advances Multi-Step Reasoning in Large Language Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 8, 6, 3

## Abstract
Chain-of-thought (CoT) prompting, which offers step-by-step problem-solving rationales, has impressively unlocked the reasoning potential of large language models (LLMs). Yet, the standard CoT is less effective in problems demanding multiple reasoning steps. This limitation arises from the complex reasoning process in multi-step problems: later stages often depend on the results of several steps earlier, not just the results of the immediately preceding step. Such complexities suggest the reasoning process is naturally represented as a graph. The almost linear and straightforward structure of CoT prompting, however, struggles to capture this complex reasoning graph. To address this challenge, we propose Residual Connection Prompting (*RESPROMPT*), a new prompting strategy that advances multi-step reasoning in LLMs. Our key idea is to reconstruct the reasoning graph within prompts. We achieve this by integrating necessary connections—links present in the reasoning graph but missing in the linear CoT flow—into the prompts. Termed “residual connections", these links are pivotal in morphing the linear CoT structure into a graph representation, effectively capturing the complex reasoning graphs inherent in multi-step problems. We evaluate *RESPROMPT* on six benchmarks across three diverse domains: math, sequential, and commonsense reasoning. For the open-sourced LLaMA family of models, *RESPROMPT* yields a significant average reasoning accuracy improvement of 12.5% on LLaMA-65B and 6.8% on LLaMA2-70B. Breakdown analysis further highlights *RESPROMPT* particularly excels in complex multi-step reasoning: for questions demanding at least five reasoning steps, *RESPROMPT* outperforms the best CoT based benchmarks by a remarkable average improvement of 21.1% on LLaMA-65B and 14.3% on LLaMA2-70B. Through extensive ablation studies and analyses, we pinpoint how to most effectively build residual connections, and assess *RESPROMPT* in view of “emergent ability", few-shot learning, and robustness, while also noting scenarios in which it might be superfluous.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work aims to improve the multi-step reasoning ability of Chain-of-Thought (CoT) prompting. When coping with complex problems, CoT tends to only leverage the reasoning outcomes from one prior steps and ignores those from several prior steps. Thus, CoT only exploits the linear reasoning structure in problem-solving and struggles to look back on several steps earlier, whose reasoning scheme is represented as a graph. To address this issue, this work introduces a new prompting method, called RESPrompt, to explicitly add connections to previous reasoning steps in prompting exemplars. Experiments on the family of Llama models show the performance improvement of RESPrompt over the Vallina CoT method.

### Strengths
Enhancing the complex reasoning ability of CoT-like prompting methods is a crucial problem in prompt-based reasoning of Large Language Models (LLMs). It is promising to construct the graph-like reasoning scheme and back-track prior reasoning steps to facilitate next-step inference for complex tasks.

### Weaknesses
However, the reviewer has several concerns about the Methodology and Experiment.  
Methodology:  
1. First, the Vallina CoT seems to already leverage the reasoning results from previous steps for next-step inference, as shown in Figure 2 (b). In Step 4, the Vallina CoT uses the results obtained in Step 1,2,3 to infer the money Tobias earned from shoveling driveways. Thus, the reviewer is quite skeptical that Vallina CoT can intrinsically leverage the results from several prior steps for next-step inference, especially on GPT-3.5 and GPT-4 models with increased model capability.   
2. Concerns about Novelty and Generality. Although RESPrompt is claimed to match the graph structure of reasoning in complex tasks, it also revokes a linearly step-by-step reasoning scheme using more crafted CoT-like prompting examples. Thus, the reviewer is concerned with the novelty of RESPrompt. Moreover, it may require extensive labor and expertise to 1. manually design the prompt examples for RESPrompt and 2. establish optimal residual connections within multi-step reasoning. These issues make RESPrompt less generalizable to new tasks.  
3. What are the 'provided question conditions' in Line 11, Page 4?


Experiments:  
1. RESPrompt is only compared with the Vallina CoT method. Empirical comparison between RESPrompt with representative methods such as Least2Most, and CoT-SC is lacking.  
2. Moreover, the reviewer is curious about whether the performance gain of RESPrompt over the Vallina CoT diminishes when testing on more powerful models such as GPT 3.5 and GPT 4.  
3. The performance gain of RESPrompt over the Vallina CoT is marginal. As shown in Figure 6, RESPrompt only enjoys significant performance gains in specific few-shot prompting settings on some datasets. This raises concerns about the effectiveness of RESPrompt in general settings and tasks.

### Questions
The authors are encouraged to address the concerns in Weaknesses.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a residual connection prompting method for enhancing multi-step reasoning in large language models. The proposed ResPrompt aims at reconstructing the reasoning graph within prompts, which is implemented by adding missing links to transform the linearly structured CoT prompts into graph-like structures. Experiments are conducted via LLaMA and LLaMA2 on 6 benchmarks covering mathematical reasoning, sequential reasoning, and commonsense reasoning. The experimental results show the benefits of the proposed ResPrompt and some interesting findings.

### Strengths
The proposed approach uses a simple method to recover complex reasoning graphs and reduce reasoning difficulty, which is easy to implement. According to the experimental results, the proposed approach shows significant improvements. The ablation study and analysis provide interesting findings.

### Weaknesses
Please see the questions listed below.

### Questions
Q1: For the performances according to the number of reasoning steps (Figure 3), it is shown that ResPrompt performances are fluctuant as the number of reasoning steps grows in the MathQA dataset, and in the questions with steps >= 5 it shows similar performance with the compared approaches. What are the possible reasons for this observation?

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
4 excellent

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduces RESPROMPT, a new prompting strategy that enhances multi-step reasoning in large language models (LLMs). By incorporating residual connections into prompts, RESPROMPT captures the complex reasoning graphs inherent in multi-step problems. Experimental results demonstrate significant improvements in reasoning accuracy, particularly for complex multi-step tasks, which confirms its motivation.

### Strengths
1.The proposed method is simple yet effective. By using the same tokens in intermediate steps to represent the connections between edges in the graph logic structure, it enhances the model's ability to solve complex multi-step reasoning problems.

2.The experimental results partially demonstrate the effectiveness of the method, but it could be more persuasive to include a broader range of baselines and benchmarks to further support the findings (see Weakness).

### Weaknesses
1.The paper's baselines only include conventional CoT methods (no matter standard, long, or short), lacking comparisons with some other approaches specifically designed to enhance CoT multi-step reasoning ability (e.g. [1][2][3]). This limitation weakens the solidity of the experimental results. Additionally, while the last paragraph in Introduction briefly mentions the advantages of the proposed method over approaches that introduce tree or graph structures during the reasoning stage, the experiments lack a proper comparison and analysis in this regard.

[1] Least-to-Most Prompting Enables Complex Reasoning in Large Language Models, ICLR 2023

[2] Complexity-Based Prompting for Multi-Step Reasoning, ICLR 2023

[3] Decomposed Prompting: A Modular Approach for Solving Complex Tasks, ICLR 2023

2.The proposed method in this paper appears to be effective primarily for mathematical and sequential reasoning benchmarks. Although the introduction and experimental setup mention commonsense reasoning, the main experiments do not include evaluations on such tasks. Despite using StrategyQA in Section 3.5 to demonstrate that RESPROMPT is not essential for simple questions, it would be valuable to showcase the effectiveness of RESPROMPT on more complex commonsense reasoning tasks such as CommonsenseQA or HotpotQA. Including experiments on these tasks would provide a comprehensive assessment of the efficacy of RESPROMPT across different domains of reasoning.

3.In section 3.3, it is mentioned that RESPROMPT demonstrates improved stability with variations in the number of exemplars in the prompt. However, it is difficult to observe from Figure 6 that RESPROMPT is more stable than other baselines and may even be less stable than the standard CoT. The results presented in the figure do not clearly indicate the claimed stability advantage of RESPROMPT.

### Questions
1.In section 3.3, it is mentioned that using the same tokens instead of symbolic variables yields better results. How would you explain this phenomenon? Intuitively, it seems challenging for the language model to reference the same tokens in the intermediate steps when they are not explicitly shown in the prompt.

2.This approach appears to somewhat decrease the readability of the reasoning process. What advantages does it offer in practical interactions with human?

3.How does RESPROMPT make the model aware of the graph structure during reasoning? Or does this linear modification in the prompt truly enable the model to learn the graph structure?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes a new prompting pipeline named ResPrompt to boost the performance of the LLMs, particularly focusing on complex reasoning tasks. The authors claim that the complex interdependence of prior reasoning steps during reasoning makes the question more challenging for LLMs. To alleviate this question, the authors propose to explicitly connect different reasoning steps and recap the LLM with prior reasoning results by repeating them in chain-of-thoughts.  Empirical evaluations demonstrate the potential of the proposed method.

### Strengths
The proposed method is neat and easy to follow. The authors provide relatively sufficient experiments to empirically demonstrate the performance improvement of the proposed method.

### Weaknesses
1. **Lacks sufficient verification of the necessity of the proposed method.** The authors claim that the complex interdependence between the current reasoning step and prior steps requires explicit connections among different reasoning steps. However, this is only an intuitive hypothesis and the authors do not verify whether it hold or not. Instead, they directly propose "repeating can help build the connections among reasoning steps and improve the performance". A better way to formulate the method could be designing simple yet straightforward experiments to verify whether the claim holds or not. For example, the authors can examine whether LLMs can make wrong references during a specific intermediate step. Consider the following CoT:

   ....So the money he earned from shoveling driveways is $110 - $15 - $60 = $35... 

   We can change the above sentence into `So the money he earned is $110 - $15 - $60 = $35.` and then query the LLM again asking where the `$35` is earned from. If the model cannot correctly answer `from shoveling driveways`, then that means the authors' claim might hold as the model does make mistakes to recovering the references from prior reasoning steps. 

2.  **Comparison with baselines might be unfair.** The authors claim that the performance can be achieved by more than 20% (relatively) on some datasets and models. However, if we look at the experiment setting and details of baselines, it seems the performance of baselines is suppressed, leading to a misunderstanding of the performance improvement. Specifically:

   a. There is no need to include the performance using the prompt used in [1], as the performance is too low.

   b. I do not understand why long prompts (`Long CoT`) lead to worse performance on the GSM8K dataset compared to short prompts (`Short CoT`), as the previous study [2] has already demonstrated that longer prompts provide better performance. The authors need clarifications and explanations for this phenomenon.

   c. It does not make sense to include questions with different numbers of reasoning steps in the `Long CoT`. Previous work [2] has already shown that using the most complex CoTs leads to the best performance. The author should include the original prompts used by [2] as another baseline. Note that the proposed method, due to repeating previous steps, uses significantly long prompts compared to baselines. Without including baselines that use prompts with similar lengths, it cannot be answered whether the performance improvement comes from longer prompts or the proposed method (repeating to connect intermediate steps)

   d. LLaMA performance on GSM8K could be much higher than the numbers in Table 1. See https://opencompass.org.cn/leaderboard-llm for a reference, where the LLaMA-2 achieves 63.5% accuracy and the LLaMA achieves 54.5% accuracy on GSM8K without any special techniques but CoTs. If we take this into consideration, the performance improvement is rather small.

3. **Cost-performance tradeoff is not discussed.** Again, the proposed method uses much longer CoTs compared to baselines. The author should discuss the inference cost given the long prompts.

4. **Additional ablation study.** I would like to emphasize this point again. The performance improvement could come from two aspects: longer prompts and proposed connections among different reasoning steps. The author should design an ablation study to demonstrate where the performance improvement comes from, rather than simply study which experiment designs can affect the final performance.

[1] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In NeurIPS, 2022b.

[2] Yao Fu, Hao Peng, Ashish Sabharwal, Peter Clark, and Tushar Khot. Complexity-based prompting for multi-step reasoning. In ICLR 2023

### Questions
Please refer to the weakness above. Although I give a low rating to this paper, I would be delighted to increase my rating given my questions addressed.

### Soundness
1 poor

### Presentation
2 fair

### Contribution
1 poor
