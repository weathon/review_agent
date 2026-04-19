# Think Beyond Size: Dynamic Prompting for More Effective Reasoning

- Decision: Reject
- Scores: 5, 3, 1, 3

## Abstract
Pretrained large language models (LLMs) are increasingly utilized across a wide range of natural language processing (NLP) tasks due to their impressive capabilities as few-shot learners. Recent techniques, such as chain-of-thought (CoT) prompting, have significantly advanced multi-step reasoning by introducing step-by-step decomposition, achieving state-of-the-art results on complex reasoning benchmarks. However, these approaches often rely on static prompting templates that do not adapt to task complexity or errors during the reasoning process. In this work, we introduce Adaptive Prompting, a dynamic and iterative framework designed to enhance reasoning by incorporating real-time adjustments to prompt structures and validation mechanisms. Experimental results demonstrate that Adaptive Prompting significantly improves performance on diverse reasoning benchmarks, including arithmetic reasoning (GSM8K, MultiArith), logical reasoning and commonsense tasks, achieving substantial accuracy gains compared to static prompting baselines. By integrating guided prompts, intermediate validation, and self-corrective steps, our approach enables smaller models to achieve competitive performance with larger counterparts, such as GPT-4, while maintaining computational efficiency. The framework achieves this without requiring fine-tuning or task-specific training data, highlighting the untapped potential of iterative reasoning methods.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces Dynamic Prompting, a framework that enhances the reasoning abilities of large language models (LLMs) by adapting prompt sequences and steps based on task complexity and model performance. Basically it uses a two-step prompting method to decompose the task and the method is tested on a group of reasoning datasets.

### Strengths
1. Using prompting to create a more efficient reasoning pipeline is helpful in industry.

### Weaknesses
1. The technical contribution is limited, as Chain-of-Thought (CoT) prompting[1] and task decomposition[2] are already well-established techniques in the field of LLMs. This paper simply use a prompt to let the LLM breakdown the question, then use the response to prompt again. 

2. The paper is not well-written. "Dynamic Prompting" is not clearly explained in the method section.

3. The quality is below the standard of ICLR. The method is too simple.


References:
[1] Wei, Jason, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V. Le, and Denny Zhou. "Chain-of-thought prompting elicits reasoning in large language models." Advances in neural information processing systems 35 (2022): 24824-24837.
[2] Shinn, Noah, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. "Reflexion: Language agents with verbal reinforcement learning." Advances in Neural Information Processing Systems 36 (2024).

### Questions
No questions.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper proposes a  real-time dynamic prompting method that can modify the number of prompting steps based on task complexity and model performance. The paper prsents empirical data that shows  that smaller LLMs can effectively leverage dynamic prompts to achieve high reasoning accuracy. As a conseqeunce, larger models may not be inherently superior to smaller LLMs if this approach is used.

### Strengths
The authors show a dynamic prompting approach that enables a system to guide the model through a detailed step-by-step process

### Weaknesses
Paper is incomplete; basis for experiments are not described with sufficient detail to understand what the results address.
Without a clear understanding of the method we cannot clearly assess the stated results.

### Questions
Can you please add more detail to help a reviewer to understand the details of your method. There is plenty of extra space.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
1

### Rating Number
1

### Confidence
5

### Summary
The authors present a prompting technique they call dynamic prompting and test it on the Gemma 9B model across ten common benchmarks. They compare the results of this technique to the results of zero shot CoT and manual CoT prompted GPT-3.5 Turbo and GPT-4.

### Strengths
* The authors test the technique across a large number of common benchmarks.
* The underlying problem of needing to massively scale up LLMs to achieve significant task performance is important and well-motivated

### Weaknesses
* The technique described in the paper ("Dynamic Prompting") is only described in very vague terms. No prompts are provided, and descriptions throughout seem to be contradictory. A detailed description of how the system works is necessary to evaluate this work's contribution and novelty.
* Note also that the description of "breaking down" a problem in one prompt and then actually solving it with another is very similar to previous work (e.g. Zhou 2023 "Least-to-Most Prompting", and many others). In general, this paper fails to engage with most previous work in this area and seems unaware of similar-looking techniques and especially of other systems that use multiple LLM prompts.
* Multiple overclaims:
  * "Dynamic prompting allows for fine-grained control over the model’s interaction with a task by breaking it down into tailored steps, adjusting the prompt based on the problem complexity or model performance." It is unclear how this is possible without some kind of grounded measure of complexity or external sound verifier.
  * "challenging the notion that larger models are inherently superior" This technique is not tested on larger models, nor is there any analysis of comparative costs or accuracy tradeoffs.
  * "It represents a shift in paradigm, where the emphasis moves away from building increasingly large models and towards optimizing the interaction between the model and the problem itself."
* The results look too good to be true (as reported in the Gemma 2 paper, Gemma 9B scores 68.6% on GSM8k. This paper reports an improvement to 98.7% with its proposed technique) and cannot be verified given the lack of useful detail about the mechanics of the prompt technique and lack of provided prompts.
* The paper would greatly benefit from removing much of the speculation (throughout) and future work, from toning down or eliding many of the claims made, and from adding in important details about the proposed contribution.

### Questions
* What was Gemma 9B's zero-shot and CoT performance on the given problems?
* How do GPT models perform when embedded in a "dynamic prompting" system?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
2

### Summary
The authors propose a dynamic prompting scheme where a two stage process is used to answer complex reasoning questions. The process first applies a prompt to the original query to generate a step by step breakdown of the question then this is added to the original context with a request to answer the question in a simple format. They argue that this dynamic prompting approach can improve the performance of smaller models over larger ones, and present some impressive looking results across a number of benchmark datasets.

### Strengths
The results look very promising.

### Weaknesses
The following issues are most significantly reduce my overall assessment of the paper:
* The authors claim to develop a novel dynamic prompting framework but don't clearly state what pre-existing dynamic prompting methods there are e.g. (Kojima et al, 2022).
* The method is described with the use of Figure 1, but the structure superficially looked very similar to previously proposed dynamic prompting methods. I think there might be something new here but this isn't clear from the description. It is the Guided prompt steps that are least clearly defined, which correspond to the main contribution of the authors as far as I can tell.
* The experiments compare two models with conventional CoT prompting, but these are both from the same family (GPT 3.5 and GPT-4) while the dynamic prompting condition is a different model (gemma 9B) which is not itself evaluated with conventional prompting approaches. I would expect at the least to see Gemma evaluated on the other prompting strategies. Otherwise we are not seeing like-with-like comparisons.
* It is also not entirely clear which variations of CoT are used for the baselines. I am guessing that the zero-shot variation is the one seen in  (Kojima et al, 2022) with the exact same choices as made by those authors . The other condition is called manual prompting but it isn't clear what this means, I am guessing this is the original method from (Wei et al 2020), described in (Zhang et al, 2022a) as manual prompting, but this is all left to the reader to work out. 
* There are no ablation studies or attempts to evaluate the importance of different aspects of the method. 
* This is a very crowded research space, and  certain concerns that may have a bearing on these approaches have arisen that are not mentioned in the paper. For instance, there is a concern that recent models have memorised key knowledge about the datasets against which these models are being evaluated, e.g. see (Zhang et al., 2022b) and (Srivastava et al.,2024), and this leads us to question how much these models are indeed *reasoning*. The language of the paper and the methodology would ideally reflect some of these concerns.


Srivastava, S., PV, A., Menon, S., Sukumar, A., Philipose, A., Prince, S., & Thomas, S. (2024). Functional benchmarks for robust evaluation of reasoning performance, and the reasoning gap. arXiv preprint arXiv:2402.19450.

Zhuosheng Zhang, Aston Zhang, Mu Li, and Alex Smola. Automatic chain of thought prompting in
large language models, 2022. (Zhang et al., 2022a)

Zhang, H., Li, L. H., Meng, T., Chang, K. W., & Broeck, G. V. D. (2022). On the paradox of learning to reason from data. arXiv preprint arXiv:2205.11502. (Zhang et al., 2022b)

(Other citations are as given in the paper)

### Questions
What is the precise method being used for dynamic prompting and how does this differ from previous dynamic prompting methods? Ideally, I would like to see an example including the potentially multiple steps in question breakdown.

### Soundness
1

### Presentation
1

### Contribution
2
