# AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models

- Avg Score: 8.00
- Decision: Accept (Oral)
- Scores: 8, 8, 8, 8

## Abstract
Large language models (LLMs)  often exhibit hallucinations, producing incorrect or outdated knowledge. Hence, model editing methods have emerged to enable targeted knowledge updates. To achieve this, a prevailing paradigm  is the locating-then-editing approach, which first locates influential parameters and then edits them by introducing a perturbation. While effective, current studies have demonstrated that this perturbation inevitably disrupt the originally preserved knowledge within LLMs, especially in sequential editing scenarios.
To address this, we introduce AlphaEdit, a novel solution that projects perturbation onto the null space of the preserved knowledge before applying it to the parameters. We theoretically prove that this projection ensures the output of post-edited LLMs remains unchanged when queried about the preserved knowledge, thereby mitigating the issue of disruption. 
Extensive experiments on various LLMs, including LLaMA3, GPT2-XL, and GPT-J, show that AlphaEdit boosts the performance of most locating-then-editing methods by an average of 36.7% with a single line of additional code for projection solely.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Post training an LLM can often cause it to disrupt its originally preserved knowledge. To circumvent this, AlphaEdit utilizes Null Space Projection to preserve old knowledge while injecting the new knowledge effectively. Results show that AlphaEdit significantly reduces the domain shift between pre and post edits compared to existing methods.

### Strengths
1. The paper is well-written and easy to follow. The authors explained the null space and how to leverage null space projection to optimize the model editing objective well.
2. I think the choice of RQs is well thought out and thorough as well. The paper answered most of the questions I had about AlphaEdit.
3. I think figure 6 is interesting to show how AlphaEdit can generalize to existing methods.

### Weaknesses
1. The paper did not mention the correlation between the accuracy and the dataset size. More concretely, how much data is needed for AlphaEdit to work well?

### Questions
1. In line 174, ‘B is in the null space of B’ -> change to ‘B is in the null space of A’
2. For figure 7(a), what does the ylabel refer to?
3. Minor presentation suggestion: In figure 5, the pre and post edited distributions are difficult to set apart due to color choice. Picking two contrasting colors similar to Memit would be great for the final version.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces AlphaEdit, a method to improve targeted knowledge editing in large language models by projecting updates onto the null space of preserved knowledge, thus reducing interference with existing information. AlphaEdit achieves this with a minimal adjustment in code, enabling it to maintain a model’s pre-existing knowledge while updating targeted information. Experimental results demonstrate AlphaEdit's effectiveness, showing a performance improvement over traditional editing methods across multiple language models.

### Strengths
1. The use of null-space projection in AlphaEdit minimizes disruption to preserved knowledge while updating new information, effectively addressing a common trade-off in model editing between knowledge update and retention.
2. The paper provides comprehensive experimental evidence that AlphaEdit outperforms existing methods on critical editing metrics such as efficacy, generalization, specificity, fluency, and consistency.

### Weaknesses
1. Accurate null-space projection may rely on high-dimensional matrix computations, which could pose scalability issues as model sizes or knowledge bases grow.
2. Limited empirical evaluation on diverse LLMs. The method is tested on models like GPT-2 XL, GPT-J, and LLaMA3 only. It would be good to see results for other models such as gemma, phi.

### Questions
1. Authors may have overlooked these methods. There are other latest methods present such as SERAC, GRACE, InstructEdit, MELO methods. Authors either provide the compared results or argue on why they have not considered these methods for comparison.
https://sites.google.com/view/serac-editing
https://arxiv.org/abs/2211.11031
https://arxiv.org/abs/2402.16123
https://arxiv.org/abs/2312.11795

2. Can authors show the results on  KnowEdit dataset as well?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This is a review of the paper entitled “AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models” submitted to ICLR 2025. The paper suggests a new approach to do targeted knowledge updates in LLMs; in particular, if an LLM tells some wrong factual information, the goal is to identify influential parameters and then introduce so-called perturbation to them that, on the one hand, repairs problematic outputs and, on the other, keeps the rest as intact as possible. The main experimental result is that the new suggested method, called AlphaEdit, performs comparably to the state of the art for single updates and shorts sequences of updates, but outperforms them dramatically for longer ones.

### Strengths
I should start by admitting that I am not a specialist in the topic of the paper, and so it is difficult for me to judge the novelty and value of the results. However, I can say that the paper is well-written: I could understand nearly everything and agree with the arguments. Moreover, for an outsider, the results look interesting and promising. Thus, I lean towards acceptance; however, of course, the opinions of reviewers who are more in the topic should be more valuable for the decision.

### Weaknesses
—

### Questions
Concrete comments:

Question: why do we need to solve the sequential editing task really sequentially, as in L (line) 244—that is, why cannot we just start from scratch with the original model and K1 being all the edits together (i.e., the union of Kp and K1 in the current equation (12)), and use Equation (11)? 
This seems to promise a better performance even for the AlphaEdit, looking at Figure 4.

Minor:
L 11: I do not think “due” is the right word here. 
L 76: “the coefficient” is unclear. Which coefficient?
L 81: besides viewing in colour, one needs a magnifier to read these figures.
L 136: “update” and “new” do not make much sense together, either one or another.
L 173: B -> A
L 177 (equation (7)): \Delta is not really \Delta here, it is projected \Delta. Which should be said or better denoted by another symbol.
L 193: it is not clear what does it mean to be “consistent” in this context

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces AlphaEdit, a novel method for knowledge editing in large language models (LLMs). The primary goal of AlphaEdit is to enable targeted knowledge updates while minimizing the disruption of existing knowledge. The authors propose projecting perturbations onto the null space of the preserved knowledge before applying them to the model parameters. This approach theoretically ensures that the output of the edited LLM remains unchanged when queried about the preserved knowledge, thereby mitigating the issue of knowledge disruption. Extensive experiments on various LLMs, including LLaMA3, GPT2-XL, and GPT-J, demonstrate that AlphaEdit significantly boosts the performance of existing model editing methods by an average of 36.4% with minimal additional code.

### Strengths
- The concept of projecting perturbations onto the null space of preserved knowledge is innovative and addresses a significant challenge in the field of knowledge editing for LLMs. The theoretical foundation provided in the paper is robust and well-explained.
- The authors conduct extensive experiments on multiple representative LLMs, demonstrating the effectiveness of AlphaEdit. The performance improvements are substantial and consistent across different models.

### Weaknesses
1. Well, actually I think the work is great and I donot see any weakness, the thing is that I think the author can do more benchmarks like the LongformEvaluation, MQUAKE which consider some more knowledge utilization ablity for knowledge editing. 
But the current evaluation is good enough.

[1] Long-form evaluation of model editing

[2] MQuAKE: Assessing Knowledge Editing in Language Models via Multi-Hop Questions

### Questions
N/A

### Soundness
4

### Presentation
3

### Contribution
4
