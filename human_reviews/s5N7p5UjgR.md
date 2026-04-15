# Markovian Transformers for Informative Language Modeling

- Decision: Reject
- Scores: 10, 6, 6, 5

## Abstract
Chain-of-Thought (CoT) reasoning holds great promise for explaining language model outputs, but recent studies have highlighted significant challenges in its practical application for interpretability. We propose to address this issue by making CoT causally essential to prediction through two key components: factoring next-token prediction through intermediate CoT text, and training CoT to predict future tokens independently of other context. This results in "Markovian" language models, where CoT serves as a fixed-size state for future token prediction. Our approach optimizes for "informativeness" – the improvement in next-token predictions using a trained CoT compared to a baseline. Using Proximal Policy Optimization (PPO) for arithmetic problems and policy gradient for GSM8K, we demonstrate effectiveness on both arithmetic problems with Mistral 7B and the GSM8K benchmark with Llama 3.1 8B, where the model learns to produce CoTs that are 33.20% more effective at predicting answers than the pre-trained baseline. The increased sensitivity of model performance to CoT perturbations provides strong evidence of CoT reliance. Furthermore, we show that CoTs trained for one model generalize to help other models predict answers, suggesting these CoTs capture reasoning patterns that transfer across different interpreters. This work advances the development of more interpretable language models, potentially enabling their extension to arbitrarily long contexts and enhancing AI reasoning capabilities across various domains.

## Human Reviews

## Human Reviewer 1

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
__Post-rebuttal update__:

After the rebuttal has concluded, I feel the need to express my strong support for this paper. I believe the proposed method has the potential to become an industry-defining standard, which ICLR should be proud to be the publisher off. The authors have done a lot to further improve the paper from a decent submission to an excellent submission that should be highlighted at a conference. While one can always conduct more experiments to support one's claims even more strongly, I think the remaining requests made by other reviewers are unrealistic. The paper should be accepted as is.

---------

The paper addresses an issue of Chain-of-Thought (CoT) reasoning in LLMs where the LM's final answer does not always depend on the CoT. The paper's idea is to enforce informativeness by conditioning the answer model on the generated CoT only without other context. To this end, the paper formally defines Markovian Language Models and (informative) update functions, from which the policy gradient procedure is derived. Applying the framework to the specific use case of CoT reasoning, the paper experiments with several RL techniques such as expert iteration, policy gradient, and PPO. The model is applied to a simple arithmetic task of adding 15 numbers as well as GSM8K, and shows that the model a) improves performance on the task b) is sensitive to perturbation in the CoT reasoning and c) produces CoTs that are sensible to a different language model such that its performance on the task is improved.

### Strengths
* The paper addresses an important limitation in Chain-of-Thought reasoning, which is of relevance to the broader ICLR community.
* The core idea is intuitive and simple.
* The paper is well written.
* The results on the simple arithmetic task and the math task are promising.
* The claims that the proposed method improves the generated CoTs in terms of interpretability and informativeness are well supported.

### Weaknesses
The method is evaluated only on few tasks and models, limiting how sure we can be that this is a useful method. Especially an application to language modeling would be very insightful and potentially extremely impactfull. However, while more is always better when it comes to experimental results, I think that this initial set of experiments support the ideas presented well and should suffice for publication.

### Questions
Please use different line styles in your figures so colorblind people can make sense of them. Otherwise Figure 2 and 3 are really hard to parse!

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work introduces and explains the construction of a Markovian Language Model to study causality in chain-of-thought reasoning. With a limited state (the previous state and its observation) on which to condition, the model is trained (fine-tuned) to maximize an informativeness objective through PPO, and is empirically shown to improve performance on mathematics tasks such as GSM8K and toy addition problems.

### Strengths
* The setup of this causally-guided model is pretty novel, and the finding that this improves performance by optimizing on an “information” metric is an impactful finding. 
* The selection of the RL training technique (PPO) is supported by highlighting the limitations of other considered methods (expert iteration and policy gradient).
* The work is written in a way that was simple to follow, which I appreciated.
* The gains on GSM8K (24.64% --> 35.71%) are meaningful (although, a bit tucked away in the paper's text).

### Weaknesses
* I understand the general intuition surrounding the design of the informativeness function, but it would be good to add some discussion on why the expected reward over the trajectory actually constitutes / addresses “informativeness” under your construction. 
* While math tasks have a more well-defined structure (the order in which their steps may be pursued), this is less clear for other tasks without such a clear structure in natural language, for instance. It would be good to examine this approach on at least one such task to further support the method’s general efficacy.
* Despite the intuition-based process of selection for the RL training strategy, there are recent works that advocate in favor of expert iteration and REINFORCE / vanilla policy gradient for LLM reasoning and RLHF [1, 2]. To this effect, including such approaches for comparison in the results section (or in the appendix) would strengthen the defense of the PPO method chosen. It would be helpful to include some evidence supporting the limitations posed.
* While I appreciate documenting the design choices in Section 4.3, some justification behind them would be beneficial, either through ablations (it’s fine for these to be in the appendix) or relevant references. 

1. Teaching Large Language Models to Reason with Reinforcement Learning. Havrilla et al. 2024 
2. Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs. Ahmadian et al. 2024

### Questions
* Is the space of states task-conditional? This isn’t apparent based on the formulation in Section 3.1 (and is unclear by the wording in line 162). If not, then it would seem that the set of relevant “CoT states” would be very sparse relative to the complete space. 
* As posed in the weaknesses section, does this method extend to other reasoning tasks (e.g. in natural language or code) whose structure is less “linear”?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a metric to measure the informativeness of CoT tokens, and then uses RL to train the model to generate highly informative CoT tokens, in order to improve the correctness of the final answer. Experiments in random addition problems and GSM8K math problems demonstrate the effectiveness of the proposed metric and RL methods. The paper also shows that more informative tokens will bring gains in interpretability.

### Strengths
The technical ideas, including the proposed metric and RL methods, are new, well-motivated, and technically reasonable.  

The experiments in random addition and GSM8K are positive.

### Weaknesses
The experiments are limited to a synthetic math problem setting and GSM8K, and the only trained model is Mistral 7B. 

The presentation needs a better organization. E.g., some major results are placed in the appendices, but training details are in the main paper.

### Questions
Have you tried other open-source models like llama? Not use CoT of Mistral in it, but use your method to finetune Llama.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper proposes a framework where the reasoning steps are used as fixed-size states, which limits the model’s context (text bottleneck), and force the model to use the reasoning steps as input. This method design is inspired by the fact that past CoT literature find that the final answer might not be sensitive to the CoT trace. In the experiment, the authors show that the model trained with this method is indeed more fragile against CoT pertubations.

### Strengths
- The Markovian framework blocks the model from attending back to the original question and force it to use the CoT context for generation. This provides a new view and framework for analyzing CoT effects.
- The reinforcement learning-based approach demonstrates improved performance on tasks requiring multiple steps, 
- The CoT steps generated from this method seem to be more interpretable, from two dimensions: 1) pertubation of the CoT could lead to more model errors 2) the reasoning can be carried over to another model.

### Weaknesses
-  While the approach improves the model’s reliance on CoT, it’s uncertain if this CoT is genuinely interpretable by humans. The transferrability between Mistral and Llama should only serve as an indirect proof.
-  It is actually fine to focus on QA task, but probably we'd like to see how this can generalize to more domains other than arithemtic. Would this paradigm also work for other reasoning task as well?
- There seem to be no baselines and ablation designed, so it is a bit hard to position the effectiveness of the method against other methods.
- The fragility analysis is insightful but lacks depth. A more detailed investigation into which types of perturbations impact CoT reliability most could provide valuable insights.
-  Writing-wise, the paper writing is clean, but probably some adjustment of the sections flow and emphasis would be nice.
   -  For instance, while the method section is quite detailed, it’s presented before establishing the limitations of existing CoT techniques clearly, which makes it harder to understand the innovation.
   - There are few tables but quite a few definitinos and equations. I'd suggest consider streamline the method descriptions and move some of them to the appendix, while adding more discussion and insights in the main body.

### Questions
- Is there a way to combine the interpretability with actual human perception? Though informativeness here can be used to improve model quality, it is also very helpful from human level. This is probably mentioned in F. But it seems to me F is more about how to encode human interpretability in training.
- In F it is mentioned that "optimal CoT would be a compression of the question, which can potentially be difficult for humans ". Is this observed in your experiments?
- Were there more ablation study or comparison conducted?

### Soundness
3

### Presentation
2

### Contribution
2
