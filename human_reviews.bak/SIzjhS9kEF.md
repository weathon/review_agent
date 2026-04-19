# Revisiting the Superficial Alignment Hypothesis

- Decision: Reject
- Scores: 6, 5, 6, 6

## Abstract
The Superficial Alignment Hypothesis posits that almost all of a language model's abilities and knowledge are learned during pre-training, while post-training is about giving a model the right style and format. We re-examine these claims by empirically studying the scaling behavior of post-training with increasing finetuning examples and evaluating them using objective task-specific standardized benchmarks. Through experiments with the Llama-3, Mistral, and Llama-2 model families of multiple sizes, we observe that, similar to the pre-training scaling laws, post-training task performance scales as a power law against the number of finetuning examples. This power law relationship holds across a broad array of capabilities, including mathematical reasoning, coding, instruction following, and multihop-reasoning. In addition, for tasks like math and multihop reasoning, we observe that a handful of examples merely align the model stylistically but do not saturate performance on the benchmarks. Model performance is instead correlated with its reasoning ability and it improves significantly with more examples, illustrating the need for holistic evaluation programs leveraging objective benchmarks in addition to measurement of alignment to human preferences. We also observe that language models are not necessarily limited to using knowledge learned during pre-training. With appropriate post-training, a model's ability to integrate new knowledge greatly improves on downstream tasks like multihop question-answering. Taken together, these results shed new light on the Superficial Alignment Hypothesis, suggesting that it is, at best, an over-simplification.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The author revisit the validity of Superficial Alignment Hypothesis made in previous work (Zhou et al., 2024). Specifically, they try to show the ground for the hypothesis is shaky because it was established on running model with chat-style data, which do not requires more knowledge. Instead, the authors choose more knowledge-intense tasks, and shows that model do learn new knowledge (beyond what's in pretraining) and such observation scales w.r.t. task data, model sizes.

### Strengths
* the work study the scaling factors in post-training alignment to better counter-argue superficial alignment hypothesis. This seems a novel thing that other people haven't broadly tried. 
* the author basically revise the SAH to be conditional on the what it really takes to accomplish some downstream tasks. If the downstream tasks don't require more knowledge, then the alignment is only style-wise. This is a good message, although a bit obvious.
* showing that win-rate and task accuracy differs is an interesting observation.
* error analysis does seems good and comprehensive;

### Weaknesses
* Missing URIAL prompt [1] as a training-free alignment baseline, since the model is trained on LIMA-1k, it adds some complexity. 

* The authors should also make the investigation/claim a bit more systematic or formal. In terms of capability X, how does superficial alignment hypothesis not hold (+ some analysis on error), how does the observation in capability  X different from observation in capability Y. I think the problem statement is not so well-stated.

* Some claims are not well-supported and overselling:
Line 310-314: an alternative explanation is that the model is getting there? Also, how does the overall accuracy changes? Are the number of incorrect responses decreasing? I don't think the author could conclude much here.

Line 316: the claim in Line 312-314 and here are cyclic and contradictory. The author is essentially saying "the model do align in intermediate steps/format/style, but the improvement is nothing and superficial" (Line 312-314); but, "look, we are making improvement and this tell us even more about superficial alignment hypothesis is bad". Here the same observation of "making improvement" are interpreted in two completely different standpoint, without good rationalization.

Line 370 "this post training is meant to impart reasoning ability": but does it actually improve reasoning ability? It sounds more like the authors are building their own instruction-tuned model instead of using an off-the-shelf one.

Line 404 "reasoning helps", Line 482 "these improvements are driven by the model’s reasoning ability": reasoning is a broad term here, and authors only test one kind of reasoning. thus, the claim is stronger than the supporting evidence.

* The author also didn't do thorough survey for related work. For example, [2] also study the same topic (maybe not explicitly mentioning "superficial alignment hypothesis") in a more thorough and focused manner.

[1] The Unlocking Spell on Base LLMs: Rethinking Alignment via In-Context Learning
[2] Instruction-tuned Language Models are Better Knowledge Learners

### Questions
* Section 3.1; could author explain why do they put heterogeneous data in the same run? Might be good to have some scaling for individual tasks, to tease apart co-learning across tasks. But this is minor and doesn't change the overall weakness.

* I note that alignment is a bit overloaded term. The purpose of the alignment could be for human-like answer, but human-like doesn't imply stronger capability. It could be interpreted that LiMA conclude the hypothesis on a specific dataset (chat-style) and should not be taken too far from its original setting?

* In Table 2 and Figure 2, instruction tuning with LiMA also seems beneficial nonetheless, where Math 10->14.7; Multihop QnA 10->21. It's just that the training instances are not as much knowledge-intense and targeted as task-1k. Doesn't this shows that the benefit of Lima data?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper presents an empirical investigation to re-evaluate the superficial alignment hypothesis, which posits that the majority of a large language model's (LLM's) abilities and knowledge are acquired during pre-training. By conducting extensive experiments with diverse pre-trained models (e.g., LLaMA) and fine-tuning tasks (e.g., mathematical reasoning), and incrementally increasing the number of fine-tuning examples, the authors demonstrate that the fine-tuning scaling law applies across a wide range of capabilities. This finding challenges the original Superficial Alignment Hypothesis, which asserts that post-training does not impart new capabilities to a model.

### Strengths
This paper aims to revisit the superficial alignment hypothesis, which posits that post-training primarily concerns style and format adjustments, while most capabilities and knowledge are learned during pre-training.The authors undertake comprehensive experiments to explore the scaling law in post-training for various reasoning abilities, thereby illuminating a feasible pathway for enhancing the advanced capabilities of large models.

### Weaknesses
The superficial alignment hypothesis has not gained widespread acceptance in practical LLM pre-training. As stated in the Llama 3 technical report [1], the importance of improving performance for specific capabilities such as code/math/reasoning/tool use, etc., during post-training is emphasized. Similar conclusions have already been observed in existing technical reports or papers (e.g., the Figure 2 of paper [2]). This suggests that the re-evaluation presented in the paper may not be as novel as claimed.

Additionally, work [2] delves into how each ability varies due to the composition of supervised fine-tuning (SFT) datasets. The authors investigate this using varying amounts of mixed data and compare the results with individual ability performance, as illustrated in the Figure 3. However, the current submission only mentions multi-task settings as a potential future direction.

The distinction between pre-training and post-training has become increasingly blurred, especially with the widespread adoption of synthetic data. It is often challenging to definitively categorize synthetic question-answering pairs as either pre-training or post-training data. Moreover, there is typically an intermediary process between pre-training and post-training. Consequently, the exclusive focus on the scaling of post-training may be overly restrictive.

[1] The Llama 3 Herd of Models, https://arxiv.org/abs/2407.21783

[2] How Abilities in Large Language Models are Affected by Supervised Fine-tuning Data Composition, https://arxiv.org/pdf/2310.05492

### Questions
The paper would benefit from expanding the size of the post-training dataset. Given that the authors are investigating the post-training scaling law, using only the Math dataset as an example for mathematical reasoning is quite limited. It is recommended that the authors consider augmenting the dataset through the use of synthetic data or by constructing a larger, more comprehensive dataset to study the scaling law (e.g., Evol-Instruct https://arxiv.org/abs/2304.12244 and WizardMath https://arxiv.org/abs/2308.09583). This would provide a more robust and comprehensive analysis of the scaling behavior in post-training.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This papre studies the superficial alignment hypothesis that  a language model's abilities and knowledge are learned during pre-training, while post-training is about giving a model the right style and format.  The authors investigate this hypothesis by assessing scaling laws in post-training across multiple tasks and model sizes, showing that fine-tuning enhances task-specific performance in reasoning and coding beyond stylistic alignment. Experiments show that while stylistic improvements are quick to saturate, task-specific reasoning, mathematical ability, and knowledge integration continue to benefit significantly with additional fine-tuning data.

### Strengths
1. The paper conducts a comprehensive evaluation using well-defined benchmarks,  providing evidence that post-training improves more than stylistic alignment.
2. New insights: by analyzing the power-law relationship in post-training, it shows that model improvements align well with scaling laws observed in pre-training.

### Weaknesses
- GPT-based Evaluation: Using GPT-based evaluation for alignment and error analysis introduces potential biases.
- Missing Citation: there are some related work that also demonstrate the scaling law in the post-training stage, sucu as this one (https://openreview.net/forum?id=MpCxUF8x61).

### Questions
1. How does the model's performance change when it is post-trained on datasets with diverse task formats, beyond those used here?
2. What are the underlying mechanisms that allow LLMs to integrate new knowledge effectively during post-training?
3. What are the long-term implications of extensive post-training on a model's performance across tasks? Does it risk overfitting to specific tasks or styles, potentially diminishing its generalization capabilities?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work challenges the superficial alignment hypothesis, which is defined by three rules:  1) A model's knowledge is learned entirely during pre-training. 2) A small number of examples can saturate a model's performance for a given task. 3) Post-training is largely about style and doesn't does not teach a model new capabilities. This work illustrates a post-training scaling law and offers several insights, claiming that the model can learn new skills during the post-training and it also requires much more training data.

### Strengths
This work presents in-depth analyses, offering insights for understanding the post-training process. It finds that the format-related errors are often saturated with a few samples, but the reasoning ability requries much more examples to improve.

### Weaknesses
Since this work aims to study the scaling law of post-training, the choice of task is limited by the size of the training set. However, the training set is still small for building a reliable scaling law. One solution is to verify the idea of synthetic tasks. 

Except for several insights of analyzing the detailed post-training process, the main findings (the key takeaways section) are not a surprise; some of them have been discussed in the community. The contribution of this work is more like to be a verficiation and provide more evidence. Besides, this work uses the term "post-training" frequently, but the actual experiments are limited to SFT rather than RLHF.

### Questions
see weakness
typos， the C3 and C2 at L055 are in the wrong order.

### Soundness
3

### Presentation
2

### Contribution
2
