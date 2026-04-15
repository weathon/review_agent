# The Impact of Depth and Width on Transformer Language Model Generalization

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 3, 6

## Abstract
Transformer language models tend to perform better the more parameters they have. Previous theoretical and empirical work suggests that the total number of parameters is not the only relevant factor, however; rather, expressivity and out-of-distribution generalization may benefit more from increasing depth than increasing width. To test this hypothesis we disentangle depth from the number of parameters, constructing families of models which trade off depth for width while keeping the total number of parameters constant. We pretrain those models and evaluate them on both language modeling and compositional generalization tasks. We report three main conclusions: (1) within each family, deeper models show better language modeling performance, but the relative benefit of additional layers diminish rapidly; (2) when fine-tuned on compositional generalization tasks, deeper models generalize better out-of-distribution than shallower models do, but returns are similarly diminishing; (3) the benefits of depth for generalization cannot be attributed solely to better performance on language modeling or in-distribution data. These results replicate in three different model families (41M, 134M and 374M parameters), suggesting that depth improves performance across model sizes.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies the effect that LLMs’ depth has on its performance on compositional genalization and language modeling tasks. To disentangle the effect of depth on performance from other factors, the authors fixed the total number of parameters of LLM by reducing the size of the transformer’s feed-forward dimension (d_model) while increasing the layers of transformer. Experiments here showed that deeper LLMs result in better language modeling and compositional generalization up until when the LLM becomes too narrow when d_ff<d_model. The authors also conducted more experiments to show that the better compositional generalization by deeper LLMs is not simply due to better language modeling performance by using pretrained deeper LLMs that have similar perplexity than the shallower counterpart. Experiments are conducted on 3 model size classes, pretrained on C4 corpus and 4 compositional generalization tasks (COGS, COGS-vf, GeoQuery, English passivization).

### Strengths
+The paper’s empirical findings contributes to the body of work that seek to better understand how to train LLMs most efficiently by choosing the best mix of model hyperparameters given a particular computational budget.

+Experiments are designed well to disentangle possible confounders (language modeling performance etc).

+The paper is generally well-written and easy to follow.

### Weaknesses
-The paper’s core contributions centers around mostly confirming existing findings (e.g. Mueller et al. (2022) and Tay et al. (2021)) with empirical results that a bigger depth improves expressiveness of neural network or LLMs, limiting the impact of the work. Making it more obvious what is different from these prior work will help readers better appreciate the paper’s contributions (e.g. in-depth analyses about why this occurs beyond empirical results on performance).  

-The experiments focus only on compositional generalization and language modeling tasks while there is a plerotha of other tasks that can be used to evaluate LLMs’ generalization capabilities.

### Questions
Can compositional generalization and language modeling tasks along stand to evaluate the generalization of the LLMs (or mostly only compositional generalization)? It would be helpful to discuss the different types of generalization if the paper is claiming generalization as a whole beyond compositional generalization.


==Post-Rebuttal==
I appreciate the authors' response and decided to keep my score.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper provides a controlled study disentangling model depth from the width and total parameters. The results support the view that depth improves generalization in transformers, with diminishing returns past a shallow threshold. The paper makes a solid contribution to understanding model architecture choices for generalization. Overall, the paper makes a valuable contribution by investigating the impact of depth on the generalization ability of Transformer language models. However, addressing the following weaknesses would enhance the comprehensiveness and applicability of the research.

### Strengths
By investigating the effect of depth on the model's generalization ability, the paper provides a valuable reference for improving and optimizing the design of language models.

### Weaknesses
1. Since the paper mainly verifies the effect of the Transformer’s “depth” on combinatorial generalization, the "depth and width" in the title of the paper is misleading.
2. While the paper primarily investigates the effect of depth, the impact of width on generalization is not extensively explored. It would be beneficial to analyze the trade-offs between depth and width and how they interact in terms of model performance and generalization.
3. The paper does not thoroughly discuss the computational implications of increasing depth or width in Transformer models. Considering the computational cost associated with deeper models, it would be useful to analyze the trade-off between improved generalization and increased computational requirements.

### Questions
1. Please double-check the reference format and standardize it.
2. In the paper, you focus on the impact of depth on Transformer language model generalization, but the analysis of width is relatively limited. Can you provide further insights into the trade-offs between depth and width? How do these two factors interact in terms of model performance and generalization? It would be helpful to explore the joint effects of depth and width and their relative importance in achieving better generalization.
3. You only conduct a single run for each experimental condition. Adding multiple runs would strengthen conclusions by quantifying uncertainty and ruling out run-specific fluctuations. Is it feasible to do multiple runs, even if a subset of conditions?
4. Is there an optimal depth where returns diminish for all model sizes and domains? Or does optimal depth keep increasing with the model scale?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper empirically studied the impact of increasing depth and width on the model's out-of-distribution generalization performance.

### Strengths
1. The paper provides some interesting experiment results which might be useful for future research.

### Weaknesses
1. The result is a bit too straightforward with only experiment results. More theoretical analysis on the difference between increasing depth and width on out-of-distribution generalization is required for a paper on venues such as ICLR.
2. Why do the authors choose to focus on decoder-only models? What can be the difference between encoder-decoder models and decoder-only models on the impact from different depths and widths?

------- post rebuttal ------
I have read the rebuttal. 
Although empirical study is also important for machine learning research, the paper lacks some insightful new information for the community. 
The rating should be between 3 or 5 but there isn't an option for 4. I'll keep my rating.

### Questions
Please see the weakness part.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper  studies how performance for "compositional generalization" in Transformers varies as a function of depth. Its main twist is to pay careful attention in keeping the number of parameters constant. Hence, when augmenting depth, it is reducing width accordingly. This is done for 3 different number of parameters.
Inspecting the results, my take-aways are the following: performance _does_ systematically get better with deeper models as long as they don’t become so narrow so as to have a width that requires reducing input dimensionality. This said, depths=3-6 look largely enough for all practical purposes, and the main way to get better performance is just to increase the number of parameters, which matches usual knowledge.

### Strengths
The paper asks a clear question: how does performance vary as a function of depth vs width for a given and fixed number of parameters for transformer based architectures on LM.
The paper provides a very clear and rigorous treatment of this question, also providing relevant literature and areas of further investigations.
I particularly like one of the final questions that is asked about "alternative approaches to controlling for total size". Universal transformers are quite an extreme way to go, with all layers sharing the same weights. Maybe you could find some alternative way, for instance by repeating blocks of layers instead of just one. Likewise, I wonder about hypernetworks. They could be used to fill out huge networks, but then constraining the number of parameters.

All in all, I think the paper may be interesting to some persons, at least as a reference on that precise question it is asking.

I think the paper is just as good as it gets to answer this question for any person that could be interested in the topic. For this, my pick is it should be accepted.

### Weaknesses
- Applicability of the study is arguable a bit weak and I would say that it mostly would serve as a reference for what is usually considered common knowledge without any rigorous treatment: "for a given parameter budget, pick depth over width".
- It remains extremely clear from this paper that beyond very small depth (as soon as we get 3~ layers), performance doesn’t really go up with depth alone: the way to go is just to add more parameters.
- As a practitioner, I would be interested by the following question: what about if my budget is not really in terms of number of paremeters, but rather in compute power or memory? Do you see the same thing happening that one should pick depth?
- p8: "when studying the the impact"

### Questions
I am not sure about the questions I should ask, since the paper really looks pretty clear to me. I guess it’s more about what’s next. Personally I didn’t find the 3rd and 4th limitations very illuminating, but liked the 2nd.

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
2 fair
