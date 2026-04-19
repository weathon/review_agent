# Advancing LLM Reasoning Generalists with Preference Trees

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6

## Abstract
We introduce EURUS, a suite of large language models (LLMs) optimized for reasoning. Finetuned from Mistral-7B, Llama-3-8B, and Mixtral-8x22B, EURUS models achieve state-of-the-art results among open-source models on a diverse set of benchmarks covering mathematics, code generation, and logical reasoning problems. Notably, EURUX-8X22B outperforms GPT-3.5 Turbo in reasoning through a comprehensive benchmarking across 12 test sets covering five tasks. The strong performance of EURUS can be primarily attributed to ULTRAINTERACT, our newly-curated large-scale, high-quality training data dataset specifically designed for complex reasoning tasks. ULTRAINTERACT can be used in both supervised fine-tuning, preference learning, and reward modeling. It pairs each instruction with a preference tree consisting of (1) reasoning chains with diverse planning strategies in a unified format, (2) multi-turn interaction trajectories with the environment and the critique, and (3) pairwise positive and negative responses to facilitate preference learning. ULTRAINTERACT allows us to conduct an in-depth exploration of preference learning for reasoning tasks. Our investigation reveals that some well-established preference learning algorithms may be less suitable for reasoning tasks compared to their effectiveness in general conversations. The hypothesis is that in reasoning tasks, the space of correct answers is much smaller than that of incorrect ones, so it is necessary to explicitly increase the reward of chosen data. Therefore, in addition to increasing the reward margin as many preference learning algorithms do, the absolute values of positive responses’ rewards should be positive and may serve as a proxy for performance. Inspired by this, we derive a novel reward modeling objective and empirically that it leads to a stable reward modeling curve and better performance. Together with ULTRAINTERACT, we obtain a strong reward model.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents EURUS, a new collection of large language models (LLMs) and a reward model (RM) designed to enhance reasoning capabilities. The authors develop ULTRAINTERACT, a dataset designed for complex reasoning tasks with 12 datasets spanning math, coding, and logical reasoning problems. ULTRAINTERACT employs preference trees, which pair each instruction with reasoning chains, interaction trajectories with feedback, and pairwise responses for preference learning.

The authors use ULTRAINTERACT to fine-tune several open-source LLMs, including Mistral-7B, Llama-3, and Mixtral-8x22B. They show that EURUS models achieve top performance on multiple reasoning benchmarks, including LeetCode and TheoremQA. EURUS-7B and LLAMA-3-EURUS-8B even surpass baselines 5 times their size, while EURUX-8X22B outperforms GPT-3.5 Turbo on 12 test sets.

They also create a reward model, EURUS-RM-7B, that excels on several reward modeling benchmarks and introduce a new reward modeling objective that merges the Bradley-Terry objective with an additional term to directly adjust the reward of chosen and rejected action

### Strengths
1. The paper introduces a novel dataset, ULTRAINTERACT, designed for complex reasoning tasks. It comprises instructions paired with preference trees, featuring reasoning chains, multi-turn interaction trajectories with feedback, and pairwise positive and negative responses. ULTRAINTERACT emphasizes complex and diverse reasoning patterns, encouraging models to break down problems into sub-problems and use tools to solve them. This dataset is a valuable contribution and can be useful for future research on LLM reasoning.

2. The proposed EURUS models achieve state-of-the-art performance on several reasoning benchmarks, demonstrating the effectiveness of ULTRAINTERACT and the proposed training methods. Notably, the smaller EURUS models outperform much larger baselines, showcasing their efficiency.

3. The paper provides valuable insights into preference learning for reasoning tasks. The analysis of reward patterns during training leads to a new reward modeling objective that improves performance, particularly on challenging problems. The authors highlight the importance of the absolute value of rewards in preference learning for reasoning, as opposed to just focusing on relative differences as in general conversation settings.

### Weaknesses
1. While the authors acknowledge the use of proprietary GPT models in data synthesis, they do not thoroughly analyze the limitations of relying on these models. It would be helpful to discuss the potential biases introduced by GPT models and explore alternative approaches for data generation that rely solely on open-source models. Though, it's worth noting that they attempt to address this by creating ULTRAINTERACT-v2 using only open-source models, which shows promising results.

2. In the paper, a few preference learning algorithms, since the preference pairs are collected in the ULTRAINTERACT, not running RL with the data seems like a big miss.

### Questions
1. In line 90-91, the statement is unclear to me. In 'a higher final reward often indicates a better reasoning capability', whose reasoning capability? Can you elaborate a bit more?

2. About the result remove in Table 3 due to data contamination. For some of the model has data contamination issue, the table suggests the TheoryQA is leaked, what about the rest dataset? If the rest doesn't has data contamination issue, should the result be compared? Without TheoryQA number, OpenChat seems like still a strong candidate.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper has several contributions. First, it builds a dataset on reasoning tasks that contain both correct and wrong steps. Second, it proposed a modified loss function for training a reward model that is better suited for reasoning tasks. Lastly, it trains a set of LLMs using the proposed dataset that have competitive performance on reasoning tasks.

### Strengths
- The paper is advancing open science by making the training data and model checkpoints public. Given the significant improvements in reasoning tasks, it is likely that these assets will be helpful to other researchers.
- The paper also proposes a new way of training reward models that is better suited to reasoning tasks. In addition, the training datasets have multi-step attempts that contain mistakes and tool usage, which is unlike other preference datasets.
- The experimental section is detailed and provides many interesting results, such as comparing three different preference optimization methods. There are many ablations provided, and evaluations are done on many tasks, which makes the results more convincing.

### Weaknesses
- The heavy reliance on GPT responses makes me feel like this is more of distilling GPT. Also, it is not clear what are the usage limitations that will arise from using a proprietary model like GPT4. As shown in tab7, this was crucial for obtaining good performance.
- The problem of the likelihood of chosen responses going down in reasoning is a known issue and studied prior work [1], which is not cited in the paper (the related work is quite short)
- The term “multi-turn action” was confusing. It seems that all the tasks require only a single correct response. None of the tasks is truly multi-turn where the model has to do multiple actions. From reading the paper, it seems the term “multi-turn” is used to describe a process where a model can try again if it makes a mistake. Actually, it is not clear how this process works, especially when training the model and evaluating it. Also, the dataset contains observations and judgements, but are they also used when training the actor? What about the python executions? There is very little detail on how the agent is trained on these and evaluated. 
- As mentioned in the previous point, there are certain steps that are not well explained. See the questions for examples. Given that the motivation is to advance open-source LLMs, I think it is important to describe the process of training in more details.

### Questions
- Is the reward model used in training the actor model?
- L148 “the actor model first decomposes the input problem into several problems” How is this done?
- L181 “we adopt more diverse reasoning patterns” How exactly is this done?
- Is python the only tool used?
- Typo in L263 reward notation
- What is "prompt level loose score" in L282
- I think the tables have too many numbers in them (tab3 has at least a hundred) and not sure if anyone will look at all of them. Instead, average scores can be put there and the detailed table can move to the appendix. This is only a suggestion though.
- Which GPT-4 model is used? I think there are multiple versions. 
- How is the reward model performance compared to ArmoRM?
- How is GPT-4 used as a reward model in tab4?
- Why does self-consistency drop in fig1 left?
- How is MCTS decoding done exactly in sec5.2?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors emphasize the performance gap between open-source LLMs and the most advanced models, particularly in reasoning capabilities. They attribute this gap to two primary factors: (1) the lack of high-quality datasets and (2) the under-exploration of preference learning techniques. To address this gap, the authors introduce a novel dataset, ULTRAINTERACT, which features a multi-turn, tree-structured format designed to enhance reasoning abilities. Additionally, they offer new insights into preference algorithms and reward modeling. They argue that effective reward modeling should consider not only the margin between rewards but also the absolute value of the reward itself.  Based on this insight, they propose a new reward model that combines two loss functions, L_{BT} and L_{DR}, demonstrating superior performance compared to existing models.

### Strengths
Authors use a new method to synthesize a dataset for SFT and preference learning, which could potentially enhance model's reasoning abilities. The intuition behind the synthesis method is straightforward and easy to be understood. I think the dataset is cool and it could be a potential approach for model to learn how to improve the response. Plus, the insights on preference learning algorithm is interesting.

### Weaknesses
1). I agree that providing trajectories to guide model improvements is a potential approach. However, during the training process, I believe that the vertical improvement information, sequential refinement across turns, may not be effectively learned. This is because current preference algorithms primarily focus on horizontal comparisons, assessing responses within the same turn. 

2). The reasons behind the better performance of EURES are hard to track and some studies will be necessary if authors want to claim that the proposed dataset is the reason. Because the baselines has different scales and training method, for example, their training dataset could have different size and their preference algorithm could be different, etc.. Plus if EURES can beat some larger model, the claim that the dataset is better will be more convincing.

3). There may be some factors contributing to the value differences observed in reward modeling, especially given the varying formulations of alignment methods. It would be valuable for the authors to offer insights into the potential reasons for these differences in the value of rewards.

### Questions
If the model is unable to effectively learn from vertical improvements, then it raises the question of why we want to synthesize the dataset with tree structure and why we are providing trajectories to the model.

### Soundness
2

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
3

### Summary
The authors explore improving large language model reasoning through the curation of high quality training data for that reasoning.
This data (UltraInteract) consists in preference trees, with nodes splitting on correct/incorrect responses; critique and refinement of rejected responses; and uses different reasoning schemas/ actor models to increase training data diversity
The actor used to generate these trajectories is GPT3.5 Turbo, with GPT4 being used as a critique model with access to an interpreter/tools.

The authors then use this dataset (alongside others) to finetune 3 language models using the following process:
1. SFT over the correct actions
2. Preference learning over correct vs incorrect actions using off the shelf preference learning algorithms

Additionally the authors also use this to derive a reward model:
3. Train a reward model, adding in terms for the difference in absolute rewards to the normal Bradley Terrey reward model.

In my view the key contributions of this paper are:
* introduction and analysis of preference-tree based instruction following data, which is scalable and effective
* introduction of improved objectives for training reward models

### Strengths
1. With regards to soundness, I feel that the necessary experiments have been run to validate the majority of claims, especially where those claims are with regards to methodological contributions. The authors have also taken pains to remove contaminated data from their work in order to make comparisons fair and meaningful, including when reporting others' work.
2. The presented language models have strong performance, and the data and reward models are in and of themselves useful contributions to the research community, removing some of the limitations of scale and quality from prior works creating preference datasets and reward models
3. The investigation surrounding the flaws of existing preference learning models is an original contribution.
4. In my view the largest contribution is the rather detailed study of creating their ultra-instruct dataset albeit moreso as an engineering challenge.
5. The experiments are run against meaningful baselines: models of similar scale, trained on similar data in similar ways.

### Weaknesses
1. As a minor point the spelling and grammar could be improved; for instance "Is proprietary models" (line 470) should be "Are proprietary models", and more generally things like "Perference Learning" (line 247). More substantially some of the references point to the wrong sections (e.g. the reference to section 5 (replaced with 6) (line 255) -- in this case harming readability (hence my review of the presentation...)
2. I feel that the modification to the reward model could be better motivated in section 3, for instance by referencing other works that maximise a similar margin loss. At the least it should be explicitly linked to the discussion in section 4.2 that actually seems to motivate it. This might be aided to seperating out the reward modelling section from the finetuning section? Since it seems to follow on more logically from the finetuning investigations
3. Section 6.1 doesn't really address the section title properly. While the performance itself does suggest that just training on open source data is sufficient (ignoring the instruction following benchmark); the body of the section just talks about mixing in this additional V2 data, and the ensuing performance gains. It would suffice to add a brief comment at the end of line 483 explaining the results of finetuning just on V2
4. As a general comment I feel that this work feels like three distinct pieces of work rather than a single cohesive one. I.e. the proposal of a new training dataset; a set of models finetuned on this dataset alongside others; and more separetely a reward model trained on a combination of dataset including the one proposed here. One way of mitigating this would be to focus on the contribution of the dataset to the reward modelling phase (using the data from the ablation studies).
5. Section 2. is a little bit confusing and could be rephrased to make it a little but clearer that it is all just an example.

### Questions
1. Did you conduct any comparative investigations over general conversational preference learning using your reward modelling objective? This would help to verify your intuition that this method is effective due to the unique features of reasoning tasks
2. Would it be possible to use the Eurus reward model for PPO-based alignment? How would this perform in comparison to the existing finetuning methods

### Soundness
3

### Presentation
3

### Contribution
3
