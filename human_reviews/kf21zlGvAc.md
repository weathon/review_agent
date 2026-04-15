# Prompt-Based Length Controlled Generation with Reinforcement Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 5, 5

## Abstract
Large language models (LLMs) like ChatGPT and GPT-4 have attracted great attention given their surprising performance on a wide range of NLP tasks. Length controlled generation of LLMs emerges as an important topic, which enables users to fully leverage the capability of LLMs in more real-world scenarios like generating a proper answer or essay of a desired length. In addition, the autoregressive generation in LLMs is extremely time-consuming, while the ability of controlling this generated length can reduce the inference cost by limiting the length.
Therefore, we propose a prompt-based length control method to achieve high-accuracy length controlled generation. In particular, we adopt reinforcement learning with the reward signal given by either trainable or rule-based reward models, which further enhances the length-control ability of LLMs by rewarding outputs that follows pre-defined control instruction. To enable rule-based inference, we also introduce standard prompt extractor to collect the standard control information from users' input. Experiments show that our method significantly improves the accuracy of prompt-based length control for summarization task on popular datasets like CNNDM and NYT. Both the standard prompt extractor and the RL-tuned model have show strong generalization ability to unseen control prompt templates.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper investigates length-controllable summarization. First (“+Prompt”), the proposed method controls the summary length by indicating the desired length in the prompt (similar to Fan et al., 2018). Second (“+RL”), to add/enhance the length-controllable capability of summarization models, the paper applies the RL method (PPO algorithm) to fine-tune GPT models using a rule-based reward. The reward simply compares the length of the generated text against the desired length, which is specified in the input prompt, and the desired length is extracted from the input prompt using a BERT/GPT-based model. Third (“+Filter”), at the inference stage, multiple summaries are sampled, and the output is the one that yields the highest reward. 

The experiments were conducted on CNN/DailyMail and NYT which are standard news summarization datasets. The paper selected three sizes of GPT models (124M, 355M, 774M) as the backbone and fine-tuned these models using their proposed methods. The prompt templates were manually crafted covering many standard length-control prompts. The results show improvements over the standard prompting method (similar to Fan et al., 2018) in terms of achieving the target length while maintaining ROUGE scores.

### Strengths
1) The paper shows improvement in length-control ability while maintaining the ROUGE scores.

2) The paper proposes and investigates different prompt extractors, and shows that a BERT-based model achieves perfect accuracy in both seen and unseen prompts.

3) The paper is the first (or one of the first) to apply the PPO algorithm to length-controllability in summarization.

4) The ablation study shows that a simple rule-based reward performs better than model-based rewards.

### Weaknesses
1) The main contributions of this paper are very incremental. For example,

    - 1.1) Controlling the length by input prompts has already been done by (Fan et al., 2018) and CTRLsum (He et al., 2022).
    - 1.2) Applying RL to controlling the length has already been done by (Bian et al., 2019)
    - 1.3) Sample filtering can be considered (I believe) as a weaker version of minimum risk decoding e.g., Freitag et al., 2022
    - 1.4) The relevant references  CTRLsum (He et al., 2022) and  (Bian et al., 2019) are missing in the paper

2) The paper mentions LLMs (e.g., GPT-4, LLaMA, etc.) which are much larger and more capable than the baseline selected in this work (GPT). So, I’m not sure if the findings in this paper would transfer to those larger models (with emergence properties). I believe these larger models are becoming more accessible to researchers now, so I’m quite surprised about the model choice in this paper. Also, there are other more commonly used models such as BART, T5, and Pegasus which have fine-tuned weights on summarization tasks.

3) This paper doesn’t compare against any existing methods. The authors list some existing approaches in Section 2.2; however, in the experiments, none of them are compared against.

Note that the weaknesses #2 and #3 are minor compared to weakness #1.  

References:
- (He et al., 2022) CTRLsum: Towards Generic Controllable Text Summarization
- (Fan et al., 2018) Controllable Abstractive Summarization
- (Bian et al., 2019) Controllable length control neural encoder-decoder via reinforcement learning
- (Freitag et al., 2022) High Quality Rather than High Model Probability: Minimum Bayes Risk Decoding with Neural Metrics

### Questions
What are your thoughts regarding the weaknesses? How do you think this approach could be applicable in the era of large language models (with stronger emergent abilities such as length control)?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper aims to control the length of generated summaries from a language model. The authors approach the goal by a prompt-based method that uses a rule-based reward function and PPO fine-tuning. The experiments conducted on the GPTs (with 124M, 355M, and 774M parameters) demonstrate that the proposed method can fine-tune the LM to be more able to be controlled the length through prompt.

### Strengths
* This paper proposes a reasonable method to control an LM to generate response with a length condition. This method is simple and can be effective. Specifically, this method mainly adopts PPO to optimize the LM with the authors’ designed rewards. The authors propose two variations as the reward function: (1) A standard prompt extractor (SPE) plus a rule-based reward function (Table1); (2) A GPT2/BERT-based trained reward model. Both variations use the synthetic, designed standard control prompts (SCP) to train the SPE or the reward model. The authors also propose to use the above reward functions to further select the generated summaries in the end.
* The experiments contain multiple quantitative analyses for reference. They already include the comparison among different control types, and out-of-domain length condition prompt templates.
* Most parts of the paper are clear.

### Weaknesses
* While this paper puts emphasis on LLM, the experiments use models with 124M, 355M and 774M, which can be controversial to be claimed as LLM. The behavior of an LM can be significantly different when the size is in Million and Billion scales. Also, which GPT is used as the main model is not specified. Because the paper only mentions the word “GPT”, I will guess it is the GPT1 (Radford et al., 2018) or the GPT2 used for the SPE.
  * Radford, Alec, et al. "Improving language understanding by generative pre-training." (2018).
* Novelty, or writing issue: Subsection 3.4 turns out to be an introduction to PPO instead of a proposed method. The added KLD penalty is also a variation proposed in (Schulman et al., 2017) and similar kinds of KLD penalty has been also added to PPO in prior work, such as (Ziegler et al., 2019). The authors can consider reorganizing the section.
  * Ziegler, Daniel M., et al. "Fine-tuning language models from human preferences." arXiv preprint arXiv:1909.08593 (2019).
* Technical issue: The definition of the advantage function is not conventional here. Specifically, A is often defined as Q(s,a)-V(s) or r + \gamma V(s’) - V(s). But in Section 3.4, the authors say the A is r - V(s,a). First, V is usually used for the state value function, whose input will only have the state. If the input includes both state and action, it is often said to be the state-action (Q) value function. Therefore, I'm wondering that is the V(s,a) should be V(s) here or the advantage used in the experiment is actually r - Q(s,a)?
* The experiments can have one baseline that does NOT use length condition prompts to fine-tune the model. This baseline can help readers understand how an “original” setup model performs on the test sets.
* The experiments miss some important details. I have checked the appendices but haven’t found them.
  * How many samples are generated for the sample filtering?
  * What is the used sampling method in the inference stage, including the hyperparameters?
* More discussion needed:
  * How do the authors view that RL+filter (BERT) receives the best BERTScore in Table 7 and 16? 
  * What do the generated examples look like?
  * What kind of errors can happen? Is there a case that the summary is within the given length but the summary is actually not complete?

### Questions
* Some typos examples:
  * In Introduction:  “It is expensive to use human for labelling…” → labeling
  * In Section 3.2: “The Appendix A.5.3” should be A.5.4 in the manuscript.
  * In Section 4.4: “THe results are give in Tabel 6” → given

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this work, for length controlled generation, The authors introduce a prompt extractor to obtain a standard control prompt, which contains metadata for controlling the length, from arbitrary user input. They define a set of standard length control types along with their corresponding rule-based reward functions. The pretrained LMs are finetuned to output while considering the standard control prompt through a modified PPO using the specified reward functions. Experimental results demonstrate an enhanced control accuracy while preserving the ability to perform downstream tasks in two summarization tasks.

### Strengths
- The proposed method is simple and efficient to control output length of LMs.
- The paper clearly defines a set of standard control types with appropriate reward functions.
- The paper is well written and easy to follow.

### Weaknesses
- It appears that there is a significant improvement in the control settings of 'Equal' and 'Between' when considering the core setting between `Prompt` and `Prompt + RL`. However, it remains unclear whether the improvement persists when the method is integrated into larger LMs such as LLaMA. This limits the extent of their contributions, despite the potential practical applicability of the method due to its simplicity.
- The paper does not compare to existing methods, such as LenAtten and LAAM, which could be adapted to the pretrained LMs used in this paper. While I understand some parts of these methods might not directly apply to this study,  at least the control target of "equal to" a specific length should be compared.
- The paper exclusively concentrates on the length control ability, rather than enhancing the downstream tasks. It would be beneficial if the reward functions for controllability and preference reward models, such as [1], were combined to enhance both the length control ability and summarization performance simultaneously. Moreover, including human evaluation for the generated summary would be valuable.
    
    [1] Learning to summarize from human feedback
    
- The paper lacks a comparison between the modified PPO and the standard PPO.

### Questions
- see weaknesses
- minor comments
    - What N of sample filtering is used?
    - What is SG and MU in Table. 6?
    - The table caption should be positioned at the top of the table.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The goal of this work is training models that can accept natural-language length constraints as part of the prompt, including "equals", "less/greater than" and "between" styles of constraint. The work uses existing task data, particularly CNNDM and NYT summarization data. They use hand-crafted prompts to add natural language constraints with various surface forms to this existing data, and train models to convert these constraints into a structured format that can be evaluated automatically. They compare various types of training including both finetuning and RL-based methods for encouraging models to follow the given length constraints.

### Strengths
- Rule based rewards seem effective and more natural than 0-1 error that is often used for constraints
- the standard prompt extractors (SPE) seem to achieve a high accuracy, including on held-out prompt templates
- Adding RL + filter seems to improve the ability of models to adhere to length constraints

### Weaknesses
- Much of the paper is focused on using existing techniques, e.g. training models on the length of existing texts (this was used for length-controlled T5 infilling for instance) and PPO RL
- While the SPE accuracy seems high on held-out templates, all templates were written by the authors and are unlikely to cover the diversity of what humans might use in the wild. It would be useful to find a way to test generalizability to real user inputs. This is particularly important because the authors specifically frame this aspect of the paper as handling diverse inputs, and so truly demonstrating that this component of the pipeline (which is a significant part of the contribution) indeed generalizes to diverse surface forms. Otherwise, it is not completely clear why the authors would not just define a standard format, as the input prompts are all defined by the authors anyway. 
- While RL does seem to result in lower constraint error, it also (by inspection, e.g. in table 3) seem to often lower the automatic quality metrics. It would also be useful for the authors to include bold and underline values in all columns, not just constraint error.

### Questions
Please correct me if there is anything I missed in terms of contributions

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
