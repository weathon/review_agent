# RePro: Training Language Models to Faithfully Recycle the Web for Pretraining

- Avg Score: 6.00
- Decision: Reject
- Scores: 6, 4, 8, 6

## Abstract
High-quality data is a cornerstone of large language model (LLM) pretraining, yet its growth has not kept pace with the needs of frontier models. In this paper, we introduce RePro, a novel web recycling method that trains a relatively small LM with reinforcement learning to generate effective and faithful rephrasings of pretraining data. Specifically, we design one *quality* reward and three *faithfulness* rewards, optimizing the LM rephraser to convert organic data into high-quality rephrasings while maintaining its core semantics and structure. In our experiment, we train a 4B rephraser to recycle 72B tokens sampled from DCLM-RefinedWeb. Pretraining results on 400M and 1.4B models demonstrate that RePro delivers 4.7\%-14.0\% relative accuracy gains over organic-only baseline on 22 downstream tasks. RePro also outperforms ReWire, the state-of-the-art web recycling method that prompts a 70B rephraser, as well as the organic baseline with a 4$\times$ larger data pool. Experiments with different amounts of recycled data highlight that RePro improves organic data efficiency by 2-3$\times$. Individual and distributional analyses validate that RePro preserves more critical information and faithfully reflects the characteristics of organic data compared to prompting-based methods. Together, these results show that RePro provides an efficient and controllable path to effectively harness organic pretraining data. Our anonymized code is available at https://anonymous.4open.science/r/RePro. We will open-source our rephraser and recycled data.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper discusses the problem of finite internet data to pretrain LLMs on. It motivates the expansion of current finite data in a way that is efficient (unlike GPT4/5 rephrasing everything) and faithful i.e. preserve underlying semantics. Towards these, they train a Qwen3/4B to rephrase data that and carry out this data augmentation to pretrain models. They use RL based loss that uses
two metrics viz. DataMan [1] score for quality assessment and BERTScore [2] for semantic similarity check. The paper name their method as "RePRO".

For experiments, they pretrain a 400M and 1.4B model and compare against baseline method (viz ReWire, which prompts llama3/70B) and baseline dataset which is standard pretraining corpus (?) and report better benchmark numbers. 



[1] Ru Peng, Kexin Yang, Yawen Zeng, Junyang Lin, Dayiheng Liu, and Junbo Zhao. DataMan: Data
manager for pre-training large language models. In Proc. of ICLR, 2025
[2] Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q Weinberger, and Yoav Artzi. BERTScore: Eval-
uating text generation with bert. In Proc. of ICLR, 2020

### Strengths
- Very well written paper, have cited important related works appropriately, even outside related work section to motivate and compare thoughtfully. 
- Nice set of ablation, easy to verify where the gains comes from
- Gain on performance (albeit marginal) against proper baselines. I also appreciate the case study in Appendix D.

All in all, this paper proposes how RLing a rephrases can make them better at augmenting internet scrap like data while optimizing for quality and faithfulness, among others.

### Weaknesses
See below

### Questions
I like this paper, to better understand if the gains are actually from the proposed method, I would need response to following questions:

- Is the Table 3 "Prompting" is prompting unoptimized Qwen3/4B with same prompt (as used for RL training)? What sampler did you use? Is it 0-temperature sampling? Can I ask to see performance with temperature 0.6, and also another with top-p 0.96 to clearly understand the gains are from RL and not just sampling. Even if this is the case, this should be weakness of RL methods in general applicable here and not of RePro directly.
- Figure 3 is missing bars for WRAP, ProX, and ReWire (+ Prompting baseline), it would be nice to see how RePro comapres to these across scales.
- Is the thinking mode of qwen3 used for this? Can I see the thinking traces for the case study in Appendix D?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper focuses on generating new pretraining data by rephrasing existing data. Specifically, the authors train a rephraser LM (4B) using RL to generate effective and faithful rephrasing of existing pretriaining data. Four reward signals, one focusing on quality and the other three focusing on faithfulness are used for RL training. Using this model, authors rephrase 72B tokens sampled from DCLM-RefinedWeb, and show that models trained with high quality rephrased data perform better than the models trained with the original data. Specifically, proposed rephrasing leads to 2-3x organic data efficiency, and also outperfoms ReWire, an existing approach that uses 70B model as rephraser.

### Strengths
Proposes an interesting approach to train a rephraser model and demonstrates the effectiveness of the trained model by comparing it with alternative rephrasing strategies

Conducts ablation studies demonstrating the effectiveness of the reward signals used for RL

### Weaknesses
* According to Sec 3.1, the entire organic pool is recycled including the high quality data D_{org-hq} (Eq. 2). The rephrased data subset used for training is selected as the highest quality samples from the entire rephrased pool (eq. 3). With this strategy, D_{rec-hq} could be dominated by rephrased version of D_{org-hq} rather than the rephrased version of D_org - D_{org-hq}. This seems to go against the goal of taking advantage of low-quality samples by rephrasing them. Authors should show how many tokens in D_{rec-hq} are derived from D_{org-hq} and how many are derived from D_org - D_{org-hq}. 
    * Is the proposed approach actually leveraging additional low-quality (but diverse) data or mainly using rephrased version of already high quality samples?
    * How would the results look if D_{rec-hq} was selected only from  D_org - D_{org-hq}?


* **Do we really need RL here or do we just need to clearly communicate with the rephraser?** - The proposed rephraser is trained with RL using four reward functions. For a fair comparison, both prompting-based rephrasing baseline and GPT prompting-based SFT data generation should incorporate the knowledge of the rewards into the prompt used for rephrasing. This could be done by describing what the reward functions are trying to capture as part of the prompt (which the authors are already doing for prompting the reward computing LLMs). This will give a clear signal if RL training is really needed or we simply need to tell the rephraser clearly what it needs to optimize when rephrasing by describing the rewards in the prompt.


* **Relation between RL reward function and the final data quality function used for selecting sample** - According to experimental results, directly using DataMan for selecting samples is worse than using DCLM-fastText. However, training with DCLM-fastText is worse than training with DataMan. This suggests that there is some disagreement between the two which warrants further attention/analysis. Authors could look at the correlation (or scatter plot) between the two to get a better understanding of this behavior

* Experiments are conducted using small models (400M and 1B), while the rephraser itself is a much bigger 4B model. It is unclear if the proposed approach would be effective for rephrasing datasets when training bigger models (let’s say 7B).


Minor suggestion:
The comparison of pretrainig data with fossil fuel is not meaningful. Used fossil fuel disappers from earth and hence the reserves are going down. In contrast, pretraining data does not disappear after being used for training.

### Questions
Is the proposed approach actually leveraging additional low-quality (but diverse) data by rephrasing?

Do we really need RL here or do we just need to clearly communicate with the rephraser?

Are DataMan and DCLM-fastText scores correlated or not?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces a data recycling framework that uses a smaller reinforcement-trained model to automatically rewrite web text into higher-quality pretraining data. Rather than filtering or prompting large models to clean data, the method trains a dedicated “rephraser” that learns to improve readability and consistency while preserving the original meaning. This approach aims to make pretraining corpora both cleaner and more diverse without heavy human curation or large-model prompting. Experiments show that models trained on the rewritten data perform better on downstream benchmarks and that the rephraser operates far more efficiently than prompt-based alternatives.

### Strengths
1. Addresses a practical problem in pretraining: improving web data quality without costly human filtering or large-model prompting, using a small reinforcement-trained rephraser.
2. Computational savings over prior works that use large models to rephrase the text, while recovering most of the performance that those methods gave 
3. Ablations confirm that each reward component (quality and faithfulness) is necessary to prevent drift and reward hacking, supporting the method’s design.

### Weaknesses
1. It would be nice to see a clearer / more varied perspective on what the rephrased text looks like. Concerns involve degenerate / repeated formatting, diversity collapse, etc. 
2. The benchmark gains are minimal in some cases (requiring 3 decimal places to see the value of it). I would love to see some discussion of what to take away from the behavior on  "core" tasks vs the other ones.
3. In some sense, the place where we need more / better data is in specialized domains like math and code. I think the authors can emphasize that their method enables straightforward adaptation of any synthetic data generator to maximize some reward (eg in math, maybe it is "diverse CoTs")

### Questions
See above

### Soundness
4

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
The authors study the rephrasing of pretraining data to increase the availability of high-quality data. They design one quality reward and three faithfulness rewards to guide the training of LM rephraser. They demonstrate their method using a 4B rephraser can outperform state-of-the-art methods using 70B models.

### Strengths
1. They demonstrated the viability of using small models to obtain high quality recycled data, while this is also shown by some concurrent works.

### Weaknesses
1. The novelty and insight feel somehow limited. One can follow up with even more design of rewards to potentially further improve the performance, while how much this could help the community is debatable
2. The scale of the pretrained model is rather limited. Notably, much smaller than the rephraser, which might impact the transfer of knowledge,

### Questions
1. Could you show some results that the baselines perform badly with the 4B models? (Even though this may be quite imaginable) Additionally, could you show some result using other model families as the rephraser, preferably the one used by the baseline (llama)?
2. Do you have some preliminary results with slightly larger pretrained models? In the literature of knowledge distillation, which is different but still somewhat related to your research direction, some has shown that distilling from a much larger teacher model might be worse than a smaller one due to gap in model capacity [1]. In your case, similarly, the small pretrained model might not be able to learn/differentiate the content generated from a larger and potentially better rephraser.

[1] Mirzadeh, S. I., Farajtabar, M., Li, A., Levine, N., Matsukawa, A., & Ghasemzadeh, H. (2020, April). Improved knowledge distillation via teacher assistant. In Proceedings of the AAAI conference on artificial intelligence (Vol. 34, No. 04, pp. 5191-5198).

### Soundness
3

### Presentation
3

### Contribution
3
