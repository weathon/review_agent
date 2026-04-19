# SuperPos-Prompt: Enhancing Soft Prompt Tuning of Language Models with Superposition of Multi Token Embeddings

- Decision: Reject
- Scores: 6, 3, 5, 5

## Abstract
Soft prompt tuning techniques have recently gained traction as an effective strategy for the parameter-efficient tuning of pretrained language models, particularly minimizing the required adjustment of model parameters. Despite their growing use, achieving optimal tuning with soft prompts, especially with smaller datasets, remains a substantial challenge. This study makes two contributions in this domain: (i) we introduce SuperPos-Prompt, a new reparameterization technique employing the superposition of multiple pretrained vocabulary embeddings to improve the learning of soft prompts.  Our experiments across several GLUE and SuperGLUE benchmarks consistently highlight SuperPos-Prompt's superiority over \textit{Residual Prompt} tuning, exhibiting an average score increase of +4.7 in T5-Small and $+3.9$ in T5-Base along with a faster convergence. Remarkably, SuperPos-Prompt occasionally outperforms even full fine-tuning methods. (ii) Additionally, we demonstrate enhanced performance and rapid convergence by omitting dropout from the frozen network, yielding consistent improvements across various scenarios and tuning methods. Unlike many existing strategies, our approach does not rely on the availability of a proficient pretrained source prompt for initialization, thereby ensuring notable flexibility and more effective combination of related prompt candidates.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduce a novel soft prompt tuning technique, which is a new reparameterization technique that employs the superposition of multiple pre-trained vocabulary embeddings to improve the learning of soft prompt turning, and is able to improve without relying on the pre-trained soft prompts.The experiments tuned the LM-adapted T5 model on a smaller scale dataset. Significant improvement was achieved compared to the Residual Prompt tuning technique.

### Strengths
The experimental results presented in this paper are significant and appealing from the point of view of the development of prompt-tuning techniques as well as fine-tuning of small-scale datasets. The authors expand the reparameterization methods in the field of soft prompt tuning techniques and provide a more detailed experimental demonstration. In conclusion, the SUPERPOS-PROMPT presented in this paper is of great value and helps to advance the development of related research. Detailed and sufficient results are provided in the main text. Thus, the SUPERPOS-PROMPT proposed in this paper is of great value and helps to promote the development of research in the field of prompt tuning.

### Weaknesses
1. the paper presents experimental results from the innovations without sufficient justification and explanation, e.g., what are the key explainable innovations that contribute to the performance improvement over similar techniques IPT, ATTEMPT, and Residual Prompt Tuning. 
2. The validity of large language models needs to be further demonstrated. 
3. The validity for LMs beyond T5 is unknown.

### Questions
Why does the author only choose T5 as the model for evaluation? Especial when the authors already noticed that T5 has checkpoints that pretrained on GLUE and SuperGLUE datasets. Therefore, I doubt the contribution of proposed method if the method does not lead to a progress on the SOTA results compared to either pretraining or other fine-tuning approaches.

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper tries to propose a different soft-prompt tuning method. The limited experiments show that the proposed method outperforms residual prompt tuning and original soft prompt tuning.

### Strengths
The method’s performance seems better than the compared baselines.

### Weaknesses
1. Unclear presentation. For example, if you finally decided to freeze E_{freeze}, how did you initialize it? Is it just randomly initialized or intialized with some pretrained token embeddings?
2. Missing baselines. For example, the authors should at least compare the proposed method with the similar methods they mentioned themselves, such as IPT and ATTEMPT, in a fair setting. “Through our experiments, we noticed that utilizing superposition is more efficient than softmax weighting.” Where is this experiment?
3. Too many ablations are missing so that I find it very hard to understand why the proposed method is so much better than residual prompt tuning. The authors mention 3 variants in the main figure (Figure 1, bef) but have shown no results regarding such variants. Where does the performance improvement come from? Is it mainly from freezing E_{freeze}, or removing softmax, or reducing parameters?

### Questions
See weakness

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
This work focuses on soft prompt tuning, a method to efficiently fine-tune pretrained language models with minimal parameter updates (PEFT methods). Soft prompts are challenging to optimize, especially with small datasets. This is empirically observed. PEFT methods such as LoRA and adapters are more popular than soft prompt tuning, prefix tuning etc. because of this reason. When finetuning an LLM on small datasets they seem to suffer from a notable performance drop compared to vanilla as observed in previous works [1]. The study contributes in two ways:
1. Introduction of SUPERPOS-PROMPT: This is a simple to use reparameterization technique that improves the learning of soft prompts. It does so by taking a linear combination of multiple pretrained vocabulary embeddings. The authors conducted experiments across various GLUE and SuperGLUE benchmarks, showing that SUPERPOS-PROMPT outperforms Residual Prompt tuning. It yields an average score improvement of +4.7 on T5-Small and +3.9 on T5-Base, along with faster convergence. SUPERPOS-PROMPT sometimes even outperforms full fine-tuning methods however I have a few questions regarding this result.
2. Omission of Dropout from Frozen Network: The authors demonstrate that excluding dropout from the frozen (unchanged) parts of the network leads to enhanced performance and faster convergence. This improvement is consistent across different scenarios and tuning methods.

Crucially, the proposed approach does not depend on having a proficient pretrained source prompt for initialization, providing significant flexibility and more effective combination of related prompt candidates. The authors argue that this is a common limitation in existing strategies, making the approach more versatile and applicable to a broader range of tasks. However, I believe comparisons with these methods would make the method more insightful.

[1] Hu, Edward J., et al. "Lora: Low-rank adaptation of large language models." arXiv preprint arXiv:2106.09685 (2021).

### Strengths
Strengths
1) I agree with the authors that the challenge of soft prompt tuning on small datasets has to be addressed. Although this technique substantially reduces the number of trainable parameters, its diminished performance and slower convergence on smaller datasets makes its use less appealing. Effectively tackling these issues would constitute a significant contribution. Authors make an attempt at addressing these issues.
2) The authors demonstrate that excluding dropout from the frozen (unchanged) parts of the network leads to enhanced performance and faster convergence. This improvement is consistent across different scenarios and tuning methods. 
3) I thank the authors for pointing this inference, “Through our experiments, we noticed that utilizing superposition is more efficient than softmax weighting.” This result has more implication since it says linear combination is more beneficial than convex combination for prompt superposition.
4) The authors propose a simple yet effective strategy (empirically) of linear combination of multiple tokens as prompt initialization.

### Weaknesses
Thank you for your work. I hope my following suggestions would make the work more robust.

1) The writing requires improvement. 
For instance, (a) the authors introduce various methods under three categories: 1) Prompt layers reparameterization, 2) Pre-trained prompts as initial states, and 3) Combined approach. Presenting this multitude of works in the introduction might overwhelm and confuse the reader. As an alternative, the authors could cite one representative work for each category and discuss the limitations of these works in the introduction. Additionally, a dedicated ‘Related Works’ section could be included, where the authors provide a more comprehensive overview of the numerous cited works, allowing readers to delve into further details. This structure would enhance the paper’s readability. (b) Another example is, authors propose “We begin with selecting a linear combination of m unique token embeddings sampled from the token embedding layer, denoted as e1 , e2 , ..., em .” However, they did not mention how to select those ‘m’. Are they randomly sampled?
(c) “Impact of Dropout, Impact of SuperPos-Prompt, Effect of the Number of Sampled Tokens” in section 5 would better fit in another "ablation study section" explaining each of the settings more thoroughly. I would also encourage the authors to explain possible reasons behind why the proposed method is able to outperform other baselines. For instance, `the impact of dropout', it is an interesting and useful study however digging further into reasons why it is performing better than without dropout would lead to more insights than empirically stating the results. Even for “Effect of the Number of Sampled Tokens”, I had to go through it multiple times to understand it is an ablation study for ‘m’ parameter. (d) In the introduction, it is not clearly explained what is meant by “multiple token embeddings.”

2) The notations in the methods section require careful attention. For example, in section 3, the authors introduce p_i as a linear combination of m embedding vectors weighted by p’_i, and they mention the dimension of E as e X m. It seems that the authors propose to randomly sample m embedding vectors for soft prompt initialization. However, in the subsequent step, they reparameterize E as follows: E = Efreeze + ∆E, with ∆Einit = 0e×n. The variable 'n' is not defined or mentioned anywhere prior to this point. Furthermore, the subsequent step, 'p_i = (Efreeze + ∆Ei)p′_i,' appears to contradict the dimensions mentioned earlier, as E was defined as having a dimension of e X m.

3) The authors deliberately choose not to include IPT and ATTEMPT in their comparative analysis, explaining that these methods depend on pretrained source prompts. This is at odds with their primary goal, which is to enhance soft prompt tuning without the need for pre-trained soft prompts. However, the rationale behind excluding these methods from the comparison is not entirely transparent. To bolster the validity of their approach, incorporating a comparison with these baseline methods would lead to more insights on the utility of this method. I would like to point out that utilizing the pretrained soft-prompts has a similar analogy as utilizing the pretrained weights for finetuning on a downstream task. I do not see a limitation in directly utilizing the pretrained soft prompts.

4) While the method is straightforward to implement, the authors have not sufficiently justified its apparent stability compared to other baseline methods for soft prompt tuning. For example, it remains unclear why the linear combination of 'm' randomly sampled token embeddings yields enhanced performance over the baseline. What specific aspects of this initialization contribute to superior results? The authors should provide additional insights to clarify this matter. A more comprehensive analysis is also needed to address the issue of inference without 'dropout' as mentioned earlier.

### Questions
1) I am not convinced with the result that, Superpos PT without dropout outperforms full finetuning by such a large margin of 16.4% on CB task on T5 small, same goes with T5 base on COPA task. It is not convincing that PEFT method is able to beat full finetuning by such a large margin. As claimed by the authors even LoRA and adapters that perform better than soft prompt based methods, occasionally beat full finetuning by very small margin. Could you please shed some light on it? Also, are the hyperparameters used to tune full model parameters consistent with the original paper.

2) The authors mention this in the introduction “Soft prompt is renowned for its exceptional parameter efficiency.” However it also mentions “finetuning soft prompts is optimization-intensive, particularly with limited data and smaller model sizes in T5 family between 50 to 300 million parameters (Lester et al., 2021);” As far as I understand the model weights are frozen. Are they contradictory statements made independently or are the authors mentioning that the convergence rate is slow for soft prompt based methods?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper focuses on soft prompt tuning of pretrained language model and proposes an approach called SuperPos-Prompt, which tunes the delta value of word embedding parameter as well as the combination weight of words for the soft prompt tokens. The authors have compared the proposed SuperPos-Prompt and existing prompt tuning methods, including Intrinsic Prompt Tuning, Residual Prompt Tuning and ATTEMPT, both theoretically and experimentally. The experimental results using the pretrained T5 model on GLUE and SuperGLUE benchmarks show the advantage of SuperPos-Prompt over existing methods. Meanwhile, as an additional contribution, the authors claim that the dropout should be omitting from the frozen network.

### Strengths
1. The proposed method is relatively easy to implement, compared with previous methods requiring heavily designed soft prompt initialization.
2. The selected setting of benchmark and pretrained model (T5) is representative.
3. Analysis of key components, including the impact of dropout, the results on different PLMs and effect of the number of sampled tokens are conducted.

### Weaknesses
1. The paper lacks sufficient analysis of SuperPos-Prompt learned soft prompts on what semantic information the prompt tokens indicate and how it contributes the results, at least some case study is needed.
2. To be honest, the novelty is fair and incremental. In my opinion, the paper is more like an experimental report over existing methods. If more analysis on the learned soft prompts from semantic perspective is proposed, I think it will be much better.

### Questions
Have you considered more recent LLMs and includes more baselines like few-shot results on ChatGPT/GPT4 for reference?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
