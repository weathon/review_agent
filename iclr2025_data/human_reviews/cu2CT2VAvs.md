## Human Reviewer 1

### Summary
The paper investigates the length generalization problem in RNN-based models, specifically on the Mamba-2 series model. The authors identify a phenomenon they term state collapse, suggesting it as the primary cause of the model's failure to generalize to longer contexts. To address this issue, they propose four mitigation methods, including three training-free approaches. Experiments conducted on the RedPajama dataset indicate that with some of these methods, the Mamba-2 model of a specific size can improve its ability to generalize over longer sequences.

### Strengths
- The paper proposes to address an important yet underexplored issue of length generalization in RNN-based models. 

- The authors identify a sharp change in the mean and variance of hidden states within Mamba-2 when processing long contexts. This observation provides a possible explanation of length generalization failures. 

- Some proposed methods show positive results in mitigating length generalization issues in Mamba-2 on a specific dataset.

### Weaknesses
Although the problem is well-motivated, the presentation significantly hinders comprehension, making it challenging to follow the authors' reasoning and understand the experimental results. A primary issue is the lack of a formal, clear definition of the new phenomenon, State Collapse (SC), which should be introduced in the early sections. However, I found no clear definition, and the descriptions provided are vague, leaving readers to infer the meaning. 

Moreover, I suspect that in the paper SC may be conflated with poor performance in length generalization. For example, in Section 4.2, the authors claim that "Figure 1 (a) and (b) shows SC on two different prompts". However, the figures appear to simply illustrate issues with length generalization. Due to this lack of clarity, it is difficult to fully assess the latter analysis of the potential causes of SC and the proposed methods to mitigate it. Even if I interpret SC as a sharp change in the hidden state values when processing long contexts, e.g., in Figure 4, 5, the latter analysis of SC in terms of over-parameterization and proposed solutions seem to be significantly disconnected from the phenomenon. 

The proposed methods and experimental results are not entirely convincing, as they are only evaluated on a single model and dataset, limiting the generalizability of the findings. Additionally, the LongMamba method is confusing; it is not clearly described in the main text, leaving it unclear whether it serves as a baseline or represents an implementation of Section 4.3.2. When comparing with the baseline [1], the authors provide only a verbal discussion in Appendix C without any qualitative results to substantiate the claims, which makes it difficult for readers to evaluate the effectiveness of the proposed methods. Finally, regarding Method 1, the legend of Figure 9 includes hyperparameters that are not explained, making it unclear how sensitive the results are to these values. Clarifying these points would enhance the reader’s understanding and confidence in the results.

[1] Ben-Kish, Assaf, et al. "DeciMamba: Exploring the Length Extrapolation Potential of Mamba." arXiv preprint arXiv:2406.14528 (2024).

### Questions
Please see the weaknesses part.

### Soundness
1

### Presentation
1

### Contribution
2

### Rating
3

### Confidence
2

---

## Human Reviewer 2

### Summary
This paper studies the performance of language model architectures with constant state size (*e.g. Mamba).* It is well-known that performance of these architectures degrades on long sequence lengths. This paper focuses on this phenomenon, which the authors call “state collapse”. The paper makes three main claims: 

1. That state collapse is due to the state size being *too big* during training, causing the model to 
2. The problem can be mitigated post-training by (a) intervening on the strength of memory decay, (b) using state normalization, and (c) and converting the model to a sliding window model. 
3. The problem can be mitigated during training by simply training on longer sequences.

### Strengths
**Significance.** The paper addresses a very important problem. The perplexity of a language model typically decreases as more context is given to the model (*i.e.* sequence length grows). However, this perplexity vs. token position plot usually flattens out or explodes for token positions beyond the sequence length on which the model was trained. A model with fixed recurrent state size for which perplexity *continues* to decrease indefinitely would represent a very significant advance. 

- Note: This paper makes some steps towards realizing this objective, but as I mention in the weaknesses, lacks evidence that the problem is really solved.

**Interesting hypothesis for state collapse (Claim 1).** The authors argue that *“SC arises from state overparameterization relative to the training length. In other words, the state capacity is excessively large for the training length, allowing the model to achieve strong language modeling performance without learning how to forget when the state is about to overflow.”* This hypothesis for the cause of state collapse is interesting and, to my knowledge, original.

**Simple methods.** The mitigation methods proposed are simple and intuitive.

### Weaknesses
**Evidence for state over-parameterization (Claim 1) is unclear and limited.** 

- *“Figure 7 shows the memory strength of the first token at different time steps, and we find that the exploded heads (heads 2, 4, and 7 in the 38th layer) have a strong inclination toward retaining all information within the training length, with a memory strength of over 0.8 at t=8K.”* It’s unclear how the layer/heads were chosen. Furthermore, the causal link between high memory strength in this one layer and state collapse is not evident.
- *“We also notice that Bt explodes earlier than ∆t. Therefore, we conclude that the collapse is largely attributable to Bt.”*  Looking at Figure 6 it’s not clear that $B_t$ “explodes” before $\Delta_t$. It also isn’t obvious why this would mean that the collapse is “largely attributable to Bt”.
- *“Finally, as we will show in Section 4.3.2, for any given training length, there exists a state size where SC will be exhibited if and only if the model’s state size is greater.”* Section 4.3.2 does not seem to address this point.
- *“It shows that SC is only exhibited by checkpoints beyond a certain amount of training, which coincides with behaviors of overfitting—a result of overparameterization.”* I found this argument confusing. Overfitting seems unrelated to the issue at hand. Training data in the language modeling setting is abundant, so it doesn’t seem like overfitting is part of the problem?

**Limitations with training-free mitigation methods (Claim 2).** There are significant limitations with the three training-free mitigation methods, which I am concerned may limit their utility in practice. 

- **Methods 1 and 3 induce forgetting.** These methods work by explicitly inducing forgetting. This is concerning given that we don’t want the model to completely forget important earlier context (*e.g.* in passkey retrieval). **
- **Perplexity still rises on longer sequences with methods 1 and 3.** In Figure 9, it looks like theperplexity at high token positions (30k - 60k) is significantly higher than perplexity at earlier positions (10k). Ideally, the model’s perplexity would continue to decrease with longer sequences (or at least stay constant).
- **Method 2** **makes training/prefill non-parallelizable.** The authors did a good job of highlighting this limitation, but am raising it again here to emphasize how this would seriously complicate model training and serving.
- **Method 2** **degrades performance substantially.** Figure 9 shows that Method 2 leads to a significant increase in perplexity at token positions within the training length. This suggests that the model may need to adapt to state normalization *during training,* which is a concern given the non-parallelizability of the architecture.

**Evaluation on a limited set of tasks (Claims 2 + 3).** The experiments supporting *Claim 2* only measures perplexity on RedPajama. Given the comment above (that training-free mitigation methods cause forgetting) the methods should be evaluated on passkey retrieval, at a minimum. Ideally, the methods would also be evaluated on long context evaluations like ∞-BENCH or Loogle [[Zhang 2024](https://aclanthology.org/2024.acl-long.814.pdf), [Li 2024](https://arxiv.org/pdf/2311.04939)]. The experiments supporting *Claim 3* only measure perplexity on the synthetic *newlines* prompt*.*The lack of experiments on real data and hard long context tasks is concerning.

### Questions
- *Why are analyses performed at the 38th layer of the Mamba model?*
- *Can Figure 9 use a log scale on the y-axis?* It is hard to discern differences in perplexity.
- *What is the color scale in Figure 11 and which mitigation method was used for this experiment?*

### Soundness
2

### Presentation
1

### Contribution
2

### Rating
3

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper investigates the poor length generalization properties of recurrent models, in particular Mamba-2, and finds that a mechanism the authors term State Collapse (SC) is responsible. Based on this insight, the authors propose three mechanisms to mitigate state collapse in Mamba models, and find that it improves the length generalization capabilities of Mamba-2 models.

### Strengths
- The work tackles an important problem from a new direction, informing long-context training methods from mechanistic understanding of recurrent models.
- The newlines prompt is a compelling synthetic (if models can’t do this, something is wrong!) and provides focus and grounding to the work.
- The provided methods all appear to improve the length-generalization capabilities of Mamba-2 models. The authors provide analysis on both synthetics and a range of language modeling tasks.

### Weaknesses
- While the three proposed methods show promise, additional comparisons between them would enhance the paper's practical value. Figure 9 provides some insights, but more detailed ablation studies comparing their relative strengths and limitations would be valuable for practitioners.
- The generalizability of these methods beyond Mamba-2 presents an opportunity for future exploration. Given that RWKV and GLA are mentioned in the introduction, including experiments with these architectures could help demonstrate the broader applicability of the proposed approach.
- The methods currently show some performance trade-offs at shorter sequence lengths. Additional analysis or ablations of this phenomenon through the paper's theoretical framework could provide valuable insights and potentially suggest ways to maintain performance across all sequence lengths.
- The paper would benefit from engaging more with relevant prior work on recurrent neural network length generalization, including recent contributions such as Jelassi, Samy, et al. "Repeat after me: Transformers are better than state space models at copying." arXiv preprint arXiv:2402.01032 (2024); Arora, Simran, et al. "Zoology: Measuring and improving recall in efficient language models." arXiv preprint arXiv:2312.04927 (2023); and Ben-Kish, Assaf, et al. "DeciMamba: Exploring the Length Extrapolation Potential of Mamba." arXiv preprint arXiv:2406.14528 (2024). While this work offers a novel perspective, situating it within the broader literature would strengthen its contribution.
- The visualization of results, particularly in figures 2, 6, and 12, could be more accessible to readers. For instance, figure 2 would benefit from clearer labeling and explanation of metrics like "answer depth," and the color scheme implications could be more explicitly defined in the caption or text.

### Questions
- How general are these methods for recurrent models? Do they transfer well beyond Mamba-2?
- Of the proposed methods, which are more or less effective, and in which contexts?
- Is the tradeoff at short sequences inherent?

### Soundness
2

### Presentation
2

### Contribution
3

### Rating
3

### Confidence
4

---

## Human Reviewer 4

### Summary
The paper looks into difficulties RNN-based language models have on long context tasks. First, the paper considers the observation that RNN based language models are unable to extrapolate to sequences longer than they were trained on. By studying the mean and variance statistics of activations in pretrained models, the authors identify an issue where some heads exhibit an exploding variance during extrapolation. Normalization then causes the information in the state to be erased. The authors argue this collapse is due to overparameterization and propose both training-free methods and a training related method to overcome this. The authors then attempt to characterize "state capacity" by sweeping training lengths and state sizes and observing the results of extrapolation. Empirical results suggest that several of the proposed methods can improve extrapolation ability.

### Strengths
- To my knowledge, this is one of the first works to take a systematic analysis of the model internals to determine the cause of RNN based models to fail to extrapolate
- The analysis of the mean and variance of the model activations is presented in a clear way making the cause of extrapolation failures clear
- The proposed training based and training-free interventions make sense and some of them appear to improve extrapolation based on the empirical results provided

### Weaknesses
- I would have liked to have seen a more precise definition of "state collapse" to provide clarity. Line 187 says it is: "a phenomenon that causes RNN models to exhibit abnormal behaviors on inputs longer than those seen during training." But what precisely constitutes an "abornomal behavior"? 
    - Is state collapse simply the failure of the model to perform extrapolation, or is it a specific behavior of the model's state that leads to this failure? If the former, there may be no need for a new name, if the latter I would expect some characteristics of state collapse related to the state to be explicitly mentioned
    - Line 058 implies the extrapolation failure is "due to a phenomenon called state collapse", implying state collapse is different from the extrapolation failure itself. In addition, Section 4.2 implies that the cause of state collapse is the explosion of variance of some outlier heads. So based on these clues, we have the variance explosion causing "state collapse", and we have "state collapse" causing extrapolation failure. But this still leaves open the question of the precise definition of "state collapse"?

- In line 329 (Section 4.2.1), we are promised that in Section 4.3.2 we will be provided a proof (or perhaps an empirical observation, it is unclear) regarding an "if and only if" relationship between state size and state collapse. However this result is never provided in 4.3.2 (neither theoretical or empirical discussion), and no figures are referenced in this section. Then in Section 5, line 400 this "if and only if" result is referred to again as if it was described in Section 4.2.1. Was this accidentally left out? Or am I somehow missing it?

- I also found the use of "state capacity" to be imprecise. Line 074 implies that state capacity and state size are different (which makes sense). State capacity is referred to multiple times in Section 4 without definition. In line 401 we are told we can empirically estimate state capacity with sweeps. Finally, in lines 407-408 we are told that "we regard the minimum training length at which SC [state collapse] does not occur as the state capacity". So we have finally defined it in Section 5 by referring to "state collapse" which has also not precisely been defined as mentioned in the point above. In addition, the definition refers to an empirical measure, which we were previously told we would use to estimate "state capacity". It all seems very circular. It appears the paper is trying to distinguish between a theoretical state capacity and an empirical state capacity, however the two appear to be conflated.

-  There are many relevant prior works related to the long context failures of RNN based models that would have been good to reference to provide the reader with a clearer view of the literature in this area. Examples: https://arxiv.org/abs/2402.01032, https://arxiv.org/abs/2402.18510, https://arxiv.org/abs/2402.04248 

- The paper only focuses on a notion of "state capacity" related to the ability to not spike perplexity on longer extrapolated sequences, or the ability to retrieve a single passkey across varying sequence lengths. This mostly seems to be related to the model's ability to ignore or forget irrelevant context. However another obvious version of "capacity" would be a multi-retrieval setting where the model has to use its fixed state to retrieve many passkeys or needles, even for shorter contexts (potentially within distribution, thus not required extrapolation), thereby stretching the memory capacity of the fixed state. This is explored synthetically in many works such as: https://arxiv.org/abs/2402.18668 for the multi-query associative recall task and in the RULER https://arxiv.org/abs/2404.06654 work which gives multi-key, multi-value and multi-query versions of passkey retrieval. This notion of capacity related to the actual information content that has to be remembered and recalled by the fixed state should at least be mentioned if not also ideally considered and analyzed in a definition of "state capacity".

### Questions
Most of my main questions and concerns I would like to have addressed are above in Weaknesses. Below is a summary of my main requests as well as a few additional questions.

1. Can you please provide a precise definition of "state collapse" as used in this paper?

2. Can you please clearly provide the proof or empirical observation regarding the relationship between state size and state collapse?

3. Can you please provide a precise definition of "state capacity" as used in this paper and ensure this is introduced earlier in the paper before it is used extensively? Can you also please clarify the relationship between state capacity, state size and state collapse as used in this paper? In addition, commenting on other notions of state/memory capacity as mentioned above would be helpful.

4. How is Method 1 of the training-free methods implemented in practice? Or what is the actual intervention? It is unclear from the text. I would recommend explicitly stating this in the Method 1 paragraph.

I like the general ideas of this paper and believe it can be interesting to the community. However I currently have concerns regarding the presentation of the ideas and technical terms in the paper as mentioned in the weaknesses section. This makes it confusing to read and clouds the message the authors are trying to get across. I can raise my score if these concerns are clarified and addressed.

### Soundness
2

### Presentation
1

### Contribution
3

### Rating
3

### Confidence
4

---

## Human Reviewer 5

### Summary
This paper investigates a critical issue called "state collapse" (SC) in RNN-based language models, particularly focusing on Mamba-2. The authors identify that SC occurs when RNNs process sequences longer than their training length, causing severe performance degradation. They propose three training-free solutions and one training-based approach to mitigate SC, enabling models to handle sequences beyond 1M tokens. They conduct experiments to understand the relationship between state capacity and model size on the passkey retrieval task.

### Strengths
- The work addresses a crucial challenge in RNN-based language models that has not been solved. The authors clearly demonstrate the SC problem (Figure 1), showing Mamba-2 and RWKV suffer from performance degradation beyond their training length.
- The proposed solutions are simple and practical. The three training-free methods (adjusting memory retention, state normalization, and sliding window) are well-motivated by their analysis of the SC phenomenon. As shown in Figure 9, considering perplexity, these solutions effectively suppress SC and allow the model to generalize well in a long context.
- The empirical results on passkey retrieval are impressive. The authors achieved near-perfect accuracy on 256K context length with only a 370M parameter model (Figure 11).

### Weaknesses
- My major concern along this line of work, i.e., RNN-based language model, is their constant hidden state size. The fundamental limitation of constant state size in RNN-based models isn't fully addressed. When the task becomes complex and requires more memory, Mamba or RWKV cannot solve them well theoretically. However, attention may not suffer from this as their hidden state size is linearly large as the input token numbers. I am not sure whether RNN-based language model is a correct way. While the paper shows good results up to 1M tokens, there's no theoretical analysis of whether these models could scale to much longer contexts. The constant state size might ultimately prove to be a bottleneck that the proposed solutions can't overcome.
- The evaluation relies heavily on the passkey retrieval task and perpelxity, which is relatively simple compared to real-world long-context tasks. The paper doesn't demonstrate whether the proposed solutions work well for more complex tasks like long document summarization or multi-step reasoning. How would the proposed solutions perform on more complex long-context tasks? As the authors acknowledge in the limitations section, "high accuracy on this task may not reflect the capabilities of the model on real-world long-context tasks."
- The state capacity analysis in Section 5 lacks clear definitions and methodology. What is the rigorous definition of memory capacity in Figure 12? While Figure 12 shows relationships between state size and capacity, it's unclear how the authors precisely define and measure "memory capacity" or "state collapse." The paper would benefit from more rigorous definitions and evaluation metrics. Also, I would like to see some rigorous theoretical analysis, as the memory capacity seems can be fomularted as a well-defined information theory problem, e.g, theoretical upper bounds of the memory capacity for RNN-based models with constant state size.

### Questions
See Weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
2