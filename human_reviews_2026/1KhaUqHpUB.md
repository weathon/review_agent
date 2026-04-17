# A Generative Approach to LLM Harmfulness Mitigation with Red Flag Tokens

- Decision: Reject
- Scores: 6, 4, 2, 6

## Abstract
Many safety post-training methods for large language models (LLMs) are designed to modify the model’s behaviour from producing unsafe answers to issuing refusals. 
However, such distribution shifts are often brittle and degrade performance on desirable tasks.
To address these pitfalls, we propose augmenting the model’s vocabulary with a special red flag token ($\langle\texttt{rf}\rangle$) and training the model to insert this token whenever harmful content is generated or imminent. 
This approach enables the model to explicitly learn the concept of harmfulness in its representations, with minimal impact on utility due to the marginal change in the generated distribution of natural language. 
Moreover, because the token is embedded in the model’s vocabulary, we can naturally leverage the LLMs' generalisation capabilities, such as in-context learning (ICL) and out-of-distribution generalisation to languages that are not formally supported (e.g., Japanese for Llama3). 
In particular, we demonstrate that through ICL alone, the model can learn to initiate reflective reasoning upon generating $\langle\texttt{rf}\rangle$ at inference, which steers the response away from harmful continuations or enables self-correction when the flag is raised falsely.
This approach is orthogonal and complementary to existing safety techniques—such as safety classifiers or standard safety training---and easier to evaluate in comparison to natural language refusals, as it does not require a human or automated judge to assess the harmlessness of the answers.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a simple technique to enhance LLM safety through the introduction of a red flag token (⟨rf⟩) that the model emits when potentially generating harmful content. By training models to insert ⟨rf⟩ whenever harmful output is imminent, the approach enables intrinsic harmfulness detection without disrupting overall language generation quality. The method is validated across multiple open-source LLMs (e.g., LLaMA3.2, MistralV3, Phi-3.5), using standard utility and robustness benchmarks, as well as cross-lingual evaluations—including Japanese, a low-resource language.

### Strengths
1. The proposed idea of incorporating a red flag token is new and technically simple. It represents a promising direction for improving LLM safety without sacrificing fluency or utility. Unlike traditional refusal-based or classifier-based systems, this generative mechanism directly embeds safety awareness within the model itself.

1. The paper is well-written and well-organized. The structure flows logically from motivation to method, followed by rigorous evaluation. The figures (especially Figures 1 and 2) clearly illustrate both conceptual design and training dynamics.

1. The authors provide a comprehensive overview of recent LLM safety research, which enables us to understand their contribution precisely. 

1. The method is technically sound and well-motivated. The loss formulation combines cross-entropy and KL-divergence terms to maintain distributional consistency, which demonstrates careful consideration of avoiding performance degradation while embedding the new safety signal.

1. Experiments are conducted across multiple models and datasets. The results convincingly show improved defense success rates under both gray-box and white-box attacks, while maintaining nearly unchanged utility scores.

1. Including results on Japanese (a low-resource, unsupported language for some models) adds considerable depth. This highlights the approach’s ability to generalize beyond English and provides valuable implications for multilingual safety research.

### Weaknesses
1. I feel this paper lacks sufficient implementation details for reproducibility. The Appendix is rather brief and does not include enough low-level information.

2. If I understand correctly, this paper does not explicitly discuss adversarial misuse or the potential for malicious users to manipulate or suppress the ⟨rf⟩ mechanism. Its threat model focuses on pre-filling and sampling attacks, but not on direct prompt-level manipulation of the red-flag behavior. Therefore, highlighting this as a limitation or open question in the revised manuscript would be valuable.

### Questions
1. Could the ⟨rf⟩ token be extended or diversified (e.g., multiple “flag” tokens for different types of risk such as toxicity, bias, or misinformation)?

1. Could you tell me the thoughts on Weakness 2?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a novel approach to LLM harmfulness mitigation. Instead of performing a post-training (alignment) of LLMs for safety, the proposed approach trains LLMs to generate a red flag token <rf> during harmful generation. The idea comes from the authors’ claim that the post-training to prevent an LLM from producing harmful answers and force it to issue refusals causes a distribution shift, and it is a cause of the performance degradation. To prevent a significant distribution shift, the LLM is trained to produce <rf> when generating harmful content, while it is maintained not to go far from the original distribution without <rf>. The red flag token is then utilized at inference time to filter the response and replace it with a safe alternative, or trigger safety-oriented reflective reasoning. The experiments show competitive adversarial robustness to the existing adversarial training-based approach (CAT) and that the usefulness of the trained model is kept as high as the baseline model.

### Strengths
A novel approach to detect harmful generations without causing a significant distribution shift. This idea itself may be used for different purposes.

### Weaknesses
Preventing a distribution shift is a reasonable idea. However, for this purpose, one can simply use filtering approraches that pre-process inputs or post-process outputs. With a filtering approach, one does not need to post-train the LLM itself, leading to no change of distribution. The current defense mechanism is actually similar to output filtering techniques. A very naive baseline approach would train a classifier for harmfulness by using the dataset prepared for the proposed approach training and use the classifier to check whether the output is harmful or not. Compared to filter-based approaches, the advantage of this approach from the perspective of defense is questionable. An experimental comparison with filtering based approach is required. 

Robustness evaluation against automated jailbreaking attack such as GCG and PAIR is provided only for a small model (LLAMA3.2-3B-IT). Moreover, although I couldn’t find the statement, it seems that the result is a single training result, meaning that the result is not reliable. While the harmless performance is better than CAT, the robustness performance is worse (10% lower than CAT for PAIR).

### Questions
How to tune hyper-parameters, alphas, in (4)? Could you provide a sensitivity analysis?

Why does it make sense to solve (5)? Does the l2-neighborhood of adversarially generated prompts in the feature space contain adversarial prompts? Does prompts generated by GCG or PAIR stay close to their original prompts in the feature space?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
They train the model to output the \<rf\> token when it generates something harmful. Unlike previous work, the model can output this token anywhere in the generation, not just in the beginning. Their loss encourages the model to give the same output as it would have without the \<rf\> token. They use few shot examples to show the model to use safety focused CoT reasoning after generating the \<rf\> token.

### Strengths
Their method is much more robust to prefill attacks than Fixed Pos RF, because that method can only output the rf token in the beginning.

They show their technique works much better than standard refusals in low resource languages, because a model trained in another language can generate the same \<rf\> token no matter the language.

I like using the \<rf\> token to encourage the Safety CoT, since it’s novel as far as I know, and it allows the model to recover from a false positive \<rf\> token.

### Weaknesses
The novelty of their core technique is relatively low, since the method is the same as previous work, but the model is able to output the \<rf\> token anywhere instead of only at the beginning.

It’s hard to tell if the technique has any benefit over CAT because their results are only a single run without error bars, and their results are similar to CAT.

Clarity
I find the loss in equation 1 confusing. It looks like it choses a random i, and then the loss encourages the model to output \<rf\> for every token between k and i. Is this correct? If so, what’s the purpose of only doing this between k and i? Why not encourage the model to output \<rf\> for every token between k and the last token?

Figure 4 should have whether the metric is DSR or utility in the figure itself, instead of just the caption (as figure 3 does).

### Questions
The paper says “To maintain model performance and reduce distribution shift as much as possible without increasing the likelihood of a harmful answer, we use a KL divergence on the tokens after the ⟨rf⟩:” If the goal is to have the model generate the same thing as it would have without the \<rf\> token, why not just avoid passing the generated \<rf\> token back to the model at all during decoding?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a generative approach to mitigating the harmfulness of LLMs by introducing a special red flag token into the model’s vocabulary. This token is trained to be generated when the LLM generates or is about to generate harmful content, enabling the model to explicitly learn the concept of harmfulness with minimal impact on its overall utility. The approach is designed to be complementary to existing safety methods and offers flexibility, allowing the token to be used either as a hard filter for unsafe responses or as a soft signal to trigger reflective, safety-focused Chain-of-Thought reasoning. Experiments show that this method, particularly when combined with adversarial training (RF-AT), provides strong robustness against various attacks, including pre-filling and sampling, while also demonstrating good cross-lingual generalization.

### Strengths
- The paper is (mostly - see [Note] in Weaknesses) well written and easy to understand
- Mitigating harmfulness in LLMs is a relevant topic
- The proposed method is well grounded and appears to achieve robust safety performance with a marginal compromise on model utility
- Despite the weaknesses listed below, the empirical evaluation of the proposed method is acceptable - but could be better

### Weaknesses
- No content provided regarding limitations or future work; See questions for examples of potential limitations
- Adding a comparison with defense methods from other categories (e.g., self-reflect and controlled text generation - example of a popular approach [1]) would make the paper's contribution more compelling
- The performance of one of the benchmarks (CAT) appears to be very comparable, if not better in some cases, to the proposed method
- Increasing the font size on some of the plots would significantly improve the paper's readability.
- [Note] On page 9, line 464-465, you have "As expected, translated attacks achieve break the model", which I'm guessing you mean "[...] translated attacks break [...]" - proofreading the paper might be necessary.
- Understandably, it is difficult to provide an anonymized version of the finetuned models. However, an anonymized repository could have been provided to assert the reproducibility of this work.

[1] Yang, K., & Klein, D. (2021). FUDGE: Controlled text generation with future discriminators. arXiv preprint arXiv:2104.05218.

### Questions
- Although this approach appears to improve safety over gradient-based attacks, isn't it still vulnerable to them? 
- Wouldn't an external safety approach be more robust to jailbreak attempts? Perhaps including a benchmark with this type of safety mechanism would allow answering this.   
- Is there a reason to use GPT-5 to judge refusals instead of models designed specifically to detect refusals like, for example, WildGuard?
- How often is the RF token triggered on harmless prompts? Perhaps it would be useful to show how a "false positive" would affect the performance (utility-wise) of the model, compared to when the model correctly assesses no risk, and/or the base model version.
- How is the DSR generalization across languages of CAT? How does your method affect utility across languages, as compared to the base model?
- How frequently is the red flag token generated more than once in a response? How does allowing the generation of this token multiple times help to increase model safety?

### Soundness
4

### Presentation
3

### Contribution
3
