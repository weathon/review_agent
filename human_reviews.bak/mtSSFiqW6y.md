# Judge Decoding: Faster Speculative Sampling Requires Going Beyond Model Alignment

- Decision: Accept (Oral)
- Scores: 6, 10, 8, 8

## Abstract
The performance of large language models (LLMs) is closely linked to their underlying size, leading to ever-growing networks and hence slower inference. Speculative decoding has been proposed as a technique to accelerate autoregressive generation, leveraging a fast draft model to propose candidate tokens, which are then verified in parallel based on their likelihood under the target model. While this approach guarantees to reproduce the target output, it incurs a substantial penalty: many high-quality draft tokens are rejected, even when they represent objectively valid continuations. Indeed, we show that even powerful draft models such as GPT-4o, as well as human text cannot achieve high acceptance rates under the standard verification scheme. This severely limits the speedup potential of current speculative decoding methods, as an early rejection becomes overwhelmingly likely when solely relying on alignment of draft and target.
We thus ask the following question: Can we adapt verification to recognize correct, but non-aligned replies? To this end, we draw inspiration from the LLM-as-a-judge framework, which demonstrated that LLMs are able to rate answers in a versatile way. We carefully design a dataset coined TokenCourt to elicit the same capability in the target model by training a compact module on top of the embeddings to produce ``judgements" of the current continuation. We showcase our strategy on the Llama-3.1 family, where our 8B/405B-Judge achieves a speedup of $9\times$ over Llama-405B, while maintaining its quality on a large range of benchmarks. These benefits remain present even in optimized inference frameworks, where our method reaches up to $141$ tokens/s for 8B/70B-Judge and $129$ tokens/s for 8B/405B on $2$ and $8$ H100s respectively.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In standard Speculative Decoding (SD), draft tokens are typically accepted or rejected during the verification phase based on whether they align with the target model distribution. However, this paper raises the concern that this verification process may be too stringent, resulting in high-quality draft tokens being rejected because they don’t align to the target distribution even though they may be valid and correct continuations. The authors try to address this concern by building upon the idea of LLM-as-a-judge and training a small judgment model that works on top of the target model latent embeddings and can be used to judge whether a draft is a correct continuation or not. Using this idea, the authors present speed ups using draft and target models in the Llama-3.1 family on a range of benchmarks and present a novel dataset (TokenCourt) used to train the target judge model.

### Strengths
- The authors clearly present and motivate their ideas around standard SD verification potentially being overly stringent and distinguishing between semantic correctness and target distribution “alignedness”. The motivation is straightforward to understand with the examples provided and it’s clear how standard verification can be arbitrarily strict depending on the target model resulting in less than desired inference efficiency using standard SD at times. 
- The reported speedups over standard SD are significant, largely in part due to the increase in the number of accepted tokens using this new judge decoding verification process compared to standard verification. However, the authors correctly point out also that this is not necessarily an apples to apples comparison given the loss of output distribution guarantees compared to standard SD.
- The investigations into token embeddings signal errors was particularly interesting in that “incorrect” statements in text could be pinpointed based on the target model’s latent embedding because of the models tendency to try to immediately rectify its response following an inaccuracy and the resulting dataset curated based on this could prove useful to others if released

### Weaknesses
- Given the fact that verification is being changed so there is no longer a guarantee of output distribution match, the speedup comparisons are no longer apples to apples and have to also be paired with quality comparisons to ensure quality is maintained. Additionally, given the way the judge model is trained on “correct” and “incorrect” inputs, it seems that for novel tasks, a new tuning dataset must be hand curated for each task to train the judgment model to recognize “correctness” making the overhead of applying a technique like Judge Decoding significant compared to other SD modifications.
- The notion of “correctness” is easiest to understand in the context of text generation tasks that are Question&Answer based. However, it’s unclear how this idea would work for more open-ended creative generation tasks where “correctness” may not be as easily defined and matching the target model output distribution may be ideal.
- An additional condition to applying Judge Decoding is for the draft model to be of high-quality that can produce longer valid generations otherwise the approach may not provide that much of a speedup. 
- The reported experimental results are somewhat limited such as the out of distribution performance results being restricted to only a few lines and the full results not being reported in the appendix at least in comparison to the baselines etc. Additionally, it’s unclear what task is being reported on for Table 1 whereas Figure 6 shows accuracy across a variety of tasks. It would be helpful to also see the relative speedups across each of these variety of tasks as well.
- For the most part, the authors are forthcoming about weaknesses and limitations to their judge decoding approach and many of these points will align. However, the amount of pre-requisites needed to apply the technique as well as required overhead for verification of maintained quality due to the loss of output distribution alignment are substantial.

### Questions
- Do the authors plan on releasing their TokenCourt dataset?
- What happens when the target LLM isn’t one of the models used to generate the correct/incorrect responses to questions in TokenCourt used to train the linear judge module? Is this a necessary requirement for this technique to work? I ask because of a scenario where one may want to re-use a curated TokenCourt dataset for a new target model from an entirely different family of models to avoid having to hand annotate again.
- Looking at the showcased TokenCourt examples, one question is if Judge Decoding may result in potentially verbose outputs. And in those cases, even though more draft tokens are being accepted and decoded per second and the full output is semantically correct, it is not as concise or clear as the original target model output if standard verification were being applied?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
The authors propose "Judge Decoding," a novel approach to speculative sampling that challenges the fundamental assumption that verification must be based on model alignment. Instead of using likelihood comparisons, they train a simple linear classifier on the last embedding layer of the target model to judge token quality directly. Their key insight is that current speculative decoding methods reject many high-quality tokens simply because they don't perfectly align with the target model's preferred phrasing. By analyzing embeddings and observing how models attempt to correct errors, they show that correctness information is already present in the token representations. They leverage this insight to create a lightweight judge module (16.4k parameters) trained on their carefully curated TokenCourt dataset. The method achieves up to 9× speedups over standard decoding, reaching unprecedented speeds of 129 tokens/s for Llama-405B, while maintaining quality across a comprehensive range of benchmarks.

### Strengths
- Clever idea that was well-executed. The paper is well-written and the experiments make sense. The results are impactful in practical terms (people should use this) and for spawning future research (there are many possible variations on how to train the linear classifier).

- I liked that the authors decided to go with the simple linear classifier since it worked best, rather than overcomplicating things

- I also liked the systematic investigation with illustrative examples for how judge decoding differs from speculative decoding

- Achieves state-of-the-art performance (141 tokens/s for 8B/70B-Judge, 129 tokens/s for 8B/405B) on standard hardware (2-8 H100 GPUs).

### Weaknesses
I would have liked if the authors addressed the possible impact of their method on less binary (correct vs incorrect) metrics like reward model scores or chatbot ELO scores. I could imagine to get a more noticeable regression on those, since the writing style of models is more important in that setting. Perhaps something for the discussion?

The Burns et al 2022 paper (https://arxiv.org/abs/2212.03827) does some nice work on eliciting concepts with linear probes which could be relevant here.

### Questions
No questions

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces Judge Decoding, a speculative decoding enhancement that improves token acceptance by evaluating contextual correctness rather than strict alignment with the target model. Traditional speculative decoding often rejects accurate tokens from draft models due to misalignments rather than incorrectness, limiting speedups. By training a lightweight “judge” module on a custom TokenCourt dataset, Judge Decoding accepts more valid tokens, achieving up to 9× faster generation speeds with maintained quality on the Llama-3.1 model family, demonstrating a promising approach for efficient large language model inference.

### Strengths
The paper is well-written. The argument is clear and reasonable. The proposed method shows effectiveness.

### Weaknesses
I do not see significant weakness.

But the sentence "To speed up inference in such a setting, Chen et al. (2023); Leviathan et al. (2023) concurrently propose speculative decoding (SD)" in the introduction section is incorrect.
At least, there is a much earlier one in 2018 that proposed speculative decoding: "Blockwise Parallel Decoding for Deep Autoregressive Models" by Mitchell Stern, Noam Shazeer, and Jakob Uszkoreit. The author skipped a large amount of literature on speculative decoding before 2023.

Authors are required to address this issue.

### Questions
NA

### Soundness
3

### Presentation
4

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
The submission proposes Judge Decoding (JD), a variation of Speculative Decoding (SD) for LLMs. Like SD, JD uses a small draft model to propose output, and uses a larger model to check the output. The difference between SD and JD is in how the check is performed: where SD focuses on checking that the draft model's output is exactly like the larger model's (i.e., effectively per-token equality), JD uses a weaker criterion, based on the embeddings of the final layer of the larger (judge) model. The downside of this weaker criterion is that the output of the model is not guaranteed to be equivalent to the larger model's output anymore, but experiments indicate a significant potential speedup.

### Strengths
* Clear presentation, covering both technical details and high-level ideas
* Comprehensive experiments that emphasise the strengths of the proposed method
* Reported speedups seem significant, and seem like they could be very useful for practical serving of LLMs

### Weaknesses
* Some of the results (e.g., lack of agreement between different models / human drafts and models) seem to be obvious and do not help to make the argument
* The use of a disjunctive criterion (standard SD accepts or JD accepts) is a bit surprising, and left me wondering in what cases (and how often) JD rejects when SD accepts. The paper would be strengthened by investigating this more closely.
* Sect. 5.3 (OOD performance) is very important for deployment of the proposed method, and so it would be useful to expand on this. In particular, Sect. 4.1/"Dataset Curation" does not go into much detail into what distribution the training data follows, and which tasks are covered well. It would be helpful to understand whether there is a strong alignment between the TokenCourt training data and the considered evals.

### Questions
* Fig 2/Fig 3: It would be useful to report the number of proposed tokens as well, either explicitly or by presenting "ratio of accepted tokens" rather than an absolute number.
* One surprising result is not investigated: using Llama 405B to draft outputs of the 8B model on GSM8K (lower right of Fig. 2) seems to work less well than the other way around, even though the models are obviously well-aligned. I would have expected a basically equal result here (following the intuition that the larger model is effectively a superset of the smaller model), and was very surprised to see a lower score.

### Soundness
4

### Presentation
4

### Contribution
4
