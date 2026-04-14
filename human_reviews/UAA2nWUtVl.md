# Cascade Reward Sampling for Efficient Decoding-Time Alignment

- Decision: Reject
- Scores: 6, 5, 6, 6

## Abstract
Aligning large language models (LLMs) with human preferences is critical for their deployment. Recently, decoding-time alignment has emerged as an effective plug-and-play technique that requires no fine-tuning of model parameters. However, generating text that achieves both high reward and high likelihood remains a significant challenge. Existing methods often fail to generate high-reward text or incur substantial computational costs. In this paper, we propose Cascade Reward Sampling (CARDS) to address both issues, guaranteeing the generation of high-reward and high-likelihood text with significantly low costs. Based on our analysis of reward models (RMs) on incomplete text and our observation that high-reward prefixes induce high-reward complete text, we use rejection sampling to iteratively generate small semantic segments to form such prefixes. The segment length is dynamically determined by the predictive uncertainty of LLMs. This strategy guarantees desirable prefixes for subsequent generations and significantly reduces wasteful token re-generations and the number of reward model scoring. Our experiments demonstrate substantial gains in both generation efficiency and alignment ratings compared to the baselines, achieving five times faster text generation and 99% win-ties in GPT-4/Claude-3 helpfulness evaluation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper focuses on reward-guided decoding-time alignment. They propose Cascade Reward Sampling (CARDS), which is mainly based on two claims:

- They claim to have analyzed the properties of reward models (RMs) on incomplete text, and hypothesize that RMs can serve as approximations for value functions.
- They show that values of predictive uncertainty can be a segmentation signal, and thus can divide the generation of an entire response into multiple steps (segments). 

Combining two claims, they establish the target distribution for sampling a new semantic segment. Therefore, they finally sample one segment each step, get the reward, and use rejection sampling to align the distribution to what they have established.

Experiments are conducted on HH-RLHF dataset. They have compared the performance of CARDS with RAD/ARGS and naive rejection sampling, showing general advantages.

### Strengths
## Originiality
- The segmentation trick with uncertainty is novel and interesting, and it deserves to be shared with a broad audience.
- The assumption on reward distribution is original, though may not be true.

## Clarity
- This paper is well-written and well-organized.
- Most figures and tables are friendly to the readers.
- The intuitions are clearly expressed.

## Significance
- The weak assumption on the expressibility of reward model is more reasonable than prior works.
- This work has improved the performance of reward-guided decoding-time alignment, compared with baselines.

### Weaknesses
## Major
- The reviewer cannot buy the idea that reward model is a value function. Only the last signal of reward model is trained, and the intermediate prediction is more like black box. There is a well-known paper [5] showing that DPO model acts like Q-function, but the result only holds for credit assignment setting.
- There is a new paper showing that "partial reward" may not be correlated to "full reward". Please see Appendix C.3 of [6].
- There is no non-trivial mathematical proof. So it would be better to remove the claim that "We first **rigorously** analyze the properties of reward models". The authors have conducted some experimets to support their intuitions or assumptions, that are great, but still far from being called "rigorous", since there is no theoretical guarantee.
- Lack of baselines. CARDS is only compared with two baselines, RAD/ARDS (these two are the same), and naive rejection sampling. Many reward-guided decoding-time alignment approaches have emerged in these two years. For example, controlled-decoding [1] can be a baseline, which samples next token from $\pi(y)\exp(r(y)/\beta)$. ([1] is different from RAD/ARDS, since it doesn't need a top-k sampling at first.) And the reviewer guesses that [1] should be faster than CARDS.
## Minor
- Equation 11(b) only holds for soft-RL, which should be made clear to the readers.
- Lack of datasets. The experimental results on HH-RLHF are acceptable but limitted. More datasets are worth exploring. For example, Ultrafeeback[2], HelpSteer[3], Beavertails[4] are worth trying.

*The empirical approach is generally fine to the reviewer, but unfortunately, the claims have many problems. The reviewer is willing to raise the rating if the issues can be solved.*

[1] Controlled Decoding from Language Models, https://arxiv.org/abs/2310.17022

[2] UltraFeedback: Boosting Language Models with Scaled AI Feedback, https://arxiv.org/abs/2310.01377

[3] HelpSteer: Multi-attribute Helpfulness Dataset for SteerLM, https://arxiv.org/abs/2311.09528

[4] BeaverTails: Towards Improved Safety Alignment of LLM via a Human-Preference Dataset, https://arxiv.org/abs/2307.04657

[5] From r to Q∗: Your Language Model is Secretly a Q-Function, https://arxiv.org/abs/2404.12358

[6] TreeBoN: Enhancing Inference-Time Alignment with Speculative Tree-Search and Best-of-N Sampling, https://arxiv.org/abs/2410.16033

### Questions
- The equation (4) is not well defined. The expectation is taken based on which distribution?
- Which reward model did you use for Figure 2b,3,4? Did you only examine one reward model to support claims shown in these figures?
- Can this approach be applied to some smaller and weaker reward models, like GPT2 models [1][2]? This would be important for some groups with restricted computation resources.
- Can this approach be possibly applied to multi-objective alignment, like what many reward-guided decoding-time algorithms [3][4] can do? (Anyway, they are concurrent works, so there is no need to compete with them.)

[1] https://huggingface.co/Tristan/gpt2_reward_summarization

[2] https://huggingface.co/Ray2333/gpt2-large-helpful-reward_model

[3] PAD: Personalized Alignment at Decoding-Time, https://arxiv.org/abs/2410.04070

[4] GenARM: Reward Guided Generation with Autoregressive Reward Model for Test-time Alignment, https://arxiv.org/abs/2410.08193

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper introduces a decoding-time alignment method for large language models (LLMs) that ensures high-reward and high-likelihood text generation with reduced computational costs. CARDS uses rejection sampling to create text by dynamically determining segment lengths based on the predictive uncertainty of LLMs, significantly enhancing efficiency. This method also maintains fluency and aligns closely with human preferences, demonstrating substantial improvements over existing methods in both speed and alignment accuracy during text generation.

### Strengths
1. The article raises an excellent question on how to enhance the efficiency of alignment during the reasoning phase.
2. Many designs in the article's methodology are interesting, such as "Our method leverages the comprehension ability of pre-trained LLMs for segmentation."
3. The experimental results of CARDS are outstanding.

### Weaknesses
1. The presentation of the article is somewhat difficult to follow. For example, Figure 1, which explicates the contributions mentioned in the introduction, requires the integration of content from many sections later in the text to be understood. Moreover, Section 4.1.1 repeatedly refers to Figure 2c without clearly explaining how Figure 2c is produced and its detailed meaning. Section 4.2.1, however, does not mention Figure 2c at all.
2. Many intermediate conclusions in the methodology lack theoretical support and are merely based on simple experiments and conjectures. For instance, "a full-response reward will be high given a high-reward prefix" and "This is because initiating a new segment is more unpredictable than continuing an existing one."
3. I think previous works [1] also assume Lemma 1 is valid; however, this paper still does not provide convincing proof, thus, the contribution in this part seems incremental.
4. CARDS appears to be a method for enhancing the efficiency of decoding-time alignment, but I have no idea why CARDS could lead to significant performance improvements in the experiment.

[1] Args: Alignment as reward-guided search.

### Questions
1. What do you mean by "However, while some of the existing decoding-time alignment methods still struggle with the trade-off between alignment and fluency, they all encounter significant efficiency challenges due to auxiliary steps added to their decoding process."?
2. In line 239, "Therefore, the above observation also suggests that RMs can be used as value functions on semantically complete preffxes." Can you explain why?
3. In Lemma 1, I still cannot understand why can we assume "Assuming the reward models are equivalent to value functions when evaluating semantically complete prefixes". Is there any previous literature proof of this?
4. What if the prefixes is not semantically complete?

### Soundness
2

### Presentation
2

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
This paper propose a novel segment-based sampling method for efficient decoding-time alignment, leveraging rejection sampling to iteratively generate small semantic segments of high reward.

### Strengths
1.	This paper conducts a rigorous analysis of reward models and demonstrates the RMs can serve as value functions on semantically complete segments.
2.	The generation method is segment-based and the length of segment is dynamic.
3.	The experiment is adequate and reasonable and the paper is well written.

### Weaknesses
1.	The experiments are conducted on 7B models. The method could be verified on more larger models.
2.	The parallelization scheme of dynamic segmentation whether has a slow inference time when the batch size is larger.

### Questions
The same as above

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
3

### Summary
- The paper focuses on enhancing decoding-time alignment methods, particularly addressing the high computational costs associated with existing approaches.
- It introduces **Cascade Reward Sampling (CARDS)**, a novel method that samples segments, leveraging the insight that reward models (RMs) favor high-reward prefixes. CARDS incrementally generates high-reward semantic segments, aiming to achieve optimal reward in the final output by prioritizing these segments.
- Experimental results demonstrate significant improvements in both performance and speed over baseline methods.

### Strengths
- The shift to segment-level sampling, along with the use of uncertainty as a termination signal for segments, presents a unique and novel approach.
- The results show significant gains on performance and speed, renders this approach very practical.

### Weaknesses
- The reliance on target score poses a notable limitation. How can one determine this? Different RMs provide values in different scales. 
    - Exploring alternative sampling strategies, like sampling multiple segments per step and selecting the highest-scoring option before proceeding greedily, could be beneficial (atleast as a baseline comparison)
- The experiments are conducted solely on the HH-RLHF dataset, limiting the generalizability of the findings. Especially, HH-RLHF is a very simple dataset as many inference time methods are already better than trained methods such as PPO and DPO. Testing on few more datasets would solidify the findings.

### Questions
Suggestions:
- One could include more baselines such as In-Context learning for alignment (Rethinking alignment via in-context learning), Best of N based on full sequence rewards

### Soundness
3

### Presentation
3

### Contribution
3
