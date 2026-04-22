# Preemptive Detection and Steering of LLM Misalignment via Latent Reachability

- Avg Score: 5.00
- Decision: Reject
- Scores: 2, 6, 6, 6

## Abstract
Large language models (LLMs) are now ubiquitous in everyday tools, raising urgent safety concerns about their tendency to generate harmful content. The dominant safety approach -- reinforcement learning from human feedback (RLHF) -- effectively shapes model behavior during training but offers no safeguards at inference time, where unsafe continuations may still arise.
We propose BRT-Align, a reachability-based framework that brings control-theoretic safety tools to LLM inference. BRT-Align models autoregressive generation as a dynamical system in latent space and learn a safety value function via backward reachability, estimating the worst-case evolution of a trajectory. This enables two complementary mechanisms: (1) a runtime monitor that forecasts unsafe completions several tokens in advance, and (2) a least-restrictive steering filter that minimally perturbs latent states to redirect generation away from unsafe regions. Experiments across multiple LLMs and toxicity benchmarks demonstrate that BRT-Align provides more accurate and earlier detection of unsafe continuations than baselines. Moreover, for LLM safety alignment, BRT-Align substantially reduces unsafe generations while preserving sentence diversity and coherence. Qualitative results further highlight emergent alignment properties: BRT-Align consistently produces responses that are less violent, less profane, less offensive, and less politically biased. Together, these findings demonstrate that reachability analysis provides a principled and practical foundation for inference-time LLM safety.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
Their method trains a classifier to determine if a partial LLM generation will be completed harmfully. In RL-BTR-ALIGN, the labels are derived using the safety of the partial generation by itself, and the discounted safety of the rest of the generation. In SAMPLE-BRT-ALIGN, the classifier is always trained to predict the safety of the final generation given the partial generation.

They also develop a method to steer the model away from the unsafe generation once it’s detected.

### Strengths
Their work is mostly easy to understand.

The novelty is sufficient with the reachability analysis and steering method. Overall I like the methodology of applying control theory to LLM safety.

Their inference time steering technique substantially increases safety rate, with only a small coherence cost. They also calculate results across multiple training seeds, making the significance clear.

### Weaknesses
I don’t think their baselines are correct in figures 2 and table 2.
Their baselines either predict everything is safe or everything is unsafe. I don’t think this is correct, because the SAP paper reports a classification accuracy of 86% for Llama2-7B on BeverTails (figure 6). (The BeaverTails dataset is relatively balanced between safe and unsafe: “44.64% were assigned the safe meta-label, while the remaining 55.36% were categorized under the unsafe meta-label.”)

Clarity

In section 4.3, they don’t explain how the argmax part of equation 1 is computed. This is a significant issue since it’s one of the main parts of the method.

The definition of ℓ(zt) is confusing. z_t is the layer-l embedding of the last emitted token, but section 5 says ℓ(zt) is calculated using a classifier given the token sequence y_0 to y_t.

They say “we assume that LLM deterministically selects the most likely token”, but it doesn't discuss the consequences of this assumption. Will their approach work with non deterministic decoding strategies? 

Minor points

The paper could better motivate why it’s necessary to classify partial generations as leading to unsafe outputs. For example, an alternative would be to generate the entire response from the LLM, and then classify the entire response as safe or unsafe before showing it to the user, or acting on it. The paper could mention that needing to wait until the entire response is generated before classifying will increase streaming latency.

### Questions
none

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes BRT-ALIGN, a reachability-based framework to conduct safety alignment for LLM during inference time. It models the autoregressive generation of LLM as a dynamical system in latent space and then learn a safety value function via backward reachability. Building on this, BRT-ALIGN is able to forecast the unsafe tokens during inference on-the-fly and take proactive intervention. It can also conduct a guided-perturbation on the latent states to steer the generation away from unsafe regions. Evaluation results on 5 LLMs demonstrates the effectiveness of BRT-ALIGN on improving safety alignment compared to baseline approaches.

### Strengths
- The paper studies the safety alignment of LLMs, which is a critical problem for the community.
- The proposed BRT-ALIGN brings the control theory and inference-time safety alignment, which is novel and very interesting.
- The evaluation results demonstrate the effectiveness of BRT-ALIGN, compared to other control-based alignment approaches.

### Weaknesses
- Lack of comparison on several SOTA alignment techniques.
- Lack of comparison on jailbreak attacks.
- The preservation of utility after applying BRT-ALIGN shall be justified by evaluating on more challenging benchmark.

### Questions
I think it is an interesting paper that introduces a novel technique. However, my main concerns lie in the evaluation section, which could be expanded to better contextualize the performance relative to existing methods.

1. Similar properties, such as preemptive detection, have been discussed in prior none control-based alignment approaches, for example, Circuit Breaker [1]. A direct comparison would help clarify the unique contribution of this work.

2. Currently, the evaluation is primarily conducted on raw harmful prompts. It would strengthen the paper to also test the proposed method against harmful prompts generated by jailbreak techniques such as GCG [2], PAIR [3], and AutoDAN [4].

3. To more comprehensively assess the utility preservation of BRT-ALIGN, I recommend including results from benchmarks designed to measure over-refusal behaviors, such as ORBench [5] and XSTest [6].

---
Reference 
---

[1] Zou A, Phan L, Wang J, et al. Improving alignment and robustness with circuit breakers[J]. Advances in Neural Information Processing Systems, 2024, 37: 83345-83373.

[2] Zou A, Wang Z, Carlini N, et al. Universal and transferable adversarial attacks on aligned language models[J]. arXiv preprint arXiv:2307.15043, 2023.

[3] Chao P, Robey A, Dobriban E, et al. Jailbreaking black box large language models in twenty queries[C]//2025 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML). IEEE, 2025: 23-42.

[4] Liu X, Xu N, Chen M, et al. Autodan: Generating stealthy jailbreak prompts on aligned large language models[J]. arXiv preprint arXiv:2310.04451, 2023.

[5] Röttger P, Kirk H R, Vidgen B, et al. Xstest: A test suite for identifying exaggerated safety behaviours in large language models[J]. arXiv preprint arXiv:2308.01263, 2023.

[6] Cui J, Chiang W L, Stoica I, et al. Or-bench: An over-refusal benchmark for large language models[J]. arXiv preprint arXiv:2405.20947, 2024.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper centres on introducing a reachability-based framework, inspired by control theory, to mitigate unsafe LLM responses by predicting potential harmful continuations and steering the model toward safer outputs, a method referred to as BRT-ALIGN. It positions this as the first application of reachability analysis for inference-time LLM safety, offering a principled approach to both anticipate and prevent harmful text generation.

### Strengths
- The paper is well-structured and clearly written, with helpful figures and explanations that make the novel reachability-based approach easy to follow.

- The approach is quite novel as this paper claims to be the first to apply backward reachability in LLM latent space, enabling formal prediction and prevention of unsafe continuations.

- The paper presents a comprehensive safety framework that integrates a reachability-based monitor for early detection with a lightweight steering filter that subtly guides the model’s latent states away from unsafe regions.

- The inclusion of multiple evaluation metric provides a holistic picture of how the proposed method outperforms the existing strategies

### Weaknesses
- There exist some methodologies that have worked on safe generation of LLMs at inference time (Safe Infer : https://arxiv.org/abs/2406.12274, Safe Decoding : https://arxiv.org/abs/2402.08983, Self-CD : https://arxiv.org/abs/2401.17633) Including these as baselines would strengthen the comparison and contextualize the proposed approach more comprehensively.

- One of the primary assumptions of the framework is greedy decoding (line 144) to create a deterministic latent trajectory. This removes randomness and makes reachability tractable, but it raises concerns as many LLM applications use sampling (top-k, nucleus) for diversity. It is unclear how well BRT-Align would generalize when the model samples instead of greedily choosing tokens. This assumption may limit real-world applicability.

- The evaluation focuses solely on offensive/toxic language as the failure mode. The failure set is defined via an offensive-language classifier. While toxicity is important, LLM misalignment encompasses other harms (misinformation, self-harm, illegal advice, privacy breaches, etc.). The paper acknowledges this limitation, but currently BRT-Align is only validated on one dimension of harm. Its effectiveness for other harmful content is untested.

- It would be interesting to see if this methodology can somehow be modified to include proprietary LLMs as currently it is limited to open source models.

- The evaluation uses automatic metrics (classifier labels for safety, cosine similarity for coherence, n-gram diversity). While these are reasonable, human evaluation of output quality and safety would strengthen the validation. 

- In some places, few notations are not previously stated making it a bit difficult to follow. For instance., W in line 134, fLM  in line 135, etc

### Questions
Please refer to the weakness section and address those concerns

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
This paper introduces BRT-ALIGN, a novel framework for inference-time safety alignment of LLMs. The core idea is to model LLM token generation as a dynamical system in a latent embedding space and apply reachability analysis, a technique from control theory. By computing a "backward reachable tube" (BRT), the method can identify latent states that will inevitably lead to unsafe completions (e.g., toxic or harmful content). This enables a steering mechanism that minimally perturbs the latent state to guide generation toward safe outputs. The authors propose two variants for learning the required safety value function: RL-BRT-ALIGN (using a Bellman recursion) and SAMPLE-BRT-ALIGN (a simpler supervised approach). Through comprehensive experiments on five open-source LLMs and three safety benchmarks, the paper demonstrates that BRT-ALIGN significantly outperforms existing control-theoretic baselines in both the accuracy of unsafe content detection and the effectiveness of safety steering, all while incurring lower computational overhead.

### Strengths
1. Framing generation as a dynamical system and using the BRT to formally define the set of "doomed" trajectories is a highly principled approach that moves beyond reactive, token-level filtering. It provides a strong theoretical foundation for anticipating misalignment behaviors.
2. A key contribution, well-supported by the results, is the model's ability to detect unsafe completions 7-10 tokens in advance. This "early warning" system is crucial for building robust and trustworthy AI systems.

### Weaknesses
1. While the application of the formalism of reachability analysis and the backward reachable tube is novel, the more general idea of learning a value function to predict future unsafe states in LLMs has been explored in prior work [1][2].
2. The evaluation is missing a crucial and straightforward baseline: prompt engineering. A simple baseline that prepends a safety instruction (e.g., "Do not generate any offensive or harmful content.") to the user's prompt could significantly improve the safety rate with little inference overhead. Without comparing against such a baseline, it is difficult to assess the practical necessity and cost-benefit of the proposed control-theoretic approach.

Related Work:

[1] Systematic Rectification of Language Models via Dead-end Analysis (https://openreview.net/forum?id=k8_yVW3Wqln)

[2] Decoding-time Realignment of Language Models (https://arxiv.org/pdf/2402.02992)

### Questions
1. For the steering filter in Equation 1, the search for a better latent state is performed by sampling within an L2-norm ball. How many samples are typically drawn to perform this maximization? 
2. The model "Mistral-8B-Instruct-2410" is consistently misspelled as "Ministral-8B-Instruct-2410" (e.g., line 245, 387, 428, and in the references).

### Soundness
3

### Presentation
3

### Contribution
2
