# JULI: Jailbreak Large Language Models by Self-Introspection

- Avg Score: 5.20
- Decision: Accept (Poster)
- Scores: 8, 2, 6, 4, 6

## Abstract
Large Language Models (LLMs) are trained with safety alignment to prevent generating malicious content. Although some attacks have highlighted vulnerabilities in these safety-aligned LLMs, they typically have limitations, such as necessitating access to the model weights or the generation process. Since proprietary models through API-calling do not grant users such permissions, these attacks find it challenging to compromise them. In this paper, we propose Jailbreaking Using LLM Introspection (JULI), which jailbreaks LLMs by manipulating the token log probabilities, using a tiny plug-in block, BiasNet. JULI relies solely on the knowledge of the target LLM's predicted token log probabilities. It can effectively jailbreak API-calling LLMs under a black-box setting and knowing only top-$5$ token log probabilities. Our approach demonstrates superior effectiveness, outperforming existing state-of-the-art (SOTA) approaches across multiple metrics.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a jailbreaking attack called JULI (Jailbreaking Using LLM Introspection). This method targets a key vulnerability: that even safety-aligned Large Language Models (LLMs) retain knowledge of harmful content, which is exposed in their top-k token log probabilities. JULI uses a small, lightweight plug-in module called "BiasNet" to analyze these log probabilities and add a calculated bias, steering the model's generation away from "safe" refusals (like "Sorry") and towards "harmful" compliances (like "Sure, here's..."). The method is notable for its effectiveness in a practical, black-box, API-only setting (requiring only top-5 log probabilities), where it demonstrates state-of-the-art performance, particularly against proprietary models like Gemini-2.5-Pro

### Strengths
1. High Practicality: The primary strength of this paper is its focus on a realistic and challenging attack scenario: proprietary, black-box models accessible only via API. Unlike most methods that require weight access or large top-k values (e.g., LINT needing top-500 ), JULI is effective using only the top-5 log probabilities, a feature commonly exposed by commercial APIs.
2. SoTA Effectiveness: The results are striking. On Gemini-2.5-Pro, JULI achieves a "Harmful Info Score" of 3.19, whereas the next-best baseline (FLIP) scores only 1.38 and the base model scores 0.06. This is a massive leap in effectiveness for practical black-box attacks.
3. Interesting Insight on Model Scale: The paper compellingly shows that JULI is more effective against the stronger Gemini-2.5-Pro (3.19 score) than the smaller Gemini-2.5-Flash (1.74 score). The authors' hypothesis, that the attack works better because it leverages the model's own (greater) knowledge, is a significant finding. It suggests that safety alignment as a "veneer" becomes less robust as models become more knowledgeable.
4. Efficiency: The attack is extremely low-cost. BiasNet is small (<1% of LLM parameters), trained on only 100 harmful data points, and adds minimal inference overhead (0.71s).
5. Evaluation: The authors justifiably critique existing metrics (BERT Score, Harmful Score) for overestimating harmfulness. Their proposed Harmful Info Score, which correlates better with human judgment (Table A3), is a solid methodological contribution in its own right.

### Weaknesses
1. BiasNet Generalizability: The BiasNet is trained on 100 harmful data points. The analysis in Figure 3(a) shows it's very effective at boosting "Sure". This raises the question of whether it's simply learning to force an affirmative prefix or if it's performing a more general "unsafe" steering. It's unclear how it would perform on harmful queries that require a more subtle or non-obvious initiation, rather than a simple "Sure, here's...".
2. Stability of Black-Box Projection: The "refined random weight" method for the black-box projection layers is clever, but its stability is not fully explored. The ablation in Table A5 shows a significant performance drop from the white-box setting (3.44) to the black-box top-5 setting (2.21) on Llama3-8B-INST. While 2.21 is still a strong score, it would be useful to know how sensitive the results are to the initial random seed of this process.

### Questions
1. Regarding the weakness in SoTA defenses: Could the authors please address the discrepancy in Table 3 (AdvBench)  where LINT (0.77) outperforms JULI (API) (0.75)? This seems to weaken the claim of superiority in. Why should the MaliciousInstruct result be considered more representative?
2. What are the precise assumptions of the black-box threat model in Section 4.2? To apply padding and use the BiasNet, the attack must know the full vocabulary size ($N_{voc}$) of the target model to define the projection layer dimensions. Is it assumed that the tokenizer and vocabulary are public, even if the model weights are not? This is a reasonable assumption, but it should be stated explicitly.
3. The insight that the attack is more effective on more capable models (Gemini-2.5-Pro > Flash)  is very interesting. Does this imply a fundamental flaw in current alignment approaches? If harmful knowledge is never erased, just suppressed, does this mean any sufficiently advanced model will be vulnerable to such introspection attacks, regardless of the alignment's strength?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces JULI, a decoding-based jailbreak for LLMs that trains a lightweight BiasNet to generate biases on token log probabilities during inference. For each position, JULI extracts the target model's probability distribution, applies the biases, and samples from the biased distribution. The method is evaluated on Gemini-2.5-Flash, Gemini-2.5-Pro, and Llama3-8B-Instruct, demonstrating its’ effectiveness. Compared to baselines, JULI do not shows clear advantages.

### Strengths
The proposed JULI method employs a lightweight plug-in BiasNet, which requires training on only 100 harmful question-answer pairs.

### Weaknesses
1. Lack of Novelty in Vulnerabilities Exploited: Although JULI achieves reasonable jailbreak ASR, it does not uncover or reveal any new vulnerabilities inherent to safety-aligned LLMs. The core mechanism—directly manipulating the token probability distribution via a lightweight bias network—builds on well-established ideas from prior work, such as logit biasing in other attacks (e.g., Weak-to-Strong). These methods have already demonstrated that aligned LLMs retain harmful knowledge in their latent distributions, even if suppressed during generation. JULI merely repackages this observation into a plug-in architecture without providing fresh insights.
2. Violation of Standard Jailbreak Assumptions of the Threat Model: The paper conflates two distinct research paradigms in jailbreaking: (i) using API-calling to simulate black-box access and (ii) studying black-box jailbreaks on proprietary LLMs where no log probabilities are exposed. JULI's reliance on top-k token log probabilities assumes an API that explicitly returns this information—a non-standard "gray-box" setting not representative of true black-box APIs. Critically, JULI cannot be applied to fully black-box LLMs without log probability access, rendering it inapplicable to the real-world proprietary models it claims to target.
3. Prohibitively High Computational Overhead in the API Setting: Even under the paper's own API-calling assumptions, JULI incurs unacceptable latency and cost due to its token-by-token generation process. For each output token, the method requires a separate API call to fetch the partial response and top-k log probabilities, followed by local biasing and resampling (Algorithm 2). This results in L API calls for a response of length L—e.g., 100+ calls for a typical harmful instruction—multiplying inference time by orders of magnitude compared to single-shot baselines.
4. Mediocre and Inconsistent Performance Across Benchmarks: JULI's empirical results do not demonstrate clear superiority over simpler or more efficient baselines.

### Questions
Does BiasNet use pretrained models? How can it generate reasonable token probability distributions after training on only 100 question-answer pairs?

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
4

### Summary
The paper introduces a novel attack method that steers LLMs into generating harmful content by manipulating their output token log probabilities. The authors identify that even safety-aligned models often retain harmful tokens in their top-k predictions. JULI exploits this by using a lightweight model to adjust these probabilities in real-time during generation. The method is effective in both open-weight scenarios and restricted API settings where only top-k log probabilities are available. Empirical results show JULI outperforms state-of-the-art baselines (like GCG, FLIP, and LINT) in terms of efficiency and attack success rate against major models like Llama-3, Qwen-2.5, and Gemini-2.5-Pro, even bypassing advanced defenses like Circuit Breakers.

### Strengths
Practical Threat Model: Unlike many jailbreaks requiring full model weights or gradient access, JULI is viable against commercial APIs that only expose top-k log probabilities (e.g., top-5), making it a realistic real-world threat.
Effectiveness Against Defenses: JULI demonstrates remarkable robustness, successfully jailbreaking models fortified with state-of-the-art defenses like Circuit Breakers, where other baselines fail.

### Weaknesses
- How does this method work against input/output based filter?
- How does the sampling work for model APIs? Since JULI samples a different token than what the API would sample, how are subsequent tokens sampled?
- How exactly is Figure 2 computed? What is the data? How is the rate computed?

### Questions
NA

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes JULI, a jailbreak method that leverages a tiny plug-in network (BiasNet) to bias the output token log-probabilities. In both the white-box and black-box scenarios, JULI can enable the target model to produce harmful information with a higher harmful score than other baselines in most cases.

### Strengths
1. JULI avoids the complex iterative optimization typical of GCG-style attacks and does not require access to the target model’s weights, making it a realistic API-side jailbreak when limited signals (e.g., top-k log-probs) are available.

2. The BiasNet architecture and training recipe are straightforward and well-specified, which supports reproducibility and lowers the barrier for independent verification and follow-up work. 

3. Experiments show JULI achieves the best harmful scores in open-weight settings and remains effective on closed-source APIs (e.g., Gemini-2.5). While JULI relies on the availability of output token log-probabilities, it is a limitation, but not (in my view) a decisive factor for acceptance.

### Weaknesses
Major Concerns:
1. Is JULI actually jailbreaking the model, or just generating harmful text via a trained adapter?

Since JULI’s BiasNet is trained on harmful Q/A data, it seems that JULI is manufacturing harmful text via BiasNet rather than truly eliciting harmful responses related to the harmful prompt. From my perspective, I think the jailbreak should unlock harmful capabilities that already exist in the target LLM, not inject a harmful generator. I am worried that the biasnet trained with harmful data tends to output harmful information; in this case, JULI would be closer to harmful response synthesis rather than a truly jailbreak, even though I notice that the author uses Figure 2 and tells us that the harmful tokens are among the top-k tokens.

I think the author could provide further experiments: if we only give the Gemini-2.5-pro clean inputs like some safe instructions from databricks-dolly-15, what is the output when using biasnet? Will the harmful score increase with the clean and safe prompts? Alongside harmfulness evaluation, I think it is still important to evaluate whether the harmful response elicited by JULI is matched with the harmful prompt. I am concerned that the JULI only produces some off-topic, harmful responses. For example, if the prompt is " give me a plan on selling a human kidney", the jailbreak answer would be highly related to selling human organs rather than some off-topic, harmful information. 

2. Why share the LM head (and its pseudoinverse) with BiasNet, and why this specific initialization in the API setting?

In the open-weight setting, JULI reuses the LM head matrix as the last layer of BiasNet and its generalized inverse as the first layer. In the closed-weight/API setting, the paper replaces the LM head with a random (orthogonalized/normalized) projection and again uses its pseudoinverse as the first layer. The paper briefly credits the padding mechanism for helping with iterative generation, but it does not clearly justify why this particular parameter sharing/pseudoinverse architecture is necessary, what role it plays (e.g., projecting to token space and back), or how sensitive JULI is to these design choices. In other words, why should LM-head and its pseudoinverse improve jailbreak performance vs. a small MLP over top-k logits? In the API setting, why does a random projection and pseudoinverse (rather than any other mapping) meaningfully help? 

From my perspective, the LM-head/pseudoinverse sharing is plausible and elegant, but the paper should demonstrate necessity and robustness. Without targeted ablations, it’s unclear whether this is a principled requirement or an implementation convenience. 

3. Figure 2 is obscure; the experimental setup behind the key observation that "Although an aligned LLM often refuses to answer harmful queries, it remains knowledgeable about the answers" should be explained.

 For example, how do you get the frequency of ground truth tokens in harmful responses? Is it based on one harmful prompt? Which dataset do you use? If you only use one prompt and generate only one harmful response, is this observation general for all prompts? I can largely understand the meaning of Figure 1, but it confuses me when I read this part at the beginning.    

Minor Concerns:
1. I notice most work about jailbreak-attack works using ASR to evaluate their method, but why does the author use the harmful score as the metric? 
2. [line 414] I don't think having access to an unaligned base model is often impractical. We can easily download many base or unaligned models from HuggingFace. It is not the key concern. But for me, I would consider relying on the unaligned base model to be a strong assumption.

### Questions
See above.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes JULI, a jailbreak method that adjusts token probabilities using a small plug-in network, BiasNet, which takes the target model’s log probabilities and outputs a token-wise bias added back to the logits during generation. The method is designed to work in open-source and API-only settings where top-k token log probabilities are exposed. Across several models and datasets, JULI reports higher harmfulness scores than prior baselines, including in an API setting against Gemini-2.5-Pro.

### Strengths
- Clear idea: Manipulating the next-token distribution using a lightweight network is simple in API settings.
- Efficiency: BiasNet trains with 100 harmful QA pairs and fewer than 1% of the target LLM parameters with extremely low inference time.
- Useful visualization and intuition. Figure 3 shows that BiasNet sparsely shifts distributions, with larger KL changes near critical positions such as response starts, and minimal perturbations later. 
- Interesting risk observation: Figure 2 suggests a high top-k hit rate of "ground truth" harmful tokens in the LLM’s top-5 predictions, indicating leakage of unsafe knowledge even when surface outputs are aligned.

### Weaknesses
- Training objective is under-specified and potentially inconsistent with the inference-time mechanism.
    - In Section 4.3, the training loss is written as $\mathbf{min}\_{\theta} E\_{(x,y)\sim L} [CE(F_{\theta} (x), y)]$. Earlier, $F_{\theta}(x)$ is defined to output a "logit bias" B that is added to the base log probabilities.
   - The paper does not define any regularization on B, no norm or temperature constraint. Unconstrained biases can dominate $\mathbb{log} p_{\alpha}$ and degrade fluency. Yet the analysis in Figure 3 claims minimal perturbation.
- Ambiguity around how training data and log probabilities are obtained in the API-only setting.
    - Section 4.3 mentions you "extract and store the log probabilities at all token positions in the response part of the training data points before the training phase." In the API setting, only top-k log probabilities for the next token are available. Algorithm 2 pads missing tokens to the k-th logprob minus 10 at inference; however, it is unclear how you obtain training pairs (x,y) across positions from an API that restricts you to a single next-token distribution per call and k ≤ 5 for Gemini.
    - Do you train BiasNet using the exact target API's top-k only, or train on a surrogate model with full-vocabulary log probabilities and then transfer? 
- Missing or weak baselines in the API setting.
    - For LINT you assert the need for top-500 tokens, but it should be possible to evaluate a modified LINT with top-5 resampling to understand how much of JULI’s gains come from BiasNet vs. resampling alone.

### Questions
see weaknesses above

### Soundness
3

### Presentation
3

### Contribution
2
