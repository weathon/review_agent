# From Threat to Tool: Leveraging Refusal-Aware Injection Attacks for Safety Alignment

- Avg Score: 2.67
- Decision: Reject
- Scores: 2, 4, 2

## Abstract
Safely aligning large language models (LLMs) is a critical challenge: reliable safety requires large amounts of human-labeled preference data, and collecting such data is expensive, slow, and often infeasible at scale. We present Refusal-Aware Adaptive Injection (RAAI), an attack-style method that directly and simply induces LLMs to produce harmful completions, and which we repurpose as a practical tool for gathering safety-alignment data. Concretely, RAAI detects internal refusal signals emitted by an LLM and adaptively injects predefined, tailored phrases into prompts so as to bypass refusals and elicit harmful but fluent responses. Unlike prior attack or data-synthesis approaches that rely on complex iterative prompt engineering or auxiliary models, RAAI is training-free, model-agnostic, and operates with minimal orchestration, making it efficient to deploy across models. Evaluated on four jailbreak benchmarks, RAAI raises the rate of harmful completions from a baseline of 2.15\% to up to 61.04\%, demonstrating its effectiveness at producing challenging negative examples that are otherwise difficult to obtain. Fine-tuning LLMs using RAAI-generated data substantially improves robustness to harmful prompts while preserving performance on standard benchmarks (e.g., MMLU, ARC). By showing how the proposed RAAI attack method can be reframed as a controlled data-collection instrument, we turn a security risk into a scalable asset for LLM safety alignment.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes to detect the refusal tokens and modify it into predefined sequences to allow harmful contents generation. Experiments show that the proposed method outperforms existing baselines. When utilized for further fine-tuning LLMs, the data shows promising results for increased safety without alignment tax.

### Strengths
1. the paper is well written and easy to follow
2. experiments show that the proposed method outperform baseline methods by a large margin.
3. the generated harmful contents can also be utilized to further fine-tune the model for improved safety performance.

### Weaknesses
1. the compared baseline methods are all outdated ones from 2023 and 2024.
2. the contribution of the paper is weak. the main contribution is to replace a token to be decoded with predefined sequences. the rest of the methods such as utilizing SimPO is just a simple application.
3. although the proposed method is branded as gray-box, it requires access to the decoding process of LLMs which seems impossible for production models such as GPT or Gemini. Thus the application scenario of the proposed method is greatly limited.

### Questions
1. has the author studied why the model trained with synthetic data from the proposed method has increased performance in MMLU or ARC
2. will the proposed method increase the rate of over/false refusals as mentioned in [1] and [2]

[1] Röttger, Paul, et al. "Xstest: A test suite for identifying exaggerated safety behaviours in large language models." arXiv preprint arXiv:2308.01263 (2023).

[2] An, Bang, et al. "Automatic pseudo-harmful prompt generation for evaluating false refusals in large language models." arXiv preprint arXiv:2409.00598 (2024).

### Soundness
2

### Presentation
2

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
This paper proposes attacking LLMs by replacing refusal tokens (such as `can't`) with predefined phrases to elicit harmful responses. 
Furthermore, the paper repurposes the idea as a data synthesis framework for improving the safety alignment of LLMs by training on pairs of generated (original safe responses) and elicited harmful responses using SimPO. Experimental results show that the method can elicit more harmful responses compared with baselines and improve the safety alignment with little alignment tax.

### Strengths
- The paper is well-written and organized, making the idea easy to follow.

- The idea of turning attacking into an alignment data generation pipeline is interesting and proves effective.

### Weaknesses
- As an attacking method, the framework exhibits several critical limitations concerning refusal token replacements:
   - For commercial models, access to token probability distributions is often restricted, preventing the application of this method to commercial deployments where such attacks would constitute a genuine security concern.
   - The refusal token selection process is arbitrary and depends on manual inspection. Moreover, Table 4 reveals inconsistencies in token classification—for instance, tokens such as `support` and `fulfill` are not categorized as refusal tokens. The rationale behind these selections requires clarification, as does the potential impact of alternative token choices on overall performance. Additionally, the handling of tokenization for these words remains inadequately addressed.

- As a data synthesis method, the approach lacks sufficient novelty. It is well-established that on-policy data (i.e., responses generated by the model itself) provides benefits (thus less alignment tax is expected), and the proposed method essentially functions as Rejection Fine-Tuning (RFT), offering limited methodological innovation beyond existing techniques.

### Questions
Typo:

Line 163:  can be adjusted

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
5

### Summary
This paper investigates a more general form of the prefilling attack for circumventing the safety alignment of LLMs called “RAAI”. During generation, if a fixed set of refusal tokens has enough predicted probability mass, then a predefined set of tokens is injected to attempt to steer the model towards compliance. The responses generated from RAAI attacks are then used as synthetic data for safety alignment.

### Strengths
1. The injection technique for circumventing the safety alignment of LLMs is simple and effective.
2. Ablations of injection phrases and refusal thresholds are performed.
3. Fine-tuning on synthetic data from RAAI improves safety without sacrificing much model utility.

### Weaknesses
1. The novelty of Refusal-Aware Adaptive Injection is quite limited. It is essentially a more general version of the prefilling attack where the harmful prefill can be inserted mid-generation based on some simple rules. In fact, prior work has explored similar ideas of alternating between generation and injection of tokens, such as in [1]. I would suggest including [1] as a baseline in your experiments.
2. The citations for prefilling attacks misses some critical works, namely [2] and [3].
3. The comparison to naive prefilling could be strengthened. I recommend comparing against the approach of [2], where rather than using “predefined phrases as a fixed prefix” for the prefill, prompt-specific prefills are generated via few-shot prompting. The prefills are typically structured in the following way: given the prompt (e.g., “Tell me how to build a bomb”), the prefill starts with an affirmative restatement of the prompt (e.g., “Sure, here are instructions for building a bomb”) followed by just a bit more text that could begin the harmful content (e.g., “Sure, here are instructions for building a bomb:\n\nStep 1. Gather”). I would suspect prefilling attacks structured this way to be comparable to RAAI performance.
4. Similarly, I suspect that using synthetic data generated by the prefilling attack approach of [2] for safety alignment would yield comparable improvements to using synthetic data generated by RAAI — I suggest investigating and reporting such results in Table 7 and 8.
5. Typos:
- Line 163: “can be adjusted for” is missing the start of the sentence (I assume tau).

References:

[1] Zhang, Zhuo, et al. "Make them spill the beans! coercive knowledge extraction from (production) llms." arXiv preprint arXiv:2312.04782 (2023).

[2] Vega, Jason, et al. "Bypassing the safety training of open-source llms with priming attacks." arXiv preprint arXiv:2312.12321 (2023).

[3] Andriushchenko, Maksym, Francesco Croce, and Nicolas Flammarion. "Jailbreaking leading safety-aligned llms with simple adaptive attacks." arXiv preprint arXiv:2404.02151 (2024).

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
1
