# A Realistic Threat Model for Large Language Model Jailbreaks

- Decision: Reject
- Scores: 8, 3, 5, 5

## Abstract
A plethora of jailbreaking attacks have been proposed to obtain harmful responses from safety-tuned LLMs. These methods largely succeed in coercing the target output in their original settings, but their attacks vary substantially in fluency and computational effort. In this work, we propose a unified threat model for the principled comparison of these methods. Our threat model combines constraints in perplexity, measuring how far a jailbreak deviates from natural text and computational budget in total FLOPs.
For the former, we build an N-gram language model on 1T tokens, which, unlike model-based perplexity, allows for an LLM-agnostic and inherently interpretable evaluation. We adapt popular attacks to this new, realistic threat model, with which we, for the first time, benchmark these attacks on equal footing. After a rigorous comparison, we find attack success rates against safety-tuned modern models to be lower than previously presented and that attacks based on discrete optimization significantly outperform recent LLM-based attacks. Being inherently interpretable, our threat model allows for a comprehensive analysis and comparison of jailbreak attacks. We find that effective attacks exploit and abuse infrequent N-grams, either selecting N-grams absent from real-world text or rare ones, e.g., specific to code datasets.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces a unified threat model for evaluating jailbreak attacks on safety-tuned LLMs. Recognizing the multitude of existing jailbreak methods that vary in success rates, fluency, and computational effort, the authors propose a framework that combines constraints on perplexity—measuring deviation from natural text—and computational budget quantified by FLOPs. To achieve an LLM-agnostic and interpretable evaluation, they construct an N-gram model based on one trillion tokens. Adapting popular attacks within this new threat model, the paper benchmarks these methods against modern safety-tuned models on equal footing. The findings indicate that attack success rates are lower than previously reported, with discrete optimization-based attacks outperforming recent LLM-based methods. Furthermore, effective attacks tend to exploit infrequent N-grams, selecting sequences that are either absent from real-world text or rare, such as those specific to code datasets.

### Strengths
* Unified Threat Model: The paper addresses the critical need for a standardized framework to compare various jailbreak attacks, providing clarity in a field crowded with disparate methods.

* Interpretability: By employing an N-gram model for perplexity measurement, the threat model remains interpretable and LLM-agnostic, facilitating a deeper understanding of why certain attacks succeed or fail.

* Comprehensive Benchmarking: Adapting and evaluating popular attacks within the proposed threat model allows for a fair and rigorous comparison, advancing the discourse on LLM vulnerabilities.

### Weaknesses
* Comparison with Existing Methods: The paper would benefit from a direct comparison with existing perplexity detectors, such as the one proposed by Alon et al. (arXiv:2308.14132). This would contextualize the proposed model within the current state-of-the-art and highlight its relative advantages.

* Perplexity Measurement Limitations: While the N-gram model offers interpretability, it may not capture the nuances of natural language as effectively as model-based perplexity measures, potentially affecting the evaluation's accuracy.

### Questions
* Clarify Equation 1: Provide a complete and formal definition of the Judge function. This will enhance the paper's clarity and reproducibility.

* Enhance Comparative Analysis: Include a comparison with existing perplexity detectors, specifically the method proposed in arXiv:2308.14132.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper introduces a new "threat model" for LLM jailbreaks (I am not convinced that this is the right framing here) using an N-gram perplexity approach. There are some interesting ideas, but I think the framing needs adjustment before being ready for publication.

### Strengths
- I thought the idea of using the N-gram was interesting.
- The paper is fairly clear written, and the plots are clearly presented.
- I thought the analysis showing an N-gram perplexity constraint increased compute time for GCG interesting, and that it reduces ASR. The analysis comparing ASR against flops was generally very interesting.

### Weaknesses
- I'm not sure if the contribution is the threat model or the N-gram language model perplexity filter?
    - As far as I understand, the "threat model" is basically assuming the N-gram approach is the right way of doing things, but I am not sure that is clearly established here? If the point of the paper is to establish the threat model, there should be lots of evidence it is an appropriate defence.
    - I don't find this evidence in the paper. It is simply assumed that this is an appropriate defence?
- I am not convinced the threat model is the best one. I think the best threat model is trying to break what Frontier AI labs have released. I think claiming the threat model here is realistic is significantly overclaiming.
- I think the benefit and results section would benefit from making clear the implications of the results much more.

### Questions
- I am generally confused about the "threat model" framing, e.g., "universal threat model for comparing attacks across different models", how does the threat model actually allow you to compare?
- The claim in Table 1 is "The Lack of an LLM-Agnostic Threat Model Renders Attacks Incomparable." But I think the attacks are comparable, just on different axises? I could not follow precisely what the claim is here? I also think the most important thing is ASR? The paper makes the claim many times that the attacks are incomparable, but I just cannot follow this.
    - In terms of needed progress, the threat model is basically what frontier AI labs release? I'm not sure a new threat model is not what is needed.
- How does the N-gram model do on longer user queries? Presumably the perplexity increases substantially with longer queries. Does this mean that this defence would not work well with long-context models? Some of the latest models from Frontier labs can have very long context lengths. This makes me think the threat model might not actually be appropriate.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
Jailbreaking attacks on large language models (LLMs) are widely studied in the literature. But those attack prompts are usually non-natural text. This paper proposes an N-gram method to measure the fluency of generated attack prompts. It shows that this simple approach can filter out several existing jailbreaking attacks and significantly reduce their attack success rates. The paper then proposes an adaptive attack which considers the N-gram of the attack prompt during generation. The results show it can boost the attack performance of existing jailbreaking methods.

### Strengths
1. A systematic study on jailbreaking attacks is an important direction and can lay the foundation for future research. This paper provides a good study in this space, which helps the community to better develop new techniques in attacking and protecting LLMs.
2. The proposed N-gram model is effective in filtering several attacks that generate jiberesh text. Those jailbreaking prompts are quite different from natural sentences, causing them easily detectable and hence not robust.
3. The evaluation is comprehensive, including multiple recent LLMs and safely aligned models. The baseline attacks are chosen from the state-of-the-arts.

### Weaknesses
1. It is known that several existing jailbreaking attacks generate non-natural text. There have been many proposed methods for filtering such jailbreaking prompts [1][2]. This insight mentioned in the paper is not new. The proposed approach of using the N-gram is straightforward. The novelty hence seems limited.
2. According to Table 2, using the perplexity measure of llama2-7b can distinguish the two optimized suffixes. Why is this approach not used to evaluate jailbreaking attacks in Tables 3 and 4? Additionally, there are many other filtering methods such as [1][2]. It is important to compare the performance of the proposed approach with these techniques.
3. The paper introduces an adaptive attack that considers the N-gram measure during adversarial prompt generation. It is strange that the paper introduces an attack against the proposed measure and then uses the same measure to evaluate the performance. It is similar to self-verifying correctness. It is suggested to use other filtering methods such as [1][2] and the llama perplexity to evaluate the final attack performance.
4. The case shown in Table 2 for the adaptive attack does not seem like natural text as well. Why does the N-gram model not filter this attack prompt? For example, the phrase “A questions their She” very unlikely exists in normal text. With the 8-gram model used in the paper, it should be able to filter out this. Could the authors explain why this case bypass the detection?


[1] Alon, Gabriel, and Michael Kamfonas. "Detecting language model attacks with perplexity." arXiv preprint arXiv:2308.14132 (2023).
[2] Inan, Hakan, et al. "Llama guard: Llm-based input-output safeguard for human-ai conversations." arXiv preprint arXiv:2312.06674 (2023).

### Questions
Please see the weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes a new threat model to compare different attacks. This threat model includes a perplexity filter based on an N-gram language model and constraints on FLOPs. A fine-tuned LLM judge measures the ASR. Many existing attacks fail in this threat model. With the consideration of the proposed perplexity filter, the adapted attacks can restore many ASRs.

### Strengths
1. A unified model to evaluate different attacks.
2. N-gram LM has certain advantages.

### Weaknesses
1. Only considered white-box attacks. In the real world, black-box attacks are more practical. As shown by Figure 13, white-box attacks in this new threat model have lower transferability. That is, this threat model cannot measure black-box attacks very well.

2. The perplexity filter is not new.

3. There are other defenses such as [instruction filter](https://arxiv.org/abs/2312.06674), and [random perturbation](https://arxiv.org/abs/2310.03684), etc. Why doesn't the threat model consider them?

4. Evidence is needed to show that N-gram LM is better than LLM. Some experiments are necessary.

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
