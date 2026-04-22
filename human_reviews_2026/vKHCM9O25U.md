# INTENTION MATCHING STOPS JAILBREAKS

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 2, 6, 4

## Abstract
Large Language Models (LLMs) are vulnerable to jailbreak attacks even with safety
alignments. Existing defenses typically lack precise localization of harmful intent,
leading to ineffective defense when faced with complex jailbreak prompts. For
precise localization, we exploit ‘semantic-consistency’ between an input-output
pair: regardless of the jailbreak input complexity, the outputs always respond
according to the actual input intents. In this paper, we present SENTINEL, a
plug and play module that can be fit into the auto-regressive generation process
for any model, systematically exploits ‘semantic-consistency’ to extract intent
for jailbreaks. Specifically, during generation process, we solve an optimization
problem to extract semantically aligned sub-sequences for an input-output pair, then
we efficiently quantify the harmfulness by using the refusal direction projection
value, and determine should we halt the generation process or not as the defense.
Experiments demonstrate that SENTINEL significantly reduces attack success
rates mostly below 5% for on various jailbreaks across all evaluated LLMs, also
we explained the fundamental mechanism as re-distributing jailbreak features from
alignment blind-spot to aligned regions.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes SENTINEL, an online jailbreak detection and defense mechanism that leverages the semantic consistency between input–output pairs to detect jailbreak intent. The paper points out that existing token-level perturbation-based defenses scale poorly with long and complex jailbreak prompts, while context-level defenses can be bypassed through overwriting attacks. To address these limitations, the proposed method exploits semantic consistency to identify the most important harm-related tokens among all input tokens and performs jailbreak detection based on these selected tokens. Experiments demonstrate that the proposed method achieves incremental improvements across multiple models and datasets while maintaining acceptable computational overhead.

### Strengths
1. The problem is interesting and important. Jailbreak attacks and defenses have been extensively explored over the past two years. Existing jailbreak defense methods can be categorized into input-based, output-based, and hidden-state-based approaches. This paper introduces a new perspective by considering the semantic consistency of input–output pairs. The idea is simple, intuitive, and likely correct, as evaluating input criticality based on output positioning has already been applied in other contexts, such as gradient heatmaps.  
2. The evaluation is comprehensive, including thorough experiments on the effectiveness of the proposed methodology, the rationale behind each component, and interpretability analyses.  
3. The paper has a clear structure and smooth flow of writing. Its problem-driven reasoning approach makes it easy to follow, while detailed methodological descriptions facilitate reproducibility.

### Weaknesses
1. The literature review is not comprehensive. The paper claims that input-perturbation-based defenses and context-level defenses face different challenges. However, other types of defenses—such as those based on output filtering or hidden-layer guidance—are not discussed. While it is impractical to cover every single study in the rapidly growing field of LLM jailbreak research, acknowledging each category of defenses would provide a more complete overview.  
2. Experimental results show that the improvements are incremental and not significant in many scenarios. As shown in Table 1, in most cases, the proposed method offers only marginal improvements over the second-best approach, which already demonstrates strong defensive capabilities. Under certain attack settings (e.g., RADICAL), the proposed method even exhibits greater vulnerability.

### Questions
A question regarding the core approach: What is the difference between evaluating input tokens based on semantic coherence and using gradient-based methods to determine which tokens most significantly affect the output? How do their respective effects compare? Intuitively, could a Grad-CAM-like approach be used to more efficiently identify key input intents?

### Soundness
3

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
This paper proposes SENTINEL, a plug-and-play jailbreak defense framework that leverages semantic consistency between input and output to extract intent-related subsequences and quantify their harmfulness via refusal direction projection. The method operates during autoregressive generation without model fine-tuning and demonstrates strong empirical performance across multiple LLMs and attack types.

### Strengths
(1) Strong empirical results: consistently reduces ASR to <5% across diverse LLMs and jailbreak methods.

(2) Low over-refusal rate on boundary cases.

(3) No model modification required—enables real-world deployment.

(4) Interpretable intent extraction via context matching.

### Weaknesses
(1) the defense is reactive (requires partial output generation), not input-only.

(2) Computational overhead from sliding windows and optimization is not quantified.

(3) Robustness against adaptive attackers who manipulate semantic consistency is not thoroughly evaluated.

(4)Refusal direction is layer- and position-specific; its optimality and transferability need deeper analysis.

### Questions
1.Can an adaptive attacker craft inputs that induce semantically inconsistent yet harmful outputs to evade matching?

2.How does SENTINEL scale to very long prompts (e.g., >2K tokens) in terms of latency?

3.Is the refusal direction stable across different model architectures (e.g., MoE models)?

4.This method requires the LLM model to output a certain length of token before it can be used, and it also incurs other computational costs. Will this limit its practicality?

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
The paper proposes a defense against jailbreaking attacks based on the assumption that the outputs are always semantically aligned with the inputs regardless of the prompt complexity. The paper proposes Sentinel to defend against jailbreak prompts while an LLM is generating content autoregressively without the need for complete generation. The paper evaluates Sentinel on three datasets and compares it against several baselines by applying popular jailbreaking attacks. Results show that Sentinel achieves the lower attack success rate on average while also not having a high false positive rate.

### Strengths
+ The assumption that inputs and outputs must bear semantic consistency is intuitive and interesting.

+ The paper seems to have good mathematical foundation to back its claims.

+ The output of the approach is explainable since it gives importance scores to each token.

+ Sentinel does not need to evaluate the complete output and can instead work on chunks as they are generated.

+ Sentinel mostly outperforms other methods on several attacks.

+ The paper presents results on adaptive attacks and the evaluation shows that adaptive Sentinel outperforms the other methods.

### Weaknesses
- Sentinel is vulnerable to multi intent or intent mixing attacks such as Radical but the paper mentions this.

- Several hyperparameters must be tuned correctly. For example, it is unclear how to choose the window size or thresholds.

- A case study showing a failure of Sentinel might include insight on why it fails.

- The adaptive attack is simple and does not consider adding irrelevant tokens or optimizing based on feedback.

### Questions
- How does your approach perform if the underlying model is not aligned?

- Can you provide a case study or failure examples to show why Sentinel might fail?

- How are the hyperparameters such as window size and thresholds chosen?

- How sensitive is Sentinel to the data used to compute the refusal direction?

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
3

### Summary
This paper propose SENTINEL, a jailbreak defense framwork based on semantic-consistency between the input and the output to detect the harmful intention, enabling more fine-grained test time denfense for LLM. SENTINEL identifies potential harmful intent by aligning semantic representations between the input and output contexts through an optimization that learns soft selection over context windows. The experimental results show that SENTINEL effectively reduces jailbreak success rates with minimal over-refusal and latency, offering a training-free and robust approch to LLM safety.

### Strengths
* A novel defense perspective based on intention dection: By modeling the semantic correspondence between input and output contexts, SENTINEL can uncover concealed harmful objectives even when the attack is obfuscated, enabling more robust and interpretable jailbreak defense.
* SENTINEL can be integrated into existing LLM without additional fine-tuning, making it a practical defense solution for real-world deployment.
* SENTINEL employs a probabilistic contextmatching mechanism that softly selects semantically aligned input–output segments to
reveal the real intention. The design enhances robustness against disguised jailbreak prompts and provides clear interpretability.

### Weaknesses
* Limitation in evaluating intent-encoding jailbreaks: SENTINEL’s malicious intention extraction is based on contiguous token windows, which may limit its robustness against intent-encoding jailbreaks like DRA[1], where harmful instructions are encoded into seperate representation. The paper does not include experiments evaluating its effectiveness under
such attack types.
* Limitation in general capability evaluation: The paper does not include experiments assessing the impact of SENTINEL on the model’s general capabilities, such as helpfulness or nonharmful task performance.
* The writing quality of this paper should be improved. I found multiple parts of this paper is hard to read (e.g., methods).
* No comparison with papers that detecting jailbreaks through input-and-output analysis.



[1]. Liu, Tong, et al. "Making them ask and answer: Jailbreaking large language models in few queries via
disguise and reconstruction." 33rd USENIX Security Symposium (USENIX Security 24). 2024.

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
2
