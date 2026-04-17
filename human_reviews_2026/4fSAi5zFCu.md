# Fixing What Fine-Tuning Breaks: A Simple and Efficient Method to Improve Safety Post Domain Adaptation

- Decision: Reject
- Scores: 2, 2, 6, 6

## Abstract
Safety-aligned language models suffer from a reduction in safety post-finetuning even on benign data. Prior works have highlighted a solution to the issue via further preference optimization in the fine-tuned models; however, this method is computationally expensive and requires domain-specific preference optimization data. In this paper, we aim to alleviate the degradation in the general safety of the fine-tuned language models via a weight steering methodology, which is both computationally inexpensive and efficient, and does not require in-domain preference optimization data. We further demonstrate that our methodology has statistically insignificant changes to the model’s general coherence and false rejection rates and retains the model’s domain-specific knowledge. Finally, we discovered that our method also increases the domain-specific safety of the language model without requiring domain-specific safety data

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper addresses safety degradation in LLMs after domain-specific fine-tuning. It proposes SPECTRA, a gradient-free method that restores safety by identifying a refusal direction from activation differences between harmful and benign prompts, then injecting it into a safety-critical subspace via activation-aware SVD. Tests on Llama2-7B and Gemma2-9B across multiple domains show improved safety and jailbreak resistance with minimal impact on performance.

### Strengths
1. The paper considers the critical and timely problem of efficient safety alignment for LLMs.
2. The empirical results demonstrate the method's positive impact when compared against the base model.
3. The exploration of safety in specialized domains like finance and law is a valuable extension.

### Weaknesses
1. The experimental evaluation is limited, as it only compares the proposed method against the base model. To properly assess its contribution, a comparison with established safety alignment baselines is necessary.
2. The contribution appears to be an incremental extension of existing techniques (e.g., Wei et al. (2024), Chhabra & Khalili (2025)). The paper should more clearly show the distinctions from prior work and provide deeper insights.
3. The academic writing needs revision for clarity and precision. Several key claims are not sufficiently supported by the empirical results.

### Questions
You claim that the impact on general coherence is "statistically insignificant." To substantiate this claim, could you please provide the specific statistical test results (e.g., p-values) from your benchmark evaluations that support this conclusion?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper examines how an LLM’s safety alignment often degrades after fine-tuning, even when the fine-tuning data are harmless. It introduces SPECTRA, which computes a safety-critical low-rank direction and injects it into the model’s weights, steering the model toward safer outputs while largely preserving capabilities on domain-specific tasks.

### Strengths
- This paper addresses a real and urgent problem in LLMs: fine-tuning can break safety, which is a major deployment concern.
- It offers a practical solution: no retraining and no domain-specific safety data are required.

### Weaknesses
1. This paper requires thorough proofreading, as the current writing makes the presentation of this work incomplete.
- Abstract does not end with a period.
- difference-in-means belrose2023diffinmeans …
- both the the Attention and MLP
- safty → safety
- neccessary → necessary
- Harmbech → Harmbench
- coherance → coherence
- The paper includes several tables (1-4) that present information better suited for inline text or an appendix.


2. You must rephrase all text in your own words, whether it presents fundamental facts or cites a source.
- Section 2 should be rewritten.
- Portions of your text appear to be copied directly from [1]. It also appears that [1] copied much of its background from [2], which must not happen.


3. The most critical problem with this paper is that its contribution appears insignificant.
This paper appears to combine existing ideas from [2] and [3]. If that is not the case, please correct me if I’m mistaken.
- The experiments present results for SPECTRA, but without baseline comparisons, it is unclear whether the proposed idea is actually effective.
    - I believe it is important to include a comparison with [2].
- “SPECTRA is the only method to improve the domain-specific safety of language models without any backpropagation.”
    - This claim seems overstated unless you demonstrate that other general, non-backpropagation methods (e.g., [2], …) are less effective in the same domain-specific setting.


---


[1] Chhabra, Vishnu Kabir, and Mohammad Mahdi Khalili. "Towards Understanding and Improving Refusal in Compressed Models via Mechanistic Interpretability." *arXiv preprint arXiv:2504.04215* (2025).

[2] Arditi, Andy, et al. "Refusal in language models is mediated by a single direction." Advances in Neural Information Processing Systems 37 (2024): 136037-136083.

[3] Wei, Boyi, et al. "Assessing the brittleness of safety alignment via pruning and low-rank modifications." arXiv preprint arXiv:2402.05162 (2024).

### Questions
- [Table 6] Did you perform any additional analysis explaining why Variant A performs best?
- [Table 6] Does “Base” refer to the fine-tuned model or the original LLM?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper discusses the issue that safety-aligned language models often lose their safety properties after being fine-tuned on domain-specific data, even when the data itself is harmless. The authors introduce SPECTRA, a lightweight method that restores safety after fine-tuning by injecting a refusal steering vector into a low-rank portion of the model’s weights. Unlike methods such as preference optimization or RLHF, SPECTRA does not require additional training, domain-specific safety data, or large computational resources. The experiments, conducted on medical, legal, and financial versions of LLaMA and Gemma, show that the method substantially lowers attack success rates in both standard and jailbreak evaluations, maintains task performance in general and domain settings, and shows almost no increase in false refusals.

### Strengths
* The paper proposes a method that improves safety without any additional training by modifying the model weights directly.
* It demonstrates the effectiveness of the method through experiments across multiple domains.
* It also reports results on general language capability, showing the negative results as well, rather than only highlighting the improvements.

### Weaknesses
* Limitations of refusal-vector–based safety
- Since the method strengthens the “refusal direction,” it mainly makes the model better at rejecting harmful queries, but it does not actually mean the model understands or reasons about harmful content. It would be helpful to include experiments that evaluate this limitation more directly.
* Insufficient evaluation of other capabilities
- The paper reports coherence and general task performance, but it would be more convincing if it also evaluated qualitative abilities such as reasoning, multi-turn dialogue quality, and instruction-following faithfulness.
* Hyperparameter choices are mostly empirical
- The selection of alpha and the rank values seems empirical. Since the cost of SVD increases as the steering scale becomes larger, it would be useful to include an analysis of the runtime or compute impact when applying this method to larger models.

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
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces SPECTRA (Scalable Projection-based Elicitation of Coherence and Trustworthy Refusal Alignment), a novel, computationally inexpensive, and efficient weight shifting methodology to counteract the degradation of safety in safety-aligned large language models (LLMs) after domain-specific fine-tuning. The degradation often occurs even when fine-tuning is performed on benign data. Unlike prior realignment methods which require computationally expensive preference optimization or domain-specific alignment data , SPECTRA works by directly injecting a refusal steering vector (calculated via difference-in-means on the base model) into a low-rank, safety-related subspace of the fine-tuned model's weight matrices.
The authors demonstrate that SPECTRA significantly improves general and in-domain safety (measured by Attack Success Rate, ASR) and robustness against jailbreaks (GCG, GPTFuzz, TAP) across Llama2-7B-Chat and Gemma2-9B-IT models fine-tuned for Medical, Law, and Finance domains. Crucially, the method is shown to incur statistically insignificant changes to the model's general and domain-specific coherence and maintains low false refusal rates, addressing common drawbacks of activation-based steering.

### Strengths
* Significance & Efficiency: SPECTRA addresses a critical, real-world issue in deploying domain-adapted LLMs (safety degradation) with an exceptionally simple, efficient, and data-light method, requiring no new domain-specific safety data or further gradient calculations. This is a major advantage over gradient-based methods.
* Quality & Robustness: The method significantly improves safety across the board for multiple LLM families (Llama2, Gemma2) and domains (Medical, Law, Finance), providing enhanced robustness against a diverse set of advanced jailbreaking attacks (GCG, GPTFuzz, TAP).
* Clarity & Non-Invasiveness: The theoretical link to activation steering (Theorem 1) provides a clear interpretability anchor. Furthermore, the empirical evidence showing statistically insignificant changes in both general and domain-specific coherence and low false refusal rates successfully mitigates major concerns associated with prior directional steering techniques.
* Originality of Application: The successful application of a base model's refusal vector (r_1) to a fine-tuned model's optimal layer (W_{l2}^f) is a novel finding in the context of weight steering for post-finetuning safety, and Variant A's effectiveness is a key result.

### Weaknesses
1. Mechanism Detail and Intuition: The explanation for why Variant A (r_1  injected into W_{l2}^f) is superior to other variants, especially Variant C (r_2 injected into W_{l2}^f ), is currently based solely on empirical results (Table 6). A deeper mechanistic interpretability analysis is needed. The authors should hypothesize why the base model's refusal vector (r_1) retains or even gains efficacy over the fine-tuned model's vector (r_2) in the new weight space, especially in the layer l_2
2. Hyperparameter Sensitivity and Selection: The paper mentions that α values and low-rank p values are model-dependent and found via "mild hyperparameter tuning" (Tables A.1, A.2). This opaque process is a critical weakness. The authors should include a sensitivity analysis (e.g., a simple sweep plot of ASR vs. α and ASR vs. p) to demonstrate that the optimal settings are not highly brittle or require extensive tuning, which would negate the "simple and efficient" claim.
3. Missing Control/Comparison to Full W′ : Equation 5 (W' =W+αrr ^TW) is a special case of SPECTRA when the projection Π is the identity matrix. The main premise of using  
$\tilde{W} = \Pi W$ is to mitigate potential downsides on coherence by only steering in the safety-critical subspace Π. The current results (Table 6) use the full W in the steering (Eq. 5) to find the best practice (Variant A) and then implicitly use SPECTRA for all subsequent tables (Tables 7, 8, 9). The paper must explicitly compare the performance (ASR vs. Coherence trade-off) of the full SPECTRA (Eq. 3) versus the simpler steering (Eq. 5) to empirically justify the necessity of the low-rank projection  \tilde{W} 

4. Refusal Layer/Token Position Explanation: The layers selected for steering (l_2) are detailed in the appendix (Table 12). The authors should discuss why fine-tuning causes a shift in the refusal layer (e.g., Llama2 Medical: 13/14) and what the mechanistic implication of this shift is.

### Questions
1. Justification for  \tilde{W} (The Low-Rank Projection): Table 6, which determines the best practice (Variant A), uses the simpler steering: 
$W' = W + \alpha r r^T W$. The core of SPECTRA is $W' = W + \alpha r r^T \tilde{W} $(Equation 3). Can the authors provide a direct comparative evaluation between SPECTRA (Eq. 3) and the simpler steering (Eq. 5), specifically showing the coherence retention benefit that \tilde{W}  is supposed to provide? Without this direct comparison, the necessity of the low-rank projection, which adds complexity (ActSVD computation, rank selection), is not sufficiently justified.

2. Mechanistic Insight into Variant A's Success: Why is Variant A (r_1 into W_{l2}^f , i.e., base model refusal vector in fine-tuned model's optimal layer) consistently the most effective? Specifically, why is it superior to Variant C (r_2 into W_{l2}^f), which uses the refusal direction calculated from the fine-tuned model itself? Does this suggest that the fine-tuning process damages the interpretability/saliency of the refusal direction more than it damages the original weights, making the base model's vector a purer signal? Please provide a detailed hypothesis.

3. Hyperparameter Sensitivity and Generalization: Please provide the sensitivity analysis plots (ASR and coherence metric changes vs. α) for at least two different models/domains (e.g., Llama2 Medical and Gemma2 Finance) to demonstrate that the optimal α and rank p values are stable. How would a practitioner determine optimal α and p without running extensive red teaming evaluations?

4. In-Domain Safety Benchmarks: For the Law and Finance domains, the authors had to create custom safety datasets (FinSafeEval, LawSafeEval). Could the authors briefly describe the nature of the harmful prompts in these datasets? For instance, do the Law prompts relate to illegal advice, and the Finance prompts to fraud/manipulation? This context is crucial for understanding the "In-Domain" safety results.

### Soundness
3

### Presentation
3

### Contribution
3
