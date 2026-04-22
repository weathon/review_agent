# Not Errors but Guardians: Understanding Sink Tokens in Multimodal LLMs

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 2, 4, 6

## Abstract
Multimodal large language models (MLLMs) achieve remarkable success in vision–language tasks but remain prone to hallucination, often attributed to abnormal attention behaviors. 
A recurring phenomenon is the emergence of attention sinks—tokens that absorb large amounts of attention despite limited semantic content. 
While previously regarded as artifacts that exacerbate hallucination, we show that in MLLMs certain tokens within system prompts act as stable, system-level attention sinks. 
Through causal interventions including masking and content substitution, we find these tokens serve critical functions: anchoring attention to ensure computational stability, influencing outputs, and implicitly tracking the model’s state. 
Building on this, we propose the Attention-Budget Hypothesis, which reframes modality bias as a trade-off in attention allocation. 
Guided by this perspective, we design SPEAR (Sink-PrEserving Attention Reallocation), an intervention that boosts visual attention while preserving sink functions, achieving effective hallucination mitigation without degrading reasoning. 
Our work provides the first systematic characterization of system-level attention sinks in MLLMs and highlights their functional role in both model stability and multimodal reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper delves into the phenomenon of attention sinks in Multimodal Large Language Models (MLLMs), specifically focusing on sink tokens within system prompts. While these tokens were initially considered problematic for hallucinations, the authors argue that they serve critical functional roles, including stabilizing attention and facilitating task progression. Through causal interventions like attention masking, zeroing, and content substitution, the authors explore how these tokens contribute to computational stability, information flow, and higher-order reasoning. Additionally, the paper introduces the Attention-Budget Hypothesis and proposes SPEAR, a novel intervention that reallocates attention to improve visual processing while preserving the stability provided by sink tokens.

### Strengths
1. The paper challenges the conventional wisdom surrounding sink tokens, proposing that they perform critical functions related to stability and reasoning, not just artifacts of hallucinations.
2.  The Attention-Budget Hypothesis provides a fresh perspective on modality bias and attention allocation, making a strong case for the trade-offs involved in boosting visual attention while maintaining system stability.
3.  The proposed SPEAR intervention is a novel and effective method for reallocating attention to mitigate hallucinations without sacrificing reasoning capabilities, outperforming the baseline and alternative methods like VAF.

### Weaknesses
1. The paper primarily compares the proposed SPEAR method with Visual Amplification Fusion (VAF). However, it lacks a broader comparison with other hallucination mitigation strategies (e.g., [1][2][3]) that could provide a more comprehensive understanding of the method’s performance
2. The paper could benefit from a more thorough error analysis, including failure modes and situations where SPEAR or other interventions might not perform as expected.
3. Some minor typos (e.g., “vi￾sion–language”) and inconsistent notation ($T_{\text{sys}\backslash\text{sink}}$ vs. $T_{\text{sys}} \setminus T_{\text{sink}}$). Figures are informative but could use clearer legends.
4. I think this version of the paper is rough, the authors should improve it before choosing to submit ICLR.

[1] VCD: Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding, CVPR 2024.

[2] OPERA: Alleviating Hallucination in Multi-Modal Large Language Models via Over-Trust Penalty and Retrospection-Allocation, CVPR 2024.

[3] Look Twice Before You Answer: Memory-Space Visual Retracing for Hallucination Mitigation in Multimodal Large Language Models, ICML 2025.

### Questions
1. How exactly do sink tokens act as “state-machine” elements? Are their activations correlated with control tokens (e.g., `<s>`, `</s>`, or position encodings)?  
2. Is this behavior consistent across transformer architectures with rotary vs. absolute position embeddings?
3. Have you tried tracking divergence or entropy in attention distributions post-intervention?
4. Equation (6) formalizes $\sum_T \Delta \alpha_i(T) = 0$. Can you empirically validate the *budget conservation* assumption across layers?  
5. Does it introduce any latency or memory overhead compared to VAF?
6. Could preserving sink tokens reinforce biases or hallucinations under adversarial prompting?  
7. The paper identifies sink tokens via high attention occupancy. Is there a dynamic component, e.g., per-layer thresholding?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper characterizes attention sinks in MLLMs. Through causal interventions, the authors demonstrate that these tokens are essential for stable inference. They propose an attention reallocation method to mitigate hallucination without decreasing attention on sink tokens.

### Strengths
1. The topic of attention sinks is interesting. Attention sinks are important tokens that stabilize inference and training.
2. The paper organization and writing are easy to follow.

### Weaknesses
1. The characterization of attention sinks in Section 3 largely recapitulates well-established findings from the LLM literature. They are common knowledge in the field and have been extensively documented in prior work. The paper does not provide novel insights beyond confirming that these patterns extend to MLLMs.

2. The functional analysis in Sections 4.1 and 4.2 is already covered by existing literature:  The finding that masking attention sinks causes collapse has been demonstrated in LLMs (e.g., Xiao et al 2023). 

Xiao et al. Efficient Streaming Language Models with Attention Sinks.

3. The claim that sink tokens function as part of an internal "state machine" in Section 4.3 is made without adequate support.

4. The proposed SPEAR method lacks technical innovation. It is a commonly adopted way widely utilized in previous works (e.g., Yang et al. 2025)

Yang et al. Understanding and Mitigating Hallucinations in Large Vision-Language Models via Modular Attribution and Intervention. ICLR 2025

5. The logic is not sound, attention sink is essential for stability does not mean that they are not harmful (not responsible for hallucination). 

5. Experiments are not sufficient to validate the effectiveness of this method. Should include other metrcis such as Chair_s and Chair_i. The improvement is also marginal.

### Questions
Attention sinks are important tokens that stabilize inference and training, but fundamental questions remain fully understudied: how they are formed, why they are necessary, whether they are sufficient, and what their side effects are.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper investigates the role of "attention sinks"—tokens that absorb a large amount of attention—within the system prompts of Multimodal Large Language Models (MLLMs). The authors challenge the prevailing view that these sinks are artifacts to be suppressed, arguing instead that they are "guardians". Through a series of causal interventions (masking, value zeroing, and value substitution), the paper claims to demonstrate that these sink tokens are critical for computational stability and serve as part of an internal "state machine" essential for multi-step reasoning. Based on this finding, the authors propose SPEAR, a plug-and-play intervention method that mitigates hallucinations by boosting visual attention while explicitly preserving these critical sink tokens. The paper claims this method is effective at reducing hallucinations without the degradation in reasoning performance seen in other methods.

### Strengths
1.The paper's primary strength is the graduated series of causal interventions in Section 4. The progression from masking attention to zeroing values to mean-value substitution  is a robust way to probe the function of these tokens.
2.The finding in Section 4.3 and Figure 4 that replacing sink token content selectively breaks multi-step reasoning while sparing simple tasks is a compelling (though qualitative) piece of evidence for their role in higher-order processing.

### Weaknesses
1. The paper's main weakness is the invalid experimental comparison in Section 6. The VAF baseline is defined as suppressing all system tokens ($\mathcal{S}_{VAF}=\mathcal{T}_{sys}$), which the paper has already proven in Section 4.1 leads to catastrophic model collapse.
2. The main results (Tables 4 & 5) are a product of this flawed setup. SPEAR outperforms VAF not because it's a better hallucination method, but because VAF (as defined by the authors) is a broken intervention that destabilizes the model. The experiment lacks a valid, strong baseline.
3. The claim that sink tokens function as an "internal 'state machine'"  is a significant exaggeration. The evidence is based on a single, qualitative example in Figure 4. While the tokens are clearly crucial for complex reasoning, "state machine" implies a level of procedural, computational function that is not fully substantiated by the provided data.

### Questions
1. Can the authors justify their definition of the VAF baseline? Did the original VAF paper (Yin et al., 2025) explicitly recommend suppressing *all* system tokens, including known stability anchors like the `<s>` token? Or is this a "strawman" definition created for this paper?
2. To demonstrate any real value, SPEAR must be compared against a *valid* baseline. How does SPEAR compare to a "VAF-Fixed" baseline—i.e., an implementation of VAF that *also* preserves the sink tokens ($\mathcal{S} = \mathcal{T}_{sys\backslash sink} \cup \mathcal{T}_{user} \cup \mathcal{T}_{out}$)?
3. Following from Question 2: The definition of SPEAR and a "VAF-Fixed" (as defined above) appear to be identical. Is the entire methodological contribution of this paper simply the observation that the VAF baseline must be implemented correctly to avoid model collapse?
4. Can the authors provide *quantitative* data to support the "state machine" claim? Specifically, what are the full benchmark scores (e.g., on SQA, MM-Vet) for the mean-value substitution intervention (Section 4.3), not just the qualitative example in Figure 4?

### Soundness
1

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
5

### Summary
This paper challenges the view that attention sinks in MLLMs are merely errors, arguing instead that specific tokens within the system prompt act as functional "guardians" essential for model stability and reasoning. Through a series of causal interventions, the authors demonstrate that these sink tokens are crucial for computational stability and act as part of an internal "state machine" for complex tasks. Building on this insight, the paper proposes the "Attention-Budget Hypothesis" and introduces SPEAR, a hallucination mitigation method that preserves these critical sink tokens while reallocating attention from non-essential text to visual tokens. Experimental results show SPEAR effectively mitigates hallucination without compromising reasoning abilities, thus validating the functional importance of sink tokens.

### Strengths
1. This work offers a compelling and insightful reframing of attention sinks. The argument that specific tokens act as functional "guardians"—essential for stability and reasoning—is a significant departure from the prevailing view that treats them as artifacts to be mitigated. This perspective moves the discourse beyond simple bug-fixing toward a more nuanced functional analysis of the model's internal mechanisms.

2. The claims are substantiated by a rigorous and systematic methodology. The use of well-designed causal intervention experiments—including attention masking and value vector manipulation—goes beyond correlational observations to convincingly dissect the functional roles of these "guardian" tokens. This robust experimental design provides strong, direct evidence for their necessity in maintaining computational stability and tracking task states, making the conclusions highly credible.

### Weaknesses
1. The conclusions are derived primarily from a specific model architecture (LLaVA) and a set of curated, synthetic tasks. It remains unclear whether the "guardian" mechanism is a universal phenomenon across MLLMs or an emergent property specific to certain architectural choices and training paradigms. The claims would be substantially strengthened by validation on a more diverse set of models, including different open-source families and closed-source systems, as well as on more complex, open-ended real-world scenarios.

2. The severity of the causal interventions—such as completely zeroing out attention scores or value vectors—may introduce confounds. This "hard" intervention forces the model into a highly unnatural, out-of-distribution state it would never encounter during training. Therefore, the observed performance collapse might not be solely due to the loss of the guardian's function, but could also be a result of the model's general fragility to such drastic internal state perturbations. Softer interventions, such as dampening activations to a low, non-zero level or replacing them with activations from a neutral token, could more precisely isolate the specific functional contribution of these tokens while minimizing the risk of inducing a general model failure.

### Questions
1. The emergence of a "guardian" token is a fascinating finding. Could you elaborate on the potential architectural or training precursors for this phenomenon? For instance, do you hypothesize that this is tied to the specific vision-language connector design in LLaVA, or perhaps to the instruction-tuning data format? How might this behavior differ in models with more deeply integrated cross-modal attention from early layers?

2. Regarding the functional role of the guardian token, your interventions effectively demonstrate its necessity. But do they reveal the dynamics of its contribution? For example, is its function a binary "on/off" switch, where any significant disruption causes catastrophic failure, or is it more of a graded, stabilizing influence? Have you considered the effects of "softer" interventions, such as dampening the guardian's value vector magnitude rather than nullifying it, to see if performance degrades more gracefully?

### Soundness
2

### Presentation
3

### Contribution
2
