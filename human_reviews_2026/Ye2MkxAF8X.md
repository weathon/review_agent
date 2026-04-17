# Benchmarking Gaslighting Negation Attacks Against Multimodal Large Language Models

- Decision: Reject
- Scores: 4, 2, 6

## Abstract
Multimodal Large Language Models (MLLMs) have exhibited remarkable advancements in integrating different modalities, excelling in complex understanding and generation tasks. Despite their success, MLLMs remain vulnerable to conversational adversarial inputs. In this paper, we systematically study gaslighting negation attacks—a phenomenon where models, despite initially providing correct answers, are persuaded by user-provided negations to reverse their outputs, often fabricating justifications. We conduct extensive evaluations of state-of the-art MLLMs across diverse benchmarks and observe substantial performance drops when negation is introduced. Notably, we introduce the first benchmark GaslightingBench, specifically designed to evaluate the vulnerability of MLLMs to negation arguments. GaslightingBench consists of multiple-choice questions curated from existing datasets, along with generated negation prompts across 20 diverse categories. Throughout extensive evaluation, we find that proprietary models such as Gemini-1.5-flash and GPT-4o demonstrate better resilience compared to open-source counterparts like Qwen2-VL and LLaVA, though even advanced reasoning-oriented models like Gemini-2.5-Pro remain susceptible. Our category level analysis further shows that subjective or socially nuanced domains (e.g., Social Relation, Image Emotion) are especially fragile, while more objective domains (e.g., Geography) exhibit relatively smaller but still notable drops. Overall, all evaluated MLLMs struggle to maintain logical consistency under gaslighting negation attack. These findings highlight a fundamental robustness gap and provide insights for developing more reliable and trustworthy multimodal AI systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents the first systematic study of “gaslighting negation attacks” against Multimodal Large Language Models (MLLMs). It defines these attacks as conversational manipulations where models, initially correct, are persuaded by user-provided negations to revise their answers incorrectly—often with fabricated justifications. The authors conclude by emphasizing that gaslighting negation attacks represent a distinct, underexplored adversarial failure mode, highlighting the need for fine-grained alignment and calibration strategies to enhance robustness and trustworthiness in multimodal AI systems.

### Strengths
1. The paper introduces “gaslighting negation” as a new class of conversational attack, distinct from jailbreak or prompt injection. It’s a subtle yet impactful vulnerability, especially in real-world dialogue contexts.

2. Proprietary models (Gemini-1.5-flash, GPT-4o, Claude-3.5) outperform open-source ones (Qwen, LLaVA) but still degrade notably.
3. Figure 7 (p.8) illustrates models contradicting earlier correct answers—sometimes even producing hallucinated justifications (“I apologize, the color is red”)—clearly conveying the behavioral risk.
4. Includes supplementary experiments on question type sensitivity (Appendix A.1) and negation phrasing effects (Figure 8, p.15).

### Weaknesses
1. The explanation of why over-alignment induces gaslighting behavior is qualitative.
2. The paper exposes the vulnerability well but provides no mitigation strategies, even conceptually, e.g., calibration, adversarial training, debate-style reinforcement
3. The study does not explore internal attention or activation traces to explain why negation overrides factual grounding—especially relevant for multimodal reasoning.

4. Minor stylistic issues (e.g., “ne￾gation,” “conversational negation attack”) indicate OCR artifacts or typesetting errors. Figures are informative but occasionally crowded.

### Questions
1. Have you examined whether negation causes greater changes in text–vision attention layers or in decoder self-attention?
2. How might calibration-aware decoding mitigate confident hallucination?
3. How consistent are generated negation prompts across linguistic forms, e.g., “not,” “no,” “never”?  
4. Would “self-consistency” or “chain-of-verification” decoding strategies resist these attacks?  
5.  Any differences in model responses to explicit vs. implied negation?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies an underexplored but important vulnerability of Multimodal Large Language Models (MLLMs): their tendency to reverse correct answers when given with user-provided negations, a phenomenon termed gaslighting negation attacks. The authors introduce GaslightingBench, a benchmark of 1,287 multimodal multiple-choice questions over 20 categories, to evaluate robustness to such attacks. They evaluate on multiple proprietary (GPT-4o, Gemini-1.5-flash, Claude-3.5-Sonnet, Gemini-2.5-Pro) and open-source (Qwen2-VL, LLaVA) models across several established multimodal datasets (MMMU, MMBench, MathVista, ChartQA, etc.), analyzing pre- and post-negation performance. Results show significant performance drop after the gaslighting attack.

### Strengths
1. The paper is clearly written and easy to follow.
2. The gaslighting attack on multimodal LLMs are under-explored (although it has been extensively studied under text-only LLMs).

### Weaknesses
### **Major**

1. **Over-simplified gaslighting prompt type:** The paper only studies direct negation and short-answered gaslighting prompt. However, I think this type of gaslighting prompt may be over-simplified, and less practical:
    - In this work, the gaslighting prompts are all directly telling the LLMs the (incorrect) answer. However, since LLMs are trained to follow user instructions. If the user directly tells the LLM what the answer should be, then it is expected that the LLM should consider the user input in the first place. The more proper gaslighting prompt should be questioning/debating as studied in existing 'gaslighting' attack in text-only domain (e.g. [1,2]), or negations with more explanation (e.g. CoT [2]).
    - In many cases when the original context is not sufficient and the MLLM is making prediction based on certain prior, if the user directly provides the (incorrect) answer, it is expected that the MLLM should change mind. For instance in the left most example in Figure 5, the MLLM predicts "professional" relation based on the people's outfits. However, if the user directly tells the MLLM they are in "family" relation, then it is expected that the MLLM should follow user's input, as it has no further prior knowledge what the relation is.
2. **Incomprehensive gaslighting type:** Based on the above consideration, I feel the proposed benchmark lacks some comprehensiveness:
    - Currently it only includes negation style prompt. I think it is important to also include questioning/debating style gaslighting.
    - Currently it focuses mostly on short-answered gaslighting attack without explanation. I think it is worthwhile to study how the model behaves when the incorrect explanation (e.g. CoT) is provided along with the gaslighting input.
3. **Evaluate on more open-sourced models:** Currently the evaluation of open-sourced models is conducted on qwen2-vl-72b and llava1.6-7b. I think more evaluation are needed:
    - qwen2-vl-72b and llava1.6-7b are very different in size and backbones, making the evaluation un-controlled and hard to draw conclusions. For instance, if you want to study the effect of LLM sizes on gaslighting attack, you should ablate on qwen2-vl-2b, 7b and 72b, etc.
    - Both qwen2-vl-72b and llava1.6-7b are not RL-finetuned. I think it's worthwhile to evaluate over RL-finetuned MLLMs such as internvl2.5/3/3.5, gemma3 and qwen2.5/3, many of which are available before ICLR paper deadline.
3. **Missing some more in-depth analysis:** 
    - The paper lacks a deeper analysis of how the model reverse decisions: no probing of attention patterns/representation shifts/intermediate reasoning traces etc.
    - I feel there could be more case studies to analyze when model predicts incorrectly after the gaslighting attack. Are they hard negative? Or are actually false negative (e.g. Figure5 left most)?
4. **Discussion on mitigation is minimal:** given that gaslighting is not a new topic in LLM literature, and there have been studies on how to mitigate [2], I feel it may be necessary to provide some baseline approaches to mitigate the impact of such attack, along with the benchmark.

### **Minor**
1. Please consider using /citep instead of /cite to put citations in parentheses for better readability.

 1 Can ChatGPT Defend its Belief in Truth? Evaluating LLM Reasoning via Debate. ACL 2023

 2 Aligning Large Language Models for Faithful Integrity Against Opposing Argument. AAAI 2025

### Questions
The paper is clearly written and straightforward, so I do not have additional questions. Please see weaknesses.

### Soundness
3

### Presentation
3

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
This paper introduces GaslightingBench, a novel benchmark designed to systematically evaluate the vulnerability of Multimodal Large Language Models (MLLMs) to gaslighting negation attacks—a form of adversarial input where models are misled into reversing their initially correct answers through user-provided negations, often fabricating justifications in the process. The authors conduct extensive experiments across multiple MLLMs (both proprietary and open-source) and existing multimodal benchmarks, demonstrating significant performance drops when negation is introduced.

### Strengths
1.The paper addresses an under-explored but critical issue—negation-induced inconsistency in MLLMs—and introduces the first dedicated benchmark (GaslightingBench) for evaluating this vulnerability.

2.The study evaluates a wide range of MLLMs across multiple datasets and question formats, providing a thorough and comparative analysis of model robustness.

3. Rigorous Methodology:The evaluation pipeline is well-structured, including negation generation, post-processing, and careful dataset curation. The use of multiple negation styles (neutral, anger, authority) adds depth to the analysis.

### Weaknesses
1.Benchmark Bias Toward MCQs: GaslightingBench is primarily based on multiple-choice questions, which may not fully capture the complexity of real-world adversarial interactions or free-form reasoning.

2. Different real-world complexity are not considered:The study focuses on controlled benchmarks; it does not test how gaslighting attacks perform in more dynamic, multi-turn, or real-world conversational settings.

3. Lack of Mitigation Strategies or insight. The paper identifies the problem but does not propose or evaluate methods to mitigate gaslighting attacks, which would have strengthened its practical impact.

### Questions
See the weakness

### Soundness
3

### Presentation
3

### Contribution
2
