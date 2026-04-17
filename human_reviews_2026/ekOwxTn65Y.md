# P$^2$-DPO: Grounding Hallucination in Perceptual Processing via Calibration Direct Preference Optimization

- Decision: Accept (Poster)
- Scores: 6, 4, 4

## Abstract
Hallucination has recently garnered significant research attention in Large Vision-Language Models (LVLMs). Direct Preference Optimization (DPO) aims to learn directly from the corrected preferences provided by humans, thereby addressing the hallucination issue. Despite its success, this paradigm has yet to specifically target the perceptual bottleneck in attended regions or address insufficient Visual Robustness against image degradation. Furthermore, existing preference pairs are often vision-agnostic and their inherently off-policy nature limits their effectiveness in guiding model learning. To address these challenges, we propose Perceptual Processing Direct Preference Optimization (P$^2$-DPO), a novel training paradigm in which the model generates and learns from its own preference pairs, thereby directly addressing the identified visual bottlenecks while inherently avoiding the issues of vision-agnostic and off-policy data. It introduces: (1) an on-policy preference pairs construction method targeting Focus-and-Enhance perception and Visual Robustness, and (2) a well-designed Calibration Loss to precisely align visual signals with the causal generation of text. Experimental results demonstrate that with a comparable amount of training data and cost, P$^2$-DPO outperforms strong baselines that rely on costly human feedback on benchmarks. Furthermore, evaluations on Attention Region Fidelity (ARF) and image degradation scenarios validate the effectiveness of P$^2$-DPO in addressing perceptual bottleneck in attended regions and improving Visual Robustness against degraded inputs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes P2-DPO (Perceptual Processing Direct Preference Optimization), a self-supervised alignment method for Large Vision-Language Models (LVLMs) designed to mitigate hallucinations by focusing on Perceptual Processing rather than Perception failures.
Unlike prior DPO or RLHF approaches that rely on human or synthetic preferences, P2-DPO generates on-policy, vision-aware preference pairs through two mechanisms: (1) Focus-and-Enhance pairs (for perceptual bottlenecks) and (2) Visual Robustness pairs (for degraded inputs) and uses them for Reinforcement Learning training. To enhance training efficacy, they further introduce a Calibration Loss to align visual evidence with text generation and a Dynamic Deficit-Weighting (DDW) scheme for adaptive balancing. Experiments on hallucination benchmarks (POPE, HallusionBench, AMBER, MMHal-Bench, TextVQA) show consistent improvements over DPO baselines and comparable results to human-feedback-based training.

### Strengths
The paper offers a compelling and underexplored perspective: separating Perceptual Processing failures (i.e., hallucinations despite correct attention) from Perception failures. This framing adds analytical depth and could motivate new diagnostic tools for LVLMs.

The idea of generating on-policy, vision-grounded preference pairs directly from the model’s own attention maps is interesting.

The experiments span several challenging hallucination benchmarks (POPE, HallusionBench, AMBER, MMHal-Bench, TextVQA) and include targeted validation (e.g., AFR, noise robustness), with clean tables and consistent results.

### Weaknesses
Although the method avoids human labels, generating multiple augmented views and attention-driven crops for each sample adds nontrivial computational overhead.  Could the authors provide results to demonstrate what is the runtime overhead (per training sample or epoch) compared to standard DPO?

The entire method depends on the quality of attention maps and cropping heuristics. If attention localization fails, the generated preference pairs may reinforce spurious regions or noise. How does the model behave when the attention maps are themselves incorrect—does this lead to reinforcing spurious crops? How would the approach handle multi-object scenes where attention maps have multiple disjoint foci?

### Questions
See weaknesses

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
In this paper, the authors propose Perceptual Processing Direct Preference Optimization, a new method to address hallucination, mainly focusing on the visual end and self-generated preference data.

### Strengths
The writing is clear, and the results are solid.

### Weaknesses
1. The model architecture is LLaVA, and the version is relatively old. It's unclear how it performs on current new models.
2. Lacks extensive comparison with related work, such as [1], which also uses model self-generated data and has a consistent idea—namely, focusing on hallucination in visual perception. Therefore, I have doubts about the novelty.

[1] Calibrated Self-Rewarding Vision Language Models

### Questions
1. How does this method perform on Qwen-VL?
2. During training, how does the attention pattern change? Does it tend to favor visual tokens rather than textual tokens?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces P²-DPO, a self-correcting framework for reducing hallucination in Large Vision-Language Models. It targets perceptual processing bottlenecks rather than perception failures, generating on-policy, vision-aware preference pairs to improve attention and robustness. With a Calibration Loss and Dynamic Deficit-Weighting, P²-DPO enhances visual grounding and robustness, outperforming human- and AI-feedback DPO baselines without external supervision.

### Strengths
- The paper provides a novel perspective on hallucination in LVLMs by distinguishing between perception failure and perceptual processing failure. It highlights the latter as an overlooked yet solvable issue that can be addressed through self-correction within the model, offering new insight into the root causes of hallucination.

- The proposed on-policy strategy, which generates vision-aware contrastive preference pairs from the model itself, effectively removes the dependence on human or AI feedback used in traditional DPO frameworks.

### Weaknesses
- Limited experimental scope: The experiments are primarily conducted on LLaVA-1.5-7B and Qwen2.5-VL-3B, but it is unclear why the authors did not include the more commonly used Qwen2.5-VL-7B model. This omission limits the completeness of the evaluation and raises questions about the method’s scalability and consistency across model sizes.

- Potential self-reinforcement bias in on-policy data: Although the on-policy preference generation strategy avoids the off-policy issue, self-generated preference pairs in the early training stage may inherit the model’s existing biases or hallucinations.

- Lack of qualitative interpretation of improvements: While quantitative results are comprehensive, the paper lacks intuitive visual evidence explaining how the Calibration Loss and Dynamic Deficit-Weighting improve the model’s attention distribution or semantic grounding. Visualizations such as attention map changes would make the effectiveness of these mechanisms clearer and more convincing.

### Questions
Please refer to the weakness part.

### Soundness
3

### Presentation
3

### Contribution
2
