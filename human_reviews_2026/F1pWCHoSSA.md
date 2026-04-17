# Unbiased Visual Reasoning with Controlled Visual Inputs

- Decision: Reject
- Scores: 4, 2, 4, 4, 6

## Abstract
End-to-end Vision-language models (VLMs) often rely on spurious visual cues, conflating perception with decision-making. We introduce VISTA (Visual Information Separation for Text-based Analysis), which enforces an explicit information bottleneck between a text-only reasoner and a stateless VLM sensor. The LLM reasoner decomposes each question and iteratively queries a VLM for visual facts; the VLM is instructed to reject queries that require high-level inference, creating an explicit information bottleneck. Trained on only 641 questions, VISTA yields large robustness gains on SpuriVerse across two vision backbones (+16.29\% with Qwen-2.5-VL-7B and +6.77\% with Llama-3.2-Vision-11B), while direct SFT or RL on the VLM fails to remedy spuriosity and can even exacerbate it. Despite never exposing the reasoner to raw pixels, VISTA slightly improves or remains on par with VLMs on everyday-scene benchmarks, including MMVP and SeedBench. Our learned reasoners transfer across sensors, indicating algorithmic rather than model-specific generalization. Together, VISTA enables spurious-resistant VQA by upgrading the brain, not the eyes.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes VISTA, a framework that improves visual reasoning by decoupling perception from reasoning. Instead of training a single vlm end-to-end, VISTA introduces a two-part system: a VLM sensor and a text-only LLM reasoner. The sensor acts like a visual probe, it sees the image but can only answer simple, factual perception questions such as object existence, color, or spatial relations, and rejects any inferential or subjective queries. The reasoner plans which visual facts to ask about, gathers them step-by-step, and then decides the final answer through reasoning alone.

This design enforces a strong information bottleneck, preventing the system from relying on spurious visual correlations (like background cues or stereotypes) that often mislead end-to-end models. Trained using reinforcement learning (GRPO) on a small set of 641 curated questions, VISTA achieves significant robustness gains on the SpuriVerse benchmark while maintaining comparable accuracy on MMVP and SeedBench. Ablation studies show that the rejection bottleneck is key to resisting bias, and removing it trades robustness for higher raw accuracy. The learned reasoning policy also transfers across unseen sensors, proving it learns algorithmic reasoning, not model-specific tricks.

### Strengths
1. VISTA introduces a clear and principled separation between perception and reasoning.
2. The paper identifies and articulates a real, under-addressed failure mode of end-to-end VLMs.
3. The sensor–reasoner design is modular, interpretable, and implementation-friendly.

### Weaknesses
1.Larger VLM sensors (e.g., Llama3.2-Vision) sometimes underperform smaller ones (e.g., Qwen2.5-VL), contrary to expected scaling trends.

Question: Can the authors explain why stronger sensors do not yield better reasoning outcomes under the VISTA setup? Is this due to an information bottleneck, training instability, or another factor?

2.The GRPO reinforcement signal improves some benchmarks (e.g., SpuriVerse) but has negligible or negative effects on others (e.g., MMVP).
Question: What causes this inconsistency? Did the authors experiment with alternative reward functions or training schedules to stabilize performance?

3. The reasoning policy is trained on only 641 curated examples, which seems insufficient for robust generalization.
Question: How sensitive are the results to this small dataset? Would scaling up training data or incorporating noisier but larger supervision alter the observed outcomes?

4. While the controlled bottleneck improves robustness, it reduces accuracy on standard benchmarks such as SeedBench.
Question: Can the authors quantify this trade-off and justify the reduction in clean-scenario performance as an acceptable cost for improved robustness?

5. Evaluation is limited to SpuriVerse and MMVP, which, while interesting, lack diversity and scale.
Question: Why were broader multimodal or reasoning benchmarks (e.g., GQA, VizWiz, ScienceQA) excluded from evaluation?

6. Ambiguous causal attribution
It remains unclear whether the robustness gains arise from the rejection rule, RL regularization, or the architectural separation itself.
Question: Did the authors isolate the effects of the architectural split from those of the training regime to identify the primary driver of improvement?

### Questions
Please see in weaknesses (6 questions)

### Soundness
3

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
The paper presents an interesting framework in which an LLM performs textual reasoning in a Chain-of-Thought (CoT) style and issues queries to a VLM that handles perception-only tasks. I regard this setup as an instantiation of the general ReAct framework, with the VLM functioning as the action executor. Nonetheless, the proposed approach remains interesting for two reasons:
 (1) it explicitly disentangles textual reasoning from visual understanding—a design philosophy commonly adopted by many visual reasoners; and
 (2) it provides some theoretical analysis of the framework, although it is unclear how this analysis connects to the central problem the paper aims to address: shortcuts that correlate spuriously with the correct answer. 

My main concern lies in the empirical results: the proposed method generally underperforms compared to end-to-end training with reinforcement learning (as shown in Table 1). I am also curious why no end-to-end (RL) results are reported for Llama3.2-Vision. Is it because VISTA performs worse than the end-to-end (RL) counterpart on this model? While the authors provide some explanations in the experimental analysis, they are not sufficiently convincing to demonstrate that the proposed method offers clear value to the community.

### Strengths
an interesting framework for visual reasoning

### Weaknesses
see my comments in Summary

### Questions
see my comments in Summary

### Soundness
2

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
3

### Summary
This paper proposes a framework named VISTA to address the issue of shortcut learning in vision-language models (VLMs), where models tend to rely on superficial visual cues rather than developing a deep understanding of the logical relationships between questions and visual inputs. The VISTA framework explicitly decomposes the reasoning process into a visual sensor (VLM) for perception and a reasoning module (LLM) for logical inference, thereby mitigating the influence of shortcut learning. Experimental results on two benchmarks, MMVP and SeedBench-500, demonstrate the effectiveness of the proposed approach.

### Strengths
1. The authors propose the VISTA framework to address shortcut learning in VLMs, which explicitly separates visual perception (sensor) from logical reasoning (reasoner) to mitigate reliance on spurious visual cues.

### Weaknesses
1. While the VISTA framework attempts to address shortcut learning by employing a dual-agent architecture (VLM + LLM), this approach does not fundamentally solve the underlying issue within the VLM itself. The VLM component remains susceptible to shortcut learning, merely transferring rather than resolving this critical limitation.

2. The evaluation is currently limited to established benchmarks. To better demonstrate the method's robustness and generalizability, performance should be validated on more recent and challenging VQA benchmarks such as MMMU and MMMU-Pro.

### Questions
1. Does the VLM component itself still suffer from shortcut learning? In the proposed agent system, the VLM appears to be reduced to a perceptual module, leaving its inherent shortcut learning issues unaddressed.

2. How does the method generalize to more comprehensive benchmarks? Evaluation on challenging benchmarks such as MMMU and MMMU-Pro would better demonstrate its generalization capability.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Visual Information Separation for Text-based Analysis (VISTA), a framework that enforces an information bottleneck between a text-only reasoner and a VLM to mitigate spurious visual correlations (hopefully). By restricting the sensor to answer only low-level perceptual queries, VISTA separates perception from reasoning and promotes evidence-seeking behaviors. On SpuriVerse, MMVP, and SeedBench, VISTA achieves claimed robustness gains while maintaining comparable general accuracy. Theoretical analysis links improved generalization to reduced information bandwidth across the sensor–reasoner interface.

### Strengths
Overall, I like the high-level motivation which limits the VLMs to do what they can do. For this direction, actually I expect to see more analysis from how to determine what VLMs can do well, instead of pretty unclear queries accept or reject in a straightforward way. Anyway, targeting spurious visual correlations in VLMs is very related to recent progress in VLMs. 

Empirical results across multiple benchmarks demonstrate certain robustness and cross-model generalization with minimal data and training cost. Some ablation studied are also included.

### Weaknesses
- The biggest weaknesses to me is the experimental settings. MMVP is such a small-scale dataset with only 150 images pair, and the author randomly 500 samples subset from SeedBench. The choice of experiments are hard to delivery something reliable. Besides, as the author mentioned the evaluated datasets are "everyday-scene benchmark". However, as this paper is motivated by "existing VLMs rely on spurious visual cues, conflating perception", there are datasets suitable for this purpose, such as ViLP (https://arxiv.org/pdf/2501.00569) and HallusionBench (https://arxiv.org/abs/2310.14566). I would recommend the authors seriously consider extending the evaluation benchmarks, not limited to what I suggested. 

- The proposed theoretical bound seems not non-trivial.

### Questions
- I am confused the difference of "Are there multiple dots and a white flag with an orange pole in the painting?" and "What is in the image?", the later question what is in the image requires more reasoning & descriptions, while the former one is evidence checking. I am actually confused what are boundaries of accepted vs. rejected queries. 

- How do you compute the advantage for the used GRPO? 

- Besides, I raised some questions above in the weakness section. 

I will adjust my final scores based on the response.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper aims to tackle the persistent issue of spurious correlations in vision–language models (VLMs), where models often conflate perception with reasoning. To address this, the authors propose VISTA (Visual Information Separation for Text-based Analysis), a framework that enforces an explicit information bottleneck between a text-only reasoner and a stateless visual sensor. The reasoner iteratively queries the sensor for perception-level facts, while the sensor rejects high-level inference requests to prevent shortcut learning. Through reinforcement learning, VISTA develops neutral, evidence-seeking reasoning policies. Experiments on SpuriVerse, MMVP, and SeedBench show substantial robustness gains against spurious cues while maintaining comparable accuracy on everyday visual tasks. The results demonstrate that decoupling perception from reasoning improves generalization and mitigates visual bias in multimodal systems.

### Strengths
1)	The paper crisply identifies spurious-cue reliance and the conflation of perception and reasoning in end-to-end VLMs, motivating a modular remedy.
2)	VISTA enforces an explicit information bottleneck between a text-only reasoner and a stateless VLM sensor, cleanly separating decision-making from raw pixels.
3)	The sensor accepts only six classes of perception queries and rejects high-level inference, with a concrete policy and examples.

### Weaknesses
1)	The proposed information bottleneck between the sensor and reasoner is conceptually interesting, but it may also introduce new risks. By restricting the reasoner’s access to full and detailed visual information, the model could miss critical cues needed for complex reasoning. Moreover, if the stateless visual sensor makes errors or misinterprets the scene, the reasoner has no means to recover or verify the missing context, potentially amplifying mistakes. The paper should further analyze and discuss this trade-off between robustness to shortcuts and vulnerability to information loss
2)	It is unclear whether the authors plan to release their code and trained models. Given that the paper’s main contribution lies in the proposed VISTA framework and its controlled perception–reasoning interface, public release is crucial for reproducibility and community validation.
3)	The paper should further analyze whether the model truly learns accurate and coherent reasoning after GRPO training. Since the reward is assigned only based on the final answer correctness, it is unclear whether the intermediate chain-of-thought steps generated by the reasoner are logically sound or merely optimized for outcome matching. Without evaluating the quality or faithfulness of these reasoning traces, the claimed improvement in reasoning robustness remains uncertain.
4)	The sensor accepts only a fixed set of perception query types, which may limit generalization to unseen reasoning formats or richer visual evidence needs.
5)	The multi-turn setup allows up to 24 rounds and 8192 tokens per episode, yet there is no throughput/cost analysis to assess deployment practicality.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
