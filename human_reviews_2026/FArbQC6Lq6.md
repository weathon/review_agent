# Accelerating Structured Chain-of-Thought in Autonomous Vehicles

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 6, 2

## Abstract
Chain-of-Thought (CoT) reasoning enhances the decision-making capabilities of vision-language-action models in autonomous driving, but its autoregressive nature introduces significant inference latency, making it impractical for real-time applications. To address this, we introduce FastDriveCoT, a novel parallel decoding method that accelerates template-structured CoT. Our approach decomposes the reasoning process into a dependency graph of distinct sub-tasks, such as identifying critical objects and summarizing traffic rules, some of which can be generated in parallel. By generating multiple independent reasoning steps concurrently within a single forward pass, we significantly reduce the number of sequential computations. Experiments demonstrate a 3-4$\times$ speedup in CoT generation and a substantial reduction in end-to-end latency across various model architectures, all while preserving the original downstream task improvements brought by incorporating CoT reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the high latency of CoT reasoning in VLA models for autonomous driving. It proposes FastDriveCoT, a parallel decoding method that leverages a structured CoT template and dependency graph to generate multiple reasoning steps concurrently, achieving significant speedup without sacrificing task performance.

### Strengths
The paper tackles a critical and practical problem: the inference latency of CoT reasoning in time-sensitive autonomous driving applications . The quality of the proposed FastDriveCoT method is demonstrated through its well-structured approach, including the use of a CoT template, a dependency graph for parallelization, and an efficient implementation leveraging custom attention masks for shared KV caching . The clarity is good, explaining the parallel decoding mechanism well. Experimental results convincingly show substantial speedups while maintaining or even slightly improving downstream task performance.

### Weaknesses
The effectiveness of FastDriveCoT relies on a predefined, template-structured CoT. While suitable for the relatively standardized reasoning in AVs, this might limit flexibility or generalizability compared to free-form CoT. The approach also depends on high-quality, template-adherent CoT data, which was auto-labeled using a large VLM, potentially introducing noise or biases . While the paper claims zero computational overhead compared to autoregressive decoding, managing the dependency graph and custom attention masks might introduce implementation complexity. A minor performance degradation was observed in the Transfusion architecture experiments.

### Questions
1. How adaptable is FastDriveCoT to different CoT templates or reasoning structures? Does modifying the template (e.g., adding/removing fields, changing dependencies) require significant re-engineering of the dependency graph or attention masks?
2. The method relies on accurately following the template. How does the parallel decoding handle cases where the model deviates from the expected structure or produces errors within a field that might affect dependent fields generated later?
3. Could you elaborate on the comparison between FastDriveCoT and other LLM acceleration techniques like speculative decoding or lookahead decoding? Are these methods complementary, or does FastDriveCoT offer distinct advantages for structured CoT?
4. Why the Transfusion architecture show a slight degradation in trajectory ADE with FastDriveCoT compared to autoregressive method, while the VLA-AR architecture always showed improvement?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces FastDriveCoT, a novel parallel decoding method to address the inference latency of Chain-of-Thought (CoT) reasoning in vision-language-action models for autonomous driving. FastDriveCoT decomposes the reasoning process into a dependency graph of distinct sub-tasks, allowing independent sub-tasks to be generated in parallel. By generating multiple independent reasoning steps concurrently in a single forward pass, the method significantly reduces sequential computations. Experiments show that FastDriveCoT achieves a 3-4x speedup in CoT generation and a substantial reduction in end-to-end latency across various model architectures, all while preserving the downstream task performance improvements brought by CoT.

### Strengths
- This paper addresses a very practical and significant problem. The robustness and interpretability of LLMs are crucial for AV systems, but their latency is a major obstacle to deployment.

- The method of decomposing CoT into a dependency graph and achieving parallel decoding in a single forward pass via a custom attention mask appears sensible. This approach can effectively utilize the parallel computing capabilities of modern GPUs and fully reuse the KV cache.

- The paper demonstrates significant speedup (3-4x in CoT time) without compromising on accuracy (i.e., not sacrificing downstream task performance).

### Weaknesses
- The core contribution heavily relies on a manually designed, highly fixed CoT template. Although the authors mention this template is an "example" (line 185), the entire methodology (including the dependency graph construction) is based on this fixed structure. Furthermore, when handling "multi-instance" fields (like lanes and critical objects), the method depends on a fixed number of slots (e.g., 3 time ranges for lanes, 4 critical objects). This might be fragile in complex, dynamic real-world scenarios.

-The experimental comparison is limited to "No CoT" and "Autoregressive CoT." The paper lacks comparisons with other SOTA LLM acceleration techniques. For instance, speculative decoding is a more general acceleration method that does not require a fixed template. Alternatively, could training a smaller model to mimic the CoT output achieve a similar balance between latency and performance? Without these comparisons, it is difficult to judge if FastDriveCoT is the best approach for this problem.

- The evaluation relies entirely on a large "internal dataset." The authors do not report results on any public, standard autonomous driving benchmarks (like nuScenes or Waymo Open Dataset). This severely harms the work's reproducibility. Although the authors promise to release a subset of the dataset (line 376), during the review period, we cannot independently verify the findings or fairly compare this method with other SOTA methods trained on public benchmarks.

### Questions
1. Have the authors considered a comparison with Speculative Decoding? It seems to be a more general acceleration technique that does not require a predefined CoT structure.

2. How does the method's performance (both speed and accuracy) degrade when the number of entities in a real-world scene (e.g., critical objects) exceeds the fixed slots in the template?

### Soundness
3

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
2

### Summary
The paper proposes FastDriveCoT, a parallel decoding method that accelerates template-structured CoT. The paper targets the problem that normal chain of thought for driving is too slow for 10 Hz control because autoregressive decoding is fully sequential. Experiments like Table 1 shows FastDriveCoT gives 3.1$\times$ to 4.1$\times$ CoT speedup while keeping or improving meta action IOU and ADE over autoregressive CoT.

### Strengths
- The reviewer found the proposed idea to predefine a CoT template and decode independent fields in parallel using a dependency graph to be interesting. 
- Again, interestingly parallel decoding even slightly improves template adherence so trajectory ADE at 3 seconds for Qwen2.5 VL 3B improves to 0.482 from 0.511 showing structure can help quality.
- Experiments in Table 1, the ablation style analysis in Figure 4, show consistent 3$\times$ to 4$\times$ CoT speedup with only small drops in some long horizon ADE.
- Writing is clear.

### Weaknesses
- Table 1 only compares no CoT and standard autoregressive CoT but it should also compare against shorter skeleton of thought decoding or speculative decoding baselines which are natural for speed claims.
- Some typos the reviewer could see: Line 115: dependecies -> dependencies; Figure 3 caption independency -> independence; Line 352 diving -> driving
- See questions below.

### Questions
- In Table 1 the Qwen2 0.5B + VLA-AR row shows meta action IOU 0.811 for autoregressive CoT but FastDriveCoT is 0.804 while trajectory ADE improves. Can you explain why meta action degrades when both use the same template.
- Figure 4 reports that speedup is almost linear in average parallel degree but Table 1 shows overall time speedup is only 1.9x to 2.4x for some models. Please clarify what overhead outside CoT causes this gap.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
In this paper, the authors are presenting "FastDriveCoT". This is a  parallel decoding method that accelerates Chai of Thought reanosing for decision making. Here the authors apply this to the field of autonomous driivng. Overall, their approach is achieving a speed up of CoT reaosning and therefore reduces the overall end-to-end latenc in the AV software stack.

### Strengths
- Good overall novelty for AV: The presented approach shows good novelty in the field of AV reasoning. 
- The authors present good technical innovation, by combining the structure CoT tmeplate, the dynamic programming algorithm and by maintining the zero extra FLOPs
- Good emperical results by showing the speed up of 3-4 times CoT reasonsing and therfore 2x faster E2E inference
- Good emperical results with different VLMs like Qwen2, Qwen2.5 and Qwen3
- Good comprehensive albation studies for effieciency and performance trade-offs

### Weaknesses
Currently, the paper shows a very good technical approach, but there are strong weaknesses that questions the papers overall impact:

- Currently, the method relies on highly structured reasoning templates that are selected by the authors for the driving task. It is unclear to the reader, if this enalbes generalization at all. My concern here is that this approach will not generalize very well to open-ended reasoning tasks. This is unfortunately something we see in AV every day.
- This dependence on predefined templates requires very specific CoT fields, dependency graphs, and manual  definitions. THis is an enginnering overhead and manual labor, that will not impact the AV world especially when we have other approaches that scale better
- Althought he authors show it can be conceptionally efficient, the custom attention maks highly overcomplicate the integration for standard frameworks
- The biggest flaws is the comparison of the approach. ALl experiments run on an internal dataset, an external validation or open benchmark on CARLA or other open datasets is not given

### Questions
1. How flexible is the COT Template design you have choosend?
2. Is the dependacy graph currently hand-crafted by you or is it learned from data?
3. How does your approach scale in multi-GPU setups?
4. How does your approach work on embedded hardware e.g. nvidia jetson so it can be implemented in a car?
5. Have you evaluated your approach on open benchmakrs or other simulations?

### Soundness
3

### Presentation
2

### Contribution
2
