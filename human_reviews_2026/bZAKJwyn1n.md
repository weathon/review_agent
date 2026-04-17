# Pursuing Minimal Sufficiency in Spatial Reasoning

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Spatial reasoning, the ability to ground language in 3D understanding, remains a persistent challenge for Vision-Language Models (VLMs). We identify two fundamental bottlenecks: \textit{inadequate} 3D understanding capabilities stemming from 2D-centric pre-training, and reasoning failures induced by \textit{redundant} 3D information.
To address these, we first construct a Minimal Sufficient Set (MSS) of information before answering a given question: a \textit{compact} selection of 3D perception results from \textit{expert models}. We introduce \textbf{MSSR} (Minimal Sufficient Spatial Reasoner), a dual-agent framework that implements this principle.  A \textit{Perception Agent} programmatically queries 3D scenes using a versatile perception toolbox to extract sufficient information, including a novel \textbf{SOG} (Situated Orientation Grounding) module that robustly extracts language-grounded directions. A \textit{Reasoning Agent} then iteratively refines this information to pursue minimality, pruning redundant details and requesting missing ones in a closed loop until the MSS is curated. 
Extensive experiments demonstrate that our method, by explicitly pursuing both sufficiency and minimality, significantly improves accuracy and achieves state-of-the-art performance across two challenging benchmarks. Furthermore, our framework produces interpretable reasoning paths, offering a promising source of high-quality training data for future models. Source code will be made publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes MSSR (Minimal Sufficient Spatial Reasoner), a dual-agent framework intended to improve spatial reasoning in vision-language models. The framework introduces a Perception Agent using a versatile perception toolbox (including a novel SOG (Situated Orientation Grounding) module) to extract 3D information, and a Reasoning Agent that iteratively prunes and augments this information to curate a minimal sufficient set (MSS) required for reasoning about a given question. Experimental results on two challenging benchmarks (MMSI-Bench and ViewSpatial-Bench) demonstrate state-of-the-art performance. The framework can also produce interpretable reasoning paths as training data for future models.

### Strengths
1. The designed dual-agent framework is meaningful and effective.
2. The proposed method is training-free, making it easy to integrate seamlessly with existing models.
3. The method achieves state-of-the-art performance, outperforming all proprietary, open-source, specialist, and agentic models.

### Weaknesses
1. The paper only selects a proprietary LLM (GPT-4o) as the backbone to demonstrate the effectiveness of the method. Additional experiments and analysis on open-source LLMs (e.g., Qwen2.5-VL) could further validate the efficacy and generalizability of the method.
2. The paper only provides the inference time per iteration. A thorough efficiency comparison with previous methods and corresponding analysis could further enhance the soundness.

### Questions
1. How is the quality of the constructed Minimal Sufficient Set ensured? Are there scenarios where the Reasoning Agent might prune critical information or request unnecessary information?
2. What are the common failure modes of the model? Does it struggle (e.g., the Reasoning Agent never makes the final decision) when the question is very complex?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes MSSR (Minimal Sufficient Spatial Reasoner), a dual-agent, zero-shot framework for improving 3D spatial reasoning in Vision-Language Models. The framework works by constructing a Minimal Sufficient Set (MSS), the smallest subset of 3D perceptual information sufficient to answer a query, by a dual agent iteratively. MSSR consists of a Perception Agent, which programmatically queries a scene using modular visual tools including a new Situated Orientation Grounding (SOG) module for robust direction grounding, and a Reasoning Agent, which prunes irrelevant information and iteratively requests only what is missing until the MSS is achieved. Experiments on MMSI-Bench and ViewSpatial-Bench show MSSR achieving state-of-the-art results, outperforming models like GPT-4o and Gemini 2.5 Pro.

### Strengths
- The paper is clearly written and logically structured, making the methodology and motivation easy to follow.
- The related work section provides a well-organized and comprehensive discussion covering VLMs for spatial reasoning, agentic framework, and visual programming paradigms.
- The proposed MSSR framework is conceptually sound and technically well-motivated.
- Extensive experiments demonstrate strong empirical performance, achieving state-of-the-art results on both MMSI-Bench and ViewSpatial-Bench while maintaining interpretability through reasoning traces.

### Weaknesses
- The paper formatting does not fully comply with ICLR submission requirements.
- The approach heavily depends on the accuracy of perception modules (e.g., reconstruction, localization, orientation grounding); sensitivity or robustness analysis is lacking.
- The criterion for minimality of the MSS is conceptually emphasized but lacks clear quantitative verification - how to ensure the generated MSS is minimal?
- Details on the prompting and coordination between the Reasoning Agent (RA) and Perception Agent (PA) are insufficient—specifically, how the RA is informed of the available tools or APIs.
- Both benchmarks used (MMSI-Bench and ViewSpatial-Bench) are newly proposed and not yet peer-reviewed; the authors are suggested to conduct evaluations on established benchmarks such as SQA3D[1] or ScanQA[2].
- The paper would benefit from qualitative examples of failure cases.


[1] Ma, Xiaojian, et al. "SQA3D: Situated Question Answering in 3D Scenes." The Eleventh International Conference on Learning Representations.

[2] Azuma, Daichi, et al. "Scanqa: 3d question answering for spatial scene understanding." *proceedings of the IEEE/CVF conference on computer vision and pattern recognition*. 2022.

### Questions
Please refer to the weakness section.

Minor typo:
- line 165, $S$ and $S^\*$  should be $\mathcal{S}$ and $\mathcal{S}^\*$

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
The paper introduces MSSR (Minimal Sufficient Spatial Reasoner), a dual-agent framework aimed at improving 3D spatial reasoning in vision-language models through explicit pursuit of minimal sufficiency. The approach separates perception and reasoning into two collaborating agents: a Perception Agent and a Reasoning Agent. The framework achieves competitive results on MMSI-Bench and ViewSpatial-Bench.

### Strengths
1. The method is well-motivated, as MSSR presents a sufficient yet non-redundant agent framework that enables effective spatial reasoning.

2. MSSR achieves strong performance on two challenging, vision-centric spatial reasoning benchmarks.

3. The paper is well-organized and easy to follow.

### Weaknesses
1. The state-of-the-art claim is unconvincing since Table 1 omits recent strong 3D-LLM baselines (e.g., Video-3D-LLM, VLM-3R) that are directly relevant to spatial reasoning. Including these comparisons is necessary for a fair evaluation.

2. Although the paper includes ablations on the PA and RA, it would be beneficial to further analyze how individual components within the Perception Agent, such as the spatial reasoning modules, foundational scene reconstruction, and global calibration, contribute to overall performance.

3. One potential concern is that MSSR appears to lack a mechanism to verify the correctness of the information provided by PA. Although the RA agent continuously queries PA and filters out irrelevant information, it remains unclear how the framework handles spurious or inaccurate inputs from PA. For future extensions, incorporating a verification module to ensure the reliability of PA’s outputs would be a valuable improvement.

### Questions
1. Does the “only PA” setting in Table 2 refer to using PA to collect all information without any filtering, or does it mean collecting subset information for the first iteration?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes MSSR, a 3D spatial reasoning framework that constructs a minimal sufficient set of spatial facts before answering a question. The model comprises two agents: A perception agent that uses visual programming with tool calls to extract 3D information, and a reasoning agent that plans and iterative requests missing information until the set if sufficient. The model is evaluated on MMSI-Bench and ViewSpatial-Bench, claiming SOTA against prior works.

### Strengths
The model gives good motivation with Figure 1, introducing attention delusion. 

The baselines compared are quite strong, including both closed and open-source models, as well as specialist and other visual programming frameworks like VADAR.

The design of the model allows plug-and-play of powerful models like VGGT, GroundingDINO, SAM2, etc, which can be easily swapped out with better models.

The ablations are quite thorough with the key components (Only PA and Only RA, without iterations, etc.).

### Weaknesses
This paper is generally quite strong in my opinion. My only issues are perhaps more datapoints for the ablation.

For Figure 5, I think the paper would be better benefitted by showing 1) more iterations to see whether more iterations would cause models to have errors in eliminating the number of sets needed for a correct answer and therefore leading to worse performance and 2) on more than just a subset of the MMSI-Bench.

For Table 2, it would be great if the authors can also provide information on ViewSpatial-Bench for a more comprehensive ablation.

However, these are just minor points in general, and I would think the paper’s idea is in fact quite solid. I am therefore leaning towards accept.

### Questions
Please see the above section for my questions regarding the ablations.

### Soundness
3

### Presentation
3

### Contribution
3
