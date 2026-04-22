# MAPA: Multi-turn Adaptive Prompting Attack On Large Vision-Language Models

- Avg Score: 2.50
- Decision: Reject
- Scores: 2, 2, 2, 4

## Abstract
Multi-turn jailbreak attacks are effective against text-only *large language models* (LLMs) by gradually introducing malicious content across turns. When extended to *large vision-language models* (LVLMs), we find that naively adding visual inputs can cause existing multi-turn jailbreaks to be easily defended. For example, overly malicious visual input will easily trigger the defense mechanism of safety-aligned LVLMs, making the response more conservative. To address this, we propose ***MAPA***: a **m**ulti-turn **a**daptive **p**rompting **a**ttack that 1) *at each turn*, alternates text-vision attack actions to elicit the most malicious response; and 2) *across turns*, adjusts the attack trajectory through iterative back-and-forth refinement to gradually amplify response maliciousness. This two-level design enables ***MAPA*** to consistently outperform state-of-the-art methods, improving attack success rates by 5-13% on HarmBench and JailbreakBench against LLaVA-v1.6-Mistral-7B, Qwen2.5-VL-7B-Instruct, and Llama-3.2-Vision-11B-Instruct. Our code is available at: https://anonymous.4open.science/r/MAPA-jailbreak.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a multi-turn jailbreak method for LVLMs. The attack is implemented through an alternative update on text-image, as well as the jailbreak trajectory across turns. The authors conduct experiments on selected subsets of two text-only datasets. From the experimental results, the attack method shows improvements on three LVLMs: LLaVA-v1.6-Mistral-7B, Qwen2.5-VL-7B, and Llama-3.2-Vision-11B-Instruct.

### Strengths
-	The studied problem is interesting and novel. Moving beyond the single-turn attack to the multi-turn jailbreak attack could be a promising research problem.
-	The authors show the effectiveness of the proposed method on three LVLMs.

### Weaknesses
-	I’m curious why the authors do not choose some multimodal benchmarks for testing, such as Jailbreak-V.
-	Even if the authors adopt the text-only benchmarks, some more single-turn baselines are also required to compare.
-	The evaluation dataset scale is greatly limited.
-	How should we judge that in the current turn, the image or the text impedes the jailbreak attack?
-	The computational cost of the proposed method is not discussed. This may contribute to a negative disadvantage of this method.

Minor:

- The first two sentences in the abstract are confusing in logic. I don’t quite understand where there is a ‘however’. It feels like the inclusion of vision inputs causes easy defense for both single and multi-turn attacks, but not particularly for multi-turn ones.
-	Figure 1 is very confusing. It is hard to grasp the idea since each column and row is not well structured.
-	Line 101 may not be right.  First, perturbation methods may not always incur great computational cost. Second, the logic is confusing. The overhead and model details unavailable are two different things.

### Questions
See weaknesses.

Overall, though the multi-turn jailbreak in LVLMs may be promising, the implementation in this work, as well as the writing, is not satisfactory. I believe the work can be greatly improved after further polishing and revision.

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
5

### Summary
This paper proposed a multi-turn jailbreak method to attack MLLMs. In their framework, the attacker has three options in each turn and utilizes the SEM metric to make the decision. Experiments are conducted on LLaVA-mistral, Qwen2.5-VL, as well as Llama-3.2-Vision, demonstrating its better performance than some of the previous methods.

### Strengths
- Clear writing, making the attack pipeline easy to follow.
- The use of semantic correlation for action choice is interesting.

### Weaknesses
- Lack of novelty. Using another LLM to attack a black-box model has been proposed[1]. It will be better if the script proves that the design of attack actions, together with semantic correlation, is better than the use of another LLM for planning. Besides, the novelty issue also lies in the attack actions—adding an SD-generated image with detoxified text prompts is proven to be effective in previous literature[2][3], and that is why I find this part more like a trick instead of a novel innovation.

- Limited comparisons. Since the attack is black-box, it is reasonable to evaluate such an attack on closed-sourced models, at least GPT-4o, GPT-5-Chat, Gemini-pro, or Cloude. Experiments on Llava-1.6-mistral are not convincing, for it does not undergo a multi-model safety alignment, and simply using samples from MM-safetybench or Figstep could get a high ASR.

- Lack of Defense. At least some defense should be evaluated to fully prove the effectiveness of such a method.
- Minor problems: some typos. For example, "a Chain-of-Thought manner (Wei et al., 2022) manner"

[1] Jailbreaking Black Box Large Language Models in Twenty Queries.
[2] MM-SafetyBench: A Benchmark for Safety Evaluation of Multimodal Large Language Models
[3] MLLM-Protector: Ensuring MLLM's Safety without Hurting Performance
[4] Do We Really Need Curated Malicious Data for Safety Alignment in Multi-modal Large Language Models?

### Questions
- Is there any specific design for the red-teaming model? How does the model generate its next-turn prompt?
- What is the performance on cutting-edge commercial models?
- Does the attack work on models with defenses, such as safety-related system prompts?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposed a multi-turn jailbreak method on VLMs, including altering text-vision attack actions and back-and-forth refinement. The paper evaluates on three open-source models using some samples from HarmBench and JailbreakBench.

### Strengths
1. The paper is easy to follow.
2. The paper proposed a multi-turn jailbreak method and it is easy to implement.

### Weaknesses
1. The method is tested only on a selected set of samples, and the limited number of experiments is insufficient to demonstrate the effectiveness of the approach. For example, evaluating on datasets like AdvBench and RedTeam-2K could provide more comprehensive validation.
2. The baseline methods are not strong enough. For multi-turn attacks, BAP attack [1] and IDEATOR [2] are existing strong jailbreak methods. The authors should include comparisons with these methods to demonstrate the effectiveness of MAPA.
3. MAPA is only evaluated on open-source models, lacking attacks on commercial models (e.g., ChatGPT, Gemini, or DeepSeek).
4. The authors should include additional metrics, such as using the GPT-judge and Perspective API, to provide a more comprehensive assessment.

[1] Jailbreak Vision Language Models via Bi-Modal Adversarial Prompt. IEEE TIFS, 2025.

[2] IDEATOR: Jailbreaking and Benchmarking Large Vision-Language Models Using Themselves. ICCV, 2025.

### Questions
1. The authors could provide a detailed greedy search process, especially in cases where the response contains many sensitive words but does not meet the jailbreak target. Would such a response have a high semantic correlation? If so, can the method avoid incorrect attack trajectories in such cases?
2. See Weaknesses.

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
This paper proposes MAPA, a framework for generating adaptive multi-turn jailbreaks that can continuously adjust to model responses in order to bypass safety mechanisms in large vision-language models (LVLMs). The experimental results demonstrate its effectiveness. However, there are still some questions that need to be addressed.

### Strengths
1. The paper is well-written, and the figures and tables are clearly presented.
2. The research question and ideas are interesting. Investigating how to jailbreak multi-turn LVLMs is a valuable direction, and the proposed method may make a meaningful contribution to this field.

### Weaknesses
1. The contribution summary should be presented more clearly, for example, by explicitly enumerating the key contributions.
2. The method section lacks clarity. Some key concepts are not explained. For instance, what do "connected" and "unconnected" mean? What is the difference between a connected and an unconnected prompt?

### Questions
1. In Equation (2), the authors write $J(R_j, t) = 0$. What does $R_j$ refer to? Should it be $r_j$? I assume this is a typographical error.

2. In line 161, the authors state: “By combining the prompt candidates $\{_{uc}Q^{T},  _cQ^V,  _cQ^V\}$”. Is this a typo? I assume the second item should be $_cQ^T$ rather than $_cQ^V$.

3. In line 189, the authors state: “It first identifies malicious concepts in the unconnected text prompt $_{uc}Q^T$ and then leverages these concepts to generate corresponding malicious images.” However, this sentence appears under the **Design Connector LLM** heading. How could the LLM generate images in this stage? Furthermore, in line 160, the authors mention generating images via the Stable Diffusion model. I assume this is also a typographical error.

4. As shown in Figure 2 and line 210, the authors use cosine similarity between responses and the harmful question to select the optimal response for the next-turn iteration, i.e., the response with the highest semantic correlation to the question. However, it is unclear why this works. Why would semantic similarity between two sentences indicate that the response actually answers the question? In other words, does selecting a response based on semantic correlation guarantee that it is the optimal one?

5. Regarding efficiency, the authors compare the number of `turns` required by different methods. While the proposed method appears efficient in terms of turns, the operations performed in each turn vary across methods. Therefore, comparing solely based on the number of turns may be misleading. Could the authors provide a more detailed and fair comparison, considering both the complexity and cost of operations per turn?

6. Will the authors consider evaluating their method on some closed-source models, such as GPT or Gemini?

I am happy to increase my score if the questions are addressed.

### Soundness
3

### Presentation
3

### Contribution
3
