# Visual Multi-Agent System: Mitigating Hallucination Snowballing via Visual Flow

- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
Multi-Agent System (MAS) powered by Visual Language Models (VLMs) enables challenging tasks but suffers from a novel failure term, multi-agent visual hallucination snowballing, where hallucinations are seeded in a single agent and amplified by following ones due to the over-reliance on textual flow to relay visual information. Through turn-, layer-, and token-wise attention analyses, we provide detailed insights into the essence of hallucination snowballing regarding the reduction of visual attention allocation. It leads us to identify a subset of vision tokens with a unimodal attention peak in middle layers that best preserve visual evidence but gradually diminish in deeper agent turns, resulting in the visual hallucination snowballing in MAS. Thus, we propose ViF, a lightweight, model-agnostic mitigation paradigm that relays inter-agent messages with Visual Flow powered by the selected visual relay tokens and applies attention reallocation to amplify this pattern. The experiment results demonstrate that our method markedly reduces hallucination snowballing, consistently improving the performance across eight benchmarks based on four common MAS structures and ten base models. The source code is publicly available at: https://github.com/YU-deep/ViF.git.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose ViF, a lightweight plug-and-play method to reduce visual hallucination snowballing in Visual Multi-Agent Systems built on Vision-Language Models. They observe that as multi-agent interactions progress, agents increasingly rely on textual context while losing attention to visual evidence, causing errors to compound. To address this, ViF introduces two mechanisms: Visual Information Relay, which passes key visual tokens between agents to preserve visual grounding, and Attention Reallocation, which restores attention to visual features in middle and deep layers. Since modern VLMs use Flash-Attention, which hides attention maps, ViF uses the key norm to identify important vision tokens efficiently. Experiments on eight benchmarks and ten base models show consistent gains of about 2–3.5% in both general and hallucination-focused evaluations. Overall, ViF is an effective and easily integrable framework that improves visual consistency and reduces hallucination propagation without retraining underlying models.

### Strengths
1. The proposed ViF method is lightweight and plug-and-play, requiring no model retraining or architecture modification, making it highly practical for integration into existing vision-language frameworks.

2. The paper provides clear empirical evidence through attention analysis that supports the identified problem and motivates the design of ViF, showing strong interpretability and reasoning behind the approach.

3. Extensive experiments across eight benchmarks and ten base models demonstrate the method’s robustness and generality, with consistent performance gains across both general and hallucination-specific evaluations.

4. The work contributes to the reliability and stability of multi-agent VLM systems, an emerging but underexplored research direction, making it a meaningful step toward safer and more grounded visual reasoning in collaborative AI agents.

### Weaknesses
1. Although the paper claims ViF reduces hallucination propagation, it does not include qualitative or human evaluations to confirm whether the mitigated errors are genuinely perceptual or semantic, limiting the interpretability of improvements.

2. The key-norm heuristic used as a substitute for attention may not perfectly capture true token importance, and the paper does not analyze failure cases or potential mismatches between key-norm values and actual visual relevance.

3. The attention reallocation strategy is presented as effective, but the specific reallocation rule (e.g., scaling factors or redistribution formula) is not fully described, reducing reproducibility.

4. The framework assumes that hallucination arises primarily from attention degradation, but other causes of hallucination are not addressed, which may limit the generality of the approach.

5. The computational overhead and latency introduced by token relaying between agents are not quantified, which could be important for real-time or large-scale multi-agent applications.

### Questions
1. How were the attention reallocation parameters determined? Are they fixed hyperparameters or adaptively tuned per model and layer?

2. When using key-norm as a proxy for attention importance, did you compare it empirically with models where attention maps are accessible to verify its reliability?

3. Could you provide qualitative examples showing how ViF changes an agent’s reasoning trajectory or visual focus compared to the baseline?

4. Since ViF passes selected vision tokens between agents, how do you prevent error propagation if the selected tokens themselves are noisy or misleading?

5. Have you tested whether combining ViF with other hallucination mitigation techniques (e.g., grounding via retrieval or contrastive filtering) yields additive improvements?

6. What is the computational cost (both in FLOPs and latency) introduced by ViF during inference, and is it negligible compared to the base MAS pipeline?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The critical failure of multi-agent systems lies in the compression and selective amplification of visual evidence by the "text stream", causing early hallucinations to spread along the information chain;ViF complements and stably transfers visual evidence through the "visual stream", and, in conjunction with attention redistribution, maintains visual dominance in the middle layer, significantly reducing the snowball effect in multi-round scenarios.

### Strengths
- **The problem definition is specific and MAS-oriented:** The paper identifies "visual hallucination snowballing" as a critical failure in multi-agent scenarios and makes a diagnosis from the triple perspectives of layer-round-token; this focus differs from the paradigm of traditional single-agent LVLM hallucination research. 

- **The design concept of the visual stream is novel:** By combining "visual relay token + attention redistribution", a non-textual channel is introduced at the cross-agent communication level, aiming to stably retain and amplify visual evidence, and conceptually has a concise and scalable appeal. 

- **Diagnosis and Ablation Adequacy:** Quantitative trends of inter-layer and round attention, hierarchical deletion of shallow/middle/deep layers, and the proportion of key subsets changing with rounds, with the overall analysis chain being complete; ablation covers the proportion of visual relay tokens and the contributions of each layer to attention redistribution, supporting the claim.

- **Compatibility Emphasis:** Positioned as "Plug and Play", it provides Key-Norm alternatives and buffer strategies on models with FlashAttention to lower the threshold for reproducibility.

### Weaknesses
1. **Computation and latency overhead not reported:** Transmitting visual relay tokens across agents increases sequence length and attention computation, and attention redistribution also introduces additional operators. It is recommended to report the end-to-end throughput/latency curves (varying with agent rounds, image resolution, model scale, and etc.), and compare them in parallel with the baseline of text-only streams.

2. **The coverage of the task domain is limited:** The main evaluation is on the understanding and question answering of static images. It is recommended to verify the generalization of ViF on agentic tasks with external tool usage, especially in multi-round planning and interaction scenarios, to examine whether visual relaying is more deeply coupled with the workflow. 

3. **Reliability and External Validity of the HS Metric:** HS relies on the judge model to evaluate "hallucination severity" and "propagation depth". It is recommended to conduct  reviewer consistency and model sensitivity analysis , and perform external validity testing on the  manually annotated  snowball diagnostic set; at the same time, disclose the version and prompt of the judge model. 

4. **Comparison with Structured Visual Features:** Besides "visual token direct transfer", schemes that transfering compressed/pooled spatial feature maps or object-level memory can be compared with ViF in terms of performance-overhead-robustness differences.

5. **Analysis of Strong Adversarial Scenarios:** When the visual judgment of the first agent is severely deviated or deliberately injected with errors, can ViF "correct rather than follow"? It is recommended to construct stronger adversarial distributions to evaluatethe cross-round correction rate of early errorsandthe anti-interference ability of the visual stream.

### Questions
1. **Cross-domain generalization:** Under video question answering, multi-image reasoning, and tool-enhanced workflows, is the benefit of ViF stable?

2. The proposed ViF approach seems to mainly focus on the impact of visual tokens that have undergone an **attention sink**. Previous studies have mainly focused on the level of single agent and advocated that alleviating the phenomenon of attention sink can improve the performance of the model.[1, 2] However, some researchers, like the author, have proposed using attention sink to enhance the performance of the model. Recent Qwen3-Next series model replaces standard attention with the combination of Gated DeltaNet and Gated Attention, which enables efficient context modeling for ultra-long context length while alleviates the attention sink. If the ViF method is used on VLMs developed based on such models in the future, can similar gains still be obtained?

ref:

[1] Huang Q, Dong X, Zhang P, et al. Opera: Alleviating hallucination in multi-modal large language models via over-trust penalty and retrospection-allocation[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 13418-13427.

[2] Chen Z, Li M, Chen Z, et al. Advancing general multimodal capability of vision-language models with pyramid-descent visual position encoding[J]. arXiv preprint arXiv:2501.10967, 2025.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a new task setting for VLM multi-agent system (MAS), hallucination snowballing. Hallucinated textual response from early agent turns may get passed down to subsequent turns, causing agents to receive spurious textual input and thus inducing further errors and hallucinations. Through extensive analysis of attention distribution over different input token segments, agent turns and transformer layers, the authors discovered that unimodal visual tokens in middle layers produced the most accuracy change when these tokens are removed. As such, unimodal tokens would potentially be most impactful for visual text interaction and is selected as visual relays tokens to carry visual information to subsequent turns. With this finding, the authors propose ViF, which concatenates these relay tokens with instruction and visual tokens to enhance downstream agent inference. Additionally, an attention reallocation module is designed to shift inactive attention to other regions and promote formation of unimodal token patterns. ViF method is extensively evaluated across 8 benchmarks on 6 different base models. Experimental results surpassed previous methods and proved the effectiveness of ViF.

### Strengths
The authors introduce a new task setting for MAS hallucination and create a multi-turn evaluation metric which could be beneficial for future researchers. 

This paper presents a rigorous evaluation of ViF, testing its generality across six base models of varying sizes and architectures. The authors also provide extensive ablation studies and parameter analysis to validate the method and demonstrate the impact of its components and parameter choices.

The authors also work to improve the method's speed and efficiency by maintaining compatibility with Flash Attention modules. Instead of using attention scores, alternative token selection methods based on key norms and value norms are tested and presented in Table 7 and Figure 8. These experiments support the validity of using key norms as a proxy for attention scores, thereby maintaining compatibility with Flash Attention.

### Weaknesses
While the paper is well-written with a clear logical flow, it omits crucial experimental details regarding the MAS evaluation. Specifically, it does not explain how conventional single VLM evaluation benchmarks are adapted for the MAS setting. This omission may hinder readers unfamiliar with multi-agent setups from fully understanding the significance of evaluation results with respect to snowballing hallucination. The paper would be strengthened by providing more prerequisite information on MAS evaluation protocols.

### Questions
1. Table 2 shows experimental results for 4 different MAS structures. How are the conventional single VLM evaluation datasets such as POPE tested in multi-agent settings? Could you please provide an example of how a Yes/No question with image is processed in multi-turn agent turns? 
2. Could you please provide more experimental setup details for comparison results in table 5? How are other SOTA methods evaluated in MAS settings following circular MAS structure? 
3. There is a potential inconsistency in classifying ViF as plug-and-play. The method applies a trained transformer block to embed visual relay tokens. If this transformer block must be retained for each new base model (particularly one with a novel visual encoder), it would constitute a model-specific training phase, which is contrary to the concept of a true plug-and-play component.
4. The paper proposes a new HS metric for snowballing hallucination. The severity of hallucination score, h, is said to be derived from a judge. Could you please provide more information about this judge and how to obtain h from the 8 benchmarks? 
5. As shown in table 1, dropping rise and fall tokens in middle layers also contributed significant accuracy change, indicating that they are also important (though not the most important) in model inferencing. Would it be possible to consider combining multiple types of visual tokens as relay tokens? How would the model perform when selecting a portion of rise, fall and unimodal tokens as relay tokens?

### Soundness
3

### Presentation
3

### Contribution
3
