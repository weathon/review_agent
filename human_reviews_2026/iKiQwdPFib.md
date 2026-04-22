# TISR: Breaking Vision–Language Models via Text–Image Segmentation and Reassembly

- Avg Score: 4.67
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 6

## Abstract
Large Vision-Language Models (LVLMs) have demonstrated remarkable multimodal understanding capabilities, yet their inherent security vulnerabilities remain a pressing concern. Prior studies have indicated that enhancing chain-of-thought reasoning can inadvertently facilitate jailbreaks; however, existing attempts often rely on image masking, which only modestly extends reasoning depth and performs poorly against advanced commercial closed-source models. To bridge this gap, we propose an effective jailbreak method, Text–Image Segmentation and Reassembly (TISR). TISR disperses harmful semantic fragments across the visual modality while employing text-based role-playing to guide their reassembly. This strategic organization conceals surface-level toxicity while preserving malicious intent, thereby extending LVLM reasoning chains and enabling successful jailbreaks. Our experiments show that LVLMs, leveraging their visual understanding and reasoning abilities, can reconstruct complete harmful semantics while failing to uphold safety constraints. Extensive evaluations across massive LVLMs demonstrate that the proposed TISR achieves state-of-the-art attack performance, surpassing prior multi-modal jailbreak methods. Specifically, it attains average Attack Success Rates (ASR) exceeding 90\% across three benchmark datasets on four leading commercial LVLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies jailbreak attack for Large Vision-Language Models. It proposes Text-Image Segmentation and Reassembly (TISR), which leverages a visual encoding-textual decoding mechanism to enhance attack of effectiveness. A novel toxicity-minimization and reasoning extension algorithm is proposed. Experiments demonstrate the effectiveness of TISR-Attack across multiple datasets.

### Strengths
- The paper studies an important tpoic of jaibreak attack. With the increasing usage of LVLMs, such security concerns have drawn increasing attention.
- The authors propose to split text at character and increase the number of image partitions, followed by shuffling instructions. These actions overcome the ineffectiveness of previous simple split-and-shuffle strategies in elongating LVLM reasoning chains.
- An encoder-reassemble process in the text prompts is introduced to guide LVLMs in reconstructing the intended semantics. The divided labor between visual-side encoding and text-side decoding constructs a jailbreak paradigm that appears low toxicity on the surface and highly adversarial in combination. 
- The authors conduct experiments on several LVLMs including commercial ones. Across different jailbreak methods, the proposed TISR achieve best performances.

### Weaknesses
- This paper follows previous works on split-and-shuffle. It applies finer-grained segmentation. The novelty from the attack framework perspective is not significant.
- I am not quite convinced by this line of attack with split-and-shuffle. The shuffled images look quite unnatural, and easy to be detected based on their patterns. Also, the text break leads to non-meaningful characters. The authors could comment on this if the semantic stealthy is important.

### Questions
- The proposed method splits image into finer segmentations. How is this related to image resolutions? Will it be easier to break semantic meanings if the image contains small objects?

### Soundness
3

### Presentation
3

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
This paper introduces a new black-box jailbreak method for Large Vision–Language Models (LVLMs).
The key idea is that by increasing the reasoning (or thinking) length—forcing the model to reconstruct scattered semantics from mixed text-image fragments—one can bypass safety alignment and induce harmful outputs.
TISR encodes harmful semantics into small, fine-grained text and image fragments, disperses them spatially, and then reassembles them.
Text plays the decoding role while images handle encoding.
This division makes the input appear harmless at the surface level while maintaining deep semantic coherence, thus allowing models to “think” their way back to the harmful meaning.
Experiments on several top commercial LVLMs and datasets show extremely high attack success rates (ASR)—up to 100% in some cases—and strong generality under both normal and defended settings.

### Strengths
- Very effective — achieves near-perfect ASR (≈ 90–100%) across major LVLMs in black-box settings.
- Evaluation breadth: multiple datasets, comparison with strong baselines, and robustness tests under defense methods (HySAC, UniGuard).
- Ablations are comprehensive.

### Weaknesses
- TISR mainly combines and extends known multimodal adversarial concepts (segmentation, reassembly, role-play) rather than introducing a fundamentally new theoretical mechanism.
- While effective as a red-team tool, the paper focuses narrowly on attack capability, with less discussion of defense implications or mitigation strategies.

### Questions
- The empirical trend between thinking length and ASR is central to the paper’s claim—showing a quantitative plot of reasoning length (e.g., token depth) versus ASR would strongly substantiate this.
- The method assumes LVLMs have consistent reconstruction ability—how does this generalize to open-source or smaller models?
- Can you please suggest some techniques to mitigate this vulnerability, either in training or inference?

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
Aiming at the jailbreak security problem faced by multimodal large models, this paper proposes Text-image Segmentation and Reassembly, which disperses harmful semantic fragments into visual modalities. Meanwhile, text-based role-playing is utilized to guide the recombination of large models, hiding the toxicity on the surface. However, it retained malicious intentions, sang the reasoning chain of VLM and successfully achieved a prison break. It has demonstrated the most advanced attack capabilities on multiple data points and models.

### Strengths
1.The motivation of this paper is clear. It proposes TISR to address the jailbreak security issue faced by multimodal large models. The existing multimodal jailbreak methods can only slightly increase the depth of model inference. This paper adopts a triple strategy of semantic splitting, modal dispersion, and role reconstruction guidance, and utilizes the cross-modal inference capability of VLM to bypass the security alignment mechanism.

2.This paper is well-experimental and has been tested on multiple datasets and models. The average ASR exceeds 90%, and in some cases, it even reaches 100%. This paper clearly explains the relevant data descriptions and experimental Settings, and compares four advanced VLM attack methods.

3.This paper conducted thorough ablation experiments, and different experimental Settings and results are presented in Table 4. This article also provides prompts for evaluation and identity construction, as well as successful cases of jailbreak, which is helpful for a better understanding and expansion of related work.

### Weaknesses
1.The test and evaluation models selected for the experiment are all closed-source models. Although this article claims to focus on black-box testing, providing experimental results from some open-source models would be more convincing. And when GPT-5-MINI is selected as the evaluation model, will there be any misjudgment when evaluating the GPT-5 model, and is it necessary to compare with other evaluation methods?

2.This article presents the high attack rate and successful attack cases of TISR. However, what I am more interested in are the few failed cases. Through some failed cases, in-depth analysis can be conducted to make the work of this article more comprehensive.

3.The core viewpoint of this article lies in extending the length of the VLM inference chain through TISR and successfully achieving jailbreaking, demonstrating an effective example to illustrate this point. However, is it necessary to quantitatively analyze the specific relationship between the length of the inference chain and the success rate of attacks to better confirm this?

### Questions
1.Provide open-source test and evaluation model experimental results.

2.Conduct an in-depth analysis of failed cases.

3.Provide quantitative analysis of the length of the inference chain and the success rate of attacks

### Soundness
3

### Presentation
3

### Contribution
3
