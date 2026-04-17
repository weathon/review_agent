# MIDAS: Multi-Image Dispersion and Semantic Reconstruction for Jailbreaking MLLMs

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Multimodal Large Language Models (MLLMs) have achieved remarkable performance but remain vulnerable to jailbreak attacks that can induce harmful content and undermine their secure deployment. Previous studies have shown that introducing additional inference steps, which disrupt security attention, can make MLLMs more susceptible to being misled into generating malicious content.
However, these methods rely on single-image masking or isolated visual cues, which only modestly extend reasoning paths and thus achieve limited effectiveness, particularly against strongly aligned commercial closed-source models. To address this problem, in this paper, we propose Multi-Image Dispersion and Semantic Reconstruction (MIDAS), a multimodal jailbreak framework that decomposes harmful semantics into risk-bearing subunits, disperses them across multiple visual clues, 
and leverages cross-image reasoning to gradually reconstruct the malicious intent, thereby bypassing existing safety mechanisms. The proposed MIDAS enforces longer and more structured multi-image chained reasoning, substantially increases the model’s reliance on visual cues while delaying the exposure of malicious semantics and significantly reducing the model’s security attention, thereby improving the performance of jailbreak against advanced MLLMs. Extensive experiments across different datasets and MLLMs demonstrate that the proposed MIDAS outperforms state-of-the-art jailbreak attacks for MLLMs and achieves an average attack success rate of 81.46\% across 4 closed-source MLLMs. Our code is available at this [link](https://github.com/Winnie-Lian/MIDAS).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces MIDAS, a jailbreak framework for multimodal large language models built on ‘multi-image dispersion + semantic reconstruction.’ The core idea is to split high-risk semantic units into subfragments and scatter them across multiple ‘game-style visual reasoning’ template images, while on the text side using placeholders and persona-driven reconstruction prompts to guide the model to decode across images step by step and fuse the meaning only at a later stage—thereby lengthening the reasoning path, delaying the exposure of sensitive content, and undermining safety-focused attention mechanisms

### Strengths
1. The combination of “multi-image dispersion” and “persona-driven textual reconstruction” ensures that harmful semantics surface only at the tail end of cross-image reasoning, representing a substantive extension beyond prior “single-image obfuscation/isolated-cue” approaches.

2. The authors conduct extensive experiments on HADES, MM-SafetyBench, and AdvBench, demonstrating the effectiveness of the proposed method.

### Weaknesses
1. How do you balance the game’s intrinsic complexity with jailbreak success? If the visual reasoning puzzle is too simple, the model may sniff out the attacker’s intent; if it’s too complex, the model may fail to grasp and execute the rules, causing the attack to fail.

2. Does relying solely on GPT-5 as the LLM judge bias the evaluation toward GPT-5’s own inductive biases, thereby undermining robustness?

3. The method splits toxic semantics into fragments embedded across multiple images to evade safety checks during inference, but ultimately those fragments must be integrated and reconstructed for the model to produce the toxic output. At that final stage, couldn’t the model still detect the attacker’s intent and block the response, causing the attack to fail?

### Questions
The questions can be found in the Weaknesses section above.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposed a method to jailbreak contemporary MLLMs. Instead of adding semantically related pictures or deceptive prompts,  they leveraged puzzles to hide sensitive words as well as elongate the reasoning chains. Experiments on multiple models demonstrate the effectiveness of such a method.

### Strengths
- Clear writing and storytelling, which is easy to follow.
- Good comparisons on several models demonstrate the effectiveness of such a method.

### Weaknesses
- Lack of novelty, which is my major concern. Previous work has demonstrated that hiding sensitive words in images can enhance the jailbreak success rate, which is why FigStep and MM-SafetyBench focus on early models. Besides, given that FigStep could be defended by GPT-4V, the author also proposed FigStep-Pro, leveraging the model's capability of analyzing each sub-token shown in the images, concatenating the words, and then answering the question. From this perspective, the novelty lies in an extra layer of encryption—using puzzles to encrypt the subwords instead of directly displaying them in the images. However, first-analysis-then-answer methods have also been proposed earlier[1][2], which weakened the innovativeness of the article.

- Following the previous point, the method used is also a little bit technical for me. It combines jailbreaking methods such as role-play and prompt injection, which are also effective on early models demonstrated by the community. Therefore, it is not surprising that adding all these ingredients for a strong jailbreaking prompt could result in a higher ASR.

[1]A Mousetrap: Fooling Large Reasoning Models for Jailbreak with Chain of Iterative Chaos
[2]GPT-4 Is Too Smart To Be Safe: Stealthy Chat with LLMs via Cipher

### Questions
I am also wondering about the performance if the model adopts some basic defensive system prompts, or some to-do tools to remind it after understanding the hidden semantics that now the model should answer the user requests.

### Soundness
4

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
The paper proposes MIDAS (Multi-Image Dispersion and Semantic Reconstruction), a compact multimodal jailbreak method that splits harmful instructions into small semantic fragments, hides those fragments across multiple benign-looking puzzle-style images, and uses a persona-guided, placeholder-filled text prompt to force cross-image reasoning. By doing so the model gradually reconstructs the malicious intent, which delays harmful-token exposure, reduces detection by input filters, and yields much higher attack success rates against strong MLLMs compared to prior single-image or single-step attacks.

### Strengths
1. The paper presents a novel jailbreak formulation that disperses harmful semantics across multiple images and reconstructs them through structured reasoning—a clear departure from prior single-image or heuristic-based attacks. The combination of game-based visual reasoning and persona-driven textual reconstruction is a creative and conceptually new approach to multimodal adversarial prompting.
2. The jailbreak method proposed by MIDAS exploits the model’s own reasoning capabilities. As modern MLLMs grow ever stronger at multi-step and cross-modal reasoning, preventing them from violating safety during extended inference becomes increasingly critical. This paper highlights that gap and argues for defenses that monitor and constrain the process of reasoning (not just surface inputs), since stronger reasoning abilities make reconstruction-based attacks like MIDAS more effective.
3. Experiments are extensive, covering multiple benchmarks and both open- and closed-source MLLMs. Ablation and efficiency analyses convincingly support the claimed improvements.

### Weaknesses
1. Limited theoretical grounding. While MIDAS provides an intuitive probabilistic formulation, the framework remains largely empirical. The paper would benefit from a deeper theoretical analysis of why cross-image dispersion weakens alignment—e.g., formalizing how reasoning-chain extension interacts with attention allocation or safety gating mechanisms.
2. Ambiguity in puzzle design generalization. The game-style visual reasoning templates are described conceptually but not quantitatively analyzed. It remains unclear how performance scales with puzzle type, visual complexity, or linguistic ambiguity. A modular evaluation of these factors could clarify which design elements drive success.
3. Lack of detailed safety or defense discussion. The paper primarily focuses on attack performance but provides limited insight into countermeasures. A more thorough discussion or empirical evaluation of potential defenses (e.g., reasoning trajectory monitoring, dynamic filtering, or representation-based anomaly detection) would strengthen the paper’s contribution to the safety community.

### Questions
Please refer to the weaknesses.

### Soundness
2

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
4

### Summary
This work presents MIDAS (Multi-Image Dispersion and Semantic Reconstruction), a multimodal jailbreak framework that decomposes a malicious query into benign-looking, game-style visual puzzles to be solved via step-wise reasoning. As the model navigates these chains, it semantically reconstructs the original intent and merges it with innocuous text, bypassing safety filters. Experiments show that the resulting longer, multi-step visual reasoning markedly increases attack success, and MIDAS achieves stronger jailbreak performance than prior state-of-the-art methods.

### Strengths
The paper presents good numerical results compared to SOTA methods. It introduces a comprehensive framework and a well-presented approach, including large-scale experiments, experiments to support the claim made, and other analyses, such as the external safety detection mechanism.

### Weaknesses
The idea that a longer chain-of-thought with innocuous early steps eases jailbreaks is not novel (for example, VisCRA, which the authors mention, already supports this idea). That said, the authors introduce a new method and cleverly engineer this idea.
I would reduce the amount of mathematical notation in favor of more detail on method design and implementation. See the details below.

### Questions
- Person-driven reasoning: This component is not well explained, yet it appears important in the ablation table. Please clarify its design and, more generally, the full ablation settings.

- Game-based reasoning templates: How are these templates defined? Are some more effective than others? Do models ever fail to complete the correct answers, and was this considered in the analysis?

- Prompt-based extractor: Could you provide further details about how this extractor is built and used?

- Efficiency: Please add a brief explanation of why the attack is faster.

### Soundness
3

### Presentation
3

### Contribution
3
