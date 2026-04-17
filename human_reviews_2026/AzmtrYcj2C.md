# GraphPrompt: Black-box Jailbreaks via Adversarial Visual Knowledge Graphs

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2, 6

## Abstract
Multimodal Large Language Models (MLLMs) introduce structured visual interaction paradigms into conversational systems, where Visual Knowledge Graphs (VKGs) are emerging as a primary input modality that models can directly parse and manipulate. VKGs significantly enhance models' ordered reasoning and planning capabilities by explicitly encoding semantic topological relationships and task workflows. However, this advancement also introduces new security attack surfaces: when sensitive or malicious intent is decomposed and implicitly encoded within graph topology and visual style cues, and further paired with surface-neutral textual descriptions, MLLMs may bypass traditional text-based safety filters and follow covert parse-then-execute pathways, exhibiting jailbreak behaviors such as instruction hiding and ambiguity amplification. The safety implications of such structured visual inputs for MLLMs nevertheless remain largely unexplored. To systematically assess this risk, we introduce GraphPrompt, a black-box jailbreak evaluation framework that exploits this attack surface through a three-layer obfuscation pipeline: (1) role-play rewriting masks harmful queries as benign tasks; (2) knowledge graph encoding decomposes procedures into entity–relation structures; and (3) visual rendering transforms graphs into adversarial VKG images. This framework automatically generates high-quality adversarial datasets while providing standardized evaluation. Systematic experiments on six state-of-the-art MLLMs reveal alarming safety risks: GraphPrompt achieves a 94\% average attack success rate with only 1.25 attempts per query on average. Ablation studies identify graph complexity and image resolution as first-order attack factors, while visual styling has minimal impact. Layer-wise analysis demonstrates that VKG inputs effectively suppress activation in safety-critical layers, providing mechanistic evidence for their jailbreak efficacy. Overall, our work establishes structured visual inputs as an under-explored attack surface and offers a reproducible framework for developing structure-aware defenses.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces GraphPrompt, a typographic jailbreaking attack on Multimodal Large Language Models (MLLMs) that utilizes a Visual Knowledge Graph (VKG). To break the alignment of the MLLM, GraphPrompt embeds a jailbreak prompt into a VKG. It begins by constructing a Knowledge Graph (KG) using a Large Language Model (LLM), which is then transformed into a VKG using Mermaid. The VKG is subsequently input into the target MLLM through the visual channel alongside a benign textual prompt. The effectiveness of the jailbreaking is evaluated using an LLM-based judge model. Experiments demonstrate that GraphPrompt successfully jailbreaks various MLLMs with a high Attacking Success Rate (ASR), underscoring the vulnerabilities of MLLMs in processing VKGs.

### Strengths
1. This paper is well organized and easy to follow.
2. Embedding malicious intention into VKG appears to be novel.
3. Experiments show GraphPrompt is promising for jailbreaking SOTA MLLM.

### Weaknesses
1. MLLMs are known to be vulnerable to typographic jailbreaking attacks, where malicious textual questions are converted into images. This approach takes advantage of the model's image understanding capabilities to circumvent textual filters, effectively "breaking the safety alignment." Consequently, the novelty of transforming harmful textual knowledge graphs into typographic images (VKGs) is somewhat limited. While GraphPrompt shows better performance than FigStep, the underlying reasons for this difference have not been thoroughly examined.

2. There is limited discussion on defense methods against GraphPrompt. Although the Related Work section includes a paragraph on defenses against jailbreaking, there are no experiments investigating how basic OCR-based filters or input-moderation defenses (e.g., converting the images into a textual description as a supplement to the benign textual input) can reduce the effectiveness of GraphPrompt.

### Questions
Please refer to the Weaknesses part.

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
3

### Summary
This paper introduces GraphPrompt, a novel black-box jailbreaking attack framework for Multimodal Large Language Models (MLLMs). The core idea is to encode harmful intent not in plain text, but within the topological structure and visual cues of Visual Knowledge Graphs (VKGs). The attack pairs these adversarial VKG images with a benign-looking textual prompt (e.g., "analyze the tasks in this graph"), which tricks MLLMs into bypassing text-based safety filters and executing the embedded harmful instruction through a "covert parsing-execution pathway".

### Strengths
1. **High Efficacy and Realistic Threat Model**: The most significant strength is the attack's high effectiveness. Achieving a 94.0% average ASR—and rates as high as 100% on Gemini and 98% on GPT-5 and Qwen2.5-VL—is impressive. This is accomplished under a strict black-box assumption (no access to weights or gradients), making it a practical and realistic threat.

2. **Data Generation Pipeline**: The framework's ability to "automatically construct high-quality adversarial sample datasets" is a useful contribution. This provides a scalable method for red-teaming MLLMs against this new structured-visual threat dimension.

3. **Ablation Studies**: The ablation studies provide a clear picture of why the attack works. The findings that topology and resolution are the dominant factors, while visual elements like color and background are "second-order", are key takeaways that can inform future defense strategies.

### Weaknesses
1. **Limited and Potentially Insufficient Evaluation Dataset**: The primary weakness is the reliance on the SafeBench-Tiny dataset, which contains only 50 harmful queries. While the authors justify this for "reproducibility and experimental control", claiming a 94-100% ASR based on such a small sample size is a major overstatement. The high success rates could be an artifact of these specific 50 queries, and the results may not generalize to a more diverse and larger-scale benchmark.

2. **Questionable Novelty Compared to Prior Work (FigStep)**: The paper claims to be the first to leverage VKGs , but its novelty relative to existing "text-in-image" attacks like FigStep  is not sufficiently established. FigStep also works by decomposing instructions into steps and rendering them in an image. While the mechanism is interesting, the motivation for jailbreaking MLLMs is a very crowded research area. The paper does not sufficiently motivate why this specific vector is substantially different or more dangerous than the "abundant llm jailbreak attacks" that already exist such as FigStep.

3. **Lack of Experimental Defense Evaluation**: The paper discusses potential defenses in the conclusion, such as "structure-aware safety filtering" and "uncertainty-aware refusal". However, it presents no experiments to evaluate the efficacy of these or any other defenses. An attack paper is made much stronger by demonstrating how the uncovered vulnerability might be patched.

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces GraphPrompt, a novel black-box jailbreaking framework that exploits the structural and semantic properties of Visual Knowledge Graphs (VKGs) to bypass safety alignments in Multimodal Large Language Models (MLLMs).

### Strengths
1. Novelty: This is the first work to systematically explore the security risks posed by VKGs in MLLMs, leveraging structural and semantic paradoxes for adversarial attacks.
2. Clarity and Coherence: The paper is well-structured and written with reasonable experimental design.
3. Practical Relevance: The attack framework is practical and poses a realistic threat to deployed MLLMs.

### Weaknesses
Methodological Detail Lacked: The paper lacks sufficient detail in key parts of the method. For example:
1.How exactly is semantic decomposition and topology-borne encoding performed?
2.How are visual encoding parameters (e.g., color, layout) adjusted during optimization?
3.How are graph size parameters (|V|, |E|) controlled or modified?

Dataset Scale: The use of SafeBench-Tiny (only 50 queries) limits the statistical reliability and generalizability of the results.

VKG Generation Process: It is unclear which model or tool is used to generate VKGs from Mermaid code, and how the quality or diversity of generated graphs is ensured.

Judge Model Validation: Although manual spot-checking is mentioned, there is no quantitative evaluation of the judge model’s accuracy or consistency.

Inconsistent Model Usage: Not all six models from Table 1 are included in the following experiments, which limits the completeness of the analysis.

### Questions
1. How was the semantic decomposition step implemented? Was it rule-based or model-based?
2. What was the rationale behind the ongoing contest scenario in the user prompt? How might this influence model behavior?
3. Why were only some of the six models used in the ablation studies?
4. Was the judge model’s performance evaluated? If so, what were the results?
5. See Weaknesses for more questions.

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
3

### Summary
This paper introduces GraphPrompt, a novel black-box jailbreaking framework that exploits Visual Knowledge Graphs (VKG) to bypass safety alignment in MLLMs. By embedding harmful intents into graph topologies and pairing them with benign textual prompts, GraphPrompt induces a "parse-then-execute" pathway that evades text-based filters. The work underscores critical vulnerabilities in MLLMs’ cross-modal reasoning and proposes future defenses centered on structure-aware safety mechanisms.

### Strengths
- Compelling experimental results: The paper achieves exceptionally high ASR under strict black-box settings, significantly outperforming strong baselines like MM-SafetyBench and FigStep.  

- Rigorous ablation analysis: The systematic ablation studies (e.g., varying node count, resolution, color schemes) clearly illustrate how graph topology and visual encoding affect VKG-driven attack success

- Insights for future defense insights: The success of VKG attack inspires future efforts toward VKG-like jailbreaks.

### Weaknesses
- Mechanistic explanation of attack efficacy: While the paper empirically demonstrates high ASR, it lacks a deeper theoretical or mechanistic explanation of why VKG so effectively bypasses safety alignment. For instance, how does graph parsing alter the model’s internal reasoning trajectory? 

- Cost analysis of black-box optimization: The feedback loop involves iterative querying, but the computational cost of generating adversarial VKGs is not quantified. 

- Reference Issue: The manuscript mentions "Appendix 4" in Section 4.1, but no appendices are included in the submission.

### Questions
Please see the weakness part.

### Soundness
3

### Presentation
3

### Contribution
3
