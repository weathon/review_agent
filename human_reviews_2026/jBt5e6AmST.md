# Evading Safety Alignment in Open-source LLM Deployment by Embedding Semantic Shift

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 2, 4

## Abstract
Large Language Models (LLMs) are increasingly distributed and deployed through public platforms such as Hugging Face. While these platforms provide basic security scanning, they often overlook subtle manipulations within the embedding layer that could lead to harmful behaviors during inference. We observed a Semantic Shift phenomenon in embedding perturbations, exposing potential security threats. Based on further analysis of this phenomenon, we propose Search-based Embedding Poisoning (SEP), a practical, systematic, and model-agnostic framework that bypasses model safety alignment by introducing carefully chosen perturbations into embeddings associated with high-risk tokens. SEP employs heuristic search to identify subtle perturbations based on linear semantic shift, dynamically adjusting search process according to model outputs until evading safety alignment. SEP achieves an average attack success rate of 96.43\% while preserving benign functionality and evading conventional detection mechanisms. Our findings highlight an overlooked attack surface in the deployment pipeline and call for embedding-level integrity checks as the core of future LLM defense strategies.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposed a new jailbreaking attack, which perturbs the embedding space to induce harmful answers. The design is based on the findings that adding different scales of perturbation on some dimentions of the embedding space could induce a variety of unexpected effects, including deviation, harmful outputs, or totally nonsense glithes. A search algorithm then find the proper perturbation scale, dimension, and location to induce the harmful outputs on the original malicious question. The evaluation shows improved attack success rate compared with baseline attacks.

### Strengths
The finding is interesting: a simple "blind" perturbation on embedding space can induce different unexpected model behaviors. Treating the embedding layer as a deployment-time attack vector (no weight edits, no prompt artifacts) is novel in the attack context. Though it requires the white-box access (can add perturbation to embedding), it fits in the scenarios that the user is using the public open-source models, or a developer uses an in-development model for malicious purposes.

It has shown promising results. The reported outcomes are strong at face value: high success rates across multiple prompts/models and signs that benign capabilities are largely retained.

### Weaknesses
The findings are largely empirical without theoretical arguments. For instance, description of finding 2 and 3 csays "similar" and "likely", without a clear definition. Are there quantitative results about how likely the findings are valid on a specific case? It looks intuitive that if adding large perturbation, the model outputs will be nonsense glitches, but the existence of that "harmful but nott deviated" region is not guaranteed. I hope the author could provide more empirical numbers or theoretical analysis.

Some of the design and evaluation details are not clear in the main paper. A key component for making the attack pipeline automated is the "multi-stage classifier". However, the classifier relies on a set of existing tools, including the keyword-matching refusal detect and model-based deviation/harmful detect. For the model-based classifier like HarmBench-Llama and LlamaGuard, are they accurate enough for making the classification accurate? Especially, I did not fully understand how HarmBench-Llama detects deviation -- the model detects a set of behaviors actually.

The paper could have more analysis of impacting factors of attack success rate or efficiency numbers, such as the categories and length of malicious questions. Intuitively, the longer malicious prompts could lead to slower optimization. Additional analysis here could help the reader understand the attack impact in depth.

Finally, the embedding space scaled perturbation idea is presented in related work, e.g., Jailbreak Antidote: Runtime Safety-Utility Balance via Sparse Representation Adjustment in Large Language Models (ICLR 2025). It (and similar efforts) should be mentioned as part of the related work.

### Questions
* Given a specific malicious prompt, how likely does the "totally harmful" region exist? Could you provide quantitative probabilities or formal analysis?
* How does the HarmBench-Llama model achieve the deviation-detect? Are there inaccuracy issues of the detection/classification?

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
This paper investigates the vulnerability of open-source LLMs to perturbations at embedding layer, introducing the phenomenon of semantic shift when attackers manipulate the embedding vector of high-risk tokens. The authors propose SEP, a framework that systematically identifies and applies minimal but targeted perturbations to token embeddings, thereby evading safety alignment mechanisms in LLMs.

### Strengths
- This paper focuses on perturbation attacks in the embedding layer of LLMs, unlike those methods that concentrate on modifying input prompts.
- The mathematical formula for the embedding perturbation process is explicit, supporting reproducibility and a clear understanding of the operational mechanism.
- The discussion section offers practical suggestions, indicating an awareness of the broader implications of safe LLM releases.

### Weaknesses
- This paper would significantly benefit from a more thorough comparison with related work. Since 2023, several studies [1,2,3] have explored how internal representations affect LLM safety. While manipulation at the embedding layer is relatively novel, the paper lacks strong empirical evidence demonstrating why attacks at the embedding layer are more advantageous than those proposed by related work.
- Only modifying the embedding layer to achieve such a high ASR seems to conflict with prior work. For instance, [2] reports that safety-related concepts typically emerge in middle layers, whereas early layers, especially layer 0, cannot be effectively controlled to achieve the attacks. I invite the authors to comment on this.
- SEP relies on an external multi-stage classifier to identify whether a response is harmful. In a black-box scenario, this is a strong and potentially unrealistic assumption.

[1] Representation Engineering: A Top-Down Approach to AI Transparency, https://arxiv.org/abs/2310.01405

[2] Uncovering Safety Risks of Large Language Models through Concept Activation Vector, https://arxiv.org/abs/2404.12038

[3] Refusal in Language Models Is Mediated by a Single Direction, https://arxiv.org/abs/2406.11717

### Questions
- Can the authors clarify the practicality of SEP in real-world LLM application scenarios?
- Can the authors provide more detailed quantitative results on whether perturbations introduced by SEP can be detected via simple embedding-space similarity checks, or whether attack traces are left that defenders could leverage?

### Soundness
3

### Presentation
2

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
This paper introduces an attack framework called Search-based Embedding Poisoning (SEP) which targets the safety alignment of open-source Large Language Models (LLMs). The authors first identify a "Semantic Shift" phenomenon, where slight, controlled perturbations to the embedding vectors of "dangerous tokens" can cause the model to bypass its safety training. The SEP framework uses a three-stage search strategy to efficiently find these specific perturbations, achieving a high attack success rate across several models

### Strengths
- The work identifies and explores a vulnerability in the embedding layer, which is a different attack surface from more common prompt-level or suffix-based attacks.
- The "Semantic Shift" phenomenon is documented through an initial empirical study, providing a clear foundation for why the proposed attack framework is effective.
- The evaluation is quite thorough, testing the SEP method against six different open-source LLMs , four baseline attack methods , and two representative safety defense techniques.

### Weaknesses
- The threat model assumes the attacker can modify the model's execution flow or files to inject the perturbation before the decoder layers. This is a strong assumption that requires more access than a standard prompt injection attack and may not be feasible in many deployed scenarios. In other words, this paper starts with a scenario where open-source models are deployed on HuggingFace, but how the attackers manipulate the embeddings? It is unclear.
- I am not sure if this paper is novel considering [1]
- The attack's effectiveness seems highly dependent on identifying and perturbing explicit "dangerous tokens". This approach might not be as effective against more implicit or complex malicious queries where the harmful intent isn't located in a few obvious words.
- The performance on Gemma-7B was notably lower in both success rate and efficiency. While the paper suggests this indicates a more robust embedding layer, the specific reasons why this model is more resistant to this particular attack aren't deeply explored.

[1] Representation Engineering: A Top-Down Approach to AI Transparency

### Questions
n/a

### Soundness
2

### Presentation
3

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
The paper probes embedding-space vulnerabilities in aligned LLMs. It observes that nudging a single embedding dimension for selected tokens gradually tips responses from refusal to harmful outputs, with small "pockets" that reliably bypass safeguards. Building on this, the authors propose SEP (Search-based Embedding Poisoning) an exponential -> binary → linear search over perturbation magnitude guided by a classifier pipeline. On several instruction-tuned models, SEP reports high attack success on a ~150-prompt set. The broader takeaway: even tiny, structured perturbations to token embeddings, without changing prompts or weights, can meaningfully degrade safety.

### Strengths
### What I see as the main contributions

- Phenomenology: a clear description of refusal → uncertainty → deviation regimes as perturbations grow, with concentrated bypass regions.
- Method: a simple, reimplementable search (SEP) to locate those regions efficiently.
- Evidence: consistent success across multiple open models, plus basic ablations (e.g., temperature, search strategy).

### Strengths

- The threat surface is concrete and practical (embedding outputs at inference), which deployment teams actually have to think about.
- SEP is easy to grasp and reproduce; the staged search hits a nice balance between speed and coverage.
- The qualitative framing (response regimes and types) helps practitioners reason about failure modes rather than treating them as one-off glitches.

### Weaknesses
### Limitations and weaknesses

- Placement in existing taxonomies is thin. The work would read more cleanly if it explicitly located itself within public threat-model frameworks that already discuss latent-space manipulations and targeted compromises. In particular, "Operationalizing a Threat Model for Red-Teaming LLMs" and "Red-teaming for generative AI: Silver bullet or security theater?" lay out categories (e.g., latent space and targeted attacks) that map closely to what SEP is doing; acknowledging that context would clarify novelty and avoid the impression of “first to discuss embedding-layer threats.”
- Narrow benchmark. The evaluation relies on ~150 malicious prompts (largely JailbreakBench/HarmBench). That’s small relative to current safety eval suites and misses long-context, tool-use, multilingual, and policy-borderline prompts that often bite in practice.
- Classifier-dependence. Success is adjudicated by a multi-stage classifier stack; disagreements among classifiers can shift measured ASR. A small dose of human labeling or cross-checking would ground the claims.
- "Dangerous tokens" is too coarse. Tokens like "marijuana" or "homicide" can be benign in legal/medical contexts. Collapsing lexical presence into hazard misses intent and context, which likely explains some of the brittle behavior you observe.


### High-leverage suggestions

- Situate SEP within public taxonomies. Briefly connect to the categories and definitions in Operationalizing a Threat Model for Red-Teaming LLMs (characterizes Schwinn et al., 2024 as a Latent Space Attack) and Red-teaming for generative AI: Silver bullet or security theater?; also distinguish your integrity-oriented embedding perturbations from "embedding inversion" (privacy) and from suffix-based adversarial prompting. Mention these related works to show continuity and difference.
- Broaden the eval slice just a bit. Add a small panel of longer, tool-use, and multilingual prompts; report per-category ASR and variance across seeds. Even 50–100 extra prompts would strengthen external validity.
- Anchor the classifier calls. Report agreement rates between your classifiers and include a modest human-rated subset; add a sensitivity sweep over thresholds so readers can see how ASR moves.
- Refine token selection. Replace a static "dangerous token" list with a lightweight contextual hazard heuristic (span-level, intent-aware) and revisit the Appendix finding, this will read less like a lexical effect and more like a policy-effect.

### Nits / small fixes

- Clean up obvious typos in figure labels (“Perturabation”, “Posioned”, “Deinal”). Mostly these typos are in figures or captions of figures. Feels like the paper was written in a rush.
- Reword the line “Embedding perturbations induce semantic shifts, resulting in Total and Deviation …” to something like: “Embedding perturbations induce semantic shifts; as the magnitude increases, responses fall into three regimes (refusal, uncertain, deviation), with harmful outputs concentrated in parts of the uncertain regime.”
- Keep notation for the perturbation magnitude and quality metrics consistent across tables and captions.

### Questions
See weaknesses above

### Soundness
3

### Presentation
1

### Contribution
3
