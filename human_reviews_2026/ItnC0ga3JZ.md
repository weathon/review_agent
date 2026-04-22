# SecMCP: Quantifying Conversation Drift in MCP via Latent Polytope

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
The Model Context Protocol (MCP) enhances large language models (LLMs) by integrating external tools, enabling dynamic aggregation of real-time data to improve task execution. However, its non-isolated execution context introduces critical security and privacy risks. In particular, adversarially crafted content can induce tool poisoning or indirect prompt injection, leading to conversation hijacking, misinformation propagation, or data exfiltration. Existing defenses, such as rule-based filters or LLM-driven detection, remain inadequate due to their reliance on static signatures, computational inefficiency, and inability to quantify conversational hijacking. To address these limitations, we propose SecMCP, a secure framework that detects and quantifies conversation drift, deviations in latent space trajectories induced by adversarial external knowledge. By modeling LLM activation vectors within a latent polytope space, SecMCP identifies anomalous shifts in conversational dynamics, enabling proactive detection of hijacking, misleading, and data exfiltration. We evaluate SecMCP on three state-of-the-art LLMs (Llama3, Vicuna, Mistral) across benchmark datasets (MS MARCO, HotpotQA, FinQA), demonstrating robust detection with AUROC scores exceeding 0.915 while maintaining system usability. Our contributions include a systematic categorization of MCP security threats, a novel latent polytope-based methodology for quantifying conversation drift, and empirical validation of SecMCP’s efficacy.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a detection framework called SECMCP (Secure Model Context Protocol), designed to quantify and identify conversation drift—semantic deviations that occur when large language models interact with external tools. By modeling the trajectories of model activation vectors within a latent polytope space, the method measures deviations between the current input and the normal semantic trajectory, enabling the detection of security risks caused by malicious external knowledge, prompt injection, or tool manipulation. The approach does not rely on prior knowledge of specific attacks and achieves high detection accuracy across multiple threat types (including tool poisoning, indirect prompt injection, and data leakage), with an average AUROC of 0.98. Its robustness is further validated across diverse models and datasets.

### Strengths
1.The method is easy to follow.
2.The paper is well-structured and clearly written.

### Weaknesses
1.The paper lacks a released code repository, which limits reproducibility.
2.Although the authors frame the work in an MCP scenario, conceptually this paper is not distinct from general LLM misbehavior detection: the MCP’s server and datasource merely act as attack initiators, while the ultimate target of attacks is still the LLM. The manuscript therefore lacks discussion of closely related work in LLM misbehavior detection and does not compare against such methods as baselines.
3.The Risk Matching section is confusing: you first describe using a threshold to make a binary decision, and then state that a decision tree is used for classification. Please clarify the module’s workflow and logic (how thresholding and the decision tree interact, what each step outputs, and why both are necessary).
4.The explanation of using the MS MARCO dataset may raise concerns about the method’s generalizability and scalability. 
5.Line 111 misuses the phrase “on the other hand / end”, which means a contrast (e.g., “on the opposite side” or “conversely”). Please correct the wording for clarity.

### Questions
1. The author should discuss and compare with the sota method of LLM misbehaviour detection.
2. Please detailed describe the risk matching section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces SECMCP, a security framework designed to detect and quantify "conversation drift" within the Model Context Protocol (MCP) used by LLMs interacting with external tools. It addresses security risks like hijacking, misleading information, and data exfiltration, which arise from adversarial content injected via MCP servers or data sources. SECMCP operates by monitoring the LLM's internal activation vectors, hypothesizing that malicious inputs cause deviations from normal conversational trajectories in the latent space. It models a "latent polytope" based on activations from benign queries, and flags incoming queries whose activations fall significantly outside this region. The method is evaluated on several LLMs and datasets, demonstrating high detection rates (AUROC > 0.915) for simulated attacks.

### Strengths
1. The paper clearly articulates the security risks inherent in MCP's non-isolated execution context and categorizes them effectively (Section 3)
2. Section 4.3 (Activation Collection and Risk Matching) details a concrete implementation based on comparing input activations to benign anchor points.
3. The high AUROC scores achieved across multiple models, datasets, and attack types in the main effectiveness evaluation (Table 1)  provide strong initial evidence for the viability of the approach.

### Weaknesses
The primary weakness is the limited robustness testing (Section 5.3) . Only evaluating against synonym replacement significantly underestimates the capabilities of adaptive adversaries. Stronger adaptive attacks, potentially optimized to minimize activation deviation while still achieving malicious goals (akin to adversarial example attacks in vision), are needed for a convincing robustness claim. The drop in AUROC (e.g., ~0.12 for Data Exfiltration) suggests vulnerability to more advanced attacks.

### Questions
Clarify my comments in the "Weaknesses" section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces SecMCP, a framework that detects and quantifies conversation drift in MCP-powered agents by analyzing deviations in LLM activation vectors within a latent polytope space. It identifies security threats of LLM like data exfiltration, achieving detection AUROC scores above 0.915 across multiple models and benchmarks. Compared to signature-based or heuristic defenses, SecMCP offers a more generalizable approach without prior knowledge of attack formats.

### Strengths
* Good Motivation: The paper effectively identifies a critical and timely problem: the security risks inherent in the non-isolated execution context of the increasingly popular MCP.

### Weaknesses
The major concern I have about this paper is the novelty:

The core technical contribution—using activation vectors to detect anomalous model behavior—lacks significant novelty. The concept of monitoring internal activations (or "steering vectors") to understand and control LLM outputs has been an active area of research, as cited in the paper itself. While the application of this technique to the specific MCP context is valuable, the MCP scenario itself does not appear to introduce fundamentally new technical challenges that would necessitate a novel detection paradigm. The paper applies a known and powerful methodology to a new application domain, but the underlying mechanism remains largely the same.

### Questions
What is the difference in using activation vectors to detect LLM misbehavior between a single LLM and an LLM agent?

### Soundness
3

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
3

### Summary
The paper studies security/privacy risks in MCP-based LLM systems arising from (i) tool poisoning via adversarial tool/server descriptions (A_ser) and (ii) indirect prompt injection via external data sources (A_ds), leading to misleading responses or full hijacking (“output a when attacker inputs p”). The proposed method is an activation-space detector. Experiments on several LLMs and MCP-like tasks report high AUROC for detecting conversational drift.

### Strengths
Main Strength: The first work to detect attacks on MCP via LLM activations.

1.	Well-motivated setting (MCP security) – The paper anchors itself in a real, currently under-specified attack surface (MCP hosts + clients + servers), and clearly separates tool poisoning vs. indirect prompt injection and the three resulting risks (hijacking, misleading, exfiltration). This framing is useful beyond this paper. 
2.	Activation-space detection, not output heuristics – Instead of relying on surface-form filters or LLM self-judging, the method uses per-layer activations of the last token and distance to anchor points to define an “authorized polytope.” This is aligned with recent activation-steering / drift-detection work but applied to MCP, which is novel in this context. 
3.	Coverage of three attack families on three models and three datasets – The experimental section is broad for an applied security paper: 3×3×3 grid, AUROC tables, plus a comparison to preventive baselines (Sandwich, Instructional, Known-answer). Reported AUROCs are strong (often >0.98). 
4.	Ablations and robustness check – The paper studies number of anchors and layer choice, and even runs a synonym-replacement adaptive attack, showing where the method degrades (notably for exfiltration). That’s good transparency. 
5.	Clear statement of limitations – The authors explicitly note lack of token-level attribution and difficulty in fully asynchronous / multi-agent settings, which are exactly the hard cases for MCP-style systems.

### Weaknesses
1.	Official implementation weakness: The code has no README, and it is unclear how to run the code, what configuration reproduces which experiment, or how the reported results were obtained. 
2.	Method underspecified: The paper says: “we compute a low-dimensional embedding vector of its activation representation using an embedding model” and then “we utilize a decision tree classifier,” but it is not stated what the activation-embedding model is nor how/when the tree is trained, on which splits, or with what class balance. This must be made explicit. 
3.	Novelty vs. existing activation-drift work needs to be sharper: The paper cites Abdelnabi et al. (2024) for “catching LLM task drift with activations,” but the current writeup makes SECMCP look structurally very close. Please clarify positioning. Also, please add related work on activation-based security, I would start here: [1].
4.	The method flags inputs as “attacked” and then detects activation drift from benign anchors. It does not verify that the model actually executed the injected instruction or that the conversation was semantically hijacked*. Thus the paper leaves successful-attack detection unresolved. This is important enough to state explicitly as a limitation / future work: LLM security tools (and consequently MCP security) ultimately need to determine whether an attack succeeded, and this has been reported as one of the hardest parts of such systems [2]. 
5.	No qualitative trace: For a security paper on “conversation drift,” at least one conversation example should be shown.

Writing: 

7.a)	Indexing in Sec. 3 mixes “1..m” even when objects do not share cardinality; 

7.b)	in Sec. 4.3, $A \subset \mathbb{R}^n$ but the distance $\mathcal{D}^l$ also sums over (n) anchors, so (n) is used both as feature dimension and number of anchors. These reduce clarity.

7.c)	“Examine the impact of … visualizations of activation deviation” is not a design factor; reserve ablation for things that change model behavior (layer, #anchors, classifier).

7.d)	Sec. 3.1 repeats material from 3.2–3.3 and can be shortened

-----

[1] Lee et. al (2025). "Programming refusal with conditional activation steering" ICLR'25‏

‏ [2] Brokman et. al (2025). “Insights and current gaps in open-source LLM vulnerability scanners: A comparative analysis” IEEE/ACM International Workshop on Responsible AI Engineering (ICSE-RAIE)

### Questions
Pleas refer to weaknesses above.

### Soundness
2

### Presentation
2

### Contribution
3
