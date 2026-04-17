# MedAgent-Pro: Towards Evidence-based Multi-modal Medical Diagnosis via Reasoning Agentic Workflow

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 2

## Abstract
Modern clinical diagnosis relies on the comprehensive analysis of multi-modal patient data, drawing on medical expertise to ensure systematic and rigorous reasoning. Recent advances in Vision–Language Models (VLMs) and agent-based methods are reshaping medical diagnosis by effectively integrating multi-modal information. However, they often output direct answers and empirical-driven conclusions without clinical evidence supported by quantitative analysis, which compromises their reliability and hinders clinical usability. 
Here we propose MedAgent-Pro, an agentic reasoning paradigm that mirrors modern diagnosis principles via a hierarchical diagnostic workflow, consisting of disease-level standardized plan generation and patient-level personalized step-by-step reasoning. To support disease-level planning, a retrieval-augmented generation agent is designed to access medical guidelines for alignment with clinical standards.  For patient-level reasoning, MedAgent-Pro leverages professional tools such as visual models to take various actions to analyze multi-modal input, and performs evidence-based reflection to iteratively adjust memory, enforcing rigorous reasoning throughout the process. Extensive experiments across a wide range of anatomical regions, imaging modalities, and diseases demonstrate the superiority of MedAgent-Pro over mainstream VLMs, agentic systems and leading expert models. Ablation studies and expert evaluation further confirm its robustness and clinical relevance. Anonymized code link is available in the reproducibility statement.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces **MedAgent-Pro**, an agent-based system designed for multi-modal medical diagnosis. The authors identify a key limitation in current Vision-Language Models (VLMs) and agentic systems: they tend to provide direct, empirical-driven answers without the step-by-step, evidence-based reasoning that is fundamental to real-world clinical practice.

To address this gap, MedAgent-Pro proposes a hierarchical workflow that mimics clinical procedure. This workflow is divided into two main stages: **(i)** Disease-Level Planning: A Retrieval-Augmented Generation (RAG) agent accesses a medical knowledge base to formulate a standardized, multi-step diagnostic plan based on established clinical guidelines. **(ii)** Patient-Level Reasoning: The agent executes this plan step-by-step, analyzing the specific patient's multi-modal data. This execution phase leverages a toolbox of "professional tools," such as segmentation models, to perform quantitative analysis (e.g., calculating the cup-to-disc ratio) alongside qualitative assessments.

The authors present an extensive evaluation across more than 10 imaging modalities and 50 diseases, claiming that MedAgent-Pro substantially outperforms general VLMs like GPT-4o, other medical agent systems, and task-specific expert models. The system's alignment with clinical workflows is further supported by positive evaluations from clinical experts. Overall, this is a fairly strong paper.

### Strengths
### **Originality and Significance:** 
The most significant strength is the novel agentic paradigm itself. By formalizing the clinical diagnostic process into a hierarchical workflow (plan-via-RAG, execute-via-tools), the paper moves the field from "black-box" pattern recognition towards auditable, evidence-based reasoning. This directly addresses the critical needs of reliability, safety, and trustworthiness for medical AI.   


### **Problem Formulation:** 
The paper does an excellent job of motivating the work, clearly articulating the gap between current VLM capabilities (which provide "empirical-driven conclusions") and the needs of clinical practice (which demands "quantitative analysis" and structured reasoning).   


### **Experimental Breadth:**
The evaluation is a major strength. The authors test their system on a very wide array of tasks, covering over 10 imaging modalities, 20+ anatomies, and 50+ diseases, providing strong evidence for the generalizability of the framework.   


### **Clinician Evaluation:** 
The inclusion of a qualitative evaluation by clinical experts is a crucial addition. Reporting that clinicians rated MedAgent-Pro's outputs higher on dimensions like "reasoning coherence" and "clinical reliability" provides strong, practical validation of the paper's central thesis.

### Weaknesses
### **Inadequate Contextualization with Prior Work:** 

The paper claims that current medical agentic systems "simply glued all tools together". This may be an oversimplification that ignores recent, highly relevant work. The authors should cite and differentiate their approach from other multi-modal medical agents like "SMR-agents" (Wang et al., [1]) and "AURA" (Fathi et al., [2]), which also appear to integrate multi-modal reasoning and tool use in a sophisticated manner. A more nuanced comparison in the Related Work section is needed to clearly establish this paper's specific contributions over the state-of-the-art.

### **The "ToolBox" is Underspecified:**
The system's performance is critically dependent on its "Available ToolBox". The paper mentions "visual/coding models" and gives MedSAM as an example, but a comprehensive list of all tools, their specific capabilities, and their individual performance is not provided. Without this, it's impossible to assess the agent workflow independently of the tools. 

### **"Evidence-Based Reflection" is Unclear:**
 This mechanism is highlighted as a key contribution but is operationally vague. How does the agent "evaluate the reliability" of a step's output? Is this a heuristic, a learned classifier, or a VLM self-critique prompt? The "Evi. Reasoning" block in Figure 2 is abstract. This core mechanism needs to be formalized.


### **Conceptual Limitation to Sequential Reasoning:** 
The proposed methodology is inherently bounded to sequential reasoning steps. The RAG agent generates a linear plan, and the execution agent follows it. The paper does not discuss or explore scenarios that might require parallel exploration of different diagnostic paths or more complex, branching decision-making. This is a notable limitation, as real-world diagnostics can often be non-linear.

### **Confusing Mathematical Notation:** 
The mathematical expressions in Section 3.2, such as representing inputs and outputs as key-value pairs (e.g., $\mathcal{P}i:r_{i}=\langle k_{r_{i}},v_{r_{i}}\rangle$), add a layer of formalism that is arguably unnecessary and confusing. A-plainer textual description of how data is passed and transformed between steps would significantly improve clarity and accessibility.

### **RAG Knowledge Base:** 
The paper relies on a "Medical Knowledge Base" sourced from MedlinePlus. The scale, and curation of this knowledge base, as well as the indexing and retrieval process, are mentioned but could be detailed further. The quality of the RAG-generated plan is a critical bottleneck that is not explicitly evaluated.

[1] Wang et al., "SMR-agents: Synergistic medical reasoning agents for zero-shot medical visual question answering with MLLMs"

[2] Fathi et al., "AURA: A Multi-modal Medical Agent for Understanding, Reasoning and Annotation"

### Questions
See "Weaknesses" for questions. **Note** that, if the related works section and other problems in Weaknesses are not addressed the rating can be subject to deduction.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work explores reasoning agentic system of medical diagnosis. By disease-level standardized plan generation and patient-level personalized step-by-step reasoning, MedAgent-Pro presents modern diagnosis principles via a hierarchical diagnostic workflow. They leverages professional tools such as visual models to take various actions to analyze multi-modal input, and performs evidence-based reflection. Extensive experiments across a wide range of anatomical regions, imaging modalities, and diseases are conducted.

### Strengths
1.	Evaluations across 10+ imaging modalities, 20+ anatomies, and 50+ diseases validate its universality.
2.	The proposed system fully leverages existing models in their toolbox to provide clinical evidence.
3.	The proposed model is trustworthy because of more visual cues and indicators compared to vanilla VLMs.
4.	Comprehensive ablations further validate the effectiveness of major components in the agentic system.

### Weaknesses
1.	The author proposed evidence-based system. However, the proposed system is more like an engineering product, which lack of architecture innovation and similar to other agentic systems[1,2].
2.	The author should provide analysis in view of visual cues compared to medical world model[3], which was also designed for treatment planning. Is the proposed system providing any visual evidence?
3.	Which part of MedAgent-pro is finetuned? Is the backbone GPT-4o finetuned on downstream data? What about the compared methods? The fairness should be ensured.
4.	In Evidence-based Reflection, is it possible that s_i is always continue, resulting in unexpected dead loop?
5.	Clinical guidelines may be inconsistent across different areas and hospitals. How can the proposed system overcome such inconsistency?

[1] Zhu Y, He Z, Hu H, et al. MedAgentBoard: Benchmarking Multi-Agent Collaboration with Conventional Methods for Diverse Medical Tasks[J]. arXiv preprint arXiv:2505.12371, 2025.
[2] Zhu Y, Qi Y, Wang Z, et al. HealthFlow: A Self-Evolving AI Agent with Meta Planning for Autonomous Healthcare Research[J]. arXiv preprint arXiv:2508.02621, 2025.
[3] Yang Y, Wang Z Y, Liu Q, et al. Medical world model: Generative simulation of tumor evolution for treatment planning[J]. arXiv preprint arXiv:2506.02327, 2025.

### Questions
Please see weaknesses

### Soundness
3

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
4

### Summary
The paper introduces MedAgent-Pro, an agentic reasoning workflow designed to emulate evidence-based clinical diagnosis. The system operates on a hierarchical structure, using planning, RAG and reasoning agents to ensure reliability.

### Strengths
The primary strength of this work is its alignment with medical practice, from an empirical one-hop VQA to a structured and evidence-based reasoning process.

### Weaknesses
My major comments are:

1. The system's effectiveness, particularly its quantitative analysis which drives the performance gains, relies on the availability of specialized visual tools for a given task. The workflow's performance for diseases or modalities lacking such robust, pre-existing tools is less clear. While the NEJM results without tools are still strong, they lack the key quantitative grounding that differentiates the method.
2. The VLM  remains a single point of failure for several critical steps. The VLM is solely responsible for generating the entire diagnostic plan from guidelines, performing all qualitative analysis, executing the evidence-based reflection, and assigning the final risk-based weights. Any hallucination or error in these steps could compromise the entire workflow.
3. The main comparison tables compare MedAgent-Pro (with tools) against general VLMs (without tools). This confounds the benefit of the agentic workflow with the benefit of simply having tool access. A more rigorous comparison, presented only in Appendix B.4, shows that while MedAgent-Pro still wins, giving baselines tool access does significantly close the gap. This stronger, more fair comparison should have been centered in the main paper.
4. The evaluation on the MITEA dataset (heart disease) was simplified from a 7-class problem to a binary classification task (healthy vs. heart disease) due to limited samples per category. This simplification is a significant departure from a realistic clinical scenario, where differentiating between various heart conditions (e.g., amyloidosis vs. hypertrophy) is the critical diagnostic task.
5. The RAG agent's knowledge base is MedlinePlus. However, clinical-grade diagnosis typically relies on more complex professional guidelines from medical societies (e.g., ACC/AHA, etc.).

### Questions
Please see the weaknesses above

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
5

### Summary
This paper introduces MedAgent-Pro, an agentic hierarchical workflow system for evidence-based, multi-modal medical diagnosis. The approach separates disease-level planning (standardized per guidelines using a retrieval-augmented generation agent) and patient-level stepwise reasoning empowered by professional tools (e.g., visual models, coding agents), integrating iterative reflection for reliability. MedAgent-Pro is benchmarked on extensive multi-modal datasets, showing performance gains over mainstream VLMs, agentic baselines, and expert models.

### Strengths
1. The hierarchical design effectively mirrors medical diagnostic principles, with disease-level planning driven by retrieval from guidelines and patient-level personalized analysis.
2. The RAG-based incorporation of external domain knowledge into the planning stage supports clinical transparency.

### Weaknesses
1. Although the hierarchical structuring and rigorous evidence focus are valuable, the architecture closely resembles recent work in multi-agent, RAG-augmented, and tool-integrated medical AI (e.g., MedAgents, MMedAgent, MDAgent, and others in Table 1). Moreover, the overall contribution of this work appears rather limited. Many of the claimed innovations rely on techniques that have already become widely adopted or established in the field. As such, the paper seems more like an integration or system-level implementation of several existing popular paradigms, rather than presenting a novel method.

2. Can the authors clarify and mathematically specify the state assessment function $\phi$ used during patient-level reasoning? How is output reliability judged, is it a learned model, rule-based, or probabilistically estimated?

3. How are risk-based weights $\mathcal{W}$ in risk score computation (p.5) defined in practice? Are they derived from clinical guidelines directly, learned via validation, or optimized by the system?

4. For clinical expert evaluation, please elaborate on the number of raters, the randomization protocol, inter-rater agreement metrics (such as Cohen’s kappa or ICC), and whether any anchoring or exposure effects could have occurred.

### Questions
see Weaknesses

### Soundness
3

### Presentation
3

### Contribution
2
