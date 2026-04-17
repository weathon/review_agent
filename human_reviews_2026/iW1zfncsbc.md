# MAC-AMP: A Closed-Loop Multi-Agent Collaboration System for Multi-Objective Antimicrobial Peptide Design

- Decision: Accept (Poster)
- Scores: 6, 2, 4, 6

## Abstract
To address the global health threat of antimicrobial resistance, antimicrobial peptides (AMP) are being explored for their potent and promising ability to fight resistant pathogens. While artificial intelligence (AI) is being employed to advance AMP discovery and design, most AMP design models struggle to balance key goals like activity, toxicity, and novelty, using rigid or unclear scoring methods that make results hard to interpret and optimize. As the capabilities of Large Language Models (LLM) advance and evolve swiftly, we turn to AI multi-agent collaboration based on such models (multi-agent LLMs), which show rapidly rising potential in complex scientific design scenarios. Based on this, we introduce $\textbf{MAC-AMP}$, a closed-loop multi-agent collaboration (MAC) system for multi-objective AMP design. The system implements a fully autonomous simulated peer review-adaptive reinforcement learning framework that requires only a task description and example dataset to design novel AMPs. The novelty of our work lies in introducing a closed-loop multi-agent system for AMP design, with cross-domain transferability, that supports multi-objective optimization while remaining explainable rather than a 'black box'. Experiments show that MAC-AMP outperforms other AMP generative models by effectively optimizing AMP generation for multiple key molecular properties, demonstrating exceptional results in antibacterial activity, AMP likeliness, toxicity compliance, and structural reliability.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
MAC-AMP introduces a closed-loop, multi-agent LLM system for multi-objective antimicrobial peptide (AMP) design. The framework integrates three stages: a property prediction module using bioinformatics tools, an AI-simulated peer review system where multiple LLMs collaboratively evaluate peptides along activity, safety, and originality dimensions, and a reinforcement learning refinement stage that converts their structured consensus into executable reward functions for the generator.

### Strengths
1. The integration of multi-agent collaboration and reinforcement learning is innovative and potentially impactful for scientific design workflows.
2. The structured peer-review mechanism and log-based audit trail improve transparency relative to typical black-box AMP generators.
3. The modular design could be adapted to other molecule or protein design domains.

### Weaknesses
The framework relies on large proprietary models (GPT-5, Perplexity, and Gemini 2.5), which are not easily accessible or affordable for most users. It would be helpful if the authors could provide an estimation of the computational cost for a single complete run, or discuss how the system performs when these agents are replaced with smaller open-source models such as Qwen. Additionally, reporting the total training time compared to a standard non-agent GRPO baseline would help clarify the efficiency and scalability of the proposed approach.

### Questions
see weakness

### Soundness
3

### Presentation
3

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
The paper proposes MAC-AMP, a closed-loop multi-agent framework for multi-objective antimicrobial peptide design. It compiles structured “peer review” consensus into PPO-executable rewards to jointly optimize activity, toxicity, and structural reliability, and reports computational improvements over several baselines.
1) The work is difficult to reproduce at review time. Code and data are not publicly available; while the authors commit to releasing them upon acceptance, independent replication and verification cannot be performed during evaluation.
2) Core evaluation and constraints depend on external tools (e.g., ToxinPred 3.0, OmegaFold, Macrel, Foldseek). The reward and selection are tightly coupled to their outputs, yet there is no systematic robustness or bias analysis to assess how these components influence optimization outcomes.
3) The use of OmegaFold pLDDT as a structural reliability proxy for short peptides is insufficiently validated. The paper lacks cross-checks against experimental structures or more rigorous dynamical simulations to establish suitability in the short-peptide regime.
4)The comparison uses a uniform protocol of generating 1,000 candidates and selecting the top-30, which can amplify scorer preferences and ranking bias. The fairness and implications of this selection strategy are not discussed in depth.
5) The paper does not provide rigorous alignment of hyperparameters, training budgets, and screening strategies across baselines, nor does it include sensitivity analyses to quantify how these choices affect reported performance.

### Strengths
The paper introduces a novel closed-loop multi-agent framework that compiles structured peer-review consensus into PPO-executable rewards, enabling interpretable and auditable multi-objective optimization for antimicrobial peptide design. Empirically, it reports consistent computational improvements on activity, toxicity, and structural reliability metrics over several baselines, supported by a clearly described system architecture and informative ablation studies.

### Weaknesses
1) The work is difficult to reproduce at review time. Code and data are not publicly available; while the authors commit to releasing them upon acceptance, independent replication and verification cannot be performed during evaluation.
2) Core evaluation and constraints depend on external tools (e.g., ToxinPred 3.0, OmegaFold, Macrel, Foldseek). The reward and selection are tightly coupled to their outputs, yet there is no systematic robustness or bias analysis to assess how these components influence optimization outcomes.
3) The use of OmegaFold pLDDT as a structural reliability proxy for short peptides is insufficiently validated. The paper lacks cross-checks against experimental structures or more rigorous dynamical simulations to establish suitability in the short-peptide regime.
4)The comparison uses a uniform protocol of generating 1,000 candidates and selecting the top-30, which can amplify scorer preferences and ranking bias. The fairness and implications of this selection strategy are not discussed in depth.
5) The paper does not provide rigorous alignment of hyperparameters, training budgets, and screening strategies across baselines, nor does it include sensitivity analyses to quantify how these choices affect reported performance.

### Questions
1)  Please specify a concrete timeline and scope for releasing code, data, and pretrained models, including licenses, repository URL, and whether an anonymized artifact will be available during rebuttal.
2) Provide an exact, machine-readable environment (Docker/Conda) with pinned versions for all dependencies, especially third-party predictors (e.g., ToxinPred, OmegaFold, Macrel, Foldseek) and their model checkpoints.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces MAC-AMP, a closed-loop multi-agent collaboration framework for multi-objective antimicrobial peptide design, tackling the complex trade-offs between activity, toxicity, and stability. The system integrates LLM-based agents (e.g., GPT-5, Gemini 2.5, Perplexity) within an AI-driven peer-review loop that evaluates candidate peptides along multiple criteria and transforms the structured consensus into executable reinforcement learning reward signals. A biomedical–computer science dual-agent team iteratively refines these rewards, guiding a PPO-based generator to optimize AMP sequences under continuous feedback—achieving interpretable and auditable improvements. Across E. coli, S. aureus, and P. aeruginosa, MAC-AMP consistently surpasses baselines such as AMP-Designer, BroadAMP-GPT, PepGAN, and Diff-AMP in antibacterial efficacy, toxicity regulation, and structural robustness, while preserving AMP-like properties and strong cross-species generalization. Collectively, MAC-AMP establishes a generalizable and interpretable paradigm for autonomous molecular design.

### Strengths
The paper introduces a closed-loop multi-agent collaboration framework that transforms AMP design into an autonomous, explainable reinforcement learning process. Its integration of AI-simulated peer review to generate executable reward signals is both novel and technically elegant. It is clearly written, with transparent modular design and reproducible details. Overall, the paper is significant, offering a transferable paradigm for multi-objective molecular design and broader autonomous scientific discovery.

### Weaknesses
While the framework is conceptually strong, several areas could be improved. 
1. The novelty is quite limited. Using existing PPO and peptide generation model (Wang et al. 2025)
2. Experimental validation is limited to in silico analyses—no wet-lab or biophysical confirmation is provided to verify that the generated AMPs exhibit real-world antimicrobial activity, which would strengthen the biological significance. 
3. Although the AI-simulated peer review concept is creative, the paper could include quantitative ablations demonstrating how each reviewer agent or the Area Chair aggregation contributes to performance, beyond qualitative explanation. 
4. The computational cost and scalability of running multi-agent deliberation and RL loops are not clearly analyzed, raising concerns about feasibility for large-scale or real-time applications. 
5. While the framework claims cross-domain generalizability, no transfer experiments to non-AMP domains are shown, leaving this claim largely theoretical. Addressing these limitations would substantially enhance the rigor and impact of the work.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces MAC-AMP, a closed-loop multi-agent collaboration (MAC) system for multi-objective antimicrobial peptide (AMP) design. The system integrates property prediction, AI-simulated peer review, reinforcement learning (RL) refinement, and peptide generation modules in a fully autonomous loop. A key novelty is the conversion of multi-agent textual consensus into executable reward signals, allowing reinforcement-based optimization without manual prompt tuning. Experiments on five bacterial targets show that MAC-AMP outperforms existing generative models in antibacterial activity, toxicity reduction, and structural reliability. Ablation studies confirm that both the AI-simulated peer review and RL refinement modules contribute critically to balanced multi-objective optimization.

### Strengths
- The paper introduces a credible pipeline that operationalizes “multi-agent collaboration” beyond conversational coordination, producing quantitative, auditable training signals.
- Transparent logging and role-based agent structure allow reproducibility and traceability uncommon in LLM-based systems.
- The system’s iterative RL refinement aligns multi-agent consensus with executable objectives, avoiding reward hacking typical in static scoring systems.
- Results across multiple bacterial targets and comparisons to several baselines demonstrate consistent gains in activity and safety metrics.
- Includes ablation analyses, sequence-level property comparisons, and visualization of learned chemical space.
- The framework is described as transferable to other molecule or material design problems with minimal adaptation.

### Weaknesses
- The quality of generated AMPs is constrained by the accuracy of property predictors (e.g., ToxinPred 3.0, OmegaFold). The paper acknowledges this but does not quantify its impact.
- While broad-spectrum testing is reported, the reasoning behind cross-species generalization (beyond physicochemical similarity) could be more rigorously analyzed.
- Although individual module ablations are provided, it remains unclear whether a simpler reward design or fewer agents would achieve similar performance.

### Questions
1. How sensitive is MAC-AMP’s performance to the specific LLMs used as reviewer agents (GPT-5, Gemini 2.5, Perplexity)? Would substituting smaller open models maintain performance trends?

2. The AI-simulated peer review uses a lexicon-tagging mechanism, how were the weights and tag mappings validated for interpretability or bias?

3. Does the system ever experience feedback collapse or overfitting to its own reviewers’ biases after multiple closed-loop iterations?

4. How is novelty quantitatively ensured beyond low Foldseek similarity, are there checks for redundancy or motif-level overlap with training data?

5. Could the authors provide empirical measures of reward variance or entropy across RL stages to confirm that adaptive optimization prevents reward hacking?

6. What are the computational costs (GPU hours, number of agent calls per iteration) and how do they scale with peptide length or number of objectives?

### Soundness
3

### Presentation
3

### Contribution
3
