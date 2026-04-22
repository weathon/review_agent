# Hypothesis Hunting with Evolving Networks of Autonomous Scientific Agents

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 6, 2

## Abstract
Large-scale scientific datasets—spanning health biobanks, cell atlases, Earth reanalyses, and more—create opportunities for exploratory discovery unconstrained by specific research questions. We term this process $\textit{hypothesis hunting}$: the cumulative search for insight through sustained exploration across vast and complex hypothesis spaces. To support it, we introduce $\texttt{AScience}$, a framework modeling discovery as the interaction of agents, networks, and evaluation norms, and implement it as $\texttt{ASCollab}$, a distributed system of LLM-based research agents with heterogeneous behaviors. These agents self-organize into evolving networks, continually producing and peer-reviewing findings under shared standards of evaluation. Experiments show that such social dynamics enable the accumulation of expert-rated results along the diversity–quality–novelty frontier, including rediscoveries of established biomarkers, extensions of known pathways, and proposals of new therapeutic targets. While wet-lab validation remains indispensable, our experiments on cancer cohorts demonstrate that socially structured, agentic networks can sustain exploratory hypothesis hunting at scale.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents "Hypothesis Hunting," an AI-driven paradigm for autonomous scientific discovery on large-scale datasets (e.g., genomics, Earth systems). It formalizes discovery in the AScience framework—modeling dynamic interactions among agents, networks, and evaluation protocols—and implements ASCollab, a decentralized LLM-based agent network with heterogeneous expertise, ReAct reasoning, shared registries, and meta-review tournaments. Evaluated on TCGA, the system rediscovers known biomarkers (e.g., BIRC5), extends pathways (e.g., ferroptosis), and proposes novel targets (e.g., SLC5A2, ABCC8), outperforming baselines on novelty and quality metrics; emergent agent behaviors and self-evolving knowledge flows are observed, and a clinical case independently validates BIRC5 and PRKD1 in renal cancer while suggesting dual-target strategies and candidate drugs. Limitations include domain specificity and the need for wet‑lab validation, but the work demonstrates AI as a collaborative catalyst for bridging data-to-knowledge gaps.

### Strengths
1. Proposes the Hypothesis Hunting paradigm for AI-driven autonomous scientific discovery, overcoming the limits of preset questions. 

2. Introduces the AScience theoretical framework that formalizes discovery as dynamic interactions among agents, networks, and evaluation protocols, and implements it via ASCollab, which integrates heterogeneous agents, self-organizing networks, and peer‑review mechanisms.

3. Implements a distributed agent system that leverages heterogeneous LLM agent behaviors and supports diverse exploration strategies. Uses a two‑tier review (expert review + meta‑review) and a shared knowledge registry to ensure hypothesis novelty and quality. Validated on TCGA cancer data: rediscovers known biomarkers (e.g., BIRC5), extends pathways (e.g., ferroptosis), and proposes new targets (e.g., SLC5A2, ABCC8).

4. Simulates competition–collaboration dynamics of scientific communities (e.g., reputation systems) to avoid premature convergence and enhance discovery diversity. Networked agents outperform independent agents, yielding more high‑quality and novel hypotheses.

5. Integrates multi‑omics data (transcriptomics, proteomics, clinical) with external knowledge bases (e.g., PubMed, DepMap) to improve the biological plausibility of proposed hypotheses.

### Weaknesses
1. Experimental results are based solely on TCGA cancer genomics data and have not been validated in other disciplines (e.g., physics, chemistry).
2. The approach assumes subsequent wet‑lab validation; the current system only provides candidate lists, and its actual biomedical value remains to be determined.
3. The distributed agent network and continuous review mechanisms impose substantial compute requirements.
4. Agent decision logic is largely a black box, which affects reproducibility of results.
5. The evolutionary mechanisms of the dynamic network structure (e.g., convergence efficiency) require further theoretical analysis.

### Questions
See weakness.

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
3

### Summary
The paper introduces the task of hypothesis hunting and presents AScience, a framework that treats scientific discovery as an interaction among agents, their networks, and shared evaluation norms. The authors implement this framework as ASCollab, a distributed system of LLM based research agents with diverse behaviors. These agents organize themselves into evolving collaboration networks and continuously produce findings while peer reviewing one another under common evaluation standards.

### Strengths
* The overall framework is fairly comprehensive and logically coherent.
* Includes concrete downstream tasks.

### Weaknesses
* The novelty of paper is limited; it essentially applies a multi-agent approach in the biology domain. What it calls “hypothesis hunting” is, in my view, simply scientific discovery.
* The experimental evaluation is insufficient; see the Questions section.

### Questions
* How does the number of agents affect performance? I suggest scaling the number of agents.
* How are other backbones used in the system, such as Qwen or LLaMA?
* I believe ablation studies removing certain types of agents are missing; for example, what are the results if the review stage is omitted?
* Please explain exactly what the difference is between scientific discovery and hypothesis hunting.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper, submitted to ICLR 2026, addresses the challenge of exploratory discovery in large-scale scientific datasets (e.g., cancer genomics, biobanks) through a framework called AScience and its instantiation ASCollab. The core concept, termed "hypothesis hunting," refers to the continuous, diverse exploration of large datasets to surface promising findings for human validation—addressing limitations of human-led research (scale and coordination).

AScience models scientific progress as a dynamic system with four components: an epistemic landscape of research approaches, heterogeneous scientific agents, attention-routing networks, and shared evaluation norms. ASCllab, a distributed system built on this framework, uses LLM-based agents (with heterogeneous expertise and behaviors) that self-organize into evolving networks. These agents generate findings, collaborate, and undergo a structured peer-review process (review + meta-review) to curate high-quality results into a shared archive.

Empirical evaluations on three TCGA cancer cohorts (KIRC, PAAD, DLBC) show that ASCllab outperforms independent agents: its findings are more novel, higher-quality, and diverse, including rediscoveries of established biomarkers (e.g., BIRC5 in KIRC), extensions of known pathways (e.g., ferroptosis), and proposals of new therapeutic targets (e.g., SLC5A2 in PAAD). The paper concludes that socially structured agent networks enable scalable, cumulative hypothesis hunting, though wet-lab validation remains necessary.

### Strengths
(1) The writing is easy to follow.

(2) This paper fills a Critical Gap. Unlike existing autonomous science systems (e.g., AI Scientist, AI Co-Scientist) that focus on answering predefined research questions, this work formalizes "hypothesis hunting" as a distinct, open-ended problem setting. It explicitly addresses the need for cumulative exploration (not just goal-driven convergence) and models scientific progress as a social process—mirroring human scientific communities (collaboration, peer review, knowledge accumulation).

### Weaknesses
(1) Incomplete experimental validation. All experiments focus on cancer genomics (TCGA). The paper does not test ASCllab on other large-scale datasets (e.g., Earth reanalyses, cell atlases) mentioned in the introduction. It is unclear if the framework’s social dynamics (e.g., collaboration, peer review) would translate to fields with different evaluation norms (e.g., physics, climate science) or data types (e.g., image-based cell atlases vs. tabular genomics).

(2) The experiments use only 16 agents over 40 rounds. While this suffices to demonstrate proof of concept, it is unclear how the system would behave with larger populations (e.g., 100+ agents) or longer timeframes. For example:
Would network dynamics become unmanageable (e.g., information overload in the archive)?
Would agent diversity persist, or would specialization converge over time?


(3) Expert Assessment is somewhat subjective: While expert evaluation is a strength, the paper relies on a single domain expert for KIRC (and two for case studies). There is no inter-rater reliability analysis—different experts might score novelty/quality differently, especially for "borderline" findings (e.g., N3/N4 novelty). Are there any cross-validation or alogrithm-based evaluation.


(4) The meta-review process uses "tournament-style" relative scoring of clustered submissions, but the paper does not explain how clusters are defined (e.g., thematic similarity metrics) or how meta-reviewers calibrate scores across clusters. This could introduce bias if clusters are unevenly sized or themed.

### Questions
Please see the weakness.

### Soundness
3

### Presentation
2

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
The paper presents AScience, a multi-agent framework for “hypothesis hunting,” i.e., the autonomous exploration of large-scale scientific datasets to surface potential insights. Each agent functions as a simulated researcher that can analyze data, generate hypotheses, peer-review others’ work, and exchange findings within an evolving social network. The authors implement this system as ASCollab and evaluate it on cancer genomics datasets (TCGA), showing that such socially structured agent networks can produce diverse, novel, and scientifically plausible hypotheses, some of which rediscover known biomarkers or propose new therapeutic leads. The approach is conceptually related to prior multi-agent simulation efforts (e.g., Simulacra, SocialLab, etc.) but extends them to emulate a full research community interacting around a dataset.

### Strengths
The vision is compelling. The paper introduces an ambitious and creative idea: deploying a network of autonomous agents to “hunt” for hypotheses within complex scientific datasets. Conceptually, this aligns with the broader movement toward AI-for-Science systems that go beyond automation to exploration and reasoning. The idea of coupling such a system with downstream validation agents (e.g., experimental design, simulation, or lab integration frameworks like Biomni) could, in principle, enable a closed loop of scientific discovery. While the current implementation is early, the long-term vision, AI agents collaborating, critiquing, and refining hypotheses like a synthetic scientific community, is inspiring and potentially high-impact.

### Weaknesses
The use of the term “hypothesis hunting” is confusing and somewhat unnecessary. The established term in the literature is hypothesis generation, and it is unclear whether the authors intend a substantive distinction or simply rebrand existing ideas. Framing aside, multi-agent hypothesis generation has already been explored in prior works cited in the related-work section. However, none of these prior systems are directly benchmarked here, the only comparison is to independent single-agent runs, making it difficult to assess what is actually improved by the proposed approach.

Methodologically, the system relies on standard multi-agent orchestration built on top of GPT-4o, a model that has since been surpassed by more capable successors (e.g., GPT-5/5-Pro). This raises an important question: would a stronger single model already outperform or render unnecessary the proposed multi-agent framework for hypothesis generation? Without such comparisons, it is difficult to isolate the contribution of the agentic coordination itself. In addition, all experiments are confined to a single dataset (TCGA), despite the introduction referencing a broader range of large-scale scientific datasets. Evaluating across multiple LLMs and diverse domains would substantially strengthen both the generality and the empirical rigor of the paper.

Overall, while the engineering is solid, the paper lacks clear methodological or conceptual novelty. If the contribution lies in a superior implementation of multi-agent hypothesis generation, this should be validated through head-to-head benchmarks against prior systems. In its current form, the paper feels more like an incremental composition of existing ideas rather than a new learning or inference principle.

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
2
