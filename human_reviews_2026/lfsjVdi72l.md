# SEED-SET: Scalable Evolving Experimental Design for System-level Ethical Testing

- Decision: Accept (Poster)
- Scores: 4, 6, 2

## Abstract
As autonomous systems such as drones, become increasingly deployed in high-stakes, human-centric domains, it is critical to evaluate the ethical alignment since failure to do so imposes imminent danger to human lives, and long term bias in decision-making. Automated ethical benchmarking of these systems is understudied due to the lack of ubiquitous, well-defined metrics for evaluation, and stakeholder-specific subjectivity, which cannot be modeled analytically.  To address these challenges, we propose SEED-SET, a Bayesian experimental design framework that incorporates domain-specific objective evaluations, and subjective value judgments from stakeholders. SEED-SET models both evaluation types separately with hierarchical Gaussian Processes, and uses a novel acquisition strategy to propose interesting test candidates based on learnt qualitative preferences and objectives that align with the stakeholder preferences.
    We validate our approach for ethical benchmarking of autonomous agents on two applications and find our method to perform the best. Our method provides an interpretable and efficient trade-off between exploration and exploitation, by generating up to $2\times$ optimal test candidates compared to baselines, with $1.25\times$ improvement in coverage of high dimensional search spaces.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes SEED-SET, a framework for ethical evaluation. The method decomposes system-level ethical assessment into a hierarchical Gaussian Process(GP) model consisting of an Objective GP $x \rightarrow y$ and a Subjective GP $y \rightarrow z$, and combines both layers within a single acquisition function that jointly accounts for mutual information and preference terms (Equation 2). The authors evaluate SEED-SET on Power Grid Resource Allocation and Fire Rescue tasks for ethical benchmarking of autonomous agents, showing that under limited sampling budgets, the method achieves an effective cost-benefit trade-off.
From a methodological standpoint, SEED-SET largely represents a structured integration of existing techniques: Variational Gaussian Processes (VGP), Bayesian Experimental Design (BED), and preference learning. Its novelty is moderate, yet introducing hierarchical modeling and a unified three-term acquisition design for ethical evaluation adds conceptual and practical value, despite relying on several handcrafted settings and strong assumptions.

### Strengths
1. The paper grounds subjective ethical criteria on observable outcomes through a hierarchical formulation (Objective GP $x \rightarrow y$; Subjective GP $y \rightarrow z$), rather than an end-to-end $x \rightarrow z$ mapping, which improves interpretability and sample efficiency.
2. Equation (2) formalizes the evaluation objective as a joint combination of objective information, subjective information, and a preference term, enabling an exploration-exploitation trade-off within the BED framework. The case-study results empirically support this formulation.
3. Addressing the stated challenges, "Measuring ethical behavior is difficult," "Value alignment is uncertain and dynamic" (though not fully elaborated), and "Ethical evaluation is expensive", the experiments on Power Grid and Fire Rescue demonstrate partial effectiveness.
4. Related work is comprehensive, and the overall structure is logically clear.

### Weaknesses
1. **Overly strong and weakly justified assumptions.** A1 fixes the policy $\pi$ and scenario space $\chi$ while assuming objective evaluations are accessible. A2 presumes users' preferences are truthful and stationary. In reality, preferences are uncertain and dynamic. These assumptions lack sufficient empirical or prior justification. Even if adopted for tractability, their rationality and limitations should be explicitly discussed, as they substantially affect the credibility of real-world ethical evaluation.
2. **Hierarchical modeling may introduce new risks.** Because subjective evaluation depends on the completeness of objective metrics, any omission or bias in these metrics may propagate to the preference layer. An end-to-end design could, in some situations, mitigate this dependency.
3. **Preference functions are manually defined.** The handcrafted weights ([0, −0.5, 1, 0] for Power Grid, [1, 0, 1] for Fire Rescue) require costly validation via TrueSkill, contradicting the stated motivation of cost efficiency and limiting scalability. The reliance on handcrafted weights weakens claims of generality.
4. **Potentially weak baselines.** It is unclear to me whether the baselines (Random, Single-GP, VS-AL-1/2) are sufficiently strong or representative of state-of-the-art Bayesian Optimization methods. More relevant or advanced baselines could strengthen the evaluation.
5. **Writing issues.**
    - Figure 1 appears early but is only explained in Section 4, with an overly brief caption.
    - The second challenge, "Value alignment is uncertain and dynamic," is not explicitly elaborated.
    - The term "both models" in the abstract is unclear until Section 4.
    - Many mathematical symbols are introduced before explanation, making it difficult for readers without a strong background to follow.

### Questions
1. Beyond empirical ablations, could the authors provide stronger theoretical motivation and analysis for each term in Equation (2)?
2. How generalizable is SEED-SET to other domains or multi-stakeholder settings? Can the approach maintain its performance across different ethical evaluation contexts?
3. How is the second challenge, "Value alignment is uncertain and dynamic", explicitly addressed in the proposed framework? The linkage between this challenge and the methodology remains unclear.
4. Please improve the clarity of your writing:
    - Move Figure 1 closer to Section 4 or expand its caption;
    - Define "both models" early;
    - Provide explicit explanations of symbols upon first appearance;
    - Justify Assumptions A1–A2 either empirically or via references to prior work.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents SEED SET, a method for testing ethical alignment in autonomous systems. 

In this paper, an autonomous system is assumed to be a black box system that gets some input values (a scenario $x \in \mathcal X$) and outputs a set of observation values ($y\in\mathcal Y$). A stakeholder, external to the system, ranks the ethical fitness of those observations, providing a real number. 

The modelling assumptions are that there exists an observation function $f_{obj}:\mathcal X \to \mathcal Y$ and an ethical evaluation function $f_{subj}:\mathcal Y \to \mathbb R$. The goal is to approximate the function $f_{subj}\circ f_{obj}:\mathcal X \to \mathbb R$, that outputs the ethical fitness of the system for each scenario.
 
For testing, we have different access to the observation and the ethical fitness functions. We can probe the observation function $f_{obj}$ and get noisy observation pairs $(x, y)$, where $y$ follows a Gaussian distribution centered in $ f(x)$. Given pairs of observations ($y_1, y_2$), we can probe an oracle to tell whether $f_{subj}(y_1) > f_{subj}(y_2)$ or vice versa.

To solve this problem, the paper proposes a Bayesian experimental design loop, where the objective and subjective functions are modeled as Gaussian processes, and at each time, the scenario to probe is the one that maximizes the expected gain of information.

In the experimental section, the paper reports on the implementation of this method on two case studies: a problem of resource allocation in a power grid and a problem of drone navigation in a rescue mission. As baselines, they compare with simple random allocation and more convoluted recent methods, also using Gaussian processes. They find that they are able to generate significantly more informative test scenarios than the baselines.

### Strengths
S1. Interesting and relevant problem for the community.

S2. Well-founded and technically sound solution.
The proposed framework builds on solid Bayesian experimental design principles and uses hierarchical variational Gaussian processes to decompose measurable system outcomes and ethical judgments. The mathematical formulation is sound, and the probabilistic modeling choices are well justified.

S3. Solid and well-executed experimental evaluation.

### Weaknesses
W1. The conceptual formulation of ethical alignment feels oversimplified and somewhat inconsistent with prior literature.

- The paper models ethical evaluation as a single real-valued score, whereas ethical considerations are often multi-dimensional or context-dependent, similar in structure to the scenario and observation spaces.

- In much of the ethical alignment literature, alignment is defined relationally, i.e., as the consistency between an agent's maximizing reward and an ethical ranking coming from other stakeholders. 

- The use of "objective" and "subjective" terminology is confusing: it implies that ethical judgment is inherently subjective, which contradicts the assumption of a fixed function $f_{subj}:\mathcal Y \to \mathbb R$ representing a universal (a.k.a. objective) ethical ranking.

W2. The proposed solution offers limited methodological novelty relative to existing Bayesian optimization and preference-learning approaches.

### Questions
Q1. Instead of getting the ethical fitness of an observation as a numerical grade from the oracle stakeholder, this paper proposes a pairwise elicitation middle layer. One of the main reasons to do so is that different stakeholders may have different grading scales, so getting pairwise comparison instead of single grades is more robust. However, since you are using an LLM (the same one) for making the ethical alignment evaluation, wouldn't it be more efficient and as robust to get the grade and avoid the pairwise comparisons? Are there any other compelling reasons for this design choice?

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
4

### Summary
The paper proposes SEED-SET, scalable framework for ethical evaluation of autonomous systems. 
It models evaluation as a hierarchy, an Objective GP learns the mapping from scenario x to measurable outcomes y and a Subjective GP maps outcomes y to a latent utility z that captures stakeholder preferences.

### Strengths
1. Joint acquisition works. Ablations show the full objective (MI + preference) beats MI-only and Pref-only, especially in higher-D Fire-Rescue.
2. Good visualizations and discussions using practical case-studies.

### Weaknesses
1. Author talks about the difficulty in measuring ethical behaviour, especially the subjectiveness of ethics, but then LLM is explicitly told which objective is primary and secondary in subjective evaluation. Hence the underlying preference function is already defined and can be directly expressed as a simple linear combination of the objectives. There is nothing genuinely subjective or hidden for the GP to learn.
2. Narrow ablation coverage. Lacks robustness to LLM model/prompt/temperature changes/ LLMs inherent biases. Does not compare against composite-BO/BOPE-style baselines (Lin et al. 2022).
3. No human-rater study. LLM-proxy labels and a hand crafted ħ(y) serve as ground truth, then TrueSkill uses the same LLM’s outcomes for triangulation, which is not independent.

### Questions
please refer to weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
