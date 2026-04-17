# LLM-Based Social Simulations Require a Boundary

- Decision: Reject
- Scores: 2, 2, 2, 4

## Abstract
This work argues that large language model (LLM)-based social simulations must establish clear boundaries to meaningfully contribute to social science research. While LLMs offer promising capabilities for modeling human-like agents compared to traditional agent-based modeling, they face fundamental limitations that constrain their reliability for social pattern discovery. The core issue lies in LLMs' tendency toward an "average persona" that lacks sufficient behavioral heterogeneity, a critical requirement for simulating complex social dynamics. We examine three key boundary problems: alignment (simulated behaviors matching real-world patterns), consistency (maintaining coherent agent behavior over time), and robustness (reproducibility under varying conditions). We propose heuristic boundaries for determining when LLM-based simulations can reliably advance social science understanding. Our analysis reveals that these simulations are most valuable when focusing on collective patterns rather than individual trajectories, when agent behaviors align with real population averages despite limited variance, and when proper validation methods confirm simulation robustness. We provide a practical checklist to guide researchers in determining the appropriate scope and claims for LLM-based social simulations.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents counterarguments against existing LLM-based social simulation work through extensive research and analysis of current approaches. This paper focuses on three key issues: alignment, consistency, and robustness. Building on this foundation, the authors propose what they consider to be the criteria for high-value social simulation, centered around the three aforementioned evaluation dimensions.

### Strengths
1. This paper focuses on the rapidly evolving field of LLM-based social simulation and conducts extensive literature research to form its conclusions. The findings provide valuable reference and guidance for relevant researchers.
2. The narrative flow and textual logic throughout the text are coherent and logical, making it easy to read and comprehend.

### Weaknesses
1. This paper is purely perspective-based and does not introduce any new techniques or present significant new experiments. It appears to fall outside the scope of the ICLR community and would be better suited for publication in journals or workshops that are more receptive to such work.
2. It is questionable whether the three evaluation aspects of alignment, consistency, and robustness proposed in this paper possess any particular characteristics specific to LLM-based social simulation. First, as emphasized by the author throughout the article, consistency and robustness are fundamental requirements for any simulation system. And alignment with the real world remains the ultimate goal of any predictive or simulation effort. Therefore, how does this paper demonstrate the innovation of the proposed evaluation system, and is this evaluation system unique to LLM-based social simulations?
3. Throughout this paper, certain sections cite literature from before the advent of LLMs as supporting material, which appears to fall outside the scope of the discussion presented herein.

### Questions
1. Can an evaluation method be designed based on the proposed evaluation framework to evalulate existing LLM-based social simulation work?
2. Does the evaluation framework proposed in this paper account for the changes introduced by LLM to social simulation?
3. Based on my understanding of the articles cited by the author, should the title of this paper be “Agent-based Social Simulation Require A Boundary”?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper discusses the background of social simulations using computational methods like agent-based modeling (ABM) to understand complex social phenomena, highlighting ABM's limitations in adaptability and representing human-like behaviors. It is motivated by the potential of large language models (LLMs) to create more flexible agents, but argues for establishing boundaries to ensure these simulations reliably contribute to social science by focusing on pattern discovery and hypothesis generation rather than replication or prediction. Key challenges include LLMs' tendency toward an "average persona" with low behavioral heterogeneity, misalignment with real-world patterns, inconsistency over time, and lack of robustness under perturbations. The solutions proposed involve heuristic boundaries emphasizing collective patterns over individual trajectories, alignment with population averages despite limited variance, and rigorous validation methods, along with a practical checklist to guide researchers on appropriate scopes and claims.

### Strengths
1. The heuristic boundaries and checklist offer actionable guidance for defining simulation scopes. They focus on collective patterns and validation availability. This bridges AI capabilities with social science needs effectively.

2. The work synthesizes extensive literature to position LLM simulations responsibly. It advocates for avoiding overclaims and focusing on beneficial applications. This fosters interdisciplinary collaboration between AI and social science fields.

### Weaknesses
1. Empirical demonstrations are absent as the paper relies solely on critiques of existing studies. No original simulations are conducted to illustrate the proposed boundaries. This limits the ability to verify the framework's practical utility.

2. Potential differences across LLM models are not differentiated in the analysis. Assumptions about universal limitations may not hold for all models or future versions. This could lead to overly broad generalizations.

3. Quantitative metrics for measuring boundaries like variance or alignment are missing. Evaluations rely on qualitative assessments. This hinders objective comparisons across simulations.

4. Accessibility is reduced by dense terminology assuming prior knowledge in both AI and social science. Key concepts like "average persona" could be clarified further. This may limit its reach to interdisciplinary audiences.

5. Overall, the major problem of this work is that its scope is too wide. It is very difficult to figure out the LLM boundary in one work. I think authors should consider shrink the scope, discuss the LLM boundary in some specific aspects and using experiments to provide actionable practical insights, not only the descriptions.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a survey and reflection on large language model-based social simulation. The authors review existing studies, discuss the boundaries of what large language model-based social simulations can or cannot achieve, and propose a checklist of methodological considerations such as alignment, temporal consistency, and robustness. The stated goal of the paper is to help researchers design more credible and interpretable social simulations that can contribute to social scientific understanding rather than merely reproducing known behaviors.

### Strengths
* The topic is timely. Large language model-based social simulation is a rapidly emerging area that indeed requires systematic methodological reflection.

* The paper is clearly written and well-organized.

* The authors try to connect social scientific perspectives with large language model agent research, which could be informative for newcomers to the field.

### Weaknesses
1. Lack of novel research contribution.

The paper primarily summarizes and comments on existing works. It does not introduce a new theoretical framework, formal model, or empirical study. ICLR normally expects some form of novel insight, methodology, or evaluation rather than a descriptive review.

2. Insufficient depth and evaluation.

The proposed “boundary checklist” remains conceptual and is not validated through adequate case studies or quantitative analysis. Without such evidence, it is difficult to assess its utility or correctness.

3. Ambiguous positioning relative to ICLR scope.

The work fits better as a perspective or survey article rather than a research contribution in machine learning. ICLR generally values algorithmic, theoretical, or experimental advances, and this paper focuses more on methodological reflection and normative guidance.

4. Some important literature is missing.

The paper omits several important studies. For example, 

[1] Gabriel, Iason, et al. "Who’s to blame when AI agents mess up? We urgently need a new system of ethics." Nature (2025): 38-40.

[2] Kozlowski, Austin C., and James Evans. "Simulating Subjects: The Promise and Peril of Artificial Intelligence Stand-Ins for Social Agents and Interactions." Sociological Methods & Research (2025): 00491241251337316.

[3] Grossmann, Igor, et al. "AI and the transformation of social science research." Science 380.6650 (2023): 1108-1109.

[4] Bail, Christopher A. "Can Generative AI improve social science?." Proceedings of the National Academy of Sciences 121.21 (2024): e2314021121.

### Questions
As the field of large language model-based agents for social science is still emerging, there are already several pioneering studies that reflect on its implications for social research and highlight methodological caveats when using such simulations. Some of these important works are missing from the current paper (for example, those listed above). Could the authors clarify how their contribution differs from and advances beyond these prior reflections, both conceptually and methodologically?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a position and analysis of the use of large language models (LLMs) for social simulation. The authors argue that for LLM-based simulations to be meaningful to social science, researchers must recognize and operate within clear boundaries. The paper identifies three primary "boundary problems": alignment (do simulated patterns match reality?), consistency (does an agent maintain its persona/behavior over time?), and robustness (is the simulation reproducible?).

The paper's core thesis is that current LLMs are constrained by an "average persona" that exhibits low behavioral variance (lacks heterogeneity), which is a fundamental limitation for simulating complex social dynamics. The authors use a mean-variance framework to analyze this alignment problem. They conclude by proposing "heuristic boundaries," chiefly that researchers should focus on collective patterns rather than individual agent trajectories, and that claims are most reliable when the mean of the collective behavior aligns with human data, even if the variance is low.

### Strengths
1. The paper's greatest strength is its clear, concise, and persuasive writing. It's an excellent summary of the key challenges in the field.

2. The "mean-variance" analysis of the alignment problem is a useful and intuitive way to decompose the "average persona" issue.

3. The paper's central message—that the field must be more critical, define its boundaries, and move beyond simple "replication"—is a crucial and timely corrective for the community.

4. The final "heuristic boundaries" (e.g., "focus on collective patterns, not individual trajectories") are practical, actionable, and well-justified by the preceding analysis.

### Weaknesses
1. The primary weakness is the paper's failure to acknowledge and differentiate itself from 
What Limits LLM-based Human Simulation: LLMs or Our Design? arXiv:2501.08579. This prior work identified the exact same "LLM-inherent limitations" (lack of diversity/heterogeneity and inconsistency) as the core bottlenecks. This submission does not offer a new conceptual leap beyond what is already present in that paper. For a position paper, where the idea is the main contribution, this overlap is a critical flaw. And this paper does not cite that paper.

2. As a position paper, it excels at identifying problems (lack of heterogeneity, consistency, robustness) but offers limited, high-level solutions. It does not propose new methods for how to measure robustness or how to quantitatively establish an "aligned mean," which are left as open challenges.

### Questions
1. Can you please explicitly state what the novel intellectual contribution of this paper is beyond the critiques already raised in that prior work? Why is your "boundary problem" framework (alignment, consistency, robustness) a more generative or insightful model than the "LLM-inherent vs. Design" framework?

2. You identify the "average persona" (low variance) as a core issue. Do you believe this is a fundamental and perhaps unsolvable limitation of the current maximum likelihood training paradigm? Or do you see it as a solvable engineering problem that can be addressed with better sampling, prompting, or fine-tuning techniques?


3. You propose focusing on "collective patterns." However, many crucial social phenomena (e.g., innovation diffusion, radicalization, market panics) are specifically driven by atypical agents or "long-tail" behaviors, not the average. Does your analysis imply that this entire class of social phenomena is fundamentally outside the boundary of what LLMs can reliably simulate today?

### Soundness
3

### Presentation
2

### Contribution
2
