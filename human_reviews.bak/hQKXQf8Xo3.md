# Multi-Session Client-Centered Treatment Outcome Evaluation in Psychotherapy

- Decision: Reject
- Scores: 3, 5, 5

## Abstract
In psychotherapy, therapeutic outcome assessment, or treatment outcome evaluation, is essential for enhancing mental health care by systematically evaluating therapeutic processes and outcomes. Existing large language model approaches often focus on therapist-centered, single-session evaluations, neglecting the client's subjective experience and longitudinal progress across multiple sessions. To address these limitations, we propose IPAEval, a client-Informed Psychological Assessment-based Evaluation framework that automates treatment outcome evaluations from the client's perspective using clinical interviews. IPAEval integrates cross-session client-contextual assessment and session-focused client-dynamics assessment to provide a comprehensive understanding of therapeutic progress. Experiments on our newly developed TheraPhase dataset demonstrate that IPAEval effectively tracks symptom severity and treatment outcomes over multiple sessions, outperforming previous single-session models and validating the benefits of items-aware reasoning mechanisms.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper focuses on evaluate therapeutic outcome via LLMs, aiming at solving a meaningful problem. It not only proposes an evaluation framework but also contributes a precious dataset. The method shifts the focus from therapist-centered, single-session assessments to a comprehensive, client-informed framework.

### Strengths
1. The paper focuses on evaluate therapeutic outcome via LLMs, aiming at solving a meaningful problem.
2. It integrates some psychology tests and metrics into LLM evaluation framework. It is very interesting.

### Weaknesses
1. The TheraPhase dataset is constructed via GPT-4. There is no human evaluation to guarantee the reliability of the dataset.
2. Experiments only conducted on one or two datasets. It would be better to evaluate on more datasets.
3. I think the proposed approach is not novel. Most parts in the approach are already existing, such as the psychometric test, client profile and PSDI.
4. In this paper, GPT-4o is selected as the Gold Model. However, as shown in Table 4, its performance is also poor. So there may be deviation in the results.

### Questions
See weaknesses above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This work presents a framework IPAEval, designed for evaluating the treatment outcome based on the cross-session clients’ context in clinical interview. It makes use of client information (including profile, session history), psychometric test before interview, client’s utterances, and “item-aware reasoning” (items classified by psychometric test, explanation) to obtain the score of PSDI (a metric adopted in Symptom Checklist-90). This work also develops a TheraPhase dataset to simulate cross-session interviews based on CPsyCoun dataset (multi-turn dialogue for chinese psychological counseling). Authors experimented on TheraPhase to evaluate the treatment outcomes between models across sessions, and further selected 2 datasets (High-low quality counseling and AnnoMI: English multi-turn) to evaluate the psychological assessment. They find that the IPAEval performs better than the single-session models on tracking the symptom severity and treatment outcomes.

### Strengths
1. Proposes a generalizable pipeline transforming from client information/dialogue to metrics (treatment outcomes)
2. Conducts an ablation study on the pipeline, specifically on removing the item-aware reasoning (items classified by psychometric test, explanation) to see if it affects the performance of tracking both psychological assessments and treatment outcomes.
3. Performs human annotations for three tasks (psychological assessments: symptom detection and severity assessment; treatment outcomes) to select the golden models as baselines

### Weaknesses
1. The main claim of the paper is to evaluate the treatment outcomes across sessions. However, there is no validation/exploration on the TheraPhase dataset to show that it is reflective of real world outcomes (i.e. to ensure the distributions of treatment outcomes across time is close to real-world clients). 
2. The TheraPhase dataset is based on a Chinese counseling dialogue dataset. The performance difference between models (e.g. Llama vs. Qwen) may be affected by the language's comprehension ability, rather than the tracking ability on treatment outcomes. The findings on this dataset is questionable.
3. The goal and contributions of the paper are unclear. The IPAEval is targeted to evaluate the treatment outcomes (by authors’ definition is the clients’ states etc) but the authors also evaluate psychological assessment (e.g. symptom detection and severity assessment).
4. The “item-aware reasoning” sounds misleading as it does not contain a multi-step thinking process on rationales but only one intermediate step of giving the rationale behind the item classification. Maybe consider changing the naming of reasoning or provide more concrete examples/analysis.

### Questions
Questions:
1. I’m confused by several terms – psychological assessment, symptom detection, severity assessment. It sounds to me that symptom detection, severity assessment are parts of psychological assessment. Can you clarify these definitions better?
2. I noticed that there are contrasting results between models on symptom severity in Figure 4 – any reason/hypothesis to explain this?

Minor:
1. Confused by the figure 2: the PSDI<0 should mean the good treatment outcomes but the sad face there indicates negative results. Can you clarify this?
2. Appreciate the formal definition on section 3 but it seems too long (~3 pages) for a 10 pages paper. Recommend to add more analysis/ elaboration on the results.
3. The fonts of the Figure 3 and Figure 4 are very small and cannot see them clearly without zooming.

Some useful but undiscussed related works are [1], [2] and [3]

In general, I can see that having a real-time therapeutic outcome assessment could be important for mental health practitioners. However, the paper lacks analysis/experiments on the effect of time (within session and cross-session) and lacks validation on the dataset (TheraPhase) to ensure it is close to real-world. I think having these may strengthen the contributions of the work and better differentiate this work to current works (like ClientCAST). Apart from that, the (arguably stat. insignificant) results (Figure 4: error bars overlapping before and after the reasoning prompt) in tracking therapeutic outcomes suggests a deeper analysis on the therapeutic outcomes e.g. which part of the therapeutic outcomes can be tracked better.

[1] https://arxiv.org/pdf/2407.00870

[2] https://arxiv.org/pdf/2405.19660

[3] https://arxiv.org/pdf/2310.13132

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper presents IPAEval, a new framework for evaluating therapy outcomes by focusing on clients’ experiences over multiple sessions rather than isolated single sessions. Unlike traditional methods that center on the therapist's role, IPAEval shifts the focus to clients, aiming to capture their mental health journey more fully.
Its key contributions include: 1) IPAEval assesses changes over time, using information from multiple therapy sessions to provide a big-picture view of clients’ progress. 2) It also evaluates each session individually, noting immediate shifts in the client’s state and responses. 3) The framework uses a sophisticated reasoning system to interpret client statements more accurately, particularly in aligning with psychometric test criteria.
Tested on the TheraPhase dataset, IPAEval showed improved performance through items-aware reasoning mechanisms in both symptom detection and outcome prediction tasks.

### Strengths
Originality: The paper offers a novel approach to psychotherapy outcome evaluation by shifting from a therapist-centered, single-session paradigm to a client-centered, multi-session framework. This approach, embodied in the IPAEval framework, introduces a unique perspective within mental health assessments by prioritizing the client’s evolving experience across sessions.

Quality: The paper's experimental design is comprehensive, testing multiple LLMs with various statistical metrics. The authors also conduct a detailed ablation study, which highlights the importance of the items-aware reasoning feature by isolating its impact on the framework’s overall performance. 

Clarity: The paper is well-organized and easy to follow. Each part of IPAEval—multi-session tracking, session-specific assessments, and items-aware reasoning—is clearly explained, and the figures and tables effectively illustrate IPAEval’s advantages and performance.

Significance: IPAEval addresses a key gap in mental health care by enabling evaluations of therapy outcomes from the client’s perspective, a shift that could substantially improve the quality of care. The framework provides practical tools that can streamline and enhance the outcome assessment process, helping practitioners to track treatment effectiveness over time and adjust interventions more responsively.

### Weaknesses
1: Despite having 110 and 800 client sessions available, the authors only annotated 30 for psychological assessment and 60 for treatment outcomes, using this small subset to establish a “Gold Model” for generating reference scores. This approach may introduce bias, as basing the Gold Model on such a small sample could affect the reliability of the labeled data, especially in representing the full diversity of client interactions.

2. The paper presents IPAEval as an improvement over models like ClientCAST and CPsyCoun, emphasizing its client-centered, multi-session approach. However, it doesn’t provide direct performance comparisons to back up these claims. While Table 1 outlines conceptual differences (such as focus and theoretical grounding), there’s no side-by-side benchmarking on tasks like symptom detection or outcome prediction. Without this empirical evidence, IPAEval’s advantages remain mostly theoretical. Adding direct comparisons with existing models would make the practical impact of IPAEval clearer and more convincing.

3. Figure 2, which aims to show the whole IPAEval framework, doesn’t clearly illustrate the Session-Focused Client-Dynamics Assessment module, especially how it tracks changes within a session from start to end. This mismatch between the figure and the text can make it confusing for readers trying to understand how session-specific changes are actually captured. Adding markers for initial and final assessments within each session would make the figure align better with the explanation in the text and help clarify how this part of the framework works.

### Questions
Could you provide more detail on why only 30 sessions were annotated for psychological assessment and 60 for treatment outcomes, given that these numbers are quite low relative to the available dataset? While I understand that annotating the entire dataset may not be feasible, expanding beyond this limited sample would likely improve the robustness of IPAEval’s evaluation.

Additionally, have you considered using multiple LLMs to cross-annotate the dataset? Cross-annotation with multiple models could help improve labeling quality and consistency while reducing the manual effort required, potentially increasing the reliability of the labeled data.

### Soundness
3

### Presentation
3

### Contribution
3
