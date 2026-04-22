# Work Zones challenge VLM Trajectory Planning: Toward Mitigation and Robust Autonomous Driving

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 8, 4, 4

## Abstract
Visual Language Models (VLMs), with powerful multimodal reasoning capabilities, are gradually integrated into autonomous driving by several automobile manufacturers to enhance planning capability in challenging environment.
However, the trajectory planning capability of VLMs in work zones, which often include irregular layouts, temporary traffic control, and dynamically changing geometric structures, is still unexplored.
To bridge this gap, we conduct the first systematic study of VLMs for work zone trajectory planning, revealing that mainstream VLMs fail to generate correct trajectories in 68.0\% of cases.
To better understand these failures, we first identify candidate patterns via subgraph mining and clustering analysis, and then confirm the validity of 8 common failure patterns through human verification.
Building on these findings, we propose REACT-Drive, a trajectory planning framework that integrates VLMs with Retrieval-Augmented Generation (RAG). Specifically,
REACT-Drive leverages VLMs to convert prior failure cases into constraint rules and executable trajectory planning code, while RAG retrieves similar patterns in new scenarios to guide trajectory generation.
Experimental results on the ROADWork dataset show that REACT-Drive yields a reduction of around $3\times$ in average displacement error relative to VLM baselines under evaluation with Qwen2.5-VL.
In addition, REACT-Drive yields the lowest inference time ($0.58$s) compared with other methods such as fine-tuning ($17.90$s).
We further conduct experiments using a real vehicle in 15 work zone scenarios in the physical world, demonstrating the strong practicality of REACT-Drive. 
Our code and demos are available on https://sites.google.com/view/react-drive.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper shows common VLMs fail on the work zone and analyze failure patterns on the ROADWork dataset. It proposes a framework called REACT-Drive that leverage constraint rules and RAG to use mitigation code for safe trajectory planning. Evaluations show REACT-Drive reduces trajectory prediction error for 3 times. Physical experiments on 15 scenarios are also conducted.

### Strengths
-the paper is overall well-written and easy to follow.

-a complete study on identifying VLMs’ weaknesses on work zone scenarios and propose an approach that largely mitigates the issue.

-though limited, the real-world evaluations strengthen the evaluation part.

### Weaknesses
-The scenario scope is relatively narrow (only the workzone scenario).

-It might be challenging for the framework (REACT-Drive) to be directly applied to handle more diverse failure scenarios.

-No closed-loop simulation is conducted for evaluation

### Questions
-How to adapt the REACT-Drive for more general failure scenarios?

-Have you tried to use VLMs in thinking mode? Will that reduce the errors?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes REACT-Drive, a novel end-to-end framework designed to address the significant limitations of vision-language models (VLMs) in trajectory planning for autonomous driving in construction zones. The core idea is to combine the reasoning capabilities of VLMs with a retrieval-augmented generation (RAG) mechanism to enhance planning robustness in complex and dynamic environments.
The framework operates through a two-stage process:
Offline stage: historical failure cases are transformed into executable constraint rules and mitigation code, building a searchable database.
Online stage: the system uses RAG to retrieve failure patterns similar to the current scenario and applies the corresponding mitigation code to guide trajectory generation.
A key innovation is the self-verification feedback mechanism, ensuring consistency between the generated code and the reference failure cases.

Experimental results on the ROADWork dataset and real-world physical scenarios demonstrate that REACT-Drive reduces trajectory prediction errors by about three times while maintaining high inference efficiency (0.58s), proving its potential for real-time deployment and practical applications.

### Strengths
- Problem-oriented and well-motivated: the paper clearly identifies pain points of VLMs in construction zone trajectory planning, supported by real accident cases, giving the work strong practical relevance.
- In-depth failure mode analysis: through systematic scene graph construction, subgraph mining, clustering, and manual verification, eight typical VLM failure modes are revealed, forming a solid analytical foundation.
- Innovative hybrid design: REACT-Drive effectively integrates VLMs’ generative capability with RAG’s retrieval mechanism, enabling the model to learn from historical experience and adapt to unseen complex scenarios.
- Robustness through self-verification: the drivability and destination constraint-based feedback mechanism ensures safety compliance and reduces erroneous trajectories.
- Excellent performance and efficiency: significant improvements in key metrics (ADE, FDE, CR) and extremely fast inference speed demonstrate clear real-time deployment potential.

### Weaknesses
- Limited scenario coverage: the model does not yet cover long-tail conditions such as extreme weather or nighttime construction, limiting generalization under diverse challenges.
Dataset and deployment constraints: evaluations rely mainly on the ROADWork dataset and self-collected data, without broader dataset diversity, and the system has not yet been tested on real autonomous vehicles.
- Reliability and safety of generated code: reliance on VLMs for generating mitigation code introduces potential risks in unseen or extreme cases, despite the self-verification mechanism.
- Maintenance of the failure mode library: as the eight failure modes were derived from current data, maintaining and extending the library for new construction configurations or dynamic events may become challenging.

### Questions
- The paper notes that extreme weather and nighttime construction zones were not systematically addressed. How can REACT-Drive’s vision encoder and VLM perception adapt in such low-visibility, perception-challenging environments? Would extra sensor modalities or specialized designs be required?
- For the offline “failure case mitigation code database,” how is its scale managed? As the database grows, will retrieval efficiency degrade? Are there mechanisms for updating or pruning outdated or inefficient mitigation codes?
- The self-verification mechanism depends on thresholds (e.g., ADE > 50px, FDE > 100px). How were these thresholds determined? Are they universally applicable across various construction zone types and safety levels?
Since the eight failure modes were manually verified, could future work automate or semi-automate the discovery of new failure modes to reduce human cost and improve adaptability?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper identifies the suboptimal trajectory planning performance of VLMs in work zones. To address this issue, this paper first conducts abnormal pattern analysis on failure cases through graph mining and human verification, and summarizes eight typical patterns. The paper further proposes the mitigation framework REACT-Drive based on RAG, which enhances the planning performance of VLM by retrieving the constructed mitigation code database. REACT-Drive demonstrated a significant improvement in planning performance and verified scalability in the physical environment.

### Strengths
1. This paper has a clear motivation to enhance VLM's planning performance in work zones, which is of practical value for autonomous driving.
2. The proposed abnormal pattern analysis and mitigation framework are well-designed and well-explained.
3. The experimental results verified the effect and efficiency of the proposed method.

### Weaknesses
1. The method involves many hyperparameters, but no ablation analysis is provided to validate the rationality of the parameter settings.
2. The paper does not provide a quantitative analysis of 8 failure patterns to prove their typicality, such as the distribution of individual failure patterns and the overall coverage rate in failure cases. Furthermore, it is unclear how many abnormal scenarios require direct handling by the VLM during RAG-based inference.
3. The explanation for the increased CR in P4 and P8 is rather superficial, lacking a analysis of the underlying causes and potential optimization strategies.
4. In the transferability experiment, further details are needed regarding the distribution of the 15 real-world work zone scenarios and a comparison other other mitigation methods.

### Questions
As discussed in weaknesses.

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
This paper presents a novel approach to addressing the failure of VLM in trajectory planning for autonomous driving, particularly in dynamic and complex work zone environments.

### Strengths
The authors systematically evaluate VLM-based planning performance on the ROADWork dataset, revealing a 68.0% failure rate and identifying eight common failure patterns. To mitigate these issues, they propose REACT-Drive, a framework that integrates RAG with constraint-based code generation to improve planning robustness and efficiency.

### Weaknesses
The entire REACT-Drive framework depends heavily on the outputs of a YOLOv12 detector fine-tuned on ROADWork and monocular depth estimation from MiDaS. Any perception errors, such as missing novel work-zone objects (enew barrier types or colored cones) or inaccurate depth estimates, would propagate through the scene graph, retrieval, and planning stages, fundamentally compromising system safety and reliability.

The decision to generate and execute Python code at runtime introduces considerable safety risks. In safety-critical systems like autonomous driving, planners are typically deterministic, rigorously tested, and verifiable. Allowing a VLM to generate executable code in real-time is highly unpredictable; even minor errors in logic or boundary conditions could lead to catastrophic failures. The current self-verification mechanism， which only checks destination proximity and drivability, is insufficient to ensure trajectory feasibility, dynamic stability, or interaction-aware behavior.

For the experiment, metrics such as ADE and FDE are reported in pixel space, which does not faithfully reflect real-world driving safety. A small pixel error may correspond to a dangerous deviation in the physical world.

Critical aspects of trajectory quality, including passenger comfort (jerk, acceleration), compliance with traffic rules (lane discipline), and interpretability, are not evaluated.

The comparison with a fine-tuned VLM baseline is arguably unfair, as REACT-Drive benefits from a database of failure cases. A more appropriate baseline would enable all models to access the same failure-case knowledge.

There is no comparison with classical or optimization-based planning methods, like MPC, leaving it unclear whether the proposed VLM-based complexity is necessary or beneficial compared to well-established methods.

The physical evaluation is conducted in an open-loop setting using only 15 scenarios (100 images). This is insufficient to support claims of generalization, especially for a long-tail problem. The study does not demonstrate performance in closed-loop simulation or real-world deployment, where interaction with other agents and control uncertainty become critical.

The pipeline involves multiple heavyweight components, including object detection, depth estimation, scene graph construction, subgraph matching, VLM-based code generation, and RAG retrieval. The end-to-end latency, including all perception modules, is likely to exceed acceptable limits for real-time autonomous driving (typically 100–200 ms), even if the planning module alone reports low latency.

The eight failure patterns are derived through clustering followed by human summarization. This process is inherently subjective, and it is unclear whether these patterns are comprehensive, mutually exclusive, or consistently identifiable—especially ambiguous ones like “overreaction to signs”.

The framework does not address how the failure-case database would be updated online or how it would handle multi-agent interactions.

The evaluation is limited to the ROADWork dataset. Broader validation on other benchmarks  nuPlan/Waymo) would better demonstrate generalizability beyond work zones. (Given that it is difficult in daily life to navigate solely through work zones, it would be highly unreasonable for this model to incur such significant computational overhead merely for work zone interactions. It should demonstrate strong generalisability.)

The dataset does not systematically include challenging conditions such as extreme weather or nighttime scenes, which are critical for assessing robustness.

From the perspectives of academic rigour and industrial deployment, it introduces excessive complexity and potential points of failure. The authors should reconsider the high-risk design of "run-time code generation". A more robust alternative would be to utilise cases retrieved via RAG to directly generate parameters for high-level semantic objectives or cost functions. These parameters could then be processed by a rigorously validated, deterministic optimiser to generate the final trajectory. This approach would incorporate VLM's semantic comprehension capabilities while ensuring the safety and reliability of the planning process.

In summary, while the paper addresses a relevant problem and presents a novel idea, fundamental concerns regarding the safety of the runtime code generation and the insufficient empirical validation prevent it from meeting the acceptance bar for ICLR in its current form. Significant additions to the experimental evaluation and a thorough reconsideration of the core planning paradigm are required.

### Questions
See the Weaknesses section

### Soundness
3

### Presentation
3

### Contribution
2
