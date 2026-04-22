# Adversarial Arena: Crowdsourcing Data Generation through Interactive Competition

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Post-training Large Language Models requires diverse, high-quality data which is rare and costly to obtain, especially in low resource domains and for multi-turn conversations. Common solutions are crowdsourcing or synthetic generation, but both often yield low-quality or low-diversity data. We introduce Adversarial Arena for building high quality conversational datasets by framing data generation as an adversarial task: attackers create prompts, and defenders generate responses. This interactive competition between multiple teams naturally produces diverse and complex data. We validated this approach by conducting a competition with 10 academic teams from top US and European universities, each building attacker or defender bots. The competition, focused on safety alignment of LLMs in cybersecurity, generated 19,683 multi-turn conversations. Fine-tuning an open-source model on this dataset produced an 18.47\% improvement in secure code generation on CyberSecEval-Instruct and 29.42\% improvement on CyberSecEval-MITRE.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces **Adversarial Arena**, a public platform for generating high-quality, diverse synthetic data through structured, multi-turn adversarial competitions between independent teams of attackers and defenders. The core idea is to frame data generation as a competitive task: attackers attempt to elicit failures (e.g., generating unsafe code), while defenders aim to produce robust, correct responses. This interactive process, orchestrated over multiple tournament rounds, naturally produces complex, multi-turn conversational data.

### Strengths
* The paper proposes **Adversarial Arena**, a public competition platform capable of generating high-quality data, and clearly elaborates on various details regarding the platform's construction, operation, and data generation process.
* Based on this public competition platform, the paper constructs a dataset for cybersecurity alignment, and experiments demonstrate the promising fine-tuning performance achieved using this dataset.

### Weaknesses
* The academic contribution of this paper is limited. The authors primarily emphasize the promising fine-tuning performance of their constructed dataset for safety alignment. However, they neither propose a new challenge (e.g., novel problems or paradigms within the safety alignment domain) nor deliberately construct data to address a specific, existing problem. While diversity is emphasized, the authors fail to provide a detailed explanation of what specific diversities were achieved. Furthermore, the authors act primarily as platform builders; the contributions of the participants and the details of their solutions lack systematic elaboration and summarization.
*   The experiments in this paper are insufficient. For instance, the fine-tuning experiment in Table 2 lacks baseline comparisons. It would be crucial to compare against recent or classic datasets for cybersecurity alignment to understand how fine-tuning on those datasets impacts a model's resilience to attacks.
*   The analysis of data diversity is somewhat superficial. Relying solely on t-SNE visualizations or embedding similarities to measure diversity and bias cannot effectively reveal novel problems or paradigms. (Even the same security issue, when phrased differently, can lead to significant shifts in vector representations). The authors should focus more on summarizing what new security issues or novel attack paradigms emerged from the competitive interactions among participants on their platform.

### Questions
Consistent with the Weaknesses section, I do not recommend the authors to proceed with a rebuttal. If they choose to do so, please just provide targeted responses addressing the points raised in the Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Adversarial Arena, a framework for crowdsourcing high-quality and diverse LLM training data through adversarial competitions between teams. The key contributions are:

1. Adversarial Arena, a general framework where "defenders" aim to make models perform well on tasks of interest while "attackers" try to elicit failures, with multiple tournaments allowing iterative improvement
2. An instantiation of this framework for cybersecurity tasks with 10 university teams (5 attackers, 5 defenders) competing over 4 rounds
3. A resulting dataset of ~19k labeled multi-turn conversations that, when used for fine-tuning, improves secure code generation by 18% and cyberattack assistance refusal by 24% (on CyberSecEval benchmarks)

### Strengths
1. **Interesting approach to data collection**: The adversarial competition framework presents an intriguing method for obtaining training data. While similar competition-approaches have been used before, the systematic application to data collection beyond adversarial tasks is novel.
2. **Good case-study design**: Given the complexity of organizing and orchestrating a multi-team competition, the case-study itself is a strong contribution. In particular, the authors also provide an orchestrator for such competitions with good documentation.
3. **Successful empirical validation**: The case study demonstrates practical effectiveness, with the resulting dataset yielding meaningful improvements on the cybersecurity task of interests. The validation methodology through fine-tuning experiments is sound, and the approach to testing diversity via semantic alignment is reasonable.
4. **Good presentation**: The paper is well-organized with helpful visualizations (Figure 3) and generally clear. The framework's instantiation for cybersecurity (Section 4), except for minor parts of the setup, is documented in depth.

### Weaknesses
**Main points**

1. **Lack of truly adaptive attacks**: The most significant limitation is that attackers and defenders interact simultaneously rather than sequentially. The paper mentions this limitation on L428; however, it only mentions either fully online settings or more frequent tournaments as a solution. In practice, defenders must commit first, allowing attackers to adapt. Hence, a more realistic approach would be turn-based, such that attackers get access to all defenses before submitting an attack strategy. The current framework hence artifically limits attackers (and thereby data quality), and it is not clear to me if the orchestrator can easily be adapted to such a turn-based setting.
2. **Scalability concerns due to manual labor**: The approach requires substantial effort from many participants (e.g., 10 academic teams in the case-study). It's unclear whether this scales to the diverse set of tasks current LLMs have to support, thus limiting the practical impact of Adversarial Arena. For example, one big competition might see a lot of participation from a diverse set of teams, but multiple weekly competitions that span many tasks might see quickly diminishing interest.

**Minor issues**

3. **High-level framework with limited novelty**: The framework itself (Section 3) is relatively abstract; it could be helpful to make it more prescriptive. The paper's significant contribution, in my opinion, is more the instantiation of this framework on cybersecurity tasks and the resulting dataset/code, not necessarily the high-level approach. In particular, using an attacker-defender setting to obtain data or improvements has been explored before (e.g., [Bartolo et al., 2020](https://arxiv.org/abs/2002.00293), [Debenedetti et al., 2024](https://arxiv.org/abs/2406.07954), or the [Generative AI Red-Teaming Challenge](https://humane-intelligence.org/get-involved/events/defcon-2023-overview/)). The core idea of instantiating a general adversarial framework including *both* attackers and defenders is still novel and has, to the best of my knowledge, not been done before. Nevertheless, I think contextualization with existing approaches would be appropriate.
4. **Diversity measurement limitations**: The diversity penalty for the case-study (L298-L307) captures only lexical diversity through BLEU scores, not semantic diversity of attack strategies. While Section 4.3 analyzes semantic diversity post-hoc, this isn't incorporated into tournament scoring; doing so would potentially better align incentives.
5. **Mixed task evaluation**: The case study combines two distinct tasks (eliciting vulnerable code and cyberattack assistance), leading to attackers focusing on the easier target (see e.g., the last paragraph of Section 4.4). Separate tournaments or independent grading per task could avoid such exploitation.
6. **Limited orthogonality testing**: Lines 182-184 emphasize testing utility on orthogonal tasks, but evaluations only examine related coding/cybersecurity tasks rather than truly orthogonal capabilities like general knowledge or reasoning. Even if the model should only work for coding tasks, I believe defenses should still be evaluated on general-purpose coding questions truly orthogonal to cybersecurity.
7. **Missing details of case-study setup**: While the evaluations of the case-study are described in detail, some points about the setup were unclear to me; see questions.

### Questions
1. Do the authors plan to publicly release the collected dataset and orchestration code? This would significantly enhance the contribution's impact.
2. Can the framework be modified to support turn-based interactions where defenders release their defenses first, allowing attackers to adapt? This would better reflect real-world scenarios and potentially yield stronger attacks (and hence better data).
3. Were defender teams allowed to fine-tune the ChallengeLLM, or were they restricted to prompting and auxiliary models?
4. Is the 45-second latency budget (L244) per conversation, per completion, or aggregated differently? Does this constraint apply to both attackers and defenders?
5. Does each tournament consist of single or multiple conversations per attacker-defender matchup?

### Soundness
4

### Presentation
3

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
The paper presents Adversarial Arena, a framework for crowdsourcing high-quality conversational datasets through structured adversarial competitions between “attacker” and “defender” teams. Each attacker attempts to elicit failures from defender models, while defenders aim to respond safely and effectively. The system reportedly produces diverse multi-turn datasets, demonstrated on a cybersecurity alignment task involving 10 university teams and ~19k conversations. Fine-tuning an open-weight model on the resulting data improves secure code generation and safety benchmarks.

### Strengths
1. Creative Framework: The notion of gamifying data generation through an attacker-defender structure is intuitively appealing and could inspire future collaborative or competitive data collection paradigms.
2. Scale and Engineering: The authors managed to coordinate 10 research teams and generate ~20k labeled dialogues, a notable engineering effort demonstrating feasibility at scale.
3. Empirical Evidence: The fine-tuning results (18.47% and 29.42% gains on security benchmarks) empirically confirm that the generated dataset is at least useful for improving safety-aligned code generation.

### Weaknesses
1. Domain Dependence.  The framework is tightly coupled to the cybersecurity task and depends heavily on fixed evaluation pipelines and manual annotation templates. The system’s success metrics rely on specific types of code vulnerabilities, which may not generalize to other domains like dialogue safety, factuality, or reasoning. The “attacker–defender” framing works mainly because cybersecurity naturally lends itself to adversarial setups; its general applicability remains unconvincing.
3. Weak Signal and Limited Generalization. The feedback signal for guiding attackers and defenders is simplistic and weak. It does not ensure meaningful improvement across rounds beyond superficial variation. While the paper emphasizes “diversity” and “richness,” it lacks concrete evidence that such signals lead to systematically better or more representative data. Furthermore, improvements are shown only for one fine-tuning case without ablation or analysis of data quality vs. quantity effects, limiting confidence in broader generalization.

### Questions
Can this idea apply to other domains for data collection?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Adversarial Arena, a tournament-style framework that crowdsources multi-turn conversational data by pairing “attackers” (prompt generators) against “defenders” (LLMs/agentic systems) and uses an automated evaluator to label each conversation as attack/defense success. A real-world case study on cybersecurity alignment involved 10 university teams (5 attackers, 5 defenders) across multiple tournaments, yielding 19,683 labeled conversations. The authors define ranking incentives (e.g., normalized ASR with a diversity multiplier; utility-aware defense score), show semantic diversity across teams/rounds (t-SNE & cosine-based analyses), and report that fine-tuning Mistral-7B-Instruct on curated subsets improves secure code generation (+18.47% on CyberSecEval-Instruct) and refusal of malicious cyberactivity (+29.42% on CyberSecEval-MITRE). The framework is backed by a serverless orchestrator (AWS Lambda/SQS/DynamoDB) to run large asynchronous, multi-turn matches.

### Strengths
Operational, repeatable framework for large-scale, multi-turn adversarial data creation with clear incentive design (normalized ASR, utility-aware defense scoring).
Demonstrated scale & impact: ~20k labeled dialogs; measurable gains on CyberSecEval-Instruct/MITRE after SFT on curated subsets.
Diversity evidence: attacker/defender/tournament-level separation via cosine distances and t-SNE plots; qualitative differences across teams and rounds.
Robust engineering: serverless orchestrator (Lambda/SQS/DynamoDB) enabling asynchronous, fault-tolerant multi-turn matches and batching, with explicit guarantees/trade-offs.
Transparency about limitations (imperfect evaluators; incentive timing; vulnerable-code vs malicious-intent skew) and concrete mitigations.

### Weaknesses
Evaluator dependence / label noise: Reliance on a single static analyzer (CodeGuru) risks false positives/negatives; human labeling process lacks detailed inter-annotator agreement (IAA) stats and calibration analysis. A small dynamic analysis slice or cross-tool ensemble would strengthen claims.
Diversity metric choice: BLEU captures lexical variety but can miss strategy-level novelty; the paper mentions considering embeddings, yet final ranking uses BLEU—an ablation comparing both (and their effect on attacker behavior) is missing.
Utility normalization and trade-offs: Defenders’ scores are aggressively penalized by utility drops, but details on the utility test construction, coverage, and ceiling effects (capping at base model) could use deeper analysis and sensitivity checks.
Generalization beyond cybersecurity: While the Discussion argues generality, only one ToI is empirically validated; even a small second domain (e.g., hallucination reduction or refusal over-agreement) would bolster generality claims.
Outcome attribution: The SFT improvements are compelling, but a breakdown by data slice (attacker team, tournament round, conversation length/turns, vulnerability type) would clarify which arena features drive the gains.

### Questions
Evaluator robustness. Can you report IAA (e.g., Cohen’s κ) and disagreement resolution for the 3-annotator panel? Any spot-checks comparing CodeGuru with a second static analyzer or dynamic tests on a held-out subset?
Diversity incentive. Why finalize on BLEU over embedding-based diversity for the ranking signal? Provide an ablation where attacker rankings and dataset properties are recomputed with an embedding-based metric.
SFT details. Share fine-tuning hyperparameters, data filtering (e.g., max turns/code length), and a per-slice contribution analysis (by attacker/defender/tournament). Do longer multi-turn dialogs help more?
Utility suites. Describe construction, difficulty, and overlap with tournament dialogs; include sensitivity of defender rankings to different utility weightings or removing the cap at base model.
Attack coverage balance. You note skew toward vulnerable-code attacks; did you trial multi-objective scoring (e.g., harmonic mean across “malicious-intent” vs “vulnerable-code” successes) and observe behavior changes?
Generalization. Any preliminary runs on a second ToI (e.g., hallucination or sycophancy) to show the arena transfers with minimal changes?

### Soundness
3

### Presentation
3

### Contribution
3
