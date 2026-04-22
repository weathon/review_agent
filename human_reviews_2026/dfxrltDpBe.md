# Can Large Language Models Really Recognize Your Name?

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 4

## Abstract
Large language models (LLMs) are increasingly being used to protect personal user data. These privacy solutions often assume that LLMs can reliably detect named entities and personally identifiable information (PII). In this paper, we challenge that assumption by revealing how LLMs can regularly overlook broad types of sensitive names even in short text snippets due to ambiguity in the contexts. We construct AMBENCH, a benchmark dataset of seemingly ambiguous yet real entity names designed around the name regularity bias phenomenon and embedded within concise text snippets containing benign prompt injections. Our experiments with state-of-the-art LLMs and specialized PII detection tools show that the recall of AMBENCH names drops by 20--40\% compared to more recognizable names. AMBENCH names are also four times more likely to be ignored in supposedly privacy-preserving LLM-powered text analysis tools adopted in the industry. Our findings showcase blind spots in current LLM-based privacy defenses and call for a systematic investigation into their privacy failure modes.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors argue that LLMs are increasingly used in privacy-preserving systems under the implicit assumption that they can recognize sensitive information such as human names. They aim to show that LLMs struggle to correctly identify rare human names that resemble other entities. By embedding real human names that are only one edit distance away from non-human entities into short sentences, the authors demonstrate that LLMs easily confuse them with non-human entities — Name Regularity Bias. Moreover, they show that when a text unintentionally contains sentences that LLMs may misinterpret as commands, it can negatively affect privacy-preserving summarization — Benign Prompt Injection. The authors conduct extensive experiments using multiple state-of-the-art LLMs, as well as tools such as Anthropic’s Clio System and Private AI.

### Strengths
While Name Regularity Bias in Named Entity Recognition has been widely studied, this paper takes an interesting approach by examining Name Regularity Bias and Prompt Injection from a privacy-preserving perspective, leveraging human names that closely resemble non-human entities — particularly through a systematic collection based on an edit distance of one. The conceptualization of benign prompt injection, which is non-malicious in nature, is also a contribution. Furthermore, the construction of over 60,000 data points using LLMs and the evaluation of performance across a variety of LLMs and NER methods are notable strengths of the paper.

### Weaknesses
Despite its interesting aspects, this paper suffers from several **fatal flaws**.

---

### 1. Issues in Dataset Construction

The first major problem lies in the dataset creation process.  
Since the primary contribution of this paper is the construction of a new benchmark, the benchmark generation pipeline is central to its value.  
However, the paper merely states that five templates were selected for each name type **without describing the criteria for template selection**.  
Moreover, there is no explanation of how the final human inspection was conducted.  
The “manual selection” process may have inadvertently amplified the adversarial nature of the benchmark.  
Inadequate manual validation could also have caused control failures in the subsequent human evaluation discussed later.

---

### 2. Poorly Defined Problem, Overclaims, and Lack of Novelty

The definitions of key problems are imprecise, the claims are overstated, and the novelty is limited.  
Core concepts such as **Name Regularity Bias (NRB)**, **Prompt Injection (PI)**, and **Personally Identifiable Information (PII)** detection are already well established.  
Rather than revealing new findings, this benchmark merely **reconfirms known phenomena** and does not propose practical mitigation strategies.

Although the authors frame the dataset’s contribution from a PII perspective, in reality, the benchmark only deals with **simple human names**.  
Each sample consists of two very simple sentences that contain almost no personal information beyond names.  
Thus, linking this benchmark to **PII performance** constitutes an overclaim, as the dataset does not provide sufficient complexity to analyze models’ PII-handling ability.

There are also issues regarding **baselines and fairness**.  
While the authors state that U.S. baby *first names* were used for the baseline (line 236), the benchmark itself appears to include *surnames* (and possibly non-U.S. ones) (line 214).  
If that is not the case, it should be clearly clarified.

Importantly, when both human and non-human entity names can fit grammatically into the same sentence, the task effectively becomes a **memorization problem** over diverse human names.  
From an application standpoint, such a problem could be trivially solved through exact matching, undermining the benchmark’s practical relevance.

---

### 3. Methodological Limitations and Serious Control Failures (Human Evaluation)

Although the paper claims to operate under *“non-adversarial settings,”*  the name selection restricted to **edit distance 1** is inherently adversarial,  and the template validation intentionally maximizes ambiguity.  This creates a substantial distributional gap from real-world data.  (Figure 5’s frequency distribution indirectly supports this concern.)
As a result, it becomes unclear whether the observed failures are truly due to **NRB** and **BPI**,  or merely reflect the **difficulty of overly ambiguous templates**.  The paper lacks ablation studies to disentangle these effects.  
In particular, the similar performance between *No BPI* and *BPI* conditions in Table 3 suggests the need for deeper analysis.  
Likewise, while the *average privacy audit score* differs significantly across baselines, the *% >= score* remains small — an inconsistency that warrants explanation.

**Table 2’s control groups** reveal serious flaws in human evaluation design.  
Control (*non-human*) entities are incorrectly classified as *PERSON* at 23–86% rates (average 40%), with *location* entities reaching 86% — indicating that linguistic cues (e.g., pronouns like “she”) override even obvious non-human names.  
Meanwhile, control (*human*) names achieve only 55–64% recall for *syndrome/mineral* types, suggesting **severe task confusion** that invalidates these results as reliable benchmarks. These control failures imply that the templates are over-engineered to create artificial difficulty, raising fundamental doubts about whether the benchmark measures realistic privacy vulnerabilities or merely **artifacts of adversarial design**.

---

### 4. Writing and Presentation Issues

The overall flow of the paper is unnatural and repetitive. The authors should revise for conciseness to improve readability and avoid redundancy.  All figures and tables should be rewritten to fully convey their intended information without omissions.  

In **Table 1**, clearly specify the criteria for boldface and include all abbreviations in full.  
The term **PII** appears in the text without first introducing its full form in the main text.

**Figure 4** is especially problematic:  
the caption **omits the y-axis description entirely** and fails to define the key metric of “consistency” anywhere in the paper.  
The x-axis explanation (“ratio of templates in which a name is classified as human”) assumes readers already know that each name appears in exactly five templates, creating unnecessary confusion.  
The vague takeaway that models are “inconsistent for at least 10%” lacks clarity — does “inconsistent” mean *consistency < 1.0*, *< 0.5*, or something else?  
Moreover, this claim **contradicts the visual evidence** in subfigures (b) and (d), which show far higher inconsistency rates.
This poor presentation raises questions about whether the manually selected templates were intentionally chosen  
to maximize the inconsistency effect.

Overall, Figure 4 exemplifies the broader issue throughout the paper:  critical experimental details are left to implicit understanding rather than explicit documentation.

---

### Questions
In addition to the weaknesses mentioned above,

did the authors consider other defense mechanisms of Clio (e.g., differential privacy in clustering)?  
As recent LLM applications often improve performance through agent systems or self-refinement techniques,
does the proposed method show any performance improvement under such approaches?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper explores how efficiently a variety of LLMs are capable of recognizing human names in the provided prompt context. Authors also proposed a dataset benchmark to evaluate the fact under the hood of privacy-preserving LLMs.

### Strengths
1.	This paper investigates a compelling issue within LLMs, efficient PII understanding in user interactions, specifically names, as well as the privacy implications under this.
2.	Proposed AmBench benchmark dataset with 60K data points, which consists of both real-world and synthetically generated scenarios.

### Weaknesses
1.	The paper lacks an explicit description of the problem and the use cases of the solution provided at the beginning of the paper. Thus, the readability lacks its quality. Describing a hypothetical scenario where the problem occurs and how the proposed solution may address the issue would also help.
2.	The problem setting is highly superficial. Does not provide a clear idea about the problem in a real-world environment.  
3.	This paper takes only one category of PII, i.e., names, into account for analyzing the issue. Does this PII misclassification/ misidentification issue exist in other PIIs, e.g., email, phone number? If so, to what extent?
4.	There is no solid evidence/ study/research outcomes provided behind the claims in lines 165-168?
5.	Why did not the authors consider the o1 or o3 reasoning models in the evaluation?
6.	The assumption made in the attack scenario stands on some assumptions which are somewhat unrealistic, e.g., the user’s conversation is included in Clio’s inputs. If the entire conversation is unveiled to the attacker, then he can directly refer to the conversation for names; no need to perform an attack.
7.	What are the potential solutions to this issue presented in the paper?
8.	The entire paper explores the vulnerabilities of SOTA LLMs in not efficiently identifying the names in the prompt context. What's the explicit/implicit relation of it with privacy leakage? It is not explained in a clear way.
9.	“LLMs are much worse at detecting ambiguous human names than popular ones.” The explanation of popular exists nowhere. Is it popular LLMs or popular names?
10.	In the abstract, line 20 should be explained in more detail on how the proposed dataset’s names are four times more likely to be ignored in the LLM-enabled privacy-preserving tools.

### Questions
Please follow the Weaknesses.

### Soundness
3

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
4

### Summary
This paper introduces AMBENCH, a benchmark designed to evaluate large language models’ ability to recognize ambiguous human names that closely resemble non-human entities. The dataset contains over 60K samples, covering human–nonhuman name pairs like Albanir–Albania and Versache–Versace, embedded in short ambiguous sentences. It systematically tests 12 different LLMs, revealing biases in name recognition and providing valuable insights into LLMs’ limitations in privacy protection and anonymization.

### Strengths
1. The paper clearly describes the benchmark construction process, offering valuable insights and methodological guidance for future benchmark design and related research on LLM safety and privacy.

2. The paper conducts a systematic and comprehensive evaluation across 12 large language models, covering both reasoning and non-reasoning, as well as open-source and closed-source models.

3. The study highlights an underexplored vulnerability—LLMs’ difficulty in recognizing ambiguous personal names, which provides a new perspective for understanding the limitations of LLM-based privacy protection and anonymization systems.

### Weaknesses
1. While the paper introduces an interesting benchmark for testing ambiguity in name recognition, the constructed scenarios appear somewhat unrealistic. In real-world contexts, human names rarely resemble non-human entities to such a degree (just one letter difference), and sentences where both usages are plausible are uncommon. As a result, the evaluated setting may not accurately reflect practical privacy risks, but rather test linguistic robustness under artificially ambiguous cases.

2. The benchmark’s scope is quite limited, as it only considers cases where human names are nearly identical to non-human entities. Such situations are rare in real-world applications. Consequently, the benchmark may have limited practical use, as few privacy or safety systems would realistically need to be evaluated under these highly specific conditions.
 
3. The reported issue appears less severe for state-of-the-art reasoning models. As shown in Table 1, reasoning-capable models (e.g., GPT-5-mini, Gemini 2.5, Claude 3.5) achieve substantially higher recall and lower false discovery rates than smaller or non-reasoning models. This suggests that the identified vulnerability may diminish as LLM reasoning abilities improve, limiting the long-term significance of the benchmark’s findings.

### Questions
Please see the weakness above.

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
3

### Summary
This paper examines how large language models (LLMs) handle non-adversarial contextual ambiguity in privacy-sensitive tasks. It identifies two factors—Name Regularity Bias (NRB) and Benign Prompt Injection (BPI)—and introduces AMBENCH, a benchmark of 12K human names orthographically or semantically similar to non-human entities. Evaluating 12 LLMs and commercial PII tools, the authors find a 20–40% recall drop on ambiguous cases, revealing persistent failures in LLM-based privacy detection systems.

### Strengths
- The paper is well structured, easy to follow, and clearly motivated.
- The experimental setup is comprehensive: the benchmark contains ~60K sentences and covers a wide range of LLMs and commercial tools, ensuring generalizability.
- Introducing the notion of non-adversarial ambiguity as a distinct failure mode in LLM privacy detection is novel and practically meaningful.

### Weaknesses
1. Semantic implausibility in "ambiguous" examples

Several examples presented in the paper appear syntactically valid but semantically implausible for human agents. These constructions make it difficult to disentangle whether model errors stem from representational bias or from the unnaturalness of the input sentences themselves. For instance:
- "I managed to find traces of Adomite at the work site." (Section 3) – The phrase "find traces of" usually applies to physical substances, not people.

These cases suggest that what the authors call contextual ambiguity may partly arise from semantic mismatch in the generated templates, rather than from genuine ambiguity in model reasoning.
Furthermore, while the authors justify such constructions as part of a “worst-case evaluation” (Section 7, Adversariality of AMBENCH), a legitimate worst-case benchmark should still preserve semantic plausibility. Evaluating models on linguistically implausible or pragmatically inconsistent inputs (e.g., “find traces of Adomite”) may conflate privacy reasoning failure with limitations in general language understanding. If real-world privacy systems were tuned to handle such implausible edge cases, their performance on normal data distributions would likely degrade severely due to excessive over-scrubbing and false positives.
This may weaken the benchmark’s construct validity, as it shifts the evaluation focus from assessing privacy reasoning to robustness under linguistically implausible inputs.

2. Ambiguity defined purely by orthographic similarity

The construction of ambiguous names relies solely on Levenshtein distance ≤ 1 between human and non-human entities (Section 4, Appendix B.1). This criterion captures surface-level resemblance but not semantic or pragmatic ambiguity. Consequently, many sentences resemble artificially mixed contexts where a human name is inserted into an otherwise non-human frame.

3. Manual template selection lacks transparency

Five templates per domain were "manually selected" (Appendix B.2), but the paper does not specify how ambiguity was judged or whether multiple annotators were involved. The absence of inter-annotator agreement raises concerns about reproducibility and objectivity of AMBENCH’s construction.

### Questions
- Some “ambiguous” examples (e.g., “find traces of Adomite”, “responsible for the infection”) appear semantically incompatible with human subjects. On what basis were these cases considered ambiguous? Did you validate their naturalness through human annotation or any linguistic plausibility check?
- Since ambiguity is defined by Levenshtein distance ≤ 1, how do you distinguish genuine contextual ambiguity from mere lexical similarity? Would you consider adding a contrastive setup (e.g., placing the same names in clearly human-centric contexts) or linguistic plausibility metrics (e.g., selectional preference or perplexity) to validate semantic naturalness?
- The five domain templates were manually selected (Appendix B.2). Could you elaborate on the selection process and whether inter-annotator agreement was measured to ensure reproducibility?

### Soundness
3

### Presentation
3

### Contribution
3
