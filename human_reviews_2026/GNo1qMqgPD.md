# VoxPrivacy: A Benchmark for Evaluating Interactional Privacy of Speech Language Models

- Decision: Accept (Poster)
- Scores: 4, 10, 4, 8

## Abstract
As Speech Language Models (SLMs) transition from personal devices to shared, multi-user environments such as smart homes, a new challenge emerges: the model is expected to distinguish between users to manage information flow appropriately. Without this capability, an SLM could reveal one user’s confidential schedule to another—a privacy failure we term **interactional privacy**. Thus, the ability to generate speaker-aware responses becomes essential for SLM safe deployment. Current SLM benchmarks test dialogue ability but overlook speaker identity. Multi-speaker benchmarks check who said what without assessing whether SLMs adapt their responses. Privacy benchmarks focus on globally sensitive data (e.g., bank passwords) while neglecting contextually sensitive information (e.g., a user’s private appointment). To address this gap, we introduce **VoxPrivacy**, the first benchmark designed to evaluate interactional privacy in SLMs. VoxPrivacy spans three tiers of increasing difficulty, from following direct secrecy commands to proactively protecting privacy. Our evaluation of nine SLMs on a 32-hour bilingual dataset reveals a widespread vulnerability: most open-source models perform close to random chance (around 50\% accuracy) on conditional privacy decisions, while even strong closed-source systems still fall short on proactive privacy inference. We further validate these findings on Real-VoxPrivacy, a human-recorded subset, confirming that the failures observed on synthetic data persist in real speech. We also demonstrate a viable path forward: by fine-tuning on a new 4,000-hour training set, we improve the model’s privacy-preserving capabilities while achieving fair robustness. To support future work, we are releasing the VoxPrivacy benchmark, the large-scale training set, and the fine-tuned model to help the development of safer and more context-aware SLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
VoxPrivacy is the first benchmark that evaluates interactional privacy for Speech Language Models (SLMs) in multi-user, spoken settings. It tests whether a model can keep user-specific secrets across three escalating tiers: (1) obeying explicit non-disclosure commands, (2) using the speaker’s voice as a key to disclose only to the original owner, and (3) proactively protecting privacy with no instruction by inferring sensitivity from content and context. Built from 7,107 utterances (32.86 hours, English/Chinese balanced) of high-quality synthetic audio with diverse speakers, VoxPrivacy pairs objective LLM-as-judge scoring with human validation. Across nine SLMs, most open-source systems hover around chance on conditional privacy decisions, revealing a core weakness in speaker-aware reasoning, not basic conversation. The authors also show a practical path forward: fine-tuning on a 4,000-hour training set substantially improves privacy compliance while preserving general abilities, though proactive, common-sense privacy remains challenging and vulnerable to spoofing attacks.

### Strengths
- Novel and interesting research questions
- Detailed and complete experiment
- Well-written paper

### Weaknesses
- Lack of practical motivations
- Data authenticity and generalization issues

### Questions
This paper presents VoxPrivacy, the first benchmark designed to evaluate interactional privacy for speech language models (SLMs) in realistic multi-user spoken scenarios. The authors construct a bilingual (English/Chinese) dataset containing 7,107 utterances (32.86 hours) of high-quality synthetic audio from diverse speakers and evaluate nine SLMs, including both open-source and proprietary systems. The benchmark employs a hybrid evaluation protocol combining LLM-as-judge scoring and human validation. Results show that open-source SLMs perform near random on speaker-conditioned privacy tasks, while closed-source models and fine-tuned versions achieve better compliance. Additional analyses reveal that the main bottleneck lies not in conversational ability but in speaker-aware reasoning and contextual privacy understanding. The paper is well-written in general. However, I do have the following concerns:

- Lack of practical motivation. 
Although it sounds fancy to leverage models' internal ability to conduct the permission verification, it is still common sense that models' responses are unreliable and random. Therefore, it may be possible to construct the permission systems instead of relied solely on large models to solve the questions. The authors also show in the paper that some jailbreak method can break the model recognition ability. Therefore, I am wondering given the current model architecture and model design, is it worth doing such a test? Should all permission-related issues be handed over to a dedicated permission system?

- Data authenticity and generalization issues
All speech data in VoxPrivacy are synthetic. WHile this choice ensures ethical safety, it limits the acoudstic and semantic diversity presnet in real spoken interactions, such as bacground noise, accent variation, and spontaneous interruptions. As a result, the benchmark may overestimate the model performance on the more realisitic data.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
10

### Rating Number
10

### Confidence
5

### Summary
Privacy studies have, up to now, mainly focused on individual users, whereas many problems of privacy occur only within the context of an interaction. This paper proposes a new dataset for evaluating interactional privacy when interacting with speech agents. This is an important contribution and a problem that has been overlooked in past works. 
The dataset is synthetic speech audio. This is a reasonable starting point, though obviously, actual recordings of human speech would be better. Recording sufficient amounts of such data, however, requires significant resources. Synthetically created speech is thus the obvious and reasonable simplification. 
Overall, I like the paper rather a lot and have only minor comments.

### Strengths
- High novelty; The addressed research problem is novel, and this is, as far as I know, the first dataset and methodology for evaluating interactional privacy.
- High quality; The proposed dataset is designed following principles of good design, the validation tests for the dataset are good, and the analysis of results is insightful. 
- Good clarity; Writing and argumentation are clear, with only minor blemishes.
- High significance: As this work addresses an important problem that has not been studied before, I believe that this can have a significant impact.

### Weaknesses
Main weaknesses:
- Argumentation: Building a dataset for SLMs was motivated by the fact that spoken dialogues have plenty of contextual information that is not available in the text only. This is true; speech is a much more informative representation than text, and my informed guess is that much of the information related to interactional privacy is available only in the voice (not in text). That said, as data is here created through synthesis from text, there is no way to confirm that information related to interactional privacy (beyond text) is included in the dataset. The question is thus whether the audio representation has any added value in comparison to text only, as long as the data is synthetic? I acknowledge that this is a difficult question that probably cannot be solved in a single paper and probably not even in a single doctoral thesis, but I would request a discussion about this issue in the paper.
- Data representativeness: A variation of the above argument is that synthetic data is always a proxy for real data, and special care must be taken to ensure or verify that it represents the true population adequately. This could be solved, for example, by adding audio from real human speakers to the test set. If the performance with synthetic and authentic samples are similar, then the synthetic data is sufficient. Again, this is not a demand but a proposal, given that it this modification can require significant resources, time and effort.

Minor comments:
- Fig 2, stage 3; spelling of "Instruction"
- Fig 2 is very dense, packed with information. I don't have a better solution to propose, but I just want to acknowledge that understanding the figure requires some effort.

### Questions
My suggestions and questions were included in the weaknesses box.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents VoxPrivacy, a benchmark for evaluating a privacy failure mode (termed "interactional privacy") in multi-user conversational context for speech language models. One user discloses secret information to the SLM, explicitly or implicitly, and the SLM needs to ensure this secret information is not disclosed when another user queries it. VoxPrivacy consists of three tiers of privacy evaluation, ranging from the most explicit (directly indicating secrecy) to the most implicit (inferring protection needs based on commonsense). Their evaluation found current SLMs, especially open-source ones, struggling with these tasks. Their fine-tuned model improved the privacy preservation capabilities without compromising the general capabilities.

### Strengths
- The paper examines contextual privacy leakage issues in speech language models and engages with the unique capabilities of SLM to process the voice which can uniquely identify a person. Hence, it makes sense to evaluate the end-to-end privacy protection for SLM.
- The paper develops a benchmark covering both direct and indirect indicators of privacy information to perform a thorough evaluation of the privacy protection capabilities of closed-source and open-source models.
- The evaluation reveals substantial gaps in the open-source models' privacy preservation capabilities.
- The fine-tuned model showed promising results improving the privacy capabilities while preserving the general capabilities.

### Weaknesses
- I can't find a realistic grounding for the privacy violations in the benchmark. The benchmark assembles the specification, instruction, and probing queries into a multi-turn dialogue, which corresponds to the situation when multiple users converse with the SLM in the same session. In these cases, people already have equal access to the output of the model, which means the sensitivity of information in the output should be determined by everyone present in the conversation, rather than just the speaker (or even people who are co-present in the context with access to the model output, regardless of whether they participate in the conversation). Also, in this case it's not natural for one speaker to describe their secret in front of another. This scenario hence feels contrived.
- In the Tier 3 example, I don't understand why "I'm worried about my medical results" implies that "medical results" should be considered a secret. I feel the bar might be set too rigidly and doesn't align with common sense.
- Gaps are exaggerated in the abstract "most models perform near random chance, about 50% accuracy on binary privacy decisions" — In fact, the closed models performed well in many of the tier 1 and 2 tasks, sometimes even better than the fine-tuned model.
- I feel it's inappropriate to call the LLM-as-a-Judge evaluation as objective evaluation and the human evaluation as subjective, because they are following the same criteria.

### Questions
- Can you discuss the validity of your threat model, and why the benchmark design appropriately reflects it?
- Can you explain how you determine the test cases in Tier 3, specifically what procedure did you follow to ensure they properly reflect social norms and commonsense?
- Are there any considerations about the potentially different cultural norms between English vs. Chinese speaking contexts? 
- How were the human annotations used to validate the LLM-as-a-Judge labels?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces VoxPrivacy, the first benchmark for assessment of interactional privacy in SLMs operating in shared, multi-user environments. The authors propose an evaluation framework that tests (1) direct command secrecy, (2) speaker-verified access, and (3) proactive privacy — reflecting practical privacy expectations in systems like smart home assistants. The benchmark consists of 7107 synthetic utterances (32.86 hours, English and Chinese). It covers a variety of privacy-sensitive categories constructed with a multi-stage pipeline and verified for quality and linguistic diversity. 9  SLMs are evaluated, showing that most open-source models perform no better than random chance on speaker-aware privacy tasks. Closed-source and fine-tuned models perform better but still show significant vulnerabilities. Analysis highlights crucial challenges in contextual integration and adversarial robustness. The authors release all resources, including the benchmark, a 4000-hour mixed-task training set, and a baseline model.

### Strengths
1. The paper addresses a novel and important problem in SLMs, interactional privacy in multi-user environments, which is underexplored. The introduction of the VoxPrivacy benchmark, based on a theoretically grounded definition of interactional privacy using Nissenbaum's Contextual Integrity.

2. The quality of the paper is good, the authors constructed a large-scale bilingual dataset with data synthesis, filtering, multi-model LLM generation, and human verification processes. The benchmark includes well-designed three-tiered tasks isolating distinct privacy capabilities, with comprehensive evaluations of nine state-of-the-art models, including open- and closed-source systems.

3. The paper is very well written.
 
4. The experimental results, based on a carefully constructed benchmark and clear evaluation methods, support the claim that current SLMs (especially open-source) struggle to reliably enforce speaker-based privacy.

5. The benchmark addresses critical safety and privacy challenges faced by SLMs in realistic shared environments such as smart homes. This benchmark will foster further research and development of practical solutions for privacy-preserving SLMs.

### Weaknesses
1. Synthetic dataset limitations (also acknowledged by the authors). The use of only synthetic, LLM-generated dialogues for privacy-sensitive utterances may reduce real-world relevance. The paper lacks user studies or comparisons with real data to confirm if synthetic secrets match actual privacy concerns.

2. Artificial dialogue structure.The fixed 3-turn dialogue pattern (secret statement → privacy instruction → probe) may not fully capture the richness and variability of natural conversations, including interruptions, multi-party interplay, and temporal gaps.
Speaker verification analysis and metric. Some details of speaker verification analysis are missing, also more conventional automatic speaker verification metric, i.e. equal error rate,  would be more appropriate.

3. Limited fine-tuning analysis. The construction of the 4000-hour mixed-task dataset for fine-tuning, including mixtures of tasks and proportions, is not fully justified. Ablation studies exploring the impact of different auxiliary tasks and the balance between privacy enhancement and general capability preservation are missing.

4. Cross-lingual performance gaps. The underperformance on Chinese w.r.t. English is observed but is not sufficiently analyzed, leaving open questions about multilingual robustness.

### Questions
1. Have the authors conducted user studies to validate that humans perceive the synthetic secrets as privacy-sensitive and expect SLMs to protect them?

2. Can the authors provide error analyses by secret categories and instruction phrasing to clarify which types of secrets leak most frequently?

3. Can the authors elaborate on the acoustic features or embeddings models use for speaker verification and their limitations, particularly regarding spoofing attacks?

4. Can the authors discuss potential extensions of the benchmark to more realistic, interactive multi-turn dialogues?
Regarding the multilingual aspect, do the authors have hypotheses or insights on why models underperform for Chinese?

### Soundness
3

### Presentation
4

### Contribution
3
