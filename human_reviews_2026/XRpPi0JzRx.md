# Strategic AI Training Sabotage: State Attacks on Advanced Systems' Development

- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Much attention has been given to the possibility that states will attempt to steal the model weights of advanced AI systems. We argue that in most situations, it is more likely that a state will attempt to sabotage the training of the models underpinning these systems. We present a threat modelling framework for sabotage of AI training, including both the necessary technical background and a taxonomy of strategic considerations and attack vectors. We then use this to examine different attacks and assess both their technical plausibility and the mitigations required to defend against them.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper “Strategic AI Sabotage: State Attacks on Advanced Systems’ Development” examines how state actors could target the training pipelines of advanced AI systems as a form of strategic sabotage. It proposes a threat modeling framework outlining sabotage goals, attack vectors, and motivations, analyzing methods such as data and model poisoning that could degrade performance or delay rival progress. Drawing parallels to Stuxnet and Cold War covert operations, the paper situates AI sabotage within modern grey-zone warfare and calls for stronger defensive measures, cooperation, and safeguards to protect AI development from state-level interference.

### Strengths
The paper is highly original, introducing the idea of state-level AI training sabotage as a new and important threat. It effectively blends technical and geopolitical analysis, supported by clear threat modeling, a well-defined taxonomy, and credible historical analogies. The writing is clear and well-structured, with visuals that enhance understanding. Overall, it makes a significant and timely contribution by spotlighting an overlooked risk at the intersection of AI safety, national security, and global governance.

### Weaknesses
The paper is largely conceptual, lacking empirical validation or quantitative modeling to support its threat scenarios. Some proposed sabotage methods are technically plausible but insufficiently detailed, leaving questions about real-world feasibility. The defensive measures are high-level and could be strengthened with concrete detection or mitigation strategies. Adding empirical case studies or simulations would make the work more practical and evidence-based.

### Questions
Can the authors provide quantitative or simulated case studies to estimate the real-world feasibility and impact of the proposed sabotage methods?

How do the identified attack vectors differ in difficulty and detectability across various AI training architectures (e.g., centralized vs. federated training)?

Could the authors expand on specific defensive mechanisms—technical or organizational—that could help detect or mitigate training sabotage in practice?

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
This paper investigates the possibility that nation states will sabotage frontier AI training runs of their rivals, causing the system to have degraded capabilities or hidden goals. Sabotage is seen as potentially less onerous than stealing model weights, which is an attack vector that has received a lot more attention. Sabotage might be less escalatory and therefore an attractive option for a nation state in a disadvantageous position compared to a more powerful rival. The paper presents some breakdown of the different types of sabotage to help understand the attack surface and attacker motivations.

### Strengths
A very interesting extension to MAIM and the RAND report on securing model weights. This paper fits more with what I think threat actors are likely to actually want to do in the real world. It is also clearly written from a position of expertise with how governments think and operate. The main text is clear and tells an interesting story.

### Weaknesses
The paper feels quite truncated now with some of the more sensitive parts removed. There aren't any experiments or even example applications of their taxonomy to different real world instances. I feel like I read a very detailed breakdown of a problem, but not a taxonomy.

It's worth noting that the citations in this work are nearly all in footnotes. I don't often see this in technical papers. I suppose it keeps the text more compact due to the citation style, but it is particularly unusual when a footnote has nothing except a citation in it. I think I would recommend merging some of the footnotes in. If the text gets too wordy, the paper can have more detailed sections that revisit topics. This is not a strong suggestion on my part, just an idea.

Specific comments on sections:

Section 2 is quite interesting and is probably new background for many readers. I think it could use more citations, especially in the second last paragraph. I had read that the Ukraine war had initially reduced the number of ransomware attacks on the rest of the world (though it's true that this trend seems to have reversed). This paragraph is also in the context of what's happening in wartime, which goes against the bullet point earlier “the situation does not yet justify overt military action”. Perhaps this list should use “or” rather than “and”. Or somehow reworded to include the Ukrainian war scenario.

The “salami slicing” done by China could use more citations as well, with the construction of artificial islands, Philippines, Vietnam, etc.

In section 3.1: The separation between the first stage (design and planning) and the second stage (data gathering and pre-processing) does not seem crisp. It seems likely that the second stage can proceed partially in parallel with the first stage. Also, creating a complete plan and deciding how large the model is going to be might depend fairly heavily on how much data can be obtained. The other stages are well known, pre-training and post training, and cannot really overlap. I wonder if it might make more sense to have a single first stage of planning, with several substeps that happen in parallel.

Typo in section 4: “efficacy of removal is a an open question.”

Minor typos in footnote 39: See is italic, and the semicolons aren't really grammatical.

### Questions
Why did you choose ICLR as a venue for this work? Overall, I am not sure whether it is the best venue. As someone in or adjacent to this field, I find it very interesting. Perhaps the machine learning community in ICLR could be one of the best audiences for this work, even if it is not a typical paper. But a dedicated AI safety conference/workshop may also be a good fit.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper argues that, for state actors, sabotaging AI training will often be more attractive than stealing model weights. It contributes (i) a threat-modeling framework covering strategic objectives (capability degradation vs. value misalignment; overt/covert; attributable/anonymous), (ii) a taxonomy of attack vectors (data poisoning, model poisoning, process disruption, with direct and indirect/organizational routes), and (iii) a walkthrough of the training pipeline highlighting plausible sabotage points and mitigations. It situates the analysis in historical context (e.g., Stuxnet; “grey-zone” operations) and discusses the MAIM (“Mutual Assured AI Malfunction”) dynamic. The Ethics Statement notes that some sensitive details were intentionally omitted; the authors also disclose substantial LLM assistance.

### Strengths
1. Makes a clear case that sabotage of training can be strategically favored in realistic scenarios (e.g., when compute deployment is infeasible for the attacker, or when preventing a rival’s capability matters more than copying it). 

2. Useful decomposition by objective (degradation vs. misalignment) and overt/covert/attributable axes; ties choices to strategic implications and examples (Table 1). 

3. Training pipeline stages are enumerated with concrete touchpoints for sabotage/mitigation—helpful for practitioners performing risk assessments. 

4. The historical review (Stuxnet; Cold War precedents; grey-zone warfare) provides external validity; the paper clarifies where MAIM might or might not hold. 

5. Dual-use considerations are acknowledged; sensitive technical details are intentionally withheld; LLM usage is disclosed.

### Weaknesses
1. Scope is largely qualitative, with limited technical novelty for ICLR. The piece reads more like a policy/security position paper than a machine-learning research paper; there are no models/algorithms, formal analyses, or empirical evaluations to test claims (e.g., attacker feasibility, detectability, or defense efficacy). Much of the content is synthesis + taxonomy. (The authors themselves note that a fuller threat-modeling study would require non-public information.) 

2. The plausibility assessments (e.g., insider-driven model poisoning, RLHF data poisoning) are plausible but lack quantitative risk estimates (likelihoods, cost-to-attack, time-to-detect), red-team case studies, or operational measurements to anchor recommendations. 

3. The paper cites J-AISI/NIST but would benefit from a side-by-side mapping showing exactly what is new (e.g., “process disruption” slice; organizational attacks) and what is reframed—ideally as a comparison table or figure. 

4. As a primarily strategic/cybersecurity paper with no ML experiments, it may struggle to meet ICLR’s contribution bar without a stronger technical component (e.g., detection methods, formal threat models with testable implications, or empirical audits across pipelines). (ICLR asks that reviews focus on value/new knowledge; here the value is practical but the ML research delta is thin.)

### Questions
1. Re pperationalization: Can you include at least one worked case study (anonymized) that quantifies attacker cost, access required, expected detection latency, and impact, for one data-poisoning and one process-disruption path? This would materially strengthen practitioner utility. 

2. Re comparative taxonomy: Provide a comparison matrix vs. J-AISI/NIST categories to make the paper’s unique framing absolutely explicit (what’s added vs. relabeled). 

3. Re MAIM implications: Could you formalize simple game-theoretic conditions (signals, attribution noise, cost curves) under which sabotage dominates theft, and simulate parameter regimes to test robustness of your MAIM-related claims? 

4. Re Defense playbooks: Your mitigations are high-level; can you attach a practitioner checklist per pipeline stage (e.g., specific logging/audit artifacts for RLHF datasets, insider-risk controls, compromise-resilience drills)? 

5. Re Responsible disclosure: Since some analyses were redacted, can you clarify what can be shared with vetted reviewers under confidentiality to better evaluate your novelty/insight claims?

### Soundness
2

### Presentation
3

### Contribution
2
