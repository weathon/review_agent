# Position: Want Better ML Reviews? Stop Asking Nicely and Start Incentivizing with a Credit System

- Avg Score: 3.20
- Decision: Reject
- Scores: 2, 2, 6, 4, 2

## Abstract
With soaring submission counts, stricter reciprocity review policies, widespread adoption of platforms like OpenReview, and without the offsetting pressure of publication fees, the machine learning (ML) community has one of the largest scholarly presences among all scientific fields. And yet, almost *everyone* has *many* unpleasant things to share about their review experience. Worse, there is little public space to seriously discuss — let alone debate — what makes a review system effective or how it might be improved.  

In this position paper, we expand our discussion from the two core problems: *How can we reasonably limit the number of submissions?* and *How can we incentivize good and discourage bad review practices?* We first assess the strengths and shortcomings of existing attempts to address such problems. Specifically, we present five takes on some popular conference mechanisms and propose two alternative designs for improvement.  

Our general position is that meaningful improvement in ML peer review won’t come from polite best-practice suggestions tucked into Calls for Papers or Reviewer Guidelines — it requires **enforceable yet fine-grained procedural safeguards** paired with **a currency-like credit system (what we call *OpenReview Points*)**. ML practitioners can “earn” such points by contributing good review practices, and “spend” them across one or multiple major conferences to redeem different kinds of “perks” — such as complimentary registration or the right to request additional review resources.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This position paper addresses how to improve peer review at major AI conferences, focusing on two key issues: limiting submission volume and incentivizing higher-quality reviews. The authors use recent evidence (e.g., CVPR's desk rejection of irresponsible reviewers, NeurIPS's submission explosion) to highlight problems in current peer review practices.

The paper critiques existing solutions. First, desk-rejecting papers from irresponsible reviewers does not scale and penalizes only a small fraction of problematic cases. Second, mandatory reviewing degrades review quality because not all reviewers are qualified. Third, reviewer discussions rarely occur due to lack of incentives.

To address these issues, the authors propose two mechanisms. The first is an OpenReview Points system where reviewers earn points through reviewing activities such as completing reviews or joining emergency review pools. These points can be used to exempt future reviewing duties or waive conference registration fees. The second is a unanimous voting mechanism for reviewer penalties, where authors can report low-quality reviews and the reported reviewer is penalized if all other reviewers and the Area Chair unanimously agree.

### Strengths
- The unanimous voting system proposed by the authors is interesting, and could be part of a credit system. It supplements the drawbacks of reviewers having to review and spend a lot of time reviewing low quality works.
- I strongly agree that an incentive-based system is needed to make peer review sustainable at scale. The proposed point system, while underdeveloped in its current form, has promising extensions. For example, if properly implemented, such a system could motivate reviewers to submit timely reviews, addressing another persistent problem in conference workflows.

- The alternative views are fair and address some of the important points (gaming, favoring researchers with more institutional support).

### Weaknesses
- Looking at the Abstract, the paper has modified the .sty file of latex. This is a violation of the formatting instruction. 
- The paper's writing style is unprofessional and unsuitable for academic publication. Phrases such as "Emergencies happen. Burnout is real. But the system doesn't care — once you're in the pool, and you're staying there" and "we have the largest scholarly firepower reserve of any scientific field" are informal and colloquial. They read more like a blog post or Twitter rant than a scholarly position paper.  The manuscript requires substantial revision to meet academic writing standards. 
- The proposed OpenReview Points system is superficial and lacks critical implementation details. The authors do not address who would maintain the system or provide precise criteria for awarding points. The current proposal relies on naive, empirical heuristics. While the authors acknowledge that "points need more sophisticated balancing," they fail to provide this framework—precisely what a position paper should offer in detail.
- Furthermore, the financial implications are unexplored. Who would cover the costs of waived registration fees? Major conferences like ICLR depend heavily on registration revenue to fund their operations. If points exempt reviewers from paying fees, the paper must explain how conference budgets would remain sustainable. Without addressing these practical concerns, the proposal remains incomplete.
- The paper identifies submission explosion as a central problem in the peer review crisis but does not explain how OpenReview Points would address this issue. The proposal focuses on incentivizing reviewers but provides no mechanism for limiting submission volume, leaving a gap between the stated problem and the proposed solution.

### Questions
see above.

### Soundness
2

### Presentation
1

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
The paper discusses the growing challenges of the machine learning peer-review system amid rising submission volumes and stricter reciprocity policies. It identifies two core issues, limiting submissions and incentivizing good reviewing practices, and critiques existing solutions. The authors propose that genuine improvement requires enforceable procedural safeguards combined with a credit-based system (OpenReview Points), allowing researchers to “spend” review credits across conferences. This framework aims to promote accountability, fairness, and sustainability throughout the publication process.

### Strengths
- The paper tackles one of the most pressing meta-scientific challenges in the ML community: the scalability and fairness of the peer-review process under exploding submission volumes. 

- The motivation is clear and resonates strongly with the current frustration many researchers face in large-scale conferences like NeurIPS, ICML, and ICLR.

### Weaknesses
- While the authors acknowledge that position papers need not present numerical results, no attempt is made to explore even minimal feasibility or simulation evidence (e.g., hypothetical modeling of point distributions, reviewer load dynamics, or incentive equilibria). Even small-scale experiments or thought experiments could have strengthened the argument’s credibility.

- Compared with prior accepted position papers at ICLR or ICML (e.g., Ngo et al., 2024; Yang, 2025), this work lacks integration of historical experience and data-informed reflection from the ML community. The proposed framework would be more convincing if supported by retrospective evidence or case studies drawn from real conference practices.

- The proposed “OpenReview Points” system remains high-level. The paper does not provide enough concrete guidance on: (1) how points would be standardized across conferences with differing review policies, (2) how abuse or collusion could be prevented, and (3) how the infrastructure and governance of such a credit market would operate in practice.
As a result, while conceptually appealing, it lacks operational realism.

- As the central stance and innovation of the paper, the proposed credit system does not sufficiently consider potential negative side effects. For example, using credits as a prerequisite for paper submission could disadvantage early-career researchers who have not yet accumulated review credits, and allowing credit redemption to skip review duties could further accelerate the loss of high-quality reviewers.


> [1] Richard Ngo, Lawrence Chan, and Sören Mindermann. The alignment problem from a deep learning perspective. In ICLR 2024.
> [2] Jing Yang. Position: The artificial intelligence and machine learning community should adopt a more transparent and regulated peer review process. In ICML 2025 Position Paper Track.

### Questions
- Could the authors provide conceptual or simulated evidence showing that a credit-based system would stabilize reviewer workload or improve review quality metrics?

- How might OpenReview Points be coordinated across independently managed conferences (e.g., ICLR, NeurIPS, ICML) with distinct review policies and timelines?

- How would gaming prevention be ensured? For instance, to avoid mutual positive flagging or collusive exchanges of review credits?

- From a design perspective, how can a credit-based review system both incentivize high-quality reviewers to participate and avoid disadvantaging newcomers to the field?

- The paper does not appear to fully utilize the available space in the ICLR format. Does this reflect a lack of additional experimental or design details, or could the authors expand with more concrete elaboration?

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
This position paper addresses persistent problems in the peer review process at large-scale ML conferences, arguing that polite guidance and voluntary best practices no longer suffice. The authors propose two concrete reforms: (1) enforceable procedural guardrails, and (2) a credit-based incentive system called OpenReview Points, which would allow participants to earn and spend credits for reviewing, opting out, or gaining privileges such as registration waivers.

I strongly agree with the central position: as venues scale, interest-based reviewer matching and informal guidelines break down. Treating large ML conferences more like economic systems — rather than purely academic gatherings — is a necessary shift. As seen in large volunteer-driven events like the Olympics, voluntarism often fails at scale unless paired with incentives or compensatory frameworks. This paper takes a bold and timely step in that direction by proposing a flexible, modular credit system that can be adapted per conference. While implementation details need to be further developed, this work opens an important conversation and presents a promising path forward.

### Strengths
* Timely and necessary: Tackles a widely recognized crisis in ML peer review, especially at scale.

* Concrete and actionable: The credit system provides clear examples of earning/spending points, offering a tangible mechanism for reform.

* Candid critique of existing systems: Identifies the limits of desk rejections, reciprocal reviewing, and submission caps with clarity.

### Weaknesses
* Implementation is underdeveloped: Lacks specifics on governance, cross-conference coordination, fraud prevention, and fair point allocation.

* No empirical or simulated analysis: A hypothetical case study or retrospective analysis using real conference data would improve feasibility.

* Assumes duty-based reviewing: Many still see reviewing as voluntary labor, not a formal obligation, which complicates enforcement.

* Operational complexity unaddressed: Running such a credit system would require major infrastructure and consensus.

### Questions
* How would OpenReview Points be initialized for new or first-time contributors?
* Would OpenReview itself manage the credit system, or would a new governance body be needed?
* Can credits be pooled, transferred, or inherited across teams or co-authors?
* How do you prevent low-effort “review farming” aimed at harvesting points?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This position paper argues that ML peer review quality cannot be improved through polite guidelines alone, proposing instead a currency-like "OpenReview Points" system combined with fine-grained procedural safeguards. The authors identify two core problems: excessive submissions and lack of accountability for reviewers. They critique existing mechanisms (submission caps, mandatory reciprocal reviewing, desk rejections) and propose a credit economy where researchers earn/spend points for various reviewing activities and privileges.

### Strengths
1. This paper discusses a timely topic : Everyone in ML has complained about bad reviews at some point. The paper tackles an issue that actually matters to the community.

2. The paper does a good job explaining the two main issues: too many submissions and no consequences for lazy reviewers. The example of SACs handling 80 papers each at NeurIPS really drives the point home.

3. The authors don't just complain - they actually look at what conferences are already trying (submission limits, mandatory reviewing, desk rejections) and explain why these don't really fix the problem.

### Weaknesses
1. The actual benefit of the proposed credit system is unclear. Research shows that incentive systems often don't improve peer review quality. For example, Gasparyan [1] studies multiple incentive types (monetary, certificates, CME credits, open recognition) and concludes no single incentive model consistently improves peer-review quality.

2. Turning reviewing into a “currency economy” risks distorting intrinsic motivations. Once credit accumulation becomes an explicit metric, reviewers may optimize for the fastest or most visible ways to earn points instead of offering thoughtful, time-consuming feedback. Despite claiming to avoid gamification, the system could still reproduce inequality or favor individuals with more time or institutional resources.



[1] Gasparyan AY, Gerasimov AN, Voronov AA, Kitas GD. Rewarding peer reviewers: maintaining the integrity of science communication. J Korean Med Sci. 2015;30(4):360-364. doi:10.3346/jkms.2015.30.4.360

### Questions
See  Weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This position paper explores how to limit the number of paper submissions to AI conferences and incentivize high-quality reviews while discouraging low-quality ones. It critiques retaliatory processes such as desk-rejecting papers from irresponsible reviewers as too coarse for handling fine-grained bad-review behavior beyond abandoning review tasks, and proposes proportional penalties. It criticizes mandatory reciprocal reviewing for degrading review quality. It suggests a system of review points to be used for "privileges," such as avoiding reviews or free conference registration.

### Strengths
The problem is important, should be discussed, and its causes and symptoms should be understood.
Free registration is a strong incentive and is practical for unlimited virtual registration.

### Weaknesses
1. Using points to avoid reviewing may send the wrong message about the value of reviews,
I disagree with the statement that avoiding reviewing and academic service is a privilege; it's the other way round, academic service is a privilege.

2. This paper takes a narrow view of handling the increasing number of papers, and is missing a key element: automated scientific discovery and AI's increasing role in performing research and writing papers.
It focuses on the symptom, the increase in submissions, rather than addressing the cause, which includes AI and automated research and writing.

3. This paper would be good as a blog, not an ICLR research paper.

### Questions
Have the authors considered posting the text as a blog or presenting their positions in a panel? to hear diverse opinions and perspectives.

### Soundness
2

### Presentation
2

### Contribution
2
