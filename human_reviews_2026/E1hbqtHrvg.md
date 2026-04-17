# Strategic Planning and Rationalizing on Trees Make LLMs Better Debaters

- Decision: Accept (Poster)
- Scores: 4, 4, 4, 6

## Abstract
Winning competitive debates requires sophisticated reasoning and argument skills. There are unique challenges in the competitive debate: (1) The time constraints force debaters to make strategic choices about which points to pursue rather than covering all possible arguments; (2) The persuasiveness of the debate relies on the back-and-forth interaction between arguments, which a single final game status cannot evaluate. To address these challenges, we propose TreeDebater, a novel debate framework that excels in competitive debate. We introduce two tree structures: the Rehearsal Tree and Debate Flow Tree. The Rehearsal Tree anticipates the attack and defenses to evaluate the strength of the claim, while the Debate Flow Tree tracks the debate status to identify the active actions. TreeDebater allocates its time budget among candidate actions and uses the speech time controller and feedback from the simulated audience to revise its statement. The human evaluation on both the stage-level and the debate-level comparison shows that our TreeDebater outperforms the state-of-the-art multi-agent debate system, with a +15.6% improvement in stage-level persuasiveness with DeepSeek and +10% debate-level opinion shift win. Further investigation shows that TreeDebater shows better strategies in limiting time to important debate actions, aligning with the strategies of human debate experts.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the task of computational argumentation and presents TreeDebater, a debate framework for LLMs that structures the debate process through two explicit tree-based representations. The Rehearsal Tree enables the model to anticipate potential attacks and defenses prior to the debate, and the Debate Flow Tree dynamically captures the evolving argumentative structure and state throughout the interaction. Additionally, a time control module is introduced to simulate real-world debate conditions by enforcing strict timing constraints. The framework is evaluated at both the stage and debate levels using an Oxford-style debate format, and human evaluations demonstrate that TreeDebater consistently outperforms the Agent4Debate multi-agent framework across multiple aspects.

### Strengths
- The paper addresses an important challenge in enabling LLMs to participate in realistic debates. Instead of focusing on static argument generation, this study simulates an interactive debate process under time constraints, making it closer to real-world settings.
- The introduction of the Rehearsal Tree and Debate Flow Tree is well-motivated and inspired by human debate strategies, effectively structuring anticipation and response during the debate.
- The evaluation is fairly comprehensive, covering both head-to-head and end-to-end comparisons to thoroughly assess the system’s performance.
- Table 3 also provides user feedbacks, which may bring insights for future research.

### Weaknesses
- The core idea centers on applying tree-based planning and reasoning for argument generation, with the main contribution being the task-specific adaptation of this structure. However, multi-step agent interactions and tree-based debate frameworks are already common strategies for enhancing LLM reasoning, which makes the overall novelty of the work somewhat limited.
- While Figure 2 presents the distribution of action types, the paper lacks fine-grained ablation studies for key components (e.g., simulated audience feedback, speech time controller, retrieval quality). This makes it difficult to isolate and understand the contribution of each module.
- The comparisons are limited to Agent4Debate, without inclusion of other LLM-based argument generation systems. Additionally, the use of powerful base models such as Gemini and DeepSeek raises questions about generalizability: how would the framework perform with smaller models (e.g., 7B parameters) as the backbone LLM?
- I appreciate the inclusion of human evaluations. However, important details are missing: How many annotators assessed each sample? What is the inter-annotator agreement, given the subjective nature of argument evaluation? Moreover, since the generated arguments may vary in claim or stance each time, how is bias controlled to ensure fair evaluation across different runs? (e.g., A reader may favor a model outputs mainly because of the claim/opinion itself rather that the quality of the outputs)
- According to line 798, it appears that 52 debates are generated for the experiments. If correct, this detail should be included in the main paper as this is an important detail. Furthermore, it would strengthen the analysis to include a diversity measure of debate topics to assess the model’s robustness across different propositions.

### Questions
Q1. From Figure 4, it appears that each node’s claim is typically concise and often limited to a single sentence. These nodes seem to represent subtopics or subclaims that illustrate the progression from the main claim to supporting points. However, in real debates, intermediate arguments are usually more complex, involving actions such as attacking a premise from the opposing side or providing detailed reasoning to defend an opinion, especially in an interactive debate setting. Therefore, this tree structure might struggle to fully capture the dynamic and intricate reasoning processes that characterize authentic debates. I would like to hear the authors’ comments on this point.

### Soundness
3

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
The paper presents TreeDebater, a framework and system for automatic debating. TreeDebater models the dynamics of a competitive debate as two trees which roughly follow the reasoning process of a human debater. The first, a Rehearsal Tree, stores a claim related to the debate topic at the root and different related arguments and counter-arguments in a tree structure below it, where each node counters its parent or enforces its grandparent. The Rehearsal Tree models the way a human debater prepares for a debate by anticipating different potential back and forth flows that may happen in the actual debate. Arguments stored in the Rehearsal Tree are scored based on the strength of their defense (of the grandparent) and attack (on the parent). The second tree, A Debate Flow Tree, follows the debate’s flow and keeps track of its status by storing in a tree structure all the claims that were made as well as related attacks and defenses that were brought up during the debate. Based on the status of the debate, the system picks the next arguments from the Rehearsal Tree. The system also includes a feedback mechanism to refine arguments and a text-to-speech system to estimate the time it takes to say a given statement, making sure that the final speech is within the allotted time. They show that humans find their system significantly more persuasive than a previous SoTA system both during certain stages of the debate as well as for the debate in its entirety.

### Strengths
The suggested system is inspired by the way humans prepare to and conduct a competitive debate. At a high level the system’s principles and architecture look reasonable but details are missing (see below). Evaluation includes both stage-level and end-to-end human preference experiments as well as additional fine-grained analysis of the debates that the system is producing. Experimental results are convincing.

### Weaknesses
Many details are missing. For example:

- It is not clear what is the action selection criteria in the paragraph starting in line 227: “Extract Candidate Actions from Debate Flow Tree”.

- When updating the Debate Flow Tree (alg. 2 in Appendix B), how is the ‘action’ being determined?

- Even after reading Appendix D, it is not clear to me how the audience feedback works.

### Questions
- Line 418: how do you determine if two claims are similar?

- What is the agreement between the annotators in the human preference experiments?

- Line 368: Figure 2 -> Table 2

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work tackles the problem of automated competitive debate, and focuses on the aspect of planning and decision making. The authors enhance an existing agent system for debating by adding capabilities of anticipating the expected flow and claims that will be made by both sides, and tracking the flow as the debate progresses. This is done via constructing trees where each node is an argument or counter-argument. Paths are chosen based on their estimated utility, in terms of the chosen action (e.g., reinforce an existing claim or rebut an opponent claim), the estimated argument strength, and semantic similarity is used to retrieve prepared arguments and track how many times an argument is addressed. The authors conduct experiments comparing their approach to the baseline agent debate system that does not incorporate this tree-based planning, performing human evaluation of per-stage debate persuasiveness and performance as well as a full head-to-head debate between the two systems.

### Strengths
1. The tree-based methods are novel and interesting. In particular, it has a nice approach of estimating the argument strength not just in terms of the argument itself but in terms of anticipating its full impact including the estimated opponent actions that will follow.
2. Human evaluation was done at the level of a specific stage as well as the overall debate impact, and shows some persuasiveness gains from the proposed approach.

### Weaknesses
1. There are many details here and no provided code, so reproducibility is a major issue. Moreover, given that the paper is focused on a comparison to Agent4Debate, I think it is missing a clearer accounting of how and in what architecture the method here integrates with the agent system described there. The diagram in Figure 1 is helpful but is quite vague in terms of understanding how the LLM agents are used in practice when generating the debate.
2. I felt that the part about simulated audience feedback (§3.4) was not sufficiently clear. Without going to the appendix, it is not explicitly stated that there is a collected dataset of human debates (the reference to "the retrieved human Debate Flow Trees" (l. 249) is too vague on its own). Crucially, I did not understand from either the main paper or the appendix how exactly this data is used.

### Questions
1. Could you explain where the list of "battlefields" included in the prompts comes from?
2. In Figure 2, what is the difference in implementation between the baseline Agent4Debate (which AFAIU doesn't use trees) and the ablation of TreeDebater without the trees? Why do they show such different behaviors?
3. To what extent would you say that the ability to anticipate the opponent behavior (Figure 3) is connected to the similarity between the two automated systems engaged in the debate (e.g., that both use the same underlying LLM and similar prompts to generate the arguments)?
4. What is the value of the decay coefficient $\gamma$?

Additional comments:
* I think it would help to highlight in the prompts (Appendix G) which parts of the prompt comes from Agent4Debate versus are new for this work.
* Appendix D mentions "two debate datasets", but then only PanelBench is mentioned.

Typos:

l. 33 argued -> argue

l. 167 this grandparent -> its grandparent

l. 240/295/302/357 the Appendix -> Appendix

l. 348 totally recruited 212 participants -> recruited 212 participants in total

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes TreeDebater, an LLM-based debating system designed for competitive, time‑limited debates. The core idea is to structure planning and tracking as two trees: 

Rehearsal Tree (pre‑debate): based on the intuition that human debaters usually prepare for the potential attacks and defenses of
their claims. The proposed rehearsal tree is designed to help the debator retrieve relevant evidence and evaluate how robust their claims are towards the attack. The Rehearsal Tree anticipates the attack and defense for each main claim in a tree format and calculates the k-step strength score to evaluate the utility of the claim.

Debate Flow Tree (in‑debate): The debate tree is designed to record the debate status, simulating the note-taking of humans. The Debate Flow Tree tracks the debate status by keeping all proposed claims with the corresponding attack and defense in a tree structure. TreeDebater can filter out the candidate actions it can take in the current stage of the debate based on the Debate Flow Tree. After getting the candidate actions, TreeDebater retrieves the prepared arguments for this action from the Rehearsal Trees.

Empirically, this paper compares TreeDebater with Agent4Debate across two backbones (Gemini‑2.0‑flash and DeepSeek‑V3) using human evaluation at (a) the stage level (head‑to‑head on the same debate context) and (b) the end‑to‑end debate level with pre/post votes. TreeDebater improves average persuasiveness and opinion‑shift win rate.

### Strengths
1. The paper tackles competitive debate where persuasiveness determines success, and it operationalizes time budgeting and action selection under evolving interaction. The two‑tree design mirrors how human debaters prepare and flow debates.
2. The Rehearsal Tree formalizes pre‑debate planning with a k‑step strength score that blends support and attack impacts.
3. The Debate Flow Tree offers a practical, auditable representation for candidate action extraction during the match.
4. Good empirical evaluation to demonstrate the effectiveness of the proposed approach. 
5. The paper is well written and easy to follow.

### Weaknesses
1. TreeDebater alone uses an iterative time controller; the baseline is given only “rough word budgets” and is then audio‑trimmed, which can truncate arguments mid‑point and plausibly depress persuasiveness.
2. Missing detailed ablations for each component of the TreeDebater in achieving the final persuasiveness. For example, the impact of Rehearsal Tree and Debate Flow Tree separately is unknown. 
3. While the overall pipeline is well designed, the fundamental idea shares similarity with Tree‑of‑Thoughts / Graph‑of‑Thoughts.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3
