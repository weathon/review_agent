# Non-Collaborative User Simulators for Tool Agents

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 4, 4, 6

## Abstract
Tool agents interact with users through multi-turn dialogues to accomplish various tasks. Recent studies have adopted user simulation methods to develop these agents in multi-turn settings. However, existing user simulators tend to be agent-friendly, exhibiting only cooperative behaviors, failing to train and test agents against non-collaborative users in the real world.
We propose a novel user simulator architecture that simulates four categories of non-collaborative behaviors: requesting unavailable services, digressing into tangential conversations, expressing impatience, and providing incomplete utterances. Our user simulator can simulate challenging and natural non-collaborative behaviors while reliably delivering all intents and information necessary to accomplish the task.
Our experiments on MultiWOZ and $\tau$-bench reveal significant performance degradation in state-of-the-art tool agents when encountering non-collaborative users, as well as agent weaknesses under each non-collaborative condition such as escalated hallucinations and dialogue breakdowns. 
Our findings point to the need for methods that can improve agent robustness to the wide range of user behaviors encountered in deployment. We release the extensible simulation framework to help the community develop and stress-test tool agents under realistic conditions within their own service domains.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a framework for non-collaborative user simulators to evaluate tool-augmented language agents. It constructs simulated users exhibiting uncooperative behaviors such as refusal, impatience, or tangential dialogue, aiming to better stress-test agent robustness. Experiments on MultiWOZ and τ-bench show that such simulated users reduce task success rates, suggesting higher difficulty for tool agents.

### Strengths
1. The paper raises an important and underexplored issue ''how to evaluate tool-using agents under non-collaborative or adversarial user behaviors'', which represents a realistic challenge in open-world interaction.

2. The categorization of user behaviors, including refusal, incomplete requests, impatience and tangentiality, provides a structured way to model human-agent mismatches and can inspire future simulation frameworks.

3. Despite relying on closed models, the paper clearly details its simulation logic, prompting schemes, and evaluation pipeline, which helps readers understand how each behavioral dimension affects agent robustness.

### Weaknesses
1. We find that the evaluation pipeline introduces self-reinforcing bias through goal alignment checking. The paper repeatedly regenerates responses until a GPT-4o-mini judge marks them as “goal-aligned,” effectively filtering out harder or ambiguous cases. This design overestimates model robustness and relies on a closed-source judge that can introduce hidden bias into evaluation outcomes.

2. The paper lacks external validity due to selective and biased environment construction. The authors rebuild the MultiWOZ environment by including only 89 cross-domain booking scenarios and excluding certain task types. Similarly, τ-bench is pruned to remove complex cases. These design choices tailor the benchmark toward the authors’ target behaviors rather than representing real-world diversity.

3. We find that the proposed simulator heavily depends on closed-source models, undermining reproducibility and comparability.
Most modules, including dialogue generation, tool unavailability detection, complaint triggering, and personality modeling, are implemented using proprietary GPT-4 series models. Model updates could unpredictably alter outcomes, preventing stable replication or fair comparison across time.

4. The paper fails to establish convincing advantages over existing baselines, and some human evaluation results contradict its main claims. Only a prompt-based baseline is compared, and human raters actually find the “tangential + incomplete” user type to be less realistic than real human dialogues. Moreover, the overall human preference rate is around 60%, which weakens the central claim that the proposed simulator generates more realistic and challenging interactions.

5. We find that the success-rate metric conflates social behavior with task completion, making conclusions highly metric-sensitive.
The evaluation treats a task as successful only when the final database state exactly matches the goal, with a fixed 30-turn limit. The authors themselves note that polite or apologetic users tend to consume more turns and thus score lower. This indicates that the metric reflects conversational verbosity rather than genuine tool competence.

### Questions
Please refer to the above-mentioned weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a user simulation framework that incorporates various non-cooperative user behaviors to evaluate different tool agents. Previous user simulation studies have primarily focused on cooperative user behaviors. The proposed framework can induce four types of non-cooperative user behaviors ( Unavailable Service, Tangential, Impatience, and Incomplete Utterances ) while maintaining goal alignment, i.e., still conveying all the information needed to complete the task. This work evaluates the tool agents on two benchmarks, MultiWOZ and τ-bench. Experimental results show that the success rate of state-of-the-art agents drops significantly under these non-cooperative behaviors. When combinations of non-cooperative behaviors are introduced, the performance of the tool agents deteriorates further.

### Strengths
(1) This paper, for the first time, shifts the focus of user simulation from cooperative to non-cooperative behavior, and clearly defines four distinct types of non-cooperative behaviors.

(2) This study demonstrates that widely used agents can perform poorly under real-world non-cooperative conditions. Conventional evaluation methods tend to overestimate their robustness.

### Weaknesses
(1) While the paper claims that the four non-collaborative behaviors are “grounded in research from marketing, open-domain dialogue, and real-world user-agent data,” the rationale for this exact categorization remains unclear. The authors do not articulate why these four were selected, whether they represent orthogonal dimensions of non-collaboration, or how they were empirically grounded in the referenced domains. A more principled taxonomy would strengthen the conceptual contribution.

(2) Although the framework is described as easily extensible, all results are based on the MultiWOZ and τ-bench benchmarks, both of which are relatively well-structured. The authors are encouraged to include at least one more complex or less-constrained benchmark to better demonstrate the framework’s generalizability and realism.

### Questions
See Weakness

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a user simulator framework that models non-collaborative user behaviors to fill a key gap in tool agent evaluation. It defines four behavior types—unavailable service requests, tangential conversation, impatience, and incomplete utterances—and extends collaborative simulators with LLM-based modules to generate realistic, goal-aligned interactions. Experiments on MultiWOZ and τ-bench show that state-of-the-art LLMs experience performance drops under non-collaborative conditions, revealing specific failure patterns.

### Strengths
- The paper addresses a genuine gap between idealized evaluation settings and real-world deployment challenges, supported by references to marketing research and real user data.
- The four categories of non-collaborative behaviors are well-defined, grounded in prior research, and cover important real-world scenarios

### Weaknesses
- The work is significant for the community, as it enables more realistic evaluation and can help prevent deployment failures. However, the novelty is somewhat incremental—the core simulation architecture builds heavily on existing work (Yao et al., 2024; Luo et al., 2024), with the main innovation being the specific non-collaborative behavior modules.
- The mixed results (Figure 5) for tangential behaviors suggest the simulator doesn't uniformly outperform baselines. More analysis of failure cases would strengthen the work
- The combination of multiple behaviors (Table 2) shows further degradation, but real-world co-occurrence patterns are not analyzed

### Questions
- In real-world deployments, what percentage of users exhibit each type of non-collaborative behavior? How should practitioners weight these scenarios?
- Do any agents show ability to recover from non-collaborative situations over longer dialogues? Is the 30-step limit preventing observation of recovery strategies?
- In the examples (Appendix F.2), users continue tangential topics even after acknowledgment. Is this realistic? How often do real users persist with tangential conversations?
- What's required to adapt your framework to non-booking domains (e.g., technical support, creative writing assistance)? How much of your taxonomy transfers?

### Soundness
2

### Presentation
3

### Contribution
3
