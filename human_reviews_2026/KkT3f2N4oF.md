# T-POP: Test-Time Personalization with Online Preference Feedback

- Decision: Reject
- Scores: 2, 2, 6, 4

## Abstract
Personalizing large language models (LLMs) to individual user preferences is a critical step beyond generating generically helpful responses. However, current personalization methods are ill-suited for new users, as they typically require either slow, resource-intensive fine-tuning or a substantial amount of pre-existing user data, creating a significant cold-start problem. To address this challenge, we introduce a new paradigm for real-time personalization by learning from online pairwise preference feedback collected during text generation. We propose T-POP (Test-Time Personalization with Online Preference Feedback), a novel algorithm that synergistically combines test-time alignment with dueling bandits. Without updating the LLM parameters, T-POP steers the decoding process of a frozen LLM by learning a reward function online that captures user preferences. By leveraging dueling bandits, T-POP intelligently queries the user to efficiently balance between exploring their preferences and exploiting the learned knowledge to generate personalized text. Extensive experiments demonstrate that T-POP achieves rapid and data-efficient personalization, significantly outperforming existing baselines and showing consistent improvement with more user interactions.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces T-POP, a novel algorithm designed to address the cold-start personalization problem for new LLM users. The core idea is to learn user preferences at test-time without updating the base LLM's parameters. T-POP frames this as an online learning problem, employing a dueling bandit framework to generate pairs of responses (one "exploitation" and one "exploration" sequence). It then queries the user for pairwise preference feedback between these sequences. This feedback is used to train a separate, lightweight reward model online. This learned reward model, in turn, steers the decoding process of the frozen base LLM to generate responses that are better aligned with the user's emerging preferences.

### Strengths
The high-level concept of this paper is its main strength. The idea of learning an additional reward function online to explicitly model a new user's personal preferences is an inspiring and innovative approach. Tackling the cold-start problem via a principled online learning framework like dueling bandits is a promising research direction.

### Weaknesses
While the core idea is novel, the paper's current execution and evaluation raise several significant concerns.

1. The experimental comparison to baselines like Linear Alignment (LA) and Amulet feels mismatched. T-POP's framework explicitly requires intermediate, online preference feedback from the user to function. In contrast, LA and Amulet are designed for a different, non-interactive scenario where they produce a single, aligned output in one go. This difference in application scenarios and assumptions limits the fairness of the comparison and suggests T-POP is only applicable to a more constrained setting where users are willing to provide continuous feedback.

2. A major practical concern is the computational overhead. The T-POP algorithm requires training and running an additional neural network (the reward model) at test time, on top of the base LLM. This seems likely to introduce significant inference latency. The paper would be much stronger if it provided a detailed analysis comparing the wall-clock inference time of T-POP (including reward model computation and online training steps) against the baselines.

### Questions
1. The current ablation study focuses on hyperparameters and model size. These results are informative but feel more like standard sensitivity analyses. My main questions relate to the core contribution of the sampling strategy, which is not ablated.

2. A key component of T-POP is the specific dueling bandit policy used to generate the "Exploitation Sequence" and "Exploration Sequence". The significance of this specific strategy is not experimentally justified.

3. Could the authors provide an ablation study that compares this sampling policy to other, perhaps simpler, methods for generating the two sequences? For example, what would be the impact of using a different policy (e.g., random sampling, or two exploitation sequences with different temperatures) to generate the pairs for the user feedback?

4. Alternatively, can the authors provide any theoretical results on the expected benefits of using the specific dueling bandit approach for this problem compared to other potential strategies?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper aims to address real-time personalization of LLMs without parameter fine-tuning. The key idea is to combine test-time alignment and neural dueling bandits to efficiently learn a user-specific reward function from online pairwise preference feedback.  
At each decoding step, T-POP adaptively generates two competing candidate responses, one focused on exploitation and the other on exploration, and queries the user to indicate preference. The reward function is updated online and used to steer future token generation via a modified scoring rule. Experiments across multiple LLM backbones and personalization attributes (creative, verbose, concise, uplifting) demonstrate consistent gains over test-time alignment and prompt-based baselines.

### Strengths
The motivation of the paper is valid and concerns practical scenarios.

### Weaknesses
1. My main concern of this paper lies in its incremental novelty and the quality of the evaluations.  
- Conceptually, T-POP resembles AMULET’s online decoding realignment, with the main difference being the use of explicit dueling-bandit uncertainty modeling. Without theoretical or empirical isolation of this component’s benefit, the contribution may appear evolutionary rather than groundbreaking. It is not clear through the current manuscripts how the dueling-bandit adoption contributes to the final performance.
- Methodology-wise, it is also not clear about the technical novelty, and how the algorithm differ from existing ones.

2. While the method borrows ingredients from neural dueling bandits, the adaptation to token-level decoding is largely heuristic. The “uncertainty bonus” and its covariance update (Eq. 5–6) are justified informally but lack a formal analysis showing regret or convergence guarantees in this setting. The current exposition risks being perceived as ad hoc rather than theoretically principled.

3. The reward model $r(\cdot;\theta)$ is defined over full sequences, but during decoding, it is applied token-by-token with gradients taken on partial sequences. It remains unclear how gradient-based uncertainty estimates correspond to true uncertainty in user preference. Authors are suggested to clearly clarify this, which would strengthen the algorithmic soundness.

4. The framing of cold-start may be overstated. The method assumes access to a few online preference interactions, but in practice even this requires user effort. The paper does not quantify the practical overhead (e.g., number of comparisons per session) or discuss scalability to more complex preference spaces.

### Questions
How sensitive is T-POP’s performance to the specific choice of uncertainty metric (Eq. 5)? Could alternative uncertainty formulations (e.g., entropy over token scores) achieve similar results?

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
This paper considers the problem of test-time personalization of LLMs and proposes a novel algorithm wherein the user preferences are collected adaptively by trading off exploitation and exploration using the dueling bandit framework. Then a reward model is trained which is then used to steer LLMs towards generating aligned outcomes without any training of the model parameters. Exhaustive experiments across different models and datasets prove the effectiveness of the algorithm.

### Strengths
The problem is well-motivated and timely, and the paper is well written. The proposed solution solves the cold-start problem, doesn't require training LLM parameters, the decoding step is simple and intuitive, and the adoption of duelling bandits is interesting. The experiments are also exhaustive with multiple models and datasets and the results seem convincing.

### Weaknesses
It seems like storing one reward model per user will be necessary compared to the RAG and prompt engineering approaches. Additionally, collecting T preferences from each user all at once and then using the final reward model is impractical, particularly since large amounts of T per user are required to properly understand their reward. The more realistic scenario is that reward model training will occur concurrently with deployment, which means the covariance matrix must also be stored. This could lead to substantial memory usage given that the number of users is very large. The authors should either acknowledge this as a limitation or provide information on how to address this problem.

### Questions
Are there any theoretical guarantees for the dueling bandit approach? (you can just restate the guarantees from the dueling bandits literature for completeness)

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a test-time personalization method for LLMs called T-POP which first trains a user-specific reward function online from pairwise preferences and use it in the decoding process of the frozen base model. During decoding, T-POP outputs two responses for expoloitation and exploration and the user is asked to choose the preferred response which will then be used to update the simple reward function. The results show that the method consistently gains over existing training-free baseline across multiple models and datasets.

### Strengths
1. Training the reward function using neural dueling bandits is a novel idea since it takes account both exploration and exploitation.
2. The inference is efficient since the architecture for the reward function is a simple head where the forward pass of the LLM backbone happens once but the simple additive decoding process made it cheap to output two responses.
3. The training is also efficient since the base model is frozen.
4. Convergence across the number of rounds of conversation shows that the reward function actually learns online.
5. Reported gains are consistent across settings suggesting practical value with low-latency personalization without fine-tuning.

### Weaknesses
1. In the beginning of the interaction with the user, online training of the reward model every round could meaningfully increase latency at inference time.
2. While LA and AMULET are included, the paper might benefit for including other decoding-time control or alignment-as-search baselines.
3. It requires users to choose a preferred response for every round (minimum of 20 turns in order to gain some performance according to the paper) which can be burden to the users.
4. There are other decoding methods for personalization using the learned reward function. Also there are papers that show combining multiple learned reward function at test-time can be used for personalizing to users. For these methods, they do not require additional training at test-time. I am not too certain what are the clear advantages of T-POP over the mentioned methods/frameworks.

### Questions
See the weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3
