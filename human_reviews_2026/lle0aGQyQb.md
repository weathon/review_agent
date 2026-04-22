# Think-While-Generating: On-the-Fly Reasoning for Personalized Long-Form Generation

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 4, 4, 6

## Abstract
Preference alignment has enabled large language models (LLMs) to better reflect human expectations, but current methods mostly optimize for population-level preferences, overlooking individual users. Personalization is essential, yet early approaches—such as prompt customization or fine-tuning—struggle to reason over implicit preferences, limiting real-world effectiveness. Recent “think-then-generate” methods address this by reasoning before response generation. However, they face challenges in long-form generation: their static one-shot reasoning must capture all relevant information for the full response generation, making learning difficult and limiting adaptability to evolving content. To address this issue, we propose **FlyThinker**, an efficient “think-while-generating” framework for personalized long-form generation. FlyThinker employs a separate reasoning model that generates latent token-level reasoning in parallel, which is fused into the generation model to dynamically guide response generation. This design enables reasoning and generation to run concurrently, ensuring inference efficiency. In addition, the reasoning model is designed to depend only on previous responses rather than its own prior outputs, which preserves training parallelism across different positions—allowing all reasoning tokens for training data to be produced in a single forward pass like standard LLM training, ensuring training efficiency. Extensive experiments on real-world benchmarks demonstrate that FlyThinker achieves better personalized generation while keeping training and inference efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces FlyThinker, an efficient “think-while-generating” framework. Unlike prior “reason-then-generate” approaches, FlyThinker employs a separate reasoning model that runs in parallel to dynamically guide the generation model. The framework achieves efficiency in both training and inference, and extensive experiments demonstrate its strong effectiveness and superior performance.

### Strengths
1. The concept of generating reasoning tokens and response tokens simultaneously is novel and intriguing to me.

2. The idea of decoupling reasoning tokens from previously generated reasoning outputs is particularly interesting, as it enables a one-pass training process and significantly improves overall efficiency.

### Weaknesses
1. The title of Figure 3 is somewhat misleading and should be revised to “Training Efficiency / Inference Efficiency.” Although the proposed method demonstrates shorter runtime, it relies on two separate models—the reasoning model and the generation model—which substantially increases memory consumption. The authors should therefore provide a comparison of the actual computational cost against other baselines to present a fair assessment.
2. Reasoning models typically show the greatest advantages on more challenging tasks, such as mathematical or scientific reasoning (e.g., AIME24, GPQA) and coding tasks. It would be more insightful if the authors evaluated the proposed method on such demanding benchmarks, as this would more convincingly highlight the true effectiveness of the reasoning model.

Typo: In line 94, the author redundantly includes an extra “First” after “Firstly.”

### Questions
As noted in the weaknesses above, please provide more details on the actual overall computational cost and evaluate the proposed method on a broader range of challenging tasks, such as mathematical or scientific reasoning and coding benchmarks, to better demonstrate its effectiveness.

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
The authors propose FlyThinker, a method for combining LLM reasoning and generation in a "think-while-generating" paradigm. In their approach, there are reasoner and generator models, where the reasoner is given the input and an in-progress generation, then produces a latent reasoning token that is fed to the generator to guide its response. They argue that this approach allows for efficient LLM personalization with latent reasoning.

### Strengths
- The proposed approach seems useful for efficiently combining reasoning and generation. It provides a simple way to align to user preferences.
- The method is flexible and can be used with models of different sizes. It is particularly convenient that a small reasoning model can be combined with a larger generation model for greater efficiency.
- The writing and presentation of the paper are clear. The method section in particular is very easy to read and clearly lays out the method.

### Weaknesses
While the proposed approach is interesting, the evaluation experiments in their current form are not sufficiently comprehensive. 
1.  The evaluation is based only on simple automated metrics (ROUGE, BLEU, METEOR, BERT-Score). To get a full understanding of how much FlyThinker improves personalization, it would be useful to have a user study or an automatic LLM evaluation of preferred personalized outputs.
2. The experiments are mostly limited to Qwen2.5-3B-Instruct. The small set of Qwen2.5-7B-Instruct experiments do not have any numbers on the axes, so it is difficult to tell how much FlyThinker improves performance over SFT. The experimental results would be strengthened by additional experiments with other models and filling out the Qwen2.5-7B-Instruct scores. 
3. From my understanding, compared to SFT, with this approach it is necessary to keep up to 2x the number of parameters in memory. The authors state that their method is efficient because both the reasoner and generator can perform inference simultaneously, but do not discuss the added memory required at training and inference time. Some discussion of training and inference memory requirements (in addition to the runtime results already included) would be appreciated.

### Questions
- Figure 5 has no values in the axes, so it is hard to tell how significant the differences in scores actually are.
- Could the authors clarify and further fill out the results in Figure 6? The authors claim that "Moderate values (0.5-2) yield the best overall performance", but do not test any intermediate values between 0.5 and 2. Also, there seems to be some instability in this range, especially for abstract generation. There are no values on the y axis for this graph, so it is difficult to tell how much $\lambda$ affects performance.
- It would be valuable to see some examples of personalized outputs produced by FlyThinker, for example for different reasoning model sizes or $\lambda$ parameters. This would make it more clear how exactly these factors affect the outputs of the generator.

Minor typo: line 180: "tought" instead of "thought"

### Soundness
2

### Presentation
3

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
This paper proposes FlyThinker, a “think-while-generating” framework for personalized long-form text generation. FlyThinker employs a Reasoner that generates latent token-level reasoning signals and a Generator that dynamically integrates these reasoning signals into its token-level predictions, enabling parallel training and inference. Experiments on three tasks from the LONGLAMP benchmark, including Product Review, Abstract Generation, and Topic Writing, demonstrate that FlyThinker achieves improvements in both personalization quality and generation efficiency over several baselines.

### Strengths
1. The “think-while-generating” paradigm is well-motivated, and the overall methodology is simple and intuitive.
2. The evaluation is comprehensive with multiple metrics, showing the effectiveness of the proposed method over baselines.
3. FlyThinker achieves training and inference efficiency comparable to SFT, which is a major advantage relative to existing reasoning-augmented methods that typically incur higher latency.

### Weaknesses
1. It is not clear whether the reported personalization results are based on the user-based split (testing on unseen users) or the temporal split (testing on later instances of seen users). Since these settings test different personalization abilities (cross-user vs. within-user), clarification or stratified results would make the findings more interpretable.

2. While the paper ablates on Reasoner size, it is not clear how the Reasoner scales relative to the Generator, e.g., what would be the smallest Reasoner that still remains effective for different Generator sizes - would it be roughly 30%, or 50% of the Generator size? Insights into this would provide very helpful practical guidance for applying FlyThinker in real-world settings.

3. Appendix H shows that adding reasoning tokens to both input and output positions yields the best performance. It is not clear how sensitive FlyThinker is to user history length, i.e., when each user has a lot of historical records with long-form generations, making the context very long. A discussion on this would strengthen the paper’s empirical insights.

4. Are the learned latent reasoning tokens interpretable, or could they be used for downstream applications such as user clustering or user preference visualization? Some discussions on this would offer insights into the interpretability of the latent reasoning tokens.

### Questions
Please refer to Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
