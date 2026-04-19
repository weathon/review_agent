# Zero-shot forecasting of chaotic systems

- Decision: Accept (Poster)
- Scores: 8, 6, 6

## Abstract
Time-series forecasting is a challenging problem that traditionally requires specialized models custom-trained for the specific task at hand. Recently, inspired by the success of large language models, foundation models pre-trained on vast amounts of time-series data from diverse domains have emerged as a promising candidate for general-purpose time-series forecasting. The defining characteristic of these foundation models is their ability to perform zero-shot learning, that is, forecasting a new system from limited context data without explicit re-training or fine-tuning. Here, we evaluate whether the zero-shot learning paradigm extends to the challenging task of forecasting chaotic systems. Across 135 distinct chaotic dynamical systems and $10^8$ timepoints, we find that foundation models produce competitive forecasts compared to custom-trained models (including NBEATS, TiDE, etc.), particularly when training data is limited. Interestingly, even after point forecasts fail, large foundation models are able to preserve the geometric and statistical properties of the chaotic attractors.
We attribute this success to foundation models' ability to perform in-context learning and identify context parroting as a simple mechanism used by these models to capture the long-term behavior of chaotic dynamical systems. Our results highlight the potential of foundation models as a tool for probing nonlinear and complex systems.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The present paper proposes to use time-series generated from chaotic systems as a new benchmark in time-series forecasting.
It makes the case for time-series foundation models (Chronos) which were pretrained on diverse synthetic and real-world data which it finds to yield better results than customly trained algorithms. This is surprising given that Chronos is a univariate prediction model and therefore sees the x, y, z components of the chaotic systems independently.

### Strengths
- The core contribution is a benchmark of foundation models on the evolution of different chaotic systems. I think this is a good contribution which goes into a similar direction as Chronos and ForecastPFN which use synthetic data (Chronos not exclusively) to pretrain their models and still obtain good performance. I agree with a reference in the conclusion stating that there appears to be common language shared by time-series as it's surprising that the existing synthetic data generators would result in such well performing models.
- I see potential for this benchmark to be used itself for more expressive synthetic data in time-series foundation models.
- I think the finding that chaotic systems can be learned better by foundation models from context rather than by a specific model is valuable and interesting.

### Weaknesses
- It would be good to give more details on the LSTM and transformer models shown in the paper in order to allow for a fair comparison. I couldn't find details on the number of layers used e.g. for LSTM after hyperparameter optimization like the number of layers or the state size of the LSTM.
- Could you clarify your rational on choosing the models you evaluated in the paper? Given the recent interest in linear RNNs like Mamba 1 / Mamba 2 / Gated Linear Attention, could you provide some results in this model class? Maybe Mamba 2, as it's fast to train.
- I think the paper in its current state is quite verbose and could benefit from a clearer structure. For example sections 5.1 and 5.2 could have individual paragraphs with titles indicating the finding which the authors would like to emphasize.

### Questions
-

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents a large-scale evaluation of the zero-shot learning capabilities of foundation models for chaotic systems. The authors investigate both short-term and long-term forecasting scenarios. They evaluate the foundation models against over 100 chaotic systems and observe several interesting findings. The key point is that foundation models not only generalize to new initial conditions but also to entirely new systems. This suggests the models have the capacity to capture the broader dynamical structures of chaotic systems.

### Strengths
The paper presents strong findings on the power of foundation models for zero-shot learning in chaotic systems. Firstly, it demonstrates that foundation models can perform as well as fully trained models across multiple scenarios, especially in long-term forecasting.

Another strength of the paper is its study and evaluation of the suitability of foundation models in the field of scientific machine learning. This proof of concept can definitely lead to new developments in the study of chaotic systems.

### Weaknesses
Further experiments are needed to assess sensitivity to initial conditions and real-world chaotic systems, as these factors currently weaken the proposed evaluation and findings.

While the application and findings of the paper are important, I believe the work in its current state lacks novelty due to limited experimental section.

### Questions
The authors aim to evaluate the proposed approach on real-world data? Interesting characteristics can be observed. 

How does foundations models forecast accuracy vary with different initial conditions within chaotic systems? This sensibility limits its generalization? 

I suggest the authors to state limitations of the zero-shot long-term prediction problem. Another possible limitation is how the foundation model zero-shot approach will behave when there is a drastic different dynamical behaviour between training and testing.

### Soundness
2

### Presentation
3

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
The article explores the application of foundation models, particularly zero-shot learning, to forecast chaotic systems. Using the foundation model Chronos, the study investigates whether these models can generate accurate short-term forecasts and capture the long-term statistical properties of chaotic attractors, which are characteristic of chaotic systems. Through an extensive evaluation on 135 chaotic systems, the authors found that, even without specific training on chaotic dynamics, Chronos provides competitive forecasts, preserving geometric and statistical attributes of chaotic attractors over the long term. The study presents these findings as a benchmark for the feasibility and limitations of using foundation models in scientific machine learning, especially in physics-informed tasks.

### Strengths
1. Extensive Benchmarking: The authors conduct a thorough and extensive evaluation, with a dataset that includes 135 chaotic systems, providing statistically robust results.
2. Foundation Model Scalability: By demonstrating the scalability of Chronos across varying chaotic systems, the study establishes the potential of foundation models for scientific machine learning in challenging domains.
3. Zero-Shot Learning Feasibility: The findings validate that foundation models can make meaningful forecasts in chaotic systems with minimal domain-specific adjustments, a promising step for generalizing AI across diverse scientific domains.
4. Insight into Long-Term Statistical Consistency: The study’s focus on long-term attractor behavior offers a novel way to assess model performance, moving beyond conventional short-term forecast accuracy.

### Weaknesses
1. The article mentions that Chronos’s forecast accuracy is sensitive to initial conditions, but the specifics of this dependency aren’t deeply explored. Could the authors provide a more systematic investigation into how initial condition variability impacts model robustness? For example, have you considered evaluating Chronos’s performance across a range of initial conditions sampled from different regions of the attractor, particularly comparing accuracy in central versus peripheral areas? Additionally, could you quantify how forecast accuracy shifts as the initial conditions deviate from those in the training data? Such analyses would offer valuable insights into Chronos’s generalization capabilities and help identify any specific sensitivities related to initial condition variability.
2. Positional Encoding and Temporal Structure: Given the chaotic nature of the systems studied, the choice of positional encodings (e.g., rotary embeddings) could significantly impact model performance, especially in maintaining temporal coherence over long horizons. However, the article lacks an in-depth discussion or ablation study regarding the choice of these encodings.
3. Limited Explanation of Performance in Extreme Cases: The authors mention that larger models tend to perform better, yet details on why model size improves stability in chaotic systems are sparse. A theoretical or empirical justification of this scaling behavior, such as through model capacity for capturing non-linear dynamics, would improve understanding.
4. Scalability of Model Parameters and Computational Requirements: Although Chronos shows promise in forecasting chaotic systems, the article could discuss practical limitations regarding computational costs, especially for larger model sizes, and the trade-offs compared to specialized models like NBEATS or TiDE.

### Questions
1. Impact of Initial Conditions: Can the authors elaborate on how initial conditions affect Chronos’s zero-shot forecasting stability? Is there a significant dependency on starting points, especially when far from the attractor’s typical state?
2. Effectiveness of Positional Encodings: How critical is the choice of positional encodings for Chronos when applied to chaotic systems? Have other encoding methods been tested, and if so, how did they affect both short-term accuracy and long-term attractor preservation?
3. Model Complexity and Forecasting Horizon: Given that larger models perform better in long-term forecasting, could the authors provide insights into the mechanisms by which increased model size aids in managing chaotic variability?
4. Comparison with In-Weights Fine-Tuning: Since the study focuses on zero-shot learning, have the authors considered comparing this approach with in-weights fine-tuning on a few chaotic trajectories to assess the performance improvement?

### Soundness
3

### Presentation
3

### Contribution
3
