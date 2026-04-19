# Test Time Learning for Time Series Forecasting

- Decision: Reject
- Scores: 3, 5, 5, 5

## Abstract
We propose the use of Test-Time Training (TTT) modules in a cascade architecture to enhance performance in long-term time series forecasting. Through extensive experiments on standard benchmark datasets, we demonstrate that TTT modules consistently outperform state-of-the-art models, including Mamba-based TimeMachine, particularly in scenarios involving extended sequence and prediction lengths. Our results show significant improvements, especially on larger datasets such as Electricity, Traffic, and Weather, underscoring the effectiveness of TTT in capturing long-range dependencies. Additionally, we explore various convolutional architectures within the TTT framework, showing that convolutional blocks as hidden layer architectures can achieve competitive results.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
It introduces a test-time learning module to replace the Timesmachine's design module for time series forecasting.

### Strengths
Using Test Time Learning for time series forecasting is worth studying.

### Weaknesses
Clarification is needed on whether this is a module or a model. If it is a module intended to enhance performance, as suggested, it should be adaptable across various models, and further experiments are necessary. If it is a standalone model, please refine the writing accordingly. Additionally, the performance improvement compared to TimeMachine is relatively minor. I typically avoid using this as a main argument, as there are other valuable insights beyond achieving state-of-the-art results, which I feel are lacking in this work.The problem it claims to address is weak, and the discussion should be more focused. The modification appears to focus on adjusting a module from the TimeMachine paper. If the goal is to address super long-term forecasting, this experiment should be highlighted in the main results table and compared with other works that claim proficiency in super long-term forecasting.

### Questions
Could you please provide more explaination for the weaknesses section?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper explores the use of Test-Time Training (TTT) modules in parallel architectures for time-series forecasting. The authors aim to enhance performance on long-term time-series forecasting by proposing an architecture that integrates TTT with various backbones. The authors also provide the numerical evaluation for convolution based TTT (according to Table 1).

### Strengths
1. Usage of test-time-training could be able to address the potential distribution drafting issues in practice.

### Weaknesses
1. The writing could be improved. At the beginning of the draft, the authors claim the proposed structure is model-agnostic. During the numerical experimental section, the main results (i.e., Table 1), it seems only a convolution network is considered but no combination with existing models (e.g., PatchTST and Itransformer) is presented. If the authors think the convolution structure is important and the proposed TTT is model-specific. Otherwise, please provide more comprehensive numerical experiments.

2. Can authors elaborate more on the difference between the test time training (TTT) and test time adaptation (TTA)? It seems several recent literature is absent. 

3. The numerical results (e.g., Table 1) are not strong enough, the TTT seems almost on par aganist timemachine model.

### Questions
Please see the weakness part.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper presents a novel approach to enhance long-term time-series forecasting (LTSF) through the use of Test-Time Training (TTT) modules. Traditional forecasting models, particularly those based on Transformers, face challenges in managing computational costs and effectively capturing long-range dependencies. To address these limitations, the authors propose integrating TTT modules into the TimeMachine model (a state-of-the-art architecture for LTSF) replacing the Mamba modules for greater adaptability and dependency capture during test time. The authors' primary contributions include the development of a new model architecture with quadruple TTT modules, specifically designed to handle extended sequence and prediction tasks. Through extensive benchmarking on datasets such as Electricity, Traffic, and Weather, the TTT-based model demonstrates notable improvements in forecasting accuracy compared to TimeMachine and other leading models. Additionally, the authors explore various convolutional configurations, such as Conv Stack 5, within the TTT framework, finding that even relatively simple configurations yield significant performance gains. This adaptability and scalability allow the TTT model to outperform competing models, especially when applied to larger prediction and sequence lengths. By establishing TTT as a powerful and scalable solution for LTSF, the paper sets a new benchmark and paves the way for further research into high-performance forecasting models.

### Strengths
1. Originality :

The paper demonstrates a high degree of originality by introducing Test-Time Training (TTT) modules specifically for time-series forecasting. This approach innovatively allows models to adapt during testing, a novel contribution in the field of forecasting where most models rely solely on training data and fixed parameters during inference. By integrating TTT modules with linear RNNs and exploring contextual hierarchies (high-resolution and low-resolution levels), the paper provides a fresh approach to capturing long-term dependencies—a significant challenge in time-series forecasting. The use of TTT for dynamic adaptation in response to evolving data patterns, especially without altering primary model parameters, marks a clear advancement over traditional models. It can be elaborated into :

Test-Time Training (TTT) Module: This module enables weight updates during the testing phase, allowing the model to adapt to varying data patterns. This makes TTT effective in capturing complex patterns for long-term predictions.

Performance Improvement: Through experiments on benchmark datasets such as Electricity, Traffic, and Weather, the TTT-based model shows significant performance gains compared to other state-of-the-art models like Transformer-based models and Mamba. Performance indicators, including Mean Squared Error (MSE) and Mean Absolute Error (MAE), demonstrate that the TTT model is better suited for predictions with longer sequence lengths and prediction horizons.

Ablation Study on Convolutional Architecture: This paper also explores several convolutional architecture configurations within the TTT framework, such as Conv Stack 5 and ModernTCN. The results indicate that simpler convolutional configurations can deliver competitive results without adding unnecessary complexity.

2. The paper’s quality is above average, particularly in its extensive and comprehensive empirical validation across multiple benchmark datasets, such as Electricity, Traffic, and Weather. The experimental design is robust, including comparisons with state-of-the-art (SOTA) models like Transformer and Mamba-based architectures. Additionally, the ablation study exploring different convolutional architectures (e.g., Conv Stack 5, ModernTCN) within the TTT framework shows careful attention to model component effectiveness and efficiency. Furthermore, performance metrics, including Mean Squared Error (MSE) and Mean Absolute Error (MAE), provide clear evidence of the model’s capabilities and improvements over existing models, adding rigor to the evaluation.

3. The clarity of the paper is commendable, especially in presenting the layered architecture of TTT modules and explaining the two-level hierarchy for capturing contextual cues at high and low resolutions. This structuring aids in understanding the model’s mechanics and how it addresses long-range dependencies. The presentation of channel mixing and independence modes is also clear, helping readers comprehend the adaptation of the model to multivariate time-series data. However, there may still be room to expand theoretical explanations of TTT’s advantages, particularly regarding why selective adaptation during test time enhances stability and adaptability, which could further elevate clarity for a broader audience.

4. The significance of this work is substantial due to its relevance in advancing time-series forecasting, particularly for long-term forecasting tasks that require handling complex temporal dependencies across large data streams. Given the growing use of time-series forecasting in fields like energy management, traffic monitoring, and financial prediction, a model that can adapt dynamically to new patterns has a wide range of practical applications. Moreover, by maintaining model efficiency through selective parameter updating during testing, TTT addresses a pressing need for models that can scale with real-time demands while remaining computationally feasible. This innovation not only pushes forward the boundaries in time-series forecasting but could also inspire similar adaptive techniques in other sequential modeling domains.

### Weaknesses
1. Theoretical Justification and Intuition

While the empirical results demonstrate TTT’s effectiveness, the paper lacks an in-depth theoretical justification for why selective adaptation during testing works effectively for capturing long-term dependencies. The proposed model could benefit from a more comprehensive discussion of the underlying mechanisms of TTT and why it succeeds in enhancing the model's ability to adapt to long-range dependencies without suffering from catastrophic forgetting. Strengthening the theoretical foundation with a mathematical analysis of TTT’s adaptability, particularly how it avoids catastrophic forgetting during selective adaptation, could be a valuable addition. It would be beneficial to reference existing theoretical frameworks on continual learning (e.g., Kirkpatrick et al., 2017 on Overcoming catastrophic forgetting in neural networks) and contrast how TTT’s unique selective adaptation minimizes memory interference, particularly in long-term sequence contexts.

2. Computational Complexity and Practicality

While TTT shows strong empirical performance, the paper does not sufficiently discuss the computational overhead associated with adapting the model during testing. Test-time training requires frequent parameter updates, which could become computationally expensive, especially for real-time applications or in settings with limited resources. There is also no analysis comparing the memory and processing requirements of TTT versus the baseline models, making it difficult to assess the model’s practical feasibility. It would be better to include a detailed analysis of computational efficiency, such as memory consumption, latency, and runtime, particularly in high-frequency and high-volume time-series contexts. If TTT proves resource-intensive, consider exploring optimization strategies, like limiting parameter updates to specific intervals or only for data points that deviate significantly from training data, which could enhance efficiency without sacrificing adaptability.

3. Insufficient Exploration of Convolutional Variants

The paper does conduct ablation studies on convolutional configurations (e.g., Conv Stack 5 and ModernTCN), but these variations remain relatively limited and do not fully explore convolutional architectures that could better capture multiscale temporal dependencies. As a result, it remains uncertain whether the TTT-based models are achieving optimal use of local temporal patterns within each channel and across channels in the multivariate context. We suggest broadening the convolutional architecture exploration to include dilated convolutions or multi-resolution features, which could potentially improve performance on long-term dependencies. Additionally, incorporating a study on whether certain convolutional structures offer better adaptability with TTT could provide deeper insights into architectural choices that optimize TTT’s efficacy.

4. Overlooked Explanation of Failure Cases

The paper does not discuss scenarios where TTT might underperform or fail to deliver the expected benefits, which could be valuable for understanding limitations and areas for further improvement. For instance, if the model encounters unexpected shifts in data distributions or operates in domains with high-frequency noise, its adaptive capabilities may struggle, impacting overall performance. Adding a failure case analysis or discussion of situations where TTT might be less effective would be beneficial for readers. An evaluation of TTT’s robustness against significant distribution shifts or noise would provide a more balanced perspective. If these analyses indicate limitations, the authors could suggest future research directions, such as integrating TTT with methods for noise-robust learning or investigating methods to dynamically adjust the degree of test-time adaptation based on input characteristics.

5. Benchmark Comparisons and Real-World Simulation

While TTT is validated on benchmark datasets, these may not fully represent real-world complexities, such as fluctuating conditions and anomalies in time-series data. Also, TTT’s performance is primarily compared with Mamba-based and Transformer models, but comparisons with state-of-the-art time-series forecasting methods that employ different adaptation or regularization mechanisms would help place TTT’s effectiveness in context. Extending the experiments to include real-world or more challenging datasets that capture diverse time series behaviors, such as anomalies or irregular patterns, would effectively stress-test TTT’s adaptability. Including comparisons with models that use temporal regularization (e.g., DeepAR, TCN) or adversarial learning techniques would also provide additional benchmarks to validate TTT’s position as a state-of-the-art adaptive method.

### Questions
What is the theoretical basis for Test-Time Training’s (TTT) adaptability without catastrophic forgetting? Could the authors elaborate on the theoretical intuition behind why selective parameter updates at test time, as done in TTT, help capture long-term dependencies without leading to catastrophic forgetting? A more detailed theoretical framework or references to relevant works (e.g., continual learning or meta-learning) would be helpful in understanding the robustness of TTT’s adaptation.

How does TTT handle extreme distribution shifts? In cases where data distributions shift significantly, such as in time-series data with high volatility or sudden anomalies, how does TTT manage to adapt without compromising previous knowledge? Empirical or anecdotal insights on TTT’s performance under extreme shifts would clarify its robustness, as well as the practical limits of test-time adaptation in real-world applications. What is the computational overhead introduced by TTT during test time?

How does TTT affect computational resources, such as memory usage, latency, and runtime, particularly in high-frequency or large-scale data scenarios? If the authors could quantify the test-time computational costs of TTT relative to baseline models (e.g., Transformer, Mamba), this would provide clarity on the practicality and scalability of TTT for real-time applications. Why were specific convolutional architectures selected for the ablation study?

Could the authors explain their choice of Conv Stack 5 and ModernTCN in the ablation study and provide any reasoning for omitting other architectures, such as dilated or multi-scale convolutions? An understanding of these choices would help readers assess the comprehensiveness of the convolutional architecture exploration and potential gains in performance. How generalizable is TTT to other sequence modeling tasks beyond time-series forecasting?

The paper demonstrates TTT’s efficacy on time-series forecasting, but to what extent might it be adaptable to other sequential tasks, such as language modeling or event prediction? If generalization is anticipated, the authors could discuss potential adaptations required for TTT in these tasks, or conversely, explain the unique suitability of TTT to time-series data. Does TTT benefit from any specific parameter initialization strategies?

Given TTT’s adaptation at test time, does it benefit from certain parameter initialization methods, particularly in the hidden states? Clarification on initialization methods, if any, that enhance TTT’s adaptability would give insight into optimal configurations and whether further fine-tuning might improve TTT’s performance.

Suggestions

1. Incorporate a failure case analysis for TTT in extreme settings.

Including a discussion or experiment that explores the limitations of TTT in scenarios with sudden distribution shifts or heavy noise would provide a more balanced perspective. If the authors have preliminary findings on failure cases, this could help readers understand when TTT might be less effective, along with areas for potential improvement.

2. Expand on the choice of hierarchical two-level context modeling.

The hierarchical design in TTT for high and low-resolution contexts is an innovative approach, but could the authors provide further intuition or references for why this structure specifically benefits time-series forecasting? This would clarify how hierarchical cues improve TTT’s ability to model long-term dependencies and adapt to multiscale patterns within time-series data.
Consider additional benchmarks with noise robustness or temporal regularization methods.

To further validate TTT’s position relative to the state of the art, additional comparisons with models that handle noise or temporal regularization (e.g., DeepAR, TCN) could strengthen empirical findings. If direct comparisons are impractical, a discussion on how TTT theoretically or practically compares with these methods would be informative.
Add computational efficiency metrics in experiments.

Including memory consumption, training time, and inference time as part of the experimental results would help assess TTT’s trade-offs in resource utilization. If TTT is found to be resource-intensive, the authors could discuss strategies for balancing adaptability and efficiency, especially in real-time or resource-constrained environments.

3. Discuss potential applications of TTT in real-world systems.

An analysis of how TTT could be applied in real-world scenarios, such as financial prediction or adaptive traffic monitoring, could showcase its practical relevance and advantages. Simulated or hypothetical case studies would also help highlight TTT’s robustness and adaptability. Consider further discussion of generalization beyond time-series forecasting.

If TTT’s applicability is anticipated for other sequence modeling domains, discussing any potential adaptations (or limitations) of TTT would be valuable. If generalization is limited, clarifying why TTT is best suited to time-series forecasting would help contextualize its strengths.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper proposes using Test-Time Training (TTT) modules in a parallel architecture for long-term time series forecasting. Experiments on benchmark datasets show that TTT modules outperform state-of-the-art models, especially in handling long sequences and predictions. The paper also explores different convolutional architectures within the TTT framework and finds that simple configurations can achieve good results.

### Strengths
1. The use of TTT modules in a parallel architecture is a novel approach for long-term time series forecasting.
2. The paper provides extensive experimental results on standard benchmark datasets, demonstrating the effectiveness of the proposed method.

### Weaknesses
1. The replacement of Mamba with TTT moderates the architectural innovation.
2. Although the paper explores different convolutional architectures, it may have yet to fully explore all possible architectures.
3. The proposed method's theoretical analysis could be further strengthened to provide a deeper understanding of its performance improvement.
4. The experiments on ETTh2 and ETTm2 reveal that TTT may not always beat TimeMachine.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
2
