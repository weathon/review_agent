# Real-Time Video Generation with Pyramid Attention Broadcast

- Decision: Accept (Poster)
- Scores: 8, 8, 6, 6

## Abstract
We present Pyramid Attention Broadcast (PAB), a real-time, high quality and training-free approach for DiT-based video generation. Our method is founded on the observation that attention difference in the diffusion process exhibits a U-shaped pattern, indicating significant redundancy. We mitigate this by broadcasting attention outputs to subsequent steps in a pyramid style. It applies different broadcast strategies to each attention based on their variance for best efficiency. We further introduce broadcast sequence parallel for more efficient distributed inference. PAB demonstrates up to 10.5x speedup across three models compared to baselines, achieving real-time generation for up to 720p videos. We anticipate that our simple yet effective method will serve as a robust baseline and facilitate future research and application for video generation.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes an accelerated algorithm for video generation that is training-free. They observe the redundancy in attention calculations and adopts different acceleration strategies for different attention mechanisms. This paper gives detailed quantitative and qualitative results.

### Strengths
1)This paper is easy to follow and the code is open source.
2)he acceleration effect is obvious from the qualitative and quantitative experimental results, 
3)The ablation experiment is thorough

### Weaknesses
1)This is essentially a cache technology, and the idea has been studied before, such as T-GATE and DeepCache.
2)The generalization of this method is questionable in scenarios with long video duration, complex textures, and drastic dynamic changes. For more details, please refer to Question 2 below.

### Questions
1)In Table 1, the PAB range is different for different models. Besides, the diffusion timesteps in the Appendix are different, such as Table 6 in the Appendix? Does this mean that these hyperparameters need to be determined experimentally? Will the hyperparameters used for the same model be very different when trained with different data?
 2)When generating long (greater than 1 minute), complex texture, high dynamic range videos, will acceleration lead to more quality loss. Can you give some quantitative results on how video quality metrics change as video length and complexity increase
 3)In Figure 11, why does the attention-related operation take up such a large proportion of the time? Are there other operations besides normalization and projection? Can you give a more detailed breakdown of the attention-related operations？

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes Pyramid Attention Broadcast (PAB), which is a real-time, high-quality, and training-free approach for DiT-based video generation. Motivated by the observation that attention difference in the diffusion process exhibits a U-shaped pattern, indicating significant redundancy. This paper broadcast attention outputs to subsequent steps in a pyramid style. PAB demonstrates up to 10.5x speedup across three models compared with baselines, achieving real-time generation for 720p vdeos.

### Strengths
1. Figure 2 vividly illustrates the motivation, and the results appear highly plausible, with the spatial attention difference being the most significant, followed by the temporal attention difference, and then the cross attention difference.

2. The proposed broadcasting method appears to be straightforward, potent, and adaptable.

3. Broadcasting the entire sequence in parallel significantly enhances inference speed. It seems that sequence parallelism is exceptionally effective for accelerating the process.

### Weaknesses
* Why opt for the Mean Squared Error (MSE) metric to assess redundancy, and does the MSE metric accurately reflect the nuances of redundancy?

* It appears that utilizing broadcasting may lead to an increase in communication time. How can we quantify this additional overhead?

* Is it feasible to apply the PAB (Parallel Attention Broadcasting) technique to T2I (Text-to-Image) models, such as FLUX? Given that current T2I models are becoming increasingly large, there is a pressing need to accelerate models that exceed 5 billion parameters.

### Questions
Overall, I find this to be a well-crafted paper. Nevertheless, I am curious if there might be a more suitable metric for assessing redundancy. Additionally, I wonder if it's possible to dynamically select layer-wise broadcasting in conjunction with distillation-based methods to facilitate real-time video generation.

### Soundness
4

### Presentation
4

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
This paper proposes Pyramid Attention Broadcast (PAB), a method for DiT-based video generation that leverages a pyramid-shaped broadcast design. This design is inspired by two key observations: (1) the attention output differences are minimal in the middle 70% of the diffusion steps, and (2) spatial, temporal, and cross-attention differences decrease in a hierarchical, pyramid-like manner. Based on these insights, PAB sets different broadcast ranges for each attention type. The method is further extended to distributed settings, supporting multi-GPU parallelism (e.g., 8-GPU, 16-GPU) to further reduce generation latency. Experiments on VBench and WebVid using three types of DiT models demonstrate that PAB effectively reduces latency and accelerates video generation.

### Strengths
The method’s effectiveness is demonstrated on multi-GPU setups, achieving real-time video generation (e.g., within 2 seconds) on an 8-card H100 configuration. This efficiency is also validated across multiple open-source Video DiT models.

### Weaknesses
1. In Figure 4, the quantitative analysis of attention differences uses 30 inference steps based on Open-So ra, but the visualized attention differences cover 50 steps, creating inconsistency. Is this visualization based on Latte? Additionally, it’s unclear whether the attention feature difference plot is derived from an entire dataset analysis or a single video generation process.

2. Some comparisons appear to be unfair. For example, in Figure 7, the original generation using a single device is compared to an 8-device setup, leading to a claimed 10.5x speedup. This comparison may be misleading and should be clarified.

3. Key implementation details should be presented in the main paper. For instance, the most crucial setting, **PAB_246**, which defines the spatial, temporal, and cross-attention broadcast ranges, should be clearly explained rather than shown in supplementary materials.

4. Broadcasting attention is not entirely new. For instance, TGATE, an image generation acceleration approach, also uses iterative caching and reusing of self-attention (SA) for acceleration. This reduces the novelty of the proposed approach.

5. The qualitative results focus primarily on natural, static scenes. In real-world applications, generating videos with complex, dynamic actions, such as those involving people or animals, is more impactful. The qualitative evaluation scope is therefore too limited to showcase the method’s full potential in diverse scenarios.

### Questions
The broadcast sequence parallelism in this work relies on DSP, so it’s unclear how much of the latency reduction is directly attributable to PAB alone. In Table 4 or in the main text, it would be helpful to further clarify PAB’s independent contribution to latency improvements.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper propose one technique, namely pyramid attention broadcast (PAB), which is a training-free for speed-up DiT-based video generation.
The idea is simple, which broadcast the attention results for the spatial, temporal, cross attention, which consumes more computation in video DiTs than that in CNN approaches.
The proposed PAB works well in OpenSora, Open-Sora-Plan, and Latte, demonstrating the effectiveness of the proposed method.

### Strengths
- The motivation is clear, which relies on one training-free method to speedup the video generation (inference).
- The attention redundancy in video DiTs are extensively studied, such as the attention cost, the attention patterns, the attention similarity and diversity. 
- With the extensive studies on attention redundancy, the authors proposed the pyramid attention broadcast. Moreover, different broadcast ranges for each attention are tailored based on the rate of change and the stability of each attention type.
- Experiments and ablation studies are conducted to demonstrate the effectiveness and efficiency of the proposed PAB metric.

### Weaknesses
- The proposed PAB is simple and effective. However, the optimal solution for PAB, such as how to determine the optimal broadcast ranges are not specified. According to the Figure. 8, the larger broadcast range, the low latency and low video generation quality. Such studies seed to be obvious to the authors. For the specific video generation model, such as open-sora, open-sora-plan, latte, how can we set the broadcast range. Please provide more discussions.
- The authors claimed that the broadcast the attention outputs other than attention scores. In my opinion, the effect seems to be the same or very similar for broadcasting the attention scores outputs. Please provide more explanation.
- One important questions is that the authors reveal that the attention outputs (spatial, temporal, cross attention) are redundancy. As such, they can broadcast the attention outputs with one training-free to speedup the video generation. In my opinion, such studies also means that the existing video generation model, such as open-sora, open-sora-plan, latte, learns a lot of redundancy information, which results in the redundancy attention outputs. Therefore, can PAB helps to learn effectively of the video generation model. Please make more discussions.

### Questions
Please check the detailed comments in the weaknesses part.

### Soundness
3

### Presentation
3

### Contribution
2
