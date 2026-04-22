# video-SALMONN S: Streaming Audio-Visual LLMs Beyond Length Limits via Memory

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 4, 6

## Abstract
Continuous, high-frame-rate, high-resolution processing of long video streams is critical for future AI agents, yet current video-understanding LLMs struggle to scale. Offline, fixed-frame-number methods require the stream length to adapt frame rates; streaming methods constrain memory by merging or discarding tokens, losing information. We propose video-SALMONN S, a streaming audio–visual LLM that processes arbitrary-length videos at 360p resolution under a fixed memory budget. The framework introduces (i) a test-time-training (TTT) memory module that continually updates token representations to capture long-range dependencies by replacing token merging, and (ii) a prompt-dependent memory reader that selectively retrieves context-relevant content from fixed-size memory. The TTT module is optimised with a Hessian-free conjugate-gradient procedure (TTT-HF) for efficient adaptation. On long-video benchmarks (Video-MME, LVBench, VideoEvalPro), video-SALMONN-S sustains high-quality understanding on multi-hour videos with >10k frames and ~1M tokens. Our 8B-parameter model achieves 74.2% overall and 67.8% on the Video-MME long split, significantly outperforming both offline and streaming baselines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a streaming-capable video-LLM architecture that processes extremely long videos via a test-time training (TTT) layer and a prompt-dependent memory-reading module. The TTT layer stores the salient history of previously seen inputs (similar to RNNs) and is optimized specifically with a Hessian-free second-order function. The outputted tokens are then further pruned using a prompt-dependent memory reading module. Extensive experiments, ablations, and discussions are provided to motivate certain choices made with their method, and quantitative results are shown on three long-video benchmarks. Specifically, they show their method can stream extremely long videos with stable accuracy and memory cost.

### Strengths
1. **Thorough discussion & ablations** - The authors provide detailed analysis of their design decisions: e.g., optimizer choice (SGD vs Muon vs. HF), optimal number of CG iterations, comparison with alternative methods of token-merging such as Mamba, etc. This supports the authors' specific design choices for each of their components.
2. **Strong Video-MME performance** - The reported performance of video-SALMONN S on Video-MME appears to SOTA for 7B-class models.
3. **Value of streaming extremely long videos** - The streaming capability of video-SALMONN S for very long-context is a strong practical contribution.

### Weaknesses
1. **Heavy reliance on prior work/limited novelty** - The authors transparently note that core modules (TTT layer, memory reader) are mostly borrowed from [1] and AdaReTaKe. While some of their adaptations are non-trivial, the novelty is still somewhat limited. The memory reader closely mirrors AdaReTaKe’s idea of prompt-guided KV selection, but adapts it for streaming long videos and modifies the budgeting strategy (one global Top-K rather than layer/time budgets). This adaptation is reasonable but modest.  Further, the TTT layer is kept exactly the same but simply trained with a different optimizer (see below).
 
2. **Missing/insignificant quantitative comparisons** - While I understand the proposed method is specifically tailored for streaming long videos, the authors do not provide a thorough comparison against non-streaming / long video understanding VLMs. For example, [2, 3] outperform almost every baseline (including video-SALMONN S) presented in this paper on LVBench, has competitive performance on MLVU, and is not included in the paper. Why are the improvements on LVBench negligible? Moreover, v-SALMONN-2 already appears to have competitive performance with v-SALMONN S across all datasets besides Video-MME, calling into question how much of an improvement the streaming aspect provides in long video understanding. This could potentially stem from long video datasets being unfit for truly multi-hour-long video streaming benchmarking.
On a similar note, some presented results show that improvements are insignificant. For example, Table 3 shows that the optimizer used in the TTT layer doesn't seem to really have any significant impact on the results (SGD vs. HF), yet is the only change the authors propose for the TTT layer and is discussed in detail in Section 3.2.

[1] Dalal, K., Koceja, D., Xu, J., Zhao, Y., Han, S., Cheung, K. C., ... & Wang, X. (2025). One-minute video generation with test-time training. In Proceedings of the Computer Vision and Pattern Recognition Conference (pp. 17702-17711).

[2] Chen, Y., Xue, F., Li, D., Hu, Q., Zhu, L., Li, X., ... & Han, S. LongVILA: Scaling Long-Context Visual Language Models for Long Videos. In The Thirteenth International Conference on Learning Representations.

[3] Zhang, Y., Wu, J., Li, W., Li, B., Ma, Z., Liu, Z., & Li, C. (2024). Video Instruction Tuning With Synthetic Data. In Transactions on Machine Learning Research

### Questions
1. Can the authors further motivate why their main contributions to the TTT layer and memory reader are significant? It appears the changes to TTT are trivial, and while the memory reader further equips v-SALMONN S to stream long contexts, it does not seem to yield large improvements on long video benchmarks.
2. Why are certain non-streaming model comparisons (see Weaknesses) left out? Either include all of them as comparisons, or discuss/exhibit why they are unfit for long-video understanding and how v-SALMONN S outperforms them.

I would be happy to increase my score if these questions (and those discussed in Weaknesses) are addressed.

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
3

### Summary
This paper proposes a method to address the memory bottleneck in long video sequence processing by applying the TTT concept to video streams. As a result, the network shows adaptive performance improvement during test time even with longer frame sequences, and is capable of processing videos of over 10K frames within a fixed memory budget. The method also features a Hessian-Free optimizer that enables efficient updates during testing.

### Strengths
- The model can handle long video sequences about 10K frames, mitigating the scalability limitation in long-term sequence modeling.
- The application of TTT to the video streaming environment is interesting, as it enables adaptive behavior at test time.

### Weaknesses
- The paper claims to preserve information via TTTHF instead of hard drop, but in practice, a cosine similarity based token discard step still exists. Because of this, the approach cannot be considered entirely different from traditional token-dropping methods, and the empirical evidence for actual information preservation remains unclear.
- Training is limited to a maximum of 1024 frames, and there is no quantitative analysis of memory efficiency (frame count vs memory usage) during training. The memory aspect thus remains insufficiently analyzed.
- The TTT concept has already been explored in prior video research (Test-Time Training on Video Streams), and similar adaptive mechanisms exist in RNN and Mamba architectures through state updates. It should be clarified whether the proposed approach's distinction is limited to the HF-based second-order optimization, or if there are structural differences beyond that.
- Experimental comparison with existing sequence compression approaches is lacking. For example, methods such as VideoLLM-online or Q-former could have been included for a fair evaluation against non-TTT token-reduction techniques, making it difficult to judge whether the proposed method is truly more effective.

### Questions
Please refer the weaknesses part.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
video-SALMONN S tackles the problem of doing audio–visual understanding on very long videos (hours, not minutes) when the model’s context and memory are fixed, which normally forces you to drop frames or heavily compress tokens and lose important events. The authors propose a streaming architecture with a test-time-training (TTT) style memory module that, as video chunks arrive, updates lightweight parameters to “write” what has happened so far, so the model doesn’t have to store every past token. They further make memory retrieval conditioned on the user’s current query, so at answer time the model can “read back” only the parts of memory that are relevant, keeping the overall memory size small but still supporting long-range reasoning. With an hessian free optimization to make the online updates stable and efficient, the system can process multi-hour videos (they report over 3 hours at 1 FPS, 360p) under a fixed memory budget and still outperform both offline and other streaming baselines on long-video benchmarks.

### Strengths
- Due to use of fixed-size memory the approach scales to really long videos

- Question/Prompt dependent token selection means efficient use of memory to store relevant tokens

- Test-time training updates the smart memory to represent past events in a compact manner

- Conjugate gradient method of test time update allows safer, better-scaled steps, while avoiding computing the full Hessian

### Weaknesses
- Prior works in streaming video models that utilize memory and token merging/reduction should be evaluated. e.g. CVPR 2025's "Streaming Dense Video Captioning". Xingyi Zhou, Anurag Arnab, Shyamal Buch, Shen Yan, Austin Myers, Xuehan Xiong, Arsha Nagrani, Cordelia Schmid


- Even though its a streaming video LLM the results are primarily on offline benchmarks, it would be better to have some results on streaming specific benchmarks like:

SVBench: A Benchmark with Temporal Multi-Turn Dialogues for Streaming Video Understanding. ICLR 2025.
Zhenyu Yang, Yuhang Hu, Zemin Du, Dizhan Xue, Shengsheng Qian, Jiahong Wu, Fan Yang, Weiming Dong, Changsheng Xu

StreamingBench: Assessing the Gap for MLLMs to Achieve Streaming Video Understanding
Junming Lin, Zheng Fang, Chi Chen, Zihao Wan, Fuwen Luo, Peng Li, Yang Liu, Maosong Sun

-

### Questions
- Since there are some efficiency considerations for this approach, could you provide some benchmarking score in terms of frames/seconds for your approach and baselines on actual hardware used for training.

### Soundness
3

### Presentation
3

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
This paper presents Video-SALMONN S, a scalable framework for **streaming multimodal video understanding** that enables large video-language models to process arbitrarily long videos under fixed memory and computation budgets. The core innovation lies in combining Test-Time Training with Hessian-Free optimization (TTTHF)—which updates visual memory parameters during inference to retain long-term information—with a prompt-dependent memory reading mechanism that selectively retrieves task-relevant visual tokens, and a similarity-based discarding strategy to prevent unbounded KV cache growth. Extensive experiments on multiple benchmarks, including Video-MME, LVBench, MLVU, and VideoEvalPro, show that Video-SALMONN S achieves superior performance and efficiency over recent streaming baselines (e.g., StreamMem, Dispider, MovieChat), supporting up to three-hour-long videos while maintaining competitive accuracy. Overall, the paper provides a well-engineered and empirically validated solution to the scalability bottleneck of streaming video-language models.

### Strengths
1. The integration of Test-Time Training with Hessian-Free optimization (TTTHF) and prompt-dependent memory reading is novel and effectively addresses the long-standing problem of unbounded visual KV cache accumulation in streaming settings. This design provides a fresh perspective by merging adaptive optimization with memory-efficient retrieval, extending prior works such as MovieChat, StreamMem, and ReTaKe toward a truly long-duration streaming paradigm. 

2. The experimental evaluation is comprehensive and convincing, spanning multiple long-form benchmarks (Video-MME, LVBench, MLVU, and VideoEvalPro). Results demonstrate consistent improvements in both accuracy and efficiency, supporting up to three-hour-long videos under fixed GPU memory—an impressive engineering achievement. Ablation studies thoroughly validate the necessity of each module (TTTHF, similarity-based discarding, and prompt-dependent retrieval), showing clear causal links between design choices and performance gains.

### Weaknesses
1. The similarity-based discarding mechanism, although efficient, inherently risks removing semantically critical frames or rare events. The paper lacks a detailed analysis of potential information loss and does not provide qualitative examples or failure cases illustrating where compression harms temporal or semantic fidelity. 

2. The proposed TTTHF module introduces additional test-time optimization overhead, yet the paper provides only high-level runtime statistics without precise measurements of latency, FLOPs, or energy consumption. Moreover, details such as GPU hours, batch configurations, and training cost breakdown are missing, making it difficult to assess real-world deployability and reproducibility. The “test-time training while streaming” paradigm incurs extremely high computational costs, making it difficult—if not impossible—for the model to achieve truly real-time streaming performance.

### Questions
1. The similarity-based discarding strategy effectively constrains memory usage but may also risk removing rare yet semantically important frames. Could the authors provide further analysis or visualization to quantify how much semantic or temporal information is lost due to discarding?

2. TTTHF adds test-time optimization overhead, yet the paper reports only aggregate runtime metrics. Could the authors include detailed latency, FLOPs, and energy profiling under different settings, along with training cost (GPU hours, batch size, memory)? Finally, are there lightweight or approximate TTTHF variants that maintain performance while reducing computation?

### Soundness
4

### Presentation
3

### Contribution
3
