# StreamingVLM: Real-Time Understanding for Infinite Video Streams

- Decision: Accept (Poster)
- Scores: 4, 8, 6, 6

## Abstract
Vision-language models (VLMs) could power real-time assistants and autonomous agents, but they face a critical challenge: understanding near-infinite video streams without escalating latency and memory usage.
Processing entire videos with full attention leads to quadratic computational costs and poor performance on long videos. Meanwhile, simple sliding window methods are also flawed, as they either break coherence or suffer from high latency due to redundant recomputation.
In this paper, we introduce **StreamingVLM**, a model designed for real-time, stable understanding of infinite visual input. Our approach is a unified framework that aligns training with streaming inference. 
During inference, we maintain a compact KV cache by reusing states of attention sinks, a short window of recent vision tokens, and a long window of recent text tokens. 
This streaming ability is instilled via a simple supervised fine-tuning (SFT) strategy that applies full attention on short, overlapped video chunks, which effectively mimics the inference-time attention pattern without training on prohibitively long contexts.
For evaluation, we build **Inf-Streams-Eval**, a new benchmark with videos averaging over two hours that requires dense, per-second alignment between frames and text.
On Inf-Streams-Eval, **StreamingVLM** achieves a **66.18%** win rate against GPT-4O mini and maintains stable, real-time performance at up to 8 FPS on a single NVIDIA H100.
Notably, our SFT strategy also enhances general VQA abilities without any VQA-specific fine-tuning, improving performance on LongVideoBench by +4.30 and OVOBench Realtime by +5.96.
Code will be released upon publication.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This work presents StreamingVLM, a vision–language model trained for near-infinite video understanding. To enable low-latency inference under a fixed GPU-memory budget, the method maintains a compact KV cache that reuses (i) persistent attention-sink states, (ii) a short window of recent vision tokens, and (iii) a long window of recent text tokens. To teach the model this input pattern without training on prohibitively long contexts, the authors adopt an overlapped-chunk, full-attention SFT strategy that mimics the inference-time attention layout. They also curate a sports-focused training set and introduce a benchmark, Inf-Streams-Eval, targeting long-video understanding(Captioning) with real-time narration. On this benchmark, StreamingVLM reports gains over strong long-video baselines and GPT-4o, and it shows modest improvements on general VQA benchmarks as well.

### Strengths
1. The KV cache design including persistent attention sinks, a long text window, a short vision window, plus contiguous RoPE shifting is conceptually sound and easy to implement. It delivers lower latency and reduced GPU memory usage while outperforming baselines on the proposed benchmark.


2. The overlapped-chunk, full-attention SFT strategy is an intuitive way to teach the model the streaming cache pattern without training on prohibitively long contexts.


3. The curated long-video captioning corpus and the Inf-Streams-Eval benchmark fill a gap for real-time, long-horizon evaluation in VLMs and should be valuable to the community.


4. The paper is well organized and clearly written, making the method and experiments easy to follow.

### Weaknesses
1. The choice to retain 512 sink tokens, a 512-token text window, and a 16-second vision window appears empirical and manually tuned. Moreover, the ablation supporting this setting is run primarily on a basketball-only subset, which raises concerns about domain generality. A single fixed policy may either waste KV budget in slow scenes or evict critical evidence too early in fast ones. Therefore, more adaptive and fine-grained design would be expected. 

2. Current KV cache eviction follows a naive FIFO rule, and there is no scoring or compression to retain semantically salient frames. This may harm long-horizon reasoning, especially for sparse-action videos. Therefore, this weakness can undermine the generalization of the proposed method.

3. The proposed benchmark has leakage risk. Sports broadcasts are heavily duplicated and often reuse the same commentary audio, so near-duplicates can slip across training dataset and benchmark. The paper does not document near-duplicate filtering, making memorization of phrasing/style a real possibility that could inflate results. 

4. This work only shows results on SFT Qwen2.5-VL-Instruct-7B, making it hard to claim the method is architecture agnostic or base model agnostic.

### Questions
1. From Table 5, we can see that when the values of $T_{sink}=512$ and $T_{window}=512$ in the inference stage are the same as those in the training stage, the overall performance is the best. This suggests the model is tightly tuned to a single window geometry, and performance degrades when the inference budget diverges, indicating limited robustness. What happens if you vary these values during training? Have you tried other values in SFT or using a curriculum scheme(start larger, anneal to target), then evaluating under both matched and mismatched inference windows? 

2. Why does StreamingVLM leverage 512 sink tokens? How sensitive is performance to reducing number of sinks​? and is there a saturation point where adding more sinks yields little or no gain?

3. Table 3 does not compare StreamingVLM with Livecc-7B-Instruct, what does Livecc-7B-Instruct perform on those general VQA tasks? 

4. How much of the reported improvement (e.g. Table 6) is due to the streaming method (KV-reuse + contiguous RoPE + in-domain SFT) versus the in-domain SFT data?

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
This paper introduces StreamingVLM, a vision-language model framework designed for real-time, long-horizon video understanding. 

This framework aligns training and inference by training on short chunks with original attention, and during inference maintaining a compact KV cache with attention sink, short visual window, and long text window.

This paper also introduces a new dataset and Inf-Streams-Eval benchmark which consisting of long sports videos with ASR and commentary annotations.

### Strengths
1.Clear motivation. The paper focuses on a real and underexplored problem—achieving real-time video understanding under limited latency and memory constraints.

2. Simple and clear method. The proposed attention sink + sliding window + contiguous RoPE mechanism is simple, elegant, and demonstrates good empirical performance.

3. Valuable dataset. The introduced dataset makes a meaningful contribution to the community of real-time long-video understanding.

4. Clear writing and easy to follow.

### Weaknesses
1. Experiments focus mainly on sports videos. It remains unclear how well the model generalizes to other domains such as egocentric or instructional videos.

2.  Although the Inf-Streams-Eval benchmark is valuable, it relies on GPT-based judgment for scoring, which may introduce bias.

### Questions
1. Could the authors provide more details on the data annotation and filtering process, such as examples of removed or edited segments?

### Soundness
3

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
4

### Summary
This paper proposes to tackle that challenge of enabling vision-language model to process long video streams in real-time. To achieve that, the authors propose to to align the training and streaming inference using two main design ideas:
* It uses a compact streaming-aware KV cache mechanism, which only keeps a small of attention-sink tokens, a text window and a short visual token window. 
* It introduces a contiguous RoPE to prevent positional drift by reindexing the tokens to stay within a bounded range. 

To evaluate the algorithm, it introduced two new datasets for streaming the video understanding. 
* It introduce Inf-streams-train, a 4000-hour dataset sports commentary dataset curated using ASR and GPT. 
* A benchmark dataset Inf-Streams-Eval with good per-second alignment of frames and text to test the infinite streaming. 
The authors provide extensive evaluations using model comparisons and show the proposed design is effective to handle long videos, and can improve step by step when incorporating the proposed mechanism.

### Strengths
Overall the paper is well motivated, and demonstrates convincing results that can advances real-time long-horizon video understanding. 
* The proposed KV-cache mechanism composed with attention-sink tokens, long text window and short vision window is effective from the evaluation, and the contiguous ROPE can further yield stable output with improved performance. 
* It further proposed a dataset that can train the model with higher quality dataset, and an evaluation mechanism to benchmark the progress. The process to create the dataset is legit and considers the important factors that limits existing datasets. If they author can share them out, it will benefit the community a lot.

### Weaknesses
Overall the paper aces well in a number of engineering factors to make the current system solid. However, there are many design choices that embed strong heuristics in them and unclear what they will terminate at. 
* The model is trained on overlapped short video chunks and never experiences true streaming behavior with recurrent KV use. It is not quite clear whether it is "training-inference alignment" claimed by the author. In Table 2, it shows alignment is important (where ReKV completely fails), while it is a very differnet mechanism and not optimized for multimodal use. It could be done better if the author can show a comparison to the same model developed (but without stream-aware training), and shows the difference. 
* There are also finite token length limit for attention-sink, visual token and text token windows. It is not quite clear to me how they impact the final results if the scenario varies. 
* From the results, we can clearly see the model can already achieve really good performance compared to the baselines (GPT-4, LiveCC) even without using T_sink or T_window. I wonder how much does the base model and training on the created dataset contribute. Ideally, we want to factor out them out in performance evaluations.

### Questions
I enumerate a few questions in the weakness part, which are mostly about how the paper improve the clarity. Hope the authors can provide me a few evidences to address them in the easiest setting, or pointing me to the right source if I missed any. 

The dataset used for training and benchmark is very important part in this part to make model great and evaluation solid. I wonder whether they are available to the community in some ways. Together with the trained model, I wonder whether are open source plans.

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
3

### Summary
This work presents StreamingVLM, a framework capable of understanding continuous visual input in real time. Specifically, the key ideas include: (1) an efficient inference scheme that maintains sink text tokens, a long window of the most recent text tokens, and a short window of the most recent vision tokens; (2) a dedicated training strategy that splits the input into overlapping chunks and trains with full attention to approximate the aforementioned efficient inference scheme; and (3) a dataset providing long-horizon data for fine-tuning and evaluation. Experiments conducted on publicly available datasets, as well as the newly created dataset, demonstrate improved real-time captioning and video understanding capabilities compared to both in-house and open-source models.

### Strengths
1. Solid presentation: I especially appreciate the contribution of the newly created SFT dataset, and I found the demo video to be a convincing demonstration of the practical value of this work.

2. Clear performance improvement over baselines: The improvements in captioning and video understanding are significant, supported by both qualitative and quantitative comparisons in the manuscript.

3. Comprehensive coverage of prior work: The paper provides a thorough literature review and is overall well written.

### Weaknesses
1. Clarification of differences in streaming-aware KV cache: The distinction between the proposed approach and StreamingLLM is not sufficiently clear. Based on my understanding, the main difference lies in using different window sizes and eviction strategies for text and visual tokens. It would be helpful to explicitly explain this difference and discuss whether the proposed StreamingVLM training strategy could be applied to StreamingLLM.
2. Generalizability of hyperparameters: The window sizes for text and visual tokens are clearly important factors (as shown in Table 5). However, I am concerned that these hyperparameters may be highly task-dependent. For example, a 16-second visual token window might work well for basketball videos but may not generalize to other scenarios. Additional discussion on how to tune or generalize these hyperparameters would strengthen the work.
3. Smaller gains in VQA compared to captioning: While the method improves performance on both VQA and captioning tasks, the gains in VQA are relatively smaller. Providing an explanation for this observation would help readers better understand the strengths and limitations of the proposed approach.

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
