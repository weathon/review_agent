# LongLive: Real-time Interactive Long Video Generation

- Decision: Accept (Poster)
- Scores: 8, 6, 6

## Abstract
We present LongLive, a frame-level autoregressive (AR) framework for real-time and interactive long video generation. Long video generation presents challenges in both efficiency and quality. Diffusion and Diffusion-Forcing models can produce high-quality videos but suffer from low efficiency due to bidirectional attention. Causal attention AR models support KV caching for faster inference but often degrade in quality on long videos due to memory challenges during long-video training. In addition, beyond static prompt-based generation, interactive capabilities, such as streaming prompt inputs, are critical for dynamic content creation, enabling users to guide narratives in real time. This interactive requirement significantly increases the complexity, especially in ensuring visual consistency and semantic coherence during prompt transitions. To address these challenges, LongLive adopts a causal, frame-level AR design that integrates a KV-recache mechanism that refreshes cached states with the new prompt for smooth, adherent switches streaming long tuning to enable long video training and to align training and inference (train-long–test-long); and short window attention paired with a frame-level attention sink, preserving long-range consistency while enabling faster generation. With these key designs, LongLive fine-tunes a 1.3B-parameter short-clip model to minute-long generation in just 32 GPU-days. At inference, LongLive sustains 20.7 FPS on a single NVIDIA H100, achieves strong performance on VBench in both short- and long-video settings. LongLive supports up to 240-second videos on a single H100 GPU. With FP8 quantization, LongLive boosts inference to 24.8 FPS with marginal quality loss.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper investigates real-time, interactive minute-level video generation and frames it as a causal, frame-level autoregressive task. It introduces three synergistic ideas—KV-recache for smooth prompt switching, streaming long tuning to close the train-short/test-long gap, and short-window attention plus frame-sink tokens for accelerating inference. Extensive experiments on VBench and a new 160 interactive suite show that a 1.3 B model fine-tuned in only 32 GPU-days reaches 20.7 FPS on one H100 and outperforms diffusion and AR baselines in both short and  long clip settings. The manuscript is clearly structured and well illustrated.

### Strengths
- This paper tackles an important practical problem—interactive long video generation—and provides the first system that delivers >20 FPS at high resolution while keeping visual consistency. 

- The authors propose KV-recache, a simple yet effective cache refresh that unsticks old-prompt semantics without breaking temporal continuity, nicely verified by ablation.

- Experiments are thorough: short vs. long, single vs. multi-prompt, throughput vs. quality, and ablations.

- The writing is accessible, with intuitive figures that contrast the proposed strategies and a reproducibility statement promising full code and weights.

### Weaknesses
Overall, the paper offers solid technical novelty and the core ideas are interesting; nevertheless, I have a few minor concerns and questions listed below.

- Why does SkyReels-V2 show a sudden performance jump in the 20–30 s interval? This spike suggests that the evaluation may be unstable. If so, I recommend re-running the evaluation with several random seeds (e.g., three) and reporting the mean and standard deviation to ensure reliability.

- The authors fine-tune on top of Wan2.1, yet the resulting model’s semantic-alignment score drops markedly, as shown in Table 1; no explanation for this degradation is provided. I encourage the authors to investigate and clarify the cause—either through analysis or additional ablation experiments—so that the community can better understand the trade-offs and further improve the model.

### Questions
See weakness

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents LongLive, a method for fine-tuning pretrained video diffusion models for long video generation. First, it introduces KV-Recache, which regenerates the KV cache using the previously generated frames together with the new prompt when a prompt change occurs. This allows the model to generate videos that remain visually coherent with past frames while adapting better to the new prompt, rather than being stuck with the previous inputs. Moreover, the authors progressively extend the generation length by fine-tuning DMDs with an increasing number of video clips, where the gradient is computed only over the last video segment. The paper also proposes several additional techniques, such as applying an attention sink by maintaining the first frame as a global anchor. Leveraging all these techniques, LongLive demonstrates real-time long video generation (up to 240 seconds) on a single H100 GPU with minimal quality degradation.

### Strengths
- The paper is generally well-written and easy to follow. 
- Real-time long video generation using pretrained diffusion models is a very challenging but important problem, and I think this paper provides quite effective techniques to solve this.
- For me, several techniques are simple (which is good) yet novel and effective, such as KV recache.
- I really like the supplementary material that the authors attached.

### Weaknesses
- There are several missing works in the literature on long video generation, such as AAPT [1], TECO [2], MALT [3], and Rolling Diffusion [4]. I recommend that the authors include a discussion of these and other relevant works that have not been covered in the paper.
- I also believe that having a long context is another important aspect in developing long video generation models. While being efficient, the proposed model inherently has a relatively short context length (mainly due to the attention sink that focuses on the first frame), which limits its effective context window. The method is indeed effective at generating visually plausible long videos in real time; however, it might struggle in scenarios that require longer temporal dependencies — for example, when a person disappears and reappears in the same scene, or when generating long videos with multiple scene transitions.

[1] Lin et al., Autoregressive Adversarial Post-Training for Real-Time Interactive Video Generation, 2025.  
[2] Yan et al., Temporally Consistent Transformers for Video Generation, ICML 2023.  
[3] Yu et al., MALT Diffusion: Memory-Augmented Latent Transformers for Any-Length Video Generation, 2025.  
[4] Ruhe et al., Rolling Diffusion Models, ICML 2024.

### Questions
- What is the empirical maximum video length that the proposed model can generate without suffering from severe quality degradation? For instance, can a model generate videos even longer than 240s?
-Why does the author choose DMD instead of other techniques, e.g., DMD2?  Is there a specific reason?

### Soundness
3

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
2

### Summary
This paper presents a frame-level AR framework for real-time interactive long video generation. The paper mainly focuses on efficiency design. The authors introduce a KV-recache mechanism that refreshes cached states to ensure smooth, adherent transitions when prompts change. They also propose a tuning strategy to address quality degradation in long videos, aligning training with inference. For efficiency, the model combines short window attention with a frame sink to maintain long-range consistency while accelerating generation. The resulting 1.3B parameter model achieves 20.7 FPS on an H100 GPU, supports videos up to 240 seconds, and demonstrates strong performance on VBench for both short and long video benchmarks.

### Strengths
1. A new KV-recache strategy is proposed for frame-level AR with multiple prompts.
2. Fairly comprehensive evluation, demonstrating the performance on different lengths of video.
3. Some efficient strategies for training and inference are proposed for long videos.

### Weaknesses
1. Generalization ability. Though the authors show some long video cases, the prompt still remains in the same scene. How does it handle multiple, rapid-fire prompts? More importantly, how does it respond to prompts that introduce abrupt semantic shifts (e.g., changing from "Iron Man fighting" to "a peaceful beach") versus the logical continuations shown in the examples?
2. The "train-long-test-long" strategy is a key strength, but its supervision (DMD loss) relies on a teacher model that itself was only trained on short clips. While the student model is exposed to long rollouts, the supervision for each new clip is still provided by this short-clip teacher. This may create a ceiling for true long-range coherence, as the teacher lacks the long-horizon context the student is trying to learn.
3. The ablation on time consuming about the KV cache, no KV Cache, and KV recache for different lengths of video is suggested to add to prove the effiency.
4. The ablation about frame sink strategy (first frame or others) is encouraged to add.

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3
