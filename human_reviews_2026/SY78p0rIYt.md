# Automated Movie Generation via Multi-Agent CoT Planning

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 6, 4, 2

## Abstract
Existing long-form video generation frameworks lack automated planning and often rely on manual intervention for storyline development, scene composition, cinematography design, and character interaction coordination, leading to high production costs and inefficiencies. To address these challenges, we present MovieAgent, an automated movie generation via multi-agent Chain of Thought (CoT) planning. MovieAgent offers two key advantages: 1) We firstly explore and define the paradigm of automated movie/long-video generation. Given a script and character bank, our MovieAgent can generates multi-scene, multi-shot long-form videos with a coherent narrative, while ensuring character consistency, synchronized subtitles, and stable audio throughout the film. 2) MovieAgent introduces a hierarchical CoT-based reasoning process to automatically structure scenes, camera settings, and cinematography, significantly reducing human effort. By employing multiple LLM agents to simulate the roles of a director, screenwriter, storyboard artist, and location manager, MovieAgent streamlines the production pipeline. Our framework represents a significant step toward fully automated movie production, bridging the gap between AI-driven video generation and high-quality, narrative-consistent filmmaking.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes MovieAgent, a multi-agent system for automated movie/long-form video generation. Unlike prior text-to-video models that generate short clips or require manual scene planning, MovieAgent uses hierarchical CoT reasoning among specialized agents (Director, Scene Planner, and Shot Planner) to automatically decompose a script into structured acts, scenes, and shots. Experimental results on a new benchmark, MoviePrompts, demonstrate improvements in visual-semantic alignment, motion smoothness, and narrative coherence compared to existing frameworks such as StoryDiffusion, DreamFactory, and Magic-Me.

### Strengths
1. The paper clearly formulates the task of automated movie generation, positioning it as a distinct and ambitious extension of existing short-video generation research.
2. The use of multiple reasoning agents to simulate filmmaking roles (director, scene, and shot planners) is conceptually elegant and well-motivated, and the introduction of internal CoT for structured narrative decomposition and cinematic planning provides interpretability and step-by-step transparency.
3. The paper compare their method with a wide range of baselines. Both automatic and human evaluations show clear improvements in narrative coherence, character consistency, and script faithfulness over baselines.
4. The paper is clearly written, includes strong qualitative visualizations, and ablation studies that isolate the contributions of CoT and multi-agent collaboration.

### Weaknesses
1. The proposed MoviePrompts dataset is small (10 scripts) and partly based on known films, which raises concerns about generalization and potential memorization effects.
2. Some metrics (like “Script Faithfulness” or “Narrative Coherence”) rely heavily on subjective human scoring; no automated measure for story structure consistency is presented.
3. Details on computational costs, failure cases, and training/inference efficiency are insufficient for assessing real-world feasibility.

### Questions
same as weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper presents MovieAgent, a multi-agent Chain-of-Thought (CoT) reasoning framework for automated long-form movie generation. The system comprises three key agents, including Director Agent, Scene Plan Agent, and Shot Plan Agent to automate script breakdown, scene planning, and shot design, enhancing efficiency and narrative coherence. The paper also proposes MoviePrompts, a new evaluation dataset, and reports both quantitative metrics and human ratings. Results show that MovieAgent outperforms prior systems in narrative coherence, character consistency, and aesthetic quality, achieving better performance in script-to-movie generation.

### Strengths
1.	The presentation is clear and easy to follow.

2.	The experiments are comprehensive across automatic and human evaluations, demonstrating improved narrative and visual consistency.

3.	The figures (e.g., Figure 1–3) effectively illustrate how CoT-based reasoning translates into automated scene and shot planning.

### Weaknesses
1.	The method primarily integrates existing models (e.g., ROICtrl, CogVideoX, Hallo2) under a planning hierarchy rather than introducing new generative architectures.

2.	The paper lacks discussion on computational efficiency or latency of the multi-agent pipeline — an important factor for large-scale or real-time production.

3.	The evaluation relies on only 10 movie prompts (8 well-known movies, 2 fictional), which may not sufficiently test generalization across genres or unseen scripts.

### Questions
1.	How efficient is the end-to-end MovieAgent pipeline in practice? What is the average processing time for generating a multi-scene movie, and how well does it scale with longer scripts?

2.	How does MovieAgent ensure cross-scene consistency in visual style and temporal continuity when using different generative models in different stages?

3.	How is audio–video synchronization quantitatively evaluated beyond qualitative visuals?

4.	Have the authors considered integrating feedback loops or self-correction between agents (e.g., re-planning if a generated scene fails to match narrative intent)?

5.	Does MovieAgent require training or fine-tuning for new characters? If so, what is the estimated time and data cost, and how would limited user-provided data affect generation quality?

6.	What is the maximum number of characters that MovieAgent can handle simultaneously in one scene? How does performance or visual quality degrade as the number of characters increases?

7.	Can MovieAgent generate realistic human characters, or is it currently limited to stylized or animated figures? If restricted to animated styles, what are the main technical challenges in extending it to photorealistic human generation?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes MovieAgent, an automated movie generation via multi-agent Chain of Thought (CoT) planning. It explores the paradigm of automated movie/long-video generation, which can generate multi-scene, multi-shot videos with a coherent narrative. MovieAgent also introduces a hierarchical CoT-based reasoning process to automatically structure scenes, camera settings, and cinematography by multiple LLM agents to simulate roles in film production. Experiments show that it achieves SOTA performances in script faithfulness, character consistency, and narrative coherence.

### Strengths
1. The paper is well-written and easy to follow.
2. Experiments show that the proposed multi-agent method achieves SOTA performances compared to baseline.

### Weaknesses
1. The proposed multi-agent framework for automated film generation appears to lack sufficient novelty. Many existing studies have explored multi-agent approaches in video or film generation (e.g., [A, B, C]), and the use of Chain-of-Thought (CoT) reasoning is already a common technique within NLP tasks.

2. The user study is limited in scale, involving only two expert evaluators and merely ten generated films. Such a small sample size restricts the reliability and generalizability of the evaluation results and may not adequately reflect the true quality of the generated films.

3. The use of VBench as an automatic evaluation metric seems misaligned with the specific characteristics of film-level generation. A more suitable benchmark, such as ViStoryBench [D], may provide a more accurate and comprehensive assessment of narrative coherence and overall film quality.

[A] Wang Q, Huang Z, Jia R, et al. MAViS: A Multi-Agent Framework for Long-Sequence Video Storytelling[J]. arXiv preprint arXiv:2508.08487, 2025.

[B] Xu Z, Wang J, Wang L, et al. Filmagent: Automating virtual film production through a multi-agent collaborative framework[M]//SIGGRAPH Asia 2024 Technical Communications. 2024: 1-4.

[C] Yuan Z, Liu Y, Cao Y, et al. Mora: Enabling generalist video generation via a multi-agent framework[J]. arXiv preprint arXiv:2403.13248, 2024.

[D] Zhuang C, Huang A, Cheng W, et al. Vistorybench: Comprehensive benchmark suite for story visualization[J]. arXiv preprint arXiv:2505.24862, 2025.

### Questions
1. What are the differences/advantages of the proposed multi-agent method, compared with the existing ones (see "weakness")?
2. What are the total costs (time/money) for generating one film?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes **MovieAgent**, a multi-agent “Director / Scene / Shot” pipeline that performs hierarchical **Chain-of-Thought (CoT)** planning to decompose a given *script synopsis + character bank* into acts, scenes, and shots. Each shot is then turned into keyframes/storyboards and subsequently into generated video. The stated goal is *automated movie / long-form video generation* with consistent characters, synchronized subtitles, and audio.

The claimed contributions are:

(1) a “first” formalization of the automated movie / long-video generation paradigm,

(2) a hierarchical CoT-based multi-agent framework for movie structure planning and execution, 

(3) state-of-the-art performance in script faithfulness, character consistency, and narrative coherence.

Evaluation is conducted on **MoviePrompts** (10 prompts), using **VBench**-style metrics and a small-scale human study with two raters. Ablations examine (i) internal CoT reasoning, (ii) different LLMs, and (iii) multi-agent vs. single-step planning.

### Strengths
1. **Clear formulation of the task and pipeline.** The system architecture mirrors real production workflow (script breakdown → scene planning → shot list → storyboard → footage). This makes the approach intuitive and easy to follow.
2. **End-to-end orchestration.** The framework attempts to cover the full stack: high-level textual planning, storyboard/keyframe generation, per-shot video synthesis, subtitle generation, and audio output.
3. **Some self-awareness about evaluation limits.** The authors acknowledge that standard image/video metrics like FID/FVD are not directly applicable without ground truth, and they note differences between automated scoring and human preference.

### Weaknesses
1. **Continuity and coherence across shots are not convincingly addressed.**
    
    The method decomposes a movie into many short, independently generated shots. However, the paper does not provide a clear mechanism to guarantee *visual continuity* across these shots — e.g., consistent character appearance, environment style, motion direction, lighting, or camera grammar. Section 3.2.3 claims that the *Shot Plan Agent* is designed to enforce visual continuity, but the description is vague, and Equation (4) plus the proposed heuristics do not actually demonstrate how separate clips are merged into a coherent multi-shot sequence without stylistic breaks.
    
    Similarly, while the system claims to produce accompanying audio and subtitles, the audio is generated separately from the video content. The paper does not explain how lip motion, speech timing, or emotional delivery are synchronized with the generated footage. Without actual AV alignment, the system is effectively still a shot-level video generator plus an unrelated audio track, rather than an integrated long-form audiovisual generator.
    
    These two issues cut to the core claim of “automated long-form movie generation.” As presented, the method behaves more like a collection of disconnected short clips. The evaluation reflects this: despite the stated goal of *multi-scene long-form narrative generation with character and plot consistency*, most reported metrics are **per-shot VBench** scores. These metrics capture short-clip qualities such as subject/background consistency, motion smoothness, and aesthetics, but not *movie-level* structure (story causality, character arcs, act/beat transitions, etc.). The ablations (CoT on/off, number of agents, LLM choice) are also analyzed only at the shot level on a very small test set. There is no quantitative measure of cross-scene identity persistence, narrative causality, or structural coherence across acts. In practice, the proposed method only slightly improves “narrative coherence,” and even that claim rests on a subjective two-rater study rather than robust movie-scale metrics.
    
2. **Evaluation design is narrow and potentially biased.**
    
    The benchmark (**MoviePrompts**) consists of just 10 prompts, 8 of which are adapted from very well-known IP (e.g., *Ne Zha 2, Frozen II, Inside Out 2*). These prompts are authored or curated by two annotators, and the human evaluation is performed by only two expert raters. This is (i) too small for statistical reliability, (ii) not blinded, and (iii) highly vulnerable to data contamination, because modern LLMs and video generators have been widely exposed to these franchises during training. In other words, the system may be doing style imitation or recall rather than true generalization.
    
3. **Metrics do not match the paper’s central claim.**
    
    The work claims advances in long-range narrative coherence, character consistency across scenes, and audio–subtitle synchronization. But the reported gains are mostly in per-shot VBench categories (e.g., Subject/Bg Consistency, Motion Smoothness, Dynamic Degree, Aesthetic Quality). These do not measure:
    
    - whether the same character looks and behaves the same in Scene 1 vs. Scene 7,
    - whether dialogue and events obey causal structure over time,
    - whether subtitle timing matches speech,
    - or whether conversations across scenes remain semantically consistent.
    
    In short, the metrics validate “this looks like a decent short clip,” not “this is a coherent movie.”
    
4. **Unsubstantiated cost/efficiency claims.**
    
    The paper frames the system as providing “near-zero cost” automated filmmaking.  Long-form video generation with multiple diffusion/transformer calls per shot, plus iterative LLM planning, is computationally and financially expensive. The paper does not report GPU hours, token usage, inference latency, or dollar cost per minute of final output. Without that, the cost claim may not stand.
    
5. **Audio evaluation is underspecified.**
    
    The method claims synchronized subtitles and stable audio generation. However, there is no quantitative evidence of audiovisual synchronization.  As a result, the audio claims are qualitative and not verifiable.

### Questions
1. **Data contamination & fairness.**
    
    Since MoviePrompts heavily relies on famous franchises (*Frozen II*, *Inside Out*, *Ne Zha*, etc.), how do you control for prior exposure of the LLMs and video generators to these IPs? Can you show results on held-out, original scripts with entirely novel characters to rule out simple style imitation or memorization?
    
2. **Movie-level narrative coherence.**
    
    Do you compute any structured, sequence-level metrics (e.g., story graph consistency, beat/act structure adherence, causal entailment across scenes)? If not, how do you justify claims of “coherent multi-scene narrative” beyond two human raters’ impressions?
    
3. **Audio / subtitle synchronization.**
    
    How is audiovisual sync actually enforced? Is there any quantitative AV-sync metric demonstrating that the generated speech matches the generated visuals?
    
4. **Cross-scene character identity and continuity.**
    
    What concrete mechanism ensures that the same character maintains consistent visual identity, style, and personality traits across different scenes and shots? 
    
5. **Cost and latency.**
    
    What is the actual cost per minute of finalized output, including failed trials and regeneration passes? Please report GPU hours, dollar estimate, and LLM token usage per finished minute. 
    
6. **Ablation granularity.**
    
    The paper reports a small numerical gain when enabling Chain-of-Thought reasoning (e.g., avg. score improves from 3.55 to 3.61). Which part of the CoT is responsible? Role decomposition (Director vs. Scene vs. Shot agent)? A more fine-grained ablation would make the contribution clearer.

### Soundness
1

### Presentation
3

### Contribution
2
