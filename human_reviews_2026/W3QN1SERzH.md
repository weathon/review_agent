# Temporally-Grounded Language Generation: A Benchmark for Real-Time Vision-Language Models

- Avg Score: 3.33
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 2

## Abstract
Vision-language models (VLMs) have shown remarkable progress in offline tasks such as image captioning and video question answering. However, real-time interactive environments impose new demands on VLMs, requiring them to generate utterances that are not only semantically accurate but also precisely timed. We identify two core capabilities necessary for such settings---$\textit{perceptual updating}$ and $\textit{contingency awareness}$---and propose a new benchmark task, $\textbf{Temporally-Grounded Language Generation (TGLG)}$, to evaluate them. TGLG requires models to generate utterances in response to streaming video such that both content and timing align with dynamic visual input. To support this benchmark, we curate evaluation datasets from sports broadcasting and egocentric human interaction domains, and introduce a new metric, $\textbf{TRACE}$, to evaluate TGLG by jointly measuring semantic similarity and temporal alignment. Finally, we present $\textbf{Vision-Language Model with Time-Synchronized Interleaving (VLM-TSI)}$, a model that interleaves visual and linguistic tokens in a time-synchronized manner, enabling real-time language generation without relying on turn-based assumptions. Experimental results show that VLM-TSI significantly outperforms a strong baseline, yet overall performance remains modest---highlighting the difficulty of TGLG and motivating further research in real-time VLMs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces Temporally-Grounded Language Generation (TGLG), a benchmark designed to evaluate the performance of real-time video language models (VLMs) that must produce time-synchronized responses while processing streaming video input. The benchmark builds on two existing datasets, SoccerNet (Cioppa et al., 2022), featuring sports broadcast videos, and HoloAssist (Wang et al., 2023), comprising egocentric human–object interaction videos. TGLG is organized around two complementary tasks: perceptual updating, which measures how proactively a model responds to evolving visual input, and contingency awareness, which assesses how effectively it provides timely, context-appropriate feedback. The authors also propose TRACE, a metric that jointly evaluates semantic similarity and temporal alignment between predicted and ground-truth utterances. The benchmark primarily evaluates VideoLLM-Online (Chen et al., CVPR 2024) and a modified variant that omits the end-of-sentence (EOS) token to avoid silence gaps. Experimental results show modest but consistent improvements with this modification.

### Strengths
- The paper targets an important and timely problem of evaluating real-time multimodal reasoning in VLMs. As such, it contributes to an emerging and rapidly evolving research direction.
- The use of two complementary datasets with distinct characteristics (third-person sports vs. first-person human interaction) is a thoughtful design choice that enhances the benchmark’s generality. The inclusion of cross-dataset evaluation (HoloAssist is used just for testing) further strengthens its robustness.
- The benchmark provides fine-grained task/action groupings, allowing analysis of model behavior under varying temporal and semantic demands. This granularity can help identify where current systems struggle with temporal grounding.

### Weaknesses
- As a benchmark paper, the experimental evaluation is rather limited. The authors only assess two versions of VideoLLM-Online, omitting several strong recent baselines such as Stream-VLM (Panchal et al., 2024), FlashVStream (Zhang et al., 2024), Dispider (Qian et al., 2025), StreamChat (Xiong et al., 2025), and StreamChat (Liu et al., 2025). Including or at least discussing results from these models would significantly strengthen the empirical validation.
- The related work section does not sufficiently situate TGLG within the landscape of contemporary benchmarks such as QEVD (Panchal et al., 2024), OmniMMI (Wang et al., CVPR 2025), and OVO-Bench (Niu et al., 2025). A detailed comparison would clarify the unique contribution and scope of TGLG.
- The proposed model modification (ignoring the EOS token) is minimal and should not be framed as a novel modeling contribution.
- The TRACE metric introduces several empirically tuned hyperparameters, which raises concerns about reproducibility and interpretability. Furthermore, it is unclear whether a joint score is preferable to reporting separate measures of semantic and temporal alignment (see related question below).
- The presentation could be improved, especially in sections explaining motivation and task setup (e.g., lines 139–147 and 256–269, and Figure 1). Given the multimodal nature of the work, clearer visual-textual illustrations would aid reader understanding. Moreover, the related work section omit a detailed discussion of most related models and benchmarks.

Minor Issues:
- Watch, Talk and Guide (Bao et al., 2023) appeared in Findings of EMNLP 2023.
- StreamBench (Xiong et al., 2025) appeared in ICLR 2025.


**Missing References**
- Liu et al. StreamChat: Chatting with Streaming Video. ArXiv Preprint arXiv:2412.08646v, March 2025.
- Niu et al. OVO-Bench: How Far is Your Video-LLMs from Real-World Online Video Understanding? CVPR 2025
- Panchal et al. What to Say and When to Say it: Live Fitness Coaching as a Testbed for Situated Interaction. NeurIPS 2024 Track on Datasets and Benchmarks. 
- Qian et al. Dispider: Enabling video LLMs with active real-time interaction via disentangled perception, decision, and reaction. CVPR 2025.
- Wang et al. OmniMMI: A Comprehensive Multi-modal Interaction Benchmark in Streaming Video Contexts. CVPR 2025.

### Questions
1. The benchmark currently evaluates two architecturally identical VLMs. How might differences in inference latency or token generation rate across models affect real-time responsiveness and TRACE scores?
2. The TRACE metric combines semantic and temporal components via a weighted sum. Why not report these two aspects independently, as done in (Panchal et al., 2024)?
3. To strengthen the empirical evaluation, please consider including additional recent real-time VLMs or at least discussing their performance relative to TGLG’s objectives. Similarly, a comparative discussion with existing streaming benchmarks (see Weaknesses) would help clarify the distinct contributions of this work.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a new task and benchmark for Temporally-Grounded Language Generation (TGLG), which aims to incorporate two core capabilities of perceptual updating and contingency awareness into real-time interactive video understanding settings. Specifically, this work builds two subsets based on real-time soccer game commentary and egocentric human interaction videos, and a new temporally synchronized token interleaving strategy is proposed to tackle the new challenges. Quantitative experiments are conducted to show the effectiveness of the proposed method and the benchmark.

### Strengths
1.The motivation of incorporating the capabilities of perceptual updating and contingency awareness into real-time video-llms is practical and intuitively reasonable.

2.The proposed benchmark, metric and method in this work are shown to be effective under real-time settings.

### Weaknesses
1.As the authors mentioned in the manuscript, the existing turn-based video-llms would give response to the environment with overly high latency, which is a major obstacle for them to handle the real-time settings. However, these video-llms are generally at a large size and have huge amout of parameters which naturally make them unsuitable for real-time response. What if the turn-based video-llms are optimized to have fewer parameters and faster response speed, for example, turn-based models could also generate response promptly if they can finish decoding before new frames come in. Are there any discussions or experiments to analyze this aspect?

2.For the proposed temporally synchronized vision and text token interleaving strategy, it is shown to be effective to generate real-time response conditioned on fastly evolving visual environments. But based on my understanding, such highly fragmented token mixing method would inevitably destroy the coherence of the visual input and also the textual context, especially when the full off-line inputs are available. So are there any experiments or analysis on the performance of such strategy under a off-line setting? Will it largely decrease the performance of the model under off-line settings?

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces a new task for testing vision-language models in online real-time settings, which is called as Temporally-Grounded Language Generation (TGLG). The proposed benchmark includes sports broadcasting and egocentric videos. Within this work, TRACE, a novel evaluation metric, is also introduced to evaluate models on the TGLG benchmark. Furthermore, the authors introduce VLM-TSI, which enables processing interleaved vision and language tokens in a time-synchronized way, later tested on the repurposed benchmark using the proposed metric TRACE. The downstream task experiments reveal that the further finetuning the baseline using VLM-TSI approach on SoccerNet and HoloAssist datasets improve real-time spatio-temporal processing based on the proposed metric.

### Strengths
- Research on an interesting direction even within the domain of spatio-temporal vision-language learning: time-synchronized processing in real-time scenarios.
- Substantial improvements over the baseline.

### Weaknesses
- No human validation study on how the proposed metric aligns with human preferences.
- I'm confused about this work's purpose. It seems that the data resources already exist, and VLM-TSI method actually proposes finetuning a suitable model on these resources. According to my current understanding, the task is actually not novel, the methodology is actually to finetune a proper model on time-synchronized interleaved video-language data. The only actual novelty is the proposed metric, which is not evaluated.
- Dataset-specific finetuning: It appears that this work performs two separate finetuning on two separate downstream datasets. It would be good to expand the experiments considering both data resources simultaneously.
- Limited evaluation: There is only one baseline in the current evaluation setup. I am aware the fact that the other models would not be good baselines as suggested by authors. However, expanding evaluations to more general zero-shot video-language benchmarks (e.g., Video-MME) could be beneficial. For instance, did the model become better in spatio-temporal processing after learning more real-time dynamics?

### Questions
- L428-L431: Could this be more related to models finding shortcuts through patterns? Penalty position could be detected by using single frame, no spatio-temporal processing is required actually.
- Fig. 1, and Fig. 4 could be combined to explain the task in a better way. Same goes for Fig 2. and Fig 3., they do not need to be separate.

### Soundness
2

### Presentation
2

### Contribution
3
