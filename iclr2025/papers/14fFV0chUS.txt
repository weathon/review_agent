

{0}------------------------------------------------

# TRACE: TEMPORAL GROUNDING VIDEO LLM VIA CAUSAL EVENT MODELING

Yongxin Guo<sup>1</sup> Jingyu Liu<sup>2</sup> Mingda Li<sup>2</sup> Qingbin Liu<sup>2</sup> Xi Chen<sup>2,\*</sup> Xiaoying Tang<sup>1,3,4,\*</sup>

<sup>1</sup>School of Science and Engineering, The Chinese University of Hong Kong, Shenzhen 518172, China

<sup>2</sup>Tencent PCG

<sup>3</sup>Shenzhen Institute of Artificial Intelligence and Robotics for Society (AIRS), Shenzhen, China

<sup>4</sup>Guangdong Provincial Key Laboratory of Future Networks of Intelligence, Shenzhen, China

## ABSTRACT

Video Temporal Grounding (VTG) is a crucial capability for video understanding models and plays a vital role in downstream tasks such as video browsing and editing. To effectively handle various tasks simultaneously and enable zero-shot prediction, there is a growing trend in employing video LLMs for VTG tasks. However, current video LLM-based methods rely exclusively on natural language generation, lacking the ability to model the clear structure inherent in videos, which restricts their effectiveness in tackling VTG tasks. To address this issue, this paper first formally introduces causal event modeling framework, which represents video LLM outputs as sequences of events, and predict the current event using previous events, video inputs, and textual instructions. Each event consists of three components: timestamps, salient scores, and textual captions. We then propose a novel task-interleaved video LLM called TRACE to effectively implement the causal event modeling framework in practice. The TRACE process visual frames, timestamps, salient scores, and text as distinct tasks, employing various encoders and decoding heads for each. Task tokens are arranged in an interleaved sequence according to the causal event modeling framework’s formulation. Extensive experiments on various VTG tasks and datasets demonstrate the superior performance of TRACE compared to state-of-the-art video LLMs. Our model and code are available at <https://github.com/gyxxyg/TRACE>.

## 1 INTRODUCTION

Video Temporal Grounding (VTG) is an important ability for video understanding models (Lin et al., 2023b), and has becoming the base of a series of downstream tasks like moment retrieval (Fabian Caba Heilbron & Niebles, 2015; Gao et al., 2017; Oncescu et al., 2021), dense video caption (Zhou et al., 2018; Tang et al., 2019), video highlight detection (Lei et al.; Liu et al., 2022), and video summarization (Song et al., 2015; Gygli et al., 2014). While non-generative models excel in moment retrieval and video highlight detection (Lei et al., 2021; Han et al., 2024; Wang et al., 2024a), they are inflexible, task-specific, and demand substantial fine-tuning for optimal performance. To tackle these challenges, recent research employs video LLMs as versatile models, integrating timestamp information into visual inputs, and fine-tuning them on VTG tasks (Ren et al., 2023; Huang et al., 2023; Wang et al., 2024b; Qian et al., 2024; Wang et al., 2024c; Wu et al., 2024) to enhance their performance and facilitate zero-shot prediction.

**Challenges posed by videos’ inherent structures.** Despite reflecting human intent, current video LLM based approaches rely on pure natural language generation. As illustrated in Figure 1(a), this approach lacks a clear structure and indiscriminately blends information, such as timestamps and text captions. In contrast, videos possess an inherent structure that transcends mere textual description. To accurately describe or reason from a video, it is insufficient to rely solely on natural language text. Instead, the corresponding timestamps and salient scores are also essential components. Together, these elements provide a more comprehensive and structured understanding of the video content. Consequently, the gap between videos’ structure and current video LLMs undermines the

\*Corresponding authors.

{1}------------------------------------------------

![Figure 1: Challenges posed by videos' inherent structures. (a) Video Structure: A video thumbnail showing potatoes being prepared, with a caption: '11.0 - 13.0 seconds, introducing and summarizing the dehydration process. 13.0 - 43.0 seconds, peeling potatoes using an apple peeler. 43.0 - 78.0 seconds, cutting the potatoes into 1-inch cubes and soaking them in water with fruit fresh added.' Below this is a table with 4 columns: Event, Start Time (s), End Time (s), and Caption. (b) Performance Gap between Models: A bar chart comparing TimeChat, VTG-LLM, and TRACE across three tasks: Moment Retrieval, Dense Video Caption, and Video Highlight Detection.](49ad3a646d84bcfeac02bdf2b3792a3e_img.jpg)

Figure 1(a) shows a video thumbnail of potatoes being prepared. The caption for the video is: "11.0 - 13.0 seconds, introducing and summarizing the dehydration process. 13.0 - 43.0 seconds, peeling potatoes using an apple peeler. 43.0 - 78.0 seconds, cutting the potatoes into 1-inch cubes and soaking them in water with fruit fresh added." Below the caption is a table representing the video structure:

| Event | Start Time (s) | End Time (s) | Caption                                                                                  |
|-------|----------------|--------------|------------------------------------------------------------------------------------------|
| 1     | 0.0            | 13.0         | introducing and summarizing the dehydration process.                                     |
| 2     | 13.0           | 43.0         | peeling potatoes using an apple peeler.                                                  |
| 3     | 43.0           | 78.0         | cutting the potatoes into 1-inch cubes and soaking them in water with Fruit Fresh added. |

Figure 1(b) is a bar chart titled "Performance on VTG Tasks." comparing three models: TimeChat, VTG-LLM, and TRACE across three tasks: Moment Retrieval, Dense Video Caption, and Video Highlight Detection. The Y-axis represents Performance (0 to 40). TRACE consistently outperforms the other two models across all tasks.

Figure 1: Challenges posed by videos' inherent structures. (a) Video Structure: A video thumbnail showing potatoes being prepared, with a caption: '11.0 - 13.0 seconds, introducing and summarizing the dehydration process. 13.0 - 43.0 seconds, peeling potatoes using an apple peeler. 43.0 - 78.0 seconds, cutting the potatoes into 1-inch cubes and soaking them in water with fruit fresh added.' Below this is a table with 4 columns: Event, Start Time (s), End Time (s), and Caption. (b) Performance Gap between Models: A bar chart comparing TimeChat, VTG-LLM, and TRACE across three tasks: Moment Retrieval, Dense Video Caption, and Video Highlight Detection.

(a) Video Structure

(b) Performance Gap between Models

Figure 1: **Challenges posed by videos' inherent structures.** Figure 1(a) shows the difference between natural language and video structure, while Figure 1(b) highlights the performance gap between SOTA video LLMs (Ren et al., 2023; Guo et al., 2024) and TRACE. We present zero-shot performance results for video LLM approaches. Specifically, we report the performance of models using the CIDEr metric for the dense video captioning task on the Youcook2 dataset,  $R@1_{IOU=0.7}$  for the moment retrieval task on the Charades-STA dataset, and HIT@1 for the highlight detection task on the QVHighlights dataset.

ability of video LLMs to effectively model video events, potentially making video LLMs difficult to achieve satisfactory results (Figure 1(b)) on VTG tasks.

**Causal event modeling as a solution.** In this paper, our primary goal is to develop a novel video LLM approach for resolving the mismatch between language modeling of LLMs and videos' inherent structure. Specifically, we concentrate on tackling two main challenges: (1) developing a theoretical framework that shifts from causal language modeling to structured event-based modeling, and (2) constructing a practical video LLM based on the theoretical framework to provide an effective solution. To accomplish this, we first introduce the causal event modeling framework, where video LLM outputs are represented as sequences of events, each containing timestamps, salient scores, and textual captions. The next events are predicted based on video inputs, text instructions, and preceding events. To effectively implement the causal event modeling framework in practice, we present a novel task-interleaved video LLM, TempoRAI grounding via Causal Event modeling (TRACE), as illustrated in Figure 2. The TRACE treats visual frames, timestamps, salient scores, and text as separate tasks, utilizing diverse encoders and decoding heads for each task, with task tokens sequenced in an interleaved manner. Furthermore, we develop an adaptive head-switching method for improved generation. Our numerical results across various VTG tasks reveal the superior performance of TRACE in comparison to state-of-the-art (SOTA) video LLMs.

### Our key contributions are summarized as follows:

- We model the videos by a series of events, and propose causal event modeling framework to capture videos' inherent structure. We then present a novel task-interleaved video LLM model, TRACE, tailored to implement the causal event modeling framework through the sequential encoding/decoding of timestamps, salient scores, and textual captions.
- We conduct comprehensive experiments on multiple VTG tasks and datasets to verify the effectiveness of TRACE. The results reveal significant improvements of TRACE in comparison to SOTA video LLMs. Notably, TRACE improves zero-shot performance by 3.1 and 4.9% on Youcook2 (CIDEr and F1 Score), by 6.5% and 3.7% in Recall ( $IOU = \{0.5, 0.7\}$ ) on Charades-STA, and by 10.3% and 9.2% for mAP and HIT@1 on QVHighlights. Moreover, surpassing existing video LLMs, TRACE achieves comparable performance to traditional non-generative and task-specific methods after fine-tuning, highlighting the potential of video LLMs to excel in VTG tasks.

## 2 RELATED WORKS

**Video temporal grounding.** Video Temporal Grounding (VTG) tasks aim to precisely identify the timestamps of events within a given video (Lin et al., 2023b). This includes various tasks such as

{2}------------------------------------------------

![Figure 2: Overview of the training process of TRACE model. The diagram illustrates the flow from video input and text inputs through various encoders into a Large Language Model (LLM), which then outputs time, score, and text via specific heads.](1b7d539e02a202c2cf2d97698b911447_img.jpg)

The diagram shows the architecture of the TRACE model. At the bottom left, a video frame of a person cooking is shown with timestamps 5.0s, 10.0s, and 15.0s. This video is processed by a **Vision Encoder** to produce **Compression** tokens. Simultaneously, **Text Inputs** (a prompt about locating events in a video) are processed by a **Text Tokenizer** to produce text tokens. The video frames are also processed by a **Time Encoder** using **Timestamp** inputs (0.0s, 10.0s) and a **Score Encoder** using **Score** inputs (4, 5). All these tokens (Compression, Time, Score, and Text) are fed into a **Large Language Model**. The LLM's output is then processed by three heads: **Time Head**, **Score Head**, and **Text Head**, which generate the final output tokens.

Figure 2: Overview of the training process of TRACE model. The diagram illustrates the flow from video input and text inputs through various encoders into a Large Language Model (LLM), which then outputs time, score, and text via specific heads.

Figure 2: Overview of the training process of TRACE model. We employ a variety of encoders and heads to handle time, score, and text inputs and outputs. The timestamps of the sampled frames are converted into time tokens and subsequently integrated into the visual tokens of each frame. In the answer section, time tokens, score tokens, and text tokens are inserted in a sequential manner. The generation process of TRACE is summarized in Figure 4.

moment retrieval (Gao et al., 2017; Zala et al., 2023b; Oncescu et al., 2021; Hendricks et al., 2018a; Boris et al., 2024), dense video caption (Zellers et al., 2021; Zala et al., 2023b; Tang et al., 2019; Fabian Caba Heilbron & Niebles, 2015; Kim et al., 2024), video summarization (Song et al., 2015; Gygli et al., 2014; Hua et al., 2024), and video highlight detection (Lei et al., 2021; Xiao et al., 2023). For tasks such as moment retrieval, video summarization, and video highlight detection, traditional approaches primarily use large-scale video-text pre-training (Xu et al., 2021; Wang et al., 2022; Yan et al., 2022; Li et al., 2023d; Chen et al., 2024b; Tong et al., 2022; Zhao et al., 2024). Subsequently, they fine-tune the pretrained models by incorporating task-specific prediction heads. While these methods have demonstrated satisfactory results, they are resource-intensive for pre-training, lack zero-shot capabilities, can only handle one specific task per model, and often require additional fine-tuning for numerous downstream tasks. For the dense video caption task, Vid2Seq employs special time tokens to represent timestamps (Yang et al., 2023). Some approaches integrate additional input information, such as text queries from training datasets (Kim et al., 2024), while other models utilize different decoding heads to decode timestamps and textual captions (Wang et al., 2021; 2023a) in parallel. However, these architectures are specifically designed for the dense video caption task, cannot be easily adapted to LLM structures to fully harness the capacity of pretrained LLMs, and also lack zero-shot capabilities.

**Video LLMs for video temporal grounding.** Large language models (LLMs) (Kaplan et al., 2020; Achiam et al., 2023; Touvron et al., 2023) have demonstrated significant potential in acquiring knowledge and addressing real-world challenges using a zero-shot approach. Recent research has focused on integrating knowledge from other modalities, such as vision (Liu et al., 2024; Li et al., 2023a) and audio (Ghosal et al., 2023), to bolster the capabilities of LLMs. Within the visual domain, video large language models (video LLMs) have emerged as a crucial research area (Lin et al., 2023a; Maaz et al., 2023; Zhu et al., 2023; Song et al., 2024a;b). Traditional video LLMs (Zhang et al., 2023; Lin et al., 2023a; Li et al., 2023b; 2024; Cheng et al., 2024; Yao et al., 2024) have made considerable performance improvements in tasks such as video question answering, reasoning, and video captioning. However, these methods encounter difficulties in precisely pinpointing event timestamps within videos. To address this issue, TimeChat (Ren et al., 2023), VTimeLLM (Huang et al., 2023), and Hawkeye (Wang et al., 2024b) have attempted to overcome this limitation by

{3}------------------------------------------------

fine-tuning the video LLMs on VTG datasets. More recently, LITA (Huang et al., 2024) introduces fast-slow visual tokens and incorporates time tokens into LLM tokenizers. Momentor (Qian et al., 2024) suggests a time encoder to address time token quantization errors. VTG-LLM (Guo et al., 2024) integrates special time tokens and time position embeddings to improve the ability of video LLMs in comprehending timestamps. However, these methods do not take into account the inherent structure of videos and still cannot achieve satisfactory performance. In this paper, we propose the causal event modeling framework to provide structured video LLM outputs and design the TRACE model to address the proposed framework. Numerical results demonstrate significant performance gains of TRACE over existing video LLMs on VTG tasks.

## 3 TRACE

In this section, we aim to develop a novel video LLM that aligns well with video structures, addressing two questions: (1) how to model the structured video LLM outputs that are aligned well with video structures, and (2) how to implement theoretical models. We start by proposing *causal event modeling* framework to tackle "how to model". Then, we introduce TRACE to address "how to implement". We have included a detailed discussion about our framework in Appendix C.

### 3.1 MODELING THE INHERENT STRUCTURES OF VIDEOS

**Formulating outputs of video LLMs by events.** Given the instruction **I** and video visual inputs **F**, we represent the outputs of video LLMs **R** as a series of events  $\{e_1, e_2, \dots, e_K\}$ , with each event  $e_k = (t_k, s_k, c_k)$  encompassing timestamps  $t_k$ , salient scores  $s_k$ , and textual captions  $c_k$ . In summary, we have

$$\mathbf{R} = \{e_1, e_2, \dots, e_K\} = \{(t_k, s_k, c_k) | 1 \leq k \leq K\}. \quad (1)$$

**Causal event modeling framework.** To effectively utilize the knowledge of pretrained LLMs, the design of causal event modeling shares the underlying intuition of causal language modeling, as formulated in the subsequent equation <sup>1</sup>.

$$\begin{aligned} \mathcal{P}(e_k | e_{1:k-1}, \mathbf{I}, \mathbf{F}) &= \mathcal{P}(t_k, s_k, c_k | e_{1:k-1}, \mathbf{I}, \mathbf{F}), \\ &= \mathcal{P}(t_k | e_{1:k-1}, \mathbf{I}, \mathbf{F}) \mathcal{P}(s_k | t_k, e_{1:k-1}, \mathbf{I}, \mathbf{F}) \mathcal{P}(c_k | s_k, t_k, e_{1:k-1}, \mathbf{I}, \mathbf{F}), \end{aligned} \quad (2)$$

The next event  $e_k$  is determined by textual instructions, visual inputs, and previous events. We can find that causal event modeling framework aligns well with the video structure (Figure 1(a)): (1) timestamps, salient scores, and textual captions are sequentially decoded within each event; (2) events are then ordered by timestamps.

### 3.2 TRACE: TASK-INTERLEAVED TEMPORAL GROUNDING VIDEO LLM

In Eq. 2, we introduce a formal causal event modeling framework to tackle the challenge of modeling structured video LLM outputs. This section illustrates the design of TRACE to implement the causal event modeling framework (Figure 2).

**Overview of TRACE.** As illustrated in Eq. 2, the causal event modeling framework necessitates encoding/decoding of visual frames (**F**), text (**I** and  $c_k$ ), timestamps ( $t_k$ ), and scores ( $s_k$ ). Consequently, the TRACE considers these elements as distinct tasks and employs the following design to efficiently manage them.

- *Separated multi-task processing.* TRACE utilizes separate encoders and decoding heads for each task to convert task inputs into task tokens and decode task tokens back to outputs (Sec. 3.2.1).
- *Task-interleaved sequence modeling.* Task tokens are sequenced in an interleaved manner according to Eq. 2 in TRACE and fed into LLM backbones (Sec. 3.2.2).
- *Adaptive head-switching mechanism for generation.* During generation, we implement an adaptive head-switching mechanism to select the appropriate decoding head for producing the next token (Sec. 3.2.3).

<sup>1</sup>Theoretically, the order of time, score, and text will not impact the results. We select one order here.

{4}------------------------------------------------

![Figure 3: Illustration of token sequence of TRACE. The diagram shows a sequence of tokens categorized into visual (orange), time (purple), score (green), and text (dark grey) tokens. The sequence starts with visual frame tokens (F) and instruction tokens (I). Event tokens (e) are structured as time tokens (t), score tokens (s), and text tokens (c), ordered chronologically. The diagram shows the sequence: F, I, e1 (t1, s1, c1), e2. Below the sequence, arrows indicate the causal event modeling formula P(e_k | e_{1:k-1}, I, F) for each event token e_k.](690fce4fb5c9cbb8beb560cb2a3fcbeb_img.jpg)

Figure 3: Illustration of token sequence of TRACE. The diagram shows a sequence of tokens categorized into visual (orange), time (purple), score (green), and text (dark grey) tokens. The sequence starts with visual frame tokens (F) and instruction tokens (I). Event tokens (e) are structured as time tokens (t), score tokens (s), and text tokens (c), ordered chronologically. The diagram shows the sequence: F, I, e1 (t1, s1, c1), e2. Below the sequence, arrows indicate the causal event modeling formula P(e\_k | e\_{1:k-1}, I, F) for each event token e\_k.

Figure 3: **Illustration of token sequence of TRACE.** Following Eq 2, the sequence begins with visual frame tokens (**F**) followed by instruction tokens (**I**). Event (**e**) tokens are structured in the sequence of time tokens (**t**), score tokens **s**, and text tokens **c**, with events ordered chronologically based on their occurrence time.

#### 3.2.1 SEPARATED MULTI-TASK PROCESSING

TRACE consists of four unique tasks: visual frames, text, timestamps, and scores. Regarding text, we directly utilize the text tokenizer and LLM head of the LLM backbone (Mistral-7B-v0.2 (Jiang et al., 2023)). Moreover, we added a special token *<sync>* for indicating the end of text tasks. The processing for the other tasks is detailed below.

**Timestamps and scores processing.** For processing timestamps and score information, we employ two separate encoders and decoding heads, both of which share the same architecture. Specifically, each encoder is initialized with a tokenizer containing 13 tokens: 11 number tokens  $\langle 0 \rangle, \dots, \langle 9 \rangle, \langle . \rangle$  for representing timestamps/scores, *<sep>* to mark the end of each timestamp/score, and *<sync>* to signify the end of the current task. Token embeddings are initialized using LLM token embeddings.

In accordance with the research in VTG-LLM (Guo et al., 2024), we format each timestamp/score to the same length, comprising 4 whole-number parts, 1 dot, and 1 fractional part<sup>2</sup>. Subsequently, *<sep>* is inserted between timestamps/scores, and *<sync>* is appended at the end of each timestamp/score input sequence. For instance, the timestamp inputs [10.23, 125.37] will be tokenized into the following sequence:  $\langle 0 \rangle \langle 0 \rangle \langle 1 \rangle \langle 0 \rangle \langle . \rangle \langle 2 \rangle \langle sep \rangle \langle 0 \rangle \langle 1 \rangle \langle 2 \rangle \langle 5 \rangle \langle . \rangle \langle 4 \rangle \langle sync \rangle$ .

**Visual frames processing.** Given a  $T$ -frame video, we initially encode the frames using the pre-trained CLIP ViT-L (Radford et al., 2021), with each frame being encoded into 576 visual tokens. Subsequently, we employ Slot-Based Compression (Guo et al., 2024) to reduce the number of visual tokens to 8 per frame. Moreover, to integrate temporal information into the visual inputs, we use a time encoder to encode the timestamps of each sampled frame and remove the *<sync>* and *<sep>* tokens, resulting in 6 time tokens for each frame. Finally, we concatenate the 8 visual tokens with the 6 time tokens to form the visual inputs for each frame.

#### 3.2.2 TASK-INTERLEAVED SEQUENCE MODELING

Utilizing the processed task tokens, we construct the sequence following Eq. 2. The token sequence order is illustrated in Figure 3.

**Inter-event sequence order.** The sequence commences with visual frame tokens **F** followed by textual instruction tokens **I**. For the events section, event tokens are sequenced according to the events' occurrence time to align with the causal event modeling formula  $\mathcal{P}(e_k | e_{1:k-1}, \mathbf{I}, \mathbf{F})$ .

**Intra-event sequence order.** For each event, in accordance with Eq. 2, tokens are arranged sequentially by time tokens ( $\mathcal{P}(t_k | e_{1:k-1}, \mathbf{I}, \mathbf{F})$ ), score tokens ( $\mathcal{P}(s_k | t_k, e_{1:k-1}, \mathbf{I}, \mathbf{F})$ ), and text tokens ( $\mathcal{P}(c_k | s_k, t_k, e_{1:k-1}, \mathbf{I}, \mathbf{F})$ ). Consequently, the causal event modeling framework (Eq. 2) emerges as a specialized autoaggressive model, featuring a unique sequence order that closely aligns with video structures.

<sup>2</sup>Different from timestamps, scores will be encoded to 3 score tokens, including 1 whole-number parts, 1 dot, and 1 fractional part.

{5}------------------------------------------------

Table 1: **Datasets used for TRACE training process.** "Compressed" indicates that datasets are condensed by retaining only one sample for samples with identical videos but varying instructions.

| Stage   | Datasets                                                                                                                              | Quantity |
|---------|---------------------------------------------------------------------------------------------------------------------------------------|----------|
| Stage 1 | Valley, LLaVA-Image, TextVR, ShareGPT4Video, VTG-IT                                                                                   | 1.9M     |
| Stage 2 | Valley (Compressed), TextVR (Compressed), ShareGPT4Video (Compressed), VTG-IT, ActivityNet Captions, VideoChatGPT, InternVid, Next-QA | 0.9M     |

#### 3.2.3 ADAPTIVE HEAD-SWITCHING MECHANISM FOR GENERATION

**Using *(sync)* token for adaptive head switching.** Since TRACE employs distinct decoding heads for various tasks during training, selecting the appropriate decoding head during generation based on previously decoded tokens is crucial. This selection is facilitated by the *(sync)* token. As illustrated in Figure 4, TRACE generates tokens in the sequence of time, score, and text tokens. Detection of the *(sync)* token prompts TRACE to switch decoding heads accordingly. The heads are cycled switched in the order of time head - score head - text head.

### 3.3 TRAINING STRATEGY AND DATA PREPARATION

This section outlines the TRACE training process, which includes two stages. For the stage 1, task modules such as the vision compression layer, task encoder, and task heads are trained for initialization. For the stage 2, the LLM backbone is fine-tuned while keeping the task modules tuned. Detailed settings and datasets are presented below. Due to the page limitation, detailed annotation examples for each task, and details about data filtering and processing are provided in Appendix A.

**Stage 1: Initialization of task modules.** In stage 1, task modules such as the vision compression layer, time encoder/head, score encoder/head, and text tokenizer/head are trained while the vision encoder and LLM backbone remain fixed. As shown in Table 1, stage 1 primarily utilizes two groups of datasets.

- *Image and video caption datasets for initializing the visual compression layer.* This group of datasets including Valley (Luo et al., 2023b), LLaVA-Image (Liu et al., 2024), TextVR (Wu et al., 2025), and a randomly sampled subset of ShareGPT4Video (Chen et al., 2024a) datasets.
- *VTG datasets for task encoder/head initialization.* We use VTG-IT dataset in this group.

For stage 1 training, we uniformly sample 128 frames from each video. The learning rate is set to 1e-3, and models are trained for one epoch with a batch size of 128.

**Stage 2: Instruction tuning for enhancing VTG capacity.** In Stage 2, the LLM backbone and task modules are fine-tuned, with only the vision encoder remaining fixed. As shown in Table 1, stage 2 primarily utilizes three groups of datasets.

- *VTG instruction tuning datasets for enhancing VTG capacity.* We use VTG-IT (Guo et al., 2024), ActivityNet Captions (Fabian Caba Heilbron & Niebles, 2015), and a subset of InternVid (Wang et al., 2023b), resulting in a total of 635K data samples. Low-quality samples were filtered out, and the VTG-IT-VHD and VTG-IT-VS datasets were re-annotated. Additional details can be found in Appendix A.
- *Video caption datasets for maintaining the quality of the visual compression layers.* We use parts of the video data from stage 1, such as Valley (Luo et al., 2023b), TextVR (Wu et al., 2025), and ShareGPT4Video (Chen et al., 2024a) datasets. These datasets are compressed by retaining only one sample for samples with identical videos but different instructions, yielding 284K data.
- *Video question answering datasets to enhance TRACE’s reasoning capabilities.* We use VideoChatGPT (Maaz et al., 2023) and Next-QA (Xiao et al., 2021) in this part.

For each video, the content is uniformly divided into 128 clips, with one frame randomly sampled from each clip. The learning rate is set to 5e-6, and the models are trained for two epochs using a batch size of 128.

{6}------------------------------------------------

![Diagram of the TRACE generation process. A 'Large Language Model' block is at the bottom. Above it, a 'Model Response' sequence is shown: [0, 0, 1, 2, 3, sep, 0, 1, 2, 5, 8, sync, 3, 6, 8, sync, 3, man, 15]. Below the sequence, three heads are labeled: 'Time Head', 'Score Head', and 'Text Head'. Arrows indicate that the 'Time Head' generates time tokens (0, 0, 1, 2, 3, sep, 0, 1, 2, 5, 8) until a 'sync' token is generated. Then, the 'Score Head' generates score tokens (3, 6, 8) until another 'sync' token is generated. Finally, the 'Text Head' generates text tokens (3, man, 15). A legend at the bottom identifies the colors: orange for visual tokens, purple for time tokens, green for score tokens, and grey for text tokens.](1956f44611abd5c3c41049836aa78ad8_img.jpg)

Diagram of the TRACE generation process. A 'Large Language Model' block is at the bottom. Above it, a 'Model Response' sequence is shown: [0, 0, 1, 2, 3, sep, 0, 1, 2, 5, 8, sync, 3, 6, 8, sync, 3, man, 15]. Below the sequence, three heads are labeled: 'Time Head', 'Score Head', and 'Text Head'. Arrows indicate that the 'Time Head' generates time tokens (0, 0, 1, 2, 3, sep, 0, 1, 2, 5, 8) until a 'sync' token is generated. Then, the 'Score Head' generates score tokens (3, 6, 8) until another 'sync' token is generated. Finally, the 'Text Head' generates text tokens (3, man, 15). A legend at the bottom identifies the colors: orange for visual tokens, purple for time tokens, green for score tokens, and grey for text tokens.

Figure 4: **Generation process of TRACE.** The TRACE generate tokens following the order of time tokens, score tokens, and text tokens. The decoding heads are switched when *<sync>* tokens are generated.

## 4 EXPERIMENTS

Detailed experimental settings and hyper-parameters can be found in Appendix B.1. Numerical results on more video understanding benchmarks and more ablation studies can be found in Appendix B.2. Case studies can be found in Appendix B.3.

### 4.1 EVALUATION DATASETS, METRICS, AND BASELINE MODELS.

We evaluate the model performance on three different tasks:

- **Dense video caption.** We use Youcook2 (Zhou et al., 2018) and ActivityNet Captions (Fabian Caba Heilbron & Niebles, 2015) datasets as the evaluation datasets. The evaluation metrics include CIDEr (Vedantam et al., 2015), METEOR (Banerjee & Lavie, 2005), and SODA<sub>c</sub> (Fujita et al., 2020) for assessing the quality of the captions. These metrics are averaged under different IoU thresholds  $\{0.3, 0.5, 0.7, 0.9\}$ , following previous studies (Ren et al., 2023; Huang et al., 2023). Additionally, we report the F1 score to measure the model’s ability to accurately locate timestamps.
- **Moment retrieval.** We utilize test set of Charades-STA (Gao et al., 2017) for the moment retrieval task and report the recall at IOU thresholds of 0.5 and 0.7. Additionally, we present the mIoU results.
- **Video highlight detection.** We employ the validation set of the QVHighlights dataset (Lei et al., 2021) and report the mean average precision (mAP) with IOU thresholds of 0.5 and 0.75, as well as the HIT@1, which represents the hit ratio of the highest scored clip.

For baseline models, we select Valley (Luo et al., 2023b), VideoChat (Li et al., 2023b), VideoChatGPT (Maaz et al., 2023), and Video-LLaMA (Zhang et al., 2023) as examples of traditional video LLMs. For video LLMs specifically designed for VTG tasks, we choose TimeChat (Ren et al., 2023), VTimeLLM (Huang et al., 2023), Momentor (Qian et al., 2024), HawkEye (Wang et al., 2024b), and VTG-LLM (Guo et al., 2024).

### 4.2 PERFORMANCE OF TRACE

**Superior zero-shot performance of TRACE over other video LLMs.** In Table 2, we show the zero-shot performance of TRACE compare to SOTA video LLM baselines. The results show that

- **Superior zero-shot performance.** As shown in Table 2, TRACE significantly outperforms other video LLMs by a substantial margin across all three datasets. Notably, it achieves a 3.1 and 4.9% performance improvement on the Youcook2 dataset using the CIDEr and F1 Score metrics; a 6.5% and 3.7% performance increase in Recall with IOU  $= \{0.5, 0.7\}$  thresholds on the Charades-STA dataset; and a 10.3% and 9.2% performance gain for the mAP and HIT@1 metrics on the QVHighlights dataset.
- **Better performance than task-specific models and larger LLMs.** As shown in Table 2, as a generalist model capable of handling various tasks, the performance of TRACE surpasses that of task-specific models like HawkEye (Wang et al., 2024b). Furthermore, the 7B TRACE model outperforms the VTimeLLM (13B) model (Huang et al., 2023), further validating the advantages of the TRACE architecture.

{7}------------------------------------------------

Table 2: **Zero-shot performance of algorithms over various tasks.** We evaluated the performance of TRACE using the Youcook2, Charades-STA, and QVHighlights datasets. We highlight the best results for each block using **bold font**. The Valley, VideoChat-Embed, and Video-LLaMA results are elaborated from previous studies (Ren et al., 2023; Huang et al., 2023; Qian et al., 2024). The results with transparent text indicates unfair comparison (13B).

| Model                                | Youcook2   |            |             | Charades-STA |              | QVHighlights |             |
|--------------------------------------|------------|------------|-------------|--------------|--------------|--------------|-------------|
|                                      | SODA_c     | CIDEr      | F1 Score    | R@1(Iou=0.5) | R@1(Iou=0.7) | mAP          | HIT@1       |
| <i>Traditional Video LLMs</i>        |            |            |             |              |              |              |             |
| Valley (7B)                          | 0.1        | 0.0        | 1.5         | 4.7          | 1.6          | 10.9         | 15.2        |
| VideoChat (7B)                       | 0.2        | 0.6        | 3.4         | 3.2          | 1.4          | 13.1         | 18.1        |
| Video-LLaMA (7B)                     | 0.0        | 0.0        | 0.1         | 2.7          | 1.2          | 11.3         | 15.6        |
| <i>Temporal Grounding Video LLMs</i> |            |            |             |              |              |              |             |
| TimeChat (7B)                        | 1.2        | 3.4        | 12.6        | 32.2         | 13.4         | 14.5         | 23.9        |
| VTimeLLM (7B)                        |            |            |             | 27.5         | 11.4         |              |             |
| VTimeLLM (13B)                       |            |            |             | 34.3         | 14.7         |              |             |
| Momentor (7B)                        |            |            |             | 26.6         | 11.6         | 7.6          |             |
| HawkEye (7B)                         |            |            |             | 31.4         | 14.5         |              |             |
| VTG-LLM (7B)                         | 1.5        | 5.0        | 17.5        | 33.8         | 15.7         | 16.5         | 33.5        |
| TRACE (7B)                           | <b>2.2</b> | <b>8.1</b> | <b>22.4</b> | <b>40.3</b>  | <b>19.4</b>  | <b>26.8</b>  | <b>42.7</b> |

**Performance of TRACE on ActivityNet Captions dataset.** In Table 4, we show the performance of TRACE on ActivityNet Captions dataset. All the reported algorithms except for Momentor (Qian et al., 2024) have incorporated the ActivityNet Captions dataset as part of the training data. Results show that the TRACE attains the best performance in moment retrieval tasks and demonstrates comparable results to VTimeLLM in dense video caption tasks.

### 4.3 ABLATION STUDIES OF TRACE.

**The causal event modeling framework enhances model performance in VTG tasks.** In the 'Ablation Studies on Architecture' section of Table 3, we conducted experiments without utilizing the causal event modeling framework. The results indicate that employing the causal event modeling framework significantly improves model performance, and TRACE can achieve better results even when sampling fewer video frames.

**Using different encoders and decoding heads for different tasks is essential for TRACE to achieve the best result.** In the "w/o independent encoder/heads" part of Table 3, we performed ablation studies by not utilizing separate encoders and decoder heads for different tasks. Instead, we directly incorporated time tokens and score tokens into the text tokenizers. The results suggest that using shared encoder/decoding heads for causal event modeling framework significantly disrupts the prelearned knowledge of LLMs, leading to irrelevant and meaningless responses.

**The performance of TRACE improves with the increase in the number of frames.** We conducted ablation studies on the number of sampled frames, as presented in Table 3. The results show that (1) the performance of TRACE enhances as the number of sampled frames increases; (2) the performance of TRACE is comparable or even superior to SOTA video LLMs like VTG-LLM and TimeChat when sampling just 8 frames, demonstrating the effectiveness of the TRACE model architecture.

**Incorporating InternVid (Wang et al., 2023b) and ActivityNet Captions (Fabian Caba Heilbron & Niebles, 2015) datasets boost TRACE performance on long videos.** As illustrated in Figure 5, we carried out ablation studies by exclusively using VTG-IT as the training data for VTG tasks. The results indicate that the performance of TRACE on long videos improves when incorporating InternVid and ActivityNet Captions datasets, leading to enhanced performance on Youcook2, QVHighlights, and ActivityNet Captions datasets. Conversely, the performance of TRACE on short videos slightly decreases (Charades-STA), suggesting that the annotations in the InternVid and ActivityNet Captions datasets may not be as accurate as those in short video annotations.

{8}------------------------------------------------

Table 3: **Ablation studies of TRACE.** All the algorithms solely utilize VTG-IT (Guo et al., 2024) during fine-tuning for efficient evaluation. The "w/o causal event modeling" approach indicates the use of natural language-style inputs similar to previous studies (Guo et al., 2024; Ren et al., 2023). The "w/o independent encoder/heads" approach signifies directly adding new tokens to the LLM tokenizer instead of employing separate encoders/heads for different tasks. We highlight the best results using **bold font** for each block.

| Models                                  | Frame Number | Youcook2          |                  |             | Charades-STA |              |
|-----------------------------------------|--------------|-------------------|------------------|-------------|--------------|--------------|
|                                         |              | SODA <sub>c</sub> | CiD <sub>E</sub> | F1 Score    | R@1(IoU=0.5) | R@1(IoU=0.7) |
| <b>Ablation Studies on Architecture</b> |              |                   |                  |             |              |              |
| w/o causal event modeling               | 96           | 1.4               | 4.3              | 17.2        | 29.7         | 14.0         |
| w/o independent encoder/heads           | 64           | —                 | —                | —           | —            | —            |
| TRACE (VTG-IT)                          | 64           | <b>1.9</b>        | <b>6.9</b>       | <b>21.4</b> | <b>37.0</b>  | <b>17.0</b>  |
| <b>Ablation Studies on Frame Number</b> |              |                   |                  |             |              |              |
| TRACE (VTG-IT)                          | 8            | 1.4               | 5.0              | 18.6        | 28.8         | 13.6         |
| TRACE (VTG-IT)                          | 64           | 1.9               | 6.9              | <b>21.4</b> | 37.0         | 17.0         |
| TRACE (VTG-IT)                          | 128          | <b>2.1</b>        | <b>7.5</b>       | <b>21.4</b> | <b>41.2</b>  | <b>20.0</b>  |

![Figure 5: Four bar charts showing performance on Youcook2, Charades-STA, QVHighlights, and ActivityNet Captions datasets. Each chart compares VTG-IT (darker red) and TRACE (lighter red) across various metrics. In all cases, TRACE outperforms VTG-IT.](bedcca5cdf168e3508ef511d94ec514c_img.jpg)

Figure 5 consists of four bar charts labeled (a) through (d), each showing performance metrics for VTG-IT (darker red bars) and TRACE (lighter red bars) on different datasets. The metrics are SODA<sub>c</sub>, CDO, F1 Score, and METEOR for Youcook2; R@1<sub>IoU=1</sub>, R@1<sub>IoU=0.5</sub>, Metrics, R@1<sub>IoU=1</sub>, and nIoU for Charades-STA; and w/pt-Goal, w/pt-Goal, Metrics, w/pt-Goal, and w/pt-Goal for QVHighlights. In all cases, TRACE consistently outperforms VTG-IT across all metrics and datasets.

Figure 5: Four bar charts showing performance on Youcook2, Charades-STA, QVHighlights, and ActivityNet Captions datasets. Each chart compares VTG-IT (darker red) and TRACE (lighter red) across various metrics. In all cases, TRACE outperforms VTG-IT.

Figure 5: **Ablation studies on data utilized while training TRACE.** We conduct experiments solely utilizing VTG-IT and compare its performance with that of the original TRACE.

### 4.4 FINE-TUNED PERFORMANCE OF TRACE.

**Competitive performance of TRACE to traditional methods after fine-tuning.** In Table 5, we fine-tune the TRACE for 3 epochs on Youcook2 and Charades-STA datasets<sup>3</sup>. The results indicate that

- *TRACE significantly outperform generalist baselines.* In contrast to TimeChat and VTG-LLM, which struggle to attain satisfactory performance even after fine-tuning, the TRACE derives significant benefits from fine-tuning and achieves notably better performance than generalist baselines. These results further substantiate that our enhancements to the model architecture are crucial for VTG tasks.
- *TRACE achieve comparable performance to non-generative and task-specific SOTAs.* As depicted in Table 5, the TRACE achieves new SOTA results on Youcook2 (without audio inputs). Furthermore, the performance of TRACE on the Charades-STA dataset is also competitive with non-generative models such as InternVideo2 and VDI. *However, these methods cannot handle various tasks simultaneously and lack zero-shot capability – the contribution of TRACE herein.*

<sup>3</sup>Results on QVHighlights can be found in Appendix B.2.

{9}------------------------------------------------

Table 4: **Performance of TRACE on ActivityNet Captions dataset.** The evaluation of TimeChat’s and VTG-LLM’s results was conducted using the official provided checkpoints. The \* indicates zero-shot evaluation. We highlight the best and the second best results using **bold** and underline.

| Models    | Dense Video Caption |                   |                  |             | Moment Retrieval         |                          |             |
|-----------|---------------------|-------------------|------------------|-------------|--------------------------|--------------------------|-------------|
|           | METEOR              | SODA <sub>c</sub> | CiD <sub>r</sub> | F1 Score    | R@I <sub>(Iou=0.5)</sub> | R@I <sub>(Iou=0.7)</sub> | mIOU        |
| VTimeLLM  | <b>6.8</b>          | <u>5.8</u>        | <b>27.6</b>      |             | <u>29.5</u>              | <u>14.2</u>              | <u>31.4</u> |
| Momentor* | 4.7                 | 2.3               | 14.9             |             | 23.0                     | 12.4                     | 29.3        |
| TimeChat  | 5.7                 | 4.7               | 19.0             | 36.9        | 4.6                      | 2.0                      | 6.9         |
| VTG-LLM   | 5.9                 | 5.1               | 20.7             | 34.8        | 8.3                      | 3.7                      | 12.0        |
| TRACE     | <u>6.4</u>          | <b>6.0</b>        | <u>25.9</u>      | <b>39.3</b> | <b>37.7</b>              | <b>24.0</b>              | <b>39.0</b> |

Table 5: **Fine-tuned performance of TRACE.** We fine-tune the TRACE for 3 epochs on the Youcook2 and Charades-STA datasets. We emphasize the best and second best results using **bold font** and underline. For Youcook2 dataset, we choose Vid2Seq (Yang et al., 2023), PDVC (Wang et al., 2021), and CM<sup>2</sup> (Kim et al., 2024) as task-specific baselines. The results depicted in gray indicate unfair comparisons due to additional audio inputs and different architectures. For charades-STA dataset, we choose InternVideo2-6B (Wang et al., 2024a), VDI (Luo et al., 2023a), and Moment-DETR (Lei et al., 2021) as examples of non-generative models.

| Model                       | Youcook2          |                  |             | Model                        | Charades-STA             |                          |
|-----------------------------|-------------------|------------------|-------------|------------------------------|--------------------------|--------------------------|
|                             | SODA <sub>c</sub> | CiD <sub>r</sub> | F1 Score    |                              | R@I <sub>(Iou=0.5)</sub> | R@I <sub>(Iou=0.7)</sub> |
| <i>Task-Specific Models</i> |                   |                  |             |                              |                          |                          |
| PDVC                        | <b>4.4</b>        | <u>22.7</u>      |             | <i>Non-Generative Models</i> |                          |                          |
| Vid2Seq (Audio Input)       | 7.9               | 47.1             | 27.3        | InternVideo2-6B              | 70.0                     | 49.0                     |
| Vid2Seq                     | <u>5.7</u>        | 25.3             | 23.5        | VDI                          | 52.3                     | 31.4                     |
| CM <sup>2</sup>             | 5.3               | 31.7             | <u>28.4</u> | Moment-DETR                  | 55.7                     | 34.2                     |
| <i>Generalist Models</i>    |                   |                  |             |                              |                          |                          |
| TimeChat                    | 3.4               | 11.0             | 19.5        | <i>Generative Models</i>     |                          |                          |
| VTG-LLM                     | 3.6               | 13.4             | 20.6        | HawkEye                      | <u>58.3</u>              | 28.8                     |
| TRACE                       | <b>6.7</b>        | <b>35.5</b>      | <b>31.8</b> | TimeChat                     | 46.7                     | 23.7                     |
|                             |                   |                  |             | VTG-LLM                      | 57.2                     | <u>33.4</u>              |
|                             |                   |                  |             | TRACE                        | <b>61.7</b>              | <b>41.4</b>              |

## 5 CONCLUSION AND FUTURE WORKS

In this paper, our goal is to address the mismatch between video structure and video LLMs on VTG tasks, and propose a causal event modeling framework and the TRACE model as a solution. Numerical results indicate the superior zero-shot performance of TRACE compared to other video LLM baselines, and TRACE also achieves competitive performance relative to traditional non-generative and task-specific models after fine-tuning. By overcoming the inherent limitations of video LLM architectures, TRACE demonstrates the potential of video LLMs on VTG tasks, and we believe that the TRACE could be a strong foundation for future research on video LLMs in VTG tasks.

However, there are future works that can further enhance the capabilities of TRACE. For instance, TRACE relies on the pre-trained decoder-only LLMs, and only using previous events to predict the next event, which may not discover the complex event relationships as pointed out by previous studies (Yi et al., 2019; Girdhar & Ramanan, 2019; Li et al., 2020). As a remedy, we can use the outputs of causality discovery models (Liang et al., 2022; Chen et al., 2024c) as supplementary inputs for TRACE to provide a more comprehensive understanding of video contents. Furthermore, expanding the annotation of more video understanding tasks by incorporating the occurrence timestamps of QA pairs and the matching score between questions and answers could significantly improve the overall performance of TRACE.

## ACKNOWLEDGMENTS

This work is supported in part by the funding from Shenzhen Institute of Artificial Intelligence and Robotics for Society, in part by the Shenzhen Key Lab of Crowd Intelligence Empowered Low-Carbon Energy Network (Grant No. ZDSYS20220606100601002), in part by Shenzhen Stability

 Rest of paper (reference and Appendix) is removed.