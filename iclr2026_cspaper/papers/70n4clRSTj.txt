000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053

# Time Blindness: Why Video-Language Models Can'T See What Humans Can?

Anonymous authors Paper under double-blind review

## Abstract

Recent advances in vision–language models (VLMs) have made impressive strides in understanding spatio-temporal relationships in videos. However, when spatial information is obscured, these models struggle to capture purely temporal patterns.

We introduce **SpookyBench**, a benchmark where information is encoded solely in temporal sequences of noise-like frames, mirroring natural phenomena from biological signaling to covert communication. Interestingly, while humans can recognize shapes, text, and patterns in these sequences with over 98% accuracy, state-of-the-art VLMs achieve 0% accuracy. This performance gap highlights a critical limitation: an over-reliance on frame-level spatial features and an inability to extract meaning from temporal cues. Overcoming this limitation will require novel architectures or training paradigms that decouple spatial dependencies from temporal processing. Our systematic analysis shows that this issue persists across model scales and architectures. We release SpookyBench to catalyze research in temporal pattern recognition and bridge the gap between human and machine video understanding. Dataset is available at this anonymous link: https://tinyurl.com/spooky-bench

## 1 Introduction

Large multimodal models have revolutionized visual understanding in both images (Liu et al., 2023; Wang et al., 2024b; Bai et al., 2025; Chen et al., 2024f; Deitke et al., 2024; Dai et al., 2024) and videos (Zhang et al., 2024b; Maaz et al., 2023; Ataallah et al., 2024; Weng et al., 2024; Wang et al., 2025). Recent Video-Vision Language Models (Video-VLMs) demonstrate impressive capabilities in various tasks, from action recognition (Wu et al., 2023; Kahatapitiya et al., 2024; Zhao et al., 2023) and visual question answering (Yu et al., 2023; Min et al., 2024; Zhong et al., 2022; Ayyubi et al., 2025; Park et al., 2024) to dense captioning (Qasim et al., 2025; Yang et al., 2023; Xu et al., 2024a; Kim et al., 2024; Chen et al., 2024d; 2025b; 2024a) and temporal grounding (Chen et al., 2024c; Wang et al., 2024a; Xu et al., 2024b). Despite this rapid progress, a fundamental limitation persists. These models excel at extracting spatial features from individual frames, but struggle with purely temporal reasoning (Cores et al., 2024; Cai et al., 2024; Li et al., 2024d), a capability that comes naturally to humans. This paper introduces **SpookyBench**, a novel benchmark designed to isolate and evaluate purely temporal understanding in video models by presenting information exclusively through temporal sequences where individual frames appear as noise. Although existing benchmarks test temporal reasoning alongside spatial understanding (Cai et al., 2024; Li et al., 2024e; Yang et al., 2025b; Li et al., 2024b), **SpookyBench** differs by completely eliminating spatial cues, forcing models to derive meaning solely from changes across frames. Current approaches to video understanding (Tang et al., 2023; Nguyen et al., 2024) typically follow a hierarchical paradigm: extract frame-level features using ViTs (Bertasius et al., 2021; Radford et al., 2021; Dosovitskiy et al., 2020), integrate these features temporally, and fuse them with language for downstream tasks (Zhang et al., 2024a; Li et al., 2024c; Wu et al., 2024; Wang et al., 2024c). This paradigm has yielded significant advances in general video understanding (Li et al., 2024a; Dubey et al., 2024; Tang et al.,
2023; Nguyen et al., 2024). However, our findings reveal a critical blind spot: when information exists purely in the temporal domain without reliable frame-level features, state-of-the-art models fail catastrophically (Figure 1). The inability to decode temporal patterns has significant implications for real-world applications. In nature, organisms such as fireflies communicate through precise temporal sequences of biolumi1

![1_image_0.png](1_image_0.png)

054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 nescence (Carlson & Copeland, 1985; Owens et al., 2022; Ram´ırez-Avila et al., 2018), encoding ´
information exclusively through timing rather than spatial arrangements. These natural examples demonstrate how temporal patterns can carry rich information even when individual observations contain minimal static content. Similarly, various human technologies from Morse code to digital communication protocols rely on temporal encoding, yet current Video-VLMs lack the fundamental mechanisms to process such information. The human visual system has evolved mechanisms for processing temporal information without relying solely on spatial cues. Neuroscience research has revealed that temporal processing is distributed across neural structures rather than centralized in a single area (Mauk & Buonomano, 2004), and the brain uses intrinsic network dynamics to perform temporal computations (Paton & Buonomano, 2018). Areas such as the parietal cortex integrate temporal information along with spatial and numeric magnitudes (Bueti & Walsh, 2009). Our experiments confirm humans' remarkable temporal perception: participants achieve over 98% accuracy on **SpookyBench** tasks without training. In stark contrast, our evaluation of 15 state-of-the-art Video-VLMs, including closed-source commercial systems such as GPT-4o (Hurst et al., 2024), and Gemini 2.0 Flash (DeepMind, 2025), reveals near-zero accuracy on these same tasks. This striking performance gap persists across model architectures, parameter scales, and pre-training strategies. Models ranging from relatively compact systems (VideoLLaMA3-2B (Zhang et al., 2025)) to massive ones (GPT-4o (Hurst et al., 2024), Qwen-VL (Wang et al., 2024b)) all struggle with purely temporal patterns. Even models specifically designed for video understanding such as LongVLM (Weng et al., 2024), LLaVA-NeXT-Interleave (Li et al., 2024c), and InternVideo2.5 (Wang et al., 2025) exhibit minimal temporal pattern recognition capability. Recent efforts to enhance temporal reasoning in Video-VLMs have explored various approaches. Models like TimeChat (Ren et al., 2024), Momentor (Qian et al., 2024), and VideoLLM (Wang et al., 2024e) incorporate specialized temporal modeling mechanisms, while ST-LLM (Liu et al., 2024b), TimeMaker (Chen et al., 2024c), and Grounded-VideoLLM (Wang et al., 2024a) focus on enhancing fine-grained temporal localization capabilities. However, our evaluation reveals that none of these approaches adequately addresses the fundamental challenge of extracting meaning from purely temporal patterns without reliable spatial features. Our findings suggest that achieving human-like video understanding requires fundamentally rethinking how neural architectures process temporal information. Rather than treating temporal integration as secondary to spatial feature extraction, future models may need dedicated mechanisms for temporal pattern recognition, possibly drawing inspiration from cognitive neuroscience research on distributed neural timing mechanisms (Paton & Buonomano, 2018; Mauk & Buonomano, 2004) and specialized brain regions for temporal processing (Bueti & Walsh, 2009; Merchant et al., 2013). The substantial gap between human and machine performance on **SpookyBench** indicates that current architectures remain fundamentally "time-blind" despite their impressive performance on standard benchmarks. By exposing this critical limitation, we hope to inspire a new wave of research into temporal reasoning in Video-VLMs, bridging the gap between human and machine perception and enabling applications that rely on precise temporal understanding, from medical diagnostics to autonomous systems that must interpret subtle temporal cues in complex environments.

## 2 Related Work 2.1 Temporal Reasoning In Video-Vlms

108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 Transformer-based Video-Vision Language Models (Video-VLMs) have advanced through several architectural families, including LLaVA variants (Liu et al., 2023; Zhang et al., 2024b; Li et al.,
2024c;a; Maaz et al., 2023; Liu et al., 2024a), the Qwen series (Wang et al., 2024b; Bai et al., 2025), and InternVL models (Chen et al., 2024g;f; Wang et al., 2025). Alternative approaches have explored dual encoders (Maaz et al., 2024), interleaved tokens (Ataallah et al., 2024; Zhu et al., 2023), compression techniques (Shen et al., 2024), and multimodal fusion (Zhang et al., 2024a; Wu et al., 2024; Wang et al., 2024c). Despite architectural diversity, these models consistently exhibit limited temporal reasoning, manifesting as hallucinations (Li et al., 2024b), grounding difficulties (Wang et al., 2024a), and a reliance on linguistic shortcuts (Ko et al., 2023) across action recognition (Wu et al., 2023; Kahatapitiya et al., 2024; Zhao et al., 2023), question answering (Yu et al., 2023; Min et al., 2024; Ayyubi et al., 2025), and captioning tasks (Yang et al., 2023; Kim et al., 2024; Chen et al., 2024a). Efforts to address these shortcomings; such as timestamp-aware encoding (Ren et al., 2024), segment-level reasoning (Qian et al., 2024), direct token processing (Liu et al., 2024b), temporal separation tokens (Chen et al., 2024c), specialized temporal streams (Wang et al., 2024a;e), and novel training paradigms (Zhang et al., 2024b; Yu et al., 2023; Tang et al., 2023; Nguyen et al., 2024); have shown incremental promise. However,

| Model                                                     | Direct Prompt   | CoT      | Params   |
|-----------------------------------------------------------|-----------------|----------|----------|
| Human Performance                                         | 98.0% ± 0.6     | N/A      | N/A      |
| Open-Source Models                                        |                 |          |          |
| VideoLLaMA3-7B (Zhang et al., 2025)                       | 0% ± 0.0        | 0% ± 0.0 | 7B       |
| VideoLLaMA3-2B (Zhang et al., 2025)                       | 0% ± 0.0        | 0% ± 0.0 | 2B       |
| TimeChat-7B (Ren et al., 2024)                            | 0% ± 0.0        | 0% ± 0.0 | 7B       |
| MiniGPT4-Video (Ataallah et al., 2024)                    | 0% ± 0.0        | 0% ± 0.0 | 7B       |
| MovieChat (Song et al., 2024)                             | 0% ± 0.0        | 0% ± 0.0 | 7B       |
| Video-ChatGPT-7B (Maaz et al., 2023)                      | 0% ± 0.0        | 0% ± 0.0 | 7B       |
| VideoGPT-plus-Phi3-mini-4k (Maaz et al., 2024)            | 0% ± 0.0        | 0% ± 0.0 | 7B       |
| VILA1.5-13bLin et al. (2024)                              | 0% ± 0.0        | 0% ± 0.0 | 13B      |
| ShareGPT4Video-8B (Chen et al., 2024a)                    | 0% ± 0.0        | 0% ± 0.0 | 8B       |
| VideoLLaMA2-7B (Cheng et al., 2024)                       | 0% ± 0.0        | 0% ± 0.0 | 7B       |
| Video-LLaVA (Zhang et al., 2024b)                         | 0% ± 0.0        | 0% ± 0.0 | 7B       |
| LLaVA-NeXT-Video (Li et al., 2024c)                       | 0% ± 0.0        | 0% ± 0.0 | 8B       |
| InternVL2-40B (Chen et al., 2024f)                        | 0% ± 0.0        | 0% ± 0.0 | 40B      |
| InternVL2-8B (Chen et al., 2024f)                         | 0% ± 0.0        | 0% ± 0.0 | 8B       |
| InternVL2.5-78B (Chen et al., 2024e)                      | 0% ± 0.0        | 0% ± 0.0 | 78B      |
| InternVL2.5-8B (Chen et al., 2024e)                       | 0% ± 0.0        | 0% ± 0.0 | 8B       |
| InternVideo2.5-Chat-8B (Wang et al., 2025)                | 0% ± 0.0        | 0% ± 0.0 | 8B       |
| InternVideo2-Chat-8B (Wang et al., 2024d)                 | 0% ± 0.0        | 0% ± 0.0 | 8B       |
| Qwen2-VL-2B-Instruct (Wang et al., 2024b)                 | 0% ± 0.0        | 0% ± 0.0 | 2B       |
| Qwen2-VL-7B-Instruct (Wang et al., 2024b)                 | 0% ± 0.0        | 0% ± 0.0 | 7B       |
| Qwen2-VL-72B-Instruct (Wang et al., 2024b)                | 0% ± 0.0        | 0% ± 0.0 | 72B      |
| Qwen2.5-VL-3B-Instruct (Bai et al., 2025)                 | 0% ± 0.0        | 0% ± 0.0 | 3B       |
| Qwen2.5-VL-7B-Instruct (Bai et al., 2025)                 | 0% ± 0.0        | 0% ± 0.0 | 7B       |
| Qwen2.5-VL-72B-Instruct (Bai et al., 2025)                | 0% ± 0.0        | 0% ± 0.0 | 72B      |
| Closed-Source Models                                      |                 |          |          |
| Gemini 1.5 Pro (Team et al., 2024)                        | 0% ± 0.0        | 0% ± 0.0 | N/A      |
| Gemini 2.0 FlashDeepMind (2025)                           | 0% ± 0.0        | 0% ± 0.0 | N/A      |
| GPT-4o (Hurst et al., 2024)                               | 0% ± 0.0        | 0% ± 0.0 | N/A      |
| Table 1: Benchmark results comparing model performance on |                 |          |          |

these methods, and even specialized video architectures like VideoGPT+ (Maaz et al., 2024), TimeChat (Ren et al., 2024), LinVT (Gao et al., 2024), LongVLM (Weng et al., 2024), and Baichuan-Omni (Li et al., 2024f), still operate on a spatialfirst paradigm where temporal understanding is secondary to spatial feature extraction. The fundamental limitations of this spatial-first approach are increasingly evidenced by temporal understanding benchmarks. TemporalBench (Cai et al., 2024) reveals a significant gap between model and human performance, while TVBench (Cores et al., 2024), VITATECS (Li et al., 2024e), and Fateh et al. (Fateh et al., 2024) confirm that many datasets inadvertently reward spatial analysis over genuine temporal reasoning. Focused evaluations further target specific failures such as temporal hallucinations with VidHalluc (Li et al., 2024b), streaming video reasoning with SVBench (Yang et al., 2025b), and challenges in temporal location, object tracking, and anomaly detection with VideoVista (Li et al., 2024g). A critical and consistent finding across these analyses is that models—including video-specific ones like LLaVA-Video (Zhang et al., 2024b), Video-ChatGPT (Maaz et al., 2023), TemporalVLM (Fateh et al., 2024), and VidChain (Lee et al., 2025)—exploit spatial shortcuts to circumvent temporal reasoning (Wang et al., 2024a; Chen et al., 2024c; Li et al., 2024b; Ko et al., 2023). Our SpookyBench benchmark is designed to directly address this issue. By deliberately obscuring spatial information, it isolates temporal pattern recognition, forcing models to derive meaning solely from temporal dynamics. This approach provides a rigorous evaluation of the "time-blindness" in current architectures, exposing fundamental limitations that remain hidden in conventional assessments. 2.2 NEUROSCIENCE INSIGHTS ON TEMPORAL PROCESSING Neuroscience research offers critical insights for addressing temporal limitations in Video-VLMs.

Mauk and Buonomano (Mauk & Buonomano, 2004) established that temporal processing is distributed across neural structures through intrinsic circuit properties, contrasting with current Video-
VLMs' sequential spatial processing. Biological systems span multiple granularities: cerebellum handles millisecond-to-second timing (Merchant et al., 2013); parietal cortex integrates temporal, spatial and numerical magnitudes (Bueti & Walsh, 2009); and neural patterns dynamically encode time through "population clocks" (Paton & Buonomano, 2018). Models could benefit from distributed temporal representations that evolve over time (Wittmann, 2009; Paton & Buonomano, 2018) rather than treating temporal integration as secondary. The performance gap on temporal tasks (Cai et al., 2024; Cores et al., 2024; Li et al., 2024b) and our SpookyBench findings demonstrate that current architectures lack mechanisms for processing purely temporal patterns—a natural capability in humans through neural systems representing time as intrinsic dynamics.

## 3 Spookybench

162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 We introduce **SpookyBench**, a novel synthetic dataset specifically designed to isolate and evaluate pure temporal understanding in video language models. The key innovation of our benchmark lies in its unique design: All meaningful information is encoded exclusively in the temporal domain through dynamic patterns of texts, images and video depth maps, while individual frames contain only structured noise. Our dataset is fundamentally different from the existing datasets used for training, fine-tuning, and evaluation of video-VLMs. Many state-of-the-art video language models employ advanced techniques, such as dynamic resolution strategies (Bai et al., 2025; Wang et al., 2024b; Chen et al., 2024f), specialized temporal encoding methods (Ren et al., 2024; Wang et al.,
2024b; Bai et al., 2025), hierarchical token merging (Weng et al., 2024; Wang et al., 2025), and joint video-motion training frameworks (Chen et al., 2024b) to capture temporal dynamics. However, these methods still rely on spatial representations extracted from individual frames, which currently remain the only viable mechanism for inferring temporal information. In contrast, **SpookyBench** forces models to depend only on temporal cues, thereby creating the first benchmark that exclusively evaluates a model's ability to process and understand pure temporal information.

![3_image_0.png](3_image_0.png)

Figure 2: Illustration of the temporal encoding framework used in SpookyBench. **Left:** Core mechanism showing how content becomes visible through opposing motion patterns. A content mask defines regions where foreground noise (moving up/left) and background noise (moving down/right) are applied. When animated, the human visual system groups pixels with similar motion, causing the content to emerge. **Right:** Comparison between moving and paused states, demonstrating how content is only perceptible during animation and disappears when static, as individual frames contain only structured noise.

## 3.1 Dataset Generation

Figure 2 shows our proposed data generation framework. The dataset consists of specially designed videos that encode three types of content - words, images, and videos - using binary noise patterns with specific motion properties. In this approach, content is embedded within noise patterns such that individual frames appear as random noise, while the content becomes perceptible only when viewed as a temporal sequence. Our dataset encodes different types of content (Figure 3) through 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 temporal noise animations in the following categories: **1) Words:** Text rendered as masks in which the background noise and foreground noise move in opposite directions, making the text visible only through temporal movement. **2)Images:** Binary masks generated using SAM2 (Ravi et al., 2024) from single-object images generated using text-to-image model Flux (Labs, 2024), encoded using the same content mask animation approach as words. **3) Dynamic Scenes:** Depth maps extracted from videos in single-object tracking datasets LaSOT (Fan et al., 2019) and OTB2015 (Wu et al., 2015) using Video Depth Anything (Chen et al., 2025a). These are encoded using a technique in which pixels above a brightness threshold move while others remain static as shown in the algorithm 2.

## 3.2 Temporal Encoding Framework

Our temporal encoding framework implements two distinct motion configurations as detailed in Algorithms 1 and 2. For words, and image masks (Algorithm 1), we employ opposing motion patterns between foreground and background. The content is first converted to a binary mask M where M(*x, y*) = 1 represents foreground pixels and M(*x, y*) = 0 represents background. We generate two separate noise patterns Nbg and Nf g consisting of random binary values (0 or 255). During animation, foreground pixels sample from Nf g with a positive offset that increases with time (y + vt mod h), while background pixels sample from Nbg with a negative offset (y − vt mod h).

This creates the perception of opposing motion within and outside the masked regions. For video depth maps (Algorithm 2), we employ a threshold-based approach. Using depth maps D extracted from videos, pixels with brightness values between lower and upper thresholds (tl ≤ d ≤ tu) are animated by sampling a noise pattern N with a time-varying offset (y + vt mod h), while pixels outside this range remain static. This creates the illusion that brighter regions (typically foreground objects) are moving while darker regions (typically background) remain static. The noise patterns are generated using binary values (0 or 255) in square blocks of variable size. We used different speckle sizes ranging from 1 × 1 to 3 × 3 pixels to investigate the effect of noise granularity on perception. For each speckle size, we also varied the noise density - the probability that a block is white versus black - using values of 10%, 30%, 50%, and 90%. These noise patterns arranged in pixel blocks create optimal perceptual conditions for human viewers while remaining challenging for vision language models. To ensure seamless animation, the noise patterns are made tileable by copying edge pixels to the opposite boundaries. All videos maintain consistent technical specifications: 960 × 540 pixel resolution, with an average duration of 7.11 seconds (ranging from 1.0 to 35.0 seconds) and an average of 333.5 frames per video. Text videos have a consistent duration of around 4 seconds; however, videos of dynamic scenes are longer, ranging up to 35 seconds. Figure 2 illustrates the structure of the data set and the encoding patterns in categories. We used binary masks for the images using SAM2 (Ravi et al., 2024). For videos, depth maps are extracted using Depth Anything V2 (Yang et al., 2025a) and Video Depth Anything (Chen et al., 2025a) from the LaSOT (Fan et al., 2019) and OTB2015 (Wu et al., 2015) datasets.

Category Basic SNR (dB) Perceptual SNR Temporal Coherence Motion Contrast Images -46.95 ± 2.40 -47.28 ± 2.28 8.00 ± 2.08 7.17 ± 5.00 Dynamic Scenes -48.95 ± 3.64 -63.43 ± 5.74 21.91 ± 5.76 -3.18 ± 10.17 Text -39.27 ± 1.58 -49.18 ± 3.31 7.84 ± 0.65 8.26 ± 6.44 Table 2: Signal-to-Noise Ratio (SNR) metrics across Spooky-
Bench categories.

## 3.3 Data Statistics

SpookyBench comprises 451 videos in three distinct categories, each requiring purely temporal reasoning for content identification. The dataset is distributed as follows: Text (46.6%, 210 videos), Object Images (40.8%, 184 videos) and Dynamic Scenes (12.6%, 57 videos). This distribution ensures comprehensive coverage of different temporal perception challenges while maintaining a natural frequency distribution that reflects real-world scenarios. Additionally, more dataset can be generated indefinitely through the data generator on our project page, thus the dataset size is essentially unlimited. The "Text" category contains common English words rendered through temporal noise patterns, enabling evaluation of models' ability to identify linguistic content through purely temporal cues. The "Object Images" category presents single objects extracted from high-quality images using segmentation techniques (Ravi et al., 2024), encoded with the same temporal animation approach. It also contains a synthetic silhouette of simple objects generated using DALL-E 3 (Betker et al., 2023) and flux (Labs, 2024).

## 3.3.1 Analysis Of Temporal Metrics

To ensure a rigorous quantification of the temporal information present in each video, we analyzed five key 2. SNR metrics that capture different aspects of the complexity and perceptibility of temporal patterns in SpookyBench, as shown in Table. These metrics provide insight into why temporal patterns might be visible to humans but challenging to detect by computational models.

270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 where PS = E[∥∇F∥
2] is motion boundary energy derived from spatial gradients of optical flow field F(*x, y*) = (Fx, Fy), and PN = Var(I0) is variance of the static frame I0.

Perceptual SNR incorporates frequencydependent visual sensitivity:

$$\mathbf{SNR}_{P}=10\log_{10}\left({\frac{\|{\mathcal{H}}(B)\odot W\|^{2}}{\|{\mathcal{H}}(N)\odot W\|^{2}}}\right)$$
$\mu$. 

![5_image_0.png](5_image_0.png)

![5_image_1.png](5_image_1.png)

$$({\mathfrak{I}})$$

Figure 3: Noise generation process: (Top) masks applied for dynamic noise video generation, (Mid) wordspecific mask, and (Bottom) depth map of video frame used for constructing noise-overlaid stimulus.

(2)
where B is the average motion boundary strength, N is the static noise frame, H is the 2D Fourier transform, ⊙ denotes element-wise multiplication, and W(f) = f · e
−f/f0is the contrast sensitivity weighting function with peak f0 ≈ 0.1 cycles/pixel.

Temporal Coherence SNR quantifies motion consistency:

$$\mathbf{SNR}_{T}=10\log_{10}\left({\frac{\mathrm{Var}(C)}{\mathbb{E}[\mathrm{Var}_{\mathrm{local}}(C)]}}\right)$$
(3)
where C = e
−Varθ(F)·1(∥F∥ > τ ) is the directional coherence map, Varθ computes circular variance of flow direction angles over time, 1 is indicator function, τ is magnitude threshold, and Varlocal computes variance over small spatial neighborhoods. Motion Contrast SNR measures foreground-background motion differentiation:

$$\mathrm{SNR}_{M}=10\log_{10}\left(\frac{\|\mu_{M}-\mu_{B}\|^{2}}{\frac{1}{2}(\sigma_{M}^{2}+\sigma_{B}^{2})}\right)$$

$$(4)$$
(4)
Basic SNR measures signal-to-noise ratio in decibels:

$$\mathbf{SNR}_{B}=10\log_{10}\left({\frac{P_{S}}{P_{N}}}\right)$$
(1)
where µM = E[F | M] and µB = E[F | ¬M] are mean flow vectors within mask region M and background region ¬M respectively, σ 2 M = E[∥F − µM∥
2| M] and σ 2 B = E[∥F − µB∥
2| ¬M]
are corresponding motion variances. The mask M is estimated from the motion boundaries.

The distribution of these metrics reveals why current vision models struggle with **SpookyBench**: they lack mechanisms to leverage temporal coherence (particularly high in Dynamic Scenes, 21.91 ±
5.76 dB) and motion contrast (negative for Dynamic Scenes, -2.20 and -3.18 dB), while text stimuli benefit from higher basic SNR (-39.27 ± 1.58 dB), explaining the observed performance gap.

## 3.3.2 Binary Snr Threshold Effect In Detection

Our analysis revealed a critical binary threshold phenomenon in detecting text within dynamic noise videos. The words exhibited negligible detection (∼0%) below 2.5dB SNR, but jumped to 85.7% accuracy above this threshold, displaying an abrupt rather than gradual transition as show in 4. Prompts performed best (40% accuracy), with Chain-of-Thought reasoning improving general identification tasks compared to direct prompting. This phenomenon parallels medical imaging diagnostics, where pathologies like microcalcifications in mammography become either entirely

| Algorithm 1 Content Mask Animation 1: Input: Content mask M, velocity v 2: Output: Animated frame Ft 3: Generate noise patterns Nbg, Nf g 4: for each pixel (x, y) do ▷ Check pixel's mask status 5: if M(x, y) then 6: Ft(x, y) ← Nf g(x, y + vt mod h) ▷ Foreground 7: else 8: Ft(x, y) ← Nbg(x, y − vt mod h) ▷ Background 9: end if 10: end for   |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

| Algorithm 2 Video Depth Map Animation 1: Input: Depth map D, thresholds (tl , tu), velocity v 2: Output: Animated frame Ft 3: Generate noise pattern N 4: for each pixel (x, y) do 5: d ← brightness from D(x, y) 6: if tl ≤ d ≤ tu then 7: Ft(x, y) ← N(x, y + vt mod h) ▷ Moving noise 8: else 9: Ft(x, y) ← N(x, y) ▷ Static noise 10: end if 11: end for   |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

visible or invisible based on specific SNR thresholds. The implications are significant: unlike perceptual phenomena that degrade gradually with noise, text detection functions as a step function, creating vulnerabilities in safety-critical applications. Just as radiologists cannot diagnose what remains invisible, language models cannot identify text below certain noise thresholds, leading to false certainties and potential catastrophic performance drops with minimal noise increases. This characteristic creates particular concerns for autonomous vehicles reading road signs or medical systems interpreting labels, while also exposing systems to adversarial attacks where slight SNR manipulations could render text completely undetectable.

## 4 Experiments 4.1 Experimental Setup

324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 To evaluate human performance against our benchmark, we designed and conducted a controlled experiment involving human participants. We recruited a total of six human participants for this study, each independently evaluating all videos. Participants were instructed to view each video carefully and subsequently record their responses on an anonymized website in the following structured form: 1) **Perceptibility Rating (1-5):** Participants rated how perceptible the Models. We evaluated **SpookyBench** on both open source models (Video-LLaVA (Zhang et al.,
2024b), LLaVA-NeXT-Video (Li et al., 2024c), TimeChat (Ren et al., 2024), InternVL2 (Chen et al.,
2024f), Qwen2-VL (Wang et al., 2024b), Qwen2.5-VL (Bai et al., 2025) etc.) and closed source models (GPT-4o (Hurst et al., 2024), Gemini 2.0 Flash (DeepMind, 2025), and Gemini 1.5 Pro (Team et al., 2024). We design different prompts for each category. All the prompts are included in the Appendix C. All prompts instruct models to respond with only 1-5 words identifying the content. We input sequences of multiple video frames simultaneously for models that do not directly support video input. Setup. We evaluate model performance using exact match accuracy between model responses and our labels. For the Text categories, each video has a single correct label yi. For Object Images and Dynamic Scenes categories, we define a set of acceptable labels Yi = {yi1, yi2*, . . . , y*in} to account for semantic ambiguity. For example, a video showing "a man playing basketball" accepts responses such as "playing basketball," "man", "human", or "woman playing basketball" as correct.

Formally, for each video i, given a model response ri and corresponding label or set of labels Li (where Li = yi for Text or Li = Yi for objects and dynamic scenes), we calculate the accuracy as: Accuracy =
1 N
PN
i=1 1(ri ∈ Li), where 1 is the indicator function that equals 1 if ri ∈ Li and 0 otherwise, and N is the total number of videos in the evaluation set. Despite this flexible evaluation protocol that accepts multiple valid responses for certain categories, none of the models tested produced responses that matched any of the acceptable options.

## 4.2 Human Evaluation

378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 presented word, shape, or object was, ranging from 1 (very difficult to perceive) to 5 (very clearly perceptible). This measure provided insights into the clarity and ease of visual grouping. 2) **Words/Objects Identification:** Participants typed out exactly what they identified in the video. This response directly tested the accuracy of their visual perception. We collect and evaluate participant responses using exact match criteria based on our predefined labels. Similar to the evaluation accuracy of the video language models for the categories of Object Images and Dynamic Scenes, we accepted multiple correct responses to avoid ambiguity. Table 3 shows the average precision and the perception rating of different annotators for different categories. The results show high human performance across all categories: participants correctly identified Words with 98% accuracy, while Object Images had 92% accuracy. We also observe a very high perceptibility rating (4.8 for texts and 4.3 and 4.0 for Object images and Dynamic scenes, respectively) across all three categories. This shows that the human brain can easily extract coherent information in videos, which seems to be very difficult for video language models.

Annotator Text Images **Dynamic Scenes**
Acc(%) Perc(1-5) Acc(%) Perc(1-5) **Acc(%) Perc(1-5)**
Annotator 1 99.5 4.7 99.5 4.7 96.5 4.3 Annotator 2 98.6 4.8 98.4 4.9 91.2 4.0 Annotator 3 99.5 4.9 97.2 4.5 94.7 4.4 Annotator 4 97.6 4.6 96.7 4.5 91.2 4.0 Annotator 5 100.0 4.8 99.5 4.7 99.0 4.7 Annotator 6 98.0 4.7 97.8 4.5 93.0 4.2 Mean 98.9±0.7 4.8±0.0 98.2±1.1 4.7±0.1 94.3±3.1 4.3±0.1 Table 3: Human evaluation results showing accuracy and perceptibility ratings across different visual categories in SpookyBench.

## 4.3 Impact Of Frame Rates On Human And Model Accuracy

To examine whether temporal sampling affects performance, we evaluated both humans and VLMs across frame rates from 1 to 30 FPS. Three human participants tested 120 randomly sampled videos (40 per category) at 1, 5, 10, 20, and 30 FPS, while four VLMs
(Qwen2-VL-7B, Qwen2.5-VL-7B, Qwen2.5- VL-3B, and GPT-4o) were evaluated using identical temporal downsampling. As shown in Tables 4 and 5, human accuracy remains above 95% at 20-30 FPS, degrades to 59.4% at 10 FPS, and drops to 0% at 1 FPS. In contrast, all VLMs achieved 0% accuracy across all frame rates. This demonstrates that temporal sampling frequency does not explain the performance gap between humans and current video-language models, indicating that VLMs lack the architectural mechanisms to process information conveyed through temporal patterns regardless of temporal resolution.

Category 1 FPS 5 FPS 10 FPS 20 FPS 30 FPS Images 0.0 12.5 80.0 95.8 97.5 Words 0.0 10.8 35.8 95.8 95.8 Videos 0.0 15.0 62.5 93.3 93.3 Average 0.0 12.8 59.4 95.0 95.6 Table 4: Human accuracy (%) across different content categories at varying frame rates. Results are averaged across 3 participants on 120 videos (40 per category).

## 4.4 Impact Of Finetuning On Model Accuracy

To investigate whether the performance gap stems from out-of-distribution data rather than architectural limitations, we finetuned two state-of-the-art video-language models on SpookyBench: InternVL2.5-8B and Qwen2-VL-7B. Both models were trained on 400 SpookyBench videos for 10 epochs using LlamaFactory (Zheng et al., 2024). Despite this targeted training on the exact task and data distribution, both models maintained 0% accuracy on the test set. This result demonstrates that the failure to decode temporal patterns is not attributable to domain mismatch or insufficient exposure to the task, but rather indicates a fundamental architectural inability to process information conveyed purely through motion without relying on spatial content.

## 5 Results And Discussion

Table 1 presents the accuracy scores on the **SpookyBench** benchmark. Human participants achieved 98% accuracy under all test conditions. In contrast, all Video-VLMs scored 0% regardless of the type, size, or origin of the model. This pattern was held across all three task categories in our benchmark: temporal symbol recognition, temporal sequence understanding, and temporal pattern reasoning. We tested two different prompting strategies to determine if performance limitations could be overcome through interface modifications. First, we used direct prompts with basic instructions asking the models to identify content in the videos. Next, we implemented 432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 chain-of-thought prompts with explicit guidance to focus on temporal patterns rather than individual frames. As shown in Table 1, none of these approaches yielded improvements. All models maintained 0% accuracy across all prompting conditions, suggesting that the limitation is inherent in the model architectures rather than a matter of optimization or prompt design. Examination of model output revealed consistent failure modes when processing **SpookyBench** videos.

Across all models tested, we observed attempts to extract information from individual frames rather than temporal patterns. When explicitly prompted to consider temporal changes, the models acknowledged the instruction but still failed to identify the patterns. Fine-tuned models produced outputs that mimicked training examples without correctly identifying test patterns. In particular, specialized temporal models like TimeChat (Ren et al., 2024), which were specifically designed for fine-grained temporal understanding, failed at the same rate as general-purpose models. This suggests that the limitation extends beyond general Video-VLMs to models explicitly optimized for temporal tasks.

Architectural Implications for Vision Models. Distinctive signal profiles in **SpookyBench**
demonstrate a fundamental gap between human and machine perception of temporal information. Current vision models struggle with **SpookyBench** stimuli primarily because they: (1) lack robust temporal integration mechanisms that could leverage high temporal coherence, (2) process information primarily through spatial rather than temporal channels, and (3) fail to perform motion-based figure-ground segregation effectively.

Model Qwen2-VL-7B Qwen2.5-VL-7B Qwen2.5-VL-3B GPT-4o Accuracy (%) 0.0 0.0 0.0 0.0 Table 5: VLM accuracy (%) averaged across all tested frame rates (1-30 FPS).

The consistently high temporal coherence values in Dynamic Scenes, coupled with their poor static-frame metrics, suggest that successful models must implement recurrent processing or attention mechanisms that operate across extended temporal windows rather than focusing on framelevel feature extraction. The negative motion contrast observed in Dynamic Scenes further indicates that models require more sophisticated motion segregation capabilities to match human perceptual abilities in dynamic visual environments. These findings highlight the need for architectural innovations that specifically address temporal processing limitations. Future models should incorporate dedicated temporal coherence pathways, motion contrast analysis, and longer temporal integration windows to bridge the perception gap demonstrated by **SpookyBench**.

## 6 Conclusion

In this paper, we introduced **SpookyBench**, a novel benchmark designed to evaluate the temporal reasoning capabilities of video-language models by isolating temporal understanding from spatial comprehension. Our experiments revealed a striking performance gap: while humans effortlessly achieve 98% accuracy on tasks requiring pure temporal pattern recognition, all tested models, including state-of-the-art open and closed-source systems, fail completely with 0% accuracy. This consistent failure across different model architectures, scales, and prompting strategies highlights a fundamental limitation in current video understanding approaches, which typically process spatial features first and then establish temporal connections, rather than integrating spatio-temporal information simultaneously. The benchmark effectively exposes the *time blindness* of current architectures that remain hidden in conventional evaluation settings where spatial features can provide shortcuts to correct answers. We hope that **SpookyBench** will inspire the development of next-generation temporal-connected models.

Figure 4: Analysis of effects of SNR on detecting words with direct prompting and chain of thought prompting.

![8_image_0.png](8_image_0.png)

## Ethics Statement 486

487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 This work introduces SpookyBench, a synthetic benchmark for evaluating temporal understanding in video-language models, and does not involve collection of personally identifiable information or creation of harmful content. The human evaluation component involved six volunteer participants who provided informed consent and were free to withdraw at any time. All participant responses were anonymized and stored securely. We used publicly available datasets (LaSOT, OTB2015) and models under their original licenses, and evaluated both open-source and commercial VLMs following their respective usage policies. While exposing fundamental limitations in current video-language models may impact their deployment in safety-critical applications, we believe this transparency is essential for responsible AI development. The synthetic nature of our dataset eliminates concerns about data consent or privacy violations. We acknowledge that improved temporal understanding capabilities could potentially be misused, but the same capabilities are fundamental for beneficial applications in medical imaging, autonomous systems, and accessibility technologies. We encourage responsible development and deployment practices, including human oversight in critical applications and adherence to existing AI safety guidelines.

## Reproducibility Statement

SpookyBench is generated using fully deterministic algorithms detailed in Algorithms 1 and 2, with specific parameters for noise generation, motion patterns, and video specifications clearly documented. We will release: (i) complete code for dataset generation with all hyperparameters (velocity values, noise densities, speckle sizes, threshold ranges); (ii) the full SpookyBench dataset with 451 videos across three categories; (iii) exact evaluation prompts for both direct and chain-of-thought strategies; (iv) model evaluation scripts with specific version numbers and inference parameters; and (v) finetuning configurations used with LlamaFactory. All SNR metrics are computed using the mathematical formulations provided in Section 3.3. We document the exact model versions evaluated (e.g., Qwen2.5-VL-7B-Instruct, InternVL2.5-8B) and will provide environment specifications including framework versions, hardware details, and computational requirements. The human evaluation methodology, including participant instructions and response collection protocols, is fully documented to enable replication of the human baseline results.

## References

Kirolos Ataallah, Xiaoqian Shen, Eslam Abdelrahman, Essam Sleiman, Deyao Zhu, Jian Ding, and Mohamed Elhoseiny. Minigpt4-video: Advancing multimodal llms for video understanding with interleaved visual-textual tokens. *arXiv preprint arXiv:2404.03413*, 2024.

Hammad Ayyubi, Junzhang Liu, Ali Asgarov, Zaber Ibn Abdul Hakim, Najibul Haque Sarker, Zhecan Wang, Chia-Wei Tang, Hani Alomari, Md Atabuzzaman, Xudong Lin, et al. Enter: Event based interpretable reasoning for videoqa. *arXiv preprint arXiv:2501.14194*, 2025.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. *arXiv preprint arXiv:2502.13923*,
2025.

Gedas Bertasius, Heng Wang, and Lorenzo Torresani. Is space-time attention all you need for video understanding? In *ICML*, volume 2, pp. 4, 2021.

James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023.

Domenica Bueti and Vincent Walsh. The parietal cortex and the representation of time, space, number and other magnitudes. *Philosophical Transactions of the Royal Society B: Biological Sciences*, 364(1525):1831–1840, 2009.

Mu Cai, Reuben Tan, Jianrui Zhang, Bocheng Zou, Kai Zhang, Feng Yao, Fangrui Zhu, Jing Gu, Yiwu Zhong, Yuzhang Shang, et al. Temporalbench: Benchmarking fine-grained temporal understanding for multimodal video models. *arXiv preprint arXiv:2410.10818*, 2024.

Albert D Carlson and Jonathan Copeland. Flash communication in fireflies. The Quarterly review of biology, 60(4):415–436, 1985.

Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Bin Lin, Zhenyu Tang, et al. Sharegpt4video: Improving video understanding and generation with better captions. *arXiv preprint arXiv:2406.04325*, 2024a.

Ling-Hao Chen, Shunlin Lu, Ailing Zeng, Hao Zhang, Benyou Wang, Ruimao Zhang, and Lei Zhang.

Motionllm: Understanding human behaviors from human motions and videos. arXiv preprint arXiv:2405.20340, 2024b.

Shimin Chen, Xiaohan Lan, Yitian Yuan, Zequn Jie, and Lin Ma. Timemarker: A versatile video-llm for long and short video understanding with superior temporal localization ability. *arXiv preprint* arXiv:2411.18211, 2024c.

Sili Chen, Hengkai Guo, Shengnan Zhu, Feihu Zhang, Zilong Huang, Jiashi Feng, and Bingyi Kang. Video depth anything: Consistent depth estimation for super-long videos. *arXiv preprint* arXiv:2501.12375, 2025a.

Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang, et al. Panda-70m: Captioning 70m videos with multiple cross-modality teachers. In *Proceedings of the IEEE/CVF* Conference on Computer Vision and Pattern Recognition, pp. 13320–13331, 2024d.

Xinlong Chen, Yuanxing Zhang, Chongling Rao, Yushuo Guan, Jiaheng Liu, Fuzheng Zhang, Chengru Song, Qiang Liu, Di Zhang, and Tieniu Tan. Vidcapbench: A comprehensive benchmark of video captioning for controllable text-to-video generation. *arXiv preprint arXiv:2502.12782*, 2025b.

Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. *arXiv preprint arXiv:2412.05271*, 2024e.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. *Science China Information Sciences*, 67(12):220101, 2024f.

Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In *Proceedings of the IEEE/CVF Conference on Computer* Vision and Pattern Recognition, pp. 24185–24198, 2024g.

Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, et al. Videollama 2: Advancing spatial-temporal modeling and audio understanding in video-llms. *arXiv preprint arXiv:2406.07476*, 2024.

Daniel Cores, Michael Dorkenwald, Manuel Mucientes, Cees GM Snoek, and Yuki M Asano.

Tvbench: Redesigning video-language evaluation. *arXiv preprint arXiv:2410.07752*, 2024.

Wenliang Dai, Nayeon Lee, Boxin Wang, Zhuolin Yang, Zihan Liu, Jon Barker, Tuomas Rintamaki, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. Nvlm: Open frontier-class multimodal llms. arXiv preprint arXiv:2409.11402, 2024.

Google DeepMind. Gemini flash, 2025. URL https://deepmind.google/
technologies/gemini/flash/. Accessed: 2025-02-24.

Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, et al. Molmo and pixmo: Open weights and open data for state-of-the-art multimodal models. *arXiv preprint arXiv:2409.17146*, 2024.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Heng Fan, Liting Lin, Fan Yang, Peng Chu, Ge Deng, Sijia Yu, Hexin Bai, Yong Xu, Chunyuan Liao, and Haibin Ling. Lasot: A high-quality benchmark for large-scale single object tracking.

In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp.

5374–5383, 2019.

Fawad Javed Fateh, Umer Ahmed, Hamza Khan, M Zeeshan Zia, and Quoc-Huy Tran. Video llms for temporal reasoning in long videos. *arXiv preprint arXiv:2412.02930*, 2024.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Lishuai Gao, Yujie Zhong, Yingsen Zeng, Haoxian Tan, Dengjie Li, and Zheng Zhao. Linvt: Empower your image-level large language model to understand videos. *arXiv preprint arXiv:2412.05185*, 2024.

Jindong Gu, Zhen Han, Shuo Chen, Ahmad Beirami, Bailan He, Gengyuan Zhang, Ruotong Liao, Yao Qin, Volker Tresp, and Philip Torr. A systematic survey of prompt engineering on vision-language foundation models. *arXiv preprint arXiv:2307.12980*, 2023.

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. *arXiv preprint* arXiv:2410.21276, 2024.

Woojeong Jin, Yu Cheng, Yelong Shen, Weizhu Chen, and Xiang Ren. A good prompt is worth millions of parameters: Low-resource prompt-based learning for vision-language models. arXiv preprint arXiv:2110.08484, 2021.

Dohwan Ko, Ji Soo Lee, Wooyoung Kang, Byungseok Roh, and Hyunwoo J Kim. Large language models are temporal and causal reasoners for video question answering. arXiv preprint arXiv:2310.15747, 2023.

Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024. Ji Soo Lee, Jongha Kim, Jeehye Na, Jinyoung Park, and Hyunwoo J Kim. Vidchain: Chain-of-tasks with metric-based direct preference optimization for dense video captioning. arXiv preprint arXiv:2501.06761, 2025.

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024a.

Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li.

Llava-next-interleave: Tackling multi-image, video, and 3d in large multimodal models. arXiv preprint arXiv:2407.07895, 2024c.

Kumara Kahatapitiya, Anurag Arnab, Arsha Nagrani, and Michael S Ryoo. Victr: Video-conditioned text representations for activity recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18547–18558, 2024.

Minkuk Kim, Hyeon Bae Kim, Jinyoung Moon, Jinwoo Choi, and Seong Tae Kim. Do you remember?

dense video captioning with cross-modal memory retrieval. In *Proceedings of the IEEE/CVF* Conference on Computer Vision and Pattern Recognition, pp. 13894–13904, 2024.

Chaoyu Li, Eun Woo Im, and Pooyan Fazli. Vidhalluc: Evaluating temporal hallucinations in multimodal large language models for video understanding. *arXiv preprint arXiv:2412.03735*,
2024b.