# HFSTI-Net: Hierarchical Frequency-spatial-temporal Interactions for Video Polyp Segmentation

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 6, 4, 4

## Abstract
Automatic video polyp segmentation (VPS) is crucial for preventing and treating colorectal cancer by ensuring accurate identification of polyps in colonoscopy examinations. However, its clinical application is hampered by two key challenges: shape collapse, which compromises structural integrity, and episodic amnesia, which causes instability in challenging video sequences. To address these challenges, we present a novel video segmentation network, \emph{HFSTI-Net}, which integrates global perception with spatiotemporal consistency in spatial, temporal, and frequency domains. Specifically, to address shape collapse under low contrast or visual ambiguity, we design a Hierarchical Frequency-spatial Interaction (HFSI) module that fuses spatial and frequency cues for fine-grained boundary localization. Furthermore,  we propose a recurrent mask-guided propagation (RMP) module that introduces a dual enhancement mechanism based on feature memory and mask alignment, effectively incorporating spatiotemporal information to alleviate inter-frame inconsistencies and ensuring long-term segmentation stability. Extensive experiments on the SUN-SEG and CVC-612 datasets demonstrate that our method achieves real-time inference and outperforms other state-of-the-art approaches. Codes are available at \url{https://github.com/Yuanqin-He/HFSTI-Net}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents the HFSTI model for video polyp segmentation to address two major challenges in current video polyp segmentation: shape collapse and episodic amnesia. The HFSTI model consists of two main modules: HFSI and RMP. The HFSI module is responsible for extracting and fusing spatial and frequency features, while the RMP module focuses on capturing long-term spatiotemporal dependencies in polyp videos.
The HFSI module comprises three components: the FFB (Frequency Filter Block), SRB (Spatial Refinement Block), and IFB (Interwoven Fusion Block). The FFB uses a frequency-domain self-attention mechanism to extract enhanced frequency-aware features, while the SRB applies spatial self-attention to capture spatial features. The IFB utilizes gated multiplicative attention to effectively fuse the enhanced frequency-aware features, spatial features, and earlier-layer features.
The RMP module captures long-term spatiotemporal dependencies by storing and retrieving historical frame features and masks from a memory bank. It consists of two submodules: TAM (Temporal Alignment Module), which generates temporally-aware features, and MAM (Mask Affinity Module), which aligns the current prediction with historical context through cross-attention.
Extensive experiments were conducted comparing the proposed method with ZoomNext, SLTnet, AutoSAM, WeakPolyp, PNS+, MAST, VPSAM, and SALI on the SUN-SEG and CVC-Clinic-612 datasets. Additionally, performance analysis was carried out on SUN-SEG with different frame rates. Visualizations were provided for the SUN-SEG dataset, and ablation studies were conducted on the HFSI, RMP modules, and submodules of HFSI.

### Strengths
One of the main innovations of this paper is the introduction of frequency features into the field of video polyp segmentation (VPS). Traditional image segmentation methods typically focus on extracting spatial features, often neglecting the potential of frequency-domain information. The authors propose the Hierarchical Frequency-Spatial Interaction (HFSI) module, which effectively integrates frequency features with spatial features. This not only improves segmentation accuracy in low-contrast regions but also addresses the common issue of shape collapse in complex backgrounds. The method innovatively combines global frequency information and local spatial details, enhancing the model's robustness in dynamic scenarios, particularly when polyps are visually similar to surrounding tissues, thus improving target localization accuracy. Additionally, the paper introduces the RMP module, which uses a memory mechanism and mask-guided propagation to effectively capture long-range spatiotemporal dependencies, solving the common problem of episodic amnesia in video sequences. This demonstrates the model's strong innovation.
The paper thoroughly validates the proposed method through comprehensive experimental design. First, the authors conduct a quantitative comparison with current state-of-the-art methods on datasets like SUN-SEG and CVC-612, showcasing the superior performance of HFSTI-Net in terms of accuracy and robustness. Moreover, the authors perform a visual analysis to demonstrate the differences in performance across various methods in complex scenarios, such as low-quality frames and background interference, followed by a performance analysis to further confirm the stability and accuracy of the proposed method. Additionally, ablation studies on the HFSI and RMP modules, as well as the submodules of HFSI, are conducted on SUN-SEG, with both quantitative and visual analysis. This indicates the model's strong reliability.
The paper also offers a candid discussion of the model's limitations, particularly in handling certain types of polyps, such as those with low contrast or drastic shape variations, where segmentation performance is somewhat limited. Furthermore, the paper suggests the potential application of this model in other medical imaging fields.

### Weaknesses
The images of the modules in this paper are not sufficiently complete and clear, and some parts are prone to misunderstandings. For example, in Figure 2, the segmentation results are denoted as Pt in the image, where t is a subscript, while in the paper, “Pt−1 are passed into the RMP module”on line 200 uses a superscript. Additionally, in line 203, P = {Pi}4i=1 is used to represent the internal components of each P, This easily causes Pi to be mistaken for Pt.In Figure 3, for (a), the layer normalization step applied to the input is not shown in the image. For (b), the paper on line 256 states, “the input feature X is first passed through a 1 × 1 convolution to encode positional information,”but the corresponding operation is not depicted in the image. For (c), after a series of operations on the input, it is passed into both the frequency and spatial branches, but this step is also not shown in the image.
Although the episodic amnesia issue is one of the core problems addressed in this study, the paper's visualizations only show two frames from a single video, which may not fully demonstrate the effectiveness of the solution. Displaying multi-frame video sequences would better validate the model's ability to capture long-term temporal dependencies and show how the RMP module alleviates episodic amnesia. A comparison with other methods would more clearly highlight the model's advantage in solving this issue.

### Questions
In Figure 2, does Memory = N refer to the number of historical frames stored in the memory? If so, what value is N set to? Was an ablation study conducted on N to determine the most suitable value?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes HFSTI-Net, a novel hierarchical model that integrates frequency, spatial, and temporal cues through two key components, i.e., Hierarchical Frequency-Spatial Interaction (HFSI) module and Recurrent Mask-Guided Propagation (RMP) module. Extensive experiments demonstrate the effectiveness of each proposed module on two polyp segmentation datasets.

### Strengths
1. Despite the model's hierarchical complexity, HFSTI-Net achieves >30 FPS on a single GPU, making it viable for clinical and real-time processing.

2. Detailed ablation and component studies validate the necessity of each sub-module (e.g. FFB, SRB, IFB).

### Weaknesses
1. The authors proposed the Recurrent Mask-Guided Propagation Module to ensure long-term consistency of the segmentation. However, the entire input sequence only contains two frames (L376-377), making the claim less convincing. 

2. The number of HFSI layers is not fully studied in the experiment.

### Questions
1. In Table 3, it is not very clear why increasing the number of frames would lead to the increase of parameters. 

2. Computing feature interaction in the frequency domain is not very convincing, the authors are encouraged to show that using FFT/IFFT is better than other approaches (e.g. simple linear transformation).

3. Can you explain how query/key/value is obtained via FFT? (L237-238)

### Soundness
3

### Presentation
2

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
This work presents HFSTI-Net, a framework for Video Polyp Segmentation (VPS). The paper aims to solve two specific  problems: "shape collapse" (poor boundary integrity) and "episodic amnesia" (temporal instability). To this end, the authors propose a dual-path architecture. The first component is the Hierarchical Frequency-spatial Interaction (HFSI) module, which attempts to improve spatial localization by fusing frequency-domain and spatial-domain features. The second component is the Recurrent Mask-guided Propagation (RMP) module, which is intended to enforce temporal consistency by propagating information via a memory bank.
From a completeness perspective, the paper presents a full pipeline, including the model architecture, training details, and extensive experiments on the SUN-SEG and CVC-612 benchmarks. The reported results suggest that this approach achieves competitive performance against current state-of-the-art methods.

### Strengths
1.The paper identifies and articulates two significant and practical challenges in clinical VPS (shape collapse and episodic amnesia) as the core motivation for the work.
2.Based on the provided tables, the method achieves strong performance, reportedly outperforming existing state-of-the-art methods on the SUN-SEG and CVC-612 datasets across standard metrics.
3.Based on the provided tables, the method achieves strong performance, reportedly outperforming existing state-of-the-art methods on the SUN-SEG and CVC-612 datasets across standard metrics.

### Weaknesses
1.On the Novelty of the HFSI Module: The core concept of the HFSI module, an "interwoven dual-path design" for frequency and spatial features, appears not to be entirely new. The innovation seems to rest on the unique Interwoven Fusion Block (IFB), which is primarily realized by the cross-concatenation of convolutional features and frequency-domain features and fused with a gating network. This concept has been proposed and implemented in considerable prior work. As such, the innovation is incremental rather than disruptive. The authors must define their claim to novelty more precisely and contrast it with a broader range of related work.
2.On the Rationale of the Interwoven Fusion Block (IFB) Design: In Figure 3(c), the IFB design is perplexing. Frequency information (from an initial FFT) is concatenated with spatial information, and subsequently, another FFT operation is applied. The rationale for applying an FFT to features that are already (at least partially) in the frequency domain is perplexing and counter-intuitive. If this "FFT-on-FFT" operation is a deliberate and innovative component of the design, it requires substantial justification. The authors must provide a detailed motivation and theoretical principle for this specific operation. However, a thorough search of both the main body and the appendix reveals no such explanation , leaving the reader confused regarding the block's fundamental working principles.
3.On the Clarity of the RMP Module: The paper claims the RMP module is "mask-guided" and that the mask $P^{t-1}$ is passed into it. However, Figures 2 and 4, along with the provided formulas, fail to show how the mask $P^{t-1}$ is stored into the Memory (e.g., via a temporal FIFO). The paper also lacks a clear description of the "Retrieve" step. This ambiguity in the RMP's core mechanics is likely to cause confusion for the reader.

### Questions
1.The RMP module's dynamic memory links temporal information across long sequences, but this introduces a significant potential risk of error propagation. If the model generates an incorrect prediction (a "bad case") and stores the associated features and mask, this error could pollute subsequent frames and lead to cascading failures. Have the authors conducted specific experimental analyses to investigate this limitation?
2.The architecture of the "Mask Encoder" shown in Figure 2 is not detailed. Could the authors please clarify its design, specifically how it encodes the 2D mask into a feature representation, and how the resulting "PEmbeddings" are subsequently fused into the model?
3.We request the authors to comment on the transferability of the RMP module's memory design. How effective is this mechanism expected to be when generalized to different datasets, which may feature distinct temporal dynamics or target characteristics?

### Soundness
3

### Presentation
3

### Contribution
2
