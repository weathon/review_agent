# WinT3R: Window-Based Streaming Reconstruction with Camera Token Pool

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
We present WinT3R, a feed-forward reconstruction model capable of online prediction of precise camera poses and high-quality point maps.
Previous methods suffer from a trade-off between reconstruction quality and real-time performance.
To address this, we first introduce a sliding window mechanism that ensures sufficient information exchange among frames within the window, thereby improving the quality of geometric predictions without introducing a large amount of extra computation.
In addition, we leverage a compact representation of cameras and maintain a global camera token pool, which enhances the reliability of camera pose estimation without sacrificing efficiency.
These designs enable WinT3R to achieve state-of-the-art performance in terms of online reconstruction quality, camera pose estimation, and reconstruction speed, as validated by extensive experiments on diverse datasets.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a feed-forward reconstruction model WinT3R, which can output precise camera poses and high-quality point maps.
The key design is a sliding window mechanism to ensure sufficient information exchange among adjacent frames, and a global camera token pool to enhance the reliability of camera pose estimation without sacrificing efficiency.
Experimental results show the effectiveness of the proposed method.

### Strengths
1. The presentation is clear and easy to follow.
2. The author conducted extensive experiments to support their statements. And results show that WinT3R outperforms other online baselines in some benchmarks.

### Weaknesses
1. The architecture is highly similar to CUT3R and VGGT. Actually, it seems like a combination of these two methods (state tokens-CUT3R, interaction within window-VGGT). And the training costs are high (compared with CUT3R's 8GPUs).
2. I want to know the relationship between the global and local tokens (image tokens and camera tokens) for each frame. From Sec.3.3, it seems that this work only supervise the point map in the local coordinate system and the relative poses between frames (following pi3). In this case, is it redundant to maintain both local and global tokens for each frame simultaneously?
3. With the window based design, I want to know the robustness of WinT3R when faced with unordered inputs. Besides, please use very long sequences (200 frames and more) to compare the reconstruction quality, as it is important in this online settings.

Please answer the above questions carefully and provide more thorough discussion and comparison.

### Questions
Please refer to the Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents WinT3R, a feed-forward model for online 3D reconstruction from streaming images. It aims to solve the critical trade-off between reconstruction quality and real-time performance. The method introduces two key contributions: an online sliding window mechanism, which allows image tokens of adjacent frames to interact directly, improving geometric quality , and a compact camera token pool, which maintains global information from all historical frames to enhance the reliability of camera pose estimation.  Experiments demonstrate its effectiveness.

### Strengths
1.	This paper presents a novel framework, WinT3R, for online 3D reconstruction, which effectively tackles the critical trade-off between reconstruction quality and real-time performance. The two core contributions, the sliding window mechanism and the camera token pool, are technically sound and directly address clear limitations in prior streaming methods. 
2.	Extensive experiments validate the effectiveness of the proposed method. 
3.	This paper is written and organized well.

### Weaknesses
1.	The paper's entire motivation is to close the quality gap between "suboptimal" online methods and high-quality offline methods (like the cited VGGT or FLARE). However, all quantitative tables (Tables 1-3) only compare against other online methods. Without including at least one offline advanced baseline, it is impossible to quantify how much of the quality gap has actually been closed, which is essential for contextualizing the paper's contribution. It will be more convincing to add these comparisons.
2.	The paper states the camera head queries the entire historical pool. This implies an attention mechanism with O(T^2) computational complexity, where T is the total number of frames. This quadratic complexity is not scalable, contradicts the real-time (17.2 FPS) claim for any reasonably long video, and is a fundamental design flaw for a streaming system. It will be interesting to discuss about it.
3.	The paper states it uses the final pose from the subsequent window for overlapping frames（L209）. However, the model's token update rule adds the tokens from the initial window to the pool(Eq.5). This raises a problem that the final chosen pose does not match the token stored in the global pool. This problem undermines the pool's global consistency.

### Questions
Refer to the Weakness.

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
4

### Summary
This paper proposes a new online streaming 3D reconstruction method called WinT3R that combines a sliding-window attention mechanism with a camera token pool memory to achieve real-time performance. It is an improvement based on the prior work CUT3R. The novelty mainly comes from the attention between image tokens within windows, and representing prior frames via compact tokens in a global camera token pool. The experimental results on standard benchmarks show strong improvements over existing online methods.

### Strengths
It provides a way of balancing the usually high-cost transformer-based 3D reconstruction model and reconstruction quality and consistency over long sequences. This enables the model to operate at real-time with streaming input. The sliding window attention effectively avoids the full casual attention cost. It maintains the simplicity of feedforward neural reconstruction models without heavy keypoint feature matching in traditional SLAM pipelines.

### Weaknesses
* It is unclear for long sequences how well the camera token pool can handle and preserve the global geometry for frames that are far apart. 
* For the efficiency and scalability claim, It lacks some experiments on demonstrating the ability of the model on long sequences and the corresponding runtime statistics. 
* The experiments on the selected scenes are mostly all indoor scenes (e.g. 7 Scenes, NRGBD). It lacks some experiments on more varying conditions such as outdoor scenes.

### Questions
It seems like the selection of the window's length and stride is important here. Different video sequences will typically have non-uniform motions at every point. Will this affect the reconstruction quality a lot?

### Soundness
2

### Presentation
4

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
This paper (“WinT3R”) presents a feed-forward architecture for predicting point maps from streaming RGB video. At a high level, it combines sliding-window attention with a global persistent state, conceptually related to CUT3R and the streaming variant of VGGT (StreamVGGT). The method uses overlapping windows to encourage direct interaction among neighboring frames while maintaining memory efficiency. In addition, a global camera token pool stores compact historical camera tokens to support future camera pose prediction. The overall idea is simple and well-motivated.

Overall, the design choices are reasonable and the empirical results are strong. My concerns are primarily about figure presentation and some missing ablations; details follow.

### Strengths
- The approach is simple, straightforward, and reasonable: combining sliding-window attention with a global state is a natural way to balance local interaction and long-range consistency.

- Experimental results indicate state-of-the-art reconstruction quality among online/streaming methods.

- The global camera token pool, while somewhat pragmatic, is an effective design specifically for camera pose estimation in a streaming setting.

### Weaknesses
- Result visualization could be improved. For example, in Figure 4 it is difficult to tell which columns are better without clearer legends or more distinct visual cues.

- Consider adding videos or point-cloud visualizations in the supplementary material to better convey qualitative improvements.

- The performance improvement over StreamVGGT is relatively modest, which could lessen the perceived contribution of this paper.

### Questions
- What are the training time and GPU resources used in the ablation study?

- Which model size (and window size) is used for those ablations?

- There is no ablation that removes the global persistent state entirely. How would performance change without it (e.g., sliding-window only, no global state)?

### Soundness
3

### Presentation
2

### Contribution
2
