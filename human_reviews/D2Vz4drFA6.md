# HyperChr: Quantization of Heterogeneously Distributed Matrices through Distribution-Aware Subspace Partitioning

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 3

## Abstract
Matrix quantization is crucial for reducing the memory footprint of matrices across various applications, including large-scale machine learning models and data compression. We have observed that matrices in different application domains exhibit heterogeneity in the distribution across columns. Leveraging this characteristic, we introduce \textit{HyperChr}, a novel matrix quantization algorithm tailored for heterogeneous data distributions prevalent across different matrix columns. Unlike traditional quantization methods, \textit{HyperChr} capitalizes on the heterogeneous distribution characteristics of each column to optimally partition high-dimensional subspaces and perform compression within each subspace. This technique enhances the compression effectiveness by grouping vectors with similar distribution ranges, enabling more precise quantization. Moreover, \textit{HyperChr} dynamically adjusts the number of centroids in each subspace based on the specific data distribution traits, optimizing both storage efficiency and data fidelity.

We evaluate \textit{HyperChr}'s performance on diverse datasets, demonstrating its superiority in reducing quantization errors compared to existing methods. Our results show that \textit{HyperChr} exhibits significant improvements at lower compression ratios ($\theta = 2-8$), reducing MAE by an average of 55.3\% and MSE by 75.3\% compared to PQ. However, at higher compression ratios ($\theta = 10-16$), the improvements are more moderate, with an average reduction of 14.9\% in MAE and 25.9\% in MSE compared to PQ. In addition, our algorithm reduces the average dequantization time by 62.9\%, which is crucial for large language model inference.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
In the paper, the authors find that the matrix data typically studied  tends to exhibit similar distributions within individual columns, but varies  across different columns. Based on this observation, they propose a column-wise, non-uniform quantization method that surpasses the widely-used PQ method.

### Strengths
Given the variability in distribution across different columns, the study incorporates non-uniform quantization into matrix quantization, resulting in lower quantization errors compared to the PQ method.

### Weaknesses
The presentation of the paper is inadequate, and it is not yet in a suitable state for publication.

1) The proposed method lacks clear description in section 3.1. Specifically, regarding matrix partition, what criteria are used to group the columns? Are they grouped randomly? Additionally, in the process of subspace generation, how is the subspace actually created? Moreover, how can we ensure that columns belonging to different subgroups but having the same indices possess similar distributions?

2) Many arguments presented in the study lack  theoretical support. For example, the assumption that the samples in each column should be drawn from the same or similar distribution solely based on the observation that half of the samples exhibit similar density maps to the whole population, while this is statistically unfounded.

### Questions
1) In my understanding, the row size $n$ of the matrix (mentioned on Line 131) represents the number of data points, while the column size $d$ denotes the data dimension. The authors should clarify the meanings of these two concepts. In this context, it is plausible to assume that the data values within the same column are drawn from the same distribution, as in data feature representation, each data dimension (column) typically corresponds to a specific data attribute. In this view, the authors should describe their quantization method using data attributes rather than matrix columns. 

2) In Definition 1, the paper establishes the objective of minimizing the memory-regularized Mean Squared Error (MSE). However, this particular goal is not explored further in the subsequent theoretical analysis. Instead, the theoretical focus shifts to a comparison with the Product Quantization (PQ) method.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper proposes a non-uniform quantization method to quantize the matrix data which have the data values within a same column drawn from a same distribution, but have the distributions varying across different columns. Empirically, the proposed method achieves obvious performance improvement compared to existing methods.

### Strengths
The performance improvement reported by the authors is remarkable.

### Weaknesses
The paper falls short in offering a comprehensive description of its method. The procedures pertaining to matrix partition and subspace generation remain ambiguous and devoid of thorough analysis. Furthermore, the regularized MSE, as outlined in Definition 1, has not been analyzed for the proposed method. In summary, the method requires substantial enhancements both in terms of its methodological description and theoretical foundation.

### Questions
1) Matrix partition (Line 207)：How are the columns of a matrix grouped, and what criteria are used for this grouping? 


2) Subspace generation (Line 213):  What is the rationale for extracting and combining the columns from different subgroups (with the same position coordinates)?

3) Synthetic dataset (Line 359): How to ensure that the distributions between columns differ when synthesizing data?

4) It appears unfair to  compare the proposed method with PQ, given that the former employs non-uniform quantization while the latter uses uniform quantization. The authors should consider examining related non-uniform quantization methods for a more comprehensive comparison.

5)  For the three contributions  summarized ( in Line 116), the first contribution concerning column distributions is empirical and superficial, lack of in-depth analysis;  and  the third contribution, which mentions  "dynamically adjusting the **intervals**" , is unclear and ambiguous.


Minor comments:

1) Line: 209:  what does the “exponential explosion” mean?

2) Line 310: what does “the **effective** variance reduction parameter” mean?

3) Line 517: As can be seen from the data in the figure, DQT remains relatively stable while QT decreases.

4) Line 522: It should be the stability of DQT.

5) line 523: The errors MAE and MSE increase (not decrease).

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposes a new method for matrix quantization, which accounts for the heterogeneous distribution of matrix columns. Compared with the PQ algorithm for matrix quantization, the new algorithm chooses different quantization codebooks for different (sets of) columns, which can be optimized for different distributions. Comparison in quantization error and time complexity against PQ is analyzed, and experiments on various datasets show the proposed algorithm outperforms baseline PQ-family algorithms in accuracy and efficiency.

### Strengths
- The presentation of the paper is generally clear.
- Experiments and comparisons with baseline algorithms are clearly presented.

### Weaknesses
- The paper proposes to exploit heterogeneous column distributions, but fails to provide sufficient insights. Certain key questions such as how to appropriately select columns for joint quantization are eluded. This makes the overall contribution rather limited, and is the main weakness of this paper.
- The introduction of related work, such as the PQ algorithms, is missing. The current Problem Setting section feels empty.

### Questions
- The current way of grouping columns is arbitrary, i.e. just based on index, but if the key idea of this paper is leveraging distribution heterogeneity, how can this grouping be improved to make the algorithm more efficient?
- When using different quantizations for different columns, there is extra overhead for storing and processing the codebooks. How does that factor in the comparison with basic PQ algorithms?
- In Figure 1, what are the dashed-line distributions supposed to show? I fail to see how the distribution of half of the data randomly sampled conveys any new information.
- Where does the name HyperChr come from? While I do believe naming is immaterial to an algorithm, this name covers up the close relationship with PQ algorithm and can be misleading in terms of its novelty.
- Notations introduced in Section 2 are almost never used again. Are these really necessary?
- In Introductions is the paragraph on lines 94-96 redundant with its next paragraph?

### Soundness
2

### Presentation
3

### Contribution
2
