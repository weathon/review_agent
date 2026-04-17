# Mitigating Exponential Mixed Frequency Growth through Frequency Selection

- Decision: Reject
- Scores: 6, 2, 2, 4

## Abstract
Quantum machine learning research has expanded rapidly due to potential computational advantages over classical methods. Angle encoding has emerged as a popular choice as feature map (FM) for embedding classical data into quantum models due to its simplicity and natural generation of truncated Fourier series, providing universal function approximation capabilities. Efficient FMs within quantum circuits can exploit exponential scaling of Fourier frequencies, with multi-dimensional inputs introducing additional exponential growth through mixed-frequency terms.
Despite this promising expressive capability, practical implementation faces significant challenges. Through controlled experiments with white-box target functions, we demonstrate that training failures can occur even when all relevant frequencies are theoretically accessible. We illustrate how two primary known causes lead to unsuccessful optimization: insufficient trainable parameters relative to the model's frequency content, and limitations imposed by the ansatz's dynamic lie algebra dimension, but also uncover an additional parameter burden: the necessity of controlling non-unique frequencies within the model. To address this, we propose near-zero weight initialization to suppress unnecessary duplicate frequencies. For target functions with a priori frequency knowledge, we introduce frequency selection as a practical solution that reduces parameter requirements and mitigates the exponential growth that would otherwise render problems intractable due to parameter insufficiency. Our frequency selection approach achieved near-optimal performance (median $R^2 \approx 0.95$) with 78\% of the parameters needed by the best standard approach in 10 randomly chosen target functions.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper considers quantum machine learning and focuses on the angle encoding problem. Due to practical implementation challenges, such as insufficient trainable parameters relative to the model’s frequency content and the ansatz’s dynamic Lie algebra dimension, the paper proposes near-zero weight initialization to suppress unnecessary duplicate frequencies, along with frequency selection as a practical solution. The experimental results demonstrate the effectiveness of the proposed approach.

### Strengths
This paper is well-written. It proposes that near-zero weight initialization can address the model’s frequency content and limitations. Furthermore, the paper introduces a frequency selection method to provide a practical solution.

### Weaknesses
The experimental results effectively illustrate the contributions, but I still have doubts about the practical applicability. How to extend the method to 2D should be clearly illustrated. Furthermore, regarding the dataset, a broader variety of data should be included in the experiments.

### Questions
see the weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper "Mitigating Exponential Mixed Frequency Growth", presents an analysis of parameter sufficiency conditions. The paper shows that duplicate frequencies are necessary for function approximation with quantum circuits. Furthermore the paper studies initialization, as well as parameter efficient training for quantum machine learning tasks.

The paper establishes minimum parameter requirements via the Frequency coverage condition,
Frequency control condition, and Optimization landscape conditions.

The paper presents quantum computer simulations, as experiments to back up its findings.

### Strengths
- Quantum machine learning can potentially unlock progress leaps in the future.

### Weaknesses
- The experimental section is does not reference or is integrated into related work.
- I am not sure it the approach to quantum machine learning presented here allows for non-linearity, which is common in the deep learning world. 
- Section 5.1 (experimental setup) does not discuss the experimental setup at all. The supplementary finally states the we are looking at simulated quantum computing results from a consumer notebook. I am uncertain. How realistic are these simulations? Will they be useful in the future?

### Questions
- Is there a way to add non-linearity to quantum circuits?
- Is it possible to run some of these experiments on publicly available machines, i.e. the free tier of IBM-Q?
- According to the supplementary the optimization runs in Jax, to the best of my knowledge Jax would not run on a quantum computer. Would it even be possible to run this code on an actual machine instead of a simulator? How far is this work away from running on an actual device?

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
2

### Summary
This paper addresses the exponential growth of parameters in angle-encoded quantum models, which is exacerbated by mixed-frequency terms in high-dimensional data. Through white-box experiments, the authors identify a novel "additional parameter burden" stemming from the need to control a large number of non-unique (degenerate) frequencies. To mitigate this, they first propose a near-zero weight initialization heuristic, which proves effective for 1D problems by suppressing these unnecessary frequencies. Second, for cases where the target spectrum is known a priori, they introduce a "frequency selection" algorithm to build a sparse model spectrum that matches the target. This selection method is shown to achieve near-optimal performance in 2D examples while using significantly fewer parameters than standard dense-spectrum model

### Strengths
The paper's main strength is its focus on a significant and unresolved problem: the exponential parameter scaling in high-dimensional quantum models, which is a major bottleneck for the field. Its originality lies  in attempting to identify a specific experimental failure (the "serial unary" case) and proposing a new "additional parameter burden" from non-unique frequencies, which seeks to challenge existing theoretical conditions. The paper's clearest contribution is the experimental demonstration of the "frequency selection" algorithm, which shows that a model with a sparse spectrum can achieve high $R^2$ scores on 2D tasks with fewer parameters than dense models. The work is structured with reasonable clarity, providing a review of the theoretical background on Fourier analysis and DLA, and its potential for reproducibility is bolstered by an extensive appendix with detailed circuit diagrams. This "frequency selection" component, which demonstrates a practical engineering approach for sparse-spectrum targets, stands as the paper's most tangible result, even as its broader theoretical claims remain unsubstantiated

### Weaknesses
The paper's primary weakness is that its central theoretical claim—the "additional parameter burden" from non-unique frequencies—appears to be founded on a single, wrongly interpreted experiment. The "serial unary" model's failure is presented as evidence for a novel phenomenon, but this result can be explained by a direct violation of the known "frequency control condition" $(p\geq|Ω|)$, which the authors themselves cite. This experiment's 45 parameters are on a single qubit (Figure 10 in App. C) and are thus not linearly independent, a specific limitation the paper explicitly describes [lines 177-178] but fails to apply to its own key experiment. 

This misinterpretation of a known failure mode as a new theoretical discovery significantly weakens the motivation for the proposed "near-zero weight initialization" heuristic. Furthermore, the paper's practical contributions are severely constrained by its own admissions: the "near-zero init" is a 1D-only solution that "could not be replicated for higher dimensional target functions". 

Additionally the "frequency selection" algorithm is practically limited to toy problems, as it "exploits complete domain knowledge" of the target spectrum a priori.

### Questions
Your paper's most novel theoretical claim, the "additional parameter burden" from non-unique frequencies , rests almost entirely on the "unexpected case" of the 1D "serial unary" model's failure. You state this model, with 45 parameters, satisfies the known requirements $p \geq |Ω|$ (45 > 25) and $p \geq dim(g)$ (45 > 3). However, this is a single-qubit architecture (Figure 10). As you correctly state in Section 3, parameters added serially on the same qubit "fail to generate additional linearly independent Fourier coefficients".   

Question: Can you provide a rigorous calculation or a detailed argument that the 45 parameters in the "serial unary" model are, in fact, linearly independent and that $p_{ind} \geq 25$? If they are not, isn't this model's failure an expected result of violating the known "frequency control condition", which would undermine the primary evidence for your "non-unique frequency" hypothesis?   

The "near-zero weight initialization" heuristic is presented as a solution to the non-unique frequency burden, but you (commendably) note in Section 6 that these successful 1D results "could not be replicated for higher dimensional target functions". This lack of generalizability to 2D is a critical limitation for a paper addressing high-dimensional scaling.   

Question: Can you provide any insight into why this heuristic fails to generalize? Does its failure in 2D suggest it is not addressing the hypothesized root cause (the non-unique frequency burden), but is perhaps an artifact of 1-qubit optimization dynamics?

The "frequency selection" algorithm is experimentally robust but is prefaced on "complete domain knowledge"  of the target function's frequency spectrum. You suggest this could be obtained via "classical Fourier analysis".   

Question: For high-dimensional datasets (the paper's core motivation), performing a classical multi-dimensional Fourier transform is itself an exponentially hard, and thus intractable, problem. Could you elaborate on a realistic path to applying this method to high-dimensional, real-world data where the target spectrum is unknown and classically intractable to obtain?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
In the manuscript titled "Mitigating Exponential Mixed Frequency Growth Through Frequency Selection," the authors propose a practical approach to address the problem of exponential frequency growth in quantum machine learning. Their main contribution lies in a frequency selection strategy that significantly reduces the number of required parameters. The experimental design is sound, employing white-box target functions (Fourier series with known frequencies) for systematic evaluation, which effectively validates the efficacy of the proposed method. The frequency selection approach achieves comparable performance using only 78% of the parameters required by the best standard method. Finally, I have the following concerns and suggestions:

1. Section 5.3, "Near-Zero Weight Initialization," in the paper states that near-zero weight initialization works well in the 1D case but performs poorly in the 2D case. The authors do not explain why this method deteriorates in higher dimensions—could a more in-depth analysis be provided?   
2. Section 3.1, "Multi-Dimensional Extensions and Mixed Frequencies," states that "every frequency in the spectrum requires individual coefficient control..." but fails to explain why, in quantum circuits, non-unique frequencies necessitate additional trainable parameters. In a classical Fourier series, each frequency component is independent—so why, in a quantum model, must extra parameters be allocated specifically to control non-unique (i.e., duplicate) frequencies? The authors are encouraged to supplement the manuscript with a theoretical analysis of the relationship between frequencies and parameters in quantum circuits, clarifying why controlling non-unique frequencies demands additional parameters.
3. The practicality of the method presented in the paper relies on prior knowledge of the target function's frequencies, which is often unrealistic in real-world scenarios. The paper does not discuss how to determine the frequencies of the target function in the absence of such prior knowledge. Could the authors provide clarification or suggestions on this aspect?
4. It is recommended to incorporate a preprocessing step—such as classical Fourier analysis (e.g., the method of Wiedmann et al., 2024, mentioned in the paper)—to extract frequencies, and to compare the performance gap and computational overhead between scenarios with and without prior frequency knowledge. This would help establish a logically complete picture of the method’s applicability.
5. In Section 4.3, "Frequency Selection," the statement "By selecting prefactors that deviate from the standard ternary..." lacks a clear description of the specific strategy for choosing these prefactors, providing only a single example with prefactors 3 and 9. The authors are encouraged to elaborate on this selection methodology.
In conclusion, before this paper is accepted for publication, the above-mentioned issues need to be addressed and revised.

### Strengths
The authors propose a practical approach to address the problem of exponential frequency growth in quantum machine learning. Their main contribution lies in a frequency selection strategy that significantly reduces the number of required parameters. The experimental design is sound, employing white-box target functions (Fourier series with known frequencies) for systematic evaluation, which effectively validates the efficacy of the proposed method. The frequency selection approach achieves comparable performance using only 78% of the parameters required by the best standard method.

### Weaknesses
1. Section 5.3, "Near-Zero Weight Initialization," in the paper states that near-zero weight initialization works well in the 1D case but performs poorly in the 2D case. The authors do not explain why this method deteriorates in higher dimensions—could a more in-depth analysis be provided?   
2. Section 3.1, "Multi-Dimensional Extensions and Mixed Frequencies," states that "every frequency in the spectrum requires individual coefficient control..." but fails to explain why, in quantum circuits, non-unique frequencies necessitate additional trainable parameters. In a classical Fourier series, each frequency component is independent—so why, in a quantum model, must extra parameters be allocated specifically to control non-unique (i.e., duplicate) frequencies? The authors are encouraged to supplement the manuscript with a theoretical analysis of the relationship between frequencies and parameters in quantum circuits, clarifying why controlling non-unique frequencies demands additional parameters.
3. The practicality of the method presented in the paper relies on prior knowledge of the target function's frequencies, which is often unrealistic in real-world scenarios. The paper does not discuss how to determine the frequencies of the target function in the absence of such prior knowledge. Could the authors provide clarification or suggestions on this aspect?
4. It is recommended to incorporate a preprocessing step—such as classical Fourier analysis (e.g., the method of Wiedmann et al., 2024, mentioned in the paper)—to extract frequencies, and to compare the performance gap and computational overhead between scenarios with and without prior frequency knowledge. This would help establish a logically complete picture of the method’s applicability.
5. In Section 4.3, "Frequency Selection," the statement "By selecting prefactors that deviate from the standard ternary..." lacks a clear description of the specific strategy for choosing these prefactors, providing only a single example with prefactors 3 and 9. The authors are encouraged to elaborate on this selection methodology.

### Questions
1. Section 5.3, "Near-Zero Weight Initialization," in the paper states that near-zero weight initialization works well in the 1D case but performs poorly in the 2D case. The authors do not explain why this method deteriorates in higher dimensions—could a more in-depth analysis be provided?   
2. Section 3.1, "Multi-Dimensional Extensions and Mixed Frequencies," states that "every frequency in the spectrum requires individual coefficient control..." but fails to explain why, in quantum circuits, non-unique frequencies necessitate additional trainable parameters. In a classical Fourier series, each frequency component is independent—so why, in a quantum model, must extra parameters be allocated specifically to control non-unique (i.e., duplicate) frequencies? The authors are encouraged to supplement the manuscript with a theoretical analysis of the relationship between frequencies and parameters in quantum circuits, clarifying why controlling non-unique frequencies demands additional parameters.
3. The practicality of the method presented in the paper relies on prior knowledge of the target function's frequencies, which is often unrealistic in real-world scenarios. The paper does not discuss how to determine the frequencies of the target function in the absence of such prior knowledge. Could the authors provide clarification or suggestions on this aspect?
4. It is recommended to incorporate a preprocessing step—such as classical Fourier analysis (e.g., the method of Wiedmann et al., 2024, mentioned in the paper)—to extract frequencies, and to compare the performance gap and computational overhead between scenarios with and without prior frequency knowledge. This would help establish a logically complete picture of the method’s applicability.
5. In Section 4.3, "Frequency Selection," the statement "By selecting prefactors that deviate from the standard ternary..." lacks a clear description of the specific strategy for choosing these prefactors, providing only a single example with prefactors 3 and 9. The authors are encouraged to elaborate on this selection methodology.

### Soundness
2

### Presentation
2

### Contribution
2
