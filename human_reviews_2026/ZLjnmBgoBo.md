# Time-series based quantum state discrimination

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 4

## Abstract
Measurement errors in quantum computers are very detrimental to quantum computations. The ability to efficiently and accurately readout quantum states is crucial for quantum error correction schemes and quantum algorithms. Readout fidelity is typically limited by a poor signal-to-noise (SNR) ratio between the quantum states we intend to classify, as well as energy relaxation (e.g., T1 decay) from an excited state to a lower state during readout. Superconducting quantum bits (qubits), one of the leading candidates for scalable quantum computing hardware, are particularly limited by energy relaxation due to their relatively short coherence times. While most approaches for classifying the results of readout on superconducting qubits typically utilize clustering algorithms (e.g., a Gaussian mixture model) on integrated readout signals, these cannot distinguish between a quantum bit that was in the ground state prior to measurement from a qubit that decays to the ground state during measurement. For this reason, we instead propose using machine learning (ML) on the raw (non-integrated) analog signal and classification models on the full time series data (i.e., the trajectory). We observe that time series classification methods, such as our chosen long short-term memory (LSTM) model, in combination with filtering and feature engineering techniques, consistently outperform clustering models.  In particular, we find that the largest improvements come from reclassifying points in the boundary regions between neighboring clusters. These boundary points correspond to measurement records that deviate from the typical cluster, likely due to transient or noisy features in the signal that are not captured when the data is integrated. By retaining temporal information, sequence-aware models such as LSTMs can better discriminate these trajectories, whereas clustering methods based on integrated values are more prone to misclassifications.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper utilizes LSTM networks to read out qutrit states from a superconducting transmon device. In typical setups, Gaussian mixture models, or GMMs, are typically employed for the task of identifying the states of the qutrits after integration is done on the raw signals. In doing so, critical information about the measurements is lost, and it can become difficult to identify states along what would be described as the decision boundary in classical machine learning. By using an LSTM, the authors argue that this important temporal information is captured, improving the classification accuracy of the states and therefore, the effective fidelity of the device. To show this, the authors perform a comparison between the GMM and the LSTM on a standard driving task by comparing the final accuracy of the two models. They find that the LSTM outperforms the GMM.

### Strengths
* The problem of error correction is, as clearly stated by the authors, a pressing issue in the community. Further, it appears that the LSTM approach improves the effective fidelity of these devices.
* The experiments were conducted on real quantum devices, removing issues with simulation parameters and noise modeling. 
* The concepts are made clear to a broader audience, and in general, the paper is written very well.

### Weaknesses
While overall the paper is quite sound and of interest, there were some parts that lacked clarity and accuracy.

* Why do the authors not show the direct classification results? Showing only where one model gets it right but the other wrong does little to highlight the performance of the models. Further, without showing where both models go wrong, it isn't clear whether this comes down to architecture or other factors.  I see that this is included as fidelity scores later in the paper, but perhaps the structure could be cleaned up to show the raw results, followed by how the LSTM beats the GMM.
* The areas where the LSTM beats the GMM are the same as where the GMM beats the LSTM. In the caption of Figure 5, only the former is highlighted as a feature. True, there are more points captured by the LSTM, but I think this needs to be elaborated on. In general, how much of that area is correctly or incorrectly classified? Judging by the fidelity results shown later, it is quite a small amount. That isn't to say it's not impactful, but I think this needs to be made clearer. 
* It is not clear that the structure processing of time in the LSTM is the reason for the improvements in the LSTM model. To make this claim, the authors would likely need to perform ablation studies with different context lengths.
* Further, would a dense network with all measurements through time also perform well? For small trajectories, this would be feasible and fast. In general, a broader comparison of machine learning architectures would be helpful rather than jumping straight from GMM to LSTM. This would also align better with the goal of integrating these models onto devices.
* I'm curious about the use of the term fidelity in describing the performance of the classification networks. This is for two reasons. One is purely from an understanding perspective. In the broader ML community, discussing fidelity in the context of classification performance will likely confuse people. The second is more technical. Should this value be referred to as a fidelity? I would consider fidelity to be how well my measurement aligned with my inputs. If I use a classification algorithm to classify my measurements, their performance isn't so much fidelity as simple algorithmic accuracy. Consider the case where a machine has perfect fidelity, but my algorithm is trained to also produce the wrong state. Does this system display a fidelity of 0 or 1? I would be interested in hearing the author's thoughts on this.
* Information about model training is very limited. It would be nice to see the training curves, sizes of networks tested, and their performance, particularly under the motivation of embedding them onto a device.

### Questions
* Can the authors quantify the preparation errors mentioned? The fidelities of the devices are already reasonably high, although it is stated that these preparation errors are small compared with noise, it would be good to have a general idea of their size.
* How large were the biggest models used?

### Soundness
2

### Presentation
2

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
The paper is concerned with quantum state discrimination: the problem of accurately reading out quantum states. Typical approaches for this problem employ Gaussian mixture model-based clustering algorithms. Here, the authors suggest to use LSTMs in combination with filtering and feature engineering approaches for quantum state discrimination based on the time series of the readout signal.

### Strengths
Currently used methods do not employ temporal information, so employing sequence models for incorporating time series data is well-motivated. Thus, the problem is very relevant for quantum computing, as quantum error correction schemes require high quantum state readout fidelity.

The experiments are based on real data and appear to be pratictically relevant. The experimental results (Table 2) indicate a consistent improvement of LSTM-based methods over the GMM-baseline method.

### Weaknesses
My main concern with this paper is that, from a machine learning perspective, the innovation is very limited. LSTMs are standard models for sequence modelling and applying them to time-series classification is well-established. Thus, there do not seem to be any insights for a broader ML audience. The main challenge and contribution of the paper appears to be in the data pre-processing step. Then, any time-series classification method could be applied. Overall, the paper is written in a way that is much more suitable to a quantum computing venue, with the main emphasis put on physical details and on applying off-the-shelf ML methods to quantum state discrimination. 

In addition, I also have several concerns regarding presentation and experiments.

Missing experimental details: the paper does not provide sufficient details on the experiments to make them reproducible. For example, the number of layers / hidden nodes in the LSTMs are not specified. The chosen bin size for the binning procedure is not specified. The paper does not appear to use a validation set for selecting these hyper-parameters. Does Table 2 report classification accuracy on the test set?

Methodology: the paper does not clearly explain how the LSTM is trained. The baseline method (GMM clustering) is an unsupervised learning method. In contrast, LSTM-based classification requires labeled training data and it remains unclear where this data is obtained from. 

Computing times: The paper states that the LSTMs make the approach feasible for deployment in real-time readout hardware (line 348), but does not report the compute times for the different methods. To support such a claim, also compute times need to be reported. 

Role of LSTMs: with the key step being the data preprocessing, the paper does not sufficiently examine the role of LSTMs for the improved accuracies. Does the improved accuracy stem from the improved data preprocessing step? Or does it actually stem from using LSTMs? Insight on this could be gained by including other time-series classification models in the comparison, for example, standard RNNs or state-of-the art time-series classifiers such as 
ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels by Angus Dempster, François Petitjean, Geoffrey I. Webb. 

Limited baselines: Since the introduction of GMM methods for this problem in 2014, there have been several other works that employ deep learning-based classifiers for quantum state discrimination (such as B. Lienhard et al. 2022 cited in the paper). It would be important to compare also to such methods. 

Wrong key reference: the paper attributes LSTMs to Bengio et al. 2000 (see line 202). However, the original reference is the very well-known paper by Hochreiter and Schmidhuber (1997). 

Activation function: the method uses a sigmoid activation function for multi-class classification, which is very uncommon. Rather, softmax activations are typically used, since they restrict the outputs to sum to one. How do you handle cases where you get very low/high probabilities for all cases?

### Questions
- How did you select the number of layers / hidden nodes in the LSTMs?
- How did you select the bin size for the binning procedure? 
- Does Table 2 report classification accuracy on the test set?
- Can you be more specific on how the labels for training the LSTMs were generated?  
- Can you report compute times for your different methods? 
- Can you provide further insights on the importance of using LSTMs vs. the improved data pre-processing step? 
- Using a sigmoid activation function for a multi-class classification problem, how do you handle cases where you get very low/high probabilities for all cases?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
In this paper, a novel approach to efficiently and accurately readout quantum states is presented. This task is curcial for quantum error correction.

The authors proposed to use machine learning for this task. In particular a LSTM model, in bomcination with filtering and feature engineering techniques is presented.

The proposed approach outperform clustering models and is better that the proposed time-series baseline (GMM).

The application of an LSTM-based classifier to bandpass-filtered readout traces is not well aligned with ICLR’s core focus on methodological advances in machine learning.

### Strengths
- Readout fidelity is a well-known bottleneck in superconducting qubit systems
- Time-series framing of qubit readout is promising
- The proposed approach is novel
- Quality of the presentation is high

### Weaknesses
- The reported improvement is small, even if significant
- Only GMM is used for comparison
- The topic is not closely aligned with ICLR’s core areas of interest

### Questions
Why do you compare only against GMMs? Are there other time-series ML models that could be used for this purpose?

### Soundness
3

### Presentation
3

### Contribution
2
