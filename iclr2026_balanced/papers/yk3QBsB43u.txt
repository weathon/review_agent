# TACKLING THE NOISY ELEPHANT IN THE ROOM: LA- BEL NOISE-ROBUST OUT-OF-DISTRIBUTION DETEC- TION VIA LOSS CORRECTION AND LOW-RANK DE
## COMPOSITION


**Anonymous authors**
Paper under double-blind review


ABSTRACT


Robust out-of-distribution (OOD) detection is an indispensable component of
modern artificial intelligence (AI) systems, especially in safety-critical applications where models must identify inputs from unfamiliar classes not seen during
training. While OOD detection has been extensively studied in the machine learning literature—with both post hoc and training-based approaches—its effectiveness under noisy training labels remains underexplored. Recent studies suggest
that label noise can significantly degrade OOD performance, yet principled solutions to this issue are lacking. In this work, we demonstrate that directly combining
existing label noise-robust methods with OOD detection strategies is insufficient
to address this critical challenge. To overcome this, we propose a _robust_ OOD
detection framework designed to _cleanse_ feature embeddings, thereby mitigating
the adverse effects of noisy labels on OOD performance. Towards this, we introduce an end-to-end training strategy that integrates loss correction methods from
the noisy-label learning literature with low-rank and sparse decomposition techniques from signal processing. Building on this strategy, we derive a novel metric
that quantifies the “OOD-ness” content within training data, which in turn leads to
a label noise-robust OOD detection scoring technique. Extensive experiments on
both synthetic and real-world datasets demonstrate that our method significantly
outperforms the state-of-the-art OOD detection techniques, particularly under severe noisy label settings.


1 INTRODUCTION


Artificial intelligence (AI) models have achieved remarkable performance across myrid of domains
including computer vision and natural language processing. Yet, a persistent challenge arises in
real-world deployment: these models often fail to recognize inputs from unfamiliar data distributions, leading to overly confident and potentially misleading predictions (Goodfellow et al., 2014).
This limitation underscores the importance of out-of-distribution (OOD) detection for building trustworthy AI systems, particularly in high-stakes domains such as autonomous driving (Geiger et al.,
2012) and medical diagnostics (Schlegl et al., 2017). The goal of OOD detection is not only to provide accurate prediction on seen data distributions but also to flag inputs from novel or unobserved
distributions (Hendrycks & Gimpel, 2016).


OOD detection has been an active topic of research in the field of AI for many decades; a recent survey can be found in (Yang et al., 2024). A key focus in this field is detecting semantic shifts—scenarios where new, previously unseen classes appear in the test data, resulting in a
mismatch between the label spaces of in-distribution (ID) and OOD samples. A wide range of
methods have been proposed for OOD detection, including softmax/logit-based post-hoc techniques
(Hendrycks & Gimpel, 2016; Liang et al., 2018; Hendrycks et al., 2022; Sun & Li, 2022; Sun et al.,
2021; Dong et al., 2022) and feature distance-based strategies (Lee et al., 2018; Sun et al., 2022;
Ming et al., 2023; Sehwag et al., 2021; Ghosal et al., 2024). As softmax/logit-based approaches
are often susceptible to overconfidence, recent research has increasingly focused on distance-based
methods—such as Mahalanobis distance (Lee et al., 2018) and _k_ -nearest neighbors ( _k_ NN)(Sun et al.,


1


Figure 1: The upper row presents the UMAP representations of the latent feature vectors _**h**_ ( _**x**_ )
learned using the cross-entropy loss–based training on the noisily labeled dataset _{_ _**x**_ _n,_ - _yn}_ for various synthetic noise rates. The clusters are more distorted for the training data, losing the ID-ness
characteristics, resulting in degraded performance in OOD detection during test time. The lower row
presents the corresponding _k_ NN score histogram for the ID and OOD datasets. The false positive
ratio (FPR) for OOD detection using the _k_ NN score is also reported.


2022)—due to their empirical success. These methods rely on the observation that feature representations extracted from OOD samples typically lie farther from the ID feature space. Since deep
neural networks naturally encode semantic similarity in their embeddings—forming well-structured
clusters for similar samples—distance-based approaches leverage this structure to better separate ID
and OOD samples. Nonetheless, most of these methods are developed under the assumption that
models are trained on _clean,_ _correctly_ labeled data. However, in practice, training datasets often
contain noisy labels, stemming from the scarcity of expert annotators and the high cost of accurate
label acquisition (Buhrmester et al., 2016). Recent empirical studies have brought serious attention
to this issue, revealing that the presence of label noise can significantly degrade the performance
of state-of-the-art OOD detection methods (Humblot-Renaux et al., 2024). This highlights a critical
gap in current research and underscores the need to develop robust OOD detection frameworks that
remain reliable under real-world label noise.


The effect of label noise on training deep learning models has been extensively studied in recent
years; see the survey (Song et al., 2022). It is now well-established that training deep neural network
(DNN) models with noisy labels can severely degrade classification performance, leading to poor
generalization and overfitting (Arpit et al., 2017; Zhang et al., 2016). To address this, a variety of
label noise-robust methods have been proposed, including loss correction strategies such as probabilistic modeling techniques (Liu & Tao, 2016; Patrini et al., 2017; Li et al., 2021b; Xia et al., 2020;
Yang et al., 2021; Cheng et al., 2020), robust loss function designs (Zhang & Sabuncu, 2018a; Lyu
& Tsang, 2019; Wang et al., 2019a), and in-built sample selection strategies (Jiang et al., 2018; Yu
et al., 2019; Nguyen et al., 2019; Han et al., 2018a; Li et al., 2020b). However, their effectiveness
in OOD detection when the training labels are noisy remains largely unexplored. The key challenge lies in the misalignment of objectives: while label noise methods aim to correct the prediction
probabilities within the training distribution, OOD detection requires learning discriminative feature
representations to detect the samples that does not belong to the training distribution. Hence, most
existing label-noise approaches exhibit poor OOD detection performance when applied directly, as
we will demonstrate in detail in subsequent sections.


**Our Contributions.** In this work, we investigate the critical challenge of robust OOD detection in
the presence of noisy labels in the training set. Unlike existing studies that focus solely on the empirical limitations of current OOD detection methods (Humblot-Renaux et al., 2024), we identify a key


2


gap: the label noise-robust methods improves generalization under noisy supervision for classification settings, yet they are largely ineffective when directly applied for OOD detection. To address
this limitation, we have the following key contributions in this work:


    - _Robust_ _OOD_ _detection_ _framework._ We propose a novel distance-based OOD detection
framework that effectively cleanses feature embeddings, enhancing robustness under noisy
training labels. Our framework is accompanied by an easy-to-implement, end-to-end learning criterion, leveraging the loss correction techniques and low-rank and sparse decomposition strategies.


    - _Novel OOD-detection scoring._ The framework allows us to quantify the “OOD-ness” content of each training sample, producing a score that reflects its difficulty in training the OOD
detector. Leveraging this score, we introduce an enhanced distance-based scoring function
that is more robust compared to existing metrics.


    - _Promising empirical evidence for OOD detection._ We conduct extensive experiments using
synthetic noise settings, several real-world noisy datasets, e.g., CIFAR-10N, CIFAR-100N
(Wei et al., 2022), and Animal10N (Song et al., 2019), and evaluate our approach across
diverse OOD datasets. Additionally, we perform detailed ablation studies on key hyperparameters to demonstrate the robustness of our approach.


To the best of our knowledge, this work is the first to offer a principled solution to the problem,
achieving substantial improvements over state-of-the-art OOD detection methods in the presence of
label noise.


2 PROBLEM STATEMENT


Consider an input feature space _X_ _⊂_ R _[D]_, where _D_ denotes the dimensionality of the input features.
Let the label space be defined as _Y_ = _{_ 1 _, . . ., K}_, corresponding to _K_ distinct classes for the
ID data. We define the training dataset _D_ as _D_ = _{_ ( _**x**_ _n, yn_ ) _}_ _[N]_ _n_ =1 _[,]_ _**x**_ _n_ _∈X_ _,_ _yn_ _∈Y_, where
_**x**_ _n_ is the feature vector of the _n_ -th training example, _yn_ is its associated _ground-truth_ class label,
and _N_ denotes the total number of training samples. Each pair ( _**x**_ _n, yn_ ) is assumed to be drawn
independently and identically distributed (i.i.d.) from an underlying joint distribution _PXY_ . Let _**h**_ :
R _[D]_ _→_ R _[L]_ denote a DNN that maps each input _**x**_ _n_ to an _L_ -dimensional _latent feature_ representation
_**h**_ ( _**x**_ _n_ ). For the task of multi-class classification, we employ a projection head _**c**_ : R _[L]_ _→_ R _[K]_
to produce pre-softmax logits. Thus, the overall label prediction function is given by _**f**_ ( _**x**_ _n_ ) =
_**σ**_ ( _**c**_ ( _**h**_ ( _**x**_ _n_ ))), where _**σ**_ denotes the softmax function that output the class probabilities. We often
learn the parameters of these functions by training via cross-entropy (CE) minimization as follows:


where _**θ**_ denotes the DNN parameters of both _**h**_ and _**c**_ functions.


**OOD Detection.** AI systems are generally learned under the closed-world assumption, where it is
presumed that test samples are drawn from the same distribution as the training data. However, this
assumption often fails in practical scenarios, where models inevitably encounter samples that lie
outside the training distribution. These unfamiliar samples are known as OOD inputs (Hendrycks
& Gimpel, 2016). In classification tasks, such distributional shift may manifest as a semantic shift,
wherein some test instances originate from an _unknown_ label space _Y_ _[o]_, disjoint from the known
label space, i.e., _Y_ _∩Y_ _[o]_ = _∅_ . The objective of OOD detection is to identify whether a given test
input belongs to the in-distribution (ID) or not, thereby preventing the model from making confident
predictions on OOD inputs. Thus, OOD detection can be considered as a binary classification task
that distinguishes ID samples from OOD ones. This can be formalized by a detection function:


�ID if _s_ ( _**x**_ ) _≥_ _τ,_
_**g**_ _τ_ ( _**x**_ ) = (2)
OOD if _s_ ( _**x**_ ) _< τ,_


where _s_ ( _**x**_ ) is a scoring function that quantifies the likelihood of _**x**_ belonging to the ID distribution,
and _τ_ is a predefined threshold.


3


_K_


I[ _yn_ = _k_ ] log( _**f**_ ( _**x**_ _n_ )))) _,_ (1)

_k_ =1


minimize _**θ**_ _L_ CE( _**θ**_ ; _{_ _**x**_ _n, yn}_ ) = _−_


_N_


_n_ =1


Typically, scoring function _s_ ( _**x**_ ) is derived from the trained parameters of the underlying DNN
model. Several scoring functions have been proposed in the literature. Early OOD detection methods directly used the softmax outputs _**f**_ ( _**x**_ ) to score “OOD-ness” (Hendrycks & Gimpel, 2016;
Liang et al., 2018), but they suffered from overconfidence issues, reducing the desired ID-OOD separability. Further, pre-softmax activations-based approaches (e.g., by using the logits _**c**_ ( _**h**_ ( _**x**_ ))) were
introduced (Hendrycks et al., 2022; Sun & Li, 2022; Sun et al., 2021; Dong et al., 2022), though
they remained sensitive to architecture and still faced overconfidence issues. Recently, distancebased methods such as those based on Mahalanobis (Lee et al., 2018) and _k_ -nearest neighbor ( _k_ NN)
(Sun et al., 2022; Ming et al., 2023; Sehwag et al., 2021; Ghosal et al., 2024) have gained traction by
leveraging the clusterability of latent feature representations _**h**_ ( _**x**_ ). In essence, the success of OOD
detection lies in the careful design and learning of the scoring function _**s**_ ( _**x**_ ) that can ensure the
ID-OOD separability during test time.


**Learning under Label Noise.** Most studies in the domain of OOD detection assume that the DNN
classifier _**f**_ and the scoring function _**s**_ are learned using ground-truth labels _yn_ . However, the lack of
access to reliable ground-truth annotations is a significant challenge for robust OOD detection–see
an example in Fig. 1 where the clusterability of the latent representations _**h**_ ( _**x**_ ) is severely compromised under label noise, leading to significant degradation in ID-OOD separability for the _k_ NN
score function. In scenarios where ground-truth labels _yn_ are difficult to obtain, we often rely on
their noisy counterparts, denoted by _y_ - _n_ _∈{_ 1 _, . . ., K}_, associated with each data item _**x**_ _n_ . In noisy


3 PROPOSED APPROACH


In this section, we present our label noise-robust OOD detection framework. Our strategy is based
on _cleansing_ the noise-corrupted latent feature space using an end-to-end training strategy, thereby
making it robust for OOD detection at test time. Towards this goal, our framework encompasses
three main components: _i)_ loss correction module _ii)_ low rank and sparse decomposition of latent
feature matrix _iii)_ robust OOD detector using distance-based metrics, e.g., _k_ NN.


**Loss Correction.** As demonstrated in Fig. 1, training directly with noisy labels (e.g., by using the
CE minimization as in equation 1 where the unobserved ground-truth labels _yn_ are replaced by the
observed noisy labels _y_ - _n_ ) leads to a corrupted latent feature space. To address this, we first integrate a loss correction module to reduce the effect of label noise in learned features. Loss correction
strategies have attracted considerable attention in noisy label learning literature. Among these, probabilistic noise modeling via the so-called _transition matrices_ (Patrini et al., 2017; Tanno et al., 2019;
Li et al., 2021b; Ibrahim et al., 2023) and robust loss function-based approaches (Zhang & Sabuncu,
2018a; Lyu & Tsang, 2019; Wang et al., 2019a) are particularly well-received, owing to their strong
theoretical foundations and robust empirical performance in classification tasks.


In general, loss correction strategies design a modified cross-entropy loss to train the classifier _**f**_ on
the noisy dataset _{_ _**x**_ _n,_ - _yn}_, while aiming to predict the ground-truth labels, i.e.,

minimize _L_ [mod] CE [(] _**[θ]**_ _[,]_ _**[ η]**_ [;] _[ {]_ _**[x]**_ _[n][,]_ [ �] _[y][n][}]_ [)] (3)
_**θ**_ _,_ _**η**_


where _**η**_ typically refers to additional model parameters according to specific loss designs. For instance, in the case of transition matrix-based approaches (Patrini et al., 2017; Tanno et al., 2019;
Li et al., 2021b; Ibrahim et al., 2023), _**η**_ refers to the noise transition probabilities that learns the
probabilistic label confusion _confusion matrix_ _**T**_, whose ( _k, k_ _[′]_ )-th entry models the label corruption
process via [ _**T**_ ] _k,k′_ = Pr( _y_ - _n_ = _k|yn_ = _k_ _[′]_ ). Intuitively, _**T**_ describes how clean labels _yn_ are stochastically flipped into noisy labels � _yn_ . The modified cross-entropy then incorporates _**T**_ by mapping the
classifier’s predicted clean-label distribution into the corresponding noisy-label distribution, so that
training can directly account for and correct the effect of label noise. In sample selection approaches
(Jiang et al., 2018; Yu et al., 2019; Nguyen et al., 2019; Han et al., 2018a; Li et al., 2020b), _**η**_ instead
represents sample-weighting terms that regulate the contributions of clean and noisy sample–label


4


pairs. In contrast, robust loss function-based methods, e.g., (Zhang & Sabuncu, 2018a; Wang et al.,
2019a), often do not introduce additional parameters—they directly design loss functions that are
inherently less sensitive to incorrect labels. Popular approaches such as symmetric cross-entropy
(SCE) (Wang et al., 2019a) and generalized cross-entropy (GCE) (Zhang & Sabuncu, 2018a) can be
viewed as hybrids of CE loss and mean absolute error (MAE) loss, thereby combining the favorable
convergence properties of CE with the robustness of MAE against outliers. For instance, the SCE
loss function is defined as follows:


where the parameters _α, β_ _>_ 0 are certain scalars, _**g**_ _∈_ R _[K]_ is a probability vector such that _**g**_ _≥_
**0** _,_ 1 _[⊤]_ _**g**_ = 1, _**y**_ _∈{_ 0 _,_ 1 _}_ _[K]_ is a one-hot label, and _**y**_ _ϵ_ takes care of the log 0 cases of the one-hot label
_**y**_ using a small constant _ϵ_ . Note that the first term in the R.H.S is the scaled CE loss, whereas the
second term in the R.H.S reduces to the MAE loss when _ϵ_ = _−_ 2 (Wang et al., 2019a).


Nonetheless, these loss correction strategies primarily operate by modifying the softmax prediction
outputs of the ID samples rather than directly _correcting_ their feature embeddings. However, feature
embeddings are often more critical for OOD detection, particularly for the distance-based OOD
metrics such as _k_ -nearest neighbor (Sun et al., 2022) and Mahalanobis (Lee et al., 2018). This
misalignment of objectives results in suboptimal performance in mitigating the effect of label noise
in OOD detection.


**Low-rank** **plus** **Sparse** **Decomposition.** To overcome the limitation of loss correction modules
in handling feature correction, we introduce the next key component of our framework. A critical
observation underlying its design is that, in the absence of label noise, latent feature vectors naturally exhibit certain clustering patterns, reflecting their low-rank structure due to their class-specific
organization—see the first UMAP plot in Fig. 1. This intrinsic structural tendency can be explicitly
leveraged in the training phase to encourage low-rank properties in the feature representations. To
this end, we adopt a low-rank and sparse decomposition strategy, drawing inspiration from classical
signal processing techniques (Cand`es et al., 2011; Zhang et al., 2011).


Consider the latent feature representation _**h**_ ( _**x**_ ) of the input image _**x**_ (e.g., the penultimate layer
encoding of the DNN model). Let us represent the latent feature matrix _**H**_ as follows: _**H**_ =

- _**h**_ ( _**x**_ 1) _, . . .,_ _**h**_ ( _**x**_ _N_ ) - _∈_ R _[D][×][N]_, where _D_ is the feature dimension and _N_ is the number of training
samples. Our goal is to promote a low-rank structure in this feature matrix (as typically observed
when labels are clean), while assuming that samples that deviate strongly from this structure (e.g.,
difficult or confusing examples that much deviate from the ID-ness characteristics) are relatively
few. Hence, in order to exploit the low-rankness of the latent matrix _**H**_ along with a sparse structure, we assume that _**H**_ _≈_ _**L**_ + _**S**_ _,_ where _**L**_ _∈_ R _[D][×][N]_ is the low-rank component and _**S**_ _∈_ R _[D][×][N]_ is
a column sparse matrix, i.e., most columns of _**S**_ have zero _ℓ_ 2 norm. That means, the low-rank term
captures the underlying class structure information, whereas the sparse term can handle the outlier
data items that do not strictly conform to the low-rank assumption.


Learning _**L**_ and _**S**_ from the observed matrix _**H**_ generally involves solving an optimization problem
of the form (Candes et al., 2009; Wang et al., 2008):


min s.t. _**H**_ = _**L**_ + _**S**_ _,_
_**L**_ _,_ _**S**_ _[∥]_ _**[L]**_ _[∥][∗]_ [+] _[ λ][∥]_ _**[S]**_ _[∥]_ [2] _[,]_ [1]


where _∥_ _**L**_ _∥∗_ denotes the nuclear norm of _**L**_ to promote the low-rankness and _∥_ _**S**_ _∥_ 2 _,_ 1 denotes the
matrix mixed norm that promotes column sparsity in _**S**_ . Here, _λ_ _>_ 0 is a regularization parameter
that balances the contributions of the low-rank and sparse terms. Directly minimizing the nuclear
norm can be computationally expensive, as it typically requires repeated singular value decompositions. To avoid this cost during training, we adopt an efficient power-iteration (PI)–based low-rank
approximation strategy (Rokhlin et al., 2010; Gu, 2015). Specifically, The method estimates the top_K_ left singular vectors of the latent representation _**H**_ by iteratively refining a randomly initialized
orthonormal basis _**Q**_ _∈_ R _[D][×][K]_ through alternating projections of _**H**_ and orthonormalization via QR
decomposition. Here, The rank _K_ can be selected according to the number of classes (or based on
the number of coarse-grained classes in the case of datasets with very large label space). Using the
learned _**Q**_, we decompose the feature matrix as


_**H**_ ID = [ _**h**_ ID( _**x**_ 1) _, . . .,_ _**h**_ ID( _**x**_ _N_ )] = ( _**QQ**_ _[⊤]_ ) _**H**_ _,_ _**H**_ OOD = [ _**h**_ OOD( _**x**_ 1) _, . . .,_ _**h**_ OOD( _**x**_ _N_ )] = _**H**_ _−_ _**H**_ ID _,_


5


_K_
�[ _**g**_ ] _k_ log[ _**y**_ _ϵ_ ] _k,_


_k_ =1


_ℓ_ SCE( _**g**_ _,_ _**y**_ ) = _−α_


_K_
�[ _**y**_ ] _k_ log[ _**g**_ ] _k −_ _β_


_k_ =1


where _**H**_ ID _∈_ R _[D][×][N]_ represents the ID component and _**H**_ OOD _∈_ R _[D][×][N]_ contains residual features
that may potentially carry non-ID information (or OOD-like components). Further, to enforce the
column sparsity in the matrix _**H**_ OOD, we employ the following regularization term:


Note that the outlier content captured in _**H**_ OOD matrix can include both spurious features learned
due to the influence of noisily labeled training examples as well as the feature components within
the correctly labeled training samples that deviate from ID characteristics. We do not attempt to
distinguish between these components explicitly, instead _**H**_ OOD is designed to capture anything that
is non-ID, while the low-rank component _**H**_ OOD focuses on learning the cleanest possible version of
the ID feature space.


Finally, the proposed method is trained by minimizing a joint objective that combines the modified
cross-entropy loss as explained in equation 3 with the regularizer in equation 4:


_LF_ = _L_ [mod] CE [+] _[ λ][ L]_ [sparse] _[,]_


where _λ_ _>_ 0 is a regularization hyperparameter that controls the strength of the column-sparsity
term. The detailed algorithm is presented in the supplementary section.


**Remark 1** _The_ _added_ _regularization_ _as_ _described_ _in_ _equation_ _4_ _is_ _implemented_ _via_ _the_ _PI–based_
_low-rank_ _approximation_ _of_ _the_ _batch-wise_ _feature_ _matrix_ _**H**_ _(see_ _the_ _detailed_ _description_ _in_ _Algo-_
_rithm 1). Since the PI method uses simple matrix–vector multiplications instead of a full SVD, the_
_additional computational complexity is approximately O_ - _T_ ( _DBK_ ) + _DK_ [2][�] _, where B is the batch_
_size,_ _K_ _is_ _the_ _number_ _of_ _classes_ _(or_ _the_ _chosen_ _rank_ _parameter),_ _D_ _is_ _the_ _feature_ _embedding_ _di-_
_mension,_ _and_ _T_ _is_ _the_ _number_ _of_ _PI_ _iterations,_ _which_ _is_ _often_ _a_ _smaller_ _number_ _in_ _practice._ _This_
_overhead is moderate and substantially cheaper than exact nuclear-norm optimization for low-rank_
_decompositions._


**Robust** **OOD** **Detection** **with** **Refined** **Feature** **Representations.** To detect OOD samples at test
time, we propose a feature distance–based score with a robust selection strategy for reference
embeddings. Specifically, for each training sample _**x**_ _n_, we assign an OOD-ness score defined as

_o_ ( _**x**_ _n_ ) = _∥_ _**h**_ ( _**x**_ _n_ ) _−_ _**h**_ ID( _**x**_ _n_ ) _∥_ 2, where larger scores indicate greater deviation from ID characteristics (e.g., noisy or ambiguous samples), while smaller scores correspond to clean, representative
ID examples—also see Fig. 3a in Sec. 4. Based on this score, we retain only the most challenging
examples by selecting


_S_ = _{n ∈_ [ _N_ ] _| o_ ( _**x**_ _n_ ) _≥_ _αp},_ _αp_ = Quantile(1 _−p_ )� _{o_ ( _**x**_ _n_ ) _}_ _[N]_ _n_ =1� _,_ (5)


so that the top (1 _−_ _p_ )% of samples with the highest OOD-ness scores are used to construct a robust
detector, where _p_ _∈_ [0 _,_ 1). For each selected sample _n_ _∈S_, we compute the _ℓ_ 2-normalized feature
vectors _**u**_ ID( _**x**_ _n_ ) = _**h**_ ID( _**x**_ _n_ ) _/∥_ _**h**_ ID( _**x**_ _n_ ) _∥_ 2. Similarly, the test sample _**x**_ _[∗]_ is mapped to its normalized
feature _**u**_ ( _**x**_ _[∗]_ ) = _**h**_ ( _**x**_ _[∗]_ ) _/∥_ _**h**_ ( _**x**_ _[∗]_ ) _∥_ 2, whose distance to the stored ID embeddings _{_ _**u**_ ID( _**x**_ _n_ ) _}n∈S_
is then evaluated. For example, using _k_ NN distance, we have the score defined as _s_ kNN( _**x**_ _[∗]_ ) =

_−_ �� _**u**_ ( _**x**_ _∗_ ) _−_ _**u**_ (ID _k_ )��2 [, where] _**[ u]**_ ID [(] _[k]_ [)] [denotes the] _[ k]_ [-th nearest embedding in] _[ S]_ [. A similar modified metric]
can be derived using Mahalanobis distance as well. A decision threshold _τ_ of the detection function
_gτ_ is chosen based on a validation set to ensure a high ID recall (e.g., 95%).


4 EXPERIMENTS


In this section, we evaluate the effectiveness of our label noise-robust OOD detection framework,
which is named as **N** oise-robust **O** ut- **O** f- **D** istribution **Le** arning (NOODLE)


**Datasets.** For synthetic label noise settings, we consider CIFAR-10 (Krizhevsky, 2009) as ID
dataset. For synthetic label noise generation, we adopt class-independent symmetric noise, where
every true label has the same probability of being corrupted, and when corrupted, it is flipped uniformly at random to any of the other _K_ _−_ 1 classes, regardless of the original class. We vary the


6


~~�~~

- _D_
��� ( _**H**_ OOD) [2] _ij_ _[.]_ (4)

_i_ =1


_L_ sparse = _∥_ _**H**_ OOD _∥_ 2 _,_ 1 =


_N_


_j_ =1


Table 1: OOD detection performance on CIFAR10 with synthetic label noise across different OOD
datasets; The top two performing algorithms (in terms of average FPR95) are highlighted in bold.


**Method** **SVHN** **FashionMNIST** **LSUN** **iSUN** **DTD/Texture** **Places365** **Average**


FPR95 _↓_ AUROC _↑_ FPR95 AUROC FPR95 AUROC FPR95 AUROC FPR95 AUROC FPR95 AUROC FPR95 AUROC


_Noise rate = 10%_


_OOD baselines_


KNN 34 _._ 23 93 _._ 92 38 _._ 51 93 _._ 57 17 _._ 71 96 _._ 66 21 _._ 72 95 _._ 93 28 _._ 58 94 _._ 04 62 _._ 26 84 _._ 62 33 _._ 84 93 _._ 12
MSP 73 _._ 54 84 _._ 82 54 _._ 92 88 _._ 64 32 _._ 84 94 _._ 49 55 _._ 18 89 _._ 70 77 _._ 02 75 _._ 86 69 _._ 01 80 _._ 33 60 _._ 42 85 _._ 64
ODIN 87 _._ 82 60 _._ 19 64 _._ 22 79 _._ 77 24 _._ 42 94 _._ 56 32 _._ 99 91 _._ 49 80 _._ 30 56 _._ 06 78 _._ 84 66 _._ 87 61 _._ 43 74 _._ 82
Energy 80 _._ 31 77 _._ 93 57 _._ 89 84 _._ 84 17 _._ 74 96 _._ 65 54 _._ 63 88 _._ 38 82 _._ 43 62 _._ 42 78 _._ 29 72 _._ 39 60 _._ 37 79 _._ 12
ReAct 96 _._ 77 53 _._ 22 63 _._ 00 87 _._ 87 49 _._ 38 90 _._ 45 72 _._ 98 81 _._ 08 92 _._ 89 44 _._ 08 75 _._ 21 75 _._ 29 75 _._ 04 72 _._ 00
Mahalanobis 31 _._ 71 91 _._ 10 72 _._ 56 74 _._ 47 28 _._ 29 93 _._ 87 52 _._ 77 81 _._ 81 49 _._ 11 80 _._ 53 94 _._ 74 44 _._ 55 54 _._ 86 77 _._ 72
CIDER 30 _._ 04 95 _._ 15 20 _._ 78 96 _._ 62 35 _._ 14 93 _._ 34 66 _._ 61 84 _._ 09 46 _._ 67 92 _._ 82 62 _._ 94 88 _._ 58 43 _._ 69 91 _._ 77
SSD+ 52 _._ 65 87 _._ 82 80 _._ 75 85 _._ 52 90 _._ 05 77 _._ 82 65 _._ 24 84 _._ 64 21 _._ 60 95 _._ 72 16 _._ 22 96 _._ 88 54 _._ 42 88 _._ 07
SNN 49 _._ 71 91 _._ 73 29 _._ 72 95 _._ 06 20 _._ 96 96 _._ 25 25 _._ 36 95 _._ 23 34 _._ 26 92 _._ 53 56 _._ 29 86 _._ 46 36 _._ 05 92 _._ 88


_Label noise-robust baselines_


SCE 5 _._ 97 98 _._ 96 12 _._ 60 97 _._ 80 2 _._ 93 99 _._ 44 15 _._ 62 97 _._ 01 27 _._ 16 94 _._ 28 59 _._ 64 85 _._ 79 20 _._ 66 95 _._ 55
GCE 7 _._ 81 98 _._ 50 16 _._ 20 97 _._ 33 5 _._ 88 98 _._ 96 12 _._ 03 97 _._ 92 31 _._ 63 93 _._ 56 47 _._ 83 89 _._ 35 20 _._ 23 95 _._ 94
Co-teaching 40 _._ 12 90 _._ 87 99 _._ 29 59 _._ 93 75 _._ 83 76 _._ 34 96 _._ 95 54 _._ 21 49 _._ 45 82 _._ 48 93 _._ 19 48 _._ 57 75 _._ 81 68 _._ 73
DivideMix 62 _._ 65 87 _._ 37 68 _._ 10 81 _._ 99 49 _._ 01 91 _._ 24 42 _._ 84 91 _._ 52 37 _._ 48 92 _._ 79 77 _._ 16 75 _._ 03 56 _._ 21 86 _._ 66
CM 6 _._ 14 98 _._ 90 15 _._ 80 97 _._ 05 6 _._ 03 98 _._ 91 10 _._ 40 98 _._ 12 19 _._ 08 96 _._ 47 53 _._ 54 87 _._ 17 18 _._ 50 96 _._ 10
VolMinNet 2 _._ 64 99 _._ 47 5 _._ 18 98 _._ 94 5 _._ 00 98 _._ 98 9 _._ 26 98 _._ 24 21 _._ 60 95 _._ 86 56 _._ 16 87 _._ 43 **16.64** 96 _._ 49


NOODLE 3 _._ 51 99 _._ 28 5 _._ 03 98 _._ 97 3 _._ 31 99 _._ 33 3 _._ 05 99 _._ 22 16 _._ 61 96 _._ 78 48 _._ 02 89 _._ 62 **13.26** 97 _._ 20


_Noise rate = 30%_


_OOD baselines_


KNN 23 _._ 80 95 _._ 80 36 _._ 15 93 _._ 43 27 _._ 04 94 _._ 61 22 _._ 03 95 _._ 86 39 _._ 50 90 _._ 40 69 _._ 76 83 _._ 53 36 _._ 38 92 _._ 27
MSP 76 _._ 88 80 _._ 34 56 _._ 16 87 _._ 14 29 _._ 90 93 _._ 92 58 _._ 82 88 _._ 30 79 _._ 04 71 _._ 54 74 _._ 98 76 _._ 61 62 _._ 63 82 _._ 97
ODIN 83 _._ 79 61 _._ 60 50 _._ 70 83 _._ 57 22 _._ 26 94 _._ 83 35 _._ 64 89 _._ 98 79 _._ 68 54 _._ 97 83 _._ 19 60 _._ 62 59 _._ 21 74 _._ 26
Energy 76 _._ 67 76 _._ 82 51 _._ 45 86 _._ 81 19 _._ 38 95 _._ 65 66 _._ 77 84 _._ 97 80 _._ 89 62 _._ 95 77 _._ 32 72 _._ 32 62 _._ 08 79 _._ 92
ReAct 88 _._ 89 67 _._ 41 62 _._ 61 85 _._ 50 21 _._ 63 95 _._ 34 90 _._ 19 65 _._ 23 91 _._ 33 51 _._ 51 82 _._ 51 68 _._ 95 72 _._ 86 72 _._ 33
Mahalanobis 37 _._ 86 90 _._ 25 50 _._ 42 85 _._ 21 26 _._ 98 93 _._ 29 60 _._ 92 79 _._ 60 52 _._ 75 77 _._ 45 95 _._ 94 40 _._ 86 54 _._ 15 77 _._ 78
CIDER 90 _._ 69 65 _._ 27 60 _._ 75 79 _._ 74 96 _._ 69 48 _._ 38 90 _._ 98 66 _._ 91 59 _._ 75 76 _._ 81 70 _._ 11 79 _._ 49 78 _._ 16 69 _._ 43
SSD+ 76 _._ 34 78 _._ 07 61 _._ 81 88 _._ 45 90 _._ 07 77 _._ 66 84 _._ 61 79 _._ 29 28 _._ 94 94 _._ 78 50 _._ 55 87 _._ 94 65 _._ 39 84 _._ 37
SNN 23 _._ 37 95 _._ 72 34 _._ 55 94 _._ 11 25 _._ 91 94 _._ 90 34 _._ 38 92 _._ 36 42 _._ 27 89 _._ 39 65 _._ 06 84 _._ 19 37 _._ 59 91 _._ 78


_Label noise-robust baselines_


SCE 19 _._ 48 96 _._ 45 25 _._ 84 95 _._ 37 16 _._ 58 96 _._ 57 61 _._ 71 87 _._ 42 35 _._ 50 91 _._ 76 74 _._ 66 79 _._ 27 38 _._ 96 91 _._ 14
GCE 58 _._ 38 91 _._ 40 20 _._ 37 96 _._ 64 11 _._ 32 97 _._ 95 12 _._ 38 97 _._ 73 30 _._ 53 94 _._ 08 51 _._ 59 88 _._ 18 30 _._ 76 94 _._ 33
Co-teaching 50 _._ 10 83 _._ 16 99 _._ 99 20 _._ 37 96 _._ 73 64 _._ 14 97 _._ 86 41 _._ 70 53 _._ 71 79 _._ 45 93 _._ 84 48 _._ 63 82 _._ 04 56 _._ 24
DivideMix 58 _._ 39 90 _._ 07 31 _._ 17 94 _._ 93 27 _._ 86 95 _._ 59 16 _._ 38 96 _._ 93 36 _._ 28 92 _._ 76 59 _._ 28 84 _._ 34 38 _._ 22 92 _._ 44
CM 22 _._ 04 96 _._ 76 8 _._ 79 98 _._ 04 10 _._ 17 98 _._ 12 23 _._ 30 95 _._ 64 23 _._ 71 94 _._ 99 55 _._ 42 86 _._ 90 **23.90** 95 _._ 08
VolMinNet 4 _._ 99 99 _._ 04 14 _._ 01 97 _._ 09 9 _._ 48 98 _._ 33 51 _._ 23 89 _._ 68 27 _._ 84 93 _._ 44 59 _._ 25 85 _._ 42 27 _._ 80 93 _._ 84


NOODLE 1 _._ 84 99 _._ 60 19 _._ 66 96 _._ 36 7 _._ 28 95 _._ 53 10 _._ 76 97 _._ 89 20 _._ 67 95 _._ 85 57 _._ 50 85 _._ 87 **19.62** 95 _._ 68


_Noise rate = 50%_


_OOD baselines_


KNN 65 _._ 53 85 _._ 64 37 _._ 84 93 _._ 71 30 _._ 61 93 _._ 38 45 _._ 41 89 _._ 21 43 _._ 81 89 _._ 06 74 _._ 98 79 _._ 58 49 _._ 70 88 _._ 43
MSP 96 _._ 92 53 _._ 68 80 _._ 68 77 _._ 70 47 _._ 78 89 _._ 67 67 _._ 84 83 _._ 23 82 _._ 50 68 _._ 45 81 _._ 12 73 _._ 63 76 _._ 14 74 _._ 40
ODIN 94 _._ 94 44 _._ 15 71 _._ 46 80 _._ 28 34 _._ 04 91 _._ 82 47 _._ 51 88 _._ 11 79 _._ 61 60 _._ 80 82 _._ 87 66 _._ 17 68 _._ 40 71 _._ 89
Energy 97 _._ 93 46 _._ 79 83 _._ 41 76 _._ 67 39 _._ 77 90 _._ 87 67 _._ 60 81 _._ 00 85 _._ 43 60 _._ 47 82 _._ 28 69 _._ 91 76 _._ 07 70 _._ 95
ReAct 99 _._ 19 24 _._ 75 90 _._ 11 64 _._ 20 50 _._ 47 85 _._ 45 78 _._ 12 67 _._ 79 93 _._ 79 39 _._ 56 86 _._ 40 62 _._ 25 83 _._ 02 57 _._ 33
Mahalanobis 55 _._ 77 83 _._ 12 59 _._ 93 85 _._ 98 31 _._ 23 93 _._ 54 45 _._ 17 88 _._ 63 48 _._ 90 81 _._ 70 93 _._ 28 51 _._ 47 55 _._ 71 80 _._ 74
CIDER 99 _._ 02 50 _._ 38 97 _._ 54 52 _._ 78 91 _._ 99 48 _._ 96 98 _._ 67 45 _._ 37 99 _._ 16 50 _._ 86 92 _._ 60 59 _._ 47 96 _._ 50 51 _._ 30
SSD+ 83 _._ 54 77 _._ 49 76 _._ 76 76 _._ 56 83 _._ 89 76 _._ 28 86 _._ 10 73 _._ 24 62 _._ 31 81 _._ 41 42 _._ 27 88 _._ 67 72 _._ 48 78 _._ 94
SNN 71 _._ 41 83 _._ 96 68 _._ 22 87 _._ 66 53 _._ 56 89 _._ 27 63 _._ 49 80 _._ 99 56 _._ 95 85 _._ 48 82 _._ 28 77 _._ 12 65 _._ 99 84 _._ 08


_Label noise-robust baselines_


SCE 14 _._ 10 97 _._ 40 42 _._ 30 90 _._ 96 25 _._ 18 94 _._ 09 67 _._ 17 80 _._ 81 51 _._ 51 84 _._ 65 70 _._ 67 77 _._ 75 45 _._ 15 87 _._ 61
GCE 19 _._ 19 96 _._ 43 29 _._ 10 95 _._ 06 22 _._ 98 95 _._ 40 53 _._ 92 86 _._ 31 48 _._ 35 87 _._ 54 65 _._ 83 83 _._ 58 39 _._ 89 90 _._ 72
Co-teaching 57 _._ 05 76 _._ 43 99 _._ 97 28 _._ 21 99 _._ 23 55 _._ 11 96 _._ 99 52 _._ 50 54 _._ 45 78 _._ 87 94 _._ 22 47 _._ 36 83 _._ 65 56 _._ 41
DivideMix 24 _._ 69 95 _._ 75 40 _._ 94 93 _._ 32 37 _._ 02 94 _._ 36 20 _._ 81 96 _._ 10 53 _._ 10 89 _._ 35 56 _._ 39 86 _._ 71 38 _._ 82 92 _._ 60
CM 17 _._ 37 96 _._ 91 21 _._ 93 95 _._ 56 17 _._ 16 96 _._ 58 39 _._ 52 92 _._ 86 30 _._ 23 93 _._ 36 61 _._ 88 84 _._ 82 **31.35** 93 _._ 35
VolMinNet 13 _._ 01 97 _._ 74 15 _._ 36 97 _._ 11 14 _._ 18 97 _._ 24 60 _._ 13 80 _._ 22 45 _._ 85 87 _._ 40 55 _._ 26 86 _._ 94 33 _._ 96 91 _._ 11


NOODLE 6 _._ 35 98 _._ 43 17 _._ 83 96 _._ 58 7 _._ 09 98 _._ 50 32 _._ 28 93 _._ 92 30 _._ 09 92 _._ 47 70 _._ 41 81 _._ 24 **27.34** 93 _._ 52


noise rate at 10%, 30%, and 50% to simulate different levels of noise severity. To test under realistic label noise, we also consider the human-annotated noisy label datasets CIFAR-10N, CIFAR100N (Wei et al., 2022), and Animal-10N (Song et al., 2019). These are annotated by the crowd
workers from the popular crowdsourcing platform Amazon Mechanical Turk (AMT). CIFAR-10N
provides five types of noisy label sets: _worst_, _aggregate_, _random1_, _random2_, and _random3_, having
40.21%, 9.03%, 17.23%, 18.12%, and 17.64% overall noise rates, respectively. Animal-10N contains human-annotated labels with noise rate 8%, whereas CIFAR-100N has noise rate of 40.20%.
As OOD datasets during test time, we consider several benchmark datasets, including SVHN (Netzer
et al., 2011), FashionMNIST (Xiao et al., 2017), LSUN (Yu et al., 2016), iSUN (Pan & Gir´o-i-Nieto,
2015), Texture (Cimpoi et al., 2013), and Places365 (Zhou et al., 2016).


**Baselines.** We compare our proposed method with several OOD detection baselines as well as different label noise-robust techniques. Regarding the OOD detection baselines, we consider MSP


7


Table 2: Average OOD detection performance on noisy real datasets; The top two performing algorithms (in terms of average FPR95) are highlighted in bold.


**Method** **CIFAR-10N** **Animal-10N** **CIFAR-100N**


Clean Worst Agg Rand1 Rand2 Rand3 FPR95 AUROC FPR95 AUROC
FPR95 AUROC FPR95 AUROC FPR95 AUROC FPR95 AUROC FPR95 AUROC FPR95 AUROC


_OOD baselines_


KNN 21 _._ 06 95 _._ 80 32 _._ 48 92 _._ 89 23 _._ 95 94 _._ 84 35 _._ 48 92 _._ 65 31 _._ 99 92 _._ 70 27 _._ 27 94 _._ 09 70 _._ 44 77 _._ 04 43 _._ 20 86 _._ 54
MSP 56 _._ 43 90 _._ 07 60 _._ 15 85 _._ 49 55 _._ 04 88 _._ 21 60 _._ 75 86 _._ 62 56 _._ 44 86 _._ 50 53 _._ 90 86 _._ 70 90 _._ 64 59 _._ 90 81 _._ 08 72 _._ 66
ODIN 33 _._ 10 92 _._ 47 45 _._ 31 86 _._ 69 43 _._ 83 89 _._ 03 49 _._ 12 86 _._ 56 46 _._ 71 84 _._ 24 41 _._ 91 87 _._ 37 76 _._ 97 62 _._ 55 71 _._ 72 76 _._ 64
Energy 39 _._ 15 92 _._ 03 47 _._ 30 87 _._ 75 56 _._ 03 87 _._ 90 54 _._ 89 87 _._ 24 50 _._ 31 86 _._ 11 42 _._ 74 88 _._ 90 75 _._ 60 74 _._ 52 78 _._ 93 51 _._ 74
ReAct 60 _._ 31 83 _._ 12 65 _._ 89 78 _._ 86 47 _._ 17 91 _._ 01 68 _._ 77 76 _._ 92 65 _._ 46 79 _._ 83 57 _._ 81 81 _._ 24 79 _._ 00 71 _._ 15 76 _._ 24 67 _._ 63
Mahalanobis 47 _._ 22 82 _._ 91 53 _._ 57 80 _._ 61 51 _._ 14 81 _._ 76 55 _._ 26 80 _._ 05 44 _._ 25 84 _._ 60 48 _._ 37 83 _._ 37 54 _._ 54 73 _._ 00 75 _._ 15 65 _._ 47
CIDER 39 _._ 33 92 _._ 49 98 _._ 35 58 _._ 92 83 _._ 54 62 _._ 71 98 _._ 32 57 _._ 88 97 _._ 59 57 _._ 59 96 _._ 48 50 _._ 41 82 _._ 64 75 _._ 09 54 _._ 01 82 _._ 50
SSD+ 33 _._ 66 92 _._ 42 39 _._ 69 89 _._ 82 45 _._ 72 86 _._ 07 41 _._ 90 88 _._ 77 39 _._ 37 90 _._ 52 32 _._ 40 92 _._ 23 66 _._ 11 75 _._ 28 50 _._ 29 83 _._ 36
SNN 22 _._ 60 95 _._ 53 **30.87** 92 _._ 78 25 _._ 14 94 _._ 18 29 _._ 87 93 _._ 76 30 _._ 74 92 _._ 94 34 _._ 26 92 _._ 12 31 _._ 43 93 _._ 65 **43.15** 87 _._ 13


_Label noise-robust baselines_


SCE 19 _._ 71 95 _._ 62 34 _._ 53 92 _._ 11 22 _._ 87 94 _._ 76 **22.90** 94 _._ 77 24 _._ 42 94 _._ 25 24 _._ 81 94 _._ 40 31 _._ 97 93 _._ 47 46 _._ 13 83 _._ 15
GCE 18 _._ 56 96 _._ 33 35 _._ 75 91 _._ 50 19 _._ 44 96 _._ 03 23 _._ 47 95 _._ 11 18 _._ 89 95 _._ 86 19 _._ 78 95 _._ 61 36 _._ 62 91 _._ 65 68 _._ 54 77 _._ 54
DivideMix 40 _._ 81 89 _._ 16 39 _._ 32 91 _._ 53 65 _._ 83 81 _._ 83 66 _._ 64 84 _._ 52 59 _._ 27 84 _._ 22 24 _._ 81 94 _._ 40 34 _._ 27 91 _._ 77 56 _._ 28 82 _._ 92
Co-teaching 81 _._ 94 58 _._ 15 82 _._ 14 60 _._ 68 77 _._ 98 63 _._ 71 53 _._ 42 74 _._ 23 77 _._ 59 62 _._ 76 81 _._ 94 58 _._ 15 68 _._ 47 61 _._ 72 81 _._ 68 59 _._ 61
CM 18 _._ 32 96 _._ 33 36 _._ 28 89 _._ 66 **21.22** 95 _._ 17 24 _._ 72 94 _._ 61 23 _._ 62 95 _._ 04 **20.70** 95 _._ 51 33 _._ 50 92 _._ 75 49 _._ 52 85 _._ 41
VolMinNet **15.00** 96 _._ 89 37 _._ 52 91 _._ 60 23 _._ 37 94 _._ 96 **22.90** 95 _._ 15 **18.80** 96 _._ 15 22 _._ 19 95 _._ 02 **29.26** 94 _._ 09 56 _._ 65 81 _._ 39


NOODLE **14.60** 96 _._ 81 **25.45** 94 _._ 71 **16.52** 96 _._ 46 **16.66** 96 _._ 42 **15.46** 96 _._ 70 **15.49** 96 _._ 50 **25.25** 95 _._ 13 **35.19** 89 _._ 52


Figure 2: The UMAP representations of the latent feature vectors _**h**_ ( _**x**_ ) learned using (a) CM approach (transition matrix-based label noise correction) with cross-entropy loss, and (b) NOODLE,
our proposed approach, on the CIFAR-10 dataset with synthetic label noise.


(Hendrycks & Gimpel, 2017), ODIN (Liang et al., 2018), Energy (Liu et al., 2020), ReAct (Sun
et al., 2021), Mahalanobis (Lee et al., 2018), KNN (Sun et al., 2022), CIDER (Ming et al., 2023),
SSD+ (Sehwag et al., 2021), and SNN (Ghosal et al., 2024). MSP, ODIN and Energy are softmaxbased approaches. MSP relies only on softmax output of the model, while ODIN uses an additional
temperature scaling hyperparameter. The Energy method computes an energy-based metric from
the model outputs, identifying test samples with higher energy as OOD. ReAct is a logit-based approach. Mahalanobis, KNN, CIDER, SSD+, and SNN are distance-based approaches. The distancebased methods—such as KNN, CIDER, and SNN all advocate the use of non-parametric _k_ NN-based
score (Sun et al., 2022) for OOD detection.


Regarding label-noise-robust learning methods, we consider different lines of approach for our evaluation. Specifically, we consider CM (Li et al., 2021a), VolMinNet (Li et al., 2021a), SCE (Wang
et al., 2019b), GCE (Zhang & Sabuncu, 2018b), DivideMix (Li et al., 2020a), and Co-Teaching (Han
et al., 2018b). Here, CM and VolMinNet are probabilistic noise-modeling approaches that rely on
transition matrices to correct label noise. GCE and SCE are robust loss function–based approaches
that are variants of the cross-entropy loss. DivideMix and Co-Teaching are sample-selection-based
approaches that focus on reweighting samples based on the presence of label noise. For OOD performance evaluation, we use the _k_ NN-based metric for all these methods, unless specified otherwise.


**Implementation** **Settings.** We use a CNN-based architecture, DenseNet-101(Huang et al., 2016),
as the backbone model for all datasets. We train the model from scratch using the ID datasets.
During training for CIFAR-10N and Animal-10N, we set the number of epochs to 100 and use a
batch size of 64. First, we extract penultimate layer’s features and then apply global average pooling
following by _ℓ_ 2-normalization before performing the PI-based low-rank decomposition module of
our NOODLE approach. We initialize the transition matrices as identity matrices of appropriate
size in the case of CM-based approaches. We set the number of iterations as _n_ iter = 3 for the
PI-based low-rank decomposition in our approach. For all datasets, we choose stochastic gradient


8


(a) (b)


Figure 3: (a) Learned “OOD-ness” score _o_ ( _**x**_ _n_ ) for CIFAR-10N training images. Top: top-10 samples with the highest _o_ ( _**x**_ _n_ ) scores. Bottom: bottom-10 samples with the lowest _o_ ( _**x**_ _n_ ) scores. (b)
OOD performance of NOODLE with ( _p_ = 0 _._ 5) and without ( _p_ = 0) robust selection strategy.


(a) (b)


Figure 4: Comparison of OOD detection performance (FPR95 _↓_ ) on CIFAR-10N. (a) Different KNN
variants of NOODLE (b) Different Mahalanobis variants of NOODLE. Here “NOODLE (X, Y)”
refers to NOODLE with X as loss correction strategy and Y as OOD distance metric.


descent (SGD) as the optimizer with a momentum of 0.9 and a weight decay of 1 _×_ 10 _[−]_ [4] . We
tune the hyperparameters _λ_ from the set of values _{_ 0 _._ 0001 _,_ 0 _._ 0005 _,_ 0 _._ 001 _,_ 0 _._ 005 _,_ 0 _._ 1 _}_ and _p_ from
_{_ 0 _,_ 0 _._ 3 _,_ 0 _._ 5 _}_ . For the NOODLE approach, we consider different options for loss correction strategies
such as CM and SCE. In terms of distance metrics in NOODLE approach, we consider both _k_ NN
and Mahalanobis scores as OOD detection metrics. We present the best performing variants of the
NOODLE approach in the main result tables, yet present the detailed ablation study across different
combinations of loss correction and distance metrics in the later sections. We evaluate the OOD
detection performance using two widely recognized metrics, i.e., the false positive rate at 95% true
positive rate (FPR95) and area under the receiver operating characteristic curve (AUROC). We also
present the ID accuracy (ID Acc.) under different settings in the supplementary section in Tables
3-6.


**Results.** Table 1 presents the OOD detection performance of the baselines and our method under
symmetric label noise across different noise rates for CIFAR-10 dataset. We can observe that OOD
detection baselines that lack label noise-robust training strategy are significantly impacted by high
levels of label noise. In contrast, the label noise-robust approaches, especially those based on probabilistic modeling such as CM and VolMinNet maintain relatively strong performance under noisy
conditions. Notably, our proposed method, NOODLE, consistently outperforms all other approaches
under test in terms of both average FPR95 and AUROC. Our approach is particularly effective at
higher noise rates. For example, at 50% noise rate, NOODLE achieves the best performance, with
an average FPR95 of 27.34% which represents a reduction in FPR95 of up to 12.5% compared to
the best baseline method.


Table 2 presents the OOD detection performance on the real noise datasets which are annotated by
unreliable crowd workers. For CIFAR-10N “worst” noise level (the noise rate is about 40.21%),
NOODLE achieves a 9.5% reduction in FPR95 and a 1.07% increase in AUROC compared to the
best performing baseline SNN. Across other noise settings of CIFAR-10N as well, NOODLE consistently outperforms all baselines. A similar trend is observed on the Animal-10N dataset: while label
noise-robust methods such as SCE, CM, and VolMinNet improve FPR95 over other non-robust techniques, NOODLE achieves an additional reduction of approximately 14% compared to the closest
baseline. For CIFAR-100N which is more challenging under noisy settings, NOODLE substantially
outperforming all the baselines with an improvement of about 15% in FPR95 compared to the best
performing baseline. More experiment results are presented in the supplementary section.


9


(a) (b) (c)


Figure 5: Comparison of OOD detection performance (FPR95 _↓_ ) on Animal-10N and CIFAR-100N
datasets. (a) CM vs NOODLE with KNN metric (b) SCE vs NOODLE with KNN metric (c) Different Mahalanobis variants of NOODLE. Here “NOODLE (X, Y)” refers to NOODLE with X as loss
correction strategy and Y as OOD distance metric.


**ID** **vs** **OOD** **Features.** From Fig. 1, it is evident that higher noise levels distort the feature space,
leading to less compact and more intermixed clusters. This feature distortion degrades the overall OOD detection performance, as we see in our experiments. To illustrate how label noise-robust
methods mitigate this issue, we examine the UMAP visualizations in Fig. 2 where the learned features _**h**_ ( _**x**_ ) of the test data for both ID and OOD samples are presented. Here, we compare the feature
representations from one of the competing baseline, i.e., CM and our approach NOODLE. For CM,
one can observe that cluster distortions are mitigated compared to the scenarios as in Fig. 1, showing that loss correction strategy helps in feature cleaning to some extend. Nonetheless, ID and OOD
sample mixups are still present significantly, especially near the boundaries. In contrast, NOODLE
produces more compact and well-separated clusters even under high noise rates with much reduced
feature mix-up between ID and OOD samples. We also observe the effectiveness of the proposed
OOD-ness score in Fig. 3. The “OOD-ness” score _o_ ( _**x**_ _n_ ) clearly differentiates the samples that are
more ID-like from those that are not in Fig. 3a. In addition, as shown in Fig. 3b, the average FPR
decreases across different real label noise settings when applying the sample selection strategy based
on the learned scores _o_ ( _**x**_ _n_ ).


**Ablation Study.** Here, we provide a detailed breakdown of each component of the NOODLE framework by studying the effect of the low-rank and sparse decomposition module in the NOODLE
framework under different loss correction strategies and OOD detection metrics. Specifically, we
analyze CM and SCE strategies for loss correction, and _k_ NN and Mahalanobis scores for the feature
distance-based OOD detection. Fig. 4a shows how the NOODLE variant with CM as loss correction
and _k_ NN as the distance metric substantially advance the OOD detection performance for CIFAR10N dataset. The SCE variant of the NOODLE version is also reasonably good, yet CM variant
(i.e., NOODLE(CM, KNN)) performs much better in all scenarios in CIFAR-10N. For example, in
worst case noise version, NOODLE (CM, KNN) reduces FPR95 to 27.94% from 32.48% by KNN,
showing its robustness even in challenging settings. We can also observe similar improvement in
performance in Fig. 4b, where CM-Mahalanobis variant of the NOODLE also exhibits impressive
OOD detection performance across scenarios. These results suggest that under different strategies of
loss correction and various OOD detection metrics, the feature cleansing strategy of the NOODLE is
effective in improving the ID-OOD separability. We also present similar analysis for other datasets
such as Animal-10N and CIFAR-100N in Fig. 5. In CIFAR-100N dataset, the SCE-Mahalanobis
variant of NOODLE achieves the best performance, likely because estimating transition matrices
for CM-based methods becomes increasingly difficult as the number of classes grows. Nevertheless,
our key idea of feature cleaning via low-rank sparse decomposition consistently enhances performance across different settings.
5 CONCLUSION


In this work, we introduce a novel framework for OOD detection under noisy labels that addresses
the limitations of existing methods by correcting label noise and enhancing OOD performance.
By leveraging low-rank ID feature representations, a carefully designed learning criterion, and robust OOD distance metric, our approach provides greater flexibility and effectiveness in improving
ID–OOD separability, even in highly noisy settings. Experimental results across multiple benchmarks and challenging OOD scenarios demonstrate the superiority of our method, highlighting its
ability to tackle the challenging problem of OOD detection under noisy labels.


10


REFERENCES


Devansh Arpit, Stanisław Jastrzundefinedbski, Nicolas Ballas, David Krueger, Emmanuel Bengio,
Maxinder S. Kanwal, Tegan Maharaj, Asja Fischer, Aaron Courville, Yoshua Bengio, and Simon
Lacoste-Julien. A closer look at memorization in deep networks. In _Proceedings of International_
_Conference on Machine Learning_, pp. 233–242, 2017.


Michael Buhrmester, Tracy Kwang, and Samuel D Gosling. Amazon’s mechanical turk: A new
source of inexpensive, yet high-quality data? 2016.


Emmanuel J. Candes, Xiaodong Li, Yi Ma, and John Wright. Robust principal component analysis?,
2009. [URL https://arxiv.org/abs/0912.3599.](https://arxiv.org/abs/0912.3599)


Emmanuel J Cand`es, Xiaodong Li, Yi Ma, and John Wright. Robust principal component analysis?
_Journal of the ACM (JACM)_, 58(3):1–37, 2011.


Jiacheng Cheng, Tongliang Liu, Kotagiri Ramamohanarao, and Dacheng Tao. Learning with
bounded instance and label-dependent label noise. In _Proceedings_ _of_ _International_ _Conference_
_on Machine Learning_, volume 119, pp. 1789–1799, 2020.


Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy Mohamed, and Andrea Vedaldi. Describing textures in the wild. _CoRR_, abs/1311.3618, 2013. [URL http://arxiv.org/abs/1311.3618.](http://arxiv.org/abs/1311.3618)


Xin Dong, Junfeng Guo, Ang Li, Wei-Te Ting, Cong Liu, and HT Kung. Neural mean discrepancy for efficient out-of-distribution detection. In _Proceedings of the IEEE/CVF Conference on_
_Computer Vision and Pattern Recognition_, pp. 19217–19227, 2022.


Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are we ready for autonomous driving? the kitti
vision benchmark suite. In _2012_ _IEEE_ _conference_ _on_ _computer_ _vision_ _and_ _pattern_ _recognition_,
pp. 3354–3361. IEEE, 2012.


Soumya Suvra Ghosal, Yiyou Sun, and Yixuan Li. How to overcome curse-of-dimensionality for
out-of-distribution detection? _Proceedings of the AAAI Conference on Artificial Intelligence_, 38
(18):19849–19857, Mar. 2024.


Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial
examples. _arXiv preprint arXiv:1412.6572_, 2014.


Ming Gu. Subspace iteration randomization and singular value problems. _SIAM Journal on Scientific_
_Computing_, 37(3):A1139–A1173, 2015.


Bo Han, Quanming Yao, Xingrui Yu, Gang Niu, Miao Xu, Weihua Hu, Ivor Tsang, and Masashi
Sugiyama. Co-teaching: Robust training of deep neural networks with extremely noisy labels.
_Advances in neural information processing systems_, 31, 2018a.


Bo Han, Quanming Yao, Xingrui Yu, Gang Niu, Miao Xu, Weihua Hu, Ivor Tsang, and Masashi
Sugiyama. Co-teaching: Robust training of deep neural networks with extremely noisy labels,
2018b. [URL https://arxiv.org/abs/1804.06872.](https://arxiv.org/abs/1804.06872)


Dan Hendrycks and Kevin Gimpel. A baseline for detecting misclassified and out-of-distribution
examples in neural networks. _arXiv preprint arXiv:1610.02136_, 2016.


Dan Hendrycks and Kevin Gimpel. A baseline for detecting misclassified and out-of-distribution
examples in neural networks. In _5th International Conference on Learning Representations, ICLR_
_2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings_ . OpenReview.net, 2017.
[URL https://openreview.net/forum?id=Hkg4TI9xl.](https://openreview.net/forum?id=Hkg4TI9xl)


Dan Hendrycks, Steven Basart, Mantas Mazeika, Andy Zou, Joseph Kwon, Mohammadreza Mostajabi, Jacob Steinhardt, and Dawn Song. Scaling out-of-distribution detection for real-world settings. In _Proceedings_ _of_ _the_ _39th_ _International_ _Conference_ _on_ _Machine_ _Learning_ _(ICML)_, pp.
8759–8773. PMLR, 2022. [URL https://proceedings.mlr.press/v162/hendrycks22a.html.](https://proceedings.mlr.press/v162/hendrycks22a.html)


Gao Huang, Zhuang Liu, and Kilian Q. Weinberger. Densely connected convolutional networks.
_CoRR_, abs/1608.06993, 2016. [URL http://arxiv.org/abs/1608.06993.](http://arxiv.org/abs/1608.06993)


11


Galadrielle Humblot-Renaux, Sergio Escalera, and Thomas B Moeslund. A noisy elephant in the
room: Is your out-of-distribution detector robust to label noise? In _Proceedings of the IEEE/CVF_
_Conference on Computer Vision and Pattern Recognition_, pp. 22626–22636, 2024.


Shahana Ibrahim, Tri Nguyen, and Xiao Fu. Deep learning from crowdsourced labels: Coupled cross-entropy minimization, identifiability, and regularization. In _Proceedings_ _of_ _Interna-_
_tional_ _Conference_ _on_ _Learning_ _Representations_, 2023. URL [https://openreview.net/forum?id=](https://openreview.net/forum?id=_qVhsWyWB9)

[qVhsWyWB9.](https://openreview.net/forum?id=_qVhsWyWB9)


Lu Jiang, Zhengyuan Zhou, Thomas Leung, Li-Jia Li, and Li Fei-Fei. Mentornet: Learning datadriven curriculum for very deep neural networks on corrupted labels. In _International conference_
_on machine learning_, pp. 2304–2313. PMLR, 2018.


Alex Krizhevsky. Learning multiple layers of features from tiny images. 2009. URL [https://api.](https://api.semanticscholar.org/CorpusID:18268744)
[semanticscholar.org/CorpusID:18268744.](https://api.semanticscholar.org/CorpusID:18268744)


Kimin Lee, Kibok Lee, Honglak Lee, and Jinwoo Shin. A simple unified framework for detecting
out-of-distribution samples and adversarial attacks. In _Advances in Neural Information Process-_
_ing Systems (NeurIPS)_ . Curran Associates, Inc., 2018.


Junnan Li, Richard Socher, and Steven C. H. Hoi. Dividemix: Learning with noisy labels as semisupervised learning, 2020a. [URL https://arxiv.org/abs/2002.07394.](https://arxiv.org/abs/2002.07394)


Junnan Li, Richard Socher, and Steven CH Hoi. Dividemix: Learning with noisy labels as semisupervised learning. _arXiv preprint arXiv:2002.07394_, 2020b.


Xuefeng Li, Tongliang Liu, Bo Han, Gang Niu, and Masashi Sugiyama. Provably end-to-end labelnoise learning without anchor points. In _International Conference on Machine Learning_, 2021a.
[URL https://api.semanticscholar.org/CorpusID:231802306.](https://api.semanticscholar.org/CorpusID:231802306)


Xuefeng Li, Tongliang Liu, Bo Han, Gang Niu, and Masashi Sugiyama. Provably end-to-end labelnoise learning without anchor points. In _Proceedings_ _of_ _International_ _Conference_ _on_ _Machine_
_Learning_, pp. 6403–6413, 2021b.


Shiyu Liang, Yixuan Li, and R. Srikant. Enhancing the reliability of out-of-distribution image detection in neural networks. In _International Conference on Learning Representations (ICLR)_, 2018.
[URL https://openreview.net/forum?id=H1VGkIxRZ.](https://openreview.net/forum?id=H1VGkIxRZ)


Tongliang Liu and Dacheng Tao. Classification with noisy labels by importance reweighting. _IEEE_
_Transactions on Pattern Analysis and Machine Intelligence_, 38:447–461, 2016.


Weitang Liu, Xiaoyun Wang, John Douglas Owens, and Yixuan Li. Energy-based out-ofdistribution detection. _ArXiv_, abs/2010.03759, 2020. URL [https://api.semanticscholar.org/](https://api.semanticscholar.org/CorpusID:222208700)
[CorpusID:222208700.](https://api.semanticscholar.org/CorpusID:222208700)


Yueming Lyu and Ivor W Tsang. Curriculum loss: Robust learning and generalization against label
corruption. _arXiv preprint arXiv:1905.10045_, 2019.


Yifei Ming, Yiyou Sun, Ousmane Dia, and Yixuan Li. How to exploit hyperspherical embeddings
for out-of-distribution detection?, 2023. [URL https://arxiv.org/abs/2203.04450.](https://arxiv.org/abs/2203.04450)


Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, and Andrew Y. Ng. Reading
digits in natural images with unsupervised feature learning. In _NIPS Workshop on Deep Learning_
_and_ _Unsupervised_ _Feature_ _Learning_ _2011_, 2011. URL [http://ufldl.stanford.edu/housenumbers/](http://ufldl.stanford.edu/housenumbers/nips2011_housenumbers.pdf)
nips2011 ~~h~~ [ousenumbers.pdf.](http://ufldl.stanford.edu/housenumbers/nips2011_housenumbers.pdf)


Duc Tam Nguyen, Chaithanya Kumar Mummadi, Thi Phuong Nhung Ngo, Thi Hoai Phuong
Nguyen, Laura Beggel, and Thomas Brox. Self: Learning to filter noisy labels with selfensembling. _arXiv preprint arXiv:1910.01842_, 2019.


Junting Pan and Xavier Gir´o-i-Nieto. End-to-end convolutional network for saliency prediction.
_CoRR_, abs/1507.01422, 2015. [URL http://arxiv.org/abs/1507.01422.](http://arxiv.org/abs/1507.01422)


12


Giorgio Patrini, Alessandro Rozza, Aditya Krishna Menon, Richard Nock, and Lizhen Qu. Making
deep neural networks robust to label noise: A loss correction approach. In _Proceedings_ _of_ _the_
_IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_, July 2017.


Vladimir Rokhlin, Arthur Szlam, and Mark Tygert. A randomized algorithm for principal component analysis. _SIAM Journal on Matrix Analysis and Applications_, 31(3):1100–1124, 2010.


Thomas Schlegl, Philipp Seeb¨ock, Sebastian M. Waldstein, Ursula Schmidt-Erfurth, and Georg
Langs. Unsupervised anomaly detection with generative adversarial networks to guide marker
discovery. In _Information_ _Processing_ _in_ _Medical_ _Imaging_, pp. 146–157. Springer International
Publishing, 2017.


Vikash Sehwag, Mung Chiang, and Prateek Mittal. SSD: A unified framework for self-supervised
outlier detection. _CoRR_, abs/2103.12051, 2021. [URL https://arxiv.org/abs/2103.12051.](https://arxiv.org/abs/2103.12051)


Hwanjun Song, Minseok Kim, and Jae-Gil Lee. SELFIE: Refurbishing unclean samples for robust
deep learning. In _ICML_, 2019.


Hwanjun Song, Minseok Kim, Dongmin Park, Yooju Shin, and Jae-Gil Lee. Learning from noisy
labels with deep neural networks: A survey. _IEEE transactions on neural networks and learning_
_systems_, 34(11):8135–8153, 2022.


Yiyou Sun and Yixuan Li. Dice: Leveraging sparsification for out-of-distribution detection. In
_Computer Vision – ECCV 2022_, pp. 691–708. Springer Nature Switzerland, 2022.


Yiyou Sun, Chuan Guo, and Yixuan Li. React: Out-of-distribution detection with rectified activations. In _Advances_ _in_ _Neural_ _Information_ _Processing_ _Systems_ _(NeurIPS)_, pp. 144–157. Curran
Associates, Inc., 2021.


Yiyou Sun, Yifei Ming, Xiaojin Zhu, and Yixuan Li. Out-of-distribution detection with deep nearest
neighbors. In _Proceedings_ _of_ _the_ _39th_ _International_ _Conference_ _on_ _Machine_ _Learning_ _(ICML)_,
pp. 20827–20840. PMLR, 2022.


Ryutaro Tanno, Ardavan Saeedi, Swami Sankaranarayanan, Daniel C Alexander, and Nathan Silberman. Learning from noisy labels by regularized estimation of annotator confusion. In _Proceedings_
_of the IEEE/CVF conference on computer vision and pattern recognition_, pp. 11244–11253, 2019.


Yilun Wang, Junfeng Yang, Wotao Yin, and Yin Zhang. A new alternating minimization algorithm
for total variation image reconstruction. _SIAM_ _J._ _Imaging_ _Sciences_, 1:248–272, 01 2008. doi:
10.1137/080724265.


Yisen Wang, Xingjun Ma, Zaiyi Chen, Yuan Luo, Jinfeng Yi, and James Bailey. Symmetric cross
entropy for robust learning with noisy labels. In _Proceedings_ _of_ _the_ _IEEE/CVF_ _international_
_conference on computer vision_, pp. 322–330, 2019a.


Yisen Wang, Xingjun Ma, Zaiyi Chen, Yuan Luo, Jinfeng Yi, and James Bailey. Symmetric cross
entropy for robust learning with noisy labels, 2019b. [URL https://arxiv.org/abs/1908.06112.](https://arxiv.org/abs/1908.06112)


Jiaheng Wei, Zhaowei Zhu, Hao Cheng, Tongliang Liu, Gang Niu, and Yang Liu. Learning with
noisy labels revisited: A study using real-world human annotations. In _International Conference_
_on Learning Representations_, 2022. [URL https://openreview.net/forum?id=TBWA6PLJZQm.](https://openreview.net/forum?id=TBWA6PLJZQm)


Xiaobo Xia, Tongliang Liu, Bo Han, Nannan Wang, Mingming Gong, Haifeng Liu, Gang Niu,
Dacheng Tao, and Masashi Sugiyama. Part-dependent label noise: Towards instance-dependent
label noise. In _Advances in Neural Information Processing Systems_, volume 33, pp. 7597–7610,
2020.


Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms. _CoRR_, abs/1708.07747, 2017. [URL http://arxiv.org/abs/1708.](http://arxiv.org/abs/1708.07747)
[07747.](http://arxiv.org/abs/1708.07747)


Jingkang Yang, Kaiyang Zhou, Yixuan Li, and Ziwei Liu. Generalized out-of-distribution detection:
A survey. _International Journal of Computer Vision_, 132(12):5635–5662, 2024.


13


Shuo Yang, Erkun Yang, Bo Han, Yang Liu, Min Xu, Gang Niu, and Tongliang Liu. Estimating
instance-dependent Bayes-label transition matrix using a deep neural network. In _Proceedings of_
_International Conference on Machine Learning_, 2021.


Fisher Yu, Ari Seff, Yinda Zhang, Shuran Song, Thomas Funkhouser, and Jianxiong Xiao. Lsun:
Construction of a large-scale image dataset using deep learning with humans in the loop, 2016.
[URL https://arxiv.org/abs/1506.03365.](https://arxiv.org/abs/1506.03365)


Xingrui Yu, Bo Han, Jiangchao Yao, Gang Niu, Ivor Tsang, and Masashi Sugiyama. How does disagreement help generalization against label corruption? In _International conference on machine_
_learning_, pp. 7164–7173. PMLR, 2019.


Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding
deep learning requires rethinking generalization. In _Proceedings of International Conference on_
_Learning Representations_, 2016.


Chunjie Zhang, Jing Liu, Qi Tian, Changsheng Xu, Hanqing Lu, and Songde Ma. Image classification by non-negative sparse coding, low-rank and sparse decomposition. In _CVPR_ _2011_, pp.
1673–1680. IEEE, 2011.


Zhilu Zhang and Mert Sabuncu. Generalized cross entropy loss for training deep neural networks
with noisy labels. _Advances in neural information processing systems_, 31, 2018a.


Zhilu Zhang and Mert R. Sabuncu. Generalized cross entropy loss for training deep neural networks
with noisy labels, 2018b. [URL https://arxiv.org/abs/1805.07836.](https://arxiv.org/abs/1805.07836)


Bolei Zhou, Aditya Khosla, Agata Lapedriza, Antonio Torralba, and Aude Oliva. [`] Places: An image
database for deep scene understanding. _CoRR_, abs/1610.02055, 2016. [URL http://arxiv.org/abs/](http://arxiv.org/abs/1610.02055)
[1610.02055.](http://arxiv.org/abs/1610.02055)


14


**Supplementary Material of “Tackling the Noisy Elephant in the Room: Label Noise-robust**
**Out-of-Distribution Detection via Loss Correction and Low-rank Decomposition”**


A NOTATION


We use the following notation throughout the paper: _x_,,, and **X** represent a scalar, a vector, a
matrix, and a tensor, respectively. Both _xi_ and [ _**x**_ ] _i_ denote the _i_ th entry of the vector . [ _**X**_ ] _i,j_ denote
the ( _i, j_ )th entry of the matrix _**X**_ . _**x**_ _i_ denotes the _i_ th row of the matrix _**X**_ ; [ _I_ ] means an integer
set _{_ 1 _,_ 2 _, . . ., I}_ . denote transpose. _**X**_ _≥_ **0** implies that all the entries of the matrix _**X**_ are nonnegative. I[ _A_ ] denotes an indicator function for the event _A_ such that I[ _A_ ] = 1 if the event _A_
happens, otherwise I[ _A_ ] = 0. CE( _**x**_ _, y_ ) = _−_ [�] _[K]_ _k_ =1 [I][[] _[y]_ [=] _[k]_ [] log(] _**[x]**_ [(] _[k]_ [))] [denotes] [the] [cross] [entropy]
function. _**I**_ denotes an identity matrix of appropriate size. **1** _K_ denotes an all-one vector of size _K_ .
_|C|_ denotes the cardinality of the set _C._ ∆ _[K]_ denotes a ( _K −_ 1)-dimensional probability simplex such
that ∆ _[K]_ = _{_ _**u**_ _∈_ R _[K]_ _|_ _**u**_ _≥_ **0** _,_ **1** _[⊤]_ _**u**_ = 1 _}_ .


B ALGORITHM DESCRIPTION


In this section, we present the NOODLE algorithm. Algorithm 1 provides the complete, step-by-step
procedure of our approach using the transition matrix-based loss correction strategy. As discussed
earlier, we obtain ID features via a low-rank sparse decomposition. The decomposition routine is
detailed in Algorithm 2.


**Algorithm 1** Proposed approach NOODLE

**Input:** Noisily labeled data _{_ ( _**x**_ _n,_   - _yn_ ) _}_ _[N]_ _n_ =1 [, where] _**[ x]**_ _[n]_ _[∈X]_ [,][ �] _[y][n]_ _[∈Y]_ [,] _[ n]_ [iter][, stopping criterion,] _[K]_
as number of classes
**Output:** Estimated parameters _**θ**_ and _**T**_
1: Initialize Transition Matrix _**T**_ to identity matrices _**I**_ _K_
2: Initialize the parameters _**θ**_ of the neural network function class _F_
3: **while** stopping criterion is not reached **do**
4: **while** stopping criterion is not reached **do**
5: Draw a random batch _B_
6: _**HB**_ _←_ [ _**h**_ ( _**x**_ 1) _, . . .,_ _**h**_ ( _**x**_ _B_ ) ] _// penultimate layer features from batch B where B denotes_
_the batch size (also see Sec. 3_ )
7: _**Q**_ _←_ ApproxTopKSingularVectors( _**HB**_ _, K, n_ iter)
8: _**H**_ ID _←_ ( _**QQ**_ _[⊤]_ ) _**HB**_
9: _**H**_ OOD _←_ _**HB**_ _−_ _**H**_ ID
10: Compute _∇LF_ ( _**T**_ _, B,_ _**H**_ _OOD_ ))
11: _**T**_ _,_ _**θ**_ _←_ SGDOptimizer( _**T**_ _, ∇LF_ ( _**T**_ _, B,_ _**H**_ OOD))
12: **end while**
13: **end while**


C MORE EXPERIMENT RESULTS


In this section, we present more detailed evaluations. While the summary results for CIFAR-10N
were reported earlier, we now provide dataset-wise OOD performance along with ID accuracy in
Table 3 and Table 4. To ensure fairness, all post-hoc methods are evaluated using the same encoder
trained with cross-entropy loss, thereby avoiding any bias in performance analysis. For CIDER
and SSD+, we follow prior work but replace their default ResNet-18 encoder with DenseNet-101
for consistency. As a result, these methods may require additional fine-tuning to fully realize their
potential. For Animal-10N, the dataset-specific results are reported in Table 5. We find that most
baseline methods struggle to achieve a good balance between ID accuracy and OOD detection. In
contrast, our proposed method NOODLE delivers consistently strong results across both metrics.


Finally, detailed results on CIFAR-100 are shown in Table 6. As expected, CIFAR-100 is considerably more challenging, leading to significant performance degradation for most baselines. Nevertheless, NOODLE achieves the best OOD detection performance while maintaining a competitive


15


**Algorithm 2** ApproxTopKSingularVectors


**Input:** Feature matrix _**HB**_ _∈_ R _[B][×][D]_, target rank _K_, number of iterations _n_ iter
**Output:** Orthonormal matrix _**Q**_ _∈_ R _[D][×][K]_ spanning the approximate top- _K_ right singular vectors of _**HB**_
1: Randomly initialize _**Q**_ _∈_ R _[D][×][K]_ _// D: feature dimension, K: target rank_
2: **for** _i_ = 1 to _n_ iter **do**
3: _**Z**_ _←_ _**HB**_ _[⊤]_ ( _**HBQ**_ ) _// project_ _**Q**_ _into column space of_ _**HB**_
4: _**Q**_ _←_ QRDecomposition( _**Z**_ ) _// obtain orthonormal basis of_ _**Z**_ _’s column space_
5: **end for**
6: **return** _**Q**_ _// spans approximate top-K_ _right singular vectors of_ _**HB**_


Table 3: OOD detection performance (FPR95 _↓_ / AUROC _↑_ ) on CIFAR-10 under different noise
settings using a DenseNet-100 encoder.


**Method** SVHN FashionMNIST LSUN iSUN Texture Places365 Average **ID Acc.**


FPR95 _↓_ AUROC _↑_ FPR95 _↓_ AUROC _↑_ FPR95 _↓_ AUROC _↑_ FPR95 _↓_ AUROC _↑_ FPR95 _↓_ AUROC _↑_ FPR95 _↓_ AUROC _↑_ FPR95 _↓_ AUROC _↑_


**Clean**


KNN 10 _._ 25 98 _._ 26 10 _._ 95 98 _._ 03 13 _._ 21 97 _._ 64 17 _._ 05 96 _._ 87 25 _._ 30 95 _._ 31 49 _._ 58 88 _._ 70 21 _._ 06 95 _._ 80 93 _._ 32
MSP 72 _._ 54 87 _._ 37 49 _._ 86 92 _._ 74 34 _._ 70 95 _._ 33 46 _._ 04 93 _._ 34 68 _._ 51 85 _._ 16 66 _._ 90 86 _._ 46 56 _._ 43 90 _._ 07 93 _._ 32
ODIN 55 _._ 88 89 _._ 16 16 _._ 01 97 _._ 23 3 _._ 01 99 _._ 12 8 _._ 47 98 _._ 20 60 _._ 46 82 _._ 84 54 _._ 77 88 _._ 30 33 _._ 10 92 _._ 47 93 _._ 32
Energy 73 _._ 05 87 _._ 23 15 _._ 64 97 _._ 18 4 _._ 44 98 _._ 86 23 _._ 22 96 _._ 16 67 _._ 02 83 _._ 48 51 _._ 52 89 _._ 30 39 _._ 15 92 _._ 03 93 _._ 32
ReAct 97 _._ 03 61 _._ 00 44 _._ 59 93 _._ 52 28 _._ 01 95 _._ 84 41 _._ 26 93 _._ 42 88 _._ 14 68 _._ 86 62 _._ 85 86 _._ 07 60 _._ 31 83 _._ 12 93 _._ 32
Mahalanobis 4 _._ 51 99 _._ 13 2 _._ 47 99 _._ 31 0 _._ 63 99 _._ 75 14 _._ 79 97 _._ 38 22 _._ 46 95 _._ 08 69 _._ 08 82 _._ 08 18 _._ 99 95 _._ 45 93 _._ 32
CIDER 18 _._ 61 96 _._ 78 27 _._ 17 95 _._ 35 31 _._ 63 94 _._ 34 42 _._ 00 93 _._ 82 56 _._ 65 89 _._ 58 59 _._ 95 85 _._ 06 39 _._ 33 92 _._ 49 90 _._ 51
SSD+ 31 _._ 05 92 _._ 81 15 _._ 85 96 _._ 54 19 _._ 28 95 _._ 86 17 _._ 41 95 _._ 87 64 _._ 27 85 _._ 06 54 _._ 11 88 _._ 39 33 _._ 66 92 _._ 42 94 _._ 03
SNN 8 _._ 68 98 _._ 35 21 _._ 49 96 _._ 22 9 _._ 22 98 _._ 42 19 _._ 46 96 _._ 72 26 _._ 99 94 _._ 97 49 _._ 74 88 _._ 52 22 _._ 60 95 _._ 53 94 _._ 15
SCE 4 _._ 59 99 _._ 13 15 _._ 47 97 _._ 06 1 _._ 96 99 _._ 58 10 _._ 60 98 _._ 06 29 _._ 31 92 _._ 84 56 _._ 36 87 _._ 03 19 _._ 71 95 _._ 62 91 _._ 09
GCE 11 _._ 33 98 _._ 02 11 _._ 73 98 _._ 00 7 _._ 16 98 _._ 73 9 _._ 06 98 _._ 30 21 _._ 70 96 _._ 15 50 _._ 39 88 _._ 80 18 _._ 56 96 _._ 33 93 _._ 54
DivideMix 22 _._ 35 94 _._ 72 36 _._ 90 91 _._ 74 33 _._ 41 90 _._ 87 46 _._ 05 87 _._ 62 47 _._ 11 86 _._ 55 58 _._ 04 84 _._ 45 40 _._ 81 89 _._ 16 81 _._ 22
Co-teaching 56 _._ 55 74 _._ 88 99 _._ 90 36 _._ 86 95 _._ 18 58 _._ 31 96 _._ 28 61 _._ 89 55 _._ 83 78 _._ 99 94 _._ 03 48 _._ 08 81 _._ 94 58 _._ 15 86 _._ 22
CM 8 _._ 35 98 _._ 64 8 _._ 39 98 _._ 49 7 _._ 16 98 _._ 67 15 _._ 19 97 _._ 31 19 _._ 72 96 _._ 36 51 _._ 14 88 _._ 54 18 _._ 32 96 _._ 33 94 _._ 39
VolMinNet 2 _._ 45 99 _._ 48 13 _._ 06 97 _._ 57 5 _._ 96 98 _._ 93 6 _._ 25 98 _._ 85 15 _._ 96 97 _._ 07 46 _._ 30 89 _._ 42 15 _._ 00 96 _._ 89 94 _._ 56


NOODLE 3 _._ 06 99 _._ 35 5 _._ 44 98 _._ 86 4 _._ 92 99 _._ 06 9 _._ 73 98 _._ 03 15 _._ 48 96 _._ 80 48 _._ 96 88 _._ 72 14 _._ 60 96 _._ 81 98 _._ 37


**Noise = Worst**


KNN 9 _._ 17 98 _._ 28 27 _._ 89 94 _._ 95 15 _._ 76 96 _._ 98 38 _._ 41 92 _._ 03 36 _._ 21 90 _._ 94 67 _._ 46 84 _._ 15 32 _._ 48 92 _._ 89 80 _._ 79
MSP 56 _._ 74 87 _._ 97 50 _._ 54 89 _._ 33 38 _._ 83 93 _._ 72 62 _._ 58 87 _._ 03 78 _._ 16 76 _._ 44 74 _._ 04 78 _._ 44 60 _._ 15 85 _._ 49 80 _._ 79
ODIN 45 _._ 45 90 _._ 96 33 _._ 09 93 _._ 31 13 _._ 41 97 _._ 43 32 _._ 36 93 _._ 02 71 _._ 68 72 _._ 39 75 _._ 89 73 _._ 02 45 _._ 31 86 _._ 69 80 _._ 79
Energy 37 _._ 62 93 _._ 17 35 _._ 78 93 _._ 28 14 _._ 57 97 _._ 00 47 _._ 39 90 _._ 98 76 _._ 06 74 _._ 61 72 _._ 39 77 _._ 45 47 _._ 30 87 _._ 75 80 _._ 79
ReAct 70 _._ 94 82 _._ 60 52 _._ 84 90 _._ 05 36 _._ 40 93 _._ 87 68 _._ 31 81 _._ 53 88 _._ 92 54 _._ 21 77 _._ 90 70 _._ 87 65 _._ 89 78 _._ 86 80 _._ 79
Mahalanobis 4 _._ 41 98 _._ 97 19 _._ 95 96 _._ 33 10 _._ 05 98 _._ 07 30 _._ 50 92 _._ 65 33 _._ 69 89 _._ 11 87 _._ 96 60 _._ 38 31 _._ 09 89 _._ 25 80 _._ 79
CIDER 99 _._ 75 58 _._ 13 98 _._ 68 66 _._ 64 99 _._ 94 52 _._ 42 99 _._ 25 61 _._ 61 93 _._ 60 65 _._ 80 98 _._ 86 48 _._ 94 98 _._ 35 58 _._ 92 80 _._ 26
SSD+ 20 _._ 08 94 _._ 86 37 _._ 05 91 _._ 58 24 _._ 16 94 _._ 97 33 _._ 27 93 _._ 97 50 _._ 12 89 _._ 22 73 _._ 44 74 _._ 29 39 _._ 69 89 _._ 82 82 _._ 87
SNN 10 _._ 98 97 _._ 95 24 _._ 49 95 _._ 67 16 _._ 61 96 _._ 76 31 _._ 84 93 _._ 96 33 _._ 92 90 _._ 69 67 _._ 36 81 _._ 67 30 _._ 87 92 _._ 78 80 _._ 57
SCE 12 _._ 39 97 _._ 67 28 _._ 98 94 _._ 52 17 _._ 71 96 _._ 62 49 _._ 37 89 _._ 63 34 _._ 61 91 _._ 13 64 _._ 11 83 _._ 10 34 _._ 53 92 _._ 11 83 _._ 48
GCE 10 _._ 53 98 _._ 18 21 _._ 72 96 _._ 09 15 _._ 71 97 _._ 05 55 _._ 81 87 _._ 25 45 _._ 69 87 _._ 86 65 _._ 02 82 _._ 54 35 _._ 75 91 _._ 50 83 _._ 49
DivideMix 44 _._ 70 93 _._ 52 48 _._ 20 89 _._ 20 38 _._ 09 90 _._ 29 15 _._ 35 96 _._ 70 32 _._ 06 94 _._ 42 57 _._ 95 85 _._ 09 39 _._ 39 91 _._ 53 80 _._ 23
Co-teaching 50 _._ 04 85 _._ 37 99 _._ 95 29 _._ 04 99 _._ 61 58 _._ 63 96 _._ 88 58 _._ 36 52 _._ 15 82 _._ 65 94 _._ 22 50 _._ 03 82 _._ 14 60 _._ 68 86 _._ 34
CM 8 _._ 61 98 _._ 29 17 _._ 92 96 _._ 78 28 _._ 37 90 _._ 87 56 _._ 29 81 _._ 97 39 _._ 24 88 _._ 19 67 _._ 23 81 _._ 87 36 _._ 28 89 _._ 66 76 _._ 26
VolMinNet 7 _._ 58 98 _._ 33 42 _._ 88 90 _._ 42 19 _._ 70 95 _._ 50 43 _._ 71 89 _._ 08 40 _._ 90 87 _._ 46 70 _._ 64 77 _._ 54 37 _._ 57 89 _._ 72 76 _._ 30


NOODLE 3 _._ 17 99 _._ 47 21 _._ 57 96 _._ 21 7 _._ 57 98 _._ 03 28 _._ 09 97 _._ 28 30 _._ 52 92 _._ 53 61 _._ 75 84 _._ 71 25 _._ 45 94 _._ 71 83 _._ 72


**Noise = Aggre**


KNN 11 _._ 37 97 _._ 91 18 _._ 32 96 _._ 70 9 _._ 93 98 _._ 11 18 _._ 12 96 _._ 35 23 _._ 69 95 _._ 43 62 _._ 26 84 _._ 54 23 _._ 95 94 _._ 84 89 _._ 93
MSP 57 _._ 75 89 _._ 45 47 _._ 16 90 _._ 89 27 _._ 74 95 _._ 11 59 _._ 89 88 _._ 27 67 _._ 84 83 _._ 15 69 _._ 89 82 _._ 42 55 _._ 04 88 _._ 21 89 _._ 93
ODIN 49 _._ 75 91 _._ 54 30 _._ 78 94 _._ 69 9 _._ 29 98 _._ 25 33 _._ 50 92 _._ 13 65 _._ 18 82 _._ 01 74 _._ 50 75 _._ 54 43 _._ 83 89 _._ 03 89 _._ 93
Energy 66 _._ 66 88 _._ 00 40 _._ 23 93 _._ 66 18 _._ 07 97 _._ 10 64 _._ 95 88 _._ 41 71 _._ 46 81 _._ 96 74 _._ 80 78 _._ 26 56 _._ 03 87 _._ 90 89 _._ 93
ReAct 66 _._ 66 88 _._ 00 40 _._ 23 93 _._ 66 18 _._ 07 97 _._ 10 64 _._ 95 88 _._ 41 71 _._ 46 81 _._ 96 21 _._ 61 96 _._ 92 47 _._ 17 91 _._ 01 89 _._ 93
Mahalanobis 4 _._ 41 98 _._ 97 19 _._ 95 96 _._ 33 10 _._ 05 98 _._ 07 30 _._ 50 92 _._ 65 33 _._ 69 89 _._ 11 87 _._ 96 60 _._ 38 31 _._ 09 89 _._ 25 89 _._ 93
CIDER 82 _._ 88 71 _._ 63 99 _._ 61 32 _._ 85 48 _._ 08 84 _._ 89 94 _._ 50 65 _._ 86 79 _._ 66 70 _._ 30 96 _._ 50 50 _._ 70 83 _._ 54 62 _._ 71 89 _._ 06
SSD+ 63 _._ 39 76 _._ 39 34 _._ 70 90 _._ 34 23 _._ 16 94 _._ 58 43 _._ 26 90 _._ 42 36 _._ 14 90 _._ 22 73 _._ 66 74 _._ 43 45 _._ 72 86 _._ 07 90 _._ 66
SNN 6 _._ 15 98 _._ 90 14 _._ 36 97 _._ 33 8 _._ 61 98 _._ 32 35 _._ 76 92 _._ 37 26 _._ 12 94 _._ 45 59 _._ 86 83 _._ 74 25 _._ 14 94 _._ 18 90 _._ 00
SCE 4 _._ 32 99 _._ 14 19 _._ 12 96 _._ 41 4 _._ 92 98 _._ 98 19 _._ 90 96 _._ 39 24 _._ 98 94 _._ 82 63 _._ 95 82 _._ 82 22 _._ 87 94 _._ 76 90 _._ 45
GCE 1 _._ 94 99 _._ 64 11 _._ 17 98 _._ 11 7 _._ 57 98 _._ 58 14 _._ 44 97 _._ 36 28 _._ 63 94 _._ 62 52 _._ 86 87 _._ 84 19 _._ 43 96 _._ 03 91 _._ 55
DivideMix 77 _._ 88 85 _._ 56 84 _._ 07 74 _._ 54 48 _._ 41 87 _._ 26 59 _._ 81 78 _._ 07 49 _._ 49 86 _._ 58 74 _._ 63 78 _._ 99 65 _._ 72 81 _._ 83 73 _._ 44
Co-teaching 48 _._ 52 80 _._ 16 99 _._ 49 39 _._ 54 94 _._ 18 58 _._ 15 86 _._ 59 72 _._ 57 45 _._ 46 84 _._ 21 93 _._ 64 47 _._ 65 77 _._ 98 63 _._ 71 86 _._ 15
CM 7 _._ 82 98 _._ 68 13 _._ 01 97 _._ 55 8 _._ 41 98 _._ 30 12 _._ 70 97 _._ 62 22 _._ 45 95 _._ 18 62 _._ 91 83 _._ 70 21 _._ 22 95 _._ 17 78 _._ 82
VolMinNet 3 _._ 88 99 _._ 17 10 _._ 28 98 _._ 08 8 _._ 46 98 _._ 32 31 _._ 84 93 _._ 84 29 _._ 41 93 _._ 06 56 _._ 36 87 _._ 27 23 _._ 37 94 _._ 96 91 _._ 86


NOODLE 1 _._ 34 99 _._ 75 17 _._ 07 97 _._ 21 2 _._ 26 98 _._ 95 8 _._ 42 98 _._ 21 17 _._ 15 96 _._ 21 52 _._ 88 88 _._ 41 16 _._ 52 96 _._ 46 91 _._ 57


and balanced ID accuracy, highlighting its robustness under difficult conditions. We also provide
ablation study of hyperparameter _λ_ in Table 7. We see the more lower value of _λ_ FPR performance
is good.Therefore, we choose low value of _λ_ in our experimental settings.We present additional
ablation study on _K_ in Table 8 and _p_ in Table 10.We also provide additional baselines for CIFAR10N(worst noise case) in Table 9


16


Table 4: OOD detection performance (FPR95 _↓_ / AUROC _↑_ ) on CIFAR-10 using a DenseNet-100
encoder under **random1**, **random2**, and **random3** noise.


**Method** SVHN FashionMNIST LSUN iSUN Texture Places365 Average **ID Acc.**


FPR95 _↓_ AUROC _↑_ FPR95 _↓_ AUROC _↑_ FPR95 _↓_ AUROC _↑_ FPR95 _↓_ AUROC _↑_ FPR95 _↓_ AUROC _↑_ FPR95 _↓_ AUROC _↑_ FPR95 _↓_ AUROC _↑_


**Noise = random1**


KNN 26 _._ 55 95 _._ 11 32 _._ 22 94 _._ 66 19 _._ 98 96 _._ 43 36 _._ 27 92 _._ 58 30 _._ 12 93 _._ 01 67 _._ 75 84 _._ 12 35 _._ 48 92 _._ 65 88 _._ 03
MSP 63 _._ 91 89 _._ 44 56 _._ 80 88 _._ 35 33 _._ 35 94 _._ 64 70 _._ 25 83 _._ 87 70 _._ 71 81 _._ 70 69 _._ 47 81 _._ 73 60 _._ 75 86 _._ 62 88 _._ 03
ODIN 55 _._ 83 87 _._ 93 43 _._ 60 91 _._ 41 11 _._ 17 98 _._ 01 46 _._ 12 89 _._ 32 65 _._ 07 77 _._ 53 72 _._ 94 75 _._ 16 49 _._ 12 86 _._ 56 88 _._ 03
Energy 81 _._ 76 85 _._ 45 39 _._ 75 92 _._ 56 11 _._ 32 97 _._ 78 63 _._ 37 86 _._ 34 68 _._ 55 80 _._ 04 64 _._ 57 81 _._ 25 54 _._ 89 87 _._ 24 88 _._ 03
ReAct 97 _._ 67 60 _._ 00 57 _._ 46 89 _._ 81 28 _._ 41 95 _._ 32 72 _._ 90 78 _._ 42 86 _._ 65 62 _._ 40 69 _._ 53 75 _._ 58 68 _._ 77 76 _._ 92 88 _._ 03
Mahalanobis 16 _._ 86 96 _._ 22 48 _._ 89 88 _._ 49 13 _._ 62 97 _._ 36 50 _._ 63 82 _._ 90 36 _._ 24 87 _._ 86 92 _._ 57 50 _._ 57 43 _._ 14 83 _._ 90 88 _._ 03
CIDER 98 _._ 03 48 _._ 59 99 _._ 91 59 _._ 70 97 _._ 94 63 _._ 79 99 _._ 45 58 _._ 63 97 _._ 02 58 _._ 86 97 _._ 60 57 _._ 74 98 _._ 32 57 _._ 88 87 _._ 74
SSD+ 18 _._ 54 93 _._ 66 51 _._ 21 87 _._ 56 20 _._ 28 95 _._ 63 36 _._ 77 91 _._ 92 51 _._ 22 89 _._ 03 73 _._ 38 74 _._ 79 41 _._ 90 88 _._ 77 90 _._ 33
SNN 16 _._ 25 97 _._ 07 35 _._ 63 93 _._ 93 15 _._ 74 97 _._ 05 23 _._ 81 95 _._ 32 26 _._ 77 93 _._ 92 61 _._ 03 85 _._ 28 29 _._ 87 93 _._ 76 87 _._ 13
SCE 17 _._ 45 97 _._ 24 10 _._ 51 97 _._ 91 5 _._ 81 98 _._ 81 14 _._ 95 97 _._ 11 25 _._ 83 93 _._ 70 62 _._ 82 83 _._ 86 22 _._ 90 94 _._ 77 89 _._ 81
GCE 6 _._ 38 98 _._ 87 11 _._ 85 97 _._ 92 11 _._ 48 97 _._ 93 23 _._ 18 96 _._ 11 31 _._ 91 93 _._ 27 56 _._ 04 86 _._ 58 23 _._ 47 95 _._ 11 90 _._ 46
DivideMix 89 _._ 21 84 _._ 82 62 _._ 24 86 _._ 97 56 _._ 76 88 _._ 52 68 _._ 00 80 _._ 74 50 _._ 85 86 _._ 88 72 _._ 78 79 _._ 19 66 _._ 64 84 _._ 52 77 _._ 14
Co-teaching 51 _._ 32 81 _._ 34 49 _._ 00 83 _._ 00 52 _._ 00 87 _._ 00 23 _._ 18 96 _._ 11 74 _._ 58 69 _._ 89 94 _._ 74 62 _._ 07 53 _._ 42 74 _._ 23 86 _._ 44
CM 3 _._ 13 99 _._ 35 18 _._ 50 96 _._ 55 15 _._ 74 97 _._ 22 21 _._ 04 96 _._ 18 25 _._ 85 94 _._ 65 64 _._ 08 83 _._ 69 24 _._ 72 94 _._ 61 90 _._ 31
VolMinNet 5 _._ 48 99 _._ 01 15 _._ 74 97 _._ 11 6 _._ 92 98 _._ 75 22 _._ 86 95 _._ 86 30 _._ 30 94 _._ 07 56 _._ 10 86 _._ 06 22 _._ 90 95 _._ 15 90 _._ 53


NOODLE 3 _._ 21 99 _._ 47 15 _._ 65 97 _._ 21 5 _._ 23 98 _._ 93 5 _._ 03 98 _._ 85 12 _._ 79 97 _._ 69 58 _._ 03 86 _._ 35 16 _._ 66 96 _._ 42 90 _._ 36


**Noise = random2**


KNN 6 _._ 29 98 _._ 73 41 _._ 05 91 _._ 75 19 _._ 28 96 _._ 04 24 _._ 92 95 _._ 02 30 _._ 85 92 _._ 20 69 _._ 53 82 _._ 43 31 _._ 99 92 _._ 70 87 _._ 79
MSP 53 _._ 36 89 _._ 13 58 _._ 24 84 _._ 86 30 _._ 82 94 _._ 26 48 _._ 26 91 _._ 41 73 _._ 00 80 _._ 23 74 _._ 98 79 _._ 10 56 _._ 44 86 _._ 50 87 _._ 79
ODIN 44 _._ 99 87 _._ 97 56 _._ 14 83 _._ 96 13 _._ 74 97 _._ 24 19 _._ 78 94 _._ 78 68 _._ 03 73 _._ 23 77 _._ 58 68 _._ 27 46 _._ 71 84 _._ 24 87 _._ 79
Energy 60 _._ 00 87 _._ 44 50 _._ 75 87 _._ 08 17 _._ 62 96 _._ 63 29 _._ 00 93 _._ 97 71 _._ 31 76 _._ 10 73 _._ 20 75 _._ 42 50 _._ 31 86 _._ 11 87 _._ 79
ReAct 78 _._ 12 80 _._ 74 56 _._ 58 86 _._ 77 30 _._ 49 94 _._ 12 62 _._ 46 86 _._ 31 87 _._ 27 61 _._ 33 77 _._ 87 69 _._ 69 65 _._ 46 79 _._ 83 87 _._ 79
Mahalanobis 11 _._ 94 96 _._ 96 69 _._ 11 81 _._ 20 23 _._ 57 95 _._ 12 36 _._ 79 87 _._ 81 38 _._ 72 86 _._ 17 94 _._ 13 48 _._ 44 45 _._ 71 82 _._ 62 87 _._ 79
CIDER 96 _._ 75 67 _._ 67 99 _._ 94 45 _._ 22 99 _._ 59 46 _._ 57 99 _._ 68 67 _._ 00 91 _._ 21 60 _._ 02 98 _._ 38 59 _._ 07 97 _._ 59 57 _._ 59 87 _._ 53
SSD+ 16 _._ 20 96 _._ 64 34 _._ 41 91 _._ 92 18 _._ 91 95 _._ 97 36 _._ 97 93 _._ 74 54 _._ 53 86 _._ 97 75 _._ 17 77 _._ 90 39 _._ 37 90 _._ 52 90 _._ 27
SNN 4 _._ 84 99 _._ 05 42 _._ 45 92 _._ 36 19 _._ 44 96 _._ 04 17 _._ 68 96 _._ 06 33 _._ 90 91 _._ 82 66 _._ 09 82 _._ 28 30 _._ 74 92 _._ 94 88 _._ 11
SCE 6 _._ 17 98 _._ 78 15 _._ 97 97 _._ 16 13 _._ 15 97 _._ 50 24 _._ 60 95 _._ 04 25 _._ 41 94 _._ 60 61 _._ 19 82 _._ 42 24 _._ 42 94 _._ 25 89 _._ 72
GCE 2 _._ 02 99 _._ 57 18 _._ 91 96 _._ 46 5 _._ 57 98 _._ 84 9 _._ 06 98 _._ 09 19 _._ 47 96 _._ 02 55 _._ 29 86 _._ 18 18 _._ 89 95 _._ 86 90 _._ 29
DivideMix 79 _._ 28 78 _._ 13 68 _._ 73 80 _._ 40 55 _._ 05 86 _._ 76 26 _._ 58 94 _._ 70 51 _._ 37 87 _._ 91 74 _._ 59 77 _._ 40 59 _._ 27 84 _._ 22 78 _._ 30
Co-teaching 51 _._ 32 81 _._ 34 99 _._ 87 32 _._ 07 74 _._ 58 69 _._ 89 94 _._ 74 62 _._ 07 52 _._ 39 81 _._ 73 92 _._ 63 49 _._ 46 77 _._ 59 62 _._ 76 85 _._ 99
CM 1 _._ 83 99 _._ 62 16 _._ 84 97 _._ 00 17 _._ 64 96 _._ 72 18 _._ 08 96 _._ 81 27 _._ 96 94 _._ 45 59 _._ 38 85 _._ 66 23 _._ 62 95 _._ 04 90 _._ 29
VolMinNet 3 _._ 95 99 _._ 25 15 _._ 59 97 _._ 29 8 _._ 74 98 _._ 36 8 _._ 31 98 _._ 42 22 _._ 82 95 _._ 84 53 _._ 37 87 _._ 74 18 _._ 80 96 _._ 15 94 _._ 35


NOODLE 2 _._ 14 99 _._ 49 12 _._ 51 97 _._ 82 3 _._ 98 99 _._ 05 5 _._ 02 98 _._ 87 17 _._ 78 95 _._ 74 52 _._ 42 87 _._ 89 15 _._ 46 96 _._ 70 90 _._ 25


**Noise = random3**


KNN 8 _._ 79 98 _._ 41 29 _._ 45 93 _._ 81 12 _._ 45 97 _._ 66 28 _._ 16 94 _._ 63 25 _._ 80 94 _._ 26 58 _._ 99 85 _._ 76 27 _._ 27 94 _._ 09 87 _._ 77
MSP 50 _._ 92 90 _._ 23 44 _._ 39 89 _._ 13 25 _._ 79 95 _._ 34 69 _._ 78 81 _._ 85 67 _._ 02 81 _._ 86 65 _._ 51 81 _._ 76 53 _._ 90 86 _._ 70 87 _._ 77
ODIN 41 _._ 84 90 _._ 63 30 _._ 77 92 _._ 45 8 _._ 82 98 _._ 25 42 _._ 48 89 _._ 24 60 _._ 35 77 _._ 22 67 _._ 20 76 _._ 41 41 _._ 91 87 _._ 37 87 _._ 77
Energy 40 _._ 77 92 _._ 63 26 _._ 49 93 _._ 06 8 _._ 85 98 _._ 19 57 _._ 99 85 _._ 82 64 _._ 08 80 _._ 66 58 _._ 27 83 _._ 06 42 _._ 74 88 _._ 90 87 _._ 77
ReAct 77 _._ 62 77 _._ 76 34 _._ 33 92 _._ 52 16 _._ 16 96 _._ 96 67 _._ 01 79 _._ 29 85 _._ 32 62 _._ 89 66 _._ 42 77 _._ 99 57 _._ 81 81 _._ 24 87 _._ 77
Mahalanobis 7 _._ 87 98 _._ 31 26 _._ 49 94 _._ 43 7 _._ 01 98 _._ 64 54 _._ 91 82 _._ 04 28 _._ 95 90 _._ 66 87 _._ 15 57 _._ 60 35 _._ 40 86 _._ 95 87 _._ 77
CIDER+ 94 _._ 37 51 _._ 43 95 _._ 95 62 _._ 08 98 _._ 10 51 _._ 80 99 _._ 37 33 _._ 13 93 _._ 19 46 _._ 85 97 _._ 89 57 _._ 20 96 _._ 48 50 _._ 41 87 _._ 16
SSD+ 31 _._ 92 93 _._ 43 29 _._ 40 93 _._ 66 12 _._ 15 97 _._ 74 25 _._ 35 93 _._ 66 36 _._ 34 91 _._ 72 59 _._ 22 83 _._ 18 32 _._ 40 92 _._ 23 90 _._ 23
SNN 12 _._ 97 97 _._ 67 34 _._ 27 91 _._ 84 16 _._ 04 96 _._ 81 53 _._ 01 89 _._ 46 29 _._ 40 92 _._ 96 59 _._ 90 83 _._ 95 34 _._ 26 92 _._ 12 87 _._ 94
SCE 6 _._ 08 98 _._ 81 16 _._ 65 96 _._ 94 13 _._ 47 96 _._ 64 24 _._ 29 95 _._ 55 23 _._ 32 95 _._ 23 65 _._ 06 83 _._ 22 24 _._ 81 94 _._ 40 89 _._ 39
GCE 7 _._ 30 98 _._ 63 17 _._ 23 96 _._ 88 5 _._ 28 98 _._ 96 10 _._ 88 97 _._ 86 20 _._ 04 95 _._ 84 57 _._ 98 85 _._ 49 19 _._ 78 95 _._ 61 90 _._ 71
DivideMix 26 _._ 15 95 _._ 47 57 _._ 09 89 _._ 78 38 _._ 77 92 _._ 91 46 _._ 40 89 _._ 32 49 _._ 93 87 _._ 99 74 _._ 53 80 _._ 78 48 _._ 81 89 _._ 38 69 _._ 20
Co-teaching 50 _._ 65 80 _._ 80 99 _._ 96 23 _._ 83 99 _._ 55 56 _._ 65 95 _._ 49 58 _._ 16 52 _._ 45 80 _._ 56 93 _._ 54 48 _._ 86 81 _._ 94 58 _._ 15 86 _._ 53
CM 2 _._ 56 99 _._ 48 12 _._ 00 97 _._ 75 6 _._ 82 98 _._ 60 23 _._ 46 95 _._ 99 23 _._ 74 94 _._ 94 55 _._ 65 86 _._ 32 20 _._ 70 95 _._ 51 90 _._ 79
VolMinNet 9 _._ 80 98 _._ 06 10 _._ 10 98 _._ 02 7 _._ 61 98 _._ 40 24 _._ 25 95 _._ 63 25 _._ 74 94 _._ 40 55 _._ 65 85 _._ 58 22 _._ 19 95 _._ 02 94 _._ 36


NOODLE 2 _._ 31 99 _._ 41 12 _._ 31 97 _._ 82 3 _._ 34 99 _._ 25 5 _._ 26 98 _._ 79 17 _._ 56 95 _._ 86 52 _._ 15 87 _._ 86 15 _._ 49 96 _._ 50 90 _._ 37


Table 5: OOD detection performance (FPR95 _↓_ / AUROC _↑_ ) on Animal-10N with real noisy labels
using a DenseNet-100 encoder.


**Method** SVHN FashionMNIST LSUN iSUN DTD Places365 Average **ID Acc.**


FPR95 _↓_ AUROC _↑_ FPR95 _↓_ AUROC _↑_ FPR95 _↓_ AUROC _↑_ FPR95 _↓_ AUROC _↑_ FPR95 _↓_ AUROC _↑_ FPR95 _↓_ AUROC _↑_ FPR95 _↓_ AUROC _↑_


KNN 45 _._ 29 89 _._ 55 58 _._ 05 90 _._ 30 66 _._ 19 79 _._ 93 90 _._ 66 62 _._ 18 73 _._ 17 74 _._ 54 89 _._ 26 65 _._ 71 70 _._ 44 77 _._ 04 81 _._ 52
MSP 93 _._ 12 64 _._ 84 82 _._ 74 79 _._ 61 84 _._ 00 74 _._ 53 96 _._ 60 34 _._ 67 96 _._ 97 41 _._ 18 90 _._ 40 64 _._ 56 90 _._ 64 59 _._ 90 81 _._ 52
ODIN 79 _._ 29 64 _._ 67 51 _._ 20 88 _._ 26 48 _._ 52 87 _._ 84 96 _._ 52 34 _._ 97 95 _._ 43 36 _._ 66 90 _._ 85 62 _._ 89 76 _._ 97 62 _._ 55 81 _._ 52
Energy 81 _._ 91 74 _._ 54 66 _._ 32 88 _._ 64 31 _._ 94 94 _._ 83 96 _._ 32 45 _._ 16 89 _._ 02 69 _._ 24 88 _._ 09 74 _._ 71 75 _._ 60 74 _._ 52 81 _._ 52
ReAct 79 _._ 18 77 _._ 62 72 _._ 85 85 _._ 98 42 _._ 96 92 _._ 02 95 _._ 69 58 _._ 85 91 _._ 88 52 _._ 46 91 _._ 43 59 _._ 99 79 _._ 00 71 _._ 15 81 _._ 52
Mahalanobis 31 _._ 73 91 _._ 49 97 _._ 47 42 _._ 44 68 _._ 73 66 _._ 87 4 _._ 87 98 _._ 87 33 _._ 39 86 _._ 64 91 _._ 04 51 _._ 67 54 _._ 54 73 _._ 00 81 _._ 52
CIDER 78 _._ 00 82 _._ 28 74 _._ 75 83 _._ 81 51 _._ 20 87 _._ 60 99 _._ 00 36 _._ 84 95 _._ 13 78 _._ 33 74 _._ 75 81 _._ 70 82 _._ 64 75 _._ 09 81 _._ 85
SSD+ 62 _._ 40 82 _._ 49 59 _._ 80 84 _._ 02 40 _._ 96 87 _._ 82 97 _._ 62 36 _._ 93 76 _._ 10 78 _._ 53 59 _._ 80 81 _._ 90 66 _._ 11 75 _._ 28 82 _._ 06
SNN 41 _._ 40 91 _._ 28 11 _._ 58 97 _._ 79 12 _._ 04 97 _._ 63 39 _._ 68 91 _._ 61 29 _._ 22 93 _._ 29 54 _._ 67 90 _._ 27 31 _._ 43 93 _._ 65 81 _._ 52
SCE 29 _._ 06 90 _._ 96 41 _._ 71 93 _._ 05 27 _._ 00 94 _._ 63 9 _._ 45 98 _._ 39 29 _._ 59 94 _._ 10 55 _._ 03 89 _._ 71 31 _._ 97 93 _._ 47 81 _._ 22
GCE 28 _._ 98 91 _._ 90 43 _._ 05 91 _._ 28 23 _._ 75 94 _._ 14 27 _._ 02 95 _._ 91 30 _._ 18 92 _._ 52 66 _._ 74 84 _._ 15 36 _._ 62 91 _._ 65 80 _._ 86
DivideMix 16 _._ 15 96 _._ 44 19 _._ 84 96 _._ 28 18 _._ 42 95 _._ 98 60 _._ 56 86 _._ 34 40 _._ 14 85 _._ 92 50 _._ 52 89 _._ 62 34 _._ 27 91 _._ 77 79 _._ 63
Co-teaching 68 _._ 04 83 _._ 45 99 _._ 90 8 _._ 68 84 _._ 54 55 _._ 97 11 _._ 71 97 _._ 11 49 _._ 72 82 _._ 39 96 _._ 92 42 _._ 70 68 _._ 47 61 _._ 72 74 _._ 08
CM 25 _._ 84 95 _._ 25 15 _._ 41 97 _._ 22 13 _._ 42 97 _._ 44 63 _._ 61 84 _._ 11 37 _._ 06 90 _._ 88 45 _._ 65 91 _._ 58 33 _._ 50 92 _._ 75 82 _._ 48
VolMinNet 15 _._ 08 96 _._ 79 29 _._ 22 94 _._ 51 11 _._ 80 97 _._ 73 40 _._ 62 90 _._ 67 25 _._ 20 94 _._ 64 53 _._ 63 90 _._ 21 29 _._ 26 94 _._ 09 81 _._ 78


NOODLE 26 _._ 49 94 _._ 77 24 _._ 41 95 _._ 75 11 _._ 36 97 _._ 86 17 _._ 29 96 _._ 70 18 _._ 21 96 _._ 33 53 _._ 76 89 _._ 37 25 _._ 25 95 _._ 13 82 _._ 98


17


Table 6: OOD detection performance (FPR95 _↓_ / AUROC _↑_ ) on CIFAR-100 with real noisy labels
using a DenseNet-100 encoder.


**Method** SVHN FashionMNIST LSUN iSUN DTD Places365 Average **ID Acc.**


FPR95 _↓_ AUROC _↑_ FPR95 _↓_ AUROC _↑_ FPR95 _↓_ AUROC _↑_ FPR95 _↓_ AUROC _↑_ FPR95 _↓_ AUROC _↑_ FPR95 _↓_ AUROC _↑_ FPR95 _↓_ AUROC _↑_


KNN 11 _._ 08 97 _._ 63 42 _._ 68 92 _._ 88 28 _._ 07 93 _._ 32 53 _._ 09 81 _._ 81 32 _._ 73 91 _._ 70 91 _._ 56 61 _._ 90 43 _._ 20 86 _._ 54 52 _._ 48
MSP 86 _._ 65 75 _._ 40 76 _._ 23 81 _._ 21 52 _._ 72 88 _._ 08 88 _._ 69 67 _._ 64 92 _._ 36 60 _._ 23 89 _._ 81 63 _._ 40 81 _._ 08 72 _._ 66 52 _._ 48
ODIN 93 _._ 41 68 _._ 56 58 _._ 06 87 _._ 70 26 _._ 85 95 _._ 10 70 _._ 69 83 _._ 42 92 _._ 78 59 _._ 70 88 _._ 51 65 _._ 36 71 _._ 72 76 _._ 64 52 _._ 48
Energy 97 _._ 07 47 _._ 12 53 _._ 27 91 _._ 52 35 _._ 35 93 _._ 49 90 _._ 27 68 _._ 41 99 _._ 29 3 _._ 46 98 _._ 31 6 _._ 41 78 _._ 93 51 _._ 74 52 _._ 48
ReAct 97 _._ 07 47 _._ 12 53 _._ 27 91 _._ 52 35 _._ 35 93 _._ 49 90 _._ 27 68 _._ 41 97 _._ 53 38 _._ 90 83 _._ 94 66 _._ 32 76 _._ 24 67 _._ 63 52 _._ 48
Mahalanobis 64 _._ 60 82 _._ 25 99 _._ 38 44 _._ 82 95 _._ 23 49 _._ 37 53 _._ 12 83 _._ 45 42 _._ 13 84 _._ 50 96 _._ 43 48 _._ 47 75 _._ 15 65 _._ 47 52 _._ 48
CIDER 28 _._ 62 94 _._ 78 14 _._ 64 97 _._ 02 66 _._ 68 77 _._ 38 73 _._ 73 72 _._ 75 57 _._ 96 80 _._ 94 82 _._ 41 72 _._ 13 54 _._ 01 82 _._ 50 57 _._ 70
SSD+ 35 _._ 23 88 _._ 50 34 _._ 22 89 _._ 82 26 _._ 73 92 _._ 94 48 _._ 84 85 _._ 22 76 _._ 06 77 _._ 32 80 _._ 69 69 _._ 36 50 _._ 29 83 _._ 86 60 _._ 06
SNN 16 _._ 78 96 _._ 78 35 _._ 98 93 _._ 88 91 _._ 30 61 _._ 59 56 _._ 80 84 _._ 25 30 _._ 82 92 _._ 35 27 _._ 24 93 _._ 93 43 _._ 15 87 _._ 13 58 _._ 06
SCE 16 _._ 87 96 _._ 64 14 _._ 13 97 _._ 12 30 _._ 67 91 _._ 97 75 _._ 99 62 _._ 21 50 _._ 27 86 _._ 01 88 _._ 84 64 _._ 92 46 _._ 13 83 _._ 15 60 _._ 74
GCE 63 _._ 18 82 _._ 35 58 _._ 27 88 _._ 42 63 _._ 09 77 _._ 88 80 _._ 78 66 _._ 52 58 _._ 03 82 _._ 63 87 _._ 90 67 _._ 42 68 _._ 54 77 _._ 54 58 _._ 21
DivideMix 30 _._ 53 94 _._ 00 37 _._ 22 93 _._ 63 67 _._ 81 83 _._ 31 67 _._ 09 70 _._ 49 48 _._ 55 85 _._ 01 86 _._ 47 71 _._ 10 56 _._ 28 82 _._ 92 33 _._ 26
Co-teaching 51 _._ 77 83 _._ 17 99 _._ 96 27 _._ 69 85 _._ 42 66 _._ 48 98 _._ 34 47 _._ 85 59 _._ 41 79 _._ 01 95 _._ 20 53 _._ 47 81 _._ 68 59 _._ 61 46 _._ 18
CM 35 _._ 24 92 _._ 09 40 _._ 79 92 _._ 55 36 _._ 79 90 _._ 95 54 _._ 59 79 _._ 49 40 _._ 46 89 _._ 36 89 _._ 26 68 _._ 02 49 _._ 52 85 _._ 41 59 _._ 06
VolMinNet 36 _._ 03 91 _._ 87 50 _._ 23 91 _._ 22 45 _._ 13 88 _._ 00 74 _._ 76 61 _._ 76 43 _._ 26 88 _._ 84 90 _._ 49 66 _._ 66 56 _._ 65 81 _._ 39 60 _._ 10


NOODLE 21 _._ 12 95 _._ 97 20 _._ 44 96 _._ 46 5 _._ 24 98 _._ 76 39 _._ 20 91 _._ 36 37 _._ 45 89 _._ 97 87 _._ 67 64 _._ 59 35 _._ 19 89 _._ 52 60 _._ 89


Table 7: OOD detection performance for different _λ_ values for the“worst” noise case. ID dataset is
CIFAR-10 and the encoder architecture is DenseNet-101.

|λ|SVHN|FashionMNIST|LSUN|iSUN|Texture|Places365|AVG|
|---|---|---|---|---|---|---|---|
||**FPR**_ ↓_**AUROC**_ ↑_|**FPR**<br>**AUROC**|**FPR**<br>**AUROC**|**FPR**<br>**AUROC**|**FPR**<br>**AUROC**|**FPR**<br>**AUROC**|**FPR**<br>**AUROC**|
|0.0001<br>0.0005<br>0.001|3.23<br>99.31<br>19.66<br>95.99<br>29.12<br>94.13|23.67<br>95.81<br>42.48<br>90.61<br>47.44<br>90.18|8.08<br>98.40<br>25.72<br>94.65<br>36.82<br>92.17|36.55<br>93.57<br>18.28<br>95.86<br>31.88<br>92.10|32.98<br>91.67<br>33.81<br>91.92<br>56.01<br>83.47|63.14<br>83.91<br>59.99<br>82.62<br>57.17<br>84.44|27.94<br>93.78<br>33.32<br>91.94<br>43.07<br>89.42|


Table 8: OOD detection performance (FPR95 _↓_ / AUROC _↑_ ) using CIFAR-100 with real noisy labels
with a DenseNet-100 encoder (varying _K_ ).

|K|SVHN|FashionMNIST|LSUN|iSUN|Texture|Places365|AVG|
|---|---|---|---|---|---|---|---|
||**FPR**_ ↓_**AUROC**_ ↑_|**FPR**<br>**AUROC **|**FPR AUROC**|**FPR**<br>**AUROC**|**FPR**<br>**AUROC**|**FPR**<br>**AUROC**|**FPR**<br>**AUROC**|
|10<br>15<br>20<br>30<br>40|21.28<br>95.95<br>17.14<br>96.42<br>21.12<br>95.97<br>26.45<br>94.70<br>25.10<br>94.81|31.66<br>94.16<br>38.31<br>92.93<br>20.44<br>96.46<br>29.85<br>95.10<br>20.04<br>96.67|6.70<br>98.51<br>6.02<br>98.68<br>5.24<br>98.76<br>6.50<br>98.50<br>6.50<br>98.73|35.01<br>91.50<br>47.71<br>90.76<br>39.20<br>91.36<br>35.50<br>93.20<br>39.08<br>93.04|31.77<br>91.58<br>28.14<br>92.57<br>37.45<br>89.97<br>42.20<br>90.00<br>33.51<br>90.82|90.59<br>63.49<br>89.62<br>64.97<br>87.67<br>64.59<br>86.30<br>65.00<br>90.72<br>65.42|36.17<br>89.20<br>37.82<br>89.39<br>35.19<br>89.52<br>37.80<br>89.40<br>35.83<br>89.92|


Table 9: OOD detection performance (FPR95 _↓_ / AUROC _↑_ ) on CIFAR-10 with real noisy labels
using a DenseNet-100 encoder.


**Method** SVHN FashionMNIST LSUN iSUN DTD Places365 Average


FPR95 _↓_ AUROC _↑_ FPR95 _↓_ AUROC _↑_ FPR95 _↓_ AUROC _↑_ FPR95 _↓_ AUROC _↑_ FPR95 _↓_ AUROC _↑_ FPR95 _↓_ AUROC _↑_ FPR95 _↓_ AUROC _↑_


**Worst**


FDBD 48.40 91.34 70.39 75.34 60.75 81.90 69.90 73.77 74.72 72.60 95.00 38.63 69.86 72.26
RankFeat 60.41 85.53 61.81 84.44 25.66 93.74 65.47 76.22 68.28 65.89 96.40 36.73 63.00 73.76
NECO 78.06 60.65 70.85 70.22 58.76 75.76 83.83 62.01 74.05 71.28 74.15 73.12 73.28 68.84
NOODLE 3.17 99.47 21.57 96.21 7.57 98.03 28.09 97.28 30.52 92.53 61.75 84.71 **25.45** **94.71**


**Agree**


FDBD 71.83 83.66 46.31 87.52 47.19 89.26 79.11 74.58 78.30 69.04 88.81 61.12 68.59 77.53
RankFeat 24.32 94.38 48.27 91.09 14.21 97.55 70.29 77.94 61.72 74.77 92.18 57.91 51.83 82.27
NECO 81.73 66.91 66.42 72.13 65.12 67.84 46.48 82.13 76.79 67.93 61.95 79.76 66.42 72.79
NOODLE 1.34 99.75 17.07 97.21 2.26 98.95 8.42 98.21 17.15 96.21 52.88 88.41 **16.52** **96.46**


**Rand1**


FDBD 48.40 91.34 70.39 75.34 60.75 81.90 69.90 73.77 74.72 72.60 95.00 38.63 69.86 72.26
RankFeat 40.54 89.96 78.08 74.02 30.94 91.98 93.35 48.12 69.47 65.98 97.83 32.54 68.37 67.10
NECO 87.57 55.73 50.18 82.85 53.82 80.93 68.28 78.11 70.27 77.44 64.11 81.02 65.70 76.01
NOODLE 3.21 99.47 15.65 97.21 5.23 98.93 5.03 98.85 12.79 97.69 58.03 86.35 **16.66** **96.42**


18


**1000**

**1001**

**1002**


**1003**

**1004**

**1005**

**1006**

**1007**

**1008**


**1009**

**1010**

**1011**

**1012**

**1013**


**1014**

**1015**

**1016**

**1017**

**1018**

**1019**


**1020**

**1021**

**1022**

**1023**

**1024**

**1025**


Table 10: OOD detection under real noisy labels (worst case) on CIFAR-10N for various _p_ values.

|p (%)|SVHN|FashionMNIST|LSUN|iSUN|Texture|Places365|AVG|
|---|---|---|---|---|---|---|---|
||**FPR**_ ↓_**AUROC**_ ↑_|**FPR**<br>**AUROC**|**FPR**<br>**AUROC**|**FPR**<br>**AUROC**|**FPR**<br>**AUROC**|**FPR**<br>**AUROC**|**FPR**<br>**AUROC**|
|0<br>20<br>30<br>40<br>50<br>70|13.52<br>97.42<br>8.63<br>98.36<br>8.24<br>98.41<br>7.86<br>98.52<br>3.17<br>99.47<br>5.68<br>98.90|29.92<br>94.26<br>30.26<br>94.28<br>30.14<br>94.31<br>30.69<br>94.37<br>21.57<br>96.21<br>30.83<br>94.31|12.52<br>97.22<br>12.05<br>97.31<br>11.98<br>97.33<br>12.13<br>97.34<br>7.57<br>98.03<br>12.05<br>97.34|28.44<br>94.74<br>17.64<br>96.35<br>17.21<br>96.48<br>15.59<br>96.85<br>28.09<br>97.28<br>13.96<br>97.11|29.65<br>92.99<br>27.75<br>93.47<br>27.22<br>93.50<br>26.91<br>93.67<br>30.52<br>92.53<br>26.10<br>93.82|63.60<br>83.41<br>63.89<br>83.21<br>64.02<br>83.14<br>65.25<br>83.02<br>61.75<br>84.71<br>67.26<br>82.40|29.61<br>93.34<br>26.70<br>93.83<br>26.47<br>93.86<br>26.40<br>93.96<br>**25.45**<br>**94.71**<br>25.98<br>93.98|


19