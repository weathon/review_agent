# DIFFERENCE BACK PROPAGATION WITH INVERSE SIG## MOID FUNCTION


**Anonymous authors**
Paper under double-blind review


ABSTRACT


Since the proposal of neural networks, the derivative-based back-propagation algorithm has been the default setting. However, the derivative for a nonlinear function is an approximation for the difference of the function values, and it would be
a more precise way to do back propagation using the difference directly instead of
the derivative. While the back propagation algorithm has been the rule-of-thumb
for neural networks, it becomes one of the bottlenecks in modern large deep learning models. With the explosion of big data and large-scale deep learning models,
a tiny change in back propagation could lead to a huge difference. Here we propose a new back propagation algorithm based on the inverse sigmoid function to
calculate the difference instead of the derivative, and we verified its effectiveness
with basic examples.


1 INTRODUCTION


Since the proposal of neural networks in 1943 (McCulloch & Pitts, 1943), the chain-rule back propagation (Dreyfus, 1962), based on derivatives, has been the only way to train neural network models.
All parameters in neural networks were updated based on the derivatives of the cost function with
respect to the parameters, calculated with a chain rule by traversing the network backward. To our
knowledge, no new method for performing backpropagation has been proposed.


Recent years have witnessed great progress on big data as well as deep learning models. To name a
few, there has been ImageNet dataset with 14 million labeled images released in 2015 (Russakovsky
et al., 2015), Twitter100k with 100,000 image-text pairs released in 2017 (Hu et al., 2017), TextCaps
with 145k captions for 28k images released in 2020 (Sidorov et al., 2020), and BuildingNet composed of 100k satellite images released in 2021 (Selvaraju et al., 2021). The size of datasets grows
rapidly, and the size of deep learning models also explodes. The model has grown from relatively
small machine learning models like the convolutional neural network (Fukushima & Miyake, 1982;
LeCun et al., 1989) with thousands of parameters to BERT (Devlin et al., 2018) with 110 million
parameters in 2018, and to V-MoE (Riquelme et al., 2021) with 15 billions parameters in 2021. Nevertheless, all these models have been using the same derivative-based back propagation algorithm.
Although the models have shown great performance, it seems we are facing a bottleneck because
nowadays we need to enlarge the models to billions of parameters to improve the accuracy by only
a few percentages.


Trying to explore more options to break the bottleneck, here we propose a new back propagation
method that makes a tiny change on the widely applied back propagation algorithm. The proposed
difference back propagation calculates the gradient based on the difference instead of the derivative,
making the gradients more reliable while propagating backward by maintaining the consistency of
the activation function. In this paper, we experimented with the new back propagation method with
sigmoid activation function in small neural networks to illustrate how it works. The algorithm is
described in Sec. 2 and the performance is shown in Sec. 3.


2 METHOD


Our method only makes changes to the activation function. Here we assume all the other parts
remain the same as a traditional neural network.


1


Figure 1: Illustration of the inconsistency of traditional back propagation. The change in _z_ is calculated based on the derivative (red dashed line), but the difference-based slope (green dashed line)
would reflect the correspondence between _z_ and _a_ more precisely.


In the traditional way, the forward and backward propagations are calculated as Eq. 1 and Eq. 2:


1
_a_ = _sigmoid_ ( _z_ ) = (1)
1 + _e_ _[−][z]_


In which _a_ _[′]_ = _a −_ _learning_ ~~_r_~~ _ate ∗_ _dl/da_, and _z_ _[′]_ = _invsig_ ( _a_ _[′]_ ). We call this formula Difference
Back Propagation (DBP) because it’s calculated based on the difference of _z_ instead of the derivative.
There are a few advantages of this method: 1) It’s consistent and precise in terms of the changes
of both _z_ and _a_, 2) It could avoid gradient vanishing from sigmoid function, and 3) DBP works
not only for sigmoid activation function, but any function that has an inverse function, even for


2


_dl_ _[da]_
_dz_ [=] _dz_


_[da]_ _dl_ _[dl]_

_dz_ _da_ [=] _[ a]_ [(1] _[ −]_ _[a]_ [)] _da_


(2)
_da_


In which _z_ and _a_ are the neuron values before and after the activation function, respectively, and
_l_ is the loss function. This chain rule works perfectly in the limit of learning rate approaching
0. However, with a finite learning rate, when _a_ is updated with Eq. 3, and the corresponding _z_ is
updated with Eq. 4 which is not consistent with the changes on _a_ .


_a_ ~~_u_~~ _pdated_ = _a −_ _[dl]_ ~~_[r]_~~ _[ate]_ (3)

_da_ _[∗]_ _[learning]_


_z_ ~~_u_~~ _pdated_ = _z −_ _[dl]_ ~~_[r]_~~ _[ate][ ̸]_ [=] _[ inv]_ ~~_[s]_~~ _[ig]_ [(] _[a]_ ~~_[u]_~~ _[pdated]_ [)] (4)

_dz_ _[∗]_ _[learning]_


_inv_ ~~_s_~~ _ig_ ( _a_ ) = _−loge_ (1 _/a −_ 1) (5)


This inconsistency is further illustrated in Fig. 1. When we perform backpropagation with optimization algorithms like gradient descent, the gradient of _a_ is first calculated based on the loss function,
and then the gradient of _z_ is calculated with chain rule. When _a_ is updated to _a_ _[′]_ based on the calculated gradients, _z_ is updated proportional to the slope of the red dashed line. However this line
doesn’t reflect the relationship between _a_ and _z_, instead, the green line indicates the corresponding
_z_ _[′]_ that satisfies the consistency: when _z_ changes to _z_ _[′]_, the corresponding _a_ changes to _a_ _[′]_ .


We propose a new formula for the back propagation chain rule as in Eq. 6


[∆] _[a]_ _dl_ _[a][′][ −]_ _[a]_

∆ _z_ _da_ [=] _z_ _[′]_ _−_ _z_


_dl_ [∆] _[a]_
_dz_ [=] ∆ _z_


_[a]_ _[a]_ _dl_ (6)

_z_ _[′]_ _−_ _z_ _da_


Figure 2: The cost functions with respect to training iterations for DBP and traditional back propagation in (1,2,1) neural network.


those functions that are not derivable or even continuous. For example, the derivative of leaky reLU
activation function at 0 is not well defined, and with DBP we don’t need to worry about it anymore.


Sigmoid activation function often faces a problem of vanishing gradient, because _a_ is too close to
1 and the derivative is too close to 0 when _z_ goes large. More specifically, with float64 precision,
when _z_ _>_ 36, computers can not distinguish a from 1, resulting in a derivative of exact 0 during the
calculation. With DBP, this issue is solved because we no longer calculate the derivative. However,
there is still a minor issue with it because the definition domain of inverse sigmoid function is (0 _,_ 1)
exclusively, so we have to constrain _a_ strictly smaller than 1, resulting that the range of _z_ is no longer
the entire set of real numbers. This problem can be solved by utilizing the Taylor Expansion and
representing _a_ as 1 _−_ _x_ when _a_ is close to 1. This is beyond the scope of this paper, and we are
setting a range constraint on _a_ along the experiments.


we conducted basic experiments to demonstrate the effectiveness of DBP, which are described in the
next section. The code repository will be open-sourced later with respect to double-blind review.


3 RESULT


The DBP method has been experimented with small neural network models based on a small set of
generative data.


The dataset consists of 100 random points with scaled cosine function. The inputs are numbers
in range ( _−_ 1 _,_ 1), and outputs are numbers in range (0 _,_ 1). The data is not split into train/test sets
because the DBP method only affect the training process and the generalizability or over-fitting is
not under consideration.


To check the effectiveness of DBP, we trained a neural network with only one hidden layer with 2
neurons. Including the input and output neurons, the neural network structure is (1 _,_ 2 _,_ 1). The cost
function is calculated as root mean squared error and the comparison between DBP and traditional
back propagation is shown in Fig. 2. As DBP is a slight modification on the back propagation algorithm, the training costs are almost identical and the resulting performances are similar. However,
there is a small but observable improvement for the convergence speed as well as the final cost,
indicating that DBP is doing a better job than the traditional back propagation algorithm.


As mentioned before, a constraint is applied over the _a_ values during the training. _a_ is restricted to
(10 _[−]_ [16] _,_ 1 _−_ 10 _[−]_ [16] ). The upper bound is used to avoid overflow in the inverse sigmoid function and
the lower bound is for the symmetry with the upper bound. In the meantime, the value of _z_ _[′]_ _−_ _z_ is
also restricted to avoid dividing by zero. Given that _z_ _[′]_ _−_ _z_ is zero only when _a_ _[′]_ _−_ _a_ is zero, so we
force the zero values in _z_ _[′]_ _−_ _z_ to 1 to make the slope zero.


3


Figure 3: The neuron values _z_ with respect to the training iterations of three randomly selected
samples, for DBP and traditional back propagation, in (1,2,1) neural network.


Figure 4: The cost functions and neuron values _z_ of a random sample with respect to the training
iterations, for DBP and traditional back propagation, in (1,2,2,1) neural network.


To further investigate what the DBP algorithm does, the _z_ values of the neurons are visualized as in
Fig.3. There are in total three neurons in the network (two hidden and one output). One data sample
is randomly picked and the convergence of the neuron values are compared between the traditional
back propagation and DBP. The two algorithms work almost the same way at the beginning of the
model training, and the difference becomes significant only when the neuron values goes far away
from zero. DBP prevents the neuron value _z_ from becoming too large or too small when the gradient
disappears, because the gradient as in Eq. 6 is smaller than the traditional back propagation as in
Eq. 2 when the updating direction is away from zero, and, on the contrary, larger in the case of
toward zero.


Larger models are also experimented with. Fig. 4 shows the convergence of the cost function and
neuron value _z_ of a random sample at a random neuron in the structure of the neural network
(1 _,_ 2 _,_ 2 _,_ 1). The results are very similar to the network (1 _,_ 2 _,_ 1): with DBP, the cost function decays slightly faster and the neuron values tend not to go far from zero so that gradient vanishing is
prevented.


Fig. 5 shows the convergence of loss function and accuracy of a transformer based classification
model on news topic classification. with all the same hyperparameters (d ~~m~~ odel=32, nlayers=2,
nhead=4, ff=64), DBP showed clear advantage on both convergence speed and final performance in
terms of accuracy (bottom two sub-figures are zoomed-in to show the difference).


4 CONCLUSION


We propose a new backpropagation algorithm, DBP, for neural networks, which is derived from
inverse sigmoid function and the difference between the _z_ values. DBP has shown a better perfor

4


Figure 5: The loss function and model accuracy with respect to iterations, for DBP (difference) and
traditional derivative back propagation (default), in basic transformer based classification using the
AG News dataset from Hugging Face’s datasets library with 4-category news topic classification


mance than the traditional derivative-based back propagation, as well as effectiveness in preventing
gradient vanishing due to sigmoid function. We believe that DBP is a more accurate way to do back
propagation because it maintains consistency between neuron values before and after the activation
function.


The DBP algorithm can also be applied to other activation functions, as long as there is an inverse
function accordingly. Without derivatives of the activation function, DBP allows for the applications
of activation functions that are not derivable or continuous.


REFERENCES


Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep
bidirectional transformers for language understanding. _arXiv preprint arXiv:1810.04805_, 2018.


Stuart Dreyfus. The numerical solution of variational problems. _Journal of Mathematical Analysis_
_and Applications_, 5(1):30–45, 1962.


Kunihiko Fukushima and Sei Miyake. Neocognitron: A self-organizing neural network model for
a mechanism of visual pattern recognition. In _Competition_ _and_ _cooperation_ _in_ _neural_ _nets_, pp.
267–285. Springer, 1982.


Yuting Hu, Liang Zheng, Yi Yang, and Yongfeng Huang. Twitter100k: A real-world dataset for
weakly supervised cross-media retrieval. _IEEE_ _Transactions_ _on_ _Multimedia_, 20(4):927–938,
2017.


Yann LeCun, Bernhard Boser, John S Denker, Donnie Henderson, Richard E Howard, Wayne Hubbard, and Lawrence D Jackel. Backpropagation applied to handwritten zip code recognition.
_Neural computation_, 1(4):541–551, 1989.


Warren S McCulloch and Walter Pitts. A logical calculus of the ideas immanent in nervous activity.
_The bulletin of mathematical biophysics_, 5(4):115–133, 1943.


Carlos Riquelme, Joan Puigcerver, Basil Mustafa, Maxim Neumann, Rodolphe Jenatton, Andr´e Susano Pinto, Daniel Keysers, and Neil Houlsby. Scaling vision with sparse mixture of experts.
_arXiv preprint arXiv:2106.05974_, 2021.


5


Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng
Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual
recognition challenge. _International journal of computer vision_, 115(3):211–252, 2015.


Pratheba Selvaraju, Mohamed Nabail, Marios Loizou, Maria Maslioukova, Melinos Averkiou, Andreas Andreou, Siddhartha Chaudhuri, and Evangelos Kalogerakis. Buildingnet: Learning to label
3d buildings. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_, pp.
10397–10407, 2021.


Oleksii Sidorov, Ronghang Hu, Marcus Rohrbach, and Amanpreet Singh. Textcaps: a dataset for
image captioning with reading comprehension. In _European Conference on Computer Vision_, pp.
742–758. Springer, 2020.


6