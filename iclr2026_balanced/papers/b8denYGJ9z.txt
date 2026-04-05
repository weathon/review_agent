# VAE BASED MULTI-FIELDS NEURAL DATA ASSIMILA
## TION FOR SEA ICE MODEL


**Anonymous authors**
Paper under double-blind review


ABSTRACT


This study presents a neural data assimilation system based on a variational autoencoder (VAE) for improving sea ice forecasts in high-resolution numerical
models. We propose a multi-field assimilation approach that simultaneously processes several physical fields, leveraging a modern VAE architecture enhanced
with pixel-wise self-attention mechanisms to capture complex spatial and crossfield correlations. Our method is validated using real-world satellite observations
(Sentinel-3 SRAL and AMSR2) and operational data from the NEMO ocean
model with integrated sea ice component (SI3). Results demonstrate that the
framework effectively assimilates sparse and noisy observations, reducing errors
in sea ice concentration estimates and improving forecast accuracy. Crucially,
we demonstrate the compatibility of the neural assimilation solution with the
NEMO restart mechanism, enabling seamless integration into operational forecasting pipelines. This work bridges the gap between machine learning-based
assimilation and practical ocean modeling, offering a scalable, non-Gaussian alternative to traditional methods like 3D-VAR.


1 INTRODUCTION


Significant trend in Earth Sciences is the growing interest in the Arctic region, driven by the rapid
decline in sea ice cover (Babb et al., 2023) and the profound environmental, economic, and geopolitical changes this entails (Kortsch et al., 2015; Shu et al., 2023; Dvoynikov et al., 2021). Numerical
models in the Arctic and Antarctic regions are notoriously harder to run and calibrate (Pan et al.,
2023; Allende et al., 2024). Sea ice forecasting requires considering the entangled relationships
between wind, ocean physics, and plastic deformations inside the ice itself to be relevant. These
forecasts hold practical value for ice-impeded navigation in addition to scientific value.


Naturally, Earth Sciences move towards increasing the resolution of climate models (MorenoChamarro et al., 2025; Olason et al., 2021; Selivanova et al., 2024) and expanding the volume of
accumulated observational data (Copernicus Climate Change Service (C3S); National Aeronautics
and Space Administration) in order to improve the prediction of the extreme events and circulations.
This trend is driven by advances in computational power, improved satellite and sensor technologies.


Numerical models are a cornerstone in ocean simulation, incorporating modern knowledge about
the complex physical processes that govern ocean dynamics. Because of the chaos, inherent to all
such models, small uncertainty of the initial conditions grows to unpredictable model behavior in a
few days horizons of simulation. Data assimilation is a necessary tool to condition these numerical
models on observations and improve their quality. Classical data assimilation algorithms rely on the
assumptions of linear model dynamics and Gaussian noise. However, as the resolution of the model
increases, these assumptions become less valid, which requires a departure from these constraints to
better capture the complexities of high-resolution systems (Carrassi et al., 2018).


The data assimilation task can be formulated as the process of iterative updating of the physical
fields of the model _xb_ (background) based on the observed data _y_ . The target is to get closer to an
unknown true state of the fields _xa_ (analysis) (Bannister, 2008a;b). The data assimilation process
must account for the spatial and temporal relationship of the physical fields.


There are several approaches to data assimilation. Non-neural approaches will be called in our text
classical data assimilation methods. In most cases, it comes down to filtering techniques such as


1


Best Linear Unbiased Estimator (BLUE) and Kalman Filter (KF) (Kalman, 1960; Jazwinski, 1970;
Evensen, 1994; 2003) or to the optimization of a cost function as 3 and 4-Dimensional Variational
(3D-VAR, 4D-VAR) (Bannister, 2008a;b; Courtier et al., 1998; Rabier et al., 2000) data assimilation.
Neural networks have shown a remarkable ability to learn complex and non-linear relationships
between physical fields in oceanography (Zhao et al., 2024). There are several studies that combine
traditional data assimilation algorithms and modern neural networks (Blanke et al., 2024; Arcucci
et al., 2021; Penny et al., 2022; Cai et al., 2024; Tian, 2024; Hatfield et al., 2021; Mack et al., 2020;
Peyron et al., 2021; Farchi et al., 2021; Barthelemy et al., 2022; Melinc & Zaplotnik, 2024).


Most existing data assimilation models focus mainly on simplified systems and benchmark examples, such as the Lorenz-63 or Lorenz-96 systems (Blanke et al., 2024; Arcucci et al., 2021; Penny
et al., 2022; Tian, 2024; Peyron et al., 2021; Farchi et al., 2021). These toy models serve as foundational testbeds for evaluating the performance of assimilation techniques due to their ability to
capture essential features of chaotic dynamics while remaining computationally tractable. However,
their simplicity often limits the direct applicability of these methods to more complex, real-world
systems characterized by high dimensionality and intricate spatial correlations.


Observation data can be sparse and contain errors and inaccuracies. The true values of the physical
field are unknown, as are the true values of the model and the observation error. We should use some
estimation for calculation and validation. This is especially noticeable when calculating the matrix
_B_ . Classical data assimilation algorithms operate under Gaussian assumptions. For example, the ice
concentration field has a significantly non-Gaussian error distribution, while the Ensemble Kalman
Filter EnKF was used to assimilate this field (Lisæter et al., 2003).


1.1 3D-VAR AND 4D-VAR


This is family of iterative data assimilation algorithms that are based on cost function optimization.
Cost function for 3D-VAR is

_J_ ( _x_ ) = ( _x −_ _xb_ ) _[T]_ _B_ _[−]_ [1] ( _x −_ _xb_ ) + ( _y −_ _Hx_ ) _[T]_ _R_ _[−]_ [1] ( _y −_ _Hx_ ) (1)


where _B_ is a background error covariance matrix, _R_ is an observation error covariance matrix, and
_H_ is the forward operator. For 4D-VAR the model operator _M_ is added:


where _x_ 0 = _x_ ( _t_ 0) are the physical fields at the beginning of the assimilation window and
_xi_ = _x_ ( _ti_ ) = ( [�] _[i]_ _k_ =1 _[M][k]_ [)] _[x]_ [(] _[t]_ [0][)][.] [It] [can] [be] [shown] [that] [Kalman] [Filter] [minimizes] [the] [same] [func-]
tion (Brasseur, 2006).


1.2 ARTIFICIAL NEURAL NETWORKS FOR DATA ASSIMILATION


An alternative to classical data assimilation schemes is the integration of artificial neural networks
(ANNs) to replace or enhance components of traditional systems. There are two main strategies
for incorporating ANNs into data assimilation. The first involves using a neural network to distill
classical methods (Arcucci et al., 2021; Farchi et al., 2021; Zavala-Romero et al., 2025), such as
3D-Var, 4D-Var, and ETKF, capturing their key features in a more computationally efficient form.
The second approach integrates ANNs directly into the assimilation process by replacing specific
components of traditional algorithms (Penny et al., 2022; Tian, 2024; Hatfield et al., 2021; Mack
et al., 2020; Peyron et al., 2021; Barthelemy et al., 2022; Melinc & Zaplotnik, 2024), potentially improving adaptability and providing more accurate error estimates. The second approach is of greater
interest because it enables a more flexible and adaptive data assimilation process. By replacing key
components of traditional algorithms with ANNs, we can take advantage of their ability to learn
complex non-linear relationships within the data. This has the potential to improve accuracy, reduce
computational costs, and enhance robustness in dynamically changing ice conditions.


Classical data assimilation algorithms face two major challenges. The first is the use of the covariance matrix, which can be extremely large and may fail to capture nonlinear interactions between
variables. The second is the dynamical operator, which is either computationally expensive or based
on overly simplistic approximations.


2


_J_ ( _x_ 0) = ( _x_ 0 _−_ _xb_ ) _[T]_ _B_ _[−]_ [1] ( _x_ 0 _−_ _xb_ ) +


_N_
�( _y −_ _H_ ( _xi_ )) _[T]_ _R_ _[−]_ [1] ( _y −_ _H_ ( _xi_ )) (2)


_i_ =1


The variational autoencoder (VAE) is widely used as a replacement or complement to the covariance
matrix, enabling dimensionality reduction while preserving complex variable interactions. VAEs
impose constraints on data distribution, making them particularly suitable for physical data fields
with nonlinear dependencies. In (Mack et al., 2020), a VAE is applied within the 3D-Var algorithm
to reduce the dimensionality of physical fields while capturing intervariable relationships, tested on
pollution data from Elephant and Castle, London. In (Peyron et al., 2021), the ensemble Kalman
filter is applied in the latent space of an autoencoder and tested on the Lorenz-96 model. In (Melinc
& Zaplotnik, 2024), an autoencoder completely replaces the covariance matrix in the 3D-Var algorithm. The approach is tested on temperature data at the 850-hPa pressure level from the ERA5
reanalysis. The latter study is conceptually similar to our work but considers only a single physical
field and does not assess the possibility of restarting the model with the assimilated field.


**This paper makes the following contributions:**


1. We propose a new data assimilation algorithm that works with multiple geophysical fields
by leveraging VAE architecture with self-attention layers in the latent space.


2. We test our algorithm on different setups and show that it outperforms the baselines: classical 3D-VAR and a VAE-based approach (Melinc & Zaplotnik, 2024).


3. We integrated our algorithm inside the operational forecasting ocean model Nucleus for
European Modeling of the Ocean (NEMO) to assimilate real sattelite observations. We
have shown that neural network based data assimilation improves the forecasts quality.


2 DATA


The area of interest for this work is the Barents Sea and the Kara Sea (see Figure 1). The Kara and
Barents Seas are not covered with multi-year ice de Gelis et al. (2021). For the selected region of
the Kara Sea, the freeze-up begins rapidly in December, with melting in April. The selected region
of the Barents Sea has a completely open sea throughout most of the year.


3 MODEL


3.1 BACKGROUND


As an assimilation background we use ocean data that are generated by our production numerical
forecasting system that consists of numerical Weather Research and Forecasting model (WRF) and
Nucleus for European Modeling of the Ocean (NEMO) of version 4.0 with integrated Sea Ice Modeling Integrated Initiative (SI3). Spatial resolution of the modeling dataset is around 3-4 km, we


3


Figure 1: Area of interest for this work.
Blue is the simulation area of NEMO model.


Figure 2: Cumulative distribution functions (CDFs)
of sea ice concentration from different data sources.


used ORCA12 grid and Drakkar configuration that is considered state of the art for high resolution
ocean modeling. Since NEMO requires atmosphere forcing we also used these data as additional
features (these data are generated by the atmosphere model Weather Research and Forecasting WRF
of version 4.4). For the period of interest, the dataset contains daily prediction of the next 72 hours
for ocean and atmosphere variables. Data range from 2015 to 2023. In this study, we focus on three
key fields: the sea ice concentration field (where data assimilation is performed), the sea ice thickness field, and the sea ice temperature field. Data from 2015 to 2021 (inclusive) are used to compute
statistical properties and train the variational autoencoder (VAE). 2022 and 2023 are reserved for
validating the accuracy of field reconstruction and conducting data assimilation experiments.


3.2 OBSERVATIONS


The following observational data was considered:


1. Sentinel-3 Altimetry (SRAL) Sea Ice Thematic Product Aublanc et al. (2025). The dataset
includes satellite tracks with an along-track resolution of up to 330 meters. The variable
”sea ~~i~~ ce ~~c~~ oncentration 20 ~~k~~ u” provides ice concentration data and is available as an auxiliary product. The primary data source is the OSI-430-a product, which has a 2-day latency
and a spatial resolution of 25 km. Therefore, greater scientific interest lies in the variable ”surf ~~t~~ ype ~~c~~ lass ~~2~~ 0 ~~k~~ u”, which characterizes surface type (open ocean, floes, lead,
unclassified) based on sea ice concentration and waveform peakiness, and the variable
”freeboard 20 ~~k~~ u” (radar freeboard). In future product versions, the latter is planned to
be converted into ice thickness.


2. ASI AMSR2 Spreen et al. (2008) sea ice concentration data. These data have a spatial
resolution of 6.25 km and are available in near-real time (NRT) with a latency about 25
hours. A key advantage of AMSR2 is its all-weather, day-and-night operational capability,
enabled by passive microwave measurements. However, a known limitation is the reduced
accuracy in sea ice concentration retrieval during melt seasons due to changes in surface
emissivity caused by snow and ice melt Pang et al. (2018).


We have determined that the most promising problem to address is the assimilation of data tracks,
especially since Sentinel-3 is expected to enhance its thematic product on sea ice with ice thickness
information. For model-to-model assimilation, we use sea ice concentration data from a different
model year, selected along SRAL tracks. For satellite-to-model assimilation, we employ AMSR2
sea ice concentration data along SRAL tracks, adjusted using the surf ~~t~~ ype ~~c~~ lass ~~2~~ 0 ~~k~~ u flag: values
are set to zero for the ”open ocean” class, while for ”floes” and ”lead” classes with zero concentration, the nearest non-zero value is assigned.


3.3 VALIDATION


For model-to-model assimilation, the model fields corresponding to the dates of the assimilation
tracks are used for validation and comparison. For satellite-to-model assimilation, AMSR2 data
serve as the validation data set.


3.4 DATA ANALYSIS


To compare sea ice concentration data from different sources along the Sentinel-3 track between December 2019 and April 2020, their cumulative distribution functions (CDF) should be analyzed (Fig.
2). Data were collected for identical dates and identical spatial domains in all sources. The NEMO
model (blue line) exhibits a lower probability of zero ice concentration compared to AMSR2 (green
line), indicating systematic overestimation of ice cover in the model output. The SRAL-derived
’sea ice ~~c~~ oncentration ~~2~~ 0 ~~k~~ u’ variable (orange line) shows markedly different slope characteristics
than both NEMO and AMSR2. This divergence stems from the coarse resolution of OSI-430-a
product (source of this variable), which introduces spatial averaging effects that increase the number
of intermediate values.


4


Figure 3: Variational Autoencoder architecture. Figure 4: Example of VAE data reconstruction.


4 MODEL


In the 3D-VAR algorithm, spatial relationships between the components of a field, as well as interfield correlations, are represented in the background error covariance matrix. In this study, we
propose using a variational autoencoder (Kingma & Welling, 2014) to capture and represent these
patterns. A related approach was previously applied to temperature data at the 850-hPa pressure level
from the ERA5 reanalysis, as reported in (Melinc & Zaplotnik, 2024). We used the VAE architecture
from (Melinc & Zaplotnik, 2024) as our baseline, denoting it as _base_ ~~_v_~~ _ae_ ~~1~~ _f_ . In our work, we
employ a different autoencoder architecture and perform experiments using both single-field and
multi-field datasets. The architecture we propose is inspired by the VAE architectures widely used
in stable diffusion models Rombach et al. (2022), which have demonstrated strong generative and
representation learning capabilities. The main characteristics of our architecture 3 are as follows:
(1) the latent space is implemented as a feature map rather than a vector, (2) ResNet-based blocks
are used throughout the encoder and decoder, and (3) attention mechanisms are introduced in the
middle layers to enhance feature interaction.


We conducted experiments with a date-conditioned autoencoder where: dates were converted to
day-of-year (DOY) values. DOY was encoded as a cyclic variable using sine/cosine transformation:
(sin(2 _π · DOY/_ 366) _,_ cos(2 _π · DOY/_ 366)) which are transformed by the linear layer and concatenated with in the latent space.


4.1 TRAINING


The variational autoencoder learns to reconstruct fields using the reparameterization trick. Mean
squared error (MSE) is used as the reconstruction loss, while the KL divergence is computed for the
Gaussian case (Kingma & Welling, 2014). We also experimented with the addition of a discriminator
and SSIM loss, but it did not produce significant improvements.


We used the Lion optimizer (Chen et al., 2023) for training, as it demonstrated a more stable convergence in our experiments. Missing data (over land) are filled with physically adequate values.


4.2 ASSIMILATION


For the baseline data assimilation experiment, we employed a 3D-VAR scheme 3 _d_ ~~_v_~~ _ar_ . The background error covariance matrix B was modeled using a fifth-order quasi-Gaussian function (Gaspari
& Cohn, 1999) with a length scale of 100 km.


The latent space assimilation algorithm (Algorithm 1) is presented below. The key feature of the
algorithm: assimilation is performed in the autoencoder’s latent space. The latent space field is
adjusted via backpropagation to ensure the reconstructed (analysis) field better approximates the
observational data.


It is important to discuss the features of the _Loss_ function. The loss function consists of three terms:
(1) the observation error, (2) the background field error, and (3) the latent-space error. The second
and third terms serve as regularizations to prevent a significant deviation from the model’s original
field and are assigned smaller weighting coefficients.


_Loss_ ( _xa, y, xb, z, z_ 0) = _wyMSE_ ( _H_ ( _xa_ ) _, y_ )
(3)
+ _wbMSE_ ( _xa, xb_ ) + _wzMSE_ ( _z, z_ 0)


5


**Algorithm 1** Latent Space Assimilation (LSA)


**Require:** _Encoder_ and _Decoder_ - part of pretrained VAE, _Loss_ - error function

**Input:** _xb_  - background fields, _y_  - observations
**Output:** _xa_  - analysis fields


Encode background: _z_ 0 _←_ _Encoder_ ( _xb_ )
Initialize optimizable latent: _z_ _←_ _z_ 0 (requires gradient)


**for** iteration i=0 to N **do**

Decode current latent: _xa_ _←_ _Decoder_ ( _z_ )
Compute loss: _Ltotal_ _←_ _Loss_ ( _xa, y, xb, z, z_ 0)
Compute gradient: _∇zLtotal_
Update latent: optimization step
**end forreturn** _xa_ _←_ Decoder( _z_ )


Figure 5: MAE metric values for _vae_ ~~4~~ _f_ model from different physical fields by day of the year on
the left. Sea Ice Concentration assimilation example for 12.04 using _vae_ ~~4~~ _f_ model on the right.


Given the slow evolution of ice concentration (typically _<_ 1% daily change), we tested assimilation
of the prior three days data.


5 EXPERIMENTS


Since we aimed at improving the quality of our forecasting system in the real world through sentinel
3 SRAL assimilation, we organized our framework as follows. First, we evaluated the quality of
the VAE reconstruction. Secondly, we experimented with model-to-model assimilation to tune out
probable overfitting and deal with general architecture search. After that we tested the best resulting
models on real-world data and tested our assimilation model inside NEMO forecasting pipeline.


The VAE should reconstruct the fields reasonably well, though perfect accuracy is not the goal, as
VAEs inherently balance reconstruction fidelity against the dimensionality of the compressed representation (i.e., latent space sparsity). The goal of the model-to-model assimilation is to estimate the
quality of assimilation in neutral setting. Since observation data and model data are noisy and usually distributed differently, thus introducing bias in the quality of assimilation. Model-to-model, on
the other hand, demonstrates the quality of assimilation in the physics-driven process that is represented by the numerical model inside NEMO. For the forecasting pipeline, let’s call this assimilation
scheme production data assimilation.


5.1 RECONSTRUCTION ERROR


The model was trained on data from 2015 to 2021 and 2022 was used for validation. The following fields were utilized: siconc (sea ice concentration), sithic (sea ice thickness), sitemp (sea


6


ice temperature) and sosstsst (sea surface temperature). An example of sea ice concentration field
reconstruction is shown in Figure 4. The results are presented in Table 1 .


The evaluation metrics used are MAE (Mean Absolute Error) and MSE (Mean Squared Error). MAE
estimates the average error magnitude, while MSE penalizes larger errors more heavily and smaller
errors less severely due to its quadratic nature. The results are presented in Table 1. Features of
the architecture and naming of models: ~~*****~~ **f** - shows how many physical fields are fed to the model
input. ~~**d**~~ ***** the latent space is transformed by a linear layer from a feature map into a vector with the
specified number of features. ~~**m**~~ the sitemp field is replaced by the sosstsst field, ~~**e**~~ **mb** - a condition
for the day of the year is added, ~~**c**~~ ***** - the number of feature maps in the latent space, 1 by default.


Table 1: VAE reconstruction quality assessment

|Model<br>name|siconc|sithic|sitemp|sosstsst|
|---|---|---|---|---|
|Model<br>name|MAE|MAE|MAE|MAE|
|base ~~v~~ae 1f|0_._028_ ±_ 0_._002|-|-|-|
|vae ~~1~~f|0_._008_ ±_ 0_._001|-|-|-|
|vae ~~1~~f ~~d~~512|0_._023_ ±_ 0_._002|-|-|-|
|vae ~~3~~f|0_._023_ ±_ 0_._002|0_._083_ ±_ 0_._006|0_._094_ ±_ 0_._007|-|
|vae ~~3~~f ~~e~~mb|0_._026_ ±_ 0_._002|0_._090_ ±_ 0_._007|0_._093_ ±_ 0_._008|-|
|vae ~~3~~f ~~m~~|0_._015_ ±_ 0_._001|0_._024_ ±_ 0_._002|-|0_._252_ ±_ 0_._006|
|vae ~~3~~f ~~m e~~mb|0_._016_ ±_ 0_._001|0_._026_ ±_ 0_._002|-|0_._254_ ±_ 0_._007|
|vae ~~4~~f|0_._024_ ±_ 0_._002|0_._083_ ±_ 0_._006|0_._085_ ±_ 0_._007|0_._265_ ±_ 0_._008|
|vae ~~4~~f ~~e~~mb|0_._032_ ±_ 0_._002|0_._099_ ±_ 0_._008|0_._083_ ±_ 0_._007|0_._267_ ±_ 0_._008|
|vae ~~4~~f ~~c~~2|0_._018_ ±_ 0_._001|0_._029_ ±_ 0_._002|0_._062_ ±_ 0_._005|0_._210_ ±_ 0_._005|
|vae ~~4~~f ~~c~~2 ~~e~~mb|0_._017_ ±_ 0_._001|0_._031_ ±_ 0_._002|0_._061_ ±_ 0_._005|0_._186_ ±_ 0_._003|


5.2 QUALITY ESTIMATION


5.2.1 MODEL TO MODEL ASSIMILATION


In this experiment, our objective was to analyze the ability of a trained VAE to capture relationships
between values and physical fields. Unfortunately, we do not know the true field values at each time
step: The model contains computational errors and depends on the initial initialization, which also
introduces uncertainties. Satellite data have measurement errors and interpretation challenges.


Therefore, we conclude that it is necessary to verify the assimilation of the model data into the
model data. We sample see ice concentration data along the SRAL tracks from the NEMO forecast
for next year _y_ ˜ = _xi_ +365[ _mask_ ] and assimilate them into current data _xi_ . To evaluate performance,
we compare the post-assimilation fields with the prediction of the model _xi_ +365 (see Algorithm 2).
Since we treat NEMO numerical modeling as physics-based, we expect that a higher quality model
will assimilate data closer to the NEMO output.


**Algorithm 2** Model-to-model (M2M) data assimilation


**Require:** _{x_ 1 _, x_ 2 _, ..., xn}_ - NEMO simulation fields, _{y_ 1 _, y_ 2 _, ..., yn}_ - AMSR corrected by SRAL
data
**for** day in assimilation cycle **do**

Create mask: _mask_ = _yi_ is not None
Sample observation: _y_ ˜ _i_ = _xi_ +365[ _mask_ ]
Make assimilation: _x_ _[a]_ _i_ [= LSA(] _[x][i]_ [,] _[y]_ [˜] _[i]_ [)]
Make validation: _xi_ and _x_ _[a]_ _i_ [vs] _[ x][i]_ [+365]
**end for**


Examples of model-in-model assimilation for 3 _d_ ~~_v_~~ _ar vae_ ~~4~~ _f_ are illustrated in figures in 7. Only sea
ice concentration values are assimilated. It can be seen that _vae_ ~~4~~ _f_ not only yields a lower error but
also produces a sharper ice-water boundary, while 3 _d_ ~~_v_~~ _ar_ tends to smooth it. The _vae_ ~~4~~ _f_ results
show that when the concentration of ice decreases, the thickness of the ice in the same region also


7


Table 2: Model to model assimilation result: average MAE

|Model name|siconc|sithic|sitemp|sosstsst|
|---|---|---|---|---|
|background|0_._112_ ±_ 0_._002|0_._242_ ±_ 0_._005|0_._67_ ±_ 0_._04|0_._86_ ±_ 0_._03|
|3d ~~v~~ar|0_._052_ ±_ 0_._001|-|-|-|
|base ~~v~~ae ~~1~~f|0_._080_ ±_ 0_._002|-|-|-|
|vae ~~1~~f|0_._051_ ±_ 0_._001|-|-|-|
|vae 1f ~~d~~512|0_._053_ ±_ 0_._001|-|-|-|
|vae ~~3~~f|0_._060_ ±_ 0_._001|0_._172_ ±_ 0_._004|0_._63_ ±_ 0_._04|-|
|vae ~~3~~f ~~e~~mb|0_._057_ ±_ 0_._001|0_._167_ ±_ 0_._004|0_._64_ ±_ 0_._04|-|
|vae ~~3~~f ~~m~~|0_._061_ ±_ 0_._001|0_._200_ ±_ 0_._004|-|0_._84_ ±_ 0_._05|
|vae ~~3~~f ~~m e~~mb|0_._062_ ±_ 0_._001|0_._213_ ±_ 0_._004|-|0_._811_ ±_ 0_._045|
|vae ~~4~~f|**0**_._**0481**_ ±_** 0**_._**0009**|**0**_.__ ±_** 0**_._|0_._615_ ±_ 0_._036|0_._732_ ±_ 0_._039|
|vae ~~4~~f ~~e~~mb|0_._0493_ ±_ 0_._0009|0_._158_ ±_ 0_._004|**0**_.__ ±_** 0**_._|**0**_.__ ±_** 0**_._|
|vae ~~4~~f ~~c~~2|0_._0508_ ±_ 0_._0010|0_._168_ ±_ 0_._003|0_._616_ ±_ 0_._035|0_._739_ ±_ 0_._034|
|vae ~~4~~f ~~c~~2 ~~e~~mb|0_._0504_ ±_ 0_._0009|0_._176_ ±_ 0_._003|**0**_.__ ±_** 0**_._|0_._724_ ±_ 0_._037|


decreases, while both the temperature of the ice and the temperature of the ocean increase. These
patterns align with the target fields and maintain physical consistency with the model. This clearly
shows that the VAE successfully captures not only relationships within individual physical variables
but also the cross-correlations between different physical parameters.


Figure 5 shows the MSE and MAE metric values for data assimilation using the _vae_ ~~4~~ _f_ model,
applied to the fields of ice concentration, ice thickness, ice temperature and upper-layer water temperature over different days of the year. The data is interrupted in mid-July when the ice in the
Barents and Kara Seas completely melts. It is evident that the ice concentration and thickness fields
are well-correlated, whereas the ice and water temperature fields show weaker dependence on the
ice concentration. Furthermore, Table 2 present the MAE metrics for assimilation across different
models. We observe that the incorporation of temperature fields improves the accuracy of the ice
concentration and thickness assimilation. The _vae_ ~~4~~ _f_ model was selected as the primary because of
its best performance in capturing the relationship between ice concentration and thickness.


5.2.2 SAT TO MODEL ASSIMILATION


The next step was data assimilation of AMSR2 ice concentration data corrected by SRAL surface
type. We evaluated the quality of assimilation against the AMSR2 data and corrected AMSR2 data
that had not yet been assimilated. The assimilation scheme works as follows.


**Algorithm 3** Satellite-to-model (S2M) data assimilation


**Require:** _{x_ 1 _, x_ 2 _, ..., xn}_ - NEMO simulation fields, _{y_ 1 _, y_ 2 _, ..., yn}_ - AMSR corrected by SRAL
data, _{v_ 1 _, v_ 2 _, ..., vn}_  - AMSR data.
**for** day in assimilation cycle **do**

Make assimilation: _x_ _[a]_ _i_ [= LSA(] _[x][i]_ [,] _[ y][i]_ [)]
Make validation: _xi_ and _x_ _[a]_ _i_ [vs] _[ v][i]_ [and] _[ y][i]_ [+1]
**end for**


The assimilation results for the SRAL-corrected AMSR2 sea ice concentration data are presented in
the Table 3. Although the _vae_ ~~3~~ _f_ ~~_e_~~ _mb_ model showed slightly better metrics, we selected the _vae_ ~~4~~ _f_
model for the final experiment. This decision is based on its superior performance in capturing ice
thickness variations in the M2M experiment, coupled with the fact that the difference in results for
this experiment is within the margin of error. Samples of assimilation results are in Appendix A.3.


5.3 PRACTICAL APPLICATION


In the last experiment, we integrate the results of the satellite data assimilation into the NEMO
pipeline using the restart mechanism. The model restart-ice fields were modified using the _sicon_
(sea ice concentration) and _sithic_ (sea ice thickness) fields, which assimilated satellite data via the


8


Table 3: Satellite to model assimilation result

|Model<br>name|AMSR2|Col3|AMSR2 corrected (track)|Col5|
|---|---|---|---|---|
|Model<br>name|MSE|MAE|MSE|MAE|
|background|0_._027_ ±_ 0_._002|0_._049_ ±_ 0_._002|0_._028_ ±_ 0_._002|0_._050_ ±_ 0_._003|
|3d var|0_._013_ ±_ 0_._001|0_._037_ ±_ 0_._002|0_._019_ ±_ 0_._002|0_._044_ ±_ 0_._002|
|base vae ~~1~~f|0_._019_ ±_ 0_._001|0_._045_ ±_ 0_._002|0_._021_ ±_ 0_._002|0_._051_ ±_ 0_._003|
|vae ~~1~~f|0_._014_ ±_ 0_._001|0_._034_ ±_ 0_._002|0_._019_ ±_ 0_._002|0_._040_ ±_ 0_._002|
|vae ~~1~~f ~~d~~512|**0**_.__ ±_** 0**_._|0_._033_ ±_ 0_._002|**0**_.__ ±_** 0**_._|0_._041_ ±_ 0_._002|
|vae ~~3~~f|**0**_.__ ±_** 0**_._|0_._033_ ±_ 0_._002|**0**_.__ ±_** 0**_._|0_._040_ ±_ 0_._002|
|vae ~~3~~f ~~e~~mb|**0**_.__ ±_** 0**_._|**0**_.__ ±_** 0**_._|**0**_.__ ±_** 0**_._|**0**_.__ ±_** 0**_._|
|vae ~~3~~f m|0_._014_ ±_ 0_._001|0_._034_ ±_ 0_._002|0_._018_ ±_ 0_._002|0_._043_ ±_ 0_._002|
|vae ~~3~~f ~~m e~~mb|0_._013_ ±_ 0_._001|0_._036_ ±_ 0_._002|0_._018_ ±_ 0_._002|0_._044_ ±_ 0_._002|
|vae ~~4~~f|0_._013_ ±_ 0_._001|0_._033_ ±_ 0_._002|0_._018_ ±_ 0_._002|0_._041_ ±_ 0_._002|
|vae ~~4~~f ~~e~~mb|**0**_.__ ±_** 0**_._|0_._039_ ±_ 0_._002|0_._018_ ±_ 0_._002|0_._046_ ±_ 0_._002|
|vae ~~4~~f ~~c~~2|0_._014_ ±_ 0_._001|0_._033_ ±_ 0_._002|0_._019_ ±_ 0_._002|0_._040_ ±_ 0_._002|
|vae ~~4~~f ~~c~~2 ~~e~~mb|0_._013_ ±_ 0_._001|0_._033_ ±_ 0_._002|0_._019_ ±_ 0_._002|0_._041_ ±_ 0_._002|


_vae_ ~~4~~ _f_ model. The _sitemp_ (ice surface temperature) and _sosstsst_ (sea surface temperature) fields
were deliberately excluded, as they demonstrated a weaker correlation with the assimilated values.
The restart data is modified as told in appendix A.1.


The experiment was organized as follows. A specific date was selected. For this date, data assimilation was performed and the restart file was modified. Subsequently, a 5-day forecast was run
using the NEMO model. The forecast results were then compared with the corresponding AMSR2
data for those days. The outcome of this comparison is presented in the Figure 8. Metrics for the
experiments are in Table 4.


The experiment demonstrated the feasibility of using fields processed by neural network-based data
assimilation within the NEMO computational model for forecasting. The predictions generated by
this approach are physically consistent and show good agreement with satellite observations.


Table 4: Application result: MAE with AMSR2 (20-02-2025)

|Experiment|1 day|2 day|3 day|4 day|5 day|
|---|---|---|---|---|---|
|model|0.142|0.123|0.081|0.084|0.081|
|model+assimilation|0.079|0.086|0.063|0.072|0.072|


6 CONCLUSION


This work is devoted to multi-field data assimilation by neural networks as an alternative to classical
approaches. For this we used a variational autoencoder with self-attention layers that helped us
to account to cross-corelation between the physics fields. We have shown that such an approach
surpasses in terms of quality both classical approach 3D-VAR and the competing approaches from
literature based on similar ideas.


Also we tested our approach on the real operative sea ice forecasting workflow in Arctic based on
Nucleus for European Modelling of the Ocean (NEMO) state-of-the-art oceam modeling framework.
We have shown that assimilation of real sattelite data in the model quickly improves the model
quality, so our approach has strong practical applications.


Possible direction of our approach improvement would be to correct not only a single snapshot
but a whole time series in order to assure physical consistency of the forecasts dynamics. The
may challenge here lies in learning a good evolution operator for the whole bunch of participating
physical fields. In order to improve quality of forecasts even further it would make sense not only to
assimilate sea ice data but also to correct atmosphere forcing and increase domain of simulations to
capture more oceanic and atmospheric processes in the Arctic.


9


REFERENCES


S. Allende, A. M. Treguier, C. Lique, C. B. Mont´egut, F. Massonnet, T. Fichefet, and A. Barthelemy.
Impact of ocean vertical-mixing parameterization on arctic sea ice and upper-ocean properties
using the nemo-si3 model. _Geoscientific_ _Model_ _Development_, 17(20):7445–7466, 2024. doi:
https://doi.org/10.5194/gmd-17-7445-2024.


R. Arcucci, J. Zhu, S. Hu, and Y. Guo. Deep data assimilation: Integrating deep learning with data
assimilation. _Applied Sciences_, 11(3), 2021. doi: https://doi.org/10.3390/app11031114.


J. Aublanc, J. Renou, F. Piras, K. Nielsen, S. K. Rose, S. B. Simonsen, S. Fleury, S. Hendricks,
N. Taburet, G. D’Apice, A. Chamayou, P. Femenias, F. Catapano, and M. Restano. Sentinel-3
altimetry thematic products for hydrology, sea ice and land ice. _Scientific_ _Data_, 12(714), 2025.
doi: https://doi.org/10.1038/s41597-025-04956-3.


D. Babb, R. Galley, S. Kirillov, J. Landy, S. Howell, J. Stroeve, W. Meier, J. Ehn, and D. Barber.
The stepwise reduction of multiyear sea ice area in the arctic ocean since 1980. _JGR Oceans_, 128
(10), 2023. doi: https://doi.org/10.1029/2023JC020157.


R. Bannister. A review of forecast error covariance statistics in atmospheric variational data assimilation. i: Characteristics and measurements of forecast error covariances. _Quarterly Journal of the_
_Royal Meteorological Society_, 134(637):1951–1970, 2008a. doi: https://doi.org/10.1002/qj.339.


R. Bannister. A review of forecast error covariance statistics in atmospheric variational data assimilation. ii: Modelling the forecast error covariance statistics. _Quarterly Journal of the Royal_
_Meteorological Society_, 134(637):1971–1996, 2008b. doi: https://doi.org/10.1002/qj.339.


S. Barthelemy, J. Brajard, L. Bertino, and F. Counillon. Super-resolution data assimilation. _Ocean_
_Dynamics_, 72:661–678, 2022. doi: https://doi.org/10.1007/s10236-022-01523-x.


M. Blanke, R. Fablet, and M. Lelarge. Neural incremental data assimilation. _arXiv:2406.15076_,
2024. [URL https://arxiv.org/abs/2406.15076.](https://arxiv.org/abs/2406.15076) Accessed: 2025-03-19.


P. Brasseur. Ocean data assimilation using sequential methods based on the kalman filter. In
_Ocean_ _Weather_ _Forecasting:_ _An_ _Integrated_ _View_ _of_ _Oceanography_, pp. 271–316. Springer Science+Business Media B.V., 2006. doi: https://doi.org/10.1007/1-4020-4028-8.


S. Cai, F. Fang, and Y. Wang. Advancing neural network-based data assimilation for large-scale
spatiotemporal systems with sparse observations. _Physics_ _of_ _Fluids_, 36(9), 2024. doi: https:
//doi.org/10.1063/5.0228384.


A. Carrassi, M. Bocquet, L. Bertino, and G. Evensen. Data assimilation in the geosciences: An
overview of methods, issues, and perspectives. _WIREs Climate Change_, 9(5), 2018. doi: https:
//doi.org/10.1002/wcc.535.


X. Chen, C. Liang, D. Huang, E. Real, K. Wang, Y. Liu, H. Pham, X. Dong, T. Luong, C.-J. Hsieh,
Y. Lu, and Q. V. Le. Symbolic discovery of optimization algorithms. 2023. doi: https://doi.org/
10.48550/arXiv.2302.06675.


Copernicus Climate Change Service (C3S). Copernicus climate data store. [Online]. [URL https:](https://cds.climate.copernicus.eu/)
[//cds.climate.copernicus.eu/.](https://cds.climate.copernicus.eu/) Accessed: 2025-03-19.


P. Courtier, E. Andersson, W. Heckley, D. Vasiljevic, M. Hamrud, A. Hollingsworth, F. Rabier,
M. Fisher, and J. Pailleux. The ecmwf implementation of three-dimensional variational assimilation (3d-var). i: Formulation. _Quarterly Journal of the Royal Meteorological Society_, 124(550):
1783–1807, 1998. doi: https://doi.org/10.1002/qj.49712455002.


I. de Gelis, A. Colin, and N. Longepe. Prediction of categorized sea ice concentration from sentinel1 sar images based on a fully convolutional network. _IEEE_ _Journal_ _of_ _Selected_ _Topics_ _in_ _Ap-_
_plied_ _Earth_ _Observations_ _and_ _Remote_ _Sensing_, 14:5831–5841, 2021. doi: 10.1109/JSTARS.
2021.3074068.


M. Dvoynikov, G. Buslaev, A. Kunshin, D. Sidorov, A. Kraslawski, and M. Budovskaya. New
concepts of hydrogen production and storage in arctic region. _Resources_, 10(3), 2021. doi:
https://doi.org/10.3390/resources10010003.


10


G. Evensen. Sequential data assimilation with a nonlinear quasi-geostrophic model using monte
carlo methods to forecast error statistics. _Journal of Geophysical Research_, 99(C5):10143–10162,
1994. doi: https://doi.org/10.1029/94JC00572.


G. Evensen. The ensemble kalman filter: theoretical formulation and practical implementation.
_Ocean Dynamics_, 53:343–367, 2003. doi: https://doi.org/10.1007/s10236-003-0036-9.


A. Farchi, M. Bocquet, P. Laloyaux, M. Bonavita, and Q. Malartic. A comparison of combined data
assimilation and machine learning methods for offline and online model error correction. _Journal_
_of Computational Science_, 55, 2021. doi: https://doi.org/10.1016/j.jocs.2021.101468.


G. Gaspari and S. Cohn. Construction of correlation functions in two and three dimensions. _Quar-_
_terly Journal of the Royal Meteorological Society_, 125(554), 1999. doi: https://doi.org/10.1002/
qj.49712555417.


S. Hatfield, M. Chantry, P. Dueben, P. Lopez, A. Geer, and T. Palmer. Building tangent-linear and
adjoint models for data assimilation with neural networks. _Journal of Advances in Modeling Earth_
_Systems_, 13(9):1–16, 2021. doi: https://doi.org/10.1029/2021MS002521.


H. Jazwinski. _Stochastic Processes and Filtering Theory_ . Academic Press, New York and London,
1970.


R. Kalman. A new approach to linear filtering and prediction problems. _Journal of Fluids Engineer-_
_ing_, 82(1):35–45, 1960. doi: https://doi.org/10.1115/1.3662552.


D. Kingma and M. Welling. Auto-encoding variational bayes. _International Conference on Learning_
_Representations (ICLR)_, 2014. doi: https://doi.org/10.48550/arXiv.1312.6114.


S. Kortsch, R. Primicerio, M. Fossheim, A. Dolgov, and M. Aschan. Climate change alters the
structure of arctic marine food webs due to poleward shifts of boreal generalists. _Proceedings of_
_the Royal Society B_, 282(1814), 2015. doi: https://doi.org/10.1098/rspb.2015.1546.


K. A. Lisæter, J. Rosanova, and G. Evensen. Assimilation of ice concentration in a coupled
ice–ocean model, using the ensemble kalman filter. _Ocean_ _Dynamics_, 53:368–388, 2003. doi:
https://doi.org/10.1007/s10236-003-0049-4.


J. Mack, R. Arcucci, M. Molina-Solana, and Y. Guo. Attention-based convolutional autoencoders
for 3d-variational data assimilation. _Computer Methods in Applied Mechanics and Engineering_,
372:1–33, 2020. doi: https://doi.org/10.1016/j.cma.2020.113291.


B. Melinc and Z. Zaplotnik. 3d-var data assimilation using a variational autoencoder. _Quarterly_
_Journal of the Royal Meteorological Society_, 150(761):2273–2295, 2024. doi: https://doi.org/10.
1002/qj.4708.


E. Moreno-Chamarro, T. Arsouze, M. Acosta, P.-A. Bretonniere, M. Castrillo, E. Ferrer, A. Frigola,
D. Kuznetsova, E. Martin-Martinez, P. Ortega, and S. Palomas. The very-high-resolution configuration of the ec-earth global model for highresmip. _Geoscientific Model Development_, 18(2):
461–482, 2025. doi: https://doi.org/10.5194/gmd-18-461-2025.


National Aeronautics and Space Administration. Nasa earth observing system data and information
system (eosdis). [Online]. [URL https://www.earthdata.nasa.gov/.](https://www.earthdata.nasa.gov/)


E. Olason, P. Rampal, and V. Dansereau. On the statistical properties of sea-ice lead fraction and
heat fluxes in the arctic. _The_ _Cryosphere_, 15:1053–1064, 2021. doi: https://doi.org/10.5194/
tc-15-1053-2021.


R. Pan, Q. Shu, Q. Wang, S. Wang, Z. Song, Y. He, and F. Qiao. Future arctic climate change in
cmip6 strikingly intensified by nemo-family climate models. _Geophysical_ _Research_ _Letters_, 50
(4), 2023. doi: https://doi.org/10.1029/2022GL102077.


X. Pang, J. Pu, X. Zhao, Q. Ji, M. Qu, and Z. Cheng. Comparison between amsr2 sea ice concentration products and pseudo-ship observations of the arctic and antarctic sea ice edge on cloud-free
days. _Remote Sensing_, 10(2):317, 2018. doi: https://doi.org/10.3390/rs10020317.


11


S. Penny, T. Smith, T. Chen, J. Platt, H. Lin, M. Goodliff, and H. Abarbanel. Integrating recurrent neural networks with data assimilation for scalable data-driven state estimation. _Jour-_
_nal_ _of_ _Advances_ _in_ _Modeling_ _Earth_ _Systems_, 14(3):1–25, 2022. doi: https://doi.org/10.1029/
2021MS002843.


M. Peyron, A. Fillion, S. Gurol, V. Marchais, S. Gratton, P. Boudier, and G. Goret. Latent space
data assimilation by using deep learning. _Quarterly Journal of the Royal Meteorological Society_,
147(740):3759–3777, 2021. doi: https://doi.org/10.1002/qj.4153.


F. Rabier, H. Jarvinen, E. Klinker, J.-F. Mahfouf, and A. Simmons. The ecmwf operational implementation of four-dimensional variational assimilation. i: Experimental results with simplified
physics. _Quarterly Journal of the Royal Meteorological Society_, 126(564):1143–1170, 2000. doi:
https://doi.org/10.1002/qj.49712656415.


R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer. High-resolution image synthesis
with latent diffusion models. In _IEEE/CVF Conference on Computer Vision and Pattern Recog-_
_nition (CVPR)_, pp. 10674–10685, 2022. doi: https://doi.org/10.1109/CVPR52688.2022.01042.


J. Selivanova, D. Iovino, and F. Cocetta. Past and future of the arctic sea ice in high-resolution model
intercomparison project (highresmip) climate models. _The Cryosphere_, 18:2739–2763, 2024. doi:
https://doi.org/10.5194/tc-18-2739-2024.


Y. Shu, Y. Zhu, F. Xu, L. Gan, P. Lee, J. Yin, and J. Chen. Path planning for ships assisted by
the icebreaker in ice-covered waters in the northern sea route based on optimal control. _Ocean_
_Engineering_, 267, 2023. doi: https://doi.org/10.1016/j.oceaneng.2022.113182.


G. Spreen, L. Kaleschke, and G. Heygster. Sea ice remote sensing using amsr-e 89 ghz channels.
_Journal of Geophysical Research_, 113:C02S03, 2008. doi: 10.1029/2005JC003384.


X. Tian. Jacobian-enforced neural networks (jenn) for improved data assimilation consistency in dynamical models. _arXiv:2412.01013_ [, 2024. URL https://arxiv.org/abs/2412.01013.](https://arxiv.org/abs/2412.01013)
Accessed: 2025-03-19.


O. Zavala-Romero, A. Bozec, E. Chassignet, and J. Miranda. Convolutional neural networks for
sea surface data assimilation in operational ocean models: test case in the gulf of mexico. _Ocean_
_Science_, 21(1):113–132, 2025. doi: https://doi.org/10.5194/os-21-113-2025.


Q. Zhao, Q. Zhao, S. Peng, J. Wang, S. Li, Z. Hou, and G. Zhong. Applications of deep learning
in physical oceanography: a comprehensive review. _Frontiers in Marine Science_, 11, 2024. doi:
https://doi.org/10.3389/fmars.2024.1396322.


A APPENDIX


A.1 ALGORITHM OF MODIFICATION OF RESTARTS IN THE NEMO OCEAN MODEL


With initial restart files, written by the ocean model and new assimilated fields that came out of
assimilation algorithm following was applied:


1. The _sicon_ concentration values were clipped to the [0, 1] range to ensure physical consistency. Values below 0.01 were treated as zero, and a mask was created based on this
threshold to distinguish ice from open water. The corrected _sicon_ data is written to the _a_ ~~_i_~~
variable.


2. Ice thickness and concentration are used to calculate the ice volume per unit area _v_ ~~_i_~~, using
the formula _v_ ~~_i_~~ = _a_ _i · sithic_ .


3. Since snow cannot lie on water, the snow volume per unit area ( _v_ ~~_s_~~ ) and the ice concentration ( _a_ ~~_i_~~ ) from the restart file are used to calculate the snow thickness. This snow thickness
is then multiplied by the new ice concentration ( _a_ ~~_i_~~ ): _v_ ~~_s_~~ = _[v]_ _a_ ~~_[s]_~~ [[] _[rest]_ _i_ [ _rest_ []] _[·][a]_ ] ~~_[i]_~~ [.]


4. The _snwice_ ~~_m_~~ _ass_ and _snwice_ ~~_m_~~ _ass_ ~~_b_~~ values were recalculated from _v_ ~~_i_~~ and _v_ ~~_s_~~ using
the average densities of sea ice and snow.


12


5. The salinity variable _sv_ ~~_i_~~ is recalculated for the new ice volume _v_ ~~_i_~~ . A value of 7.4 psu
was assigned for newly formed ice.


6. The values of _e_ ~~_s_~~ ~~_l_~~ 01, _e_ ~~_i_~~ ~~_l_~~ 01, and _e_ ~~_i_~~ ~~_l_~~ 02 are recalculated proportionally to _v_ ~~_s_~~ and _v_ ~~_i_~~
taking into account the heat capacity and latent heat of fusion for snow and ice.


7. The internal stress variables _stress_ 1 ~~_i_~~, _stress_ 2 ~~_i_~~, and _stress_ 12 ~~_i_~~ are recalculated proportionally to the ice volume _v_ ~~_i_~~ .


8. The _oa_ ~~_i_~~ value is scaled proportionally with _a_ ~~_i_~~ .


9. The ocean state components of the restart file were left unmodified, as the change in ice
concentration was not substantial.


A.2 ADDITIONAL ASSIMILATION PROCESS ILLUSTRATIONS


These are examples of assimilated sea ice fields for model-to-model assimilation experiments and
sat-to-model experiments.


Figure 6: Results M2M assimilation for 12 April with use 3 _d_ ~~_v_~~ _ar_ model. sicinc _−_ sea ice concentration. Columns: background _−_ initial values, assimilation result _−_ M2M output, target _−_ reference
values, background error = background _−_ target (initial field deviation), assimilation error = assimilation result _−_ target (post assimilation deviation), change = assimilation result _−_ background
(assimilation induced adjustment).


A.3 THE USE OF LARGE LANGUAGE MODELS (LLMS)


The text in the paper was initially written by humans and then sent to LLM in the process to suggest
stylistic improvements and to correct grammar and punctuation mistakes.


13


Figure 7: Results M2M assimilation for 30 May with use _vae_ ~~4~~ _f_ model. Rows: siconc _−_ sea ice
concentration, sithic _−_ sea ice thickness, sosstsst _−_ sea surface temperature. Columns: background

_−_ initial values, assimilation result _−_ M2M output, target _−_ reference values, background error =
background _−_ target (initial field deviation), assimilation error = assimilation result _−_ target (post
assimilation deviation), change = assimilation result _−_ background (assimilation induced adjustment).


14


Figure 8: NEMO model prediction results. The forecast was initialized on February 22, 2023, and
run for 5 days. Top row: Ice concentration values predicted by the model without data assimilation.
Middle row: Ice concentration values predicted by the model with data assimilation (assimilation
was performed only on the initial day). Bottom row: Reference AMSR2 satellite ice concentration
data for comparison.


15