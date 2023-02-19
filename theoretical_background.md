# ShakemapSim - Theoretical Background

This short document provides the main theoretical background of the ShakemapSim tool. Conditional on a specified earthquake rupture, the first section explains the general procedure to simulate fields of ground-motion intensity measures (IMs) at spatially distributed sites of interest when no recordings are available. This simulation requires a specification of the joint distribution of ground-motion IMs at the corresponding sites. 

The second section shows how to perform such simulations if recordings are available, in which case the conditional joint distribution of ground-motion IMs is required. 

## Empirical models to simulate ground-motion IMs 

The following explanations are partially copied from our recent study, Bodenmann et al. (2023)[^1], and the notation is mostly based on the book of Baker et al. (2021)[^2]. The latter is also an excellent resource for readers that want to know more about seismic hazard and risk analysis.

### Ground-motion models

We focus on empirically derived ground-motion models (GMMs) that predict a ground-motion IM at site $i$ induced by an earthquake rupture $k$ as
$$
\ln \textit{IM}_{ki} = \mu_{\ln IM}(rup_k, site_i; \boldsymbol{\psi}_{\mathtt{GMM}}) + \delta B_k + \delta W_{ki}~,
$$
where $\mu_{\ln \textit{IM}}(\cdot)$ is the predicted mean $\ln {\textit{IM}}$ value as a function of rupture ($rup$) and site ($site$) characteristics. Amongst others, these typically include earthquake magnitude, rupture mechanism, source-to-site distance and site-specific geological information. The parameters of the GMM are denoted as $\boldsymbol{\psi}_{\mathtt{GMM}}$. The between-event and within-event residuals, $\delta B_k$ and $\delta W_{ki}$, are assumed to be independent, normally distributed variables with standard deviations $\tau$ and $\phi$, respectively. Empirical GMMs provide the mean function $\mu_{\ln \textit{IM}}(\cdot)$, as well as the standard deviations $\tau$ and $\phi$. For a specific event, the between-event residual denotes a common deviation from the predicted mean that is constant for all sites, whereas the within-event residuals vary in space. This is discussed in the following section. 

> **_Note:_** We use GMM implementations from OpenQuake. Currently we only used the models `AkkarEtAl2014Rjb` and `CaucciEtAl2014`, with each having 35% weight in the USGS logic tree used for the M7.8 earthquake at the border of Turkey and Syria. Other models can easily be added by following the OpenQuake documentation. However, some models may require additional $site$ and/or $rup$ characteristics than the ones considered so far.

### Spatial correlation models

Conditional on rupture $rup_k$, the joint distribution of ground-motion IMs at $n$ spatially distributed sites $\mathbf{IM}_k=(\textit{IM}_{k1},\ldots,\textit{IM}_{kn})^\top$ is commonly assumed to be a multivariate lognormal distribution 
$$
\ln{\mathbf{IM}_k} \sim \mathcal{N}(\boldsymbol{\mu},\tau^2 + \phi^2 \cdot \mathbf{C} )~,
$$
where $\boldsymbol{\mu}$ is the mean vector with entries derived from the mean function $\mu_{\ln \textit{IM}}(\cdot)$ of the GMM, and $\mathbf{C}$ is the correlation matrix of the within-event residuals $\boldsymbol{\delta} \mathbf{W}_{k}$. To compute the entries of this matrix we employ a model $\rho(\cdot)$ that predicts the correlation between two sites $site_i$ and $site_j$ as 
$$
[\mathbf{C}]_{ij}=\rho(site_i,site_j; \boldsymbol{\psi}_{\mathtt{SCM}})~, 
$$
where $\boldsymbol{\psi}_{\mathtt{SCM}}$ denotes the spatial correlation model (SCM) parameters. Commonly, these models are defined for a distance metric $d$ between two sites. Thus, we often denote the correlation model as $\rho(d; \boldsymbol{\psi}_{\mathtt{SCM}})$. The vast majority of SCMs proposed in the literature assume that correlation decreases exponentially with the Euclidean distance, $d_\mathrm{E}$, between two sites as
$$
\rho(d_\mathrm{E}; \boldsymbol{\psi}_{\mathtt{SCM}})=\exp\left[- \left(\frac{d_\mathrm{E}}{\ell}\right)^\gamma\right]~,
$$
where parameters $\boldsymbol{\psi}_{\mathtt{SCM}}=(\ell,\gamma)$, denote the lengthscale and the exponent, respectively. 

Having computed the parameters of the multivariate normal distribution, sampling of $\ln im$ values is done by using methods implemented in other packages (such as `numpy`). 

> **_Note:_** Currently we implemented following models: `EspositoIervolino2012esm`, `HeresiMiranda2019` and `BodenmannEtAl2023`. The first two models depend only on $d_\mathrm{E}$, while the last model additionally accounts for site and path effects, and thus also depends on the earthquake rupture. You find all references in the documentation of the [source files](modules/spatialcorrelation.py) which should also help you to easily add other SCMs.

## Account for recorded intensity measure values
We use the subscript $\mathcal{S}$ to indicate seismic network stations, with site characteristics $\mathbf{sites}_\mathcal{S}=\{site_{i} | i\in\mathcal{S}\}$, and recorded $im$ values $\mathbf{im}_\mathcal{S}$. Consequently, $\boldsymbol{\mu}_\mathcal{S}$ and $\mathbf{C}_\mathcal{SS}$ denote the mean vector, derived from the GMM, and the correlation matrix, derived from the SCM, respectively. 

The objective is to compute the parameters of the joint distribution of IM values at target sites $\mathbf{sites}_\mathcal{T}=\{site_{i} | i\in\mathcal{T}\}$ conditional on recordings $\mathbf{im}_\mathcal{S}$ from the seismic network stations. For that purpose, we follow the two-step procedure proposed by Worden et al. (2018)[^3], also discussed in Bodenmann et al. (2022)[^4]. 

### Compute the between-event residual
From above we recall that before seeing any data the between-event residual is assumed to be normally distributed with zero mean and standard deviation $\tau$. After observing data $\mathbf{im}_\mathcal{S}$, the parameters of the normal distribution are updated, e.g., $\delta B_k | \mathbf{im}_\mathcal{S} \sim \mathcal{N}(\mu_{\delta B|\mathbf{im}_\mathcal{S}}, \sigma_{\tau|\mathbf{im}_\mathcal{S}})$, with remaining variance 
$$
\sigma_{\delta B|\mathbf{im}_\mathcal{S}}^2 = \left( \frac{1}{\tau^2} + \frac{\mathbf{1}^\top \mathbf{C}_\mathcal{SS}^{-1} \mathbf{1}}{\phi^2} \right)^{-1} ~,
$$
and mean
$$
\mu_{\delta B|\mathbf{im}_\mathcal{S}} = \frac{\sigma_{\delta B|\mathbf{im}_\mathcal{S}}^2}{\phi^2} \left(\mathbf{1}^\top \mathbf{C}_\mathcal{SS}^{-1} (\ln{\mathbf{im}_\mathcal{S}}-\boldsymbol{\mu}_\mathcal{S})\right)~.
$$
> **_Note:_** To make the workflow more efficient, the above computations are conducted when initializing the `Shakemap` object where we also cache $\mathbf{C}_\mathcal{SS}^{-1}$. See the corresponding [source file](modules/shakemap.py) for further information.

### Compute the conditional predictive distribution
Then we estimate the conditional predictive distribution of logarithmic $im$ at the target sites $\mathbf{sites}_\mathcal{T}$, which is again a multivariate normal distribution 
$$
\ln{\mathbf{IM}}_\mathcal{T} | \mathbf{im}_\mathcal{S} \sim \mathcal{N}(\overline{\boldsymbol{\mu}}_{\mathcal{T}},\overline{\boldsymbol{\Sigma}}_{\mathcal{TT}}) ~.
$$ 

The mean vector of the conditional predictive distribution is computed as
$$
\overline{\boldsymbol{\mu}}_\mathcal{T} = \boldsymbol{\mu}_{\mathcal{T}} + \mu_{\delta B|\mathbf{im}_\mathcal{S}} + \mathbf{C}_{\mathcal{TS}}\mathbf{C}_{\mathcal{SS}}^{-1}(\ln{\mathbf{im}_\mathcal{S}}-\boldsymbol{\mu}_{\mathcal{S}}-\mu_{\delta B|\mathbf{im}_\mathcal{S}})~, 
$$
and the corresponding covariance matrix is
$$
\overline{\boldsymbol{\Sigma}}_{\mathcal{TT}} = (\phi^2+\sigma_{\delta B|\mathbf{im}_\mathcal{S}}^2) \left(\mathbf{C}_{\mathcal{TT}} - \mathbf{C}_{\mathcal{TS}}\mathbf{C}_{\mathcal{SS}}^{-1}\mathbf{C}_{\mathcal{TS}}^\top\right)~,
$$
where matrices $\mathbf{C}_{\mathcal{TS}}$ and $\mathbf{C}_{\mathcal{TT}}$ denote the correlation matrices between the target points and seismic stations and between the target points themselves derived with the SCM.

Having computed the parameters of the multivariate normal distribution, sampling of conditional $\ln im$ values is done by using methods implemented in other packages (such as `numpy`). 

## References

[^1]: Bodenmann L., Baker J. and Stojadinovic B. (2023): "Accounting for site and path effects in spatial ground-motion correlation models using Bayesian inference", Nat. Hazards Earth Syst. Sci. Discuss. (in review) doi: [10.5194/nhess-2022-267](https://doi.org/10.5194/nhess-2022-267)

[^2]: Baker J., Bradley B., and Stafford P. (2021): "Seismic Hazard and Risk Analysis", Cambridge University Press, doi: [10.1017/9781108425056](https://doi.org/10.1017/9781108425056)

[^3]: Worden B.C., Thompson E.M., Baker J.W., Bradley B., Luco N., and Wald D. (2018): "Spatial and Spectral Interpolation of Ground-Motion Intensity Measure Observations", Bulletin of the Seismological Society of America, doi: [10.1785/0120170201](https://doi.org/10.1785/0120170201)

[^4]: Bodenmann L., Reuland Y., and Stojadinović B. (2022): Dynamic post-earthquake updating of regional damage estimates using Gaussian processes (in review), doi: [10.31224/2205](https://doi.org/10.31224/2205)