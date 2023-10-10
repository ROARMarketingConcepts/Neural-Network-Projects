# KL Divergence Derivation

- $P(x)$ unknown distribution,

- $Q(x)$ known distribution (e.g., Gaussian distribution)

### Forward KL-Divergence:

$D_{KL}(P\parallel Q) = \sum P(X)\log\left(\frac{P(x)}{Q(x)}\right)$

### Reverse KL-Divergence (used in VAE):

$D_{KL}(Q\parallel P) = \sum Q(X)\log\left(\frac{Q(x)}{P(x)}\right)$

More specifically,

$D_{KL}\left[Q(z|X)\parallel P(z|X)\right] = \sum_{z}Q(z|X)\log\left(\frac{Q(z|X)}{P(z|X)}\right) = E\left[\log\left(\frac{Q(z|X)}{P(z|X)}\right)\right]$

$= E\left[\log Q(z|X) - \log P(z|X)\right]$

$= E\left[\log Q(z|X) - \log \frac{P(X|z)P(z)}{P(X)}\right]$, using Bayes Theorem equivalent for $P(z|X)$

$D_{KL}\left[Q(z|X)\parallel P(z|X)\right]= E\left[\log Q(z|X) - (\log P(X|z)+\log P(z)-\log P(X)) \right]$

$= E\left[\log Q(z|X) -\log P(X|z)-\log P(z)+\log P(X) \right]$

Note that $E$ is dependent on $z$, so we can take $\log P(X)$ out of the brackets.

$D_{KL}\left[Q(z|X)\parallel P(z|X)\right]=E\left[\log Q(z|X) -\log P(X|z)-\log P(z)\right]+\log P(X)$ 

$D_{KL}\left[Q(z|X)\parallel P(z|X)\right]-\log P(X)=E\left[\log Q(z|X) -\log P(X|z)-\log P(z)\right]$ 

$\log P(X)-D_{KL}\left[Q(z|X)\parallel P(z|X)\right]=E\left[\log P(X|z)+\log P(z)-\log Q(z|X)\right]$ 

$=E[\log P(X|z)]-E[\log Q(z|X)-\log P(z)] = E[\log P(X|z)]-D_{KL}[Q(z|X)\parallel P(z)]$

$\therefore \space \boxed{\log P(X)-D_{KL}\left[Q(z|X)\parallel P(z|X)\right]=E[\log P(X|z)]-D_{KL}[Q(z|X)\parallel P(z)]}$

Our objective is $E[\log P(X|z)]-D_{KL}[Q(z|X)\parallel P(z)]$,  which we need to maximize.  We will multiply the objective by $-1$ so that we can minimize (i.e., gradient descent).

Therefore, our loss function is $-E[\log P(X|z)]-D_{KL}[Q(z|X)\parallel P(z)]$, where $\log P(X|z)$ is the reconstruction loss mapping from $z$ to $X$.

Moreover, we want to minimize $D_{KL}[Q(z|X)\parallel P(z)]$, which is the difference between our simple distribution, $Q(z|X)$, and the latent distribution, $P(z)$, which is $\mathcal{N} (0,1)$.  We want $P(z)$ to be as close to $Q(z|X)$ as possible, so that we can sample it easily.

So if we want $Q(z|X)$ to be Gaussian/Normal with learned parameters $\mu(X)$ and variance $\sigma^2(X)$, the KL Divergence is:

$D_{KL}\left[\mathcal{N}(\mu(X),\sigma^2(X))\parallel \mathcal{N}(0,1)\right]=\frac{1}{2}\left[tr(\sigma^2(X))+\mu(X)^T\mu(X)-k-\log \det(\sigma^2(X))\right]$, 

where $k$ is the dimension of our Gaussian and $tr(X)$ is the trace function, i.e., the sum of the diagonal of the matrix $X$.

