# Random-Sample-Error-for-Canopy-Classification
A simulation of error margins in canopy classification using random sample point generation, similar to the i-Tree Canopy methods.

# i-Tree Canopy Calculation of Standard Error and Confidence Intervals
Source: https://canopy.itreetools.org/references

In photo‐interpretation, randomly selected points are laid over aerial imagery and an interpreter classifies each point into a cover class (e.g., tree, building, water). From this classification of points, a statistical estimate of the
amount or percent cover in each cover class can be calculated along with an estimate of uncertainty of the estimate (standard error (SE)). To illustrate how this is done, let us assume 1,000 points have been interpreted and 
classified within a city as either “tree” or “non‐tree” as a means to ascertain the tree cover within that city, and 330 points were classified as “tree”.

**To calculate the percent tree cover and SE, let:**  
N = total number of sampled points (i.e, 1,000)  
n = total number of points classified as tree (i.e., 330), and  
p = n/N (i.e., 330/1,000 = 0.33)  
q = 1 – p (i.e., 1 ‐ 0.33 = 0.67)

And using the formula
<div align="center">
SE = √ (pq/N)  
</div>

<br>Thus in this example, tree cover in the city is estimated at 33% with a SE of 1.5%. Based on the SE formula, SE is greatest when p=0.5 and least when p is very small or very large (Table 1).
<div align="center">
SE = √ (0.33 x 0.67 / 1,000) = 0.0149
</div>

<br>**Table 1. Estimate of SE**  
(N = 1000) with varying p.  
p | SE  
0.01 | 0.0031  
0.10 | 0.0095  
0.30 | 0.0145  
0.50 | 0.0158  
0.70 | 0.0145  
0.90 | 0.0095  
0.99 | 0.0031

In the case above, a 95% confidence interval can be calculated. “Under simple random sampling, a 95% confidence interval procedure has the interpretation that for 95% of the possible samples of size n, the interval covers the true 
value of the population mean” (Thompson 2002). The 95% confidence interval for the above example is between 30.1% and 35.9%. To calculate a 95% confidence interval (if N>=30) the SE x 1.96 (i.e., 0.0149 x 1.96 = 0.029) is added 
to and subtracted from the estimate (i.e., 0.33) to obtain the confidence interval.

**Less than 10 points classified in category**  
If the number of points classified in a category (n) is less than 10, a different SE formula (Poisson) should be used as the normal approximation cannot be relied upon with a small sample size (<10) (Hodges and Lehmann, 1964).  
In this case, standard error is calculated using the formula  
<div align="center">
SE = (√n) / N
</div>

<br>For example, if n = 5 and N = 1000, p = n/N (i.e., 5/1,000 = 0.005) and SE = √5 / 1000 = 0.0022. Thus the tree cover estimate would be 0.5% with a SE of 0.22%.

# The Impact of Spatial Autocorrelation
Standard Error (SE) as calculated by i-Tree Canopy assumes independence among the underlying observations.

The SE formula

<div align="center">
SE = √ (pq/N)  
</div>

<br> is derived from the binomial distribution, which models the number of successes (e.g., tree cover) in a fixed number of independent Bernoulli trials. The key assumptions are:

1. Independence: Each point (e.g., each pixel or sample location) must be classified independently of others.
2. Identical probability: Each point has the same probability 𝑝 of being classified as "tree".

If the points are spatially autocorrelated (common in land cover data), the SE will be **underestimated**, leading to **overconfidence** in the precision of your estimate.

One approach is to calculate an adjusted standard error as

<div align="center">
SEadjusted = √ (p(1 - p)/Neff) = √(p(1-p)/N) x √ DEFF   
</div>

<br>where

<div align="center">
DEFF = 1 + (nc - 1) * ρ 
</div>

<br> in which nc is the avearge cluster size and ρ is intraclass correlation or spatial autocorrelation coefficient, which can be esimated via Moran's i.

<br> or where

<div align="center">
Neff = N/ (1 + (N - 1)ρ
</div>

<br> This yields the revised formula

<div align="center">
Neff = N/ (1 + (N - 1)ρ
</div>
