# Bootstrapping
Lucas De Oliveira, Kooha Kwon, Melvin Vellera

## Traditional vs. Boostrap Estimations

### Estimation of Population Mean Using Tradtional and Bootstrap Methods

Now let's use the real dataset to compare the traditional and bootstrap statistical estimations. Here, we have 252 rows of data of bodyfat percentage. We are going to treat this dataset as a population. The population mean of the data is 18.94%.

However, we are going to assume we only have 10 data points and build a 95% Confidence Interval with two methods. The data points were randomly sampled from the population using the following code:

```
samp_n = 10
sample_bfp = bfp.loc[random.sample(range(252), samp_n)]
sample_bfp['BODYFAT']
```

Traditionally, we can use Student's t-statistics in order to calculate the interval. The methodology requires estimation of standard deviation, standard error, and critical t*. The following equations can be used to calculate the 95% CI of the population mean.

![CI_Equations](https://github.com/lbdeoliveira/msds610_code_presentation/blob/master/assets/images/CI_Equations.jpg)

Now, let's use bootstrapping to estimate. As stated before, we first need to resample the sample multiple times to simulate sample distribution. And then, we calculate the mean for each simulation and collect all the means into a list. 

```
resample_means = []

for i in range(10_000):                              # Repeating Simulation 10,000 Times
    inds = np.random.randint(0, samp_n-1, samp_n-1)  # Randomly Pulling Samples
    mean = sample_bfp['BODYFAT'].iloc[inds].mean()   # Calculating Simulated Mean
    resample_means.append(mean)                      # Collecting Means into a List

```

Using the list of simulated means, we can draw a histogram. From the histogram, we could simply locate 2.5th and 97.5th percentile values, and that will be the lower and upper mean of our distribution.

![Bootstrap_Mean_Histogram](https://github.com/lbdeoliveira/msds610_code_presentation/blob/master/assets/images/Bootstrap_Mean_Histogram.jpg){:height="50%" width="50%"}

The table below shows the true population distribution, mean estimation using the traditional method, and estimation using the bootstrapping method. Both traditional and bootstrapping methods were able to successfully include the true population mean within their 95% CI. However, the bootstrapping method outperforms the traditional method with a tighter interval range.

![Mean_Results](https://github.com/lbdeoliveira/msds610_code_presentation/blob/master/assets/images/Mean_Results.jpg){:height="50%" width="50%"}


### What if you do not know the analytical solution for calculating the confidence interval of a statistic?

Let's take the example of Adjusted R2 in our example. Also, for the sake of simplicity we will refer to Adjusted R2 as R2 from now.

We first calculate R2 for the entire dataset and assume that this is the population R2 that we need to estimate from a sample.
```
model = smf.ols('BODYFAT ~ ABDOMEN + WRIST + ANKLE + AGE', data=bfp).fit()
pop_adj_rsquared = model.rsquared_adj
print(f'Population Adjusted R2: {pop_adj_rsquared:.2f}')
```
> **Population Adjusted R2: 0.71**

We now take a random sample of 30 rows from the entired dataset and assume that we do not have the rest of the data
```
samp_n = 30
sample_bfp = bfp.loc[random.sample(range(pop_n), samp_n)]
```
Let's see what the sample R2 looks like
```
sample_model = smf.ols('BODYFAT ~ ABDOMEN + WRIST + ANKLE + AGE',
                       data=sample_bfp).fit()
samp_adj_rsquared = sample_model.rsquared_adj
print(f'Sample Adjusted R2: {samp_adj_rsquared:.2f}')
```
> **Sample Adjusted R2: 0.75**

As you see, the sample R2 might not match the population R2. It could also be wildly different if our sample is not representative of the population. So for bootstrapping to work, it is essential for the sample to be representative of the population. 
We know, by design, that our sample is representative of the population.
Now let's calculate a bootstrapped confidence interval for R2 from our sample.
```
adj_rsquareds = []
for i in range(1_000):                                            # Iterate a 1000 times 
    inds = np.random.randint(0, samp_n-1, samp_n-1)               # Random sampling with replacement 
    model = smf.ols('BODYFAT ~ ABDOMEN + WRIST + ANKLE + AGE',    # Fit model
                    data=sample_bfp.iloc[inds]).fit()             
    rsq_adj = model.rsquared_adj                                  # Get R2
    adj_rsquareds.append(rsq_adj)                                 # Store R2

boot_rsq_adj_mean = np.mean(adj_rsquareds)                        # Calculate mean of 1000 R2 values
boot_rsq_adj_ci = [np.quantile(adj_rsquareds, 0.025),             # Calculate 2.5th and 97.5th percentiles (R2)
                   np.quantile(adj_rsquareds, 0.975)]

print(f'Bootstrap Confidence Interval (Adj Rsquared): '
      + f'''{[f'{e:.2f}' for e in boot_rsq_adj_ci]}''')
print(f'Population Rsquared_adj: {pop_adj_rsquared:.2f}')
```
>**Bootstrap Confidence Interval (Adjusted R2): ['0.57', '0.93']<br>
>Population Adjusted R2: 0.71**

**As seen from the above output, the population R2 is indeed contained in the 95% confidence interval created from our sample!**

Let's see the histogram for our 1000 R2 values:
```
fig, ax = plt.subplots(figsize=(10, 8))

ax.hist(adj_rsquareds, alpha=.5, edgecolor='grey')
ax.vlines(x=boot_rsq_adj_mean, ymin=0, ymax=300, color='red')
ax.vlines(x=boot_rsq_adj_ci[0], ymin=0, ymax=80, color='red')
ax.vlines(x=boot_rsq_adj_ci[1], ymin=0, ymax=80, color='red')
ax.vlines(x=boot_rsq_adj_mean, ymin=0, ymax=280, color='red')

ax.annotate(f'{boot_rsq_adj_mean:.2f} (Mean)', (boot_rsq_adj_mean-0.007, 305))
ax.annotate(f'{boot_rsq_adj_ci[0]:.2f} (2.5th)',
            (boot_rsq_adj_ci[0]-0.007, 85))
ax.annotate(f'{boot_rsq_adj_ci[1]:.2f} (97.5th)',
            (boot_rsq_adj_ci[1]-0.007, 85))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_title('Distribution of Bootstrapped Sample Adj Rsquareds', size=16)

plt.show()
```
![R2 Histogram](https://github.com/lbdeoliveira/msds610_code_presentation/blob/master/assets/images/R2_Histogram.png)

From the above output, we can see that boostrapping can work even if the histogram (sampling distribution) is not a symmetric normal distribution. In fact, it does not even have to be normal! You just need the percentile values to calculate confidence intervals or to perform hypothesis testing.

### Summary
1. Bootstrapping work wells for **very small sample sizes**
2. For such small sample sizes, it can give more **precise confidence intervals** (i.e. smaller intervals) as compared to standard statistical methods 
3. Boostrapping can be used for almost ANY statistic! (even for ones that do not have a **normally distributed** sampling distribution, or for ones for which statistical calculations have **not been discovered** yet!)

To conclude, when everything else fails, you can always pull yourself up by your bootstraps and start bootstrapping to get some confidence (intervals).

**Thank you for reading!**
