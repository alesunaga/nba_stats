import numpy as np
import pandas as pd
from scipy.stats import pearsonr, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress scientific notation and set precision for NumPy outputs
np.set_printoptions(suppress=True, precision=2)

# Load the NBA dataset
nba = pd.read_csv('./nba_games.csv')

# Subset the dataset for 2010 and 2014 seasons
nba_2010 = nba[nba.year_id == 2010]
nba_2014 = nba[nba.year_id == 2014]

# ---------------------------------------------
# Comparing Knicks and Nets in the 2010 season
# ---------------------------------------------
# Extract points scored by Knicks and Nets
knicks_pts = nba_2010[nba_2010.fran_id == 'Knicks']['pts']
nets_pts = nba_2010[nba_2010.fran_id == 'Nets']['pts']

# Calculate the difference in average points between Knicks and Nets
diff_means_2010 = knicks_pts.mean() - nets_pts.mean()
print(f"2010 Season - Average points difference (Knicks - Nets): {diff_means_2010:.2f}")

# Plot histograms of points scored by Knicks and Nets
plt.hist(knicks_pts, alpha=0.8, density=True, label='Knicks', bins=15)
plt.hist(nets_pts, alpha=0.8, density=True, label='Nets', bins=15)
plt.legend()
plt.title("2010 Season - Knicks vs Nets")
plt.xlabel("Points")
plt.ylabel("Frequency")

# Save the histogram as a PNG file
plt.savefig('2010_season_histogram.png', dpi=300)  # dpi=300 for higher resolution
plt.show()

# ---------------------------------------------
# Comparing Knicks and Nets in the 2014 season
# ---------------------------------------------
# Extract points scored by Knicks and Nets
knicks_pts_2014 = nba_2014[nba_2014.fran_id == 'Knicks']['pts']
nets_pts_2014 = nba_2014[nba_2014.fran_id == 'Nets']['pts']

# Calculate the difference in average points between Knicks and Nets
diff_means_2014 = knicks_pts_2014.mean() - nets_pts_2014.mean()
print(f"2014 Season - Average points difference (Knicks - Nets): {diff_means_2014:.2f}")

# Plot histograms of points scored by Knicks and Nets
plt.hist(knicks_pts_2014, alpha=0.8, density=True, label='Knicks', bins=15)
plt.hist(nets_pts_2014, alpha=0.8, density=True, label='Nets', bins=15)
plt.legend()
plt.title("2014 Season - Knicks vs Nets")
plt.xlabel("Points")
plt.ylabel("Frequency")

# Save the histogram as a PNG file
plt.savefig('2014_season_histogram.png', dpi=300)  # dpi=300 for higher resolution
plt.show()

# ---------------------------------------------
# Boxplot for points scored in the 2010 season
# ---------------------------------------------
# Create a boxplot of points scored by teams in the 2010 season
plt.clf()  # Clear previous plots
sns.boxplot(data=nba_2010, x='fran_id', y='pts')
plt.title("2010 Season - Points by Team")
plt.xlabel("Team")
plt.ylabel("Points")
plt.show()

# ---------------------------------------------
# Chi-Square Test on Game Location vs Result
# ---------------------------------------------
# Create a contingency table for game location and result
location_result_freq = pd.crosstab(nba_2010.game_result, nba_2010.game_location)
print("Contingency Table - Game Result vs Location:")
print(location_result_freq)

# Calculate proportions
location_result_proportions = location_result_freq / len(nba_2010)
print("Proportion Table - Game Result vs Location:")
print(location_result_proportions)

# Perform a Chi-Square test
chi2, pval, dof, expected = chi2_contingency(location_result_proportions)
print("Expected Frequencies (if no association):")
print(expected)
print(f"Chi-Square Statistic: {chi2:.2f}")
print(f"P-Value: {pval:.4f}")

# ---------------------------------------------
# Correlation between Forecast and Point Differential
# ---------------------------------------------
# Calculate covariance and correlation
covariance = np.cov(nba_2010.forecast, nba_2010.point_diff)[0, 1]
print(f"Covariance between forecast and point differential: {covariance:.2f}")

corr, pval = pearsonr(nba_2010.forecast, nba_2010.point_diff)
print(f"Pearson Correlation: {corr:.2f}, P-Value: {pval:.4f}")

# Scatter plot of forecast vs point differential
plt.clf()  # Clear previous plots
plt.scatter(nba_2010.forecast, nba_2010.point_diff, alpha=0.7)
plt.title("2010 Season - Forecast vs Point Differential")
plt.xlabel("Forecasted Win Probability")
plt.ylabel("Point Differential")
plt.show()
