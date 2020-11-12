from matplotlib import rcParams
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

df = pd.read_csv('cardiovascular-disease-dataset/cardio.csv')
df = df.drop('id', axis=1)

pd.options.display.max_columns = df.shape[1]

print(df.describe())

labels = ['Diastolični krvni tlak', 'Sistolični krvni tlak']
x = np.arange(len(labels))

blood_pressure = df.loc[:, ['ap_lo', 'ap_hi']]
sns.boxplot(x='variable', y='value', data=blood_pressure.melt())
plt.xlabel("Krvna tlaka")
plt.ylabel("Vrednost")
# plt.savefig('figures/tlaka-outliers', bbox_inches='tight')
plt.clf()

print("Diastilic pressure is higher than systolic one in {0} cases".format(df[df['ap_lo'] > df['ap_hi']].shape[0]))
df.drop(df[(df['ap_hi'] > df['ap_hi'].quantile(0.9945)) | (df['ap_hi'] < df['ap_hi'].quantile(0.0046))].index, inplace=True)
df.drop(df[(df['ap_lo'] > df['ap_lo'].quantile(0.9842)) | (df['ap_lo'] < df['ap_lo'].quantile(0.002))].index, inplace=True)
df.drop(df[df['ap_lo'] > df['ap_hi']].index, inplace=True)

blood_pressure = df.loc[:, ['ap_lo', 'ap_hi']]
sns.boxplot(x='variable', y='value', data=blood_pressure.melt())
plt.xlabel("Krvna tlaka")
plt.ylabel("Vrednost")
# plt.savefig('figures/tlaka-popravljeno', bbox_inches='tight')
plt.clf()
print("Diastilic pressure is higher than systolic one in {0} cases".format(df[df['ap_lo'] > df['ap_hi']].shape[0]))

print(df.describe())

height_weight = df.loc[:, ['height', 'weight']]
sns.boxplot(x='variable', y='value', data=height_weight.melt())
plt.xlabel("")
plt.ylabel("centimetri / kilogrami")
# plt.savefig('figures/height-weight-range', bbox_inches='tight')
# plt.show()
plt.clf()

df.drop(df[(df['height'] > df['height'].quantile(0.999)) | (df['height'] < df['height'].quantile(0.001))].index, inplace=True)
df.drop(df[(df['weight'] > df['weight'].quantile(0.999)) | (df['weight'] < df['weight'].quantile(0.001))].index, inplace=True)

height_weight = df.loc[:, ['height', 'weight']]
sns.boxplot(x='variable', y='value', data=height_weight.melt())
plt.xlabel("")
plt.ylabel("centimetri / kilogrami")
# plt.savefig('figures/height-weight-range-fixed', bbox_inches='tight')
# plt.show()
plt.clf()

df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
df = df[['age', 'gender', 'height', 'weight', 'bmi', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']]

print(df.describe())
df_train = df.sample(frac=0.5, random_state=1)
temp_df = df.drop(df_train.index)
df_validation = temp_df.sample(frac=0.5, random_state=1)
df_test = temp_df.drop(df_validation.index)

df.to_csv('cardiovascular-disease-dataset/cardio_cleaned.csv', index=False)
df_train.to_csv('cardiovascular-disease-dataset/cardio_cleaned_train.csv', index=False)
df_validation.to_csv('cardiovascular-disease-dataset/cardio_cleaned_validation.csv', index=False)
df_test.to_csv('cardiovascular-disease-dataset/cardio_cleaned_test.csv', index=False)

bmi = df.loc[:, ['bmi']]
sns.boxplot(x='variable', y='value', data=bmi.melt())
plt.xlabel("")
plt.ylabel("Vrednost")
# plt.savefig('figures/height-weight-range-fixed', bbox_inches='tight')
# plt.show()
plt.clf()

corr = df.corr()
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, annot=True, square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.xticks(rotation=90)
# plt.savefig('figures/heatmap', bbox_inches='tight')
# plt.show()
plt.clf()


rcParams['figure.figsize'] = 11, 8
df['years'] = (df['age'] / 365).round().astype('int')
sns.countplot(x='years', hue='cardio', data=df)
plt.xlabel("Starost")
plt.ylabel("Število pacientov")
# plt.savefig('figures/influence_age_cardio', bbox_inches='tight')
# plt.show()
plt.clf()
