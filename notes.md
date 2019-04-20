# data preparation notes

## drop all missing values

- observations 5114 to 719. GCV: 117.594
- all observations: 182.5323

- imputing many missing values deteriorated model performance signficantly: 55.22% increase in GCV

## features with missing numbers

features with missing values: `pm10median`, `pm25median`, `so2median`

- `pm25median` is missing many values and they're not distributed randomly. Drop.
- `pm10median` isn't distributed randomly. Try to predict with simple linear model.
- `so2median` is probably distributed randomly. Fill missing with median value.

## gamma distribution

- 4 features: 0.0125