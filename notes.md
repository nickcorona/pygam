# data preparation notes

## features with missing numbers

features with missing values: `pm10median`, `pm25median`, `so2median`

- `pm25median` is missing many values and they're not distributed randomly. Drop.
- `pm10median` isn't distributed randomly. Try to predict with simple linear model.
- `so2median` is probably distributed randomly. Fill missing with median value.