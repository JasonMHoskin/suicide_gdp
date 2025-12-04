### SETUP & DATA LOADING ###
if (!require(tidyverse))
  install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if (!require(caret))
  install.packages("caret", repos = "http://cran.us.r-project.org")
if (!require(gridExtra))
  install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if (!require(ranger))
  install.packages("ranger", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(gridExtra)
library(ranger)

# Import dataset
master <- read_csv("https://raw.githubusercontent.com/JasonMHoskin/suicide_gdp/main/master.csv")

# Initial Name Standardization
master <- master %>%
  rename(
    suicide_100k_rate = `suicides/100k pop`,
    country_year = `country-year`,
    hdi_for_year = `HDI for year`,
    gdp_for_year = `gdp_for_year ($)`,
    gdp_per_capita = `gdp_per_capita ($)`,
    suicides_no = `suicides_no`
  ) %>%
  mutate(
    country = as.factor(country),
    sex = as.factor(sex),
    age = as.factor(age),
    generation = as.factor(generation)
  )

### Data Splitting ###
# Set seed for consistent results
set.seed(123)

# Create partition on the data
test_index <- createDataPartition(
  y = master$suicide_100k_rate,
  times = 1,
  p = 0.1,
  list = FALSE
)

train_raw <- master[-test_index, ]
holdout_raw <- master[test_index, ]

# Ensure holdout countries exist in train_raw
holdout_set <- holdout_raw %>% semi_join(train_raw, by = "country")
removed <- anti_join(holdout_raw, holdout_set, by = "country")
train_raw <- rbind(train_raw, removed)

# Determine new partition percentages
nrow(holdout_set) / nrow(master)
nrow(train_raw) / nrow(master)

# Remove now superfluous vectors
rm(test_index, holdout_raw, removed, master)

### Exploratory Data Analysis ###
# Verify missing data
print(colSums(is.na(train_raw)))

# Visualize data skewness
p_skew_gdp <- ggplot(train_raw, aes(x = gdp_per_capita)) +
  geom_histogram(bins = 30,
                 fill = "firebrick",
                 color = "black") +
  labs(title = "GDP", x = "GDP per Capita") +
  theme_minimal()

p_skew_suicide <- ggplot(train_raw, aes(x = suicide_100k_rate)) +
  geom_histogram(bins = 30,
                 fill = "firebrick",
                 color = "black") +
  labs(title = "Suicide Rate", x = "Suicides per 100k") +
  theme_minimal()

grid.arrange(p_skew_gdp, p_skew_suicide, ncol = 2)

# Imputation statistics—Calculated on Train Only
hdi_means <- train_raw %>%
  group_by(country) %>%
  summarize(mean_hdi = mean(hdi_for_year, na.rm = TRUE),
            .groups = "drop")
global_mean <- mean(train_raw$hdi_for_year, na.rm = TRUE)

# Function to apply cleaning and transformation
process_data <- function(df, hdi_ref) {
  df %>%
    left_join(hdi_ref, by = "country") %>%
    mutate(
      hdi_filled = ifelse(is.na(hdi_for_year), mean_hdi, hdi_for_year),
      hdi_filled = ifelse(is.na(hdi_filled), global_mean, hdi_filled),
      log_gdp = log(gdp_per_capita + 1),
      log_suicide_rate = log(suicide_100k_rate + 1),
      age_numeric = case_when(
        age == "5-14 years" ~ 10,
        age == "15-24 years" ~ 20,
        age == "25-34 years" ~ 30,
        age == "35-54 years" ~ 45,
        age == "55-74 years" ~ 65,
        age == "75+ years" ~ 80,
        TRUE ~ NA_real_
      ),
      is_male = ifelse(sex == "male", 1, 0),
      age = factor(
        age,
        levels = c(
          "5-14 years",
          "15-24 years",
          "25-34 years",
          "35-54 years",
          "55-74 years",
          "75+ years"
        )
      )
    ) %>%
    select(
      country,
      sex,
      age,
      age_numeric,
      is_male,
      generation,
      log_gdp,
      hdi_filled,
      log_suicide_rate
    ) %>%
    na.omit()
}

# Apply processing
train_set <- process_data(train_raw, hdi_means)
holdout_set <- process_data(holdout_set, hdi_means)

## Confirm issues are resolved
# Verify missing data
print(colSums(is.na(train_set)))

# Visualize data skewness
p_log_gdp <- ggplot(train_set, aes(x = log_gdp)) +
  geom_histogram(bins = 30,
                 fill = "navy",
                 color = "black") +
  labs(title = "GDP", x = "GDP per Capita") +
  theme_minimal()

p_log_suicide <- ggplot(train_set, aes(x = log_suicide_rate)) +
  geom_histogram(bins = 30,
                 fill = "navy",
                 color = "black") +
  labs(title = "Suicide Rate", x = "Suicides per 100k") +
  theme_minimal()

grid.arrange(p_log_gdp, p_log_suicide, ncol = 2)

### Results ###
# Plot 1: Non-Linearity
p1 <- ggplot(train_set, aes(x = log_gdp, y = log_suicide_rate)) +
  geom_point(alpha = 0.1, color = "gray") +
  geom_smooth(method = "gam",
              color = "blue",
              se = FALSE) +
  geom_smooth(
    method = "lm",
    color = "red",
    linetype = "dashed",
    se = FALSE
  ) +
  labs(title = "Non-Linearity: Log GDP vs Log Suicide", x = "Log GDP", y = "Log Suicide Rate") +
  theme_minimal()

# Plot 2: Interaction Effects
p2 <- train_set %>%
  group_by(age, sex) %>%
  summarize(mean_rate = mean(exp(log_suicide_rate) - 1), .groups = "drop") %>%
  ggplot(aes(x = age, y = mean_rate, fill = sex)) +
  geom_col(position = "dodge") +
  labs(title = "Interaction: Age & Sex", y = "Mean Suicide Rate", x = "Age Group") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

grid.arrange(p1, p2, ncol = 2)

# Training and Analysis
# Define Control
ctrl <- trainControl(method = "cv", number = 5)

# Model 1: Linear Regression
fit_lm <- train(
  log_suicide_rate ~ sex + age_numeric + log_gdp + hdi_filled + generation,
  method = "lm",
  data = train_set,
  trControl = ctrl
)

# Model 2: Random Forest
fit_rf <- train(
  log_suicide_rate ~ sex + age_numeric + log_gdp + hdi_filled + generation,
  method = "ranger",
  data = train_set,
  trControl = ctrl,
  tuneGrid = data.frame(
    mtry = c(3, 5),
    splitrule = "variance",
    min.node.size = 5
  ),
  importance = "impurity"
)

# Predictions
pred_lm <- predict(fit_lm, holdout_set)
pred_rf <- predict(fit_rf, holdout_set)

error_lm <- RMSE(holdout_set$log_suicide_rate, pred_lm)
error_rf <- RMSE(holdout_set$log_suicide_rate, pred_rf)

results <- tibble(
  Model = c("Linear Regression", "Random Forest"),
  RMSE_Holdout = c(error_lm, error_rf)
)

print(results)

# Assess variable importance
varImp(fit_rf) %>%
  ggplot(aes(x = reorder(row.names(.), Overall), y = Overall)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Feature Importance (Random Forest)", x = "Feature", y = "Importance") +
  theme_minimal()