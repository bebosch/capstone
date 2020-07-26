#############################################################################################################
# Create train, test and validation set with each row representing a rating given by one user to one movie
#############################################################################################################

# install all required libraries (note: this process could take a couple of minutes)

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# load all required libraries
library(tidyverse)
library(caret)
library(data.table)
library(lubridate)

# set digit places to 5
options(digits = 5)

# download metadata
dl <- tempfile()
url = "http://files.grouplens.org/datasets/movielens/ml-10m.zip"
download.file(url, dl)
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# convert downloaded metadata to data frame
movies <- as.data.frame(movies) %>% 
  mutate(movieId = as.numeric(levels(movieId))[movieId],
         title = as.character(title),
         genres = as.character(genres))

# split movie title and year
movies <- movies %>% 
  extract(title, c("title", "year"), regex = "([A-Za-z\\,\\s]*)\\s(\\(\\d{4}\\))") %>% 
  mutate(year = str_replace_all(year, "[\\(\\)]", ""))

# combine movie metadata and user ratings
movielens <- left_join(ratings, movies, by = "movieId")
class(movielens$year)

# transform timestamp to readable date format and round by week
movielens <- movielens %>% 
  mutate(date = as_datetime(timestamp)) %>% 
  mutate(date = round_date(date, unit = "week")) %>% 
  select(-timestamp)

# create edx and validation set (10% of movielens data)
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# create train and test set within edx set
edx_test_index <- createDataPartition(edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-edx_test_index,]
test_set <- edx[edx_test_index,]
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")
rm(dl, ratings, movies, test_index, temp, removed)

#############################################################################################################
# To compare different models, we need a loss function in order to quantify what it means to do well
#############################################################################################################

# RMSE (residual mean squared error) function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings) ^ 2))
}

#############################################################################################################
# Building the simplest possible recommendation system by predicting the same rating for all movies
#############################################################################################################

# average rating across all users
mu <- mean(train_set$rating)

# use mu to compute RMSE on the test set
mu_rmse <- RMSE(mu, test_set$rating)

# use mu_rmse to create rmse_results table (= Average Model)
rmse_results <- data_frame(method = "Average Model", RMSE = mu_rmse)

#############################################################################################################
# Incorporating a movie effect takes into account that on average some movies are rated higher than others
#############################################################################################################

# demonstrate movie effect (differing average ratings) graphically
movie_mu <- movielens %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating)) %>%
  summarize(movie_mu = mean(b_i)) %>%
  .$movie_mu
movielens %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating)) %>%
  ggplot(aes(b_i)) + 
  geom_histogram(bins = 30, color = "black") + 
  geom_vline(xintercept = movie_mu, color = "red")

# calculate the movie effect
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

# calculate predicted ratings when incorporating the movie effect
movie_effect_predicted_ratings <- mu + test_set %>% 
  left_join(movie_avgs, by = 'movieId') %>% 
  .$b_i

# use predicted ratings by incorporating movie effect to compute RMSE on the test set
movie_effect_rmse <- RMSE(test_set$rating, movie_effect_predicted_ratings)

# add movie_effect_rmse to rmse_results table (= Movie Effect Model)
rmse_results <- bind_rows(rmse_results, data_frame(method = "Movie Effect Model", RMSE = movie_effect_rmse))

#############################################################################################################
# Incorporating a user effect takes into account that on average some users give better ratings than others
#############################################################################################################

# demonstrate user effect (differing average ratings by users) graphically
user_mu <- movielens %>%
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>%
  summarize(user_mu = mean(b_u)) %>%
  .$user_mu
movielens %>%
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  ggplot(aes(b_u)) +
  geom_histogram(bins = 30, color = "black") +
  geom_vline(xintercept = user_mu, color = "red")

# calculate the user effect
user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# calculate predicted ratings when additionally incorporating the user effect
user_effect_predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by = 'movieId') %>% 
  left_join(user_avgs, by = 'userId') %>% 
  mutate(pred = mu + b_i + b_u) %>% 
  .$pred

# use predicted ratings by incorporating user effect to compute RMSE on the test set
user_effect_rmse <- RMSE(test_set$rating, user_effect_predicted_ratings)

# add user_effect_rmse to rmse_results table (= Movie + User Effect Model)
rmse_results <- bind_rows(rmse_results, data_frame(method = "Movie + User Effect Model", 
                                                   RMSE = user_effect_rmse))

#############################################################################################################
# Incorporating a time effect takes into account that average ratings vary across time
#############################################################################################################

# demonstrate time effect (differing average ratings across time) graphically
time_mu <- movielens %>%
  group_by(date) %>% 
  summarize(b_t = mean(rating)) %>%
  summarize(time_mu = mean(b_t)) %>%
  .$time_mu
movielens %>%
  group_by(date) %>%
  summarize(b_t = mean(rating)) %>%
  ggplot(aes(date, b_t)) +
  geom_point(alpha = 0.2) +
  geom_smooth() +
  geom_hline(yintercept = time_mu, color = "red")

# calculate the time effect
week_avgs <- train_set %>% 
  left_join(movie_avgs, by = 'movieId') %>% 
  left_join(user_avgs, by = 'userId') %>%
  group_by(date) %>% 
  summarize(b_t = mean(rating - mu - b_i - b_u))

# calculate predicted ratings when additionally incorporating the time effect
time_effect_predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by = 'movieId') %>% 
  left_join(user_avgs, by = 'userId') %>% 
  left_join(week_avgs, by = "date") %>%
  mutate(pred = mu + b_i + b_u + b_t) %>% 
  .$pred

# use predicted ratings by incorporating time effect to compute RMSE on the test set
time_effect_rmse <- RMSE(test_set$rating, time_effect_predicted_ratings)

# add time_effect_rmse to rmse_results table (= Movie + User + Time Effect Model)
rmse_results <- bind_rows(rmse_results, data_frame(method = "Movie + User + Time Effect Model", 
                                                   RMSE = time_effect_rmse))

#############################################################################################################
# Incorporating a genre effect takes into account that average ratings vary by genre type and combination
#############################################################################################################

# demonstrate genre effect (differing average ratings across genre) graphically
genre_mu <- movielens %>%
  group_by(genres) %>% 
  summarize(b_g = mean(rating)) %>%
  summarize(genre_mu = mean(b_g)) %>%
  .$genre_mu
movielens %>%
  group_by(genres) %>% 
  summarize(b_g = mean(rating)) %>% 
  ggplot(aes(b_g)) +
  geom_histogram(bins = 30, color = "black") +
  geom_vline(xintercept = genre_mu, color = "red")

# calculate the genre effect
genre_avgs <- train_set %>% 
  left_join(movie_avgs, by ='movieId') %>%
  left_join(user_avgs, by = "userId") %>% 
  left_join(week_avgs, by = "date") %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u - b_t))

# calculate predicted ratings when additionally incorporating the genre effect
genre_effect_predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by = 'movieId') %>% 
  left_join(user_avgs, by = 'userId') %>% 
  left_join(week_avgs, by = "date") %>%
  left_join(genre_avgs, by = "genres") %>% 
  mutate(pred = mu + b_i + b_u + b_t + b_g) %>% 
  .$pred

# use predicted ratings by incorporating genre effect to compute RMSE on the test set
genre_effect_rmse <- RMSE(test_set$rating, genre_effect_predicted_ratings)

# add genre_effect_rmse to rmse_results table (= Movie + User + Time + Genre Effect Model)
rmse_results <- bind_rows(rmse_results, data_frame(method = "Movie + User + Time + Genre Effect Model", 
                                                   RMSE = genre_effect_rmse))

#############################################################################################################
# Incorporating a publish effect takes into account that ratings vary depending on the age of the movie
#############################################################################################################

# demonstrate publish effect (differing average ratings across year of publication) graphically
publish_mu <- movielens %>%
  group_by(year) %>% 
  summarize(b_p = mean(rating)) %>%
  summarize(publish_mu = mean(b_p)) %>%
  .$publish_mu
movielens %>%
  mutate(year = as.Date(year, "%Y")) %>%
  group_by(year) %>%
  summarize(b_p = mean(rating)) %>%
  ggplot(aes(year, b_p)) +
  geom_point(alpha = 0.2) +
  geom_smooth() +
  geom_hline(yintercept = publish_mu, color = "red")

# calculate publish effect
publish_avgs <- train_set %>% 
  left_join(movie_avgs, by ='movieId') %>%
  left_join(user_avgs, by = "userId") %>% 
  left_join(week_avgs, by = "date") %>%
  left_join(genre_avgs, by = "genres") %>% 
  group_by(year) %>%
  summarize(b_p = mean(rating - mu - b_i - b_u - b_t - b_g))

# calculate predicted ratings when additionally incorporating the genre effect
publish_effect_predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by = 'movieId') %>% 
  left_join(user_avgs, by = 'userId') %>% 
  left_join(week_avgs, by = "date") %>%
  left_join(genre_avgs, by = "genres") %>% 
  left_join(publish_avgs, by = "year") %>%
  mutate(pred = mu + b_i + b_u + b_t + b_g + b_p) %>% 
  .$pred

# use predicted ratings by incorporating publish effect to compute RMSE on the test set
publish_effect_rmse <- RMSE(test_set$rating, publish_effect_predicted_ratings)

# add publish_effect_rmse to rmse_results table (= Movie + User + Time + Genre + Publish Effect Model)
rmse_results <- bind_rows(rmse_results, 
                          data_frame(method = "Movie + User + Time + Genre + Publish Effect Model", 
                                     RMSE = publish_effect_rmse))

#############################################################################################################
# Regularization limits the variability of effect sizes by penalizing large estimates from small sample sizes
#############################################################################################################

# find optimal lambda (note: this process takes several minutes)
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  b_t <- train_set %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(date) %>%
    summarize(b_t = sum(rating - b_u - b_i - mu)/(n()+l))
  b_g <- train_set %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    left_join(b_t, by="date") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_t - b_u - b_i - mu)/(n()+l))
  b_p <- train_set %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    left_join(b_t, by="date") %>%
    left_join(b_g, by="genres") %>%
    group_by(year) %>%
    summarize(b_p = sum(rating - b_g - b_t - b_u - b_i - mu)/(n()+l))
  predicted_ratings <- test_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_t, by = "date") %>%
    left_join(b_g, by = "genres") %>%
    left_join(b_p, by = "year") %>%
    mutate(pred = mu + b_i + b_u + b_t + b_g + b_p) %>%
    .$pred
  return(RMSE(test_set$rating, predicted_ratings))
})

# calculate and demonstrate optimal lambda graphically
lambda <- lambdas[which.min(rmses)]
qplot(lambdas, rmses)

# perform regularization with optimal lambda (equivalent to minmum of rmses)
regularized_rmse = min(rmses)

# add regularized_rmse to rmse_results table (= Regularized Movie + User + Time + Genre + Year Effect Model)  
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie + User + Time + Genre + Year Effect Model", 
                                     RMSE = regularized_rmse))

#############################################################################################################
# Apply final algrorithm (regularized model) on on the validation set to return final RMSE
#############################################################################################################

# calculate different regularized effects on whole edx set (before only test set)
movie_reg_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n())
user_reg_avgs <- edx %>% 
  left_join(movie_reg_avgs, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda), n_u = n())
time_reg_avgs <- edx %>%
  left_join(movie_reg_avgs, by = 'movieId') %>%
  left_join(user_reg_avgs, by = 'userId') %>%
  group_by(date) %>%
  summarize(b_t = sum(rating - b_u - b_i - mu)/(n()+lambda), n_t = n())
genre_reg_avgs <- edx %>%
  left_join(movie_reg_avgs, by = 'movieId') %>%
  left_join(user_reg_avgs, by = 'userId') %>%
  left_join(time_reg_avgs, by = 'date') %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - b_t - b_u - b_i - mu)/(n()+lambda), n_g = n())
publish_reg_avgs <- edx %>%
  left_join(movie_reg_avgs, by = 'movieId') %>%
  left_join(user_reg_avgs, by = 'userId') %>%
  left_join(time_reg_avgs, by = 'date') %>%
  left_join(genre_reg_avgs, by = 'genres') %>%
  group_by(year) %>%
  summarize(b_p = sum(rating - b_g - b_t - b_u - b_i - mu)/(n()+lambda), n_p = n())

# use the regularized model to calculate predicted ratings on the validation set
validation_predicted_ratings <- validation %>%
  left_join(movie_reg_avgs, by='movieId') %>%
  left_join(user_reg_avgs, by = "userId") %>%
  left_join(time_reg_avgs, by = "date") %>%
  left_join(genre_reg_avgs, by = "genres") %>%
  left_join(publish_reg_avgs, by = "year") %>%
  mutate(pred = mu + b_i + b_u + b_t + b_g + b_p) %>%
  .$pred

# calculate final RMSE on validation set
validation_rmse <- RMSE(validation$rating, validation_predicted_ratings)

# add validation_rmse to rmse_results table (= Final RMSE on Validation Set) and print table
rmse_results <- bind_rows(rmse_results,data_frame(method="Final RMSE on Validation Set",  
                                     RMSE = validation_rmse))
rmse_results %>% knitr::kable()

#############################################################################################################
cat("The final RMSE on the validation set is", validation_rmse)
#############################################################################################################
