##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(scales)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#Project

#Exploring Data

head(edx)

glimpse(edx)


#number of unique users provided  ratings and number of unique movies were rated

edx %>%
  summarize(n_user = n_distinct(userId), n_movie = n_distinct(movieId))

# Separating the year of released from the title

pattern <- "[/(]\\d{4}[/)]$"
st_year_p<- str_extract(edx$title, pattern)
st_year <- str_extract(st_year_p, regex("\\d{4}"))


t <- str_remove(edx$title, pattern)

edxnew <- edx %>% mutate(year_released = as.numeric(st_year), title = t)

#10 best movies (most rated with the highest rate)
top_movies <- edxnew %>%
  group_by(movieId) %>%
  summarize(n=n(), s=sum(rating), title = first(title)) %>%
  top_n(10, n*s) 
top_movies

# plot number of rating for each year of released
movie <- edxnew %>% 
  select(movieId,year_released) %>%
  group_by(year_released) %>%
  summarise(n_rating=n())

movie$year_released[which.max(movie$n_rating)]

movie %>%
  ggplot(aes(year_released,n_rating )) +
  geom_line()



#Number of movies for each genres 
movie_generes <- edxnew %>% separate_rows(genres,sep="\\|")
genres <- levels(as.factor(movie_generes$genres))

sapply(genres, function(g) {
  sum(str_detect(edx$genres, g))
})

# which genres is rated more
movie_generes%>%
  group_by(genres)%>% 
  summarise(n=n())%>%
  mutate(genres=reorder(genres, n))%>%
  ggplot(aes(genres, n)) + 
  geom_bar(width = 0.5, stat = "identity", color = "black") +
  coord_flip()

# Distribution of movies getting rated

edxnew %>% group_by(movieId)%>% summarise(n=n())%>%
  ggplot(aes(n)) + 
  geom_histogram(bins = 20, color = "black") + 
  scale_x_log10() +
  ggtitle("Movies")


# Distribution of users rating the movies

edxnew %>% group_by(userId)%>% summarise(n=n())%>%
  ggplot(aes(n)) + 
  geom_histogram(bins = 20, color = "black") + 
  scale_x_log10() +
  ggtitle("Users")


#Prediction


#Create test set and train set

set.seed(1)
test_index <- createDataPartition(y=edxnew$rating, times=1, p=0.1, list=FALSE)

train_set <- edxnew[-test_index,]
test_set_temp <- edxnew[test_index,]



# Make sure userId and movieId in test_set  are also in train_set_temp set
test_set <- test_set_temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add rows removed from test_set  back into train_set
removed <- anti_join(test_set_temp, test_set)
train_set <- rbind(train_set, removed)

# Delete extra object from memory
rm(test_index,test_set_temp, removed)


#function that computes the RMSE for vectors of ratings and corresponding predictors.

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings-predicted_ratings)^2))
}


#Model No.1 (Average of Rating-Naive Model)

mu <- mean(train_set$rating)
naive_rmse <- RMSE(test_set$rating,mu)

#Creating a result table
RMSE_results <- tibble(method ="Average of Rating", RMSE = naive_rmse)


#Model No.2 (Movie Effect b_i)

movie_effects <- train_set %>% 
  group_by(movieId)%>%
  summarize(b_i= mean(rating-mu))

# We can see our estimates
qplot(movie_effects$b_i,  bins = 30, color = I("black"))

b_i_test <- test_set %>% 
  left_join(movie_effects, by="movieId") %>%
  pull(b_i)

predicted_ratings <- mu + b_i_test


RMSE_model <- RMSE(test_set$rating, predicted_ratings)

RMSE_results <- bind_rows(RMSE_results,
                          data_frame(method="Movie Effect Model",RMSE = RMSE_model ))

RMSE_results %>% knitr::kable()





#Model No.3 (User Effect b_u)

user_effects <- train_set %>% 
  left_join(movie_effects, by="movieId")%>% 
  group_by(userId)%>%
  summarize(b_u= mean(rating-mu-b_i))

# We can see our estimates
qplot(user_effects$b_u,  bins = 30, color = I("black"))


b_u_test <- test_set %>%
  left_join(user_effects,by="userId")%>%
  pull(b_u)

predicted_ratings <- mu + b_i_test + b_u_test 

RMSE_model <- RMSE(test_set$rating, predicted_ratings)

RMSE_results <- bind_rows(RMSE_results,
                          data_frame(method="Movie Effect and User Effect Model",RMSE = RMSE_model ))

RMSE_results %>% knitr::kable()


#Model No.4 (Genres Effect b_g)

genres_effects <- train_set %>% 
  left_join(movie_effects, by="movieId")%>% 
  left_join(user_effects, by="userId")%>%
  group_by(genres)%>%
  summarize(b_g= mean(rating-mu-b_i-b_u))

qplot(genres_effects$b_g,  bins = 30, color = I("black"))

b_g_test <- test_set %>%
  left_join(genres_effects,by="genres")%>%
  pull(b_g)

predicted_ratings <- mu + b_i_test + b_u_test + b_g_test

RMSE_model <- RMSE(test_set$rating, predicted_ratings)

RMSE_results <- bind_rows(RMSE_results,
                          data_frame(method="Movie Effect and User Effect and Genres Effect Model",RMSE = RMSE_model ))

RMSE_results %>% knitr::kable()

# We made a mistake in the model No.2
test_set %>%
  left_join(movie_effects, by="movieId") %>%
  mutate(residual=rating-(b_i+mu)) %>%
  arrange(desc(abs(residual))) %>%
  select(title,residual,b_i)%>%
  slice (1:10) 


#10 best movie according to our estimate
train_set %>%
  select(movieId,title) %>%
  distinct() %>%
  left_join(movie_effects, by="movieId") %>%
  arrange(desc(b_i)) %>%
  select(title, b_i) %>%
  slice(1:10)

#10 worst movie according to our estimate
train_set %>%
  select(movieId,title) %>%
  distinct() %>%
  left_join(movie_effects, by="movieId") %>%
  arrange(b_i) %>%
  select(title , b_i) %>%
  slice(1:10)


#we want to find how much these two group rated 
# we can find the times of rating for each movieId
train_set_n_rating <- train_set %>%
  count(movieId) %>%
  select(movieId,n)

# number of rating for 10 best movies according to our estimate
train_set %>%
  select(movieId,title) %>%
  distinct() %>%
  left_join(movie_effects, by="movieId") %>%
  arrange(desc(b_i)) %>%
  select(title, movieId) %>%
  slice(1:10) %>%
  left_join(train_set_n_rating)

# number of rating for 10 worst movies according to our estimate
train_set %>%
  select(movieId,title) %>%
  distinct() %>%
  left_join(movie_effects, by="movieId") %>%
  arrange(b_i) %>%
  select(title ,movieId) %>%
  slice(1:10)%>%
  left_join(train_set_n_rating)

# Model No.5  Regularized Movie Effect Model

mu <- mean(train_set$rating)

lambdas <- seq(0, 10, 0.25)


RMSE <-sapply(lambdas,function(l){
  
  movie_effects_reg <- train_set %>% 
    group_by(movieId)%>%
    summarise(b_i= sum(rating-mu)/(n()+l),n_i=n())
  
  b_i_reg_test <- test_set %>% 
    left_join(movie_effects_reg, by="movieId") %>%
    pull(b_i)
  
  predicted_ratings <- mu + b_i_reg_test
  
  
  RMSE_reg_test <- RMSE(test_set$rating, predicted_ratings)
  
  return(RMSE_reg_test)
  
})




qplot(lambdas, RMSE)
lambdas[which.min(RMSE)]

lambdas <- lambdas[which.min(RMSE)]

movie_effects_reg <- train_set %>% 
  group_by(movieId)%>%
  summarise(b_i= sum(rating-mu)/(n()+lambdas),n_i=n())

b_i_reg_test <- test_set %>% 
  left_join(movie_effects_reg, by="movieId") %>%
  pull(b_i)

predicted_ratings <- mu + b_i_reg_test


RMSE_reg_test <- RMSE(test_set$rating, predicted_ratings)


RMSE_results <- bind_rows(RMSE_results,
                          data_frame(method="Regularized Movie Effect Model",RMSE = RMSE_reg_test ))

RMSE_results %>% knitr::kable()

# Model No.6 Regularized Movie Effect and User Effect Model

mu <- mean(train_set$rating)

lambdas <- seq(0, 10, 0.25)

RMSE <- sapply(lambdas, function(l){
  
  movie_effects_reg <- train_set %>% 
    group_by(movieId)%>%
    summarize(b_i= sum(rating-mu)/(n()+l),n_i=n())
  
  b_i_reg_test <- test_set %>% 
    left_join(movie_effects_reg, by="movieId") %>%
    pull(b_i)
  
  user_effects_reg <- train_set %>% 
    left_join(movie_effects_reg, by="movieId")%>% 
    group_by(userId)%>%
    summarize(b_u= sum(rating-mu-b_i)/(n()+l) ,n_i=n())
  
  b_u_reg_test <- test_set %>% 
    left_join(user_effects_reg, by="userId") %>%
    pull(b_u)
  
  
  predicted_ratings <- mu + b_i_reg_test + b_u_reg_test
  RMSE_reg_test <- RMSE(test_set$rating, predicted_ratings)
  return(RMSE_reg_test)
  
})


qplot(lambdas, RMSE)
lambdas <- lambdas[which.min(RMSE)]


mu <- mean(train_set$rating)
movie_effects_reg <- train_set %>% 
  group_by(movieId)%>%
  summarize(b_i= sum(rating-mu)/(n()+lambdas),n_i=n())

b_i_reg_test <- test_set %>% 
  left_join(movie_effects_reg, by="movieId") %>%
  pull(b_i)

user_effects_reg <- train_set %>% 
  left_join(movie_effects_reg, by="movieId")%>% 
  group_by(userId)%>%
  summarize(b_u= sum(rating-mu-b_i)/(n()+lambdas) ,n_i=n())

b_u_reg_test <- test_set %>% 
  left_join(user_effects_reg, by="userId") %>%
  pull(b_u)


predicted_ratings <- mu + b_i_reg_test + b_u_reg_test


RMSE_reg_test <- RMSE(test_set$rating, predicted_ratings)


RMSE_results <- bind_rows(RMSE_results,
                          data_frame(method="Regularized Movie Effect and User Effect Model",RMSE = RMSE_reg_test ))

RMSE_results %>% knitr::kable()

# Model No.7 Regularized Movie Effect and User Effect and Genres Effect Model

mu <- mean(train_set$rating)
lambdas <- seq(0, 10, 0.25)

RMSE <- sapply(lambdas, function(l){
  
  movie_effects_reg <- train_set %>% 
    group_by(movieId)%>%
    summarize(b_i= sum(rating-mu)/(n()+l),n_i=n())
  
  b_i_reg_test <- test_set %>% 
    left_join(movie_effects_reg, by="movieId") %>%
    pull(b_i)
  
  user_effects_reg <- train_set %>% 
    left_join(movie_effects_reg, by="movieId")%>% 
    group_by(userId)%>%
    summarize(b_u= sum(rating-mu-b_i)/(n()+l) ,n_i=n())
  
  b_u_reg_test <- test_set %>% 
    left_join(user_effects_reg, by="userId") %>%
    pull(b_u)
  
  
  genres_effects_reg <- train_set %>% 
    left_join(movie_effects_reg, by="movieId")%>% 
    left_join(user_effects_reg, by="userId")%>% 
    group_by(genres)%>%
    summarize(b_g= sum(rating-mu-b_i-b_u)/(n()+l) ,n_i=n())
  
  b_g_reg_test <-test_set %>% 
    left_join(genres_effects_reg, by="genres") %>%
    pull(b_g)
  
  
  predicted_ratings <- mu + b_i_reg_test + b_u_reg_test + b_g_reg_test
  RMSE_reg_test <- RMSE(test_set$rating, predicted_ratings)
  return(RMSE_reg_test)
  
})


qplot(lambdas, RMSE)

lambdas <- lambdas[which.min(RMSE)]



movie_effects_reg <- train_set %>% 
  group_by(movieId)%>%
  summarize(b_i= sum(rating-mu)/(n()+lambdas),n_i=n())

b_i_reg_test <- test_set %>% 
  left_join(movie_effects_reg, by="movieId") %>%
  pull(b_i)

user_effects_reg <- train_set %>% 
  left_join(movie_effects_reg, by="movieId")%>% 
  group_by(userId)%>%
  summarize(b_u= sum(rating-mu-b_i)/(n()+lambdas) ,n_i=n())

b_u_reg_test <- test_set %>% 
  left_join(user_effects_reg, by="userId") %>%
  pull(b_u)


genres_effects_reg <- train_set %>% 
  left_join(movie_effects_reg, by="movieId")%>% 
  left_join(user_effects_reg, by="userId")%>% 
  group_by(genres)%>%
  summarize(b_g= sum(rating-mu-b_i-b_u)/(n()+lambdas) ,n_i=n())

b_g_reg_test <-test_set %>% 
  left_join(genres_effects_reg, by="genres") %>%
  pull(b_g)


predicted_ratings <- mu + b_i_reg_test + b_u_reg_test + b_g_reg_test
RMSE_reg_test <- RMSE(test_set$rating, predicted_ratings)




RMSE_results <- bind_rows(RMSE_results,
                          data_frame(method="Regularized Movie Effect and User Effect and Genres Effect Model",RMSE = RMSE_reg_test ))

RMSE_results %>% knitr::kable()


#Final RMSE

test_set <- validation

mu <- mean(train_set$rating)
movie_effects_reg <- train_set %>% 
  group_by(movieId)%>%
  summarize(b_i= sum(rating-mu)/(n()+lambdas),n_i=n())

b_i_reg_test <- test_set %>% 
  left_join(movie_effects_reg, by="movieId") %>%
  pull(b_i)

user_effects_reg <- train_set %>% 
  left_join(movie_effects_reg, by="movieId")%>% 
  group_by(userId)%>%
  summarize(b_u= sum(rating-mu-b_i)/(n()+lambdas) ,n_i=n())

b_u_reg_test <- test_set %>% 
  left_join(user_effects_reg, by="userId") %>%
  pull(b_u)


genres_effects_reg <- train_set %>% 
  left_join(movie_effects_reg, by="movieId")%>% 
  left_join(user_effects_reg, by="userId")%>% 
  group_by(genres)%>%
  summarize(b_g= sum(rating-mu-b_i-b_u)/(n()+lambdas) ,n_i=n())

b_g_reg_test <-test_set %>% 
  left_join(genres_effects_reg, by="genres") %>%
  pull(b_g)


predicted_ratings <- mu + b_i_reg_test + b_u_reg_test + b_g_reg_test
RMSE_reg_test <- RMSE(test_set$rating, predicted_ratings)




RMSE_results <- bind_rows(RMSE_results,
                          data_frame(method="Final RMSE",RMSE = RMSE_reg_test ))

RMSE_results %>% knitr::kable()

