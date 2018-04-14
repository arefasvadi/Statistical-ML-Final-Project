# This is a preprocessing file for HackerRank Survey that was downloaded from
# Kaggle at https://www.kaggle.com/hackerrank/developer-survey-2018
# 
# Only the following fields are selected to build models from their joint 
# distribution:
#   age_begin_coding, age, gender, education, degree_focus, job_level,
#   current_role, industry, hiring_manager, vim_emacs, 
#   Not yet selected [Languages {C, C++
#   C#, Java, Ruby, Python, Java Script, R, Lua, Rust, Scala, Go, PHP}].
#


library(dplyr)
library(tidyr)

#Santitize data

#set current working directory
wd <- "C:\\Users\\axa159430\\Desktop\\Statistical-ML-Final-Project"
setwd(wd)

#load the initial dataset
hk.country.codes <- read.csv("dataset/Country-Code-Mapping.csv")
hk.code.books <- read.csv("dataset/HackerRank-Developer-Survey-2018-Codebook.csv")
hk.num.mapping <- read.csv("dataset/HackerRank-Developer-Survey-2018-Numeric-Mapping.csv")
hk.numeric <- read.csv("dataset/HackerRank-Developer-Survey-2018-Numeric.csv")
hk.vals <- read.csv("dataset/HackerRank-Developer-Survey-2018-Values.csv")
#removing null values from begin age field
hk.vals <- hk.vals %>% filter(q1AgeBeginCoding != '#NULL!')
hk.numeric <- hk.numeric %>% filter(q1AgeBeginCoding != '#NULL!')
#removing null values from age field
hk.vals <- hk.vals %>% filter(q2Age != '#NULL!')
hk.numeric <- hk.numeric %>% filter(q2Age != '#NULL!')
#removing null values from gender field
hk.vals <- hk.vals %>% filter(q3Gender != '#NULL!')
hk.numeric <- hk.numeric %>% filter(q3Gender != '#NULL!')
#removing null values from Education field
hk.vals <- hk.vals %>% filter(q4Education != '#NULL!')
hk.numeric <- hk.numeric %>% filter(q4Education != '#NULL!')
#removing null values from degree focus field
hk.vals <- hk.vals %>% filter(q5DegreeFocus != '#NULL!')
hk.numeric <- hk.numeric %>% filter(q5DegreeFocus != '#NULL!')
hk.vals <- hk.vals %>% filter(q5DegreeFocus != "")
hk.numeric <- hk.numeric %>% filter(q5DegreeFocus != 0)
#removing values from job level field
hk.vals <- hk.vals %>% filter(q8JobLevel != '#NULL!')
hk.numeric <- hk.numeric %>% filter(q8JobLevel != '#NULL!')
hk.vals <- hk.vals %>% filter(q8JobLevel != "")
hk.numeric <- hk.numeric %>% filter(q8JobLevel != 0)
#removing null values from current role
hk.vals <- hk.vals %>% filter(q9CurrentRole != '#NULL!')
hk.numeric <- hk.numeric %>% filter(q9CurrentRole != '#NULL!')
hk.vals <- hk.vals %>% filter(q9CurrentRole != "")
hk.numeric <- hk.numeric %>% filter(q9CurrentRole != 0)
#reoving null values from industry
hk.vals <- hk.vals %>% filter(q10Industry != '#NULL!')
hk.numeric <- hk.numeric %>% filter(q10Industry != '#NULL!')
hk.vals <- hk.vals %>% filter(q10Industry != "")
hk.numeric <- hk.numeric %>% filter(q10Industry != 0)
#removing null values from hiring manager
#It's clean
#removing null values from emacs vs vim
#maybe later can be used as latent variables
hk.vals <- hk.vals %>% filter(q24VimorEmacs != '#NULL!')
hk.numeric <- hk.numeric %>% filter(q24VimorEmacs != '#NULL!')
hk.vals <- hk.vals %>% filter(q24VimorEmacs != "")
hk.numeric <- hk.numeric %>% filter(q24VimorEmacs != 0)
#select only related columns
hk.vals <- hk.vals %>% select(q1AgeBeginCoding,q2Age,q3Gender,q4Education,
                              q5DegreeFocus,q8JobLevel,q9CurrentRole,
                              q10Industry,q24VimorEmacs)
hk.numeric <- hk.numeric %>% select(q1AgeBeginCoding,q2Age,q3Gender,q4Education,
                              q5DegreeFocus,q8JobLevel,q9CurrentRole,
                              q10Industry,q24VimorEmacs)
# write to output files
# 15334 record selected
write.csv(hk.numeric,file="dataset/hk.numeric.csv")
write.csv(hk.vals,file="dataset/hk.vals.csv")

