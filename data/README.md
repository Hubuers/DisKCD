# JAD and SDP

#### JAD and SDP consist of student response data extracted from the final exams of two courses, "Java Application Development" and "Software Design Patterns", offered during the 2021-2022 academic year in the software engineering major at a Chinese university. 

Note that the exercises in these exams are open questions rather than fill-in-the-blank questions or multiple-choice questions.

##### data/data_original/id_exercise.csv

`exercise_text`：The practice text content collected and organized in the exam papers.

##### data/data_original/id_TKC.csv

`TKC_text`：The text content covering the tested knowledge concepts that we have collected.

##### data/data_original/id_UKC.csv

`UKC_text`：The text content not covering the tested knowledge concepts that we have collected.

##### data/data_original/tested_exer_log.csv

`stu_id`：The IDs of students participating in the exam.

`exer_id`：The IDs of the exercises defined by us that are tested in the exam paper.

`response`：The student's response. If the student answered the question correctly, response=1; otherwise, response=0.

`TKC_id`：The TKC IDs related to the exercises.

##### data/data_original/untested_exer_log.csv

`stu_id`：The IDs of students participating in the exam.

`exer_id`：We assume the IDs of exercises that are not tested in the exam paper.

`response`：The student's response. If the student answered the question correctly, response=1; otherwise, response=0.

`UKC_id`：The UKC IDs related to the exercises.

##### data/data_original/stu-z-score.csv

Each row represents a student, and each column represents the z-score value of the historical grades for that student in a particular course.

##### data/data_original/TKC_ppt.txt and UKC_ppt.txt

Each row represents the learning resource text content corresponding to a knowledge concept.

##### data/data_original/q.csv

Q-matrix, where each row represents an exercise, each column represents a knowledge concept, and a value of 1 indicates the relevance of the exercise to that knowledge concept.

##### data/data_original/exercise_em.npy 、TKC_em.npy and UKC_em.npy

The output of exercises, TKC and UKC entity nodes through a bidirectional LSTM network.

# Junyi

#### **Junyi** is sourced from the Chinese e-learning platform Junyi Academy, which is widely applied  in CDM. To more effectively differentiate between TKCs and UKCs, we extracted response logs of 10,000 students in the Junyi dataset for our experiments. 