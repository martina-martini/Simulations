import random
import statistics
from collections import Counter
import numpy as np
################################## input parameters ##################################
grade_probabilities = {
                 18: 87/1849,
                 19: 62/1849,
                 20: 74/1849,
                 21: 55/1849,
                 22: 99/1849,
                 23: 94/1849,
                 24: 117/1849,
                 25: 117/1849,
                 26: 136/1849,
                 27: 160/1849,
                 28: 215/1849,
                 29: 160/1849,
                 30: 473/1849
}
TOT_COURSES = 15
NUM_SESSIONS_PER_YEAR = 6 # we assume one session every 2 months
probability_of_passing = 0.7 # it is same for all exams
# probability_of_accepting = 0.8 # it is also same for all the passed exams
# it is better to set the probability of accepting an exam based on the received grade (see later)
NUM_EXAMS_PER_SESSION = 4
TOT_STUDENTS = 100
# these are all modifiable parameters

# add random seed for reproducibility purposes
seed = 1
random.seed(seed)

# class Student: it contains the id, the number of left exams before graduating, all the grades achieved
# the count of the number of sessions taken to graduate, the years before graduating (one year is considered
# equal to 6 sessions), the final graduation grade and the list of exams that the student didn't pass
class Student():
    def __init__(self, id):
        self.id = id
        self.num_left_exams = TOT_COURSES
        # the number of left exams to pass is set to the total number of courses because when the student starts
        # his career he must have already to pass all the courses' exams
        self.grades = []
        self.sessions_to_graduate = 0
        self.years_to_graduate = 0
        self.final_grade = 0
        self.failed_exams = []

# initialization of the student id -> it will be updated at the end of the loop
id = 0
################ empty lists for further analyses #############################
students = []
graduation_grades = []
averages = []
sessions = []
failures = []
max_retaken_times = []
most_difficult_courses = []
average_times_retake = []
times_for_the_last = []

######################## beginning of the simulation ##########################
for student in range(TOT_STUDENTS):
    courses = [i for i in range(1, TOT_COURSES + 1)]
    print(f'\ncourses to attend: {[i for i in courses]}')
    student = Student(id)
    print(f'student {id}')
    print(f'left exams: {student.num_left_exams}')
    times_to_pass_last = 0 # how many times the student take before passing the last exam before the graduation

    while student.num_left_exams != 0: # the loop goes on until all the exams are passed
        for session in range(NUM_SESSIONS_PER_YEAR):
            if courses:
                print(f'\tsession {session}')
                # if the remaining exams are more than the maximum number of exams that a student can take in 1 session
                # then the exams that a student takes in this session is chosen randomly among all the remaining courses
                if student.num_left_exams > NUM_EXAMS_PER_SESSION:
                    exams_for_this_session = random.sample(courses, NUM_EXAMS_PER_SESSION)
                # else if the number of remaining exams is less than the exams that a student can take in a session
                # then the exams that the student takes in this session is equal to the left exams to pass
                else:
                    exams_for_this_session = courses
                print(f'\t\texams for session {session}: {exams_for_this_session}')
                for exam in exams_for_this_session:
                    result = random.choices(['passed', 'failed'], [probability_of_passing, 1 - probability_of_passing])[
                        0]
                    # for each taken exam, the chances to pass or fail the exam is given by the Bernoulli distribution
                    # with probability of passing fixed for each exam and tunable at the beginning of the simulation
                    if result == 'passed':
                        grade = random.choices(list(grade_probabilities.keys()), list(grade_probabilities.values()))[0]
                        student.grades.append(grade)

                        if grade > 28:
                          probability_of_accepting = 1
                        elif grade <= 28 and grade > 25:
                          probability_of_accepting = 0.8
                        elif grade <= 25 and grade > 22:
                          probability_of_accepting = 0.6

                        else:
                          probability_of_accepting = 0.5
                        # the probability of accepting an exam is depending linearly on the acquired grade

                        accept = random.choices(['accepted', 'rejected'],
                                                [probability_of_accepting, 1 - probability_of_accepting])[0]
                        # if the exam is passed, the grade is assigned based on the probability distribution shown at
                        # the beginning of the simulation
                        # also, the probability of accepting the grade follows a Bernoulli distribution with probability
                        # chosen at the beginning
                        if accept == 'accepted':
                            student.num_left_exams -= 1
                            courses.remove(exam)
                            print(
                                f'\t\t\texam {exam} is {result} with grade {grade} and it is {accept} '
                                f'so now the number of left exams is {student.num_left_exams}')
                        # if the exam grade is accepted, the number of remaining exams for a student decreases by 1
                        else:  # reject
                            print(f'\t\t\texam {exam} is {result} but it is {accept}')
                            pass
                    else:  # failed
                        print(
                            f'\t\t\tthe exam {exam} is {result}, so the number of left exams is {student.num_left_exams}')
                        student.failed_exams.append(exam)
                        pass
                if student.num_left_exams <= 1:
                    times_to_pass_last += 1
                # if the number of missing exams to graduate is equal to 1, then count the number of times
                # that the student takes to pass the last exam
                print(f'the remaining courses are: {courses}')

            else: # if no courses left, stop the simulation
                break
            student.sessions_to_graduate += 1
        student.years_to_graduate += 1 # when the number of sessions becomes equal to 6, a year is finished

####################################### OUTPUT METRICS ################################################################
    print(f'the student {id + 1} took the following grades: {student.grades}')

    avg_exams = sum(student.grades) / len(student.grades)
    print(f'average grade considering {TOT_COURSES} exams: {avg_exams}')
    print(f'number of sessions to graduate: {student.sessions_to_graduate}')
    print(f'number of years to graduate: {student.years_to_graduate}')
    print(f'number of failed exams: {len(student.failed_exams)}')
    print(
        f'most retaken exam before passing it:  Course {statistics.mode(student.failed_exams)} -> Course {statistics.mode(student.failed_exams)} was failed {max(Counter(student.failed_exams).values())} times')
    print(
        f'average of the times the student retook an exam: {statistics.mean(Counter(student.failed_exams).values())} times')
    print(f'number of times to pass the last exam before graduating is: {times_to_pass_last}')

    graduation_grade = (avg_exams / 30) * 110 + random.uniform(0, 4) + random.uniform(0, 2) + random.uniform(0, 2)
    # note: the assignment of the bonus points is uniformly distributed
    # assumption: there is no addition time taken to write the thesis, even if the bones points for it are considered
    if graduation_grade > 112.5:
        print(f'the graduation grade of student {student.id} is 110 cum laude')
        graduation_grades.append('110 cum laude')
    else:
        print(f'the graduation grade of student {student.id} is {graduation_grade}')
        graduation_grades.append(graduation_grade)

############################################ lists are fulfilled for further analyses #################################
    students.append(student.id)
    # graduation_grades.append(graduation_grade)
    averages.append(avg_exams)
    sessions.append(student.sessions_to_graduate)
    failures.append(len(student.failed_exams))
    max_retaken_times.append(max(Counter(student.failed_exams).values()))
    most_difficult_courses.append(statistics.mode(student.failed_exams))
    average_times_retake.append(statistics.mean(Counter(student.failed_exams).values()))
    times_for_the_last.append(times_to_pass_last)

    # the identification number of each student is updated at the end of the loop
    id += 1

######################################## ANALYSES PART #################################################################
########################### confidence level, it can be tuned #########################################################
confidence = 0.90

import matplotlib.pyplot as plt
import scipy.stats as stats

plt.figure(figsize=(22, 18))

##################################### graduation grades plot + confidence interval #####################################
grades_mean = np.mean(graduation_grades)
grades_sample_variance = np.var(graduation_grades, ddof=1)
alpha = 1 - confidence
t_value = stats.t.ppf(1 - alpha / 2, TOT_STUDENTS - 1)
margin = t_value * (grades_sample_variance / (TOT_STUDENTS ** 0.5))

g_confidence_interval_lower = grades_mean - margin
g_confidence_interval_upper = grades_mean + margin

relative_error = t_value/grades_mean
g_accuracy = 1 - relative_error

print(f"Confidence Interval for graduation grades: [{g_confidence_interval_lower}, {g_confidence_interval_upper}]")
print(f'The accuracy for graduation grades is {g_accuracy}')
print(f'The maximum graduation grade is {max(graduation_grades)} while the minimum graduation grade is {min(graduation_grades)}')

plt.subplot(5, 1, 1)
plt.plot(students, graduation_grades, color='blue', label='Graduation Grades')
plt.title('Graduation Grades, Averages, and Sessions for Students')
plt.xlabel('Student ID')
plt.ylabel('Graduation Grade')
plt.axhline(y=g_confidence_interval_lower, color='red', linestyle='--', label='CI Lower Bound')
plt.axhline(y=g_confidence_interval_upper, color='red', linestyle='--', label='CI Upper Bound')
plt.legend(loc = 'upper right')
plt.xticks(students) # Set x-axis ticks to show every student ids
plt.grid(True)

########################################### average grades plot + confidence interval ##################################
avg_mean = np.mean(averages)
avg_sample_variance = np.var(averages, ddof=1)

avg_confidence_interval_lower = avg_mean - margin
avg_confidence_interval_upper = avg_mean + margin

relative_error = t_value/avg_mean
avg_accuracy = 1 - relative_error

print(f"\nConfidence Interval for average grades: [{avg_confidence_interval_lower}, {avg_confidence_interval_upper}]")
print(f'The accuracy for average grades is {avg_accuracy}')
print(f'The maximum average grade is {max(averages)} while the minimum average grade is {min(averages)}')

plt.subplot(5, 1, 2)
plt.plot(students, averages, color='green', label='Averages')
plt.xlabel('Student ID')
plt.ylabel('Average Exam Grade')
plt.axhline(y=avg_confidence_interval_lower, color='red', linestyle='--', label='CI Lower Bound')
plt.axhline(y=avg_confidence_interval_upper, color='red', linestyle='--', label='CI Upper Bound')
plt.legend(loc = 'upper right')
plt.xticks(students)
plt.grid(True)

################################### number of sessions plot + confidence interval ######################################
s_mean = np.mean(sessions)
s_sample_variance = np.var(sessions, ddof=1)

s_confidence_interval_lower = s_mean - margin
s_confidence_interval_upper = s_mean + margin

relative_error = t_value/s_mean
s_accuracy = 1 - relative_error

print(f"\nConfidence Interval for the number of sessions: [{s_confidence_interval_lower}, {s_confidence_interval_upper}]")
print(f'The accuracy for the number of sessions is {s_accuracy}')
print(f'The maximum number of sessions taken to graduate is {max(sessions)} while the minimum number of sessions to graduate is {min(sessions)}')

plt.subplot(5, 1, 3)
plt.plot(students, sessions, color='orange', label='Number of Sessions')
plt.xlabel('Student ID')
plt.ylabel('Number of Sessions to Graduate')
plt.axhline(y=s_confidence_interval_lower, color='red', linestyle='--', label='CI Lower Bound')
plt.axhline(y=s_confidence_interval_upper, color='red', linestyle='--', label='CI Upper Bound')
plt.legend(loc = 'upper right')
plt.xticks(students)
plt.grid(True)

##################### max number of attempts for an exam plot + confidence interval ####################################
# it counts how many times the student fails an exam and must retake it -> the maximum number of times that a student
# must retake an exam is stored and plotted here
att_mean = np.mean(max_retaken_times)
att_sample_variance = np.var(max_retaken_times, ddof=1)

att_confidence_interval_lower = att_mean - margin
att_confidence_interval_upper = att_mean + margin

relative_error = t_value/att_mean
att_accuracy = 1 - relative_error

print(f"\nConfidence Interval for max attempts of retaking an exam is: [{att_confidence_interval_lower}, {s_confidence_interval_upper}]")
print(f'The accuracy for max attempts of retaking an exam is {att_accuracy}\n')
print(f'The maximum number of attempts of retaking an exam is {max(max_retaken_times)} while the minimum number of attempts of retaking an exam is {min(max_retaken_times)}\n')

plt.subplot(5, 1, 4)
plt.plot(students, max_retaken_times, color='black', label='Max number of times the student retake an exam')
plt.xlabel('Student ID')
plt.ylabel('Times to retake an exam')
plt.axhline(y=att_confidence_interval_lower, color='red', linestyle='--', label='CI Lower Bound')
plt.axhline(y=att_confidence_interval_upper, color='red', linestyle='--', label='CI Upper Bound')
plt.legend(loc = 'upper right')
plt.xticks(students)
plt.grid(True)

############################################## print the most difficult courses ########################################
# for each student it is computed the number of times that he retake the exams -> the exam associated with the maximum
# number of times that the student fails the exam is named as the most difficult exam to pass for him
plt. subplot(5, 1, 5)
plt.scatter(students, most_difficult_courses, color='purple', label='Most difficult exam to pass')
plt.ylabel('Most difficult exam to pass')
plt.xlabel('Student ID')
plt.legend(loc = 'upper right')
plt.xticks(students)
plt.yticks(range(0, 16))
plt.grid(True)


plt.tight_layout()
plt.show()

print('\n')
######################################## count what are the most difficult courses to pass #############################
# to compute this metric, the idea is to count how many students retake a specific exam
occurrences = Counter(most_difficult_courses)
ordered_occurrences = sorted(occurrences.items(), key=lambda x: x[1], reverse=True)
for course in ordered_occurrences:
  print(f'The course {course[0]} is the most difficult for {course[1]} students')

########################## number of times retaking the last exam before passing it ####################################
# it computes the number of times that a student takes doing the last exam (i.e., student.num_left_courses == 1) before
# finishing all the exams to attend and graduating
print(f'\nThe number of times retaking the last exam before graduating is {max(times_for_the_last)} '
      f'while the minimum is {min(times_for_the_last)}\n')


############################################ FURTHER INTERESTING ANALYSES ##############################################
import pandas as pd

# create a dataframe to better understand
students_serie = pd.Series(students, name='Student id')
sessions_serie = pd.Series(sessions, name='Sessions to graduate')
graduation_grades_serie = pd.Series(graduation_grades, name='Graduation grade')
average_times_retake_serie = pd.Series(average_times_retake, name='Average times retaking an exam')
times_for_the_last_serie = pd.Series(times_for_the_last, name='Times to pass the last exam')

df = pd.DataFrame({'Student id':students_serie, 'Sessions to graduate':sessions_serie, 'Graduation grade':graduation_grades_serie, 'Average times retaking an exam': average_times_retake, 'Times to pass the last exam': times_for_the_last_serie})
df.set_index('Student id', inplace=True)
print(df.describe())
# this method allows to display some statistics about the data, like the mean, the standard deviation, the minimum, the
# maximum and the quantile (25%, 50%, 75%) for each considered attribute of teh dataframe

# additional analyses can be done comparing and filtering the students in the dataframe
# for example, it can be computed the most talented students by filtering the student ids who
# - took the minimum number of sessions to graduate
# - got the graduation grade greater than 100
# - got the average times to retake an exam equal to the absolute minimum computed
# - took the minimum number of times for passing the last exam
best_students = df[(df['Sessions to graduate'] == min(sessions))
                  & (df['Graduation grade'] > 100)
                  & (df['Average times retaking an exam'] == min(average_times_retake))
                  & (df['Times to pass the last exam'] == min(times_for_the_last))]
print(best_students)

