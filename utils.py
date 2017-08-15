import os
import csv
import random


def padding(student_tuple, target_length):
    num_problems_answered = student_tuple[0]
    question_seq = student_tuple[1]
    question_corr = student_tuple[2]
    
    pad_length = target_length - num_problems_answered
    question_seq += [-1]*pad_length
    question_corr += [0]*pad_length
    
    new_student_tuple = (num_problems_answered, question_seq, question_corr)
    return new_student_tuple

def read_data_from_csv(filename, shuffle=False):
    rows = []
    max_num_problems_answered = 0
    num_problems = 0
    
    print("Reading {0}".format(filename))
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            rows.append(row)
    print("{0} lines was read".format(len(rows)))
    
    # tuples stores the student answering sequence as 
    # ([num_problems_answered], [problem_ids], [is_corrects])
    tuples = []
    for i in range(0, len(rows), 3):
        # numbers of problem a student answered
        num_problems_answered = int(rows[i][0])
        
        # only keep student with at least 3 records.
        if num_problems_answered < 3:
            continue
        
        problem_ids = rows[i+1]
        is_corrects = rows[i+2]
        
        invalid_ids_loc = [i for i, pid in enumerate(problem_ids) if pid=='']        
        for invalid_loc in invalid_ids_loc:
            del problem_ids[invalid_loc]
            del is_corrects[invalid_loc]
        
        tup =(num_problems_answered, problem_ids, is_corrects)
        tuples.append(tup)
        
        if max_num_problems_answered < num_problems_answered:
            max_num_problems_answered = num_problems_answered
        
        pid = max(int(pid) for pid in problem_ids if pid!='')
        if num_problems < pid:
            num_problems = pid
    # add 1 to num_problems because 0 is in the pid
    num_problems+=1

    #shuffle the tuple
    if shuffle:
        random.shuffle(tuples)

    print ("max_num_problems_answered:", max_num_problems_answered)
    print ("num_problems:", num_problems)
    print("The number of students is {0}".format(len(tuples)))
    print("Finish reading data.")
    
    return tuples, max_num_problems_answered, num_problems


def load_padded_train_data(train_path):
    students_train, max_num_problems_answered_train, num_problems_train = \
    read_data_from_csv(train_path)

    students_train = [padding(student_tuple, max_num_problems_answered_train) 
                  for student_tuple in students_train]
    return students_train, max_num_problems_answered_train, num_problems_train

def load_padded_test_data(test_path):
    students_test, max_num_problems_answered_test, num_problems_test = \
    read_data_from_csv(test_path)

    students_test = [padding(student_tuple, max_num_problems_answered_test) 
                  for student_tuple in students_test]
    
    return students_test, max_num_problems_answered_test, num_problems_test

def load_train_test(train_path, test_path):
    students_train, max_num_steps_train, num_problems_train = \
    read_data_from_csv(train_path)

    students_test, max_num_steps_test, num_problems_test = \
    read_data_from_csv(test_path)
    
    max_num_steps = max(max_num_steps_train, max_num_steps_test)
    
    num_problems = max(num_problems_train, num_problems_test)
    
    
    students_train = [padding(student_tuple, max_num_steps) 
                      for student_tuple in students_train]

    students_test = [padding(student_tuple, max_num_steps) 
                      for student_tuple in students_test]
    
    return students_train, students_test, max_num_steps, num_problems