"""
Created on Sat Apr 25 10:50:19 2015

@author: Janak Ramachandran

KDD Cup 2014 - Donors Choose
Problem Scope:  The 2014 KDD Cup asks participants to help DonorsChoose.org identify projects that are exceptionally exciting to the business,
at the time of posting. While all projects on the site fulfill some kind of need, certain projects have a quality above and beyond what is typical.
By identifying and recommending such projects early, they will improve funding outcomes, better the user experience,
and help more students receive the materials they need to learn. Also, correlate exciting DonorsChoose.org projects to US states and
school poverty levels identifying the top 2-5 projects per state that would benefit poorer schools.

"""
def load_all_data():   
    import pandas as pd
    "Read in and save the csv files in a tabular structure using dataframes"
    outcomes = pd.read_csv('outcomes.csv')
    projects = pd.read_csv('projects.csv')
    donations = pd.read_csv('donations.csv')
    resources = pd.read_csv('resources.csv')
    "Sort the projects dataframes based on date"
    projects = projects.sort(['date_posted'],ascending=[True])
    return (outcomes, projects, donations, resources)

def consolidate_training_data(projects, outcomes):
    import pandas as pd
    "Merge (inner join) the outcomes and projects dataframe tables"
    merged_training_table = pd.merge(outcomes, projects)
    return merged_training_table
    
def extract_test_data(projects, outcomes):
    "Extract test data based on the train dataset's size.  Since projects is sorted test data will be all data on or after 2014-01-01"
    training_set_size = outcomes.shape[0]
    test_set = projects[training_set_size+1:projects.shape[0]]
    return test_set

def encode_string_label(feature_to_encode):
    from sklearn import preprocessing
    "Function that encodes labels from 't/f' to '1/0'"
    le = preprocessing.LabelEncoder()
    le.fit(feature_to_encode)
    list(le.classes_)
    feature_to_encode = le.transform(feature_to_encode)
    return feature_to_encode
    
def extract_relevant_features(projects, outcomes):
    import numpy as np
    "Extract groundtruth and encode it from t/f or 1/0"
    groundtruth = merged_training_table.is_exciting
    groundtruth = encode_string_label(groundtruth)
    "Extract the relevant features from the training dataset and build a feature vector"
    school_charter = encode_string_label(merged_training_table.school_charter)
    school_magnet = encode_string_label(merged_training_table.school_magnet)
    school_year_round = encode_string_label(merged_training_table.school_year_round)
    school_nlns = encode_string_label(merged_training_table.school_nlns)
    school_kipp = encode_string_label(merged_training_table.school_kipp)
    school_charter_ready_promise = encode_string_label(merged_training_table.school_charter_ready_promise)
    teacher_teach_for_america = encode_string_label(merged_training_table.teacher_teach_for_america)
    teacher_ny_teaching_fellow = encode_string_label(merged_training_table.teacher_ny_teaching_fellow)
    eligible_double_your_impact_match = encode_string_label(merged_training_table.eligible_double_your_impact_match)
    eligible_almost_home_match = encode_string_label(merged_training_table.eligible_almost_home_match)
    fulfillment_labor_materials = np.asarray(merged_training_table.fulfillment_labor_materials)
    total_price_excluding_optional_support = np.asarray(merged_training_table.total_price_excluding_optional_support)
    total_price_including_optional_support = np.asarray(merged_training_table.total_price_including_optional_support)
    students_reached = np.asarray(merged_training_table.students_reached)
    
    feature_vector = np.vstack([school_charter, school_magnet, school_year_round, school_nlns, \
                                        school_kipp, school_charter_ready_promise, teacher_teach_for_america, teacher_ny_teaching_fellow,  \
                                         eligible_double_your_impact_match, eligible_almost_home_match, fulfillment_labor_materials, \
                                         total_price_excluding_optional_support, total_price_including_optional_support, students_reached]).T
    feature_vector[np.isnan(feature_vector)] = 0
    return (feature_vector, groundtruth)

def classify_dataset_crossvalidation(feature_vector, groundtruth):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import cross_validation
    "Build and train a random forest classifier and perform 5-fold cross-validation to compute performance using AUC"   
    rf_clf = RandomForestClassifier(n_estimators=100)
    scores_rf = cross_validation.cross_val_score(rf_clf, feature_vector, groundtruth, cv=5, scoring='roc_auc')
    print("Accuracy of RF classifier: %0.2f (+/- %0.2f)" % (scores_rf.mean(), scores_rf.std() * 2))  
    return scores_rf
    
def extract_features_for_testset(test_set):
    import numpy as np
    "Extract the same relevant features from the test dataset and build feature vector"
    school_charter = encode_string_label(test_set.school_charter)
    school_magnet = encode_string_label(test_set.school_magnet)
    school_year_round = encode_string_label(test_set.school_year_round)
    school_nlns = encode_string_label(test_set.school_nlns)
    school_kipp = encode_string_label(test_set.school_kipp)
    school_charter_ready_promise = encode_string_label(test_set.school_charter_ready_promise)
    teacher_teach_for_america = encode_string_label(test_set.teacher_teach_for_america)
    teacher_ny_teaching_fellow = encode_string_label(test_set.teacher_ny_teaching_fellow)
    eligible_double_your_impact_match = encode_string_label(test_set.eligible_double_your_impact_match)
    eligible_almost_home_match = encode_string_label(test_set.eligible_almost_home_match)
    fulfillment_labor_materials = np.asarray(test_set.fulfillment_labor_materials)
    total_price_excluding_optional_support = np.asarray(test_set.total_price_excluding_optional_support)
    total_price_including_optional_support = np.asarray(test_set.total_price_including_optional_support)
    students_reached = np.asarray(test_set.students_reached)
    
    feature_vector_testset = np.vstack([school_charter, school_magnet, school_year_round, school_nlns, \
                                        school_kipp, school_charter_ready_promise, teacher_teach_for_america, teacher_ny_teaching_fellow,  \
                                         eligible_double_your_impact_match, eligible_almost_home_match, fulfillment_labor_materials, \
                                         total_price_excluding_optional_support, total_price_including_optional_support, students_reached]).T
    feature_vector_testset[np.isnan(feature_vector_testset)] = 0
    return feature_vector_testset
    
def classify_testset(feature_vector, groundtruth, feature_vector_testset):
    from sklearn.ensemble import RandomForestClassifier
    "Build a random forest classifier and predict on the test set"
    rf_clf = RandomForestClassifier(n_estimators=100)
    rf_clf.fit(feature_vector, groundtruth)
    testset_outcome = rf_clf.predict(feature_vector_testset) 
    return testset_outcome
    
def consolidate_results(test_set, testset_outcome):
    import pandas as pd  
    "Consolidate relevant columns with projectid and test set outcome and save it in a csv"
    testset_outcome_df = pd.DataFrame({'projectid':test_set.projectid,'schoolid':test_set.schoolid,'school_state':test_set.school_state, 'poverty_level':test_set.poverty_level,'is_exciting':testset_outcome,'students_reached':test_set.students_reached})
    testset_outcome_df.to_csv('test_set_with_outcome.csv')
    
    "Select only projects that were predicted as 'is_exciting'"
    exciting_projects_only = testset_outcome_df[testset_outcome_df['is_exciting'] == 1]
    "Select only projects that had 'highest poverty' level"
    exciting_projects_highest_poverty = exciting_projects_only[exciting_projects_only['poverty_level'] == 'highest poverty']
    "Sort by ascending 'state name' and descending 'students reached' criteria"
    exciting_projects_poverty_state_impact = exciting_projects_highest_poverty.sort(['school_state', 'students_reached'],ascending=[True, False])
    exciting_projects_poverty_state_impact.to_csv('exciting_projects_poverty_state_impact.csv')
    
    "Select atmost 5 exciting projects per state that will have impact on most number of students"
    "The idea here is that poorest schools that have exciting projects with most students reached will have a greater impact"
    top_projects_per_state_poverty_level = exciting_projects_poverty_state_impact.groupby('school_state').head(5)
    top_projects_per_state_poverty_level.to_csv('top_projects_per_state_poverty_level.csv')
    return (testset_outcome_df, exciting_projects_poverty_state_impact, top_projects_per_state_poverty_level)
              
outcomes, projects, donations, resources = load_all_data()  
merged_training_table = consolidate_training_data(projects, outcomes)
test_set =  extract_test_data(projects, outcomes)
feature_vector, groundtruth = extract_relevant_features(projects, outcomes)
scores_rf = classify_dataset_crossvalidation(feature_vector, groundtruth)
feature_vector_testset = extract_features_for_testset(test_set)
testset_outcome = classify_testset(feature_vector, groundtruth, feature_vector_testset)
testset_outcome_df, exciting_projects_poverty_state_impact, top_projects_per_state_poverty_level = consolidate_results(test_set, testset_outcome)

