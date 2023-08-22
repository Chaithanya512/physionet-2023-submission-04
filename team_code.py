#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

from helper_code import *
import numpy as np, os, sys
import mne
from sklearn.impute import SimpleImputer
import joblib
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import pandas as pd
import tensorflow as tf
from tsfresh.feature_extraction import extract_features


################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose >= 1:
        print('Finding the Challenge data...')

    patient_ids = find_data_folders(data_folder)
    num_patients = len(patient_ids)

    if num_patients==0:
        raise FileNotFoundError('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    features = list()
    outcomes = list()
    cpcs = list()
    
    for i in range(num_patients):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patients))

        current_features = get_features(data_folder, patient_ids[i])
        features.append(current_features)

        # Extract labels.
        patient_metadata = load_challenge_data(data_folder, patient_ids[i])
        current_outcome = get_outcome(patient_metadata)
        outcomes.append(current_outcome)
        current_cpc = get_cpc(patient_metadata)
        cpcs.append(current_cpc)

    features = np.vstack(features)
    outcomes = np.vstack(outcomes)
    cpcs = np.vstack(cpcs)
    
    
   
    # Train the models.
    if verbose >= 1:
        print('Training the Challenge model on the Challenge data...')
    imputer = SimpleImputer().fit(features)
    features = imputer.transform(features)
    

    outcome_model , cpc_model = train_model2(features, outcomes, cpcs)
    # Save the models.
    save_challenge_model(model_folder, imputer, outcome_model, cpc_model)

    if verbose >= 1:
        print('Done.')


def train_model2(features,outcomes,cpcs):
    param_grid_xgb = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2,0.02],
    'n_estimators': [200,250,300,350],
    }
    
    class_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(features.shape[1],)),
        tf.keras.layers.Dense(64, activation='selu'),
        tf.keras.layers.Dense(64, activation='selu'),
        tf.keras.layers.Dense(32, activation='selu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Assuming binary classification
    ])
    lr = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=10000,
        decay_rate=0.9)
    class_model.compile(tf.keras.optimizers.SGD(learning_rate= lr, clipnorm= 1.0, clipvalue=0.5), 
                                                loss='BinaryFocalCrossentropy', metrics=['accuracy'])

    # Train the classification model
    class_model.fit(features, outcomes.ravel(), epochs=50, batch_size=16)

    # Define an ANN model for regression
    xgb_regressor = xgb.XGBRegressor()
    
    
    # Perform grid search using cross-validation for regression
    grid_search_reg = GridSearchCV(xgb_regressor, param_grid_xgb, cv=4)
    grid_search_reg.fit(features,cpcs.ravel())
    best_reg = grid_search_reg.best_estimator_
    best_reg.fit(features,cpcs.ravel())
    return class_model,best_reg
    
# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.

def load_challenge_models(model_folder, verbose):
    filename = os.path.join(model_folder, 'models.sav')
    return joblib.load(filename)

# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose):
    imputer = models['imputer']
    outcome_model = models['outcome_model']
    cpc_model = models['cpc_model']

    # Extract features.
    features = get_features(data_folder, patient_id)
    features = features.reshape(1, -1)

    # Impute missing data.
    features = imputer.transform(features)


    outcome_probability = outcome_model.predict(features).flatten()[0]
    cpc = cpc_model.predict(features)[0]
    if outcome_probability>=0.5:
        outcome=1
    else:
        outcome=0
    

    # Ensure that the CPC score is between (or equal to) 1 and 5.
    cpc = np.clip(cpc, 1, 5)

    return outcome, outcome_probability, cpc

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder, imputer, outcome_model, cpc_model):
    d = {'imputer': imputer, 'outcome_model': outcome_model, 'cpc_model': cpc_model}
    filename = os.path.join(model_folder, 'models.sav')
    joblib.dump(d, filename, protocol=0)

# Preprocess data.
def preprocess_data(data, sampling_frequency, utility_frequency):
    # Define the bandpass frequencies.
    passband = [0.1, 30.0]

    # Promote the data to double precision because these libraries expect double precision.
    data = np.asarray(data, dtype=np.float64)

    # If the utility frequency is between bandpass frequencies, then apply a notch filter.
    if utility_frequency is not None and passband[0] <= utility_frequency <= passband[1]:
        data = mne.filter.notch_filter(data, sampling_frequency, utility_frequency, n_jobs=4, verbose='error')

    # Apply a bandpass filter.
    data = mne.filter.filter_data(data, sampling_frequency, passband[0], passband[1], n_jobs=4, verbose='error')

    # Resample the data.
    if sampling_frequency % 2 == 0:
        resampling_frequency = 128
    else:
        resampling_frequency = 125
    lcm = np.lcm(int(round(sampling_frequency)), int(round(resampling_frequency)))
    up = int(round(lcm / sampling_frequency))
    down = int(round(lcm / resampling_frequency))
    resampling_frequency = sampling_frequency * up / down
    data = scipy.signal.resample_poly(data, up, down, axis=1)


    return data, resampling_frequency

# Extract features.
def get_features(data_folder, patient_id):

    # Load patient data.
    patient_metadata = load_challenge_data(data_folder, patient_id)
    recording_ids = find_recording_files(data_folder, patient_id)
    num_recordings = len(recording_ids)

    # Extract patient features.
    patient_features = get_patient_features(patient_metadata)

    # Extract EEG features.
    
    eeg_data=get_eeg_data(data_folder, patient_id)
    
    eeg_features=get_eeg_features(eeg_data)
    
    # Extract ECG features.


    # Extract features.
    return np.hstack((patient_features, eeg_features))

# Extract patient features from the data.
def get_patient_features(data):
    age = get_age(data)
    sex = get_sex(data)
    rosc = get_rosc(data)
    ohca = get_ohca(data)
    shockable_rhythm = get_shockable_rhythm(data)
    ttm = get_ttm(data)

    sex_features = np.zeros(2, dtype=int)
    if sex == 'Female':
        female = 1
        male   = 0
        other  = 0
    elif sex == 'Male':
        female = 0
        male   = 1
        other  = 0
    else:
        female = 0
        male   = 0
        other  = 1

    features = np.array((age, female, male, other, rosc, ohca, shockable_rhythm, ttm))

    return features



def get_eeg_data(data_folder, patient_id):
   
    recording_ids = find_recording_files(data_folder, patient_id)
    eeg_group = 'EEG'
   
    eeg_data_exists = any(
        os.path.exists(os.path.join(data_folder, patient_id, '{}_{}.hea'.format(recording_id, eeg_group)))
        for recording_id in recording_ids
    )
   
    eeg_channels = ['Fp1', 'F7', 'T3', 'T5', 'O1', 'Fp2', 'F8', 'T4', 'T6', 'O2', 'F3', 'C3', 'P3', 'F4', 'C4', 'P4', 'Fz', 'Cz', 'Pz']
   
   
# Bipolar channels: ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fz-Cz', 'Cz-Pz']
   
   
   
    if eeg_data_exists:
       
        total_length=0
        stacked_eeg=[]
        for recording_id in reversed(recording_ids):
            
            if total_length>=175000:
                
                return np.hstack((stacked_eeg[::-1]))
            recording_location = os.path.join(data_folder, patient_id, '{}_{}'.format(recording_id, eeg_group))
            if os.path.exists(recording_location + '.hea'):
                data, channels, sampling_frequency = load_recording_data(recording_location)
                utility_frequency = get_utility_frequency(recording_location + '.hea')
                data = reorder_recording_channels(data, channels, eeg_channels)
                data, sampling_frequency = preprocess_data(data, sampling_frequency, utility_frequency)
                data = np.array([data[0, :] - data[1, :], data[1, :] - data[2, :], data[2, :]- data[3, :], data[3, :] - data[4, :], data[5, :] - data[6, :], data[6, :] - data[7, :], data[7, :] - data[8, :], data[8, :] - data[9, :], data[0, :] - data[10, :], data[10, :] - data[11, :], data[11, :] - data[12, :], data[12, :] - data[4, :], data[5, :] - data[13, :], data[13, :] - data[14, :], data[14, :] - data[15, :], data[15, :] - data[9, :], data[16, :] - data[17, :], data[17, :] - data[18, :]]) # Convert to bipolar montage
                length =data.shape[1]
                stacked_eeg.append(data)
                total_length += length
        return None
    else:
        return None

       
        
   
def get_eeg_features(eeg_data):
   
    trimmed_eeg_data_list = list()
    if eeg_data is None:
        return float("nan")*np.ones(198)
    if eeg_data.shape[1]>=175000:
        for row in eeg_data:
            # trimmed_eeg_data_list.append(row[:176768]) 
            trimmed_eeg_data_list.append(row[len(row)-175000:])
       
        trimmed_eeg_data = np.array(trimmed_eeg_data_list)
        features = get_tsfresh_features(trimmed_eeg_data)
        return np.hstack((features))
    else:
        return float("nan")*np.ones(198)
        
        
def reorder_recording_channels(current_data, current_channels, reordered_channels):
   
   
    if current_channels == reordered_channels:
        return current_data
    else:
        indices = list()
        for channel in reordered_channels:
            if channel in current_channels:
                i = current_channels.index(channel)
                indices.append(i)
        num_channels = len(reordered_channels)
        num_samples = np.shape(current_data)[1]
        reordered_data = np.zeros((num_channels, num_samples))
        reordered_data[:, :] = current_data[indices, :]
       

       
        return reordered_data
def get_tsfresh_features(data_rocket):
    data_rocket = pd.DataFrame(data_rocket)
    reshaped_data = pd.DataFrame({
        'time': np.tile(np.arange(data_rocket.shape[1]), data_rocket.shape[0]),
        'id': np.zeros(data_rocket.shape[0] * data_rocket.shape[1], dtype=int),
        'kind': np.repeat(np.arange(data_rocket.shape[0]), data_rocket.shape[1]),
        'signal_value': data_rocket.values.ravel()
    })
    
    # Print the reshaped DataFrame
    
    
    
    fc_parameters = {
        'sum_values':None,
        "absolute_maximum":None,
        "median":None,
        "mean":None,
        "standard_deviation":None,
        "variance":None,
        "length":None,
        "maximum":None,
        "minimum":None,
        "root_mean_square":None,
        "fourier_entropy":[{'bins': 20}],
        # "spkt_welch_density": [{'coeff': 0}, {'coeff': 1}, {'coeff': 2}, {'coeff': 3}, {'coeff': 4}, {'coeff': 5}],
        # "autocorrelation": [{'lag': 5}],
        # "cid_ce": [{'normalize': True}],
        
        # "abs_energy":None,
        # # "benford_correlation":None,
        # # "binned_entropy": [{'max_bins': 10}],
        # "cwt_coefficients": [{'widths': (2, 5, 10, 20), 'coeff': 0, 'w': 2}, {'widths': (2, 5, 10, 20), 'coeff': 0, 'w': 5}],
        # "fft_coefficient": [{'coeff': 0,'attr':'real'}, {'coeff': 1,'attr':'real'}],
        # "kurtosis":None,
        # "number_cwt_peaks": [{'n': 2}],
        # "partial_autocorrelation": [{'lag': 5}],
        # "permutation_entropy": [{'tau': 1, 'dimension': 3}],
        # "variation_coefficient":None,
    }
    
    extracted_features = extract_features(reshaped_data,column_id="id", column_sort='time', column_kind="kind", column_value="signal_value",default_fc_parameters=fc_parameters,n_jobs=6)
    extracted_features = extracted_features.to_numpy()
    eeg_features = extracted_features.flatten()
   # print(eeg_features.shape)

    return eeg_features

    
