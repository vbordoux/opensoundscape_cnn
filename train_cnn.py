# from opensoundscape.annotations import BoxedAnnotations
import os
import pandas as pd
import numpy as np
import torch
import random
from opensoundscape import CNN, SpectrogramPreprocessor
from opensoundscape.preprocess.utils import show_tensor_grid
from opensoundscape import AudioFileDataset
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve, auc
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf #TF needed to import pickle
import gc
import matplotlib.pyplot as plt
from opensoundscape.preprocess.actions import Action, MelScale
from opensoundscape.preprocess.action_functions import pcen
from opensoundscape.spectrogram import Spectrogram
from opensoundscape.preprocess.actions import ACTION_FN_DICT
import math

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def create_train_valid_set(experiment_name, samples_dir, fish_sound, fish_sound_folder, window_len_s):
    
    train_set_dir = os.path.join(samples_dir, experiment_name)

    pos_samples_file = os.listdir(os.path.join(train_set_dir, fish_sound_folder)) 
    neg_samples_file = os.listdir(os.path.join(train_set_dir, 'Unknown'))

    pos_samples_file = [os.path.join(train_set_dir, fish_sound_folder, file) for file in pos_samples_file]
    neg_samples_file = [os.path.join(train_set_dir, 'Unknown', file) for file in neg_samples_file]

    df_pos = pd.DataFrame()
    df_neg = pd.DataFrame()

    # Add positive sample file in the dataframe in a column file and create a column label with 1
    df_pos['file'] = pos_samples_file
    df_pos['start_time'] = 0.0
    df_pos['end_time'] = window_len_s
    df_pos[fish_sound] = 1

    # Add negative sample file in the dataframe in a column file and create a column label with 0
    df_neg['file'] = neg_samples_file
    df_neg['start_time'] = 0.0
    df_neg['end_time'] = window_len_s
    df_neg[fish_sound] = 0

    # concatenate the two dataframe
    df_trainset = pd.concat([df_pos, df_neg], ignore_index=True)
    df_trainset[fish_sound].value_counts()

    # Save the df as csv in the current repo
    df_trainset.to_csv('train_set_baseline.csv', index=False)
    print('train set saved in train_set_baseline.csv')

    df_trainset = pd.read_csv('train_set_baseline.csv', index_col=[0,1,2])
    return df_trainset



# def create_test_set_with_opensoundscape(test_files_path, fish_sound):
    
#     test_files_list = os.listdir(test_files_path)

#     # Create a list of all wav files with files ending by .wav
#     wav_files = sorted([file for file in test_files_list if file.endswith('.wav')])
#     annot_files = sorted([file for file in test_files_list if file.endswith('.txt')])

#     wav_files = [os.path.join(test_files_path, file) for file in wav_files]
#     annot_files = [os.path.join(test_files_path, file) for file in annot_files]

#     print("Checking files order\n", wav_files[:3])
#     print(annot_files[:3])

#     selection_files = annot_files
#     audio_files = wav_files

#     annotations = BoxedAnnotations.from_raven_files(raven_files=selection_files, audio_files=audio_files, annotation_column='Type')

#     clip_duration = 5.0
#     clip_overlap = 0
#     min_label_overlap = 0.2
#     species_of_interest = [fish_sound]

#     clip_labels = annotations.clip_labels(
#         clip_duration=clip_duration,
#         clip_overlap=clip_overlap,
#         min_label_overlap=min_label_overlap,
#         class_subset=species_of_interest)
    
#     return clip_labels



def load_test_set(test_set_path, test_files_path, fish_sound):
    # tensorflow is needed to import the pickle because it contains tensors
    test_set_df = pd.read_pickle(os.path.join(test_set_path, 'test_set.pkl'))
    
    # First drop embeddings
    test_set_df = test_set_df.drop(columns=['Embedding'])

    # Modify the format of the pickle to be compatible with opensoundscape
    test_set_df[fish_sound] = (test_set_df['Label'] == fish_sound)
    test_set_df = test_set_df.drop(columns=['Label'])

    test_set_df = test_set_df.rename(columns={'Starttime': 'start_time', 'Endtime': 'end_time', 'filename': 'file'})
    test_set_df['file'] = test_files_path + test_set_df['file']
    test_set_df = test_set_df.set_index(['file', 'start_time', 'end_time'])

    return test_set_df


# 1. Define the function at the top level of the file
def apply_mel(spec_obj, mel_transform):
    # Convert the immutable spec data into a PyTorch tensor
    spec_tensor = torch.tensor(spec_obj.spectrogram, dtype=torch.float32)
    
    # Apply the mel matrix transform
    mel_tensor = mel_transform(spec_tensor)
    
    # Generate the new frequency mapping axis
    mel_frequencies = np.linspace(60, 16000, 128)
    
    # Instantiate and return a clean, new Spectrogram object
    return Spectrogram(
        spectrogram=mel_tensor.numpy(),
        frequencies=mel_frequencies,
        times=spec_obj.times
    )

# 2. Tell OpenSoundscape how to save/load this custom function name
from opensoundscape.preprocess.actions import ACTION_FN_DICT
ACTION_FN_DICT['apply_mel'] = apply_mel


def setup_preprocessor(sample_rate, window_len):

    assert sample_rate == 32000, "SurfPerch expects 32 kHz audio"
    assert window_len == 5.0,    "SurfPerch expects 5-second windows"

    preprocessor = SpectrogramPreprocessor(window_len, sample_rate=sample_rate)
    preprocessor.pipeline.bandpass.bypass = True

    preprocessor.pipeline.to_spec.set(
        window_samples=640,      # 20 ms @ 32 kHz
        overlap_samples=320,     # overlap = window - hop → 10 ms hop
        fft_size=1024,
    )

    mel_transform = MelScale(
        n_mels=128,
        sample_rate=sample_rate,
        f_min=60.0,
        f_max=16000.0,
        n_stft=513,   # fft_size (1024) // 2 + 1
        norm=None,
    )

    mel_action = Action(
        fn=apply_mel,
        mel_transform=mel_transform
    )
    preprocessor.insert_action('to_mel', mel_action, after_key='to_spec')

    # PCEN
    pcen_action = Action(
        pcen,
        gain=0.8,
        bias=10.0,
        power=0.25,
        time_constant=0.06,
        eps=1e-6,
    )
    preprocessor.insert_action('pcen', pcen_action, after_key='to_mel')

    # ── Augmentations: all off ─────────────────────────────
    preprocessor.pipeline.random_trim_audio.bypass = True
    preprocessor.pipeline.overlay.bypass = True
    preprocessor.pipeline.time_mask.bypass = True
    preprocessor.pipeline.frequency_mask.bypass = True
    preprocessor.pipeline.adaptive_random_gain.bypass = True
    preprocessor.pipeline.adaptive_random_noise.bypass = True
    preprocessor.pipeline.random_wrap.bypass = True

    # # Augmentations
    # preprocessor.pipeline.random_trim_audio.bypass = True
    # preprocessor.pipeline.overlay.bypass = True
    # # model.preprocessor.pipeline.overlay.overlay_df = (train_df.astype(int))
    # model.preprocessor.pipeline.overlay.set(overlay_class=fish_sound)

    # preprocessor.pipeline.time_mask.bypass = True
    # preprocessor.pipeline.time_mask.set(max_masks=2, max_width=0.1)

    # preprocessor.pipeline.frequency_mask.bypass = True
    # preprocessor.pipeline.frequency_mask.set(max_masks=2, max_width=0.1)
    
    return preprocessor



def compute_metrics(y_true, y_pred, fish_sound):
    # Extract target values
    true_labels = y_true[fish_sound].values
    pred_probs = y_pred[fish_sound].values
    pred_binary = pred_probs.round()  # Threshold = 0.5
    
    # Calculate metrics
    precision = precision_score(true_labels, pred_binary, pos_label=1, average='binary')
    recall = recall_score(true_labels, pred_binary, pos_label=1, average='binary')
    f1 = f1_score(true_labels, pred_binary, pos_label=1, average='binary')
    auc_roc = roc_auc_score(true_labels, pred_probs)
    precision_curve, recall_curve, _ = precision_recall_curve(true_labels, pred_probs)
    auc_pr = auc(recall_curve, precision_curve)
    
    # Return metrics as a dictionary for better access
    return precision, recall, f1, auc_roc, auc_pr



if __name__ == "__main__":
    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Hard coded params for now - If properly given here, the rest of the code should run
    # TODO get them from the dict config eventually
    # fish_sound = 'Jackhammer'
    # fish_sound_folder = fish_sound
    # window_len_s = 5.0
    # sample_rate = 32000
    # experiment_name = 'jackhammer_surfperch_hockey_11_20250326'
    # samples_dir = '/home/reindert/Valentin_REVO/experiments_paper_only/output/grafton_deployment/surfperch/labeled_outputs/'
    # testset_files_dir = '/home/reindert/Valentin_REVO/experiments_paper_only/dataset/grafton_deployment/test_set/'
    # testset_pickle_dir = '/home/reindert/Valentin_REVO/experiments_paper_only/output/grafton_deployment/surfperch/test_set/'

    fish_sound = 'downsweep'
    fish_sound_folder = fish_sound
    window_len_s = 5.0
    sample_rate = 32000
    experiment_name = 'texel_baseline_20250331'
    samples_dir = '/home/reindert/Valentin_REVO/surfperch_toshare/eval_texel Outputs/september 2024/surfperch/labeled_outputs/'
    testset_files_dir = '/home/reindert/Valentin_REVO/surfperch_toshare/eval_texel Data/september 2024/test_set/'
    testset_pickle_dir = '/home/reindert/Valentin_REVO/surfperch_toshare/eval_texel Outputs/september 2024/surfperch/test_set/'


    # RUN PARAMS
    nbr_epochs = 128
    batch_size = 12
    num_workers = 20


    # NOT USED anymore - Create test set with opensoundscape - can be used to compare perf of both test set
    # test_set_df = create_test_set_with_opensoundscape(testset_files_dir, fish_sound)

    # Load test set
    test_set_df = load_test_set(testset_pickle_dir, testset_files_dir, fish_sound)

    # Create the 5-fold training and validation sets
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    df_trainset = create_train_valid_set(experiment_name, samples_dir, fish_sound, fish_sound_folder, window_len_s)
    validation_metrics_dic = {}
    testset_metrics_dic = {}

    # breakpoint()

    # Iterate over the folds
    for i, (train_index, valid_index) in enumerate(kf.split(df_trainset, df_trainset[fish_sound])):

        print("\n\n----------------------------------------")
        print(f"Running experiment for Fold {i+1}/5")
        print("----------------------------------------")

        train_df, valid_df = df_trainset.iloc[train_index], df_trainset.iloc[valid_index]

        # upsample (repeat samples) so that all classes have 200 samples
        # train_df = resample(train_df, n_samples_per_class=500, n_samples_without_labels=500, random_state=0)

        # We use Resnet as most common architeture used in Bioacoustics (ref: Stowell 2022)
        # Resnet 18 because how dataset is small (avoid overfitting)
        
        # Opensoundscape is likely loading pretrained weigths for the model from pytorch
        architecture = 'efficientnet_b0' 
        class_list = [fish_sound]
        model = CNN(architecture=architecture,
                    sample_rate=sample_rate,
                    classes=class_list,
                    sample_duration=5.0)
        # print(f"model.device is {model.device}")
        model.optimizer_params["kwargs"]["lr"] = 0.001

        preprocessor = setup_preprocessor(sample_rate=sample_rate, window_len=window_len_s)
        model.preprocessor = preprocessor
        print(model.preprocessor.pipeline)
        print(model.optimizer_params)

        # Display samples
        # dataset = AudioFileDataset(train_df, model.preprocessor)
        # tensors = [dataset[i].data for i in range(9)]
        # sample_labels = [list(dataset[i].labels[dataset[i].labels > 0].index) for i in range(9)]
        # _ = show_tensor_grid(tensors, 3, labels=sample_labels)

        checkpoint_folder = Path("model_training_checkpoints")
        checkpoint_folder.mkdir(exist_ok=True)


        steps_per_epoch = math.ceil(len(train_df) / batch_size)
        nbr_steps = steps_per_epoch * nbr_epochs

        model.train(
            train_df,
            valid_df,
            steps=nbr_steps,
            batch_size=batch_size,
            num_workers=num_workers,
            progress_bar=True,
            save_interval=steps_per_epoch,  # save checkpoint every 10 epochs
            save_path=checkpoint_folder,  # location to save checkpoints
        )

        # Load the best model - I think otherwise the model at the end of the training is used, which likely overfit data
        ACTION_FN_DICT['__main__.apply_mel'] = apply_mel
        model.load("./model_training_checkpoints/best.model")

        # Run predictions on the validation set and on the test set
        predict_validset = model.predict(valid_df,
                                        batch_size=batch_size, 
                                        num_workers=num_workers, 
                                        activation_layer='sigmoid', 
                                        progress_bar=False)
        
        pred_testset = model.predict(test_set_df, 
                                     batch_size=batch_size, 
                                     num_workers=num_workers, 
                                     activation_layer='sigmoid')

        # Compute metrics
        prec_valid, recall_valid, f1_valid, auc_roc_valid, auc_pr_valid = compute_metrics(valid_df, predict_validset, fish_sound)
        prec_test, recall_test, f1_test, auc_roc_test, auc_pr_test = compute_metrics(test_set_df, pred_testset, fish_sound)

        # Save metrics of the fold iteration
        validation_metrics_dic[i] = [prec_valid, recall_valid, f1_valid, auc_roc_valid, auc_pr_valid]
        testset_metrics_dic[i] = [prec_test, recall_test, f1_test, auc_roc_test, auc_pr_test]

        # Clean cuda memory to avoid OOM
        # model.to('cpu')
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Compute average of all metrics across all folds
    prec_valid = np.mean([metrics[0] for metrics in validation_metrics_dic.values()])
    recall_valid = np.mean([metrics[1] for metrics in validation_metrics_dic.values()])
    f1_valid = np.mean([metrics[2] for metrics in validation_metrics_dic.values()])
    auc_roc_valid = np.mean([metrics[3] for metrics in validation_metrics_dic.values()])
    auc_pr_valid = np.mean([metrics[4] for metrics in validation_metrics_dic.values()])

    prec_test = np.mean([metrics[0] for metrics in testset_metrics_dic.values()])
    recall_test = np.mean([metrics[1] for metrics in testset_metrics_dic.values()])
    f1_test = np.mean([metrics[2] for metrics in testset_metrics_dic.values()])
    auc_roc_test = np.mean([metrics[3] for metrics in testset_metrics_dic.values()])
    auc_pr_test = np.mean([metrics[4] for metrics in testset_metrics_dic.values()])

    # print all average metrics
    print("Validation Set")
    print("Precision valid: ", prec_valid)
    print("Recall valid: ", recall_valid)
    print("F1 valid: ", f1_valid)
    print("AUC ROC valid: ", auc_roc_valid)
    print("AUC precision recall: ", auc_pr_valid)

    print("\nTest Set")
    print("Precision test: ", prec_test)
    print("Recall test: ", recall_test)
    print("F1 test: ", f1_test)
    print("AUC ROC test: ", auc_roc_test)
    print("AUC precision recall: ", auc_pr_test)

    # Save all metrics in a csv file
    metrics_df = pd.DataFrame({'fold': list(validation_metrics_dic.keys()),
                            'precision_valid': [metrics[0] for metrics in validation_metrics_dic.values()],
                            'recall_valid': [metrics[1] for metrics in validation_metrics_dic.values()],
                            'f1_valid': [metrics[2] for metrics in validation_metrics_dic.values()],
                            'auc_roc_valid': [metrics[3] for metrics in validation_metrics_dic.values()],
                            'auc_pr_valid': [metrics[4] for metrics in validation_metrics_dic.values()],
                            'precision_test': [metrics[0] for metrics in testset_metrics_dic.values()],
                            'recall_test': [metrics[1] for metrics in testset_metrics_dic.values()],
                            'f1_test': [metrics[2] for metrics in testset_metrics_dic.values()],
                            'auc_roc_test': [metrics[3] for metrics in testset_metrics_dic.values()],
                            'auc_pr_test': [metrics[4] for metrics in testset_metrics_dic.values()]})
    
    # create results dir
    Path("./results").mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(f"./results/{experiment_name}_metrics.csv", index=False)
    print(f"Results saved in ./results/{experiment_name}_metrics.csv")
    
    # Make a boxplot of the test set metrics
    metrics_df.boxplot(column=['precision_test', 'recall_test', 'f1_test', 'auc_roc_test', 'auc_pr_test'], grid=False)
    plt.show()


# Perf on last run
'''
TEXEL
Validation Set
Precision valid:  0.9866518353726363
Recall valid:  0.9
F1 valid:  0.9397985731967393
AUC ROC valid:  0.9960000000000001
AUC precision recall:  0.9958641285399313

Test Set
Precision test:  0.48166056166056165
Recall test:  0.6978723404255319
F1 test:  0.5572425578509381
AUC ROC test:  0.94016846327102
AUC precision recall:  0.6889680280556431

BPNS
Validation Set
Precision valid:  0.9154160654160653
Recall valid:  0.9133333333333333
F1 valid:  0.9112923374266387
AUC ROC valid:  0.9772222222222222
AUC precision recall:  0.9797900039815197

Test Set
Precision test:  0.19879453541769868
Recall test:  1.0
F1 test:  0.3303786921374322
AUC ROC test:  0.9997169725189445
AUC precision recall:  0.8985410998866794
''' 