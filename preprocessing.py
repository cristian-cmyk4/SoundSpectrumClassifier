import os
import torch
import torchaudio
import pandas as pd
import shutil

def copy_files(source_folder, destination_folder):
   
    files = os.listdir(source_folder)
    for file in files:
        source_file = os.path.join(source_folder, file)
        destination_file = os.path.join(destination_folder, file)
        shutil.copy(source_file, destination_file)

def extract_label_from_filename(filename):
    label = ''.join(filter(lambda x: not x.isdigit(), filename))
    return label.lower()

def separate_audio_with_txt(audio_path, txt_path, output_folder):
 
    with open(txt_path, 'r') as txtfile:
        lines = txtfile.readlines()
        sections = [line.strip().split() for line in lines]

    audio_filename = os.path.splitext(os.path.basename(audio_path))[0]
    waveform, sample_rate = torchaudio.load(audio_path)

    samples_start = [int(float(start_time) * sample_rate) for start_time, _, _ in sections]
    samples_end = [int(float(end_time) * sample_rate) for _, end_time, _ in sections]
    labels = [label for _, _, label in sections]

    separated_files = []
    for i, (start, end) in enumerate(zip(samples_start, samples_end)):
        audio_section = waveform[:, start:end]
        output_filename = f"{output_folder}/{audio_filename}_section_{i+1}.wav"
        separated_files.append((os.path.basename(output_filename), labels[i]))
        torchaudio.save(output_filename, audio_section, sample_rate)

    return separated_files

if __name__ == "__main__":
    
    audio_folder_path_train = "data/dcase2016_task2_train_dev/dcase2016_task2_train"
    audio_folder_path_dev = "data/dcase2016_task2_train_dev/dcase2016_task2_dev/sound"
    audio_folder_path_test = "data/dcase2016_task2_test_public/sound"
    
    txt_folder_path_dev = "data/dcase2016_task2_train_dev/dcase2016_task2_dev/annotation"
    txt_folder_path_test = "data/dcase2016_task2_test_public/annotation"
    
    output_folder_path_train = "data_train/audio"
    output_csv_path_train = "data_train/meta/meta.csv"
    output_folder_path_test = "data_test/audio"
    output_csv_path_test = "data_test/meta/meta.csv"
    
    audio_files_train = os.listdir(audio_folder_path_train)
    audio_files_dev = os.listdir(audio_folder_path_dev)
    audio_files_test = os.listdir(audio_folder_path_test)
    
    all_data_dev = []
    for audio_file in audio_files_dev:
        if audio_file.endswith(".wav") and "poly_0" in audio_file:
            audio_path = os.path.join(audio_folder_path_dev, audio_file)
            txt_file = os.path.splitext(audio_file)[0] + ".txt"
            txt_path = os.path.join(txt_folder_path_dev, txt_file)
            separated_files = separate_audio_with_txt(audio_path, txt_path, output_folder_path_train)
            all_data_dev.extend(separated_files)
            
    audio_names, labels = [],[]
    for audio_file in audio_files_train:
        if audio_file.endswith(".wav"):
            audio_name = os.path.splitext(audio_file)[0]
            label = extract_label_from_filename(audio_name)
            audio_names.append(audio_file)
            labels.append(label)

    df_dev = pd.DataFrame(all_data_dev, columns=["Audio_File", "Label"])
    df_train = pd.DataFrame({'Audio_File': audio_names, 'Label': labels})
    df_meta_train = pd.concat([df_dev, df_train], ignore_index=True)
    df_meta_train.to_csv(output_csv_path_train, index=False)
    
    copy_files(audio_folder_path_train, output_folder_path_train)
    
    all_data_test = []
    for audio_file in audio_files_test:
        if audio_file.endswith(".wav") and "poly_0" in audio_file:
            audio_path = os.path.join(audio_folder_path_test, audio_file)
            txt_file = os.path.splitext(audio_file)[0] + ".txt"
            txt_path = os.path.join(txt_folder_path_test, txt_file)
            separated_files = separate_audio_with_txt(audio_path, txt_path, output_folder_path_test)
            all_data_test.extend(separated_files)
            
    df_meta_test = pd.DataFrame(all_data_test, columns=["Audio_File", "Label"])
    df_meta_test.to_csv(output_csv_path_test, index=False)
