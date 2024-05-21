import os
import pandas as pd


def create_ravdess_df(path):
    ravdess_directory_list = os.listdir(path)
    file_emotion = []
    file_path = []

    for dir in ravdess_directory_list:
        actor = os.listdir(path + dir)
        for file in actor:
            part = file.split('.')[0].split('-')
            file_emotion.append(int(part[2]))
            file_path.append(path + dir + '/' + file)

    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    path_df = pd.DataFrame(file_path, columns=['Path'])
    ravdess_df = pd.concat([emotion_df, path_df], axis=1)
    ravdess_df.Emotions.replace({1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
                                5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'}, inplace=True)
    return ravdess_df


def create_crema_df(path):
    crema_directory_list = os.listdir(path)
    file_emotion = []
    file_path = []

    for file in crema_directory_list:
        file_path.append(path + file)
        part = file.split('_')
        if part[2] == 'SAD':
            file_emotion.append('sad')
        elif part[2] == 'ANG':
            file_emotion.append('angry')
        elif part[2] == 'DIS':
            file_emotion.append('disgust')
        elif part[2] == 'FEA':
            file_emotion.append('fear')
        elif part[2] == 'HAP':
            file_emotion.append('happy')
        elif part[2] == 'NEU':
            file_emotion.append('neutral')
        else:
            file_emotion.append('Unknown')

    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    path_df = pd.DataFrame(file_path, columns=['Path'])
    crema_df = pd.concat([emotion_df, path_df], axis=1)
    return crema_df


def create_tess_df(path):
    tess_directory_list = os.listdir(path)
    file_emotion = []
    file_path = []

    for dir in tess_directory_list:
        directories = os.listdir(path + dir)
        for file in directories:
            part = file.split('.')[0].split('_')[2]
            if part == 'ps':
                file_emotion.append('surprise')
            else:
                file_emotion.append(part)
            file_path.append(path + dir + '/' + file)

    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    path_df = pd.DataFrame(file_path, columns=['Path'])
    tess_df = pd.concat([emotion_df, path_df], axis=1)
    return tess_df


def create_savee_df(path):
    savee_directory_list = os.listdir(path)
    file_emotion = []
    file_path = []

    for file in savee_directory_list:
        file_path.append(path + file)
        part = file.split('_')[1][:-6]
        if part == 'a':
            file_emotion.append('angry')
        elif part == 'd':
            file_emotion.append('disgust')
        elif part == 'f':
            file_emotion.append('fear')
        elif part == 'h':
            file_emotion.append('happy')
        elif part == 'n':
            file_emotion.append('neutral')
        elif part == 'sa':
            file_emotion.append('sad')
        else:
            file_emotion.append('surprise')

    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    path_df = pd.DataFrame(file_path, columns=['Path'])
    savee_df = pd.concat([emotion_df, path_df], axis=1)
    return savee_df


def create_data_path_csv(ravdess_path, crema_path, tess_path, savee_path):
    ravdess_df = create_ravdess_df(ravdess_path)
    crema_df = create_crema_df(crema_path)
    tess_df = create_tess_df(tess_path)
    savee_df = create_savee_df(savee_path)

    data_path = pd.concat([ravdess_df, crema_df, tess_df, savee_df], axis=0)
    data_path.to_csv("data_path.csv", index=False)
    return data_path
