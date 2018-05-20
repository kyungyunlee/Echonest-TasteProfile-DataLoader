import os
import sys
import scipy.sparse as sparse
import numpy as np
import pickle
import pandas as pd
import argparse

total_users = 1019318 
total_songs = 384546

parser = argparse.ArgumentParser()
parser.add_argument('--users', type=int, default=total_users)
parser.add_argument('--songs', type=int, default=total_songs)
args = parser.parse_args()
print(args)


base_dir = '/media/bach4/kylee/Deep-content-data/' # change base dir to where the txt file is saved  
audio_dir = '/media/bach2/dataset/MSD/songs/'

def create_sparse_matrix(txtfile, num_subset_songs, num_subset_users):
    ''' Creates a csr matrix of user data from csvfile 
    Params :
        txtfile : path to taste profile textfile
        num_subset_songs : number of subset songs
        num_subset_users : number of subset users
    Return :
        subset_matrix : sorted itemwise sparse matrix of item x user
        song_list : list of unique song ids sorted by popularity
        user_list : list of unique user ids
    '''
    
    df = pd.read_csv(txtfile, sep='\t', names=['user', 'song', 'play_count'])

    # sort songs and users by playcounts
    sorted_song = df.groupby(['song'])[['play_count']].sum().sort_values('play_count', ascending=False)
    sorted_user = df.groupby(['user'])[['play_count']].sum().sort_values('play_count', ascending=False)

    # take subset 
    subset_songs = sorted_song.ix[0:num_subset_songs].index.get_level_values('song').tolist()
    subset_users = sorted_user.ix[0:num_subset_users].index.get_level_values('user').tolist()
    filtered_df = df[df['song'].isin(subset_songs) & df['user'].isin(subset_users)]

    # map index to songs and index to users for sparse matrix 
    idx_songs = np.arange(num_subset_songs)
    idx_users = np.arange(num_subset_users)
    song_to_idx = dict(zip(subset_songs, idx_songs))
    user_to_idx = dict(zip(subset_users, idx_users))
    
    # replace df values to indices 
    filtered_df['song'] = filtered_df['song'].map(song_to_idx)
    filtered_df['user'] = filtered_df['user'].map(user_to_idx)
    # filtered_df = filtered_df.replace({'user':user_to_idx})
    # filtered_df = filtered_df.replace({'song':song_to_idx})

    # rows : songs, cols : users 
    song_user_matrix = sparse.csr_matrix((filtered_df.values[:, 2], (filtered_df.values[:, 1], filtered_df.values[:, 0])))

    print (song_user_matrix.shape)

    idx_to_song = dict(zip(song_to_idx.values(), song_to_idx.keys()))
    idx_to_user = dict(zip(user_to_idx.values(), user_to_idx.keys()))
    
    song_list = list()
    user_list = list()
    for s in range(len(song_to_idx)) : 
        song_list.append(idx_to_song[s])
    for u in range(len(user_to_idx)) :
        user_list.append(idx_to_user[u])
    
    return song_user_matrix, song_list, user_list  


def calculate_sparsity(user_song_sparse):
    ''' calculate how sparse the matrix is in percentage '''
    matrix_size = user_song_sparse.shape[0]*user_song_sparse.shape[1]
    total_plays = len(user_song_sparse.nonzero()[0])
    sparsity = 100*(1 - (total_plays/matrix_size))
    print(sparsity)
    if sparsity > 99.5:
        print ("Matrix may be too sparse")


def filter_missing_audio(audio_dir, taste_profile):
    ''' filter entries from user data that is missing audio
    Args : 
        audio_dir : directory to audio files 
        taste_profile : text file containig user,song, play_count data 
    '''
    audio_list = []
    # get all existing audios 
    for root, directories, filenames in os.walk(audio_dir):
        print(root, directories, filenames)
        for filename in filenames:
            if '.clip.mp3' in filename : 
                audio_list.append(filename.replace('.clip.mp3', ''))


    print ("%d audios present"%len(audio_list))

    echonest_id_to_MSD_id = pickle.load(open('echonest_id_to_MSD_id.pkl', 'rb')) 
    MSD_id_to_7D_id = pickle.load(open('/media/bach4/kylee/MSD_mel/MSD_split/MSD_id_to_7D_id.pkl','rb'))
    
    # remove from txt file 
    filtered_taste_profile = open(os.path.join(base_dir, 'filtered_train_triplets.txt'),'w')
    with open(taste_profile, 'r') as f :
        for line in f:
            songid = line.split('\t')[1]
            trackid = MSD_id_to_7D_id[echonest_id_to_MSD_id[songid]]
            # print (trackid)
            if trackid in audio_list : 
                filtered_taste_profile.write(line)
            else :
                print ("missing", songid)
    
    print ("%d audios exist"%count)
    filtered_taste_profile.close()


if __name__=='__main__':
    num_subset_users = args.users
    num_subset_songs = args.songs
   
    # handle missing audio --- it takes long only do it once!  
    '''
    taste_profile = os.path.join(base_dir, 'train_triplets.txt')
    filter_missing_audio(audio_dir, taste_profile)
    '''

    txt = os.path.join(base_dir, 'train_triplets.txt')
    
    song_user_matrix, songs, users = create_sparse_matrix(txt, num_subset_songs, num_subset_users)
    print (len(songs), len(users))
    # save outputs 
    filename_tag = '_' + str(args.users) + '_'+ str(args.songs)
    sparse.save_npz('song_user_matrix' + filename_tag +  '.npz', song_user_matrix)
    np.save('subset_songs' + filename_tag + '.npy', np.array(songs))
    np.save('subset_users' + filename_tag + '.npy', np.array(users))

    # calculate sparsity 
    calculate_sparsity(song_user_matrix)

