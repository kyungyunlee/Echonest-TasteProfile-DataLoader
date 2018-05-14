import os
import scipy.sparse as sparse
import numpy as np
import pickle
import pandas as pd
import argparse
from utils import txt_to_csv

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
        csvfile : path to taste profile csvfile 
        num_subset_songs : number of subset songs
        num_subset_users : number of subset users
    Return :
        subset_matrix : sorted itemwise sparse matrix of item x user
        subset_songs : numpy array of unique song ids sorted by popularity
        subset_users : numpy array of unique user ids
    '''
    
    # df = pd.read_csv(csvfile, usecols=[0,1,2], names=['user', 'song', 'play_count'])
    f = open(txtfile, 'r')
    songs = list()
    users = list ()
    plays = list ()
    for line in f :
        split_line = line.split('\t')
        users.append(split_line[0])
        songs.append(split_line[1])
        plays.append(float(split_line[2]))

    # unique_songs = list(df.song.unique())
    # unique_users = list(df.user.unique())
    '''
    songs = list(df.song)
    users = list(df.user)
    plays = list(df.play_count.astype(np.float))
    '''
    print (len(songs), len(users), len(plays))

    unique_songs = list(set(songs))
    unique_users = list(set(users))
    print (len(unique_songs), len(unique_users))
    song_idx_dict = {}
    user_idx_dict = {}

    for i, song in enumerate(unique_songs) : 
        song_idx_dict[song] = i

    pickle.dump(song_idx_dict, open("song_idx.pkl", 'wb'))
    
    for i, user in enumerate(unique_users):
        user_idx_dict[user] = i 
    pickle.dump(user_idx_dict, open("user_idx.pkl", 'wb'))
    
    idx_song_dict = dict(zip(song_idx_dict.values(), song_idx_dict.keys()))
    idx_user_dict = dict(zip(user_idx_dict.values(), user_idx_dict.keys()))
    
    song_to_idx = list()
    user_to_idx = list()
    for s in songs : 
        song_to_idx.append(song_idx_dict[s])
    for u in users :
        user_to_idx.append(user_idx_dict[u])

    # rows = df['song'].astype('category', categories=unique_songs, ordered=True)
    # cols = df['user'].astype('category', categories=unique_users, ordered=True)
    # sparse_matrix = sparse.csr_matrix((plays,(rows.cat.codes.copy(), cols.cat.codes.copy())), shape=(len(songs), len(users)))
    
    # 주의 : row == song , column == user 순서중요 
    rows = song_to_idx
    cols = user_to_idx
    # rows = user_to_idx
    # cols = song_to_idx

    # coo matrix를 쓰는게 맞는지.. 
    sparse_matrix = sparse.coo_matrix((plays,(rows, cols)), shape=(len(unique_songs), len(unique_users)))
    # print (sparse_matrix)

    print ("Sparse matrix shape : ", sparse_matrix.shape)
    print(sparse_matrix.tocsr()[:3])

    # 1. sort by song popularity
    itemwise_sum = sparse_matrix.sum(axis=1).A.reshape(-1) 
    idx = np.argsort(itemwise_sum)[::-1] # song index 
    print (np.sort(itemwise_sum)[::-1])
    print (idx)
   

    sorted_sparse_matrix = sparse_matrix.tocsr()[idx, :] # 무조건 indexing문제다.. 
    print(sparse_matrix)
    # print (sorted_sparse_matrix)
    check_sum = sorted_sparse_matrix.sum(axis=1).A.reshape(-1)
    # print (check_sum[0], sorted_sparse_matrix[:,0])

    # subset of songs
    songwise_subset_matrix = sorted_sparse_matrix[:num_subset_songs]
    
     
    # 2. sort users 
    userwise_sum = sparse_matrix.sum(axis=0).A.reshape(-1)
     
    # print (userwise_sum)
    print (np.sort(userwise_sum)[::-1][:5])
    user_idx = np.argsort(userwise_sum)[::-1]
    user_sorted_matrix = songwise_subset_matrix.tocsc()[:,user_idx] # 여기도 indexing문제

    subset_matrix = user_sorted_matrix[:, :num_subset_users].tocsr()

    
    # check 
    for i in idx[:5] :
        print (i,idx_song_dict[i])
   
    for i in user_idx[:5]:
        print (i,idx_user_dict[i]) 

    print ("subset songs , subset users", num_subset_songs, num_subset_users)
    subset_songs = list(np.array(unique_songs)[idx][:num_subset_songs])
    # subset_users = unique_users[:num_subset_users]
    subset_users = list(np.array(unique_users)[user_idx][:num_subset_users])
    

    print (subset_matrix.shape)

    return subset_matrix, subset_songs, subset_users


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
    count = 0
    with open(taste_profile, 'r') as f :
        for line in f:
            if count == 10: 
                break
            songid = line.split('\t')[1]
            trackid = MSD_id_to_7D_id[echonest_id_to_MSD_id[songid]]
            # print (trackid)
            if trackid in audio_list : 
                filtered_taste_profile.write(line)
                count +=1 
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

    # txt to csvfile 
    # filtered_taste_profile = os.path.join(base_dir, 'filtered_train_triplets.txt')
    # taste_profile_csv = txt_to_csv(filtered_taste_profile)

    
    # 실제로 할때는 small_train_triplets.txt -> train_triplets.txt 로 바꿔야함 
    filtered_taste_profile = os.path.join(base_dir, 'small_train_triplets.txt')
    # taste_profile_csv = txt_to_csv(filtered_taste_profile)
    
    song_user_matrix, songs, users = create_sparse_matrix(filtered_taste_profile, num_subset_songs, num_subset_users)
    print (len(songs), len(users))
    # save outputs 
    filename_tag = '_' + str(args.users) + '_'+ str(args.songs)
    sparse.save_npz('song_user_matrix' + filename_tag +  '.npz', song_user_matrix)
    np.save('subset_songs' + filename_tag + '.npy', songs)
    np.save('subset_users' + filename_tag + '.npy', users)

    # calculate sparsity 
    calculate_sparsity(song_user_matrix)

