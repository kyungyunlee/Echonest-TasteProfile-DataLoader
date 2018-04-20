import os
import scipy.sparse as sparse
import numpy as np
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

def create_sparse_matrix(csvfile, num_subset_songs, num_subset_users):
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
    
    df = pd.read_csv(csvfile, usecols=[0,1,2], names=['user', 'song', 'play_count'])

    songs = list(df.song.unique())
    users = list(df.user.unique())
    plays = list(df.play_count.astype(np.float))

    rows = df['song'].astype('category', categories=songs)
    cols = df['user'].astype('category', categories=users)
    sparse_matrix = sparse.csr_matrix((plays,(rows.cat.codes.copy(), cols.cat.codes.copy())), shape=(len(songs), len(users)))
    print ("Sparse matrix shape : ", sparse_matrix.shape)

    # sort by item popularity
    itemwise_sum = sparse_matrix.sum(axis=1).A.T[0]
    print (itemwise_sum.shape)
    idx = np.argsort(itemwise_sum)[::-1]

    sorted_songs = list(np.array(songs)[idx])
    sorted_sparse_matrix = sparse_matrix[idx]

    # subset 
    subset_matrix = sorte_sparse_matrix[:num_subset_songs][:, :num_subset_users]
    subset_songs = sorted_songs[:num_subset_songs]
    subset_users = users[:num_subset_users]

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



if __name__=='__main__':
    base_dir = '/media/bach4/kylee/Deep-content-data/'
    num_subset_users = args.users
    num_subset_songs = args.songs
    
    # txt to csvfile 
    taste_profile = os.path.join(base_dir, 'train_triplets.txt')
    taste_profile_csv = txt_to_csv(taste_profile)
    
    song_user_matrix, songs, users = create_sparse_matrix(taste_profile_csv, num_subset_users, num_subset_songs)

    # save outputs 
    sparse.save_npz('song_user_matrix.npz', song_user_matrix)
    np.save('subset_songs.npy', songs)
    np.save('subset_users.npy', users)

    # calculate sparsity 
    calculate_sparsity(user_song_matrix)



