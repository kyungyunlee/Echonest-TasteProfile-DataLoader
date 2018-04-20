## Echo Nest Taste Profile User Data Processor 
Python script to create a sparse matrix (scipy csr matrix) of the user data provided by Echo Nest.  
Resulting matrix is song x user matrix with play count as the data.   

### Data format
* User data is provided by Echo Nest in a tab delimited .txt file. Each row contains user_id, song_id and play_count.

### To use 
* Download user data (`train_triplets.txt`) from [MSD website](https://labrosa.ee.columbia.edu/millionsong/tasteprofile) 
* If subset of the data is used  (ex 20000 users, 10000 songs)   
`python load_user_data.py --users 20000 --songs 10000`
* If all data is used  
`python load_user_data.py`
* After running the script file, three files will be created in the same directory as the script file :  
	* `song_user_matrix.npz` : sparse song x user matrix
	* `subset_songs.npy` : numpy array of songs in the sparse matrix (order preserved)
	* `subset_users.npy` : numpy array of users in the sparset matrix (order preserved)
* To load user matrix  
` song_user_matrix = scipy.sparse.load_npz('song_user_matrix.npz') `  

