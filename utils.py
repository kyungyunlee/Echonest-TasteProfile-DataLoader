import os 
import csv

def txt_to_csv(txtfile, csvfile=None):
    ''' Convert txtfile to csvfile 
    Params: 
        txtfile : path to txtfile 
        csvfile : path to new csvfile 
    Return :
        csvfile : path to saved csvfile 
    ''' 
    if csvfile == None:
        csvfile_name = txtfile.strip().split('/')[-1].split('.')[0] + '.csv'
        csvfile = os.path.join('/'.join(txtfile.strip().split('/')[:-1]), csvfile_name)

    with open(txtfile, 'r') as f:
        data = (line.strip().split('\t') for line in f)
        with open(csvfile, 'w+') as out_f:
            writer = csv.writer(out_f)
            writer.writerows(data)
    
    return csvfile 
