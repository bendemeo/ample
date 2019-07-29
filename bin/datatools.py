from dataset import *
import csv
import os
from sklearn import preprocessing

def open_data(NAMESPACE, parent_dir='data/', delimiter = '\t', dimred=True, annos = []):

    filename = 'dimred' if dimred else 'full'
    with open(parent_dir+NAMESPACE+'/'+filename+'.txt') as f:
        reader = csv.reader(f, delimiter = delimiter)
        data = np.array(list(reader)).astype(float) #assumes clean data, no column names

    result = dataset(data, path=parent_dir+NAMESPACE+'/')


    for anno in annos: #search for the indicated annotation file
        if os.path.exists(parent_dir+NAMESPACE+'/'+anno+'.txt'):
            with open(parent_dir+NAMESPACE+'/'+anno+'.txt') as f:
                reader = csv.reader(f, delimiter = delimiter)
                labels = np.array(list(reader))
                labels = [x[0] for x in labels]
                result.data[anno] = labels
        else:
            print('WARNING: could not find annotation {}'.format(anno))


    #search for subsamples
    if os.path.exists(parent_dir+NAMESPACE+'/ft.txt'):
        result.load_subsample(parent_dir+NAMESPACE+'/ft.txt', name='ft', delimiter=delimiter)


    return(result)


# def open_data(NAMESPACE, parent_dir='data/', delimiter = '\t', dimred=True, annos = []):
#
#     filename = 'dimred' if dimred else 'full'
#     with open(parent_dir+NAMESPACE+'/'+filename+'.txt') as f:
#         reader = csv.reader(f, delimiter = delimiter)
#         data = np.array(list(reader)).astype(float) #assumes clean data, no column names
#
#     result = dataset(data, path=parent_dir+NAMESPACE+'/')
#
#
#     for anno in annos: #search for the indicated annotation file
#         if os.path.exists(parent_dir+NAMESPACE+'/'+anno+'.txt'):
#             with open(parent_dir+NAMESPACE+'/'+anno+'.txt') as f:
#                 reader = csv.reader(f, delimiter = delimiter)
#                 labels = np.array(list(reader))
#                 labels = [x[0] for x in labels]
#                 result.data[anno] = labels
#         else:
#             print('WARNING: could not find annotation {}'.format(anno))
#
#
#     #search for subsamples
#     if os.path.exists(parent_dir+NAMESPACE+'/ft.txt'):
#         result.load_subsample(parent_dir+NAMESPACE+'/ft.txt', name='ft', delimiter=delimiter)
#
#
#     return(result)
