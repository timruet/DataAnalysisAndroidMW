import numpy as np
import pandas as pd

data=pd.read_csv('../datasets/data-transcending-paper/meta_info_file.tsv',sep='\t')


def getFamilies(data):
    families = []
    for item in data.loc[:,'families']:
        try:
            if item.find('FAM') != -1:
                idxFAM = item.find('FAM') +4 
                idxSep = item.find('|', idxFAM)
                families.append(item[idxFAM:idxSep])
        except AttributeError:
            pass
    return families

def countFamilies(families):
    count = dict({})
    for family in families:
        if family in count:
            count[family] = count[family]+1
        else:
            count[family] = 0
    count = dict(reversed(sorted(count.items(), key=lambda item: item[1])))
    return count

print(countFamilies(getFamilies(data)))
    