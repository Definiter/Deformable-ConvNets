import os

for m in os.listdir('.'):
    if os.path.isdir(m):
        print m
        count = []
        for f in os.listdir('{}/leftImg8bit_train'.format(m)):
            if ('params' in f or 'states' in f):
                count.append(f[-11:-7])
                #print '{}/leftImg8bit_train/{}'.format(m, f)
        count = sorted(count)
        print count
        if (len(count) != 0):
            for f in os.listdir('{}/leftImg8bit_train'.format(m)):
                if ('params' in f or 'states' in f) and (not count[-1] in f):
                    print '{}/leftImg8bit_train/{}'.format(m, f)
                    #os.remove('{}/leftImg8bit_train/{}'.format(m, f))
