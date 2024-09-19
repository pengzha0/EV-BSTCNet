import glob
import numpy as np
import json

def analyse(acc_all):
    # print(acc_all)
    temp = acc_all.split('\n')
    temp = [float(i[34:42]) for i in temp]

    best_index = temp.index(max(temp))


    best_acc = acc_all.split('\n')[best_index]
    mean_acc = str(np.mean(temp))

    add = '\n\n\nbest: {}\nmean: {}'.format(best_acc, mean_acc)
    return add




def fuse(result_dir):
    results = [r for r in glob.glob('{}/*'.format(result_dir)) if not r.endswith('.txt')]

    best_acc_all = []
    for result in results:
        f = open('{}/train_log.json'.format(result), 'r')
        train_res = json.load(f)
        f.close()


        if not train_res['finished']:
            continue
        # log = [l.split(' ')[-1] for l in log]
        time = result.replace('\\', '/').split('/')[-1]
        k = ', '.join(['{}: {}'.format(kk, train_res[kk]) for kk in ['best_acc', 'best_epoch']])
        res = 'time: {}, res: {}'.format(time, k)
        best_acc_all.append(res)
    best_acc_all = '\n'.join(best_acc_all)
    # print(best_acc_all)
    ana = analyse(best_acc_all)
    best_acc_all += ana
    f = open('{}/fuse.txt'.format(result_dir), 'w+')
    f.write(best_acc_all)
    f.close()




if __name__ == '__main__':
    fuse('../result/20220928141029_test_repeat')





