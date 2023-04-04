import os

def get_combined_results(left_results, right_results):
    results = {}
    count   = float(left_results['count'])

    results['left_mr']	= round(left_results ['mr'] /count, 5)
    results['left_mrr']	= round(left_results ['mrr']/count, 5)
    results['right_mr']	= round(right_results['mr'] /count, 5)
    results['right_mrr']	= round(right_results['mrr']/count, 5)
    results['mr']		= round((left_results['mr']  + right_results['mr']) /(2*count), 5)
    results['mrr']		= round((left_results['mrr'] + right_results['mrr'])/(2*count), 5)

    for k in range(10):
        # results['left_hits@{}'.format(k+1)]	= round(left_results ['hits@{}'.format(k+1)]/count, 5)
        # results['right_hits@{}'.format(k+1)]	= round(right_results['hits@{}'.format(k+1)]/count, 5)
        results['hits@{}'.format(k+1)]		= round((left_results.get('hits@{}'.format(k+1), 0.0) + right_results.get('hits@{}'.format(k+1), 0.0))/(2*count), 5)
    return results

def set_gpu(gpus):
    os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus