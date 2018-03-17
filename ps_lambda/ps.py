from mxnet import ndarray as nd


def push(w, b, params_s3_url):
    nd.save(params_s3_url, [w, b])


def pull(params_s3_url):
    w, b = nd.load(params_s3_url)
    return w, b
