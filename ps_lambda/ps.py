from mxnet import ndarray as nd
import boto3


def push(w, b, params_s3_url):
    fname = params_s3_url.split('/')[-1]
    nd.save('./%s' % fname, [w, b])
    s3 = boto3.resource('s3')
    s3.meta.client.upload_file('./%s' % fname, 'ps-lambda-mxnet', fname)


def pull(params_s3_url):
    fname = params_s3_url.split('/')[-1]
    s3 = boto3.resource('s3')
    s3.meta.client.download_file('ps-lambda-mxnet', fname, './%s' % fname)
    w, b = nd.load('./%s' % fname)
    return w, b
