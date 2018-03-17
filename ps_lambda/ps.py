from mxnet import ndarray as nd
import boto3

#todo: fix the bug of using nd.save/load


def push(data, params_s3_url):
    fname = params_s3_url.split('/')[-1]
    nd.save('./%s' % fname, data)
    s3 = boto3.resource('s3')
    s3.meta.client.upload_file('./%s' % fname, 'ps-lambda-mxnet', fname)


def pull(params_s3_url):
    fname = params_s3_url.split('/')[-1]
    s3 = boto3.resource('s3')
    s3.meta.client.download_file('ps-lambda-mxnet', fname, './%s' % fname)
    return nd.load('./%s' % fname)
