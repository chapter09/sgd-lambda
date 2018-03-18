from mxnet import ndarray as nd
import boto3
import os

#todo: fix the bug of using nd.save/load


def push(data, url, is_lambda=True):
    fname = url.split('/')[-1]
    fname = fname.split('.')[0]

    if is_lambda:
        fpath = '/tmp/%s' % fname
    else:
        fpath = './%s' % fname

    nd.save(fpath, data)
    s3 = boto3.resource('s3')
    s3.meta.client.upload_file(fpath, 'ps-lambda-mxnet', fname)


def pull(url, is_lambda=True):
    fname = url.split('/')[-1]
    fname = fname.split('.')[0]
    if is_lambda:
        fpath = '/tmp/%s' % fname
    else:
        fpath = './%s' % fname

    s3 = boto3.resource('s3')
    s3.meta.client.download_file('ps-lambda-mxnet', fname, fpath)
    return nd.load(fpath)
