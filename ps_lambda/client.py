import boto3
import json
import mxnet as mx
from mxnet import nd, gluon
from multiprocessing import Pool
import argparse

from ps import push, pull


def real_fn(X):
    return 2 * X[:, 0] - 3.4 * X[:, 1] + 4.2


def gen_data(num_examples, num_inputs):
    X = nd.random_normal(shape=(num_examples, num_inputs))
    noise = .1 * nd.random_normal(shape=(num_examples,))
    y = real_fn(X) + noise
    return X, y


# lambda invoke
def lambda_call(payload):
    client = boto3.client('lambda')
    r = client.invoke(FunctionName='ps-lambda',
                      InvocationType="RequestResponse",
                      Payload=payload)['Payload']

    result = json.loads(r.read())
    print("Lambda result: " + str(result))
    return result


# upload data
def upload_input_data(data, s3_url):
    push(data, s3_url, False)
    return s3_url


# train
def train(batch_size, num_lambda, lr, epochs, s3_url, kv_url):

    pool = Pool(int(num_lambda)+1)
    procs= []

    for rank in range(0, int(num_lambda)):
        payload = json.dumps({
            "batch-size": batch_size,
            "learning-rate": lr,
            "s3-url": s3_url,
            "kv-url": kv_url,
            "rank": rank,
            "epochs": epochs
        })
        print("Launch AWS Lambda #%d" % rank)
        procs.append(pool.apply_async(lambda_call, (payload, )))

    res = [proc.get() for proc in procs]

    pool.close()
    pool.join()

    # total_loss = [np.mean(square_loss(net(X), y).asnumpy())]


def main():

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data', dest='is_data_ready', default=True,
                        action='store_false', help='is data ready in S3')
    args = parser.parse_args()

    epochs = 10
    learning_rate = .5
    batch_size = 100

    num_inputs = 2
    num_outputs = 1
    num_examples = 10000

    kv_url = "s3://ps-lambda-mxnet/w-b-%d" % num_examples
    s3_url = "s3://ps-lambda-mxnet/X-y-%d" % num_examples

    num_lambda = num_examples / batch_size

    X, y = gen_data(num_examples, num_inputs)

    if not args.is_data_ready:
        upload_input_data([X, y], s3_url)

    # init params
    w = nd.random_normal(shape=(num_inputs, num_outputs))
    b = nd.random_normal(shape=num_outputs)

    # push params to kvstore
    push([w, b], kv_url, False)

    for i in range(0, 10):
        train(batch_size, num_lambda, learning_rate, epochs, s3_url, kv_url)

    # collect final results
    w, b = pull(kv_url, False)
    print("Weight: ", w)
    print("Bias:", b)


if __name__ == '__main__':
    main()
