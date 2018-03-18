#!/usr/bin/env bash

zip -jg ../deploy/lambda_function.zip ../ps_lambda/ps.py
zip -jg ../deploy/lambda_function.zip ../ps_lambda/worker.py
#
#aws lambda create-function --function-name ps-lambda \
#--zip-file fileb://../deploy/lambda_function.zip --runtime python2.7 \
#--region us-east-2 --role arn:aws:iam::971057488869:role/lambda_basic_execution \
#--handler worker.lambda_handler --memory-size 1024 --timeout 300

echo "Upload to AWS Lambda"

aws lambda update-function-code --function-name ps-lambda \
--zip-file fileb://../deploy/lambda_function.zip