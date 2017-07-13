#!/bin/bash
mkdir ~/.aws/
aws configure set default.s3.signature_version s3v4
git pull
cd "deeprl-torch/experiments/2016 - Experience Retention/"
th AUTOMATED_EXPERIMENTS.lua