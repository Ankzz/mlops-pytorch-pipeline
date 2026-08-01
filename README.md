# mlops-pytorch-pipeline
This repository is part of IITM Term-3 Assignment-3

## Overview
In this assignment, you will take a PyTorch image classification model through the full deployment lifecycle: from
local development with proper Git workflows, to containerized training with Docker, to orchestrated deployment on
Kubernetes. By the end, you will have a production-style ML pipeline that can train and serve predictions at scale.

## Learning Objectives

By completing this assignment, you will be able to:
    1 Structure an ML project repository with proper Git practices (branching, PRs, .gitignore, secrets management)
    2 Write multi-stage Dockerfiles optimized for ML workloads
    3 Deploy PyTorch training jobs on Kubernetes using Jobs and persistent storage
    4 Serve a trained model via a Kubernetes Deployment with health checks
    5 Use ConfigMaps and Secrets for environment-specific configuration

## Prerequisites
    ● Python 3.10+, PyTorch experience
    ● Docker Desktop installed (or access to a Docker-enabled VM)
    ● kubectl CLI installed
    ● A Kubernetes cluster (Minikube, kind, or a cloud-managed cluster)
    ● A GitHub account
